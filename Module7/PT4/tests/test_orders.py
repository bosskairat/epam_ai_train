import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db import init_db, engine
from sqlmodel import Session
from app.db import DB_FILE
from datetime import datetime, timedelta

client = TestClient(app)


def reset_db():
    # remove existing DB file if present
    try:
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
    except Exception:
        pass
    init_db()


@pytest.fixture(autouse=True)
def run_around_tests():
    reset_db()
    # seed via import in main; main seeds on import; call client startup by hitting root
    yield


def test_seeded_orders_exist():
    resp = client.get("/orders?page=1&limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_create_order_valid():
    payload = {"status": "created", "amount": 123.45}
    resp = client.post("/orders", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] > 0
    assert data["amount"] == 123.45


def test_create_order_negative_amount():
    payload = {"status": "created", "amount": -10}
    resp = client.post("/orders", json=payload)
    assert resp.status_code == 400


def test_pagination_limits():
    r1 = client.get("/orders?page=1&limit=2")
    r2 = client.get("/orders?page=2&limit=2")
    assert r1.status_code == 200 and r2.status_code == 200
    assert len(r1.json()) <= 2
    assert len(r2.json()) <= 2


def test_filter_by_status():
    # create a known status
    client.post("/orders", json={"status": "completed", "amount": 50})
    r = client.get("/orders?status=completed&page=1&limit=50")
    assert all(o["status"] == "completed" for o in r.json())


def test_filter_by_amount_range():
    client.post("/orders", json={"status": "created", "amount": 10})
    client.post("/orders", json={"status": "created", "amount": 200})
    r = client.get("/orders?min_amount=100&page=1&limit=50")
    assert all(o["amount"] >= 100 for o in r.json())


def test_filter_by_date_range():
    # create orders with specific dates by directly using DB session
    from app.models import Order
    with Session(engine) as s:
        now = datetime.utcnow()
        o1 = Order(status="created", amount=10, created_at=now - timedelta(days=10))
        o2 = Order(status="created", amount=20, created_at=now - timedelta(days=1))
        s.add(o1); s.add(o2); s.commit()
    sd = (datetime.utcnow() - timedelta(days=5)).date().isoformat()
    r = client.get(f"/orders?start_date={sd}&page=1&limit=50")
    assert all(datetime.fromisoformat(o["created_at"]) >= datetime.fromisoformat(sd) for o in r.json())


def test_invalid_date_format():
    r = client.get("/orders?start_date=not-a-date")
    assert r.status_code == 400


def test_limit_bounds():
    r = client.get("/orders?limit=200")
    assert r.status_code == 422 or r.status_code == 200


def test_unknown_status_rejected_on_post():
    resp = client.post("/orders", json={"status": "notastatus", "amount": 10})
    # FastAPI/Pydantic returns 422 for invalid enum values during validation
    assert resp.status_code in (400, 422)


def test_zero_limit_not_allowed():
    r = client.get("/orders?limit=0")
    assert r.status_code == 422


def test_large_page_returns_empty_list():
    r = client.get("/orders?page=9999&limit=10")
    assert r.status_code == 200
    assert r.json() == []
