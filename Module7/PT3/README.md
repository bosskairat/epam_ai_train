Orders Management API (FastAPI)

Quick start

- Create virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Run app:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Seed happens automatically on first import.

API endpoints

- POST /orders  — create order
- GET /orders?page=1&limit=10&status=created&min_amount=10&max_amount=100&start_date=2024-01-01&end_date=2024-12-31

Tests

```bash
pytest -q
```
