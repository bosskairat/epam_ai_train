"""
tests/conftest.py
------------------
Shared pytest configuration and fixtures.

Environment variables are set at module level — before any app import —
so the application never requires real API keys or a production database.
"""

from __future__ import annotations

import os
import tempfile

# ── Test environment (must happen before any app.* import) ───────────────────

_TEST_DB = tempfile.mktemp(suffix=".db", prefix="fin_test_")

os.environ["DB_PATH"]                    = _TEST_DB
os.environ["JWT_SECRET"]                 = "test-secret-key-32-chars-minimum-ok!"
os.environ["AUTH_BOOTSTRAP_USERNAME"]    = "admin"
os.environ["AUTH_BOOTSTRAP_PASSWORD"]    = "admin1234"
os.environ["ADMIN_USERNAMES"]            = "admin"
os.environ["ALLOW_USER_REGISTRATION"]    = "true"
os.environ["ENABLE_CONTENT_MODERATION"]  = "false"
os.environ.setdefault("OPENAI_API_KEY",    "")
os.environ.setdefault("NEWS_API_KEY",      "")
os.environ.setdefault("FINNHUB_API_KEY",   "test-key")
os.environ.setdefault("LOG_LEVEL",         "WARNING")

# ── Shared stub data ─────────────────────────────────────────────────────────

STUB_MARKET_DATA = {
    "TSLA": {
        "ticker": "TSLA",
        "company_name": "Tesla, Inc.",
        "current_price": 185.40,
        "change_pct": -3.21,
        "volume": 95_000_000,
        "market_cap": 590_000_000_000,
        "sector": "Consumer Cyclical",
        "history": [],
        "fetched_at": "2024-01-15T12:00:00",
    }
}

STUB_ARTICLES = [
    {
        "title": "Tesla shares plunge after disappointing delivery report",
        "summary": "Tesla missed Q4 delivery estimates amid growing competition.",
        "source": "Reuters",
        "url": "https://reuters.com/article/1",
        "published_at": "2024-01-15T08:00:00Z",
        "text": "Tesla Inc. stock fell 5% after the EV maker missed delivery targets.",
    },
    {
        "title": "EV market faces headwinds as interest rates stay high",
        "summary": "Higher borrowing costs are slowing EV adoption globally.",
        "source": "Bloomberg",
        "url": "https://bloomberg.com/article/2",
        "published_at": "2024-01-15T06:30:00Z",
        "text": "Rising rates continue to weigh on electric vehicle demand worldwide.",
    },
]

STUB_ANALYSIS = {
    "summary": "Tesla shares declined sharply following weaker-than-expected delivery numbers.",
    "sentiment": "Bearish",
    "key_drivers": ["Missed delivery estimates", "Rising EV competition", "High interest rates"],
    "risk_factors": ["Further demand slowdown", "Price war escalation"],
    "insight": (
        "Tesla's stock movements often correlate strongly with delivery data. "
        "Investors monitor quarterly deliveries as a proxy for demand health."
    ),
    "sources_used": ["https://reuters.com/article/1", "https://bloomberg.com/article/2"],
    "disclaimer": "This is not financial advice.",
}

STUB_STATE: dict = {
    "query": "Why did Tesla stock drop today?",
    "tickers": ["TSLA"],
    "market_data": STUB_MARKET_DATA,
    "articles": STUB_ARTICLES,
    "analysis": STUB_ANALYSIS,
    "rag_sources": ["market:TSLA — 2024-01-15", "news — 2024-01-15"],
    "token_usage": {"prompt": 420, "completion": 180, "total": 600},
    "total_latency_s": 2.1,
    "agent_log": [
        "data_agent: fetched 1 tickers in 0.8s",
        "news_agent: fetched 2 articles in 0.5s",
        "analysis_agent: done in 0.8s | tokens=600",
    ],
    "hallucination_score": 0.15,
    "verification": {"hallucination_score": 0.15},
    "trace_id": "test-trace-001",
}

# ── Fixtures ──────────────────────────────────────────────────────────────────

import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture
def stub_state() -> dict:
    """Return a fresh copy of STUB_STATE so tests can mutate freely."""
    import copy
    return copy.deepcopy(STUB_STATE)


@pytest.fixture
def mock_pipeline(stub_state):
    """Patch run_pipeline at both the module definition and the import site in routes."""
    with patch(
        "app.agents.supervisor_agent.run_pipeline",
        new_callable=AsyncMock,
        return_value=stub_state,
    ), patch(
        "app.api.routes.run_pipeline",
        new_callable=AsyncMock,
        return_value=stub_state,
    ) as mock:
        yield mock, stub_state


def _make_token(username: str = "testuser", user_id: int = 1) -> str:
    """Mint a valid JWT for the given username using the test secret."""
    from app.core.jwt_tokens import create_access_token
    return create_access_token(username=username, user_id=user_id)


@pytest.fixture
def user_token() -> str:
    return _make_token("testuser")


@pytest.fixture
def admin_token() -> str:
    return _make_token("admin")


@pytest.fixture
def auth_client(mock_pipeline):
    """
    FastAPI TestClient with auth dependencies stubbed out.
    Use for endpoint tests that are NOT about the auth system itself.
    """
    from fastapi.testclient import TestClient
    from app.api.app import create_app
    from app.api.deps import require_auth_principal, require_admin

    app = create_app()

    async def _stub_user():
        return {"sub": "testuser", "user_id": 1, "is_admin": False}

    async def _stub_admin():
        return {"sub": "admin", "user_id": 1, "is_admin": True}

    app.dependency_overrides[require_auth_principal] = _stub_user
    app.dependency_overrides[require_admin] = _stub_admin

    return TestClient(app)


@pytest.fixture
def raw_client():
    """
    FastAPI TestClient with real auth (no overrides).
    Use for testing login, register, and token validation.
    """
    from fastapi.testclient import TestClient
    from app.api.app import create_app
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def disable_rate_limits():
    """Disable SlowAPI rate limiting for the duration of each test."""
    try:
        from app.api.limits import limiter
        limiter.enabled = False
        yield
        limiter.enabled = True
    except Exception:
        yield


@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    """
    Point settings.DB_PATH at a per-test SQLite file and re-initialise tables.
    Use in unit tests for users_store or history that need a clean slate.
    """
    db_path = str(tmp_path / "isolated.db")
    from app.core import config
    monkeypatch.setattr(config.settings, "DB_PATH", db_path)
    from app.core import history as hist
    from app.core import users_store as usr
    hist.init_db()
    usr.init_db()
    yield db_path
