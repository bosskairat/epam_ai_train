"""
app/core/history.py
--------------------
SQLite-backed conversation history store.
"""

from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from app.core.config import settings


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.HISTORY_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT    NOT NULL,
                query       TEXT    NOT NULL,
                sentiment   TEXT,
                summary     TEXT,
                tickers     TEXT,
                rag_sources TEXT,
                token_total INTEGER,
                latency_s   REAL,
                full_state  TEXT
            )
        """)


def save(query: str, state: dict) -> int:
    """Persist one pipeline result; returns the new row id."""
    analysis = state.get("analysis", {})
    row = {
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "query":       query,
        "sentiment":   analysis.get("sentiment", ""),
        "summary":     analysis.get("summary", ""),
        "tickers":     json.dumps(state.get("tickers", [])),
        "rag_sources": json.dumps(state.get("rag_sources", [])),
        "token_total": state.get("token_usage", {}).get("total", 0),
        "latency_s":   state.get("total_latency_s", 0.0),
        "full_state":  json.dumps(state),
    }
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO conversations
               (created_at, query, sentiment, summary, tickers,
                rag_sources, token_total, latency_s, full_state)
               VALUES
               (:created_at, :query, :sentiment, :summary, :tickers,
                :rag_sources, :token_total, :latency_s, :full_state)""",
            row,
        )
        return cur.lastrowid


def get_history(limit: int = 50) -> list[dict]:
    """Return the most recent `limit` conversations, newest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["tickers"] = json.loads(d["tickers"] or "[]")
        d["rag_sources"] = json.loads(d["rag_sources"] or "[]")
        d["full_state"] = json.loads(d["full_state"] or "{}")
        result.append(d)
    return result


def clear() -> int:
    """Delete all rows; returns count deleted."""
    with _conn() as conn:
        cur = conn.execute("DELETE FROM conversations")
        return cur.rowcount
