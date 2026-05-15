"""
app/core/history.py
--------------------
SQLite-backed conversation history store.
"""

from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from app.core.config import settings
from app.core.pii import redact_state, redact_pii


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT    NOT NULL,
                username    TEXT,
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
        _migrate(conn)


def save(query: str, state: dict, username: str | None = None) -> int:
    """Persist one pipeline result; returns the new row id."""
    analysis = state.get("analysis", {})
    consent = False
    if isinstance(state, dict):
        consent = bool(state.get("consent") or (state.get("meta") and state["meta"].get("consent")))

    if consent:
        stored_state = state
    else:
        # Redact only user-provided text (query already stored separately).
        # System-generated fields (rag_sources, analysis, agent_log, tickers,
        # market_data, token_usage) must not be touched — they contain dates,
        # ticker symbols and other patterns that false-positive on PII regexes.
        stored_state = dict(state)
        if stored_state.get("articles"):
            stored_state["articles"] = [
                {**a, "text": redact_pii(a["text"])} if a.get("text") else a
                for a in stored_state["articles"]
            ]
        stored_state["_pii_redacted"] = True

    row = {
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "username":    username,
        "query":       query,
        "sentiment":   analysis.get("sentiment", ""),
        "summary":     analysis.get("summary", ""),
        "tickers":     json.dumps(state.get("tickers", [])),
        "rag_sources": json.dumps(state.get("rag_sources", [])),
        "token_total": state.get("token_usage", {}).get("total", 0),
        "latency_s":   state.get("total_latency_s", 0.0),
        "full_state":  json.dumps(stored_state),
    }
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO conversations
               (created_at, username, query, sentiment, summary, tickers,
                rag_sources, token_total, latency_s, full_state)
               VALUES
               (:created_at, :username, :query, :sentiment, :summary, :tickers,
                :rag_sources, :token_total, :latency_s, :full_state)""",
            row,
        )
        return cur.lastrowid


def get_history(limit: int = 50, username: str | None = None) -> list[dict]:
    """Return the most recent `limit` conversations, filtered by username if given."""
    with _conn() as conn:
        if username:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE username = ? ORDER BY id DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        else:
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


def _migrate(conn: sqlite3.Connection) -> None:
    """Add missing columns safely (idempotent)."""
    cur = conn.execute("PRAGMA table_info(conversations)")
    cols = {row[1] for row in cur.fetchall()}
    alters = []
    if "user_rating" not in cols:
        alters.append("ALTER TABLE conversations ADD COLUMN user_rating INTEGER")
    if "feedback_text" not in cols:
        alters.append("ALTER TABLE conversations ADD COLUMN feedback_text TEXT")
    if "username" not in cols:
        alters.append("ALTER TABLE conversations ADD COLUMN username TEXT")
    for sql in alters:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass
    conn.commit()


def update_feedback(
    conv_id: int,
    rating: int | None = None,
    feedback_text: str | None = None,
    username: str | None = None,
) -> bool:
    """Update user_rating / feedback_text. Checks username ownership when provided."""
    sets: list[str] = []
    params: list[object] = []
    if rating is not None:
        if rating < 1 or rating > 5:
            raise ValueError("rating must be between 1 and 5")
        sets.append("user_rating = ?")
        params.append(rating)
    if feedback_text is not None:
        sets.append("feedback_text = ?")
        params.append(redact_pii(feedback_text)[:2000])
    if not sets:
        return False
    params.append(conv_id)
    if username:
        sql = f"UPDATE conversations SET {', '.join(sets)} WHERE id = ? AND username = ?"
        params.append(username)
    else:
        sql = f"UPDATE conversations SET {', '.join(sets)} WHERE id = ?"
    with _conn() as conn:
        cur = conn.execute(sql, params)
        return cur.rowcount > 0


def clear(username: str | None = None) -> int:
    """Delete conversations. When username is given, deletes only that user's rows."""
    with _conn() as conn:
        if username:
            cur = conn.execute("DELETE FROM conversations WHERE username = ?", (username,))
        else:
            cur = conn.execute("DELETE FROM conversations")
        return cur.rowcount


def erase_by_trace_id(trace_id: str) -> int:
    """Delete conversations whose persisted `full_state` contains `trace_id`.

    Returns the number of rows deleted.
    """
    if not trace_id:
        return 0
    with _conn() as conn:
        rows = conn.execute("SELECT id, full_state FROM conversations").fetchall()
        to_delete = []
        for r in rows:
            try:
                fs = json.loads(r["full_state"] or "{}")
                if fs.get("trace_id") == trace_id:
                    to_delete.append(r["id"])
            except Exception:
                continue
        deleted = 0
        for idv in to_delete:
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (idv,))
            deleted += cur.rowcount
        return deleted


def get_user_tokens(username: str | None = None) -> int:
    """Return total tokens consumed by username (all users if None)."""
    with _conn() as conn:
        if username:
            row = conn.execute(
                "SELECT COALESCE(SUM(token_total), 0) FROM conversations WHERE username = ?",
                (username,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COALESCE(SUM(token_total), 0) FROM conversations",
            ).fetchone()
    return int(row[0]) if row else 0


def purge_older_than(days: int) -> int:
    """Purge conversations older than `days`. Returns number deleted."""
    if not days or days <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.isoformat()
    with _conn() as conn:
        cur = conn.execute("DELETE FROM conversations WHERE created_at < ?", (cutoff_iso,))
        return cur.rowcount
