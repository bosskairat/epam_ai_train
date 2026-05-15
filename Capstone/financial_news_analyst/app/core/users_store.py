"""
SQLite-backed user accounts (username + bcrypt password hash).
Uses the same database file as conversation history (DB_PATH → app.db).
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from typing import List

import bcrypt

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_USERNAME_RE = re.compile(r"^[\w.-]{3,64}$")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("ascii"),
        )
    except (ValueError, TypeError):
        return False


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                is_blocked    INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        # Safe migration: add is_blocked if table already exists without it
        try:
            conn.execute("ALTER TABLE users ADD COLUMN is_blocked INTEGER NOT NULL DEFAULT 0")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


def valid_username(username: str) -> bool:
    return bool(username and _USERNAME_RE.match(username))


def create_user(username: str, password: str) -> int:
    """Insert a new user. Raises sqlite3.IntegrityError if username exists."""
    if not valid_username(username):
        raise ValueError("Username must be 3–64 characters: letters, digits, _, ., or -")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    h = _hash_password(password)
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
            (username, h, now),
        )
        return int(cur.lastrowid)


def upsert_user_password(username: str, password: str) -> int:
    """Create or replace password for username (bootstrap / admin reset)."""
    if not valid_username(username):
        raise ValueError("Invalid username for bootstrap.")
    if len(password) < 8:
        raise ValueError("Bootstrap password must be at least 8 characters.")
    h = _hash_password(password)
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row:
            uid = int(row["id"])
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (h, uid))
            return uid
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
            (username, h, now),
        )
        return int(cur.lastrowid)


def verify_credentials(username: str, password: str) -> int | None:
    """Return user id if credentials are valid and account is not blocked, else None."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, password_hash, is_blocked FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if not row:
        return None
    if row["is_blocked"]:
        return None
    if not _verify_password(password, row["password_hash"]):
        return None
    return int(row["id"])


def is_blocked(username: str) -> bool:
    """Return True if the account exists and is blocked."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT is_blocked FROM users WHERE username = ?", (username,)
        ).fetchone()
    return bool(row and row["is_blocked"])


def list_users() -> List[dict]:
    """Return all users (without password hashes)."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, username, created_at, is_blocked FROM users ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def block_user(username: str) -> bool:
    """Block a user account. Returns True if the user was found."""
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE users SET is_blocked = 1 WHERE username = ?", (username,)
        )
        return cur.rowcount > 0


def unblock_user(username: str) -> bool:
    """Unblock a user account. Returns True if the user was found."""
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE users SET is_blocked = 0 WHERE username = ?", (username,)
        )
        return cur.rowcount > 0


def delete_user(username: str) -> bool:
    """Permanently delete a user account. Returns True if deleted."""
    with _conn() as conn:
        cur = conn.execute("DELETE FROM users WHERE username = ?", (username,))
        return cur.rowcount > 0


def reset_password(username: str, new_password: str) -> bool:
    """Reset a user's password (admin action). Returns True if the user was found."""
    if len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    h = _hash_password(new_password)
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?", (h, username)
        )
        return cur.rowcount > 0


def bootstrap_from_env() -> None:
    """Ensure AUTH_BOOTSTRAP_* account exists on startup."""
    u, p = settings.AUTH_BOOTSTRAP_USERNAME, settings.AUTH_BOOTSTRAP_PASSWORD
    if not u or not p:
        return
    try:
        upsert_user_password(u, p)
        logger.info(f"Bootstrap user ensured: {u!r}")
    except ValueError as exc:
        logger.warning(f"Bootstrap user skipped: {exc}")
