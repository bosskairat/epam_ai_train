"""
JWT access tokens for authenticated users (HS256).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import settings


def create_access_token(*, username: str, user_id: int) -> str:
    if not settings.JWT_SECRET or len(settings.JWT_SECRET) < 16:
        raise RuntimeError("JWT_SECRET must be set to at least 16 characters when issuing tokens.")
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=max(5, settings.JWT_EXPIRE_MINUTES))
    payload = {
        "sub": username,
        "uid": user_id,
        "typ": "access",
        "iat": int(now.timestamp()),
        "exp": exp,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def decode_access_token(token: str) -> dict | None:
    if not settings.JWT_SECRET:
        return None
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
