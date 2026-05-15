"""
FastAPI dependency: Bearer JWT authentication.
Returns None when AUTH_ENABLED=false (open API).
"""

from __future__ import annotations

from typing import Any

from fastapi import Header, HTTPException

from app.core.config import settings
from app.core.jwt_tokens import decode_access_token


def _admin_names() -> list[str]:
    return [n.strip() for n in settings.ADMIN_USERNAMES.split(",") if n.strip()]


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return authorization[7:].strip() or None


async def require_auth_principal(
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Requires Authorization: Bearer <JWT>. Always enforced."""
    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")

    payload = decode_access_token(token)
    if payload and payload.get("typ") == "access" and payload.get("sub"):
        username = str(payload["sub"])
        return {
            "sub": username,
            "user_id": int(payload.get("uid", 0)),
            "is_admin": username in _admin_names(),
        }

    raise HTTPException(status_code=403, detail="Invalid or expired token.")


async def require_admin(
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Like require_auth_principal but additionally enforces admin role."""
    principal = await require_auth_principal(authorization)
    if principal is None or not principal.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin privileges required.")
    return principal


require_api_key = require_auth_principal
