"""
Login / registration endpoints (username + password → JWT).
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.limits import limiter
from app.core import users_store as usr
from app.core.config import settings
from app.core.jwt_tokens import create_access_token, decode_access_token
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_ADMIN_NAMES = lambda: [n.strip() for n in settings.ADMIN_USERNAMES.split(",") if n.strip()]  # noqa: E731


@router.get("/config")
def auth_config():
    """Public: tells the UI how authentication is configured."""
    return {
        "auth_required": settings.AUTH_ENABLED,
        "registration_enabled": settings.AUTH_ENABLED and settings.ALLOW_USER_REGISTRATION,
    }


@router.get("/me")
async def get_me(authorization: str | None = Header(None)):
    """Return info about the currently authenticated user."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    token = authorization[7:].strip()
    payload = decode_access_token(token)
    if not payload or payload.get("typ") != "access":
        raise HTTPException(status_code=403, detail="Invalid token.")
    username = str(payload["sub"])
    return {
        "username": username,
        "user_id": int(payload.get("uid", 0)),
        "is_admin": username in _ADMIN_NAMES(),
    }


class LoginBody(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


@router.post("/login")
@limiter.limit("12/minute")
async def login(request: Request, body: LoginBody):
    name = body.username.strip()

    # Check if account is blocked before verifying password (avoids timing leak)
    if usr.is_blocked(name):
        raise HTTPException(status_code=403, detail="Account is blocked. Contact an administrator.")

    uid = usr.verify_credentials(name, body.password)
    if uid is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    try:
        token = create_access_token(username=name, user_id=uid)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": max(5, settings.JWT_EXPIRE_MINUTES) * 60,
        "username": name,
        "is_admin": name in _ADMIN_NAMES(),
    }


class RegisterBody(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8, max_length=256)


@router.post("/register")
@limiter.limit("5/minute")
async def register(request: Request, body: RegisterBody):
    if not settings.ALLOW_USER_REGISTRATION:
        raise HTTPException(status_code=403, detail="Registration is disabled.")
    name = body.username.strip()
    try:
        uid = usr.create_user(name, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Username already taken.") from None

    logger.info(f"New user registered: {name!r}")

    try:
        token = create_access_token(username=name, user_id=uid)
    except RuntimeError:
        return {"ok": True, "username": name, "access_token": None}

    return {
        "ok": True,
        "username": name,
        "access_token": token,
        "token_type": "bearer",
        "expires_in": max(5, settings.JWT_EXPIRE_MINUTES) * 60,
        "is_admin": name in _ADMIN_NAMES(),
    }
