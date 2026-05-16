"""
tests/test_auth.py
-------------------
Authentication endpoint tests: login, register, JWT validation, access control.

Uses raw_client (no dependency overrides) to test the real auth flow.
"""

from __future__ import annotations

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _register(client, username: str, password: str = "password123"):
    return client.post("/api/v1/auth/register", json={"username": username, "password": password})


def _login(client, username: str, password: str = "password123"):
    return client.post("/api/v1/auth/login", json={"username": username, "password": password})


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ═══════════════════════════════════════════════════════════════════════════════
# /auth/config
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuthConfig:

    def test_config_is_public(self, raw_client):
        resp = raw_client.get("/api/v1/auth/config")
        assert resp.status_code == 200

    def test_config_has_required_keys(self, raw_client):
        data = raw_client.get("/api/v1/auth/config").json()
        assert "auth_required" in data
        assert "registration_enabled" in data

    def test_auth_required_is_true(self, raw_client):
        data = raw_client.get("/api/v1/auth/config").json()
        assert data["auth_required"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# /auth/register
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegister:

    def test_register_new_user(self, raw_client, isolated_db):
        resp = _register(raw_client, "newuser_reg")
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "newuser_reg"
        assert data.get("access_token") is not None

    def test_register_returns_jwt(self, raw_client, isolated_db):
        resp = _register(raw_client, "tokenuser_reg")
        assert resp.status_code == 200
        token = resp.json().get("access_token")
        assert token and len(token) > 20

    def test_register_auto_login_token_is_valid(self, raw_client, isolated_db):
        token = _register(raw_client, "autologin_user").json()["access_token"]
        me_resp = raw_client.get("/api/v1/auth/me", headers=_bearer(token))
        assert me_resp.status_code == 200
        assert me_resp.json()["username"] == "autologin_user"

    def test_register_duplicate_username_returns_409(self, raw_client, isolated_db):
        _register(raw_client, "dupuser_reg")
        resp = _register(raw_client, "dupuser_reg")
        assert resp.status_code == 409

    def test_register_short_password_returns_422(self, raw_client, isolated_db):
        resp = raw_client.post(
            "/api/v1/auth/register",
            json={"username": "weakpw_user", "password": "short"},
        )
        assert resp.status_code == 422

    def test_register_too_short_username_returns_422(self, raw_client, isolated_db):
        resp = raw_client.post(
            "/api/v1/auth/register",
            json={"username": "ab", "password": "password123"},
        )
        assert resp.status_code == 422

    def test_register_is_not_admin(self, raw_client, isolated_db):
        data = _register(raw_client, "regular_user_reg").json()
        assert data.get("is_admin") is False


# ═══════════════════════════════════════════════════════════════════════════════
# /auth/login
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogin:

    def test_login_valid_credentials(self, raw_client, isolated_db):
        _register(raw_client, "loginuser")
        resp = _login(raw_client, "loginuser")
        assert resp.status_code == 200
        assert resp.json().get("access_token") is not None

    def test_login_returns_username(self, raw_client, isolated_db):
        _register(raw_client, "nameduser")
        data = _login(raw_client, "nameduser").json()
        assert data["username"] == "nameduser"

    def test_login_wrong_password_returns_401(self, raw_client, isolated_db):
        _register(raw_client, "wrongpw_user")
        resp = _login(raw_client, "wrongpw_user", "wrongpassword!")
        assert resp.status_code == 401

    def test_login_nonexistent_user_returns_401(self, raw_client, isolated_db):
        resp = _login(raw_client, "ghost_user_xyz")
        assert resp.status_code == 401

    def test_login_blocked_user_returns_403(self, raw_client, isolated_db):
        from app.core import users_store as usr
        _register(raw_client, "blocked_login_user")
        usr.block_user("blocked_login_user")
        resp = _login(raw_client, "blocked_login_user")
        assert resp.status_code == 403

    def test_login_token_is_decodable(self, raw_client, isolated_db):
        _register(raw_client, "decode_user")
        token = _login(raw_client, "decode_user").json()["access_token"]
        from app.core.jwt_tokens import decode_access_token
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "decode_user"
        assert payload["typ"] == "access"

    def test_admin_login_returns_is_admin_true(self, raw_client, isolated_db):
        from app.core import users_store as usr
        usr.upsert_user_password("admin", "admin1234")
        data = raw_client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin1234"},
        ).json()
        assert data.get("is_admin") is True


# ═══════════════════════════════════════════════════════════════════════════════
# /auth/me
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetMe:

    def test_me_with_valid_token(self, raw_client, isolated_db):
        token = _register(raw_client, "me_user").json()["access_token"]
        resp = raw_client.get("/api/v1/auth/me", headers=_bearer(token))
        assert resp.status_code == 200
        assert resp.json()["username"] == "me_user"

    def test_me_without_token_returns_401(self, raw_client):
        resp = raw_client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_me_with_garbage_token_returns_403(self, raw_client):
        resp = raw_client.get("/api/v1/auth/me", headers=_bearer("garbage.token.value"))
        assert resp.status_code == 403

    def test_me_returns_is_admin_field(self, raw_client, isolated_db):
        token = _register(raw_client, "admin_check_user").json()["access_token"]
        data = raw_client.get("/api/v1/auth/me", headers=_bearer(token)).json()
        assert "is_admin" in data


# ═══════════════════════════════════════════════════════════════════════════════
# Access control — protected routes require valid Bearer token
# ═══════════════════════════════════════════════════════════════════════════════

class TestAccessControl:

    def test_analyze_without_token_returns_401(self, raw_client):
        resp = raw_client.post("/api/v1/analyze", json={"query": "Tesla"})
        assert resp.status_code == 401

    def test_history_without_token_returns_401(self, raw_client):
        resp = raw_client.get("/api/v1/history")
        assert resp.status_code == 401

    def test_admin_endpoint_with_user_token_returns_403(self, raw_client, isolated_db):
        token = _register(raw_client, "nonadmin_access").json()["access_token"]
        resp = raw_client.get("/api/v1/admin/users", headers=_bearer(token))
        assert resp.status_code == 403

    def test_admin_endpoint_with_admin_token(self, raw_client, isolated_db):
        from app.core import users_store as usr
        usr.upsert_user_password("admin", "admin1234")
        token = raw_client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin1234"},
        ).json()["access_token"]
        resp = raw_client.get("/api/v1/admin/users", headers=_bearer(token))
        assert resp.status_code == 200
