"""
tests/test_users_store.py
--------------------------
Unit tests for app/core/users_store.py:
  - User creation (positive + negative)
  - Credential verification
  - Block / unblock
  - Delete
  - Password reset
  - List users
"""

from __future__ import annotations

import sqlite3
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# create_user
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateUser:

    def test_create_valid_user_returns_id(self, isolated_db):
        from app.core import users_store as usr
        uid = usr.create_user("alice", "securepass1")
        assert isinstance(uid, int)
        assert uid > 0

    def test_created_user_can_log_in(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("bob", "securepass1")
        assert usr.verify_credentials("bob", "securepass1") is not None

    def test_duplicate_username_raises_integrity_error(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("dupuser", "securepass1")
        with pytest.raises(sqlite3.IntegrityError):
            usr.create_user("dupuser", "different1234")

    def test_short_username_raises_value_error(self, isolated_db):
        from app.core import users_store as usr
        with pytest.raises(ValueError, match="Username"):
            usr.create_user("ab", "securepass1")

    def test_invalid_chars_in_username_raises(self, isolated_db):
        from app.core import users_store as usr
        with pytest.raises(ValueError):
            usr.create_user("bad user!", "securepass1")

    def test_short_password_raises_value_error(self, isolated_db):
        from app.core import users_store as usr
        with pytest.raises(ValueError, match="Password"):
            usr.create_user("gooduser", "short")

    def test_username_with_dots_and_dashes_valid(self, isolated_db):
        from app.core import users_store as usr
        uid = usr.create_user("john.doe-99", "securepass1")
        assert uid > 0


# ═══════════════════════════════════════════════════════════════════════════════
# verify_credentials
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerifyCredentials:

    def test_correct_credentials_return_user_id(self, isolated_db):
        from app.core import users_store as usr
        uid_created = usr.create_user("verifyme", "mypassword1")
        uid_returned = usr.verify_credentials("verifyme", "mypassword1")
        assert uid_returned == uid_created

    def test_wrong_password_returns_none(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("wrongpw", "correctpass1")
        assert usr.verify_credentials("wrongpw", "wrongpassword!") is None

    def test_unknown_username_returns_none(self, isolated_db):
        from app.core import users_store as usr
        assert usr.verify_credentials("nobody_xyz", "anypassword") is None

    def test_blocked_user_returns_none(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("blocked_cred", "mypassword1")
        usr.block_user("blocked_cred")
        assert usr.verify_credentials("blocked_cred", "mypassword1") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Block / unblock
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlockUnblock:

    def test_block_user(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("blockme", "mypassword1")
        assert usr.block_user("blockme") is True
        assert usr.is_blocked("blockme") is True

    def test_unblock_user(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("unblockme", "mypassword1")
        usr.block_user("unblockme")
        assert usr.unblock_user("unblockme") is True
        assert usr.is_blocked("unblockme") is False

    def test_block_nonexistent_user_returns_false(self, isolated_db):
        from app.core import users_store as usr
        assert usr.block_user("ghost_xyz") is False

    def test_unblock_nonexistent_user_returns_false(self, isolated_db):
        from app.core import users_store as usr
        assert usr.unblock_user("ghost_xyz") is False

    def test_is_blocked_false_for_active_user(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("active_user", "mypassword1")
        assert usr.is_blocked("active_user") is False


# ═══════════════════════════════════════════════════════════════════════════════
# Delete
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeleteUser:

    def test_delete_existing_user_returns_true(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("deleteme", "mypassword1")
        assert usr.delete_user("deleteme") is True

    def test_deleted_user_cannot_log_in(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("gone_user", "mypassword1")
        usr.delete_user("gone_user")
        assert usr.verify_credentials("gone_user", "mypassword1") is None

    def test_delete_nonexistent_user_returns_false(self, isolated_db):
        from app.core import users_store as usr
        assert usr.delete_user("nobody_xyz") is False

    def test_delete_removes_from_list(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("list_user", "mypassword1")
        usr.delete_user("list_user")
        names = [u["username"] for u in usr.list_users()]
        assert "list_user" not in names


# ═══════════════════════════════════════════════════════════════════════════════
# Password reset
# ═══════════════════════════════════════════════════════════════════════════════

class TestResetPassword:

    def test_reset_allows_new_login(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("resetme", "oldpassword1")
        usr.reset_password("resetme", "newpassword1")
        assert usr.verify_credentials("resetme", "newpassword1") is not None

    def test_reset_invalidates_old_password(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("resetme2", "oldpassword1")
        usr.reset_password("resetme2", "newpassword1")
        assert usr.verify_credentials("resetme2", "oldpassword1") is None

    def test_reset_short_password_raises(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("resetme3", "oldpassword1")
        with pytest.raises(ValueError):
            usr.reset_password("resetme3", "short")

    def test_reset_nonexistent_user_returns_false(self, isolated_db):
        from app.core import users_store as usr
        assert usr.reset_password("nobody_xyz", "newpassword1") is False


# ═══════════════════════════════════════════════════════════════════════════════
# List users
# ═══════════════════════════════════════════════════════════════════════════════

class TestListUsers:

    def test_list_includes_created_user(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("listable", "mypassword1")
        names = [u["username"] for u in usr.list_users()]
        assert "listable" in names

    def test_list_does_not_include_password_hash(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("nohash", "mypassword1")
        for u in usr.list_users():
            assert "password_hash" not in u

    def test_list_includes_is_blocked_field(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("fieldcheck", "mypassword1")
        for u in usr.list_users():
            assert "is_blocked" in u

    def test_blocked_user_shows_in_list_as_blocked(self, isolated_db):
        from app.core import users_store as usr
        usr.create_user("will_block", "mypassword1")
        usr.block_user("will_block")
        user = next(u for u in usr.list_users() if u["username"] == "will_block")
        assert bool(user["is_blocked"]) is True
