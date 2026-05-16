"""
tests/test_history.py
----------------------
Tests for conversation history: per-user isolation, feedback, token usage,
PII redaction on save, and retention purge.
"""

from __future__ import annotations

import pytest


SAMPLE_STATE = {
    "analysis": {
        "sentiment": "Bullish",
        "summary": "Markets are up.",
        "key_drivers": ["Strong earnings"],
        "risk_factors": [],
        "insight": "Earnings beat expectations.",
        "sources_used": ["https://example.com"],
        "disclaimer": "This is not financial advice.",
    },
    "tickers": ["AAPL"],
    "rag_sources": ["market:AAPL — 2024-01-15"],
    "token_usage": {"prompt": 100, "completion": 50, "total": 150},
    "total_latency_s": 1.5,
    "agent_log": ["data_agent: 1 ticker"],
    "articles": [{"text": "Apple stock rose 3% after earnings beat.", "url": "https://x.com"}],
    "consent": False,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Per-user isolation
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerUserIsolation:

    def test_user_sees_only_own_history(self, isolated_db):
        from app.core import history as hist
        hist.save("Query A", SAMPLE_STATE, username="alice")
        hist.save("Query B", SAMPLE_STATE, username="bob")

        alice_hist = hist.get_history(limit=50, username="alice")
        assert all(r["username"] == "alice" for r in alice_hist)
        assert len(alice_hist) == 1
        assert alice_hist[0]["query"] == "Query A"

    def test_user_does_not_see_other_users_records(self, isolated_db):
        from app.core import history as hist
        hist.save("Bob private query", SAMPLE_STATE, username="bob")
        alice_hist = hist.get_history(limit=50, username="alice")
        assert not any(r["username"] == "bob" for r in alice_hist)

    def test_admin_none_sees_all(self, isolated_db):
        from app.core import history as hist
        hist.save("Query X", SAMPLE_STATE, username="user1")
        hist.save("Query Y", SAMPLE_STATE, username="user2")
        all_hist = hist.get_history(limit=50, username=None)
        users = {r["username"] for r in all_hist}
        assert "user1" in users
        assert "user2" in users

    def test_clear_deletes_only_own_records(self, isolated_db):
        from app.core import history as hist
        hist.save("Keep this", SAMPLE_STATE, username="keeper")
        hist.save("Delete this", SAMPLE_STATE, username="cleaner")
        hist.clear(username="cleaner")
        assert hist.get_history(username="keeper")
        assert not hist.get_history(username="cleaner")

    def test_clear_all_deletes_everything(self, isolated_db):
        from app.core import history as hist
        hist.save("Q1", SAMPLE_STATE, username="u1")
        hist.save("Q2", SAMPLE_STATE, username="u2")
        hist.clear(username=None)
        assert hist.get_history(limit=100, username=None) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Save & retrieve
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveAndRetrieve:

    def test_save_returns_row_id(self, isolated_db):
        from app.core import history as hist
        row_id = hist.save("test query", SAMPLE_STATE, username="test")
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_saved_fields_are_correct(self, isolated_db):
        from app.core import history as hist
        hist.save("Tesla drop", SAMPLE_STATE, username="testuser")
        records = hist.get_history(username="testuser")
        assert records
        r = records[0]
        assert r["query"] == "Tesla drop"
        assert r["sentiment"] == "Bullish"
        assert r["username"] == "testuser"
        assert r["token_total"] == 150
        assert r["latency_s"] == 1.5

    def test_tickers_stored_as_list(self, isolated_db):
        from app.core import history as hist
        hist.save("Apple", SAMPLE_STATE, username="u")
        r = hist.get_history(username="u")[0]
        assert isinstance(r["tickers"], list)
        assert "AAPL" in r["tickers"]

    def test_rag_sources_stored_as_list(self, isolated_db):
        from app.core import history as hist
        hist.save("Apple", SAMPLE_STATE, username="u2")
        r = hist.get_history(username="u2")[0]
        assert isinstance(r["rag_sources"], list)


# ═══════════════════════════════════════════════════════════════════════════════
# Feedback
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeedback:

    def test_update_rating(self, isolated_db):
        from app.core import history as hist
        rid = hist.save("query", SAMPLE_STATE, username="rater")
        ok = hist.update_feedback(rid, rating=5, username="rater")
        assert ok
        r = hist.get_history(username="rater")[0]
        assert r["full_state"].get("user_rating") is None  # stored in column, not full_state
        # Re-fetch via get_feedback_all
        fb = hist.get_feedback_all()
        assert any(row["id"] == rid and row["user_rating"] == 5 for row in fb)

    def test_update_feedback_text(self, isolated_db):
        from app.core import history as hist
        rid = hist.save("query", SAMPLE_STATE, username="commenter")
        hist.update_feedback(rid, feedback_text="Great analysis!", username="commenter")
        fb = hist.get_feedback_all()
        row = next((r for r in fb if r["id"] == rid), None)
        assert row and row["feedback_text"] == "Great analysis!"

    def test_wrong_owner_cannot_update_feedback(self, isolated_db):
        from app.core import history as hist
        rid = hist.save("private", SAMPLE_STATE, username="owner")
        ok = hist.update_feedback(rid, rating=1, username="attacker")
        assert not ok

    def test_rating_out_of_range_raises(self, isolated_db):
        from app.core import history as hist
        rid = hist.save("q", SAMPLE_STATE, username="u")
        with pytest.raises(ValueError):
            hist.update_feedback(rid, rating=6, username="u")

    def test_no_fields_returns_false(self, isolated_db):
        from app.core import history as hist
        rid = hist.save("q", SAMPLE_STATE, username="u2")
        assert hist.update_feedback(rid, username="u2") is False


# ═══════════════════════════════════════════════════════════════════════════════
# Token usage
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenUsage:

    def test_user_token_sum(self, isolated_db):
        from app.core import history as hist
        state_a = {**SAMPLE_STATE, "token_usage": {"total": 200}}
        state_b = {**SAMPLE_STATE, "token_usage": {"total": 350}}
        hist.save("Q1", state_a, username="token_user")
        hist.save("Q2", state_b, username="token_user")
        assert hist.get_user_tokens("token_user") == 550

    def test_other_user_tokens_excluded(self, isolated_db):
        from app.core import history as hist
        state = {**SAMPLE_STATE, "token_usage": {"total": 999}}
        hist.save("Q", state, username="other_token_user")
        assert hist.get_user_tokens("my_user") == 0

    def test_global_token_sum(self, isolated_db):
        from app.core import history as hist
        state = {**SAMPLE_STATE, "token_usage": {"total": 100}}
        hist.save("Q1", state, username="gu1")
        hist.save("Q2", state, username="gu2")
        assert hist.get_user_tokens(None) >= 200


# ═══════════════════════════════════════════════════════════════════════════════
# PII redaction on save
# ═══════════════════════════════════════════════════════════════════════════════

class TestPIIOnSave:

    def test_article_text_pii_redacted_when_no_consent(self, isolated_db):
        from app.core import history as hist
        pii_state = {
            **SAMPLE_STATE,
            "consent": False,
            "articles": [{"text": "Contact me at user@example.com for more info.", "url": ""}],
        }
        rid = hist.save("pii query", pii_state, username="pii_user")
        record = hist.get_history(username="pii_user")[0]
        articles = record["full_state"].get("articles", [])
        assert articles
        assert "user@example.com" not in articles[0].get("text", "")
        assert "[REDACTED_EMAIL]" in articles[0].get("text", "")

    def test_system_fields_not_redacted(self, isolated_db):
        from app.core import history as hist
        rid = hist.save("sys fields", SAMPLE_STATE, username="sys_user")
        record = hist.get_history(username="sys_user")[0]
        fs = record["full_state"]
        assert "AAPL" in str(fs.get("tickers", []))
        assert "market:AAPL" in str(fs.get("rag_sources", []))


# ═══════════════════════════════════════════════════════════════════════════════
# Retention purge
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetentionPurge:

    def test_purge_old_records(self, isolated_db):
        from app.core import history as hist
        import sqlite3
        rid = hist.save("old query", SAMPLE_STATE, username="olduser")
        # Backdate the record
        db_path = isolated_db
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE conversations SET created_at = '2020-01-01T00:00:00+00:00' WHERE id = ?",
            (rid,),
        )
        conn.commit()
        conn.close()
        deleted = hist.purge_older_than(days=30)
        assert deleted >= 1

    def test_purge_zero_days_does_nothing(self, isolated_db):
        from app.core import history as hist
        hist.save("keep this", SAMPLE_STATE, username="keepuser")
        deleted = hist.purge_older_than(days=0)
        assert deleted == 0
        assert len(hist.get_history(username="keepuser")) == 1
