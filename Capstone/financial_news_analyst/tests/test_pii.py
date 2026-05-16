"""
tests/test_pii.py
------------------
Unit tests for PII detection and redaction (app/core/pii.py).

Positive cases: patterns that MUST be redacted.
Negative cases: strings that must NOT be redacted (false-positive regression).
"""

from __future__ import annotations

import pytest
from app.core.pii import redact_pii


# ═══════════════════════════════════════════════════════════════════════════════
# Email
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmailRedaction:

    def test_simple_email_redacted(self):
        assert "[REDACTED_EMAIL]" in redact_pii("Contact me at user@example.com please")

    def test_subdomain_email_redacted(self):
        assert "[REDACTED_EMAIL]" in redact_pii("Send to alice@mail.company.org")

    def test_email_not_in_output(self):
        result = redact_pii("My email is bob@test.io")
        assert "@test.io" not in result

    def test_no_false_positive_on_ticker(self):
        result = redact_pii("Buy AAPL and TSLA today")
        assert "[REDACTED" not in result

    def test_no_false_positive_on_url(self):
        result = redact_pii("See https://reuters.com/article/123 for details")
        assert "[REDACTED_EMAIL]" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# SSN
# ═══════════════════════════════════════════════════════════════════════════════

class TestSSNRedaction:

    def test_ssn_redacted(self):
        assert "[REDACTED_SSN]" in redact_pii("My SSN is 123-45-6789")

    def test_ssn_not_in_output(self):
        result = redact_pii("SSN: 987-65-4321 is private")
        assert "987-65-4321" not in result

    def test_no_false_positive_on_date(self):
        result = redact_pii("Published on 2024-01-15")
        assert "[REDACTED_SSN]" not in result

    def test_no_false_positive_on_zip(self):
        result = redact_pii("ZIP code 10001")
        assert "[REDACTED_SSN]" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# API keys
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIKeyRedaction:

    def test_sk_prefix_key_redacted(self):
        assert "[REDACTED_KEY]" in redact_pii("My key is sk-abc123def456ghi789jkl")

    def test_api_prefix_key_redacted(self):
        assert "[REDACTED_KEY]" in redact_pii("api-key: api_ABCDEFGHIJKLMNOP1234567890")

    def test_short_token_not_redacted(self):
        result = redact_pii("Error code sk-123")
        assert "sk-123" in result  # too short to be a real API key


# ═══════════════════════════════════════════════════════════════════════════════
# Phone numbers (strict US format — must NOT match dates or source labels)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhoneRedaction:

    def test_us_phone_redacted(self):
        result = redact_pii("Call me at (212) 555-1234 anytime")
        assert "[REDACTED_PHONE]" in result

    def test_dashed_phone_redacted(self):
        result = redact_pii("Phone: 800-555-9876")
        assert "[REDACTED_PHONE]" in result

    # ── Critical false-positive regressions ──────────────────────────────────

    def test_date_not_redacted_yyyy_mm_dd(self):
        result = redact_pii("market:TSLA — 2024-01-15")
        assert "[REDACTED_PHONE]" not in result

    def test_date_in_parens_not_redacted(self):
        result = redact_pii("(2026-05-15)")
        assert "[REDACTED_PHONE]" not in result

    def test_source_label_not_redacted(self):
        result = redact_pii("news — 2024-01-15")
        assert "[REDACTED_PHONE]" not in result

    def test_pipeline_label_not_redacted(self):
        result = redact_pii("pipeline — 2024-01-15")
        assert "[REDACTED_PHONE]" not in result

    def test_version_number_not_redacted(self):
        result = redact_pii("version 1.2.3456")
        assert "[REDACTED_PHONE]" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_string_unchanged(self):
        assert redact_pii("") == ""

    def test_none_returns_none(self):
        assert redact_pii(None) is None

    def test_non_pii_text_unchanged(self):
        text = "Tesla stock fell 3% after weak delivery numbers."
        assert redact_pii(text) == text

    def test_multiple_pii_types_all_redacted(self):
        text = "Email user@x.com SSN 111-22-3333 key sk-abcdefghijklmnopqrst"
        result = redact_pii(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_SSN]" in result
        assert "[REDACTED_KEY]" in result
        assert "user@x.com" not in result
        assert "111-22-3333" not in result

    def test_financial_text_preserved(self):
        text = "AAPL up 2.5% | MSFT market cap $3T | S&P 500 hit record high"
        result = redact_pii(text)
        assert "AAPL" in result
        assert "MSFT" in result
        assert "S&P 500" in result
        assert "[REDACTED" not in result
