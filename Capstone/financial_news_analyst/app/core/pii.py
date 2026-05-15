"""
app/core/pii.py
----------------
Small helper utilities for simple PII detection and redaction.
This is intentionally conservative and regex-based for initial rollout.
"""

import re
from typing import Any

# Common PII-like patterns (simple heuristics)
_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_PHONE = re.compile(
    r"\b(?:\+?1[-.\s]?)?"          # optional US/CA country code
    r"\(?[2-9]\d{2}\)?[-.\s]"      # area code: exactly 3 digits, first digit 2-9
    r"[2-9]\d{2}[-.\s]?\d{4}\b"    # exchange + subscriber: 3+4 digits
)
_API_KEY = re.compile(r"\b(?:sk|pk|ak|secret|api)[-_]?[A-Za-z0-9]{16,}\b", re.IGNORECASE)


def redact_pii(text: str) -> str:
    """Return a copy of `text` with common PII patterns replaced by placeholders.

    This function uses conservative regexes to avoid over-redaction where possible.
    """
    if not isinstance(text, str) or not text:
        return text

    out = _EMAIL.sub("[REDACTED_EMAIL]", text)
    out = _SSN.sub("[REDACTED_SSN]", out)
    out = _CC.sub("[REDACTED_CC]", out)
    out = _API_KEY.sub("[REDACTED_KEY]", out)
    out = _PHONE.sub("[REDACTED_PHONE]", out)

    return out


def _redact_recursive(obj: Any) -> Any:
    """Recursively redact strings inside dicts/lists/tuples.

    Non-string primitives (int/float/bool/None) are returned unchanged.
    """
    if isinstance(obj, str):
        return redact_pii(obj)
    if isinstance(obj, dict):
        return {k: _redact_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_recursive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_redact_recursive(v) for v in obj)
    return obj


def redact_state(state: Any) -> Any:
    """Convenience wrapper to redact an application state object.

    Use before persisting user-provided data when consent is not given.
    """
    return _redact_recursive(state)
