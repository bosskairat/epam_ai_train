"""
app/core/security.py
---------------------
Input validation, sanitisation, and prompt-injection defence.
"""

import re
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Injection patterns to reject outright ────────────────────────────────────
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(everything|all)",
    r"you\s+are\s+now\s+(?:a|an)\s+\w+",
    r"act\s+as\s+(?:a|an)\s+\w+",
    r"jailbreak",
    r"dan\s+mode",
    r"do\s+anything\s+now",
    r"override\s+(?:your\s+)?(?:safety|rules|instructions)",
    r"system\s*:\s*you\s+are",
    r"</?(system|user|assistant)>",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

MAX_QUERY_LENGTH = 500  # characters


class ValidationError(ValueError):
    """Raised when a user query fails validation."""


def validate_query(query: str) -> str:
    """
    Validate and sanitise a user query.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Reject empty or too-long inputs.
    3. Scan for prompt-injection patterns.
    4. Remove control characters.

    Returns the cleaned query string or raises ValidationError.
    """
    if not isinstance(query, str):
        raise ValidationError("Query must be a string.")

    query = query.strip()

    if not query:
        raise ValidationError("Query must not be empty.")

    if len(query) > MAX_QUERY_LENGTH:
        raise ValidationError(
            f"Query is too long ({len(query)} chars). Maximum is {MAX_QUERY_LENGTH}."
        )

    # Check for prompt injection
    for pattern in _COMPILED:
        if pattern.search(query):
            logger.warning(f"Potential prompt injection detected: '{query[:80]}…'")
            raise ValidationError(
                "Query contains disallowed patterns. Please ask a financial question."
            )

    # Strip non-printable control chars (keep newlines for readability)
    query = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", query)

    logger.debug(f"Query validated: '{query[:80]}'")
    return query
