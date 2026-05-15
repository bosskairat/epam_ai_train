"""
app/core/security.py
---------------------
Input validation, sanitisation, and prompt-injection defence.
"""

import re
from app.core.logger import get_logger
from app.core.pii import redact_pii, redact_state
from app.core.config import settings

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

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


def sanitize_for_prompt(text: str) -> str:
    """Sanitise arbitrary text before including it in an LLM prompt.

    This performs control-character removal, whitespace collapsing and
    conservative PII redaction. Designed for downstream prompt construction.
    """
    if not isinstance(text, str):
        return text

    t = text.strip()
    # Strip non-printable control chars (keep newlines normalized)
    t = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", t)
    # Collapse excessive whitespace
    t = re.sub(r"\s+", " ", t)
    # Conservative PII redaction
    t = redact_pii(t)

    logger.debug(f"Sanitized text for prompt (len={len(t)}): '{t[:80]}…'")
    return t


def sanitize_document(text: str) -> str:
    """Sanitise documents before upsert: remove role/system markers and
    apply prompt sanitisation rules."""
    if not isinstance(text, str):
        return text
    # Remove common role prefixes and XML-like tags
    t = re.sub(r"(?im)^(system|user|assistant)\s*:\s*", "", text)
    t = re.sub(r"</?(system|user|assistant)>", "", t)
    # Collapse excessive whitespace and redact PII
    t = sanitize_for_prompt(t)
    return t


def moderate_text(text: str) -> dict:
    """Moderation wrapper. Returns {'severity': 'allow'|'flag'|'block', 'categories': {...}}.

    If `OPENAI_API_KEY` is configured, attempts to use OpenAI moderation endpoint,
    otherwise falls back to simple heuristics.
    """
    if not text:
        return {"severity": "allow", "categories": {}}

    # Try remote moderation if available
    if settings.OPENAI_API_KEY and OpenAI is not None:
        try:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            # Use the moderation endpoint if available; fall back safely.
            resp = getattr(client, "moderations", None)
            if resp is not None:
                result = client.moderations.create(model="omni-moderation-latest", input=text)
                # Interpret a conservative heuristic: any category flagged -> 'flag'
                categories = getattr(result, "results", [{}])[0].get("categories", {})
                flagged = any(categories.values())
                severity = "flag" if flagged else "allow"
                return {"severity": severity, "categories": categories}
        except Exception:
            # Fall through to local heuristics
            pass

    # Local heuristic checks
    lowered = text.lower()
    categories = {}
    # Block obvious secrets / PII exfiltration patterns
    if re.search(r"\b(password|ssn|social security|credit card|ccv|cvv)\b", lowered):
        return {"severity": "block", "categories": {"pii": True}}

    # Block violent threats
    if re.search(r"\b(kill|bomb|attack|terroris)\b", lowered):
        return {"severity": "block", "categories": {"violence": True}}

    # Flag potentially risky content (adult, policy-edge)
    if re.search(r"\b(sex|porn|escort)\b", lowered):
        return {"severity": "flag", "categories": {"sexual": True}}

    return {"severity": "allow", "categories": {}}
