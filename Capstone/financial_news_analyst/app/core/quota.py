"""
app/core/quota.py
------------------
Redis-backed simple rate limiter and token quota tracker.

Provides:
- `check_rate_limit(key, limit_per_minute)` — enforces a per-minute quota.
- `increment_tokens(key, tokens)` — track tokens consumed by key.
- `get_usage(key)` — return tracked token usage.

If `REDIS_URL` is not configured, falls back to an in-memory best-effort store.
"""

from __future__ import annotations

import time
from typing import Optional
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_in_memory_rl = {}
_in_memory_tokens = {}


def _get_redis() -> Optional[object]:
    try:
        if not settings.REDIS_URL:
            return None
        import redis

        return redis.from_url(settings.REDIS_URL)
    except Exception as exc:
        logger.warning(f"Redis unavailable for quotas: {exc}")
        return None


def check_rate_limit(key: str, limit_per_minute: int) -> bool:
    """Return True if under limit, False if rate-limited."""
    if not key:
        key = "anon"
    r = _get_redis()
    cur_min = int(time.time() // 60)
    rl_key = f"rl:{key}:{cur_min}"
    try:
        if r:
            count = r.incr(rl_key)
            if count == 1:
                r.expire(rl_key, 65)
            return count <= limit_per_minute
        else:
            val = _in_memory_rl.get(rl_key, 0) + 1
            _in_memory_rl[rl_key] = val
            return val <= limit_per_minute
    except Exception as exc:
        logger.warning(f"Rate limit check error: {exc}")
        return True


def increment_tokens(key: str, tokens: int) -> None:
    if not key:
        key = "anon"
    r = _get_redis()
    try:
        if r:
            r.incrby(f"tokens:{key}", tokens)
        else:
            _in_memory_tokens[key] = _in_memory_tokens.get(key, 0) + tokens
    except Exception as exc:
        logger.warning(f"Failed to increment tokens: {exc}")


def get_usage(key: str) -> dict:
    r = _get_redis()
    try:
        if r:
            tokens = int(r.get(f"tokens:{key}") or 0)
        else:
            tokens = _in_memory_tokens.get(key, 0)
        return {"tokens": tokens}
    except Exception as exc:
        logger.warning(f"Failed to fetch usage: {exc}")
        return {"tokens": 0}


def get_all_usage() -> dict:
    """Return a mapping of keys -> token usage. Best-effort; may be approximate."""
    r = _get_redis()
    try:
        if r:
            out = {}
            for k in r.scan_iter(match="tokens:*"):
                try:
                    raw = r.get(k)
                    if not raw:
                        continue
                    # k is bytes or str like 'tokens:thekey'
                    if isinstance(k, bytes):
                        k_str = k.decode()
                    else:
                        k_str = k
                    key_name = k_str.split("tokens:", 1)[-1]
                    out[key_name] = int(raw)
                except Exception:
                    continue
            return out
        else:
            return dict(_in_memory_tokens)
    except Exception as exc:
        logger.warning(f"Failed to fetch all usage: {exc}")
        return {}
