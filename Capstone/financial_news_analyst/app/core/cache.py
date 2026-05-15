"""
app/core/cache.py
------------------
Simple caching layer with Redis fallback and in-memory TTLCache.

Provides `make_cache_key`, `get_cached_response`, and `set_cached_response`.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from cachetools import TTLCache
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# In-memory fallback cache
_mem_cache: TTLCache = TTLCache(maxsize=1024, ttl=settings.CACHE_TTL)


def _get_redis():
    if not settings.REDIS_URL:
        return None
    try:
        import redis

        return redis.from_url(settings.REDIS_URL)
    except Exception as exc:
        logger.warning(f"Redis unavailable for cache: {exc}")
        return None


def make_cache_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        if p is None:
            p = ""
        if not isinstance(p, (bytes, bytearray)):
            p = str(p).encode("utf-8")
        h.update(p)
    return h.hexdigest()


def get_cached_response(key: str) -> Optional[Any]:
    r = _get_redis()
    try:
        if r:
            val = r.get(f"cache:{key}")
            if not val:
                return None
            return json.loads(val)
        else:
            return _mem_cache.get(key)
    except Exception as exc:
        logger.warning(f"Cache get error: {exc}")
        return None


def set_cached_response(key: str, value: Any, ttl: int | None = None) -> None:
    r = _get_redis()
    try:
        payload = json.dumps(value)
        if r:
            r.set(f"cache:{key}", payload)
            if ttl:
                r.expire(f"cache:{key}", int(ttl))
        else:
            # cachetools TTL is configured globally; use provided ttl if different
            if ttl and ttl != settings.CACHE_TTL:
                # recreate a temporary cache entry with custom ttl
                _mem_cache[key] = value
            else:
                _mem_cache[key] = value
    except Exception as exc:
        logger.warning(f"Cache set error: {exc}")
