"""
app/core/logger.py
------------------
Structured logging setup used across the entire application.
Tracks latency, token usage, and agent decisions.
"""

import logging
import time
import functools
from typing import Any, Callable
from app.core.config import settings
from app.core.pii import redact_pii
from app.observability.tracing import get_request_id


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s %(trace_id)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        # Redact common PII from log messages before they are emitted.
        class TraceIdFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    record.trace_id = get_request_id() or "-"
                except Exception:
                    record.trace_id = "-"
                return True

        class RedactionFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    # Use conservative redaction to avoid leaking secrets in logs.
                    redacted = redact_pii(record.getMessage())
                    record.msg = redacted
                    record.args = ()
                except Exception:
                    # Never fail logging due to redaction errors.
                    pass
                return True

        handler.addFilter(TraceIdFilter())
        handler.addFilter(RedactionFilter())
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return logger


def log_latency(logger: logging.Logger):
    """Decorator: logs execution time of any async or sync function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
            return result

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def estimate_tokens(text: str) -> int:
    """Rough token estimator: ~4 chars per token (GPT convention)."""
    return max(1, len(text) // 4)
