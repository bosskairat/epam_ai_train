"""
app/observability/tracing.py
----------------------------
Simple request trace id context management and helpers for LLM spans.

This module provides a ContextVar-based `trace_id` that is set by
FastAPI middleware and can be read anywhere in the code to attach
request identifiers to logs and traces.
"""

from __future__ import annotations

import uuid
import contextvars
from contextlib import contextmanager
from time import perf_counter
from typing import Generator

# Context variable holding the current trace id (string) for the active context
_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)


def make_trace_id() -> str:
    """Generate a new compact trace id."""
    return uuid.uuid4().hex


def set_request_id(trace_id: str) -> None:
    _trace_id.set(trace_id)


def get_request_id() -> str | None:
    return _trace_id.get()


@contextmanager
def trace_context(trace_id: str | None = None) -> Generator[str, None, None]:
    """Context manager that sets a trace id for the enclosed block.

    If `trace_id` is omitted a new id is generated. Useful for background
    tasks or synthetic traces.
    """
    if trace_id is None:
        trace_id = make_trace_id()
    token = _trace_id.set(trace_id)
    try:
        yield trace_id
    finally:
        _trace_id.reset(token)


@contextmanager
def llm_span(name: str = "llm_call"):
    """Simple span-like context manager to time LLM interactions.

    Yields a dict with `span_id`, `name`, and `start_ts`. The caller may
    augment this dict with latency/tokens and export it to metrics or a DB.
    """
    span = {
        "span_id": uuid.uuid4().hex,
        "name": name,
        "start_ts": perf_counter(),
    }
    try:
        yield span
    finally:
        span["end_ts"] = perf_counter()
