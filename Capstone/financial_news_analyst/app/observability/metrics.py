"""
app/observability/metrics.py
----------------------------
Prometheus metrics helpers and lightweight wrappers used by agents.

This module exposes a small set of metrics and helper functions to
record LLM, embedding, and retrieval events. It intentionally avoids
high-cardinality labels.
"""

from prometheus_client import Counter, Histogram
from prometheus_client import Gauge

# LLM metrics
llm_requests_total = Counter("llm_requests_total", "Total LLM calls", ["model"])
llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM call latency in seconds",
    ["model"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
llm_tokens_total = Counter("llm_tokens_total", "Tokens consumed by LLM calls", ["model"])

# Embedding & retrieval metrics
embedding_requests_total = Counter("embedding_requests_total", "Embedding requests total")
retrieval_requests_total = Counter("retrieval_requests_total", "Retrieval requests total")

# API-level metrics
api_requests_total = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
api_errors_total = Counter(
    "api_errors_total", "Total API errors", ["endpoint", "error_type"]
)
api_latency_seconds = Histogram(
    "api_latency_seconds",
    "End-to-end API request latency",
    ["endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)
hallucination_score = Gauge(
    "hallucination_score_last", "Hallucination score of the last analysis (0=supported, 1=hallucinated)"
)


# System / process metrics
system_cpu_percent = Gauge("system_cpu_percent", "System CPU percent")
system_memory_percent = Gauge("system_memory_percent", "System memory percent")
process_cpu_percent = Gauge("process_cpu_percent", "Process CPU percent")
process_memory_mb = Gauge("process_memory_mb", "Process memory RSS in MB")


def observe_api_request(method: str, endpoint: str, status: int, latency_s: float) -> None:
    api_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    api_latency_seconds.labels(endpoint=endpoint).observe(latency_s)


def observe_api_error(endpoint: str, error_type: str) -> None:
    api_errors_total.labels(endpoint=endpoint, error_type=error_type).inc()


def observe_llm_call(model: str | None, latency_s: float, tokens: int | None = None) -> None:
    model_label = model or "unknown"
    llm_requests_total.labels(model=model_label).inc()
    llm_latency_seconds.labels(model=model_label).observe(latency_s)
    if tokens:
        llm_tokens_total.labels(model=model_label).inc(tokens)


def observe_embedding() -> None:
    embedding_requests_total.inc()


def observe_retrieval() -> None:
    retrieval_requests_total.inc()
