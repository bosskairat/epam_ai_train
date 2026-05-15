"""
app/api/app.py
---------------
FastAPI application factory with middleware, CORS, and rate limiting.
"""

import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.api.limits import limiter
from app.api.routes import router
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial News Analyst",
        description="Multi-Agent GenAI system for financial research",
        version="1.0.0",
    )

    # ── Rate limiting ─────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── API metrics middleware ────────────────────────────────────────────────
    @app.middleware("http")
    async def api_metrics_middleware(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - t0
        try:
            from app.observability.metrics import observe_api_request, observe_api_error
            path = request.url.path
            observe_api_request(request.method, path, response.status_code, latency)
            if response.status_code >= 500:
                observe_api_error(path, "server_error")
            elif response.status_code == 429:
                observe_api_error(path, "rate_limited")
        except Exception:
            pass
        return response

    # Apply existing slowapi rate limit decorator to the analyze endpoint
    rate = f"{settings.RATE_LIMIT_PER_MINUTE}/minute"
    limiter.limit(rate)(router.routes[2].endpoint)  # /analyze is index 2

    # Add Redis-backed rate limiting middleware when REDIS_URL is configured
    if settings.REDIS_URL:
        from app.core.quota import check_rate_limit
        from fastapi.responses import JSONResponse

        @app.middleware("http")
        async def redis_rate_limit(request: Request, call_next):
            key = request.headers.get("X-API-Key") or (request.client.host if request.client else "anon")
            allowed = True
            try:
                allowed = check_rate_limit(key, settings.RATE_LIMIT_PER_MINUTE)
            except Exception:
                allowed = True
            if not allowed:
                return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
            return await call_next(request)

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    # ── Observability: metrics endpoint and trace id middleware -------------
    try:
        # Mount Prometheus ASGI app if prometheus_client is installed.
        from prometheus_client import make_asgi_app
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
    except Exception:
        logger.warning("prometheus_client not available; /metrics not mounted")

    @app.middleware("http")
    async def add_trace_id(request: Request, call_next):
        try:
            from app.observability.tracing import make_trace_id, set_request_id
            trace_id = request.headers.get("X-Trace-Id") or make_trace_id()
            set_request_id(trace_id)
        except Exception:
            trace_id = None
        response = await call_next(request)
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    logger.info("FastAPI app created")
    # Startup tasks: resource monitoring and retention purge
    @app.on_event("startup")
    async def _startup_tasks():
        try:
            from app.observability.resource import start_resource_monitoring

            try:
                await start_resource_monitoring()
                logger.info("Resource monitoring started")
            except Exception as exc:
                logger.warning(f"Failed to start resource monitor: {exc}")
        except Exception:
            logger.debug("Resource monitoring not configured")

        # retention purge loop (runs once a day)
        try:
            import asyncio

            from app.core import history as hist

            async def _retention_loop():
                while True:
                    try:
                        deleted = hist.purge_older_than(settings.HISTORY_RETENTION_DAYS)
                        logger.info(f"Retention purge removed {deleted} rows")
                    except Exception as exc:
                        logger.warning(f"Retention purge failed: {exc}")
                    await asyncio.sleep(24 * 3600)

            asyncio.create_task(_retention_loop())
        except Exception:
            logger.debug("Retention purge not scheduled")

    return app
