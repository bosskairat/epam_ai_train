"""
app/api/routes.py
------------------
FastAPI router — exposes the multi-agent pipeline via REST.

Endpoints:
  POST   /analyze        → run the full pipeline and save to history
  GET    /health         → liveness check
  GET    /rag/stats      → vector store document count
  GET    /history        → list past conversations
  DELETE /history        → clear all conversation history
"""

from __future__ import annotations
import time
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator
from app.agents.supervisor_agent import run_pipeline
from app.core.security import validate_query, ValidationError
from app.rag.vector_store import get_vector_store
from app.core import history as hist
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialise the history DB table when the module loads.
hist.init_db()


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def sanitize(cls, v: str) -> str:
        try:
            return validate_query(v)
        except ValidationError as e:
            raise ValueError(str(e))


class AnalyzeResponse(BaseModel):
    query: str
    analysis: dict
    tickers: list[str]
    articles_count: int
    rag_sources: list[str]
    token_usage: dict
    total_latency_s: float
    agent_log: list[str]
    history_id: int | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    """Simple liveness probe."""
    return {"status": "ok", "timestamp": time.time()}


@router.get("/rag/stats")
def rag_stats():
    """Return the number of documents stored in the vector store."""
    store = get_vector_store()
    return {"document_count": store.count()}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Run the full multi-agent pipeline and persist result to history.
    """
    logger.info(f"[API] POST /analyze query='{request.query[:80]}'")

    try:
        state = await run_pipeline(request.query)
    except Exception as exc:
        logger.error(f"[API] Pipeline error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # Persist to SQLite history
    history_id: int | None = None
    try:
        history_id = hist.save(request.query, state)
    except Exception as exc:
        logger.warning(f"[API] Failed to save history: {exc}")

    return AnalyzeResponse(
        query=request.query,
        analysis=state["analysis"],
        tickers=state["tickers"],
        articles_count=len(state.get("articles", [])),
        rag_sources=state["rag_sources"],
        token_usage=state["token_usage"],
        total_latency_s=state["total_latency_s"],
        agent_log=state["agent_log"],
        history_id=history_id,
    )


@router.get("/history")
def get_history(limit: int = 50):
    """Return the last `limit` conversation records."""
    try:
        records = hist.get_history(limit=limit)
        return {"conversations": records, "count": len(records)}
    except Exception as exc:
        logger.error(f"[API] History fetch error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/history")
def clear_history():
    """Delete all conversation history."""
    try:
        deleted = hist.clear()
        return {"deleted": deleted}
    except Exception as exc:
        logger.error(f"[API] History clear error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
