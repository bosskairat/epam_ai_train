"""
app/api/routes.py
------------------
FastAPI router — exposes the multi-agent pipeline via REST.

Endpoints:
  POST   /analyze                  → run the full pipeline and save to history
  GET    /health                   → liveness check
  GET    /rag/stats                → vector store document count
  GET    /history                  → list past conversations
  DELETE /history                  → clear all conversation history
  PATCH  /history/{id}/feedback    → attach rating/feedback to a conversation
  GET    /usage                    → token usage for the current user
"""

from __future__ import annotations
import time
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field, field_validator
from app.agents.supervisor_agent import run_pipeline
from app.core.quota import get_usage
from app.core.security import validate_query, ValidationError
from app.rag.vector_store import get_vector_store
from app.core import history as hist
from app.core import users_store as usr
from app.api import auth_routes
from app.api.deps import require_auth_principal, require_admin
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)
router = APIRouter()

# Initialise DB tables on module load.
hist.init_db()
usr.init_db()
usr.bootstrap_from_env()
router.include_router(auth_routes.router)


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    query: str
    consent: bool = False

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
    hallucination_score: float | None = None
    trace_id: str | None = None


class FeedbackBody(BaseModel):
    rating: int | None = Field(None, ge=1, le=5)
    feedback_text: str | None = Field(None, max_length=2000)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@router.get("/rag/stats")
def rag_stats():
    store = get_vector_store()
    return {
        "document_count": store.count(),
        "top_k": settings.TOP_K_RESULTS,
        "min_similarity": settings.RAG_MIN_SIMILARITY,
        "embedding_model": settings.EMBEDDING_MODEL,
        "qdrant_path": settings.QDRANT_PATH,
    }


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    http_request: Request,
    request: AnalyzeRequest,
    principal: dict | None = Depends(require_auth_principal),
):
    """Run the full multi-agent pipeline and persist result to history."""
    logger.info(f"[API] POST /analyze query='{request.query[:80]}' | consent={request.consent}")

    # Quota key: authenticated username or client IP
    key = (principal or {}).get("sub") or (
        http_request.client.host if http_request.client else "anon"
    )

    if settings.TOKEN_QUOTA_PER_KEY and settings.TOKEN_QUOTA_PER_KEY > 0:
        try:
            usage = get_usage(key)
            if usage.get("tokens", 0) >= settings.TOKEN_QUOTA_PER_KEY:
                raise HTTPException(status_code=429, detail="Token quota exceeded.")
        except HTTPException:
            raise
        except Exception:
            pass

    try:
        state = await run_pipeline(request.query, consent=request.consent, api_key=key)
    except Exception as exc:
        logger.error(f"[API] Pipeline error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    history_id: int | None = None
    try:
        state["consent"] = bool(request.consent)
        history_id = hist.save(request.query, state, username=key)
        from app.core.quota import increment_tokens
        try:
            increment_tokens(key, state.get("token_usage", {}).get("total", 0))
        except Exception:
            pass
    except Exception as exc:
        logger.warning(f"[API] Failed to save history: {exc}")

    verification = state.get("verification", {})
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
        hallucination_score=verification.get("hallucination_score"),
        trace_id=state.get("trace_id"),
    )


@router.get("/history")
def get_history(
    limit: int = 50,
    principal: dict | None = Depends(require_auth_principal),
):
    try:
        username = (principal or {}).get("sub")
        records = hist.get_history(limit=limit, username=username)
        return {"conversations": records, "count": len(records)}
    except Exception as exc:
        logger.error(f"[API] History fetch error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/history/{history_id}/feedback")
def patch_feedback(
    history_id: int,
    body: FeedbackBody,
    principal: dict | None = Depends(require_auth_principal),
):
    """Attach user rating (1–5) and/or free-text feedback to a saved conversation."""
    if body.rating is None and body.feedback_text is None:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: rating, feedback_text",
        )
    try:
        username = (principal or {}).get("sub")
        ok = hist.update_feedback(
            history_id,
            rating=body.rating,
            feedback_text=body.feedback_text,
            username=username,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation id not found.")
    return {"ok": True, "id": history_id}


@router.delete("/history")
def clear_history(principal: dict | None = Depends(require_auth_principal)):
    try:
        username = (principal or {}).get("sub")
        deleted = hist.clear(username=username)
        return {"deleted": deleted}
    except Exception as exc:
        logger.error(f"[API] History clear error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Usage (own tokens only) ───────────────────────────────────────────────────

@router.get("/usage")
def get_usage_endpoint(
    principal: dict | None = Depends(require_auth_principal),
):
    """Return token usage for the current user, calculated from persistent history."""
    try:
        username = (principal or {}).get("sub")
        tokens = hist.get_user_tokens(username)
        return {"username": username or "anon", "tokens_used": tokens}
    except Exception as exc:
        logger.error(f"[API] Usage error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── RAG Evaluation ────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    query: str
    state: dict


@router.post("/rag/evaluate")
def rag_evaluate(
    body: EvalRequest,
    principal: dict = Depends(require_auth_principal),
):
    """
    Run all five RAG quality evaluations on a completed pipeline result.
    Returns retrieval accuracy, answer relevance, source attribution,
    hallucination score, and bias assessment.
    """
    try:
        from app.rag.evaluation import evaluate_pipeline_result
        username = (principal or {}).get("sub")
        history = hist.get_history(limit=100, username=None)   # all users for bias dist
        report = evaluate_pipeline_result(body.query, body.state, history_rows=history)
        return report
    except Exception as exc:
        logger.error(f"[API] RAG evaluate error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/evaluate/history")
def rag_evaluate_history(
    sample: int = 20,
    _: dict = Depends(require_admin),
):
    """Admin: aggregate RAG quality metrics over recent history."""
    try:
        from app.rag.evaluation import evaluate_from_history
        rows = hist.get_history(limit=200, username=None)
        return evaluate_from_history(rows, sample=sample)
    except Exception as exc:
        logger.error(f"[API] RAG eval history error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Admin: user management ────────────────────────────────────────────────────

class ResetPasswordBody(BaseModel):
    new_password: str = Field(..., min_length=8, max_length=256)


@router.get("/admin/feedback")
def admin_get_feedback(
    limit: int = 200,
    _: dict = Depends(require_admin),
):
    """Return all conversations that have a user rating or feedback text."""
    try:
        records = hist.get_feedback_all(limit=limit)
        return {"feedback": records, "count": len(records)}
    except Exception as exc:
        logger.error(f"[API] Admin feedback error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/admin/users")
def admin_list_users(
    _: dict = Depends(require_admin),
):
    """List all registered users with their token usage (from persistent history)."""
    users = usr.list_users()
    result = []
    for u in users:
        tokens = hist.get_user_tokens(u["username"])
        result.append({
            "id": u["id"],
            "username": u["username"],
            "created_at": u["created_at"],
            "is_blocked": bool(u["is_blocked"]),
            "tokens_used": tokens,
        })
    return {"users": result, "total": len(result)}


@router.patch("/admin/users/{username}/block")
def admin_block_user(
    username: str,
    principal: dict = Depends(require_admin),
):
    """Block a user account."""
    if username == principal["sub"]:
        raise HTTPException(status_code=400, detail="Cannot block your own account.")
    if not usr.block_user(username):
        raise HTTPException(status_code=404, detail="User not found.")
    logger.info(f"[Admin] {principal['sub']} blocked user {username!r}")
    return {"ok": True, "username": username, "is_blocked": True}


@router.patch("/admin/users/{username}/unblock")
def admin_unblock_user(
    username: str,
    _: dict = Depends(require_admin),
):
    """Unblock a user account."""
    if not usr.unblock_user(username):
        raise HTTPException(status_code=404, detail="User not found.")
    return {"ok": True, "username": username, "is_blocked": False}


@router.delete("/admin/users/{username}")
def admin_delete_user(
    username: str,
    principal: dict = Depends(require_admin),
):
    """Permanently delete a user account."""
    if username == principal["sub"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account.")
    if not usr.delete_user(username):
        raise HTTPException(status_code=404, detail="User not found.")
    logger.info(f"[Admin] {principal['sub']} deleted user {username!r}")
    return {"ok": True, "username": username}


@router.post("/admin/users/{username}/reset-password")
def admin_reset_password(
    username: str,
    body: ResetPasswordBody,
    _: dict = Depends(require_admin),
):
    """Reset a user's password."""
    try:
        if not usr.reset_password(username, body.new_password):
            raise HTTPException(status_code=404, detail="User not found.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, "username": username}
