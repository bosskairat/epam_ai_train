"""
app/rag/verify.py
------------------
Simple RAG-based claim verification and hallucination scoring.

This module provides lightweight utilities to parse short claims from
analysis text, re-query the vector store for supporting documents, and
compute per-claim support scores and an overall hallucination metric.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import List, Dict

from app.rag.vector_store import get_vector_store
from app.core.logger import get_logger

logger = get_logger(__name__)


def _split_into_claims(text: str) -> List[str]:
    if not text:
        return []
    # Split by sentence terminators, keep short heuristic filters
    parts = re.split(r"(?<=[\.\?!])\s+", text.strip())
    claims = [p.strip().rstrip(".!") for p in parts if p.strip()]
    # Filter out very short fragments
    claims = [c for c in claims if len(c) > 20]
    return claims


def verify_text(text: str, k: int = 3, support_threshold: float = 0.6) -> Dict:
    """Verify claims found in `text` using the vector store.

    Returns a dict with per-claim support and an overall hallucination score
    where 0.0 = fully supported and 1.0 = fully hallucinated.
    """
    store = get_vector_store()
    claims = _split_into_claims(text)
    if not claims:
        return {"claims": [], "overall_support": 0.0, "hallucination_score": 1.0}

    results = []
    max_supports = []
    for claim in claims:
        docs = store.query(claim, k=k)
        # docs are expected to include 'score' (higher is better) and 'payload'
        supports = []
        for d in docs:
            score = d.get("score") or 0.0
            supports.append({
                "id": d.get("id"),
                "score": score,
                "snippet": (d.get("payload", {}).get("text", ""))[:300],
            })

        max_score = max((s["score"] for s in supports), default=0.0)
        supported = max_score >= support_threshold
        max_supports.append(max_score)
        results.append(
            {
                "claim": claim,
                "supported": supported,
                "max_support": round(max_score, 4),
                "supporting_docs": supports,
            }
        )

    overall_support = mean(max_supports) if max_supports else 0.0
    hallucination_score = round(1.0 - overall_support, 4)

    logger.info(
        f"RAG verify: {len(results)} claims, overall_support={overall_support:.3f}, hallucination_score={hallucination_score}"
    )

    return {
        "claims": results,
        "overall_support": round(overall_support, 4),
        "hallucination_score": hallucination_score,
    }


def verify_texts(texts: List[str], **kwargs) -> Dict:
    """Verify multiple text blocks and aggregate results."""
    aggregate_claims = []
    supports = []
    for t in texts:
        v = verify_text(t, **kwargs)
        aggregate_claims.extend(v.get("claims", []))
        supports.append(v.get("overall_support", 0.0))

    overall = mean(supports) if supports else 0.0
    return {
        "claims": aggregate_claims,
        "overall_support": round(overall, 4),
        "hallucination_score": round(1.0 - overall, 4),
    }
