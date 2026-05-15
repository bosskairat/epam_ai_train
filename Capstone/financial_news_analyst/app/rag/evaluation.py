"""
app/rag/evaluation.py
----------------------
RAG quality evaluation across all five NFR dimensions:

  1. Retrieval Accuracy  — precision@k, MRR, source diversity
  2. Answer Relevance    — query-answer semantic similarity
  3. Source Attribution  — LLM citation vs retrieved source coverage
  4. Hallucination       — claim-level support scores (via verify.py)
  5. Bias Assessment     — sentiment distribution & keyword imbalance
"""

from __future__ import annotations

import re
import math
from statistics import mean, stdev
from typing import Any
from collections import Counter

from app.rag.vector_store import get_vector_store
from app.rag.verify import verify_texts
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_RELEVANCE_THRESHOLD = 0.35   # cosine score above which a doc is "relevant"
_SUPPORT_THRESHOLD   = 0.50   # min score to call a claim "supported"

_BIAS_BULLISH_KW = {
    "surge", "rally", "soar", "gain", "bullish", "outperform",
    "beat", "record high", "upgrade", "positive", "growth",
}
_BIAS_BEARISH_KW = {
    "plunge", "crash", "drop", "decline", "bearish", "underperform",
    "miss", "record low", "downgrade", "negative", "loss",
}


# ── 1. Retrieval Accuracy ─────────────────────────────────────────────────────

def evaluate_retrieval(query: str, k: int | None = None) -> dict:
    """
    Retrieve top-k docs for a query and compute:
      - scores        : list of cosine similarity scores
      - mean_score    : average retrieval confidence
      - precision_at_k: fraction of docs above _RELEVANCE_THRESHOLD
      - mrr           : Mean Reciprocal Rank (first relevant doc position)
      - source_diversity: number of distinct source_tag prefixes
      - doc_count     : docs actually returned (may be < k when store is small)
    """
    from app.core.config import settings
    k = k or settings.TOP_K_RESULTS
    store = get_vector_store()
    docs = store.query(query, k=k)

    if not docs:
        return {
            "doc_count": 0,
            "scores": [],
            "mean_score": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "source_diversity": 0,
            "relevant_docs": 0,
        }

    scores = [float(d.get("score") or 0.0) for d in docs]
    relevant = [s >= _RELEVANCE_THRESHOLD for s in scores]
    relevant_count = sum(relevant)

    # MRR — reciprocal rank of first relevant hit
    mrr = 0.0
    for rank, is_rel in enumerate(relevant, start=1):
        if is_rel:
            mrr = 1.0 / rank
            break

    # Source diversity: count unique top-level source types (market / news / pipeline)
    tags = set()
    for d in docs:
        tag = d.get("source_tag", "") or ""
        tags.add(tag.split(":")[0] if ":" in tag else tag)

    return {
        "doc_count": len(docs),
        "scores": [round(s, 4) for s in scores],
        "mean_score": round(mean(scores), 4) if scores else 0.0,
        "precision_at_k": round(relevant_count / len(docs), 4),
        "mrr": round(mrr, 4),
        "source_diversity": len(tags),
        "relevant_docs": relevant_count,
    }


# ── 2. Answer Relevance ───────────────────────────────────────────────────────

def evaluate_answer_relevance(query: str, answer_text: str) -> dict:
    """
    Measure how semantically relevant the generated answer is to the original
    query using the same embedding model as the vector store.

    Returns:
      - cosine_similarity : float [0, 1]
      - token_overlap     : ROUGE-1-like unigram overlap (recall)
      - relevance_label   : "high" / "medium" / "low"
    """
    store = get_vector_store()
    try:
        q_vec = store._embed([query])[0]
        a_vec = store._embed([answer_text])[0]
        cosine = _cosine(q_vec, a_vec)
    except Exception as exc:
        logger.warning(f"[Eval] Embedding call failed: {exc}")
        cosine = 0.0

    # Token-level overlap (recall of query tokens in answer)
    q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
    a_tokens = set(re.findall(r"\b\w+\b", answer_text.lower()))
    overlap = len(q_tokens & a_tokens) / len(q_tokens) if q_tokens else 0.0

    label = "high" if cosine >= 0.7 else ("medium" if cosine >= 0.45 else "low")

    return {
        "cosine_similarity": round(cosine, 4),
        "token_overlap": round(overlap, 4),
        "relevance_label": label,
    }


# ── 3. Source Attribution ─────────────────────────────────────────────────────

def evaluate_source_attribution(
    retrieved_sources: list[str],
    llm_cited_urls: list[str],
    analysis_text: str,
) -> dict:
    """
    Measure how well the LLM cites sources that were actually retrieved.

    - citation_rate      : cited URLs / total retrieved source tags
    - url_citation_count : number of valid http URLs in llm_cited_urls
    - source_coverage    : fraction of retrieved source prefixes that appear
                           in the answer text
    - traceability_score : composite [0, 1]
    """
    valid_urls = [u for u in (llm_cited_urls or []) if str(u).startswith("http")]
    url_count  = len(valid_urls)

    # How many of the retrieved source labels appear in the analysis text?
    covered = 0
    for src in retrieved_sources:
        # strip date part: "market:TSLA (2026-05-15)" → "market"
        base = src.split(":")[0].split(" ")[0].lower()
        if base and base in analysis_text.lower():
            covered += 1

    coverage = covered / len(retrieved_sources) if retrieved_sources else 0.0
    citation_rate = url_count / max(len(retrieved_sources), 1)

    # Traceability: weight url citations 60%, source coverage 40%
    traceability = round(min(1.0, citation_rate) * 0.6 + coverage * 0.4, 4)

    return {
        "retrieved_sources_count": len(retrieved_sources),
        "llm_cited_urls_count": url_count,
        "source_coverage": round(coverage, 4),
        "citation_rate": round(citation_rate, 4),
        "traceability_score": traceability,
        "cited_urls": valid_urls[:10],
    }


# ── 4. Hallucination ──────────────────────────────────────────────────────────

def evaluate_hallucination(summary: str, insight: str) -> dict:
    """Thin wrapper around verify.py adding a human-readable label."""
    texts = [t for t in (summary, insight) if t and t.strip()]
    if not texts:
        return {
            "hallucination_score": 1.0,
            "overall_support": 0.0,
            "claims_total": 0,
            "claims_supported": 0,
            "label": "unknown",
        }

    result = verify_texts(texts, k=3, support_threshold=_SUPPORT_THRESHOLD)
    score = result.get("hallucination_score", 1.0)
    claims = result.get("claims", [])
    supported = sum(1 for c in claims if c.get("supported"))

    label = "low" if score < 0.3 else ("medium" if score < 0.6 else "high")

    return {
        "hallucination_score": score,
        "overall_support": result.get("overall_support", 0.0),
        "claims_total": len(claims),
        "claims_supported": supported,
        "label": label,
    }


# ── 5. Bias Assessment ────────────────────────────────────────────────────────

def evaluate_bias(analysis_text: str, sentiment: str, history_rows: list[dict]) -> dict:
    """
    Two complementary bias checks:

    A) Lexical bias in the current analysis:
       - Count bullish vs bearish keyword occurrences
       - Compute imbalance score (0 = balanced, 1 = fully one-sided)

    B) Sentiment distribution over historical conversations:
       - Proportion of Bullish / Bearish / Neutral / Mixed responses
       - Flag if any single sentiment exceeds 70 % of all responses
    """
    words = re.findall(r"\b\w+\b", analysis_text.lower())
    bull_hits = sum(1 for w in words if w in _BIAS_BULLISH_KW)
    bear_hits = sum(1 for w in words if w in _BIAS_BEARISH_KW)
    total_kw  = bull_hits + bear_hits

    if total_kw == 0:
        lexical_imbalance = 0.0
        dominant_direction = "neutral"
    else:
        bull_ratio = bull_hits / total_kw
        bear_ratio = bear_hits / total_kw
        lexical_imbalance = round(abs(bull_ratio - bear_ratio), 4)
        dominant_direction = "bullish" if bull_ratio > bear_ratio else "bearish"

    # Historical sentiment distribution
    sentiments = [r.get("sentiment", "Neutral") for r in history_rows if r.get("sentiment")]
    dist: dict[str, int] = Counter(sentiments)
    total_hist = sum(dist.values()) or 1
    dist_pct = {k: round(v / total_hist, 4) for k, v in dist.items()}

    bias_flag = False
    bias_reason = None
    for sent, pct in dist_pct.items():
        if pct > 0.70:
            bias_flag = True
            bias_reason = f"{sent} sentiment dominates {pct*100:.0f}% of responses"

    return {
        "lexical": {
            "bullish_keywords": bull_hits,
            "bearish_keywords": bear_hits,
            "imbalance_score": lexical_imbalance,
            "dominant_direction": dominant_direction,
        },
        "historical_distribution": dist_pct,
        "history_sample_size": len(sentiments),
        "bias_flag": bias_flag,
        "bias_reason": bias_reason,
        "current_sentiment": sentiment,
    }


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def evaluate_pipeline_result(
    query: str,
    state: dict,
    history_rows: list[dict] | None = None,
) -> dict:
    """
    Run all five evaluations on a completed pipeline result.

    `state` is the dict returned by run_pipeline() / the API response.
    `history_rows` is an optional list of past conversations for bias analysis.

    Returns a structured report under five keys:
        retrieval, answer_relevance, source_attribution, hallucination, bias
    """
    analysis    = state.get("analysis", {})
    summary     = analysis.get("summary", "")
    insight     = analysis.get("insight", "")
    sentiment   = analysis.get("sentiment", "Neutral")
    llm_urls    = analysis.get("sources_used", [])
    rag_sources = state.get("rag_sources", [])

    answer_text = f"{summary} {insight}".strip()

    logger.info(f"[RAGEval] Starting evaluation for query='{query[:60]}'")

    report: dict[str, Any] = {}

    # 1 — Retrieval
    try:
        report["retrieval"] = evaluate_retrieval(query)
    except Exception as exc:
        logger.warning(f"[RAGEval] Retrieval eval failed: {exc}")
        report["retrieval"] = {"error": str(exc)}

    # 2 — Answer relevance
    try:
        report["answer_relevance"] = evaluate_answer_relevance(query, answer_text)
    except Exception as exc:
        logger.warning(f"[RAGEval] Answer relevance eval failed: {exc}")
        report["answer_relevance"] = {"error": str(exc)}

    # 3 — Source attribution
    try:
        report["source_attribution"] = evaluate_source_attribution(
            rag_sources, llm_urls, answer_text
        )
    except Exception as exc:
        logger.warning(f"[RAGEval] Source attribution eval failed: {exc}")
        report["source_attribution"] = {"error": str(exc)}

    # 4 — Hallucination
    try:
        report["hallucination"] = evaluate_hallucination(summary, insight)
    except Exception as exc:
        logger.warning(f"[RAGEval] Hallucination eval failed: {exc}")
        report["hallucination"] = {"error": str(exc)}

    # 5 — Bias
    try:
        report["bias"] = evaluate_bias(answer_text, sentiment, history_rows or [])
    except Exception as exc:
        logger.warning(f"[RAGEval] Bias eval failed: {exc}")
        report["bias"] = {"error": str(exc)}

    # Composite quality score [0, 1] — higher is better
    try:
        retrieval_score   = report["retrieval"].get("precision_at_k", 0.0)
        relevance_score   = report["answer_relevance"].get("cosine_similarity", 0.0)
        attribution_score = report["source_attribution"].get("traceability_score", 0.0)
        grounding_score   = report["hallucination"].get("overall_support", 0.0)
        bias_penalty      = report["bias"]["lexical"].get("imbalance_score", 0.0) * 0.5

        composite = round(
            retrieval_score   * 0.25 +
            relevance_score   * 0.25 +
            attribution_score * 0.20 +
            grounding_score   * 0.30 -
            bias_penalty      * 0.10,
            4,
        )
        report["composite_score"] = max(0.0, min(1.0, composite))
    except Exception:
        report["composite_score"] = None

    logger.info(f"[RAGEval] Done. composite={report.get('composite_score')}")
    return report


# ── Aggregate evaluation over history ─────────────────────────────────────────

def evaluate_from_history(history_rows: list[dict], sample: int = 20) -> dict:
    """
    Compute aggregate RAG evaluation metrics across recent history rows.
    Used by the admin dashboard to show system-level quality trends.

    Only rows that have full_state with analysis data are processed.
    `sample` limits the number of rows to avoid slow embedding calls.
    """
    rows = [r for r in history_rows if r.get("full_state")][:sample]
    if not rows:
        return {"error": "No history rows with full_state available."}

    retrieval_scores: list[float] = []
    relevance_scores: list[float] = []
    attribution_scores: list[float] = []
    hallucination_scores: list[float] = []
    sentiments: list[str] = []

    for row in rows:
        state    = row.get("full_state", {})
        query    = row.get("query", "")
        analysis = state.get("analysis", {}) if isinstance(state, dict) else {}

        if not query or not analysis:
            continue

        sentiments.append(analysis.get("sentiment", "Neutral"))

        try:
            r = evaluate_retrieval(query)
            retrieval_scores.append(r["precision_at_k"])
        except Exception:
            pass

        try:
            summary = analysis.get("summary", "")
            insight = analysis.get("insight", "")
            if summary:
                rel = evaluate_answer_relevance(query, f"{summary} {insight}")
                relevance_scores.append(rel["cosine_similarity"])
        except Exception:
            pass

        try:
            rag_src  = state.get("rag_sources", [])
            llm_urls = analysis.get("sources_used", [])
            attr = evaluate_source_attribution(rag_src, llm_urls, summary)
            attribution_scores.append(attr["traceability_score"])
        except Exception:
            pass

        try:
            h = evaluate_hallucination(
                analysis.get("summary", ""), analysis.get("insight", "")
            )
            hallucination_scores.append(h["hallucination_score"])
        except Exception:
            pass

    def _agg(vals: list[float]) -> dict:
        if not vals:
            return {"mean": None, "min": None, "max": None, "std": None, "n": 0}
        return {
            "mean": round(mean(vals), 4),
            "min":  round(min(vals), 4),
            "max":  round(max(vals), 4),
            "std":  round(stdev(vals), 4) if len(vals) > 1 else 0.0,
            "n":    len(vals),
        }

    sent_dist = Counter(sentiments)
    total = sum(sent_dist.values()) or 1

    return {
        "sample_size": len(rows),
        "retrieval_precision": _agg(retrieval_scores),
        "answer_relevance":    _agg(relevance_scores),
        "source_attribution":  _agg(attribution_scores),
        "hallucination":       _agg(hallucination_scores),
        "sentiment_distribution": {k: round(v / total, 4) for k, v in sent_dist.items()},
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
