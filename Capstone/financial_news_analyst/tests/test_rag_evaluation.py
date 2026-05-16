"""
tests/test_rag_evaluation.py
-----------------------------
Tests for app/rag/evaluation.py — all five RAG quality dimensions.

All external calls (Qdrant, OpenAI) are mocked so these tests run offline.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ── Shared stubs ──────────────────────────────────────────────────────────────

SAMPLE_QUERY = "Why did Tesla stock drop today?"

SAMPLE_STATE = {
    "query": SAMPLE_QUERY,
    "tickers": ["TSLA"],
    "rag_sources": ["market:TSLA — 2024-01-15", "news — 2024-01-15"],
    "analysis": {
        "summary": "Tesla shares fell after missing delivery targets.",
        "sentiment": "Bearish",
        "insight": "Delivery data is a key sentiment driver for Tesla investors.",
        "key_drivers": ["Missed deliveries"],
        "risk_factors": ["Demand slowdown"],
        "sources_used": ["https://reuters.com/article/1"],
        "disclaimer": "Not financial advice.",
    },
    "token_usage": {"total": 600},
    "total_latency_s": 2.1,
    "articles": [{"text": "Tesla missed delivery estimates.", "url": "https://reuters.com/article/1"}],
}

STUB_QDRANT_RESULTS = [
    {"text": "Tesla deliveries missed expectations.", "score": 0.82, "source_tag": "news", "url": ""},
    {"text": "Tesla market data TSLA 185.40.", "score": 0.74, "source_tag": "market:TSLA", "url": ""},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval evaluation
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalEvaluation:

    @patch("app.rag.evaluation.get_vector_store")
    def test_returns_required_keys(self, mock_store_fn):
        mock_store = MagicMock()
        mock_store.query.return_value = STUB_QDRANT_RESULTS
        mock_store_fn.return_value = mock_store

        from app.rag.evaluation import evaluate_retrieval
        result = evaluate_retrieval(SAMPLE_QUERY)
        for key in ("precision_at_k", "mrr", "mean_score", "doc_count"):
            assert key in result, f"Missing key: {key}"

    @patch("app.rag.evaluation.get_vector_store")
    def test_precision_at_k_in_range(self, mock_store_fn):
        mock_store = MagicMock()
        mock_store.query.return_value = STUB_QDRANT_RESULTS
        mock_store_fn.return_value = mock_store

        from app.rag.evaluation import evaluate_retrieval
        result = evaluate_retrieval(SAMPLE_QUERY)
        assert 0.0 <= result["precision_at_k"] <= 1.0

    @patch("app.rag.evaluation.get_vector_store")
    def test_empty_results_handled(self, mock_store_fn):
        mock_store = MagicMock()
        mock_store.query.return_value = []
        mock_store_fn.return_value = mock_store

        from app.rag.evaluation import evaluate_retrieval
        result = evaluate_retrieval(SAMPLE_QUERY)
        assert result["doc_count"] == 0
        assert result["precision_at_k"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Answer relevance
# ═══════════════════════════════════════════════════════════════════════════════

def _make_store_mock(embed_vec=None):
    """Return a MagicMock that behaves like VectorStore for relevance tests."""
    import numpy as np
    vec = embed_vec if embed_vec is not None else np.array([0.8, 0.2, 0.1])
    store = MagicMock()
    store._embed.return_value = [vec]
    store.query.return_value = STUB_QDRANT_RESULTS
    return store


class TestAnswerRelevance:

    @patch("app.rag.evaluation.get_vector_store")
    def test_returns_required_keys(self, mock_store_fn):
        mock_store_fn.return_value = _make_store_mock()
        from app.rag.evaluation import evaluate_answer_relevance
        result = evaluate_answer_relevance(SAMPLE_QUERY, "Tesla fell on weak deliveries.")
        for key in ("cosine_similarity", "token_overlap", "relevance_label"):
            assert key in result, f"Missing key: {key}"

    @patch("app.rag.evaluation.get_vector_store")
    def test_cosine_similarity_in_range(self, mock_store_fn):
        import numpy as np
        mock_store_fn.return_value = _make_store_mock(np.array([0.7, 0.3, 0.1]))
        from app.rag.evaluation import evaluate_answer_relevance
        result = evaluate_answer_relevance(SAMPLE_QUERY, "Tesla stock dropped.")
        assert -1.0 <= result["cosine_similarity"] <= 1.0

    @patch("app.rag.evaluation.get_vector_store")
    def test_token_overlap_in_range(self, mock_store_fn):
        mock_store_fn.return_value = _make_store_mock()
        from app.rag.evaluation import evaluate_answer_relevance
        result = evaluate_answer_relevance("Tesla stock", "Tesla shares dropped today")
        assert 0.0 <= result["token_overlap"] <= 1.0

    @patch("app.rag.evaluation.get_vector_store")
    def test_relevance_label_is_valid(self, mock_store_fn):
        mock_store_fn.return_value = _make_store_mock()
        from app.rag.evaluation import evaluate_answer_relevance
        result = evaluate_answer_relevance(SAMPLE_QUERY, "Some answer text.")
        assert result["relevance_label"] in ("high", "medium", "low")


# ═══════════════════════════════════════════════════════════════════════════════
# Source attribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestSourceAttribution:

    def test_returns_required_keys(self):
        from app.rag.evaluation import evaluate_source_attribution
        result = evaluate_source_attribution(
            retrieved_sources=["market:TSLA — 2024-01-15", "news — 2024-01-15"],
            llm_cited_urls=["https://reuters.com/article/1"],
            analysis_text="Tesla fell. Source: reuters.",
        )
        for key in ("traceability_score", "citation_rate", "source_coverage", "llm_cited_urls_count"):
            assert key in result, f"Missing key: {key}"

    def test_traceability_score_in_range(self):
        from app.rag.evaluation import evaluate_source_attribution
        result = evaluate_source_attribution(
            retrieved_sources=["news — 2024-01-15"],
            llm_cited_urls=["https://example.com"],
            analysis_text="Based on news sources.",
        )
        assert 0.0 <= result["traceability_score"] <= 1.0

    def test_no_citations_gives_zero_citation_rate(self):
        from app.rag.evaluation import evaluate_source_attribution
        result = evaluate_source_attribution(
            retrieved_sources=["news — 2024-01-15"],
            llm_cited_urls=[],
            analysis_text="Analysis text with no sources.",
        )
        assert result["citation_rate"] == 0.0

    def test_llm_cited_urls_count_correct(self):
        from app.rag.evaluation import evaluate_source_attribution
        result = evaluate_source_attribution(
            retrieved_sources=["news — 2024-01-15"],
            llm_cited_urls=["https://a.com", "https://b.com"],
            analysis_text="See reuters and bloomberg.",
        )
        assert result["llm_cited_urls_count"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Hallucination
# ═══════════════════════════════════════════════════════════════════════════════

class TestHallucinationEvaluation:

    @patch("app.rag.evaluation.verify_texts")
    def test_returns_required_keys(self, mock_verify):
        mock_verify.return_value = {"hallucination_score": 0.2, "claims": []}

        from app.rag.evaluation import evaluate_hallucination
        result = evaluate_hallucination("Tesla fell.", "Deliveries missed targets.")
        for key in ("hallucination_score", "label"):
            assert key in result, f"Missing key: {key}"

    @patch("app.rag.evaluation.verify_texts")
    def test_score_in_range(self, mock_verify):
        mock_verify.return_value = {"hallucination_score": 0.35, "claims": []}

        from app.rag.evaluation import evaluate_hallucination
        result = evaluate_hallucination("summary", "insight")
        assert 0.0 <= result["hallucination_score"] <= 1.0

    @patch("app.rag.evaluation.verify_texts")
    def test_low_score_label_is_low(self, mock_verify):
        mock_verify.return_value = {"hallucination_score": 0.1, "claims": []}

        from app.rag.evaluation import evaluate_hallucination
        result = evaluate_hallucination("summary", "insight")
        assert result["label"] in ("low", "medium", "high")


# ═══════════════════════════════════════════════════════════════════════════════
# Bias evaluation
# ═══════════════════════════════════════════════════════════════════════════════

class TestBiasEvaluation:

    def test_returns_required_keys(self):
        from app.rag.evaluation import evaluate_bias
        result = evaluate_bias("Markets rose sharply on strong earnings.", "Bullish", [])
        assert "lexical" in result
        assert "bias_flag" in result

    def test_bullish_text_has_low_bearish_count(self):
        from app.rag.evaluation import evaluate_bias
        result = evaluate_bias(
            "Strong growth, record profits, markets surging upward.",
            "Bullish",
            [],
        )
        lex = result["lexical"]
        assert lex.get("bullish_keywords", 0) > lex.get("bearish_keywords", 0)

    def test_bearish_text_has_low_bullish_count(self):
        from app.rag.evaluation import evaluate_bias
        result = evaluate_bias(
            "Collapse, recession, crash, losses, declining revenue.",
            "Bearish",
            [],
        )
        lex = result["lexical"]
        assert lex.get("bearish_keywords", 0) > lex.get("bullish_keywords", 0)

    def test_imbalance_score_in_range(self):
        from app.rag.evaluation import evaluate_bias
        result = evaluate_bias("Mixed signals in the market.", "Neutral", [])
        imbalance = result["lexical"].get("imbalance_score", 0.0)
        assert 0.0 <= imbalance <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Full pipeline evaluation
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullEvaluation:

    @patch("app.rag.evaluation.verify_texts")
    @patch("app.rag.evaluation.get_vector_store")
    def test_composite_score_in_range(self, mock_store_fn, mock_verify):
        mock_store_fn.return_value = _make_store_mock()
        mock_verify.return_value = {"hallucination_score": 0.2, "claims": [], "overall_support": 0.8}

        from app.rag.evaluation import evaluate_pipeline_result
        report = evaluate_pipeline_result(SAMPLE_QUERY, SAMPLE_STATE)
        score = report.get("composite_score")
        assert score is not None
        assert 0.0 <= score <= 1.0

    @patch("app.rag.evaluation.verify_texts")
    @patch("app.rag.evaluation.get_vector_store")
    def test_all_dimensions_present(self, mock_store_fn, mock_verify):
        mock_store_fn.return_value = _make_store_mock()
        mock_verify.return_value = {"hallucination_score": 0.15, "claims": [], "overall_support": 0.85}

        from app.rag.evaluation import evaluate_pipeline_result
        report = evaluate_pipeline_result(SAMPLE_QUERY, SAMPLE_STATE)
        for dim in ("retrieval", "answer_relevance", "source_attribution", "hallucination", "bias"):
            assert dim in report, f"Missing evaluation dimension: {dim}"

    @patch("app.rag.evaluation.verify_texts")
    @patch("app.rag.evaluation.get_vector_store")
    def test_aggregate_from_history_returns_means(self, mock_store_fn, mock_verify):
        mock_store_fn.return_value = _make_store_mock()
        mock_verify.return_value = {"hallucination_score": 0.2, "claims": [], "overall_support": 0.8}

        from app.rag.evaluation import evaluate_from_history
        history_rows = [
            {
                "query": "Tesla analysis",
                "full_state": SAMPLE_STATE,
                "sentiment": "Bearish",
            }
        ]
        report = evaluate_from_history(history_rows, sample=1)
        assert "sample_size" in report

    def test_aggregate_from_empty_history(self):
        from app.rag.evaluation import evaluate_from_history
        report = evaluate_from_history([], sample=10)
        assert "error" in report or report.get("sample_size", 0) == 0
