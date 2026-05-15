"""
app/agents/analysis_agent.py
------------------------------
Analysis Agent — LLM-powered reasoning via OpenAI.

Ingests market data and news articles directly into the Qdrant in-memory
vector store, retrieves relevant context, then calls gpt-4o-mini to
synthesise a structured JSON insight report.

⚠ DISCLAIMER: Output is for educational/informational purposes only.
  It is NOT financial advice.
"""

from __future__ import annotations
import json
import time

from openai import OpenAI
from app.core.config import settings
from app.core.logger import get_logger, estimate_tokens
from app.observability.tracing import llm_span, get_request_id
from app.observability.metrics import observe_llm_call
from app.core.cache import make_cache_key, get_cached_response, set_cached_response
from app.core.quota import get_usage
from app.rag.verify import verify_texts
from app.rag.vector_store import get_vector_store
from app.rag.retriever import retrieve_context

logger = get_logger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


SYSTEM_PROMPT = """You are a financial research assistant.
Your role is to synthesise market data, news headlines, and historical context
into clear, structured investment research summaries.

Rules:
- Be factual, balanced, and concise.
- For sources_used, extract the full URL from each "URL: ..." line in the context. Only include actual URLs (starting with http). Never use labels like "news", "market", or "pipeline".
- NEVER give specific buy/sell recommendations or guarantee returns.
- End every summary with: "This is not financial advice."
- Respond ONLY in valid JSON matching the schema below.

Response schema:
{
  "summary":       "<5-7 sentence overview>",
  "sentiment":     "<Bullish | Bearish | Neutral | Mixed>",
  "key_drivers":   ["<driver 1>", "<driver 2>", ...],
  "risk_factors":  ["<risk 1>", "<risk 2>", ...],
  "insight":       "<1 paragraph educational insight>",
  "sources_used":  ["<https://...>", ...],
  "disclaimer":    "This is not financial advice."
}
"""


def _build_user_prompt(
    query: str,
    market_data: dict,
    articles: list[dict],
    context_text: str,
) -> str:
    market_block = (
        "\n\n".join(
            f"[{ticker}]\n{text}" for ticker, text in market_data.items() if text
        )
        or "No market data available."
    )

    news_block = (
        "\n\n".join(a.get("text", "") for a in articles[:5] if a.get("text"))
        or "No recent news found."
    )

    return f"""User Question: {query}

--- LIVE MARKET DATA ---
{market_block}

--- RECENT NEWS ---
{news_block}

--- HISTORICAL CONTEXT (RAG) ---
{context_text}

Based on ALL the above information, generate your structured JSON analysis.
Remember to cite the sources provided."""


async def run(
    query: str,
    market_data: dict,
    articles: list[dict],
    consent: bool = False,
    api_key: str | None = None,
) -> dict:
    """
    Entry point called by the Supervisor Agent.

    Returns:
        {
          "analysis":    {...},
          "rag_sources": [...],
          "token_usage": {...},
          "latency_s":   float,
        }
    """
    logger.info(f"[AnalysisAgent] Running for query: '{query[:80]}'")

    # Ingest market data and news into the shared vector store
    store = get_vector_store()
    try:
        for ticker, text in market_data.items():
            if text and not text.startswith("[Market data unavailable"):
                store.upsert([text], source_tag=f"market:{ticker}", consent=consent)
        news_texts = [a.get("text", "") for a in articles if a.get("text")]
        if news_texts:
            store.upsert(news_texts, source_tag="news", consent=consent)
    except Exception as exc:
        logger.warning(f"[AnalysisAgent] RAG upsert failed: {exc}")

    # Retrieve relevant context from the vector store
    try:
        ctx = retrieve_context(query)
        context_text = ctx.get("context_text", "No historical context available.")
        rag_sources = ctx.get("sources", ["pipeline"])
    except Exception as exc:
        logger.warning(f"[AnalysisAgent] RAG retrieval failed: {exc}")
        context_text = "No historical context available."
        rag_sources = []

    user_prompt = _build_user_prompt(query, market_data, articles, context_text)
    prompt_tokens = estimate_tokens(SYSTEM_PROMPT + user_prompt)
    logger.info(f"[AnalysisAgent] Estimated prompt tokens: ~{prompt_tokens}")

    # Check token quota (best-effort) if configured
    try:
        if api_key and settings.TOKEN_QUOTA_PER_KEY and settings.TOKEN_QUOTA_PER_KEY > 0:
            usage = get_usage(api_key)
            if usage.get("tokens", 0) >= settings.TOKEN_QUOTA_PER_KEY:
                logger.warning(f"[AnalysisAgent] Token quota exceeded for key={api_key}")
                raise Exception("Token quota exceeded")
    except Exception:
        # swallow and continue — API route typically enforces quota earlier
        pass

    # Attempt to serve from cache
    try:
        cache_key = make_cache_key(settings.LLM_MODEL, SYSTEM_PROMPT, user_prompt)
        cached = get_cached_response(cache_key)
        if cached:
            logger.info("[AnalysisAgent] Returning cached analysis result")
            # Annotate cached result
            cached.setdefault("cached", True)
            return cached
    except Exception:
        pass

    # Graceful degradation without API key
    if not settings.OPENAI_API_KEY:
        logger.warning("[AnalysisAgent] No OPENAI_API_KEY – returning stub analysis")
        return {
            "analysis": _stub_analysis(query, market_data, articles),
            "rag_sources": ["pipeline"],
            "token_usage": {"prompt": prompt_tokens, "completion": 0, "total": prompt_tokens},
            "latency_s": 0.0,
        }

    client = _get_client()
    latency = None
    usage = None
    # Wrap the LLM call in a span to capture timing and metrics
    with llm_span("analysis_agent.llm_call") as span:
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency = time.perf_counter() - t0
        span["latency_s"] = latency
        # Attach request trace id when available
        span["trace_id"] = get_request_id()
        try:
            usage = response.usage
            total_tokens = usage.total_tokens if usage else prompt_tokens
            span["tokens"] = total_tokens
            try:
                observe_llm_call(settings.LLM_MODEL, latency, total_tokens)
            except Exception:
                pass
        except Exception:
            usage = None

    raw_text = response.choices[0].message.content or "{}"
    logger.debug(f"[AnalysisAgent] Raw LLM output:\n{raw_text}")

    # Moderate LLM output when content moderation is enabled
    if settings.ENABLE_CONTENT_MODERATION:
        try:
            from app.core.security import moderate_text
            mod = moderate_text(raw_text)
            if mod.get("severity") == "block":
                logger.warning("[AnalysisAgent] LLM output blocked by content moderation")
                return {
                    "analysis": _stub_analysis(query, {}, []),
                    "rag_sources": rag_sources,
                    "token_usage": token_info if "token_info" in dir() else {"prompt": prompt_tokens, "completion": 0, "total": prompt_tokens},
                    "latency_s": round(latency, 3),
                    "trace_id": get_request_id(),
                    "verification": {"claims": [], "overall_support": 0.0, "hallucination_score": 1.0},
                    "moderation_blocked": True,
                }
        except Exception as exc:
            logger.warning(f"[AnalysisAgent] Output moderation check failed: {exc}")

    try:
        analysis = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("[AnalysisAgent] Failed to parse LLM JSON – returning raw text")
        analysis = {"raw": raw_text, "parse_error": True}

    token_info = {
        "prompt":     usage.prompt_tokens if usage else prompt_tokens,
        "completion": usage.completion_tokens if usage else 0,
        "total":      usage.total_tokens if usage else prompt_tokens,
    }

    logger.info(f"[AnalysisAgent] Done — latency={latency:.2f}s | tokens={token_info} | trace={get_request_id()}")
    # Run a lightweight RAG-based verification for summary/insight if present
    try:
        texts_to_check = []
        if isinstance(analysis, dict):
            if analysis.get("summary"):
                texts_to_check.append(analysis.get("summary"))
            if analysis.get("insight"):
                texts_to_check.append(analysis.get("insight"))
        if texts_to_check:
            verification = verify_texts(texts_to_check)
        else:
            verification = {"claims": [], "overall_support": 0.0, "hallucination_score": 1.0}
    except Exception as exc:
        logger.warning(f"[AnalysisAgent] Verification failed: {exc}")
        verification = {"claims": [], "overall_support": 0.0, "hallucination_score": 1.0}
    result = {
        "analysis":    analysis,
        "rag_sources": rag_sources,
        "token_usage": token_info,
        "latency_s":   round(latency, 3),
        "trace_id":    get_request_id(),
        "verification": verification,
    }
    # Best-effort: cache the result for subsequent identical prompts
    try:
        cache_key = make_cache_key(settings.LLM_MODEL, SYSTEM_PROMPT, user_prompt)
        set_cached_response(cache_key, result, ttl=settings.CACHE_TTL)
    except Exception:
        pass
    return result


def _stub_analysis(query: str, market_data: dict, articles: list[dict]) -> dict:
    tickers = list(market_data.keys())
    return {
        "summary": (
            f"Analysis for '{query}'. "
            f"Live data retrieved for: {', '.join(tickers) or 'N/A'}. "
            f"{len(articles)} news articles processed. "
            "(OpenAI key not set — this is a stub response.)"
        ),
        "sentiment": "Neutral",
        "key_drivers": ["Market data successfully fetched via MCP", "News articles retrieved via MCP"],
        "risk_factors": ["OpenAI API key not configured", "Stub mode active"],
        "insight": (
            "Set OPENAI_API_KEY in your .env file to enable full LLM-powered analysis. "
            "The MCP data pipeline (market + news) is fully operational."
        ),
        "sources_used": ["pipeline"],
        "disclaimer": "This is not financial advice.",
    }
