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
                store.upsert([text], source_tag=f"market:{ticker}")
        news_texts = [a.get("text", "") for a in articles if a.get("text")]
        if news_texts:
            store.upsert(news_texts, source_tag="news")
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

    # Graceful degradation without API key
    if not settings.OPENAI_API_KEY:
        logger.warning("[AnalysisAgent] No OPENAI_API_KEY – returning stub analysis")
        return {
            "analysis": _stub_analysis(query, market_data, articles),
            "rag_sources": ["pipeline"],
            "token_usage": {"prompt": prompt_tokens, "completion": 0, "total": prompt_tokens},
            "latency_s": 0.0,
        }

    t0 = time.perf_counter()
    client = _get_client()
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

    raw_text = response.choices[0].message.content or "{}"
    logger.debug(f"[AnalysisAgent] Raw LLM output:\n{raw_text}")

    try:
        analysis = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("[AnalysisAgent] Failed to parse LLM JSON – returning raw text")
        analysis = {"raw": raw_text, "parse_error": True}

    usage = response.usage
    token_info = {
        "prompt":     usage.prompt_tokens if usage else prompt_tokens,
        "completion": usage.completion_tokens if usage else 0,
        "total":      usage.total_tokens if usage else prompt_tokens,
    }

    logger.info(f"[AnalysisAgent] Done — latency={latency:.2f}s | tokens={token_info}")
    return {
        "analysis":    analysis,
        "rag_sources": rag_sources,
        "token_usage": token_info,
        "latency_s":   round(latency, 3),
    }


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
