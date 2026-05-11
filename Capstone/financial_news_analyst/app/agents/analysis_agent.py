"""
app/agents/analysis_agent.py
------------------------------
Analysis Agent — the main LLM-powered reasoning agent.

Combines:
  • Live market data (from Data Agent)
  • Recent news articles (from News Agent)
  • Historical context retrieved via RAG

Generates a structured insight report using GPT-4o-mini.

⚠ DISCLAIMER: Output is for educational/informational purposes only.
  It is NOT financial advice.
"""

from __future__ import annotations
import json
import time
from datetime import datetime
from openai import OpenAI
from app.rag.retriever import retrieve_context
from app.core.config import settings
from app.core.logger import get_logger, estimate_tokens, log_latency

logger = get_logger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial research assistant.
Your role is to synthesise market data, news headlines, and historical context
into clear, structured investment research summaries.

Rules:
- Be factual, balanced, and concise.
- Always cite the sources provided in the context.
- NEVER give specific buy/sell recommendations or guarantee returns.
- End every summary with: "⚠ This is not financial advice."
- Respond ONLY in valid JSON matching the schema below.

Response schema:
{
  "summary":       "<2-3 sentence overview>",
  "sentiment":     "<Bullish | Bearish | Neutral | Mixed>",
  "key_drivers":   ["<driver 1>", "<driver 2>", ...],
  "risk_factors":  ["<risk 1>", "<risk 2>", ...],
  "insight":       "<1 paragraph educational insight>",
  "sources_used":  ["<source 1>", ...],
  "disclaimer":    "This is not financial advice."
}
"""


def _build_user_prompt(
    query: str,
    market_data: dict,
    articles: list[dict],
    rag_context: dict,
) -> str:
    """Assemble the full user-turn prompt from all data sources."""

    # ── Market data block ────────────────────────────────────────────────────
    market_block = "No market data available."
    if market_data:
        lines = []
        for ticker, data in market_data.items():
            if data.get("error"):
                lines.append(f"• {ticker}: data unavailable ({data['error']})")
            else:
                lines.append(
                    f"• {data.get('company_name', ticker)} ({ticker}): "
                    f"${data.get('current_price')} "
                    f"({data.get('change_pct', 0):+.2f}%) | "
                    f"Vol: {data.get('volume', 'N/A')}"
                )
        market_block = "\n".join(lines)

    # ── News block ───────────────────────────────────────────────────────────
    news_block = "No recent news found."
    if articles:
        news_lines = []
        for a in articles[:5]:  # top 5 to stay within token budget
            news_lines.append(
                f"• [{a.get('source', '?')}] {a.get('title', '')} "
                f"({a.get('published_at', '')[:10]})"
            )
        news_block = "\n".join(news_lines)

    prompt = f"""User Question: {query}

--- LIVE MARKET DATA ---
{market_block}

--- RECENT NEWS ---
{news_block}

--- HISTORICAL CONTEXT (RAG, {rag_context['doc_count']} docs retrieved) ---
{rag_context['context_text']}

Based on ALL the above information, generate your structured JSON analysis.
Remember to cite the sources provided."""

    return prompt


@log_latency(logger)
def run(
    query: str,
    market_data: dict,
    articles: list[dict],
) -> dict:
    """
    Entry point called by the Supervisor Agent.

    Args:
        query:       Original user question.
        market_data: Output from Data Agent.
        articles:    Output from News Agent.

    Returns:
        {
          "analysis":      {...},   # parsed JSON from LLM
          "rag_sources":   [...],
          "token_usage":   {...},
          "latency_s":     float,
        }
    """
    logger.info(f"[AnalysisAgent] Running for query: '{query[:80]}'")

    # ── RAG Retrieval ─────────────────────────────────────────────────────────
    rag_context = retrieve_context(query)
    logger.info(f"[AnalysisAgent] RAG retrieved {rag_context['doc_count']} docs")

    # ── Build prompt ──────────────────────────────────────────────────────────
    user_prompt = _build_user_prompt(query, market_data, articles, rag_context)

    # Log prompt token estimate
    prompt_tokens = estimate_tokens(SYSTEM_PROMPT + user_prompt)
    logger.info(f"[AnalysisAgent] Estimated prompt tokens: ~{prompt_tokens}")
    logger.debug(f"[AnalysisAgent] Full prompt:\n{user_prompt}")

    # ── LLM Call ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    if not settings.OPENAI_API_KEY:
        # Graceful degradation: return a stub response without a real API key
        logger.warning("[AnalysisAgent] No OPENAI_API_KEY – returning stub analysis")
        analysis = _stub_analysis(query, market_data, articles, rag_context)
        latency = time.perf_counter() - t0
        return {
            "analysis": analysis,
            "rag_sources": rag_context["sources"],
            "token_usage": {"prompt": prompt_tokens, "completion": 0, "total": prompt_tokens},
            "latency_s": round(latency, 3),
        }

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

    # ── Parse response ────────────────────────────────────────────────────────
    raw_text = response.choices[0].message.content or "{}"
    logger.debug(f"[AnalysisAgent] Raw LLM output:\n{raw_text}")

    try:
        analysis = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("[AnalysisAgent] Failed to parse LLM JSON – returning raw text")
        analysis = {"raw": raw_text, "parse_error": True}

    usage = response.usage
    token_info = {
        "prompt": usage.prompt_tokens if usage else prompt_tokens,
        "completion": usage.completion_tokens if usage else 0,
        "total": usage.total_tokens if usage else prompt_tokens,
    }

    logger.info(
        f"[AnalysisAgent] Done — latency={latency:.2f}s | tokens={token_info}"
    )

    return {
        "analysis": analysis,
        "rag_sources": rag_context["sources"],
        "token_usage": token_info,
        "latency_s": round(latency, 3),
    }


# ── Stub for offline / no-key mode ────────────────────────────────────────────

def _stub_analysis(query, market_data, articles, rag_context) -> dict:
    """Return a plausible stub when no OpenAI key is configured."""
    tickers = list(market_data.keys())
    return {
        "summary": (
            f"Analysis for '{query}'. "
            f"Live data retrieved for: {', '.join(tickers) or 'N/A'}. "
            f"{len(articles)} news articles and {rag_context['doc_count']} RAG docs processed. "
            "(OpenAI key not set — this is a stub response.)"
        ),
        "sentiment": "Neutral",
        "key_drivers": ["Market data successfully fetched", "News articles retrieved"],
        "risk_factors": ["OpenAI API key not configured", "Stub mode active"],
        "insight": (
            "Set OPENAI_API_KEY in your .env file to enable full LLM-powered analysis. "
            "The data pipeline (yfinance + news + RAG) is fully operational."
        ),
        "sources_used": rag_context["sources"],
        "disclaimer": "This is not financial advice.",
    }
