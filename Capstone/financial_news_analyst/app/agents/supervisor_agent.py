"""
app/agents/supervisor_agent.py
--------------------------------
Supervisor Agent — pure orchestration, no data-fetching logic.

Runs the three agents sequentially:
  data_agent  → fetches market data via market MCP server
  news_agent  → fetches news via news MCP server
  analysis_agent → ingests data into RAG, retrieves context, calls OpenAI

LangGraph is replaced by simple async sequential Python. The public API
(run_pipeline) is unchanged from the caller's perspective.
"""

from __future__ import annotations
import time
from app.agents import data_agent, news_agent, analysis_agent
from app.core.logger import get_logger

logger = get_logger(__name__)


async def run_pipeline(query: str) -> dict:
    """
    Execute the full multi-agent pipeline for a user query.

    Returns a dict matching the original AgentState schema so that
    routes.py and tests need minimal changes.
    """
    logger.info(f"[Supervisor] Pipeline started for: '{query[:80]}'")
    t_start = time.perf_counter()
    agent_log: list[str] = []

    # ── Step 1: Data Agent ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    data_result = await data_agent.run(query)
    elapsed = round(time.perf_counter() - t0, 3)
    agent_log.append(
        f"data_agent: fetched {len(data_result['tickers'])} tickers in {elapsed}s"
    )
    logger.info(f"[Supervisor] data_agent done ({elapsed}s)")

    # ── Step 2: News Agent ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    news_result = await news_agent.run(query, tickers=data_result["tickers"])
    elapsed = round(time.perf_counter() - t0, 3)
    agent_log.append(
        f"news_agent: fetched {len(news_result['articles'])} articles in {elapsed}s"
    )
    logger.info(f"[Supervisor] news_agent done ({elapsed}s)")

    # ── Step 3: Analysis Agent ────────────────────────────────────────────────
    t0 = time.perf_counter()
    analysis_result = await analysis_agent.run(
        query,
        data_result["market_data"],
        news_result["articles"],
    )
    elapsed = round(time.perf_counter() - t0, 3)
    agent_log.append(
        f"analysis_agent: generated analysis in {elapsed}s | "
        f"tokens={analysis_result['token_usage'].get('total', '?')} | "
        f"sentiment={analysis_result['analysis'].get('sentiment', '?')}"
    )
    logger.info(f"[Supervisor] analysis_agent done ({elapsed}s)")

    total_latency = round(time.perf_counter() - t_start, 3)
    logger.info(
        f"[Supervisor] Pipeline complete — total={total_latency}s | trace={agent_log}"
    )

    return {
        "query":          query,
        "tickers":        data_result["tickers"],
        "market_data":    data_result["market_data"],
        "articles":       news_result["articles"],
        "analysis":       analysis_result["analysis"],
        "rag_sources":    analysis_result["rag_sources"],
        "token_usage":    analysis_result["token_usage"],
        "total_latency_s": total_latency,
        "agent_log":      agent_log,
    }
