"""
app/agents/supervisor_agent.py
--------------------------------
Supervisor Agent — the orchestration layer.

Implements a simple LangGraph-style state machine:

  START
    │
    ▼
  data_node   ←─ fetches market data, ingests into vector store
    │
    ▼
  news_node   ←─ fetches news, ingests into vector store
    │
    ▼
  analysis_node ←─ RAG retrieval + LLM synthesis
    │
    ▼
  END

The graph is intentionally linear for clarity. Extend with
conditional edges if you need routing logic.
"""

from __future__ import annotations
import time
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from app.agents import data_agent, news_agent, analysis_agent
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    tickers: list[str]
    market_data: dict
    articles: list[dict]
    analysis: dict
    rag_sources: list[str]
    token_usage: dict
    total_latency_s: float
    agent_log: list[str]   # trace of agent decisions for observability


# ── Node functions ────────────────────────────────────────────────────────────

def data_node(state: AgentState) -> AgentState:
    """Run the Data Agent and update state."""
    logger.info("[Supervisor] → data_node")
    t0 = time.perf_counter()

    result = data_agent.run(state["query"])
    elapsed = round(time.perf_counter() - t0, 3)

    state["tickers"] = result["tickers"]
    state["market_data"] = result["market_data"]
    state["agent_log"].append(
        f"data_node: fetched {len(result['tickers'])} tickers in {elapsed}s"
    )
    logger.info(f"[Supervisor] data_node done ({elapsed}s)")
    return state


def news_node(state: AgentState) -> AgentState:
    """Run the News Agent and update state."""
    logger.info("[Supervisor] → news_node")
    t0 = time.perf_counter()

    result = news_agent.run(state["query"], tickers=state.get("tickers", []))
    elapsed = round(time.perf_counter() - t0, 3)

    state["articles"] = result["articles"]
    state["agent_log"].append(
        f"news_node: fetched {len(result['articles'])} articles in {elapsed}s"
    )
    logger.info(f"[Supervisor] news_node done ({elapsed}s)")
    return state


def analysis_node(state: AgentState) -> AgentState:
    """Run the Analysis Agent and update state."""
    logger.info("[Supervisor] → analysis_node")
    t0 = time.perf_counter()

    result = analysis_agent.run(
        query=state["query"],
        market_data=state.get("market_data", {}),
        articles=state.get("articles", []),
    )
    elapsed = round(time.perf_counter() - t0, 3)

    state["analysis"] = result["analysis"]
    state["rag_sources"] = result["rag_sources"]
    state["token_usage"] = result["token_usage"]
    state["agent_log"].append(
        f"analysis_node: generated analysis in {elapsed}s | "
        f"tokens={result['token_usage'].get('total', '?')}"
    )
    logger.info(f"[Supervisor] analysis_node done ({elapsed}s)")
    return state


# ── Build graph ───────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("data_node", data_node)
    builder.add_node("news_node", news_node)
    builder.add_node("analysis_node", analysis_node)

    builder.set_entry_point("data_node")
    builder.add_edge("data_node", "news_node")
    builder.add_edge("news_node", "analysis_node")
    builder.add_edge("analysis_node", END)

    return builder.compile()


_graph = None  # Lazy-compiled singleton


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(query: str) -> dict:
    """
    Execute the full multi-agent pipeline for a user query.

    Returns the final AgentState dict.
    """
    logger.info(f"[Supervisor] Pipeline started for: '{query[:80]}'")
    t_start = time.perf_counter()

    initial_state: AgentState = {
        "query": query,
        "tickers": [],
        "market_data": {},
        "articles": [],
        "analysis": {},
        "rag_sources": [],
        "token_usage": {},
        "total_latency_s": 0.0,
        "agent_log": [],
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)
    final_state["total_latency_s"] = round(time.perf_counter() - t_start, 3)

    logger.info(
        f"[Supervisor] Pipeline complete — "
        f"total={final_state['total_latency_s']}s | "
        f"trace={final_state['agent_log']}"
    )

    return final_state
