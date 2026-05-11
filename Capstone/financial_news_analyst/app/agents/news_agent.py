"""
app/agents/news_agent.py
-------------------------
News Agent — fetches recent financial news and ingests it into ChromaDB.

Responsibilities:
  1. Build a news search query from the user question.
  2. Fetch articles via news_tool (NewsAPI → RSS fallback).
  3. Store each article snippet in the vector store.
  4. Return structured article list to the Supervisor.
"""

from __future__ import annotations
from app.tools.news_tool import fetch_news, format_for_rag
from app.rag.vector_store import get_vector_store
from app.core.logger import get_logger, log_latency

logger = get_logger(__name__)


def _build_search_query(user_query: str, tickers: list[str]) -> str:
    """
    Construct a focused news search string by combining the user question
    with identified ticker symbols.
    """
    # Add ticker symbols so news is financially relevant
    ticker_terms = " OR ".join(tickers[:3]) if tickers else ""
    combined = f"{user_query} {ticker_terms}".strip()
    # Trim to NewsAPI's query length limit
    return combined[:500]


@log_latency(logger)
def run(query: str, tickers: list[str] | None = None) -> dict:
    """
    Entry point called by the Supervisor Agent.

    Args:
        query:   Original user question.
        tickers: Ticker symbols identified by the Data Agent.

    Returns:
        {
          "articles":     [...],   # raw article dicts
          "rag_ingested": int,
        }
    """
    logger.info(f"[NewsAgent] Running for query: '{query[:80]}'")

    tickers = tickers or []
    search_query = _build_search_query(query, tickers)

    articles = fetch_news(search_query, max_articles=8)

    # Prepare text snippets for RAG
    snippets = format_for_rag(articles)
    metadatas = [
        {"source": a.get("source", "unknown"), "published_at": a.get("published_at", "")}
        for a in articles
    ]

    store = get_vector_store()
    ingested = store.upsert(
        texts=snippets,
        metadatas=metadatas,
        source_tag="news",
    )

    logger.info(f"[NewsAgent] Done — {len(articles)} articles, {ingested} docs ingested")

    return {
        "articles": articles,
        "rag_ingested": ingested,
    }
