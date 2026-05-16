"""
app/agents/news_agent.py
-------------------------
News Agent — fetches recent financial news via the news MCP server.

Responsibilities:
  1. Build a focused search query from the user question + tickers.
  2. Call news_server.py (MCP) to fetch articles.
  3. Return article snippets to the Supervisor.

RAG ingestion is handled downstream by analysis_agent directly.
"""

from __future__ import annotations
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from app.core.logger import get_logger

logger = get_logger(__name__)

NEWS_SERVER = Path(__file__).parent.parent / "mcp_servers" / "news_server.py"
ARTICLE_SEPARATOR = "\n\n---\n\n"


def _build_search_query(user_query: str, tickers: list[str]) -> str:
    # Cap at 3 tickers to keep the NewsAPI query focused; more terms dilute relevance.
    # 500-char limit matches NewsAPI's recommended query length for best results.
    ticker_terms = " OR ".join(tickers[:3]) if tickers else ""
    combined = f"{user_query} {ticker_terms}".strip()
    return combined[:500]


async def run(query: str, tickers: list[str] | None = None) -> dict:
    """
    Entry point called by the Supervisor Agent.

    Returns:
        {
          "articles": [{"text": "<formatted snippet>"}, ...],
        }
    """
    logger.info(f"[NewsAgent] Running for query: '{query[:80]}'")

    tickers = tickers or []
    search_query = _build_search_query(query, tickers)

    params = StdioServerParameters(command=sys.executable, args=[str(NEWS_SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                result = await session.call_tool(
                    "fetch_financial_news",
                    {"query": search_query, "max_articles": 8},
                )
                raw = result.content[0].text if result.content else ""
            except Exception as exc:
                logger.warning(f"[NewsAgent] MCP call failed: {exc}")
                raw = ""

    articles = [{"text": a.strip()} for a in raw.split(ARTICLE_SEPARATOR) if a.strip()]
    logger.info(f"[NewsAgent] Done — {len(articles)} articles fetched")
    return {"articles": articles}
