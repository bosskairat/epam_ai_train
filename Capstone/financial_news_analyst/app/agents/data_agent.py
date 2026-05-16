"""
app/agents/data_agent.py
-------------------------
Data Agent — fetches live market data via the market MCP server.

Responsibilities:
  1. Extract ticker symbols from a user query (heuristic).
  2. Call market_server.py (MCP) for each ticker.
  3. Return structured market data text to the Supervisor.

RAG ingestion is handled downstream by analysis_agent directly.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from app.core.logger import get_logger

logger = get_logger(__name__)

MARKET_SERVER = Path(__file__).parent.parent / "mcp_servers" / "market_server.py"

# Maps common company/asset names to their exchange ticker symbols so natural-language
# queries ("Tesla", "bitcoin") resolve without requiring the user to know the ticker.
_ALIAS_MAP = {
    "tesla": "TSLA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "sp500": "SPY",
    "s&p": "SPY",
    "nasdaq": "QQQ",
    "dow": "DIA",
}


def extract_tickers(query: str) -> list[str]:
    """Heuristically extract ticker symbols or company names from a free-text query."""
    tickers: set[str] = set()
    q_lower = query.lower()

    for name, symbol in _ALIAS_MAP.items():
        if name in q_lower:
            tickers.add(symbol)

    # Uppercase tokens of 1–5 chars are candidate ticker symbols, but common English
    # words (pronouns, articles, prepositions) must be excluded to avoid false matches.
    for match in re.findall(r"\b[A-Z]{1,5}\b", query):
        if match not in {"I", "A", "THE", "AND", "OR", "IN", "IS", "FOR", "OF", "TO", "IT"}:
            tickers.add(match)

    for match in re.findall(r"\b[A-Z]{2,5}-USD\b", query):
        tickers.add(match)

    logger.info(f"Extracted tickers: {tickers}")
    return list(tickers) if tickers else ["SPY"]


async def run(query: str) -> dict:
    """
    Entry point called by the Supervisor Agent.

    Returns:
        {
          "tickers":     list[str],
          "market_data": {TICKER: "formatted text", ...},
        }
    """
    logger.info(f"[DataAgent] Running for query: '{query[:80]}'")

    tickers = extract_tickers(query)
    market_data: dict[str, str] = {}

    params = StdioServerParameters(command=sys.executable, args=[str(MARKET_SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for ticker in tickers:
                try:
                    result = await session.call_tool("get_market_data", {"ticker": ticker})
                    market_data[ticker] = result.content[0].text if result.content else ""
                except Exception as exc:
                    logger.warning(f"[DataAgent] MCP call failed for {ticker}: {exc}")
                    market_data[ticker] = f"[Market data unavailable for {ticker}: {exc}]"

    logger.info(f"[DataAgent] Done — {len(tickers)} tickers fetched")
    return {"tickers": tickers, "market_data": market_data}
