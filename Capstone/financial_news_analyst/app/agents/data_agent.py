"""
app/agents/data_agent.py
-------------------------
Data Agent — fetches live market data and ingests it into the vector store.

Responsibilities:
  1. Extract ticker symbols from a user query.
  2. Call yfinance via financial_tool.
  3. Store a text snippet in ChromaDB for future RAG retrieval.
  4. Return structured market data to the Supervisor.
"""

from __future__ import annotations
import re
from app.tools.financial_tool import fetch_market_data, format_for_rag
from app.rag.vector_store import get_vector_store
from app.core.logger import get_logger, log_latency

logger = get_logger(__name__)

# Well-known ticker aliases in natural language
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
    """
    Heuristically extract ticker symbols or company names from a free-text query.

    Returns a list of yfinance-compatible symbols.
    """
    tickers = set()
    q_lower = query.lower()

    # 1. Map well-known aliases
    for name, symbol in _ALIAS_MAP.items():
        if name in q_lower:
            tickers.add(symbol)

    # 2. Uppercase 1-5 letter words that look like tickers (e.g. TSLA, AAPL)
    for match in re.findall(r"\b[A-Z]{1,5}\b", query):
        # Skip common English words
        if match not in {"I", "A", "THE", "AND", "OR", "IN", "IS", "FOR", "OF", "TO", "IT"}:
            tickers.add(match)

    # 3. Detect crypto patterns like BTC-USD
    for match in re.findall(r"\b[A-Z]{2,5}-USD\b", query):
        tickers.add(match)

    logger.info(f"Extracted tickers: {tickers}")
    return list(tickers) if tickers else ["SPY"]   # default: S&P 500 ETF


@log_latency(logger)
def run(query: str) -> dict:
    """
    Entry point called by the Supervisor Agent.

    Returns:
        {
          "tickers": [...],
          "market_data": { TICKER: {...}, ... },
          "rag_ingested": int,   # number of docs added to vector store
        }
    """
    logger.info(f"[DataAgent] Running for query: '{query[:80]}'")

    tickers = extract_tickers(query)
    market_data = {}
    snippets = []

    for ticker in tickers:
        data = fetch_market_data(ticker)
        market_data[ticker] = data
        snippets.append(format_for_rag(data))

    # Ingest into vector store for future RAG retrieval
    store = get_vector_store()
    ingested = store.upsert(
        texts=snippets,
        metadatas=[{"ticker": t} for t in tickers],
        source_tag="market_data",
    )

    logger.info(f"[DataAgent] Done — {len(tickers)} tickers, {ingested} docs ingested")

    return {
        "tickers": tickers,
        "market_data": market_data,
        "rag_ingested": ingested,
    }
