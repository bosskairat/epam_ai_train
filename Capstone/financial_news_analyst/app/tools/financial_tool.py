"""
app/tools/financial_tool.py
-----------------------------
Fetches live market data using yfinance.
Returns structured JSON suitable for the Data Agent and RAG ingestion.
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import Optional
import yfinance as yf
from app.core.logger import get_logger, log_latency

logger = get_logger(__name__)


@log_latency(logger)
def fetch_market_data(ticker: str, period: str = "5d") -> dict:
    """
    Retrieve price history + key info for a stock/crypto ticker.

    Args:
        ticker: Symbol such as 'TSLA', 'BTC-USD', 'SPY'.
        period: yfinance period string ('1d', '5d', '1mo', …).

    Returns:
        Structured dict with:
          - ticker
          - company_name
          - current_price
          - change_pct  (vs previous close)
          - volume
          - market_cap
          - history     (list of OHLCV dicts for the period)
          - fetched_at  (ISO timestamp)
    """
    logger.info(f"Fetching market data for {ticker} (period={period})")

    try:
        stock = yf.Ticker(ticker)
        try:
            info = stock.info or {}
        except Exception as info_exc:
            logger.warning(f"Could not fetch info for {ticker} (using price history only): {info_exc}")
            info = {}
        hist = stock.history(period=period)

        if hist.empty:
            logger.warning(f"No history data for {ticker}")
            return _empty_result(ticker, error="no history data returned")

        latest = hist.iloc[-1]
        prev_close = hist.iloc[-2]["Close"] if len(hist) > 1 else latest["Close"]
        change_pct = ((latest["Close"] - prev_close) / prev_close * 100) if prev_close else 0.0

        history_records = [
            {
                "date": str(idx.date()),
                "open": round(row["Open"], 4),
                "high": round(row["High"], 4),
                "low": round(row["Low"], 4),
                "close": round(row["Close"], 4),
                "volume": int(row["Volume"]),
            }
            for idx, row in hist.iterrows()
        ]

        result = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName") or info.get("shortName") or ticker,
            "current_price": round(float(latest["Close"]), 4),
            "change_pct": round(change_pct, 2),
            "volume": int(latest["Volume"]),
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "pe_ratio": info.get("trailingPE"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "history": history_records,
            "period": period,
            "fetched_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"{ticker}: price={result['current_price']}, change={result['change_pct']}%"
        )
        return result

    except Exception as exc:
        logger.error(f"Failed to fetch market data for {ticker}: {exc}")
        return _empty_result(ticker, error=str(exc))


def _empty_result(ticker: str, error: Optional[str] = None) -> dict:
    return {
        "ticker": ticker.upper(),
        "company_name": ticker,
        "current_price": None,
        "change_pct": None,
        "volume": None,
        "market_cap": None,
        "history": [],
        "fetched_at": datetime.utcnow().isoformat(),
        "error": error,
    }


def format_for_rag(data: dict) -> str:
    """Convert market data dict to a plain-text snippet for RAG storage."""
    if data.get("error"):
        return f"[Market data unavailable for {data['ticker']}: {data['error']}]"

    lines = [
        f"Ticker: {data['ticker']} ({data['company_name']})",
        f"Price: ${data['current_price']} ({data['change_pct']:+.2f}% vs prev close)",
        f"Volume: {data['volume']:,}" if data['volume'] else "Volume: N/A",
        f"Market Cap: ${data['market_cap']:,}" if data['market_cap'] else "Market Cap: N/A",
        f"Sector: {data.get('sector', 'N/A')} | Industry: {data.get('industry', 'N/A')}",
        f"52W High: {data.get('52w_high')} | 52W Low: {data.get('52w_low')}",
        f"P/E Ratio: {data.get('pe_ratio', 'N/A')}",
        f"As of: {data['fetched_at']}",
    ]
    return "\n".join(lines)
