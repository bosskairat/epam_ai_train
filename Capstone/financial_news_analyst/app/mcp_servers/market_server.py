"""
app/mcp_servers/market_server.py
----------------------------------
MCP server exposing live market data tools via the Finnhub API (free tier).

Tools:
  get_market_data(ticker) → formatted price/company text

Transport: stdio (spawned as subprocess by data_agent.py).
Run standalone: python app/mcp_servers/market_server.py
"""

from __future__ import annotations
import os
import sys

# Add project root to sys.path so `app.*` imports resolve when run as subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime
from typing import Optional
import finnhub
from mcp.server.fastmcp import FastMCP
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)
mcp = FastMCP("market-data")

_client: Optional[finnhub.Client] = None

_CRYPTO_MAP: dict[str, str] = {
    "BTC-USD":  "BINANCE:BTCUSDT",
    "ETH-USD":  "BINANCE:ETHUSDT",
    "BNB-USD":  "BINANCE:BNBUSDT",
    "SOL-USD":  "BINANCE:SOLUSDT",
    "ADA-USD":  "BINANCE:ADAUSDT",
    "XRP-USD":  "BINANCE:XRPUSDT",
    "DOGE-USD": "BINANCE:DOGEUSDT",
}


def _get_client() -> finnhub.Client:
    global _client
    if _client is None:
        _client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)
    return _client


def _resolve(ticker: str) -> tuple[str, bool]:
    upper = ticker.upper()
    if upper in _CRYPTO_MAP:
        return _CRYPTO_MAP[upper], True
    return upper, False


def _format(data: dict) -> str:
    if data.get("error"):
        return f"[Market data unavailable for {data['ticker']}: {data['error']}]"
    lines = [
        f"Ticker: {data['ticker']} ({data['company_name']})",
        f"Price: {data['current_price']} ({data['change_pct']:+.2f}% vs prev close)",
        f"Market Cap: {data['market_cap']:,}" if data["market_cap"] else "Market Cap: N/A",
        f"Industry: {data.get('industry', 'N/A')}",
        f"Day High: {data.get('52w_high')} | Day Low: {data.get('52w_low')}",
        f"As of: {data['fetched_at']}",
    ]
    return "\n".join(lines)


@mcp.tool()
def get_market_data(ticker: str) -> str:
    """Fetch current price, change %, market cap, and industry for a stock or crypto ticker (e.g. AAPL, TSLA, BTC-USD)."""
    logger.info(f"[market_server] get_market_data({ticker})")

    if not settings.FINNHUB_API_KEY:
        return f"[Market data unavailable for {ticker.upper()}: FINNHUB_API_KEY not configured]"

    symbol, is_crypto = _resolve(ticker)
    client = _get_client()

    try:
        quote = client.quote(symbol)
        current_price = quote.get("c")
        prev_close    = quote.get("pc")
        day_high      = quote.get("h")
        day_low       = quote.get("l")

        if not current_price:
            return f"[Market data unavailable for {ticker.upper()}: no quote data returned]"

        change_pct = quote.get("dp") or (
            (current_price - prev_close) / prev_close * 100 if prev_close else 0.0
        )

        company_name = ticker
        market_cap   = None
        industry     = None

        if not is_crypto:
            try:
                profile      = client.company_profile2(symbol=symbol)
                company_name = profile.get("name") or ticker
                mc           = profile.get("marketCapitalization")
                market_cap   = int(mc * 1_000_000) if mc else None
                industry     = profile.get("finnhubIndustry")
            except Exception as exc:
                logger.warning(f"Could not fetch profile for {ticker}: {exc}")

        data = {
            "ticker":        ticker.upper(),
            "company_name":  company_name,
            "current_price": round(float(current_price), 4),
            "change_pct":    round(float(change_pct), 2),
            "market_cap":    market_cap,
            "industry":      industry,
            "52w_high":      day_high,
            "52w_low":       day_low,
            "fetched_at":    datetime.utcnow().isoformat(),
        }
        return _format(data)

    except finnhub.FinnhubAPIException as exc:
        if exc.status_code == 403:
            return f"[Market data unavailable for {ticker.upper()}: Finnhub API key missing or invalid (403)]"
        return f"[Market data unavailable for {ticker.upper()}: {exc}]"
    except Exception as exc:
        logger.error(f"Failed to fetch market data for {ticker}: {exc}")
        return f"[Market data unavailable for {ticker.upper()}: {exc}]"


if __name__ == "__main__":
    mcp.run(transport="stdio")
