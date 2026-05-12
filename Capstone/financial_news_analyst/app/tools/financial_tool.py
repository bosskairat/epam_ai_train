"""
app/tools/financial_tool.py
-----------------------------
Fetches live market data using the Finnhub API (free tier).

Free-tier endpoints used:
  /quote            — current price, prev close, day high/low, change %
  /stock/profile2   — company name, market cap, industry

Set FINNHUB_API_KEY in .env (https://finnhub.io/register).
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
import finnhub
from app.core.config import settings
from app.core.logger import get_logger, log_latency

logger = get_logger(__name__)

_client: Optional[finnhub.Client] = None


def _get_client() -> finnhub.Client:
    global _client
    if _client is None:
        _client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)
    return _client


# Crypto tickers → Finnhub exchange-prefixed symbols
_CRYPTO_MAP: dict[str, str] = {
    "BTC-USD":  "BINANCE:BTCUSDT",
    "ETH-USD":  "BINANCE:ETHUSDT",
    "BNB-USD":  "BINANCE:BNBUSDT",
    "SOL-USD":  "BINANCE:SOLUSDT",
    "ADA-USD":  "BINANCE:ADAUSDT",
    "XRP-USD":  "BINANCE:XRPUSDT",
    "DOGE-USD": "BINANCE:DOGEUSDT",
}


def _resolve(ticker: str) -> tuple[str, bool]:
    """Return (finnhub_symbol, is_crypto)."""
    upper = ticker.upper()
    if upper in _CRYPTO_MAP:
        return _CRYPTO_MAP[upper], True
    return upper, False


@log_latency(logger)
def fetch_market_data(ticker: str, period: str = "5d") -> dict:
    """
    Retrieve current price + company info for a stock/crypto ticker.

    Uses only Finnhub free-tier endpoints (/quote, /stock/profile2).
    Returns a structured dict compatible with format_for_rag().
    """
    logger.info(f"Fetching market data for {ticker} via Finnhub")

    if not settings.FINNHUB_API_KEY:
        logger.warning("FINNHUB_API_KEY not set — returning empty market data")
        return _empty_result(ticker, error="FINNHUB_API_KEY not configured")

    symbol, is_crypto = _resolve(ticker)
    client = _get_client()

    try:
        # ── Quote (current price, prev close, day range) ──────────────────────
        quote = client.quote(symbol)
        current_price = quote.get("c")
        prev_close    = quote.get("pc")
        day_high      = quote.get("h")
        day_low       = quote.get("l")

        if not current_price:
            return _empty_result(ticker, error="no quote data returned")

        change_pct = quote.get("dp") or (
            (current_price - prev_close) / prev_close * 100 if prev_close else 0.0
        )

        # ── Company profile (stocks only, best-effort) ────────────────────────
        company_name = ticker
        market_cap   = None
        industry     = None

        if not is_crypto:
            try:
                profile      = client.company_profile2(symbol=symbol)
                company_name = profile.get("name") or ticker
                mc           = profile.get("marketCapitalization")
                market_cap   = int(mc * 1_000_000) if mc else None  # Finnhub gives millions
                industry     = profile.get("finnhubIndustry")
            except Exception as profile_exc:
                logger.warning(f"Could not fetch profile for {ticker}: {profile_exc}")

        result = {
            "ticker":        ticker.upper(),
            "company_name":  company_name,
            "current_price": round(float(current_price), 4),
            "change_pct":    round(float(change_pct), 2),
            "volume":        None,
            "market_cap":    market_cap,
            "sector":        None,
            "industry":      industry,
            "pe_ratio":      None,
            "52w_high":      day_high,
            "52w_low":       day_low,
            "history":       [],
            "period":        period,
            "fetched_at":    datetime.utcnow().isoformat(),
        }

        logger.info(
            f"{ticker}: price={result['current_price']}, change={result['change_pct']}%"
        )
        return result

    except finnhub.FinnhubAPIException as exc:
        if exc.status_code == 403:
            logger.error(
                f"Finnhub 403 for {ticker} — check FINNHUB_API_KEY in .env "
                "(get a free key at https://finnhub.io/register)"
            )
            return _empty_result(ticker, error="Finnhub API key missing or invalid (403)")
        logger.error(f"Finnhub API error for {ticker}: {exc}")
        return _empty_result(ticker, error=str(exc))
    except Exception as exc:
        logger.error(f"Failed to fetch market data for {ticker}: {exc}")
        return _empty_result(ticker, error=str(exc))


def _empty_result(ticker: str, error: Optional[str] = None) -> dict:
    return {
        "ticker":        ticker.upper(),
        "company_name":  ticker,
        "current_price": None,
        "change_pct":    None,
        "volume":        None,
        "market_cap":    None,
        "history":       [],
        "fetched_at":    datetime.utcnow().isoformat(),
        "error":         error,
    }


def format_for_rag(data: dict) -> str:
    """Convert market data dict to a plain-text snippet for RAG storage."""
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
