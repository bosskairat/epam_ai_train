"""
app/mcp_servers/news_server.py
--------------------------------
MCP server exposing financial news fetching tools.

Tries NewsAPI first; falls back to RSS feeds (stdlib xml.etree.ElementTree).

Tools:
  fetch_financial_news(query, max_articles) → articles as formatted text

Transport: stdio (spawned as subprocess by news_agent.py).
Run standalone: python app/mcp_servers/news_server.py
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import requests
from mcp.server.fastmcp import FastMCP
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)
mcp = FastMCP("news")

RSS_FEEDS = {
    # Major finance
    "yahoo_finance":  "https://finance.yahoo.com/news/rssindex",
    "reuters_biz":    "https://feeds.reuters.com/reuters/businessNews",
    "bloomberg":      "https://feeds.bloomberg.com/markets/news.rss",
    "marketwatch":    "https://feeds.content.dowjones.io/public/rss/mw_topstories",

    # Investing / analysis
    "investing_com":  "https://www.investing.com/rss/news.rss",
    "zerohedge":      "https://feeds.feedburner.com/zerohedge/feed",

    # Economy / macro
    "cnbc":           "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "economist_finance": "https://www.economist.com/finance-and-economics/rss.xml",

    # Crypto
    "coindesk":       "https://www.coindesk.com/arc/outboundfeeds/rss/",

    # Quant / tech markets
    "wsj_markets":    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",

    # Central banks / policy
    "federal_reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
}

ARTICLE_SEPARATOR = "\n\n---\n\n"


def _fetch_newsapi(query: str, max_articles: int) -> list[dict]:
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_articles,
                "apiKey": settings.NEWS_API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "title":        a.get("title", ""),
                "summary":      (a.get("description") or a.get("content", ""))[:400],
                "source":       a.get("source", {}).get("name", "NewsAPI"),
                "url":          a.get("url", ""),
                "published_at": a.get("publishedAt", datetime.utcnow().isoformat()),
            }
            for a in data.get("articles", [])
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
    except Exception as exc:
        logger.warning(f"NewsAPI failed: {exc}")
        return []


def _tag(element: ET.Element, tag: str) -> str:
    child = element.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _fetch_one_feed(feed_name: str, url: str, keywords: set[str]) -> list[dict]:
    results = []
    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for item in root.findall(".//item"):
            title   = _tag(item, "title")
            summary = _tag(item, "description")[:400]
            link    = _tag(item, "link")
            pubdate = _tag(item, "pubDate") or datetime.utcnow().isoformat()
            if keywords and not any(kw in (title + summary).lower() for kw in keywords):
                continue
            results.append({
                "title":        title,
                "summary":      summary,
                "source":       feed_name.replace("_", " ").title(),
                "url":          link,
                "published_at": pubdate,
            })
    except ET.ParseError as exc:
        logger.warning(f"RSS XML parse error for {feed_name}: {exc}")
    except Exception as exc:
        logger.warning(f"RSS feed {feed_name} failed: {exc}")
    return results


def _fetch_rss(query: str, max_articles: int) -> list[dict]:
    keywords = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 3}
    articles: list[dict] = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_fetch_one_feed, name, url, keywords): name
            for name, url in RSS_FEEDS.items()
        }
        for future in as_completed(futures, timeout=12):
            try:
                articles.extend(future.result())
            except Exception as exc:
                logger.warning(f"RSS future error: {exc}")

    return articles[:max_articles]


def _format_article(a: dict) -> str:
    return (
        f"Headline: {a['title']}\n"
        f"Source: {a['source']} | Published: {a['published_at']}\n"
        f"Summary: {a['summary']}\n"
        f"URL: {a['url']}"
    )


@mcp.tool()
def fetch_financial_news(query: str, max_articles: int = 8) -> str:
    """Fetch recent financial news articles relevant to the query. Returns formatted article snippets separated by '---'."""
    logger.info(f"[news_server] fetch_financial_news(query='{query[:60]}')")

    articles: list[dict] = []
    if settings.NEWS_API_KEY:
        articles = _fetch_newsapi(query, max_articles)

    if not articles:
        logger.info("[news_server] Falling back to RSS feeds")
        articles = _fetch_rss(query, max_articles)

    articles = articles[:max_articles]
    logger.info(f"[news_server] Fetched {len(articles)} articles")

    if not articles:
        return "No news articles found."

    return ARTICLE_SEPARATOR.join(_format_article(a) for a in articles)


if __name__ == "__main__":
    mcp.run(transport="stdio")
