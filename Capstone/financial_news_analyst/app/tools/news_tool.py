"""
app/tools/news_tool.py
-----------------------
Fetches financial news from NewsAPI (if key available) with an RSS fallback.
RSS is parsed with stdlib xml.etree.ElementTree — no feedparser needed.
Returns structured articles ready for analysis and RAG ingestion.
"""

from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
from app.core.config import settings
from app.core.logger import get_logger, log_latency

logger = get_logger(__name__)

# Free financial RSS feeds used as fallback / supplement
RSS_FEEDS = {
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "reuters_biz":   "https://feeds.reuters.com/reuters/businessNews",
    "investing_com": "https://www.investing.com/rss/news.rss",
    "seeking_alpha": "https://seekingalpha.com/feed.xml",
}


@log_latency(logger)
def fetch_news(query: str, max_articles: int = 8) -> list[dict]:
    """
    Fetch financial news articles for a query.
    Tries NewsAPI first; falls back to RSS (stdlib xml) if no key.
    """
    logger.info(f"Fetching news for query: '{query}'")
    articles = []

    if settings.NEWS_API_KEY:
        articles = _fetch_newsapi(query, max_articles)

    if not articles:
        logger.info("Falling back to RSS feeds (stdlib xml.etree)")
        articles = _fetch_rss(query, max_articles)

    logger.info(f"Fetched {len(articles)} articles")
    return articles[:max_articles]


def _fetch_newsapi(query: str, max_articles: int) -> list[dict]:
    """Call NewsAPI /v2/everything endpoint."""
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


def _fetch_rss(query: str, max_articles: int) -> list[dict]:
    """
    Parse RSS 2.0 feeds using stdlib xml.etree.ElementTree.
    No feedparser or third-party XML library required.
    """
    keywords = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 3}
    articles = []

    for feed_name, url in RSS_FEEDS.items():
        if len(articles) >= max_articles:
            break
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            for item in root.findall(".//item"):
                title   = _tag(item, "title")
                summary = _tag(item, "description")[:400]
                link    = _tag(item, "link")
                pubdate = _tag(item, "pubDate") or datetime.utcnow().isoformat()

                # Simple keyword relevance filter
                if keywords and not any(kw in (title + summary).lower() for kw in keywords):
                    continue

                articles.append({
                    "title":        title,
                    "summary":      summary,
                    "source":       feed_name.replace("_", " ").title(),
                    "url":          link,
                    "published_at": pubdate,
                })

                if len(articles) >= max_articles:
                    break

        except ET.ParseError as exc:
            logger.warning(f"RSS XML parse error for {feed_name}: {exc}")
        except Exception as exc:
            logger.warning(f"RSS feed {feed_name} failed: {exc}")

    return articles


def _tag(element: ET.Element, tag: str) -> str:
    """Safely extract text from an XML child element."""
    child = element.find(tag)
    return (child.text or "").strip() if child is not None else ""


def format_for_rag(articles: list[dict]) -> list[str]:
    """Convert article list to plain-text snippets for RAG storage."""
    return [
        f"Headline: {a['title']}\n"
        f"Source: {a['source']} | Published: {a['published_at']}\n"
        f"Summary: {a['summary']}\n"
        f"URL: {a['url']}"
        for a in articles
    ]
