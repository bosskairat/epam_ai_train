import feedparser

SOURCES = {
    "astana_times": "https://astanatimes.com/feed/",
    "qazmonitor": "https://en.qazmonitor.com/rss",
    "kazakhstan_today": "https://www.kt.kz/eng/rss",
}

class NewsMCP:

    def latest(self, topic: str | None = None, limit: int = 3):
        articles = []

        for source, url in SOURCES.items():
            try:
                feed = feedparser.parse(url)

                if feed.bozo:
                    continue  # broken feed → skip

                for entry in feed.entries[:limit]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title} {summary}".lower()

                    if topic is None or topic.lower() in text:
                        articles.append({
                            "source": source,
                            "title": title,
                            "link": entry.get("link")
                        })
            
            except Exception:
                continue  # fallback to next source
        
        if articles:
            return  {"ok": True, "data": articles}
        else:
            return {
                "ok": False,
                "message": "All news sources are currently unavailable"
            }
    
        
