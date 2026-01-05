import feedparser

class NewsMCP:
    """
    MCP news client fetching top news from Astana Times RSS feed.
    """

    RSS_URL = "https://astanatimes.com/feed/"

    def latest(self, topic: str | None = None, limit: int = 5):
        """
        Fetch latest news articles.

        :param topic: Optional keyword to filter news
        :param limit: Number of articles to return
        :return: dict with 'data' key containing list of articles
        """
        feed = feedparser.parse(self.RSS_URL)
        articles = []

        for entry in feed.entries:
            title = entry.title
            link = entry.link
            summary = getattr(entry, "summary", "")  # some RSS items may have a summary

            # filter by topic if provided
            if topic is None or topic.lower() in title.lower() or topic.lower() in summary.lower():
                articles.append({
                    "title": title,
                    "link": link,
                    "summary": summary
                })

            if len(articles) >= limit:
                break

        return {"data": articles}
