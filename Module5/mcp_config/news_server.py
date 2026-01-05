from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import feedparser

server = Server("news-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_news",
            description="Fetch latest news headlines from BBC RSS feed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of headlines to return", "default": 5}
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_news":
        limit = arguments.get("limit", 5)
        feed = feedparser.parse("http://feeds.bbci.co.uk/news/rss.xml")
        headlines = [entry.title for entry in feed.entries[:limit]]
        return [TextContent(type="text", text="\n".join(headlines))]
    return []

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())