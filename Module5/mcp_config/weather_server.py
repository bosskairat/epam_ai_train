from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import requests
import json

server = Server("weather-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description="Fetch current weather and forecast for a location using Open-Meteo API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "Latitude of the location"},
                    "longitude": {"type": "number", "description": "Longitude of the location"},
                    "forecast_days": {"type": "integer", "description": "Number of forecast days (1-16)", "default": 1}
                },
                "required": ["latitude", "longitude"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_weather":
        lat = arguments["latitude"]
        lon = arguments["longitude"]
        days = arguments.get("forecast_days", 1)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,weathercode&forecast_days={days}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return [TextContent(type="text", text=json.dumps(data, indent=2))]
        else:
            return [TextContent(type="text", text=f"Error fetching weather: {response.status_code}")]
    return []

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())