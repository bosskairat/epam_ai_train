import re
from agents.intent import parse_intent
from agents.llm import OpenAILLM
from mcp_config.weather_mcp import WeatherMCP
from mcp_config.news_mcp import NewsMCP


class AgentOrchestrator:
    def __init__(self):
        self.weather = WeatherMCP()
        self.news = NewsMCP()
        self.llm = OpenAILLM()
        self.city_cache = {}


    def get_city_coords(self, city_name: str):
        """
        Ask the LLM for latitude and longitude of any city.
        Returns (lat, lon) as floats, or None if unknown.
        """
        prompt = f"""
You are a helpful assistant that converts a city name to geographic coordinates.
Return ONLY JSON in the format: {{"latitude": float, "longitude": float}}

City name: "{city_name}"
"""
        response = self.llm.chat(system="City to coordinates converter", user=prompt)

        try:
            coords = eval(response)  # simple JSON-like parsing
            return coords["latitude"], coords["longitude"]
        except Exception:
            return None, None


    def get_days_from_intent(self, intent: dict) -> int:
        date_str = intent.get("date", "today")
        if not date_str:
            return 1
        if date_str == "today":
            return 1
        elif date_str == "tomorrow":
            return 2
        else:
            match = re.search(r'next (\d+) days?', date_str)
            if match:
                return int(match.group(1))
        return 1
    

    def handle(self, query: str, context: list):

        # Build conversation history for LLM
        history_text = ""
        for msg in context[-5:]:  # last 5 messages
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        # Parse intent
        intent = parse_intent(query, history_text)
        tool_results = {}

        # Determine city from intent or last context
        city = intent.get("city")
        if not city:
            # Look for last city mentioned in context
            for msg in reversed(context):
                if msg["role"] == "user":
                    prev_intent = parse_intent(msg["content"], history_text)
                    if prev_intent.get("city"):
                        city = prev_intent["city"]
                        break

        # Weather MCP
        if city and intent.get("weather"):
            if city in self.city_cache:
                lat, lon = self.city_cache[city]
            else:
                lat, lon = self.get_city_coords(city)
                if lat is not None:
                    self.city_cache[city] = (lat, lon)

            days = self.get_days_from_intent(intent)        
            
            if lat is not None and lon is not None:
                weather_result = self.weather.get_weather(lat, lon, days)
                if weather_result["ok"]:
                    tool_results["weather"] = weather_result["data"]
                else:
                    tool_results["weather_error"] = weather_result["message"]
            else:
                tool_results["weather_error"] = {"error": f"Coordinates not found for city '{city}'"}

        # News MCP
        if intent.get("news"):
            news_result = self.news.latest(intent.get("news_topic"))
            if news_result["ok"]:
                tool_results["news"] = news_result["data"]
            else:
                tool_results["news_error"] = news_result["message"]

        # LLM synthesis prompt
        synthesis_prompt = f"""
You are a helpful assistant answering questions about weather and news.
Use the following conversation history and tool data to respond.

Conversation history:
{history_text}

Current user question:
{query}

Available tool data (JSON):
{tool_results}
If tool data contains *_error fields:
- Explain politely that the service is temporarily unavailable
- Do NOT invent data
- Suggest retrying later

Respond clearly, friendly, and concisely.

"""

        return self.llm.chat(
            system="You are a helpful weather and news assistant.",
            user=synthesis_prompt
        )
