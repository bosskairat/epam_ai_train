from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import requests
import feedparser
import json


# Load environment variables from .env file
load_dotenv()


@tool
def get_weather(latitude: float, longitude: float, forecast_days: int = 1) -> str:
    """Fetch current weather and forecast for a location using Open-Meteo API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max,temperature_2m_min,weathercode&forecast_days={forecast_days}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return json.dumps(data, indent=2)
    else:
        return f"Error fetching weather: {response.status_code}"


@tool
def get_news(limit: int = 5) -> str:
    """Fetch latest news headlines from BBC RSS feed."""
    feed = feedparser.parse("http://feeds.bbci.co.uk/news/rss.xml")
    headlines = [entry.title for entry in feed.entries[:limit]]
    return "\n".join(headlines)


llm = ChatOpenAI(temperature=0)
tools = [get_weather, get_news]
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)


def query_agent(question):
    return agent.run(question)