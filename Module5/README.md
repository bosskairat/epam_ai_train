# Weather and News Query App

This is a Python-based application using Streamlit that answers user questions about current weather conditions and latest news. It leverages agent orchestration patterns and Model Context Protocol (MCP) servers to integrate external data sources.

## Features

- Query weather information using Open-Meteo (free, no API key)
- Fetch latest news from BBC RSS feed (free, no API key)
- Agent orchestration using LangChain with OpenAI
- Streamlit web interface with conversation history
- MCP server implementations for weather and news (in mcp_config/)

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set your OpenAI API key: `export OPENAI_API_KEY=your_key_here`
3. Run the app: `streamlit run app.py`

## Project Structure

- `app.py`: Main Streamlit application
- `agents/orchestrator.py`: LangChain agent with tools for weather and news
- `mcp_config/`: MCP server implementations (weather_server.py, news_server.py)
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Usage

- Enter questions like "What's the weather in New York?" or "Latest news headlines"
- The agent will parse the query and fetch relevant data
- Conversation history is maintained in the session

## MCP Servers

The project includes custom MCP servers for demonstration:
- `weather_server.py`: Fetches weather from Open-Meteo
- `news_server.py`: Fetches news from BBC RSS

To run MCP servers separately: `python mcp_config/weather_server.py` (requires MCP SDK)