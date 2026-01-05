# Weather & News Agent

## Overview
The **Weather & News Agent** is a Python-based AI application that allows users to ask natural language questions about:

- Current weather conditions  
- Latest news headlines

The agent uses **Streamlit** for a web interface and integrates **external data sources** through **MCP (Model Context Protocol) servers**, providing a robust, multi-turn conversational experience.

Key features:
- Multi-turn conversations with memory of previous questions
- Weather forecasts including today, tomorrow, and the next N days
- Top news headlines with optional topic filtering
- Dynamic city recognition via LLM

---

## Architecture

The application uses a **modular agent orchestration pattern**:

```
User Input → Intent Detection → Tool Routing → MCP Servers → LLM Synthesis → User Response
```

**Components:**

1. **Intent Detection**
   - Parses user queries to determine whether the request is for weather, news, or both  
   - Extracts entities such as city names, dates (“tomorrow”, “next N days”), and news topics  

2. **Agent Orchestrator**
   - Routes requests to the appropriate MCP servers (weather or news)  
   - Aggregates and formats the results  
   - Maintains conversational context for follow-up questions  

3. **MCP Servers**
   - Standardized interface to external tools:
     - **Weather:** Open-Meteo (free, no API key)  
     - **News:** Astana Times RSS feed (no API key required)  
   - Handles errors and provides fallback responses if services are unavailable  

4. **LLM Integration**
   - OpenAI GPT-4o-mini is used for:
     - Intent parsing  
     - City and date resolution  
     - Synthesizing user-friendly responses from tool data  
   - Ensures natural, concise, and context-aware answers  

---

## Features

- **Natural Language Queries**
  - Ask anything like:
    - “What’s the weather in Astana today?”  
    - “Will it rain tomorrow in Almaty?”  
    - “Show me the latest news about AI”

- **Multi-turn Conversations**
  - “And tomorrow?”  
  - “What about the next 3 days?”  
  - The agent remembers previous cities and topics  

- **Dynamic City Recognition**
  - Supports any city worldwide using the LLM to generate latitude and longitude  

- **Flexible Date Parsing**
  - Supports: `today`, `tomorrow`, `next N days`  

- **News Filtering**
  - Optionally filter news by keywords or topics  

- **Streamlit Web Interface**
  - Pinned input at the bottom  
  - Scrollable chat history  
  - Clear separation of user queries and agent responses  

---

## Setup Instructions

1. **Clone the repository**
```bash
unzip weather-news-agent.zip
cd weather-news-agent
```

2. **Create and activate a Python virtual environment**
```bash
python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

5. **Open the web interface**
- The app will open at `http://localhost:8501/` by default

---

## Project Structure

```
project/
├── app.py                  # Streamlit UI
├── agents/
│   ├── context.py          # Conversation history tracking
│   ├── intent.py           # Intent parsing using LLM
│   ├── llm.py              # OpenAI LLM wrapper
│   └── orchestrator.py     # Agent orchestrator
├── mcp_config/
│   ├── news_mcp.py         # News MCP (Astana Times RSS)
│   └── weather_mcp.py      # Weather MCP (Open-Meteo)
├── requirements.txt
└── README.md
```

---

## Usage Examples

**Weather queries**
```
You: What's the weather in Almaty today?
Assistant: The current weather in Almaty is 5°C with clear skies.
You: And tomorrow?
Assistant: Tomorrow: Max 7°C, Min 1°C, 10% chance of precipitation.
You: Show me the next 3 days forecast.
Assistant: Day 1: ..., Day 2: ..., Day 3: ...
```

**News queries**
```
You: Latest news about AI
Assistant: Top AI news from Astana Times:
1. "Kazakhstan AI Startups Gain Momentum" – link
2. "AI in Education: Opportunities and Challenges" – link
3. ...
```

**Combined queries**
```
You: What's the weather in Astana and any news about politics?
Assistant: Weather in Astana: ..., News: Top headlines about politics: ...
```

---

## Notes

- The app requires **internet access** to fetch data from Open-Meteo and Astana Times RSS  
- No API keys are required for either service  
- LLM responses depend on OpenAI API access — ensure your API key is set in the environment if needed  

---

## Future Improvements

- Add **more news sources** with RSS feeds  
- Add **response caching** for performance  
- Add **multi-language support**  
- Include **graphs for weather trends** in Streamlit  

---

