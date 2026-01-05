# Weather & News Agent (Streamlit + MCP)

## Overview
This project is a Python-based AI agent that answers questions about:
- Current weather conditions
- Latest news headlines

It uses:
- Streamlit for UI
- Agent orchestration pattern
- Model Context Protocol (MCP) for external tools
- Open-Meteo and TheNews API (no API keys)

## Architecture
- Intent detection → Tool routing → Response aggregation
- MCP servers standardize external data access
- Supports multi-turn conversations

## LLM Integration
This project uses OpenAI GPT-4o-mini as the reasoning engine:
- Intent detection
- Entity extraction
- Tool routing decisions
- Final response synthesis

External data access is performed exclusively via MCP servers.

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py



