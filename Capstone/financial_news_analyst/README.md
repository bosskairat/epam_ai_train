# Financial News Analyst — Multi-Agent GenAI System

A capstone project demonstrating a **multi-agent AI architecture** that combines live financial data, real-time news, and RAG-powered historical context to generate structured investment research summaries.

> **Disclaimer:** All outputs are for educational and informational purposes only. This system does **not** provide financial advice.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Supervisor Agent              │
│     (async sequential orchestration)   │
└──────┬──────────────┬───────────────────┘
       │              │
       ▼              ▼
 ┌──────────┐   ┌──────────┐
 │  Data    │   │  News    │
 │  Agent   │   │  Agent   │
 │(Finnhub  │   │(Finnhub/ │
 │  MCP)    │   │NewsAPI   │
 │          │   │  MCP)    │
 └────┬─────┘   └────┬─────┘
      │              │
      └──────┬───────┘
             ▼
      ┌─────────────┐
      │   Qdrant    │  ← Persistent Vector Store
      │ (embeddings)│
      └──────┬──────┘
             ▼
      ┌─────────────────┐
      │ Analysis Agent  │
      │  (GPT-4o-mini)  │
      │  RAG + LLM      │
      └────────┬────────┘
               ▼
        Structured Report
   (summary / sentiment / drivers)
```

### Agents

| Agent | Role | Transport |
|---|---|---|
| **Supervisor** | Async sequential orchestrator — runs agents in order, collects results | In-process |
| **Data Agent** | Fetches live market data (price, volume, fundamentals) via Finnhub | MCP subprocess |
| **News Agent** | Fetches recent financial news via Finnhub / NewsAPI | MCP subprocess |
| **Analysis Agent** | Ingests data into Qdrant, retrieves RAG context, calls GPT-4o-mini | In-process |

### MCP Servers

| Server | Purpose |
|---|---|
| `market_server.py` | Wraps Finnhub market data API — called by Data Agent |
| `news_server.py` | Wraps Finnhub / NewsAPI news fetch — called by News Agent |

---

## Project Structure

```
financial_news_analyst/
├── app/
│   ├── agents/
│   │   ├── supervisor_agent.py   # Orchestration
│   │   ├── data_agent.py         # Finnhub market data via MCP
│   │   ├── news_agent.py         # News fetch via MCP
│   │   └── analysis_agent.py     # RAG ingest + LLM synthesis
│   ├── mcp_servers/
│   │   ├── market_server.py      # MCP tool: get_market_data
│   │   └── news_server.py        # MCP tool: fetch_financial_news
│   ├── rag/
│   │   ├── vector_store.py       # Qdrant persistent wrapper
│   │   └── retriever.py          # Top-k retrieval + prompt formatting
│   ├── api/
│   │   ├── app.py                # FastAPI factory + middleware
│   │   └── routes.py             # REST endpoints
│   └── core/
│       ├── config.py             # Settings from environment variables
│       ├── logger.py             # Structured logging
│       └── security.py           # Input validation + injection defence
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py
├── ui/
│   └── streamlit_app.py
├── main.py                       # FastAPI entry point
├── run.bat                       # Windows: start FastAPI + Streamlit together
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd financial_news_analyst

# Conda (recommended)
conda create -n capstone python=3.12
conda activate capstone

# Or plain venv
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Fill in your API keys
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes (for LLM) | OpenAI API key — [platform.openai.com](https://platform.openai.com) |
| `FINNHUB_API_KEY` | Yes (for market data) | Finnhub API key — free at [finnhub.io](https://finnhub.io) |
| `NEWS_API_KEY` | Optional | NewsAPI key — free at [newsapi.org](https://newsapi.org). Falls back to Finnhub news if absent. |
| `LLM_MODEL` | Optional | Default: `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Optional | Default: `text-embedding-3-small` |
| `QDRANT_PATH` | Optional | Persistent vector store path. Default: `./qdrant_db` |
| `TOP_K_RESULTS` | Optional | RAG documents to retrieve. Default: `4` |
| `RATE_LIMIT_PER_MINUTE` | Optional | API rate limit. Default: `10` |
| `LOG_LEVEL` | Optional | `DEBUG`, `INFO`, `WARNING`. Default: `INFO` |

> **No OpenAI key?** The pipeline still runs — market data and news are fetched normally; the analysis step returns a structured stub response instead of calling the LLM.

> **No OpenAI key for embeddings?** Automatically falls back to `all-MiniLM-L6-v2` (sentence-transformers, runs locally, zero API cost).

---

## Running

### FastAPI server

```bash
python main.py
# API at   http://localhost:8000
# Docs at  http://localhost:8000/docs
```

### Streamlit UI

```bash
streamlit run ui/streamlit_app.py
# Opens at http://localhost:8501
```

### Both at once (Windows)

```bat
run.bat
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness check |
| `GET` | `/api/v1/rag/stats` | Vector store document count |
| `POST` | `/api/v1/analyze` | Run full pipeline |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Why did Tesla stock drop today?"}'
```

**Response schema:**

```json
{
  "query": "Why did Tesla stock drop today?",
  "analysis": {
    "summary": "...",
    "sentiment": "Bearish",
    "key_drivers": ["..."],
    "risk_factors": ["..."],
    "insight": "...",
    "sources_used": ["..."],
    "disclaimer": "This is not financial advice."
  },
  "tickers": ["TSLA"],
  "articles_count": 6,
  "rag_sources": ["pipeline (2024-01-15)"],
  "token_usage": {"prompt": 820, "completion": 310, "total": 1130},
  "total_latency_s": 3.4,
  "agent_log": ["data_agent: ...", "news_agent: ...", "analysis_agent: ..."]
}
```

---

## Tests

```bash
# All tests (no API keys needed — external calls are mocked)
pytest

# Verbose
pytest -v

# Specific class
pytest tests/test_pipeline.py::TestInputValidation -v

# With coverage
pip install pytest-cov
pytest --cov=app --cov-report=term-missing
```

| Test Class | Coverage |
|---|---|
| `TestInputValidation` | Empty, too-long, and injected inputs |
| `TestTickerExtraction` | Alias mapping, uppercase detection, default fallback |
| `TestPositiveOutputStructure` | Required keys, valid sentiment values, latency |
| `TestNegativeCases` | Edge cases: empty news, missing market data |
| `TestHallucinationDetection` | Rule-based detection of LLM refusal phrases |
| `TestRAGStore` | Qdrant upsert + retrieval + empty store behaviour |
| `TestAPIEndpoints` | FastAPI routes with mocked pipeline |

---

## Demo Queries

```
Why did Tesla stock drop today?
Summarize current market sentiment for S&P 500
What's happening with Bitcoin prices?
How is Nvidia performing this week?
Give me an overview of AAPL and MSFT
What are the key risks in the EV sector?
```

---

## Observability

The system emits structured logs at every pipeline step:

```
[INFO]  supervisor_agent  — Pipeline started for: 'Why did Tesla stock drop today?'
[INFO]  data_agent        — Extracted tickers: {'TSLA'}
[INFO]  data_agent        — Done — 1 tickers fetched
[INFO]  news_agent        — Done — 6 articles fetched
[INFO]  vector_store      — VectorStore loaded — Qdrant persistent at './qdrant_db' (14 docs)
[INFO]  vector_store      — Upserted 7 docs (source_tag=pipeline)
[INFO]  retriever         — RAG context built: 4 docs, ~390 tokens
[INFO]  analysis_agent    — Done — latency=1.94s | tokens={...}
[INFO]  supervisor_agent  — Pipeline complete — total=4.12s
```

---

## Cost Notes

- **Deduplication** — Qdrant uses a content-hash as point ID; re-ingesting the same document is a no-op.
- **Small model** — `gpt-4o-mini` keeps LLM costs low (~10× cheaper than GPT-4).
- **Token-efficient prompts** — top-5 news snippets only; market data compacted to a single text block.
- **Local embedding fallback** — `all-MiniLM-L6-v2` runs offline at zero API cost when no OpenAI key is set.

---

## Security

- Control characters stripped from all user input.
- Query length capped at 500 characters.
- 8 regex patterns block common prompt injection attempts (`ignore instructions`, `act as`, `jailbreak`, etc.).
- API rate limiting via `slowapi` (default: 10 req/min, configurable).
- All secrets loaded from environment variables — nothing hardcoded.
