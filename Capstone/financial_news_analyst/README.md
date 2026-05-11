# 📈 Financial News Analyst — Multi-Agent GenAI System

A production-ready capstone project demonstrating a **multi-agent AI architecture** that combines live financial data, real-time news, and RAG-powered historical context to generate investment research summaries.

> ⚠️ **Disclaimer:** All outputs are for educational and informational purposes only. This system does **not** provide financial advice.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│            Supervisor Agent                  │
│         (LangGraph state machine)            │
└──────┬──────────────┬───────────────────────┘
       │              │
       ▼              ▼
 ┌──────────┐   ┌──────────┐
 │  Data    │   │  News    │
 │  Agent   │   │  Agent   │
 │(yfinance)│   │(NewsAPI/ │
 │          │   │  RSS)    │
 └────┬─────┘   └────┬─────┘
      │              │
      └──────┬───────┘
             ▼
      ┌─────────────┐
      │  ChromaDB   │  ← RAG Vector Store
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

| Agent | Role |
|---|---|
| **Supervisor** | LangGraph orchestrator — routes tasks, manages execution flow |
| **Data Agent** | Fetches live market data via `yfinance`, ingests into ChromaDB |
| **News Agent** | Fetches recent news via NewsAPI (RSS fallback), ingests into ChromaDB |
| **Analysis Agent** | RAG retrieval + GPT-4o-mini synthesis → structured JSON report |

---

## 📁 Project Structure

```
financial_news_analyst/
├── app/
│   ├── agents/
│   │   ├── supervisor_agent.py   # LangGraph orchestration
│   │   ├── data_agent.py         # yfinance market data
│   │   ├── news_agent.py         # NewsAPI / RSS
│   │   └── analysis_agent.py     # LLM + RAG analysis
│   ├── rag/
│   │   ├── vector_store.py       # ChromaDB wrapper
│   │   └── retriever.py          # Top-k retrieval + formatting
│   ├── tools/
│   │   ├── financial_tool.py     # yfinance helper
│   │   └── news_tool.py          # NewsAPI + RSS helper
│   ├── api/
│   │   ├── app.py                # FastAPI factory + middleware
│   │   └── routes.py             # REST endpoints
│   └── core/
│       ├── config.py             # Settings from env vars
│       ├── logger.py             # Structured logging + latency tracking
│       └── security.py           # Input validation + injection defence
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py          # Full test suite (40+ tests)
├── ui/
│   └── streamlit_app.py          # Streamlit front-end
├── main.py                       # Entry point (server + demo mode)
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone & create virtual environment

```bash
git clone <repo-url>
cd financial_news_analyst

python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | **Yes** (for LLM) | OpenAI API key — get one at platform.openai.com |
| `NEWS_API_KEY` | Optional | NewsAPI key — free at newsapi.org. Falls back to RSS if absent. |
| `LLM_MODEL` | Optional | Default: `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Optional | Default: `text-embedding-3-small` |
| `CHROMA_PERSIST_DIR` | Optional | Default: `./chroma_db` |
| `TOP_K_RESULTS` | Optional | Number of RAG docs to retrieve. Default: `4` |
| `RATE_LIMIT_PER_MINUTE` | Optional | API rate limit. Default: `10` |
| `LOG_LEVEL` | Optional | `DEBUG`, `INFO`, `WARNING`. Default: `INFO` |

> **No API key?** The system still runs in stub mode — the data pipeline (yfinance + news + ChromaDB) works fully; only the LLM analysis step returns a placeholder.

---

## 🚀 Running the System

### Option A — FastAPI Server

```bash
python main.py
# Server starts at http://localhost:8000
# Docs at   http://localhost:8000/docs
```

#### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness check |
| `GET` | `/api/v1/rag/stats` | ChromaDB document count |
| `POST` | `/api/v1/analyze` | Run full pipeline |

Example request:

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Why did Tesla stock drop today?"}'
```

### Option B — Streamlit UI

```bash
streamlit run ui/streamlit_app.py
# Opens at http://localhost:8501
```

### Option C — Demo mode (no server)

```bash
python main.py --demo
```

---

## 🧪 Running Tests

```bash
# All tests (no API keys required — all external calls are mocked)
pytest

# Verbose output
pytest -v

# Specific test class
pytest tests/test_pipeline.py::TestInputValidation -v

# With coverage
pip install pytest-cov
pytest --cov=app --cov-report=term-missing
```

### Test Categories

| Class | What it tests |
|---|---|
| `TestInputValidation` | Empty/long/injected inputs |
| `TestTickerExtraction` | Alias mapping, uppercase detection |
| `TestPositiveOutputStructure` | Schema validation, required keys |
| `TestNegativeCases` | Edge cases, empty news, conflicting data |
| `TestHallucinationDetection` | Rule-based hallucination phrase detection |
| `TestRAGStore` | ChromaDB upsert + retrieval |
| `TestAPIEndpoints` | FastAPI routes with mocked pipeline |

---

## 🎬 Demo Queries

```
Why did Tesla stock drop today?
Summarize current market sentiment for S&P 500
What's happening with Bitcoin prices?
How is Nvidia performing this week?
Give me an overview of AAPL and MSFT
What are the key risks in the EV sector?
```

### Sample Output (stub mode)

```json
{
  "summary": "Tesla shares declined sharply following weaker-than-expected delivery numbers.",
  "sentiment": "Bearish",
  "key_drivers": [
    "Missed Q4 delivery estimates",
    "Rising EV competition from BYD",
    "High interest rates dampening demand"
  ],
  "risk_factors": [
    "Further demand slowdown in key markets",
    "Price war escalation"
  ],
  "insight": "Tesla's stock movements often correlate strongly with quarterly delivery data...",
  "sources_used": ["Reuters", "Bloomberg", "market_data (2024-01-15)"],
  "disclaimer": "This is not financial advice."
}
```

---

## 📊 Observability

The system logs at every step:

```
[2024-01-15 12:00:01] INFO  app.agents.supervisor_agent — [Supervisor] Pipeline started
[2024-01-15 12:00:01] INFO  app.agents.data_agent — Extracted tickers: {'TSLA'}
[2024-01-15 12:00:02] INFO  app.tools.financial_tool — TSLA: price=185.40, change=-3.21%
[2024-01-15 12:00:02] INFO  app.rag.vector_store — Upserted 1 docs (source_tag=market_data)
[2024-01-15 12:00:03] INFO  app.agents.news_agent — Fetched 6 articles
[2024-01-15 12:00:03] INFO  app.rag.retriever — RAG context built: 4 docs, ~380 tokens
[2024-01-15 12:00:05] INFO  app.agents.analysis_agent — Done — latency=1.82s | tokens=620
[2024-01-15 12:00:05] INFO  app.agents.supervisor_agent — Pipeline complete — total=3.21s
```

---

## 💰 Cost Optimisation

- **Embedding caching** — ChromaDB deduplicates via content hash; identical documents are never re-embedded.
- **Small model** — `gpt-4o-mini` instead of GPT-4 (≈10× cheaper per token).
- **Token-efficient prompts** — Top-5 news articles only; market data is compacted to a single text block.
- **Local fallback embeddings** — `all-MiniLM-L6-v2` via sentence-transformers if no OpenAI key (zero API cost).

---

## 🔐 Security

- Input sanitisation strips control characters.
- Query length capped at 500 characters.
- 8 regex patterns block common prompt injection attempts.
- API rate limiting via `slowapi` (configurable per minute).
- No secrets in code — all sensitive values via environment variables.

---

## 🛠️ Extending the System

- **Add a ticker** — just add to `_ALIAS_MAP` in `data_agent.py`.
- **Add a news source** — add URL to `RSS_FEEDS` in `news_tool.py`.
- **Add an agent** — create a new node function and wire it into the LangGraph in `supervisor_agent.py`.
- **Switch LLM** — change `LLM_MODEL` in `.env` (any OpenAI-compatible model).
- **Use a cloud vector DB** — swap `chromadb.PersistentClient` for Pinecone/Weaviate client in `vector_store.py`.
