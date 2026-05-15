# Financial News Analyst — Architecture Blueprint

## 1. Problem Statement

Real-time financial research requires synthesising live market prices, breaking news,
and historical context into actionable insights. Manual aggregation is slow and
error-prone. This system automates the full pipeline using a multi-agent GenAI
architecture: dedicated agents for market data, news retrieval, and LLM-powered
analysis — connected through a RAG knowledge base and served via a secure REST API
with a Streamlit front-end.

---

## 2. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Orchestration | Custom async Python (sequential agents) | LangGraph overhead not justified for 3-agent linear pipeline; pure async is transparent and debuggable |
| LLM | OpenAI `gpt-4o-mini` | Best cost/quality ratio for structured JSON output; falls back to local stub when key absent |
| Embeddings | OpenAI `text-embedding-3-small` / `all-MiniLM-L6-v2` (local) | Dual-mode: cloud quality or fully offline |
| Vector Store | Qdrant (disk-persistent) | Free, runs locally, survives restarts, cosine similarity search |
| API Framework | FastAPI + Uvicorn | Async-native, auto OpenAPI docs, Pydantic validation |
| UI | Streamlit 1.57 | Rapid iteration; cookie-based session persistence |
| Database | SQLite (`app.db`) | Zero-ops; stores users + conversation history + token usage |
| Auth | JWT HS256 (PyJWT) + bcrypt | Stateless tokens, industry-standard password hashing |
| Market Data | Finnhub API via MCP server | Free tier, real-time quotes + company profiles |
| News | NewsAPI + 11 RSS feeds via MCP server | Dual fallback; completely free when NewsAPI quota exhausted |
| Observability | Prometheus + psutil | Standard metrics export; scrape-ready `/metrics` endpoint |
| Rate Limiting | SlowAPI + in-memory fallback (Redis optional) | Per-IP limiting; Redis upgrades to durable distributed limiting |
| Caching | `cachetools.TTLCache` + Redis (optional) | LLM response deduplication; configurable TTL |
| MCP Protocol | `mcp[cli]` stdio transport | Decouples data tools from agent code; swappable without refactor |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (:8501)                      │
│  Login/Register · Analyze tab · History tab · Admin panel   │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTPS + JWT Bearer
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI REST API (:8000)                        │
│  /auth/*  /analyze  /history  /usage  /admin/*  /metrics    │
│  ├── JWT auth middleware (deps.py)                          │
│  ├── SlowAPI rate limiter                                   │
│  ├── Prometheus metrics middleware                          │
│  └── PII redaction on persist                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Supervisor Agent (orchestrator)                 │
│  Sequential pipeline: DataAgent → NewsAgent → AnalysisAgent │
└──────┬──────────────────┬───────────────────────┬───────────┘
       │                  │                       │
       ▼                  ▼                       ▼
┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐
│  Data Agent  │  │  News Agent  │  │    Analysis Agent       │
│  MCP stdio   │  │  MCP stdio   │  │  RAG retrieve → LLM    │
└──────┬───────┘  └──────┬───────┘  └──────────┬─────────────┘
       │                  │                     │
       ▼                  ▼                     ▼
┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐
│ market_server│  │ news_server  │  │  Qdrant VectorStore     │
│ (Finnhub API)│  │ (NewsAPI+RSS)│  │  (disk-persistent)      │
└──────────────┘  └──────────────┘  └────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  SQLite  app.db                              │
│  tables: users · conversations                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Agent Roles & Data Flow

```
User query
    │
    ▼
[Input validation & prompt-injection scan]  ← security.py
    │
    ▼
[DataAgent]
    • Extracts ticker symbols via heuristics + alias map
    • Calls MCP market_server → Finnhub: price, profile, market cap
    │
    ▼
[NewsAgent]
    • Builds focused search query from user question + tickers
    • Calls MCP news_server → NewsAPI (primary) + 11 RSS feeds (fallback)
    │
    ▼
[AnalysisAgent]
    • Upserts market + news text into Qdrant (PII-redacted when no consent)
    • Retrieves top-k similar chunks (cosine ≥ RAG_MIN_SIMILARITY)
    • Calls GPT-4o-mini with structured JSON schema
    • Runs RAG-based hallucination verification (verify.py)
    • Caches result (TTL configurable)
    │
    ▼
[API response]  →  persist to app.db  →  Streamlit render
```

---

## 5. MCP Tool Selections & Rationale

| MCP Server | Tools exposed | Why MCP |
|---|---|---|
| `market_server.py` | `get_market_data(ticker)` | Isolates Finnhub SDK; agent calls tool without knowing API details |
| `news_server.py` | `fetch_financial_news(query, max_articles)` | Hides NewsAPI + RSS complexity; agent gets normalised article list |

Both servers run as stdio subprocesses started per-request. Replacing a data source
(e.g. switching from Finnhub to Alpha Vantage) requires only changing the MCP server,
not the agent code.

---

## 6. Non-Functional Requirements Coverage

### 📊 Observability & Monitoring

| NFR | Implementation | File(s) |
|---|---|---|
| **LLM Tracing** | `llm_span()` context manager records `span_id`, `trace_id`, `latency_s`, `tokens` per call. Trace ID propagated through the full request via `ContextVar` and returned in the API response header `X-Trace-Id`. | `app/observability/tracing.py`, `app/agents/analysis_agent.py` |
| **Performance Metrics** | Prometheus `Histogram` for end-to-end API latency (`api_latency_seconds`), LLM latency (`llm_latency_seconds`), and counters for requests/errors per endpoint (`api_requests_total`, `api_errors_total`). Scraped at `/metrics`. | `app/observability/metrics.py`, `app/api/app.py` |
| **Error Tracking** | Structured logging via Python `logging` with PII-redacting filter. `api_errors_total` Prometheus counter distinguishes `server_error` vs `rate_limited`. All agent exceptions caught and logged with `exc_info=True`. | `app/core/logger.py`, `app/api/app.py` |
| **User Feedback** | `PATCH /history/{id}/feedback` accepts rating (1–5) and free-text feedback. Stored in `conversations.user_rating / feedback_text` columns. Feedback UI rendered per conversation in History tab. | `app/core/history.py`, `app/api/routes.py`, `ui/streamlit_app.py` |
| **Resource Usage** | Background coroutine samples CPU % and RAM (RSS MB) every 5 s via `psutil` and updates Prometheus `Gauge` metrics (`system_cpu_percent`, `process_memory_mb`). | `app/observability/resource.py`, `app/observability/metrics.py` |

---

### 🔒 Security & Safety

| NFR | Implementation | File(s) |
|---|---|---|
| **Input Validation** | `validate_query()` strips control chars, enforces 500-char limit, and blocks 10+ prompt-injection patterns (jailbreak, role override, etc.) with regex. Pydantic models validate all API request bodies. | `app/core/security.py`, `app/api/routes.py` |
| **Content Filtering** | `moderate_text()` — local heuristic (PII exfiltration, violent keywords) with optional OpenAI Moderation API upgrade. Applied on RAG document upsert AND on LLM output before returning. Controlled by `ENABLE_CONTENT_MODERATION`. | `app/core/security.py`, `app/rag/vector_store.py`, `app/agents/analysis_agent.py` |
| **Privacy Protection** | Conservative regex-based PII redaction (`pii.py`): email, SSN, credit card, API keys. Applied to article text before persistence. System-generated fields (source labels, analysis JSON) are excluded to avoid false positives. Consent flag stored per conversation. | `app/core/pii.py`, `app/core/history.py` |
| **Access Control** | JWT HS256 tokens issued on login (`/auth/login`). `require_auth_principal` FastAPI dependency enforces Bearer token on all protected endpoints. Admin role (`ADMIN_USERNAMES`) gates user management endpoints. Blocked accounts rejected at login before password check. | `app/api/deps.py`, `app/api/auth_routes.py`, `app/core/users_store.py` |
| **Rate Limiting** | SlowAPI per-IP limiter (`RATE_LIMIT_PER_MINUTE`). Login endpoint: 12/min. Registration: 5/min. Optional Redis-backed middleware for distributed deployments. Per-user token quota (`TOKEN_QUOTA_PER_KEY`). | `app/api/limits.py`, `app/api/app.py`, `app/core/quota.py` |

---

### ✓ RAG Quality Assurance

| NFR | Implementation | File(s) |
|---|---|---|
| **Retrieval Accuracy** | Qdrant cosine similarity with `score_threshold = RAG_MIN_SIMILARITY` (default 0.28) filters low-quality matches before they reach the LLM prompt. Top-K configurable via `TOP_K_RESULTS`. | `app/rag/vector_store.py`, `app/core/config.py` |
| **Answer Relevance** | `verify_texts()` splits LLM output (`summary`, `insight`) into sentences, re-queries the vector store per claim, and computes a `hallucination_score` (0 = fully grounded, 1 = unsupported). Score surfaced in API response and UI badge. | `app/rag/verify.py`, `app/agents/analysis_agent.py` |
| **Source Attribution** | Every retrieved chunk carries `source_tag` (e.g. `market:TSLA`, `news`) and `ingested_at` date. The LLM system prompt instructs citation of URLs from context. `rag_sources` list returned in API response and rendered as chips in UI. | `app/rag/retriever.py`, `app/agents/analysis_agent.py` |
| **Hallucination Detection** | See *Answer Relevance* above. `hallucination_score` propagated from `AnalysisAgent` → `SupervisorAgent` → API response → Streamlit metric tile (🟢/🟡/🔴 thresholds at 0.3/0.6). | `app/rag/verify.py`, `app/agents/supervisor_agent.py`, `ui/streamlit_app.py` |
| **Bias Assessment** | **Not implemented.** Financial analysis bias (e.g. sentiment skewed toward bull/bear) would require a specialised financial-domain classifier or a second LLM call for cross-checking — disproportionate cost for a capstone project. The `disclaimer` field in every response and the "not financial advice" notice mitigate the impact. |  |

---

### 💰 Cost & Resource Management

| NFR | Implementation | File(s) |
|---|---|---|
| **Local-First Architecture** | When `OPENAI_API_KEY` is absent, the system uses `all-MiniLM-L6-v2` (SentenceTransformer, CPU) for embeddings and returns a stub analysis — fully functional data pipeline with no cloud dependency. News fetches 11 RSS feeds for free when NewsAPI quota is exhausted. | `app/rag/vector_store.py`, `app/agents/analysis_agent.py`, `app/mcp_servers/news_server.py` |
| **Free Tier Optimization** | Default model is `gpt-4o-mini` (lowest OpenAI cost). Finnhub free tier used. NewsAPI free tier with RSS fallback. Qdrant runs on local disk. SQLite requires no server. | `app/core/config.py` |
| **Efficient Processing** | LLM responses cached by SHA-256 hash of `(model, system_prompt, user_prompt)` using `TTLCache` (configurable TTL via `CACHE_TTL`). Cache hit logged and returned without an API call. | `app/core/cache.py`, `app/agents/analysis_agent.py` |
| **Scalability** | FastAPI is async; multiple concurrent requests are handled by the event loop. SlowAPI + optional Redis enforce fair per-user limits. Token quota (`TOKEN_QUOTA_PER_KEY`) prevents single-user overconsumption. **Note:** MCP servers are spawned as subprocesses per-request — this limits high-concurrency throughput. A connection-pool approach would be the next optimisation. | `app/api/app.py`, `app/core/quota.py` |
| **Data Management** | `purge_older_than(days)` deletes conversations older than `HISTORY_RETENTION_DAYS`. Runs as a daily background asyncio task at startup. Consent-aware PII redaction minimises personal data at rest. | `app/core/history.py`, `app/api/app.py` |

---

### ⚖️ Compliance & Ethics

| NFR | Implementation | File(s) |
|---|---|---|
| **Industry Standards** | Every LLM response includes a mandatory `disclaimer: "This is not financial advice."` enforced in the system prompt and validated in the response schema. UI renders the disclaimer in a warning box on every result. | `app/agents/analysis_agent.py`, `ui/streamlit_app.py` |
| **Transparency** | `GET /auth/config` exposes auth configuration to the UI. `agent_log` field in every response shows the full agent execution trace (timings, token counts, sentiment). Hallucination score and RAG source chips make reasoning visible to the user. | `app/api/auth_routes.py`, `app/agents/supervisor_agent.py`, `ui/streamlit_app.py` |
| **Consent Management** | `consent` checkbox in the Analyze form. When unchecked, article text is PII-redacted before storage. Consent flag persisted with the conversation row. `erase_by_trace_id` endpoint enables right-to-erasure workflows. | `app/core/history.py`, `app/api/routes.py`, `ui/streamlit_app.py` |
| **Audit Trail** | Every conversation stored in SQLite with `username`, `created_at`, `trace_id`, `token_total`, `latency_s`. Trace ID propagated through logs and response headers. Admin panel shows per-user token consumption. | `app/core/history.py`, `app/observability/tracing.py`, `ui/streamlit_app.py` |
| **Graceful Degradation** | No `OPENAI_API_KEY` → stub analysis returned (no crash). NewsAPI failure → RSS feeds used automatically. Qdrant empty → pipeline continues with "no historical context". Redis unavailable → in-memory fallback for rate limiting and caching. Any agent exception is caught, logged, and returns HTTP 500 with a descriptive message rather than an unhandled crash. | `app/agents/analysis_agent.py`, `app/mcp_servers/news_server.py`, `app/core/quota.py`, `app/core/cache.py` |

---

## 7. Project Structure

```
financial_news_analyst/
├── main.py                     # Uvicorn entry point
├── run.py                      # Parallel launcher (FastAPI + Streamlit)
├── app/
│   ├── agents/
│   │   ├── supervisor_agent.py # Orchestration
│   │   ├── data_agent.py       # Ticker extraction + market MCP
│   │   ├── news_agent.py       # News MCP
│   │   └── analysis_agent.py   # RAG + LLM + verification
│   ├── api/
│   │   ├── app.py              # FastAPI factory + middleware
│   │   ├── routes.py           # All REST endpoints
│   │   ├── auth_routes.py      # /auth/login, /register, /me
│   │   ├── deps.py             # JWT dependency injection
│   │   └── limits.py           # Shared SlowAPI limiter
│   ├── core/
│   │   ├── config.py           # Settings from env vars
│   │   ├── history.py          # Conversation persistence
│   │   ├── users_store.py      # User accounts (bcrypt)
│   │   ├── jwt_tokens.py       # Token create/decode
│   │   ├── security.py         # Validation + moderation
│   │   ├── pii.py              # PII detection/redaction
│   │   ├── cache.py            # TTLCache + Redis adapter
│   │   ├── quota.py            # Rate limit + token quota
│   │   └── logger.py           # Structured logging + PII filter
│   ├── rag/
│   │   ├── vector_store.py     # Qdrant wrapper
│   │   ├── retriever.py        # Context retrieval + formatting
│   │   └── verify.py           # Hallucination scoring
│   ├── mcp_servers/
│   │   ├── market_server.py    # Finnhub MCP tool
│   │   └── news_server.py      # NewsAPI + RSS MCP tool
│   └── observability/
│       ├── metrics.py          # Prometheus counters/histograms
│       ├── resource.py         # CPU/RAM background sampler
│       └── tracing.py          # Trace ID context var + llm_span
└── ui/
    └── streamlit_app.py        # Full-stack UI (login, analyze, history, admin)
```

---

## 8. Key Configuration (`.env`)

```env
# Required
OPENAI_API_KEY=...
FINNHUB_API_KEY=...
JWT_SECRET=<at-least-32-random-chars>

# Auth bootstrap (admin account created on startup)
AUTH_BOOTSTRAP_USERNAME=admin
AUTH_BOOTSTRAP_PASSWORD=<min-8-chars>
ADMIN_USERNAMES=admin

# Registration
ALLOW_USER_REGISTRATION=true

# Optional tuning
LLM_MODEL=gpt-4o-mini
RAG_MIN_SIMILARITY=0.28
TOP_K_RESULTS=4
RATE_LIMIT_PER_MINUTE=10
TOKEN_QUOTA_PER_KEY=0        # 0 = unlimited
CACHE_TTL=300                # seconds
HISTORY_RETENTION_DAYS=90
```
