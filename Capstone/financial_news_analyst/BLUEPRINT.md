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
| RAG Evaluation | Custom `evaluation.py` (cosine, MRR, ROUGE-1, bias) | No external eval framework needed; runs against live Qdrant store |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI (:8501)                        │
│  Login/Register · Analyze · History · Admin (Users + RAG Eval)  │
└───────────────────────┬─────────────────────────────────────────┘
                        │ HTTP + JWT Bearer
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                FastAPI REST API (:8000)                          │
│  /auth/*  /analyze  /history  /usage                            │
│  /rag/stats  /rag/evaluate  /rag/evaluate/history               │
│  /admin/users/*  /metrics                                       │
│  ├── JWT auth middleware (deps.py)                              │
│  ├── SlowAPI rate limiter                                       │
│  ├── Prometheus metrics middleware                              │
│  └── PII redaction on persist                                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              Supervisor Agent (orchestrator)                     │
│  Sequential pipeline: DataAgent → NewsAgent → AnalysisAgent     │
└──────┬──────────────────┬───────────────────────┬───────────────┘
       │                  │                       │
       ▼                  ▼                       ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐
│  Data Agent  │  │  News Agent  │  │      Analysis Agent        │
│  MCP stdio   │  │  MCP stdio   │  │  RAG → LLM → verify.py    │
└──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘
       │                  │                     │
       ▼                  ▼                     ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐
│ market_server│  │ news_server  │  │  Qdrant VectorStore        │
│ (Finnhub API)│  │ (NewsAPI+RSS)│  │  (disk-persistent)         │
└──────────────┘  └──────────────┘  └──────────┬───────────────┘
                                                │
                                                ▼
                                    ┌──────────────────────────┐
                                    │   evaluation.py           │
                                    │  Precision@K · MRR        │
                                    │  Cosine Sim · ROUGE-1     │
                                    │  Traceability · Bias      │
                                    └──────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      SQLite  app.db                              │
│  tables: users · conversations                                  │
└─────────────────────────────────────────────────────────────────┘
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
    │
    ▼
[RAG Evaluation]  ← evaluation.py  (async, best-effort)
    • Retrieval accuracy: Precision@K, MRR, source diversity
    • Answer relevance:  query-answer cosine similarity, token overlap
    • Source attribution: traceability score, citation rate
    • Hallucination:     claim-level support via verify.py
    • Bias:              lexical imbalance + historical sentiment distribution
    │
    ▼
[Composite quality score]  →  UI expander  /  Admin RAG Eval tab
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
| **User Feedback** | `PATCH /history/{id}/feedback` accepts rating (1–5) and free-text feedback. Stored in `conversations.user_rating / feedback_text` columns. Feedback form rendered at the bottom of every Analyze result. History tab displays submitted feedback as a formatted section (star rating metric + info box). Admin Feedback tab aggregates all responses with avg rating. | `app/core/history.py`, `app/api/routes.py`, `ui/streamlit_app.py` |
| **Resource Usage** | Background coroutine samples CPU % and RAM (RSS MB) every 5 s via `psutil` and updates Prometheus `Gauge` metrics (`system_cpu_percent`, `process_memory_mb`). | `app/observability/resource.py`, `app/observability/metrics.py` |
| **Test Coverage** | 174 automated tests across 6 files: input validation, ticker extraction, output structure, RAG store, API endpoint integration (auth mocked via FastAPI dependency override), auth flows, per-user history isolation, feedback, PII redaction, user management, and all 5 RAG evaluation dimensions. Async pipeline tests use `AsyncMock`. SlowAPI rate limiter disabled via `limiter.enabled = False` in autouse fixture. | `tests/` |

---

### 🔒 Security & Safety

| NFR | Implementation | File(s) |
|---|---|---|
| **Input Validation** | `validate_query()` strips control chars, enforces 500-char limit, and blocks 10+ prompt-injection patterns (jailbreak, role override, etc.) with regex. Pydantic models validate all API request bodies. | `app/core/security.py`, `app/api/routes.py` |
| **Content Filtering** | `moderate_text()` — local heuristic (PII exfiltration, violent keywords) with optional OpenAI Moderation API upgrade. Applied on RAG document upsert AND on LLM output before returning. Controlled by `ENABLE_CONTENT_MODERATION`. | `app/core/security.py`, `app/rag/vector_store.py`, `app/agents/analysis_agent.py` |
| **Privacy Protection** | Conservative regex-based PII redaction (`pii.py`): email, SSN, credit card, API keys. Applied to article text before persistence. System-generated fields (source labels, analysis JSON) are excluded to avoid false positives. Consent flag stored per conversation. | `app/core/pii.py`, `app/core/history.py` |
| **Access Control** | JWT HS256 tokens issued on login. Authentication always required (`AUTH_ENABLED = True`). `require_auth_principal` / `require_admin` FastAPI dependencies gate all endpoints. Admin role (`ADMIN_USERNAMES`) controls user management. Blocked accounts rejected at login before password check. Cookie-based session persistence across browser refreshes. | `app/api/deps.py`, `app/api/auth_routes.py`, `app/core/users_store.py`, `ui/streamlit_app.py` |
| **Rate Limiting** | SlowAPI per-IP limiter (`RATE_LIMIT_PER_MINUTE`). Login: 12/min. Registration: 5/min. Optional Redis-backed middleware for distributed deployments. Per-user token quota (`TOKEN_QUOTA_PER_KEY`). | `app/api/limits.py`, `app/api/app.py`, `app/core/quota.py` |

---

### ✓ RAG Quality Assurance

| NFR | Implementation | File(s) |
|---|---|---|
| **Retrieval Accuracy** | Qdrant cosine similarity with `score_threshold = RAG_MIN_SIMILARITY` (default 0.28) filters low-quality matches. `evaluation.py` computes **Precision@K** (fraction of docs above threshold), **MRR** (Mean Reciprocal Rank), and **source diversity** per query. Aggregate trends available in Admin → RAG Evaluation tab. | `app/rag/vector_store.py`, `app/rag/evaluation.py`, `app/core/config.py` |
| **Answer Relevance** | `evaluate_answer_relevance()` embeds both the query and the generated answer, computes **cosine similarity**, and measures **ROUGE-1 token overlap**. Returns a `relevance_label` (high / medium / low). Shown in the per-analysis evaluation expander. | `app/rag/evaluation.py`, `app/api/routes.py` |
| **Source Attribution** | `evaluate_source_attribution()` computes a **traceability score**: fraction of LLM-cited URLs that came from retrieved context + coverage of source tags in the answer text. Every chunk carries `source_tag` + `ingested_at`; `rag_sources` rendered as chips in UI. | `app/rag/evaluation.py`, `app/rag/retriever.py`, `ui/streamlit_app.py` |
| **Hallucination Detection** | `verify_texts()` splits summary + insight into sentences, re-queries Qdrant per claim, and returns `hallucination_score` [0 = grounded, 1 = unsupported]. `evaluate_hallucination()` adds `claims_supported` count and a human label. Score shown as 🟢/🟡/🔴 tile in UI and tracked in Admin aggregate report. | `app/rag/verify.py`, `app/rag/evaluation.py`, `app/agents/analysis_agent.py`, `ui/streamlit_app.py` |
| **Bias Assessment** | `evaluate_bias()` counts bullish/bearish keyword occurrences and computes a **lexical imbalance score** [0 = balanced, 1 = one-sided]. Also analyses **historical sentiment distribution** across all conversations and flags if any sentiment exceeds 70 % of responses. Results visible in per-analysis expander and Admin RAG Evaluation tab. | `app/rag/evaluation.py`, `ui/streamlit_app.py` |

---

### 💰 Cost & Resource Management

| NFR | Implementation | File(s) |
|---|---|---|
| **Local-First Architecture** | When `OPENAI_API_KEY` is absent, the system uses `all-MiniLM-L6-v2` (SentenceTransformer, CPU) for embeddings and returns a stub analysis — fully functional data pipeline with no cloud dependency. News fetches 11 RSS feeds for free when NewsAPI quota is exhausted. | `app/rag/vector_store.py`, `app/agents/analysis_agent.py`, `app/mcp_servers/news_server.py` |
| **Free Tier Optimization** | Default model is `gpt-4o-mini` (lowest OpenAI cost). Finnhub free tier. NewsAPI free tier with RSS fallback. Qdrant on local disk. SQLite requires no server. Token usage calculated from persisted SQLite history — no Redis required. | `app/core/config.py`, `app/core/history.py` |
| **Efficient Processing** | LLM responses cached by SHA-256 hash of `(model, system_prompt, user_prompt)` using `TTLCache` (configurable TTL). Cache hit returned without an API call. RAG evaluation runs post-response (best-effort, timeout 30 s) to avoid blocking the user. | `app/core/cache.py`, `app/agents/analysis_agent.py`, `ui/streamlit_app.py` |
| **Scalability** | FastAPI is async; multiple concurrent requests handled by the event loop. SlowAPI + optional Redis enforce fair per-user limits. Token quota prevents single-user overconsumption. **Note:** MCP servers are spawned as subprocesses per-request — connection pooling is the next optimisation for high concurrency. | `app/api/app.py`, `app/core/quota.py` |
| **Data Management** | `purge_older_than(days)` deletes conversations older than `HISTORY_RETENTION_DAYS`. Runs daily as a background asyncio task. Token usage derived from `SUM(token_total)` in SQLite — survives restarts. Consent-aware PII redaction minimises personal data at rest. | `app/core/history.py`, `app/api/app.py` |

---

### ⚖️ Compliance & Ethics

| NFR | Implementation | File(s) |
|---|---|---|
| **Industry Standards** | Every LLM response includes a mandatory `disclaimer: "This is not financial advice."` enforced in the system prompt. UI renders the disclaimer in a warning box on every result. | `app/agents/analysis_agent.py`, `ui/streamlit_app.py` |
| **Transparency** | Analysis results are rendered in a fixed, readable order: Tickers → Sentiment → Summary → Key Drivers/Risk Factors → Educational Insight → LLM-cited sources → Disclaimer → RAG Quality Evaluation → Agent Execution Trace (includes total pipeline latency as the final line) → Rate this analysis. Hallucination score and composite RAG quality visible in the evaluation expander. `GET /auth/config` exposes auth state to the UI. | `app/api/auth_routes.py`, `app/agents/supervisor_agent.py`, `ui/streamlit_app.py` |
| **Consent Management** | `consent` checkbox in the Analyze form. When unchecked, article text is PII-redacted before storage; system-generated fields are preserved. Consent flag persisted per conversation. `erase_by_trace_id` endpoint enables right-to-erasure workflows. | `app/core/history.py`, `app/api/routes.py`, `ui/streamlit_app.py` |
| **Audit Trail** | Every conversation stored with `username`, `created_at`, `trace_id`, `token_total`, `latency_s`. Token usage aggregated per user from persistent SQLite (survives restarts). Admin panel shows per-user consumption. Trace ID propagated through logs and response headers. | `app/core/history.py`, `app/observability/tracing.py`, `ui/streamlit_app.py` |
| **Graceful Degradation** | No `OPENAI_API_KEY` → stub analysis (pipeline still runs). NewsAPI failure → RSS feeds. Qdrant empty → "no historical context". Redis unavailable → in-memory fallback. RAG evaluation failure → silently skipped, main response unaffected. Any agent exception returns HTTP 500 with a descriptive message. | `app/agents/analysis_agent.py`, `app/mcp_servers/news_server.py`, `app/core/quota.py`, `app/rag/evaluation.py` |

---

## 7. RAG Evaluation Details

`POST /rag/evaluate` — runs on each analysis result (called from UI after pipeline completes).
`GET  /rag/evaluate/history?sample=N` — admin endpoint, aggregates metrics over N recent conversations.

### Metrics & Computation

| Metric | Signal | Formula |
|---|---|---|
| **Precision@K** | Retrieval accuracy | `relevant_docs / k`, where relevant = cosine score ≥ 0.35 |
| **MRR** | Ranking quality | `1 / rank_of_first_relevant_doc` |
| **Source Diversity** | Coverage breadth | Count of distinct `source_tag` prefixes (market / news / pipeline) |
| **Cosine Similarity** | Answer relevance | `dot(q_emb, a_emb) / (‖q‖·‖a‖)` using same embedder as retrieval |
| **Token Overlap** | Lexical relevance | ROUGE-1 recall: `|query_tokens ∩ answer_tokens| / |query_tokens|` |
| **Traceability Score** | Source attribution | `citation_rate × 0.6 + source_coverage × 0.4` |
| **Hallucination Score** | Grounding | `1 − mean(max_support_per_claim)` via Qdrant re-query |
| **Lexical Imbalance** | Bias | `|bullish_kw − bearish_kw| / (bullish_kw + bearish_kw)` |
| **Composite Score** | Overall quality | `retrieval×0.25 + relevance×0.25 + attribution×0.20 + grounding×0.30 − bias×0.10` |

---

## 8. Project Structure

```
financial_news_analyst/
├── main.py                     # Uvicorn entry point
├── run.py                      # Parallel launcher (FastAPI + Streamlit)
├── architecture.png            # System architecture diagram
├── BLUEPRINT.md                # This document
├── app/
│   ├── agents/
│   │   ├── supervisor_agent.py # Orchestration + pipeline state
│   │   ├── data_agent.py       # Ticker extraction + market MCP
│   │   ├── news_agent.py       # News MCP
│   │   └── analysis_agent.py   # RAG + LLM + verification + caching
│   ├── api/
│   │   ├── app.py              # FastAPI factory + middleware
│   │   ├── routes.py           # All REST endpoints incl. /rag/evaluate
│   │   ├── auth_routes.py      # /auth/login, /register, /me, /config
│   │   ├── deps.py             # JWT dependency injection + require_admin
│   │   └── limits.py           # Shared SlowAPI limiter
│   ├── core/
│   │   ├── config.py           # Settings (all env vars, AUTH_ENABLED=True)
│   │   ├── history.py          # Conversation persistence + get_user_tokens
│   │   ├── users_store.py      # User accounts, block/unblock, bcrypt
│   │   ├── jwt_tokens.py       # Token create/decode (HS256)
│   │   ├── security.py         # Input validation + content moderation
│   │   ├── pii.py              # PII detection/redaction
│   │   ├── cache.py            # TTLCache + Redis adapter
│   │   ├── quota.py            # Rate limit + token quota
│   │   └── logger.py           # Structured logging + PII filter
│   ├── rag/
│   │   ├── vector_store.py     # Qdrant wrapper (score_threshold applied)
│   │   ├── retriever.py        # Context retrieval + source formatting
│   │   ├── verify.py           # Claim-level hallucination scoring
│   │   └── evaluation.py       # Full RAG evaluation (5 dimensions)
│   ├── mcp_servers/
│   │   ├── market_server.py    # Finnhub MCP tool
│   │   └── news_server.py      # NewsAPI + 11 RSS feeds MCP tool
│   └── observability/
│       ├── metrics.py          # Prometheus counters/histograms/gauges
│       ├── resource.py         # CPU/RAM background sampler (psutil)
│       └── tracing.py          # Trace ID ContextVar + llm_span
└── ui/
    └── streamlit_app.py        # Full-stack UI:
                                #   Login/Register (JWT cookie persistence)
                                #   Analyze + RAG Evaluation expander
                                #   History + feedback
                                #   Admin: User Mgmt + RAG Eval aggregate
```

---

## 9. Key Configuration (`.env`)

```env
# Required
OPENAI_API_KEY=...
FINNHUB_API_KEY=...
JWT_SECRET=<at-least-32-random-chars>

# Auth — always required (AUTH_ENABLED is hardcoded True)
AUTH_BOOTSTRAP_USERNAME=admin
AUTH_BOOTSTRAP_PASSWORD=<min-8-chars>
ADMIN_USERNAMES=admin           # comma-separated for multiple admins

# Registration
ALLOW_USER_REGISTRATION=true

# Optional tuning
LLM_MODEL=gpt-4o-mini
RAG_MIN_SIMILARITY=0.28
TOP_K_RESULTS=4
RATE_LIMIT_PER_MINUTE=10
TOKEN_QUOTA_PER_KEY=0           # 0 = unlimited
CACHE_TTL=300                   # seconds
HISTORY_RETENTION_DAYS=90
ENABLE_CONTENT_MODERATION=true

# Optional infrastructure
REDIS_URL=                      # leave empty for in-memory fallback
NEWS_API_KEY=                   # leave empty to use RSS feeds only
DB_PATH=./app.db                # SQLite file (users + conversations)
QDRANT_PATH=./qdrant_db         # Vector store directory
```
