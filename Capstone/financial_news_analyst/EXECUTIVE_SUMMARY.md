# Executive Summary — Financial News Analyst

## Problem Statement

Staying informed about financial markets requires synthesising three streams of information simultaneously: live price movements, breaking news, and historical context. Today this is done manually — an analyst opens multiple tabs, reads multiple sources, and forms a judgment under time pressure. The process is slow (hours per research session), inconsistent across individuals, and prone to gaps when relevant news is missed or context is forgotten.

The objective of this project was to automate that entire workflow using a multi-agent AI system: given a plain-language question such as *"Why did Tesla stock drop today?"*, the system fetches live market data, retrieves recent news, draws on a historical knowledge base, and returns a structured research brief — in under 60 seconds, with every claim traceable to a source.

---

## Technical Approach

The system is built on a **three-agent pipeline**, each agent with a single responsibility:

- **Data Agent** — identifies ticker symbols in the user's question and retrieves live price, volume, and company profile data from Finnhub
- **News Agent** — constructs a focused search query and fetches recent articles from NewsAPI with automatic fallback to 11 RSS feeds
- **Analysis Agent** — embeds all retrieved content into a local vector database (Qdrant), retrieves the most relevant context by semantic similarity, and calls GPT-4o-mini to generate a structured brief (sentiment, summary, key drivers, risks, and educational insight)

Each data tool runs as an **isolated subprocess** via the Model Context Protocol (MCP), meaning the underlying data provider can be swapped — for example, replacing Finnhub with Bloomberg — without touching the agent logic.

Three architectural decisions stand out:

1. **Cost-first design.** The system defaults to `gpt-4o-mini` (roughly 10× cheaper than GPT-4) and uses a local sentence-embedding model when no OpenAI key is present. Free data tiers (Finnhub, NewsAPI, RSS) cover the full data pipeline at zero marginal cost. Repeated identical queries are served from a local cache without any API call.

2. **Quality is measured, not assumed.** Every analysis result is automatically evaluated across five RAG quality dimensions: retrieval precision, answer relevance, source traceability, hallucination score, and bias assessment. A composite quality score is shown to the user, making the system's confidence explicit rather than hidden.

3. **Security and privacy by default.** Authentication is always required (JWT Bearer tokens). User-submitted text is scanned for prompt-injection attempts before reaching the pipeline. Personal data in stored articles is redacted by default; users can opt in to full storage by checking a consent checkbox.

---

## Results and Business Value

The delivered system is fully functional and production-ready in structure:

| Dimension | Outcome |
|---|---|
| **End-to-end latency** | 30–60 seconds per analysis (live data + LLM) |
| **Data coverage** | Real-time market data + up to 8 news articles per query; 11 RSS feeds as zero-cost fallback |
| **Offline resilience** | Full pipeline runs without any paid API key; local embeddings replace cloud model |
| **Quality transparency** | Hallucination score, source attribution, and composite RAG quality score on every result |
| **Access control** | Role-based (admin / user), per-user history, token quotas, blocked-account enforcement |
| **Test coverage** | 174 automated tests covering the pipeline, all API endpoints, authentication, PII redaction, and RAG evaluation |

For a business context, this system compresses a research workflow that takes a skilled analyst 1–3 hours into a 60-second automated brief. It is explainable (every claim links to a source), auditable (full conversation history with token usage per user), and extensible (adding a new data source requires only a new MCP server).

---

## Lessons Learned

**What worked well.** The MCP tool-isolation pattern proved its value: agents remained clean and testable while the data layer could be iterated independently. Building RAG quality evaluation as a first-class feature — not an afterthought — made the system's limitations visible and actionable. Starting with a security-first mindset (prompt injection blocking, PII redaction, JWT everywhere) avoided retrofitting security onto a working system.

**What was harder than expected.** Evaluating RAG quality without labelled ground truth required careful proxy design — cosine similarity, ROUGE-1 overlap, and claim-level re-querying each measure a different facet but none is definitive. PII redaction with simple regular expressions requires tight scope control; overly broad patterns silently corrupt system-generated fields like market dates and ticker symbols.

---

## Potential Next Steps

The current system is a solid foundation for several extensions of immediate business value:

- **Watchlist and portfolio tracking** — proactive alerts when monitored tickers move beyond a threshold
- **Multi-turn conversation** — follow-up questions that retain context from the previous analysis
- **Richer data sources** — SEC filings, earnings transcripts, and analyst reports via additional MCP servers
- **Production deployment** — containerisation (Docker), a managed vector database, and Redis-backed caching for multi-user scale
- **Feedback-driven improvement** — use the collected user ratings to fine-tune retrieval thresholds and prompt templates over time

The architecture was designed to accommodate these extensions without restructuring: new data sources plug in as MCP servers, new evaluation metrics extend the existing five-dimension framework, and the agent pipeline accepts additional stages without changing the orchestrator.
