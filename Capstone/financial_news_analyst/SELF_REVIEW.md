# Self-Review — Financial News Analyst

## Overview

This document is an honest self-assessment of the Financial News Analyst capstone project:
what was built well, where shortcuts were taken, known limitations, and what I would do
differently with more time or a second iteration.

---

## What Went Well

### Architecture decisions held up under pressure
The choice to use MCP stdio transport for data tools turned out to be genuinely useful, not
just architecturally tidy. When switching the market data source from yFinance to Finnhub
mid-project, only the MCP server file changed — the agent code was untouched. This validated
the isolation principle in practice, not just in theory.

### Security was built in from the start
Authentication (JWT), input validation (prompt-injection blocking), PII redaction, content
moderation, and rate limiting were all designed in from the beginning rather than bolted on
at the end. This avoided the common pattern of "working system with security as an
afterthought." The result is that the security layer is coherent — every request travels
through the same validation chain.

### RAG quality evaluation as a first-class feature
Making the five-dimension RAG evaluation visible to the user (not just logged internally)
was the right call. It makes the system's uncertainty explicit and gives users and reviewers
a basis for trust — or scepticism — about specific outputs. Most RAG demos hide this entirely.

### Test coverage is meaningful, not cosmetic
174 tests across 6 files cover real failure modes: per-user history isolation, PII redaction
scope (false positives on dates were a real bug caught by tests), blocked-account enforcement,
JWT expiry, and hallucination score range validation. The tests were written after the feature
was built in some cases, which led to finding the `severity` variable bug in `vector_store.py`
that would have caused silent failures in production with content moderation disabled.

---

## Known Limitations

### MCP servers are spawned per-request
Every call to DataAgent or NewsAgent starts a fresh subprocess for the MCP server. This is
fine at one user, noticeably slow at ten concurrent users, and unacceptable at scale. The fix
is an MCP connection pool or switching to a persistent HTTP-based MCP transport, but that
adds complexity that was out of scope for this project.

### RAG evaluation uses proxy metrics only
None of the five evaluation dimensions use labelled ground truth — they are all proxies
(cosine similarity, ROUGE-1 overlap, keyword counting). This means the composite score is
directionally useful but not a reliable absolute measure. A hallucination score of 0.2 means
"80% of claims found supporting evidence in the vector store" — it does not mean "the output
is 80% factually correct." This distinction is not clearly communicated to the user.

### Qdrant is a single shared collection
All users write to the same Qdrant collection. This means one user's queries can influence
another user's RAG context. For a personal research tool this is acceptable, but for a
multi-tenant product it is a data isolation problem. The fix (per-user namespacing or
separate collections) was considered and deferred.

### No streaming output
The pipeline runs to completion before showing any result. A 45-second spinner with no
intermediate feedback is poor UX. Streaming — showing the agent log entries as they arrive,
then the analysis when it's ready — would dramatically improve the perceived responsiveness.
FastAPI supports streaming responses; Streamlit supports `st.write_stream`. This was scoped
out due to time constraints.

### Phone number PII regex was a late fix
The initial phone regex was too broad and matched financial date patterns like `2026-05-15`,
causing `[REDACTED_PHONE]` to appear in RAG source labels. This was caught visually during
testing, not by automated tests. The fix was straightforward (tighter US-format regex) but
the root cause was inadequate test coverage for PII false positives at the time the feature
was written. The regression tests were added after the fact.

---

## What I Would Do Differently

### Write PII tests before the PII module
The false-positive phone bug would have been caught immediately if the test suite had been
written first. The `test_pii.py` file now contains explicit false-positive regression tests
for dates, source labels, and version numbers — these should have existed from day one.

### Separate the embedding concern from VectorStore
`VectorStore` currently owns both the Qdrant client and the embedding model. This made the
RAG evaluation tests awkward to write (had to mock `store._embed` via `get_vector_store`).
A cleaner design would inject the embedder as a dependency, making both the store and the
evaluation module independently testable without mocking internal methods.

### Use `asyncio.gather` for parallel agent execution
The supervisor runs DataAgent and NewsAgent sequentially. They are independent — market data
fetch does not depend on news, and vice versa. Running them in parallel with
`asyncio.gather` would cut the pipeline latency roughly in half for the data-fetching phase.
This was a deliberate simplification for transparency during development that should be
revisited before any production use.

### Plan the database schema before writing history.py
The `conversations` table was extended three times with `ALTER TABLE` migrations: first for
`user_rating` and `feedback_text`, then for `username`. SQLite's limited ALTER TABLE support
made this safe but awkward. Starting with the full schema — including username and feedback
columns — would have been cleaner and avoided the migration logic entirely.

### Add a real embedding cache
The current LLM response cache (SHA-256 keyed TTLCache) deduplicates identical queries but
does nothing for semantically similar ones. A simple nearest-neighbour cache on the query
embedding — returning a cached result if a new query is within cosine distance 0.05 of a
recent one — would reduce API calls significantly for common financial topics like "S&P 500
sentiment." This is a known optimisation pattern for RAG systems that was not implemented.

---

## Overall Assessment

The project meets its stated objectives: a working multi-agent financial research system with
authentication, RAG, quality evaluation, an admin panel, and comprehensive tests. The
architecture is sound and the core design decisions (MCP isolation, security-first, visible
quality metrics) are defensible.

The main weaknesses are in scalability (per-request MCP subprocesses, shared Qdrant
collection) and user experience (no streaming, no real-time updates). These are known
trade-offs made deliberately to keep the codebase navigable for a capstone project rather
than production engineering failures.

If this were a production system rather than a capstone, the next iteration would prioritise:
connection pooling for MCP, per-user Qdrant namespacing, streaming output, and a proper
embedding cache — in that order.
