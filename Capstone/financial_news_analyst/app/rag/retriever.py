"""
app/rag/retriever.py
---------------------
High-level RAG retrieval helper used by the Analysis Agent.

Wraps VectorStore.query() and formats retrieved documents
into a prompt-ready context block with source citations.
"""

from __future__ import annotations
from app.rag.vector_store import get_vector_store
from app.core.config import settings
from app.core.logger import get_logger, estimate_tokens

logger = get_logger(__name__)


def retrieve_context(query: str, k: int | None = None) -> dict:
    """
    Retrieve top-k documents from the vector store.

    Returns:
        {
          "context_text": str,       # formatted block ready for LLM prompt
          "sources":      list[str], # human-readable source labels
          "doc_count":    int,
          "token_estimate": int,
        }
    """
    k = k or settings.TOP_K_RESULTS
    store = get_vector_store()
    docs = store.query(query, k=k)

    if not docs:
        return {
            "context_text": "No historical context available.",
            "sources": [],
            "doc_count": 0,
            "token_estimate": 0,
        }

    # Build numbered context block
    sections = []
    seen: set[str] = set()
    sources = []
    for i, doc in enumerate(docs, start=1):
        tag = doc["source_tag"]
        date = doc["ingested_at"][:10] if doc["ingested_at"] else "unknown date"
        sections.append(
            f"[Context {i} | {date}]\n{doc['text']}"
        )
        label = f"{tag} ({date})"
        if label not in seen:
            seen.add(label)
            sources.append(label)

    context_text = "\n\n".join(sections)
    token_est = estimate_tokens(context_text)

    logger.info(
        f"RAG context built: {len(docs)} docs, ~{token_est} tokens"
    )

    return {
        "context_text": context_text,
        "sources": sources,
        "doc_count": len(docs),
        "token_estimate": token_est,
    }
