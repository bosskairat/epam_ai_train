"""
app/rag/vector_store.py
------------------------
ChromaDB-backed vector store for financial/news context.

Features:
  • Persistent storage across runs
  • Embedding caching to avoid repeated API calls
  • Top-k semantic retrieval with source attribution
"""

from __future__ import annotations
import hashlib
import json
from datetime import datetime
from typing import Optional
import chromadb

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "financial_context"


class VectorStore:
    """Thin wrapper around a ChromaDB persistent collection."""

    def __init__(self):
        # Persistent client – data survives restarts
        self._client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

        # Use OpenAI embeddings if key is set, otherwise fall back to the
        # built-in sentence-transformers model (works offline)
        if settings.OPENAI_API_KEY:
            embed_fn = OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.EMBEDDING_MODEL,
            )
            embed_type = f"openai:{settings.EMBEDDING_MODEL}"
            logger.info(f"Using OpenAI embeddings ({settings.EMBEDDING_MODEL})")
        else:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
            embed_fn = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            embed_type = "local:all-MiniLM-L6-v2"
            logger.info("Using local SentenceTransformer embeddings (offline mode)")

        # Check if collection exists and matches our embedding function
        try:
            existing_collection = self._client.get_collection(COLLECTION_NAME)
            existing_embed_type = existing_collection.metadata.get("embedding_type")
            if existing_embed_type != embed_type:
                logger.warning(
                    f"Collection embedding type mismatch: {existing_embed_type} != {embed_type}. "
                    "Deleting and recreating collection."
                )
                self._client.delete_collection(COLLECTION_NAME)
                raise ValueError("Collection recreated due to embedding type change")
        except ValueError:
            # Collection was deleted, will be recreated below
            pass
        except Exception:
            # Collection doesn't exist, will be created below
            pass

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine", "embedding_type": embed_type},
        )
        logger.info(
            f"VectorStore ready — collection '{COLLECTION_NAME}' "
            f"({self._collection.count()} docs)"
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        source_tag: str = "unknown",
    ) -> int:
        """
        Add or update documents in the collection.
        Uses a content-hash as ID so duplicates are naturally deduplicated.

        Returns the number of documents added/updated.
        """
        if not texts:
            return 0

        ids, docs, metas = [], [], []
        for i, text in enumerate(texts):
            doc_id = _hash(text)
            meta = (metadatas[i] if metadatas else {}) or {}
            meta.setdefault("source_tag", source_tag)
            meta.setdefault("ingested_at", datetime.utcnow().isoformat())
            # Chroma requires string values in metadata
            meta = {k: str(v) for k, v in meta.items()}

            ids.append(doc_id)
            docs.append(text)
            metas.append(meta)

        try:
            self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"Upserted {len(ids)} docs (source_tag={source_tag})")
            return len(ids)
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"Dimension mismatch during upsert: {e}. Recreating collection.")
                # Delete and recreate collection with correct embedding function
                self._client.delete_collection(COLLECTION_NAME)
                if settings.OPENAI_API_KEY:
                    embed_fn = OpenAIEmbeddingFunction(
                        api_key=settings.OPENAI_API_KEY,
                        model_name=settings.EMBEDDING_MODEL,
                    )
                    embed_type = f"openai:{settings.EMBEDDING_MODEL}"
                else:
                    from chromadb.utils.embedding_functions import (
                        SentenceTransformerEmbeddingFunction,
                    )
                    embed_fn = SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                    embed_type = "local:all-MiniLM-L6-v2"
                self._collection = self._client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=embed_fn,
                    metadata={"hnsw:space": "cosine", "embedding_type": embed_type},
                )
                # Retry upsert
                self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
                logger.info(f"Upserted {len(ids)} docs after collection recreation (source_tag={source_tag})")
                return len(ids)
            else:
                raise

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(self, query_text: str, k: Optional[int] = None) -> list[dict]:
        """
        Retrieve the top-k most relevant documents for a query.

        Returns list of dicts with:
          - text
          - source_tag
          - ingested_at
          - distance  (lower = more similar)
        """
        k = k or settings.TOP_K_RESULTS
        count = self._collection.count()

        if count == 0:
            logger.warning("Vector store is empty – no context retrieved")
            return []

        # Cannot request more results than exist
        k = min(k, count)

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"Dimension mismatch during query: {e}. Recreating collection.")
                # Delete and recreate collection with correct embedding function
                self._client.delete_collection(COLLECTION_NAME)
                if settings.OPENAI_API_KEY:
                    embed_fn = OpenAIEmbeddingFunction(
                        api_key=settings.OPENAI_API_KEY,
                        model_name=settings.EMBEDDING_MODEL,
                    )
                    embed_type = f"openai:{settings.EMBEDDING_MODEL}"
                else:
                    from chromadb.utils.embedding_functions import (
                        SentenceTransformerEmbeddingFunction,
                    )
                    embed_fn = SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                    embed_type = "local:all-MiniLM-L6-v2"
                self._collection = self._client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=embed_fn,
                    metadata={"hnsw:space": "cosine", "embedding_type": embed_type},
                )
                # Retry query on empty collection
                logger.warning("Collection recreated - returning empty results for query")
                return []
            else:
                raise

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        retrieved = []
        for doc, meta, dist in zip(docs, metas, dists):
            retrieved.append(
                {
                    "text": doc,
                    "source_tag": meta.get("source_tag", "unknown"),
                    "ingested_at": meta.get("ingested_at", ""),
                    "distance": round(dist, 4),
                }
            )

        logger.info(
            f"Retrieved {len(retrieved)} docs for query '{query_text[:60]}…'"
        )
        for r in retrieved:
            logger.debug(
                f"  [{r['source_tag']}] dist={r['distance']} — {r['text'][:80]}…"
            )

        return retrieved

    def count(self) -> int:
        return self._collection.count()


# ── Singleton ─────────────────────────────────────────────────────────────────
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the module-level singleton VectorStore (lazy init)."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash(text: str) -> str:
    """SHA-256 content hash, truncated to 16 hex chars."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
