"""
app/rag/vector_store.py
------------------------
Qdrant persistent vector store for financial/news context.
Data survives app restarts via local on-disk storage.
"""

from __future__ import annotations
import hashlib
from datetime import datetime, timezone
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.config import settings
from app.core.logger import get_logger
from app.observability.metrics import observe_embedding, observe_retrieval
from app.observability.tracing import get_request_id
from app.core.pii import redact_pii

logger = get_logger(__name__)

COLLECTION_NAME = "financial_context"
_OPENAI_VECTOR_SIZE = 1536   # text-embedding-3-small / ada-002
_LOCAL_VECTOR_SIZE = 384     # all-MiniLM-L6-v2


class VectorStore:
    def __init__(self):
        self._qdrant = QdrantClient(path=settings.QDRANT_PATH)
        self._embed = self._make_embedder()
        vector_size = _OPENAI_VECTOR_SIZE if settings.OPENAI_API_KEY else _LOCAL_VECTOR_SIZE

        existing = {c.name for c in self._qdrant.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"VectorStore created — Qdrant persistent at '{settings.QDRANT_PATH}', dim={vector_size}")
        else:
            count = self._qdrant.count(collection_name=COLLECTION_NAME).count
            logger.info(f"VectorStore loaded — Qdrant persistent at '{settings.QDRANT_PATH}' ({count} docs)")

    def _make_embedder(self):
        if settings.OPENAI_API_KEY:
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            model = settings.EMBEDDING_MODEL
            logger.info(f"Using OpenAI embeddings ({model})")

            def embed(texts: list[str]) -> list[list[float]]:
                # Record an embedding request metric and call provider
                try:
                    observe_embedding()
                except Exception:
                    pass
                resp = client.embeddings.create(input=texts, model=model)
                return [item.embedding for item in resp.data]
        else:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Using local SentenceTransformer embeddings (offline mode)")

            def embed(texts: list[str]) -> list[list[float]]:
                try:
                    observe_embedding()
                except Exception:
                    pass
                return st_model.encode(texts).tolist()

        return embed

    def upsert(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        source_tag: str = "unknown",
        consent: bool = False,
    ) -> int:
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return 0

        vectors = self._embed(texts)
        points = []
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            meta = (metadatas[i] if metadatas else {}) or {}
            from app.core.security import sanitize_document, moderate_text

            sanitized = sanitize_document(text)
            if settings.ENABLE_CONTENT_MODERATION:
                mod = moderate_text(sanitized)
                severity = mod.get("severity", "allow")
                if severity == "block":
                    logger.warning(f"Skipping document upsert due to moderation block (source={source_tag})")
                    continue

            # Decide stored text based on consent
            stored_text = sanitized if consent else redact_pii(sanitized)
            payload = {
                "text": stored_text,
                "source_tag": source_tag,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "pii_redacted": stored_text != sanitized,
                "moderation": severity,
                **{k: str(v) for k, v in meta.items()},
            }
            if consent:
                payload["original_text"] = sanitized

            points.append(PointStruct(id=_hash_to_id(sanitized), vector=vector, payload=payload))

        self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"Upserted {len(points)} docs (source_tag={source_tag})")
        return len(points)

    def query(self, query_text: str, k: Optional[int] = None) -> list[dict]:
        k = k or settings.TOP_K_RESULTS
        total = self.count()
        if total == 0:
            logger.warning("Vector store is empty – no context retrieved")
            return []

        [query_vector] = self._embed([query_text])
        result = self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=min(k, total),
            with_payload=True,
            score_threshold=settings.RAG_MIN_SIMILARITY,
        )
        # Emit retrieval metric
        try:
            observe_retrieval()
        except Exception:
            pass

        retrieved = []
        for h in result.points:
            payload = h.payload or {}
            # Skip documents that were explicitly blocked by moderation
            if payload.get("moderation") == "block":
                continue
            retrieved.append(
                {
                    "id": getattr(h, "id", None),
                    "text": payload.get("text", ""),
                    "payload": payload,
                    "source_tag": payload.get("source_tag", "unknown"),
                    "ingested_at": payload.get("ingested_at", ""),
                    "score": getattr(h, "score", None),
                    "distance": round(1 - (h.score or 0), 4),
                }
            )
        logger.info(f"Retrieved {len(retrieved)} docs for query '{query_text[:60]}…' | trace={get_request_id()}")
        return retrieved

    def count(self) -> int:
        result = self._qdrant.count(collection_name=COLLECTION_NAME)
        return result.count


# ── Singleton ─────────────────────────────────────────────────────────────────
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


# ── Helpers ───────────────────────────────────────────────────────────────────
def _hash_to_id(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
