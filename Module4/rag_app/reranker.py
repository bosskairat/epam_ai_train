import os
from typing import List, Optional
import numpy as np
from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    """Reranker abstraction that tries a CrossEncoder first and falls back to
    embedding-based cosine similarity when the cross-encoder is unavailable.

    Usage:
        r = Reranker()
        ranked = r.rerank(query, docs, top_k=5)
    """

    def __init__(self, reranker_name=RERANKER_MODEL_NAME):
        
        # Try loading cross-encoder
        try:
            self.reranker = CrossEncoder(reranker_name)
        except Exception as e:
            print(f"⚠️  Could not load CrossEncoder '{reranker_name}': {e}")


    def rerank(self, query: str, docs: List[str], top_k: Optional[int] = None) -> List[dict]:
        """Rerank documents given a query.

        Returns a list of dicts ordered by descending relevance with keys:
          - 'score': float
          - 'index': int (original index in `docs`)
          - 'doc': str

        If top_k is provided, returns only the top_k entries.
        """
        if not docs:
            return []

        # Use CrossEncoder when available
        if self.reranker is not None:
            try:
                pairs = [(query, d) for d in docs]
                scores = self.reranker.predict(pairs)
                ranked_indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
                results = [{"score": float(scores[i]), "index": int(i), "doc": docs[i]} for i in ranked_indices]
                return results[:top_k] if top_k is not None else results
            except Exception as e:
                print(f"⚠️  CrossEncoder prediction failed: {e}. Falling back to embedding similarity.")

