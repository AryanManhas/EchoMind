from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import numpy as np
import faiss

from services.embedding_service import EmbeddingService


class SearchService:
    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service
        self.dimension = 384  # all-MiniLM-L6-v2 vector dimension
        # Industry Grade: Using FAISS Inner Product for extremely fast cosine similarity search
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_map = {}  # faiss_id -> memory_record

    def _build_faiss_index(self, memories: list[dict[str, Any]]):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_map.clear()
        
        vectors = []
        for idx, item in enumerate(memories):
            emb = item.get("embedding")
            if not emb:
                continue
            v = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm  # Normalize for cosine similarity using Inner Product index
            vectors.append(v)
            self.id_map[idx] = item
            
        if vectors:
            self.index.add(np.array(vectors))

    def search(self, query: str, memories: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not memories:
            return []
        if self.embedding_service.enabled:
            return self._semantic_search(query, memories, top_k)
        return self._keyword_fuzzy_search(query, memories, top_k)

    def _semantic_search(
        self, query: str, memories: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        query_vec = self.embedding_service.embed_text(query)
        if query_vec is None:
            return self._keyword_fuzzy_search(query, memories, top_k)

        # Build FAISS index dynamically. In full production, this would be appended to linearly.
        self._build_faiss_index(memories)

        if self.index.ntotal == 0:
            return self._keyword_fuzzy_search(query, memories, top_k)

        q = np.array([query_vec], dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
             q = q / norm

        distances, indices = self.index.search(q, min(top_k, self.index.ntotal))

        scored: list[dict[str, Any]] = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx in self.id_map:
                item_copy = dict(self.id_map[idx])
                item_copy["score"] = float(distances[0][i])
                scored.append(item_copy)

        return scored

    @staticmethod
    def _keyword_fuzzy_search(
        query: str, memories: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        query_lower = query.lower().strip()
        query_tokens = set(query_lower.split())
        scored: list[dict[str, Any]] = []

        for item in memories:
            text = (item.get("text") or "").lower()
            tokens = set(text.split())
            overlap = len(query_tokens.intersection(tokens)) / max(len(query_tokens), 1)
            fuzzy = SequenceMatcher(None, query_lower, text).ratio()
            score = 0.65 * overlap + 0.35 * fuzzy
            item_copy = dict(item)
            item_copy["score"] = score
            scored.append(item_copy)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
