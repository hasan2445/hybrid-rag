"""In-memory dense retriever for smoke tests and small corpora.

Alternative to the Qdrant-backed DenseRetriever when you don't want to run
a separate vector database server. Uses numpy cosine similarity for <100k docs.

For production: use retrieval.DenseRetriever (Qdrant) instead.
"""
from __future__ import annotations

import hashlib

import numpy as np

from retrieval import RetrievedDoc


class _HashEmbedder:
    """Deterministic hash-based embedder for smoke tests.

    NOT semantically meaningful — just a reproducible way to get dense vectors
    without downloading sentence-transformers. Uses multi-hash projection.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for t in texts:
            tokens = t.lower().split()
            vec = np.zeros(self.dim, dtype=np.float32)
            for tok in tokens:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                for i in range(self.dim):
                    vec[i] += ((h >> i) & 1) * 2 - 1
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
        return np.array(embeddings)


class InMemoryDenseRetriever:
    """Dense retriever with numpy cosine similarity over an in-memory matrix.

    Works with either a real SentenceTransformer encoder or the deterministic
    hash embedder used for smoke tests.
    """

    def __init__(self, corpus: list[dict], encoder=None, embed_dim: int = 64):
        self.corpus = corpus
        self.doc_ids = [d["id"] for d in corpus]
        self.encoder = encoder or _HashEmbedder(dim=embed_dim)

        texts = [d["text"] for d in corpus]
        self.embeddings = self._encode_passages(texts)

    def _encode_passages(self, texts: list[str]) -> np.ndarray:
        # Real sentence-transformers accepts normalize_embeddings kwarg,
        # our hash embedder already returns normalized vectors.
        try:
            return self.encoder.encode(
                [f"passage: {t}" for t in texts],
                normalize_embeddings=True,
            )
        except TypeError:
            return self.encoder.encode([f"passage: {t}" for t in texts])

    def _encode_query(self, query: str) -> np.ndarray:
        try:
            return self.encoder.encode([f"query: {query}"], normalize_embeddings=True)[0]
        except TypeError:
            return self.encoder.encode([f"query: {query}"])[0]

    def search(self, query: str, top_k: int = 50) -> list[RetrievedDoc]:
        q_vec = self._encode_query(query)
        scores = self.embeddings @ q_vec
        top_indices = np.argsort(-scores)[:top_k]
        return [
            RetrievedDoc(
                doc_id=self.doc_ids[i],
                text=self.corpus[i]["text"],
                score=float(scores[i]),
                source="dense",
                metadata=self.corpus[i].get("metadata"),
            )
            for i in top_indices
        ]
