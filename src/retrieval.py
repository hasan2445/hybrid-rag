"""Hybrid retrieval: BM25 + dense embeddings + Reciprocal Rank Fusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

# Heavy optional deps — only needed for the full Qdrant-backed DenseRetriever.
# BM25Retriever, reciprocal_rank_fusion, and smoke tests work without them.
try:
    from qdrant_client import QdrantClient
    _HAS_QDRANT = True
except ImportError:
    QdrantClient = None  # type: ignore
    _HAS_QDRANT = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    source: str  # "bm25" / "dense" / "hybrid"
    metadata: dict | None = None


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists via RRF.

    rrf_score(doc) = sum over rankings of 1 / (k + rank_in_ranking)

    Args:
        rankings: list of lists where each inner list is doc_ids ordered best-first
        k: smoothing constant, default 60 (standard)

    Returns:
        list of (doc_id, fused_score) sorted by score descending
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


class BM25Retriever:
    """Lightweight BM25 index in memory. For production with >1M docs, swap for Elasticsearch."""

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self.doc_ids = [d["id"] for d in corpus]
        tokenized = [d["text"].lower().split() for d in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 50) -> list[RetrievedDoc]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(-scores)[:top_k]
        return [
            RetrievedDoc(
                doc_id=self.doc_ids[i],
                text=self.corpus[i]["text"],
                score=float(scores[i]),
                source="bm25",
                metadata=self.corpus[i].get("metadata"),
            )
            for i in top_indices
        ]


class DenseRetriever:
    """Qdrant-backed dense retrieval with e5-large embeddings."""

    def __init__(
        self,
        collection: str,
        qdrant_url: str = "http://localhost:6333",
        model_name: str = "intfloat/e5-large-v2",
    ):
        if not _HAS_QDRANT:
            raise ImportError(
                "qdrant_client not installed. Install with `pip install qdrant-client` "
                "or use InMemoryDenseRetriever from retrieval_inmemory.py for small corpora."
            )
        if not _HAS_ST:
            raise ImportError(
                "sentence-transformers not installed. Install with "
                "`pip install sentence-transformers`."
            )
        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection
        self.encoder = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> np.ndarray:
        # e5 models want "query: " / "passage: " prefixes for best performance
        return self.encoder.encode(f"query: {query}", normalize_embeddings=True)

    def search(self, query: str, top_k: int = 50) -> list[RetrievedDoc]:
        vector = self.embed_query(query)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector.tolist(),
            limit=top_k,
        )
        return [
            RetrievedDoc(
                doc_id=str(h.id),
                text=h.payload.get("text", ""),
                score=float(h.score),
                source="dense",
                metadata=h.payload.get("metadata"),
            )
            for h in hits
        ]


class HybridRetriever:
    """BM25 + dense combined via RRF."""

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 50, candidates_per_retriever: int = 50) -> list[RetrievedDoc]:
        bm25_hits = self.bm25.search(query, top_k=candidates_per_retriever)
        dense_hits = self.dense.search(query, top_k=candidates_per_retriever)

        # Build rankings (doc_id lists)
        rankings = [
            [h.doc_id for h in bm25_hits],
            [h.doc_id for h in dense_hits],
        ]
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)

        # Lookup full text by doc_id (prefer BM25 text, fall back to dense)
        id_to_doc: dict[str, RetrievedDoc] = {}
        for hit in bm25_hits + dense_hits:
            if hit.doc_id not in id_to_doc:
                id_to_doc[hit.doc_id] = hit

        results = []
        for doc_id, fused_score in fused[:top_k]:
            base = id_to_doc.get(doc_id)
            if base is None:
                continue
            results.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    text=base.text,
                    score=fused_score,
                    source="hybrid",
                    metadata=base.metadata,
                )
            )
        return results
