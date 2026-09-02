"""Cross-encoder re-ranking on top-N candidates.

Cross-encoders concatenate (query, doc) and jointly score relevance. Much more
accurate than bi-encoder cosine similarity but 10-100× slower per pair, so we
only run it on a pre-filtered candidate set from hybrid retrieval.
"""
from __future__ import annotations

import torch
from sentence_transformers import CrossEncoder

from retrieval import RetrievedDoc


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDoc],
        top_k: int = 10,
    ) -> list[RetrievedDoc]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        # Attach new scores and re-sort
        reranked = []
        for doc, score in zip(candidates, scores):
            reranked.append(
                RetrievedDoc(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=float(score),
                    source="cross-encoder",
                    metadata=doc.metadata,
                )
            )
        reranked.sort(key=lambda d: -d.score)
        return reranked[:top_k]
