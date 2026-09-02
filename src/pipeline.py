"""End-to-end RAG pipeline: hybrid retrieval → cross-encoder rerank → LLM answer."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from generate import RAGAnswer, generate_answer
from rerank import CrossEncoderReranker
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RetrievedDoc


@dataclass
class PipelineResult:
    query: str
    answer: RAGAnswer
    retrieved_docs: list[RetrievedDoc]
    reranked_docs: list[RetrievedDoc]
    latency_ms: dict[str, float]


class RAGPipeline:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: CrossEncoderReranker,
        llm_model: str = "gpt-4o-mini",
        retrieve_top_k: int = 50,
        rerank_top_k: int = 10,
    ):
        self.hybrid = HybridRetriever(bm25, dense)
        self.reranker = reranker
        self.llm_model = llm_model
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k

    def run(self, query: str) -> PipelineResult:
        latency = {}

        t0 = time.perf_counter()
        retrieved = self.hybrid.search(query, top_k=self.retrieve_top_k)
        latency["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        reranked = self.reranker.rerank(query, retrieved, top_k=self.rerank_top_k)
        latency["rerank_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        answer = generate_answer(query, reranked, model=self.llm_model)
        latency["generation_ms"] = (time.perf_counter() - t2) * 1000

        latency["total_ms"] = sum(latency.values())
        return PipelineResult(
            query=query,
            answer=answer,
            retrieved_docs=retrieved,
            reranked_docs=reranked,
            latency_ms=latency,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--corpus-path", default="data/corpus.jsonl")
    args = parser.parse_args()

    # Build indices
    import json
    with open(args.corpus_path) as f:
        corpus = [json.loads(line) for line in f]

    bm25 = BM25Retriever(corpus=corpus)
    dense = DenseRetriever(collection=args.collection)
    reranker = CrossEncoderReranker()

    pipeline = RAGPipeline(bm25, dense, reranker, llm_model=args.llm_model)
    result = pipeline.run(args.query)

    print(f"\n=== Query: {args.query} ===")
    print(f"\n=== Answer ===\n{result.answer.answer}")
    print(f"\n=== Sources ===\n{result.answer.sources}")
    print(f"\n=== Latency ===")
    for k, v in result.latency_ms.items():
        print(f"  {k}: {v:.1f} ms")


if __name__ == "__main__":
    main()
