"""Evaluation harness: retrieval metrics + LLM-as-Judge.

Computes:
- Retrieval: nDCG@k, MRR@k, Recall@k
- Generation: faithfulness / relevance / coverage via LLM-as-Judge
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from judge import RAGJudge
from pipeline import RAGPipeline
from rerank import CrossEncoderReranker
from retrieval import BM25Retriever, DenseRetriever


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances[:k]))


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    rel = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids[:k]]
    dcg = dcg_at_k(rel, k)
    ideal = dcg_at_k(sorted(rel, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


@dataclass
class EvalResult:
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_50: float
    faithfulness: float
    relevance: float
    coverage: float


def evaluate(pipeline: RAGPipeline, judge: RAGJudge, queries_path: str, out_path: str) -> EvalResult:
    with open(queries_path) as f:
        queries = [json.loads(line) for line in f]

    ndcg_sum, mrr_sum, recall_sum = 0.0, 0.0, 0.0
    faith_sum, rel_sum, cov_sum = 0.0, 0.0, 0.0
    n = 0
    results = []

    for q in tqdm(queries):
        query = q["query"]
        relevant_ids = set(q.get("relevant_doc_ids", []))

        pipeline_result = pipeline.run(query)
        retrieved_ids = [d.doc_id for d in pipeline_result.reranked_docs]

        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k=10)
        mrr = mrr_at_k(retrieved_ids, relevant_ids, k=10)
        recall = recall_at_k(retrieved_ids, relevant_ids, k=50)

        # LLM-as-Judge
        context_text = "\n".join(
            f"[{i+1}] {d.text}" for i, d in enumerate(pipeline_result.reranked_docs)
        )
        score = judge.score(query, context_text, pipeline_result.answer.answer)

        ndcg_sum += ndcg
        mrr_sum += mrr
        recall_sum += recall
        faith_sum += score.faithfulness
        rel_sum += score.relevance
        cov_sum += score.coverage
        n += 1

        results.append({
            "query": query,
            "ndcg@10": ndcg,
            "mrr@10": mrr,
            "recall@50": recall,
            "faithfulness": score.faithfulness,
            "relevance": score.relevance,
            "coverage": score.coverage,
            "judge_reasoning": score.reasoning,
        })

    # Write per-query
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    eval_result = EvalResult(
        ndcg_at_10=ndcg_sum / n,
        mrr_at_10=mrr_sum / n,
        recall_at_50=recall_sum / n,
        faithfulness=faith_sum / n,
        relevance=rel_sum / n,
        coverage=cov_sum / n,
    )

    print("\n=== Evaluation Summary ===")
    print(f"  nDCG@10:       {eval_result.ndcg_at_10:.3f}")
    print(f"  MRR@10:        {eval_result.mrr_at_10:.3f}")
    print(f"  Recall@50:     {eval_result.recall_at_50:.3f}")
    print(f"  Faithfulness:  {eval_result.faithfulness:.2f}/5")
    print(f"  Relevance:     {eval_result.relevance:.2f}/5")
    print(f"  Coverage:      {eval_result.coverage:.2f}/5")
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--corpus-path", default="data/corpus.jsonl")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--out", default="outputs/eval_results.jsonl")
    args = parser.parse_args()

    with open(args.corpus_path) as f:
        corpus = [json.loads(line) for line in f]

    bm25 = BM25Retriever(corpus=corpus)
    dense = DenseRetriever(collection=args.collection)
    reranker = CrossEncoderReranker()
    pipeline = RAGPipeline(bm25, dense, reranker, llm_model=args.llm_model)
    judge = RAGJudge(model=args.judge_model)

    evaluate(pipeline, judge, args.queries, args.out)
