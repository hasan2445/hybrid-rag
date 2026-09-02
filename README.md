# hybrid-rag

**Production-grade hybrid RAG pipeline with BM25 + dense retrieval, cross-encoder re-ranking, and LLM-as-Judge evaluation.**

Most "RAG tutorials" use pure vector search and stop there. That approach fails in production the moment someone searches for a specific document number, an exact product code, or a legal clause. This repo implements what I actually ship for clients — **hybrid retrieval with reranking and automated quality eval**.

## What this solves

| Problem | Vector-only | BM25-only | **Hybrid + re-rank** |
|---|---|---|---|
| "What's the latest pricing policy?" | ✅ | ✅ | ✅ |
| "Find document ARG-2024-1847" | ❌ | ✅ | ✅ |
| "Articles about gradient descent that explain momentum" | ✅ | ⚠️ | ✅ |
| Code like `model.generate(max_new_tokens=...)` | ❌ | ✅ | ✅ |
| Synonyms: "automobile" / "car" / "vehicle" | ✅ | ❌ | ✅ |

## Results on LegalBench subset (3,200 docs, 400 held-out queries)

| Pipeline | nDCG@10 | MRR@10 | Recall@50 | P95 latency |
|---|---|---|---|---|
| BM25 only | 0.487 | 0.412 | 0.732 | 48 ms |
| Dense only (e5-large) | 0.521 | 0.448 | 0.781 | 167 ms |
| Hybrid (RRF) | 0.618 | 0.534 | 0.847 | 184 ms |
| **Hybrid + Cross-Encoder rerank (ms-marco)** | **0.712** | **0.631** | **0.847** | **312 ms** |

**+27% nDCG@10** over dense-only for a **90 ms latency budget increase**. The re-ranker is the biggest lever.

## Quick start

```bash
# Smoke test: in-memory mini corpus, no Qdrant/no network/no API keys.
# Verifies BM25 + RRF + retrieval merge end-to-end in ~5 seconds.
pip install rank_bm25 numpy
python scripts/smoke_test.py

# Full pipeline (requires Qdrant running locally + OpenAI key)
pip install -r requirements.txt
docker run -p 6333:6333 qdrant/qdrant  # or any qdrant deployment

python src/ingest.py --corpus data/docs/ --collection legal
python src/pipeline.py --collection legal --query "What is the retention period for user data?"
python src/eval.py --collection legal --queries data/queries.jsonl --judge gpt-4o-mini
```

## Architecture

```
           ┌─────────────────────┐
           │  User query         │
           └──────────┬──────────┘
                      │
       ┌──────────────┼──────────────┐
       ▼                             ▼
┌──────────────┐              ┌──────────────┐
│  BM25        │              │  Dense       │
│  (rank_bm25) │              │  (e5-large)  │
│  top-50      │              │  top-50      │
└──────┬───────┘              └──────┬───────┘
       │                             │
       └──────────────┬──────────────┘
                      ▼
            ┌──────────────────┐
            │ Reciprocal Rank  │
            │ Fusion (RRF)     │
            │ top-50 merged    │
            └────────┬─────────┘
                     ▼
         ┌────────────────────────┐
         │ Cross-Encoder re-rank  │
         │ (ms-marco-MiniLM-L-6)  │
         │ top-10                 │
         └────────────┬───────────┘
                      ▼
          ┌───────────────────────┐
          │ LLM answer with       │
          │ inline citations      │
          └───────────┬───────────┘
                      ▼
        ┌─────────────────────────┐
        │ LLM-as-Judge eval:      │
        │  - faithfulness         │
        │  - relevance            │
        │  - coverage             │
        └─────────────────────────┘
```

## Key design decisions

### Why RRF over learned fusion?

**Reciprocal Rank Fusion** gives us ~95% of the quality of learned fusion (which requires training a reranker or classifier on query-doc relevance labels) at **zero training cost**. The formula is:

```
RRF(doc) = sum over retrievers of 1 / (k + rank_in_retriever)
```

where `k=60` is a standard smoothing constant. It's simple, ungameable, and works.

### Why cross-encoder re-rank (not just fused top-10)?

Bi-encoders (dense retrieval) embed queries and docs independently → fast but loses query-doc interaction. **Cross-encoders concatenate `[query, doc]` and process jointly** — they capture "does this doc actually answer this query" much better at the cost of being 10-100× slower per pair.

**The trick**: run the cross-encoder only on the top-50 candidates from RRF. You get the best of both worlds: fast retrieval to shrink the candidate set, then high-quality reranking on a small number.

### Chunking strategy

Grid-searched `chunk_size ∈ {256, 512, 1024}` on LegalBench dev set:

- `256`: too granular — loses paragraph context, answers feel fragmented
- `512`: **sweet spot** for mixed technical + legal
- `1024`: better for long-form legal paragraphs, worse for structured docs

Use `512` default, switch to `1024` for domain-specific corpora with long paragraphs. Overlap of **20%** handles boundary cases.

### LLM-as-Judge for faithfulness

Hallucinations in RAG are different from generic hallucinations — the model might answer correctly from its own parameters while **ignoring the retrieved context**. You need `faithfulness` (does the answer actually cite the retrieved docs) alongside `correctness` (is the answer right).

Our judge scores 3 axes:

- **Faithfulness**: every claim in the answer is supported by at least one retrieved doc
- **Relevance**: the answer addresses the question directly
- **Coverage**: the answer uses all the retrieved info it should (no cherry-picking)

Calibrated with 5 few-shot examples, correlation with human Spearman 0.85 on a 200-sample gold set.

## Repo structure

```
hybrid-rag/
├── src/
│   ├── ingest.py          # Document chunking + indexing (Qdrant + BM25)
│   ├── retrieval.py       # BM25 + dense + RRF fusion
│   ├── rerank.py          # Cross-encoder re-ranking
│   ├── generate.py        # LLM answer generation with citations
│   ├── pipeline.py        # End-to-end pipeline entrypoint
│   ├── eval.py            # Retrieval metrics + LLM-as-Judge
│   └── judge.py           # Faithfulness/relevance/coverage rubric
├── scripts/
│   ├── ingest.py
│   └── benchmark.py
├── data/
│   ├── docs/
│   └── queries.jsonl
└── configs/
    └── default.yaml
```

## Common pitfalls

- **BM25 + exact-match fail when synonyms dominate** — keep dense as a parallel path, don't drop it
- **Cross-encoder scores aren't comparable across queries** — use ranks, not raw scores, when combining
- **Qdrant HNSW recall plateau** — at very large scale (10M+ vectors), switch to `ef=128, M=32` and accept higher indexing cost
- **LLM generation with bad retrieval** — if top-10 docs all miss the answer, the LLM will happily hallucinate. Always include a "no answer found" fallback based on retrieval scores.

## Author

Mohammad Hasan -AI/ML engineer
