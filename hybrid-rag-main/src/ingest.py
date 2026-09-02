"""Document ingestion: chunk → embed → index into Qdrant + BM25 corpus.

Chunking uses a recursive splitter with overlap, preserving sentence boundaries
where possible. Tuned for 512-token chunks (sweet spot from grid search) with
20% overlap to handle boundary queries.
"""
from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Iterable

import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap_ratio: float = 0.2,
    encoder: tiktoken.Encoding | None = None,
) -> list[str]:
    """Split text into ~chunk_size token chunks with overlap, preserving paragraphs."""
    encoder = encoder or tiktoken.get_encoding("cl100k_base")
    overlap = int(chunk_size * overlap_ratio)

    # Split on paragraph, then sentences
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append("\n\n".join(buffer))
            # Preserve last ~overlap tokens of buffer for the next chunk
            tail = []
            tail_tokens = 0
            for piece in reversed(buffer):
                t = count_tokens(piece, encoder)
                if tail_tokens + t > overlap:
                    break
                tail.insert(0, piece)
                tail_tokens += t
            buffer = tail
            buffer_tokens = tail_tokens

    for para in paragraphs:
        para_tokens = count_tokens(para, encoder)

        # Single paragraph exceeds chunk_size — split on sentences
        if para_tokens > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for s in sentences:
                s_tokens = count_tokens(s, encoder)
                if buffer_tokens + s_tokens > chunk_size:
                    flush_buffer()
                buffer.append(s)
                buffer_tokens += s_tokens
            continue

        if buffer_tokens + para_tokens > chunk_size:
            flush_buffer()
        buffer.append(para)
        buffer_tokens += para_tokens

    if buffer:
        chunks.append("\n\n".join(buffer))

    return chunks


def ingest(
    corpus_dir: str,
    collection: str,
    qdrant_url: str = "http://localhost:6333",
    chunk_size: int = 512,
    overlap_ratio: float = 0.2,
    embed_model: str = "intfloat/e5-large-v2",
    out_corpus: str = "data/corpus.jsonl",
) -> None:
    encoder = tiktoken.get_encoding("cl100k_base")
    client = QdrantClient(url=qdrant_url)
    model = SentenceTransformer(embed_model)
    vector_dim = model.get_sentence_embedding_dimension()

    # Ensure collection exists
    collections = client.get_collections().collections
    if not any(c.name == collection for c in collections):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{collection}' (dim={vector_dim})")

    # Iterate over .txt files
    corpus_path = Path(corpus_dir)
    files = list(corpus_path.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files under {corpus_dir}")

    bm25_corpus = []
    points = []
    batch_size = 64

    for file in tqdm(files, desc="Ingesting"):
        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap_ratio=overlap_ratio, encoder=encoder)

        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            metadata = {"source": str(file.relative_to(corpus_path)), "chunk_index": i}
            bm25_corpus.append({"id": doc_id, "text": chunk, "metadata": metadata})

            # Dense side: batch and flush
            points.append({"id": doc_id, "text": chunk, "metadata": metadata})
            if len(points) >= batch_size:
                _flush_qdrant_batch(client, model, collection, points)
                points = []

    if points:
        _flush_qdrant_batch(client, model, collection, points)

    # Write bm25 corpus
    Path(out_corpus).parent.mkdir(parents=True, exist_ok=True)
    with open(out_corpus, "w") as f:
        for d in bm25_corpus:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Ingested {len(bm25_corpus)} chunks from {len(files)} files")
    print(f"BM25 corpus written to {out_corpus}")


def _flush_qdrant_batch(client: QdrantClient, model, collection: str, points: list[dict]) -> None:
    texts = [f"passage: {p['text']}" for p in points]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    qdrant_points = [
        PointStruct(
            id=p["id"],
            vector=emb.tolist(),
            payload={"text": p["text"], "metadata": p["metadata"]},
        )
        for p, emb in zip(points, embeddings)
    ]
    client.upsert(collection_name=collection, points=qdrant_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--out-corpus", default="data/corpus.jsonl")
    args = parser.parse_args()
    ingest(
        args.corpus,
        args.collection,
        args.qdrant_url,
        args.chunk_size,
        args.overlap_ratio,
        out_corpus=args.out_corpus,
    )
