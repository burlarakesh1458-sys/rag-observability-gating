"""Ingestion: load -> clean -> chunk -> embed -> persist (Chroma + BM25 + manifest)."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

from .chunking import Chunk, chunk_sections
from .config import get_settings
from .loaders import discover, load_file
from .retrieval import Embedder, HybridIndex

log = logging.getLogger(__name__)


def build_chunks(raw_dir: str | Path, cfg: dict) -> list[Chunk]:
    c = cfg["chunking"]
    chunks: list[Chunk] = []
    for path in discover(raw_dir):
        sections = load_file(path, c.get("drop_non_latin_lines", True))
        doc_chunks = chunk_sections(
            sections, c["chunk_tokens"], c["overlap_tokens"], c["min_chunk_tokens"], c["tokenizer"]
        )
        log.info("%s: %d sections -> %d chunks", path.name, len(sections), len(doc_chunks))
        chunks.extend(doc_chunks)
    return chunks


def ingest(raw_dir: str | Path | None = None, settings: dict | None = None) -> dict:
    cfg = settings or get_settings()
    raw_dir = raw_dir or cfg["paths"]["raw_dir"]
    t0 = time.perf_counter()
    chunks = build_chunks(raw_dir, cfg)
    if not chunks:
        raise RuntimeError(f"no documents found under {raw_dir}")
    t_chunk = time.perf_counter() - t0
    emb = Embedder(
        cfg["embeddings"]["model"],
        cfg["embeddings"].get("query_instruction", ""),
        cfg["embeddings"].get("batch_size", 32),
    )
    t1 = time.perf_counter()
    HybridIndex(cfg["paths"]["index_dir"], emb).build(chunks)
    t_index = time.perf_counter() - t1
    tok = [c.n_tokens for c in chunks]
    stats = {
        "documents": len({c.source for c in chunks}),
        "chunks": len(chunks),
        "chunks_per_type": dict(Counter(c.doc_type for c in chunks)),
        "tokens_total": sum(tok),
        "tokens_per_chunk_mean": round(sum(tok) / len(tok), 1),
        "tokens_per_chunk_max": max(tok),
        "load_chunk_s": round(t_chunk, 2),
        "embed_index_s": round(t_index, 2),
        "embedding_model": cfg["embeddings"]["model"],
    }
    Path(cfg["paths"]["index_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["index_dir"]) / "ingest_stats.json").write_text(json.dumps(stats, indent=2))
    return stats
