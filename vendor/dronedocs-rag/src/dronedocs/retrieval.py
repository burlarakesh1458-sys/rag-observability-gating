"""Hybrid retrieval: Chroma vector search + BM25 keyword search fused with RRF."""

from __future__ import annotations

import contextlib
import json
import logging
import pickle
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9_]+(?:\.[0-9]+)?")
_STOP_WORDS = (
    "a an the of to in on for and or is are be by with as at from "
    "that this it its what which who how when does do can i my under any per"
)
_STOP = set(_STOP_WORDS.split())


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps parameter names like RTL_ALT_M and numbers like 2.5 intact."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    chunk_id: str
    score: float
    rank: int
    method: str
    text: str = ""
    metadata: dict = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """RRF(d) = sum_i w_i / (k + rank_i(d)), ranks are 1-based. Returns ids sorted by fused score."""
    weights = list(weights) if weights else [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for w, ranking in zip(weights, rankings, strict=True):
        for r, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + r)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Embedder:
    def __init__(self, model_name: str, query_instruction: str = "", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.query_instruction = query_instruction
        self.batch_size = batch_size

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([self.query_instruction + text], normalize_embeddings=True)[0].tolist()


class HybridIndex:
    """Owns the Chroma collection, the BM25 index and the chunk manifest on disk."""

    COLLECTION = "dronedocs"

    def __init__(self, index_dir: str | Path, embedder: Embedder | None = None):
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self.chunks: dict[str, Chunk] = {}
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._collection = None

    # ------------------------------------------------------------------ build
    def _client(self):
        import chromadb
        from chromadb.config import Settings

        return chromadb.PersistentClient(
            path=str(self.index_dir / "chroma"), settings=Settings(anonymized_telemetry=False)
        )

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        if self.embedder is None:
            raise RuntimeError("embedder required to build the index")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        client = self._client()
        with contextlib.suppress(Exception):
            client.delete_collection(self.COLLECTION)
        col = client.create_collection(self.COLLECTION, metadata={"hnsw:space": "cosine"})
        texts = [c.text for c in chunks]
        vecs = self.embedder.embed_docs(texts)
        for i in range(0, len(chunks), 256):
            sl = slice(i, i + 256)
            col.add(
                ids=[c.chunk_id for c in chunks[sl]],
                embeddings=vecs[sl],
                documents=texts[sl],
                metadatas=[c.to_metadata() for c in chunks[sl]],
            )
        self._collection = col
        self.chunks = {c.chunk_id: c for c in chunks}
        self._bm25_ids = [c.chunk_id for c in chunks]
        self._bm25 = BM25Okapi([bm25_tokenize(c.text + " " + c.locator) for c in chunks])
        with open(self.index_dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c.__dict__) + "\n")
        with open(self.index_dir / "bm25.pkl", "wb") as fh:
            pickle.dump((self._bm25_ids, self._bm25), fh)

    # ------------------------------------------------------------------ load
    def load(self) -> HybridIndex:
        manifest = self.index_dir / "chunks.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"no index at {self.index_dir}; run `make ingest` first")
        with open(manifest, encoding="utf-8") as fh:
            self.chunks = {d["chunk_id"]: Chunk(**d) for d in map(json.loads, fh)}
        with open(self.index_dir / "bm25.pkl", "rb") as fh:
            self._bm25_ids, self._bm25 = pickle.load(fh)  # noqa: S301 - our own artifact
        self._collection = self._client().get_collection(self.COLLECTION)
        return self

    # ------------------------------------------------------------------ search
    def vector_search(self, query: str, k: int) -> list[Hit]:
        qv = self.embedder.embed_query(query)
        res = self._collection.query(query_embeddings=[qv], n_results=min(k, len(self.chunks)))
        hits = []
        for r, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0], strict=True), start=1):
            hits.append(Hit(cid, 1.0 - float(dist), r, "vector", self.chunks[cid].text))
        return hits

    def bm25_search(self, query: str, k: int) -> list[Hit]:
        import numpy as np

        scores = self._bm25.get_scores(bm25_tokenize(query))
        order = np.argsort(-scores)[:k]
        return [
            Hit(self._bm25_ids[i], float(scores[i]), r, "bm25", self.chunks[self._bm25_ids[i]].text)
            for r, i in enumerate(order, start=1)
            if scores[i] > 0
        ]

    def hybrid_search(
        self, query: str, vector_k: int = 20, bm25_k: int = 20, rrf_k: int = 60, fused_k: int = 20, mode: str = "hybrid"
    ) -> tuple[list[Hit], dict]:
        vec = self.vector_search(query, vector_k) if mode in ("hybrid", "vector") else []
        bm = self.bm25_search(query, bm25_k) if mode in ("hybrid", "bm25") else []
        fused = reciprocal_rank_fusion([[h.chunk_id for h in vec], [h.chunk_id for h in bm]], k=rrf_k)
        hits = [Hit(cid, s, r, mode, self.chunks[cid].text) for r, (cid, s) in enumerate(fused[:fused_k], 1)]
        debug = {
            "vector": [(h.chunk_id, round(h.score, 4)) for h in vec],
            "bm25": [(h.chunk_id, round(h.score, 3)) for h in bm],
        }
        return hits, debug
