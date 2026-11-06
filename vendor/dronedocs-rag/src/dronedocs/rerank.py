"""Second-stage rerankers: local cross-encoder (default), Cohere (optional), or passthrough."""

from __future__ import annotations

import os
from typing import Protocol

from .retrieval import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...


class CrossEncoderReranker:
    """Scores (query, chunk) pairs jointly with ms-marco-MiniLM-L-6-v2 (CPU friendly, ~22M params)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 256,
        candidates: int | None = None,
    ):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu", max_length=max_length)
        self.candidates = candidates

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        hits = hits[: self.candidates] if self.candidates else hits
        scores = self.model.predict([(query, h.text) for h in hits], show_progress_bar=False)
        ranked = sorted(zip(hits, scores, strict=True), key=lambda x: -float(x[1]))[:top_n]
        return [
            Hit(h.chunk_id, float(s), r, "cross_encoder", h.text, {"pre_rank": h.rank})
            for r, (h, s) in enumerate(ranked, start=1)
        ]


class CohereReranker:
    def __init__(self, model: str = "rerank-english-v3.0"):
        import httpx

        self.model = model
        self._http = httpx.Client(timeout=30, headers={"Authorization": f"Bearer {os.environ['COHERE_API_KEY']}"})

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        r = self._http.post(
            "https://api.cohere.com/v2/rerank",
            json={"model": self.model, "query": query, "documents": [h.text for h in hits], "top_n": top_n},
        )
        r.raise_for_status()
        out = []
        for rank, item in enumerate(r.json()["results"], start=1):
            h = hits[item["index"]]
            out.append(Hit(h.chunk_id, float(item["relevance_score"]), rank, "cohere", h.text, {"pre_rank": h.rank}))
        return out


class PassthroughReranker:
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        return [
            Hit(h.chunk_id, h.score, i, "none", h.text, {"pre_rank": h.rank})
            for i, h in enumerate(hits[:top_n], start=1)
        ]


def build_reranker(cfg: dict) -> Reranker:
    provider = cfg.get("provider", "cross_encoder")
    if provider == "cohere":
        return CohereReranker(cfg.get("cohere_model", "rerank-english-v3.0"))
    if provider == "none":
        return PassthroughReranker()
    return CrossEncoderReranker(
        cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"), cfg.get("max_length", 256), cfg.get("candidates")
    )
