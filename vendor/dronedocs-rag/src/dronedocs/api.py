"""FastAPI service: ``POST /ask``, ``POST /ingest``, ``GET /health`` and a minimal HTML UI at ``/``."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings

app = FastAPI(title="DroneDocs RAG", version=__version__)

_lock = threading.Lock()
_state: dict = {"pipeline": None}


def get_pipeline():
    """Lazily build the pipeline (models load on first request, not at import)."""
    with _lock:
        if _state["pipeline"] is None:
            from .pipeline import RAGPipeline

            _state["pipeline"] = RAGPipeline()
        return _state["pipeline"]


def set_pipeline(p) -> None:  # used by tests and by the observability layer
    _state["pipeline"] = p


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    include_debug: bool = False


class CitationOut(BaseModel):
    chunk_id: str
    source: str
    locator: str
    snippet: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    refused: bool
    refusal_reason: str | None = None
    citations: list[CitationOut]
    prompt_version: str
    model: str
    latency_ms: dict[str, float]
    debug: dict | None = None


class IngestRequest(BaseModel):
    raw_dir: str | None = None


@app.get("/health")
def health() -> dict:
    cfg = get_settings()
    index_ready = (Path(cfg["paths"]["index_dir"]) / "chunks.jsonl").exists()
    return {
        "status": "ok" if index_ready else "degraded",
        "index_ready": index_ready,
        "version": __version__,
        "pipeline_loaded": _state["pipeline"] is not None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        pipe = get_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    res = pipe.ask(req.question)
    if res.error:
        raise HTTPException(status_code=502, detail=res.error)
    return AskResponse(
        request_id=res.request_id,
        answer=res.answer,
        refused=res.refused,
        refusal_reason=res.refusal_reason,
        citations=[CitationOut(**c.__dict__) for c in res.citations],
        prompt_version=res.prompt_version,
        model=res.model,
        latency_ms={k: round(v * 1000, 1) for k, v in res.stage_timings.items()},
        debug={"reranked": res.reranked, "usage": res.usage, "citation_check": res.citation_check}
        if req.include_debug
        else None,
    )


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest) -> dict:
    from .ingest import ingest

    stats = ingest(req.raw_dir)
    with _lock:
        _state["pipeline"] = None  # reload the index on next request
    return stats


_HTML = Path(__file__).with_name("static") / "index.html"


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return _HTML.read_text(encoding="utf-8")
