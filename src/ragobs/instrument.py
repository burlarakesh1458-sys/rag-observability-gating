"""Build an instrumented DroneDocs pipeline (tracer + metrics store attached as observers)."""

from __future__ import annotations

from .config import obs_settings, rag_settings, resolve
from .store import MetricsStore
from .tracing import RAGTracer, build_tracer


def build_store() -> MetricsStore:
    return MetricsStore(resolve(obs_settings()["storage"]["sqlite_path"]))


def build_instrumented_pipeline(
    prompt_file: str | None = None,
    model: str | None = None,
    store=None,
    clock=None,
    tags: dict | None = None,
    shared_components=None,
    simple_spans: bool = False,
    llm=None,
):
    """Returns ``(pipeline, tracer)``.

    ``shared_components`` = (index, reranker) lets several pipelines (e.g. two prompt versions during
    a replay) share the loaded embedding / cross-encoder models instead of loading them twice.
    """
    from dronedocs.pipeline import RAGPipeline

    cfg = rag_settings(prompt_file, model)
    tracer: RAGTracer = build_tracer(obs_settings(), store=store, clock=clock, tags=tags, simple_spans=simple_spans)
    index, reranker = shared_components if shared_components else (None, None)
    pipe = RAGPipeline(settings=cfg, index=index, reranker=reranker, llm=llm, observers=[tracer])
    return pipe, tracer
