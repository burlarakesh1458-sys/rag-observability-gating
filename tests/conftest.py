"""Offline fakes for the vendored pipeline: no network, no model downloads."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DRONEDOCS_ROOT", str(ROOT / "vendor" / "dronedocs-rag"))
sys.path.insert(0, str(ROOT / "src"))

from dronedocs.chunking import Chunk  # noqa: E402
from dronedocs.llm import LLMResponse  # noqa: E402
from dronedocs.retrieval import Hit  # noqa: E402


class FakeIndex:
    def __init__(self):
        self.chunks = {
            "rules.pdf#c000": Chunk(
                "rules.pdf#c000",
                "rules.pdf",
                "pdf",
                "page 27",
                "Nano unmanned aircraft system: weighing less than or equal to 250 grams",
                15,
                [27],
                [],
            ),
            "px4.md#c000": Chunk(
                "px4.md#c000",
                "px4.md",
                "markdown",
                "section: Auto-Disarming",
                "COM_DISARM_LAND Time-out for auto disarm after landing. Default: 2s",
                15,
                [],
                ["Auto"],
            ),
        }

    def hybrid_search(self, query, vector_k, bm25_k, rrf_k, fused_k, mode="hybrid"):
        return [Hit(c, 1 / (i + 1), i + 1, mode, ch.text) for i, (c, ch) in enumerate(self.chunks.items())], {
            "vector": [["rules.pdf#c000", 0.8]],
            "bm25": [["px4.md#c000", 3.1]],
        }


class FakeReranker:
    def rerank(self, query, hits, top_n):
        return [
            Hit(h.chunk_id, 4.0 - i, i + 1, "fake", h.text, {"pre_rank": h.rank}) for i, h in enumerate(hits[:top_n])
        ]


class FakeLLM:
    def __init__(self, text, model="openai/gpt-oss-120b"):
        self.text, self.model = text, model

    def chat(self, messages, **kw):
        return LLMResponse(text=self.text, model=self.model, prompt_tokens=2000, completion_tokens=100, latency_s=0.01)


@pytest.fixture
def make_traced_pipeline(tmp_path):
    from dronedocs.config import load_prompt_bundle, load_settings
    from dronedocs.pipeline import RAGPipeline

    from ragobs.config import obs_settings
    from ragobs.store import MetricsStore
    from ragobs.tracing import JSONLTraceExporter, OTelExporter, RAGTracer

    def _make(answer, model="openai/gpt-oss-120b", clock=None):
        cfg = load_settings(ROOT / "vendor" / "dronedocs-rag" / "configs" / "settings.yaml")
        store = MetricsStore(tmp_path / "m.sqlite")
        otel = OTelExporter("test", span_file=tmp_path / "spans.jsonl", simple=True)
        tracer = RAGTracer(
            [JSONLTraceExporter(tmp_path / "traces"), otel, store], obs_settings()["pricing"], clock=clock
        )
        pipe = RAGPipeline(
            settings=cfg,
            index=FakeIndex(),
            reranker=FakeReranker(),
            llm=FakeLLM(answer, model),
            prompt_bundle=load_prompt_bundle(ROOT / "configs" / "prompts" / "answer_v1.yaml"),
            observers=[tracer],
        )
        return pipe, tracer, store, tmp_path

    return _make
