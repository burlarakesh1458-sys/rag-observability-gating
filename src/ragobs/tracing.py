"""Request tracing for the DroneDocs RAG pipeline.

``RAGTracer`` plugs into the pipeline's observer hooks (``on_stage`` / ``on_result``), so the
RAG package needs no tracing dependency. For every request it captures:

  retrieve   - BM25 + vector candidates and the RRF-fused list (ids, scores, sources)
  rerank     - the cross-encoder order with each chunk's pre-rerank position
  evidence_gate / refuse
  generate   - model, prompt version, the exact final prompt, the response, token usage, cost
  validate_citations - cited / valid / invalid chunk ids and sentence-level coverage

and fans the finished trace out to exporters:
  * JSONLTraceExporter  - one JSON document per request in traces/YYYY-MM-DD.jsonl (always on)
  * OTelExporter        - real OpenTelemetry spans (GenAI semantic conventions + Langfuse attributes)
                          to OTLP/HTTP (Langfuse self-hosted/cloud, Jaeger, Tempo...) and/or a local
                          span JSONL file when no collector is configured
  * MetricsStore        - one row per request in SQLite for dashboards / anomaly detection
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .cost import request_cost_usd

log = logging.getLogger(__name__)


@dataclass
class Span:
    name: str
    start_s: float  # epoch seconds
    end_s: float
    attributes: dict = field(default_factory=dict)


@dataclass
class Trace:
    trace_id: str
    request_id: str
    ts: str  # ISO-8601 UTC (may be simulated during backfill replay)
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    error: str | None
    model: str
    prompt_version: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: dict
    citation_coverage: float
    n_citations: int
    n_invalid_citations: int
    best_rerank_score: float | None
    spans: list[Span]
    tags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["spans"] = [s.__dict__ for s in self.spans]
        return d


class Exporter(Protocol):
    def export(self, trace: Trace) -> None: ...


class RAGTracer:
    """Pipeline observer that assembles a ``Trace`` per request and exports it."""

    def __init__(
        self,
        exporters: list[Exporter],
        pricing: dict,
        clock=None,
        tags: dict | None = None,
        max_attr_chars: int = 20000,
        capture_prompt: bool = True,
    ):
        self.exporters = exporters
        self.pricing = pricing
        self.clock = clock or time.time  # replaceable for backfill replays
        self.tags = dict(tags or {})
        self.max_attr_chars = max_attr_chars
        self.capture_prompt = capture_prompt
        self._spans: dict[str, list[Span]] = {}
        self._lock = threading.Lock()
        self.last_trace: Trace | None = None

    def _clip(self, v: Any) -> Any:
        if isinstance(v, str) and len(v) > self.max_attr_chars:
            return v[: self.max_attr_chars] + "...[truncated]"
        return v

    # ---- pipeline hooks
    def on_stage(self, request_id: str, stage: str, payload: dict, duration_s: float) -> None:
        end = self.clock()
        attrs = {k: self._clip(v) for k, v in payload.items()}
        if stage == "generate" and not self.capture_prompt:
            attrs.pop("messages", None)
        with self._lock:
            self._spans.setdefault(request_id, []).append(Span(stage, end - duration_s, end, attrs))

    def on_result(self, res) -> None:
        with self._lock:
            spans = self._spans.pop(res.request_id, [])
        end = self.clock()
        total = res.stage_timings.get("total", 0.0)
        root = Span("rag.request", end - total, end, {"question": res.question})
        pt, ct = res.usage.get("prompt_tokens", 0), res.usage.get("completion_tokens", 0)
        chk = res.citation_check or {}
        trace = Trace(
            trace_id=os.urandom(16).hex(),
            request_id=res.request_id,
            ts=datetime.fromtimestamp(end, UTC).isoformat(timespec="seconds"),
            question=res.question,
            answer=res.answer,
            refused=res.refused,
            refusal_reason=res.refusal_reason,
            error=res.error,
            model=res.model,
            prompt_version=res.prompt_version,
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=request_cost_usd(res.model, pt, ct, self.pricing),
            latency_ms={k: round(v * 1000, 2) for k, v in res.stage_timings.items()},
            citation_coverage=float(chk.get("coverage", 0.0)) if not res.refused else 0.0,
            n_citations=len(chk.get("valid_ids", [])),
            n_invalid_citations=len(chk.get("invalid_ids", [])),
            best_rerank_score=res.reranked[0]["score"] if res.reranked else None,
            spans=[root, *spans],
            tags=dict(self.tags),
        )
        self.last_trace = trace
        for ex in self.exporters:
            try:
                ex.export(trace)
            except Exception:
                log.exception("exporter %s failed", type(ex).__name__)


# --------------------------------------------------------------------------- exporters
class JSONLTraceExporter:
    def __init__(self, trace_dir: str | Path):
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, trace: Trace) -> None:
        path = self.dir / f"{trace.ts[:10]}.jsonl"
        line = json.dumps(trace.to_dict(), default=str)
        with self._lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class _JSONLSpanExporter:
    """OpenTelemetry SpanExporter writing OTLP-shaped JSON lines (fallback when no collector)."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult

        with open(self.path, "a", encoding="utf-8") as fh:
            for s in spans:
                fh.write(s.to_json(indent=None) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis: int = 30000):
        return True


def langfuse_otlp_config(host: str | None = None) -> tuple[str, dict] | None:
    """Langfuse ingests OTLP/HTTP at /api/public/otel with Basic auth (public:secret key)."""
    pk, sk = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        return None
    host = (host or os.environ.get("LANGFUSE_HOST") or "http://localhost:3000").rstrip("/")
    token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    return f"{host}/api/public/otel/v1/traces", {"Authorization": f"Basic {token}"}


class OTelExporter:
    """Re-emits each finished trace as OpenTelemetry spans with their real start/end times.

    Attribute conventions: OpenTelemetry GenAI (``gen_ai.*``) plus Langfuse's OTel mapping
    (``langfuse.observation.type=generation``, ``langfuse.trace.metadata.*``), so the same spans
    render as a proper trace tree with a token-counted generation in Langfuse.
    """

    def __init__(
        self,
        service_name: str = "dronedocs-rag",
        span_file: str | Path | None = None,
        otlp_endpoint: str | None = None,
        otlp_headers: dict | None = None,
        simple: bool = False,
    ):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        self.provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        proc = SimpleSpanProcessor if simple else BatchSpanProcessor
        self.targets: list[str] = []
        if otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            self.provider.add_span_processor(proc(OTLPSpanExporter(endpoint=otlp_endpoint, headers=otlp_headers or {})))
            self.targets.append(otlp_endpoint)
        if span_file:
            self.provider.add_span_processor(proc(_JSONLSpanExporter(Path(span_file))))
            self.targets.append(str(span_file))
        self.tracer = self.provider.get_tracer("ragobs")

    @staticmethod
    def _ns(t: float) -> int:
        return int(t * 1e9)

    @staticmethod
    def _attrs(span: Span) -> dict:
        out = {}
        for k, v in span.attributes.items():
            if isinstance(v, (str, bool, int, float)):
                out[f"rag.{k}"] = v
            elif v is None:
                continue
            else:
                out[f"rag.{k}"] = json.dumps(v, default=str)[:20000]
        return out

    def export(self, trace: Trace) -> None:
        from opentelemetry import trace as ot

        root = trace.spans[0]
        root_attrs = {
            "langfuse.trace.name": "dronedocs.ask",
            "langfuse.trace.input": trace.question,
            "langfuse.trace.output": trace.answer,
            "langfuse.trace.metadata.prompt_version": trace.prompt_version,
            "langfuse.trace.metadata.request_id": trace.request_id,
            "langfuse.trace.tags": ["refused"] if trace.refused else ["answered"],
            "rag.prompt_version": trace.prompt_version,
            "rag.refused": trace.refused,
            "rag.citation_coverage": trace.citation_coverage,
            "rag.cost_usd": trace.cost_usd,
            **{f"rag.tag.{k}": str(v) for k, v in trace.tags.items()},
        }
        parent = self.tracer.start_span("rag.request", start_time=self._ns(root.start_s), attributes=root_attrs)
        ctx = ot.set_span_in_context(parent)
        for s in trace.spans[1:]:
            attrs = self._attrs(s)
            if s.name == "generate":
                attrs.update(
                    {
                        "langfuse.observation.type": "generation",
                        "gen_ai.system": "openai_compatible",
                        "gen_ai.request.model": trace.model,
                        "gen_ai.usage.input_tokens": trace.prompt_tokens,
                        "gen_ai.usage.output_tokens": trace.completion_tokens,
                        "langfuse.observation.input": json.dumps(s.attributes.get("messages", []))[:20000],
                        "langfuse.observation.output": str(s.attributes.get("response", ""))[:20000],
                        "langfuse.observation.prompt.name": "dronedocs-answer",
                        "langfuse.observation.metadata.prompt_version": trace.prompt_version,
                        "langfuse.observation.cost_details": json.dumps({"total": trace.cost_usd}),
                    }
                )
            else:
                attrs["langfuse.observation.type"] = "span"
            child = self.tracer.start_span(
                f"rag.{s.name}", context=ctx, start_time=self._ns(s.start_s), attributes=attrs
            )
            child.end(end_time=self._ns(s.end_s))
        if trace.error:
            parent.set_status(ot.Status(ot.StatusCode.ERROR, trace.error))
        parent.end(end_time=self._ns(root.end_s))

    def flush(self) -> None:
        self.provider.force_flush()

    def shutdown(self) -> None:
        self.provider.shutdown()


def build_tracer(obs: dict, store=None, clock=None, tags: dict | None = None, simple_spans: bool = False) -> RAGTracer:
    from .config import resolve

    t = obs["tracing"]
    exporters: list[Any] = []
    trace_dir = resolve(obs["storage"]["trace_dir"])
    if t.get("jsonl", True):
        exporters.append(JSONLTraceExporter(trace_dir))
    endpoint, headers = None, None
    lf = langfuse_otlp_config(
        os.environ.get(t.get("langfuse_host_env", "LANGFUSE_HOST")) or t.get("default_langfuse_host")
    )
    if lf:
        endpoint, headers = lf
    elif os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"].rstrip("/") + "/v1/traces"
    span_file = trace_dir / "otel_spans.jsonl" if t.get("otel_spans_jsonl", True) else None
    if endpoint or span_file:
        exporters.append(
            OTelExporter(t.get("service_name", "dronedocs-rag"), span_file, endpoint, headers, simple=simple_spans)
        )
    if store is not None:
        exporters.append(store)
    return RAGTracer(
        exporters,
        obs["pricing"],
        clock=clock,
        tags=tags,
        max_attr_chars=t.get("max_attr_chars", 20000),
        capture_prompt=t.get("capture_prompt", True),
    )
