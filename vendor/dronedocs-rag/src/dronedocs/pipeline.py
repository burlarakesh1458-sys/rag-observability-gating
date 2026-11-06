"""RAG pipeline orchestrated as a LangGraph state machine.

    retrieve (BM25 + vector, RRF) -> rerank (cross-encoder) -> evidence_gate
        -> [refuse]                         (best rerank score below threshold)
        -> generate -> validate_citations   (refuse if citations missing / invented)

Every node reports to optional observers (``on_stage``) so an observability layer can trace
each step without this package depending on any tracing vendor.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, TypedDict

from .citations import REFUSAL_TEXT, CitationCheck, check_citations, normalize_citations, strip_invalid_citations
from .config import get_settings, load_prompt_bundle
from .llm import ChatClient, LLMResponse
from .rerank import Reranker, build_reranker
from .retrieval import Embedder, Hit, HybridIndex

log = logging.getLogger(__name__)


class Observer(Protocol):
    def on_stage(self, request_id: str, stage: str, payload: dict, duration_s: float) -> None: ...


@dataclass
class Citation:
    chunk_id: str
    source: str
    locator: str
    snippet: str


@dataclass
class AskResult:
    request_id: str
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    prompt_version: str
    model: str
    stage_timings: dict[str, float]
    retrieved: list[dict] = field(default_factory=list)
    reranked: list[dict] = field(default_factory=list)
    context_ids: list[str] = field(default_factory=list)
    prompt_messages: list[dict] = field(default_factory=list)
    raw_answer: str = ""
    usage: dict = field(default_factory=dict)
    citation_check: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self, include_prompt: bool = False) -> dict:
        d = asdict(self)
        if not include_prompt:
            d.pop("prompt_messages")
        return d


class RAGState(TypedDict, total=False):
    request_id: str
    question: str
    fused: list[Hit]
    retrieval_debug: dict
    reranked: list[Hit]
    gate_passed: bool
    messages: list[dict]
    llm: LLMResponse
    check: CitationCheck
    answer: str
    timings: dict[str, float]
    error: str


class RAGPipeline:
    def __init__(
        self,
        settings: dict | None = None,
        index: HybridIndex | None = None,
        reranker: Reranker | None = None,
        llm: ChatClient | None = None,
        prompt_bundle: dict | None = None,
        observers: list[Observer] | None = None,
    ):
        self.cfg = settings or get_settings()
        if index is None:
            emb = Embedder(self.cfg["embeddings"]["model"], self.cfg["embeddings"].get("query_instruction", ""))
            index = HybridIndex(self.cfg["paths"]["index_dir"], emb).load()
        self.index = index
        self.reranker = reranker or build_reranker(self.cfg["rerank"])
        self.llm = llm or ChatClient.from_settings(self.cfg["llm"], self.cfg["paths"])
        self.prompts = prompt_bundle or load_prompt_bundle(self.cfg["app"]["prompt_file"])
        self.observers: list[Observer] = list(observers or [])
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ helpers
    def _emit(self, state: RAGState, stage: str, payload: dict, dt: float) -> None:
        state.setdefault("timings", {})[stage] = dt
        for ob in self.observers:
            try:
                ob.on_stage(state["request_id"], stage, payload, dt)
            except Exception:
                log.exception("observer %r failed on %s", ob, stage)

    def _hit_dict(self, h: Hit) -> dict:
        c = self.index.chunks[h.chunk_id]
        return {
            "chunk_id": h.chunk_id,
            "score": round(h.score, 4),
            "rank": h.rank,
            "source": c.source,
            "locator": c.locator,
            "pre_rank": h.metadata.get("pre_rank"),
            "n_tokens": c.n_tokens,
        }

    def build_messages(self, question: str, hits: list[Hit]) -> list[dict]:
        p = self.prompts
        ctx = "\n".join(
            p["context_item"].format(
                chunk_id=h.chunk_id,
                source=self.index.chunks[h.chunk_id].source,
                locator=self.index.chunks[h.chunk_id].locator,
                text=h.text,
            )
            for h in hits
        )
        return [
            {"role": "system", "content": p["system"].strip()},
            {"role": "user", "content": p["user"].format(context=ctx, question=question)},
        ]

    # ------------------------------------------------------------------ nodes
    def _retrieve(self, state: RAGState) -> RAGState:
        r = self.cfg["retrieval"]
        t0 = time.perf_counter()
        hits, debug = self.index.hybrid_search(
            state["question"], r["vector_k"], r["bm25_k"], r["rrf_k"], r["fused_k"], r.get("mode", "hybrid")
        )
        dt = time.perf_counter() - t0
        state.update(fused=hits, retrieval_debug=debug)
        self._emit(
            state,
            "retrieve",
            {
                "mode": r.get("mode", "hybrid"),
                "fused": [self._hit_dict(h) for h in hits],
                "vector": debug["vector"][:10],
                "bm25": debug["bm25"][:10],
            },
            dt,
        )
        return state

    def _rerank(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        top = self.reranker.rerank(state["question"], state["fused"], self.cfg["rerank"]["top_n"])
        dt = time.perf_counter() - t0
        state["reranked"] = top
        self._emit(
            state, "rerank", {"provider": self.cfg["rerank"]["provider"], "order": [self._hit_dict(h) for h in top]}, dt
        )
        return state

    def _gate(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        top = state.get("reranked") or []
        thr = self.cfg["guardrails"]["min_rerank_score"]
        is_ce = self.cfg["rerank"]["provider"] == "cross_encoder"
        passed = bool(top) and (not is_ce or top[0].score >= thr)
        state["gate_passed"] = passed
        self._emit(
            state,
            "evidence_gate",
            {"passed": passed, "best_score": top[0].score if top else None, "threshold": thr},
            time.perf_counter() - t0,
        )
        return state

    def _generate(self, state: RAGState) -> RAGState:
        msgs = self.build_messages(state["question"], state["reranked"])
        state["messages"] = msgs
        t0 = time.perf_counter()
        resp = self.llm.chat(msgs)
        dt = time.perf_counter() - t0
        state["llm"] = resp
        self._emit(
            state,
            "generate",
            {
                "model": resp.model,
                "prompt_version": self.prompts["version"],
                "messages": msgs,
                "response": resp.text,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "retries": resp.retries,
                "cached": resp.cached,
            },
            dt,
        )
        return state

    def _validate(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        g = self.cfg["guardrails"]
        ctx_ids = {h.chunk_id for h in state["reranked"]}
        chk = check_citations(
            state["llm"].text,
            ctx_ids,
            set(self.index.chunks),
            self.prompts.get("refusal_sentinel", "INSUFFICIENT_EVIDENCE"),
            g.get("require_citations", True),
            g.get("max_uncited_sentence_ratio", 0.5),
        )
        answer = (
            REFUSAL_TEXT
            if chk.refused
            else strip_invalid_citations(normalize_citations(state["llm"].text), chk.invalid_ids)
        )
        state.update(check=chk, answer=answer)
        self._emit(state, "validate_citations", {**asdict(chk), "coverage": chk.coverage}, time.perf_counter() - t0)
        return state

    def _refuse(self, state: RAGState) -> RAGState:
        state["answer"] = REFUSAL_TEXT
        state["check"] = CitationCheck(status="refused_by_gate")
        self._emit(state, "refuse", {"reason": "refused_by_gate"}, 0.0)
        return state

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(RAGState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("rerank", self._rerank)
        g.add_node("evidence_gate", self._gate)
        g.add_node("generate", self._generate)
        g.add_node("validate_citations", self._validate)
        g.add_node("refuse", self._refuse)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "rerank")
        g.add_edge("rerank", "evidence_gate")
        g.add_conditional_edges(
            "evidence_gate",
            lambda s: "generate" if s["gate_passed"] else "refuse",
            {"generate": "generate", "refuse": "refuse"},
        )
        g.add_edge("generate", "validate_citations")
        g.add_edge("validate_citations", END)
        g.add_edge("refuse", END)
        return g.compile()

    # ------------------------------------------------------------------ public
    def ask(self, question: str, request_id: str | None = None) -> AskResult:
        rid = request_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        state: dict[str, Any] = {"request_id": rid, "question": question, "timings": {}}
        error = None
        try:
            state = self.graph.invoke(state)
        except Exception as e:
            log.exception("pipeline failed")
            error = f"{type(e).__name__}: {e}"
            state["answer"] = "The system hit an error while answering. Please retry."
            state["check"] = CitationCheck(status="error")
        timings = dict(state.get("timings", {}))
        timings["total"] = time.perf_counter() - t0
        chk: CitationCheck = state["check"]
        cites = []
        if not chk.refused:
            for cid in chk.valid_ids:
                c = self.index.chunks[cid]
                cites.append(Citation(cid, c.source, c.locator, c.text[:240]))
        llm: LLMResponse | None = state.get("llm")
        res = AskResult(
            request_id=rid,
            question=question,
            answer=state["answer"],
            refused=chk.refused,
            refusal_reason=None if not chk.refused else chk.status,
            citations=cites,
            prompt_version=self.prompts["version"],
            model=self.llm.model,
            stage_timings=timings,
            retrieved=[self._hit_dict(h) for h in state.get("fused", [])],
            reranked=[self._hit_dict(h) for h in state.get("reranked", [])],
            context_ids=[h.chunk_id for h in state.get("reranked", [])],
            prompt_messages=state.get("messages", []),
            raw_answer=llm.text if llm else "",
            usage={
                "prompt_tokens": llm.prompt_tokens,
                "completion_tokens": llm.completion_tokens,
                "retries": llm.retries,
            }
            if llm
            else {"prompt_tokens": 0, "completion_tokens": 0, "retries": 0},
            citation_check={**asdict(chk), "coverage": chk.coverage},
            error=error,
        )
        for ob in self.observers:
            if hasattr(ob, "on_result"):
                try:
                    ob.on_result(res)
                except Exception:
                    log.exception("observer on_result failed")
        return res
