"""Offline evaluation over the golden set, with threshold gating for CI.

    python -m dronedocs.eval.run_eval            # full run, exits 1 if a metric is below threshold
    dronedocs eval --limit 10 --no-gate          # quick smoke run

Writes ``reports/eval_<timestamp>.json`` (per-item rows + summary) and ``reports/latest.json``.
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_settings, load_prompt_bundle, project_root
from ..llm import ChatClient, LLMError
from . import metrics as M

log = logging.getLogger(__name__)
_CK_LOCK = threading.Lock()


def load_golden(path: str | Path, limit: int | None = None) -> list[dict]:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        # keep the answerable / unanswerable mix in small smoke runs
        ans = [r for r in rows if r["answerable"]]
        una = [r for r in rows if not r["answerable"]]
        k_una = max(1, round(limit * len(una) / len(rows)))
        rows = ans[: limit - k_una] + una[:k_una]
    return rows


def build_judge(cfg: dict) -> ChatClient:
    j = cfg["judge"]
    return ChatClient.from_settings(
        cfg["llm"],
        cfg["paths"],
        model=j["model"],
        temperature=j.get("temperature", 0.0),
        max_tokens=j.get("max_tokens", 700),
    )


def eval_item(pipeline, judge: ChatClient | None, judge_prompt: str, item: dict) -> dict:
    res = pipeline.ask(item["question"], request_id=f"eval-{item['id']}")
    chunks = pipeline.index.chunks
    ctx = [chunks[c] for c in res.context_ids]
    rels = [M.chunk_is_relevant(c, item) for c in ctx]
    raw_ids = res.citation_check.get("cited_ids", [])
    valid_ids = res.citation_check.get("valid_ids", [])
    cited = [chunks[c] for c in valid_ids if c in chunks]
    row = {
        "id": item["id"],
        "question": item["question"],
        "answerable": item["answerable"],
        "answer": res.answer,
        "raw_answer": res.raw_answer,
        "refused": res.refused,
        "refusal_reason": res.refusal_reason,
        "error": res.error,
        "context_ids": res.context_ids,
        "cited_ids": valid_ids,
        "raw_cited_ids": raw_ids,
        "best_rerank_score": res.reranked[0]["score"] if res.reranked else None,
        "latency_s": res.stage_timings,
        "usage": res.usage,
        "refusal_correct": res.refused != item["answerable"],
    }
    if item["answerable"]:
        fused_ids = [h["chunk_id"] for h in res.retrieved]
        row.update(
            context_recall=float(M.evidence_in(ctx, item) or any(rels)),
            context_precision=M.average_precision(rels),
            fused_recall=float(any(M.chunk_is_relevant(chunks[c], item) for c in fused_ids)),
            mrr=M.reciprocal_rank(rels),
        )
        if not res.refused:
            row["citation_accuracy"] = float(any(M.chunk_is_relevant(c, item) for c in cited))
            row["citation_validity"] = (
                (len([c for c in raw_ids if c in res.context_ids]) / len(raw_ids)) if raw_ids else 0.0
            )
            if judge is not None:
                try:
                    jr = M.judge_answer(judge, item, res.answer, cited, judge_prompt)
                except LLMError as e:  # judge unavailable (e.g. daily quota): row is not scored, not failed
                    row["error"] = f"judge {e}"[:600]
                    return row
                row.update(
                    faithfulness=jr.faithfulness,
                    answer_relevancy=jr.answer_relevancy,
                    judge_claims=jr.n_claims,
                    judge_supported=jr.n_supported,
                    judge_reason=jr.reason,
                    judge_tokens=jr.tokens,
                )
        else:
            row["answer_relevancy"] = 0.0  # a false refusal answers nothing
    return row


def _mean(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def is_quota_error(row: dict) -> bool:
    """Provider daily-quota failures say nothing about answer quality: they are reported as
    'not evaluated' instead of being scored as refusals."""
    return bool(row.get("error")) and "quota exhausted" in str(row["error"])


def summarize(rows: list[dict]) -> dict:
    skipped = [r["id"] for r in rows if is_quota_error(r)]
    rows = [r for r in rows if not is_quota_error(r)]
    if not rows:
        return {"n_items": 0, "n_not_evaluated_quota": len(skipped), "not_evaluated_ids": skipped}
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    lat = {}
    for stage in ("retrieve", "rerank", "generate", "total"):
        vals = [r["latency_s"].get(stage) for r in rows if r["latency_s"].get(stage) is not None]
        lat[stage] = {
            "p50_ms": round(1000 * (_pct(vals, 0.5) or 0), 1),
            "p95_ms": round(1000 * (_pct(vals, 0.95) or 0), 1),
        }
    return {
        "n_items": len(rows),
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "n_not_evaluated_quota": len(skipped),
        "not_evaluated_ids": skipped,
        "faithfulness": _mean([r.get("faithfulness") for r in answered]),
        "answer_relevancy": _mean([r.get("answer_relevancy") for r in ans]),
        "context_recall": _mean([r["context_recall"] for r in ans]),
        "context_precision": _mean([r["context_precision"] for r in ans]),
        "citation_accuracy": _mean([r.get("citation_accuracy") for r in answered]),
        "citation_validity": _mean([r.get("citation_validity") for r in answered]),
        "refusal_correctness": _mean([float(r["refusal_correct"]) for r in rows]),
        "answer_rate_answerable": round(len(answered) / len(ans), 4) if ans else None,
        "refusal_rate_unanswerable": _mean([float(r["refused"]) for r in una]),
        "fused_recall@20": _mean([r["fused_recall"] for r in ans]),
        "mrr@top_n": _mean([r["mrr"] for r in ans]),
        "errors": sum(1 for r in rows if r.get("error")),
        "latency": lat,
        "tokens": {
            "prompt": sum(r["usage"].get("prompt_tokens", 0) for r in rows),
            "completion": sum(r["usage"].get("completion_tokens", 0) for r in rows),
            "judge": sum(r.get("judge_tokens", 0) for r in rows),
        },
        "mean_prompt_tokens": round(
            statistics.mean(r["usage"].get("prompt_tokens", 0) for r in rows if r["usage"].get("prompt_tokens")), 1
        )
        if any(r["usage"].get("prompt_tokens") for r in rows)
        else None,
    }


def check_thresholds(summary: dict, thresholds: dict) -> list[str]:
    failures = []
    for metric, thr in thresholds.items():
        val = summary.get(metric)
        if val is None or val < thr:
            failures.append(f"{metric}={val} < {thr}")
    return failures


def evaluate(
    pipeline,
    items: list[dict],
    judge: ChatClient | None,
    concurrency: int = 2,
    judge_prompt: str | None = None,
    checkpoint: str | Path | None = None,
) -> tuple[list[dict], dict]:
    """Evaluate items; with ``checkpoint`` every finished row is appended to a JSONL file and rows that
    already completed without error are reused, so a run killed by OOM / quota can be resumed."""
    jp = judge_prompt or load_prompt_bundle("configs/prompts/judge_v1.yaml")["judge"]
    done: dict[str, dict] = {}
    ck = Path(checkpoint) if checkpoint else None
    if ck and ck.exists():
        for line in ck.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row.get("error"):
                done[row["id"]] = row
        log.info("resuming: %d/%d items already evaluated", len(done), len(items))
    todo = [it for it in items if it["id"] not in done]

    def _run(it: dict) -> dict:
        row = eval_item(pipeline, judge, jp, it)
        if ck:
            with _CK_LOCK, open(ck, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        fresh = {r["id"]: r for r in ex.map(_run, todo)}
    rows = [done.get(it["id"]) or fresh[it["id"]] for it in items]
    return rows, summarize(rows)


def run(
    limit: int | None = None,
    use_ragas: bool = True,
    gate: bool = True,
    out: str | None = None,
    settings: dict | None = None,
    pipeline=None,
    resume: bool = False,
    report_only: bool = False,
) -> int:
    from ..pipeline import RAGPipeline

    cfg = settings or get_settings()
    ecfg = cfg["eval"]
    items = load_golden(ecfg["golden_path"], limit)
    judge = build_judge(cfg)
    judge_bundle = load_prompt_bundle("configs/prompts/judge_v1.yaml")
    t0 = time.perf_counter()
    ck_dir = Path(ecfg.get("report_dir", "reports"))
    ck_dir = ck_dir if ck_dir.is_absolute() else project_root() / ck_dir
    ck_dir.mkdir(parents=True, exist_ok=True)
    prompt_bundle = load_prompt_bundle(cfg["app"]["prompt_file"])
    ck = ck_dir / f"checkpoint_{prompt_bundle['version']}_{cfg['llm']['model'].replace('/', '_')}.jsonl"
    if report_only:
        # rebuild the report from the checkpoint without any model or LLM calls (last row per id wins)
        last: dict[str, dict] = {}
        for line in ck.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] not in last or not row.get("error") or last[row["id"]].get("error"):
                last[row["id"]] = row
        rows = [last[it["id"]] for it in items if it["id"] in last]
        summary = summarize(rows)
        use_ragas = False

        class _P:  # minimal stand-in for the report header
            prompts = prompt_bundle
            llm = type("L", (), {"model": cfg["llm"]["model"]})()

        pipeline = _P()
    else:
        pipeline = pipeline or RAGPipeline(cfg)
        if not resume and ck.exists():
            ck.unlink()
        rows, summary = evaluate(pipeline, items, judge, ecfg.get("concurrency", 2), judge_bundle["judge"], ck)
    summary["wall_time_s"] = None if report_only else round(time.perf_counter() - t0, 1)

    ragas_summary = None
    if use_ragas and ecfg.get("ragas", True):
        try:
            from .ragas_eval import run_ragas

            ragas_summary = run_ragas(cfg, items, rows, pipeline.index.chunks, ecfg.get("ragas_sample", 8))
        except Exception as e:
            log.warning("Ragas evaluation unavailable, using built-in judge only: %s", e)
            ragas_summary = {"status": "unavailable", "error": f"{type(e).__name__}: {e}"[:400]}

    failures = check_thresholds(summary, ecfg["thresholds"])
    if summary.get("n_not_evaluated_quota"):
        log.warning(
            "%d items not evaluated (provider quota): %s",
            summary["n_not_evaluated_quota"],
            summary["not_evaluated_ids"],
        )
    report = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator_model": pipeline.llm.model,
        "judge_model": judge.model,
        "prompt_version": pipeline.prompts["version"],
        "judge_prompt_version": judge_bundle["version"],
        "retrieval": cfg["retrieval"],
        "rerank": {k: cfg["rerank"][k] for k in ("provider", "model", "top_n")},
        "chunking": cfg["chunking"],
        "embedding_model": cfg["embeddings"]["model"],
        "thresholds": ecfg["thresholds"],
        "gate_failures": failures,
        "passed": not failures,
        "partial": bool(summary.get("n_not_evaluated_quota")),
        "summary": summary,
        "ragas": ragas_summary,
        "rows": rows,
    }
    rep_dir = Path(ecfg.get("report_dir", "reports"))
    if not rep_dir.is_absolute():
        rep_dir = project_root() / rep_dir
    rep_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out) if out else rep_dir / f"eval_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    (rep_dir / "latest.json").write_text(json.dumps(report, indent=2, default=str))
    print_summary(report)
    print(f"\nreport: {out_path}")
    if failures:
        print("QUALITY GATE FAILED: " + "; ".join(failures))
        return 1 if gate else 0
    print("QUALITY GATE PASSED")
    return 0


def print_summary(report: dict) -> None:
    s = report["summary"]
    thr = report["thresholds"]
    print(
        f"\nDroneDocs eval  gen={report['generator_model']}  judge={report['judge_model']}  "
        f"prompt={report['prompt_version']}  n={s['n_items']} ({s['n_answerable']} answerable)"
    )
    print(f"{'metric':<28}{'value':>8}{'threshold':>11}")
    for k in (
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
        "citation_accuracy",
        "citation_validity",
        "refusal_correctness",
        "answer_rate_answerable",
        "refusal_rate_unanswerable",
        "fused_recall@20",
        "mrr@top_n",
    ):
        v = s.get(k)
        t = thr.get(k)
        mark = "" if t is None else ("  ok" if v is not None and v >= t else "  FAIL")
        print(f"{k:<28}{'n/a' if v is None else f'{v:.3f}':>8}{'' if t is None else f'{t:.2f}':>11}{mark}")
    if s.get("n_not_evaluated_quota"):
        print(
            f"PARTIAL: {s['n_not_evaluated_quota']} items not evaluated (provider quota): "
            + ", ".join(s["not_evaluated_ids"])
        )
    lat = s["latency"]
    print("latency p50/p95 (ms): " + ", ".join(f"{k} {v['p50_ms']:.0f}/{v['p95_ms']:.0f}" for k, v in lat.items()))
    if report.get("ragas"):
        print(f"ragas: {json.dumps({k: v for k, v in report['ragas'].items() if k != 'rows'})[:400]}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(run(gate="--no-gate" not in sys.argv, use_ragas="--no-ragas" not in sys.argv))
