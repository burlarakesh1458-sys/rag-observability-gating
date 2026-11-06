"""CI regression gate.

Runs the golden eval (from the vendored DroneDocs package) through the *instrumented* pipeline
with the prompt/config in this commit, compares to the stored baseline and exits non-zero when
a key metric regresses:

    fail if  value < floor            (absolute quality bar)
    fail if  baseline - value > max_drop   (relative regression vs. main)

The prompt version is recorded on every trace and in the gate report, so a failing PR points
straight at the prompt/config change that caused it.

    python -m ragobs.gate                       # CI subset, compare to baseline
    python -m ragobs.gate --subset full
    python -m ragobs.gate --update-baseline     # after an intentional, reviewed improvement
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import UTC, datetime

from .config import ROOT, VENDOR, obs_settings, resolve

KEY_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    "citation_accuracy",
    "refusal_correctness",
)


def compare(current: dict, baseline: dict | None, rules: dict) -> list[dict]:
    checks = []
    for metric, rule in rules.items():
        val = current.get(metric)
        base = (baseline or {}).get(metric)
        status, why = "pass", ""
        if val is None:
            status, why = "fail", "metric missing"
        elif val < rule["floor"]:
            status, why = "fail", f"below floor {rule['floor']}"
        elif base is not None and base - val > rule["max_drop"]:
            status, why = "fail", f"dropped {base - val:.3f} > {rule['max_drop']} vs baseline"
        checks.append(
            {
                "metric": metric,
                "value": val,
                "baseline": base,
                "floor": rule["floor"],
                "max_drop": rule["max_drop"],
                "status": status,
                "why": why,
            }
        )
    return checks


def markdown(report: dict) -> str:
    lines = [
        f"### RAG regression gate: {'PASSED' if report['passed'] else 'FAILED'}",
        f"prompt `{report['prompt_version']}` · model `{report['model']}` · judge `{report['judge_model']}` · "
        f"subset `{report['subset']}` (n={report['summary']['n_items']})",
        "",
        "| metric | value | baseline | floor | max drop | status |",
        "|---|---|---|---|---|---|",
    ]
    for c in report["checks"]:
        v = "n/a" if c["value"] is None else f"{c['value']:.3f}"
        b = "n/a" if c["baseline"] is None else f"{c['baseline']:.3f}"
        lines.append(
            f"| {c['metric']} | {v} | {b} | {c['floor']} | {c['max_drop']} | "
            f"{'✅' if c['status'] == 'pass' else '❌ ' + c['why']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--subset", choices=["ci", "full"], default="ci")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from dronedocs.eval.run_eval import build_judge, evaluate, load_golden

    from .instrument import build_instrumented_pipeline, build_store

    obs = obs_settings()
    gate_cfg = obs["regression_gate"]
    items = load_golden(VENDOR / "eval" / "golden.jsonl")
    if args.subset == "ci":
        keep = set(gate_cfg["ci_subset"])
        items = [it for it in items if it["id"] in keep]
    store = build_store()
    pipe, _ = build_instrumented_pipeline(
        args.prompt_file, args.model, store=store, tags={"source": "regression-gate", "subset": args.subset}
    )
    judge = build_judge(pipe.cfg)
    t0 = time.perf_counter()
    rows, summary = evaluate(pipe, items, judge, concurrency=1)
    summary["wall_time_s"] = round(time.perf_counter() - t0, 1)

    base_path = resolve(gate_cfg["baseline_file"])
    baseline = json.loads(base_path.read_text())["summary"] if base_path.exists() else None
    checks = compare(summary, baseline, gate_cfg["metrics"])
    passed = all(c["status"] == "pass" for c in checks)
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    report = {
        "timestamp": ts,
        "prompt_version": pipe.prompts["version"],
        "prompt_file": pipe.prompts["_path"],
        "model": pipe.llm.model,
        "judge_model": judge.model,
        "subset": args.subset,
        "passed": passed,
        "checks": checks,
        "summary": summary,
        "rows": rows,
        "baseline_file": str(base_path.relative_to(ROOT)),
        "baseline_found": baseline is not None,
    }
    out = resolve(args.out or f"reports/gate_{pipe.prompts['version']}_{args.subset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    store.record_eval(
        uuid.uuid4().hex[:12], ts, pipe.prompts["version"], pipe.llm.model, judge.model, args.subset, passed, summary
    )
    md = markdown(report)
    print(md)
    (resolve("reports") / "gate_summary.md").write_text(md + "\n")
    if args.update_baseline:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(
            json.dumps(
                {k: report[k] for k in ("timestamp", "prompt_version", "model", "judge_model", "subset", "summary")},
                indent=2,
            )
        )
        print(f"baseline updated -> {base_path}")
        return 0
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
