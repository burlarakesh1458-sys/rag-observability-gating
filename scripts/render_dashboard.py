"""Render the dashboard + gate comparison to PNG (docs/) and dump the daily metrics table."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragobs.analytics import aggregate, by_prompt_version, detect_anomalies  # noqa: E402
from ragobs.charts import IST, dashboard_figure, gate_figure  # noqa: E402
from ragobs.config import obs_settings, resolve  # noqa: E402
from ragobs.store import MetricsStore  # noqa: E402


def main() -> None:
    obs = obs_settings()
    store = MetricsStore(resolve(obs["storage"]["sqlite_path"]))
    rows = [r for r in store.requests() if json.loads(r["tags"] or "{}").get("source") == "replay-backfill"]
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    if rows:
        buckets = aggregate(rows, "day", tz=IST)
        a = obs["anomaly"]
        flags = detect_anomalies(buckets, a["rules"], a["window_days"], a["min_requests"])
        versions = by_prompt_version(rows)
        fig = dashboard_figure(
            buckets, flags, f"DroneDocs RAG - daily health ({len(rows)} real requests, IST days)", versions
        )
        fig.savefig(docs / "dashboard.png", dpi=110)
        out = {"buckets": buckets, "anomalies": flags, "by_prompt_version": versions}
        (ROOT / "reports" / "daily_metrics.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"{len(buckets)} days, {len(flags)} anomaly flags")
        for f in flags:
            print("  ", f["bucket"], f["metric"], f["reason"], f["prompt_versions"])
    gates = [json.loads(p.read_text()) for p in sorted((ROOT / "reports").glob("gate_answer-*_ci.json"))]
    if gates:
        gates.sort(key=lambda r: (not r["passed"], r["prompt_version"]))
        gate_figure(gates).savefig(docs / "regression_gate.png", dpi=110)
        print("gate chart:", [(g["prompt_version"], g["passed"]) for g in gates])


if __name__ == "__main__":
    main()
