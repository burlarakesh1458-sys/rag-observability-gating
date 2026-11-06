"""Matplotlib renderings of the dashboard panels (used for docs/ PNGs and by the Streamlit app)."""

from __future__ import annotations

from datetime import timedelta, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IST = timezone(timedelta(hours=5, minutes=30))
RED = "#d62728"


def _flag_marks(ax, xs, ys, flags_for_metric):
    fx = [x for x, b in zip(xs, flags_for_metric, strict=True) if b]
    fy = [y for y, b in zip(ys, flags_for_metric, strict=True) if b]
    if fx:
        ax.scatter(fx, fy, s=160, facecolors="none", edgecolors=RED, linewidths=2.2, zorder=5, label="anomaly")


def dashboard_figure(buckets: list[dict], flags: list[dict], title: str, versions: list[dict] | None = None):
    xs = [b["bucket"][5:] for b in buckets]
    flagged = {(f["bucket"], f["metric"]) for f in flags}

    def fl(metric):
        return [(b["bucket"], metric) in flagged for b in buckets]

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.6))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.bar(xs, [b["requests"] for b in buckets], color="#4c72b0")
    ax.set_title("Requests per day")

    ax = axes[0, 1]
    for stage, color in (("total", "#4c72b0"), ("generate", "#dd8452"), ("rerank", "#55a868"), ("retrieve", "#8172b3")):
        ax.plot(xs, [b[f"latency_{stage}_p95_ms"] for b in buckets], marker="o", color=color, label=f"{stage} p95")
    ax.plot(xs, [b["latency_total_p50_ms"] for b in buckets], ls="--", color="#4c72b0", label="total p50")
    _flag_marks(ax, xs, [b["latency_total_p95_ms"] for b in buckets], fl("latency_total_p95_ms"))
    ax.set_title("Latency by stage (ms)")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ys = [1000 * b["cost_per_request_usd"] for b in buckets]
    ax.plot(xs, ys, marker="o", color="#937860")
    _flag_marks(ax, xs, ys, fl("cost_per_request_usd"))
    ax.set_title("Cost per request (USD x 1e-3)")

    ax = axes[1, 0]
    ys = [b["citation_coverage"] for b in buckets]
    ax.plot(xs, ys, marker="o", color="#55a868", label="citation coverage")
    ax.plot(xs, [b["grounded_rate"] for b in buckets], marker="s", color="#2a9d8f", label="grounded answer rate")
    _flag_marks(ax, xs, ys, fl("citation_coverage"))
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Grounding")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ys = [b["failure_rate"] for b in buckets]
    ax.plot(xs, ys, marker="o", color=RED, label="failure rate (error + unsupported)")
    ax.plot(xs, [b["refusal_rate"] for b in buckets], marker="s", color="#8c8c8c", label="refusal rate (no evidence)")
    _flag_marks(ax, xs, ys, fl("failure_rate"))
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Failure / refusal rate")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    if versions:
        names = [v["bucket"] for v in versions]
        x = range(len(names))
        w = 0.27
        ax.bar(
            [i - w for i in x],
            [v["citation_coverage"] for v in versions],
            w,
            label="citation coverage",
            color="#55a868",
        )
        ax.bar(list(x), [v["failure_rate"] for v in versions], w, label="failure rate", color=RED)
        ax.bar([i + w for i in x], [v["refusal_rate"] for v in versions], w, label="refusal rate", color="#8c8c8c")
        ax.set_xticks(list(x), [f"{n}\n(n={v['requests']})" for n, v in zip(names, versions, strict=True)], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    ax.set_title("By prompt version")

    for a in axes.flat:
        a.grid(alpha=0.3)
        for lbl in a.get_xticklabels():
            lbl.set_fontsize(8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def gate_figure(reports: list[dict]):
    """Side-by-side gate metrics for several gate reports (e.g. baseline prompt vs regressed prompt)."""
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "citation_accuracy", "refusal_correctness"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.8 / max(1, len(reports))
    colors = ["#55a868", RED, "#4c72b0", "#dd8452"]
    for i, r in enumerate(reports):
        raw = [r["summary"].get(m) for m in metrics]
        vals = [v or 0 for v in raw]
        xs = [j + (i - (len(reports) - 1) / 2) * w for j in range(len(metrics))]
        bars = ax.bar(
            xs,
            vals,
            w,
            label=f"{r['prompt_version']} ({'PASS' if r['passed'] else 'FAIL'})",
            color=colors[i % len(colors)],
        )
        for b, v in zip(bars, raw, strict=True):
            # n/a = nothing to score (e.g. every answer was rejected as uncited) - that fails the gate too
            label = "n/a" if v is None else f"{v:.2f}"
            ax.text(b.get_x() + b.get_width() / 2, (v or 0) + 0.01, label, ha="center", fontsize=8)
    floors = reports[0]["checks"]
    for j, m in enumerate(metrics):
        fl = next((c["floor"] for c in floors if c["metric"] == m), None)
        if fl is not None:
            ax.hlines(fl, j - 0.45, j + 0.45, colors="black", linestyles="dotted")
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_ylim(0, 1.12)
    ax.set_title("CI regression gate on the golden subset (dotted = hard floor; n/a = no answer left to score)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
