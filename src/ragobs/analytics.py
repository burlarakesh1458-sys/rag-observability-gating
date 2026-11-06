"""Aggregate request rows into time buckets and flag anomalies against a trailing baseline.

Metric definitions (per bucket):
  latency_<stage>_p50/p95_ms  percentiles, never just the mean (means hide the tail)
  cost_per_request_usd        mean USD cost from token usage x pricing table
  citation_coverage           mean share of sentences carrying a valid citation, over answered requests
  grounded_rate               answered with >=1 valid citation / all requests
  refusal_rate                declined because evidence was insufficient (gate or model) / all requests
  unsupported_rate            model produced an answer the citation validator rejected / all requests
  error_rate                  pipeline exceptions / all requests
  failure_rate                error_rate + unsupported_rate  ("errors out or produces a response it cannot support")
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime

from .store import UNSUPPORTED


def pct(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def bucket_key(ts: str, granularity: str = "day") -> str:
    dt = datetime.fromisoformat(ts)
    return dt.strftime("%Y-%m-%d") if granularity == "day" else dt.strftime("%Y-%m-%d %H:00")


def aggregate(rows: list[dict], granularity: str = "day", tz=None) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        ts = r["ts"]
        if tz is not None:
            ts = datetime.fromisoformat(ts).astimezone(tz).isoformat()
        groups[bucket_key(ts, granularity)].append(r)
    out = []
    for key in sorted(groups):
        g = groups[key]
        n = len(g)
        answered = [r for r in g if not r["refused"] and not r["error"]]
        unsupported = [r for r in g if r["refusal_reason"] in UNSUPPORTED]
        declined = [r for r in g if r["refused"] and r["refusal_reason"] not in UNSUPPORTED and not r["error"]]
        errors = [r for r in g if r["error"]]
        b = {
            "bucket": key,
            "requests": n,
            "prompt_versions": sorted({r["prompt_version"] for r in g}),
            "cost_per_request_usd": statistics.mean(r["cost_usd"] or 0 for r in g),
            "cost_total_usd": sum(r["cost_usd"] or 0 for r in g),
            "tokens_per_request": statistics.mean((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in g),
            "citation_coverage": statistics.mean(r["citation_coverage"] for r in answered) if answered else 0.0,
            "grounded_rate": sum(1 for r in answered if r["n_citations"] > 0) / n,
            "refusal_rate": len(declined) / n,
            "unsupported_rate": len(unsupported) / n,
            "error_rate": len(errors) / n,
        }
        b["failure_rate"] = b["error_rate"] + b["unsupported_rate"]
        for stage in ("retrieve", "rerank", "generate", "total"):
            vals = [r[f"latency_{stage}_ms"] for r in g if r[f"latency_{stage}_ms"] is not None]
            b[f"latency_{stage}_p50_ms"] = pct(vals, 0.5)
            b[f"latency_{stage}_p95_ms"] = pct(vals, 0.95)
        out.append(b)
    return out


def by_prompt_version(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["prompt_version"]].append(r)
    res = []
    for pv, g in sorted(groups.items()):
        agg = aggregate([{**r, "ts": "2000-01-01T00:00:00+00:00"} for r in g])[0]
        agg["bucket"] = pv
        res.append(agg)
    return res


def detect_anomalies(buckets: list[dict], rules: dict, window: int = 3, min_requests: int = 5) -> list[dict]:
    """Flag buckets whose metric moved past a rule threshold vs. the median of the previous ``window`` buckets."""
    flags = []
    for i, b in enumerate(buckets):
        prev = [p for p in buckets[max(0, i - window) : i] if p["requests"] >= min_requests]
        if len(prev) < min(2, window) or b["requests"] < min_requests:
            continue
        for metric, rule in rules.items():
            cur = b.get(metric)
            base_vals = [p[metric] for p in prev if p.get(metric) is not None]
            if cur is None or not base_vals:
                continue
            base = statistics.median(base_vals)
            reason = None
            if "abs_increase" in rule and cur - base > rule["abs_increase"]:
                reason = f"+{cur - base:.2f} abs vs baseline {base:.2f}"
            if "abs_drop" in rule and base - cur > rule["abs_drop"]:
                reason = f"-{base - cur:.2f} abs vs baseline {base:.2f}"
            if "rel_increase" in rule and base > 0 and (cur - base) / base > rule["rel_increase"]:
                reason = f"+{100 * (cur - base) / base:.0f}% vs baseline {base:.4g}"
            if reason:
                flags.append(
                    {
                        "bucket": b["bucket"],
                        "metric": metric,
                        "value": cur,
                        "baseline": base,
                        "reason": reason,
                        "prompt_versions": b["prompt_versions"],
                    }
                )
    return flags
