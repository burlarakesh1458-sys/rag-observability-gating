from datetime import datetime

from ragobs.analytics import aggregate, by_prompt_version, detect_anomalies, pct
from ragobs.cost import request_cost_usd
from ragobs.gate import compare
from ragobs.replay import IST, SimClock, build_schedule, prompt_at


def _row(day, refused=False, reason=None, cov=1.0, total=1000.0, pv="answer-v1.2", cost=0.001, err=None):
    return {
        "ts": f"2026-11-{day:02d}T10:00:00+05:30",
        "refused": int(refused),
        "refusal_reason": reason,
        "error": err,
        "citation_coverage": cov,
        "n_citations": 0 if refused else 1,
        "cost_usd": cost,
        "prompt_tokens": 2000,
        "completion_tokens": 100,
        "prompt_version": pv,
        "latency_retrieve_ms": 50.0,
        "latency_rerank_ms": 300.0,
        "latency_generate_ms": total - 350,
        "latency_total_ms": total,
    }


def test_percentiles():
    assert pct([1, 2, 3, 4, 5], 0.5) == 3
    assert pct([10, 20], 0.95) == 19.5
    assert pct([], 0.5) is None


def test_aggregate_rates():
    rows = [_row(1), _row(1, True, "refused_by_gate"), _row(1, True, "refused_no_citations"), _row(1, err="boom")]
    b = aggregate(rows)[0]
    assert b["requests"] == 4
    assert b["refusal_rate"] == 0.25 and b["unsupported_rate"] == 0.25 and b["error_rate"] == 0.25
    assert b["failure_rate"] == 0.5 and b["grounded_rate"] == 0.25


def test_anomaly_flags_incident_day_only():
    rows = []
    for d in (1, 2, 3, 5):
        rows += [_row(d) for _ in range(6)]
    rows += [_row(4, True, "refused_no_citations", pv="answer-v1.3") for _ in range(4)] + [_row(4) for _ in range(2)]
    flags = detect_anomalies(
        aggregate(rows),
        {"failure_rate": {"abs_increase": 0.15}, "citation_coverage": {"abs_drop": 0.15}},
        window=3,
        min_requests=5,
    )
    assert {f["bucket"] for f in flags} == {"2026-11-04"}
    assert any(f["metric"] == "failure_rate" and "answer-v1.3" in f["prompt_versions"] for f in flags)


def test_by_prompt_version():
    res = by_prompt_version([_row(1), _row(2, True, "refused_uncited", pv="answer-v1.3")])
    assert [r["bucket"] for r in res] == ["answer-v1.2", "answer-v1.3"] and res[1]["failure_rate"] == 1.0


def test_cost_table():
    pricing = {"m": {"input": 0.15, "output": 0.60}, "default": {"input": 0, "output": 0}}
    assert request_cost_usd("m", 1_000_000, 1_000_000, pricing) == 0.75
    assert request_cost_usd("llama3:8b", 5000, 500, pricing) == 0


def test_gate_compare_floor_and_drop():
    rules = {"faithfulness": {"max_drop": 0.05, "floor": 0.8}, "citation_accuracy": {"max_drop": 0.1, "floor": 0.8}}
    ok = compare(
        {"faithfulness": 0.93, "citation_accuracy": 0.95}, {"faithfulness": 0.95, "citation_accuracy": 0.95}, rules
    )
    assert all(c["status"] == "pass" for c in ok)
    drop = compare(
        {"faithfulness": 0.86, "citation_accuracy": 0.95}, {"faithfulness": 0.95, "citation_accuracy": 0.95}, rules
    )
    assert drop[0]["status"] == "fail" and "dropped" in drop[0]["why"]
    floor = compare({"faithfulness": 0.7, "citation_accuracy": None}, None, rules)
    assert [c["status"] for c in floor] == ["fail", "fail"]


def test_deploy_timeline_and_clock():
    assert prompt_at(datetime(2026, 11, 3, 9, 59, tzinfo=IST)).endswith("answer_v1.yaml")
    assert prompt_at(datetime(2026, 11, 3, 10, 0, tzinfo=IST)).endswith("answer_v1_3_concise.yaml")
    assert prompt_at(datetime(2026, 11, 4, 11, 0, tzinfo=IST)).endswith("answer_v1.yaml")
    sched = build_schedule(datetime(2026, 10, 31, tzinfo=IST), 6, 8)
    assert len(sched) == 48 and sched == sorted(sched)
    c = SimClock()
    c.set(datetime(2026, 11, 3, 12, 0, tzinfo=IST))
    assert abs(c() - datetime(2026, 11, 3, 12, 0, tzinfo=IST).timestamp()) < 1
