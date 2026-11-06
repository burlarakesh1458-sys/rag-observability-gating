import json


def test_trace_captures_every_stage(make_traced_pipeline):
    pipe, tracer, _store, _tmp = make_traced_pipeline("Nano drones weigh at most 250 g [rules.pdf#c000].")
    res = pipe.ask("max nano weight?")
    t = tracer.last_trace
    assert [s.name for s in t.spans] == [
        "rag.request",
        "retrieve",
        "rerank",
        "evidence_gate",
        "generate",
        "validate_citations",
    ]
    gen = next(s for s in t.spans if s.name == "generate")
    assert "rules.pdf#c000" in gen.attributes["messages"][1]["content"]  # final prompt captured
    assert gen.attributes["response"].startswith("Nano drones")
    rr = next(s for s in t.spans if s.name == "rerank")
    assert rr.attributes["order"][0]["pre_rank"] == 1  # rerank order + pre-rank kept
    assert t.prompt_version == "answer-v1.2" and t.request_id == res.request_id
    assert t.cost_usd == (2000 * 0.15 + 100 * 0.60) / 1e6
    assert t.citation_coverage == 1.0 and t.n_citations == 1


def test_trace_written_to_jsonl_sqlite_and_otel(make_traced_pipeline):
    pipe, _tracer, store, tmp = make_traced_pipeline("Default is 2 s [px4.md#c000].")
    pipe.ask("PX4 auto disarm?")
    files = list((tmp / "traces").glob("*.jsonl"))
    assert len(files) == 1 and json.loads(files[0].read_text().splitlines()[0])["spans"]
    rows = store.requests()
    assert len(rows) == 1 and rows[0]["prompt_version"] == "answer-v1.2" and rows[0]["latency_total_ms"] > 0
    spans = [json.loads(line) for line in (tmp / "spans.jsonl").read_text().splitlines()]
    names = {s["name"] for s in spans}
    assert {"rag.request", "rag.generate", "rag.rerank"} <= names
    gen = next(s for s in spans if s["name"] == "rag.generate")
    assert gen["attributes"]["langfuse.observation.type"] == "generation"
    assert gen["attributes"]["gen_ai.usage.input_tokens"] == 2000
    root = next(s for s in spans if s["name"] == "rag.request")
    assert gen["parent_id"] == root["context"]["span_id"]
    assert root["attributes"]["langfuse.trace.metadata.prompt_version"] == "answer-v1.2"


def test_refusal_is_traced_with_reason(make_traced_pipeline):
    pipe, _tracer, store, _ = make_traced_pipeline("It is Rs 9999 [rules.pdf#c999].")
    pipe.ask("fee?")
    row = store.requests()[0]
    assert row["refused"] == 1 and row["refusal_reason"] == "refused_no_citations" and row["n_invalid_citations"] == 1


def test_simulated_clock_sets_timestamp(make_traced_pipeline):
    pipe, _tracer, store, _ = make_traced_pipeline("2 s [px4.md#c000].", clock=lambda: 1793700000.0)
    pipe.ask("disarm?")
    assert store.requests()[0]["ts"].startswith("2026-11-03")


def test_exporter_failure_never_breaks_request(make_traced_pipeline):
    pipe, tracer, store, _ = make_traced_pipeline("2 s [px4.md#c000].")

    class Down:
        def export(self, t):
            raise ConnectionError("langfuse down")

    tracer.exporters.insert(0, Down())
    res = pipe.ask("disarm?")
    assert not res.refused and len(store.requests()) == 1
