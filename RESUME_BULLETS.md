# Resume bullets: RAG Observability (all numbers measured in this repo)

- Instrumented a hybrid-retrieval RAG pipeline (BM25 + vector + cross-encoder + LLM) with per-stage
  tracing: retrieved/reranked chunks, exact prompt, response, tokens and cost. Exported to JSONL,
  SQLite and OpenTelemetry/OTLP (Langfuse-compatible), with no tracing code inside the RAG package.
- Built quality SLO dashboards (p50/p95 latency per stage, cost per request, citation coverage,
  failure and refusal rates) with trailing-median anomaly detection. Over a 48-request replay it flagged the
  one bad day (citation coverage 1.00 → 0.00, failure rate 0.00 → 0.75), while error rate stayed at 0.00.
- Wrote a root-cause incident report tying the drop to a "token diet" prompt change that removed the
  citation rules. The change cut tokens per request by 31% (1,668 → 1,158) but took grounded answers from
  74% to 0%.
- Added a CI regression gate (golden subset, hard floor + max-drop vs. baseline, versioned prompts): it
  passes the production prompt (faithfulness 1.00, citation accuracy 1.00) and blocks the regressed prompt
  (answer relevancy 0.75 → 0.00, refusal correctness 0.80 → 0.17).
