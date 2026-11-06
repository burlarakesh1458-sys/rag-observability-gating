"""Serve the DroneDocs API with tracing on every request, plus metrics endpoints.

uvicorn ragobs.api:app --port 8000
"""

from __future__ import annotations

import json

from dronedocs import api as rag_api

from .analytics import aggregate, by_prompt_version, detect_anomalies
from .charts import IST
from .config import obs_settings
from .instrument import build_instrumented_pipeline, build_store

app = rag_api.app
_store = build_store()


@app.on_event("startup")
def _install_instrumented_pipeline() -> None:
    pipe, _ = build_instrumented_pipeline(store=_store, tags={"source": "api"})
    rag_api.set_pipeline(pipe)


@app.get("/metrics/daily")
def metrics_daily(source: str | None = None) -> dict:
    rows = _store.requests()
    if source:
        rows = [r for r in rows if json.loads(r["tags"] or "{}").get("source") == source]
    buckets = aggregate(rows, "day", tz=IST)
    a = obs_settings()["anomaly"]
    return {
        "buckets": buckets,
        "anomalies": detect_anomalies(buckets, a["rules"], a["window_days"], a["min_requests"]),
        "by_prompt_version": by_prompt_version(rows) if rows else [],
    }
