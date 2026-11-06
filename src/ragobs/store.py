"""SQLite metrics store: one row per request + one row per offline eval run."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    trace_id TEXT,
    ts TEXT NOT NULL,
    question TEXT,
    model TEXT,
    prompt_version TEXT,
    refused INTEGER,
    refusal_reason TEXT,
    error TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    latency_retrieve_ms REAL,
    latency_rerank_ms REAL,
    latency_generate_ms REAL,
    latency_total_ms REAL,
    citation_coverage REAL,
    n_citations INTEGER,
    n_invalid_citations INTEGER,
    best_rerank_score REAL,
    tags TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    prompt_version TEXT,
    model TEXT,
    judge_model TEXT,
    subset TEXT,
    passed INTEGER,
    summary TEXT
);
"""

# Requests the validator rejected (model produced something it could not support) vs. clean declines.
UNSUPPORTED = ("refused_no_citations", "refused_uncited")


class MetricsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # exporter protocol
    def export(self, trace) -> None:
        lat = trace.latency_ms
        row = (
            trace.request_id,
            trace.trace_id,
            trace.ts,
            trace.question,
            trace.model,
            trace.prompt_version,
            int(trace.refused),
            trace.refusal_reason,
            trace.error,
            trace.prompt_tokens,
            trace.completion_tokens,
            trace.cost_usd,
            lat.get("retrieve"),
            lat.get("rerank"),
            lat.get("generate"),
            lat.get("total"),
            trace.citation_coverage,
            trace.n_citations,
            trace.n_invalid_citations,
            trace.best_rerank_score,
            json.dumps(trace.tags),
        )
        with self._lock:
            self.db.execute(f"INSERT OR REPLACE INTO requests VALUES ({','.join('?' * len(row))})", row)
            self.db.commit()

    def record_eval(
        self,
        run_id: str,
        ts: str,
        prompt_version: str,
        model: str,
        judge_model: str,
        subset: str,
        passed: bool,
        summary: dict,
    ) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO eval_runs VALUES (?,?,?,?,?,?,?,?)",
                (run_id, ts, prompt_version, model, judge_model, subset, int(passed), json.dumps(summary)),
            )
            self.db.commit()

    def requests(self, where: str = "", params: tuple = ()) -> list[dict]:
        q = "SELECT * FROM requests " + (f"WHERE {where} " if where else "") + "ORDER BY ts"
        with self._lock:
            return [dict(r) for r in self.db.execute(q, params).fetchall()]

    def eval_runs(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM eval_runs ORDER BY ts").fetchall()]
