"""Traffic generator / replayer.

Sends REAL questions through the instrumented pipeline (real retrieval, real LLM calls, real
latency and token usage) and records traces + metrics.

``--backfill`` mode lays the requests out over past days on a simulated clock so the dashboard has
history: each request's *timestamp* is simulated, every measured value is real. It also replays the
deployment timeline below, which includes the 2026-11-03 (Tuesday) prompt change that caused the
quality incident documented in docs/incident-2026-11-03.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime, timedelta, timezone

from .config import ROOT, VENDOR, resolve

IST = timezone(timedelta(hours=5, minutes=30))

# (effective-from, prompt file). The prompt in force at a request's simulated time serves it.
DEPLOYMENTS = [
    (datetime(2026, 10, 1, 0, 0, tzinfo=IST), "configs/prompts/answer_v1.yaml"),
    (datetime(2026, 11, 3, 10, 0, tzinfo=IST), "configs/prompts/answer_v1_3_concise.yaml"),  # Tue: "token diet" PR
    (datetime(2026, 11, 4, 11, 0, tzinfo=IST), "configs/prompts/answer_v1.yaml"),  # Wed: rollback
]

# Real-user style paraphrases on top of the golden questions (traffic is not just the test set).
EXTRA_QUESTIONS = [
    "how heavy can a nano drone be in india",
    "Do I need a licence to fly a micro drone for fun?",
    "what's the penalty if I break the drone rules?",
    "When do I have to report a drone crash?",
    "ArduPilot copter battery failsafe default voltage?",
    "what altitude does ardupilot RTL climb to by default",
    "How long after landing does PX4 disarm automatically?",
    "Which PX4 parameter controls what happens on geofence breach?",
    "Is a type certificate needed for model aircraft?",
    "Can a 17 year old get a remote pilot certificate?",
    "How do I arm an ArduPilot quad with the sticks?",
    "What does the PX4 data link loss failsafe do?",
    "What is the max speed of a DJI Mini 4 Pro?",
    "Where can I buy drone insurance in Mumbai?",
]


class SimClock:
    """Wall clock shifted to a simulated instant; real durations are preserved."""

    def __init__(self):
        self.offset = 0.0

    def set(self, instant: datetime) -> None:
        self.offset = instant.timestamp() - time.time()

    def __call__(self) -> float:
        return time.time() + self.offset


def prompt_at(ts: datetime) -> str:
    current = DEPLOYMENTS[0][1]
    for start, pf in DEPLOYMENTS:
        if ts >= start:
            current = pf
    return current


def build_schedule(start: datetime, days: int, per_day: int, seed: int = 7) -> list[datetime]:
    rng = random.Random(seed)
    out = []
    for d in range(days):
        day = start + timedelta(days=d)
        for _ in range(per_day):
            out.append(day.replace(hour=9) + timedelta(minutes=rng.randint(0, 12 * 60)))
    # make sure the incident window gets traffic on both sides of the deploy / rollback
    return sorted(out)


def question_pool(seed: int = 7) -> list[str]:
    golden = [
        json.loads(line)["question"]
        for line in (VENDOR / "eval" / "golden.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pool = golden + EXTRA_QUESTIONS
    random.Random(seed).shuffle(pool)
    return pool


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    from .instrument import build_instrumented_pipeline, build_store

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Generate real RAG traffic and record traces/metrics")
    ap.add_argument("--backfill", action="store_true", help="simulated timestamps over past days + deploy timeline")
    ap.add_argument("--start", default="2026-10-31", help="first backfill day (IST)")
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--per-day", type=int, default=8)
    ap.add_argument("-n", type=int, default=10, help="live mode: number of requests")
    ap.add_argument("--sleep", type=float, default=1.0, help="pause between requests (rate-limit friendly)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip requests already in reports/replay_log.jsonl (after a quota stop or a crash)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    store = build_store()
    clock = SimClock()
    pool = question_pool(args.seed)
    pipes: dict[str, object] = {}
    shared = None

    def pipe_for(pf: str):
        nonlocal shared
        if pf not in pipes:
            p, _ = build_instrumented_pipeline(
                prompt_file=pf,
                store=store,
                clock=clock,
                tags={"source": "replay-backfill" if args.backfill else "replay-live"},
                shared_components=shared,
            )
            shared = shared or (p.index, p.reranker)
            pipes[pf] = p
        return pipes[pf]

    if args.backfill:
        start = datetime.fromisoformat(args.start).replace(tzinfo=IST)
        schedule = build_schedule(start, args.days, args.per_day, args.seed)
    else:
        schedule = [None] * args.n

    log_path = resolve("reports") / "replay_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if args.resume and log_path.exists():
        done = {json.loads(line)["i"] for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    elif log_path.exists():
        log_path.unlink()
    for i, ts in enumerate(schedule):
        if i in done:
            continue
        q = pool[i % len(pool)]
        pf = prompt_at(ts) if ts else "configs/prompts/answer_v1.yaml"
        pipe = pipe_for(str(resolve(pf)))
        if ts:
            clock.set(ts)
        else:
            clock.offset = 0.0
        request_id = f"rp-{(ts or datetime.now(IST)).strftime('%m%d%H%M')}-{i:03d}"
        res = pipe.ask(q, request_id=request_id)
        if res.error and "quota exhausted" in res.error:
            # Provider daily quota, not a product failure: drop the row and stop; rerun with --resume later.
            with store._lock:
                store.db.execute("DELETE FROM requests WHERE request_id = ?", (request_id,))
                store.db.commit()
            print(f"[{i + 1}/{len(schedule)}] provider quota exhausted; stopping. Rerun with --resume.")
            return 2
        rec = {
            "i": i,
            "sim_ts": ts.isoformat() if ts else None,
            "prompt_version": res.prompt_version,
            "refused": res.refused,
            "reason": res.refusal_reason,
            "error": res.error,
            "total_ms": round(res.stage_timings["total"] * 1000),
            "q": q,
        }
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"[{i + 1}/{len(schedule)}] {rec['sim_ts'] or 'live'} {res.prompt_version:<22} "
            f"{'REFUSED ' + str(res.refusal_reason) if res.refused else 'answered'} {rec['total_ms']}ms  {q[:60]}"
        )
        time.sleep(args.sleep)
    for p in pipes.values():
        for ob in p.observers:
            for ex in getattr(ob, "exporters", []):
                if hasattr(ex, "flush"):
                    ex.flush()
    print(f"done: {len(schedule)} requests -> {store.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
