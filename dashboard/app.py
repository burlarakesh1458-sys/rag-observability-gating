"""Streamlit observability dashboard over data/metrics.sqlite and traces/*.jsonl.

streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragobs.analytics import aggregate, by_prompt_version, detect_anomalies  # noqa: E402
from ragobs.charts import IST, dashboard_figure  # noqa: E402
from ragobs.config import obs_settings, resolve  # noqa: E402
from ragobs.store import MetricsStore  # noqa: E402

st.set_page_config(page_title="DroneDocs RAG - Observability", layout="wide")
obs = obs_settings()
store = MetricsStore(resolve(obs["storage"]["sqlite_path"]))
rows = store.requests()
st.title("DroneDocs RAG - observability")
if not rows:
    st.info("No traffic recorded yet. Run `make replay` to generate traffic.")
    st.stop()

gran = st.sidebar.radio("Bucket", ["day", "hour"], index=0)
sources = sorted({json.loads(r["tags"] or "{}").get("source", "unknown") for r in rows})
pick = st.sidebar.multiselect(
    "Traffic source", sources, default=[s for s in sources if s != "regression-gate"] or sources
)
rows = [r for r in rows if json.loads(r["tags"] or "{}").get("source", "unknown") in pick]
buckets = aggregate(rows, gran, tz=IST)
a = obs["anomaly"]
flags = detect_anomalies(buckets, a["rules"], a["window_days"], a["min_requests"])

n = len(rows)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Requests", n)
c2.metric("p95 latency (s)", f"{sorted(r['latency_total_ms'] for r in rows)[int(0.95 * (n - 1))] / 1000:.2f}")
c3.metric("Cost / request (USD)", f"{sum(r['cost_usd'] for r in rows) / n:.5f}")
answered = [r for r in rows if not r["refused"] and not r["error"]]
c4.metric("Citation coverage", f"{(sum(r['citation_coverage'] for r in answered) / max(1, len(answered))):.0%}")
c5.metric("Anomalies flagged", len(flags))

st.pyplot(dashboard_figure(buckets, flags, "Daily health", by_prompt_version(rows)))
if flags:
    st.subheader("Anomalies")
    st.dataframe(flags, use_container_width=True)

st.subheader("Trace explorer")
rid = st.selectbox("Request", [r["request_id"] for r in reversed(rows)])
day = next(r["ts"][:10] for r in rows if r["request_id"] == rid)
trace_file = resolve(obs["storage"]["trace_dir"]) / f"{day}.jsonl"
if trace_file.exists():
    for line in trace_file.read_text().splitlines():
        t = json.loads(line)
        if t["request_id"] == rid:
            st.write(f"**Q:** {t['question']}")
            st.write(f"**A:** {t['answer']}")
            st.caption(
                f"prompt {t['prompt_version']} - {t['model']} - {t['prompt_tokens']}+{t['completion_tokens']} tokens "
                f"- ${t['cost_usd']:.5f}"
            )
            for s in t["spans"]:
                with st.expander(f"{s['name']}  ({1000 * (s['end_s'] - s['start_s']):.0f} ms)"):
                    st.json(s["attributes"])
            break
