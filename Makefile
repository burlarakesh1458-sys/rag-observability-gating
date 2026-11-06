PY ?= python
export PYTHONPATH := src:vendor/dronedocs-rag/src
export DRONEDOCS_ROOT := $(CURDIR)/vendor/dronedocs-rag

.PHONY: install install-dev ingest serve replay replay-live dashboard charts gate gate-full baseline test lint langfuse-up langfuse-down clean

install:
	$(PY) -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps

install-dev: install
	$(PY) -m pip install -r requirements-dev.txt

ingest:          ## build the vendored RAG's index (vendor/dronedocs-rag/data/index)
	$(PY) -m dronedocs.cli -v ingest

serve:           ## traced API on :8000 (+ /metrics/daily)
	$(PY) -m uvicorn ragobs.api:app --port 8000

replay:          ## real traffic, backfilled over 6 days incl. the 2026-11-03 incident (48 LLM calls)
	$(PY) -m ragobs.replay --backfill --days 6 --per-day 8

replay-live:     ## N real requests with real timestamps
	$(PY) -m ragobs.replay -n 10

dashboard:       ## interactive dashboard on :8501
	streamlit run dashboard/app.py

charts:          ## static PNGs into docs/
	$(PY) scripts/render_dashboard.py

gate:            ## CI regression gate on the stratified golden subset (exit 1 on regression)
	$(PY) -m ragobs.gate --subset ci

gate-full:
	$(PY) -m ragobs.gate --subset full

baseline:        ## re-baseline after an intentional, reviewed change
	$(PY) -m ragobs.gate --subset ci --update-baseline

test:
	$(PY) -m pytest

lint:
	ruff check src tests scripts dashboard

langfuse-up:     ## self-hosted Langfuse on :3000 (needs Docker)
	docker compose -f docker-compose.langfuse.yml up -d

langfuse-down:
	docker compose -f docker-compose.langfuse.yml down

clean:
	rm -rf data/metrics.sqlite traces/*.jsonl reports/replay_log.jsonl
