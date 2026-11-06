from datetime import datetime

from ragobs.config import rag_settings
from ragobs.replay import IST, build_schedule, prompt_at


def test_deploy_timeline_serves_concise_prompt_only_during_incident():
    assert prompt_at(datetime(2026, 11, 3, 9, 59, tzinfo=IST)).endswith("answer_v1.yaml")
    assert prompt_at(datetime(2026, 11, 3, 10, 0, tzinfo=IST)).endswith("answer_v1_3_concise.yaml")
    assert prompt_at(datetime(2026, 11, 4, 10, 59, tzinfo=IST)).endswith("answer_v1_3_concise.yaml")
    assert prompt_at(datetime(2026, 11, 4, 11, 0, tzinfo=IST)).endswith("answer_v1.yaml")


def test_backfill_schedule_is_deterministic_and_inside_business_hours():
    start = datetime(2026, 10, 31, tzinfo=IST)
    a, b = build_schedule(start, 6, 8, seed=7), build_schedule(start, 6, 8, seed=7)
    assert a == b and len(a) == 48 and a == sorted(a)
    assert all(9 <= t.hour <= 21 for t in a)


def test_deployment_overrides_are_applied_to_vendored_settings():
    cfg = rag_settings()
    assert cfg["rerank"]["top_n"] == 3
    assert cfg["llm"]["model"] != cfg["judge"]["model"]
