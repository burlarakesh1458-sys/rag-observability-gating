"""Per-request cost from token usage and the pricing table in configs/observability.yaml."""

from __future__ import annotations


def request_cost_usd(model: str, prompt_tokens: int, completion_tokens: int, pricing: dict) -> float:
    price = pricing.get(model) or pricing.get("default") or {"input": 0.0, "output": 0.0}
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000
