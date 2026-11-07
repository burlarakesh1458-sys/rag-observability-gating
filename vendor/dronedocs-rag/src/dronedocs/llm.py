"""Provider-agnostic chat client for any OpenAI-compatible endpoint.

Groq, OpenAI, Together, vLLM and Ollama all speak ``/v1/chat/completions``; only the
base url, key and model name change. Adds exponential backoff on 429/5xx (Groq free tier
is 8k tokens/minute), honours ``retry-after``, paces itself from the ``x-ratelimit-*`` headers so it
waits *before* a request would be throttled, records token usage + latency and keeps an optional
on-disk response cache (sqlite) so a resumed or repeated run never pays for the same call twice.

``ReplayClient`` serves recorded answers by question with no network at all; the CI quality gate
uses it so the build is deterministic and needs no API key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    retries: int = 0
    cached: bool = False
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    pass


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, k: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
            self._db.commit()


class ChatClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_s: float = 60,
        max_retries: int = 6,
        reasoning_effort: str | None = None,
        cache_path: str | Path | None = None,
        transport: httpx.BaseTransport | None = None,
        max_retry_after_s: float = 120.0,
        min_interval_s: float = 0.0,
    ):
        self.model = model
        self.base_url = (base_url or "https://api.groq.com/openai/v1").rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self._cache = _Cache(Path(cache_path)) if cache_path else None
        # a retry-after longer than this means a daily (TPD/RPD) cap, not a per-minute one: stop the call
        self.max_retry_after_s = max_retry_after_s
        self.min_interval_s = min_interval_s  # client-side pacing between requests (0 = off)
        self._pace_lock = threading.Lock()
        self._next_ok = 0.0  # monotonic time before which no request is sent
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(timeout=timeout_s, headers=headers, transport=transport)

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_settings(
        cls, llm_cfg: dict, paths: dict | None = None, model: str | None = None, **overrides
    ) -> ChatClient:
        base_url = os.environ.get(llm_cfg.get("base_url_env", "LLM_BASE_URL")) or llm_cfg.get("default_base_url")
        api_key = (
            os.environ.get(llm_cfg.get("api_key_env", "LLM_API_KEY"))
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        cache_path = None
        if llm_cfg.get("cache") and paths:
            cache_path = Path(paths["cache_dir"]) / "llm_cache.sqlite"
        kw = dict(
            model=model or llm_cfg["model"],
            base_url=base_url,
            api_key=api_key,
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 800),
            timeout_s=llm_cfg.get("timeout_s", 60),
            max_retries=llm_cfg.get("max_retries", 6),
            reasoning_effort=llm_cfg.get("reasoning_effort"),
            cache_path=cache_path,
            max_retry_after_s=float(llm_cfg.get("max_retry_after_s", 120)),
            min_interval_s=float(llm_cfg.get("min_interval_s", 0)),
        )
        kw.update(overrides)
        return cls(**kw)

    # ------------------------------------------------------------------ call
    def _payload(
        self, messages: list[dict], temperature: float | None, max_tokens: int | None, json_mode: bool
    ) -> dict:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.reasoning_effort and "gpt-oss" in self.model:
            body["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        body = self._payload(messages, temperature, max_tokens, json_mode)
        key = hashlib.sha256(json.dumps([self.base_url, body], sort_keys=True).encode()).hexdigest()
        if self._cache and (hit := self._cache.get(key)):
            return LLMResponse(**hit, cached=True)

        t0 = time.perf_counter()
        attempt = 0
        last_err: str = ""
        while attempt <= self.max_retries:
            self._wait_turn()
            try:
                r = self._http.post(f"{self.base_url}/chat/completions", json=body)
                self._update_pacing(r, body)
            except httpx.HTTPError as e:  # network / timeout
                last_err = f"{type(e).__name__}: {e}"
                r = None
            if r is not None and r.status_code == 200:
                data = r.json()
                msg = data["choices"][0]["message"]
                usage = data.get("usage") or {}
                resp = LLMResponse(
                    text=(msg.get("content") or "").strip(),
                    model=data.get("model", self.model),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    latency_s=time.perf_counter() - t0,
                    retries=attempt,
                )
                if self._cache:
                    d = resp.__dict__.copy()
                    d.pop("raw", None)
                    d.pop("cached", None)
                    self._cache.put(key, d)
                return resp
            if r is not None:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 403, 404) and not (
                    r.status_code == 400 and json_mode and "json" in r.text.lower()
                ):
                    raise LLMError(last_err)
                if r.status_code == 400 and json_mode:  # model refused json mode -> drop it once
                    body.pop("response_format", None)
            if r is not None and r.status_code == 429 and _retry_after(r) > self.max_retry_after_s:
                # daily quota (TPD/RPD) exhausted: waiting minutes per call would stall the whole run
                raise LLMError(f"quota exhausted (retry-after {_retry_after(r):.0f}s): {last_err}")
            wait = _retry_wait(r, attempt)
            log.warning("LLM call failed (%s); retry %d in %.1fs", last_err[:400], attempt + 1, wait)
            time.sleep(wait)
            attempt += 1
        raise LLMError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    # ------------------------------------------------------------------ pacing
    def _wait_turn(self) -> None:
        with self._pace_lock:
            delay = self._next_ok - time.monotonic()
        if delay > 0:
            log.info("pacing: waiting %.1fs for the provider's rate window", delay)
            time.sleep(delay)

    def _update_pacing(self, r: httpx.Response, body: dict) -> None:
        """Push the next allowed send time forward when the token window is nearly empty."""
        now = time.monotonic()
        nxt = now + self.min_interval_s
        try:
            remaining = float(r.headers.get("x-ratelimit-remaining-tokens", "inf"))
        except ValueError:
            remaining = float("inf")
        need = _estimate_tokens(body)
        if remaining < need:
            nxt = max(nxt, now + _parse_duration(r.headers.get("x-ratelimit-reset-tokens", "")) + 0.25)
        with self._pace_lock:
            self._next_ok = max(self._next_ok, nxt)


class ReplayClient:
    """Serves recorded model answers keyed by question text; no network, fully deterministic.

    Recordings are JSONL rows ``{"question": ..., "raw_answer": ..., "usage": {...}}`` captured from a
    live eval run (``scripts/record_answers.py``). A question without a recording raises ``LLMError``
    with ``not recorded`` so the eval marks the item as not evaluated instead of scoring it.
    """

    def __init__(self, path: str | Path, model: str = "replay"):
        self.model = model
        self._rows: dict[str, dict] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self._rows[_norm_q(row["question"])] = row
        if self._rows:
            self.model = next(iter(self._rows.values())).get("model", model)

    def __len__(self) -> int:
        return len(self._rows)

    def lookup(self, question: str) -> dict | None:
        return self._rows.get(_norm_q(question))

    def chat(self, messages: list[dict], **_: object) -> LLMResponse:
        user = _norm_q(next((m["content"] for m in reversed(messages) if m["role"] == "user"), ""))
        matches = [q for q in self._rows if q and q in user]
        if not matches:
            raise LLMError("not recorded: no recorded answer for this question")
        row = self._rows[max(matches, key=len)]  # longest match: one question may contain another
        u = row.get("usage") or {}
        return LLMResponse(
            text=row["raw_answer"],
            model=self.model,
            prompt_tokens=int(u.get("prompt_tokens", 0)),
            completion_tokens=int(u.get("completion_tokens", 0)),
            cached=True,
        )


def _norm_q(q: str) -> str:
    return " ".join(q.split()).lower()


def _estimate_tokens(body: dict) -> int:
    chars = sum(len(str(m.get("content", ""))) for m in body.get("messages", []))
    return chars // 4 + int(body.get("max_tokens") or 0)


def _parse_duration(s: str) -> float:
    """Parse Groq/OpenAI reset durations such as ``577ms``, ``7.66s``, ``2m59.56s``, ``1h2m3s``."""
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", s or ""):
        total += float(num) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return total


def _retry_after(r: httpx.Response) -> float:
    try:
        return float(r.headers.get("retry-after", "0"))
    except ValueError:
        return 0.0


def _retry_wait(r: httpx.Response | None, attempt: int) -> float:
    if r is not None:
        ra = r.headers.get("retry-after")
        if ra:
            try:
                return min(float(ra) + random.uniform(0.1, 0.8), 65.0)
            except ValueError:
                pass
    return min(2**attempt + random.uniform(0, 1), 60.0)


def parse_json_loose(text: str) -> dict:
    """Parse a JSON object from a model reply that may contain fences or prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{") :] if "{" in t else t
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in: {text[:200]}")
    return json.loads(t[start : end + 1])
