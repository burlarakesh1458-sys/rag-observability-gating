"""Evaluation metrics.

Deterministic, reference-based metrics (no LLM needed):
  * context_recall     - gold evidence quote is present in the top-k prompt context
  * context_precision  - rank-weighted precision of the prompt context (Ragas-style average
                         precision), where a passage is relevant if it holds the gold evidence
                         or comes from the gold page / section
  * citation_accuracy  - an answered question cites at least one passage from the gold
                         source page / section
  * citation_validity  - share of chunk ids in the *raw* model output that were really in context
  * refusal_correctness- answerable -> answered, unanswerable -> refused

LLM-judge metrics (``judge_answer``): faithfulness (share of atomic claims supported by the
cited passages) and answer_relevancy (does it answer the question with the reference facts).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..chunking import Chunk
from ..llm import ChatClient, parse_json_loose


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _refs(item: dict) -> list[dict]:
    return [
        {"source": item["source"], "locator": item["locator"], "evidence": item["evidence"]},
        *(item.get("alt_evidence") or []),
    ]


def _matches(chunk: Chunk, ref: dict) -> bool:
    if chunk.source != ref["source"]:
        return False
    if norm(ref["evidence"]) in norm(chunk.text):
        return True
    loc = ref["locator"]
    if loc.startswith("page "):
        return int(loc.split()[1]) in chunk.pages
    sec = loc.removeprefix("section: ")
    return any(s == sec or s.startswith(sec + " >") for s in chunk.sections)


def chunk_is_relevant(chunk: Chunk, item: dict) -> bool:
    """Relevant = from a gold page/section, or containing a gold evidence quote (primary or alternate)."""
    if not item.get("answerable"):
        return False
    return any(_matches(chunk, ref) for ref in _refs(item))


def evidence_in(chunks: list[Chunk], item: dict) -> bool:
    return any(norm(ref["evidence"]) in norm(c.text) for c in chunks for ref in _refs(item))


def average_precision(rels: list[bool]) -> float:
    """Ragas context_precision@k: mean of precision@i over relevant positions i."""
    hits, total = 0, 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def reciprocal_rank(rels: list[bool]) -> float:
    for i, r in enumerate(rels, start=1):
        if r:
            return 1.0 / i
    return 0.0


JUDGE_PROMPT = """You are a strict grader for a retrieval-augmented QA system about drone regulations and autopilots.

QUESTION: {question}
REFERENCE ANSWER (verified by a human): {reference}
SYSTEM ANSWER: {answer}

CITED PASSAGES (the evidence the system pointed to):
{context}

Tasks:
1. Split the SYSTEM ANSWER into atomic factual claims (ignore the [chunk_id] markers). For each claim decide
   if it is directly supported by the CITED PASSAGES (paraphrase is fine, outside knowledge is not).
2. answer_relevancy: 1.0 if the SYSTEM ANSWER answers the QUESTION with the key facts of the REFERENCE,
   0.5 if partially / missing a key fact, 0.0 if it does not answer or contradicts the reference.

Return ONLY a JSON object:
{{"claims": [{{"claim": "...", "supported": true}}], "answer_relevancy": 0.0, "reason": "<one short sentence>"}}"""


@dataclass
class JudgeResult:
    faithfulness: float | None
    answer_relevancy: float | None
    n_claims: int
    n_supported: int
    reason: str
    tokens: int


def judge_answer(
    judge: ChatClient, item: dict, answer: str, cited: list[Chunk], prompt_template: str | None = None
) -> JudgeResult:
    ctx = "\n\n".join(f"[{c.chunk_id}] ({c.source}, {c.locator})\n{c.text}" for c in cited) or "(none)"
    prompt = (prompt_template or JUDGE_PROMPT).format(
        question=item["question"], reference=item["reference"], answer=answer, context=ctx
    )
    last_err = ""
    tokens = 0
    for _ in range(2):  # one re-ask on malformed JSON
        resp = judge.chat([{"role": "user", "content": prompt}], json_mode=True)
        tokens += resp.total_tokens
        try:
            data = parse_json_loose(resp.text)
            claims = data.get("claims") or []
            sup = sum(1 for c in claims if c.get("supported") in (True, "true", "yes", 1))
            faith = sup / len(claims) if claims else None
            rel = float(data.get("answer_relevancy", 0.0))
            return JudgeResult(
                faith, max(0.0, min(1.0, rel)), len(claims), sup, str(data.get("reason", ""))[:300], tokens
            )
        except (ValueError, TypeError, AttributeError) as e:
            last_err = str(e)
    return JudgeResult(None, None, 0, 0, f"judge_parse_error: {last_err[:200]}", tokens)
