"""Citation parsing and enforcement.

The model must cite chunk ids like ``[drone_rules_2021.pdf#c012]``. After generation we:
1. extract every cited id,
2. verify each id exists AND was actually in the prompt context (no invented / stale ids),
3. check that enough factual sentences carry a citation,
and convert the answer into a refusal when the evidence does not hold up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITE_RE = re.compile(r"\[([^\[\]]+?#c\d{3}(?:\s*[,;]\s*[^\[\]]+?#c\d{3})*)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")

REFUSAL_TEXT = (
    "I can't answer that from the indexed documents: the retrieved passages do not "
    "contain enough evidence to support an answer."
)


@dataclass
class CitationCheck:
    status: str  # ok | refused_by_model | refused_no_citations | refused_uncited
    cited_ids: list[str] = field(default_factory=list)
    valid_ids: list[str] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)
    n_sentences: int = 0
    n_cited_sentences: int = 0

    @property
    def refused(self) -> bool:
        return self.status != "ok"

    @property
    def coverage(self) -> float:
        return self.n_cited_sentences / self.n_sentences if self.n_sentences else 0.0


_BRACKETS = str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]", "〔": "[", "〕": "]"})


def normalize_citations(text: str) -> str:
    """gpt-oss models often emit their native 【id】 citation brackets; map them to [id]."""
    return text.translate(_BRACKETS)


def extract_citations(text: str) -> list[str]:
    ids: list[str] = []
    for group in CITE_RE.findall(text):
        for cid in re.split(r"\s*[,;]\s*", group):
            cid = cid.strip()
            if cid and cid not in ids:
                ids.append(cid)
    return ids


def split_sentences(text: str) -> list[str]:
    parts = []
    for block in text.split("\n"):
        block = block.strip(" -*\t")
        if block:
            parts.extend(s.strip() for s in _SENT_SPLIT.split(block) if s.strip())
    return parts


def is_refusal(text: str, sentinel: str = "INSUFFICIENT_EVIDENCE") -> bool:
    t = text.strip()
    return not t or sentinel in t or t.lower().startswith("i can't answer that from the indexed documents")


def check_citations(
    answer: str,
    context_ids: set[str],
    known_ids: set[str] | None = None,
    sentinel: str = "INSUFFICIENT_EVIDENCE",
    require_citations: bool = True,
    max_uncited_sentence_ratio: float = 0.5,
) -> CitationCheck:
    answer = normalize_citations(answer)
    if is_refusal(answer, sentinel):
        return CitationCheck(status="refused_by_model")
    cited = extract_citations(answer)
    known = known_ids if known_ids is not None else context_ids
    valid = [c for c in cited if c in context_ids and c in known]
    invalid = [c for c in cited if c not in valid]
    sents = [s for s in split_sentences(answer) if len(s) > 15]
    n_cited = sum(1 for s in sents if CITE_RE.search(s))
    chk = CitationCheck("ok", cited, valid, invalid, len(sents), n_cited)
    if require_citations and not valid:
        chk.status = "refused_no_citations"
    elif sents and (1 - n_cited / len(sents)) > max_uncited_sentence_ratio:
        chk.status = "refused_uncited"
    return chk


def strip_invalid_citations(answer: str, invalid_ids: list[str]) -> str:
    for cid in invalid_ids:
        answer = answer.replace(f"[{cid}]", "")
    return re.sub(r"\s{2,}", " ", answer).strip()
