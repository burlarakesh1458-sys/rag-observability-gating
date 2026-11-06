"""Token-aware chunker with overlap that keeps exact page / section provenance.

Sections of one document are flattened into a paragraph stream. A window of
``chunk_tokens`` slides over that stream with ``overlap_tokens`` of overlap; the window
end snaps back to a paragraph boundary when one is close, so sentences are rarely cut.
Each chunk records every page / section its tokens came from.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from .loaders import Section


@lru_cache(maxsize=4)
def get_encoder(name: str = "cl100k_base"):
    import tiktoken

    return tiktoken.get_encoding(name)


@dataclass
class Chunk:
    chunk_id: str
    source: str
    doc_type: str
    locator: str  # human readable: "page 27" / "pages 27-28" / "section: A > B"
    text: str
    n_tokens: int
    pages: list[int] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    content_hash: str = ""

    def to_metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d["pages"] = ",".join(map(str, self.pages))
        d["sections"] = " | ".join(self.sections)
        return d


def make_locator(pages: list[int], sections: list[str]) -> str:
    if pages:
        lo, hi = min(pages), max(pages)
        return f"page {lo}" if lo == hi else f"pages {lo}-{hi}"
    if sections:
        uniq = list(dict.fromkeys(sections))
        return "section: " + uniq[0] + (f" (+{len(uniq) - 1} more)" if len(uniq) > 1 else "")
    return "document"


def chunk_sections(
    sections: list[Section],
    chunk_tokens: int = 550,
    overlap_tokens: int = 100,
    min_chunk_tokens: int = 40,
    tokenizer: str = "cl100k_base",
    snap_window: int = 150,
) -> list[Chunk]:
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    if not sections:
        return []
    enc = get_encoder(tokenizer)
    source, doc_type = sections[0].source, sections[0].doc_type

    tokens: list[int] = []
    owner: list[int] = []  # section index owning each token
    boundaries: set[int] = set()  # token offsets where a paragraph starts
    for si, sec in enumerate(sections):
        for para in [p for p in sec.text.split("\n\n") if p.strip()]:
            boundaries.add(len(tokens))
            ids = enc.encode(para.strip() + "\n\n")
            tokens.extend(ids)
            owner.extend([si] * len(ids))
    boundaries.add(len(tokens))

    chunks: list[Chunk] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_tokens, n)
        if end < n:  # snap to a paragraph boundary if one is near the end of the window
            candidates = [b for b in boundaries if end - snap_window <= b < end and b > start + overlap_tokens]
            if candidates:
                end = max(candidates)
        window = tokens[start:end]
        if len(window) >= min_chunk_tokens or not chunks:
            secs_idx = sorted(set(owner[start:end]))
            pages = sorted({sections[i].page for i in secs_idx if sections[i].page is not None})
            sec_names = [sections[i].section for i in secs_idx if sections[i].section]
            text = enc.decode(window).strip()
            cid = f"{source}#c{len(chunks):03d}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    source=source,
                    doc_type=doc_type,
                    locator=make_locator(pages, sec_names),
                    text=text,
                    n_tokens=len(window),
                    pages=pages,
                    sections=list(dict.fromkeys(sec_names)),
                    content_hash=hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()[:12],
                )
            )
        if end >= n:
            break
        start = max(end - overlap_tokens, start + 1)
    return chunks
