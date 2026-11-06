"""Document loaders for PDF, Markdown and HTML.

Each loader yields ``Section`` objects that carry a precise locator (PDF page number or
heading path) so every chunk can be cited back to an exact page / section.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf": "pdf", ".md": "markdown", ".markdown": "markdown", ".html": "html", ".htm": "html"}

_QUOTES = str.maketrans(
    {"‗": "'", "―": '"', "‖": '"', "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "\u00a0": " "}
)


_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_GAZETTE_HEADER = re.compile(r"THE GAZETTE OF INDIA\s*:\s*EXTRAORDINARY", re.I)
_ADMONITION = re.compile(r"^\s*:::\s*\w*.*$")
_ANCHOR = re.compile(r"\s*\{#[^}]*\}")


@dataclass
class Section:
    source: str  # file name, e.g. drone_rules_2021.pdf
    doc_type: str  # pdf | markdown | html
    locator: str  # "page 27" or "section: Failsafe Actions > Battery"
    text: str
    page: int | None = None
    section: str | None = None


def _latin_ratio(line: str) -> float:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return latin / len(letters)


def clean_text(text: str, drop_non_latin_lines: bool = True) -> str:
    text = text.translate(_QUOTES)
    lines = []
    for line in text.splitlines():
        if drop_non_latin_lines and _latin_ratio(line) < 0.6:
            continue  # bilingual Gazette: skip Devanagari lines
        if _TABLE_RULE.match(line):
            continue  # markdown table separator rows carry no content, only tokens
        if _GAZETTE_HEADER.search(line):
            continue  # running page header of the Gazette of India
        line = re.sub(r"[ \t]{2,}", " ", line).rstrip()
        lines.append(line)
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def load_pdf(path: Path, drop_non_latin_lines: bool = True) -> Iterator[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        txt = clean_text(page.extract_text() or "", drop_non_latin_lines)
        if len(re.sub(r"\W", "", txt)) < 40:  # header-only / Hindi-only pages
            continue
        yield Section(path.name, "pdf", f"page {i}", txt, page=i)


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_by_headings(
    path: Path, lines: list[tuple[int, str]], doc_type: str, drop_non_latin_lines: bool
) -> Iterator[Section]:
    """lines: (level, text) where level>0 marks a heading."""
    stack: list[str] = []
    buf: list[str] = []

    def flush():
        body = clean_text("\n".join(buf), drop_non_latin_lines)
        if len(body) >= 30:
            sec = " > ".join(stack) if stack else path.stem
            yield Section(path.name, doc_type, f"section: {sec}", body, section=sec)

    for level, text in lines:
        if level:
            yield from flush()
            buf = []
            stack = [*stack[: level - 1], text.strip()]
        else:
            buf.append(text)
    yield from flush()


def load_markdown(path: Path, drop_non_latin_lines: bool = True) -> Iterator[Section]:
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw)  # images
    raw = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", raw)  # links -> text
    raw = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", raw)  # reference-style links
    raw = re.sub(r"(?m)^\[[^\]]+\]:\s+\S+.*$", "", raw)  # link reference definitions
    raw = re.sub(r"<br\s*/?>", "\n", raw)
    raw = re.sub(r"<[^>]+>", "", raw)  # img, Badge, anchors...
    raw = "\n".join(line for line in raw.splitlines() if not _ADMONITION.match(line))
    lines: list[tuple[int, str]] = []
    in_code = False
    for line in raw.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        m = _MD_HEADING.match(line) if not in_code else None
        lines.append((len(m.group(1)), _ANCHOR.sub("", m.group(2))) if m else (0, line))
    yield from _split_by_headings(path, lines, "markdown", drop_non_latin_lines)


def load_html(path: Path, drop_non_latin_lines: bool = True) -> Iterator[Section]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    body = soup.select_one("div[itemprop=articleBody]") or soup.find("main") or soup.body or soup
    for tag in body.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()
    for a in body.select("a.headerlink"):
        a.decompose()
    lines: list[tuple[int, str]] = []
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr", "pre", "dt", "dd"]):
        if el.find_parent("table") and el.name != "tr":
            continue  # table content is emitted row-by-row below
        if el.name in ("li", "dd") and el.find(["p", "li", "table"]):
            continue  # avoid double-counting nested blocks
        if el.name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["td", "th"])]
            text = " | ".join(cells)
        else:
            text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            lines.append((int(el.name[1]), text.replace("¶", "").strip()))
        else:
            lines.append((0, ("- " if el.name == "li" else "") + text))
    yield from _split_by_headings(path, lines, "html", drop_non_latin_lines)


def load_file(path: Path, drop_non_latin_lines: bool = True) -> list[Section]:
    kind = SUPPORTED.get(path.suffix.lower())
    if kind == "pdf":
        return list(load_pdf(path, drop_non_latin_lines))
    if kind == "markdown":
        return list(load_markdown(path, drop_non_latin_lines))
    if kind == "html":
        return list(load_html(path, drop_non_latin_lines))
    raise ValueError(f"unsupported file type: {path}")


def discover(raw_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(raw_dir).rglob("*") if p.suffix.lower() in SUPPORTED)
