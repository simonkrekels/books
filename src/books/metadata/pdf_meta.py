"""Sniff DOI / arXiv ID / ISBN from PDF files.

The strategy is to gather text from several locations (embedded metadata,
the front-matter pages, and the last page), regex-scan each for the three
identifier shapes, and score candidates by *where they were found*.
Front-matter sources outweigh deeper / back-matter pages because the body
typically contains the bibliography (DOIs of *other* works) and content
chapters that may incidentally mention numbers shaped like an identifier.

Page coverage:

* Page 0 — paper-style canonical front matter (journal info, abstract).
* Pages 1..5 — book-style copyright pages (ISBN, edition DOI), and paper
  early body. Treated as moderately authoritative.
* Last page — back-matter (mostly references). Low confidence.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # type: ignore[import-untyped]

# --- DOI ---

# Excluded characters: whitespace, quotes, brackets/braces, pipe, caret, comma,
# semicolon. Comma/semicolon would otherwise cause the regex to greedily eat
# across DOI list separators (e.g. "10.X/a, 10.Y/b" → one giant match).
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>{}|^,;]+", re.IGNORECASE)

# --- arXiv ---

# Modern arXiv IDs always appear with an explicit context marker
# ("arXiv:" or "arxiv.org/abs/") so we don't false-match four-digit
# numbers that happen to be followed by a dot in body text.
ARXIV_NEW_RE = re.compile(
    r"(?:arxiv\s*[:.]?|arxiv\.org/abs/)\s*(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)

# Pre-2007 IDs use a category prefix, e.g. "hep-th/9711200".
ARXIV_OLD_RE = re.compile(
    r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\b",
    re.IGNORECASE,
)

# Bare modern-style ID. Used only against the filename, where a bare match
# is informative; in body text it produces too many false positives.
ARXIV_BARE_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")

# --- ISBN ---

# Labeled ISBN: "ISBN", optionally "-10" or "-13", optional separators, then
# 9–17 chars of digits/hyphens/spaces ending in a digit or X. High precision.
ISBN_LABELED_RE = re.compile(
    r"ISBN(?:[-\s]?1[03])?[:\s]*([\d\-\s]{9,21}[\dXx])",
    re.IGNORECASE,
)

# Bare ISBN-13: 13 digits starting with 978 or 979, with optional hyphens.
# Negative lookbehind/lookahead reject matches inside longer numeric runs
# (DOIs of the form "10.NNNN/9781..." in particular).
ISBN13_BARE_RE = re.compile(r"(?<![\d/])(97[89](?:[-\s]?\d){10})(?!\d)")

# Trailing punctuation we strip from a captured DOI (a sentence ending with
# "...10.1234/abc." should yield "10.1234/abc", not "10.1234/abc.").
_DOI_TRAIL_PUNCT = ".,;:)]}'\""

# --- Page weights ---

# A DOI / ISBN that appears in PDF metadata is almost certainly the work's
# own; one on page 0 (paper style) is usually the journal-printed canonical;
# pages 1–5 cover book-style copyright info and paper early body — moderate
# trust; deeper pages are heavily contaminated by bibliography entries.
_W_METADATA = 10
_W_PAGE_ZERO = 5
_W_FRONT_MATTER = 2  # pages 1..5
_W_OTHER_PAGE = 1

# How many leading pages are treated as "front matter" for weighting.
_FRONT_MATTER_PAGES = 6

# Default DOI-confidence threshold. With the weights above, this requires
# a DOI to either appear in PDF metadata, on page 0, on multiple front-matter
# pages, or in any single front-matter page — body-only matches (citations)
# never reach it.
_DEFAULT_DOI_THRESHOLD = 2


@dataclass
class SniffResult:
    """Outcome of scanning a PDF for identifiers."""

    doi: str | None
    arxiv_id: str | None
    isbn: str | None
    pdf_metadata: dict[str, str] = field(default_factory=dict)


# A weighted text source: (text, weight). Higher weight ⇒ candidates found
# in this source contribute more to the final score.
WeightedSource = tuple[str, int]


def sniff_pdf(path: Path, *, min_doi_score: int | None = None) -> SniffResult:
    """Inspect a PDF and return the best-scoring DOI / arXiv ID / ISBN.

    Reads PDF metadata, pages 0..5, and the last page; aggregates candidate
    scores across these sources; picks the highest-scoring DOI, arXiv ID,
    and ISBN. ISBN has no confidence threshold (the regex + checksum makes
    false positives rare); DOI requires reaching ``min_doi_score`` so a
    citation in body text doesn't masquerade as the paper's own DOI.
    Filename is consulted as a last resort for arXiv IDs.
    """
    if min_doi_score is None:
        min_doi_score = _DEFAULT_DOI_THRESHOLD

    with fitz.open(path) as doc:
        meta = {k: (v or "") for k, v in (doc.metadata or {}).items()}
        page_texts = _gather_pages(doc)

    sources = _build_sources(meta, page_texts)

    doi_scores = score_dois(sources)
    best_doi = _best(doi_scores)
    if best_doi is not None and doi_scores[best_doi] < min_doi_score:
        # Looks like a citation, not the work's own DOI.
        best_doi = None

    return SniffResult(
        doi=best_doi,
        arxiv_id=_best(score_arxiv(sources, filename=path.stem)),
        isbn=_best(score_isbn(sources)),
        pdf_metadata=meta,
    )


def _build_sources(
    meta: dict[str, str], page_texts: list[tuple[int, str]]
) -> list[WeightedSource]:
    """Assemble the (text, weight) tuples that the scorers consume."""
    sources: list[WeightedSource] = []
    for v in meta.values():
        if v:
            sources.append((v, _W_METADATA))
    for page_index, text in page_texts:
        if page_index == 0:
            weight = _W_PAGE_ZERO
        elif page_index < _FRONT_MATTER_PAGES:
            weight = _W_FRONT_MATTER
        else:
            weight = _W_OTHER_PAGE
        sources.append((text, weight))
    return sources


def _gather_pages(doc: fitz.Document) -> list[tuple[int, str]]:
    """Return text from pages 0..5 (front matter) plus the last page.

    Pages with empty extracted text are kept (with empty string) — the
    caller's regexes simply find nothing in them. We dedupe by page index
    so very short documents don't yield duplicate entries.
    """
    n = len(doc)
    indices = list(range(min(_FRONT_MATTER_PAGES, n)))
    if n - 1 not in indices and n - 1 >= 0:
        indices.append(n - 1)
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for i in indices:
        if i in seen or not (0 <= i < n):
            continue
        seen.add(i)
        out.append((i, doc[i].get_text()))
    return out


# --- DOI scoring ---

def score_dois(sources: list[WeightedSource]) -> Counter[str]:
    """Sum DOI-match weights across all sources.

    Returns a Counter where each key is a normalized DOI and each value is
    the total weight contributed by every source it was found in.
    """
    scores: Counter[str] = Counter()
    for text, weight in sources:
        for m in DOI_RE.finditer(text):
            scores[_clean_doi(m.group())] += weight
    return scores


# --- arXiv scoring ---

def score_arxiv(
    sources: list[WeightedSource], *, filename: str | None = None
) -> Counter[str]:
    """Sum arXiv-ID match weights across all sources.

    If no contextual match is found, falls back to a bare-pattern match on
    the filename with weight 1 (a filename like ``2604.00777v1.pdf`` is a
    strong implicit signal even without an "arXiv:" prefix).
    """
    scores: Counter[str] = Counter()
    for text, weight in sources:
        for m in ARXIV_NEW_RE.finditer(text):
            scores[m.group(1)] += weight
        for m in ARXIV_OLD_RE.finditer(text):
            scores[m.group(1).lower()] += weight
    if not scores and filename:
        m = ARXIV_BARE_RE.search(filename)
        if m:
            scores[m.group(1)] = 1
    return scores


# --- ISBN scoring ---

def score_isbn(sources: list[WeightedSource]) -> Counter[str]:
    """Sum ISBN-match weights across all sources.

    Two-pass strategy: prefer ISBNs found with an explicit "ISBN:" label
    (very high precision), and only fall back to bare 978/979-prefixed
    matches if no labeled candidate exists. Both pass through
    :func:`normalize_isbn` (checksum-validated, ISBN-10 → ISBN-13).
    """
    labeled: Counter[str] = Counter()
    for text, weight in sources:
        for m in ISBN_LABELED_RE.finditer(text):
            isbn = normalize_isbn(m.group(1))
            if isbn:
                labeled[isbn] += weight
    if labeled:
        return labeled

    bare: Counter[str] = Counter()
    for text, weight in sources:
        for m in ISBN13_BARE_RE.finditer(text):
            isbn = normalize_isbn(m.group(1))
            if isbn:
                bare[isbn] += weight
    return bare


def normalize_isbn(raw: str) -> str | None:
    """Strip separators, validate checksum, return canonical ISBN-13.

    Accepts ISBN-10 or ISBN-13 in any common format (with or without
    hyphens / spaces). Returns ``None`` if the cleaned string is the wrong
    length, contains non-digit characters in the wrong positions, or fails
    its checksum.
    """
    s = re.sub(r"[^\dXx]", "", raw).upper()
    if len(s) == 10 and _isbn10_valid(s):
        return _isbn10_to_13(s)
    if len(s) == 13 and _isbn13_valid(s):
        return s
    return None


def _isbn10_valid(s: str) -> bool:
    """Verify an ISBN-10 checksum (mod 11 over weighted digits, last may be X)."""
    total = 0
    for i, c in enumerate(s):
        if c == "X":
            if i != 9:  # X is only valid as the check digit
                return False
            v = 10
        elif c.isdigit():
            v = int(c)
        else:
            return False
        total += v * (10 - i)
    return total % 11 == 0


def _isbn13_valid(s: str) -> bool:
    """Verify an ISBN-13 checksum (mod 10 over alternating-weight digits)."""
    if not s.isdigit():
        return False
    total = 0
    for i, c in enumerate(s):
        v = int(c)
        total += v if i % 2 == 0 else 3 * v
    return total % 10 == 0


def _isbn10_to_13(s10: str) -> str:
    """Convert an ISBN-10 to canonical ISBN-13 form (978-prefix, recompute check)."""
    body = "978" + s10[:9]
    total = sum(int(c) if i % 2 == 0 else 3 * int(c) for i, c in enumerate(body))
    check = (10 - total % 10) % 10
    return body + str(check)


# --- helpers ---

def _best(scores: Counter[str]) -> str | None:
    if not scores:
        return None
    return scores.most_common(1)[0][0]


def _clean_doi(s: str) -> str:
    """Lowercase the DOI and strip trailing sentence punctuation."""
    return s.rstrip(_DOI_TRAIL_PUNCT).lower()


# Backward-compatible flat-list helpers used by the regex-only unit tests.
def doi_candidates(*texts: str) -> list[str]:
    """Return every DOI substring found across the given texts, unweighted."""
    out: list[str] = []
    for t in texts:
        for m in DOI_RE.finditer(t):
            out.append(_clean_doi(m.group()))
    return out


def arxiv_candidates(*texts: str, filename: str | None = None) -> list[str]:
    """Return every arXiv ID found across the texts; falls back to filename."""
    out: list[str] = []
    for t in texts:
        for m in ARXIV_NEW_RE.finditer(t):
            out.append(m.group(1))
        for m in ARXIV_OLD_RE.finditer(t):
            out.append(m.group(1).lower())
    if not out and filename:
        m = ARXIV_BARE_RE.search(filename)
        if m:
            out.append(m.group(1))
    return out


def isbn_candidates(*texts: str) -> list[str]:
    """Return every valid ISBN found across the texts (canonical ISBN-13 form)."""
    out: list[str] = []
    for t in texts:
        for m in ISBN_LABELED_RE.finditer(t):
            isbn = normalize_isbn(m.group(1))
            if isbn:
                out.append(isbn)
        for m in ISBN13_BARE_RE.finditer(t):
            isbn = normalize_isbn(m.group(1))
            if isbn:
                out.append(isbn)
    return out
