"""Sniff DOI and arXiv identifiers from PDF files.

The strategy is to gather text from several locations (embedded metadata,
the first/second/last pages, and as a last resort the filename), regex-scan
each for DOI / arXiv-style patterns, and score candidates by *where they
were found*. Front-matter sources outweigh body matches because the body
typically contains the bibliography of the paper, which is full of DOIs of
*other* works. The DOI/arXiv ID printed on page 1 is almost always the
paper's own.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # type: ignore[import-untyped]

# Public regexes (kept module-level so tests can target them directly).
# Excluded characters: whitespace, quotes, brackets/braces, pipe, caret, comma,
# semicolon. Comma/semicolon would otherwise cause the regex to greedily eat
# across DOI list separators (e.g. "10.X/a, 10.Y/b" → one giant match).
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>{}|^,;]+", re.IGNORECASE)

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

# Trailing punctuation we strip from a captured DOI (a sentence ending with
# "...10.1234/abc." should yield "10.1234/abc", not "10.1234/abc.").
_DOI_TRAIL_PUNCT = ".,;:)]}'\""

# Per-source weights used by sniff_pdf to score candidates.
# A DOI that appears in PDF metadata is almost certainly the paper's own;
# a DOI on page 1 is usually the paper's own header; deeper pages are
# heavily contaminated by bibliography entries.
_W_METADATA = 10
_W_FIRST_PAGE = 5
_W_OTHER_PAGE = 1


@dataclass
class SniffResult:
    """Outcome of scanning a PDF for identifiers.

    Attributes:
        doi: Best-scoring DOI candidate, or None if nothing matched.
        arxiv_id: Best-scoring arXiv ID candidate, or None.
        pdf_metadata: Raw metadata dict as returned by PyMuPDF.
    """

    doi: str | None
    arxiv_id: str | None
    pdf_metadata: dict[str, str] = field(default_factory=dict)


# A weighted text source: (text, weight). Higher weight ⇒ candidates found
# in this source contribute more to the final score.
WeightedSource = tuple[str, int]


def sniff_pdf(path: Path, *, min_doi_score: int | None = None) -> SniffResult:
    """Inspect a PDF and return the best-scoring DOI / arXiv ID.

    Reads PDF metadata, the first two pages, and the last page; aggregates
    candidate scores across these sources; picks the highest-scoring DOI
    and the highest-scoring arXiv ID. Falls back to the filename for
    arXiv IDs only.

    Args:
        path: PDF file to scan.
        min_doi_score: If set, the best DOI is dropped (returned as None)
            unless its score reaches this threshold. Default is
            ``_W_FIRST_PAGE``, meaning the DOI must appear in PDF metadata
            or on page 1 to be trusted as the paper's own — body-only
            matches are usually citations of *other* papers.
    """
    if min_doi_score is None:
        min_doi_score = _W_FIRST_PAGE

    with fitz.open(path) as doc:
        meta = {k: (v or "") for k, v in (doc.metadata or {}).items()}
        page_texts = _gather_pages(doc)

    sources = _build_sources(meta, page_texts)

    doi_scores = score_dois(sources)
    best_doi = _best(doi_scores)
    if best_doi is not None and doi_scores[best_doi] < min_doi_score:
        # Looks like a citation, not the paper's own DOI.
        best_doi = None

    return SniffResult(
        doi=best_doi,
        arxiv_id=_best(score_arxiv(sources, filename=path.stem)),
        pdf_metadata=meta,
    )


def _build_sources(
    meta: dict[str, str], page_texts: list[tuple[int, str]]
) -> list[WeightedSource]:
    """Assemble the list of (text, weight) tuples that the scorers consume."""
    sources: list[WeightedSource] = []
    for v in meta.values():
        if v:
            sources.append((v, _W_METADATA))
    for page_index, text in page_texts:
        weight = _W_FIRST_PAGE if page_index == 0 else _W_OTHER_PAGE
        sources.append((text, weight))
    return sources


def _gather_pages(doc: fitz.Document) -> list[tuple[int, str]]:
    """Return text from pages 0, 1, and the last page (0-indexed)."""
    n = len(doc)
    indices = [i for i in (0, 1, n - 1) if 0 <= i < n]
    seen: set[int] = set()
    out: list[tuple[int, str]] = []
    for i in indices:
        if i in seen:
            continue
        seen.add(i)
        out.append((i, doc[i].get_text()))
    return out


def score_dois(sources: list[WeightedSource]) -> Counter[str]:
    """Sum DOI-match weights across all sources.

    Returns a Counter where each key is a normalized DOI and each value is
    the total weight contributed by every source it was found in. The same
    DOI appearing in multiple sources accumulates their weights.
    """
    scores: Counter[str] = Counter()
    for text, weight in sources:
        for m in DOI_RE.finditer(text):
            scores[_clean_doi(m.group())] += weight
    return scores


def score_arxiv(
    sources: list[WeightedSource], *, filename: str | None = None
) -> Counter[str]:
    """Sum arXiv-ID match weights across all sources.

    If no contextual match is found, falls back to a bare-pattern match on
    the filename with weight 1 (a filename like ``2604.00777v1.pdf`` is a
    strong implicit signal even without "arXiv:" prefix).
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
