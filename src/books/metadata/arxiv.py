"""arXiv REST client.

Calls ``GET /api/query?id_list=...`` (https://export.arxiv.org/api/query) and
parses the Atom XML response into a :class:`~books.metadata.models.PaperMatch`.
Used as a fallback when no DOI is available, or when Crossref doesn't have
the work yet (preprints).
"""

import xml.etree.ElementTree as ET

import httpx

from books.metadata.models import Author, PaperMatch

ARXIV_URL = "https://export.arxiv.org/api/query"

# Atom-extension namespaces; arXiv-specific fields like <arxiv:doi> use the
# second namespace.
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivError(Exception):
    """Raised for unexpected arXiv API failures."""


def lookup(arxiv_id: str, *, client: httpx.Client | None = None) -> PaperMatch | None:
    """Fetch metadata for ``arxiv_id`` from the arXiv API.

    ``client`` may be supplied for testing.
    """
    own_client = client is None
    client = client or httpx.Client(timeout=20.0)
    try:
        resp = client.get(ARXIV_URL, params={"id_list": arxiv_id})
        resp.raise_for_status()
        return _parse(resp.text, arxiv_id)
    finally:
        if own_client:
            client.close()


def _parse(xml_text: str, arxiv_id: str) -> PaperMatch | None:
    """Translate an arXiv Atom feed into :class:`PaperMatch`.

    arXiv returns a one-entry feed for valid IDs, an entry with an error
    title for invalid ones — distinguished by inspecting ``<atom:id>`` (real
    hits include ``arxiv.org/abs/``).
    """
    root = ET.fromstring(xml_text)
    entry = root.find("atom:entry", NS)
    if entry is None:
        return None

    id_el = entry.find("atom:id", NS)
    if id_el is not None and "arxiv.org/abs/" not in (id_el.text or ""):
        return None

    title = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip()
    summary = (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip()
    published = entry.findtext("atom:published", default="", namespaces=NS) or ""
    year = int(published[:4]) if published[:4].isdigit() else None
    doi = entry.findtext("arxiv:doi", default=None, namespaces=NS)

    authors: list[Author] = []
    for a in entry.findall("atom:author", NS):
        name = (a.findtext("atom:name", default="", namespaces=NS) or "").strip()
        if not name:
            continue
        given, family = _split_name(name)
        authors.append(Author(family=family, given=given))

    return PaperMatch(
        source="arxiv",
        arxiv_id=arxiv_id,
        doi=doi,
        title=title,
        authors=authors,
        year=year,
        abstract=summary,
        type="preprint",
        # We keep a small forensic blob — the full Atom XML is not stored.
        raw={"id": arxiv_id, "summary": summary, "published": published},
    )


def _split_name(full: str) -> tuple[str, str]:
    """Split a full name into (given, family) using a last-token heuristic.

    arXiv returns names as a single string (e.g. ``"Jane Q. Smith"``). We
    treat the last whitespace-separated token as the family name. This is
    wrong for compound family names (e.g. ``"de Broglie"``) but those are
    rare enough to handle on a case-by-case basis if/when they cause issues.
    """
    parts = full.split()
    if len(parts) == 1:
        return ("", parts[0])
    return (" ".join(parts[:-1]), parts[-1])
