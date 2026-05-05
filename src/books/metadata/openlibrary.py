"""Open Library REST client.

Fallback for ISBN-only PDFs (typically books). Crossref carries some book
metadata but coverage is patchy; Open Library is broad, free, and needs no
auth. Endpoint:

  https://openlibrary.org/api/books?bibkeys=ISBN:9780805382914&format=json&jscmd=data
"""

import re
from typing import Any

import httpx

from books.metadata.models import Author, PaperMatch

OPENLIB_URL = "https://openlibrary.org/api/books"


class OpenLibraryError(Exception):
    """Raised for unexpected Open Library API failures."""


def lookup(isbn: str, *, client: httpx.Client | None = None) -> PaperMatch | None:
    """Fetch metadata for ``isbn`` from Open Library.

    Returns ``None`` if Open Library has no record for the given ISBN
    (the API responds with an empty JSON object rather than a 404).
    """
    own_client = client is None
    client = client or httpx.Client(timeout=20.0)
    try:
        resp = client.get(
            OPENLIB_URL,
            params={
                "bibkeys": f"ISBN:{isbn}",
                "format": "json",
                "jscmd": "data",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        # The keyed payload uses the original "ISBN:..." string as the key.
        key = f"ISBN:{isbn}"
        if key not in data:
            return None
        return _parse(data[key], isbn)
    finally:
        if own_client:
            client.close()


def _parse(payload: dict[str, Any], isbn: str) -> PaperMatch:
    """Translate an Open Library ``data`` payload into :class:`PaperMatch`."""
    title = payload.get("title", "[untitled]")
    if subtitle := payload.get("subtitle"):
        title = f"{title}: {subtitle}"

    authors: list[Author] = []
    for a in payload.get("authors") or []:
        name = (a.get("name") or "").strip()
        if not name:
            continue
        # Open Library returns full names. Reuse the simple heuristic from
        # the arXiv client (last whitespace-separated token = family name).
        from books.metadata.arxiv import _split_name

        given, family = _split_name(name)
        authors.append(Author(family=family, given=given))

    # publish_date is a free-form string ("September 21, 2017", "2017", etc).
    # Pull out the first 4-digit run as the year.
    year = None
    publish_date = payload.get("publish_date", "") or ""
    if m := re.search(r"\b(\d{4})\b", publish_date):
        year = int(m.group(1))

    publishers = payload.get("publishers") or []
    publisher = (publishers[0] or {}).get("name") if publishers else None

    return PaperMatch(
        source="openlibrary",
        isbn=isbn,
        title=title,
        authors=authors,
        year=year,
        publisher=publisher,
        type="book",
        raw=payload,
    )
