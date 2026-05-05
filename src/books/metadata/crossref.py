"""Crossref REST client.

Calls ``GET /works/{doi}`` (https://api.crossref.org) and converts the JSON
into a :class:`~books.metadata.models.PaperMatch`. We send a User-Agent with
a mailto qualifier to land in Crossref's "polite pool" (better rate limits).
"""

from typing import Any

import httpx

from books import config
from books.metadata.models import Author, PaperMatch

CROSSREF_URL = "https://api.crossref.org/works/{doi}"


class CrossrefError(Exception):
    """Raised for unexpected Crossref API failures (after raise_for_status)."""


def lookup(doi: str, *, client: httpx.Client | None = None) -> PaperMatch | None:
    """Fetch metadata for ``doi`` from Crossref.

    Returns ``None`` for 404 (DOI not registered with Crossref). Other HTTP
    errors propagate. ``client`` may be supplied for testing — when omitted,
    a short-lived client is created with the configured User-Agent.
    """
    own_client = client is None
    client = client or httpx.Client(
        timeout=20.0,
        headers={"User-Agent": config.crossref_user_agent()},
    )
    try:
        resp = client.get(CROSSREF_URL.format(doi=doi))
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _parse(resp.json())
    finally:
        if own_client:
            client.close()


def _parse(payload: dict[str, Any]) -> PaperMatch:
    """Translate a Crossref ``works`` response into :class:`PaperMatch`."""
    msg = payload.get("message") or {}

    # Title is always returned as a list (multiple language variants); take
    # the first as canonical.
    titles = msg.get("title") or []
    title = titles[0] if titles else "[untitled]"

    authors = []
    for a in msg.get("author") or []:
        family = a.get("family")
        if not family:
            continue  # corporate authors arrive without a family name
        authors.append(
            Author(
                family=family,
                given=a.get("given"),
                orcid=_clean_orcid(a.get("ORCID")),
            )
        )

    # Crossref's "issued" comes as {date-parts: [[YYYY, MM, DD]]}.
    issued = msg.get("issued") or {}
    date_parts = issued.get("date-parts") or [[]]
    year = date_parts[0][0] if date_parts and date_parts[0] else None

    containers = msg.get("container-title") or []
    journal = containers[0] if containers else None

    return PaperMatch(
        source="crossref",
        doi=msg.get("DOI"),
        title=title,
        authors=authors,
        year=int(year) if year else None,
        journal=journal,
        publisher=msg.get("publisher"),
        abstract=msg.get("abstract"),
        type=msg.get("type"),
        raw=msg,
    )


def _clean_orcid(orcid: str | None) -> str | None:
    """Normalise Crossref's ``https://orcid.org/0000-...`` URLs to the bare ID."""
    if not orcid:
        return None
    return orcid.rsplit("/", 1)[-1]
