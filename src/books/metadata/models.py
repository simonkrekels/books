"""Pydantic models for canonical bibliographic records.

A ``PaperMatch`` is the shape every metadata source must produce. The
importer and DB layer only know about ``PaperMatch`` — Crossref and arXiv
adapters live in their own modules and translate API responses into this
common form.
"""

from typing import Any

from pydantic import BaseModel, Field


class Author(BaseModel):
    """Single author record. ``given`` / ``orcid`` may be missing on older works."""

    family: str
    given: str | None = None
    orcid: str | None = None


class PaperMatch(BaseModel):
    """Canonical bibliographic record returned by metadata sources.

    The ``source`` field tags the producer (``"crossref"``, ``"arxiv"``,
    ``"manual"``); ``raw`` carries the original API response for forensics
    and is persisted as ``papers.metadata_json``.
    """

    source: str
    doi: str | None = None
    arxiv_id: str | None = None
    isbn: str | None = None
    title: str
    authors: list[Author] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    publisher: str | None = None
    abstract: str | None = None
    type: str | None = None  # journal-article, preprint, book-chapter, ...
    raw: dict[str, Any] = Field(default_factory=dict)
