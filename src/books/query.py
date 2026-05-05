"""Compile CLI flags + positional terms into a parameterized SQL query.

Used by ``book ls``. Returns a tuple of (sql, params) so callers run it
through ``conn.execute(sql, params)`` without any string-interpolation risk.
"""

from typing import Any


def build_papers_query(
    *,
    terms: list[str] | None = None,
    author: str | None = None,
    year: int | None = None,
    journal: str | None = None,
    tag: str | None = None,
    doi: str | None = None,
    isbn: str | None = None,
) -> tuple[str, list[Any]]:
    """Build a parameterized SELECT for the papers table.

    All filters AND together. Free-text ``terms`` perform LIKE-anywhere match
    against title/abstract. Returns (sql, params).
    """
    joins: list[str] = []
    where: list[str] = []
    params: list[Any] = []

    if author:
        # Author filter requires the join chain papers -> paper_authors -> authors.
        # SELECT DISTINCT (below) handles the row-multiplication side effect.
        joins.append("JOIN paper_authors pa ON pa.paper_id = p.id")
        joins.append("JOIN authors a ON a.id = pa.author_id")
        where.append("a.family_name LIKE ?")
        params.append(f"%{author}%")
    if year is not None:
        where.append("p.year = ?")
        params.append(year)
    if journal:
        where.append("p.journal LIKE ?")
        params.append(f"%{journal}%")
    if tag:
        joins.append("JOIN tags t ON t.paper_id = p.id")
        where.append("t.tag = ?")
        params.append(tag)
    if doi:
        where.append("p.doi = ?")
        params.append(doi)
    if isbn:
        where.append("p.isbn = ?")
        params.append(isbn)
    for term in terms or []:
        where.append("(p.title LIKE ? OR p.abstract LIKE ?)")
        like = f"%{term}%"
        params.extend([like, like])

    parts = ["SELECT DISTINCT p.* FROM papers p"]
    # dict.fromkeys preserves insertion order while deduping repeated joins
    # (e.g., asking for both an author and a tag wouldn't add the author
    # joins twice).
    parts.extend(dict.fromkeys(joins).keys())
    if where:
        parts.append("WHERE " + " AND ".join(where))
    parts.append("ORDER BY p.year DESC NULLS LAST, p.title")
    return " ".join(parts), params
