"""Build the LLM prompt and the citation reference list from packed chunks.

The context is numbered ``[1]``, ``[2]``, … so the model can cite inline; the
same numbering is returned as a list of :class:`Reference` objects for the
command to render after the answer.
"""

import sqlite3
from dataclasses import dataclass

from books import db
from books.retrieval import SearchResult

DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant answering questions strictly from the "
    "provided excerpts of the user's paper library. Use only information "
    "contained in the numbered excerpts. Cite every claim inline with the "
    "bracketed number of the excerpt(s) it draws on, e.g. [1] or [2][3]. If "
    "the excerpts do not contain enough information to answer, say so plainly "
    "rather than guessing. Be concise and precise."
)


@dataclass
class Reference:
    """One numbered citation target, mapped 1:1 to a packed chunk."""

    n: int
    paper_id: int
    author: str
    year: int | None
    page: int
    title: str

    def label(self) -> str:
        """Short one-line citation, e.g. ``Smith et al. (2024) p.3  id=12``."""
        year = self.year if self.year is not None else "?"
        return f"{self.author} ({year}) p.{self.page}  id={self.paper_id}"


def _author_str(authors: list[sqlite3.Row]) -> str:
    """Render an author label: ``Surname`` or ``Surname et al.`` or ``[unknown]``."""
    if not authors:
        return "[unknown]"
    first = authors[0]["family_name"]
    return f"{first} et al." if len(authors) > 1 else first


def build_context(
    conn: sqlite3.Connection, packed: list[SearchResult]
) -> tuple[str, list[Reference]]:
    """Return ``(context_text, references)`` for the packed chunks.

    Each chunk becomes a numbered block carrying its source label and the chunk
    text; ``references[i]`` describes block ``[i + 1]``.
    """
    blocks: list[str] = []
    refs: list[Reference] = []
    for i, r in enumerate(packed, start=1):
        row = conn.execute(
            "SELECT title, year FROM papers WHERE id = ?", (r.paper_id,)
        ).fetchone()
        authors = db.get_authors(conn, r.paper_id) if row else []
        author = _author_str(authors)
        title = (row["title"] if row else "") or ""
        year = row["year"] if row else None

        refs.append(
            Reference(
                n=i,
                paper_id=r.paper_id,
                author=author,
                year=year,
                page=r.page,
                title=title,
            )
        )
        header = f"[{i}] {author} ({year if year is not None else '?'}), " \
                 f"\"{title}\", p.{r.page}"
        blocks.append(f"{header}\n{r.text.strip()}")

    return "\n\n".join(blocks), refs


def build_user_message(context: str, question: str) -> str:
    """Stitch the numbered context and the user's question into one message."""
    return (
        f"Excerpts from the library:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above, citing inline with [n]."
    )
