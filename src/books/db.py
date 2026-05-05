"""SQLite persistence layer for the bibliography library.

Schema lives at the top of this module; access goes through ``connect()`` (a
context-managed ``sqlite3.Connection`` with row factory + foreign keys on).
The DB stores canonical bibliographic data — vector chunks live in Chroma,
not here, and can be rebuilt from PDFs at any time.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from books import config
from books.metadata.models import Author, PaperMatch

# Bumped manually whenever schema changes; mismatches raise — migrations are
# not implemented yet (this is v0.1).
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  id INTEGER PRIMARY KEY,
  doi TEXT UNIQUE,
  arxiv_id TEXT UNIQUE,
  title TEXT NOT NULL,
  year INTEGER,
  journal TEXT,
  publisher TEXT,
  abstract TEXT,
  type TEXT,
  file_path TEXT NOT NULL,
  source_pdf_hash TEXT UNIQUE,    -- sha256 of the imported file; blocks duplicates
  imported_at TEXT NOT NULL,      -- ISO8601, UTC
  needs_reindex INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT              -- raw API response, kept for forensics
);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);

CREATE TABLE IF NOT EXISTS authors (
  id INTEGER PRIMARY KEY,
  family_name TEXT NOT NULL,
  given_name TEXT,
  orcid TEXT,
  UNIQUE(family_name, given_name)
);

CREATE TABLE IF NOT EXISTS paper_authors (
  paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  author_id INTEGER NOT NULL REFERENCES authors(id),
  position INTEGER NOT NULL,      -- 0-indexed authorship order
  PRIMARY KEY (paper_id, position)
);
CREATE INDEX IF NOT EXISTS idx_paper_authors_author ON paper_authors(author_id);

CREATE TABLE IF NOT EXISTS tags (
  paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  tag TEXT NOT NULL,
  PRIMARY KEY (paper_id, tag)
);

CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY
);
"""


def init_db(path: Path | None = None) -> None:
    """Create or upgrade the SQLite database at ``path`` (defaults to config).

    Idempotent: re-running on an initialized DB is a no-op. Raises if the
    on-disk schema version does not match :data:`SCHEMA_VERSION` (no
    migrations are defined yet).
    """
    target = path or config.db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _raw_connect(target) as conn:
        conn.executescript(SCHEMA)
        cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
        elif row["version"] != SCHEMA_VERSION:
            raise RuntimeError(
                f"db schema version {row['version']} != expected {SCHEMA_VERSION}; "
                "no migrations defined yet"
            )


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a configured sqlite3.Connection. Initialises the DB on first use.

    The connection commits on clean exit and rolls back on exceptions.
    """
    target = path or config.db_path()
    if not target.exists():
        init_db(target)
    with _raw_connect(target) as conn:
        yield conn


@contextmanager
def _raw_connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Bare connection helper — used internally during init/connect.

    Configures Row factory (column-by-name access) and turns on foreign-key
    enforcement (off by default in SQLite).
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_paper(
    conn: sqlite3.Connection,
    match: PaperMatch,
    *,
    file_path: str,
    source_pdf_hash: str,
) -> int:
    """Insert a paper row plus its authors. Returns the new ``papers.id``.

    Raises :class:`sqlite3.IntegrityError` on duplicate DOI / arXiv ID / hash.
    """
    cur = conn.execute(
        """
        INSERT INTO papers
          (doi, arxiv_id, title, year, journal, publisher, abstract, type,
           file_path, source_pdf_hash, imported_at, needs_reindex, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (
            match.doi,
            match.arxiv_id,
            match.title,
            match.year,
            match.journal,
            match.publisher,
            match.abstract,
            match.type,
            file_path,
            source_pdf_hash,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            json.dumps(match.raw, default=str) if match.raw else None,
        ),
    )
    paper_id = int(cur.lastrowid)
    for position, author in enumerate(match.authors):
        author_id = _upsert_author(conn, author)
        conn.execute(
            "INSERT INTO paper_authors(paper_id, author_id, position) VALUES (?, ?, ?)",
            (paper_id, author_id, position),
        )
    return paper_id


def _upsert_author(conn: sqlite3.Connection, author: Author) -> int:
    """Find-or-insert an author row by (family_name, given_name); returns id.

    Two SELECTs because SQLite's ``=`` does not match NULL (only ``IS`` does).
    We try both forms to handle the case where ``given_name`` is None.
    """
    conn.execute(
        "INSERT OR IGNORE INTO authors(family_name, given_name, orcid) VALUES (?, ?, ?)",
        (author.family, author.given, author.orcid),
    )
    cur = conn.execute(
        "SELECT id FROM authors WHERE family_name = ? AND given_name IS ?",
        (author.family, author.given),
    )
    row = cur.fetchone()
    if row is None:
        cur = conn.execute(
            "SELECT id FROM authors WHERE family_name = ? AND given_name = ?",
            (author.family, author.given),
        )
        row = cur.fetchone()
    return int(row["id"])


def find_by_hash(conn: sqlite3.Connection, sha256: str) -> sqlite3.Row | None:
    """Look up a paper by the sha256 of its imported PDF (used for dedup)."""
    return conn.execute(
        "SELECT * FROM papers WHERE source_pdf_hash = ?", (sha256,)
    ).fetchone()


def find_by_doi(conn: sqlite3.Connection, doi: str) -> sqlite3.Row | None:
    """Look up a paper by exact DOI."""
    return conn.execute("SELECT * FROM papers WHERE doi = ?", (doi,)).fetchone()


def find_by_arxiv(conn: sqlite3.Connection, arxiv_id: str) -> sqlite3.Row | None:
    """Look up a paper by exact arXiv ID."""
    return conn.execute(
        "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()


def get_paper(conn: sqlite3.Connection, ident: str) -> sqlite3.Row | None:
    """Resolve an identifier to a paper row.

    Tries ``id`` (if numeric), then DOI, then arXiv ID. Returns None if none match.
    """
    if ident.isdigit():
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (int(ident),)).fetchone()
        if row is not None:
            return row
    return find_by_doi(conn, ident) or find_by_arxiv(conn, ident)


def get_authors(conn: sqlite3.Connection, paper_id: int) -> list[sqlite3.Row]:
    """Return the authors of a paper, in authorship order."""
    return list(
        conn.execute(
            """
            SELECT a.* FROM authors a
            JOIN paper_authors pa ON pa.author_id = a.id
            WHERE pa.paper_id = ?
            ORDER BY pa.position
            """,
            (paper_id,),
        )
    )


def get_tags(conn: sqlite3.Connection, paper_id: int) -> list[str]:
    """Return the user-applied tags on a paper, sorted alphabetically."""
    return [
        r["tag"]
        for r in conn.execute(
            "SELECT tag FROM tags WHERE paper_id = ? ORDER BY tag", (paper_id,)
        )
    ]


def delete_paper(conn: sqlite3.Connection, paper_id: int) -> None:
    """Delete a paper. Cascades remove its author links and tags."""
    conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))


def set_needs_reindex(conn: sqlite3.Connection, paper_id: int, value: bool) -> None:
    """Mark a paper as needing re-chunking/re-embedding (or clear the flag)."""
    conn.execute(
        "UPDATE papers SET needs_reindex = ? WHERE id = ?",
        (1 if value else 0, paper_id),
    )


def papers_needing_reindex(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all papers flagged ``needs_reindex = 1``."""
    return list(conn.execute("SELECT * FROM papers WHERE needs_reindex = 1"))
