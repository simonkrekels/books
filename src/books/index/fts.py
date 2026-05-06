"""SQLite FTS5 index for BM25 full-text search over chunk text.

The ``chunks`` table is the canonical text store; ``chunks_fts`` is a
standalone FTS5 virtual table whose rowids mirror ``chunks.id``.  The two
tables are kept in sync manually — every write to ``chunks`` must be paired
with a corresponding write to ``chunks_fts``.

Typical call sites:

* :func:`upsert_paper`  — called by the indexer after embedding.
* :func:`delete_paper`  — called before removing a paper from the library.
* :func:`rebuild`       — called after bulk reindex to cheaply resync FTS.
* :func:`search`        — called by the search command for BM25 results.
* :func:`has_content`   — graceful fallback check in the search command.
"""

import sqlite3


def upsert_paper(
    conn: sqlite3.Connection, paper_id: int, chunks: list
) -> None:
    """Replace all stored chunks for *paper_id* and rebuild FTS entries.

    ``chunks`` is a list of :class:`books.index.chunker.Chunk` objects (or
    any objects with ``chunk_index``, ``page_number``, and ``text`` attrs).
    Existing rows for the paper are removed first so a re-index never leaves
    stale chunks behind.
    """
    delete_paper(conn, paper_id)
    for chunk in chunks:
        cur = conn.execute(
            "INSERT INTO chunks(paper_id, chunk_index, page, text) VALUES(?, ?, ?, ?)",
            (paper_id, chunk.chunk_index, chunk.page_number, chunk.text),
        )
        conn.execute(
            "INSERT INTO chunks_fts(rowid, text) VALUES(?, ?)",
            (cur.lastrowid, chunk.text),
        )


def delete_paper(conn: sqlite3.Connection, paper_id: int) -> None:
    """Remove all chunks (and their FTS entries) for *paper_id*.

    Must be called *before* the paper row is deleted from ``papers``, so that
    the ``chunks`` table still contains the rows needed for FTS cleanup.
    """
    rows = conn.execute(
        "SELECT id FROM chunks WHERE paper_id = ?", (paper_id,)
    ).fetchall()
    for row in rows:
        conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (row["id"],))
    conn.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))


def rebuild(conn: sqlite3.Connection) -> None:
    """Rebuild the FTS index from scratch using the current ``chunks`` table.

    Use after bulk operations (e.g. ``book reindex --all``) where incremental
    FTS updates have already been applied per-paper but you want to compact
    the index.  Rebuilding is idempotent and safe to call at any time.
    """
    conn.execute("DELETE FROM chunks_fts")
    conn.execute(
        "INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks"
    )


def search(
    conn: sqlite3.Connection, query: str, n: int
) -> list[tuple[str, float]]:
    """Return up to *n* BM25-ranked chunks matching *query*.

    Returns a list of ``(chunk_id, score)`` pairs where ``chunk_id`` is the
    Chroma-compatible ``"{paper_id}:{chunk_index}"`` format and ``score`` is
    a positive float (higher = more relevant).  Returns ``[]`` when the index
    is empty or the query contains no usable terms.
    """
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []
    try:
        rows = conn.execute(
            """
            SELECT c.paper_id, c.chunk_index, bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score          -- bm25() is negative; lower = better
            LIMIT ?
            """,
            (fts_query, n),
        ).fetchall()
    except sqlite3.OperationalError:
        # Malformed query or empty index — degrade silently.
        return []
    # Negate so higher score = more relevant (consistent with cosine convention).
    return [
        (f"{row['paper_id']}:{row['chunk_index']}", -row["bm25_score"])
        for row in rows
    ]


def has_content(conn: sqlite3.Connection) -> bool:
    """Return True if the chunks table contains at least one row."""
    row = conn.execute("SELECT COUNT(*) FROM chunks LIMIT 1").fetchone()
    return bool(row and row[0] > 0)


def rrf_fuse(
    lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    Each list is ``[(id, score), ...]`` sorted by score descending.  Returns
    a merged list sorted by fused score descending.  The constant *k* (default
    60, from the original RRF paper) controls rank-weight decay.
    """
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, (item_id, _) in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _build_fts_query(query: str) -> str:
    """Convert a plain-text query into an FTS5 MATCH expression.

    Each whitespace-separated token is double-quoted (making it a single-term
    phrase query) and joined with implicit AND.  Double-quotes inside tokens
    are escaped by doubling them, per the FTS5 spec.
    """
    tokens = query.split()
    if not tokens:
        return ""
    escaped = ['"' + t.replace('"', '""') + '"' for t in tokens]
    return " ".join(escaped)
