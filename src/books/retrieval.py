"""Shared retrieval pipeline — embed a query and return ranked chunks.

This module holds the *retrieval* half of search (embedding + Chroma cosine +
FTS5 BM25 + RRF fusion), independent of any rendering.  Both ``book search``
(:mod:`books.commands.search_cmd`) and ``book ask``
(:mod:`books.commands.ask_cmd`) build on :func:`retrieve`.

Heavy dependencies (torch via the embedder, chromadb) are imported lazily
inside :func:`retrieve` so importing this module for :class:`SearchResult`
alone stays cheap.
"""

import sqlite3
from dataclasses import dataclass

from books import config


class RetrievalUnavailable(Exception):
    """Raised when the vector index holds no chunks to search."""


@dataclass
class SearchResult:
    """A single chunk result, normalized from either Chroma or FTS5."""

    paper_id: int
    chunk_index: int
    page: int
    text: str
    score: float  # cosine similarity, negated-BM25, or RRF score


def retrieve(
    conn: sqlite3.Connection,
    query: str,
    n: int,
    *,
    allowed_ids: set[int] | None = None,
) -> list[SearchResult]:
    """Embed *query* and return up to *n* ranked chunks.

    Uses hybrid search (cosine + BM25 fused with RRF) when hybrid mode is
    enabled and the FTS index has content; otherwise falls back to cosine-only.
    ``allowed_ids`` restricts results to those papers (``None`` = no filter).

    Raises :class:`RetrievalUnavailable` if nothing has been indexed yet.
    """
    from books.index import fts as fts_index
    from books.index.chroma import ChromaIndex
    from books.index.indexer import get_embedder

    embedder = get_embedder()
    prompt = config.query_prompt()
    [query_vec] = embedder.embed([prompt + query if prompt else query])

    index = ChromaIndex(config.chroma_dir())
    if index.count() == 0:
        raise RetrievalUnavailable

    if config.hybrid_search() and fts_index.has_content(conn):
        return hybrid_search(conn, index, query_vec, query, n, allowed_ids)
    return cosine_search(index, query_vec, n, allowed_ids)


def cosine_search(
    index, query_vec: list[float], n: int, allowed_ids: set[int] | None = None
) -> list[SearchResult]:
    """Return up to *n* chunks ranked by cosine similarity."""
    where: dict | None = None
    if allowed_ids is not None:
        if len(allowed_ids) == 1:
            where = {"paper_id": next(iter(allowed_ids))}
        else:
            where = {"paper_id": {"$in": list(allowed_ids)}}
    res = index.query(query_embedding=query_vec, n_results=n, where=where)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]
    return [
        SearchResult(
            paper_id=int(meta["paper_id"]),
            chunk_index=int(meta["chunk_index"]),
            page=int(meta["page"]),
            text=doc,
            score=1.0 - dist,
        )
        for doc, meta, dist in zip(docs, metas, distances)
    ]


def hybrid_search(
    conn,
    index,
    query_vec: list[float],
    query_text: str,
    n: int,
    allowed_ids: set[int] | None = None,
) -> list[SearchResult]:
    """Fuse cosine + BM25 results with Reciprocal Rank Fusion."""
    from books.index import fts as fts_index

    cosine_hits = cosine_search(index, query_vec, n, allowed_ids)
    bm25_hits = fts_index.search(conn, query_text, n, allowed_ids)

    cosine_ranked = [(f"{r.paper_id}:{r.chunk_index}", r.score) for r in cosine_hits]
    fused = fts_index.rrf_fuse([cosine_ranked, bm25_hits])

    # Look up text + page for each fused chunk from the SQLite chunks table.
    results: list[SearchResult] = []
    for chunk_id, rrf_score in fused[:n]:
        paper_id, chunk_index = map(int, chunk_id.split(":"))
        row = conn.execute(
            "SELECT page, text FROM chunks WHERE paper_id = ? AND chunk_index = ?",
            (paper_id, chunk_index),
        ).fetchone()
        if row is None:
            continue
        results.append(
            SearchResult(
                paper_id=paper_id,
                chunk_index=chunk_index,
                page=row["page"],
                text=row["text"],
                score=rrf_score,
            )
        )
    return results
