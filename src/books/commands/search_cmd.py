"""``book search`` — semantic search across indexed PDFs."""

from collections import defaultdict
from dataclasses import dataclass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from books import config, db

console = Console()

# Number of candidate results fetched from each source before fusion.
# Larger windows improve recall at the cost of extra FTS/Chroma work.
_FETCH_MULTIPLIER = 5


@dataclass
class SearchResult:
    """A single chunk result, normalized from either Chroma or FTS5."""

    paper_id: int
    chunk_index: int
    page: int
    text: str
    score: float  # cosine similarity, negated-BM25, or RRF score


def run(
    query: str = typer.Argument(..., help="Free-text search query."),
    k: int = typer.Option(5, "-k", "--top-k", help="Number of results (papers when grouped, chunks otherwise)."),
    group: bool = typer.Option(True, "--group/--no-group", help="Group chunks by paper (default: on)."),
) -> None:
    """Embed the query, fetch top-k results from the index, render results.

    With ``--group`` (the default), chunks are aggregated by paper and ranked
    by best chunk score — one panel per paper.  With ``--no-group``, one panel
    per chunk is shown.

    When the FTS5 BM25 index is populated (after ``book reindex --all``), BM25
    and cosine scores are fused via Reciprocal Rank Fusion for better recall on
    exact-term queries.  Falls back to cosine-only with a warning if the index
    is empty.
    """
    from books.index.chroma import ChromaIndex
    from books.index.indexer import get_embedder

    embedder = get_embedder()
    [query_vec] = embedder.embed([query])

    index = ChromaIndex(config.chroma_dir())
    if index.count() == 0:
        console.print("[yellow]no chunks indexed yet[/yellow] — try `book reindex --all`")
        raise typer.Exit(code=1)

    fetch_n = k * _FETCH_MULTIPLIER

    with db.connect() as conn:
        from books.index import fts as fts_index

        if config.hybrid_search() and fts_index.has_content(conn):
            results = _hybrid_search(conn, index, query_vec, query, fetch_n)
        else:
            if config.hybrid_search():
                console.print(
                    "[yellow]BM25 index empty[/yellow] — run `book reindex --all` to enable hybrid search"
                )
            results = _cosine_search(index, query_vec, fetch_n)

        if not results:
            console.print("[dim]no matches[/dim]")
            raise typer.Exit(code=1)

        if group:
            _render_grouped(conn, query, results, k)
        else:
            _render_flat(conn, query, results[:k])


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------


def _cosine_search(index, query_vec: list[float], n: int) -> list[SearchResult]:
    """Return up to *n* chunks ranked by cosine similarity."""
    res = index.query(query_embedding=query_vec, n_results=n)
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


def _hybrid_search(
    conn,
    index,
    query_vec: list[float],
    query_text: str,
    n: int,
) -> list[SearchResult]:
    """Fuse cosine + BM25 results with Reciprocal Rank Fusion."""
    from books.index import fts as fts_index

    cosine_hits = _cosine_search(index, query_vec, n)
    bm25_hits = fts_index.search(conn, query_text, n)

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


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_grouped(conn, query: str, results: list[SearchResult], k: int) -> None:
    """One panel per paper; chunks within a paper sorted by page number."""
    groups: dict[int, dict] = defaultdict(
        lambda: {"chunks": [], "best_score": 0.0, "row": None, "authors": []}
    )
    for r in results:
        g = groups[r.paper_id]
        g["chunks"].append(r)
        if r.score > g["best_score"]:
            g["best_score"] = r.score

    for paper_id, g in groups.items():
        g["row"] = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        g["authors"] = db.get_authors(conn, paper_id) if g["row"] else []

    ranked = sorted(groups.items(), key=lambda kv: kv[1]["best_score"], reverse=True)[:k]

    for paper_id, g in ranked:
        row = g["row"]
        authors = g["authors"]
        title = (row["title"] if row else "") or ""
        year = row["year"] if row else None
        author_str = ", ".join(a["family_name"] for a in authors[:3]) or "[unknown]"

        header = Text()
        header.append(f"id={paper_id}  ", style="dim")
        header.append(f"score {g['best_score']:.3f}", style="green")

        body = Text()
        body.append(f"{author_str} ({year or '?'})\n", style="bold")
        body.append(f"{title}\n", style="italic")

        for chunk in sorted(g["chunks"], key=lambda c: c.page):
            body.append(f"\n[page {chunk.page}  score {chunk.score:.3f}]\n", style="cyan")
            body.append(_snippet(chunk.text, query))

        console.print(Panel(body, title=header, border_style="blue"))


def _render_flat(conn, query: str, results: list[SearchResult]) -> None:
    """One panel per chunk result."""
    for r in results:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (r.paper_id,)).fetchone()
        authors = db.get_authors(conn, r.paper_id) if row else []
        title = (row["title"] if row else "") or ""
        year = row["year"] if row else None
        author_str = ", ".join(a["family_name"] for a in authors[:3]) or "[unknown]"

        header = Text()
        header.append(f"id={r.paper_id}  ", style="dim")
        header.append(f"page {r.page}  ", style="cyan")
        header.append(f"score {r.score:.3f}", style="green")

        body = Text()
        body.append(f"{author_str} ({year or '?'})\n", style="bold")
        body.append(f"{title}\n", style="italic")
        body.append("\n")
        body.append(_snippet(r.text, query))

        console.print(Panel(body, title=header, border_style="blue"))


def _snippet(text: str, query: str, max_len: int = 600) -> Text:
    """Return a query-centred snippet with query terms highlighted."""
    text = " ".join(text.split())  # collapse whitespace from PDF extraction
    if len(text) > max_len:
        lower = text.lower()
        pos = -1
        for term in query.lower().split():
            pos = lower.find(term)
            if pos >= 0:
                break
        if pos < 0:
            text = text[:max_len] + "…"
        else:
            start = max(0, pos - max_len // 2)
            text = ("…" if start > 0 else "") + text[start : start + max_len] + "…"
    out = Text(text)
    for term in query.split():
        out.highlight_words([term], "bold magenta")
    return out
