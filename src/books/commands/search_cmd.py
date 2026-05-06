"""``book search`` — semantic search across indexed PDFs."""

from collections import defaultdict

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from books import config, db

console = Console()


def run(
    query: str = typer.Argument(..., help="Free-text search query."),
    k: int = typer.Option(5, "-k", "--top-k", help="Number of results (papers when grouped, chunks otherwise)."),
    group: bool = typer.Option(True, "--group/--no-group", help="Group chunks by paper (default: on)."),
) -> None:
    """Embed the query, fetch top-k chunks from Chroma, render results.

    With ``--group`` (the default), chunks are aggregated by paper and ranked
    by best chunk score — one panel per paper. With ``--no-group``, one panel
    per chunk is shown (the original behaviour).
    """
    from books.index.chroma import ChromaIndex
    from books.index.indexer import get_embedder

    embedder = get_embedder()
    [query_vec] = embedder.embed([query])

    index = ChromaIndex(config.chroma_dir())
    if index.count() == 0:
        console.print("[yellow]no chunks indexed yet[/yellow] — try `book reindex --all`")
        raise typer.Exit(code=1)

    # Fetch more chunks than needed when grouping, to ensure we surface k
    # distinct papers even if one paper dominates the raw top results.
    fetch_n = k * 5 if group else k
    res = index.query(query_embedding=query_vec, n_results=fetch_n)
    # Chroma returns lists-of-lists (one inner list per query vector); we
    # always submit exactly one, so unwrap.
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]

    if not docs:
        console.print("[dim]no matches[/dim]")
        raise typer.Exit(code=1)

    with db.connect() as conn:
        if group:
            _render_grouped(conn, query, docs, metas, distances, k)
        else:
            _render_flat(conn, query, docs, metas, distances)


def _render_grouped(conn, query: str, docs, metas, distances, k: int) -> None:
    """One panel per paper, showing all matching chunks; ranked by best score."""
    # Accumulate per-paper data keyed by paper_id.
    groups: dict[int, dict] = defaultdict(lambda: {"chunks": [], "best_score": 0.0, "row": None, "authors": []})

    for doc, meta, dist in zip(docs, metas, distances):
        paper_id = int(meta["paper_id"])
        score = 1 - dist
        g = groups[paper_id]
        g["chunks"].append((int(meta["page"]), doc, score))
        if score > g["best_score"]:
            g["best_score"] = score

    # Fetch DB rows once per unique paper.
    for paper_id, g in groups.items():
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        g["row"] = row
        g["authors"] = db.get_authors(conn, paper_id) if row else []

    # Sort groups by best score and take top k.
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

        # Sort chunks within this paper by page number.
        for page, doc, score in sorted(g["chunks"], key=lambda t: t[0]):
            body.append(f"\n[page {page}  score {score:.3f}]\n", style="cyan")
            body.append(_snippet(doc, query))

        console.print(Panel(body, title=header, border_style="blue"))


def _render_flat(conn, query: str, docs, metas, distances) -> None:
    """Original per-chunk rendering, one panel per result."""
    for doc, meta, dist in zip(docs, metas, distances):
        paper_id = int(meta["paper_id"])
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        authors = db.get_authors(conn, paper_id) if row else []
        title = (row["title"] if row else meta.get("title", "")) or ""
        year = row["year"] if row else None
        author_str = ", ".join(a["family_name"] for a in authors[:3]) or "[unknown]"

        header = Text()
        header.append(f"id={paper_id}  ", style="dim")
        header.append(f"page {meta['page']}  ", style="cyan")
        header.append(f"score {1 - dist:.3f}", style="green")

        body = Text()
        body.append(f"{author_str} ({year or '?'})\n", style="bold")
        body.append(f"{title}\n", style="italic")
        body.append("\n")
        body.append(_snippet(doc, query))

        console.print(Panel(body, title=header, border_style="blue"))


def _snippet(doc: str, query: str, max_len: int = 600) -> Text:
    """Return a query-centred snippet with the query terms bold-highlighted."""
    text = " ".join(doc.split())  # collapse whitespace from PDF extraction
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
