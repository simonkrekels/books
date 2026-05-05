"""``book reindex`` — rebuild Chroma chunks for selected papers."""

import typer
from rich.console import Console

from books import config, db

console = Console()


def run(
    idents: list[str] = typer.Argument(None, help="Paper id(s) or DOI(s)."),
    all_: bool = typer.Option(False, "--all", help="Reindex every paper."),
) -> None:
    """Re-chunk and re-embed papers in the vector index.

    Resolution rules:

    * ``--all`` → every paper in the library.
    * one or more positional idents → those specific papers.
    * neither → every paper currently flagged ``needs_reindex = 1``.

    Per-paper failures are logged but don't stop the run; exit code is 1 if
    anything failed.
    """
    from books.index.indexer import delete_paper_chunks, index_paper

    rows = _resolve(idents, all_)
    if not rows:
        console.print("[dim]nothing to reindex[/dim]")
        raise typer.Exit(code=1)

    library = config.library_dir()
    failed = 0
    for row in rows:
        paper_id = int(row["id"])
        pdf_path = library / row["file_path"]
        if not pdf_path.exists():
            console.print(f"[yellow]missing file (id={paper_id}):[/yellow] {pdf_path}")
            failed += 1
            continue
        try:
            # Delete first so a re-chunked paper with fewer chunks doesn't
            # leave stale chunk IDs lingering in the collection.
            delete_paper_chunks(paper_id)
            n = index_paper(
                paper_id=paper_id,
                pdf_path=pdf_path,
                title=row["title"],
                doi=row["doi"],
            )
            with db.connect() as conn:
                db.set_needs_reindex(conn, paper_id, False)
            console.print(f"[green]id={paper_id}[/green] {row['title']}: {n} chunks")
        except Exception as e:
            console.print(f"[red]id={paper_id} failed:[/red] {e}")
            failed += 1

    if failed:
        raise typer.Exit(code=1)


def _resolve(idents: list[str] | None, all_: bool) -> list:
    """Resolve the CLI args to an actual list of paper rows."""
    with db.connect() as conn:
        if all_:
            return list(conn.execute("SELECT * FROM papers ORDER BY id"))
        if idents:
            rows = []
            for ident in idents:
                row = db.get_paper(conn, ident)
                if row is None:
                    console.print(f"[yellow]not found:[/yellow] {ident}")
                else:
                    rows.append(row)
            return rows
        # Default behaviour: pick up the breadcrumbs from any failed
        # post-import indexing runs.
        return db.papers_needing_reindex(conn)
