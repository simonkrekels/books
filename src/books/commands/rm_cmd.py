"""``book rm`` — remove a paper from the library."""

import typer
from rich.console import Console

from books import config, db

console = Console()


def run(
    ident: str = typer.Argument(..., help="Paper id or DOI."),
    keep_file: bool = typer.Option(
        True,
        "--keep-file/--delete-file",
        help="Leave the PDF on disk (default) or also delete it.",
    ),
) -> None:
    """Remove a paper from the library.

    Drops the SQLite row (cascading author / tag links) and deletes Chroma
    chunks. By default the PDF file on disk is left in place — pass
    ``--delete-file`` to remove it as well.
    """
    with db.connect() as conn:
        row = db.get_paper(conn, ident)
        if row is None:
            console.print(f"[red]no paper with id/doi:[/red] {ident}")
            raise typer.Exit(code=1)
        paper_id = int(row["id"])
        file_path = config.library_dir() / row["file_path"]
        db.delete_paper(conn, paper_id)

    # Best-effort Chroma cleanup. If Chroma isn't reachable (e.g. corrupt
    # store), we still want the SQLite delete to stand — the chunks become
    # orphans but `book reindex --all` can clean them up later.
    try:
        from books.index.indexer import delete_paper_chunks

        delete_paper_chunks(paper_id)
    except Exception as e:
        console.print(f"[yellow]could not remove chunks from chroma:[/yellow] {e}")

    console.print(f"removed [bold]{row['title']}[/bold] (id={paper_id})")
    if not keep_file:
        try:
            file_path.unlink()
            console.print(f"deleted file [dim]{file_path}[/dim]")
        except FileNotFoundError:
            console.print(f"[yellow]file already missing:[/yellow] {file_path}")
    else:
        console.print(f"kept file [dim]{file_path}[/dim]")
