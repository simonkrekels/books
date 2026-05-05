"""``book show`` — display every field of a single paper."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from books import config, db

console = Console()


def run(ident: str = typer.Argument(..., help="Paper id or DOI.")) -> None:
    """Show full details for a paper, including authors, tags, and abstract."""
    with db.connect() as conn:
        row = db.get_paper(conn, ident)
        if row is None:
            console.print(f"[red]no paper with id/doi:[/red] {ident}")
            raise typer.Exit(code=1)
        authors = db.get_authors(conn, row["id"])
        tags = db.get_tags(conn, row["id"])

    fields = Table.grid(padding=(0, 1))
    fields.add_column(style="bold cyan")
    fields.add_column()
    fields.add_row("title", row["title"])
    fields.add_row(
        "authors",
        ", ".join(
            f"{a['given_name'] or ''} {a['family_name']}".strip() for a in authors
        )
        or "[unknown]",
    )
    fields.add_row("year", str(row["year"] or ""))
    fields.add_row("journal", row["journal"] or "")
    fields.add_row("doi", row["doi"] or "")
    fields.add_row("arxiv", row["arxiv_id"] or "")
    fields.add_row("isbn", row["isbn"] or "")
    fields.add_row("type", row["type"] or "")
    fields.add_row("file", str(config.library_dir() / row["file_path"]))
    fields.add_row("imported", row["imported_at"])
    fields.add_row("tags", ", ".join(tags) if tags else "")
    fields.add_row("needs reindex", "yes" if row["needs_reindex"] else "no")
    console.print(Panel(fields, title=f"paper {row['id']}"))

    # Render the abstract in its own panel — abstracts are often paragraph-long
    # and look bad alongside the metadata key-value grid.
    if row["abstract"]:
        console.print(Panel(row["abstract"], title="abstract", border_style="dim"))
