"""``book ls`` — list / filter papers in the library."""

import typer
from rich.console import Console
from rich.table import Table

from books import db, query

console = Console()


def run(
    terms: list[str] = typer.Argument(None, help="Free-text terms (match title/abstract)."),
    author: str = typer.Option(None, "--author", help="Filter by author family name."),
    year: int = typer.Option(None, "--year", help="Filter by publication year."),
    journal: str = typer.Option(None, "--journal", help="Filter by journal."),
    tag: str = typer.Option(None, "--tag", help="Filter by tag."),
    doi: str = typer.Option(None, "--doi", help="Filter by DOI."),
) -> None:
    """List papers in the library.

    All filters AND together. Free-text positional terms perform a
    LIKE-anywhere match against title/abstract. Exits non-zero when no row
    matches (handy for shell pipelines).
    """
    sql, params = query.build_papers_query(
        terms=terms or None,
        author=author,
        year=year,
        journal=journal,
        tag=tag,
        doi=doi,
    )
    with db.connect() as conn:
        rows = list(conn.execute(sql, params))
        # Pre-fetch authors for every row so the table-render loop stays
        # outside the DB context manager.
        author_lookup = {r["id"]: db.get_authors(conn, r["id"]) for r in rows}

    if not rows:
        console.print("[dim]no matches[/dim]")
        raise typer.Exit(code=1)

    table = Table(show_header=True, header_style="bold")
    table.add_column("id", justify="right", style="dim")
    table.add_column("year", justify="right")
    table.add_column("authors")
    table.add_column("title")
    table.add_column("source", style="dim")
    for r in rows:
        authors = author_lookup[r["id"]]
        # Cap to first 3 authors and indicate the rest with a trailing "+N";
        # papers with 30+ co-authors would otherwise blow out the column.
        names = ", ".join(a["family_name"] for a in authors[:3])
        if len(authors) > 3:
            names += f" +{len(authors) - 3}"
        source = r["doi"] or r["arxiv_id"] or ""
        table.add_row(
            str(r["id"]),
            str(r["year"] or ""),
            names or "[unknown]",
            r["title"],
            source,
        )
    console.print(table)
