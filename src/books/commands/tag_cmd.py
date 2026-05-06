"""``book tag`` — manage tags on library entries."""

import typer
from rich.console import Console
from rich.table import Table

from books import db

app = typer.Typer(help="Add, remove, or list tags.", no_args_is_help=True)
console = Console()


@app.command("add")
def tag_add(
    ident: str = typer.Argument(..., help="Paper id, DOI, arXiv ID, or ISBN."),
    tag: str = typer.Argument(..., help="Tag to apply."),
) -> None:
    """Apply TAG to the paper identified by IDENT."""
    with db.connect() as conn:
        row = db.get_paper(conn, ident)
        if row is None:
            console.print(f"[red]not found:[/red] {ident}")
            raise typer.Exit(code=1)
        db.add_tag(conn, int(row["id"]), tag)
    console.print(f"[green]tagged[/green] id={row['id']} ← {tag!r}")


@app.command("rm")
def tag_rm(
    ident: str = typer.Argument(..., help="Paper id, DOI, arXiv ID, or ISBN."),
    tag: str = typer.Argument(..., help="Tag to remove."),
) -> None:
    """Remove TAG from the paper identified by IDENT."""
    with db.connect() as conn:
        row = db.get_paper(conn, ident)
        if row is None:
            console.print(f"[red]not found:[/red] {ident}")
            raise typer.Exit(code=1)
        db.remove_tag(conn, int(row["id"]), tag)
    console.print(f"[dim]removed tag[/dim] {tag!r} from id={row['id']}")


@app.command("ls")
def tag_ls(
    ident: str = typer.Argument(
        None, help="Paper id/DOI/arXiv/ISBN — omit to list all tags."
    ),
) -> None:
    """List tags for one paper, or all tags with counts across the library."""
    with db.connect() as conn:
        if ident is not None:
            row = db.get_paper(conn, ident)
            if row is None:
                console.print(f"[red]not found:[/red] {ident}")
                raise typer.Exit(code=1)
            tags = db.get_tags(conn, int(row["id"]))
            if not tags:
                console.print(f"[dim]no tags on id={row['id']}[/dim]")
            else:
                console.print(", ".join(tags))
        else:
            rows = db.list_all_tags(conn)
            if not rows:
                console.print("[dim]no tags in library[/dim]")
                return
            table = Table(show_header=True, header_style="bold")
            table.add_column("tag")
            table.add_column("papers", justify="right")
            for r in rows:
                table.add_row(r["tag"], str(r["count"]))
            console.print(table)
