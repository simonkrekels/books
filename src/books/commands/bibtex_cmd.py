"""``book bibtex`` — export bibliography entries to a .bib file."""

import re
import sqlite3
from pathlib import Path

import typer
from rich.console import Console

from books import db
from books.paths import slugify

console = Console()


def run(
    bib_file: Path = typer.Argument(..., help="Target .bib file (appended to)."),
    ident_or_tag: str = typer.Argument(
        ..., help="Paper id/DOI/arXiv/ISBN, or a tag name to export all tagged papers."
    ),
) -> None:
    """Append one or more BibTeX entries to BIB_FILE.

    IDENT_OR_TAG is resolved as a paper identifier first; if no match is
    found it is treated as a tag and all papers carrying that tag are exported.
    Entries already present in the file (matched by BibTeX key) are skipped.
    """
    with db.connect() as conn:
        papers = _resolve(conn, ident_or_tag)

    if not papers:
        console.print(f"[red]nothing found for[/red] {ident_or_tag!r}")
        raise typer.Exit(code=1)

    existing_keys = _read_keys(bib_file)
    added = skipped = 0

    with bib_file.open("a", encoding="utf-8") as f:
        for row, authors in papers:
            key = _make_key(row, authors)
            if key in existing_keys:
                console.print(f"[dim]skip (already in file):[/dim] @{key}")
                skipped += 1
                continue
            entry = _format_entry(key, row, authors)
            f.write(entry + "\n\n")
            console.print(f"[green]added[/green] @{key}")
            added += 1

    console.print(f"[dim]{added} added, {skipped} skipped → {bib_file}[/dim]")


def _resolve(
    conn: sqlite3.Connection, ident_or_tag: str
) -> list[tuple[sqlite3.Row, list[sqlite3.Row]]]:
    """Return a list of (paper_row, authors) for the given identifier or tag."""
    row = db.get_paper(conn, ident_or_tag)
    if row is not None:
        return [(row, db.get_authors(conn, int(row["id"])))]
    # Treat as tag.
    rows = list(
        conn.execute(
            """
            SELECT DISTINCT p.* FROM papers p
            JOIN tags t ON t.paper_id = p.id
            WHERE t.tag = ?
            ORDER BY p.year DESC NULLS LAST, p.title
            """,
            (ident_or_tag,),
        )
    )
    return [(r, db.get_authors(conn, int(r["id"]))) for r in rows]


def _read_keys(path: Path) -> set[str]:
    """Collect all BibTeX cite-keys already present in *path* (if it exists)."""
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\{([^,\s]+),", text))


def _make_key(row: sqlite3.Row, authors: list[sqlite3.Row]) -> str:
    """Generate a BibTeX cite-key: SluggedFamilyYear (e.g. ``Smith2023``)."""
    family = (authors[0]["family_name"] if authors else "") or "Unknown"
    year = str(row["year"] or "")
    base = slugify(family).capitalize().replace("-", "") + year
    return base or "Unknown"


def _format_entry(key: str, row: sqlite3.Row, authors: list[sqlite3.Row]) -> str:
    """Render a single BibTeX entry string."""
    entry_type = _bibtex_type(row["type"])
    fields: list[tuple[str, str]] = []

    title = row["title"] or ""
    fields.append(("title", _brace(title)))

    if authors:
        author_str = " and ".join(
            f"{a['family_name']}, {a['given_name']}".rstrip(", ")
            for a in authors
        )
        fields.append(("author", _brace(author_str)))

    if row["year"]:
        fields.append(("year", str(row["year"])))
    if row["journal"]:
        fields.append(("journal", _brace(row["journal"])))
    if row["publisher"]:
        fields.append(("publisher", _brace(row["publisher"])))
    if row["doi"]:
        fields.append(("doi", row["doi"]))
    if row["isbn"]:
        fields.append(("isbn", row["isbn"]))
    if row["arxiv_id"]:
        fields.append(("eprint", row["arxiv_id"]))
        fields.append(("archivePrefix", "arXiv"))

    body = ",\n".join(f"  {k} = {{{v}}}" if not v.startswith("{") else f"  {k} = {v}" for k, v in fields)
    return f"@{entry_type}{{{key},\n{body}\n}}"


def _bibtex_type(paper_type: str | None) -> str:
    mapping = {
        "journal-article": "article",
        "book": "book",
        "book-chapter": "inbook",
        "proceedings-article": "inproceedings",
    }
    return mapping.get(paper_type or "", "misc")


def _brace(text: str) -> str:
    """Wrap in double braces to preserve capitalisation."""
    return "{" + text + "}"
