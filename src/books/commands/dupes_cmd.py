"""``book dupes`` — find and interactively remove duplicate papers."""

import re
import unicodedata
from collections import defaultdict

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from books import config, db

console = Console()


def run(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List duplicates without prompting to remove."
    ),
    fuzzy: bool = typer.Option(
        False,
        "--fuzzy",
        help="Also surface near-matches (≥80%% title-token overlap, e.g. preprint vs. published).",
    ),
    delete_file: bool = typer.Option(
        False,
        "--delete-file",
        help="Delete the PDF on disk when removing a paper (default: keep file).",
    ),
) -> None:
    """Find duplicate papers and interactively choose which copy to keep.

    Default detection: papers whose normalised titles are identical.
    ``--fuzzy`` widens the net to papers sharing ≥ 80 % of title tokens,
    catching preprint / published-version pairs and minor title variations.

    For each duplicate group you are shown the candidates, numbered, and
    prompted to pick one to keep; the rest are removed from the library
    (identical to ``book rm``).  ``--dry-run`` lists groups without any
    prompting or removal.
    """
    with db.connect() as conn:
        rows = list(conn.execute("SELECT * FROM papers ORDER BY id"))
        author_lookup = {r["id"]: db.get_authors(conn, r["id"]) for r in rows}

    groups = _find_duplicates(rows, fuzzy=fuzzy)

    if not groups:
        console.print("[green]no duplicates found[/green]")
        return

    console.print(
        f"found [bold]{len(groups)}[/bold] duplicate group(s)\n"
    )

    if dry_run:
        for group in groups:
            console.print(_group_panel(group, author_lookup))
        return

    removed = 0
    for group in groups:
        console.print(_group_panel(group, author_lookup))
        n = len(group)
        choices = [str(i + 1) for i in range(n)] + ["s", "q"]
        choice = Prompt.ask(
            "[bold cyan]Keep which? ("
            + "/".join(str(i + 1) for i in range(n))
            + ", [S]kip, [Q]uit)[/bold cyan]",
            choices=choices,
            default="s",
            show_choices=False,
        )
        if choice == "q":
            break
        if choice == "s":
            continue
        keep_idx = int(choice) - 1
        for i, paper in enumerate(group):
            if i != keep_idx:
                _remove_paper(paper, delete_file=delete_file)
                removed += 1

    if removed:
        console.print(f"\n[green]removed {removed} duplicate(s)[/green]")


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    """Lowercase, remove punctuation and extra whitespace for comparison."""
    title = unicodedata.normalize("NFKD", title)
    title = re.sub(r"[^\w\s]", "", title)
    return " ".join(title.lower().split())


def _jaccard(a: str, b: str) -> float:
    """Token Jaccard similarity between two normalised title strings."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _find_duplicates(rows: list, *, fuzzy: bool = False) -> list[list]:
    """Return groups of duplicate paper rows (each group has ≥ 2 members).

    Uses Union-Find so transitive near-matches are clustered together.
    """
    norms = [(row, _normalize_title(row["title"])) for row in rows]
    n = len(norms)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(n):
        norm_a = norms[i][1]
        if not norm_a:
            continue
        for j in range(i + 1, n):
            norm_b = norms[j][1]
            if not norm_b:
                continue
            if norm_a == norm_b or (fuzzy and _jaccard(norm_a, norm_b) >= 0.8):
                union(i, j)

    clusters: dict[int, list] = defaultdict(list)
    for i, (row, _) in enumerate(norms):
        clusters[find(i)].append(row)

    return [g for g in clusters.values() if len(g) > 1]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _group_panel(group: list, author_lookup: dict) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")  # number
    table.add_column(style="dim")                          # id
    table.add_column()                                     # year
    table.add_column()                                     # authors
    table.add_column()                                     # title / source

    for i, row in enumerate(group):
        authors = author_lookup.get(row["id"], [])
        names = ", ".join(a["family_name"] for a in authors[:3])
        if len(authors) > 3:
            names += f" +{len(authors) - 3}"
        source = row["doi"] or row["arxiv_id"] or row["isbn"] or ""
        title_line = Text(row["title"])
        if source:
            title_line.append(f"\n{source}", style="dim")
        table.add_row(
            str(i + 1),
            f"id={row['id']}",
            str(row["year"] or "?"),
            names or "[unknown]",
            title_line,
        )

    return Panel(table, border_style="yellow")


# ---------------------------------------------------------------------------
# Removal (mirrors rm_cmd logic)
# ---------------------------------------------------------------------------


def _remove_paper(paper, *, delete_file: bool) -> None:
    paper_id = int(paper["id"])
    file_path = config.library_dir() / paper["file_path"]

    with db.connect() as conn:
        from books.index.fts import delete_paper as fts_delete

        fts_delete(conn, paper_id)
        db.delete_paper(conn, paper_id)

    try:
        from books.index.indexer import delete_paper_chunks

        delete_paper_chunks(paper_id)
    except Exception as e:
        console.print(f"[yellow]could not remove Chroma chunks for id={paper_id}:[/yellow] {e}")

    console.print(f"  removed [bold]{paper['title']}[/bold] (id={paper_id})")
    if delete_file:
        try:
            file_path.unlink()
            console.print(f"  deleted [dim]{file_path}[/dim]")
        except FileNotFoundError:
            console.print(f"  [yellow]file already missing:[/yellow] {file_path}")
    else:
        console.print(f"  kept file [dim]{file_path}[/dim]")
