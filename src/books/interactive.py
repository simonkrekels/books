"""Rich-based interactive prompts used by the importer.

Entry points and their return codes:

* :func:`confirm_match`         — show a candidate; returns ``apply``, ``skip``,
                                   ``manual``, ``quit``.
* :func:`no_match_prompt`       — no metadata found; returns ``retry``,
                                   ``use_pdf``, ``manual_entry``, ``manual_doi``,
                                   ``skip``, ``quit``.
* :func:`manual_doi_lookup`     — read a DOI and resolve via Crossref.
* :func:`manual_entry_form`     — prompt for title / authors / year → PaperMatch.
* :func:`build_match_from_pdf_meta` — build a minimal PaperMatch from sniffed PDF
                                       embedded metadata.
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from books.metadata import crossref
from books.metadata.models import Author, PaperMatch
from books.metadata.pdf_meta import SniffResult

console = Console()


def confirm_match(pdf: Path, match: PaperMatch) -> str:
    """Display ``match`` for ``pdf`` and ask the user how to proceed.

    Returns one of: ``apply``, ``skip``, ``manual``, ``quit``.
    """
    console.print(_match_panel(pdf, match))
    choice = Prompt.ask(
        "[bold cyan][A]pply / [S]kip / [M]anual DOI / [Q]uit[/bold cyan]",
        choices=["a", "s", "m", "q"],
        default="a",
        show_choices=False,
    )
    return {"a": "apply", "s": "skip", "m": "manual", "q": "quit"}[choice]


def no_match_prompt(pdf: Path, sniff: SniffResult) -> str:
    """Prompt when no metadata could be fetched for ``pdf``.

    Returns one of: ``retry``, ``use_pdf``, ``manual_entry``, ``manual_doi``,
    ``skip``, ``quit``.
    """
    console.print(f"[yellow]no metadata match for[/yellow] {pdf.name}")
    if sniff.doi:
        console.print(f"  sniffed DOI (lookup failed): {sniff.doi}")
    if sniff.arxiv_id:
        console.print(f"  sniffed arXiv: {sniff.arxiv_id}")
    title = sniff.pdf_metadata.get("title", "").strip()
    if title:
        console.print(f"  pdf title: {title}")
    choice = Prompt.ask(
        "[bold cyan][R]etry / [U]se PDF metadata / [M]anual entry / [E]nter DOI / [S]kip / [Q]uit[/bold cyan]",
        choices=["r", "u", "m", "e", "s", "q"],
        default="s",
        show_choices=False,
    )
    return {
        "r": "retry",
        "u": "use_pdf",
        "m": "manual_entry",
        "e": "manual_doi",
        "s": "skip",
        "q": "quit",
    }[choice]


def manual_doi_lookup(default: str | None = None) -> PaperMatch | None:
    """Ask the user for a DOI and resolve it via Crossref.

    Returns ``None`` if the user submits an empty input, or if the lookup
    raises (e.g. network failure). ``default`` pre-populates the prompt
    with a previously sniffed DOI when one is available.
    """
    doi = Prompt.ask(
        "DOI to look up (empty to cancel)",
        default=default or "",
        show_default=bool(default),
    ).strip()
    if not doi:
        return None
    try:
        return crossref.lookup(doi)
    except Exception as e:
        console.print(f"[red]lookup failed:[/red] {e}")
        return None


def manual_entry_form() -> PaperMatch | None:
    """Interactively prompt for title, authors, and year.

    Author input format: ``Family, Given; Family, Given`` (semicolon-separated).
    A bare name without a comma is treated as a family-only author.
    Returns ``None`` if the user leaves title blank.
    """
    title = Prompt.ask("Title (empty to cancel)").strip()
    if not title:
        return None

    author_raw = Prompt.ask("Author(s) [Family, Given; ...] (empty to skip)", default="").strip()
    authors: list[Author] = []
    for part in author_raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if "," in part:
            family, _, given = part.partition(",")
            authors.append(Author(family=family.strip(), given=given.strip() or None))
        else:
            authors.append(Author(family=part))

    year_raw = Prompt.ask("Year (empty to skip)", default="").strip()
    year: int | None = None
    if year_raw.isdigit():
        year = int(year_raw)

    return PaperMatch(source="manual", title=title, authors=authors, year=year)


def build_match_from_pdf_meta(sniff: SniffResult) -> PaperMatch | None:
    """Build a minimal PaperMatch from embedded PDF metadata.

    Returns ``None`` when the PDF title field is blank (unusable for import).
    The ``author`` field from PDF metadata is treated as a single family-name
    entry unless it contains a comma, in which case it is split as
    ``Family, Given``.
    """
    title = sniff.pdf_metadata.get("title", "").strip()
    if not title:
        return None

    authors: list[Author] = []
    author_raw = sniff.pdf_metadata.get("author", "").strip()
    if author_raw:
        if "," in author_raw:
            family, _, given = author_raw.partition(",")
            authors.append(Author(family=family.strip(), given=given.strip() or None))
        else:
            authors.append(Author(family=author_raw))

    return PaperMatch(
        source="pdf_meta",
        title=title,
        authors=authors,
        doi=sniff.doi,
        arxiv_id=sniff.arxiv_id,
        isbn=sniff.isbn,
    )


def _match_panel(pdf: Path, match: PaperMatch) -> Panel:
    """Build the rich Panel that shows a candidate match summary."""
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("source", match.source)
    table.add_row("title", match.title)
    # Cap displayed authors at 5 — long author lists make the panel unreadable.
    table.add_row(
        "authors",
        ", ".join(
            f"{a.given or ''} {a.family}".strip() for a in match.authors[:5]
        )
        + (f" +{len(match.authors) - 5}" if len(match.authors) > 5 else ""),
    )
    table.add_row("year", str(match.year or ""))
    table.add_row("journal", match.journal or "")
    table.add_row("doi", match.doi or "")
    table.add_row("arxiv", match.arxiv_id or "")
    return Panel(table, title=f"match for {pdf.name}", border_style="green")
