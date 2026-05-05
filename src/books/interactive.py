"""Rich-based interactive prompts used by the importer.

Three entry points, each returns a small string code so the importer can
branch on the user's choice without leaking ``rich`` into the import logic:

* :func:`confirm_match`   — show a candidate and accept / skip / replace / quit.
* :func:`no_match_prompt` — when nothing was found; offer manual DOI / skip / quit.
* :func:`manual_doi_lookup` — read a DOI from stdin and look it up via Crossref.
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from books.metadata import crossref
from books.metadata.models import PaperMatch
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

    Shows whatever was sniffed (DOI / arXiv ID / PDF title) so the user can
    decide whether to enter a DOI manually. Returns one of: ``manual``,
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
        "[bold cyan][E]nter DOI / [S]kip / [Q]uit[/bold cyan]",
        choices=["e", "s", "q"],
        default="s",
        show_choices=False,
    )
    return {"e": "manual", "s": "skip", "q": "quit"}[choice]


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
