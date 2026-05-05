"""``book import`` — add PDFs to the library."""

from pathlib import Path

import typer

from books.importer import import_paths


def run(
    paths: list[Path] = typer.Argument(..., exists=True, help="PDF file(s) or directory."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Auto-accept the top match."),
) -> None:
    """Import PDF(s) into the library.

    Each path may be a single PDF or a directory (recursed for ``*.pdf``).
    Without ``--quiet``, the importer prompts to confirm each match. Exit
    code is 1 if any file failed to import.
    """
    summary = import_paths(paths, quiet=quiet)
    if summary.by_status("failed"):
        raise typer.Exit(code=1)
