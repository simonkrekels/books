"""``book import`` — add PDFs to the library."""

from pathlib import Path

import typer

from books.importer import import_paths


def run(
    paths: list[Path] = typer.Argument(..., exists=True, help="PDF file(s) or directory."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Auto-accept the top match."),
    copy: bool = typer.Option(
        False,
        "--copy",
        help="Copy the source PDF instead of moving it (default is move).",
    ),
) -> None:
    """Import PDF(s) into the library.

    Each path may be a single PDF or a directory (recursed for ``*.pdf``).
    Without ``--quiet``, the importer prompts to confirm each match. By
    default the source PDF is moved into the library (the original is gone
    after a successful import); use ``--copy`` to keep the source. Exit
    code is 1 if any file failed to import.
    """
    summary = import_paths(paths, quiet=quiet, mode_override="copy" if copy else None)
    if summary.by_status("failed"):
        raise typer.Exit(code=1)
