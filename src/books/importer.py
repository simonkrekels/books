"""Import workflow — orchestrates the full pipeline for one or more PDFs.

For each PDF:

1. Hash the file (sha256) and skip if already imported.
2. Sniff DOI / arXiv ID with weighted scoring against PDF metadata + page 1
   + later pages (see :mod:`books.metadata.pdf_meta`).
3. Look up canonical metadata via Crossref (DOI) or arXiv (ID), preferring
   Crossref when both are available.
4. In interactive mode, ask the user to confirm; in ``--quiet`` mode, accept
   the top match silently.
5. Render the templated path, copy/move/symlink the PDF in place, and
   commit the SQLite row.
6. Post-commit: chunk + embed + upsert into Chroma. Failures here flag the
   paper ``needs_reindex`` rather than rolling back the SQLite import.
"""

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from books import config, db, interactive, paths
from books.metadata import arxiv, crossref, openlibrary
from books.metadata.models import PaperMatch
from books.metadata.pdf_meta import sniff_pdf

console = Console()


@dataclass
class ImportOutcome:
    """Result of attempting to import a single PDF.

    ``status`` is one of: ``imported``, ``skipped``, ``duplicate``,
    ``failed``, ``quit``. ``message`` is a short human-readable note;
    ``paper_id`` is set when a row exists in the DB after the call.
    """

    path: Path
    status: str
    message: str = ""
    paper_id: int | None = None


@dataclass
class ImportSummary:
    """Aggregate of all per-file outcomes for a single import invocation."""

    outcomes: list[ImportOutcome] = field(default_factory=list)
    quit_early: bool = False  # True if the user chose [Q]uit mid-batch

    def record(self, outcome: ImportOutcome) -> None:
        self.outcomes.append(outcome)

    def by_status(self, status: str) -> list[ImportOutcome]:
        """Filter outcomes by status string."""
        return [o for o in self.outcomes if o.status == status]

    def render(self) -> None:
        """Print a coloured summary table to the console."""
        if not self.outcomes:
            return
        table = Table(title="import summary", show_header=True, header_style="bold")
        table.add_column("status")
        table.add_column("file")
        table.add_column("note", style="dim")
        for o in self.outcomes:
            colour = {
                "imported": "green",
                "skipped": "yellow",
                "duplicate": "yellow",
                "failed": "red",
                "quit": "dim",
            }.get(o.status, "")
            table.add_row(
                f"[{colour}]{o.status}[/]" if colour else o.status,
                o.path.name,
                o.message,
            )
        console.print(table)


def import_paths(
    paths_in: list[Path],
    *,
    quiet: bool = False,
    mode_override: str | None = None,
) -> ImportSummary:
    """Import every PDF reachable from ``paths_in``.

    Each entry may be a PDF file or a directory (recursed for ``*.pdf``).
    In ``quiet`` mode no prompts appear: top matches are auto-applied; PDFs
    without a confident DOI/arXiv ID are skipped. ``mode_override`` (one of
    ``copy``/``move``/``symlink``) takes precedence over ``config.import_mode``
    for this invocation only.
    """
    summary = ImportSummary()
    pdfs = _expand_paths(paths_in)
    if not pdfs:
        console.print("[yellow]no PDFs found[/yellow]")
        return summary
    mode = mode_override or config.import_mode()
    for pdf in pdfs:
        outcome = _import_one(pdf, quiet=quiet, mode=mode)
        summary.record(outcome)
        # Honour the user's [Q]uit choice — stop the whole batch.
        if outcome.status == "quit":
            summary.quit_early = True
            break
    summary.render()
    return summary


def _expand_paths(paths_in: list[Path]) -> list[Path]:
    """Flatten a mix of files/directories into a deterministic list of PDFs."""
    out: list[Path] = []
    for p in paths_in:
        if p.is_dir():
            out.extend(sorted(p.rglob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            out.append(p)
    return out


def _import_one(pdf: Path, *, quiet: bool, mode: str) -> ImportOutcome:
    """Process a single PDF end-to-end. Returns the outcome dataclass.

    The function is intentionally linear (no early returns from sub-helpers)
    so the control flow — sniff, lookup, confirm, place, commit, index — is
    easy to read top-to-bottom.
    """
    # 1. Dedup against previous imports by file hash.
    sha = _sha256(pdf)
    with db.connect() as conn:
        existing = db.find_by_hash(conn, sha)
    if existing is not None:
        return ImportOutcome(
            pdf,
            "duplicate",
            f"already imported as id={existing['id']}",
            paper_id=int(existing["id"]),
        )

    console.print(f"\n[bold]importing[/bold] {pdf.name}")

    # 2-3. Sniff identifiers and look up canonical metadata.
    sniff = sniff_pdf(pdf)
    match = _lookup(sniff.doi, sniff.arxiv_id, sniff.isbn)

    # 4. Either prompt the user, or short-circuit (quiet mode).
    if match is None:
        if quiet:
            return ImportOutcome(pdf, "skipped", "no DOI/arXiv match found")
        action = interactive.no_match_prompt(pdf, sniff)
        if action == "quit":
            return ImportOutcome(pdf, "quit")
        if action == "skip":
            return ImportOutcome(pdf, "skipped")
        match = interactive.manual_doi_lookup(sniff.doi)
        if match is None:
            return ImportOutcome(pdf, "skipped", "manual lookup abandoned")
    elif not quiet:
        decision = interactive.confirm_match(pdf, match)
        if decision == "quit":
            return ImportOutcome(pdf, "quit")
        if decision == "skip":
            return ImportOutcome(pdf, "skipped")
        if decision == "manual":
            replacement = interactive.manual_doi_lookup(match.doi)
            if replacement is None:
                return ImportOutcome(pdf, "skipped", "manual lookup abandoned")
            match = replacement

    # 5. Place file on disk according to the configured template + mode.
    rel_path = paths.render_template(
        config.path_template(),
        paper=match.model_dump(),
    )
    dest = config.library_dir() / rel_path

    if dest.exists():
        return ImportOutcome(pdf, "failed", f"target file exists: {dest}")

    try:
        paths.place_pdf(pdf, dest, mode)
    except OSError as e:
        return ImportOutcome(pdf, "failed", f"could not place file: {e}")

    # 6. Commit the DB row. On IntegrityError (duplicate DOI/arXiv ID), undo
    # the file placement so the user can fix and retry. We only undo for
    # copy/symlink — undoing a "move" would require restoring the original.
    try:
        with db.connect() as conn:
            paper_id = db.insert_paper(
                conn,
                match,
                file_path=rel_path,
                source_pdf_hash=sha,
            )
    except sqlite3.IntegrityError as e:
        if mode in ("copy", "symlink"):
            try:
                dest.unlink()
            except OSError:
                pass
        return ImportOutcome(pdf, "failed", f"db insert: {e}")

    # 7. Post-commit: index for vector search. Failures here are non-fatal.
    _index_post_commit(paper_id, dest, match)
    return ImportOutcome(pdf, "imported", f"id={paper_id} -> {rel_path}", paper_id=paper_id)


def _index_post_commit(paper_id: int, pdf_path: Path, match: PaperMatch) -> None:
    """Chunk + embed + upsert the just-imported paper into Chroma.

    Failures are recoverable: the row is flagged ``needs_reindex=1`` so the
    user can retry later via ``book reindex`` without re-importing.
    """
    # Lazy import: the indexer pulls in torch / sentence-transformers, which
    # we don't want loaded for unrelated commands like ``book ls``.
    from books.index.indexer import index_paper

    try:
        n = index_paper(
            paper_id=paper_id,
            pdf_path=pdf_path,
            title=match.title,
            doi=match.doi,
        )
        console.print(f"[dim]indexed {n} chunks[/dim]")
    except Exception as e:
        console.print(f"[yellow]indexing failed:[/yellow] {e}")
        try:
            with db.connect() as conn:
                db.set_needs_reindex(conn, paper_id, True)
        except Exception:
            # If even the reindex flag can't be set, swallow — the user can
            # still rebuild from PDFs via `book reindex --all`.
            pass


def _lookup(
    doi: str | None, arxiv_id: str | None, isbn: str | None
) -> PaperMatch | None:
    """Try the available metadata sources in priority order.

    Crossref (DOI) is preferred because it carries the richest published
    metadata. arXiv is the natural fallback for preprints. Open Library
    (ISBN) covers books that don't show up in either, and is also useful
    when the DOI lookup 404s on a book DOI.
    """
    if doi:
        try:
            match = crossref.lookup(doi)
            if match is not None:
                # Carry the sniffed ISBN through so it gets persisted even
                # when Crossref didn't volunteer it.
                if match.isbn is None and isbn:
                    match.isbn = isbn
                return match
        except Exception as e:
            console.print(f"[yellow]crossref lookup failed:[/yellow] {e}")
    if arxiv_id:
        try:
            match = arxiv.lookup(arxiv_id)
            if match is not None:
                return match
        except Exception as e:
            console.print(f"[yellow]arxiv lookup failed:[/yellow] {e}")
    if isbn:
        try:
            return openlibrary.lookup(isbn)
        except Exception as e:
            console.print(f"[yellow]open library lookup failed:[/yellow] {e}")
    return None


def _sha256(path: Path) -> str:
    """Stream the file and return its hex sha256."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        # 64 KiB chunks: large enough to amortize the syscall, small enough
        # to keep memory flat for huge PDFs.
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
