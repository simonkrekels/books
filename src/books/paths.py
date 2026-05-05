"""On-disk path utilities for the library.

Two responsibilities:

* :func:`render_template` — turn a config path template (e.g.
  ``"{author_last}/{year}/{title_slug}.pdf"``) into a concrete relative path
  using fields from a paper dict.
* :func:`place_pdf` — copy/move/symlink a source PDF to its target.

The slug helper is shared because every templated key needs the same
ASCII-safe, hyphen-separated form.
"""

import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any


def slugify(text: str, max_len: int = 80) -> str:
    """Return a filesystem-safe, ASCII-only, lowercase, hyphenated slug.

    Empty input or input that contains no alphanumerics yields ``"untitled"``.
    Long results are truncated at ``max_len`` characters.
    """
    if not text:
        return "untitled"
    # NFKD splits accented chars into base + combining marks; the ASCII-encode
    # then drops the combining marks ("Schrödinger" -> "Schrodinger").
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Common in titles; preserves intent better than dropping the ampersand.
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text or "untitled"


def render_template(template: str, *, paper: dict[str, Any]) -> str:
    """Render a path template using fields derived from a paper dict.

    Available substitution keys (always present, missing values fall back to
    ``"unknown"``): ``author_last``, ``author_last_first``, ``year``,
    ``title_slug``, ``doi_slug``, ``journal_slug``.
    """
    return template.format(**_keys(paper))


def _keys(paper: dict[str, Any]) -> dict[str, str]:
    """Build the substitution dict consumed by :func:`render_template`."""
    authors = paper.get("authors") or []
    first = authors[0] if authors else {}
    family = first.get("family") or "unknown"
    given = first.get("given") or ""
    return {
        "author_last": slugify(family),
        "author_last_first": slugify(f"{family} {given}".strip()),
        "year": str(paper.get("year") or "unknown"),
        "title_slug": slugify(paper.get("title") or "untitled"),
        # DOIs always contain a "/" — replace with "_" before slugifying so
        # the structure (registrant / suffix) is preserved as one token.
        "doi_slug": slugify((paper.get("doi") or "").replace("/", "_")),
        "journal_slug": slugify(paper.get("journal") or "unknown"),
    }


def place_pdf(src: Path, dest: Path, mode: str) -> None:
    """Copy, move, or symlink ``src`` to ``dest`` according to ``mode``.

    Creates parent directories as needed. Raises ``FileExistsError`` if the
    target already exists — callers decide how to recover.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"target already exists: {dest}")
    if mode == "copy":
        shutil.copy2(src, dest)
    elif mode == "move":
        shutil.move(str(src), str(dest))
    elif mode == "symlink":
        # Resolve the source so the symlink points at an absolute path that
        # remains valid even if the source was relative to a different cwd.
        dest.symlink_to(src.resolve())
    else:
        raise ValueError(f"unknown mode: {mode}")
