"""PDF text extraction with page mapping (PyMuPDF)."""

from pathlib import Path

import fitz  # type: ignore[import-untyped]


def extract_text(path: Path) -> list[tuple[int, str]]:
    """Read a PDF and return ``[(page_number, text), ...]`` (1-indexed pages).

    Pages with no extractable text are skipped — this includes scanned-only
    pages and blank pages. The chunker downstream relies on the page numbers
    to label each chunk with its page-of-origin for the search UI.
    """
    out: list[tuple[int, str]] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if text and text.strip():
                out.append((i + 1, text))
    return out
