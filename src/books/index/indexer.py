"""Indexing orchestration — extract → chunk → embed → upsert.

Acts as the boundary between the ingestion-time workflow (importer) and the
query-time workflow (search). The embedder is cached as a module-level
singleton because loading the model is the dominant cost when batch-importing.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from books import config

if TYPE_CHECKING:
    # Type-only import; runtime import happens inside `get_embedder` so torch
    # stays out of import paths that don't need it.
    from books.index.embedder import Embedder

_EMBEDDER: "Embedder | None" = None


def get_embedder() -> "Embedder":
    """Return the process-wide embedder, constructing it on first call.

    The first call is expensive (loads the model from disk / HuggingFace).
    Subsequent calls are O(1) and reuse the loaded model. This singleton is
    intentional: we want batch imports to share the loaded weights.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        from books.index.embedder import make_embedder

        _EMBEDDER = make_embedder(config.embedder_model(), config.embedder_device())
    return _EMBEDDER


def index_paper(
    *,
    paper_id: int,
    pdf_path: Path,
    title: str,
    doi: str | None,
) -> int:
    """Run the full pipeline for a single paper. Returns chunk count.

    Steps:

    1. Extract page-mapped text from the PDF.
    2. Chunk it (token-budgeted, with overlap).
    3. Embed every chunk.
    4. Upsert into the persistent Chroma collection.

    A PDF that produces no extractable text (image-only / scanned) is a
    no-op and returns 0.
    """
    # Lazy imports keep `book ls` etc. from pulling in torch + chromadb.
    from books.index.chroma import ChromaIndex
    from books.index.chunker import chunk_pages
    from books.index.extract import extract_text

    pages = extract_text(pdf_path)
    chunks = chunk_pages(
        pages,
        chunk_tokens=config.chunk_tokens(),
        overlap_tokens=config.chunk_overlap(),
    )
    if not chunks:
        return 0

    embedder = get_embedder()
    embeddings = embedder.embed([c.text for c in chunks])

    index = ChromaIndex(config.chroma_dir())
    index.upsert(
        paper_id=paper_id,
        chunks=chunks,
        embeddings=embeddings,
        title=title,
        doi=doi,
    )
    return len(chunks)


def delete_paper_chunks(paper_id: int) -> None:
    """Remove every Chroma chunk for ``paper_id``. Safe to call on absent papers."""
    from books.index.chroma import ChromaIndex

    index = ChromaIndex(config.chroma_dir())
    index.delete_paper(paper_id)
