"""Thin wrapper around Chroma's PersistentClient for our single ``papers`` collection.

We disable Chroma's auto-embedding and embed externally (see
:mod:`books.index.embedder`) so the embedder can be swapped without touching
collection state. Chunks are addressed by the synthetic ID
``"{paper_id}:{chunk_index}"``, which makes per-paper deletes cheap.
"""

from pathlib import Path
from typing import Any

import chromadb

from books.index.chunker import Chunk

COLLECTION = "papers"


class ChromaIndex:
    """Persistent vector index for paper chunks.

    Each chunk is stored with metadata ``{paper_id, page, chunk_index, doi,
    title}`` so search hits can be rendered without joining back to SQLite
    when needed (the ``search`` command does join, for fresher data).
    """

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        # `embedding_function=None` opts out of Chroma's built-in embedder —
        # we always supply pre-computed vectors via `embeddings=...`.
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=None,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        *,
        paper_id: int,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        title: str,
        doi: str | None,
    ) -> None:
        """Insert or update all chunks for one paper.

        Chunk IDs are deterministic, so re-running upsert with re-chunked
        text replaces the previous version cleanly.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        ids = [f"{paper_id}:{c.chunk_index}" for c in chunks]
        metas: list[dict[str, Any]] = [
            {
                "paper_id": int(paper_id),
                "page": int(c.page_number),
                "chunk_index": int(c.chunk_index),
                "doi": doi or "",
                "title": title,
            }
            for c in chunks
        ]
        self._collection.upsert(
            ids=ids,
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=metas,
        )

    def delete_paper(self, paper_id: int) -> None:
        """Remove every chunk belonging to ``paper_id`` from the index."""
        self._collection.delete(where={"paper_id": int(paper_id)})

    def query(
        self,
        *,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Run a single-vector nearest-neighbour search.

        Returns Chroma's raw response shape: ``{ids, documents, metadatas,
        distances, embeddings}``, each a list of lists (one entry per query
        vector — we always pass exactly one).
        """
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

    def count(self) -> int:
        """Number of chunks currently stored in the index."""
        return int(self._collection.count())
