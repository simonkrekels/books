"""Tests for the shared retrieval pipeline (books.retrieval).

The backends are driven with a fake Chroma index and a real FTS-backed tmp_db;
``retrieve`` is exercised by monkeypatching the lazily-imported embedder and
ChromaIndex so no model loads or network calls occur.
"""

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from books import db, retrieval
from books.metadata.models import Author, PaperMatch


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    p = tmp_path / "library.db"
    db.init_db(p)
    return p


@dataclass
class _Chunk:
    chunk_index: int
    page_number: int
    text: str


def _paper(tmp_db: Path, doi: str = "10.1/test") -> int:
    match = PaperMatch(
        source="crossref",
        doi=doi,
        title="Test Paper",
        authors=[Author(family="Smith", given="Alice")],
        year=2024,
    )
    with db.connect(tmp_db) as conn:
        return db.insert_paper(conn, match, file_path=f"{doi}.pdf", source_pdf_hash=doi)


class _FakeIndex:
    """Minimal stand-in for ChromaIndex returning canned cosine hits."""

    def __init__(self, hits: list[tuple[int, int, int, str, float]]):
        # hits: (paper_id, chunk_index, page, text, distance)
        self._hits = hits

    def count(self) -> int:
        return len(self._hits)

    def query(self, *, query_embedding, n_results, where=None):
        hits = self._hits[:n_results]
        return {
            "ids": [[f"{p}:{c}" for p, c, _, _, _ in hits]],
            "documents": [[t for _, _, _, t, _ in hits]],
            "metadatas": [[
                {"paper_id": p, "chunk_index": c, "page": pg}
                for p, c, pg, _, _ in hits
            ]],
            "distances": [[d for _, _, _, _, d in hits]],
        }


def test_cosine_search_maps_distance_to_score():
    index = _FakeIndex([(1, 0, 3, "alpha", 0.1), (2, 1, 5, "beta", 0.4)])
    results = retrieval.cosine_search(index, [0.0], n=5)
    assert [r.paper_id for r in results] == [1, 2]
    assert results[0].text == "alpha"
    assert results[0].page == 3
    assert results[0].score == pytest.approx(0.9)


def test_hybrid_search_fuses_cosine_and_bm25(tmp_db: Path):
    from books.index import fts as fts_index

    paper_id = _paper(tmp_db)
    chunks = [
        _Chunk(0, 1, "entropy production in statistical mechanics"),
        _Chunk(1, 2, "Boltzmann equation and kinetic theory"),
    ]
    with db.connect(tmp_db) as conn:
        fts_index.upsert_paper(conn, paper_id, chunks)
        # Cosine ranks chunk 1 first; BM25 on "entropy" ranks chunk 0 first.
        index = _FakeIndex([
            (paper_id, 1, 2, "Boltzmann equation and kinetic theory", 0.2),
            (paper_id, 0, 1, "entropy production in statistical mechanics", 0.3),
        ])
        results = retrieval.hybrid_search(conn, index, [0.0], "entropy", n=5)

    ids = {(r.paper_id, r.chunk_index) for r in results}
    assert (paper_id, 0) in ids  # the BM25-matched chunk is present
    # Text + page are reconstructed from the chunks table.
    chunk0 = next(r for r in results if r.chunk_index == 0)
    assert "entropy" in chunk0.text
    assert chunk0.page == 1


def test_retrieve_raises_when_index_empty(tmp_db: Path, monkeypatch):
    _install_fakes(monkeypatch, _FakeIndex([]))
    with db.connect(tmp_db) as conn:
        with pytest.raises(retrieval.RetrievalUnavailable):
            retrieval.retrieve(conn, "anything", 5)


def test_retrieve_uses_cosine_when_fts_empty(tmp_db: Path, monkeypatch):
    index = _FakeIndex([(1, 0, 1, "some chunk text", 0.25)])
    _install_fakes(monkeypatch, index)
    with db.connect(tmp_db) as conn:  # chunks table empty → no hybrid
        results = retrieval.retrieve(conn, "query", 5)
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.75)


def _install_fakes(monkeypatch, index) -> None:
    """Patch the lazy imports inside retrieve() with fakes."""
    fake_embedder = types.SimpleNamespace(embed=lambda texts: [[0.0, 0.0]])
    indexer_mod = types.ModuleType("books.index.indexer")
    indexer_mod.get_embedder = lambda: fake_embedder
    monkeypatch.setitem(sys.modules, "books.index.indexer", indexer_mod)

    chroma_mod = types.ModuleType("books.index.chroma")
    chroma_mod.ChromaIndex = lambda path: index
    monkeypatch.setitem(sys.modules, "books.index.chroma", chroma_mod)

    monkeypatch.setattr(retrieval.config, "query_prompt", lambda: "")
    monkeypatch.setattr(retrieval.config, "chroma_dir", lambda: Path("/tmp/x"))
    monkeypatch.setattr(retrieval.config, "hybrid_search", lambda: True)
