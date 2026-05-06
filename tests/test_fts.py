"""Tests for the FTS5 BM25 index (books.index.fts)."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from books import db
from books.index import fts as fts_index
from books.index.fts import _build_fts_query, rrf_fuse as _rrf_fuse
from books.metadata.models import Author, PaperMatch


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    p = tmp_path / "library.db"
    db.init_db(p)
    return p


def _paper(tmp_db: Path, doi: str = "10.1/test") -> int:
    """Insert a minimal paper and return its id."""
    match = PaperMatch(
        source="crossref",
        doi=doi,
        title="Test Paper",
        authors=[Author(family="Smith", given="Alice")],
        year=2024,
    )
    with db.connect(tmp_db) as conn:
        return db.insert_paper(conn, match, file_path="smith/2024/test.pdf", source_pdf_hash=doi)


@dataclass
class _Chunk:
    chunk_index: int
    page_number: int
    text: str


# ---------------------------------------------------------------------------
# upsert_paper / search
# ---------------------------------------------------------------------------


def test_upsert_and_search(tmp_db: Path):
    paper_id = _paper(tmp_db)
    chunks = [
        _Chunk(0, 1, "entropy production in statistical mechanics"),
        _Chunk(1, 2, "Boltzmann equation and kinetic theory"),
    ]
    with db.connect(tmp_db) as conn:
        fts_index.upsert_paper(conn, paper_id, chunks)
        results = fts_index.search(conn, "entropy", 5)

    assert len(results) >= 1
    chunk_id, score = results[0]
    assert chunk_id == f"{paper_id}:0"
    assert score > 0


def test_search_matches_stemmed_term(tmp_db: Path):
    paper_id = _paper(tmp_db)
    # "running" and "run" are the canonical Porter stemmer example — both
    # reduce to "run", so searching for one should match the other.
    chunks = [_Chunk(0, 1, "running simulations in parallel")]
    with db.connect(tmp_db) as conn:
        fts_index.upsert_paper(conn, paper_id, chunks)
        results = fts_index.search(conn, "run", 5)
    assert any(cid == f"{paper_id}:0" for cid, _ in results)


def test_search_returns_empty_for_unknown_term(tmp_db: Path):
    paper_id = _paper(tmp_db)
    chunks = [_Chunk(0, 1, "statistical thermodynamics")]
    with db.connect(tmp_db) as conn:
        fts_index.upsert_paper(conn, paper_id, chunks)
        results = fts_index.search(conn, "xyzzy_nonexistent_term", 5)
    assert results == []


def test_search_returns_empty_on_blank_query(tmp_db: Path):
    with db.connect(tmp_db) as conn:
        results = fts_index.search(conn, "", 5)
    assert results == []


def test_search_scores_order(tmp_db: Path):
    paper_id = _paper(tmp_db)
    chunks = [
        _Chunk(0, 1, "entropy entropy entropy thermodynamics"),  # highly relevant
        _Chunk(1, 2, "something completely unrelated"),
    ]
    with db.connect(tmp_db) as conn:
        fts_index.upsert_paper(conn, paper_id, chunks)
        results = fts_index.search(conn, "entropy", 5)
    # chunk 0 should outrank chunk 1
    ids = [cid for cid, _ in results]
    assert f"{paper_id}:0" in ids
    if len(ids) > 1:
        assert ids.index(f"{paper_id}:0") < ids.index(f"{paper_id}:1")


# ---------------------------------------------------------------------------
# delete_paper
# ---------------------------------------------------------------------------


def test_delete_paper_removes_chunks(tmp_db: Path):
    paper_id = _paper(tmp_db)
    chunks = [_Chunk(0, 1, "unique phrase xyzzy_marker")]
    with db.connect(tmp_db) as conn:
        fts_index.upsert_paper(conn, paper_id, chunks)
        fts_index.delete_paper(conn, paper_id)
        results = fts_index.search(conn, "xyzzy_marker", 5)
        count = conn.execute("SELECT COUNT(*) FROM chunks WHERE paper_id = ?", (paper_id,)).fetchone()[0]
    assert results == []
    assert count == 0


def test_upsert_replaces_existing_chunks(tmp_db: Path):
    paper_id = _paper(tmp_db)
    old_chunks = [_Chunk(0, 1, "old content about thermodynamics")]
    new_chunks = [_Chunk(0, 1, "new content about quantum mechanics")]
    with db.connect(tmp_db) as conn:
        fts_index.upsert_paper(conn, paper_id, old_chunks)
        fts_index.upsert_paper(conn, paper_id, new_chunks)
        old_results = fts_index.search(conn, "thermodynamics", 5)
        new_results = fts_index.search(conn, "quantum", 5)
        count = conn.execute("SELECT COUNT(*) FROM chunks WHERE paper_id = ?", (paper_id,)).fetchone()[0]
    assert old_results == []
    assert len(new_results) == 1
    assert count == 1


# ---------------------------------------------------------------------------
# rebuild
# ---------------------------------------------------------------------------


def test_rebuild_restores_fts(tmp_db: Path):
    paper_id = _paper(tmp_db)
    chunks = [_Chunk(0, 1, "renormalization group theory")]
    with db.connect(tmp_db) as conn:
        fts_index.upsert_paper(conn, paper_id, chunks)
        # Simulate FTS corruption by wiping the FTS table directly.
        conn.execute("DELETE FROM chunks_fts")
        assert fts_index.search(conn, "renormalization", 5) == []
        # Rebuild should recover from the chunks table.
        fts_index.rebuild(conn)
        results = fts_index.search(conn, "renormalization", 5)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# has_content
# ---------------------------------------------------------------------------


def test_has_content_false_when_empty(tmp_db: Path):
    with db.connect(tmp_db) as conn:
        assert fts_index.has_content(conn) is False


def test_has_content_true_after_upsert(tmp_db: Path):
    paper_id = _paper(tmp_db)
    with db.connect(tmp_db) as conn:
        fts_index.upsert_paper(conn, paper_id, [_Chunk(0, 1, "some text")])
        assert fts_index.has_content(conn) is True


# ---------------------------------------------------------------------------
# _rrf_fuse
# ---------------------------------------------------------------------------


def test_rrf_fuse_single_list():
    ranked = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    result = _rrf_fuse([ranked])
    ids = [r[0] for r in result]
    assert ids == ["a", "b", "c"]


def test_rrf_fuse_boosts_overlap():
    list_a = [("x", 0.9), ("y", 0.8), ("z", 0.1)]
    list_b = [("y", 0.9), ("z", 0.8), ("w", 0.7)]
    result = _rrf_fuse([list_a, list_b])
    ids = [r[0] for r in result]
    # "y" appears at rank 1 in both lists → should have highest fused score
    assert ids[0] == "y"


def test_rrf_fuse_empty_lists():
    assert _rrf_fuse([]) == []
    assert _rrf_fuse([[]]) == []


# ---------------------------------------------------------------------------
# _build_fts_query
# ---------------------------------------------------------------------------


def test_build_fts_query_single_word():
    assert _build_fts_query("entropy") == '"entropy"'


def test_build_fts_query_multiple_words():
    assert _build_fts_query("entropy production") == '"entropy" "production"'


def test_build_fts_query_empty():
    assert _build_fts_query("") == ""
    assert _build_fts_query("   ") == ""


def test_build_fts_query_escapes_quotes():
    assert _build_fts_query('say "hello"') == '"say" """hello"""'
