"""Tests for duplicate-detection logic in books.commands.dupes_cmd."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from books import db
from books.commands.dupes_cmd import _find_duplicates, _jaccard, _normalize_title
from books.metadata.models import Author, PaperMatch


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


def test_normalize_title_lowercases():
    assert _normalize_title("Entropy Production") == "entropy production"


def test_normalize_title_strips_punctuation():
    assert _normalize_title("Hello, World!") == "hello world"


def test_normalize_title_collapses_whitespace():
    assert _normalize_title("  a   b  ") == "a b"


def test_normalize_title_unicode():
    # Accented characters — decomposition shouldn't break words
    assert "renormalization" in _normalize_title("Renormalization")


def test_jaccard_identical():
    assert _jaccard("a b c", "a b c") == 1.0


def test_jaccard_disjoint():
    assert _jaccard("a b", "c d") == 0.0


def test_jaccard_partial():
    j = _jaccard("quantum mechanics introduction", "quantum field theory introduction")
    # shared: quantum, introduction (2); union: 5
    assert abs(j - 2 / 5) < 1e-9


def test_jaccard_empty():
    assert _jaccard("", "a b") == 0.0


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    p = tmp_path / "library.db"
    db.init_db(p)
    return p


_counter = 0


def _insert(tmp_db: Path, title: str, doi: str | None = None) -> int:
    global _counter
    _counter += 1
    unique_hash = f"hash-{_counter}"
    match = PaperMatch(
        source="crossref",
        doi=doi or f"10.99/{_counter}",
        title=title,
        authors=[Author(family="Smith", given="Alice")],
        year=2024,
    )
    with db.connect(tmp_db) as conn:
        return db.insert_paper(
            conn, match, file_path=f"smith/2024/{_counter}.pdf", source_pdf_hash=unique_hash
        )


# ---------------------------------------------------------------------------
# _find_duplicates (exact)
# ---------------------------------------------------------------------------


def test_find_duplicates_exact_match(tmp_db: Path):
    with db.connect(tmp_db) as conn:
        rows = list(conn.execute("SELECT * FROM papers ORDER BY id"))
    assert _find_duplicates(rows) == []


def test_find_duplicates_no_dupes(tmp_db: Path):
    _insert(tmp_db, "Quantum Mechanics", doi="10.1/a")
    _insert(tmp_db, "Classical Mechanics", doi="10.1/b")
    with db.connect(tmp_db) as conn:
        rows = list(conn.execute("SELECT * FROM papers ORDER BY id"))
    assert _find_duplicates(rows) == []


def test_find_duplicates_exact_title(tmp_db: Path):
    # Two papers with identical normalised titles
    _insert(tmp_db, "Quantum Mechanics", doi="10.1/a")
    _insert(tmp_db, "quantum mechanics", doi="10.1/b")
    with db.connect(tmp_db) as conn:
        rows = list(conn.execute("SELECT * FROM papers ORDER BY id"))
    groups = _find_duplicates(rows)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_find_duplicates_punctuation_ignored(tmp_db: Path):
    _insert(tmp_db, "Entropy: A Story", doi="10.1/a")
    _insert(tmp_db, "Entropy A Story", doi="10.1/b")
    with db.connect(tmp_db) as conn:
        rows = list(conn.execute("SELECT * FROM papers ORDER BY id"))
    groups = _find_duplicates(rows)
    assert len(groups) == 1


def test_find_duplicates_three_copies(tmp_db: Path):
    _insert(tmp_db, "The Same Paper", doi="10.1/a")
    _insert(tmp_db, "The Same Paper", doi="10.1/b")
    _insert(tmp_db, "the same paper", doi="10.1/c")
    with db.connect(tmp_db) as conn:
        rows = list(conn.execute("SELECT * FROM papers ORDER BY id"))
    groups = _find_duplicates(rows)
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_find_duplicates_distinct_groups(tmp_db: Path):
    _insert(tmp_db, "Alpha Paper", doi="10.1/a")
    _insert(tmp_db, "Alpha Paper", doi="10.1/b")
    _insert(tmp_db, "Beta Paper", doi="10.1/c")
    _insert(tmp_db, "Beta Paper", doi="10.1/d")
    with db.connect(tmp_db) as conn:
        rows = list(conn.execute("SELECT * FROM papers ORDER BY id"))
    groups = _find_duplicates(rows)
    assert len(groups) == 2


# ---------------------------------------------------------------------------
# _find_duplicates (fuzzy)
# ---------------------------------------------------------------------------


def test_find_duplicates_fuzzy_near_match(tmp_db: Path):
    # 4 shared tokens, union = 5 → Jaccard = 4/5 = 0.8 (meets threshold)
    _insert(tmp_db, "Quantum Mechanics Thermodynamics Introduction")
    _insert(tmp_db, "Quantum Mechanics Thermodynamics Introduction Extended")
    with db.connect(tmp_db) as conn:
        rows = list(conn.execute("SELECT * FROM papers ORDER BY id"))
    # Should NOT appear in exact mode
    assert _find_duplicates(rows, fuzzy=False) == []
    # Should appear in fuzzy mode
    groups = _find_duplicates(rows, fuzzy=True)
    assert len(groups) == 1


def test_find_duplicates_fuzzy_no_false_positives(tmp_db: Path):
    _insert(tmp_db, "Quantum Field Theory", doi="10.1/a")
    _insert(tmp_db, "Statistical Mechanics Review", doi="10.1/b")
    with db.connect(tmp_db) as conn:
        rows = list(conn.execute("SELECT * FROM papers ORDER BY id"))
    assert _find_duplicates(rows, fuzzy=True) == []
