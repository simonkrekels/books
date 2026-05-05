from pathlib import Path

import pytest

from books import db
from books.metadata.models import Author, PaperMatch


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    p = tmp_path / "library.db"
    db.init_db(p)
    return p


def _sample_match(**overrides) -> PaperMatch:
    base = dict(
        source="crossref",
        doi="10.1234/abcd",
        title="A Test Paper",
        authors=[
            Author(family="Knuth", given="Donald E."),
            Author(family="Turing", given="Alan"),
        ],
        year=1968,
        journal="Test Journal",
    )
    base.update(overrides)
    return PaperMatch(**base)


def test_init_creates_schema(tmp_db: Path):
    with db.connect(tmp_db) as conn:
        rows = list(conn.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    names = {r["name"] for r in rows}
    assert {"papers", "authors", "paper_authors", "tags", "schema_version"} <= names


def test_insert_and_lookup_by_doi(tmp_db: Path):
    with db.connect(tmp_db) as conn:
        pid = db.insert_paper(
            conn,
            _sample_match(),
            file_path="knuth/1968/a-test-paper.pdf",
            source_pdf_hash="deadbeef",
        )
        row = db.find_by_doi(conn, "10.1234/abcd")
        assert row is not None
        assert row["id"] == pid
        assert row["title"] == "A Test Paper"
        assert row["year"] == 1968


def test_get_authors_in_order(tmp_db: Path):
    with db.connect(tmp_db) as conn:
        pid = db.insert_paper(
            conn,
            _sample_match(),
            file_path="x.pdf",
            source_pdf_hash="aa",
        )
        authors = db.get_authors(conn, pid)
        assert [a["family_name"] for a in authors] == ["Knuth", "Turing"]


def test_duplicate_hash_rejected(tmp_db: Path):
    import sqlite3

    with db.connect(tmp_db) as conn:
        db.insert_paper(
            conn,
            _sample_match(),
            file_path="a.pdf",
            source_pdf_hash="hash1",
        )
    with db.connect(tmp_db) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            db.insert_paper(
                conn,
                _sample_match(doi="10.0000/other"),
                file_path="b.pdf",
                source_pdf_hash="hash1",
            )


def test_delete_cascades(tmp_db: Path):
    with db.connect(tmp_db) as conn:
        pid = db.insert_paper(
            conn,
            _sample_match(),
            file_path="x.pdf",
            source_pdf_hash="hh",
        )
        db.delete_paper(conn, pid)
        rem = list(conn.execute("SELECT * FROM paper_authors WHERE paper_id = ?", (pid,)))
        assert rem == []
