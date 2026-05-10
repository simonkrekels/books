from pathlib import Path

import pytest

from books import db, query
from books.metadata.models import Author, PaperMatch


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
    p = tmp_path / "library.db"
    db.init_db(p)
    with db.connect(p) as conn:
        db.insert_paper(
            conn,
            PaperMatch(
                source="crossref",
                doi="10.1/quanta",
                title="On the Theory of Quanta",
                authors=[Author(family="de Broglie", given="Louis")],
                year=1924,
                journal="Annales de Physique",
            ),
            file_path="de-broglie/1924/quanta.pdf",
            source_pdf_hash="h1",
        )
        db.insert_paper(
            conn,
            PaperMatch(
                source="arxiv",
                arxiv_id="2604.00777",
                title="A Quantum Algorithm",
                authors=[Author(family="Smith", given="Jane")],
                year=2024,
                abstract="A new quantum algorithm for transformer attention.",
            ),
            file_path="smith/2024/quantum-algo.pdf",
            source_pdf_hash="h2",
        )
        pid = db.insert_paper(
            conn,
            PaperMatch(
                source="crossref",
                doi="10.2/foo",
                title="Classical Mechanics",
                authors=[Author(family="Smith", given="John")],
                year=2019,
            ),
            file_path="smith/2019/classical.pdf",
            source_pdf_hash="h3",
        )
        conn.execute("INSERT INTO tags(paper_id, tag) VALUES (?, ?)", (pid, "textbook"))
    return p


def test_filter_by_author(populated_db: Path):
    sql, params = query.build_papers_query(author="Smith")
    with db.connect(populated_db) as conn:
        rows = list(conn.execute(sql, params))
    assert {r["title"] for r in rows} == {"A Quantum Algorithm", "Classical Mechanics"}


def test_filter_by_year(populated_db: Path):
    sql, params = query.build_papers_query(year=1924)
    with db.connect(populated_db) as conn:
        rows = list(conn.execute(sql, params))
    assert [r["title"] for r in rows] == ["On the Theory of Quanta"]


def test_filter_by_term_in_abstract(populated_db: Path):
    sql, params = query.build_papers_query(terms=["transformer"])
    with db.connect(populated_db) as conn:
        rows = list(conn.execute(sql, params))
    assert [r["title"] for r in rows] == ["A Quantum Algorithm"]


def test_filter_by_tag(populated_db: Path):
    sql, params = query.build_papers_query(tag="textbook")
    with db.connect(populated_db) as conn:
        rows = list(conn.execute(sql, params))
    assert [r["title"] for r in rows] == ["Classical Mechanics"]


def test_combined_filters(populated_db: Path):
    sql, params = query.build_papers_query(author="Smith", year=2024)
    with db.connect(populated_db) as conn:
        rows = list(conn.execute(sql, params))
    assert [r["title"] for r in rows] == ["A Quantum Algorithm"]


def test_order_year_desc(populated_db: Path):
    sql, params = query.build_papers_query()
    with db.connect(populated_db) as conn:
        rows = list(conn.execute(sql, params))
    assert [r["year"] for r in rows] == [2024, 2019, 1924]


# ---------------------------------------------------------------------------
# resolve_paper_ids
# ---------------------------------------------------------------------------


def test_resolve_no_filters_returns_none(populated_db: Path):
    with db.connect(populated_db) as conn:
        assert query.resolve_paper_ids(conn) is None


def test_resolve_by_author(populated_db: Path):
    with db.connect(populated_db) as conn:
        ids = query.resolve_paper_ids(conn, author="Smith")
        all_rows = list(conn.execute("SELECT id, title FROM papers"))
    smith_ids = {r["id"] for r in all_rows if "Smith" in r["title"] or True}
    # Both Smith papers should be present; de Broglie should not
    de_broglie_id = next(r["id"] for r in all_rows if "Quanta" in r["title"])
    assert de_broglie_id not in ids
    assert len(ids) == 2


def test_resolve_by_tag(populated_db: Path):
    with db.connect(populated_db) as conn:
        ids = query.resolve_paper_ids(conn, tag="textbook")
        textbook_id = conn.execute(
            "SELECT p.id FROM papers p JOIN tags t ON t.paper_id = p.id WHERE t.tag = 'textbook'"
        ).fetchone()["id"]
    assert ids == {textbook_id}


def test_resolve_year_min(populated_db: Path):
    with db.connect(populated_db) as conn:
        ids = query.resolve_paper_ids(conn, year_min=2000)
        all_rows = list(conn.execute("SELECT id, year FROM papers"))
    assert ids == {r["id"] for r in all_rows if r["year"] and r["year"] >= 2000}


def test_resolve_year_max(populated_db: Path):
    with db.connect(populated_db) as conn:
        ids = query.resolve_paper_ids(conn, year_max=2000)
        all_rows = list(conn.execute("SELECT id, year FROM papers"))
    assert ids == {r["id"] for r in all_rows if r["year"] and r["year"] <= 2000}


def test_resolve_year_range(populated_db: Path):
    with db.connect(populated_db) as conn:
        ids = query.resolve_paper_ids(conn, year_min=2019, year_max=2019)
        row = conn.execute("SELECT id FROM papers WHERE year = 2019").fetchone()
    assert ids == {row["id"]}


def test_resolve_no_match_returns_empty_set(populated_db: Path):
    with db.connect(populated_db) as conn:
        ids = query.resolve_paper_ids(conn, author="Nonexistent")
    assert ids == set()
