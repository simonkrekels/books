"""Tests for context + reference building (books.rag.prompt)."""

from pathlib import Path

import pytest

from books import db
from books.rag import prompt
from books.metadata.models import Author, PaperMatch
from books.retrieval import SearchResult


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    p = tmp_path / "library.db"
    db.init_db(p)
    return p


def _paper(tmp_db: Path, doi: str, title: str, authors: list[Author], year: int) -> int:
    match = PaperMatch(source="crossref", doi=doi, title=title, authors=authors, year=year)
    with db.connect(tmp_db) as conn:
        return db.insert_paper(conn, match, file_path=f"{doi}.pdf", source_pdf_hash=doi)


def _result(paper_id: int, page: int, text: str) -> SearchResult:
    return SearchResult(paper_id=paper_id, chunk_index=0, page=page, text=text, score=0.5)


def test_build_context_numbers_contiguously(tmp_db: Path):
    p1 = _paper(tmp_db, "10.1/a", "Paper A", [Author(family="Smith", given="A")], 2020)
    p2 = _paper(tmp_db, "10.1/b", "Paper B", [Author(family="Jones", given="B")], 2021)
    packed = [_result(p1, 3, "alpha content"), _result(p2, 7, "beta content")]
    with db.connect(tmp_db) as conn:
        context, refs = prompt.build_context(conn, packed)

    assert [r.n for r in refs] == [1, 2]
    assert "[1]" in context and "[2]" in context
    assert "alpha content" in context and "beta content" in context


def test_references_map_to_packed_chunks(tmp_db: Path):
    p1 = _paper(tmp_db, "10.1/a", "Attention Is All You Need",
                [Author(family="Vaswani", given="A"), Author(family="Shazeer", given="N")], 2017)
    packed = [_result(p1, 3, "transformer text")]
    with db.connect(tmp_db) as conn:
        _, refs = prompt.build_context(conn, packed)

    assert len(refs) == len(packed)
    ref = refs[0]
    assert ref.paper_id == p1
    assert ref.year == 2017
    assert ref.page == 3
    assert ref.author == "Vaswani et al."  # multiple authors → "et al."
    assert ref.title == "Attention Is All You Need"


def test_single_author_label(tmp_db: Path):
    p1 = _paper(tmp_db, "10.1/solo", "Solo", [Author(family="Hawking", given="S")], 1988)
    with db.connect(tmp_db) as conn:
        _, refs = prompt.build_context(conn, [_result(p1, 1, "text")])
    assert refs[0].author == "Hawking"


def test_reference_label_format(tmp_db: Path):
    p1 = _paper(tmp_db, "10.1/a", "T", [Author(family="Smith", given="A")], 2024)
    with db.connect(tmp_db) as conn:
        _, refs = prompt.build_context(conn, [_result(p1, 5, "text")])
    assert refs[0].label() == f"Smith (2024) p.5  id={p1}"


def test_build_user_message_contains_question_and_context():
    msg = prompt.build_user_message("[1] excerpt", "What is X?")
    assert "[1] excerpt" in msg
    assert "What is X?" in msg
