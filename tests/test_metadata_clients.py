import httpx
from pytest_httpx import HTTPXMock

from books.metadata import arxiv, crossref


def test_crossref_parses_response(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.crossref.org/works/10.1234/test",
        json={
            "status": "ok",
            "message": {
                "DOI": "10.1234/test",
                "title": ["Bound States Between Active Particles"],
                "author": [
                    {"family": "Dolai", "given": "Pradeep"},
                    {"family": "Das", "given": "Arnab"},
                ],
                "issued": {"date-parts": [[2022, 5, 15]]},
                "container-title": ["Physical Review E"],
                "publisher": "American Physical Society",
                "type": "journal-article",
                "abstract": "<p>We show...</p>",
            },
        },
    )
    with httpx.Client() as c:
        match = crossref.lookup("10.1234/test", client=c)
    assert match is not None
    assert match.doi == "10.1234/test"
    assert match.title.startswith("Bound States")
    assert match.year == 2022
    assert match.journal == "Physical Review E"
    assert [a.family for a in match.authors] == ["Dolai", "Das"]
    assert match.type == "journal-article"


def test_crossref_returns_none_on_404(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.crossref.org/works/10.1234/missing",
        status_code=404,
    )
    with httpx.Client() as c:
        match = crossref.lookup("10.1234/missing", client=c)
    assert match is None


ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2604.00777v1</id>
    <title>A Test Preprint Title</title>
    <summary>This is the abstract text of a fake arXiv paper.</summary>
    <published>2026-04-05T00:00:00Z</published>
    <author><name>Jane Q. Smith</name></author>
    <author><name>John Doe</name></author>
    <arxiv:doi>10.9999/foo</arxiv:doi>
  </entry>
</feed>
"""


def test_arxiv_parses_response(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://export.arxiv.org/api/query?id_list=2604.00777",
        text=ARXIV_XML,
    )
    with httpx.Client() as c:
        match = arxiv.lookup("2604.00777", client=c)
    assert match is not None
    assert match.title == "A Test Preprint Title"
    assert match.year == 2026
    assert match.doi == "10.9999/foo"
    assert match.arxiv_id == "2604.00777"
    assert [a.family for a in match.authors] == ["Smith", "Doe"]
    assert match.authors[0].given == "Jane Q."
    assert match.type == "preprint"
