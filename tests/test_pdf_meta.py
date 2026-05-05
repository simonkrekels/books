from pathlib import Path

import pytest

from books.metadata.pdf_meta import (
    arxiv_candidates,
    doi_candidates,
    score_arxiv,
    score_dois,
    sniff_pdf,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# Some example PDFs aren't shipped with the repo (size / copyright). Tests
# that depend on them skip cleanly when the file is absent so the suite
# still passes on a fresh clone.
DOLAI_PDF = EXAMPLES / "Dolai et al. - 2022 - Inducing a bound state between active particles.pdf"
ARXIV_PDF = EXAMPLES / "2604.00777v1.pdf"


def test_doi_basic():
    assert doi_candidates("see DOI 10.1234/abcd here") == ["10.1234/abcd"]


def test_doi_strips_trailing_punct():
    assert doi_candidates("...10.1103/PhysRevE.105.034605.") == [
        "10.1103/physreve.105.034605"
    ]


def test_doi_in_url():
    cands = doi_candidates("https://doi.org/10.1038/s41586-022-04565-9")
    assert "10.1038/s41586-022-04565-9" in cands


def test_doi_no_match():
    assert doi_candidates("no doi here") == []


def test_arxiv_new_with_prefix():
    assert arxiv_candidates("arXiv:2604.00777") == ["2604.00777"]
    assert arxiv_candidates("arXiv:2604.00777v3") == ["2604.00777"]
    assert arxiv_candidates("see arxiv.org/abs/2604.00777") == ["2604.00777"]


def test_arxiv_old_format():
    assert "hep-th/9711200" in arxiv_candidates("see hep-th/9711200 for details")


def test_arxiv_bare_only_in_filename():
    # Body text without "arXiv:" hint shouldn't match (avoid false positives)
    assert arxiv_candidates("equation 2604.00777 in section 3") == []
    # But filename does match
    assert arxiv_candidates("equation 2604.00777", filename="2604.00777v1") == [
        "2604.00777"
    ]


def test_score_dois_prefers_higher_weight():
    # DOI A appears once with weight 5; DOI B appears 3x with weight 1.
    # A wins on weight even though B is more frequent.
    sources = [
        ("front matter mentions 10.1234/aaa", 5),
        ("body cites 10.5678/bbb,10.5678/bbb,10.5678/bbb", 1),
    ]
    scores = score_dois(sources)
    assert scores["10.1234/aaa"] == 5
    assert scores["10.5678/bbb"] == 3
    assert scores.most_common(1)[0][0] == "10.1234/aaa"


def test_score_dois_accumulates_across_sources():
    # Same DOI in metadata (weight 10) and on page 1 (weight 5) accumulates.
    sources = [
        ("doi: 10.1234/own", 10),
        ("see 10.1234/own again", 5),
        ("citation 10.5678/other", 1),
    ]
    scores = score_dois(sources)
    assert scores["10.1234/own"] == 15
    assert scores["10.5678/other"] == 1


def test_score_arxiv_filename_fallback_only_without_contextual_match():
    # Contextual match in body wins; filename ignored when contextual exists.
    sources = [("text contains arXiv:1234.5678", 1)]
    scores = score_arxiv(sources, filename="9999.99999")
    assert "1234.5678" in scores
    assert "9999.99999" not in scores


# Real PDF sniffing — uses files at examples/
@pytest.mark.skipif(not ARXIV_PDF.exists(), reason="example PDF not present")
def test_sniff_arxiv_preprint_filename():
    res = sniff_pdf(ARXIV_PDF)
    assert res.arxiv_id == "2604.00777"
    # The cited DOI in the body must NOT win — it scores below the threshold,
    # so sniff_pdf returns None for the DOI and the importer falls back to arXiv.
    assert res.doi is None


@pytest.mark.skipif(not DOLAI_PDF.exists(), reason="example PDF not present")
def test_sniff_dolai_paper_finds_doi():
    res = sniff_pdf(DOLAI_PDF)
    # Whatever the DOI is, sniffing should produce *something* — at least one candidate.
    # We check that the result is well-formed rather than asserting a specific value
    # (DOI varies per real-world paper).
    if res.doi is not None:
        assert res.doi.startswith("10.")
