"""Tests for the token-budget chunk packer (books.rag.budget)."""

from books.rag import budget
from books.retrieval import SearchResult


def _result(i: int, text: str, score: float) -> SearchResult:
    return SearchResult(paper_id=i, chunk_index=0, page=1, text=text, score=score)


def test_estimate_tokens_rounds_up():
    assert budget.estimate_tokens("") == 0
    assert budget.estimate_tokens("abc") == 1  # 3 chars / 4 → ceil = 1
    assert budget.estimate_tokens("a" * 8) == 2


def test_pack_respects_token_budget():
    # Each chunk is 40 chars ≈ 10 tokens; budget of 25 fits two.
    results = [_result(i, "x" * 40, score=1.0 - i * 0.1) for i in range(5)]
    packed = budget.pack(results, budget_tokens=25, max_chunks=100)
    assert len(packed) == 2
    total = sum(budget.estimate_tokens(r.text) for r in packed)
    assert total <= 25


def test_pack_respects_max_chunks():
    results = [_result(i, "tiny", score=1.0 - i * 0.1) for i in range(10)]
    packed = budget.pack(results, budget_tokens=10_000, max_chunks=3)
    assert len(packed) == 3


def test_pack_preserves_rank_order():
    results = [_result(i, "word " * 5, score=1.0 - i * 0.1) for i in range(4)]
    packed = budget.pack(results, budget_tokens=10_000, max_chunks=100)
    assert [r.paper_id for r in packed] == [0, 1, 2, 3]


def test_pack_takes_first_chunk_even_if_over_budget():
    # A single oversized top hit is still returned rather than an empty context.
    results = [_result(0, "x" * 4000, score=0.9)]
    packed = budget.pack(results, budget_tokens=10, max_chunks=100)
    assert len(packed) == 1


def test_pack_skips_oversized_later_chunks_but_keeps_smaller_ones():
    results = [
        _result(0, "x" * 20, score=0.9),   # 5 tokens
        _result(1, "x" * 4000, score=0.8),  # 1000 tokens — too big
        _result(2, "x" * 20, score=0.7),   # 5 tokens — still fits
    ]
    packed = budget.pack(results, budget_tokens=12, max_chunks=100)
    assert [r.paper_id for r in packed] == [0, 2]


def test_pack_empty_input():
    assert budget.pack([], budget_tokens=100, max_chunks=10) == []
