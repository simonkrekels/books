"""Pack ranked retrieval results into a model-sized token budget.

This is where "the number of chunks fed is optimized to the model" lives: the
user sets ``rag.context_budget`` to suit the chosen model's context window, and
chunks are taken in rank order until that budget (or ``rag.max_chunks``) is hit.

Token counts are estimated at ~4 characters/token — the same ratio the chunker
uses (see :mod:`books.index.chunker`).  An estimate is sufficient here: it keeps
the dependency-free packer in lockstep with how chunks were sized at index time,
and the model is given headroom via ``rag.reserve_output`` regardless.
"""

from books.retrieval import SearchResult

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token count of *text* (~4 chars/token, rounded up)."""
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def pack(
    results: list[SearchResult],
    *,
    budget_tokens: int,
    max_chunks: int,
) -> list[SearchResult]:
    """Return the rank-order prefix of *results* that fits the budget.

    Walks *results* (already sorted best-first) and accepts each chunk whose
    estimated tokens still fit within ``budget_tokens``, stopping at
    ``max_chunks`` chunks.  A single chunk larger than the whole budget is
    still taken if it would be the first one, so ``ask`` never returns an empty
    context purely because the top hit is large.
    """
    packed: list[SearchResult] = []
    used = 0
    for r in results:
        if len(packed) >= max_chunks:
            break
        cost = estimate_tokens(r.text)
        if packed and used + cost > budget_tokens:
            continue
        packed.append(r)
        used += cost
    return packed
