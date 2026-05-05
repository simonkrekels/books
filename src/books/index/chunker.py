"""Chunk page-mapped text into overlapping pieces for embedding.

Wraps `semantic-text-splitter`_'s character-mode splitter and rebuilds the
page-of-origin for each chunk afterwards by tracking byte offsets through a
flattened version of the input.

.. _semantic-text-splitter: https://github.com/benbrandt/text-splitter
"""

from dataclasses import dataclass

from semantic_text_splitter import TextSplitter


@dataclass
class Chunk:
    """A single chunk of text ready to be embedded.

    ``page_number`` is the page where the chunk's *first character* lives.
    Long chunks may straddle a page boundary; we report the starting page.
    """

    text: str
    page_number: int
    chunk_index: int  # 0-based position within the paper


def chunk_pages(
    pages: list[tuple[int, str]],
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Split ``pages`` (from :func:`books.index.extract.extract_text`) into chunks.

    The token budget is converted to characters at ~4 chars/token (a common
    English approximation). For more precise budgeting, swap to
    ``TextSplitter.from_huggingface_tokenizer(...)`` matched to the
    embedding model. Good enough as a v1 default.
    """
    if not pages:
        return []

    chars_per_token = 4
    capacity = chunk_tokens * chars_per_token
    overlap = overlap_tokens * chars_per_token
    splitter = TextSplitter(capacity, overlap=overlap)

    full_text, page_starts = _concat(pages)
    chunks = list(splitter.chunks(full_text))

    out: list[Chunk] = []
    cursor = 0
    for idx, chunk_text in enumerate(chunks):
        # `find` recovers each chunk's offset in the concatenated text. The
        # cursor moves forward only — splitter chunks are non-decreasing in
        # offset, so we don't re-scan from the start each time.
        loc = full_text.find(chunk_text, cursor)
        if loc < 0:
            # Defensive fallback: if find fails (shouldn't happen, but the
            # splitter is allowed to massage whitespace), use the cursor.
            loc = cursor
        cursor = loc + max(len(chunk_text), 1)
        page = _page_at_offset(loc, page_starts)
        out.append(Chunk(text=chunk_text, page_number=page, chunk_index=idx))
    return out


# Two newlines separate each page in the flattened text — same heuristic
# the chunker would otherwise use as a paragraph boundary.
_PAGE_SEP = "\n\n"


def _concat(pages: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int]]]:
    """Flatten pages into one string + return ``[(page_number, start_offset), ...]``.

    The returned offset table lets :func:`_page_at_offset` resolve a chunk's
    byte offset back to a page number in O(P) where P = page count.
    """
    parts: list[str] = []
    starts: list[tuple[int, int]] = []
    cursor = 0
    for page_num, text in pages:
        starts.append((page_num, cursor))
        parts.append(text)
        cursor += len(text) + len(_PAGE_SEP)
        parts.append(_PAGE_SEP)
    return "".join(parts), starts


def _page_at_offset(offset: int, starts: list[tuple[int, int]]) -> int:
    """Return the page whose start offset is the largest one ≤ ``offset``."""
    page = starts[0][0]
    for p, start in starts:
        if start <= offset:
            page = p
        else:
            break
    return page
