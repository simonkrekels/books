from books.index.chunker import chunk_pages


def test_chunker_preserves_page_for_short_text():
    pages = [(1, "Page one content."), (2, "Page two content."), (3, "Page three content.")]
    chunks = chunk_pages(pages, chunk_tokens=200, overlap_tokens=10)
    assert chunks
    # All chunks should have page numbers from the input set
    assert all(c.page_number in {1, 2, 3} for c in chunks)
    # Chunk indices are sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunker_long_text_produces_multiple_chunks():
    long_text = ("This is a sentence. " * 500)  # ~10000 chars
    chunks = chunk_pages([(1, long_text)], chunk_tokens=128, overlap_tokens=16)
    assert len(chunks) > 1
    # All from page 1
    assert all(c.page_number == 1 for c in chunks)


def test_chunker_empty_input():
    assert chunk_pages([], chunk_tokens=128, overlap_tokens=0) == []


def test_chunker_assigns_correct_page_for_split():
    # Both pages large enough that each spans multiple chunks.
    page1 = "alpha " * 200
    page2 = "beta " * 200
    chunks = chunk_pages(
        [(1, page1), (2, page2)], chunk_tokens=64, overlap_tokens=0
    )
    pages = {c.page_number for c in chunks}
    assert pages == {1, 2}
    # First chunk is page 1, last is page 2.
    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 2
