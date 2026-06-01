from __future__ import annotations

from ollama_orchestra.chunking import TextChunk, TextChunker, chunk_text


def test_short_text_returns_one_chunk():
    chunks = chunk_text("hello world" * 10, min_size=10)

    assert len(chunks) == 1
    assert chunks[0].is_primary is True


def test_long_text_splits_into_multiple_chunks():
    text = "\n\n".join(["paragraph " + str(i) + " " + ("x" * 80) for i in range(8)])
    chunker = TextChunker(target_size=180, max_size=300, overlap=100, min_size=10)

    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    assert chunks[0].index == 0
    assert chunks[0].is_primary is True
    assert all(chunk.index == index for index, chunk in enumerate(chunks))


def test_split_chunk_to_max_length_respects_limit():
    chunker = TextChunker(overlap=10, min_size=1)
    chunk = TextChunk("x" * 55, 0, 0, 55, True)

    split = chunker.split_chunk_to_max_length(chunk, 20)

    assert len(split) > 1
    assert all(len(item.text) <= 20 for item in split)


def test_tiny_text_below_min_size_is_ignored():
    assert chunk_text("small", min_size=100) == []
