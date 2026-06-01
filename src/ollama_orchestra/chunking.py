from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    index: int
    start_offset: int
    end_offset: int
    is_primary: bool


class TextChunker:
    """Semantic text chunker for embedding pipelines."""

    def __init__(
        self,
        *,
        target_size: int = 1500,
        max_size: int = 2500,
        overlap: int = 200,
        min_size: int = 100,
    ) -> None:
        self.target_size = target_size
        self.max_size = max_size
        self.overlap = overlap
        self.min_size = min_size

    def chunk_text(self, text: str, *, include_overlap: bool = True) -> list[TextChunk]:
        if not text or len(text.strip()) < self.min_size:
            return []

        clean_text = text.strip()
        if len(clean_text) <= self.target_size:
            return [TextChunk(clean_text, 0, 0, len(clean_text), True)]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean_text) if p.strip()]
        if not paragraphs:
            paragraphs = [clean_text]

        chunks: list[TextChunk] = []
        current_parts: list[str] = []
        current_length = 0
        current_start = 0
        text_offset = 0

        for paragraph in paragraphs:
            paragraph_len = len(paragraph)
            if paragraph_len > self.max_size:
                if current_parts:
                    self._append_chunk(chunks, "\n\n".join(current_parts), current_start)
                    current_parts = []
                    current_length = 0

                sentence_parts: list[str] = []
                sentence_len = 0
                sentence_start = text_offset
                for sentence in self._split_into_sentences(paragraph):
                    if sentence_len + len(sentence) > self.target_size and sentence_parts:
                        chunk = " ".join(sentence_parts)
                        self._append_chunk(chunks, chunk, sentence_start)
                        if include_overlap:
                            overlap_text = (
                                " ".join(sentence_parts[-2:]) if len(sentence_parts) >= 2 else ""
                            )
                            if len(overlap_text) > self.overlap:
                                overlap_text = overlap_text[-self.overlap :]
                            sentence_parts = [overlap_text] if overlap_text else []
                            sentence_len = len(overlap_text)
                        else:
                            sentence_parts = []
                            sentence_len = 0
                        sentence_start = text_offset
                    sentence_parts.append(sentence)
                    sentence_len += len(sentence) + 1

                if sentence_parts:
                    current_parts = sentence_parts
                    current_length = sentence_len
                    current_start = sentence_start
            elif current_length + paragraph_len + 2 > self.target_size and current_parts:
                self._append_chunk(chunks, "\n\n".join(current_parts), current_start)
                if include_overlap and len(current_parts[-1]) <= self.overlap:
                    overlap_para = current_parts[-1]
                    current_parts = [overlap_para, paragraph]
                    current_length = len(overlap_para) + paragraph_len + 2
                else:
                    current_parts = [paragraph]
                    current_length = paragraph_len
                current_start = text_offset
            else:
                current_parts.append(paragraph)
                current_length += paragraph_len + 2

            text_offset += paragraph_len + 2

        if current_parts:
            self._append_chunk(chunks, "\n\n".join(current_parts), current_start)

        for index, chunk in enumerate(chunks):
            chunk.index = index
            chunk.is_primary = index == 0
        return chunks

    def split_chunk_to_max_length(self, chunk: TextChunk, max_length: int) -> list[TextChunk]:
        if len(chunk.text) <= max_length:
            return [chunk]

        step = max(max_length - self.overlap, 1)
        sub_chunks: list[TextChunk] = []
        start = 0
        while start < len(chunk.text):
            end = min(start + max_length, len(chunk.text))
            sub_text = chunk.text[start:end].strip()
            if sub_text:
                sub_chunks.append(
                    TextChunk(
                        text=sub_text,
                        index=len(sub_chunks),
                        start_offset=chunk.start_offset + start,
                        end_offset=chunk.start_offset + end,
                        is_primary=len(sub_chunks) == 0 and chunk.is_primary,
                    )
                )
            if end >= len(chunk.text):
                break
            start += step
        return sub_chunks

    def _append_chunk(self, chunks: list[TextChunk], text: str, start_offset: int) -> None:
        if len(text) < self.min_size:
            return
        chunks.append(
            TextChunk(text, len(chunks), start_offset, start_offset + len(text), len(chunks) == 0)
        )

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        return [s.strip() for s in sentence_endings.split(text) if s.strip()]


def chunk_text(text: str, **kwargs) -> list[TextChunk]:
    return TextChunker(**{k: v for k, v in kwargs.items() if k != "include_overlap"}).chunk_text(
        text,
        include_overlap=kwargs.get("include_overlap", True),
    )
