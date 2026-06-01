from __future__ import annotations

import pytest
import respx
from httpx import Response

from ollama_orchestra.chunking import TextChunker
from ollama_orchestra.embeddings import EmbeddingService


@pytest.mark.asyncio
@respx.mock
async def test_embed_text_returns_embedding_from_first_endpoint():
    respx.post("http://one.test/api/embeddings").mock(
        return_value=Response(200, json={"embedding": [1.0, 2.0, 3.0]})
    )
    service = EmbeddingService("embed-model", ["http://one.test"])

    try:
        assert await service.embed_text("hello") == [1.0, 2.0, 3.0]
    finally:
        await service.close()


@pytest.mark.asyncio
@respx.mock
async def test_embedding_falls_back_to_second_endpoint_and_alerts():
    alerts: list[str] = []
    respx.post("http://one.test/api/embeddings").mock(return_value=Response(500))
    respx.post("http://two.test/api/embeddings").mock(
        return_value=Response(200, json={"embedding": [2.0, 4.0]})
    )
    service = EmbeddingService(
        "embed-model",
        ["http://one.test", "http://two.test"],
        alert_cb=alerts.append,
        quarantine_seconds=60,
    )

    try:
        assert await service.embed_text("hello") == [2.0, 4.0]
        assert alerts
        assert service._endpoint_down_until["http://one.test"] > 0
    finally:
        await service.close()


@pytest.mark.asyncio
@respx.mock
async def test_long_text_uses_mean_pooling():
    calls = 0

    def responder(_request):
        nonlocal calls
        calls += 1
        return Response(200, json={"embedding": [float(calls), float(calls * 2)]})

    respx.post("http://one.test/api/embeddings").mock(side_effect=responder)
    chunker = TextChunker(target_size=80, max_size=120, overlap=0, min_size=10)
    service = EmbeddingService("embed-model", ["http://one.test"], chunker=chunker)
    service.MAX_EMBEDDING_TEXT_LENGTH = 80

    try:
        text = "\n\n".join(["paragraph " + str(i) + " " + ("x" * 70) for i in range(4)])
        embedding = await service.embed_text(text)
        assert embedding is not None
        assert calls > 1
        assert embedding[1] == embedding[0] * 2
    finally:
        await service.close()


@pytest.mark.asyncio
@respx.mock
async def test_embed_texts_preserves_order():
    respx.post("http://one.test/api/embeddings").mock(
        return_value=Response(200, json={"embedding": [1.0]})
    )
    service = EmbeddingService("embed-model", ["http://one.test"])

    try:
        assert await service.embed_texts(["a", "b"], batch_size=2) == [[1.0], [1.0]]
    finally:
        await service.close()
