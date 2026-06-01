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
    metrics: list[dict] = []
    respx.post("http://one.test/api/embeddings").mock(return_value=Response(500))
    respx.post("http://two.test/api/embeddings").mock(
        return_value=Response(200, json={"embedding": [2.0, 4.0]})
    )
    service = EmbeddingService(
        "embed-model",
        ["http://one.test", "http://two.test"],
        alert_cb=alerts.append,
        metrics_cb=metrics.append,
        quarantine_seconds=60,
    )

    try:
        assert await service.embed_text("hello") == [2.0, 4.0]
        assert alerts
        assert service._endpoint_down_until["http://one.test"] > 0
        assert [event["event"] for event in metrics] == [
            "endpoint_score_updated",
            "embedding_failure",
            "embedding_endpoint_quarantined",
            "endpoint_score_updated",
            "embedding_success",
        ]
    finally:
        await service.close()


def test_first_endpoint_down_alert_is_not_suppressed_on_low_uptime():
    alerts: list[str] = []
    service = EmbeddingService(
        "embed-model",
        ["http://one.test"],
        alert_cb=alerts.append,
        quarantine_seconds=60,
    )

    service._alert_endpoint_down("http://one.test", "HTTP 500")

    assert len(alerts) == 1
    assert service._endpoint_down_until["http://one.test"] > 0


def test_repeated_endpoint_down_alert_is_suppressed_during_cooldown():
    alerts: list[str] = []
    service = EmbeddingService(
        "embed-model",
        ["http://one.test"],
        alert_cb=alerts.append,
        quarantine_seconds=60,
    )

    service._alert_endpoint_down("http://one.test", "first")
    service._alert_endpoint_down("http://one.test", "second")

    assert len(alerts) == 1


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


@pytest.mark.asyncio
@respx.mock
async def test_endpoint_scoring_prioritizes_successful_endpoint():
    calls: list[str] = []

    def one_responder(_request):
        calls.append("one")
        return Response(500)

    def two_responder(_request):
        calls.append("two")
        return Response(200, json={"embedding": [2.0]})

    respx.post("http://one.test/api/embeddings").mock(side_effect=one_responder)
    respx.post("http://two.test/api/embeddings").mock(side_effect=two_responder)
    service = EmbeddingService(
        "embed-model",
        ["http://one.test", "http://two.test"],
        quarantine_seconds=0,
    )

    try:
        assert await service.embed_text("first") == [2.0]
        assert await service.embed_text("second") == [2.0]
        assert calls == ["one", "two", "two"]
    finally:
        await service.close()
