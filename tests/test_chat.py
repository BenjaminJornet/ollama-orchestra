from __future__ import annotations

import pytest
import respx
from httpx import Response

from ollama_orchestra.chat import OrchestratedChat


@pytest.mark.asyncio
@respx.mock
async def test_orchestrated_chat_returns_message():
    respx.post("http://one.test/api/chat").mock(
        return_value=Response(
            200, json={"message": {"role": "assistant", "content": "hello"}}
        )
    )
    service = OrchestratedChat("test-model", ["http://one.test"])

    result = await service.chat([{"role": "user", "content": "hi"}])

    assert result is not None
    assert result["message"]["content"] == "hello"


@pytest.mark.asyncio
@respx.mock
async def test_orchestrated_chat_falls_back_and_scores():
    calls: list[str] = []
    metrics: list[dict] = []

    def one_responder(_request):
        calls.append("one")
        return Response(500)

    def two_responder(_request):
        calls.append("two")
        return Response(
            200, json={"message": {"role": "assistant", "content": "response"}}
        )

    respx.post("http://one.test/api/chat").mock(side_effect=one_responder)
    respx.post("http://two.test/api/chat").mock(side_effect=two_responder)

    service = OrchestratedChat(
        "test-model",
        ["http://one.test", "http://two.test"],
        metrics_cb=metrics.append,
        quarantine_seconds=60,
    )

    result = await service.chat([{"role": "user", "content": "hi"}])

    assert result is not None
    assert result["message"]["content"] == "response"

    # Second call should prioritize http://two.test directly because it has a better score
    result2 = await service.chat([{"role": "user", "content": "hi"}])
    assert result2 is not None

    assert calls == ["one", "two", "two"]
    assert any(m["event"] == "chat_success" and m["url"] == "http://two.test" for m in metrics)
    assert any(m["event"] == "chat_failure" and m["url"] == "http://one.test" for m in metrics)


@pytest.mark.asyncio
@respx.mock
async def test_endpoint_status_reports_scores_and_quarantine():
    respx.post("http://one.test/api/chat").mock(return_value=Response(500))
    respx.post("http://two.test/api/chat").mock(
        return_value=Response(200, json={"message": {"role": "assistant", "content": "ok"}})
    )
    service = OrchestratedChat(
        "test-model",
        ["http://one.test", "http://two.test"],
        quarantine_seconds=60,
    )

    await service.chat([{"role": "user", "content": "hi"}])
    status = service.endpoint_status()

    assert status[0]["url"] == "http://two.test"
    one = next(item for item in status if item["url"] == "http://one.test")
    assert one["failures"] == 1
    assert one["quarantined"] is True
    assert one["quarantine_remaining_seconds"] > 0
