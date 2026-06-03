from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

_REASONING_BLOCK_RE = re.compile(
    r"<(think|reasoning|thought)>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.MULTILINE)


def strip_reasoning(text: str) -> str:
    """Remove reasoning tags and leftover Markdown fences from model output."""
    if not text:
        return text
    cleaned = _REASONING_BLOCK_RE.sub("", text)
    cleaned = _FENCE_RE.sub("", cleaned)
    return cleaned.strip()


async def chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    think: bool = False,
    strip: bool = True,
    timeout: float = 120.0,
    **opts: Any,
) -> dict[str, Any]:
    """Call Ollama `/api/chat` with top-level `think: false` by default.

    Some reasoning models can spend the whole `num_predict` budget inside hidden
    reasoning and return an empty visible message with `done_reason: "length"`.
    Ollama supports disabling this via top-level `think: false`; putting it
    inside `options` does not work.
    """
    url = base_url.rstrip("/").removesuffix("/v1")
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if think is False:
        body["think"] = False
    elif think is True:
        body["think"] = True
    if opts:
        body["options"] = opts

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{url}/api/chat", json=body)
        response.raise_for_status()
        data = response.json()

    if strip:
        message = data.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = strip_reasoning(message["content"])
    return data


async def stream_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    think: bool = False,
    strip: bool = True,
    timeout: float = 120.0,
    **opts: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Stream Ollama `/api/chat` JSON lines with top-level `think: false` by default."""
    url = base_url.rstrip("/").removesuffix("/v1")
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
    if think is False:
        body["think"] = False
    elif think is True:
        body["think"] = True
    if opts:
        body["options"] = opts

    async with (
        httpx.AsyncClient(timeout=timeout) as client,
        client.stream("POST", f"{url}/api/chat", json=body) as response,
    ):
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if strip:
                message = chunk.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    message["content"] = strip_reasoning(message["content"])
            yield chunk
