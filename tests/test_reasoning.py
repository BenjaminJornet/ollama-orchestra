from __future__ import annotations

import respx
from httpx import Response

from ollama_orchestra.reasoning import chat, stream_chat, strip_reasoning


def test_strip_reasoning_removes_reasoning_blocks_and_fences():
    text = """
<think>hidden chain</think>
```markdown
Visible answer
```
<reasoning>more hidden text</reasoning>
"""

    assert strip_reasoning(text) == "Visible answer"


@respx.mock
async def test_chat_sets_think_false_top_level_and_strips_output():
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(200, json={"message": {"content": "<think>x</think>answer"}})
    )

    result = await chat(
        "http://localhost:11434",
        "example-model",
        [{"role": "user", "content": "hi"}],
        num_predict=32,
    )

    body = route.calls.last.request.content.decode()
    assert '"think":false' in body.replace(" ", "")
    assert '"options":{"num_predict":32}' in body.replace(" ", "")
    assert result["message"]["content"] == "answer"


@respx.mock
async def test_stream_chat_yields_json_lines_and_sets_think_false():
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200,
            content=(
                b'{"message":{"content":"<think>x</think>hel"},"done":false}\n'
                b'{"message":{"content":"lo"},"done":true}\n'
            ),
        )
    )

    chunks = [
        chunk
        async for chunk in stream_chat(
            "http://localhost:11434",
            "example-model",
            [{"role": "user", "content": "hi"}],
            num_predict=32,
        )
    ]

    body = route.calls.last.request.content.decode()
    assert '"stream":true' in body.replace(" ", "")
    assert '"think":false' in body.replace(" ", "")
    assert chunks[0]["message"]["content"] == "hel"
    assert chunks[1]["done"] is True
