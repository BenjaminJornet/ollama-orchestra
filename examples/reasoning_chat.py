from __future__ import annotations

import asyncio
import os

from ollama_orchestra import chat


async def main() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "your-model")
    prompt = os.getenv("OLLAMA_PROMPT", "Summarize why top-level think:false matters.")

    result = await chat(
        base_url,
        model,
        [{"role": "user", "content": prompt}],
        think=False,
        num_predict=256,
    )
    print(result.get("message", {}).get("content", result))


if __name__ == "__main__":
    asyncio.run(main())
