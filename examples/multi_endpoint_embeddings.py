from __future__ import annotations

import asyncio
import os

from ollama_orchestra import EmbeddingService


async def main() -> None:
    urls = [
        item.strip()
        for item in os.getenv("OLLAMA_EMBEDDING_URLS", "http://localhost:11434").split(",")
        if item.strip()
    ]
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "your-embedding-model")
    text = os.getenv("OLLAMA_EMBEDDING_TEXT", "This text will be embedded with endpoint fallback.")

    service = EmbeddingService(model=model, urls=urls, alert_cb=lambda msg: print(f"alert: {msg}"))
    try:
        vector = await service.embed_text(text)
        print({"dimensions": len(vector or []), "ok": vector is not None})
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
