from __future__ import annotations

import asyncio

from ollama_orchestra import OllamaSemaphorePool, RoundRobinOllama, TextChunker, strip_reasoning


async def main() -> None:
    pool = OllamaSemaphorePool(local_hosts={"gpu-a.local"})
    assert pool.is_local_ollama_url("http://gpu-a.local:8080")
    async with pool.semaphore("http://gpu-a.local:8080"):
        pass

    rr = RoundRobinOllama(["a", "b"])
    assert [await rr.next_url(), await rr.next_url(), await rr.next_url()] == ["a", "b", "a"]

    chunks = TextChunker(min_size=1, target_size=20).chunk_text("hello world")
    assert len(chunks) == 1
    assert strip_reasoning("<think>x</think>answer") == "answer"
    print("smoke ok")


if __name__ == "__main__":
    asyncio.run(main())
