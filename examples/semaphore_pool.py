from __future__ import annotations

import asyncio
import time

from ollama_orchestra import OllamaSemaphorePool, RoundRobinOllama


async def fake_request(pool: OllamaSemaphorePool, url: str, index: int) -> None:
    async with pool.semaphore(url):
        print(f"start task={index} url={url} ts={time.monotonic():.3f}")
        await asyncio.sleep(0.2)
        print(f"end task={index} url={url} ts={time.monotonic():.3f}")


async def main() -> None:
    urls = ["http://gpu-a.local:11434", "http://gpu-b.local:11434"]
    pool = OllamaSemaphorePool(local_hosts={"gpu-a.local", "gpu-b.local"})
    rr = RoundRobinOllama(urls)

    tasks = []
    for index in range(6):
        tasks.append(fake_request(pool, await rr.next_url(), index))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
