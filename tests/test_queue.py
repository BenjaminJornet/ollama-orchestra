from __future__ import annotations

import asyncio

import pytest

from ollama_orchestra.queue import OllamaSemaphorePool, RoundRobinOllama


@pytest.mark.asyncio
async def test_local_ollama_url_allows_one_concurrent_slot():
    pool = OllamaSemaphorePool()
    inside = 0
    max_inside = 0

    async def worker():
        nonlocal inside, max_inside
        async with pool.semaphore("http://localhost:11434"):
            inside += 1
            max_inside = max(max_inside, inside)
            await asyncio.sleep(0.01)
            inside -= 1

    await asyncio.gather(worker(), worker(), worker())

    assert max_inside == 1


@pytest.mark.asyncio
async def test_cloud_url_uses_higher_concurrency_limit():
    pool = OllamaSemaphorePool(cloud_limit=8)
    sem = await pool.get_semaphore("https://api.example.test/v1")

    assert sem._value == 8


@pytest.mark.asyncio
async def test_injected_local_hosts_are_treated_as_local():
    pool = OllamaSemaphorePool(local_hosts={"gpu-box.local"})

    assert pool.is_local_ollama_url("http://gpu-box.local:8080") is True


@pytest.mark.asyncio
async def test_round_robin_cycles_urls():
    rr = RoundRobinOllama(["http://a", "http://b"])

    assert [await rr.next_url(), await rr.next_url(), await rr.next_url()] == [
        "http://a",
        "http://b",
        "http://a",
    ]


@pytest.mark.asyncio
async def test_round_robin_requires_configuration():
    rr = RoundRobinOllama()

    with pytest.raises(RuntimeError):
        await rr.next_url()
