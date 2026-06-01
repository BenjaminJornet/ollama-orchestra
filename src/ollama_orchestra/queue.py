from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or parsed.path
    port = parsed.port or (443 if scheme == "https" else 80)
    return f"{scheme}://{host}:{port}"


class OllamaSemaphorePool:
    """Per-endpoint async semaphores for local Ollama and higher-throughput APIs."""

    def __init__(
        self,
        local_hosts: set[str] | None = None,
        *,
        local_limit: int = 1,
        cloud_limit: int = 8,
    ) -> None:
        self.local_hosts = set(local_hosts or set())
        self.local_limit = local_limit
        self.cloud_limit = cloud_limit
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    def is_local_ollama_url(self, url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return port == 11434 or host in self.local_hosts

    async def get_semaphore(self, base_url: str) -> asyncio.Semaphore:
        key = _normalize_url(base_url)
        if key not in self._semaphores:
            async with self._lock:
                if key not in self._semaphores:
                    limit = (
                        self.local_limit
                        if self.is_local_ollama_url(base_url)
                        else self.cloud_limit
                    )
                    self._semaphores[key] = asyncio.Semaphore(limit)
                    logger.info("ollama_semaphore_created url=%s concurrency=%d", key, limit)
        return self._semaphores[key]

    @asynccontextmanager
    async def semaphore(self, base_url: str) -> AsyncIterator[None]:
        sem = await self.get_semaphore(base_url)
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()


class RoundRobinOllama:
    """Async-safe round-robin URL dispatcher."""

    def __init__(self, urls: list[str] | None = None) -> None:
        self._urls: list[str] = urls or []
        self._cycle = itertools.cycle(self._urls) if self._urls else None
        self._lock = asyncio.Lock()

    def configure(self, urls: list[str]) -> None:
        self._urls = list(urls)
        self._cycle = itertools.cycle(self._urls) if self._urls else None
        logger.info("round_robin_configured urls=%s parallelism=%d", self._urls, len(self._urls))

    @property
    def parallelism(self) -> int:
        return len(self._urls)

    @property
    def urls(self) -> list[str]:
        return list(self._urls)

    async def next_url(self) -> str:
        if not self._cycle:
            raise RuntimeError("RoundRobinOllama is not configured; call configure() first")
        async with self._lock:
            return next(self._cycle)
