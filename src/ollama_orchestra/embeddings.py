from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import httpx
import pybreaker

from .chunking import TextChunk, TextChunker

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Ollama embedding client with endpoint fallback, quarantine, and mean pooling."""

    MAX_EMBEDDING_TEXT_LENGTH = 1500

    def __init__(
        self,
        model: str,
        urls: list[str],
        *,
        timeout: float = 300.0,
        connect_timeout: float = 10.0,
        alert_cb: Callable[[str], None] | None = None,
        metrics_cb: Callable[[dict], None] | None = None,
        quarantine_seconds: float = 300.0,
        chunker: TextChunker | None = None,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if not urls:
            raise ValueError("at least one Ollama URL is required")
        self.model = model
        self.urls = [self._normalize_base_url(url) for url in urls]
        self.alert_cb = alert_cb
        self.metrics_cb = metrics_cb
        self.quarantine_seconds = quarantine_seconds
        self.chunker = chunker or TextChunker()
        self._timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self._client: httpx.AsyncClient | None = None
        self._breakers = {
            url: pybreaker.CircuitBreaker(fail_max=3, reset_timeout=30, name=f"ollama_{idx}")
            for idx, url in enumerate(self.urls)
        }
        self._endpoint_last_alert: dict[str, float] = {}
        self._endpoint_down_until: dict[str, float] = {}

    async def embed_text(self, text: str) -> list[float] | None:
        if not text or not text.strip():
            return None
        clean_text = text.strip()
        if len(clean_text) <= self.MAX_EMBEDDING_TEXT_LENGTH:
            return await self._embed_single_chunk(clean_text)
        return await self._embed_long_text_with_pooling(clean_text)

    async def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 1,
    ) -> list[list[float] | None]:
        if not texts:
            return []
        semaphore = asyncio.Semaphore(batch_size)

        async def _embed(index: int, text: str) -> tuple[int, list[float] | None]:
            async with semaphore:
                return index, await self.embed_text(text)

        completed = await asyncio.gather(
            *[_embed(index, text) for index, text in enumerate(texts)],
            return_exceptions=True,
        )
        indexed: list[tuple[int, list[float] | None]] = []
        for item in completed:
            if isinstance(item, Exception):
                logger.error("batch_embedding_error error=%s", item)
                indexed.append((len(indexed), None))
            else:
                indexed.append(item)
        indexed.sort(key=lambda pair: pair[0])
        return [embedding for _, embedding in indexed]

    async def embed_chunks(
        self,
        chunks: list[TextChunk],
        *,
        batch_size: int = 1,
    ) -> list[tuple[TextChunk, list[float] | None]]:
        if not chunks:
            return []
        semaphore = asyncio.Semaphore(batch_size)

        async def _embed(index: int, chunk: TextChunk) -> tuple[int, list[float] | None]:
            async with semaphore:
                return index, await self._embed_single_chunk(chunk.text)

        completed = await asyncio.gather(
            *[_embed(index, chunk) for index, chunk in enumerate(chunks)],
            return_exceptions=True,
        )
        indexed: list[tuple[int, list[float] | None]] = []
        for item in completed:
            if isinstance(item, Exception):
                logger.error("batch_chunk_embedding_error error=%s", item)
                indexed.append((len(indexed), None))
            else:
                indexed.append(item)
        indexed.sort(key=lambda pair: pair[0])
        embeddings = [embedding for _, embedding in indexed]
        return list(zip(chunks, embeddings, strict=False))

    async def chunk_and_embed(
        self,
        text: str,
        *,
        include_overlap: bool = True,
    ) -> list[tuple[TextChunk, list[float] | None]]:
        chunks = self.chunker.chunk_text(text, include_overlap=include_overlap)
        return await self.embed_chunks(chunks) if chunks else []

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _embed_single_chunk(self, text: str) -> list[float] | None:
        client = await self._get_client()
        last_error = None
        for url in self.urls:
            if time.monotonic() < self._endpoint_down_until.get(url, 0.0):
                continue
            try:

                async def _call_ollama(endpoint: str = url) -> list[float] | None:
                    response = await client.post(
                        f"{endpoint}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return data.get("embedding")
                    raise RuntimeError(f"HTTP {response.status_code}")

                guarded = self._breakers[url](_call_ollama)
                embedding = await guarded()
                if embedding:
                    self._emit_metric({"event": "embedding_success", "url": url})
                    return embedding
                logger.warning("no_embedding_in_response url=%s", url)
                return None
            except pybreaker.CircuitBreakerError:
                logger.warning("embedding_circuit_open url=%s", url)
                self._emit_metric({"event": "embedding_circuit_open", "url": url})
                continue
            except Exception as exc:
                last_error = str(exc)
                logger.warning("embedding_request_failed url=%s error=%s", url, exc)
                self._emit_metric(
                    {"event": "embedding_failure", "url": url, "error": str(exc)}
                )
                self._alert_endpoint_down(url, str(exc))
                continue

        self._alert(f"All Ollama embedding endpoints failed. Last error: {last_error}")
        return None

    async def _embed_long_text_with_pooling(self, text: str) -> list[float] | None:
        chunks = self.chunker.chunk_text(text, include_overlap=True)
        safe_chunks: list[TextChunk] = []
        for chunk in chunks:
            safe_chunks.extend(
                self.chunker.split_chunk_to_max_length(chunk, self.MAX_EMBEDDING_TEXT_LENGTH)
            )

        chunk_embeddings = await self.embed_chunks(safe_chunks)
        valid_vectors = [embedding for _, embedding in chunk_embeddings if embedding is not None]
        if not valid_vectors:
            return None

        dimension = len(valid_vectors[0])
        mean_vector = [0.0] * dimension
        vector_count = 0
        for vector in valid_vectors:
            if len(vector) != dimension:
                logger.warning(
                    "embedding_dimension_mismatch expected=%d actual=%d",
                    dimension,
                    len(vector),
                )
                continue
            vector_count += 1
            for index, value in enumerate(vector):
                mean_vector[index] += value

        if vector_count == 0:
            return None
        return [value / vector_count for value in mean_vector]

    def _alert_endpoint_down(self, url: str, error: str) -> None:
        now = time.monotonic()
        last_alert = self._endpoint_last_alert.get(url)
        if last_alert is not None and now - last_alert < 1800:
            return
        self._endpoint_last_alert[url] = now
        self._endpoint_down_until[url] = now + self.quarantine_seconds
        self._emit_metric(
            {
                "event": "embedding_endpoint_quarantined",
                "url": url,
                "seconds": self.quarantine_seconds,
            }
        )
        self._alert(f"Ollama embedding endpoint unavailable: {url}. Error: {error[:200]}")

    def _alert(self, message: str) -> None:
        if not self.alert_cb:
            return
        try:
            self.alert_cb(message)
        except Exception:
            logger.exception("embedding_alert_callback_failed")

    def _emit_metric(self, event: dict) -> None:
        if not self.metrics_cb:
            return
        try:
            self.metrics_cb(event)
        except Exception:
            logger.exception("embedding_metrics_callback_failed")

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        base = url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return base
