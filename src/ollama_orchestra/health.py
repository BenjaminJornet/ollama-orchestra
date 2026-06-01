from __future__ import annotations

import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)


async def check_server_health(base_url: str, *, timeout: float = 5.0) -> bool:
    """Return True when an Ollama server responds to `GET /api/tags`."""
    url = base_url.rstrip("/").removesuffix("/v1")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/api/tags")
        ok = response.status_code == 200
        logger.info("ollama_health url=%s ok=%s status=%d", url, ok, response.status_code)
        return ok
    except Exception as exc:
        logger.warning("ollama_health_unreachable url=%s error=%s", url, exc)
        return False


async def prewarm_model(
    base_url: str,
    model: str,
    *,
    keep_alive: str = "30m",
    timeout: float = 30.0,
) -> bool:
    """Pre-load a model on one Ollama server using an empty generation request."""
    url = base_url.rstrip("/").removesuffix("/v1")
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{url}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": keep_alive},
            )
        elapsed = round(time.monotonic() - start, 1)
        ok = response.status_code == 200
        logger.info("ollama_prewarm url=%s model=%s ok=%s elapsed_s=%s", url, model, ok, elapsed)
        return ok
    except Exception as exc:
        logger.warning("ollama_prewarm_error url=%s model=%s error=%s", url, model, exc)
        return False


async def prewarm_all_servers(
    urls: list[str],
    model: str,
    *,
    keep_alive: str = "30m",
) -> dict[str, bool]:
    """Prewarm a model on every configured server concurrently."""
    results = await asyncio.gather(
        *[prewarm_model(url, model, keep_alive=keep_alive) for url in urls],
        return_exceptions=True,
    )
    status: dict[str, bool] = {}
    for url, result in zip(urls, results, strict=False):
        status[url] = False if isinstance(result, Exception) else bool(result)
    return status
