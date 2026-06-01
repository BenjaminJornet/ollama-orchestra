from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import pybreaker

from .embeddings import EndpointScore
from .queue import OllamaSemaphorePool
from .reasoning import chat

logger = logging.getLogger(__name__)


class OrchestratedChat:
    """Ollama chat client with concurrency pool, scoring, and reasoning stripping."""

    def __init__(
        self,
        model: str,
        urls: list[str],
        *,
        pool: OllamaSemaphorePool | None = None,
        timeout: float = 120.0,
        alert_cb: Callable[[str], None] | None = None,
        metrics_cb: Callable[[dict], None] | None = None,
        quarantine_seconds: float = 300.0,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if not urls:
            raise ValueError("at least one Ollama URL is required")
        self.model = model
        self.urls = [self._normalize_base_url(url) for url in urls]
        self.pool = pool or OllamaSemaphorePool()
        self.alert_cb = alert_cb
        self.metrics_cb = metrics_cb
        self.quarantine_seconds = quarantine_seconds
        self._timeout = timeout
        self._breakers = {
            url: pybreaker.CircuitBreaker(
                fail_max=3, reset_timeout=30, name=f"ollama_chat_{idx}"
            )
            for idx, url in enumerate(self.urls)
        }
        self._endpoint_scores = {url: EndpointScore() for url in self.urls}
        self._endpoint_last_alert: dict[str, float] = {}
        self._endpoint_down_until: dict[str, float] = {}

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        think: bool = False,
        strip: bool = True,
        **opts: Any,
    ) -> dict[str, Any] | None:
        if not messages:
            return None

        last_error = None
        for url in self._ranked_urls():
            if time.monotonic() < self._endpoint_down_until.get(url, 0.0):
                continue
            try:
                start = time.monotonic()

                async def _call_ollama(endpoint: str = url) -> dict[str, Any]:
                    async with self.pool.semaphore(endpoint):
                        return await chat(
                            endpoint,
                            self.model,
                            messages,
                            think=think,
                            strip=strip,
                            timeout=self._timeout,
                            **opts,
                        )

                guarded = self._breakers[url](_call_ollama)
                response = await guarded()
                if response:
                    self._record_endpoint_success(url, time.monotonic() - start)
                    self._emit_metric({"event": "chat_success", "url": url})
                    return response
                logger.warning("no_response_from_chat url=%s", url)
                return None
            except pybreaker.CircuitBreakerError:
                logger.warning("chat_circuit_open url=%s", url)
                self._record_endpoint_failure(url)
                self._emit_metric({"event": "chat_circuit_open", "url": url})
                continue
            except Exception as exc:
                last_error = str(exc)
                logger.warning("chat_request_failed url=%s error=%s", url, exc)
                self._record_endpoint_failure(url)
                self._emit_metric(
                    {"event": "chat_failure", "url": url, "error": str(exc)}
                )
                self._alert_endpoint_down(url, str(exc))
                continue

        self._alert(f"All Ollama chat endpoints failed. Last error: {last_error}")
        return None

    def _ranked_urls(self) -> list[str]:
        return sorted(self.urls, key=lambda url: self._endpoint_scores[url].value, reverse=True)

    def _record_endpoint_success(self, url: str, latency: float) -> None:
        score = self._endpoint_scores[url]
        score.record_success(latency)
        self._emit_metric(
            {
                "event": "endpoint_score_updated",
                "url": url,
                "score": score.value,
                "successes": score.successes,
                "failures": score.failures,
                "avg_latency": score.avg_latency,
            }
        )

    def _record_endpoint_failure(self, url: str) -> None:
        score = self._endpoint_scores[url]
        score.record_failure()
        self._emit_metric(
            {
                "event": "endpoint_score_updated",
                "url": url,
                "score": score.value,
                "successes": score.successes,
                "failures": score.failures,
                "avg_latency": score.avg_latency,
            }
        )

    def _alert_endpoint_down(self, url: str, error: str) -> None:
        now = time.monotonic()
        last_alert = self._endpoint_last_alert.get(url)
        if last_alert is not None and now - last_alert < 1800:
            return
        self._endpoint_last_alert[url] = now
        self._endpoint_down_until[url] = now + self.quarantine_seconds
        self._emit_metric(
            {
                "event": "chat_endpoint_quarantined",
                "url": url,
                "seconds": self.quarantine_seconds,
            }
        )
        self._alert(f"Ollama chat endpoint unavailable: {url}. Error: {error[:200]}")

    def _alert(self, message: str) -> None:
        if not self.alert_cb:
            return
        try:
            self.alert_cb(message)
        except Exception:
            logger.exception("chat_alert_callback_failed")

    def _emit_metric(self, event: dict) -> None:
        if not self.metrics_cb:
            return
        try:
            self.metrics_cb(event)
        except Exception:
            logger.exception("chat_metrics_callback_failed")

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        base = url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return base
