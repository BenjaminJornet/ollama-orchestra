# ollama-orchestra

[![CI](https://github.com/BenjaminJornet/ollama-orchestra/actions/workflows/ci.yml/badge.svg)](https://github.com/BenjaminJornet/ollama-orchestra/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ollama-orchestra.svg)](https://pypi.org/project/ollama-orchestra/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Production helpers for running Ollama under concurrent load.

Ollama is excellent for local models, but production pipelines quickly hit coordination problems: one GPU should usually receive one request at a time, multi-GPU ingestion needs endpoint rotation, embedding endpoints need fallback, and reasoning models may burn their token budget before producing visible content.

`ollama-orchestra` packages those patterns into small async utilities.

## Install

```bash
uv add ollama-orchestra
```

## Concurrency control

```python
from ollama_orchestra import OllamaSemaphorePool, RoundRobinOllama

pool = OllamaSemaphorePool(local_hosts={"gpu-a.local", "gpu-b.local"})
rr = RoundRobinOllama(["http://gpu-a.local:11434", "http://gpu-b.local:11434"])

url = await rr.next_url()
async with pool.semaphore(url):
    # Call your Ollama client here. Local Ollama endpoints default to 1 slot.
    ...
```

Ports `11434` are treated as local Ollama endpoints by default. Other URLs default to higher concurrency for OpenAI-compatible gateways or cloud APIs.

## Reasoning models gotcha

Some Ollama reasoning models can spend the whole `num_predict` budget inside hidden reasoning and return an empty visible message with `done_reason: "length"`.

Ollama expects `think: false` at the top level of the request body, not inside `options`.

```python
from ollama_orchestra import chat

result = await chat(
    "http://localhost:11434",
    "your-model",
    [{"role": "user", "content": "Summarize this log"}],
    think=False,
    num_predict=256,
)
```

The helper also strips leftover `<think>`, `<reasoning>`, `<thought>`, and simple Markdown fences from returned content by default.

## Embeddings with fallback

```python
from ollama_orchestra import EmbeddingService

service = EmbeddingService(
    model="your-embedding-model",
    urls=["http://gpu-a.local:11434", "http://gpu-b.local:11434"],
)

vector = await service.embed_text("Long text is chunked and mean-pooled automatically.")
await service.close()
```

Features:

- endpoint fallback
- per-endpoint circuit breakers
- temporary quarantine for failing endpoints
- optional alert callback
- long-text chunking and mean pooling

## Health and prewarm

```python
from ollama_orchestra import check_server_health, prewarm_all_servers

healthy = await check_server_health("http://localhost:11434")
status = await prewarm_all_servers(["http://localhost:11434"], model="your-model")
```

## Documentation and examples

- `docs/reasoning-models.md` explains Ollama's top-level `think: false` gotcha.
- `docs/production-patterns.md` documents concurrency, round-robin, prewarm, and fallback patterns.
- `examples/reasoning_chat.py` calls Ollama chat with reasoning disabled.
- `examples/multi_endpoint_embeddings.py` demonstrates embedding fallback across endpoints.
- `examples/semaphore_pool.py` demonstrates per-endpoint concurrency control.

## Roadmap

- Adaptive concurrency based on latency and endpoint health.
- Endpoint scoring for embedding workloads.
- Streaming chat helper.
- Additional gateway-compatible health checks.

## Metrics hooks

Both semaphore and embedding workflows accept optional callbacks for lightweight instrumentation:

```python
events = []
pool = OllamaSemaphorePool(metrics_cb=events.append)
service = EmbeddingService("your-embedding-model", ["http://localhost:11434"], metrics_cb=events.append)
```

Events are dictionaries with an `event` key, such as `semaphore_acquired`, `embedding_failure`, or `embedding_endpoint_quarantined`.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run pytest
uv run python scripts/smoke.py
uv build
```

## License

MIT
