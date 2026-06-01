# Production Patterns

`ollama-orchestra` packages patterns that show up when Ollama moves from a single-request demo to concurrent ingestion or agent workflows.

## One request per local GPU endpoint

Local Ollama servers often run on a single GPU. Sending many concurrent requests to one endpoint usually increases latency instead of throughput because work queues inside the server.

Use `OllamaSemaphorePool` to serialize requests per local endpoint:

```python
pool = OllamaSemaphorePool()

async with pool.semaphore("http://localhost:11434"):
    ...
```

## Round-robin across endpoints

If you have multiple local endpoints, distribute chunks across them while keeping each endpoint serialized:

```python
rr = RoundRobinOllama(["http://gpu-a.local:11434", "http://gpu-b.local:11434"])
url = await rr.next_url()
```

## Prewarm before batch jobs

The first request can be slow while a model loads into memory. Prewarm before ingestion jobs:

```python
await prewarm_all_servers(urls, model="your-model")
```

## Embedding fallback and quarantine

Embedding endpoints can fail due to cold starts, memory pressure, or restarts. `EmbeddingService` uses:

- per-endpoint circuit breakers
- fallback to the next endpoint
- temporary quarantine for endpoints that just failed
- optional alert callback

## Orchestrated Chat with fallback and scoring

To route chat and reasoning requests across multiple endpoints safely, use `OrchestratedChat`. It handles round-robin routing, serializes concurrent requests per GPU using `OllamaSemaphorePool`, checks circuit breakers, quarantines failing endpoints, and ranks URLs dynamically based on success rates and latencies (endpoint scoring):

```python
from ollama_orchestra import OrchestratedChat

chat_service = OrchestratedChat(
    model="your-reasoning-model",
    urls=["http://gpu-a.local:11434", "http://gpu-b.local:11434"],
    quarantine_seconds=300,
)

response = await chat_service.chat([{"role": "user", "content": "hi"}], think=False)
```

## Maintenance roadmap

- Metrics callback hooks.
- Adaptive concurrency based on latency.
- Streaming chat helpers.
- More health-check variants for gateways.
