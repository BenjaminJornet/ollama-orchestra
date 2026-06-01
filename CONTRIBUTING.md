# Contributing

Thanks for helping improve `ollama-orchestra`.

## Development setup

```bash
uv sync --dev
uv run ruff check .
uv run pytest
```

## Contribution guidelines

- Keep helpers async-first.
- Keep defaults safe for local single-GPU Ollama servers.
- Add tests for concurrency, fallback, timeouts, and error paths.
- Do not hardcode hostnames, private IPs, model names, or credentials.
- Prefer explicit constructor parameters over environment-specific settings.

## Useful PRs

- Metrics hooks for semaphore and embedding workflows.
- Additional health checks for OpenAI-compatible gateways.
- Adaptive concurrency policies based on latency or failures.
