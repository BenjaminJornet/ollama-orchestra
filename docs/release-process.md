# Release Process

## Checklist

1. Run `uv run ruff check .`.
2. Run `uv run pytest`.
3. Run `uv build`.
4. Run local examples when Ollama is available.
5. Update `CHANGELOG.md`.
6. Tag the release.
7. Create a GitHub release.
8. Publish through the manual `Publish to PyPI` workflow.

## Optional local smoke checks

```bash
uv run python examples/semaphore_pool.py
OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=your-model uv run python examples/reasoning_chat.py
OLLAMA_EMBEDDING_URLS=http://localhost:11434 OLLAMA_EMBEDDING_MODEL=your-embedding-model uv run python examples/multi_endpoint_embeddings.py
```

The Ollama examples require local models chosen by the user. Do not hardcode model names in release artifacts.
