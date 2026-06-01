# Release Process

## Checklist

1. Merge feature PRs to `main` after CI, build, and smoke checks pass.
2. Run `python scripts/prepare-release.py X.Y.Z`.
3. Review `git diff` and commit the release prep.
4. Push `main` and wait for CI, build, and smoke checks to pass.
5. Run `bash scripts/create-release.sh X.Y.Z`.
6. Confirm the `Publish to PyPI` workflow succeeds.
7. Run `bash scripts/run-pypi-install-check.sh`.

## Optional local smoke checks

```bash
bash scripts/validate-release.sh
uv run python examples/semaphore_pool.py
OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=your-model uv run python examples/reasoning_chat.py
OLLAMA_EMBEDDING_URLS=http://localhost:11434 OLLAMA_EMBEDDING_MODEL=your-embedding-model uv run python examples/multi_endpoint_embeddings.py
```

The Ollama examples require local models chosen by the user. Do not hardcode model names in release artifacts.
