# Changelog

## Unreleased

## 0.1.3

- Added metrics callbacks for semaphore and embedding workflows.

## 0.1.2

- Added maintainer documentation, security policy, release process, examples, and issue/PR templates.
- Added production-pattern and reasoning-model documentation.
- Added build and manual publish workflows.
- Fixed first endpoint alert cooldown behavior on low-uptime CI runners.
- Added PyPI metadata, badges, and typed package marker.

## 0.1.0

Initial public release.

- Added per-endpoint Ollama semaphore pool.
- Added async round-robin endpoint dispatcher.
- Added text chunking and embedding fallback service.
- Added reasoning-model helper with top-level `think: false` support.
- Added health and prewarm helpers.
- Added tests and CI.
