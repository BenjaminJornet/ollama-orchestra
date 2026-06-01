# Maintainers

## Primary maintainer

- Benjamin Jornet (`@BenjaminJornet`)

## Maintainer responsibilities

- Review changes to concurrency, fallback, and HTTP behavior.
- Triage reports with minimal endpoint configuration and expected behavior.
- Keep examples generic and free of private infrastructure details.
- Maintain release notes and publish tested releases.

## Review priorities

1. No private hostnames, IPs, credentials, or hardcoded model choices.
2. Async behavior must be tested.
3. Concurrency limits should remain safe for local Ollama servers.
4. Error paths should fail open only when documented.
