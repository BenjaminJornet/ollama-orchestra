# Security Policy

## Supported versions

Only the latest minor release receives security fixes while the project is pre-1.0.

## Reporting a vulnerability

Please open a private GitHub security advisory if available, or contact the maintainer through GitHub.

## Security notes

- Do not put API keys or credentials in endpoint URLs.
- Treat model prompts and outputs as untrusted data.
- Avoid logging full prompts when they may contain sensitive text.
- Configure local hostnames explicitly instead of hardcoding private infrastructure details.
