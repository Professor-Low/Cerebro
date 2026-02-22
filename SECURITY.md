# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via [GitHub Security Advisories](https://github.com/Professor-Low/Cerebro/security/advisories/new).

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include:
- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting)
- Full paths of source file(s) related to the issue
- Location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue and how an attacker might exploit it

## Preferred Languages

We prefer all communications to be in English.

## Security Design Principles

Cerebro is designed with security as a core principle:

- **Local-first**: All data stays on your machine by default
- **No phone home**: Zero telemetry, no external data collection
- **Secret detection**: Built-in scanning prevents accidental credential storage
- **Sandboxed execution**: Agent actions require explicit approval levels
- **Encrypted at rest**: Optional encryption for stored conversations and embeddings
