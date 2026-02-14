# Contributing to Cerebro

Thank you for your interest in contributing to Cerebro! This guide will help you get started.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/Professor-Low/Cerebro/issues)
2. If not, create a new issue using the [Bug Report template](https://github.com/Professor-Low/Cerebro/issues/new?template=bug_report.yml)
3. Include as much detail as possible: OS, Python version, steps to reproduce

### Suggesting Features

1. Check [existing feature requests](https://github.com/Professor-Low/Cerebro/issues?q=label%3Aenhancement)
2. Create a new issue using the [Feature Request template](https://github.com/Professor-Low/Cerebro/issues/new?template=feature_request.yml)

### Pull Requests

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/Cerebro.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes** following the conventions below
5. **Test** your changes: `pytest`
6. **Commit** with a conventional message (see below)
7. **Push** to your fork: `git push origin feature/your-feature-name`
8. **Open a Pull Request** against `main`

### Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new MCP tool for timeline visualization
fix: resolve memory leak in embedding cache
docs: update deployment guide for Docker
refactor: simplify causal reasoning engine
test: add tests for session handoff
chore: update dependencies
```

### Development Setup

```bash
# Clone
git clone https://github.com/Professor-Low/Cerebro.git
cd Cerebro

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/

# Run the MCP server locally
cerebro serve
```

## Architecture Overview

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system architecture.

Cerebro Memory is an MCP server providing 49 tools across 10 categories:
- **Core Memory** — Save, search, and retrieve conversations
- **Knowledge Base** — Facts, entities, and semantic knowledge
- **Session Continuity** — Handoff and resume across sessions
- **Reasoning** — Causal models, predictions, insights
- **Learning** — Solutions, antipatterns, skill tracking
- **Self-Awareness** — Confidence, introspection, hallucination detection
- **User Profile** — Preferences, goals, personality evolution
- **Quality** — Deduplication, decay, consolidation
- **Privacy** — Secret detection, data redaction
- **Meta-Learning** — Strategy optimization, A/B testing

## Style Guide

- **Python**: Follow PEP 8, use type hints, docstrings for public APIs
- **Commits**: Conventional Commits format
- **PRs**: One feature/fix per PR, include tests when applicable

## License

By contributing, you agree that your contributions will be licensed under the [AGPL-3.0 License](LICENSE).
