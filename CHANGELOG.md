# Changelog

All notable changes to Cerebro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.3] - 2026-02-21

### Fixed
- MCP tool hang: per-call `ThreadPoolExecutor(max_workers=1)` replaces shared default executor — prevents thread pool starvation from orphan threads left by timed-out calls
- SQLite thread safety: added `check_same_thread=False` to keyword index connection — fixes crash when search runs in different thread than init
- DGX empty result fallback: changed `if dgx_result.get("results"):` to `if dgx_result is not None:` — empty list is falsy in Python, was incorrectly triggering 15s+ TensorFlow model load as local fallback
- DGX search/embedding clients: replaced all `loop.run_in_executor(None, ...)` with per-call fresh thread executors
- Added `OMP_NUM_THREADS=2` limit to prevent FAISS/NumPy spawning excessive OpenMP threads in MCP subprocess

## [0.1.2] - 2026-02-19

### Fixed
- MCP search hang: replaced `ThreadPoolExecutor` with `threading.Thread` in embeddings engine — executor fails in MCP subprocess context with "cannot schedule new futures after interpreter shutdown"
- Replaced `loop.run_in_executor()` with `asyncio.to_thread()` in MCP server async wrapper
- Removed slow NAS keyword search fallback (816MB read over CIFS causing 45s+ timeouts) — returns empty instead

### Added
- `rebuild_keyword_index` MCP tool for rebuilding SQLite FTS5 index on demand
- Auto-populate keyword index on startup if empty (background rebuild)
- Incremental keyword index updates when saving conversations
- `add_chunks()` method on `KeywordIndex` for incremental FTS5 inserts
- Background NAS-to-local-cache sync after loading FAISS from local cache

## [0.1.0] - 2026-02-14

### Added
- Initial open-source release of Cerebro Memory
- MCP server with 49 tools across 10 categories
- 3-tier memory architecture (episodic, semantic, working memory)
- Causal reasoning engine with "what-if" simulation
- Predictive failure anticipation
- Self-awareness module with hallucination detection
- Learning system with auto-promoting patterns
- Session continuity and handoff system
- Privacy-first design with built-in secret detection
- CLI with `cerebro init`, `cerebro serve`, `cerebro doctor`
- Docker deployment support
- Cross-platform support (Windows, Linux, macOS)
- FAISS vector search with sentence-transformers
- Optional GPU acceleration for embeddings
- Redis integration for advanced features

[unreleased]: https://github.com/Professor-Low/Cerebro/compare/v1.5.3...HEAD
[1.5.3]: https://github.com/Professor-Low/Cerebro/compare/v0.1.2...v1.5.3
[0.1.2]: https://github.com/Professor-Low/Cerebro/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/Professor-Low/Cerebro/releases/tag/v0.1.0
