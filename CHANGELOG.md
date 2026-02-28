# Changelog

All notable changes to Cerebro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.1.0] - 2026-02-27

### Fixed
- `get_chunk` tool crashes with `NameError: AI_MEMORY_PATH` — replaced with `AI_MEMORY_BASE` (already imported from config)
- Docker entrypoint reads wrong env var `AI_MEMORY_PATH` instead of `CEREBRO_DATA_DIR`, defaults to `/data/memory` instead of `/data`
- Docker entrypoint directory list mismatched with `config.py:ensure_directories()` — now creates the same directories Python expects
- `config/mcp-desktop.json` used broken Docker config with incorrect env vars — replaced with native `cerebro serve` config

## [3.0.0] - 2026-02-27

### Fixed
- 11 MCP tool handlers crash with `slice(None, 10, None)` when NAS is unreachable — dict error responses from `do_search()`/`do_find()` were being iterated and sliced as if they were lists
- Affected tools: `search_knowledge_base`, `find_file_paths`, `semantic_search`, `hybrid_search`, `get_rag_context`, `find_code`, `find_duplicate_memories`, `search_images`, `get_active_sessions`, `find_solution`, `find_antipatterns`
- All handlers now short-circuit with a clean error response when the underlying function returns an error dict

## [2.0.0] - 2026-02-22

### Changed
- Complete rewrite as `cerebro-ai` PyPI package

### Added
- Self-evolving memory with confidence scoring and fact provenance
- Working memory for reasoning continuity across compactions
- Causal modeling and predictive simulation
- Meta-learning for retrieval strategy optimization
- Active memory consolidation (clustering, abstraction, pruning)
- Multi-device support with device tagging
- Session handoff system
- Goal tracking and proactive context
- Privacy/secret scanning
- Episodic vs semantic memory separation

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

[unreleased]: https://github.com/Professor-Low/Cerebro/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/Professor-Low/Cerebro/compare/v1.5.3...v2.0.0
[1.5.3]: https://github.com/Professor-Low/Cerebro/compare/v0.1.2...v1.5.3
[0.1.2]: https://github.com/Professor-Low/Cerebro/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/Professor-Low/Cerebro/releases/tag/v0.1.0
