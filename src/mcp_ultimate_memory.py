#!/usr/bin/env python3
"""
Cerebro Memory - MCP Server
Copyright (C) 2026 Michael Lopez (Professor-Low)
Licensed under AGPL-3.0 — see LICENSE for details.

MCP Server for Ultimate AI Memory System with RAG
Exposes comprehensive memory capabilities via Model Context Protocol

OPTIMIZED FOR SPEED:
- Lazy loading of services (only load what's needed)
- Pre-initialized at startup for instant response
- Shorter timeouts with fast fallbacks
- No blocking operations in the main async loop

SANITIZATION:
- All responses are sanitized to prevent terminal rendering issues
- Newlines replaced with " | " in content fields
- Content truncated to reasonable lengths
- Special characters escaped
"""
import os
import sys

# ============================================================
# CRITICAL: Redirect stdout to stderr BEFORE any imports.
# The MCP protocol uses stdout for JSON-RPC messages.
# Any print() to stdout corrupts the protocol stream and
# causes Claude Code MCP calls to hang indefinitely.
# ============================================================
_original_stdout = sys.stdout  # Save for MCP protocol
sys.stdout = sys.stderr        # All print() now goes to stderr

# Embeddings enabled — uses DGX Spark for embedding generation (lazy-loaded, won't block startup)
os.environ.setdefault('ENABLE_EMBEDDINGS', '1')
os.environ.setdefault('CEREBRO_DGX_HOST', '')
os.environ.setdefault('DGX_EMBEDDING_HOST', '')
os.environ.setdefault('DGX_EMBEDDING_PORT', '8781')
os.environ.setdefault('CEREBRO_DEVICE', 'auto')

import asyncio
import concurrent.futures
import datetime as dt
import json
import re
import uuid
from functools import partial
from pathlib import Path

# Ensure bare imports resolve to this package directory
# (needed when running as an installed package via pip)
_pkg_dir = str(Path(__file__).parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Device registry for multi-device tagging
try:
    from .device_registry import get_device_metadata
except ImportError:
    from device_registry import get_device_metadata

# Cross-platform configuration
try:
    from .config import AI_MEMORY_BASE, get_base_path
    from .config import NAS_IP as _CONFIG_NAS_IP
except ImportError:
    from config import AI_MEMORY_BASE, get_base_path
    from config import NAS_IP as _CONFIG_NAS_IP

# ============== CONTENT SANITIZATION ==============
MAX_CONTENT_LENGTH = 500  # Max chars per content field
MAX_CONTEXT_LENGTH = 200  # Max chars for context fields
MAX_RESULTS = 10  # Max results to return


def sanitize_text(text: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """
    Sanitize text content to prevent terminal rendering issues.
    - Replace newlines with " | "
    - Remove control characters
    - Truncate to max length
    """
    if not text:
        return ""

    # Replace newlines and tabs with " | "
    text = re.sub(r'[\n\r\t]+', ' | ', text)

    # Remove other control characters (but keep basic printable chars)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Trim whitespace
    text = text.strip()

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length - 3] + "..."

    return text


def sanitize_result(result: dict) -> dict:
    """Sanitize a search result dictionary."""
    sanitized = result.copy()

    # Sanitize content field
    if "content" in sanitized:
        sanitized["content"] = sanitize_text(sanitized["content"], MAX_CONTENT_LENGTH)

    # Sanitize context field
    if "context" in sanitized:
        sanitized["context"] = sanitize_text(sanitized["context"], MAX_CONTEXT_LENGTH)

    # Sanitize context_before field
    if "context_before" in sanitized:
        sanitized["context_before"] = sanitize_text(sanitized["context_before"], MAX_CONTEXT_LENGTH)

    return sanitized


def sanitize_results(results: list) -> list:
    """Sanitize a list of search results."""
    # Limit number of results
    results = results[:MAX_RESULTS]
    return [sanitize_result(r) for r in results]


def safe_json_dumps(obj: dict, indent: int = 2) -> str:
    """
    Safe JSON serialization that ensures clean output.
    Uses ensure_ascii=True to escape all non-ASCII characters.
    """
    return json.dumps(obj, indent=indent, ensure_ascii=True)

# Initialize MCP server
server = Server("cerebro-memory")

# Thread pool for blocking I/O operations - more workers to handle stuck threads
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# SHORTER timeout for NAS operations - fail fast!
NAS_TIMEOUT = 30

# NAS configuration for socket-based reachability check
NAS_IP = _CONFIG_NAS_IP or os.environ.get("CEREBRO_NAS_IP", "")
NAS_SMB_PORT = 445  # SMB port


def is_nas_reachable(timeout: float = 2.0) -> bool:
    """
    Fast NAS reachability check using socket + filesystem test.
    Socket alone isn't enough - filesystem can hang even when socket works.
    Returns True if NAS is reachable AND filesystem responds, False otherwise.
    """
    import socket
    import threading

    # Step 1: Socket check (fast network test)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((NAS_IP, NAS_SMB_PORT))
        sock.close()
        if result != 0:
            return False
    except Exception:
        return False

    # Step 2: Actual filesystem test with threading timeout
    # This catches cases where socket works but SMB file ops hang
    fs_result = [False]

    def check_filesystem():
        try:
            fs_result[0] = AI_MEMORY_BASE.exists()
        except Exception:
            fs_result[0] = False

    thread = threading.Thread(target=check_filesystem, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Filesystem check didn't complete in time - NAS is slow/hanging
        return False

    return fs_result[0]

# Pre-initialized services (loaded at startup)
_memory = None
_embeddings = None
_initialized = False
_init_event = None  # Created in main() when event loop is running

# DGX embedding integration - use GPU for embedding generation without blocking MCP startup
_dgx_embedding_available = None  # Cached availability check


def is_dgx_embedding_available_cached() -> bool:
    """Check if DGX embedding service is available (cached for performance)."""
    global _dgx_embedding_available
    if _dgx_embedding_available is not None:
        return _dgx_embedding_available
    try:
        from dgx_embedding_client import is_dgx_embedding_available_sync
        _dgx_embedding_available = is_dgx_embedding_available_sync()
    except ImportError:
        _dgx_embedding_available = False
    return _dgx_embedding_available


def embed_chunks_via_dgx(chunks: list, batch_size: int = 128) -> list:
    """
    Generate embeddings for chunks using DGX GPU service.
    Returns chunks with embeddings attached, or original chunks if DGX fails.
    """
    if not is_dgx_embedding_available_cached():
        return chunks  # Return without embeddings

    try:
        import datetime

        from dgx_embedding_client import dgx_embed_sync

        texts = [chunk["content"] for chunk in chunks]
        embeddings = dgx_embed_sync(texts, batch_size=batch_size)

        if embeddings is None:
            return chunks

        # Attach embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
            chunk["embedding_model"] = "all-mpnet-base-v2"
            chunk["embedding_dim"] = 768
            chunk["embedded_at"] = datetime.datetime.now().isoformat()
            chunk["embedded_via"] = "dgx"

        return chunks
    except Exception as e:
        print(f"[MCP] DGX embedding failed: {e}")
        return chunks  # Return without embeddings


async def wait_for_init(timeout: float = 20.0) -> bool:
    """
    Wait for background initialization to complete.
    Returns True if ready, False if timed out.
    Call this at the start of any tool that needs _memory or _embeddings.
    """
    if _initialized:
        return True
    if _init_event is None:
        # Event not created yet - shouldn't happen but handle gracefully
        await asyncio.sleep(0.1)
        return _initialized
    try:
        await asyncio.wait_for(_init_event.wait(), timeout=timeout)
        return _initialized  # Return actual state, not just "event was set"
    except asyncio.TimeoutError:
        return False


def _init_memory():
    """Initialize memory service (called once at startup)"""
    global _memory
    if _memory is None:
        # Add src to path if needed
        src_path = os.path.dirname(os.path.abspath(__file__))
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from ai_memory_ultimate import UltimateMemoryService
        _memory = UltimateMemoryService(base_path=get_base_path())
    return _memory


def _init_embeddings():
    """Initialize embeddings engine and warm up FAISS cache (called once at startup)"""
    global _embeddings
    if _embeddings is None:
        src_path = os.path.dirname(os.path.abspath(__file__))
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from ai_embeddings_engine import EmbeddingsEngine
        _embeddings = EmbeddingsEngine(base_path=get_base_path())

        # Pre-load FAISS index into memory cache for instant search
        # Runs during background init (_blocking_init via _background_init)
        # so it cannot block MCP startup handshake
        try:
            _embeddings.warmup_cache()
        except Exception as e:
            sys.stderr.write(f"[Embeddings] Cache warmup failed (non-fatal): {e}\n")

    return _embeddings


async def run_in_thread(func, *args, timeout=NAS_TIMEOUT):
    """Run blocking function in thread pool with timeout"""
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, partial(func, *args)),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout}s")


@server.list_tools()
async def list_tools():
    """List all available tools"""
    return [
        Tool(
            name="save_conversation_ultimate",
            description="Save a conversation with COMPREHENSIVE extraction: facts, file paths, user preferences, goals, technical details, entities, actions, decisions, problems/solutions, and code snippets. Creates searchable knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "List of conversation messages with role and content",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"}
                            }
                        }
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session identifier"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata (session_type, issue, priority, etc.)"
                    }
                },
                "required": ["messages"]
            }
        ),
        Tool(
            name="search_knowledge_base",
            description="Search the central knowledge base for facts, learnings, and discoveries. Returns all facts matching the query with their source conversations. By default, superseded (corrected) facts are filtered out.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (searches fact content)"
                    },
                    "fact_type": {
                        "type": "string",
                        "description": "Optional filter by type: technical_limitation, technical_capability, discovery, location, usage, configuration"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum results to return (default 20)"
                    },
                    "include_superseded": {
                        "type": "boolean",
                        "description": "Include superseded (corrected) facts in results. Default false - only active facts are returned."
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_entity_info",
            description="Get information about a specific entity (tool, network, person, etc.) including all conversations where it was mentioned.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "description": "Type of entity: tools, networks, people, servers, technologies",
                        "enum": ["tools", "networks", "people", "servers", "technologies"]
                    },
                    "entity_name": {
                        "type": "string",
                        "description": "Name of the entity (e.g. 'Python', 'NAS', 'Docker')"
                    }
                },
                "required": ["entity_type", "entity_name"]
            }
        ),
        Tool(
            name="get_timeline",
            description="Get timeline of actions and decisions for a specific time period. Shows chronological history of what was done and decided.",
            inputSchema={
                "type": "object",
                "properties": {
                    "year_month": {
                        "type": "string",
                        "description": "Year and month in YYYY-MM format (e.g. '2025-12')"
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Optional filter by event type: action or decision",
                        "enum": ["action", "decision", "all"]
                    }
                },
                "required": ["year_month"]
            }
        ),
        Tool(
            name="find_file_paths",
            description="Find all file paths that were mentioned in conversations, with their purpose and context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path_pattern": {
                        "type": "string",
                        "description": "Pattern to match (e.g. 'claude', 'hooks', 'NAS')"
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Optional filter by purpose: configuration, script, log, data_storage, hook_or_command"
                    }
                },
                "required": ["path_pattern"]
            }
        ),
        Tool(
            name="get_user_context",
            description="Get comprehensive user context: goals, preferences, technical environment. Perfect for understanding the user's overall setup and intentions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Number of recent conversations to analyze (default 10)"
                    }
                }
            }
        ),
        Tool(
            name="get_user_profile",
            description="Get the user's personal profile - everything learned about the user including identity, relationships (pets, family, friends), projects, companies, preferences, and goals. This is the centralized personal knowledge hub.",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter: identity, relationships, projects, preferences, goals, technical_environment, or 'all' (default)",
                        "enum": ["all", "identity", "relationships", "projects", "preferences", "goals", "technical_environment"]
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Include metadata like completeness score and last update timestamp (default: true)"
                    }
                }
            }
        ),
        # CONSOLIDATED: search (replaces semantic_search, hybrid_search, get_rag_context)
        Tool(
            name="search",
            description="Search memory. Modes: hybrid (default, recommended), semantic, rag. Hybrid combines AI + keyword search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "rag"],
                        "description": "Search mode: hybrid (default), semantic, rag"
                    },
                    "top_k": {
                        "type": "number",
                        "description": "Number of results (default 5)"
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Semantic vs keyword weight for hybrid (0-1, default 0.7)"
                    },
                    "chunk_type": {
                        "type": "string",
                        "description": "Filter by type: message, fact, file_path, goal, problem_solution, summary"
                    },
                    "boost_recency": {
                        "type": "boolean",
                        "description": "Boost newer content (default true)"
                    },
                    "recency_days": {
                        "type": "number",
                        "description": "Recency decay half-life in days (default 30)"
                    },
                    "max_tokens": {
                        "type": "number",
                        "description": "Max tokens for results (default 500, use for context-aware retrieval)"
                    },
                    "compact": {
                        "type": "boolean",
                        "description": "Return bullet-point format (default false)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="rebuild_vector_index",
            description="Rebuild the vector search index. Call this after saving multiple conversations or if search results seem outdated. Takes a few seconds to complete.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_corrections",
            description="Get corrections that Claude learned from the user. Use this before answering to avoid repeating known mistakes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional topic filter (network, file_system, configuration, etc.)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional search query"
                    },
                    "importance": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Filter by importance level"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum results (default 10)"
                    }
                }
            }
        ),
        # CONSOLIDATED: projects (replaces get_project_state, get_active_projects, get_stale_projects)
        Tool(
            name="projects",
            description="Get project info. Actions: state (single project), active (list), stale (inactive), auto_update (run status transitions), activity (get activity summary).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["state", "active", "stale", "auto_update", "activity"],
                        "description": "Action: state (single project), active (list), stale (inactive), auto_update (run transitions), activity (summary)"
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Project ID (for action=state)"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "File path to auto-detect project (for action=state)"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "paused", "completed", "all"],
                        "description": "Status filter (for action=active)"
                    },
                    "stale_days": {
                        "type": "number",
                        "description": "Days inactive to consider stale (for action=stale, default 14)"
                    },
                    "days": {
                        "type": "number",
                        "description": "Days to analyze for activity summary (for action=activity, default 7)"
                    }
                }
            }
        ),
        Tool(
            name="trigger_search_visualization",
            description="Trigger real-time search visualization in the 3D brain graph. Shows which nodes were accessed during search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_results": {
                        "type": "array",
                        "description": "Search results to visualize (from semantic_search or hybrid_search)"
                    }
                },
                "required": ["search_results"]
            }
        ),
        # CONSOLIDATED: project_evolution (replaces record/get/supersede/list evolution tools)
        Tool(
            name="project_evolution",
            description="Track project versions. Actions: record, timeline, supersede, list.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["record", "timeline", "supersede", "list"],
                        "description": "Action: record (new version), timeline (history), supersede (mark old), list (all)"
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Project identifier"
                    },
                    "summary": {
                        "type": "string",
                        "description": "What changed (for action=record)"
                    },
                    "version": {
                        "type": "string",
                        "description": "Version label (for action=record)"
                    },
                    "old_version": {
                        "type": "string",
                        "description": "Version being superseded (for action=supersede)"
                    },
                    "new_version": {
                        "type": "string",
                        "description": "Version that supersedes (for action=supersede)"
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords for matching"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why superseded (for action=supersede)"
                    }
                }
            }
        ),
        # CONSOLIDATED: preferences (replaces get_user_preferences, update_user_preference)
        # Phase 2: Added evolution actions (evolved, decay, contradictions, migrate, stats)
        Tool(
            name="preferences",
            description="Get or update user preferences. Actions: get, update, evolved, decay, contradictions, migrate, stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get", "update", "evolved", "decay", "contradictions", "migrate", "stats"],
                        "description": "Action: get (legacy), update (add/reinforce), evolved (weighted prefs), decay (check staleness), contradictions (find conflicts), migrate (legacy->evolved), stats (evolution metrics)"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["communication_style", "workflow", "technical"],
                        "description": "Preference category (for action=update)"
                    },
                    "preference": {
                        "type": "string",
                        "description": "Preference to add (for action=update)"
                    },
                    "positive": {
                        "type": "boolean",
                        "description": "True for 'prefers', False for 'dislikes' (for action=update)"
                    },
                    "include_stale": {
                        "type": "boolean",
                        "description": "Include stale preferences in results (for action=evolved, default false)"
                    }
                }
            }
        ),
        # Phase 3 & 6 - Brain Evolution: Confidence and Provenance tracking for facts
        Tool(
            name="confidence",
            description="Manage fact confidence scores and provenance. Actions: stats, get, provenance (full history), reinforce, decay, low, quarantine, source_chain (origin), corrections (correction history), reinforcements (confirmation history), detect_contradictions, contradiction_stats, resolve_contradiction, batch_resolve, pending_contradictions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["stats", "get", "provenance", "reinforce", "decay", "low", "quarantine", "source_chain", "corrections", "reinforcements", "reinforcement_stats", "highly_reinforced", "detect_contradictions", "contradiction_stats", "resolve_contradiction", "batch_resolve", "pending_contradictions"],
                        "description": "Action: stats (confidence overview), get (get fact confidence), provenance (full history), reinforce (boost fact), decay (apply decay), low (low-confidence facts), quarantine (quarantine low facts), source_chain (fact origin), corrections (correction history), reinforcements (fact's confirmations), reinforcement_stats (system reinforcement stats), highly_reinforced (facts with 3+ confirmations), detect_contradictions (find conflicts), contradiction_stats (contradiction summary), resolve_contradiction (resolve single), batch_resolve (batch resolve), pending_contradictions (list pending)"
                    },
                    "fact_id": {
                        "type": "string",
                        "description": "Fact ID (for action=get, provenance, reinforce, source_chain, corrections, reinforcements)"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Confidence threshold 0-1 (for action=low/quarantine, default 0.5/0.4)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for reinforcement (for action=reinforce)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Fact content to check for contradictions (for action=detect_contradictions)"
                    },
                    "fact_id_a": {
                        "type": "string",
                        "description": "First fact ID (for action=resolve_contradiction)"
                    },
                    "fact_id_b": {
                        "type": "string",
                        "description": "Second fact ID (for action=resolve_contradiction)"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["confirmed", "dismissed"],
                        "description": "Resolution status (for action=resolve_contradiction)"
                    },
                    "resolution": {
                        "type": "string",
                        "description": "Resolution explanation (for action=resolve_contradiction)"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of contradictions to resolve (for action=batch_resolve, default 100)"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["keep_newer", "keep_higher_confidence", "dismiss_all"],
                        "description": "Batch resolution strategy (for action=batch_resolve)"
                    }
                }
            }
        ),
        # CONSOLIDATED: code (replaces find_code, get_code_stats)
        Tool(
            name="code",
            description="Search code snippets or get stats. Actions: search, stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "stats"],
                        "description": "Action: search (find snippets), stats (get statistics)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for action=search)"
                    },
                    "language": {
                        "type": "string",
                        "description": "Filter by language (for action=search)"
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project (for action=search)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Max results (for action=search, default 10)"
                    }
                }
            }
        ),
        # CONSOLIDATED: analyze (replaces detect_recurring_patterns, find_knowledge_gaps, track_skill_development)
        # Phase 4 Enhancement: Added validated_patterns type
        Tool(
            name="analyze",
            description="Analyze patterns, knowledge gaps, or skill development. Types: patterns, knowledge_gaps, skill, validated_patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["patterns", "knowledge_gaps", "skill", "validated_patterns"],
                        "description": "Analysis type: patterns, knowledge_gaps, skill, validated_patterns (validated and ready to apply)"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum occurrences (for patterns/knowledge_gaps, default 3)"
                    },
                    "skill": {
                        "type": "string",
                        "description": "Technology or skill name (for type=skill)"
                    },
                    "max_per_type": {
                        "type": "number",
                        "description": "Max patterns per type (for validated_patterns, default 5)"
                    }
                }
            }
        ),
        # CONSOLIDATED: quality (replaces find_duplicate_memories, merge_duplicates, score_memory_quality, get_quality_stats)
        Tool(
            name="quality",
            description="Memory quality management. Actions: stats, score, duplicates, merge, fact_duplicates, fact_merge, fact_links.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["stats", "score", "duplicates", "merge", "fact_duplicates", "fact_merge", "fact_links"],
                        "description": "Action: stats, score, duplicates (chunks), merge (chunks), fact_duplicates (find duplicate facts), fact_merge (merge facts), fact_links (link facts)"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Similarity threshold 0-1 (for duplicates, default 0.90)"
                    },
                    "auto_merge": {
                        "type": "boolean",
                        "description": "Auto-merge >0.95 similarity (for merge, default false)"
                    }
                }
            }
        ),
        # Phase 4: Active Decay - storage management and intelligent decay
        Tool(
            name="decay",
            description="Storage decay management. Actions: run (execute decay), preview (dry-run), stats (current state), storage (size report), golden (manage protected items).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["run", "preview", "stats", "storage", "golden"],
                        "description": "Action: run (execute decay), preview (dry-run), stats (get statistics), storage (size report), golden (list/add/remove protected items)"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force run even if already ran today (for action=run)"
                    },
                    "include_summaries": {
                        "type": "boolean",
                        "description": "Include summary decay (for action=run, default true)"
                    },
                    "include_conversations": {
                        "type": "boolean",
                        "description": "Include conversation decay (for action=run, default true)"
                    },
                    "include_facts": {
                        "type": "boolean",
                        "description": "Include fact decay (for action=run, default true)"
                    },
                    "item_id": {
                        "type": "string",
                        "description": "Item ID to mark/unmark as golden (for action=golden)"
                    },
                    "golden_action": {
                        "type": "string",
                        "enum": ["list", "add", "remove"],
                        "description": "Golden item action: list, add, or remove"
                    }
                }
            }
        ),
        # Phase 7: Self-improvement metrics and reporting
        Tool(
            name="self_report",
            description="Generate self-improvement reports. Actions: metrics (performance trends), improvements (before/after tracking), report (full analysis), record_metric, record_improvement.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["metrics", "improvements", "report", "record_metric", "record_improvement"],
                        "description": "Action: metrics (get trends), improvements (list tracked), report (full analysis), record_metric, record_improvement"
                    },
                    "days": {
                        "type": "number",
                        "description": "Days to analyze (for metrics/report, default 30)"
                    },
                    "metric_name": {
                        "type": "string",
                        "description": "Metric name (for record_metric)"
                    },
                    "value": {
                        "type": "number",
                        "description": "Metric value (for record_metric)"
                    },
                    "improvement_name": {
                        "type": "string",
                        "description": "Improvement name (for record_improvement or measure)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Improvement description (for record_improvement)"
                    },
                    "baseline_value": {
                        "type": "number",
                        "description": "Baseline value before improvement (for record_improvement)"
                    }
                }
            }
        ),
        # Phase 6: Personality Evolution
        Tool(
            name="personality",
            description="Personality evolution tracking. Actions: traits (get all), evolution (summary), consistency (check), evolve (from correction/feedback), sync (to profile).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["traits", "evolution", "consistency", "evolve", "sync"],
                        "description": "Action: traits (get all), evolution (summary), consistency (check), evolve (from feedback), sync (to profile)"
                    },
                    "days": {
                        "type": "number",
                        "description": "Days to analyze (for evolution, default 30)"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["communication", "technical", "workflow", "emotional", "learning"],
                        "description": "Trait category filter (optional)"
                    },
                    "feedback_type": {
                        "type": "string",
                        "enum": ["correction", "suggestion", "response"],
                        "description": "Type of feedback (for action=evolve)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Feedback content (for action=evolve)"
                    },
                    "accepted": {
                        "type": "boolean",
                        "description": "Whether feedback was accepted/positive (for action=evolve)"
                    }
                }
            }
        ),
        # CONSOLIDATED: conversation (replaces tag_conversation, add_note_to_conversation, set_conversation_relevance)
        Tool(
            name="conversation",
            description="Manage conversation: tag, note, or set relevance. Actions: tag, note, relevance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["tag", "note", "relevance"],
                        "description": "Action: tag, note, relevance"
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Conversation ID"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to add (for action=tag)"
                    },
                    "note": {
                        "type": "string",
                        "description": "Note content (for action=note)"
                    },
                    "relevance": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low", "archive"],
                        "description": "Relevance level (for action=relevance)"
                    }
                },
                "required": ["conversation_id"]
            }
        ),
        # CONSOLIDATED: images (replaces save_screenshot, search_images)
        Tool(
            name="images",
            description="Save or search images. Actions: save, search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["save", "search"],
                        "description": "Action: save (screenshot), search"
                    },
                    "image_data": {
                        "type": "string",
                        "description": "Base64 image or file path (for action=save)"
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Conversation ID (for action=save)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Image description (for action=save)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for action=search)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Max results (for action=search, default 10)"
                    }
                }
            }
        ),
        # CONSOLIDATED: branch (replaces create_branch, mark_branch_status)
        Tool(
            name="branch",
            description="Exploration branches. Actions: create, mark (chosen/abandoned).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "mark"],
                        "description": "Action: create (new branch), mark (status)"
                    },
                    "name": {
                        "type": "string",
                        "description": "Branch name (for action=create)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Branch description (for action=create)"
                    },
                    "parent_conversation_id": {
                        "type": "string",
                        "description": "Parent conversation (for action=create)"
                    },
                    "branch_id": {
                        "type": "string",
                        "description": "Branch ID (for action=mark)"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["chosen", "abandoned"],
                        "description": "Status to set (for action=mark)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for status (for action=mark)"
                    }
                }
            }
        ),
        # CONSOLIDATED: session (replaces get_active_sessions, get_session_summary, detect_session_continuation, get_conversation_thread)
        Tool(
            name="session",
            description="Session info. Actions: thread, active, summary, detect.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["thread", "active", "summary", "detect"],
                        "description": "Action: thread, active (list), summary, detect (continuation)"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (for thread/summary)"
                    },
                    "days_back": {
                        "type": "number",
                        "description": "Days to look back (for active, default 7)"
                    },
                    "include_last_n": {
                        "type": "number",
                        "description": "Recent segments to include (for summary, default 1)"
                    },
                    "user_prompt": {
                        "type": "string",
                        "description": "User prompt to analyze (for detect)"
                    }
                }
            }
        ),
        Tool(
            name="system_health_check",
            description="Check health of all AI Memory components (NAS, embeddings, indexes, database, MCP). Returns overall status and detailed component diagnostics.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="suggest_questions",
            description="Get suggested questions to ask user to fill knowledge gaps in user profile. Returns prioritized questions based on profile completeness and recent context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum questions to return (default 3)"
                    },
                    "importance": {
                        "type": "string",
                        "description": "Filter by importance (critical/helpful/all, default all)",
                        "enum": ["critical", "helpful", "all"]
                    }
                }
            }
        ),
        # ============== ANTICIPATION ENGINE ==============
        Tool(
            name="get_suggestions",
            description="Get proactive suggestions based on current context. Analyzes user message, finds similar past situations, relevant solutions/antipatterns, and generates ranked suggestions. Use at conversation start or when user seems stuck.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_message": {
                        "type": "string",
                        "description": "The user's current message or question"
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Current working directory (optional, for project detection)"
                    },
                    "recent_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of recently used tool names (optional, for stage detection)"
                    },
                    "conversation_length": {
                        "type": "number",
                        "description": "Number of messages in current conversation (optional)"
                    },
                    "max_suggestions": {
                        "type": "number",
                        "description": "Maximum suggestions to return (default 3)"
                    }
                },
                "required": ["user_message"]
            }
        ),
        # CONSOLIDATED: device (replaces get_current_device, get_all_devices, register_device)
        Tool(
            name="device",
            description="Device info and registration. Actions: current, all, register.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["current", "all", "register"],
                        "description": "Action: current (this device), all (list), register"
                    },
                    "friendly_name": {
                        "type": "string",
                        "description": "Friendly name (for action=register)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Device description (for action=register)"
                    }
                }
            }
        ),
        Tool(
            name="search_by_device",
            description="Search conversations filtered by device. Use to find content from a specific device (e.g., only DGX Spark conversations).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "device_tag": {
                        "type": "string",
                        "description": "Device tag to filter by (e.g., 'dgx_spark', 'windows_pc')"
                    },
                    "include_untagged": {
                        "type": "boolean",
                        "description": "Include conversations without device tags (default true)"
                    }
                },
                "required": ["query"]
            }
        ),
        # ============== STARTUP CONTINUATION TOOLS ==============
        Tool(
            name="check_session_continuation",
            description="IMPORTANT: Call this at the START of every new conversation to check if there's recent work to continue. Returns summary of last continuable session if one exists. Intelligently distinguishes between real work-in-progress and one-off tasks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "number",
                        "description": "How many hours back to look for continuable sessions (default 48)"
                    }
                }
            }
        ),
        Tool(
            name="get_continuation_context",
            description="Get full context for continuing a previous session. Call this AFTER check_session_continuation if user wants to continue. Returns relevant files, key points, and pending work.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session ID to continue (from check_session_continuation)"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="update_active_work",
            description="Update the active_work section in quick_facts.json. Use this to track current project state for session handoff. Call this when: (1) completing a phase/milestone, (2) starting a new task, (3) before ending a session with pending work.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name (e.g., 'Brain Evolution')"
                    },
                    "current_phase": {
                        "type": "string",
                        "description": "Current phase number/identifier (e.g., '1.5', '2')"
                    },
                    "phase_name": {
                        "type": "string",
                        "description": "Human-readable phase name (e.g., 'Session Handoff')"
                    },
                    "next_action": {
                        "type": "string",
                        "description": "What to do next - this is the KEY field for continuation"
                    },
                    "last_completed": {
                        "type": "string",
                        "description": "What was just completed (e.g., 'Phase 1: Truth Maintenance')"
                    },
                    "key_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key file paths relevant to current work"
                    },
                    "plan_file": {
                        "type": "string",
                        "description": "Path to plan/roadmap file if exists"
                    },
                    "clear": {
                        "type": "boolean",
                        "description": "If true, clears the active_work section entirely"
                    },
                    "workstream_id": {
                        "type": "string",
                        "description": "Target a specific workstream by ID (e.g., 'ws_a1b2c3')"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["in_progress", "paused", "completed"],
                        "description": "Set workstream status"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Short description of the workstream"
                    },
                    "remove_workstream": {
                        "type": "string",
                        "description": "Remove a workstream by ID"
                    },
                    "list_workstreams": {
                        "type": "boolean",
                        "description": "Just return current workstreams state without modifying"
                    }
                }
            }
        ),
        # CONSOLIDATED: record_learning (replaces record_solution, record_failure, record_antipattern, confirm_solution)
        Tool(
            name="record_learning",
            description="Record solution, failure, or antipattern. Types: solution, failure, antipattern, confirm.",
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["solution", "failure", "antipattern", "confirm"],
                        "description": "Learning type: solution, failure, antipattern, confirm"
                    },
                    "problem": {
                        "type": "string",
                        "description": "Problem description (for solution/antipattern)"
                    },
                    "solution": {
                        "type": "string",
                        "description": "Solution description (for solution)"
                    },
                    "what_not_to_do": {
                        "type": "string",
                        "description": "What failed (for antipattern)"
                    },
                    "why_it_failed": {
                        "type": "string",
                        "description": "Why it failed (for antipattern/failure)"
                    },
                    "solution_id": {
                        "type": "string",
                        "description": "Solution ID (for failure/confirm)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context"
                    }
                }
            }
        ),
        # CONSOLIDATED: find_learning (replaces find_solution, find_antipatterns, get_solution_chain, get_learnings_summary)
        Tool(
            name="find_learning",
            description="Find solutions, antipatterns, or learning summary. Types: solution, antipattern, chain, summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["solution", "antipattern", "chain", "summary"],
                        "description": "Search type: solution, antipattern, chain, summary"
                    },
                    "problem": {
                        "type": "string",
                        "description": "Problem to find solutions for (for solution/antipattern)"
                    },
                    "solution_id": {
                        "type": "string",
                        "description": "Solution ID (for chain)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to filter by"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Max results (default 10)"
                    }
                }
            }
        ),
        Tool(
            name="analyze_conversation_learnings",
            description="Extract learnings from a conversation. Identifies solutions, failures, and patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "Conversation ID to analyze"
                    },
                    "save": {
                        "type": "boolean",
                        "description": "Save extracted learnings (default true)"
                    }
                },
                "required": ["conversation_id"]
            }
        ),
        Tool(
            name="get_recent_learnings",
            description="Get recently recorded learnings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Max results (default 10)"
                    }
                }
            }
        ),
        # Phase 5: Privacy management tool
        Tool(
            name="privacy",
            description="Privacy and secret management. Actions: scan (check text for secrets), stats (get redaction statistics), sensitive (list sensitive conversations).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["scan", "stats", "sensitive"],
                        "description": "Action: scan (check text), stats (redaction stats), sensitive (list sensitive convos)"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to scan for secrets (for action=scan)"
                    },
                    "redact": {
                        "type": "boolean",
                        "description": "Return redacted version (for action=scan, default false)"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold 0-1 (for action=scan, default 0.7)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Max results (for action=sensitive, default 20)"
                    }
                }
            }
        ),
        # ============== PHASE 1: EPISODIC vs SEMANTIC MEMORY (v6.0) ==============
        Tool(
            name="memory_type",
            description="Query episodic (events) vs semantic (general facts) memories. Actions: query_episodic (what happened on date X), query_semantic (general facts), save_episodic (save event), save_semantic (save fact), stats (get statistics), link (link episodic to semantic), migrate (migrate existing facts).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["query_episodic", "query_semantic", "save_episodic", "save_semantic", "stats", "link", "migrate"],
                        "description": "Action: query_episodic (events by date/actor/emotion), query_semantic (facts by domain/keyword), save_episodic (save event), save_semantic (save fact), stats (statistics), link (link episode to semantic), migrate (migrate existing facts)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for query_episodic/query_semantic)"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (for query_episodic)"
                    },
                    "date_range": {
                        "type": "object",
                        "description": "Date range {start: YYYY-MM-DD, end: YYYY-MM-DD} (for query_episodic)",
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"}
                        }
                    },
                    "actor": {
                        "type": "string",
                        "description": "Actor filter (for query_episodic, e.g., 'Professor', 'Claude')"
                    },
                    "emotion": {
                        "type": "string",
                        "description": "Emotional state filter (for query_episodic, e.g., 'frustrated', 'relieved')"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Domain filter (for query_semantic, e.g., 'infrastructure', 'debugging')"
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords to search (for query_semantic)"
                    },
                    "event": {
                        "type": "string",
                        "description": "Event description (for save_episodic)"
                    },
                    "outcome": {
                        "type": "string",
                        "description": "Event outcome (for save_episodic)"
                    },
                    "emotional_state": {
                        "type": "string",
                        "description": "Emotional state during event (for save_episodic)"
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Link to conversation (for save_episodic)"
                    },
                    "fact": {
                        "type": "string",
                        "description": "Fact content (for save_semantic)"
                    },
                    "episode_id": {
                        "type": "string",
                        "description": "Episode ID (for link action)"
                    },
                    "semantic_id": {
                        "type": "string",
                        "description": "Semantic memory ID (for link action)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Max results (default 10)"
                    }
                }
            }
        ),
        # ============== PHASE 2: WORKING MEMORY (v6.0) ==============
        Tool(
            name="working_memory",
            description="Manage working memory - active reasoning state that persists across compactions. Actions: get_active (get current session), create (new session), add_chain (add reasoning hypothesis), update_chain (add evidence/evaluation), add_note (scratch pad), get_summary, export (for handoff), import (from handoff), stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get_active", "create", "add_chain", "update_chain", "add_note", "get_summary", "export", "import", "stats", "archive", "extend"],
                        "description": "Action: get_active (current session), create (new), add_chain (reasoning hypothesis), update_chain (add evidence/update), add_note (scratch), get_summary, export (handoff), import (restore), stats, archive, extend"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (auto-detected for most actions)"
                    },
                    "active_goal": {
                        "type": "string",
                        "description": "Current task/goal (for create/get_active)"
                    },
                    "hypothesis": {
                        "type": "string",
                        "description": "Hypothesis to test (for add_chain)"
                    },
                    "chain_id": {
                        "type": "string",
                        "description": "Reasoning chain ID (for update_chain)"
                    },
                    "evidence": {
                        "type": "array",
                        "description": "Evidence to add [{type, content, confidence}] (for update_chain)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["observation", "test_result", "memory_recall", "inference", "user_input", "assumption"]},
                                "content": {"type": "string"},
                                "confidence": {"type": "number"}
                            }
                        }
                    },
                    "evaluation": {
                        "type": "string",
                        "enum": ["untested", "testing", "supported", "partially_supported", "refuted", "inconclusive"],
                        "description": "Evaluation status (for update_chain)"
                    },
                    "next_step": {
                        "type": "string",
                        "description": "What to do next (for update_chain)"
                    },
                    "outcome": {
                        "type": "string",
                        "description": "Final outcome (for update_chain when concluding)"
                    },
                    "note": {
                        "type": "string",
                        "description": "Note content (for add_note)"
                    },
                    "note_category": {
                        "type": "string",
                        "description": "Note category: notes, temp_facts, key_observations (for add_note)"
                    },
                    "handoff_data": {
                        "type": "object",
                        "description": "Handoff data to import (for import action)"
                    },
                    "hours": {
                        "type": "number",
                        "description": "Hours to extend (for extend action, default 4)"
                    }
                }
            }
        ),
        # ============== PHASE 3: CAUSAL MODEL (v6.0) ==============
        Tool(
            name="causal",
            description="Manage causal models - store WHY things happen. Actions: add_link (add cause-effect), find_causes (what causes X), find_effects (what X causes), get_interventions (how to prevent X), what_if (simulate intervention), search, reinforce, weaken, extract (from text), stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add_link", "find_causes", "find_effects", "get_interventions", "what_if", "search", "reinforce", "weaken", "extract", "stats"],
                        "description": "Action: add_link (create causal link), find_causes (what causes effect), find_effects (what cause produces), get_interventions (solutions), what_if (simulate), search, reinforce, weaken, extract (from text), stats"
                    },
                    "cause": {
                        "type": "string",
                        "description": "Cause description (for add_link)"
                    },
                    "effect": {
                        "type": "string",
                        "description": "Effect description (for add_link, find_causes, get_interventions)"
                    },
                    "mechanism": {
                        "type": "string",
                        "description": "How cause produces effect (for add_link)"
                    },
                    "counterfactual": {
                        "type": "string",
                        "description": "What would happen without cause (for add_link)"
                    },
                    "interventions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Possible interventions (for add_link)"
                    },
                    "intervention": {
                        "type": "string",
                        "description": "Intervention to simulate (for what_if)"
                    },
                    "link_id": {
                        "type": "string",
                        "description": "Causal link ID (for reinforce/weaken)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for search action)"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to extract causal links from (for extract)"
                    },
                    "use_llm": {
                        "type": "boolean",
                        "description": "Use DGX LLM for extraction (for extract, default false)"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence (default 0.5)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Max results (default 10)"
                    }
                }
            }
        ),
        # ============== PHASE 4: ACTIVE REASONING (v6.0) ==============
        Tool(
            name="reason",
            description="Active reasoning over memories - generate insights without prompting. Actions: analyze (full reasoning), find_insights (relevant to query), proactive (during conversation), validate (confirm/refute insight), goal (reason about achieving goal), stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["analyze", "find_insights", "proactive", "validate", "goal", "stats"],
                        "description": "Action: analyze (full reasoning over memories), find_insights (relevant to query), proactive (surface insights during conversation), validate (confirm/refute), goal (reason about goal), stats"
                    },
                    "memories": {
                        "type": "array",
                        "description": "Memories to reason over (for analyze action)"
                    },
                    "context": {
                        "type": "string",
                        "description": "Context for focused reasoning"
                    },
                    "query": {
                        "type": "string",
                        "description": "Query to find relevant insights (for find_insights)"
                    },
                    "user_message": {
                        "type": "string",
                        "description": "User's current message (for proactive)"
                    },
                    "insight_id": {
                        "type": "string",
                        "description": "Insight ID (for validate)"
                    },
                    "is_valid": {
                        "type": "boolean",
                        "description": "Whether insight is valid (for validate)"
                    },
                    "goal": {
                        "type": "string",
                        "description": "Goal to reason about (for goal action)"
                    },
                    "use_llm": {
                        "type": "boolean",
                        "description": "Use DGX LLM for deeper reasoning (default false)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Max results (default 10)"
                    }
                }
            }
        ),
        # ============== PHASE 5: SESSION HANDOFF (v6.0) ==============
        Tool(
            name="session_handoff",
            description="Manage session handoffs for reasoning continuity. Actions: save (save handoff), get_latest, get_recent, restore (working memory from handoff).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["save", "get_latest", "get_recent", "restore"],
                        "description": "Action: save (create handoff), get_latest, get_recent, restore (working memory from handoff)"
                    },
                    "handoff_data": {
                        "type": "object",
                        "description": "Handoff data (for save action, typically from working_memory export)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why handoff is happening (compaction, session_end, etc.)"
                    },
                    "handoff_id": {
                        "type": "string",
                        "description": "Handoff ID (for restore action)"
                    },
                    "hours": {
                        "type": "number",
                        "description": "Hours to look back (for get_recent, default 48)"
                    }
                }
            }
        ),
        # ============== PHASE 6: GOAL-DIRECTED ACCESS (v6.0) ==============
        Tool(
            name="goals",
            description="Track user goals for proactive memory access. Actions: detect (from text), add, get, update, complete, list_active, proactive_context (get relevant context for message), find_relevant, stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["detect", "add", "get", "update", "complete", "list_active", "proactive_context", "find_relevant", "stats"],
                        "description": "Action: detect (from text), add (manual), get, update, complete, list_active, proactive_context (get context), find_relevant, stats"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to detect goals from (for detect/proactive_context)"
                    },
                    "goal_id": {
                        "type": "string",
                        "description": "Goal ID (for get/update/complete)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Goal description (for add)"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Goal priority (for add/update)"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "paused", "completed", "abandoned"],
                        "description": "Goal status (for update)"
                    },
                    "add_subgoal": {
                        "type": "string",
                        "description": "Subgoal to add (for update)"
                    },
                    "add_blocker": {
                        "type": "string",
                        "description": "Known blocker to add (for update)"
                    },
                    "context": {
                        "type": "string",
                        "description": "Context to find relevant goals (for find_relevant)"
                    }
                }
            }
        ),
        # ============== PHASE 7: PREDICTIVE SIMULATION (v6.0) ==============
        Tool(
            name="predict",
            description="Predictive simulation using causal model and history. Actions: from_causal (predict outcome), anticipate_failures (get warnings), check_pattern (specific failure pattern), preventive_actions (suggestions), verify (record outcome), stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["from_causal", "anticipate_failures", "check_pattern", "preventive_actions", "verify", "stats"],
                        "description": "Action: from_causal (predict), anticipate_failures, check_pattern, preventive_actions, verify (outcome), stats"
                    },
                    "action_text": {
                        "type": "string",
                        "description": "Action being taken (for from_causal prediction)"
                    },
                    "context": {
                        "type": "string",
                        "description": "Context for prediction/anticipation"
                    },
                    "pattern_type": {
                        "type": "string",
                        "enum": ["timeout", "encoding", "path", "network", "permission"],
                        "description": "Specific pattern to check (for check_pattern)"
                    },
                    "prediction_id": {
                        "type": "string",
                        "description": "Prediction ID (for verify)"
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["correct", "incorrect", "partial"],
                        "description": "Actual outcome (for verify)"
                    }
                }
            }
        ),
        # ============== PHASE 8: META-LEARNING (v6.0) ==============
        Tool(
            name="meta_learn",
            description="Meta-learning for retrieval strategy optimization. Actions: record_query (track performance), feedback (user feedback), recommend (get strategy), get_stats, create_experiment (A/B test), experiment_result, list_experiments, tune_parameter, suggest_parameter.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["record_query", "feedback", "recommend", "get_stats", "create_experiment", "experiment_result", "list_experiments", "tune_parameter", "suggest_parameter", "set_default"],
                        "description": "Action to perform"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for record_query, recommend)"
                    },
                    "strategy_id": {
                        "type": "string",
                        "description": "Strategy identifier (for record_query)"
                    },
                    "latency_ms": {
                        "type": "number",
                        "description": "Query latency in ms (for record_query)"
                    },
                    "result_count": {
                        "type": "number",
                        "description": "Number of results (for record_query)"
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether query was successful (for record_query, experiment_result)"
                    },
                    "query_id": {
                        "type": "string",
                        "description": "Query ID (for feedback)"
                    },
                    "positive": {
                        "type": "boolean",
                        "description": "Positive feedback? (for feedback)"
                    },
                    "experiment_name": {
                        "type": "string",
                        "description": "Experiment name (for create_experiment)"
                    },
                    "strategy_a": {
                        "type": "string",
                        "description": "First strategy (for create_experiment)"
                    },
                    "strategy_b": {
                        "type": "string",
                        "description": "Second strategy (for create_experiment)"
                    },
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment ID (for experiment_result)"
                    },
                    "param_name": {
                        "type": "string",
                        "description": "Parameter name (for tune_parameter, suggest_parameter)"
                    },
                    "param_value": {
                        "type": "number",
                        "description": "Parameter value (for tune_parameter)"
                    },
                    "param_min": {
                        "type": "number",
                        "description": "Minimum value (for tune_parameter)"
                    },
                    "param_max": {
                        "type": "number",
                        "description": "Maximum value (for tune_parameter)"
                    },
                    "param_step": {
                        "type": "number",
                        "description": "Step size (for tune_parameter)"
                    },
                    "score": {
                        "type": "number",
                        "description": "Performance score (for tune_parameter record)"
                    }
                }
            }
        ),
        # ============== PHASE 9: CONTINUOUS SELF-MODELING (v6.0) ==============
        Tool(
            name="self_model",
            description="Continuous self-modeling for real-time self-awareness. Actions: get_state, update_confidence, add_uncertainty, add_limitation, add_strength, update_quality, assess_text, introspect, hallucination_check, take_snapshot, get_snapshot, reset_daily, stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get_state", "update_confidence", "add_uncertainty", "add_limitation", "add_strength", "update_quality", "assess_text", "introspect", "hallucination_check", "take_snapshot", "get_snapshot", "reset_daily", "stats"],
                        "description": "Action to perform"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence level 0-1 (for update_confidence)"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic for uncertainty/limitation/strength"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to assess/introspect (for assess_text, introspect, hallucination_check)"
                    },
                    "context": {
                        "type": "string",
                        "description": "Context for analysis (for introspect, take_snapshot)"
                    },
                    "task_clarity": {
                        "type": "number",
                        "description": "Task clarity 0-1 (for update_quality)"
                    },
                    "evidence_sufficiency": {
                        "type": "number",
                        "description": "Evidence sufficiency 0-1 (for update_quality)"
                    },
                    "hallucination_risk": {
                        "type": "number",
                        "description": "Hallucination risk 0-1 (for update_quality)"
                    },
                    "snapshot_id": {
                        "type": "string",
                        "description": "Snapshot ID (for get_snapshot)"
                    },
                    "use_llm": {
                        "type": "boolean",
                        "description": "Use DGX LLM for deep analysis (for introspect)"
                    }
                }
            }
        ),
        # ============== PHASE 10: ACTIVE MEMORY CONSOLIDATION (v6.0) ==============
        Tool(
            name="consolidate",
            description="Active memory consolidation - cluster episodes, create abstractions, strengthen connections, prune redundancies. Actions: run (trigger consolidation), get_state, get_run, list_runs, get_abstractions, schedule, stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["run", "get_state", "get_run", "list_runs", "get_abstractions", "schedule", "stats"],
                        "description": "Action to perform"
                    },
                    "full": {
                        "type": "boolean",
                        "description": "Process all memories (for run, default false = recent only)"
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Run ID (for get_run)"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Filter abstractions by domain (for get_abstractions)"
                    },
                    "hours": {
                        "type": "number",
                        "description": "Hours until next run (for schedule, default 24)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Limit results (for list_runs, get_abstractions)"
                    }
                }
            }
        ),
        # ============== CHUNK RETRIEVAL FOR CONTEXT INJECTION ==============
        Tool(
            name="get_chunk",
            description="Retrieve specific chunk(s) by ID for context injection. Used by Cerebro to fetch referenced content without sending full text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "16-char hex chunk ID"
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Conversation ID for efficient lookup"
                    },
                    "include_context": {
                        "type": "boolean",
                        "description": "Include surrounding context (default true)"
                    }
                },
                "required": ["chunk_id", "conversation_id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls - OPTIMIZED: only load what's needed"""

    # AGENT 16: Basic logging for all tool calls
    start_time = None
    try:
        import time
        start_time = time.time()
    except:
        pass

    # Tools that DON'T need memory/embeddings — can run before init completes
    # These only read JSON files directly (quick_facts, learnings, etc.)
    no_init_tools = {
        "check_session_continuation",
        "get_recent_learnings",
        "get_corrections",
        "update_active_work",
        "find_learning",
        "record_learning",
        "system_health_check",
        "get_suggestions",
        "suggest_questions",
        "device",
    }

    _tool_t0 = time.time()

    if name not in no_init_tools:
        # Wait for background initialization to complete (max 45s)
        sys.stderr.write(f"MCP TOOL [{name}]: waiting for init...\n"); sys.stderr.flush()
        if not await wait_for_init(timeout=20.0):
            return [TextContent(
                type="text",
                text="AI Memory is still initializing (NAS/model warmup in progress). Please try again in a few seconds."
            )]
        sys.stderr.write(f"MCP TOOL [{name}]: init done in {(time.time()-_tool_t0)*1000:.0f}ms\n"); sys.stderr.flush()
    else:
        sys.stderr.write(f"MCP TOOL [{name}]: fast-path (no init wait)\n"); sys.stderr.flush()

    try:
        # OPTIMIZATION: Only load services that are actually needed for this tool
        # Memory-only tools (fast, no chunking)

        # Embeddings-required tools (including save for auto-chunking)
        embeddings_tools = {
            "save_conversation_ultimate",  # Now auto-chunks!
            "search",  # consolidated search tool
            "rebuild_vector_index",
            # Legacy names for backward compatibility
            "semantic_search",
            "hybrid_search",
            "get_rag_context"
        }

        # Load only what we need
        memory = _memory  # Already initialized at startup
        embeddings = _embeddings if name in embeddings_tools else None

        if name == "save_conversation_ultimate":
            # Auto-generate session_id if not provided (Phase 3.1 fix)
            session_id = arguments.get("session_id")
            if not session_id:
                session_id = f"live_{uuid.uuid4().hex[:8]}"

            # Extract metadata before nested functions (fixes 'metadata not defined' bug)
            metadata = arguments.get("metadata", {})

            # AUTO-TAG WITH DEVICE INFO (Multi-device awareness)
            try:
                device_meta = get_device_metadata()
                metadata["device_tag"] = device_meta.get("device_tag", "unknown")
                metadata["device_name"] = device_meta.get("device_name", "Unknown")
                metadata["device_hostname"] = device_meta.get("hostname", "")
                metadata["device_os"] = device_meta.get("os", "")
                metadata["device_architecture"] = device_meta.get("architecture", "")
            except Exception:
                # If device detection fails, continue without it
                metadata["device_tag"] = "unknown"

            def do_save():
                return memory.save_conversation(
                    messages=arguments.get("messages", []),
                    session_id=session_id,
                    metadata=metadata
                )

            conv_id = await run_in_thread(do_save)

            def read_stats():
                conv_file = memory.conversations_path / f"{conv_id}.json"
                with open(conv_file, "r", encoding="utf-8") as f:
                    return json.load(f)

            saved = await run_in_thread(read_stats)

            # AUTO-CHUNK: Create searchable chunks for this conversation
            # INCREMENTAL OPTIMIZATION (Agent 7): Only embed new chunks
            # DGX INTEGRATION (Phase 4): Use DGX GPU for embedding generation
            def do_chunk():
                chunks = _embeddings.chunk_conversation(saved)
                if not chunks:
                    return 0, False

                # Check if this is an incremental save
                previous_chunk_count = metadata.get("previous_chunk_count", 0)

                # Try DGX embedding if available (doesn't block MCP startup!)
                used_dgx = False
                if is_dgx_embedding_available_cached():
                    chunks = embed_chunks_via_dgx(chunks, batch_size=128)
                    if chunks and "embedding" in chunks[0]:
                        used_dgx = True

                if previous_chunk_count > 0:
                    # INCREMENTAL: Only save and embed new chunks
                    _embeddings.save_incremental_chunks(
                        chunks,
                        conv_id,
                        previous_chunk_count=previous_chunk_count
                    )
                    return len(chunks), used_dgx
                else:
                    # FULL SAVE: Save all chunks
                    _embeddings.save_chunks(chunks, conv_id)
                    # Also save vectors if embeddings exist
                    if chunks and "embedding" in chunks[0]:
                        _embeddings.save_vectors(chunks, conv_id)
                    return len(chunks), used_dgx

            chunk_result = await run_in_thread(do_chunk)
            chunk_count = chunk_result[0] if isinstance(chunk_result, tuple) else chunk_result
            used_dgx = chunk_result[1] if isinstance(chunk_result, tuple) else False

            if used_dgx:
                embedding_stats = {"chunks": chunk_count, "embedded": True, "embedded_via": "dgx", "indexed": True, "note": "Chunks embedded via DGX GPU"}
                # Notify DGX search service to reindex (non-blocking)
                try:
                    from dgx_search_client import invalidate_dgx_cache
                    invalidate_dgx_cache()
                except Exception:
                    pass
            else:
                embedding_stats = {"chunks": chunk_count, "embedded": False, "indexed": True, "note": "Chunks created for RAG search (keyword only)"}

            # Phase 7: Analyze feedback signals in this conversation
            feedback_analysis = {}
            try:
                def do_feedback_analysis():
                    from feedback_detector import FeedbackDetector
                    detector = FeedbackDetector()
                    messages = arguments.get("messages", [])
                    if messages:
                        analysis = detector.analyze_conversation_feedback(messages, conv_id)
                        # If strong feedback signals, link to solutions if possible
                        return {
                            "success_signals": len(analysis.get("success_signals", [])),
                            "failure_signals": len(analysis.get("failure_signals", [])),
                            "overall_outcome": analysis.get("overall_outcome", "neutral"),
                            "net_sentiment": analysis.get("net_sentiment", 0)
                        }
                    return {}
                feedback_analysis = await run_in_thread(do_feedback_analysis)
            except Exception:
                feedback_analysis = {}  # Non-critical, don't fail save

            return [TextContent(
                type="text",
                text=safe_json_dumps({
                    "success": True,
                    "conversation_id": conv_id,
                    "extraction_stats": {
                        "facts": len(saved["extracted_data"]["facts"]),
                        "file_paths": len(saved["extracted_data"]["file_paths"]),
                        "actions": len(saved["extracted_data"]["actions_taken"]),
                        "decisions": len(saved["extracted_data"]["decisions_made"]),
                        "problems": len(saved["extracted_data"]["problems_solved"]),
                        "user_preferences": len(saved["extracted_data"]["user_preferences"]),
                        "goals": len(saved["extracted_data"]["goals_and_intentions"]),
                        "tags": len(saved["metadata"]["tags"]),
                        "topics": len(saved["metadata"]["topics"])
                    },
                    "embedding_stats": embedding_stats,
                    "feedback_analysis": feedback_analysis,
                    "device": {
                        "tag": metadata.get("device_tag", "unknown"),
                        "name": metadata.get("device_name", "Unknown"),
                        "hostname": metadata.get("device_hostname", "")
                    },
                    "tags": saved["metadata"]["tags"][:10],  # Limit tags
                    "topics": saved["metadata"]["topics"][:10],  # Limit topics
                    "importance": saved["metadata"]["importance"],
                    "message": f"Conversation saved from {metadata.get('device_name', 'Unknown')}: {len(saved['extracted_data']['facts'])} facts extracted"
                })
            )]

        elif name == "get_chunk":
            # Retrieve a specific chunk by ID for context injection
            chunk_id = arguments.get("chunk_id")
            conversation_id = arguments.get("conversation_id")
            include_context = arguments.get("include_context", True)

            if not chunk_id or not conversation_id:
                return [TextContent(type="text", text=safe_json_dumps({
                    "error": "chunk_id and conversation_id are required"
                }))]

            def do_get_chunk():
                # Check for regular conversation chunks
                chunk_file = Path(AI_MEMORY_PATH) / "embeddings" / "chunks" / f"{conversation_id}.jsonl"

                # Also check for agent output chunks
                agent_chunk_file = Path(AI_MEMORY_PATH) / "embeddings" / "chunks" / f"agent-{conversation_id}.jsonl"

                target_file = None
                if chunk_file.exists():
                    target_file = chunk_file
                elif agent_chunk_file.exists():
                    target_file = agent_chunk_file

                if not target_file:
                    return {"error": f"Conversation {conversation_id} not found", "chunk_id": chunk_id}

                try:
                    for line in target_file.read_text(encoding='utf-8').splitlines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        if chunk.get("chunk_id") == chunk_id:
                            result = {
                                "chunk_id": chunk_id,
                                "conversation_id": conversation_id,
                                "content": chunk.get("content", ""),
                                "role": chunk.get("role", "unknown"),
                                "chunk_type": chunk.get("chunk_type", "message"),
                                "metadata": chunk.get("metadata", {})
                            }
                            if include_context and chunk.get("context_before"):
                                result["context_before"] = chunk["context_before"]
                            return result
                except Exception as e:
                    return {"error": f"Failed to read chunk file: {str(e)}", "chunk_id": chunk_id}

                return {"error": f"Chunk {chunk_id} not found in conversation {conversation_id}"}

            result = await run_in_thread(do_get_chunk)
            return [TextContent(type="text", text=safe_json_dumps(result))]

        # ============================================================
        # CONSOLIDATED TOOL HANDLERS
        # ============================================================

        elif name == "search":
            # Consolidated: replaces semantic_search, hybrid_search, get_rag_context
            # OPTIMIZED: Try DGX search service first (fast), fall back to local
            query = arguments.get("query")
            mode = arguments.get("mode", "hybrid")
            top_k = arguments.get("top_k", 5)
            alpha = arguments.get("alpha", 0.7)
            chunk_type = arguments.get("chunk_type")
            boost_recency = arguments.get("boost_recency", True)
            recency_days = arguments.get("recency_days", 30)
            max_tokens = arguments.get("max_tokens")
            compact = arguments.get("compact", False)

            # Try DGX search service first (sub-50ms, GPU accelerated)
            dgx_result = None
            try:
                from dgx_search_client import dgx_search, is_dgx_available
                if await is_dgx_available():
                    dgx_result = await dgx_search(query, top_k=top_k, mode=mode, alpha=alpha)
            except ImportError:
                pass  # DGX client not available
            except Exception as e:
                sys.stderr.write(f"[Search] DGX search failed: {e}\n")

            if dgx_result and dgx_result.get("results"):
                # DGX search successful - use those results
                results = dgx_result.get("results", [])
                results = sanitize_results(results)
                search_source = "dgx"
                latency_ms = dgx_result.get("latency_ms", 0)
            else:
                # Fall back to local search
                search_source = "local"
                latency_ms = None

                def do_search():
                    global _embeddings
                    # Fast NAS check - fail fast if NAS unavailable
                    if not is_nas_reachable(timeout=2.0):
                        return {"error": "NAS not reachable", "results": [], "count": 0, "nas_status": "unavailable"}

                    # Lazy initialize embeddings if not already done
                    emb = _embeddings
                    if emb is None:
                        emb = _init_embeddings()
                        if emb is None:
                            return {"error": "Embeddings engine not available", "results": [], "count": 0}

                    filters = {}
                    if chunk_type:
                        filters["chunk_type"] = chunk_type

                    if mode == "semantic":
                        return emb.semantic_search(
                            query, top_k=top_k, filters=filters,
                            boost_by_recency=boost_recency,
                            recency_decay_days=recency_days
                        )
                    elif mode == "rag":
                        return emb.hybrid_search(query, top_k=top_k)
                    else:  # hybrid (default)
                        return emb.hybrid_search(
                            query, top_k=top_k, alpha=alpha,
                            boost_by_recency=boost_recency,
                            recency_decay_days=recency_days
                        )

                results = await run_in_thread(do_search, timeout=45)
                results = sanitize_results(results)

            # Apply token-limited compression if requested
            if compact or max_tokens:
                try:
                    from result_compressor import compress_for_mcp
                    token_limit = max_tokens if max_tokens else 500
                    compressed = compress_for_mcp(results, max_tokens=token_limit)
                    compressed_response = {
                        "query": query,
                        "mode": mode,
                        "formatted": compressed["formatted"],
                        "token_count": compressed["token_count"],
                        "stats": compressed["stats"],
                        "recency_boost_enabled": boost_recency,
                        "search_source": search_source
                    }
                    if latency_ms is not None:
                        compressed_response["latency_ms"] = latency_ms
                    return [TextContent(type="text", text=safe_json_dumps(compressed_response))]
                except Exception:
                    # Fall back to regular output on error
                    pass

            response_data = {
                "query": query,
                "mode": mode,
                "results_count": len(results),
                "results": results,
                "recency_boost_enabled": boost_recency,
                "search_source": search_source
            }
            if latency_ms is not None:
                response_data["latency_ms"] = latency_ms

            return [TextContent(type="text", text=safe_json_dumps(response_data))]

        elif name == "projects":
            # Consolidated: replaces get_project_state, get_active_projects, get_stale_projects
            action = arguments.get("action", "active")
            try:
                from project_tracker import ProjectTracker

                def do_projects():
                    # Fast NAS check with socket timeout
                    if not is_nas_reachable(timeout=2.0):
                        return {"error": "NAS not reachable (socket timeout)"}

                    tracker = ProjectTracker()
                    if action == "state":
                        project_id = arguments.get("project_id")
                        file_path = arguments.get("file_path")
                        if file_path:
                            project = tracker.search_project_by_path(file_path)
                        elif project_id:
                            project = tracker.get_project(project_id)
                        else:
                            return {"error": "Must provide either project_id or file_path"}
                        if not project:
                            return {"error": "Project not found", "project": None}
                        return {
                            "project_id": project["project_id"],
                            "name": project["name"],
                            "status": project["status"],
                            "last_worked": project["last_worked"],
                            "current_focus": sanitize_text(project.get("current_focus", ""), 200),
                            "priority": project["priority"],
                            "blockers": [sanitize_text(b, 150) for b in project.get("blockers", [])[:5]],
                            "next_steps": [sanitize_text(s, 150) for s in project.get("next_steps", [])[:5]],
                            "files": project.get("files", [])[:10]
                        }
                    elif action == "auto_update":
                        # Phase 3: Auto-update project states
                        from project_auto_updater import ProjectAutoUpdater
                        auto_updater = ProjectAutoUpdater()

                        # Run status transitions (time-based)
                        transitions = auto_updater.run_status_transitions()

                        return {
                            "action": "auto_update",
                            "projects_checked": transitions.get("checked", 0),
                            "transitions": transitions.get("transitions", []),
                            "timestamp": transitions.get("timestamp"),
                            "message": f"{len(transitions.get('transitions', []))} projects changed status"
                        }
                    elif action == "activity":
                        # Phase 3: Get activity summary
                        from project_auto_updater import ProjectAutoUpdater
                        auto_updater = ProjectAutoUpdater()

                        days = arguments.get("days", 7)
                        summary = auto_updater.get_activity_summary(days=days)

                        return {
                            "action": "activity",
                            "period_days": summary.get("period_days"),
                            "total_projects": summary.get("total_projects"),
                            "active_projects": summary.get("active_projects"),
                            "stale_projects": summary.get("stale_projects"),
                            "inactive_projects": summary.get("inactive_projects"),
                            "activity_events": summary.get("total_activity_events"),
                            "most_active": summary.get("most_active_projects", [])[:5],
                            "recent_transitions": summary.get("recent_transitions", [])
                        }
                    elif action == "stale":
                        from pattern_detector import PatternDetector
                        detector = PatternDetector()
                        stale_days = arguments.get("stale_days", 14)
                        stale = detector.detect_stale_projects(stale_days=stale_days)
                        return {"stale_projects": stale, "count": len(stale), "threshold_days": stale_days}
                    else:  # active (default)
                        status = arguments.get("status", "active")
                        if status == "all":
                            projects = list(tracker.projects.values())
                        else:
                            projects = tracker.get_active_projects(status=status)
                        projects_summary = []
                        for p in projects[:20]:
                            projects_summary.append({
                                "project_id": p["project_id"],
                                "name": p["name"],
                                "status": p["status"],
                                "last_worked": p["last_worked"],
                                "current_focus": sanitize_text(p.get("current_focus", ""), 100),
                                "priority": p["priority"]
                            })
                        return {"projects": projects_summary, "count": len(projects_summary), "stats": tracker.get_stats()}

                result = await run_in_thread(do_projects, timeout=15)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "project_evolution":
            # Consolidated: replaces record/get/supersede/list evolution tools
            action = arguments.get("action", "list")
            try:
                from project_evolution import ProjectEvolutionTracker

                def do_evolution():
                    # Fast NAS check - fail fast if NAS unavailable
                    if not is_nas_reachable(timeout=2.0):
                        return {"error": "NAS not reachable", "nas_status": "unavailable", "suggestion": "Check if NAS is powered on and Z: drive is mounted"}
                    tracker = ProjectEvolutionTracker()
                    project_id = arguments.get("project_id")

                    if action == "record":
                        evolution = tracker.record_evolution(
                            project_id=project_id,
                            conversation_id=arguments.get("conversation_id", "manual"),
                            summary=arguments.get("summary"),
                            version=arguments.get("version"),
                            supersedes_version=arguments.get("supersedes_version"),
                            keywords=arguments.get("keywords", [])
                        )
                        return {"success": True, "project_id": project_id, "evolution": evolution}
                    elif action == "timeline":
                        timeline = tracker.get_project_timeline(project_id)
                        current = tracker.get_current_version(project_id)
                        return {"project_id": project_id, "current_version": current.get("version") if current else None, "version_count": len(timeline), "timeline": timeline}
                    elif action == "supersede":
                        success = tracker.mark_superseded(
                            project_id=project_id,
                            old_version=arguments.get("old_version"),
                            new_version=arguments.get("new_version"),
                            reason=arguments.get("reason"),
                            conversation_ids=arguments.get("conversation_ids", [])
                        )
                        return {"success": success, "message": f"Marked {arguments.get('old_version')} as superseded"}
                    else:  # list
                        projects = tracker.get_all_projects()
                        return {"tracked_projects": projects, "count": len(projects)}

                result = await run_in_thread(do_evolution)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "preferences":
            # Consolidated: replaces get_user_preferences, update_user_preference
            # Phase 2: Added evolution actions
            action = arguments.get("action", "get")
            try:
                from preference_manager import PreferenceManager

                def do_preferences():
                    # Fast NAS check with socket timeout - BEFORE touching filesystem
                    if not is_nas_reachable(timeout=2.0):
                        return {"error": "NAS not reachable (socket timeout)", "success": False}

                    manager = PreferenceManager()

                    if action == "update":
                        return manager.update_preference_manual(
                            arguments.get("category"),
                            arguments.get("preference"),
                            arguments.get("positive", True)
                        )
                    elif action == "evolved":
                        # Get weighted/evolved preferences
                        include_stale = arguments.get("include_stale", False)
                        return manager.get_evolved_preferences(include_stale=include_stale)
                    elif action == "decay":
                        # Check for stale preferences
                        return manager.check_preference_decay()
                    elif action == "contradictions":
                        # Find contradicting preferences
                        return {"contradictions": manager.detect_contradictions()}
                    elif action == "migrate":
                        # Migrate legacy preferences to evolution system
                        return manager.migrate_to_evolution()
                    elif action == "stats":
                        # Get evolution statistics
                        return manager.get_evolution_stats()
                    else:  # get (legacy)
                        return manager.get_preferences_summary()

                result = await run_in_thread(do_preferences, timeout=15)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "confidence":
            # Phase 3 & 6 - Brain Evolution: Confidence and Provenance tracking for facts
            action = arguments.get("action", "stats")
            try:
                from confidence_tracker import ConfidenceTracker
                from provenance_tracker import ProvenanceTracker

                def do_confidence():
                    # Fast NAS check - fail fast if NAS unavailable
                    if not is_nas_reachable(timeout=2.0):
                        return {"error": "NAS not reachable", "nas_status": "unavailable", "suggestion": "Check if NAS is powered on and Z: drive is mounted"}
                    conf_tracker = ConfidenceTracker()
                    prov_tracker = ProvenanceTracker()

                    if action == "stats":
                        # Combined stats from both trackers
                        conf_stats = conf_tracker.get_stats()
                        prov_stats = prov_tracker.get_stats()
                        return {
                            "confidence": conf_stats,
                            "provenance": prov_stats
                        }
                    elif action == "get":
                        fact_id = arguments.get("fact_id")
                        if not fact_id:
                            return {"error": "fact_id required"}
                        confidence = conf_tracker.get_confidence(fact_id)
                        if confidence is None:
                            return {"error": f"No confidence record for fact_id: {fact_id}"}
                        record = conf_tracker.get_record(fact_id)
                        return {"fact_id": fact_id, "confidence": confidence, "record": record}
                    elif action == "provenance":
                        # Phase 6: Full provenance (source + corrections + reinforcements)
                        fact_id = arguments.get("fact_id")
                        if not fact_id:
                            return {"error": "fact_id required"}

                        # Get full provenance from provenance tracker
                        full_prov = prov_tracker.get_full_provenance(fact_id)
                        if full_prov is None:
                            # Try lazy migration
                            prov_tracker.migrate_fact_provenance(fact_id)
                            full_prov = prov_tracker.get_full_provenance(fact_id)

                        # Also get confidence provenance
                        conf_prov = conf_tracker.get_provenance(fact_id)

                        return {
                            "fact_id": fact_id,
                            "provenance": full_prov,
                            "confidence_events": conf_prov,
                            "confidence_event_count": len(conf_prov) if conf_prov else 0
                        }
                    elif action == "source_chain":
                        # Phase 6: Just the source chain (origin info)
                        fact_id = arguments.get("fact_id")
                        if not fact_id:
                            return {"error": "fact_id required"}
                        source_chain = prov_tracker.get_source_chain(fact_id)
                        if source_chain is None:
                            # Try lazy migration
                            prov_tracker.migrate_fact_provenance(fact_id)
                            source_chain = prov_tracker.get_source_chain(fact_id)
                        return {"fact_id": fact_id, "source_chain": source_chain}
                    elif action == "corrections":
                        # Phase 6: Correction history for a fact
                        fact_id = arguments.get("fact_id")
                        if not fact_id:
                            return {"error": "fact_id required"}
                        corrections = prov_tracker.get_correction_history(fact_id)
                        return {
                            "fact_id": fact_id,
                            "corrections": corrections,
                            "correction_count": len(corrections)
                        }
                    elif action == "reinforcements":
                        # Phase 6: Reinforcement history for a fact
                        fact_id = arguments.get("fact_id")
                        if not fact_id:
                            return {"error": "fact_id required"}
                        reinforcements = prov_tracker.get_reinforcements(fact_id)
                        return {
                            "fact_id": fact_id,
                            "reinforcements": reinforcements,
                            "reinforcement_count": len(reinforcements)
                        }
                    elif action == "reinforcement_stats":
                        # Phase 5.2: System-wide reinforcement statistics
                        stats = conf_tracker.get_reinforcement_stats()
                        return {"reinforcement_stats": stats}
                    elif action == "highly_reinforced":
                        # Phase 5.2: Get facts with 3+ reinforcements
                        min_count = int(arguments.get("threshold", 3))
                        facts = conf_tracker.get_highly_reinforced_facts(min_reinforcements=min_count)
                        return {
                            "highly_reinforced_facts": facts,
                            "count": len(facts),
                            "min_reinforcements": min_count
                        }
                    elif action == "detect_contradictions":
                        # Phase 6: Find contradictions for given content
                        content = arguments.get("content")
                        fact_id = arguments.get("fact_id", "temp_check")
                        if not content:
                            return {"error": "content required for contradiction detection"}
                        contradictions = prov_tracker.detect_contradictions(
                            new_fact_id=fact_id,
                            new_fact_content=content
                        )
                        return {
                            "checked_content": content[:100] + "..." if len(content) > 100 else content,
                            "contradictions_found": len(contradictions),
                            "contradictions": [
                                {
                                    "fact_id": c.fact_id_a,
                                    "similarity": round(c.similarity_score, 2),
                                    "type": c.contradiction_type,
                                    "evidence": c.evidence
                                }
                                for c in contradictions
                            ]
                        }
                    elif action == "contradiction_stats":
                        # Phase 6: Statistics about detected contradictions
                        stats = prov_tracker.get_contradiction_stats()
                        return {"contradiction_stats": stats}
                    elif action == "reinforce":
                        fact_id = arguments.get("fact_id")
                        if not fact_id:
                            return {"error": "fact_id required"}
                        reason = arguments.get("reason", "User reinforced")
                        new_confidence = conf_tracker.reinforce(fact_id, source="user_action", reason=reason)
                        if new_confidence is None:
                            return {"error": f"No confidence record for fact_id: {fact_id}"}

                        # Phase 6: Also log reinforcement in provenance tracker
                        prov_tracker.add_reinforcement(
                            fact_id=fact_id,
                            conversation_id="mcp_action",
                            reinforcement_type="explicit",
                            context=reason
                        )
                        return {"fact_id": fact_id, "new_confidence": new_confidence, "action": "reinforced"}
                    elif action == "decay":
                        result = conf_tracker.apply_decay_all()
                        return {"action": "decay_applied", **result}
                    elif action == "low":
                        threshold = arguments.get("threshold", 0.5)
                        low_facts = conf_tracker.get_low_confidence_facts(threshold=threshold)
                        return {"threshold": threshold, "count": len(low_facts), "facts": low_facts}
                    elif action == "quarantine":
                        threshold = arguments.get("threshold", 0.4)
                        result = conf_tracker.quarantine_low_confidence(threshold=threshold)
                        return {"action": "quarantine", **result}
                    elif action == "resolve_contradiction":
                        fact_id_a = arguments.get("fact_id_a")
                        fact_id_b = arguments.get("fact_id_b")
                        status = arguments.get("status", "dismissed")
                        resolution = arguments.get("resolution", "Manually resolved")
                        if not fact_id_a or not fact_id_b:
                            return {"error": "fact_id_a and fact_id_b required"}
                        success = prov_tracker.resolve_contradiction(fact_id_a, fact_id_b, status, resolution)
                        return {"resolved": success, "fact_id_a": fact_id_a, "fact_id_b": fact_id_b, "status": status}
                    elif action == "batch_resolve":
                        count = arguments.get("count", 100)
                        strategy = arguments.get("strategy", "keep_newer")
                        result = prov_tracker.batch_resolve_contradictions(count=count, strategy=strategy)
                        return {"action": "batch_resolve", **result}
                    elif action == "pending_contradictions":
                        limit = arguments.get("count", 20)
                        pending = prov_tracker.get_pending_contradictions(limit=limit)
                        return {"pending_count": len(pending), "contradictions": pending}
                    else:
                        return {"error": f"Unknown action: {action}. Valid actions: stats, get, provenance, source_chain, corrections, reinforcements, detect_contradictions, contradiction_stats, resolve_contradiction, batch_resolve, pending_contradictions, reinforce, decay, low, quarantine"}

                result = await run_in_thread(do_confidence)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "code":
            # Consolidated: replaces find_code, get_code_stats
            action = arguments.get("action", "search")
            # Fast NAS check - fail fast if NAS unavailable
            if not is_nas_reachable(timeout=2.0):
                return [TextContent(type="text", text=safe_json_dumps({"error": "NAS not reachable", "nas_status": "unavailable", "suggestion": "Check if NAS is powered on and Z: drive is mounted"}))]
            try:
                from code_indexer import CodeIndexer

                def do_code():
                    indexer = CodeIndexer()
                    if action == "stats":
                        return indexer.get_code_stats()
                    else:  # search
                        results = indexer.search_code(
                            query=arguments.get("query", ""),
                            language=arguments.get("language"),
                            project=arguments.get("project"),
                            limit=arguments.get("limit", 10)
                        )
                        return {"code_snippets": results, "count": len(results)}

                result = await run_in_thread(do_code)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "analyze":
            # Consolidated: replaces detect_recurring_patterns, find_knowledge_gaps, track_skill_development
            # Phase 4 Enhancement: Added validated_patterns type
            analysis_type = arguments.get("type", "patterns")
            try:
                if analysis_type == "skill":
                    from skill_tracker import SkillTracker
                    def do_analyze():
                        tracker = SkillTracker()
                        skill = arguments.get("skill")
                        if not skill:
                            return {"error": "skill parameter required"}
                        return tracker.track_skill_development(skill)
                elif analysis_type == "validated_patterns":
                    # Phase 4 Enhancement: Get validated patterns ready for injection
                    from patterns.applier import PatternApplier
                    from patterns.validator import PatternValidator
                    def do_analyze():
                        validator = PatternValidator()
                        max_per_type = arguments.get("max_per_type", 5)
                        validated = validator.get_validated_patterns(max_per_type=max_per_type)

                        # Format for display
                        result = {
                            "stats": validated.get("stats", {}),
                            "topics": [],
                            "problems": [],
                            "knowledge_gaps": [],
                            "injection_preview": ""
                        }

                        for r in validated.get("topics", []):
                            result["topics"].append({
                                "topic": r.pattern_data.get("topic", "unknown"),
                                "confidence": round(r.confidence, 2),
                                "occurrences": r.pattern_data.get("occurrences", 0)
                            })

                        for r in validated.get("problems", []):
                            result["problems"].append({
                                "problem": r.pattern_data.get("problem", "unknown")[:60],
                                "confidence": round(r.confidence, 2),
                                "occurrences": r.pattern_data.get("occurrences", 0)
                            })

                        for r in validated.get("knowledge_gaps", []):
                            result["knowledge_gaps"].append({
                                "concept": r.pattern_data.get("concept", "unknown"),
                                "confidence": round(r.confidence, 2),
                                "times_explained": r.pattern_data.get("times_explained", 0)
                            })

                        # Generate injection preview
                        applier = PatternApplier(max_patterns=5)
                        result["injection_preview"] = applier.get_injection_context(validated)

                        return result
                else:
                    from pattern_detector import PatternDetector
                    def do_analyze():
                        detector = PatternDetector()
                        threshold = arguments.get("threshold", 3)
                        if analysis_type == "knowledge_gaps":
                            gaps = detector.find_knowledge_gaps(min_explanations=threshold)
                            return {"knowledge_gaps": gaps, "count": len(gaps)}
                        else:  # patterns
                            topics = detector.detect_recurring_topics(threshold)
                            problems = detector.detect_recurring_problems()
                            return {"recurring_topics": topics[:10], "recurring_problems": problems[:10]}

                result = await run_in_thread(do_analyze)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "quality":
            # Consolidated: replaces find_duplicate_memories, merge_duplicates, score_memory_quality, get_quality_stats
            # Phase 2 Enhancement: Added fact_duplicates, fact_merge, fact_links
            action = arguments.get("action", "stats")
            try:
                def do_quality():
                    if action in ["duplicates", "merge"]:
                        # Chunk-level deduplication (original)
                        from deduplicator import MemoryDeduplicator
                        dedup = MemoryDeduplicator()
                        threshold = arguments.get("threshold", 0.90)
                        duplicates = dedup.find_duplicates(threshold)
                        if action == "merge":
                            auto = arguments.get("auto_merge", False)
                            return dedup.merge_duplicates(duplicates, auto_merge=auto)
                        return {"duplicates": duplicates[:20], "total_found": len(duplicates)}
                    elif action in ["fact_duplicates", "fact_merge"]:
                        # Fact-level deduplication (Phase 2 Enhancement)
                        from knowledge.fact_deduplicator import FactDeduplicator
                        dedup = FactDeduplicator()
                        threshold = arguments.get("threshold", 0.85)
                        if action == "fact_duplicates":
                            return dedup.deduplicate(threshold=threshold, dry_run=True)
                        else:  # fact_merge
                            return dedup.deduplicate(threshold=threshold, dry_run=False)
                    elif action == "fact_links":
                        # Fact linking (Phase 2 Enhancement)
                        from knowledge.fact_deduplicator import FactDeduplicator
                        from knowledge.fact_linker import FactLinker
                        dedup = FactDeduplicator()
                        linker = FactLinker()
                        facts = dedup.load_all_facts()
                        if not facts:
                            return {"error": "No facts found to link"}
                        graph = linker.build_link_graph(facts[:100])  # Limit for performance
                        linker.save_links(graph)
                        return {
                            "nodes": graph["node_count"],
                            "edges": graph["edge_count"],
                            "sample_links": graph["edges"][:10],
                        }
                    else:
                        from quality_scorer_v2 import QualityScorerV2
                        scorer = QualityScorerV2()
                        if action == "score":
                            scored = scorer.score_all_conversations()
                            return {"scored_conversations": scored[:50], "total": len(scored)}
                        else:  # stats
                            return scorer.get_quality_stats()

                result = await run_in_thread(do_quality, timeout=180)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "decay":
            # Phase 4: Active Decay - storage management and intelligent decay
            action = arguments.get("action", "stats")
            try:
                def do_decay():
                    if action == "run":
                        from decay_pipeline import run_decay
                        force = arguments.get("force", False)
                        include_summaries = arguments.get("include_summaries", True)
                        include_conversations = arguments.get("include_conversations", True)
                        include_facts = arguments.get("include_facts", True)
                        return run_decay(
                            force=force,
                            include_summaries=include_summaries,
                            include_conversations=include_conversations,
                            include_facts=include_facts
                        )
                    elif action == "preview":
                        from decay_pipeline import get_decay_preview
                        return get_decay_preview()
                    elif action == "stats":
                        from decay_pipeline import get_decay_stats
                        return get_decay_stats()
                    elif action == "storage":
                        from storage_manager import StorageManager
                        manager = StorageManager(AI_MEMORY_BASE)
                        return manager.get_storage_report()
                    elif action == "golden":
                        from usage_tracker import UsageTracker
                        tracker = UsageTracker(AI_MEMORY_BASE)
                        golden_action = arguments.get("golden_action", "list")
                        item_id = arguments.get("item_id")

                        if golden_action == "list":
                            items = tracker.get_golden_items()
                            return {
                                "golden_items": items,
                                "count": len(items)
                            }
                        elif golden_action == "add" and item_id:
                            tracker.mark_golden(item_id, reason="Marked via MCP tool")
                            return {
                                "success": True,
                                "message": f"Marked {item_id} as golden (never decay)"
                            }
                        elif golden_action == "remove" and item_id:
                            result = tracker.unmark_golden(item_id)
                            return {
                                "success": result,
                                "message": f"Removed golden status from {item_id}" if result else f"{item_id} was not golden"
                            }
                        else:
                            return {"error": "For add/remove, item_id is required"}
                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_decay, timeout=300)  # 5 min timeout for decay
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "privacy":
            # Phase 5: Privacy and secret management
            action = arguments.get("action", "stats")
            try:
                def do_privacy():
                    from privacy_filter import PrivacyFilter
                    from secret_detector import SecretDetector

                    if action == "scan":
                        text = arguments.get("text", "")
                        if not text:
                            return {"error": "text parameter required for scan action"}

                        redact = arguments.get("redact", False)
                        min_confidence = arguments.get("min_confidence", 0.7)

                        # Detect secrets
                        detector = SecretDetector()
                        secrets = detector.scan(text, min_confidence=min_confidence)

                        result = {
                            "secrets_found": len(secrets),
                            "details": [
                                {
                                    "type": s.secret_type.value,
                                    "confidence": s.confidence,
                                    "pattern": s.pattern_name,
                                    "position": {"start": s.start, "end": s.end}
                                }
                                for s in secrets
                            ]
                        }

                        if redact:
                            filter = PrivacyFilter(min_confidence=min_confidence)
                            redaction = filter.redact(text)
                            result["redacted_text"] = redaction.redacted_text
                            result["sensitivity_level"] = redaction.sensitivity_level.value

                        return result

                    elif action == "stats":
                        # Get redaction statistics from saved conversations
                        conversations_path = AI_MEMORY_BASE / "conversations"

                        total_convos = 0
                        secrets_redacted = 0
                        sensitivity_counts = {}

                        if conversations_path.exists():
                            for conv_file in conversations_path.glob("*.json"):
                                try:
                                    with open(conv_file) as f:
                                        conv = json.load(f)
                                    total_convos += 1
                                    privacy = conv.get("privacy", {})
                                    if privacy.get("enabled"):
                                        secrets_redacted += privacy.get("secrets_redacted", 0)
                                        level = privacy.get("sensitivity_level", "public")
                                        sensitivity_counts[level] = sensitivity_counts.get(level, 0) + 1
                                except:
                                    pass

                        return {
                            "total_conversations": total_convos,
                            "total_secrets_redacted": secrets_redacted,
                            "sensitivity_distribution": sensitivity_counts,
                            "privacy_filter_enabled": True
                        }

                    elif action == "sensitive":
                        # List sensitive conversations
                        conversations_path = AI_MEMORY_BASE / "conversations"
                        limit = arguments.get("limit", 20)

                        sensitive_convos = []
                        if conversations_path.exists():
                            for conv_file in sorted(conversations_path.glob("*.json"), reverse=True):
                                if len(sensitive_convos) >= limit:
                                    break
                                try:
                                    with open(conv_file) as f:
                                        conv = json.load(f)
                                    privacy = conv.get("privacy", {})
                                    level = privacy.get("sensitivity_level", "public")
                                    if level in ("confidential", "secret", "top_secret"):
                                        sensitive_convos.append({
                                            "id": conv.get("id"),
                                            "timestamp": conv.get("timestamp"),
                                            "sensitivity_level": level,
                                            "secrets_redacted": privacy.get("secrets_redacted", 0),
                                            "secret_types": privacy.get("secret_types", [])
                                        })
                                except:
                                    pass

                        return {
                            "sensitive_conversations": sensitive_convos,
                            "count": len(sensitive_convos)
                        }

                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_privacy, timeout=60)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # PHASE 1: EPISODIC vs SEMANTIC MEMORY (v6.0)
        # ============================================================
        elif name == "memory_type":
            action = arguments.get("action", "stats")
            try:
                def do_memory_type():
                    from episodic_memory import EpisodicMemoryManager
                    from memory_types import EpisodicMemory
                    from semantic_memory import SemanticMemoryManager

                    episodic = EpisodicMemoryManager(base_path=str(AI_MEMORY_BASE))
                    semantic = SemanticMemoryManager(base_path=str(AI_MEMORY_BASE))

                    if action == "query_episodic":
                        # Query episodic memories (events)
                        query = arguments.get("query")
                        date = arguments.get("date")
                        date_range = arguments.get("date_range")
                        actor = arguments.get("actor")
                        emotion = arguments.get("emotion")
                        limit = arguments.get("limit", 10)

                        if date:
                            episodes = episodic.query_by_date(date)
                        elif date_range:
                            episodes = episodic.query_by_date_range(
                                date_range.get("start"),
                                date_range.get("end")
                            )
                        elif actor:
                            episodes = episodic.query_by_actor(actor)
                        elif emotion:
                            episodes = episodic.query_by_emotion(emotion)
                        elif query:
                            episodes = episodic.search_episodes(query, limit=limit)
                        else:
                            episodes = episodic.get_recent_episodes(days=7, limit=limit)

                        return {
                            "type": "episodic",
                            "count": len(episodes),
                            "episodes": [e.to_dict() for e in episodes[:limit]]
                        }

                    elif action == "query_semantic":
                        # Query semantic memories (facts)
                        query = arguments.get("query")
                        domain = arguments.get("domain")
                        keywords = arguments.get("keywords", [])
                        limit = arguments.get("limit", 10)

                        if domain:
                            memories = semantic.query_by_domain(domain, limit=limit)
                        elif keywords:
                            memories = semantic.query_by_keywords(keywords, limit=limit)
                        elif query:
                            memories = semantic.search(query, limit=limit)
                        else:
                            memories = semantic.get_high_confidence(threshold=0.7, limit=limit)

                        return {
                            "type": "semantic",
                            "count": len(memories),
                            "facts": [m.to_dict() for m in memories[:limit]]
                        }

                    elif action == "save_episodic":
                        # Save an episodic memory
                        event = arguments.get("event")
                        if not event:
                            return {"error": "event parameter required"}

                        episode = EpisodicMemory(
                            id=episodic._generate_id(),
                            timestamp=dt.datetime.now().isoformat(),
                            event=event,
                            outcome=arguments.get("outcome"),
                            emotional_state=arguments.get("emotional_state"),
                            actors=["Professor", "Claude"],
                            conversation_id=arguments.get("conversation_id")
                        )
                        episode_id = episodic.save_episode(episode)
                        return {
                            "success": True,
                            "episode_id": episode_id,
                            "event": event[:200]
                        }

                    elif action == "save_semantic":
                        # Save a semantic memory
                        fact = arguments.get("fact")
                        if not fact:
                            return {"error": "fact parameter required"}

                        mem, created = semantic.find_or_create(
                            fact=fact,
                            domain=arguments.get("domain")
                        )
                        return {
                            "success": True,
                            "semantic_id": mem.id,
                            "fact": fact[:200],
                            "created": created,
                            "confidence": mem.confidence
                        }

                    elif action == "link":
                        # Link episodic to semantic
                        episode_id = arguments.get("episode_id")
                        semantic_id = arguments.get("semantic_id")
                        if not episode_id or not semantic_id:
                            return {"error": "episode_id and semantic_id required"}

                        episodic.link_to_semantic(episode_id, semantic_id)
                        return {
                            "success": True,
                            "linked": f"{episode_id} -> {semantic_id}"
                        }

                    elif action == "migrate":
                        # Migrate existing facts to semantic memory
                        migrated = semantic.migrate_from_facts()
                        return {
                            "success": True,
                            "facts_migrated": migrated
                        }

                    elif action == "stats":
                        # Get statistics
                        ep_stats = episodic.get_stats()
                        sem_stats = semantic.get_stats()
                        return {
                            "episodic": ep_stats,
                            "semantic": sem_stats
                        }

                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_memory_type, timeout=60)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # PHASE 2: WORKING MEMORY (v6.0)
        # ============================================================
        elif name == "working_memory":
            action = arguments.get("action", "get_active")
            try:
                def do_working_memory():
                    from working_memory import WorkingMemoryManager

                    wm = WorkingMemoryManager(base_path=str(AI_MEMORY_BASE))

                    if action == "get_active":
                        # Get active session (create if none exists)
                        active_goal = arguments.get("active_goal")
                        session = wm.get_or_create_session(active_goal=active_goal)
                        return {
                            "session_id": session["session_id"],
                            "active_goal": session.get("active_goal"),
                            "expires_at": session.get("expires_at"),
                            "chain_count": len(session.get("reasoning_chains", [])),
                            "note_count": len(session.get("scratch_pad", {}).get("notes", []))
                        }

                    elif action == "create":
                        # Create new session
                        active_goal = arguments.get("active_goal")
                        session = wm.create_session(active_goal=active_goal)
                        return {
                            "success": True,
                            "session_id": session["session_id"],
                            "active_goal": active_goal
                        }

                    elif action == "add_chain":
                        # Add a reasoning chain
                        session_id = arguments.get("session_id")
                        if not session_id:
                            session = wm.get_active_session()
                            session_id = session["session_id"] if session else None
                        if not session_id:
                            return {"error": "No active session found"}

                        hypothesis = arguments.get("hypothesis")
                        if not hypothesis:
                            return {"error": "hypothesis parameter required"}

                        chain = wm.add_reasoning_chain(session_id, hypothesis)
                        return {
                            "success": True,
                            "chain_id": chain.chain_id,
                            "hypothesis": hypothesis[:100]
                        }

                    elif action == "update_chain":
                        # Update a reasoning chain
                        session_id = arguments.get("session_id")
                        if not session_id:
                            session = wm.get_active_session()
                            session_id = session["session_id"] if session else None
                        if not session_id:
                            return {"error": "No active session found"}

                        chain_id = arguments.get("chain_id")
                        if not chain_id:
                            return {"error": "chain_id parameter required"}

                        chain = wm.update_reasoning_chain(
                            session_id=session_id,
                            chain_id=chain_id,
                            evidence=arguments.get("evidence"),
                            evaluation=arguments.get("evaluation"),
                            next_step=arguments.get("next_step"),
                            outcome=arguments.get("outcome")
                        )

                        if chain:
                            return {
                                "success": True,
                                "chain_id": chain_id,
                                "evaluation": chain.evaluation,
                                "next_step": chain.next_step
                            }
                        return {"error": f"Chain {chain_id} not found"}

                    elif action == "add_note":
                        # Add scratch pad note
                        session_id = arguments.get("session_id")
                        if not session_id:
                            session = wm.get_active_session()
                            session_id = session["session_id"] if session else None
                        if not session_id:
                            return {"error": "No active session found"}

                        note = arguments.get("note")
                        if not note:
                            return {"error": "note parameter required"}

                        category = arguments.get("note_category", "notes")
                        wm.add_scratch_note(session_id, note, category)
                        return {"success": True, "note_added": note[:100]}

                    elif action == "get_summary":
                        # Get session summary
                        session_id = arguments.get("session_id")
                        if not session_id:
                            session = wm.get_active_session()
                            session_id = session["session_id"] if session else None
                        if not session_id:
                            return {"error": "No active session found"}

                        return wm.get_summary(session_id)

                    elif action == "export":
                        # Export for handoff
                        session_id = arguments.get("session_id")
                        if not session_id:
                            session = wm.get_active_session()
                            session_id = session["session_id"] if session else None
                        if not session_id:
                            return {"error": "No active session found"}

                        return wm.export_for_handoff(session_id)

                    elif action == "import":
                        # Import from handoff
                        handoff_data = arguments.get("handoff_data")
                        if not handoff_data:
                            return {"error": "handoff_data parameter required"}

                        new_session_id = wm.import_from_handoff(handoff_data)
                        return {
                            "success": True,
                            "new_session_id": new_session_id
                        }

                    elif action == "archive":
                        # Archive a session
                        session_id = arguments.get("session_id")
                        if not session_id:
                            return {"error": "session_id parameter required"}
                        wm.archive_session(session_id)
                        return {"success": True, "archived": session_id}

                    elif action == "extend":
                        # Extend session expiry
                        session_id = arguments.get("session_id")
                        if not session_id:
                            session = wm.get_active_session()
                            session_id = session["session_id"] if session else None
                        if not session_id:
                            return {"error": "No active session found"}

                        hours = arguments.get("hours", 4)
                        wm.extend_session(session_id, hours)
                        return {"success": True, "extended_hours": hours}

                    elif action == "stats":
                        return wm.get_stats()

                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_working_memory, timeout=60)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # PHASE 3: CAUSAL MODEL (v6.0)
        # ============================================================
        elif name == "causal":
            action = arguments.get("action", "stats")
            try:
                def do_causal():
                    from causal_extractor import CausalExtractor
                    from causal_model import CausalModelManager

                    cm = CausalModelManager(base_path=str(AI_MEMORY_BASE))

                    if action == "add_link":
                        cause = arguments.get("cause")
                        effect = arguments.get("effect")
                        if not cause or not effect:
                            return {"error": "cause and effect parameters required"}

                        link = cm.add_link(
                            cause=cause,
                            effect=effect,
                            mechanism=arguments.get("mechanism"),
                            counterfactual=arguments.get("counterfactual"),
                            interventions=arguments.get("interventions", [])
                        )
                        return {
                            "success": True,
                            "link_id": link.link_id,
                            "cause": cause[:100],
                            "effect": effect[:100],
                            "confidence": link.confidence,
                            "evidence_count": link.evidence_count
                        }

                    elif action == "find_causes":
                        effect = arguments.get("effect")
                        if not effect:
                            return {"error": "effect parameter required"}

                        threshold = arguments.get("confidence_threshold", 0.5)
                        links = cm.find_causes(effect, threshold=threshold)
                        return {
                            "effect": effect[:100],
                            "count": len(links),
                            "causes": [
                                {
                                    "link_id": l.link_id,
                                    "cause": l.cause,
                                    "mechanism": l.mechanism,
                                    "confidence": l.confidence,
                                    "interventions": l.interventions
                                }
                                for l in links
                            ]
                        }

                    elif action == "find_effects":
                        cause = arguments.get("cause")
                        if not cause:
                            return {"error": "cause parameter required"}

                        threshold = arguments.get("confidence_threshold", 0.5)
                        links = cm.find_effects(cause, threshold=threshold)
                        return {
                            "cause": cause[:100],
                            "count": len(links),
                            "effects": [
                                {
                                    "link_id": l.link_id,
                                    "effect": l.effect,
                                    "mechanism": l.mechanism,
                                    "confidence": l.confidence
                                }
                                for l in links
                            ]
                        }

                    elif action == "get_interventions":
                        effect = arguments.get("effect")
                        if not effect:
                            return {"error": "effect parameter required"}

                        interventions = cm.get_interventions(effect)
                        return {
                            "effect": effect[:100],
                            "count": len(interventions),
                            "interventions": interventions
                        }

                    elif action == "what_if":
                        intervention = arguments.get("intervention")
                        if not intervention:
                            return {"error": "intervention parameter required"}

                        predictions = cm.simulate_what_if(intervention)
                        return {
                            "intervention": intervention[:100],
                            "count": len(predictions),
                            "predictions": predictions
                        }

                    elif action == "search":
                        query = arguments.get("query")
                        if not query:
                            return {"error": "query parameter required"}

                        limit = arguments.get("limit", 10)
                        links = cm.search_links(query, limit=limit)
                        return {
                            "query": query,
                            "count": len(links),
                            "links": [l.to_dict() for l in links]
                        }

                    elif action == "reinforce":
                        link_id = arguments.get("link_id")
                        if not link_id:
                            return {"error": "link_id parameter required"}

                        link = cm.reinforce_link(link_id)
                        if link:
                            return {
                                "success": True,
                                "link_id": link_id,
                                "new_confidence": link.confidence,
                                "evidence_count": link.evidence_count
                            }
                        return {"error": f"Link {link_id} not found"}

                    elif action == "weaken":
                        link_id = arguments.get("link_id")
                        if not link_id:
                            return {"error": "link_id parameter required"}

                        link = cm.weaken_link(link_id)
                        if link:
                            return {
                                "success": True,
                                "link_id": link_id,
                                "new_confidence": link.confidence
                            }
                        return {"error": f"Link {link_id} not found"}

                    elif action == "extract":
                        text = arguments.get("text")
                        if not text:
                            return {"error": "text parameter required"}

                        extractor = CausalExtractor()
                        use_llm = arguments.get("use_llm", False)
                        results = extractor.extract_from_text(text, use_llm=use_llm)

                        # Optionally save extracted links
                        saved_links = []
                        for r in results:
                            if r.get("confidence", 0) >= 0.6:
                                link = cm.add_link(
                                    cause=r["cause"],
                                    effect=r["effect"],
                                    mechanism=r.get("mechanism"),
                                    counterfactual=r.get("counterfactual"),
                                    interventions=r.get("interventions", []),
                                    confidence=r.get("confidence", 0.6)
                                )
                                saved_links.append(link.link_id)

                        return {
                            "extracted": len(results),
                            "saved": len(saved_links),
                            "links": results,
                            "saved_link_ids": saved_links
                        }

                    elif action == "stats":
                        return cm.get_stats()

                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_causal, timeout=60)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # PHASE 4: ACTIVE REASONING (v6.0)
        # ============================================================
        elif name == "reason":
            action = arguments.get("action", "stats")
            try:
                def do_reason():
                    from active_reasoner import ActiveReasoner

                    reasoner = ActiveReasoner(base_path=str(AI_MEMORY_BASE))

                    if action == "analyze":
                        memories = arguments.get("memories", [])
                        context = arguments.get("context")
                        use_llm = arguments.get("use_llm", False)

                        if not memories:
                            # Load recent conversations as memories
                            convs_path = AI_MEMORY_BASE / "conversations"
                            if convs_path.exists():
                                for conv_file in sorted(convs_path.glob("*.json"), reverse=True)[:10]:
                                    try:
                                        with open(conv_file) as f:
                                            memories.append(json.load(f))
                                    except:
                                        pass

                        result = reasoner.reason_about_memories(memories, context, use_llm=use_llm)
                        return result

                    elif action == "find_insights":
                        query = arguments.get("query")
                        if not query:
                            return {"error": "query parameter required"}

                        limit = arguments.get("limit", 5)
                        insights = reasoner.find_relevant_insights(query, limit=limit)
                        return {
                            "query": query,
                            "count": len(insights),
                            "insights": [i.to_dict() for i in insights]
                        }

                    elif action == "proactive":
                        user_message = arguments.get("user_message")
                        if not user_message:
                            return {"error": "user_message parameter required"}

                        # Load recent memories
                        recent_memories = []
                        convs_path = AI_MEMORY_BASE / "conversations"
                        if convs_path.exists():
                            for conv_file in sorted(convs_path.glob("*.json"), reverse=True)[:5]:
                                try:
                                    with open(conv_file) as f:
                                        recent_memories.append(json.load(f))
                                except:
                                    pass

                        insights = reasoner.proactive_insights(user_message, recent_memories)
                        return {
                            "insights": insights,
                            "count": len(insights)
                        }

                    elif action == "validate":
                        insight_id = arguments.get("insight_id")
                        is_valid = arguments.get("is_valid")

                        if not insight_id:
                            return {"error": "insight_id parameter required"}
                        if is_valid is None:
                            return {"error": "is_valid parameter required"}

                        return reasoner.validate_insight(insight_id, is_valid)

                    elif action == "goal":
                        goal = arguments.get("goal")
                        if not goal:
                            return {"error": "goal parameter required"}

                        # Load semantic memories as available knowledge
                        knowledge = []
                        semantic_path = AI_MEMORY_BASE / "semantic"
                        if semantic_path.exists():
                            for sem_file in semantic_path.glob("sem_*.json"):
                                try:
                                    with open(sem_file) as f:
                                        knowledge.append(json.load(f))
                                except:
                                    pass

                        return reasoner.reason_about_goal(goal, knowledge)

                    elif action == "stats":
                        return reasoner.get_stats()

                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_reason, timeout=90)  # Longer timeout for reasoning
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # PHASE 5: SESSION HANDOFF (v6.0)
        # ============================================================
        elif name == "session_handoff":
            action = arguments.get("action", "get_latest")
            try:
                def do_handoff():
                    from session_continuity import SessionContinuityManager
                    from working_memory import WorkingMemoryManager

                    scm = SessionContinuityManager(base_path=str(AI_MEMORY_BASE))

                    if action == "save":
                        handoff_data = arguments.get("handoff_data")
                        if not handoff_data:
                            # Auto-export from working memory
                            wm = WorkingMemoryManager(base_path=str(AI_MEMORY_BASE))
                            session = wm.get_active_session()
                            if session:
                                handoff_data = wm.export_for_handoff(session["session_id"])
                            else:
                                return {"error": "No handoff_data and no active working memory session"}

                        reason = arguments.get("reason", "manual")
                        handoff_id = scm.save_handoff(handoff_data, reason)

                        # Also update quick_facts
                        scm.integrate_with_quick_facts(handoff_data)

                        return {
                            "success": True,
                            "handoff_id": handoff_id,
                            "continuation_prompt": scm._generate_continuation_prompt(handoff_data)
                        }

                    elif action == "get_latest":
                        handoff = scm.get_latest_handoff()
                        if handoff:
                            return handoff
                        return {"error": "No handoffs found"}

                    elif action == "get_recent":
                        hours = arguments.get("hours", 48)
                        handoffs = scm.get_recent_handoffs(hours)
                        return {
                            "count": len(handoffs),
                            "handoffs": handoffs
                        }

                    elif action == "restore":
                        handoff_id = arguments.get("handoff_id")
                        if handoff_id:
                            # Find specific handoff
                            handoffs_path = AI_MEMORY_BASE / "session_handoffs"
                            handoff_file = handoffs_path / f"{handoff_id}.json"
                            if handoff_file.exists():
                                with open(handoff_file) as f:
                                    handoff = json.load(f)
                            else:
                                return {"error": f"Handoff {handoff_id} not found"}
                        else:
                            # Use latest
                            handoff = scm.get_latest_handoff()
                            if not handoff:
                                return {"error": "No handoffs to restore from"}

                        # Restore to working memory
                        wm = WorkingMemoryManager(base_path=str(AI_MEMORY_BASE))
                        new_session_id = wm.import_from_handoff(handoff)

                        return {
                            "success": True,
                            "restored_from": handoff.get("id"),
                            "new_session_id": new_session_id,
                            "active_goal": handoff.get("active_goal"),
                            "incomplete_reasoning": len(handoff.get("incomplete_reasoning", []))
                        }

                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_handoff, timeout=30)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # PHASE 6: GOAL-DIRECTED ACCESS (v6.0)
        # ============================================================
        elif name == "goals":
            action = arguments.get("action", "list_active")
            try:
                def do_goals():
                    from goal_tracker import Goal, GoalTracker
                    from proactive_retrieval import ProactiveRetrieval

                    gt = GoalTracker(base_path=str(AI_MEMORY_BASE))
                    pr = ProactiveRetrieval(base_path=str(AI_MEMORY_BASE))

                    if action == "detect":
                        text = arguments.get("text")
                        if not text:
                            return {"error": "text parameter required"}

                        result = pr.detect_and_track_goal(text)
                        return result

                    elif action == "add":
                        description = arguments.get("description")
                        if not description:
                            return {"error": "description parameter required"}

                        goal = Goal(
                            goal_id=gt._generate_id(),
                            description=description,
                            inferred_from="explicit",
                            priority=arguments.get("priority", "medium")
                        )
                        goal_id = gt.add_goal(goal)
                        return {
                            "success": True,
                            "goal_id": goal_id,
                            "description": description[:100]
                        }

                    elif action == "get":
                        goal_id = arguments.get("goal_id")
                        if not goal_id:
                            return {"error": "goal_id parameter required"}

                        goal = gt.get_goal(goal_id)
                        if goal:
                            return goal.to_dict()
                        return {"error": f"Goal {goal_id} not found"}

                    elif action == "update":
                        goal_id = arguments.get("goal_id")
                        if not goal_id:
                            return {"error": "goal_id parameter required"}

                        goal = gt.update_goal(
                            goal_id,
                            status=arguments.get("status"),
                            add_subgoal=arguments.get("add_subgoal"),
                            add_blocker=arguments.get("add_blocker")
                        )
                        if goal:
                            return {"success": True, "goal": goal.to_dict()}
                        return {"error": f"Goal {goal_id} not found"}

                    elif action == "complete":
                        goal_id = arguments.get("goal_id")
                        if not goal_id:
                            return {"error": "goal_id parameter required"}

                        success = gt.complete_goal(goal_id)
                        return {"success": success, "goal_id": goal_id}

                    elif action == "list_active":
                        goals = gt.get_active_goals()
                        return {
                            "count": len(goals),
                            "goals": [g.to_dict() for g in goals]
                        }

                    elif action == "proactive_context":
                        text = arguments.get("text")
                        if not text:
                            return {"error": "text parameter required"}

                        context = pr.get_proactive_context(text)

                        # Also check for pitfalls
                        pitfall = pr.surface_pitfall_warning(text)
                        if pitfall:
                            context["warnings"].append(pitfall)

                        return context

                    elif action == "find_relevant":
                        context = arguments.get("context")
                        if not context:
                            return {"error": "context parameter required"}

                        goals = gt.find_relevant_goals(context)
                        return {
                            "count": len(goals),
                            "goals": [g.to_dict() for g in goals]
                        }

                    elif action == "stats":
                        return pr.get_stats()

                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_goals, timeout=30)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # PHASE 7: PREDICTIVE SIMULATION (v6.0)
        # ============================================================
        elif name == "predict":
            action = arguments.get("action", "stats")
            try:
                def do_predict():
                    from failure_anticipator import FailureAnticipator
                    from predictor import Predictor

                    pred = Predictor(base_path=str(AI_MEMORY_BASE))
                    fa = FailureAnticipator(base_path=str(AI_MEMORY_BASE))

                    if action == "from_causal":
                        action_text = arguments.get("action_text")
                        if not action_text:
                            return {"error": "action_text parameter required"}

                        context = arguments.get("context")
                        prediction = pred.predict_from_causal(action_text, context)
                        return prediction.to_dict()

                    elif action == "anticipate_failures":
                        context = arguments.get("context")
                        if not context:
                            return {"error": "context parameter required"}

                        warnings = fa.anticipate_failures(context)
                        return {
                            "count": len(warnings),
                            "warnings": warnings
                        }

                    elif action == "check_pattern":
                        pattern_type = arguments.get("pattern_type")
                        context = arguments.get("context")
                        if not pattern_type or not context:
                            return {"error": "pattern_type and context parameters required"}

                        warning = fa.check_specific_pattern(pattern_type, context)
                        if warning:
                            return {"pattern_detected": True, "warning": warning}
                        return {"pattern_detected": False}

                    elif action == "preventive_actions":
                        context = arguments.get("context")
                        if not context:
                            return {"error": "context parameter required"}

                        actions = fa.get_preventive_actions(context)
                        return {
                            "count": len(actions),
                            "actions": actions
                        }

                    elif action == "verify":
                        prediction_id = arguments.get("prediction_id")
                        outcome = arguments.get("outcome")
                        if not prediction_id or not outcome:
                            return {"error": "prediction_id and outcome parameters required"}

                        result = pred.verify_prediction(prediction_id, outcome)
                        return result

                    elif action == "stats":
                        pred_stats = pred.get_stats()
                        fa_stats = fa.get_stats()
                        return {
                            "predictor": pred_stats,
                            "failure_anticipator": fa_stats
                        }

                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_predict, timeout=30)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # PHASE 8: META-LEARNING (v6.0)
        # ============================================================
        elif name == "meta_learn":
            action = arguments.get("action", "get_stats")
            try:
                def do_meta_learn():
                    from meta_learner import MetaLearner
                    from strategy_optimizer import StrategyOptimizer

                    ml = MetaLearner(base_path=str(AI_MEMORY_BASE))
                    so = StrategyOptimizer(base_path=str(AI_MEMORY_BASE))

                    if action == "record_query":
                        query = arguments.get("query")
                        strategy_id = arguments.get("strategy_id")
                        latency_ms = arguments.get("latency_ms", 0)
                        result_count = arguments.get("result_count", 0)
                        success = arguments.get("success")

                        if not query or not strategy_id:
                            return {"error": "query and strategy_id parameters required"}

                        query_id = ml.record_query(
                            query=query,
                            strategy_id=strategy_id,
                            latency_ms=latency_ms,
                            result_count=result_count,
                            success=success
                        )
                        return {
                            "success": True,
                            "query_id": query_id,
                            "query_type": ml.classify_query_type(query)
                        }

                    elif action == "feedback":
                        query_id = arguments.get("query_id")
                        positive = arguments.get("positive")
                        if not query_id or positive is None:
                            return {"error": "query_id and positive parameters required"}

                        return ml.record_feedback(query_id, positive)

                    elif action == "recommend":
                        query = arguments.get("query")
                        if not query:
                            return {"error": "query parameter required"}

                        return ml.recommend_strategy(query)

                    elif action == "get_stats":
                        ml_stats = ml.get_stats()
                        so_stats = so.get_stats()
                        return {
                            "meta_learner": ml_stats,
                            "strategy_optimizer": so_stats
                        }

                    elif action == "create_experiment":
                        name = arguments.get("experiment_name")
                        strategy_a = arguments.get("strategy_a")
                        strategy_b = arguments.get("strategy_b")
                        if not name or not strategy_a or not strategy_b:
                            return {"error": "experiment_name, strategy_a, and strategy_b required"}

                        return so.create_experiment(name, strategy_a, strategy_b)

                    elif action == "experiment_result":
                        experiment_id = arguments.get("experiment_id")
                        strategy_id = arguments.get("strategy_id")
                        success = arguments.get("success")
                        if not experiment_id or not strategy_id or success is None:
                            return {"error": "experiment_id, strategy_id, and success required"}

                        return so.record_experiment_result(experiment_id, strategy_id, success)

                    elif action == "list_experiments":
                        return {
                            "experiments": so.list_experiments(),
                            "count": len(so.list_experiments())
                        }

                    elif action == "tune_parameter":
                        param_name = arguments.get("param_name")
                        if not param_name:
                            return {"error": "param_name required"}

                        # If score provided, record performance
                        score = arguments.get("score")
                        if score is not None:
                            value = arguments.get("param_value")
                            if value is None:
                                return {"error": "param_value required when recording score"}
                            return so.record_parameter_performance(param_name, value, score)

                        # Otherwise, setup tuning
                        current = arguments.get("param_value")
                        min_val = arguments.get("param_min")
                        max_val = arguments.get("param_max")
                        step = arguments.get("param_step")
                        if None in [current, min_val, max_val, step]:
                            return {"error": "param_value, param_min, param_max, param_step required for setup"}

                        return so.setup_parameter_tuning(param_name, current, min_val, max_val, step)

                    elif action == "suggest_parameter":
                        param_name = arguments.get("param_name")
                        if not param_name:
                            return {"error": "param_name required"}

                        return so.suggest_parameter_value(param_name)

                    elif action == "set_default":
                        strategy_id = arguments.get("strategy_id")
                        if not strategy_id:
                            return {"error": "strategy_id required"}

                        return ml.set_default_strategy(strategy_id)

                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_meta_learn, timeout=30)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # PHASE 9: CONTINUOUS SELF-MODELING (v6.0)
        # ============================================================
        elif name == "self_model":
            action = arguments.get("action", "get_state")
            try:
                def do_self_model():
                    from introspection import Introspector
                    from self_model import SelfModelManager

                    sm = SelfModelManager(base_path=str(AI_MEMORY_BASE))
                    intro = Introspector(base_path=str(AI_MEMORY_BASE))

                    if action == "get_state":
                        return sm.get_current_state()

                    elif action == "update_confidence":
                        confidence = arguments.get("confidence")
                        topic = arguments.get("topic")
                        if confidence is None:
                            return {"error": "confidence parameter required"}

                        return sm.update_confidence(confidence, topic)

                    elif action == "add_uncertainty":
                        topic = arguments.get("topic")
                        if not topic:
                            return {"error": "topic parameter required"}

                        return sm.add_uncertainty(topic)

                    elif action == "add_limitation":
                        topic = arguments.get("topic")
                        if not topic:
                            return {"error": "topic parameter required"}

                        return sm.add_limitation(topic)

                    elif action == "add_strength":
                        topic = arguments.get("topic")
                        if not topic:
                            return {"error": "topic parameter required"}

                        return sm.add_strength(topic)

                    elif action == "update_quality":
                        return sm.update_reasoning_quality(
                            task_clarity=arguments.get("task_clarity"),
                            evidence_sufficiency=arguments.get("evidence_sufficiency"),
                            hallucination_risk=arguments.get("hallucination_risk")
                        )

                    elif action == "assess_text":
                        text = arguments.get("text")
                        if not text:
                            return {"error": "text parameter required"}

                        return sm.assess_text(text)

                    elif action == "introspect":
                        text = arguments.get("text")
                        if not text:
                            return {"error": "text parameter required"}

                        context = arguments.get("context")
                        use_llm = arguments.get("use_llm", False)

                        if use_llm:
                            result = intro.evaluate_with_llm(text, context)
                        else:
                            result = intro.analyze_reasoning(text, context)

                        return result.to_dict()

                    elif action == "hallucination_check":
                        text = arguments.get("text")
                        if not text:
                            return {"error": "text parameter required"}

                        return intro.detect_hallucination_risk(text)

                    elif action == "take_snapshot":
                        context = arguments.get("context")
                        snapshot_id = sm.take_snapshot(context)
                        return {"success": True, "snapshot_id": snapshot_id}

                    elif action == "get_snapshot":
                        snapshot_id = arguments.get("snapshot_id")
                        if not snapshot_id:
                            return {"error": "snapshot_id required"}

                        snapshot = sm.get_snapshot(snapshot_id)
                        if snapshot:
                            return snapshot
                        return {"error": f"Snapshot {snapshot_id} not found"}

                    elif action == "reset_daily":
                        return sm.reset_daily()

                    elif action == "stats":
                        sm_stats = sm.get_stats()
                        intro_stats = intro.get_stats()
                        return {
                            "self_model": sm_stats,
                            "introspection": intro_stats
                        }

                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_self_model, timeout=30)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # PHASE 10: ACTIVE MEMORY CONSOLIDATION (v6.0)
        # ============================================================
        elif name == "consolidate":
            action = arguments.get("action", "stats")
            try:
                def do_consolidate():
                    from abstractor import Abstractor
                    from consolidator import Consolidator

                    cons = Consolidator(base_path=str(AI_MEMORY_BASE))
                    abstr = Abstractor(base_path=str(AI_MEMORY_BASE))

                    if action == "run":
                        full = arguments.get("full", False)
                        return cons.run_consolidation(full=full)

                    elif action == "get_state":
                        return cons.get_state()

                    elif action == "get_run":
                        run_id = arguments.get("run_id")
                        if not run_id:
                            return {"error": "run_id required"}

                        run = cons.get_run(run_id)
                        if run:
                            return run
                        return {"error": f"Run {run_id} not found"}

                    elif action == "list_runs":
                        limit = arguments.get("limit", 10)
                        return {
                            "runs": cons.get_recent_runs(limit=limit),
                            "count": len(cons.get_recent_runs(limit=limit))
                        }

                    elif action == "get_abstractions":
                        domain = arguments.get("domain")
                        abstractions = abstr.get_all_abstractions(domain=domain)
                        limit = arguments.get("limit", 20)
                        return {
                            "abstractions": abstractions[:limit],
                            "total": len(abstractions)
                        }

                    elif action == "schedule":
                        hours = arguments.get("hours", 24)
                        return cons.schedule_next_run(hours_from_now=hours)

                    elif action == "stats":
                        cons_stats = cons.get_stats()
                        abstr_stats = abstr.get_stats()
                        return {
                            "consolidator": cons_stats,
                            "abstractor": abstr_stats
                        }

                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_consolidate, timeout=120)  # Longer timeout for consolidation
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "self_report":
            # Phase 7: Self-improvement metrics and reporting
            action = arguments.get("action", "report")
            try:
                def do_self_report():
                    from reflection.improvement_tracker import ImprovementTracker
                    from reflection.performance_tracker import TRACKED_METRICS, PerformanceTracker

                    perf_tracker = PerformanceTracker()
                    imp_tracker = ImprovementTracker()

                    if action == "metrics":
                        days = arguments.get("days", 30)
                        summary = perf_tracker.get_all_metrics_summary(days=days)
                        derived = perf_tracker.calculate_derived_metrics()
                        summary["derived_metrics"] = derived
                        return summary

                    elif action == "improvements":
                        return imp_tracker.get_all_improvements()

                    elif action == "record_metric":
                        metric_name = arguments.get("metric_name")
                        value = arguments.get("value")
                        if not metric_name or value is None:
                            return {"error": "metric_name and value required"}
                        success = perf_tracker.record_metric(
                            metric_name=metric_name,
                            value=value,
                            context=arguments.get("context")
                        )
                        return {"success": success, "metric": metric_name, "value": value}

                    elif action == "record_improvement":
                        name = arguments.get("improvement_name")
                        desc = arguments.get("description", "")
                        metric = arguments.get("metric_name", "context_relevance")
                        baseline = arguments.get("baseline_value", 0)
                        if not name:
                            return {"error": "improvement_name required"}
                        return imp_tracker.record_improvement(
                            name=name,
                            description=desc,
                            metric=metric,
                            baseline_value=baseline
                        )

                    else:  # report (default)
                        days = arguments.get("days", 30)

                        # Get metrics summary
                        metrics = perf_tracker.get_all_metrics_summary(days=days)
                        derived = perf_tracker.calculate_derived_metrics()

                        # Get improvement report
                        improvements = imp_tracker.generate_improvement_report()

                        # Get comprehensive self-evaluation from self_evaluation.py
                        try:
                            from self_evaluation import SelfEvaluator
                            evaluator = SelfEvaluator()
                            full_eval = evaluator.generate_full_report(days=days)
                            self_eval = {
                                "overall_score": full_eval.get("summary", {}).get("overall_score"),
                                "grade": full_eval.get("summary", {}).get("grade"),
                                "correction_rate": full_eval.get("correction_rate", {}).get("correction_rate"),
                                "solution_success_rate": full_eval.get("solution_success", {}).get("success_rate"),
                                "fact_health_score": full_eval.get("fact_health", {}).get("health_score"),
                                "areas_for_improvement": full_eval.get("summary", {}).get("areas_for_improvement", []),
                                "strengths": full_eval.get("summary", {}).get("strengths", []),
                                "pattern_promotion_candidates": full_eval.get("pattern_promotion", {}).get("candidates_found", 0),
                            }
                        except Exception as eval_err:
                            self_eval = {"error": str(eval_err)}

                        # Get feedback statistics
                        try:
                            from feedback_detector import FeedbackDetector
                            detector = FeedbackDetector()
                            feedback_stats = detector.get_feedback_stats(days=days)
                        except Exception:
                            feedback_stats = {}

                        # Combine into comprehensive report
                        return {
                            "period": f"Last {days} days",
                            "generated_at": dt.datetime.now().isoformat(),
                            "self_evaluation": self_eval,
                            "feedback": {
                                "total_success": feedback_stats.get("total_success", 0),
                                "total_failure": feedback_stats.get("total_failure", 0),
                                "success_rate": feedback_stats.get("success_rate"),
                            },
                            "metrics": {
                                "tracked": metrics.get("metrics_with_data", 0),
                                "total_available": len(TRACKED_METRICS),
                                "summaries": {
                                    k: {
                                        "has_data": v.get("has_data"),
                                        "trend": v.get("trend", {}).get("direction") if v.get("has_data") else None,
                                        "latest": v.get("statistics", {}).get("latest") if v.get("has_data") else None,
                                    }
                                    for k, v in metrics.get("summaries", {}).items()
                                },
                                "derived": derived,
                            },
                            "improvements": improvements.get("summary", {}),
                            "recommendations": improvements.get("recommendations", []),
                            "available_metrics": list(TRACKED_METRICS.keys()),
                        }

                result = await run_in_thread(do_self_report, timeout=60)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "personality":
            # Phase 6: Personality evolution tracking
            action = arguments.get("action", "traits")
            try:
                def do_personality():
                    from personality.consistency_checker import ConsistencyChecker
                    from personality.evolution_engine import PersonalityEvolutionEngine
                    from personality.trait_tracker import TraitTracker

                    trait_tracker = TraitTracker()
                    evolution_engine = PersonalityEvolutionEngine()
                    consistency_checker = ConsistencyChecker()

                    if action == "traits":
                        # Get all traits, optionally filtered by category
                        all_traits = trait_tracker.get_all_traits()
                        category = arguments.get("category")
                        if category:
                            by_cat = all_traits.get("by_category", {})
                            if category in by_cat:
                                return {
                                    "category": category,
                                    "prefers": by_cat[category].get("prefers", []),
                                    "dislikes": by_cat[category].get("dislikes", []),
                                    "count": len(by_cat[category].get("prefers", [])) + len(by_cat[category].get("dislikes", []))
                                }
                            return {"error": f"Category '{category}' not found"}
                        return all_traits

                    elif action == "evolution":
                        days = arguments.get("days", 30)
                        return evolution_engine.get_evolution_summary(days)

                    elif action == "consistency":
                        result = consistency_checker.check_all_traits()
                        suggestions = consistency_checker.get_resolution_suggestions()
                        result["resolution_suggestions"] = suggestions[:5]
                        return result

                    elif action == "evolve":
                        feedback_type = arguments.get("feedback_type", "correction")
                        content = arguments.get("content", "")
                        accepted = arguments.get("accepted", True)

                        if feedback_type == "correction":
                            correction = {
                                "content": content,
                                "topic": "general",
                                "importance": "medium",
                                "id": f"manual_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"
                            }
                            changes = evolution_engine.evolve_from_correction(correction)
                            return {
                                "action": "evolved_from_correction",
                                "changes": changes,
                                "traits_affected": len(changes)
                            }
                        else:
                            result = evolution_engine.evolve_from_feedback(
                                feedback_type=feedback_type,
                                context=content,
                                accepted=accepted
                            )
                            return result

                    elif action == "sync":
                        return evolution_engine.synchronize_with_profile()

                    else:
                        return {"error": f"Unknown action: {action}"}

                result = await run_in_thread(do_personality, timeout=60)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "conversation":
            # Consolidated: replaces tag_conversation, add_note_to_conversation, set_conversation_relevance
            action = arguments.get("action", "tag")
            conv_id = arguments.get("conversation_id")
            try:
                def do_conversation():
                    if action == "note":
                        success = memory.add_user_note(conv_id, arguments.get("note"))
                        return {"success": success, "conversation_id": conv_id}
                    elif action == "relevance":
                        success = memory.set_conversation_relevance(conv_id, arguments.get("relevance"))
                        return {"success": success, "conversation_id": conv_id, "relevance": arguments.get("relevance")}
                    else:  # tag
                        tags = arguments.get("tags", [])
                        success = memory.add_user_tags(conv_id, tags)
                        return {"success": success, "conversation_id": conv_id, "tags_added": tags}

                result = await run_in_thread(do_conversation)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "images":
            # Consolidated: replaces save_screenshot, search_images
            action = arguments.get("action", "search")
            try:
                from image_processor import ImageProcessor

                def do_images():
                    # Fast NAS check - fail fast if NAS unavailable
                    if not is_nas_reachable(timeout=2.0):
                        return {"error": "NAS not reachable", "nas_status": "unavailable", "suggestion": "Check if NAS is powered on and Z: drive is mounted"}
                    processor = ImageProcessor()
                    if action == "save":
                        return processor.save_screenshot(
                            arguments.get("image_data"),
                            arguments.get("conversation_id"),
                            arguments.get("description", "")
                        )
                    else:  # search
                        results = processor.search_images(
                            arguments.get("query", ""),
                            arguments.get("limit", 10)
                        )
                        return {"images": results, "count": len(results)}

                result = await run_in_thread(do_images)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "branch":
            # Consolidated: replaces create_branch, mark_branch_status
            action = arguments.get("action", "create")
            try:
                from branch_tracker import BranchTracker

                def do_branch():
                    tracker = BranchTracker()
                    if action == "mark":
                        branch_id = arguments.get("branch_id")
                        status = arguments.get("status")
                        reason = arguments.get("reason")
                        if status == "chosen":
                            tracker.mark_branch_chosen(branch_id, reason)
                        else:
                            tracker.mark_branch_abandoned(branch_id, reason)
                        return {"branch_id": branch_id, "status": status, "reason": reason}
                    else:  # create
                        name = arguments.get("name")
                        description = arguments.get("description", "")
                        parent = arguments.get("parent_conversation_id")
                        branch_id = tracker.create_branch(name, description, parent)
                        return {"branch_id": branch_id, "name": name}

                result = await run_in_thread(do_branch)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "session":
            # Consolidated: replaces get_conversation_thread, get_active_sessions, get_session_summary, detect_session_continuation
            action = arguments.get("action", "active")
            try:
                def do_session():
                    # Fast NAS check - fail fast if NAS unavailable
                    if not is_nas_reachable(timeout=2.0):
                        return {"error": "NAS not reachable", "nas_status": "unavailable", "suggestion": "Check if NAS is powered on and Z: drive is mounted"}
                    if action == "thread":
                        session_id = arguments.get("session_id")
                        if not session_id:
                            return {"error": "session_id is required"}
                        thread = memory.get_conversation_thread(session_id)
                        thread_summary = []
                        for conv in thread:
                            thread_summary.append({
                                "conversation_id": conv["id"],
                                "timestamp": conv["timestamp"],
                                "message_count": len(conv.get("messages", [])),
                                "topics": conv.get("metadata", {}).get("topics", [])
                            })
                        return {"session_id": session_id, "thread_length": len(thread_summary), "conversations": thread_summary}
                    else:
                        from session_continuity import SessionContinuityManager
                        manager = SessionContinuityManager()
                        if action == "summary":
                            session_id = arguments.get("session_id")
                            include_last_n = arguments.get("include_last_n", 1)
                            return manager.get_session_summary(session_id, include_last_n)
                        elif action == "detect":
                            user_prompt = arguments.get("user_prompt")
                            recent_sessions = manager.get_active_sessions(days_back=7)
                            detected = manager.detect_continuation(user_prompt, recent_sessions)
                            return {"continuation_detected": detected is not None, "suggested_session_id": detected, "recent_sessions": recent_sessions[:3]}
                        else:  # active
                            days_back = arguments.get("days_back", 7)
                            sessions = manager.get_active_sessions(days_back=days_back)
                            return {"active_sessions": sessions, "count": len(sessions)}

                result = await run_in_thread(do_session)
                return [TextContent(type="text", text=safe_json_dumps(result) if isinstance(result, dict) else result)]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "device":
            # Consolidated: replaces get_current_device, get_all_devices, register_device
            action = arguments.get("action", "current")
            try:
                def do_device():
                    # Import inside function to avoid closure issues with thread executor
                    from device_registry import get_device_registry as get_registry
                    # Fast NAS check - fail fast if NAS unavailable
                    if not is_nas_reachable(timeout=2.0):
                        return {"error": "NAS not reachable", "nas_status": "unavailable", "suggestion": "Check if NAS is powered on and Z: drive is mounted"}
                    registry = get_registry()
                    if action == "all":
                        devices = registry.get_all_devices()
                        return {"devices": devices, "count": len(devices), "current_device": registry.get_device_tag()}
                    elif action == "register":
                        device = registry.register_device(
                            friendly_name=arguments.get("friendly_name"),
                            description=arguments.get("description")
                        )
                        return {"success": True, "device_tag": device.get("device_type", "unknown"), "device_name": device.get("friendly_name", "Unknown")}
                    else:  # current
                        device = registry.get_current_device()
                        return {
                            "device_tag": device.get("device_type", "unknown"),
                            "device_name": device.get("friendly_name", device.get("device_name", "Unknown")),
                            "hostname": device.get("hostname", ""),
                            "os": device.get("os", ""),
                            "is_current": True
                        }

                result = await run_in_thread(do_device)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "record_learning":
            # LOCAL-FIRST: Write to local disk DIRECTLY - NO thread pool
            # The thread pool has stuck NAS threads - bypass it completely
            learning_type = arguments.get("type", "solution")
            try:
                from local_brain import get_local_brain
                brain = get_local_brain()

                # Direct synchronous calls - local disk is instant
                if learning_type == "failure":
                    result = brain.save_failure(
                        solution_id=arguments.get("solution_id", ""),
                        failure_description=arguments.get("why_it_failed", ""),
                        error_message=arguments.get("context", ""),
                        conversation_id=arguments.get("conversation_id")
                    )
                elif learning_type == "antipattern":
                    result = brain.save_antipattern(
                        what_not_to_do=arguments.get("what_not_to_do", ""),
                        why_it_failed=arguments.get("why_it_failed", ""),
                        error_details=arguments.get("context", ""),
                        original_problem=arguments.get("problem", ""),
                        conversation_id=arguments.get("conversation_id"),
                        tags=arguments.get("tags", [])
                    )
                elif learning_type == "confirm":
                    result = {"success": True, "message": "Confirmation noted (will sync when NAS available)"}
                else:  # solution/learning
                    result = brain.save_solution(
                        problem=arguments.get("problem", ""),
                        solution=arguments.get("solution", ""),
                        context=arguments.get("context", ""),
                        tags=arguments.get("tags", []),
                        conversation_id=arguments.get("conversation_id")
                    )

                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "find_learning":
            # Consolidated: replaces find_solution, find_antipatterns, get_solution_chain, get_learnings_summary
            learning_type = arguments.get("type", "solution")
            try:
                from solution_tracker import SolutionTracker

                def do_find():
                    # Fast NAS check with socket timeout
                    if not is_nas_reachable(timeout=2.0):
                        return {"error": "NAS not reachable", "solutions": [], "count": 0}

                    tracker = SolutionTracker()
                    if learning_type == "antipattern":
                        results = tracker.find_antipatterns(
                            problem=arguments.get("problem"),
                            tags=arguments.get("tags")
                        )
                        return {"antipatterns": results[:10], "count": len(results), "note": "These approaches FAILED - avoid them!"}
                    elif learning_type == "chain":
                        return tracker.get_solution_chain(arguments.get("solution_id", ""))
                    elif learning_type == "summary":
                        return tracker.get_learnings_summary()
                    else:  # solution
                        results = tracker.find_solution(arguments.get("problem", ""))
                        return {"solutions": results[:10], "count": len(results)}

                # Short timeout
                result = await run_in_thread(do_find, timeout=15)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "search_knowledge_base":
            query = arguments.get("query", "").lower()
            fact_type = arguments.get("fact_type")
            limit = arguments.get("limit", 20)
            include_superseded = arguments.get("include_superseded", False)

            def do_search():
                # Fast NAS check - fail fast if NAS unavailable
                if not is_nas_reachable(timeout=2.0):
                    return {"error": "NAS not reachable", "results": [], "count": 0, "nas_status": "unavailable"}
                results = []

                # Search facts.jsonl
                kb_file = memory.facts_path / "facts.jsonl"
                if kb_file.exists():
                    with open(kb_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                fact = json.loads(line)
                                # TRUTH MAINTENANCE: Skip superseded facts unless explicitly requested
                                if not include_superseded:
                                    if fact.get("superseded_by") or fact.get("status") == "superseded":
                                        continue
                                content = fact.get("content") or fact.get("learning", "")
                                if fact_type and fact.get("type") != fact_type:
                                    continue
                                if query in content.lower():
                                    results.append(fact)
                                if len(results) >= limit:
                                    break
                            except json.JSONDecodeError:
                                continue

                # Also search individual fact JSON files if not enough results
                if len(results) < limit:
                    for fact_file in memory.facts_path.glob("*.json"):
                        if len(results) >= limit:
                            break
                        try:
                            with open(fact_file, "r", encoding="utf-8") as f:
                                fact = json.load(f)
                            # TRUTH MAINTENANCE: Skip superseded facts
                            if not include_superseded:
                                if fact.get("superseded_by") or fact.get("status") == "superseded":
                                    continue
                            content = fact.get("content") or fact.get("learning", "")
                            if fact_type and fact.get("type") != fact_type:
                                continue
                            if query in content.lower():
                                # Avoid duplicates
                                if not any(r.get("id") == fact.get("id") for r in results):
                                    results.append(fact)
                        except (json.JSONDecodeError, Exception):
                            continue

                return results

            results = await run_in_thread(do_search)
            # Sanitize fact content
            for r in results:
                if "content" in r:
                    r["content"] = sanitize_text(r["content"], MAX_CONTENT_LENGTH)
                if "learning" in r:
                    r["learning"] = sanitize_text(r["learning"], MAX_CONTENT_LENGTH)
            results = results[:MAX_RESULTS]
            return [TextContent(type="text", text=safe_json_dumps({"results": results, "count": len(results), "query": query}))]

        elif name == "get_entity_info":
            entity_type = arguments.get("entity_type")
            entity_name = arguments.get("entity_name")

            def do_lookup():
                # Fast NAS check - fail fast if NAS unavailable
                if not is_nas_reachable(timeout=2.0):
                    return {"error": "NAS not reachable", "found": False, "nas_status": "unavailable"}
                entity_file = memory.entities_path / f"{entity_type}.json"
                if not entity_file.exists():
                    return {"found": False, "message": f"No {entity_type} database found"}

                with open(entity_file, "r", encoding="utf-8") as f:
                    entities = json.load(f)

                entity_info = entities.get(entity_name)
                if not entity_info:
                    return {"found": False, "message": f"{entity_name} not found in {entity_type}"}

                return {"found": True, "entity_type": entity_type, "entity_name": entity_name, "info": entity_info}

            result = await run_in_thread(do_lookup)
            return [TextContent(type="text", text=safe_json_dumps(result))]

        elif name == "get_timeline":
            year_month = arguments.get("year_month")
            event_type = arguments.get("event_type", "all")

            def do_timeline():
                # Fast NAS check - fail fast if NAS unavailable
                if not is_nas_reachable(timeout=2.0):
                    return {"error": "NAS not reachable", "events": [], "nas_status": "unavailable"}
                timeline_file = memory.timeline_path / f"{year_month}.jsonl"
                if not timeline_file.exists():
                    return {"events": [], "message": f"No timeline for {year_month}"}

                events = []
                with open(timeline_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                            if event_type == "all" or event.get("type") == event_type:
                                events.append(event)
                        except json.JSONDecodeError:
                            continue
                return {"period": year_month, "event_count": len(events), "events": events}

            result = await run_in_thread(do_timeline)
            # Sanitize event descriptions
            for event in result.get("events", []):
                if "description" in event:
                    event["description"] = sanitize_text(event["description"], MAX_CONTENT_LENGTH)
            return [TextContent(type="text", text=safe_json_dumps(result))]

        elif name == "find_file_paths":
            path_pattern = arguments.get("path_pattern", "").lower()
            purpose_filter = arguments.get("purpose")

            def do_find():
                # Fast NAS check - fail fast if NAS unavailable
                if not is_nas_reachable(timeout=2.0):
                    return {"error": "NAS not reachable", "paths": [], "nas_status": "unavailable"}
                all_paths = []
                try:
                    for conv_file in memory.conversations_path.glob("*.json"):
                        try:
                            with open(conv_file, "r", encoding="utf-8") as f:
                                conv = json.load(f)
                                for path_info in conv.get("extracted_data", {}).get("file_paths", []):
                                    if path_pattern in path_info.get("path", "").lower():
                                        if not purpose_filter or path_info.get("purpose") == purpose_filter:
                                            all_paths.append({**path_info, "conversation_id": conv.get("id", "unknown")})
                        except Exception:
                            continue
                except Exception:
                    pass
                return all_paths

            all_paths = await run_in_thread(do_find)
            # Sanitize context in file paths
            for p in all_paths:
                if "context" in p:
                    p["context"] = sanitize_text(p["context"], MAX_CONTEXT_LENGTH)
            all_paths = all_paths[:MAX_RESULTS]
            return [TextContent(type="text", text=safe_json_dumps({"pattern": path_pattern, "found_count": len(all_paths), "paths": all_paths}))]

        elif name == "get_user_context":
            limit = arguments.get("limit", 10)

            def do_context():
                # Fast NAS check - fail fast if NAS unavailable
                if not is_nas_reachable(timeout=2.0):
                    return {"error": "NAS not reachable", "goals": [], "preferences": [], "technical_environment": {}, "nas_status": "unavailable"}
                all_goals = []
                all_preferences = []
                all_tech_details = {"tools": set(), "ips": set(), "capacities": {}}

                try:
                    conv_files = sorted(memory.conversations_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

                    for conv_file in conv_files:
                        try:
                            with open(conv_file, "r", encoding="utf-8") as f:
                                conv = json.load(f)
                                extracted = conv.get("extracted_data", {})
                                all_goals.extend(extracted.get("goals_and_intentions", []))
                                all_preferences.extend(extracted.get("user_preferences", []))
                                tech = extracted.get("technical_details", {})
                                if tech.get("tools"):
                                    all_tech_details["tools"].update(tech["tools"])
                                if tech.get("ips"):
                                    all_tech_details["ips"].update(tech["ips"])
                                all_tech_details["capacities"].update(tech.get("capacities", {}))
                        except Exception:
                            continue

                    # Load user profile for comprehensive user context
                    user_summary = None
                    try:
                        prof_path = Path(memory.base_path) / "user" / "profile.json"
                        if prof_path.exists():
                            with open(prof_path, "r", encoding="utf-8") as f:
                                profile = json.load(f)
                                user_summary = {
                                    "identity": profile.get("identity", {}),
                                    "relationships_count": {
                                        "pets": len(profile.get("relationships", {}).get("pets", [])),
                                        "family": len(profile.get("relationships", {}).get("family", [])),
                                        "colleagues": len(profile.get("relationships", {}).get("colleagues", [])),
                                        "friends": len(profile.get("relationships", {}).get("friends", []))
                                    },
                                    "projects_count": {
                                        "companies_owned": len(profile.get("projects", {}).get("companies_owned", [])),
                                        "clients": len(profile.get("projects", {}).get("clients", [])),
                                        "active_projects": len(profile.get("projects", {}).get("active_projects", []))
                                    },
                                    "profile_completeness": profile.get("metadata", {}).get("profile_completeness", 0)
                                }
                    except Exception:
                        pass  # User profile optional

                    result = {
                        "goals": all_goals,
                        "preferences": all_preferences,
                        "technical_environment": {
                            "tools": sorted(list(all_tech_details["tools"])),
                            "ips": sorted(list(all_tech_details["ips"])),
                            "capacities": all_tech_details["capacities"]
                        },
                        "conversations_analyzed": len(conv_files)
                    }

                    # Include user summary if available
                    if user_summary:
                        result["user_profile_summary"] = user_summary

                    return result

                except Exception as e:
                    return {"error": str(e), "goals": [], "preferences": [], "technical_environment": {}}

            result = await run_in_thread(do_context)
            # Sanitize goals and preferences
            if "goals" in result:
                for g in result["goals"]:
                    if isinstance(g, dict) and "goal" in g:
                        g["goal"] = sanitize_text(g["goal"], MAX_CONTENT_LENGTH)
                result["goals"] = result["goals"][:MAX_RESULTS]
            if "preferences" in result:
                for p in result["preferences"]:
                    if isinstance(p, dict) and "preference" in p:
                        p["preference"] = sanitize_text(p["preference"], MAX_CONTENT_LENGTH)
                result["preferences"] = result["preferences"][:MAX_RESULTS]
            return [TextContent(type="text", text=safe_json_dumps(result))]

        elif name == "get_user_profile":
            category = arguments.get("category", "all")
            include_metadata = arguments.get("include_metadata", True)

            def do_get_profile():
                # Fast NAS check - fail fast if NAS unavailable
                if not is_nas_reachable(timeout=2.0):
                    return {"error": "NAS not reachable", "profile_exists": False, "nas_status": "unavailable", "suggestion": "Check if NAS is powered on and Z: drive is mounted"}
                try:
                    # Load user profile from NAS
                    prof_path = Path(memory.base_path) / "user" / "profile.json"

                    if not prof_path.exists():
                        return {
                            "error": "User profile not found. Run build_user_profile.py first.",
                            "profile_exists": False
                        }

                    with open(prof_path, "r", encoding="utf-8") as f:
                        profile = json.load(f)

                    # Filter by category if specified
                    if category != "all":
                        if category in profile:
                            result = {
                                category: profile[category],
                                "profile_exists": True
                            }
                            if include_metadata:
                                result["metadata"] = profile.get("metadata", {})
                            return result
                        else:
                            return {
                                "error": f"Category '{category}' not found in profile",
                                "available_categories": list(profile.keys())
                            }

                    # Return full profile
                    result = {
                        "identity": profile.get("identity", {}),
                        "relationships": profile.get("relationships", {}),
                        "projects": profile.get("projects", {}),
                        "preferences": profile.get("preferences", {}),
                        "goals": profile.get("goals", []),
                        "technical_environment": profile.get("technical_environment", {}),
                        "profile_exists": True
                    }

                    if include_metadata:
                        result["metadata"] = profile.get("metadata", {})

                    return result

                except Exception as e:
                    return {
                        "error": f"Failed to load user profile: {str(e)}",
                        "profile_exists": False
                    }

            result = await run_in_thread(do_get_profile)
            return [TextContent(type="text", text=safe_json_dumps(result))]

        elif name == "semantic_search":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            chunk_type = arguments.get("chunk_type")
            boost_by_recency = arguments.get("boost_by_recency", True)
            recency_decay_days = arguments.get("recency_decay_days", 30)

            def do_search():
                filters = {}
                if chunk_type:
                    filters["chunk_type"] = chunk_type
                return embeddings.semantic_search(
                    query, top_k=top_k, filters=filters,
                    boost_by_recency=boost_by_recency,
                    recency_decay_days=recency_decay_days
                )

            results = await run_in_thread(do_search)
            # Sanitize results to prevent terminal rendering issues
            results = sanitize_results(results)

            # AGENT 13: Include confidence note in output
            return [TextContent(type="text", text=safe_json_dumps({
                "query": query,
                "results_count": len(results),
                "results": results,
                "recency_boost_enabled": boost_by_recency,
                "note": "Results include confidence scores (HIGH/MEDIUM/LOW), similarity_score (0-1), and recency_multiplier"
            }))]

        elif name == "hybrid_search":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            alpha = arguments.get("alpha", 0.7)
            boost_by_recency = arguments.get("boost_by_recency", True)
            recency_decay_days = arguments.get("recency_decay_days", 30)

            def do_search():
                return embeddings.hybrid_search(
                    query, top_k=top_k, alpha=alpha,
                    boost_by_recency=boost_by_recency,
                    recency_decay_days=recency_decay_days
                )

            results = await run_in_thread(do_search)
            # Sanitize results to prevent terminal rendering issues
            results = sanitize_results(results)

            # AGENT 13: Include confidence note in output
            return [TextContent(type="text", text=safe_json_dumps({
                "query": query,
                "search_type": "hybrid",
                "semantic_weight": alpha,
                "keyword_weight": 1 - alpha,
                "results_count": len(results),
                "results": results,
                "recency_boost_enabled": boost_by_recency,
                "note": "Results include confidence scores (HIGH/MEDIUM/LOW), similarity_score (0-1), and recency_multiplier"
            }))]

        elif name == "get_rag_context":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            use_hybrid = arguments.get("use_hybrid", True)

            def do_rag():
                # Get raw results instead of formatted string
                if use_hybrid:
                    results = embeddings.hybrid_search(query, top_k=top_k)
                else:
                    results = embeddings.semantic_search(query, top_k=top_k)
                return results

            results = await run_in_thread(do_rag)
            # Sanitize results to prevent terminal rendering issues
            results = sanitize_results(results)

            # AGENT 13: Return as clean JSON with confidence scores
            return [TextContent(type="text", text=safe_json_dumps({
                "query": query,
                "search_type": "hybrid" if use_hybrid else "semantic",
                "sources_count": len(results),
                "results": results,
                "note": "Results include confidence scores (HIGH/MEDIUM/LOW) and similarity_score (0-1)"
            }))]

        elif name == "rebuild_vector_index":
            def do_rebuild():
                import time as _t
                _start = _t.monotonic()

                # Count vector files to estimate timeout dynamically
                vector_files = list(embeddings.vectors_path.glob("*.npy"))
                vector_count = sum(1 for f in vector_files if "_metadata" not in f.name)
                print(f"[FAISS Rebuild] Starting full rebuild: {vector_count} vector files to process")

                index = embeddings.build_faiss_index(rebuild=True)
                elapsed = _t.monotonic() - _start
                total = index.ntotal if index else 0
                print(f"[FAISS Rebuild] Done: {total} vectors indexed in {elapsed:.1f}s")
                return {"total": total, "elapsed": elapsed, "files_processed": vector_count}

            # Dynamic timeout: 30s base + 0.5s per vector file + 120s buffer for
            # NAS I/O and FAISS training. Minimum 120s, scales with data growth.
            try:
                file_count = sum(1 for f in embeddings.vectors_path.glob("*.npy") if "_metadata" not in f.name)
            except Exception:
                file_count = 2000  # Safe fallback estimate
            dynamic_timeout = max(120, 30 + int(file_count * 0.5) + 120)
            print(f"[FAISS Rebuild] Timeout set to {dynamic_timeout}s for {file_count} vector files")

            result = await run_in_thread(do_rebuild, timeout=dynamic_timeout)
            return [TextContent(type="text", text=safe_json_dumps({
                "success": True,
                "message": f"Vector index rebuilt successfully in {result['elapsed']:.1f}s",
                "total_vectors": result["total"],
                "files_processed": result["files_processed"],
                "elapsed_seconds": round(result["elapsed"], 1),
                "timeout_used": dynamic_timeout
            }))]

        elif name == "get_corrections":
            # Import corrections tracker
            try:
                from corrections_tracker import CorrectionsTracker

                def do_get_corrections():
                    # Fast NAS check with socket timeout
                    if not is_nas_reachable(timeout=2.0):
                        return []  # Return empty list, don't block

                    tracker = CorrectionsTracker()
                    topic = arguments.get("topic")
                    query = arguments.get("query")
                    importance = arguments.get("importance")
                    limit = arguments.get("limit", 10)

                    results = []

                    if importance == "high":
                        results = tracker.get_high_importance_corrections(limit=limit)
                    elif topic:
                        results = tracker.get_corrections_by_topic(topic, limit=limit)
                    elif query:
                        results = tracker.search_corrections(query, limit=limit)
                    else:
                        results = tracker.get_recent_corrections(days=30, limit=limit)

                    # Sanitize results
                    return sanitize_results(results)

                results = await run_in_thread(do_get_corrections, timeout=15)

                response = {
                    "corrections": results,
                    "count": len(results),
                    "source": "corrections_tracker"
                }

                return [TextContent(type="text", text=safe_json_dumps(response))]

            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({
                    "error": f"Corrections retrieval failed: {str(e)}",
                    "corrections": []
                }))]

        elif name == "get_project_state":
            # Import project tracker
            try:
                from project_tracker import ProjectTracker

                def do_get_project():
                    tracker = ProjectTracker()
                    project_id = arguments.get("project_id")
                    file_path = arguments.get("file_path")

                    if file_path:
                        project = tracker.search_project_by_path(file_path)
                    elif project_id:
                        project = tracker.get_project(project_id)
                    else:
                        return {"error": "Must provide either project_id or file_path"}

                    if not project:
                        return {"error": "Project not found", "project": None}

                    # Sanitize project data
                    project_clean = {
                        "project_id": project["project_id"],
                        "name": project["name"],
                        "status": project["status"],
                        "last_worked": project["last_worked"],
                        "current_focus": sanitize_text(project.get("current_focus", ""), 200),
                        "priority": project["priority"],
                        "blockers": [sanitize_text(b, 150) for b in project.get("blockers", [])[:5]],
                        "next_steps": [sanitize_text(s, 150) for s in project.get("next_steps", [])[:5]],
                        "files": project.get("files", [])[:10],
                        "technologies": project.get("technologies", [])[:10],
                        "milestones_completed": len(project.get("milestones", {}).get("completed", [])),
                        "milestones_in_progress": project.get("milestones", {}).get("in_progress", [])[:3]
                    }

                    return project_clean

                result = await run_in_thread(do_get_project)
                return [TextContent(type="text", text=safe_json_dumps(result))]

            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({
                    "error": f"Project state retrieval failed: {str(e)}"
                }))]

        elif name == "get_active_projects":
            # Import project tracker
            try:
                from project_tracker import ProjectTracker

                def do_get_active():
                    tracker = ProjectTracker()
                    status = arguments.get("status", "active")

                    if status == "all":
                        projects = list(tracker.projects.values())
                    else:
                        projects = tracker.get_active_projects(status=status)

                    # Sanitize and summarize
                    projects_summary = []
                    for p in projects[:20]:  # Limit to 20
                        projects_summary.append({
                            "project_id": p["project_id"],
                            "name": p["name"],
                            "status": p["status"],
                            "last_worked": p["last_worked"],
                            "current_focus": sanitize_text(p.get("current_focus", ""), 100),
                            "priority": p["priority"],
                            "blocker_count": len(p.get("blockers", [])),
                            "next_step_count": len(p.get("next_steps", []))
                        })

                    # Get stats
                    stats = tracker.get_stats()

                    return {
                        "projects": projects_summary,
                        "count": len(projects_summary),
                        "stats": stats
                    }

                result = await run_in_thread(do_get_active)
                return [TextContent(type="text", text=safe_json_dumps(result))]

            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({
                    "error": f"Active projects retrieval failed: {str(e)}",
                    "projects": []
                }))]

        elif name == "get_conversation_thread":
            # Import memory service
            try:
                def do_get_thread():
                    session_id = arguments.get("session_id")
                    if not session_id:
                        return {"error": "session_id is required"}

                    # Get thread
                    thread = memory.get_conversation_thread(session_id)

                    # Sanitize thread
                    thread_summary = []
                    for conv in thread:
                        thread_summary.append({
                            "conversation_id": conv["id"],
                            "timestamp": conv["timestamp"],
                            "message_count": len(conv.get("messages", [])),
                            "topics": conv.get("metadata", {}).get("topics", []),
                            "thread_position": conv.get("metadata", {}).get("conversation_thread", {}).get("thread_position", 0),
                            "summary": sanitize_text(
                                conv.get("search_index", {}).get("summary", ""),
                                200
                            )
                        })

                    return {
                        "session_id": session_id,
                        "thread_length": len(thread_summary),
                        "conversations": thread_summary
                    }

                result = await run_in_thread(do_get_thread)
                return [TextContent(type="text", text=safe_json_dumps(result))]

            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({
                    "error": f"Thread retrieval failed: {str(e)}",
                    "conversations": []
                }))]

        elif name == "trigger_search_visualization":
            # Trigger visualization via WebSocket
            try:
                import requests

                def do_trigger():
                    search_results = arguments.get("search_results", [])

                    # Send to visualization server
                    response = requests.post(
                        "http://localhost:8080/api/notify",
                        json={
                            "type": "search_performed",
                            "results": search_results[:10]  # Top 10 only
                        },
                        timeout=2
                    )

                    if response.status_code == 200:
                        return {
                            "success": True,
                            "message": "Search visualization triggered",
                            "results_count": len(search_results)
                        }
                    else:
                        return {
                            "success": False,
                            "error": "Visualization server not responding"
                        }

                result = await run_in_thread(do_trigger)
                return [TextContent(type="text", text=safe_json_dumps(result))]

            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({
                    "success": False,
                    "error": f"Visualization trigger failed: {str(e)}"
                }))]

        elif name == "get_stale_projects":
            try:
                from pattern_detector import PatternDetector

                def do_get_stale():
                    detector = PatternDetector()
                    stale_days = arguments.get("stale_days", 14)
                    stale = detector.detect_stale_projects(stale_days=stale_days)

                    return {
                        "stale_projects": stale,
                        "count": len(stale),
                        "threshold_days": stale_days
                    }

                result = await run_in_thread(do_get_stale)
                return [TextContent(type="text", text=safe_json_dumps(result))]

            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({
                    "error": f"Stale project detection failed: {str(e)}",
                    "stale_projects": []
                }))]

        # PROJECT EVOLUTION HANDLERS
        elif name == "record_project_evolution":
            try:
                from project_evolution import ProjectEvolutionTracker

                def do_record():
                    tracker = ProjectEvolutionTracker()
                    evolution = tracker.record_evolution(
                        project_id=arguments.get("project_id"),
                        conversation_id=arguments.get("conversation_id", "manual"),
                        summary=arguments.get("summary"),
                        version=arguments.get("version"),
                        supersedes_version=arguments.get("supersedes_version"),
                        keywords=arguments.get("keywords", [])
                    )
                    return {
                        "success": True,
                        "project_id": arguments.get("project_id"),
                        "evolution": evolution,
                        "message": f"Recorded evolution: {evolution.get('version')} - {arguments.get('summary')[:50]}..."
                    }

                result = await run_in_thread(do_record)
                return [TextContent(type="text", text=safe_json_dumps(result))]

            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({
                    "success": False,
                    "error": f"Failed to record evolution: {str(e)}"
                }))]

        elif name == "get_project_evolution":
            try:
                from project_evolution import ProjectEvolutionTracker

                def do_get_timeline():
                    tracker = ProjectEvolutionTracker()
                    project_id = arguments.get("project_id")
                    timeline = tracker.get_project_timeline(project_id)
                    current = tracker.get_current_version(project_id)

                    return {
                        "project_id": project_id,
                        "current_version": current.get("version") if current else None,
                        "version_count": len(timeline),
                        "timeline": timeline,
                        "has_superseded": any(v.get("is_superseded") for v in timeline)
                    }

                result = await run_in_thread(do_get_timeline)
                return [TextContent(type="text", text=safe_json_dumps(result))]

            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({
                    "error": f"Failed to get project evolution: {str(e)}",
                    "timeline": []
                }))]

        elif name == "mark_content_superseded":
            try:
                from project_evolution import ProjectEvolutionTracker

                def do_mark():
                    tracker = ProjectEvolutionTracker()
                    success = tracker.mark_superseded(
                        project_id=arguments.get("project_id"),
                        old_version=arguments.get("old_version"),
                        new_version=arguments.get("new_version"),
                        reason=arguments.get("reason"),
                        conversation_ids=arguments.get("conversation_ids", [])
                    )
                    return {
                        "success": success,
                        "message": f"Marked {arguments.get('old_version')} as superseded by {arguments.get('new_version')}"
                    }

                result = await run_in_thread(do_mark)
                return [TextContent(type="text", text=safe_json_dumps(result))]

            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({
                    "success": False,
                    "error": f"Failed to mark content as superseded: {str(e)}"
                }))]

        elif name == "list_tracked_evolutions":
            try:
                from project_evolution import ProjectEvolutionTracker

                def do_list():
                    tracker = ProjectEvolutionTracker()
                    projects = tracker.get_all_projects()
                    return {
                        "tracked_projects": projects,
                        "count": len(projects)
                    }

                result = await run_in_thread(do_list)
                return [TextContent(type="text", text=safe_json_dumps(result))]

            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({
                    "error": f"Failed to list evolutions: {str(e)}",
                    "tracked_projects": []
                }))]

        elif name == "get_user_preferences":
            try:
                from preference_manager import PreferenceManager

                def do_get_prefs():
                    # Fast NAS check with socket timeout
                    if not is_nas_reachable(timeout=2.0):
                        return {"error": "NAS not reachable (socket timeout)"}

                    manager = PreferenceManager()
                    return manager.get_preferences_summary()

                result = await run_in_thread(do_get_prefs, timeout=15)
                return [TextContent(type="text", text=safe_json_dumps(result))]

            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({
                    "error": f"Preference retrieval failed: {str(e)}"
                }))]

        elif name == "update_user_preference":
            try:
                from preference_manager import PreferenceManager

                def do_update_pref():
                    # Fast NAS check with socket timeout - BEFORE touching filesystem
                    if not is_nas_reachable(timeout=2.0):
                        return {"error": "NAS not reachable (socket timeout)", "success": False}

                    manager = PreferenceManager()
                    category = arguments.get("category")
                    preference = arguments.get("preference")
                    positive = arguments.get("positive", True)

                    # Update preference using the manager's method
                    result = manager.update_preference_manual(category, preference, positive)
                    return result

                result = await run_in_thread(do_update_pref, timeout=15)
                return [TextContent(type="text", text=safe_json_dumps(result))]

            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({
                    "error": f"Preference update failed: {str(e)}"
                }))]

        elif name == "find_code":
            try:
                from code_indexer import CodeIndexer

                def do_find():
                    indexer = CodeIndexer()
                    return indexer.search_code(
                        query=arguments.get("query"),
                        language=arguments.get("language"),
                        project=arguments.get("project"),
                        limit=arguments.get("limit", 10)
                    )

                result = await run_in_thread(do_find)
                return [TextContent(type="text", text=safe_json_dumps({"code_snippets": result, "count": len(result)}))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e), "code_snippets": []}))]

        elif name == "detect_recurring_patterns":
            try:
                from pattern_detector import PatternDetector

                def do_detect():
                    detector = PatternDetector()
                    topics = detector.detect_recurring_topics(arguments.get("threshold", 3))
                    problems = detector.detect_recurring_problems()
                    return {"recurring_topics": topics[:10], "recurring_problems": problems[:10]}

                result = await run_in_thread(do_detect)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "find_knowledge_gaps":
            try:
                from pattern_detector import PatternDetector

                def do_find():
                    detector = PatternDetector()
                    threshold = arguments.get("threshold", 3)
                    return detector.find_knowledge_gaps(min_explanations=threshold)

                result = await run_in_thread(do_find, timeout=30)
                return [TextContent(type="text", text=safe_json_dumps({
                    "knowledge_gaps": result,
                    "count": len(result),
                    "note": "These topics are explained repeatedly - consider adding to permanent context"
                }))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e), "knowledge_gaps": []}))]

        elif name == "track_skill_development":
            try:
                from skill_tracker import SkillTracker

                def do_track():
                    tracker = SkillTracker()
                    skill = arguments.get("skill")
                    if not skill:
                        return {"error": "skill parameter required"}
                    return tracker.track_skill_development(skill)

                result = await run_in_thread(do_track)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "get_code_stats":
            try:
                from code_indexer import CodeIndexer

                def do_stats():
                    indexer = CodeIndexer()
                    return indexer.get_code_stats()

                result = await run_in_thread(do_stats)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "find_duplicate_memories":
            try:
                from deduplicator import MemoryDeduplicator

                def do_find():
                    dedup = MemoryDeduplicator()
                    threshold = arguments.get("threshold", 0.90)
                    return dedup.find_duplicates(threshold)

                result = await run_in_thread(do_find, timeout=120)  # Longer timeout
                return [TextContent(type="text", text=safe_json_dumps({
                    "duplicates": result[:20],  # Limit to first 20
                    "total_found": len(result)
                }))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e), "duplicates": []}))]

        elif name == "merge_duplicates":
            try:
                from deduplicator import MemoryDeduplicator

                def do_merge():
                    dedup = MemoryDeduplicator()
                    # First find duplicates
                    duplicates = dedup.find_duplicates(0.90)
                    # Then merge
                    auto = arguments.get("auto_merge", False)
                    return dedup.merge_duplicates(duplicates, auto_merge=auto)

                result = await run_in_thread(do_merge, timeout=180)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "score_memory_quality":
            try:
                # Use v2 with better thresholds (Phase 3.2 fix)
                from quality_scorer_v2 import QualityScorerV2

                def do_score():
                    scorer = QualityScorerV2()
                    return scorer.score_all_conversations()

                result = await run_in_thread(do_score, timeout=60)
                return [TextContent(type="text", text=safe_json_dumps({
                    "scored_conversations": result[:50],  # Top 50
                    "total": len(result)
                }))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "get_quality_stats":
            try:
                # Use v2 with better thresholds (Phase 3.2 fix)
                from quality_scorer_v2 import QualityScorerV2

                def do_stats():
                    scorer = QualityScorerV2()
                    return scorer.get_quality_stats()

                result = await run_in_thread(do_stats)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "tag_conversation":
            try:
                def do_tag():
                    conv_id = arguments.get("conversation_id")
                    tags = arguments.get("tags", [])
                    success = memory.add_user_tags(conv_id, tags)
                    return {"success": success, "conversation_id": conv_id, "tags_added": tags}
                result = await run_in_thread(do_tag)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "add_note_to_conversation":
            try:
                def do_note():
                    conv_id = arguments.get("conversation_id")
                    note = arguments.get("note")
                    success = memory.add_user_note(conv_id, note)
                    return {"success": success, "conversation_id": conv_id}
                result = await run_in_thread(do_note)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "set_conversation_relevance":
            try:
                def do_relevance():
                    conv_id = arguments.get("conversation_id")
                    relevance = arguments.get("relevance")
                    success = memory.set_conversation_relevance(conv_id, relevance)
                    return {"success": success, "conversation_id": conv_id, "relevance": relevance}
                result = await run_in_thread(do_relevance)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "save_screenshot":
            try:
                from image_processor import ImageProcessor
                def do_save():
                    processor = ImageProcessor()
                    image_data = arguments.get("image_data")
                    conv_id = arguments.get("conversation_id")
                    description = arguments.get("description", "")
                    return processor.save_screenshot(image_data, conv_id, description)
                result = await run_in_thread(do_save)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "search_images":
            try:
                from image_processor import ImageProcessor
                def do_search():
                    processor = ImageProcessor()
                    query = arguments.get("query")
                    limit = arguments.get("limit", 10)
                    return processor.search_images(query, limit)
                result = await run_in_thread(do_search)
                return [TextContent(type="text", text=safe_json_dumps({"images": result, "count": len(result)}))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e), "images": []}))]

        elif name == "create_branch":
            try:
                from branch_tracker import BranchTracker
                def do_create():
                    tracker = BranchTracker()
                    name = arguments.get("name")
                    description = arguments.get("description", "")
                    parent = arguments.get("parent_conversation_id")
                    branch_id = tracker.create_branch(name, description, parent)
                    return {"branch_id": branch_id, "name": name}
                result = await run_in_thread(do_create)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "mark_branch_status":
            try:
                from branch_tracker import BranchTracker
                def do_mark():
                    tracker = BranchTracker()
                    branch_id = arguments.get("branch_id")
                    status = arguments.get("status")
                    reason = arguments.get("reason")

                    if status == "chosen":
                        tracker.mark_branch_chosen(branch_id, reason)
                    else:
                        tracker.mark_branch_abandoned(branch_id, reason)

                    return {"branch_id": branch_id, "status": status, "reason": reason}
                result = await run_in_thread(do_mark)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "get_active_sessions":
            try:
                from session_continuity import SessionContinuityManager

                def do_get_active():
                    manager = SessionContinuityManager()
                    days_back = arguments.get("days_back", 7)
                    return manager.get_active_sessions(days_back=days_back)

                result = await run_in_thread(do_get_active)
                return [TextContent(type="text", text=safe_json_dumps({
                    "active_sessions": result,
                    "count": len(result)
                }))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "get_session_summary":
            try:
                from session_continuity import SessionContinuityManager

                def do_get_summary():
                    manager = SessionContinuityManager()
                    session_id = arguments.get("session_id")
                    include_last_n = arguments.get("include_last_n", 1)
                    return manager.get_session_summary(session_id, include_last_n)

                result = await run_in_thread(do_get_summary)
                return [TextContent(type="text", text=result or "No summary available")]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "detect_session_continuation":
            try:
                from session_continuity import SessionContinuityManager

                def do_detect():
                    manager = SessionContinuityManager()
                    user_prompt = arguments.get("user_prompt")
                    recent_sessions = manager.get_active_sessions(days_back=7)
                    detected_session = manager.detect_continuation(user_prompt, recent_sessions)

                    return {
                        "continuation_detected": detected_session is not None,
                        "suggested_session_id": detected_session,
                        "recent_sessions": recent_sessions[:3]  # Top 3 for context
                    }

                result = await run_in_thread(do_detect)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "system_health_check":
            try:
                # Use FastHealthChecker for instant response (<100ms)
                # No thread pool needed - it's designed to be instant
                from health_checker import run_fast_health_check
                result = run_fast_health_check()
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "suggest_questions":
            try:
                from question_generator import QuestionGenerator

                def do_suggest_questions():
                    generator = QuestionGenerator()
                    limit = arguments.get("limit", 3)
                    importance = arguments.get("importance", "all")

                    # Load profile and recent conversations
                    profile = generator.load_profile()
                    recent_convs = generator.get_recent_conversations(limit=5)
                    history = generator.load_question_history()

                    # Select questions
                    questions = generator.select_questions_to_display(
                        profile, recent_convs, history
                    )

                    # Filter by importance if specified
                    if importance != "all":
                        questions = [q for q in questions if q.get('importance') == importance or q.get('tier') == 1]

                    # Limit results
                    questions = questions[:limit]

                    return {
                        "questions": questions,
                        "count": len(questions)
                    }

                result = await run_in_thread(do_suggest_questions)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # ANTICIPATION ENGINE - Proactive Suggestions
        # ============================================================

        elif name == "get_suggestions":
            try:
                from anticipation_engine import AnticipationEngine

                def do_get_suggestions():
                    engine = AnticipationEngine()

                    user_message = arguments.get("user_message", "")
                    cwd = arguments.get("cwd")
                    recent_tools = arguments.get("recent_tools", [])
                    conversation_length = arguments.get("conversation_length", 0)
                    max_suggestions = arguments.get("max_suggestions", 3)

                    result = engine.get_suggestions(
                        user_message=user_message,
                        cwd=cwd,
                        recent_tools=recent_tools,
                        conversation_length=conversation_length,
                        max_suggestions=max_suggestions
                    )

                    return result

                result = await run_in_thread(do_get_suggestions)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # STARTUP CONTINUATION - Smart Session Resume
        # ============================================================

        elif name == "check_session_continuation":
            try:
                import time as _csc_time
                _csc_t0 = _csc_time.time()
                sys.stderr.write("CSC: start\n"); sys.stderr.flush()

                from device_registry import get_current_device_tag
                sys.stderr.write(f"CSC: import done {(_csc_time.time()-_csc_t0)*1000:.0f}ms\n"); sys.stderr.flush()

                def do_check():
                    hours = arguments.get("hours", 48)
                    sys.stderr.write(f"CSC: do_check entered {(_csc_time.time()-_csc_t0)*1000:.0f}ms\n"); sys.stderr.flush()

                    # Get current device for cross-device comparison
                    try:
                        current_device = get_current_device_tag()
                    except:
                        current_device = "unknown"
                    sys.stderr.write(f"CSC: device={current_device} {(_csc_time.time()-_csc_t0)*1000:.0f}ms\n"); sys.stderr.flush()

                    # ========== FAST NAS CHECK with socket timeout ==========
                    if not is_nas_reachable(timeout=2.0):
                        return {
                            "has_continuation": False,
                            "session_id": "",
                            "summary": "",
                            "confidence": 0.0,
                            "reason": "NAS not reachable (socket timeout)",
                            "message": "NAS offline - cannot check continuation",
                            "current_device": current_device,
                            "same_device": True,
                            "cross_device_warning": False,
                            "source": "none"
                        }

                    sys.stderr.write(f"CSC: NAS reachable {(_csc_time.time()-_csc_t0)*1000:.0f}ms\n"); sys.stderr.flush()
                    quick_facts_path = AI_MEMORY_BASE / "quick_facts.json"

                    # ========== PHASE 1: CHECK quick_facts.json FIRST ==========
                    # This is the PRIMARY source - fast single-file read
                    try:
                        if quick_facts_path.exists():
                            with open(quick_facts_path, 'r', encoding='utf-8') as f:
                                quick_facts = json.load(f)
                            sys.stderr.write(f"CSC: quick_facts loaded {(_csc_time.time()-_csc_t0)*1000:.0f}ms\n"); sys.stderr.flush()

                            active_work = quick_facts.get("active_work", {})
                            if active_work:
                                # Check if active_work is recent (within hours parameter)
                                last_updated = active_work.get("last_updated", "")
                                is_recent = True
                                if last_updated:
                                    try:
                                        if "T" in last_updated:
                                            work_date = dt.datetime.fromisoformat(last_updated)
                                        else:
                                            work_date = dt.datetime.strptime(last_updated, "%Y-%m-%d")
                                        cutoff = dt.datetime.now() - dt.timedelta(hours=hours)
                                        is_recent = work_date > cutoff
                                    except:
                                        is_recent = True

                                # --- Multi-workstream support ---
                                workstreams = active_work.get("workstreams", [])
                                sys.stderr.write(f"CSC: ws={len(workstreams)} is_recent={is_recent} {(_csc_time.time()-_csc_t0)*1000:.0f}ms\n"); sys.stderr.flush()
                                if workstreams and is_recent:
                                    # Filter to active workstreams
                                    active_ws = [ws for ws in workstreams if ws.get("status") in ("in_progress", "paused")]
                                    sys.stderr.write(f"CSC: active_ws={len(active_ws)} {(_csc_time.time()-_csc_t0)*1000:.0f}ms RETURNING\n"); sys.stderr.flush()
                                    if active_ws:
                                        # Sort by priority
                                        active_ws.sort(key=lambda w: w.get("priority", 99))
                                        primary = active_ws[0]
                                        return {
                                            "has_continuation": True,
                                            "session_id": primary.get("session_id", f"active_work_{primary.get('project', 'unknown')}"),
                                            "summary": f"[ACTIVE PROJECT] {primary.get('project', 'Unknown')} - {primary.get('summary', '')}",
                                            "confidence": 0.95,
                                            "reason": "Found active workstreams in quick_facts.json",
                                            "timestamp": active_work.get("last_updated", dt.datetime.now().isoformat()),
                                            "device_tag": current_device,
                                            "current_device": current_device,
                                            "same_device": primary.get("device", "") == current_device,
                                            "cross_device_warning": primary.get("device", "") != current_device and primary.get("device", "") != "",
                                            "message": f"Active project: {primary.get('project')} | Next: {primary.get('next_action', '')}",
                                            "active_work": active_work,
                                            "workstreams": active_ws,
                                            "workstream_count": len(active_ws),
                                            "next_action": primary.get("next_action", ""),
                                            "key_files": primary.get("key_files", []),
                                            "plan_file": primary.get("plan_file", ""),
                                            "last_completed": primary.get("last_completed", ""),
                                            "source": "quick_facts_workstreams"
                                        }

                                # --- Legacy flat format fallback ---
                                if is_recent and active_work.get("next_action"):
                                    return {
                                        "has_continuation": True,
                                        "session_id": f"active_work_{active_work.get('project', 'unknown')}",
                                        "summary": f"[ACTIVE PROJECT] {active_work.get('project', 'Unknown')} - Phase {active_work.get('current_phase', '?')}: {active_work.get('phase_name', '')}",
                                        "confidence": 0.95,
                                        "reason": "Found active_work in quick_facts.json (legacy format)",
                                        "timestamp": last_updated or dt.datetime.now().isoformat(),
                                        "device_tag": current_device,
                                        "current_device": current_device,
                                        "same_device": True,
                                        "cross_device_warning": False,
                                        "message": f"Active project: {active_work.get('project')} | Next: {active_work.get('next_action', '')}",
                                        "active_work": active_work,
                                        "next_action": active_work.get("next_action", ""),
                                        "key_files": active_work.get("key_files", []),
                                        "plan_file": active_work.get("plan_file", ""),
                                        "last_completed": active_work.get("last_completed", ""),
                                        "source": "quick_facts"
                                    }

                                # --- Stale workstreams: return them anyway (don't scan NAS) ---
                                if not is_recent and workstreams:
                                    paused_ws = [ws for ws in workstreams if ws.get("status") in ("in_progress", "paused")]
                                    if paused_ws:
                                        primary = paused_ws[0]
                                        return {
                                            "has_continuation": True,
                                            "session_id": primary.get("session_id", f"active_work_{primary.get('project', 'unknown')}"),
                                            "summary": f"[STALE - last updated {last_updated[:10]}] {primary.get('project', 'Unknown')} - {primary.get('summary', '')}",
                                            "confidence": 0.6,
                                            "reason": f"Workstreams found but last updated {last_updated[:19]} (older than {hours}h)",
                                            "timestamp": last_updated or dt.datetime.now().isoformat(),
                                            "device_tag": current_device,
                                            "current_device": current_device,
                                            "same_device": primary.get("device", "") == current_device,
                                            "cross_device_warning": primary.get("device", "") != current_device and primary.get("device", "") != "",
                                            "message": f"Stale project: {primary.get('project')} | Next: {primary.get('next_action', '')}",
                                            "active_work": active_work,
                                            "workstreams": paused_ws,
                                            "workstream_count": len(paused_ws),
                                            "next_action": primary.get("next_action", ""),
                                            "source": "quick_facts_workstreams_stale"
                                        }
                    except Exception:
                        pass  # quick_facts check failed, fall back to conversation analysis

                    # ========== PHASE 2: FALLBACK - Conversation analysis ==========
                    # Only runs if quick_facts had NO active_work at all
                    # FAST: cap file scan to avoid NAS timeout
                    try:
                        from session_analyzer import SessionAnalyzer
                        analyzer = SessionAnalyzer()
                        candidate = analyzer.get_best_continuation_candidate(
                            hours=hours,
                            device_tag=current_device
                        )

                        if candidate and candidate["confidence"] >= 0.4:
                            result = {
                                "has_continuation": True,
                                "session_id": candidate["session_id"],
                                "summary": candidate["summary"],
                                "confidence": round(candidate["confidence"], 2),
                                "reason": candidate["reason"],
                                "timestamp": candidate["timestamp"],
                                "device_tag": candidate.get("device_tag", "unknown"),
                                "current_device": current_device,
                                "same_device": candidate.get("same_device", True),
                                "cross_device_warning": candidate.get("cross_device_warning", False),
                                "message": f"Found continuable work: {candidate['summary'][:100]}...",
                                "source": "conversation_analysis"
                            }

                            if candidate.get("cross_device_warning"):
                                session_device = candidate.get("device_tag", "unknown")
                                result["warning_message"] = (
                                    f"Note: This session was from '{session_device}' device. "
                                    f"You're currently on '{current_device}'. "
                                    "Some paths or configurations may differ."
                                )

                            return result
                    except Exception:
                        pass  # SessionAnalyzer failed, return no continuation

                    return {
                        "has_continuation": False,
                        "session_id": "",
                        "summary": "",
                        "confidence": 0.0,
                        "reason": "No recent continuable work found (only quick tasks or old sessions)",
                        "message": "No work to continue - starting fresh",
                        "current_device": current_device,
                        "same_device": True,
                        "cross_device_warning": False,
                        "source": "none"
                    }

                # Use shorter timeout - this should complete in <5s
                sys.stderr.write(f"CSC: calling run_in_thread {(_csc_time.time()-_csc_t0)*1000:.0f}ms\n"); sys.stderr.flush()
                result = await run_in_thread(do_check, timeout=15)
                elapsed_ms = (_csc_time.time()-_csc_t0)*1000
                sys.stderr.write(f"CSC: run_in_thread done {elapsed_ms:.0f}ms\n"); sys.stderr.flush()
                # DEBUG: Write timing to file so we can verify externally
                try:
                    with open(str(AI_MEMORY_BASE / "_csc_timing.txt"), "w") as _tf:
                        _tf.write(f"{_csc_time.strftime('%Y-%m-%d %H:%M:%S', _csc_time.localtime())} completed in {elapsed_ms:.0f}ms source={result.get('source','?')}\n")
                except: pass
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                sys.stderr.write(f"CSC: EXCEPTION {e} {(_csc_time.time()-_csc_t0)*1000:.0f}ms\n"); sys.stderr.flush()
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e), "has_continuation": False}))]

        elif name == "get_continuation_context":
            try:
                from context_injector import ContextInjector
                from session_analyzer import SessionAnalyzer

                def do_get_context():
                    session_id = arguments.get("session_id", "")
                    if not session_id:
                        return {"error": "session_id is required"}

                    # Fast NAS check with socket timeout - BEFORE touching filesystem
                    if not is_nas_reachable(timeout=2.0):
                        return {"error": "NAS not reachable (socket timeout)"}

                    # ========== PHASE 1.5: CHECK quick_facts.json FIRST ==========
                    # If session_id starts with "active_work_", this came from quick_facts
                    # Return the structured active_work data directly
                    if session_id.startswith("active_work_"):
                        try:
                            quick_facts_path = AI_MEMORY_BASE / "quick_facts.json"
                            if quick_facts_path.exists():
                                with open(quick_facts_path, 'r', encoding='utf-8') as f:
                                    quick_facts = json.load(f)

                                active_work = quick_facts.get("active_work", {})
                                if active_work:
                                    # Build actionable context from structured data
                                    next_action = active_work.get("next_action", "")
                                    key_files = active_work.get("key_files", [])
                                    plan_file = active_work.get("plan_file", "")
                                    last_completed = active_work.get("last_completed", "")
                                    project = active_work.get("project", "Unknown")
                                    phase = active_work.get("current_phase", "?")
                                    phase_name = active_work.get("phase_name", "")

                                    # Build formatted context
                                    formatted_lines = [
                                        f"# Active Work: {project}",
                                        f"## Current Phase: {phase} - {phase_name}",
                                        "",
                                        f"**Last Completed:** {last_completed}",
                                        f"**Next Action:** {next_action}",
                                        "",
                                        "## Key Files:",
                                    ]
                                    for f in key_files:
                                        formatted_lines.append(f"- {f}")
                                    if plan_file:
                                        formatted_lines.append(f"\n**Plan File:** {plan_file}")
                                        formatted_lines.append("\n> **Recommended:** Read the plan file first for full context.")

                                    return {
                                        "session_id": session_id,
                                        "summary": f"{project} - Phase {phase}: {phase_name}",
                                        "key_points": [
                                            f"Last completed: {last_completed}",
                                            f"Next action: {next_action}"
                                        ],
                                        "pending_work": [next_action] if next_action else [],
                                        "files_loaded": len(key_files),
                                        "formatted_context": "\n".join(formatted_lines),
                                        "message": f"Loaded context for session: {session_id}",
                                        # STRUCTURED DATA - the key improvement
                                        "active_work": active_work,
                                        "next_action": next_action,
                                        "key_files": key_files,
                                        "plan_file": plan_file,
                                        "last_completed": last_completed,
                                        "source": "quick_facts"
                                    }
                        except Exception:
                            pass  # Fall back to conversation analysis

                    # ========== FALLBACK: Conversation analysis ==========
                    # Get suggested context from analyzer
                    analyzer = SessionAnalyzer()
                    sessions = analyzer.get_recent_sessions(hours=168)
                    suggested_context = []

                    for session in sessions:
                        if session["id"] == session_id:
                            analysis = analyzer.analyze_session(session["conversation"])
                            suggested_context = analysis.get("suggested_context", [])
                            break

                    # Prepare full context
                    injector = ContextInjector()
                    context = injector.prepare_continuation_context(session_id, suggested_context)

                    return {
                        "session_id": session_id,
                        "summary": context["conversation_summary"],
                        "key_points": context["key_points"],
                        "pending_work": context["pending_work"],
                        "files_loaded": len(context["files_context"]),
                        "formatted_context": context["formatted_prompt"],
                        "message": f"Loaded context for session: {session_id}",
                        "source": "conversation_analysis"
                    }

                result = await run_in_thread(do_get_context)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "update_active_work":
            try:
                def do_update_active_work():

                    # Fast NAS check with socket timeout - BEFORE touching filesystem
                    if not is_nas_reachable(timeout=2.0):
                        return {"success": False, "error": "NAS not reachable (socket timeout)"}

                    quick_facts_path = AI_MEMORY_BASE / "quick_facts.json"
                    lock_path = AI_MEMORY_BASE / ".quick_facts.lock"

                    # Acquire file lock to prevent concurrent write races
                    lock_fd = None
                    try:
                        import fcntl
                        lock_fd = open(lock_path, 'w')
                        fcntl.flock(lock_fd, fcntl.LOCK_EX)
                    except (ImportError, OSError):
                        pass  # Windows or lock failure - proceed without lock

                    try:
                        return _update_active_work_inner(quick_facts_path, arguments)
                    finally:
                        if lock_fd:
                            try:
                                import fcntl
                                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                            except (ImportError, OSError):
                                pass
                            lock_fd.close()

                def _update_active_work_inner(quick_facts_path, arguments):
                    import hashlib

                    # Load existing quick_facts
                    quick_facts = {}
                    if quick_facts_path.exists():
                        try:
                            with open(quick_facts_path, 'r', encoding='utf-8') as f:
                                quick_facts = json.load(f)
                        except Exception:
                            pass

                    # Handle clear request
                    if arguments.get("clear"):
                        quick_facts["active_work"] = {"last_updated": dt.datetime.now().isoformat(), "last_device": "", "workstreams": []}
                        with open(quick_facts_path, 'w', encoding='utf-8') as f:
                            json.dump(quick_facts, f, indent=2)
                        return {
                            "success": True,
                            "action": "cleared",
                            "message": "active_work workstreams cleared"
                        }

                    # --- Migrate flat format to workstreams if needed ---
                    active_work = quick_facts.get("active_work", {})
                    if "workstreams" not in active_work:
                        old_ws = []
                        if active_work.get("project") and active_work.get("next_action"):
                            old_ws.append({
                                "id": "ws_" + hashlib.md5(active_work["project"].encode()).hexdigest()[:6],
                                "project": active_work["project"],
                                "summary": active_work.get("phase_name", ""),
                                "status": "in_progress",
                                "next_action": active_work.get("next_action", ""),
                                "last_completed": active_work.get("last_completed", ""),
                                "key_files": active_work.get("key_files", []),
                                "device": "",
                                "started_at": active_work.get("last_updated", dt.datetime.now().isoformat()),
                                "updated_at": dt.datetime.now().isoformat(),
                                "session_id": "",
                                "priority": 1
                            })
                        active_work = {
                            "last_updated": dt.datetime.now().isoformat(),
                            "last_device": "",
                            "workstreams": old_ws
                        }

                    workstreams = active_work.get("workstreams", [])
                    now = dt.datetime.now().isoformat()

                    # Get current device tag
                    try:
                        device_tag = get_device_metadata().get("device_tag", "unknown")
                    except Exception:
                        device_tag = "unknown"

                    # --- Handle list_workstreams ---
                    if arguments.get("list_workstreams"):
                        active_work["last_device"] = device_tag
                        return {
                            "success": True,
                            "action": "listed",
                            "active_work": active_work,
                            "workstream_count": len(workstreams),
                            "in_progress": len([w for w in workstreams if w.get("status") == "in_progress"]),
                            "message": f"{len(workstreams)} workstreams ({len([w for w in workstreams if w.get('status') == 'in_progress'])} active)"
                        }

                    # --- Handle remove_workstream ---
                    if arguments.get("remove_workstream"):
                        rm_id = arguments["remove_workstream"]
                        before = len(workstreams)
                        workstreams = [w for w in workstreams if w.get("id") != rm_id]
                        active_work["workstreams"] = workstreams
                        active_work["last_updated"] = now
                        active_work["last_device"] = device_tag
                        quick_facts["active_work"] = active_work
                        with open(quick_facts_path, 'w', encoding='utf-8') as f:
                            json.dump(quick_facts, f, indent=2)
                        removed = before - len(workstreams)
                        return {
                            "success": True,
                            "action": "removed",
                            "removed_count": removed,
                            "remaining": len(workstreams),
                            "message": f"Removed {removed} workstream(s), {len(workstreams)} remaining"
                        }

                    # --- Find or create workstream ---
                    target_ws = None
                    target_idx = None

                    # By workstream_id
                    if arguments.get("workstream_id"):
                        for i, ws in enumerate(workstreams):
                            if ws.get("id") == arguments["workstream_id"]:
                                target_ws = ws
                                target_idx = i
                                break

                    # By project name
                    if target_ws is None and arguments.get("project"):
                        project_name = arguments["project"]
                        for i, ws in enumerate(workstreams):
                            if ws.get("project", "").lower() == project_name.lower():
                                target_ws = ws
                                target_idx = i
                                break

                        # Create new if not found
                        if target_ws is None:
                            max_priority = max((ws.get("priority", 0) for ws in workstreams), default=0)
                            target_ws = {
                                "id": "ws_" + hashlib.md5(project_name.encode()).hexdigest()[:6],
                                "project": project_name,
                                "summary": "",
                                "status": "in_progress",
                                "next_action": "",
                                "last_completed": "",
                                "key_files": [],
                                "device": device_tag,
                                "started_at": now,
                                "updated_at": now,
                                "session_id": "",
                                "priority": max_priority + 1
                            }
                            workstreams.append(target_ws)
                            target_idx = len(workstreams) - 1

                    # If we still have no target, update the first in_progress or create generic
                    if target_ws is None:
                        if workstreams:
                            for i, ws in enumerate(workstreams):
                                if ws.get("status") == "in_progress":
                                    target_ws = ws
                                    target_idx = i
                                    break
                        if target_ws is None:
                            # No project specified and no existing workstreams - create placeholder
                            target_ws = {
                                "id": "ws_" + hashlib.md5(now.encode()).hexdigest()[:6],
                                "project": "Unknown Project",
                                "summary": "",
                                "status": "in_progress",
                                "next_action": "",
                                "last_completed": "",
                                "key_files": [],
                                "device": device_tag,
                                "started_at": now,
                                "updated_at": now,
                                "session_id": "",
                                "priority": 1
                            }
                            workstreams.append(target_ws)
                            target_idx = len(workstreams) - 1

                    # --- Apply updates to target workstream ---
                    if arguments.get("project"):
                        target_ws["project"] = arguments["project"]
                    if arguments.get("summary"):
                        target_ws["summary"] = arguments["summary"]
                    if arguments.get("status"):
                        target_ws["status"] = arguments["status"]
                    if arguments.get("next_action"):
                        target_ws["next_action"] = arguments["next_action"]
                    if arguments.get("last_completed"):
                        target_ws["last_completed"] = arguments["last_completed"]
                    if arguments.get("key_files"):
                        target_ws["key_files"] = arguments["key_files"]
                    if arguments.get("plan_file"):
                        target_ws["plan_file"] = arguments["plan_file"]
                    if arguments.get("current_phase"):
                        target_ws["current_phase"] = arguments["current_phase"]
                    if arguments.get("phase_name"):
                        target_ws["phase_name"] = arguments["phase_name"]

                    target_ws["device"] = device_tag
                    target_ws["updated_at"] = now

                    # Update in list
                    workstreams[target_idx] = target_ws

                    # --- Cap at 10 workstreams ---
                    if len(workstreams) > 10:
                        in_prog = [w for w in workstreams if w.get("status") == "in_progress"]
                        others = [w for w in workstreams if w.get("status") != "in_progress"]
                        others.sort(key=lambda w: w.get("updated_at", ""), reverse=True)
                        workstreams = in_prog + others[:10 - len(in_prog)]

                    # --- Save back ---
                    active_work["workstreams"] = workstreams
                    active_work["last_updated"] = now
                    active_work["last_device"] = device_tag
                    quick_facts["active_work"] = active_work

                    with open(quick_facts_path, 'w', encoding='utf-8') as f:
                        json.dump(quick_facts, f, indent=2)

                    return {
                        "success": True,
                        "action": "updated",
                        "workstream": target_ws,
                        "workstream_count": len(workstreams),
                        "message": f"Updated workstream: {target_ws.get('project', 'Unknown')} - {target_ws.get('next_action', 'No action set')}"
                    }

                # Short timeout - this is a single file read+write, should be <5s
                result = await run_in_thread(do_update_active_work, timeout=15)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # DEVICE REGISTRY - Multi-Device Awareness
        # ============================================================

        elif name == "get_current_device":
            try:
                def do_get_device():
                    from device_registry import get_device_registry as get_registry
                    registry = get_registry()
                    device = registry.get_current_device()
                    return {
                        "device_tag": device.get("device_type", "unknown"),
                        "device_name": device.get("friendly_name", device.get("device_name", "Unknown")),
                        "hostname": device.get("hostname", ""),
                        "os": device.get("os", ""),
                        "os_version": device.get("os_version", ""),
                        "architecture": device.get("architecture", ""),
                        "is_current": True,
                        "conversation_count": device.get("conversation_count", 0),
                        "last_seen": device.get("last_seen", "")
                    }

                result = await run_in_thread(do_get_device)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "get_all_devices":
            try:
                def do_get_all():
                    from device_registry import get_device_registry as get_registry
                    registry = get_registry()
                    devices = registry.get_all_devices()
                    return {
                        "devices": devices,
                        "count": len(devices),
                        "current_device": registry.get_device_tag()
                    }

                result = await run_in_thread(do_get_all)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "register_device":
            try:
                def do_register():
                    from device_registry import get_device_registry as get_registry
                    registry = get_registry()
                    friendly_name = arguments.get("friendly_name")
                    description = arguments.get("description")

                    device = registry.register_device(
                        friendly_name=friendly_name,
                        description=description
                    )

                    return {
                        "success": True,
                        "device_tag": device.get("device_type", "unknown"),
                        "device_name": device.get("friendly_name", "Unknown"),
                        "hostname": device.get("hostname", ""),
                        "message": f"Device registered as '{device.get('friendly_name')}'"
                    }

                result = await run_in_thread(do_register)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "search_by_device":
            try:
                def do_device_search():
                    from device_registry import get_device_registry as get_registry
                    get_registry()
                    query = arguments.get("query", "")
                    device_tag = arguments.get("device_tag")
                    include_untagged = arguments.get("include_untagged", True)

                    # Tokenize query into individual words for scoring
                    import re as _re
                    stop_words = {"the", "a", "an", "is", "was", "were", "be", "been",
                                  "being", "have", "has", "had", "do", "does", "did",
                                  "will", "would", "could", "should", "may", "might",
                                  "shall", "can", "to", "of", "in", "for", "on", "with",
                                  "at", "by", "from", "as", "into", "about", "it", "its",
                                  "and", "or", "but", "not", "this", "that", "what", "which"}
                    query_words = [w.lower() for w in _re.findall(r'\w+', query) if len(w) > 1]
                    query_words_filtered = [w for w in query_words if w not in stop_words]
                    # Use all words if filtering removes everything
                    if not query_words_filtered:
                        query_words_filtered = query_words

                    # Search conversations with relevance scoring
                    scored_results = []
                    conv_path = memory.conversations_path

                    # FAST approach: read only metadata (first ~4KB), skip message bodies
                    # Sort by mtime newest first, cap at 100 files
                    try:
                        all_files = sorted(conv_path.glob("*.json"),
                                          key=lambda f: f.stat().st_mtime, reverse=True)[:100]
                    except Exception:
                        all_files = list(conv_path.glob("*.json"))[:100]

                    for conv_file in all_files:
                        try:
                            # Read only first 8KB to get metadata without loading full messages
                            with open(conv_file, "r", encoding="utf-8") as f:
                                head = f.read(8192)

                            # Quick device_tag check via string search before full parse
                            if device_tag:
                                tag_needle = f'"device_tag": "{device_tag}"'
                                has_tag = tag_needle in head
                                if not has_tag:
                                    if not include_untagged:
                                        continue
                                    # Check if it has ANY device_tag (then skip — wrong device)
                                    if '"device_tag"' in head:
                                        continue

                            # Now parse the full file for matching ones only
                            with open(conv_file, "r", encoding="utf-8") as f:
                                conv = json.load(f)

                            conv_device = conv.get("metadata", {}).get("device_tag")
                            meta = conv.get("metadata", {})
                            searchable_parts = [
                                meta.get("summary", ""),
                                " ".join(meta.get("tags", [])),
                                " ".join(meta.get("topics", [])),
                            ]
                            # search_index only (skip raw messages — too slow over NAS)
                            si = conv.get("search_index", {})
                            if isinstance(si, dict):
                                kw = si.get("keywords", "")
                                searchable_parts.append(" ".join(kw) if isinstance(kw, list) else str(kw))
                                searchable_parts.append(str(si.get("summary", "")))

                            searchable = " ".join(searchable_parts).lower()

                            score = 0
                            matched_words = []
                            for word in query_words_filtered:
                                if word in searchable:
                                    score += 1
                                    matched_words.append(word)

                            if score > 0:
                                scored_results.append({
                                    "conversation_id": conv.get("id", conv_file.stem),
                                    "device_tag": conv_device or "untagged",
                                    "device_name": meta.get("device_name", "Unknown"),
                                    "timestamp": conv.get("timestamp", ""),
                                    "summary": meta.get("summary", "")[:200],
                                    "relevance_score": score,
                                    "matched_words": matched_words,
                                    "match_ratio": round(score / len(query_words_filtered), 2)
                                })

                        except Exception:
                            continue

                    scored_results.sort(
                        key=lambda x: (x.get("relevance_score", 0), x.get("timestamp", "")),
                        reverse=True
                    )

                    return {
                        "results": scored_results[:20],
                        "count": len(scored_results),
                        "device_filter": device_tag or "all",
                        "query": query,
                        "query_words": query_words_filtered,
                        "files_scanned": len(all_files)
                    }

                result = await run_in_thread(do_device_search, timeout=30)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        # ============================================================
        # LEARNING SYSTEM - Solution Tracking & Anti-Patterns
        # ============================================================

        elif name == "record_solution":
            try:
                from solution_tracker import SolutionTracker

                def do_record():
                    tracker = SolutionTracker()
                    return tracker.record_solution(
                        problem=arguments.get("problem", ""),
                        solution=arguments.get("solution", ""),
                        context=arguments.get("context", ""),
                        tags=arguments.get("tags", []),
                        conversation_id=arguments.get("conversation_id"),
                        supersedes=arguments.get("supersedes"),
                        caused_by_failure=arguments.get("caused_by_failure")
                    )

                result = await run_in_thread(do_record)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "record_failure":
            try:
                from solution_tracker import SolutionTracker

                def do_record():
                    tracker = SolutionTracker()
                    return tracker.record_failure(
                        solution_id=arguments.get("solution_id", ""),
                        failure_description=arguments.get("failure_description", ""),
                        error_message=arguments.get("error_message", ""),
                        conversation_id=arguments.get("conversation_id")
                    )

                result = await run_in_thread(do_record)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "record_antipattern":
            try:
                from solution_tracker import SolutionTracker

                def do_record():
                    tracker = SolutionTracker()
                    return tracker.record_antipattern(
                        what_not_to_do=arguments.get("what_not_to_do", ""),
                        why_it_failed=arguments.get("why_it_failed", ""),
                        error_details=arguments.get("error_details", ""),
                        original_problem=arguments.get("original_problem", ""),
                        conversation_id=arguments.get("conversation_id"),
                        tags=arguments.get("tags", [])
                    )

                result = await run_in_thread(do_record)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "find_solution":
            try:
                from solution_tracker import SolutionTracker

                def do_find():
                    tracker = SolutionTracker()
                    return tracker.find_solution(arguments.get("problem", ""))

                result = await run_in_thread(do_find)
                return [TextContent(type="text", text=safe_json_dumps({
                    "solutions": result[:10],
                    "count": len(result)
                }))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "find_antipatterns":
            try:
                from solution_tracker import SolutionTracker

                def do_find():
                    tracker = SolutionTracker()
                    return tracker.find_antipatterns(
                        problem=arguments.get("problem"),
                        tags=arguments.get("tags")
                    )

                result = await run_in_thread(do_find)
                return [TextContent(type="text", text=safe_json_dumps({
                    "antipatterns": result[:10],
                    "count": len(result),
                    "note": "These are approaches that FAILED - avoid them!"
                }))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "get_solution_chain":
            try:
                from solution_tracker import SolutionTracker

                def do_get():
                    tracker = SolutionTracker()
                    return tracker.get_solution_chain(arguments.get("solution_id", ""))

                result = await run_in_thread(do_get)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "confirm_solution":
            try:
                from solution_tracker import SolutionTracker

                def do_confirm():
                    tracker = SolutionTracker()
                    return tracker.confirm_solution_works(
                        solution_id=arguments.get("solution_id", ""),
                        conversation_id=arguments.get("conversation_id")
                    )

                result = await run_in_thread(do_confirm)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "get_learnings_summary":
            try:
                from solution_tracker import SolutionTracker

                def do_get():
                    tracker = SolutionTracker()
                    return tracker.get_learnings_summary()

                result = await run_in_thread(do_get)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "analyze_conversation_learnings":
            try:
                from learning_extractor import LearningExtractor

                def do_analyze():
                    extractor = LearningExtractor()
                    conv_id = arguments.get("conversation_id", "")

                    # Load conversation
                    conv_path = AI_MEMORY_BASE / "conversations" / f"{conv_id}.json"
                    if not conv_path.exists():
                        return {"error": f"Conversation {conv_id} not found"}

                    with open(conv_path, 'r', encoding='utf-8') as f:
                        conv = json.load(f)

                    # Analyze
                    learnings = extractor.analyze_conversation(conv)

                    # Save if requested
                    if arguments.get("save", True):
                        extractor.save_learnings(learnings)

                    return learnings

                result = await run_in_thread(do_analyze)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        elif name == "get_recent_learnings":
            try:
                from learning_extractor import LearningExtractor

                def do_get():
                    extractor = LearningExtractor()
                    limit = arguments.get("limit", 10)
                    return {
                        "learnings": extractor.get_recent_learnings(limit),
                        "count": limit
                    }

                result = await run_in_thread(do_get)
                return [TextContent(type="text", text=safe_json_dumps(result))]
            except Exception as e:
                return [TextContent(type="text", text=safe_json_dumps({"error": str(e)}))]

        else:
            return [TextContent(type="text", text=safe_json_dumps({"error": f"Unknown tool: {name}"}))]

    except TimeoutError as e:
        # AGENT 16: Log timeout errors
        if start_time:
            try:
                duration_ms = (time.time() - start_time) * 1000
                sys.stderr.write(f"MCP Tool TIMEOUT: {name} after {duration_ms:.0f}ms | {str(e)[:100]}\n")
                sys.stderr.flush()
            except:
                pass
        return [TextContent(type="text", text=safe_json_dumps({"error": sanitize_text(str(e), 200), "suggestion": "NAS may be slow or unavailable"}))]
    except Exception as e:
        # AGENT 16: Log general errors
        if start_time:
            try:
                duration_ms = (time.time() - start_time) * 1000
                sys.stderr.write(f"MCP Tool ERROR: {name} after {duration_ms:.0f}ms | {str(e)[:100]}\n")
                sys.stderr.flush()
            except:
                pass
        return [TextContent(type="text", text=safe_json_dumps({"error": sanitize_text(str(e), 200)}))]
    finally:
        # AGENT 16: Log successful tool completion
        if start_time:
            try:
                duration_ms = (time.time() - start_time) * 1000
                sys.stderr.write(f"MCP Tool: {name} completed in {duration_ms:.0f}ms\n")
                sys.stderr.flush()
            except:
                pass


def _blocking_init():
    """Synchronous initialization - runs in thread to avoid blocking the event loop.

    All operations here are blocking (imports, model loading, FAISS index, etc.)
    and MUST NOT run on the async event loop or they'll prevent the MCP protocol
    handshake from completing, causing Claude Code to time out.

    FAST-FAIL BRAIN: Uses quick NAS check (2s timeout) to decide initialization path.
    """
    global _memory, _embeddings
    import time as _time

    # FAST-FAIL: Quick NAS check with latency measurement
    start = _time.time()
    nas_available = is_nas_reachable(timeout=2.0)
    latency_ms = round((_time.time() - start) * 1000)

    # Clear startup status banner
    sys.stderr.write("\n")
    sys.stderr.write("=" * 50 + "\n")

    if not nas_available:
        # NAS unavailable - use local-only mode
        sys.stderr.write("  AI Memory: NAS OFFLINE - local mode\n")
        sys.stderr.write("  > record_learning saves locally, syncs later\n")
        sys.stderr.write("  > search/profile unavailable until NAS online\n")
        sys.stderr.write("=" * 50 + "\n\n")
        sys.stderr.flush()
        _memory = None
        _embeddings = None
        return

    # NAS available - show status with latency
    sys.stderr.write(f"  AI Memory: NAS connected ({latency_ms}ms latency)\n")
    sys.stderr.flush()

    # Initialize memory service
    memory_ok = False
    try:
        _memory = _init_memory()
        memory_ok = True
    except Exception as e:
        sys.stderr.write(f"  > Memory service: FAILED ({e})\n")
        _memory = None

    # Initialize embeddings engine (constructor is now fast due to lazy trackers)
    embeddings_ok = False
    try:
        _embeddings = _init_embeddings()
        embeddings_ok = True
    except Exception as e:
        sys.stderr.write(f"  > Search: FAILED ({e})\n")
        _embeddings = None

    # Show status
    if memory_ok:
        sys.stderr.write("  > Memory service: ready\n")
    if embeddings_ok:
        sys.stderr.write("  > Search: ready (model pre-warmed)\n")

    sys.stderr.write("=" * 50 + "\n\n")
    sys.stderr.flush()


async def _background_init():
    """Initialize services in background after server starts accepting connections.

    Runs all blocking work in a thread executor so the async event loop stays
    free to handle the MCP protocol handshake (initialize request/response).
    Hard-capped at 45s to prevent hanging if NAS is slow/unresponsive.
    """
    global _initialized, _init_event

    try:
        loop = asyncio.get_event_loop()
        await asyncio.wait_for(
            loop.run_in_executor(_executor, _blocking_init),
            timeout=45.0
        )

        _initialized = True
        _init_event.set()  # Signal waiters that init is complete
        sys.stderr.write("AI Memory MCP: Ready for queries!\n")
        sys.stderr.flush()

    except asyncio.TimeoutError:
        sys.stderr.write("AI Memory MCP: Warning - initialization timed out after 45s (NAS slow?)\n")
        sys.stderr.flush()
        _initialized = False
        _init_event.set()  # Signal waiters so they don't hang forever

    except Exception as e:
        sys.stderr.write(f"AI Memory MCP: Warning - {e}\n")
        sys.stderr.flush()
        _initialized = False
        _init_event.set()  # Signal waiters even on failure so they don't hang


async def main():
    """Run the MCP server - accepts connections immediately, initializes in background"""
    global _init_event

    sys.stderr.write("AI Memory MCP: Starting server (initialization runs in background)...\n")
    sys.stderr.flush()

    # Create the init event now that we have an event loop
    _init_event = asyncio.Event()

    # Use the ORIGINAL stdout (saved before redirect) for MCP protocol.
    # sys.stdout was redirected to stderr early in module load to prevent
    # print() from corrupting the MCP JSON-RPC stream.
    from io import TextIOWrapper

    import anyio

    _mcp_stdout = anyio.wrap_file(
        TextIOWrapper(_original_stdout.buffer, encoding="utf-8")
    )

    # Start the server FIRST so Claude Code can connect immediately
    async with stdio_server(stdout=_mcp_stdout) as (read_stream, write_stream):
        # Kick off background initialization (non-blocking)
        asyncio.create_task(_background_init())

        # Server is now accepting connections while init runs in background
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
