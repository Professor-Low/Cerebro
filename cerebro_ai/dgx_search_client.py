"""
DGX Search Service Client

Provides async HTTP client for the DGX-based search service.
Falls back gracefully if DGX is unavailable.

Usage:
    from dgx_search_client import dgx_search, is_dgx_available

    # Check if DGX is available
    if await is_dgx_available():
        results = await dgx_search("my query", top_k=10, mode="hybrid")
    else:
        # Fall back to local search
        ...
"""

import asyncio
import concurrent.futures
import json
import os
import socket
from functools import partial
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


async def _run_in_fresh_thread(func, *args, timeout=5.0):
    """Run blocking func in a per-call thread to avoid pool starvation."""
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dgx-search")
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(executor, partial(func, *args) if args else func),
            timeout=timeout
        )
    finally:
        executor.shutdown(wait=False)

# Configuration
DGX_HOST = os.environ.get("DGX_SEARCH_HOST", os.environ.get("CEREBRO_DGX_HOST", ""))
DGX_PORT = int(os.environ.get("DGX_SEARCH_PORT", "8780"))
DGX_TIMEOUT = float(os.environ.get("DGX_SEARCH_TIMEOUT", "5.0"))

DGX_SEARCH_URL = f"http://{DGX_HOST}:{DGX_PORT}"

# Cache for DGX availability (avoid repeated health checks)
_dgx_available: Optional[bool] = None
_dgx_check_time: float = 0
_DGX_CHECK_INTERVAL = 30  # Re-check every 30 seconds


def _is_dgx_reachable() -> bool:
    """Quick socket check if DGX is reachable"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        result = sock.connect_ex((DGX_HOST, DGX_PORT))
        sock.close()
        return result == 0
    except Exception:
        return False


async def is_dgx_available() -> bool:
    """
    Check if DGX search service is available.
    Caches result for _DGX_CHECK_INTERVAL seconds.
    Uses per-call ThreadPoolExecutor to avoid thread pool starvation
    when timed-out calls leave orphan threads in the default pool.
    """
    global _dgx_available, _dgx_check_time
    import time

    now = time.time()
    if _dgx_available is not None and (now - _dgx_check_time) < _DGX_CHECK_INTERVAL:
        return _dgx_available

    try:
        reachable = await _run_in_fresh_thread(_is_dgx_reachable, timeout=2.0)

        if not reachable:
            _dgx_available = False
            _dgx_check_time = now
            return False

        # Try health endpoint
        available = await _run_in_fresh_thread(_check_health, timeout=3.0)

        _dgx_available = available
        _dgx_check_time = now
        return available

    except asyncio.TimeoutError:
        _dgx_available = False
        _dgx_check_time = now
        return False
    except Exception:
        _dgx_available = False
        _dgx_check_time = now
        return False


def _check_health() -> bool:
    """Blocking health check"""
    try:
        req = Request(f"{DGX_SEARCH_URL}/health")
        req.add_header("Accept", "application/json")
        with urlopen(req, timeout=DGX_TIMEOUT) as response:
            if response.status == 200:
                data = json.loads(response.read())
                return data.get("status") == "healthy"
        return False
    except Exception:
        return False


def _do_search(query: str, top_k: int = 10, mode: str = "hybrid", alpha: float = 0.7) -> Dict[str, Any]:
    """Blocking search call"""
    try:
        import urllib.parse
        params = urllib.parse.urlencode({
            "q": query,
            "top_k": top_k,
            "mode": mode,
            "alpha": alpha
        })

        url = f"{DGX_SEARCH_URL}/search?{params}"
        req = Request(url)
        req.add_header("Accept", "application/json")

        with urlopen(req, timeout=DGX_TIMEOUT) as response:
            if response.status == 200:
                return json.loads(response.read())
            else:
                return {"error": f"HTTP {response.status}", "results": []}

    except HTTPError as e:
        return {"error": f"HTTP error: {e.code}", "results": []}
    except URLError as e:
        return {"error": f"Connection error: {e.reason}", "results": []}
    except Exception as e:
        return {"error": str(e), "results": []}


async def dgx_search(
    query: str,
    top_k: int = 10,
    mode: str = "hybrid",
    alpha: float = 0.7,
    timeout: float = None
) -> Optional[Dict[str, Any]]:
    """
    Search using DGX search service.

    Args:
        query: Search query
        top_k: Number of results to return
        mode: Search mode (hybrid, keyword, semantic)
        alpha: Semantic weight for hybrid search (0-1)
        timeout: Request timeout (default: DGX_TIMEOUT)

    Returns:
        Search results dict or None if DGX unavailable
    """
    if timeout is None:
        timeout = DGX_TIMEOUT

    try:
        result = await _run_in_fresh_thread(_do_search, query, top_k, mode, alpha, timeout=timeout)

        if "error" in result and not result.get("results"):
            # DGX returned error with no results
            return None

        return result

    except asyncio.TimeoutError:
        return None
    except Exception:
        return None


async def dgx_keyword_search(query: str, top_k: int = 10) -> Optional[Dict[str, Any]]:
    """Keyword-only search via DGX"""
    return await dgx_search(query, top_k=top_k, mode="keyword")


async def dgx_semantic_search(query: str, top_k: int = 10) -> Optional[Dict[str, Any]]:
    """Semantic-only search via DGX"""
    return await dgx_search(query, top_k=top_k, mode="semantic")


async def dgx_stats() -> Optional[Dict[str, Any]]:
    """Get DGX search service stats"""
    def _get_stats():
        try:
            req = Request(f"{DGX_SEARCH_URL}/stats")
            req.add_header("Accept", "application/json")
            with urlopen(req, timeout=DGX_TIMEOUT) as response:
                if response.status == 200:
                    return json.loads(response.read())
        except Exception:
            pass
        return None

    try:
        return await _run_in_fresh_thread(_get_stats, timeout=DGX_TIMEOUT)
    except (asyncio.TimeoutError, TimeoutError):
        return None


async def dgx_reindex() -> bool:
    """Trigger reindex on DGX"""
    def _trigger_reindex():
        try:
            req = Request(f"{DGX_SEARCH_URL}/reindex", method="POST")
            req.add_header("Accept", "application/json")
            with urlopen(req, timeout=DGX_TIMEOUT) as response:
                return response.status == 200
        except Exception:
            return False

    try:
        return await _run_in_fresh_thread(_trigger_reindex, timeout=DGX_TIMEOUT)
    except (asyncio.TimeoutError, TimeoutError):
        return False


# Invalidate cache (call after saving new conversations)
def invalidate_dgx_cache():
    """Force re-check of DGX availability on next call"""
    global _dgx_available, _dgx_check_time
    _dgx_available = None
    _dgx_check_time = 0
