"""
DGX Embedding Service Client

Provides async HTTP client for the DGX-based embedding service.
Falls back gracefully if DGX is unavailable.

Usage:
    from dgx_embedding_client import dgx_embed, is_dgx_embedding_available

    # Check if DGX is available
    if await is_dgx_embedding_available():
        vectors = await dgx_embed(["text1", "text2"], batch_size=128)
        # vectors is np.ndarray shape (2, 768)
    else:
        # Fall back to local embedding or keyword search
        ...
"""

import asyncio
import concurrent.futures
import json
import os
import socket
from functools import partial
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


async def _run_in_fresh_thread(func, *args, timeout=5.0):
    """Run blocking func in a per-call thread to avoid pool starvation."""
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dgx-embed")
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(executor, partial(func, *args) if args else func),
            timeout=timeout
        )
    finally:
        executor.shutdown(wait=False)

# Configuration
DGX_HOST = os.environ.get("DGX_EMBEDDING_HOST", os.environ.get("CEREBRO_DGX_HOST", ""))
DGX_PORT = int(os.environ.get("DGX_EMBEDDING_PORT", "8781"))
DGX_TIMEOUT = float(os.environ.get("DGX_EMBEDDING_TIMEOUT", "10.0"))

DGX_EMBEDDING_URL = f"http://{DGX_HOST}:{DGX_PORT}"

# Cache for DGX availability (avoid repeated health checks)
_dgx_available: Optional[bool] = None

# If no DGX host configured, mark as unavailable immediately
if not DGX_HOST:
    _dgx_available = False
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


async def is_dgx_embedding_available() -> bool:
    """
    Check if DGX embedding service is available.
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
        req = Request(f"{DGX_EMBEDDING_URL}/health")
        req.add_header("Accept", "application/json")
        with urlopen(req, timeout=DGX_TIMEOUT) as response:
            if response.status == 200:
                data = json.loads(response.read())
                return data.get("status") == "healthy"
        return False
    except Exception:
        return False


def _do_embed(texts: List[str], batch_size: int = 128, normalize: bool = True) -> Optional[np.ndarray]:
    """Blocking embed call"""
    try:
        payload = json.dumps({
            "texts": texts,
            "batch_size": batch_size,
            "normalize": normalize
        }).encode("utf-8")

        req = Request(
            f"{DGX_EMBEDDING_URL}/embed",
            data=payload,
            method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")

        with urlopen(req, timeout=DGX_TIMEOUT) as response:
            if response.status == 200:
                data = json.loads(response.read())
                embeddings = data.get("embeddings", [])
                return np.array(embeddings, dtype=np.float32)
            else:
                return None

    except HTTPError as e:
        print(f"[DGX Embed] HTTP error: {e.code}")
        return None
    except URLError as e:
        print(f"[DGX Embed] Connection error: {e.reason}")
        return None
    except Exception as e:
        print(f"[DGX Embed] Error: {e}")
        return None


async def dgx_embed(
    texts: List[str],
    batch_size: int = 128,
    normalize: bool = True,
    timeout: float = None
) -> Optional[np.ndarray]:
    """
    Generate embeddings using DGX embedding service.

    Args:
        texts: List of strings to embed
        batch_size: Processing batch size (default 128, max 512)
        normalize: Whether to L2 normalize for cosine similarity (default True)
        timeout: Request timeout (default: DGX_TIMEOUT)

    Returns:
        Numpy array of embeddings (N x 768) or None if DGX unavailable
    """
    if timeout is None:
        timeout = DGX_TIMEOUT

    if not texts:
        return np.array([], dtype=np.float32)

    try:
        result = await _run_in_fresh_thread(_do_embed, texts, batch_size, normalize, timeout=timeout)
        return result

    except (asyncio.TimeoutError, TimeoutError):
        print(f"[DGX Embed] Request timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"[DGX Embed] Error: {e}")
        return None


async def dgx_embed_batch(
    texts: List[str],
    max_batch: int = 512,
    internal_batch: int = 128
) -> Optional[np.ndarray]:
    """
    Embed large number of texts by splitting into chunks.

    Args:
        texts: List of strings to embed
        max_batch: Maximum texts per API call (default 512)
        internal_batch: Batch size for GPU processing (default 128)

    Returns:
        Numpy array of all embeddings concatenated
    """
    if not texts:
        return np.array([], dtype=np.float32)

    all_embeddings = []

    for i in range(0, len(texts), max_batch):
        batch = texts[i:i+max_batch]
        embeddings = await dgx_embed(batch, batch_size=internal_batch)

        if embeddings is None:
            return None  # Fail fast if DGX fails

        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings) if all_embeddings else None


async def dgx_embedding_stats() -> Optional[dict]:
    """Get DGX embedding service stats"""
    def _get_stats():
        try:
            req = Request(f"{DGX_EMBEDDING_URL}/stats")
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


def invalidate_dgx_embedding_cache():
    """Force re-check of DGX availability on next call"""
    global _dgx_available, _dgx_check_time
    _dgx_available = None
    _dgx_check_time = 0


# Synchronous wrappers for non-async code
def dgx_embed_sync(texts: List[str], batch_size: int = 128) -> Optional[np.ndarray]:
    """Synchronous version of dgx_embed"""
    return _do_embed(texts, batch_size)


def is_dgx_embedding_available_sync() -> bool:
    """Synchronous version of is_dgx_embedding_available"""
    return _is_dgx_reachable() and _check_health()
