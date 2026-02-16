"""
AI Embeddings Engine - Enterprise RAG System
Provides semantic search and accurate retrieval without hallucinations
Uses local sentence-transformers for privacy and zero API costs

OPTIMIZED FOR MCP: All NAS operations are non-blocking with timeouts
RECENCY BOOST: Newer content ranks higher (Project Evolution Tracker)
"""
import concurrent.futures
import hashlib
import json
import os
import pickle
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from config import EMBEDDING_DEVICE, EMBEDDING_DIM, EMBEDDING_MODEL
except ImportError:
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    EMBEDDING_DIM = 768
    EMBEDDING_DEVICE = "auto"

# Import Project Evolution Tracker for recency weighting
try:
    from project_evolution import ProjectEvolutionTracker, apply_recency_weighting
    EVOLUTION_TRACKING_ENABLED = True
except ImportError:
    EVOLUTION_TRACKING_ENABLED = False
    print("Warning: project_evolution not found. Recency weighting disabled.")

# Import Confidence Tracker for fact confidence weighting (Phase 3 - Brain Evolution)
try:
    from confidence_tracker import ConfidenceTracker
    CONFIDENCE_TRACKING_ENABLED = True
except ImportError:
    CONFIDENCE_TRACKING_ENABLED = False
    print("Warning: confidence_tracker not found. Confidence weighting disabled.")


class EmbeddingsEngine:
    """
    Vector embedding engine for semantic search and RAG.
    Uses sentence-transformers for local, privacy-preserving embeddings.

    IMPORTANT: Optimized for MCP usage - no blocking operations in constructor.
    AGENT 15: Enhanced with class-level model caching and GPU support.
    LOCAL CACHE: FAISS index cached on local SSD for instant loading.
    """

    # Class-level thread pool for async-safe file I/O
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    # Timeout for NAS operations (seconds)
    # NAS timeout for embeddings operations (independent of MCP wrapper timeout)
    NAS_TIMEOUT = 30
    # Local cache timeout - increased because id_mapping can be large
    # Pickle loads ~10x faster than JSON, but 44MB still takes a moment
    LOCAL_TIMEOUT = 5

    # AGENT 15: Class-level model cache (shared across all instances)
    _model_cache = None
    _model_lock_class = threading.Lock()

    # LOCAL CACHE: Store FAISS index on local SSD for instant loading
    # This avoids slow NAS reads on every search
    LOCAL_CACHE_PATH = Path(os.path.expanduser("~")) / ".claude" / "local_brain" / "faiss_cache"
    _index_cache = None  # Class-level index cache (shared across instances)
    _index_cache_lock = threading.Lock()

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.embeddings_path = self.base_path / "embeddings"
        self.vectors_path = self.embeddings_path / "vectors"
        self.chunks_path = self.embeddings_path / "chunks"
        self.index_path = self.embeddings_path / "indexes"

        # LAZY directory creation - don't touch NAS in constructor!
        self._directories_created = False

        # Model loading is controlled by ENABLE_EMBEDDINGS environment variable
        # Checked dynamically when model is accessed
        self._model = None
        self._model_loaded = False
        self._model_load_failed = False  # Will check environment variable when loading
        self._model_lock = threading.Lock()

        self.embedding_dim = EMBEDDING_DIM
        self.index = None
        self.id_mapping = []

        # Model disabled by default for MCP stability
        # Keyword search will be used instead (still works well!)

        # LAZY INITIALIZATION: Don't create trackers in constructor - they hit NAS
        # They will be initialized on first access via _ensure_trackers()
        self._evolution_tracker = None
        self._confidence_tracker = None
        self._trackers_initialized = False

    def _ensure_directories(self) -> bool:
        """Lazily create directories only when needed, with timeout protection"""
        if self._directories_created:
            return True

        try:
            def create_dirs():
                for path in [self.vectors_path, self.chunks_path, self.index_path]:
                    path.mkdir(parents=True, exist_ok=True)
                return True

            future = self._executor.submit(create_dirs)
            result = future.result(timeout=self.NAS_TIMEOUT)
            self._directories_created = result
            return result
        except concurrent.futures.TimeoutError:
            print(f"WARNING: Directory creation timed out after {self.NAS_TIMEOUT}s")
            return False
        except Exception as e:
            print(f"WARNING: Failed to create directories: {e}")
            return False

    def _is_nas_available(self, timeout: float = 2.0) -> bool:
        """
        Fast NAS reachability check using socket + filesystem test.
        Returns True if NAS is reachable AND filesystem responds.
        """
        import socket

        NAS_IP = os.environ.get("CEREBRO_NAS_IP", "")
        NAS_SMB_PORT = 445

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

        # Step 2: Quick filesystem check with threading timeout
        fs_result = [False]

        def check_filesystem():
            try:
                fs_result[0] = self.base_path.exists()
            except Exception:
                fs_result[0] = False

        thread = threading.Thread(target=check_filesystem, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return False

        return fs_result[0]

    def _ensure_trackers(self) -> bool:
        """
        Lazy-initialize trackers only when needed, with NAS guard.
        Returns True if trackers are available, False otherwise.
        """
        if self._trackers_initialized:
            return True

        # Check NAS first - don't hang on tracker initialization
        if not self._is_nas_available(timeout=2.0):
            print("[Embeddings] NAS unavailable, skipping tracker initialization")
            return False

        try:
            if EVOLUTION_TRACKING_ENABLED and self._evolution_tracker is None:
                try:
                    self._evolution_tracker = ProjectEvolutionTracker(base_path=str(self.base_path))
                except Exception as e:
                    print(f"Warning: Evolution tracker init failed: {e}")

            if CONFIDENCE_TRACKING_ENABLED and self._confidence_tracker is None:
                try:
                    self._confidence_tracker = ConfidenceTracker(base_path=str(self.base_path))
                except Exception as e:
                    print(f"Warning: Confidence tracker init failed: {e}")

            self._trackers_initialized = True
            return True
        except Exception as e:
            print(f"Warning: Tracker initialization failed: {e}")
            return False

    @property
    def model(self):
        """Lazy-load the model only when actually needed"""
        # Check environment variable dynamically
        if os.environ.get('ENABLE_EMBEDDINGS', '0') != '1':
            return None

        if self._model_load_failed:
            return None
        if self._model is not None:
            return self._model

        with self._model_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return self._model
            if self._model_load_failed:
                return None

            self._model = self._load_model_with_timeout()
            return self._model

    def _has_gpu(self) -> bool:
        """Check if GPU should be used. Respects CEREBRO_DEVICE env var."""
        if EMBEDDING_DEVICE == "cpu":
            return False
        if EMBEDDING_DEVICE == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[Embeddings] WARNING: CEREBRO_DEVICE=cuda but CUDA not available, falling back to CPU")
                    return False
                return True
            except ImportError:
                print("[Embeddings] WARNING: CEREBRO_DEVICE=cuda but torch not installed, falling back to CPU")
                return False
        # "auto" (default): detect
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model_with_timeout(self, timeout: int = 60):
        """
        Load the model with a timeout to prevent hanging.
        AGENT 15: Enhanced with class-level caching and GPU support.
        DGX-FIRST: Skip local model if DGX embedding service is available.
        """
        # AGENT 15: Check class-level cache first
        if EmbeddingsEngine._model_cache is not None:
            print("[Embeddings] Using cached model (instant load)")
            return EmbeddingsEngine._model_cache

        # DGX-FIRST: If DGX is available, skip local model entirely
        # DGX handles embedding via HTTP API, no local model needed
        try:
            from dgx_embedding_client import is_dgx_embedding_available_sync
            if is_dgx_embedding_available_sync():
                print("[Embeddings] DGX embedding service available — skipping local model load")
                self._model_load_failed = True  # Prevent retry
                return None
        except ImportError:
            pass
        except Exception:
            pass

        result = [None]
        error = [None]

        def load():
            try:
                # Double-check cache inside thread (thread-safe)
                with EmbeddingsEngine._model_lock_class:
                    if EmbeddingsEngine._model_cache is not None:
                        result[0] = EmbeddingsEngine._model_cache
                        return

                    # Disable progress bars and reduce verbosity
                    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

                    from sentence_transformers import SentenceTransformer

                    print("[Embeddings] Loading model (one-time)...")

                    # AGENT 15: Use GPU if available, otherwise CPU
                    device = 'cuda' if self._has_gpu() else 'cpu'

                    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

                    if device == 'cuda':
                        print("[Embeddings] Using GPU acceleration")
                    else:
                        print("[Embeddings] Using CPU")

                    # AGENT 15: Cache for future use (class-level)
                    EmbeddingsEngine._model_cache = model
                    result[0] = model

            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=load)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Model loading timed out
            self._model_load_failed = True
            print(f"WARNING: Model loading timed out after {timeout}s. Falling back to keyword search.")
            return None

        if error[0]:
            self._model_load_failed = True
            print(f"WARNING: Failed to load model: {error[0]}. Falling back to keyword search.")
            return None

        if result[0]:
            self._model_loaded = True
            print(f"[Embeddings] Model '{EMBEDDING_MODEL}' loaded on {device} ({self.embedding_dim}d)")
            return result[0]

        return None

    def is_model_available(self) -> bool:
        """Check if embedding model is available without blocking"""
        return self._model is not None and not self._model_load_failed

    def warmup_cache(self) -> bool:
        """
        Pre-load FAISS index AND embedding model into memory for instant first search.
        DGX-FIRST: Skips local model if DGX embedding service is available.
        """
        print("[Embeddings] Warming up cache...")

        # Step 1: Load FAISS index
        try:
            self.build_faiss_index(rebuild=False)
            if self.index is not None:
                print(f"[Embeddings] FAISS index warm: {self.index.ntotal} vectors ready")
            else:
                print("[Embeddings] No FAISS index available, will use keyword search")
        except Exception as e:
            print(f"[Embeddings] FAISS warmup failed: {e}")

        # Step 2: Pre-warm embedding model (respects DGX-first)
        dgx_available = False
        try:
            from dgx_embedding_client import is_dgx_embedding_available_sync
            dgx_available = is_dgx_embedding_available_sync()
        except ImportError:
            pass
        except Exception:
            pass

        if dgx_available:
            print("[Embeddings] DGX embedding service available — skipping local model warmup")
            return self.index is not None

        # No DGX: pre-load local model so first search is instant
        if os.environ.get('ENABLE_EMBEDDINGS', '0') == '1':
            print("[Embeddings] Pre-loading embedding model...")
            model = self.model  # Triggers lazy load via @property
            if model is not None:
                print("[Embeddings] Model pre-warmed and ready!")
            else:
                print("[Embeddings] Model load failed/disabled, keyword search available")

        return (self.index is not None) or (self._model is not None)

    def invalidate_cache(self):
        """Clear memory cache to force reload from disk on next search"""
        with self._index_cache_lock:
            EmbeddingsEngine._index_cache = None
        self.index = None
        self.id_mapping = []
        print("[Embeddings] Cache invalidated")

    def chunk_conversation(self, conversation: Dict) -> List[Dict]:
        """
        Intelligently chunk conversation into semantic units.
        Strategy: Message-based chunking with context windows
        """
        chunks = []
        messages = conversation.get("messages", [])
        conv_id = conversation.get("id", "unknown")

        # Strategy 1: Individual messages (for precise retrieval)
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if len(content) < 50:  # Skip very short messages
                continue

            # Skip short filler messages
            content_lower = content.lower().strip()
            filler_starts = ('hi ', 'hello', 'hey ', 'thanks', 'thank you', 'ok ', 'sure', 'got it', 'sounds good')
            if content_lower.startswith(filler_starts) and len(content) < 100:
                continue

            # Add context from previous message for coherence
            context = ""
            if i > 0:
                context = messages[i-1].get("content", "")[:200]

            chunk = {
                "chunk_id": self._generate_chunk_id(conv_id, i),
                "conversation_id": conv_id,
                "chunk_type": "message",
                "chunk_index": i,
                "role": msg.get("role"),
                "content": content,
                "context_before": context,
                "metadata": {
                    "timestamp": conversation.get("timestamp"),
                    "tags": conversation.get("metadata", {}).get("tags", []),
                    "topics": conversation.get("metadata", {}).get("topics", []),
                    "importance": conversation.get("metadata", {}).get("importance", "medium")
                }
            }
            chunks.append(chunk)

        # Strategy 2: Extracted facts (for fact retrieval)
        facts = conversation.get("extracted_data", {}).get("facts", [])
        for i, fact in enumerate(facts):
            chunk = {
                "chunk_id": self._generate_chunk_id(conv_id, f"fact-{i}"),
                "conversation_id": conv_id,
                "chunk_type": "fact",
                "chunk_index": i,
                "content": fact.get("content", ""),
                "metadata": {
                    "fact_type": fact.get("type"),
                    "confidence": fact.get("confidence"),
                    "timestamp": conversation.get("timestamp"),
                    "tags": conversation.get("metadata", {}).get("tags", [])
                }
            }
            chunks.append(chunk)

        # Strategy 3: File paths with context (for file location queries)
        file_paths = conversation.get("extracted_data", {}).get("file_paths", [])
        for i, path_info in enumerate(file_paths):
            content = f"{path_info['path']} - {path_info.get('purpose', 'unknown')}: {path_info.get('context', '')}"
            chunk = {
                "chunk_id": self._generate_chunk_id(conv_id, f"path-{i}"),
                "conversation_id": conv_id,
                "chunk_type": "file_path",
                "chunk_index": i,
                "content": content,
                "metadata": {
                    "path": path_info["path"],
                    "purpose": path_info.get("purpose"),
                    "timestamp": conversation.get("timestamp"),
                    "tags": conversation.get("metadata", {}).get("tags", [])
                }
            }
            chunks.append(chunk)

        # Strategy 4: Goals and preferences (for understanding user intent)
        goals = conversation.get("extracted_data", {}).get("goals_and_intentions", [])
        for i, goal in enumerate(goals):
            chunk = {
                "chunk_id": self._generate_chunk_id(conv_id, f"goal-{i}"),
                "conversation_id": conv_id,
                "chunk_type": "goal",
                "chunk_index": i,
                "content": goal.get("goal", ""),
                "metadata": {
                    "priority": goal.get("priority"),
                    "timestamp": conversation.get("timestamp"),
                    "tags": conversation.get("metadata", {}).get("tags", [])
                }
            }
            chunks.append(chunk)

        # Strategy 5: Problems and solutions (for troubleshooting queries)
        problems = conversation.get("extracted_data", {}).get("problems_solved", [])
        for i, problem in enumerate(problems):
            problem_text = problem.get('problem', '')
            solution_text = problem.get('solution', '')
            # Skip garbage chunks
            if len(problem_text) < 20:
                continue
            if solution_text and solution_text not in ("Not explicitly stated", "Not yet resolved") and len(solution_text) < 20:
                continue
            content = f"Problem: {problem_text} Solution: {solution_text}"
            chunk = {
                "chunk_id": self._generate_chunk_id(conv_id, f"problem-{i}"),
                "conversation_id": conv_id,
                "chunk_type": "problem_solution",
                "chunk_index": i,
                "content": content,
                "metadata": {
                    "status": problem.get("status"),
                    "timestamp": conversation.get("timestamp"),
                    "tags": conversation.get("metadata", {}).get("tags", [])
                }
            }
            chunks.append(chunk)

        # Strategy 6: Full conversation summary (for broad queries)
        summary = conversation.get("search_index", {}).get("summary", "")
        if summary:
            chunk = {
                "chunk_id": self._generate_chunk_id(conv_id, "summary"),
                "conversation_id": conv_id,
                "chunk_type": "summary",
                "chunk_index": 0,
                "content": summary,
                "metadata": {
                    "timestamp": conversation.get("timestamp"),
                    "tags": conversation.get("metadata", {}).get("tags", []),
                    "topics": conversation.get("metadata", {}).get("topics", []),
                    "importance": conversation.get("metadata", {}).get("importance")
                }
            }
            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, conv_id: str, chunk_identifier: Any) -> str:
        """Generate unique chunk ID"""
        combined = f"{conv_id}:{chunk_identifier}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for chunks"""
        if not self.model:
            # Return chunks without embeddings if model not available
            return chunks

        # Extract content for embedding
        texts = [chunk["content"] for chunk in chunks]

        # AGENT 15: Use batch embedding for better performance
        embeddings = self.embed_batch(texts, batch_size=32)

        # Attach embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
            chunk["embedding_model"] = EMBEDDING_MODEL
            chunk["embedding_dim"] = self.embedding_dim
            chunk["embedded_at"] = datetime.now().isoformat()
            embedded_chunks.append(chunk)

        return embedded_chunks

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts in batches for better performance (AGENT 15).

        Args:
            texts: List of texts to embed
            batch_size: Batch size (32 is optimal for most GPUs/CPUs)

        Returns:
            Numpy array of embeddings
        """
        # Strategy 1: DGX GPU embedding (preferred)
        try:
            from dgx_embedding_client import dgx_embed_sync, is_dgx_embedding_available_sync
            if is_dgx_embedding_available_sync():
                result = dgx_embed_sync(texts, batch_size=batch_size)
                if result is not None:
                    print(f"[Embeddings] Batch of {len(texts)} embedded via DGX")
                    return result.astype(np.float32)
        except ImportError:
            pass
        except Exception as e:
            print(f"[Embeddings] DGX batch embedding failed: {e}")

        # Strategy 2: Local model fallback
        if not self.model:
            # Return zero embeddings if no embedding method available
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        # Process in batches via local model
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False  # Faster
            )
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def save_chunks(self, chunks: List[Dict], conversation_id: str):
        """Save chunks to disk with timeout protection"""
        if not self._ensure_directories():
            print("WARNING: Could not create directories, skipping chunk save")
            return

        def do_save():
            chunk_file = self.chunks_path / f"{conversation_id}.jsonl"
            with open(chunk_file, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk) + "\n")

        try:
            future = self._executor.submit(do_save)
            future.result(timeout=self.NAS_TIMEOUT)
        except concurrent.futures.TimeoutError:
            print(f"WARNING: Saving chunks timed out after {self.NAS_TIMEOUT}s")
        except Exception as e:
            print(f"WARNING: Failed to save chunks: {e}")

    def save_incremental_chunks(self, chunks: List[Dict], conversation_id: str,
                               previous_chunk_count: int = 0) -> int:
        """
        Save only NEW chunks without re-embedding old ones.
        INCREMENTAL OPTIMIZATION (Agent 7)

        Args:
            chunks: List of all chunks (old + new)
            conversation_id: ID of conversation
            previous_chunk_count: Number of chunks already embedded

        Returns:
            Number of NEW chunks added
        """
        # Extract only NEW chunks
        new_chunks = chunks[previous_chunk_count:]

        if not new_chunks:
            return 0  # Nothing new to add

        if not self._ensure_directories():
            print("WARNING: Could not create directories, skipping incremental save")
            return 0

        def do_save():
            chunk_file = self.chunks_path / f"{conversation_id}.jsonl"

            # If file exists, append; otherwise create
            if chunk_file.exists():
                # Append new chunks
                with open(chunk_file, "a", encoding="utf-8") as f:
                    for chunk in new_chunks:
                        f.write(json.dumps(chunk) + "\n")
            else:
                # Create new file with new chunks
                with open(chunk_file, "w", encoding="utf-8") as f:
                    for chunk in new_chunks:
                        f.write(json.dumps(chunk) + "\n")

        try:
            future = self._executor.submit(do_save)
            future.result(timeout=self.NAS_TIMEOUT)

            # Generate embeddings ONLY for new chunks
            if self.model:
                new_embedded = self.embed_chunks(new_chunks)

                # INCREMENTAL INDEX UPDATE
                self._append_to_faiss_index(new_embedded, conversation_id)

            return len(new_chunks)

        except concurrent.futures.TimeoutError:
            print(f"WARNING: Incremental save timed out after {self.NAS_TIMEOUT}s")
            return 0
        except Exception as e:
            print(f"WARNING: Failed to save incremental chunks: {e}")
            return 0

    def _append_to_faiss_index(self, new_chunks: List[Dict], conversation_id: str):
        """
        Append new vectors to existing FAISS index without rebuilding.
        INCREMENTAL OPTIMIZATION (Agent 7)

        Args:
            new_chunks: New chunks with embeddings
            conversation_id: ID for tracking
        """
        if not new_chunks or "embedding" not in new_chunks[0]:
            return  # No embeddings to add

        try:
            import faiss
        except ImportError:
            print("WARNING: faiss not installed, skipping index update")
            return

        if not self._ensure_directories():
            return

        def do_append():
            index_file = self.index_path / "faiss_index.bin"
            mapping_file = self.index_path / "id_mapping.json"

            # Load existing index or create new
            if index_file.exists():
                index = faiss.read_index(str(index_file))
                with open(mapping_file, "r", encoding="utf-8") as f:
                    id_mapping = json.load(f)
            else:
                # Create new index (first time)
                index = faiss.IndexFlatIP(self.embedding_dim)
                id_mapping = []

            # Get current index size (for mapping)
            current_size = index.ntotal

            # Extract embeddings and ensure float32 type
            new_vectors = np.array(
                [chunk["embedding"] for chunk in new_chunks],
                dtype=np.float32
            )

            # Ensure contiguous memory layout
            if not new_vectors.flags['C_CONTIGUOUS']:
                new_vectors = np.ascontiguousarray(new_vectors)

            # Normalize for cosine similarity
            faiss.normalize_L2(new_vectors)

            # ADD new vectors to index (incremental!)
            index.add(new_vectors)

            # Update ID mapping - include content to avoid NAS reads during search
            for i, chunk in enumerate(new_chunks):
                id_mapping.append({
                    'index_id': current_size + i,
                    'conversation_id': conversation_id,
                    'chunk_id': chunk.get('chunk_id'),
                    'chunk_type': chunk.get('chunk_type', 'message'),
                    'content': chunk.get('content', ''),  # Store content for fast search
                    'context': chunk.get('context_before', ''),  # Store context too
                    'metadata': chunk.get('metadata', {})
                })

            # Save updated index and mapping
            faiss.write_index(index, str(index_file))
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(id_mapping, f, indent=2)

            # Update in-memory references
            self.index = index
            self.id_mapping = id_mapping

            return index.ntotal  # Total vectors now in index

        try:
            future = self._executor.submit(do_append)
            total = future.result(timeout=self.NAS_TIMEOUT)
            print(f"[Incremental] Added {len(new_chunks)} chunks. Total index size: {total}")
        except concurrent.futures.TimeoutError:
            print(f"WARNING: Index append timed out after {self.NAS_TIMEOUT}s")
        except Exception as e:
            print(f"WARNING: Failed to append to FAISS index: {e}")

    def save_vectors(self, chunks: List[Dict], conversation_id: str):
        """Save vectors separately for efficient loading - with timeout protection"""
        # Only save if chunks have embeddings
        if not chunks or "embedding" not in chunks[0]:
            return

        if not self._ensure_directories():
            print("WARNING: Could not create directories, skipping vector save")
            return

        def do_save():
            vectors_file = self.vectors_path / f"{conversation_id}.npy"
            # Extract embeddings and ensure float32 type (FAISS requirement)
            embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
            # Save as numpy array
            np.save(vectors_file, embeddings)

            # Save metadata separately
            metadata_file = self.vectors_path / f"{conversation_id}_metadata.json"
            metadata = [
                {
                    "chunk_id": chunk["chunk_id"],
                    "chunk_type": chunk["chunk_type"],
                    "metadata": chunk["metadata"]
                }
                for chunk in chunks
            ]
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        try:
            future = self._executor.submit(do_save)
            future.result(timeout=self.NAS_TIMEOUT)
        except concurrent.futures.TimeoutError:
            print(f"WARNING: Saving vectors timed out after {self.NAS_TIMEOUT}s")
        except Exception as e:
            print(f"WARNING: Failed to save vectors: {e}")

    def _ensure_local_cache_dir(self):
        """Ensure local cache directory exists"""
        try:
            self.LOCAL_CACHE_PATH.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"WARNING: Could not create local cache dir: {e}")
            return False

    def _get_file_mtime(self, path: Path, timeout: float = 3.0) -> float:
        """Get file modification time with timeout (NAS paths can hang).
        Returns 0 if file doesn't exist or operation times out."""
        result = [0.0]

        def _stat():
            try:
                result[0] = path.stat().st_mtime if path.exists() else 0.0
            except Exception:
                result[0] = 0.0

        thread = threading.Thread(target=_stat, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            print(f"WARNING: stat() timed out for {path} after {timeout}s")
            return 0.0
        return result[0]

    def _load_from_local_cache(self):
        """Try to load FAISS index from local SSD cache (fast)"""
        try:
            import faiss
        except ImportError:
            return None, None

        local_index = self.LOCAL_CACHE_PATH / "faiss_index.bin"
        local_mapping_pkl = self.LOCAL_CACHE_PATH / "id_mapping.pkl"
        local_mapping_json = self.LOCAL_CACHE_PATH / "id_mapping.json"

        if not local_index.exists():
            return None, None

        try:
            def load_local():
                idx = faiss.read_index(str(local_index))
                # Prefer pickle (10x faster than JSON for large files)
                if local_mapping_pkl.exists():
                    with open(local_mapping_pkl, "rb") as f:
                        mapping = pickle.load(f)
                elif local_mapping_json.exists():
                    # Fallback to JSON, but convert to pickle for next time
                    with open(local_mapping_json, "r", encoding="utf-8") as f:
                        mapping = json.load(f)
                    # Save as pickle for faster future loads
                    with open(local_mapping_pkl, "wb") as f:
                        pickle.dump(mapping, f)
                    print("[Embeddings] Converted JSON mapping to pickle for faster loading")
                else:
                    return None, None
                return idx, mapping

            future = self._executor.submit(load_local)
            idx, mapping = future.result(timeout=self.LOCAL_TIMEOUT)
            print(f"[Embeddings] Loaded from LOCAL CACHE ({idx.ntotal} vectors) - fast!")
            return idx, mapping
        except Exception as e:
            print(f"[Embeddings] Local cache load failed: {e}")
            return None, None

    def _save_to_local_cache(self, index, id_mapping):
        """Save FAISS index to local SSD cache for fast future loads"""
        try:
            import faiss
        except ImportError:
            return False

        if not self._ensure_local_cache_dir():
            return False

        local_index = self.LOCAL_CACHE_PATH / "faiss_index.bin"
        local_mapping_pkl = self.LOCAL_CACHE_PATH / "id_mapping.pkl"

        try:
            def save_local():
                faiss.write_index(index, str(local_index))
                # Use pickle instead of JSON - 10x faster loading
                with open(local_mapping_pkl, "wb") as f:
                    pickle.dump(id_mapping, f)
                return True

            future = self._executor.submit(save_local)
            future.result(timeout=self.LOCAL_TIMEOUT)
            print("[Embeddings] Saved to local cache for instant future loads")
            return True
        except Exception as e:
            print(f"WARNING: Failed to save to local cache: {e}")
            return False

    def _is_local_cache_stale(self) -> bool:
        """Check if local cache is older than NAS version.
        Returns False (not stale) if NAS is unreachable — use local cache as-is."""
        # Guard: don't hang trying to stat NAS files if NAS is down
        if not self._is_nas_available(timeout=2.0):
            return False

        nas_index = self.index_path / "faiss_index.bin"
        local_index = self.LOCAL_CACHE_PATH / "faiss_index.bin"

        nas_mtime = self._get_file_mtime(nas_index, timeout=3.0)
        local_mtime = self._get_file_mtime(local_index, timeout=2.0)

        # If NAS is newer, local cache is stale
        return nas_mtime > local_mtime

    def build_faiss_index(self, rebuild: bool = False):
        """
        Build FAISS index for fast similarity search.

        LOCAL CACHE STRATEGY:
        1. Check class-level memory cache first (instant)
        2. Check local SSD cache (fast, ~100ms)
        3. Fall back to NAS (slow, can timeout)
        4. After loading from NAS, update local cache
        """
        try:
            import faiss
        except ImportError:
            print("WARNING: faiss not installed. Using keyword search only.")
            return None

        # STEP 1: Check class-level memory cache (shared across instances)
        with self._index_cache_lock:
            if EmbeddingsEngine._index_cache is not None and not rebuild:
                self.index, self.id_mapping = EmbeddingsEngine._index_cache
                print(f"[Embeddings] Using MEMORY CACHE ({self.index.ntotal} vectors) - instant!")
                return self.index

        index_file = self.index_path / "faiss_index.bin"
        mapping_file = self.index_path / "id_mapping.json"

        if not rebuild:
            # STEP 2: Try local SSD cache first (fast)
            local_idx, local_mapping = self._load_from_local_cache()
            if local_idx is not None:
                # Check if we should update from NAS (background, non-blocking)
                if self._is_local_cache_stale():
                    print("[Embeddings] Local cache is stale, will sync from NAS in background")
                    # Use stale cache now, update later (fast user experience)
                    self.index = local_idx
                    self.id_mapping = local_mapping
                    # Update class-level cache
                    with self._index_cache_lock:
                        EmbeddingsEngine._index_cache = (self.index, self.id_mapping)
                    # TODO: Background sync from NAS
                    return self.index
                else:
                    self.index = local_idx
                    self.id_mapping = local_mapping
                    # Update class-level cache
                    with self._index_cache_lock:
                        EmbeddingsEngine._index_cache = (self.index, self.id_mapping)
                    return self.index

            # STEP 3: Fall back to NAS (slow but authoritative)
            # Guard: skip NAS load entirely if NAS is unreachable
            if not self._is_nas_available(timeout=2.0):
                print("[Embeddings] NAS unreachable, skipping NAS index load")
                return None
            if index_file.exists():
                try:
                    def load_index():
                        idx = faiss.read_index(str(index_file))
                        with open(mapping_file, "r", encoding="utf-8") as f:
                            mapping = json.load(f)
                        return idx, mapping

                    future = self._executor.submit(load_index)
                    self.index, self.id_mapping = future.result(timeout=self.NAS_TIMEOUT)
                    print(f"[Embeddings] FAISS index loaded from NAS ({self.index.ntotal} vectors)")

                    # STEP 4: Save to local cache for next time
                    self._save_to_local_cache(self.index, self.id_mapping)

                    # Update class-level cache
                    with self._index_cache_lock:
                        EmbeddingsEngine._index_cache = (self.index, self.id_mapping)

                    return self.index
                except concurrent.futures.TimeoutError:
                    print(f"WARNING: FAISS index loading timed out after {self.NAS_TIMEOUT}s. Falling back to keyword search.")
                    return None
                except Exception as e:
                    print(f"WARNING: Failed to load index: {e}")
                    # Fall through to rebuild

        # Build new index
        all_vectors = []
        id_mapping = []

        # Load all vector files with progress tracking
        import time as _time
        _rebuild_start = _time.monotonic()
        vector_files = [f for f in self.vectors_path.glob("*.npy") if "_metadata" not in f.name]
        total_files = len(vector_files)
        loaded = 0
        skipped = 0

        print(f"[Embeddings] Rebuild: loading {total_files} vector files from NAS...")

        for vector_file in vector_files:
            conv_id = vector_file.stem
            try:
                vectors = np.load(vector_file)
            except Exception as e:
                print(f"WARNING: Failed to load vectors from {vector_file}: {e}")
                skipped += 1
                continue

            # Load metadata
            metadata_file = self.vectors_path / f"{conv_id}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                for i, meta in enumerate(metadata):
                    id_mapping.append({
                        "conversation_id": conv_id,
                        "chunk_id": meta["chunk_id"],
                        "chunk_type": meta["chunk_type"],
                        "metadata": meta["metadata"]
                    })

                all_vectors.append(vectors)

            loaded += 1
            if loaded % 200 == 0:
                elapsed = _time.monotonic() - _rebuild_start
                print(f"[Embeddings] Rebuild progress: {loaded}/{total_files} files loaded ({elapsed:.1f}s)")

        load_elapsed = _time.monotonic() - _rebuild_start
        print(f"[Embeddings] Loaded {loaded} files ({skipped} skipped) in {load_elapsed:.1f}s")

        if not all_vectors:
            return None

        # Combine all vectors and ensure float32 type (FAISS requirement)
        all_vectors = np.vstack(all_vectors).astype(np.float32)

        # Make sure array is contiguous in memory
        if not all_vectors.flags['C_CONTIGUOUS']:
            all_vectors = np.ascontiguousarray(all_vectors)

        # Create FAISS index
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(all_vectors)

        # Use IndexIVFFlat for large datasets (faster search)
        if len(all_vectors) > 1000:
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            n_clusters = min(int(np.sqrt(len(all_vectors))), 100)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters)
            self.index.train(all_vectors)
            self.index.add(all_vectors)
            self.index.nprobe = 10  # Search 10 clusters
        else:
            # Use flat index for small datasets (exact search)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(all_vectors)

        # Save index to NAS
        faiss.write_index(self.index, str(index_file))

        # Save mapping to NAS
        self.id_mapping = id_mapping
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(id_mapping, f, indent=2)

        # Also save to local cache for fast future loads
        self._save_to_local_cache(self.index, self.id_mapping)

        # Update class-level memory cache
        with self._index_cache_lock:
            EmbeddingsEngine._index_cache = (self.index, self.id_mapping)

        print(f"[Embeddings] Index rebuilt: {self.index.ntotal} vectors, saved to NAS + local cache")
        return self.index

    def validate_and_rebuild_if_needed(self) -> Dict[str, Any]:
        """
        Validate index integrity and rebuild if out of sync.
        Call this at startup to ensure index matches chunks.

        Returns dict with status and action taken.
        """
        result = {
            "status": "unknown",
            "action": None,
            "chunk_count": 0,
            "index_size": 0,
            "details": ""
        }

        try:
            # Count total chunks across all chunk files WITH TIMEOUT
            # See failure_memory fail_001: NAS operations can hang
            def count_chunks():
                count = 0
                if self.chunks_path.exists():
                    for chunk_file in self.chunks_path.glob("*.jsonl"):
                        try:
                            with open(chunk_file, "r", encoding="utf-8") as f:
                                count += sum(1 for _ in f)
                        except Exception:
                            continue
                return count

            try:
                future = self._executor.submit(count_chunks)
                total_chunks = future.result(timeout=self.NAS_TIMEOUT)
            except concurrent.futures.TimeoutError:
                print(f"WARNING: Chunk counting timed out after {self.NAS_TIMEOUT}s, skipping validation")
                result["status"] = "timeout"
                result["details"] = "Chunk counting timed out, skipping validation"
                return result

            result["chunk_count"] = total_chunks

            # Check index size
            index_file = self.index_path / "faiss_index.bin"
            mapping_file = self.index_path / "id_mapping.json"

            index_size = 0
            if index_file.exists() and mapping_file.exists():
                try:
                    with open(mapping_file, "r", encoding="utf-8") as f:
                        id_mapping = json.load(f)
                    index_size = len(id_mapping)
                except Exception:
                    index_size = 0

            result["index_size"] = index_size

            # Decision logic
            if total_chunks == 0:
                result["status"] = "empty"
                result["details"] = "No chunks found, nothing to index"
                return result

            if index_size == 0:
                # Index missing - rebuild
                result["status"] = "missing"
                result["action"] = "rebuild"
                result["details"] = f"Index missing, rebuilding from {total_chunks} chunks"
                print(f"[Embeddings] {result['details']}")
                self.build_faiss_index(rebuild=True)
                return result

            # Check sync: allow 10% tolerance for minor discrepancies
            sync_ratio = index_size / total_chunks if total_chunks > 0 else 0

            if sync_ratio < 0.9:  # Index has less than 90% of chunks
                result["status"] = "out_of_sync"
                result["action"] = "rebuild"
                result["details"] = f"Index out of sync ({index_size}/{total_chunks} = {sync_ratio:.1%}), rebuilding"
                print(f"[Embeddings] {result['details']}")
                self.build_faiss_index(rebuild=True)
                return result

            # Index is healthy
            result["status"] = "healthy"
            result["details"] = f"Index in sync ({index_size}/{total_chunks} chunks)"
            return result

        except Exception as e:
            result["status"] = "error"
            result["details"] = f"Validation failed: {str(e)}"
            return result

    def semantic_search(self, query: str, top_k: int = 10,
                       filters: Optional[Dict] = None, boost_by_importance: bool = False,
                       boost_by_recency: bool = True, recency_decay_days: int = 30,
                       conversation_type: str = None) -> List[Dict]:
        """
        Perform semantic search on the vector database.
        Falls back to keyword search if model not available.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters for chunk_type, tags, etc.
            boost_by_importance: If True, boost results by conversation quality/importance
            boost_by_recency: If True, boost newer content (decay older results)
            recency_decay_days: Half-life for recency decay (default 30 days)
            conversation_type: Filter by type ("meta_system" or "user_work") - Enhancement 2
        """
        # Try to get query embedding via DGX first, then local model, then keyword fallback
        query_embedding = None

        # Strategy 1: DGX GPU embedding (preferred — no local model needed)
        try:
            from dgx_embedding_client import dgx_embed_sync, is_dgx_embedding_available_sync
            if is_dgx_embedding_available_sync():
                dgx_result = dgx_embed_sync([query])
                if dgx_result is not None and len(dgx_result) > 0:
                    query_embedding = dgx_result.astype(np.float32)
                    print(f"[Embeddings] Query embedded via DGX ({query_embedding.shape[1]}d)")
        except ImportError:
            pass
        except Exception as e:
            print(f"[Embeddings] DGX query embedding failed: {e}")

        # Strategy 2: Local model fallback (only if DGX failed)
        if query_embedding is None and self.model:
            query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
            print("[Embeddings] Query embedded via local model")

        # Strategy 3: Keyword search fallback (no embedding available)
        if query_embedding is None:
            print("[Embeddings] No embedding available, using keyword search")
            return self._keyword_search(query, top_k=top_k)

        if not self.index:
            # Try to load index
            self.build_faiss_index(rebuild=False)
            if not self.index:
                # Fall back to keyword search
                return self._keyword_search(query, top_k=top_k)

        try:
            import faiss
        except ImportError:
            return self._keyword_search(query, top_k=top_k)
        if not query_embedding.flags['C_CONTIGUOUS']:
            query_embedding = np.ascontiguousarray(query_embedding)
        faiss.normalize_L2(query_embedding)

        # Search
        k = min(top_k * 2, self.index.ntotal)  # Get more for filtering
        distances, indices = self.index.search(query_embedding, k)

        # Retrieve results with metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue

            # Get metadata
            if idx >= len(self.id_mapping):
                continue
            result_meta = self.id_mapping[idx]

            # Apply filters
            if filters:
                if "chunk_type" in filters and result_meta["chunk_type"] != filters["chunk_type"]:
                    continue
                if "tags" in filters:
                    meta_tags = result_meta["metadata"].get("tags", [])
                    if not any(tag in meta_tags for tag in filters["tags"]):
                        continue

            # Enhancement 2: Filter by conversation_type
            if conversation_type:
                # Load conversation to check type
                conv_file = self.base_path / "conversations" / f"{result_meta['conversation_id']}.json"
                if conv_file.exists():
                    try:
                        with open(conv_file, 'r', encoding='utf-8') as f:
                            conv_data = json.load(f)
                        conv_type = conv_data.get('metadata', {}).get('conversation_type', 'user_work')
                        if conv_type != conversation_type:
                            continue
                    except Exception:
                        pass  # If we can't check, include the result

            # OPTIMIZATION: Check if content is cached in id_mapping first (fast path)
            # This avoids slow NAS file reads during search
            similarity_score = float(dist)
            confidence = self._score_to_confidence(similarity_score)

            # Fast path: content stored in id_mapping (new format after re-indexing)
            if "content" in result_meta:
                results.append({
                    "similarity": similarity_score,
                    "similarity_score": similarity_score,
                    "score": similarity_score,
                    "confidence": confidence,
                    "conversation_id": result_meta["conversation_id"],
                    "chunk_type": result_meta["chunk_type"],
                    "content": result_meta["content"],
                    "metadata": result_meta["metadata"],
                    "context": result_meta.get("context", "")
                })
            else:
                # Slow path: read from chunk file (only for top results, with timeout)
                # Skip if we already have enough results
                if len(results) >= top_k:
                    # Just return metadata, skip content for lower-ranked results
                    results.append({
                        "similarity": similarity_score,
                        "similarity_score": similarity_score,
                        "score": similarity_score,
                        "confidence": confidence,
                        "conversation_id": result_meta["conversation_id"],
                        "chunk_type": result_meta["chunk_type"],
                        "content": f"[Content in conversation {result_meta['conversation_id']}]",
                        "metadata": result_meta["metadata"],
                        "context": ""
                    })
                else:
                    # Try to read content, but with timeout protection
                    chunk_file = self.chunks_path / f"{result_meta['conversation_id']}.jsonl"
                    content_found = False
                    if chunk_file.exists():
                        try:
                            with open(chunk_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    try:
                                        chunk = json.loads(line)
                                        if chunk.get("chunk_id") == result_meta["chunk_id"]:
                                            results.append({
                                                "similarity": similarity_score,
                                                "similarity_score": similarity_score,
                                                "score": similarity_score,
                                                "confidence": confidence,
                                                "conversation_id": result_meta["conversation_id"],
                                                "chunk_type": result_meta["chunk_type"],
                                                "content": chunk.get("content", ""),
                                                "metadata": result_meta["metadata"],
                                                "context": chunk.get("context_before", "")
                                            })
                                            content_found = True
                                            break
                                    except json.JSONDecodeError:
                                        continue
                        except Exception:
                            pass  # Continue without content on error

                    if not content_found:
                        # Fallback if content couldn't be loaded
                        results.append({
                            "similarity": similarity_score,
                            "similarity_score": similarity_score,
                            "score": similarity_score,
                            "confidence": confidence,
                            "conversation_id": result_meta["conversation_id"],
                            "chunk_type": result_meta["chunk_type"],
                            "content": f"[Content in conversation {result_meta['conversation_id']}]",
                            "metadata": result_meta["metadata"],
                            "context": ""
                        })

            if len(results) >= top_k * 2:  # Get extra for importance boosting
                break

        # BOOST BY IMPORTANCE (Agent 11)
        if boost_by_importance:
            try:
                from quality_scorer import QualityScorer
                scorer = QualityScorer(base_path=str(self.base_path))

                for result in results:
                    conv_id = result.get('conversation_id')
                    if conv_id:
                        # Load conversation
                        conv_file = self.base_path / "conversations" / f"{conv_id}.json"
                        if conv_file.exists():
                            try:
                                with open(conv_file, 'r', encoding='utf-8') as f:
                                    conv = json.load(f)

                                quality = scorer.score_conversation(conv)
                                multiplier = quality['search_boost_multiplier']

                                # Apply boost
                                result['score'] = result['similarity'] * multiplier
                                result['importance_boost'] = multiplier
                                result['importance'] = quality['importance']
                                result['quality_score'] = quality['overall_score']
                            except Exception:
                                # If error loading conversation, keep original score
                                result['score'] = result['similarity']
                                result['importance_boost'] = 1.0

                # Re-sort by boosted scores
                results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            except ImportError:
                # QualityScorer not available, keep original scores
                pass
            except Exception as e:
                print(f"WARNING: Importance boosting failed: {e}")

        # BOOST BY RECENCY (Project Evolution Tracker)
        # Newer content ranks higher, superseded content is penalized
        # Lazy-init trackers only when needed (avoids constructor hangs)
        if boost_by_recency and EVOLUTION_TRACKING_ENABLED:
            self._ensure_trackers()  # Lazy init with NAS guard
            if self._evolution_tracker:
                try:
                    results = apply_recency_weighting(
                        results,
                        tracker=self._evolution_tracker,
                        query=query,
                        decay_days=recency_decay_days
                    )
                except Exception as e:
                    print(f"WARNING: Recency boosting failed: {e}")

        # BOOST BY CONFIDENCE (Phase 3 - Brain Evolution)
        # Facts with higher confidence rank higher in results
        if CONFIDENCE_TRACKING_ENABLED:
            self._ensure_trackers()  # Lazy init with NAS guard
            if self._confidence_tracker:
                try:
                    for result in results:
                        # Try to find fact_id in the result metadata or content
                        fact_id = result.get('metadata', {}).get('fact_id')

                        if fact_id:
                            # Apply confidence weighting
                            fact_confidence = self._confidence_tracker.get_confidence(fact_id)
                            if fact_confidence is not None:
                                # Combined score = current score * fact confidence
                                result['fact_confidence'] = fact_confidence
                                result['score'] = result.get('score', result.get('similarity', 0.5)) * fact_confidence
                            else:
                                result['fact_confidence'] = 0.60  # Default for unknown facts
                        else:
                            result['fact_confidence'] = None  # Not a tracked fact

                    # Re-sort by confidence-weighted scores
                    results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
                except Exception as e:
                    print(f"WARNING: Confidence boosting failed: {e}")

        return results[:top_k]

    def hybrid_search(self, query: str, top_k: int = 10,
                     alpha: float = 0.7, boost_by_recency: bool = True,
                     recency_decay_days: int = 30,
                     conversation_type: str = None) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword search.
        alpha: weight for semantic search (1.0 = pure semantic, 0.0 = pure keyword)
        boost_by_recency: If True, boost newer content (decay older results)
        recency_decay_days: Half-life for recency decay (default 30 days)
        conversation_type: Filter by type ("meta_system" or "user_work") - Enhancement 2
        Falls back to keyword-only if model not available.
        """
        # If model not available, use keyword search only
        if not self.model:
            results = self._keyword_search(query, top_k=top_k)
            # Apply recency to keyword-only results too
            # Lazy-init trackers only when needed (avoids constructor hangs)
            if boost_by_recency and EVOLUTION_TRACKING_ENABLED:
                self._ensure_trackers()  # Lazy init with NAS guard
                if self._evolution_tracker:
                    try:
                        results = apply_recency_weighting(
                            results, tracker=self._evolution_tracker,
                            query=query, decay_days=recency_decay_days
                        )
                    except Exception:
                        pass
            # Apply confidence weighting to keyword-only results
            if CONFIDENCE_TRACKING_ENABLED:
                self._ensure_trackers()  # Lazy init with NAS guard
                if self._confidence_tracker:
                    try:
                        results = self._apply_confidence_weighting(results)
                    except Exception:
                        pass
            return results

        # Semantic search results (already has recency if enabled)
        semantic_results = self.semantic_search(
            query, top_k=top_k, boost_by_recency=boost_by_recency,
            recency_decay_days=recency_decay_days,
            conversation_type=conversation_type
        )

        # Keyword search results
        keyword_results = self._keyword_search(query, top_k=top_k)

        # Combine and re-rank
        combined = {}

        for i, result in enumerate(semantic_results):
            chunk_id = result.get("content", "")[:100]  # Use content snippet as key
            combined[chunk_id] = {
                **result,
                "score": alpha * result.get("score", result.get("similarity", 0)) + (1 - alpha) * (1.0 - i / max(len(semantic_results), 1))
            }

        for i, result in enumerate(keyword_results):
            chunk_id = result.get("content", "")[:100]
            if chunk_id in combined:
                combined[chunk_id]["score"] += (1 - alpha) * (1.0 - i / max(len(keyword_results), 1))
            else:
                combined[chunk_id] = {
                    **result,
                    "score": (1 - alpha) * (1.0 - i / max(len(keyword_results), 1))
                }

        # Sort by combined score
        results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _score_to_confidence(self, similarity: float) -> str:
        """
        Convert similarity score to confidence label.
        AGENT 13: Make confidence scores human-readable.
        """
        if similarity >= 0.85:
            return "HIGH"
        elif similarity >= 0.70:
            return "MEDIUM"
        else:
            return "LOW"

    def _apply_confidence_weighting(self, results: List[Dict]) -> List[Dict]:
        """
        Apply fact confidence weighting to search results.
        Phase 3 - Brain Evolution: Facts with higher confidence rank higher.

        Args:
            results: List of search result dicts

        Returns:
            Results with confidence weighting applied and re-sorted
        """
        if not self._confidence_tracker:
            return results

        for result in results:
            # Try to find fact_id in the result metadata
            fact_id = result.get('metadata', {}).get('fact_id')

            if fact_id:
                # Apply confidence weighting
                fact_confidence = self._confidence_tracker.get_confidence(fact_id)
                if fact_confidence is not None:
                    result['fact_confidence'] = fact_confidence
                    result['score'] = result.get('score', result.get('similarity', 0.5)) * fact_confidence
                else:
                    result['fact_confidence'] = 0.60  # Default for unknown facts
            else:
                result['fact_confidence'] = None  # Not a tracked fact

        # Re-sort by confidence-weighted scores
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)

    def _keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Fast keyword search using SQLite FTS5 index.
        Falls back to slow file scan if index not available.
        """
        try:
            # Try fast SQLite-based search first
            from keyword_index import get_keyword_index
            idx = get_keyword_index(self.chunks_path)

            # Check if index has data
            if idx.get_indexed_count() > 0:
                results = idx.search(query, top_k=top_k)
                if results is not None:  # Fixed: empty list [] is valid, don't fallback to slow search
                    # Ensure all required fields are present
                    for r in results:
                        if "similarity_score" not in r:
                            r["similarity_score"] = r.get("similarity", 0.5)
                        if "confidence" not in r:
                            r["confidence"] = self._score_to_confidence(r.get("similarity", 0.5))
                        if "context" not in r:
                            r["context"] = ""
                    return results

        except ImportError:
            print("[Embeddings] keyword_index not available, using slow fallback")
        except Exception as e:
            print(f"[Embeddings] SQLite search failed: {e}, using slow fallback")

        # Fallback to slow file-based search (with timeout)
        return self._keyword_search_slow(query, top_k)

    def _keyword_search_slow(self, query: str, top_k: int = 10) -> List[Dict]:
        """Slow fallback keyword search - reads all files from NAS"""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        def do_search():
            results = []
            try:
                if not self.chunks_path.exists():
                    return results

                chunk_files = list(self.chunks_path.glob("*.jsonl"))
                for chunk_file in chunk_files:
                    try:
                        with open(chunk_file, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    chunk = json.loads(line)
                                    content_lower = chunk.get("content", "").lower()
                                    matches = sum(1 for word in query_words if word in content_lower)
                                    if matches > 0:
                                        similarity = matches / len(query_words)
                                        results.append({
                                            "similarity": similarity,
                                            "similarity_score": similarity,
                                            "score": similarity,
                                            "confidence": self._score_to_confidence(similarity),
                                            "conversation_id": chunk.get("conversation_id", "unknown"),
                                            "chunk_type": chunk.get("chunk_type", "unknown"),
                                            "content": chunk.get("content", ""),
                                            "metadata": chunk.get("metadata", {}),
                                            "context": chunk.get("context_before", "")
                                        })
                                except json.JSONDecodeError:
                                    continue
                    except Exception:
                        continue
            except Exception as e:
                print(f"WARNING: Error in keyword search: {e}")
            return results

        try:
            future = self._executor.submit(do_search)
            results = future.result(timeout=self.NAS_TIMEOUT)
            results = sorted(results, key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        except concurrent.futures.TimeoutError:
            print(f"WARNING: Keyword search timed out after {self.NAS_TIMEOUT}s")
            return []
        except Exception as e:
            print(f"WARNING: Keyword search failed: {e}")
            return []
            return []

    def process_conversation(self, conversation: Dict) -> str:
        """
        Complete pipeline: chunk, embed, and index a conversation.
        Call this for every new conversation.
        """
        conv_id = conversation.get("id")

        # Step 1: Chunk
        chunks = self.chunk_conversation(conversation)

        # Step 2: Embed (if model available)
        if self.model:
            embedded_chunks = self.embed_chunks(chunks)
        else:
            embedded_chunks = chunks

        # Step 3: Save chunks and vectors
        self.save_chunks(embedded_chunks, conv_id)
        if self.model and "embedding" in (embedded_chunks[0] if embedded_chunks else {}):
            self.save_vectors(embedded_chunks, conv_id)

        # Step 4: Rebuild index (incremental would be better for production)
        if self.model:
            self.build_faiss_index(rebuild=True)

        return conv_id

    def get_rag_context(self, query: str, top_k: int = 5,
                       use_hybrid: bool = True) -> str:
        """
        Get RAG context for a query.
        Returns formatted context string for LLM consumption.
        """
        if use_hybrid:
            results = self.hybrid_search(query, top_k=top_k)
        else:
            results = self.semantic_search(query, top_k=top_k)

        if not results:
            return "No relevant information found in memory."

        context_parts = ["# Relevant Information from Memory:\n"]

        for i, result in enumerate(results, 1):
            chunk_type = result["chunk_type"]
            content = result["content"]
            similarity = result.get("similarity", result.get("score", 0))
            conv_id = result["conversation_id"]

            context_parts.append(
                f"\n## Result {i} (similarity: {similarity:.3f}, type: {chunk_type})\n"
                f"Conversation: {conv_id}\n"
                f"{content}\n"
            )

        return "\n".join(context_parts)


# Example usage
if __name__ == "__main__":
    engine = EmbeddingsEngine()

    # Test with a sample conversation
    test_conversation = {
        "id": "test-001",
        "timestamp": datetime.now().isoformat(),
        "messages": [
            {"role": "user", "content": "What's my NAS IP address?"},
            {"role": "assistant", "content": "Your NAS is configured and accessible"}
        ],
        "extracted_data": {
            "facts": [
                {"type": "location", "content": "NAS is configured and accessible", "confidence": "high"}
            ],
            "file_paths": [],
            "goals_and_intentions": [],
            "problems_solved": []
        },
        "metadata": {
            "tags": ["nas", "network"],
            "topics": ["configuration"],
            "importance": "high"
        },
        "search_index": {
            "summary": "User asked about NAS IP address"
        }
    }

    # Process conversation
    conv_id = engine.process_conversation(test_conversation)
    print(f"\nProcessed conversation: {conv_id}")

    # Test search
    query = "where is my NAS?"
    results = engine.semantic_search(query, top_k=3)
    print(f"\nSearch results for '{query}':")
    for result in results:
        print(f"  - {result['content']} (similarity: {result['similarity']:.3f})")
