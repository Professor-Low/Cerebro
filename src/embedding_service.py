"""
Background Embedding Service - Queue-based Processing
AGENT 7 OPTIMIZATION: Offload embedding work to background process

This service keeps the model loaded in memory and processes
embedding requests asynchronously through a queue.
"""
import os
import queue
import threading
import time
from typing import Callable, List, Optional

try:
    from config import EMBEDDING_DEVICE, EMBEDDING_MODEL
except ImportError:
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    EMBEDDING_DEVICE = "auto"


class EmbeddingService:
    """
    Background embedding service with queue-based processing.
    Keeps model loaded and processes requests asynchronously.
    """

    def __init__(self, model_name=None):
        if model_name is None:
            model_name = EMBEDDING_MODEL
        self.model_name = model_name
        self.model = None
        self.request_queue = queue.Queue()
        self.running = False
        self.worker_thread = None

        # Start the service
        self.start()

    @staticmethod
    def _get_device() -> str:
        """Determine compute device, respecting CEREBRO_DEVICE."""
        if EMBEDDING_DEVICE == "cpu":
            return "cpu"
        if EMBEDDING_DEVICE == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            print("[EmbeddingService] WARNING: CEREBRO_DEVICE=cuda but CUDA not available")
            return "cpu"
        # "auto"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def start(self):
        """Start the background worker thread"""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="EmbeddingWorker"
        )
        self.worker_thread.start()

        print("[EmbeddingService] Started background worker")

    def _worker_loop(self):
        """Background worker that processes embedding requests"""
        # Load model once at startup
        print(f"[EmbeddingService] Loading model: {self.model_name}")

        try:
            # Disable progress bars for cleaner output
            os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

            from sentence_transformers import SentenceTransformer
            device = self._get_device()
            self.model = SentenceTransformer(self.model_name, device=device)
            print(f"[EmbeddingService] Model loaded on {device}")
        except Exception as e:
            print(f"[EmbeddingService] ERROR: Failed to load model: {e}")
            self.running = False
            return

        # Process requests from queue
        while self.running:
            try:
                # Get request from queue (blocks with timeout)
                request = self.request_queue.get(timeout=1.0)

                # Process the request
                self._process_request(request)

                self.request_queue.task_done()

            except queue.Empty:
                # No requests, continue waiting
                continue
            except Exception as e:
                print(f"[EmbeddingService] Error processing request: {e}")

    def _process_request(self, request):
        """Process a single embedding request"""
        texts = request['texts']
        callback = request['callback']
        request_id = request['id']

        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Call the callback with results
            callback(embeddings, request_id)

        except Exception as e:
            print(f"[EmbeddingService] Error embedding texts: {e}")
            callback(None, request_id)

    def embed_async(self, texts: List[str], callback: Callable,
                   request_id: Optional[str] = None) -> str:
        """
        Queue texts for embedding asynchronously.

        Args:
            texts: List of text strings to embed
            callback: Function to call with results (embeddings, request_id)
            request_id: Optional ID to track this request

        Returns:
            request_id for tracking
        """
        if not request_id:
            request_id = str(time.time())

        request = {
            'id': request_id,
            'texts': texts,
            'callback': callback
        }

        self.request_queue.put(request)
        return request_id

    def embed_sync(self, texts: List[str], timeout: float = 30.0):
        """
        Synchronous embedding (blocks until complete).

        Args:
            texts: List of text strings to embed
            timeout: Maximum seconds to wait

        Returns:
            Embedding vectors or None if timeout
        """
        result = {'embeddings': None, 'done': False}

        def callback(embeddings, request_id):
            result['embeddings'] = embeddings
            result['done'] = True

        self.embed_async(texts, callback)

        # Wait for result
        start_time = time.time()
        while not result['done']:
            if time.time() - start_time > timeout:
                return None  # Timeout
            time.sleep(0.1)

        return result['embeddings']

    def stop(self):
        """Stop the background worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        print("[EmbeddingService] Stopped")

    def is_ready(self) -> bool:
        """Check if the service is running and model is loaded"""
        return self.running and self.model is not None

    def queue_size(self) -> int:
        """Get the current number of pending requests"""
        return self.request_queue.qsize()


# Global instance (singleton pattern)
_embedding_service = None


def get_embedding_service() -> Optional[EmbeddingService]:
    """Get or create the global embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def shutdown_embedding_service():
    """Shutdown the global embedding service"""
    global _embedding_service
    if _embedding_service is not None:
        _embedding_service.stop()
        _embedding_service = None


# Example usage
if __name__ == "__main__":
    print("Testing Background Embedding Service...")

    service = get_embedding_service()

    # Wait for service to load (model loading takes time)
    print("Waiting for model to load...")
    max_wait = 15
    for i in range(max_wait):
        if service.is_ready():
            print(f"Model ready after {i+1} seconds")
            break
        time.sleep(1)
    else:
        print(f"Model still loading after {max_wait} seconds, continuing anyway...")

    if service.is_ready():
        # Test synchronous embedding
        print("\nTest 1: Synchronous embedding")
        texts = ["Hello world", "This is a test", "Background processing"]
        embeddings = service.embed_sync(texts, timeout=10.0)

        if embeddings is not None:
            print(f"[OK] Generated {len(embeddings)} embeddings")
            print(f"  Embedding shape: {embeddings.shape}")
        else:
            print("[FAIL] Failed to generate embeddings")

        # Test asynchronous embedding
        print("\nTest 2: Asynchronous embedding")

        def on_complete(embeddings, req_id):
            if embeddings is not None:
                print(f"[OK] Request {req_id} complete: {len(embeddings)} vectors")
            else:
                print(f"[FAIL] Request {req_id} failed")

        req_id = service.embed_async(
            ["async test 1", "async test 2"],
            callback=on_complete,
            request_id="test_async"
        )

        print(f"  Queued request: {req_id}")
        print(f"  Queue size: {service.queue_size()}")

        # Wait for async to complete
        time.sleep(2)
    else:
        print("[FAIL] Service failed to initialize")

    # Cleanup
    shutdown_embedding_service()
    print("\n[OK] Service shutdown complete")
