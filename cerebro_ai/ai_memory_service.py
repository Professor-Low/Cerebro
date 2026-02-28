"""
Local AI Memory Service
Runs locally, stores data to configured base directory.
"""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class MemoryService:
    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.conversations_path = self.base_path / "conversations"
        self.facts_path = self.base_path / "facts"
        self.context_path = self.base_path / "context"

    def save_memory(self, content: str, memory_type: str = "conversation",
                   metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Save a memory if it's worth keeping
        Returns: memory_id if saved, None if rejected
        """
        if not self._is_worth_saving(content):
            return None

        memory_id = self._generate_id(content)
        memory_record = {
            "id": memory_id,
            "timestamp": datetime.now().isoformat(),
            "type": memory_type,
            "content": content,
            "metadata": metadata or {},
            "word_count": len(content.split())
        }

        file_path = self._get_path_for_type(memory_type) / f"{memory_id}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory_record, f, indent=2, ensure_ascii=False)

        return memory_id

    def retrieve_memory(self, memory_id: str, memory_type: str = "conversation") -> Optional[Dict]:
        """Retrieve a specific memory by ID"""
        file_path = self._get_path_for_type(memory_type) / f"{memory_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def search_memories(self, memory_type: str = "conversation", limit: int = 100) -> list:
        """Get recent memories of a type"""
        path = self._get_path_for_type(memory_type)
        if not path.exists():
            return []

        memories = []
        for file_path in sorted(path.glob("*.json"), key=os.path.getmtime, reverse=True)[:limit]:
            with open(file_path, 'r', encoding='utf-8') as f:
                memories.append(json.load(f))
        return memories

    def _is_worth_saving(self, content: str) -> bool:
        """Determine if content is worth saving"""
        # Simple heuristics
        if len(content.strip()) < 10:
            return False

        # Skip common filler phrases
        skip_phrases = ["ok", "thanks", "yes", "no", "hello", "hi"]
        if content.lower().strip() in skip_phrases:
            return False

        return True

    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content and timestamp"""
        timestamp = datetime.now().isoformat()
        hash_input = f"{content}{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _get_path_for_type(self, memory_type: str) -> Path:
        """Get storage path for memory type"""
        type_mapping = {
            "conversation": self.conversations_path,
            "fact": self.facts_path,
            "context": self.context_path
        }
        return type_mapping.get(memory_type, self.conversations_path)


# Simple HTTP API wrapper (optional)
if __name__ == "__main__":
    import urllib.parse
    from http.server import BaseHTTPRequestHandler, HTTPServer

    service = MemoryService()

    class MemoryHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            memory_id = service.save_memory(
                content=data.get('content', ''),
                memory_type=data.get('type', 'conversation'),
                metadata=data.get('metadata', {})
            )

            response = {"memory_id": memory_id, "saved": memory_id is not None}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def do_GET(self):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)

            if 'id' in params:
                memory = service.retrieve_memory(params['id'][0], params.get('type', ['conversation'])[0])
                response = memory or {"error": "not found"}
            else:
                memories = service.search_memories(params.get('type', ['conversation'])[0])
                response = {"memories": memories}

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    print("AI Memory Service running on http://localhost:8765")
    server = HTTPServer(('localhost', 8765), MemoryHandler)
    server.serve_forever()
