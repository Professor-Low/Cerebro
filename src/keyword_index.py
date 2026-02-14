"""
Fast Keyword Search using SQLite FTS5
Indexes all chunks once, then searches in milliseconds
"""
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List


class KeywordIndex:
    """SQLite-based keyword search for fast queries"""

    # Store index locally for speed
    LOCAL_INDEX_PATH = Path(os.path.expanduser("~")) / ".claude" / "local_brain" / "keyword_index.db"

    def __init__(self, chunks_path: Path = None):
        self.chunks_path = chunks_path
        self._ensure_dir()
        self.conn = None
        self._connect()

    def _ensure_dir(self):
        self.LOCAL_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        """Connect to SQLite database"""
        self.conn = sqlite3.connect(str(self.LOCAL_INDEX_PATH), timeout=10)
        self.conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self):
        """Create FTS5 table if not exists"""
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                conversation_id,
                chunk_id,
                chunk_type,
                content,
                metadata,
                tokenize='porter unicode61'
            )
        """)
        # Track indexed files to avoid re-indexing
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_files (
                filename TEXT PRIMARY KEY,
                mtime REAL,
                chunk_count INTEGER
            )
        """)
        self.conn.commit()

    def get_indexed_count(self) -> int:
        """Get number of indexed chunks"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunks_fts")
        return cursor.fetchone()[0]

    def needs_rebuild(self) -> bool:
        """Check if index needs rebuilding"""
        # If no chunks indexed, definitely need rebuild
        if self.get_indexed_count() == 0:
            return True
        return False

    def build_index(self, chunks_path: Path, progress_callback=None) -> int:
        """Build keyword index from chunk files"""
        if not chunks_path.exists():
            return 0

        chunk_files = list(chunks_path.glob("*.jsonl"))
        total_files = len(chunk_files)
        indexed = 0

        for i, chunk_file in enumerate(chunk_files):
            try:
                # Check if already indexed with same mtime
                mtime = chunk_file.stat().st_mtime
                cursor = self.conn.execute(
                    "SELECT mtime FROM indexed_files WHERE filename = ?",
                    (chunk_file.name,)
                )
                row = cursor.fetchone()
                if row and row[0] == mtime:
                    continue  # Already indexed

                # Get conversation_id from filename (filename is conv_id.jsonl)
                conv_id = chunk_file.stem

                # Remove old entries for this conversation
                self.conn.execute(
                    "DELETE FROM chunks_fts WHERE conversation_id = ?",
                    (conv_id,)
                )

                # Index chunks from file
                chunk_count = 0
                with open(chunk_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            chunk = json.loads(line)
                            self.conn.execute("""
                                INSERT INTO chunks_fts (conversation_id, chunk_id, chunk_type, content, metadata)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                chunk.get("conversation_id", ""),
                                chunk.get("chunk_id", ""),
                                chunk.get("chunk_type", "message"),
                                chunk.get("content", ""),
                                json.dumps(chunk.get("metadata", {}))
                            ))
                            chunk_count += 1
                        except json.JSONDecodeError:
                            continue

                # Record indexed file
                self.conn.execute("""
                    INSERT OR REPLACE INTO indexed_files (filename, mtime, chunk_count)
                    VALUES (?, ?, ?)
                """, (chunk_file.name, mtime, chunk_count))

                indexed += chunk_count

                if progress_callback and i % 100 == 0:
                    progress_callback(i, total_files)

            except Exception as e:
                print(f"Warning: Failed to index {chunk_file}: {e}")
                continue

        self.conn.commit()
        return indexed

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Fast keyword search using FTS5"""
        try:
            # FTS5 query - escape special characters
            safe_query = query.replace('"', '""')

            cursor = self.conn.execute("""
                SELECT
                    conversation_id,
                    chunk_id,
                    chunk_type,
                    content,
                    metadata,
                    bm25(chunks_fts) as score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
            """, (safe_query, top_k))

            # Collect raw results first
            raw_results = []
            for row in cursor:
                try:
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                except:
                    metadata = {}

                raw_results.append({
                    "conversation_id": row["conversation_id"],
                    "chunk_id": row["chunk_id"],
                    "chunk_type": row["chunk_type"],
                    "content": row["content"],
                    "metadata": metadata,
                    "score": -row["score"],  # BM25 returns negative scores
                    "confidence": "MEDIUM"
                })

            # Normalize scores against max in result set (0.0-1.0 range)
            if raw_results:
                max_raw = max(r["score"] for r in raw_results) if raw_results else 1.0
                for r in raw_results:
                    r["similarity"] = min(1.0, r["score"] / max_raw) if max_raw > 0 else 0.0

            return raw_results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def close(self):
        if self.conn:
            self.conn.close()


# Singleton instance
_keyword_index = None

def get_keyword_index(chunks_path: Path = None) -> KeywordIndex:
    """Get or create the keyword index singleton"""
    global _keyword_index
    if _keyword_index is None:
        _keyword_index = KeywordIndex(chunks_path)
    return _keyword_index
