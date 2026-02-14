"""
Memory Deduplicator - Agent 11
Finds and merges duplicate memory chunks to maintain clean memory
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


class MemoryDeduplicator:
    """
    Find and merge duplicate memory chunks.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.chunks_path = self.base_path / "embeddings" / "chunks"
        self.vectors_path = self.base_path / "embeddings" / "vectors"
        self.conversations_path = self.base_path / "conversations"

    def find_duplicates(self, threshold: float = 0.90) -> List[Dict]:
        """
        Find chunks with similarity above threshold.

        Args:
            threshold: Similarity threshold (0.0-1.0, default 0.90)

        Returns:
            List of duplicate pairs with similarity scores
        """
        duplicates = []

        # Load all chunks
        all_chunks = []
        chunk_files = list(self.chunks_path.glob('*.jsonl'))

        print(f"[Deduplicator] Scanning {len(chunk_files)} chunk files...")

        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            chunk = json.loads(line)
                            chunk['source_file'] = chunk_file.name
                            all_chunks.append(chunk)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"[Deduplicator] Error reading {chunk_file}: {e}")
                continue

        print(f"[Deduplicator] Found {len(all_chunks)} total chunks")

        # Load embeddings engine
        from ai_embeddings_engine import EmbeddingsEngine
        embeddings_engine = EmbeddingsEngine()

        # Compare chunks pairwise (optimized with early stopping)
        for i in range(len(all_chunks)):
            for j in range(i + 1, len(all_chunks)):
                chunk1 = all_chunks[i]
                chunk2 = all_chunks[j]

                # Skip if from same conversation (those are expected to be related)
                if chunk1.get('conversation_id') == chunk2.get('conversation_id'):
                    continue

                # Compute similarity
                similarity = self._compute_similarity(chunk1['content'], chunk2['content'], embeddings_engine)

                if similarity >= threshold:
                    duplicates.append({
                        'chunk1': {
                            'content': chunk1['content'][:200],  # Truncate for display
                            'conversation_id': chunk1.get('conversation_id'),
                            'chunk_id': chunk1.get('chunk_id')
                        },
                        'chunk2': {
                            'content': chunk2['content'][:200],
                            'conversation_id': chunk2.get('conversation_id'),
                            'chunk_id': chunk2.get('chunk_id')
                        },
                        'similarity': round(similarity, 3),
                        'suggestion': 'merge' if similarity > 0.95 else 'review'
                    })

            # Progress update every 100 chunks
            if (i + 1) % 100 == 0:
                print(f"[Deduplicator] Processed {i + 1}/{len(all_chunks)} chunks, found {len(duplicates)} duplicates")

        # Sort by similarity
        duplicates.sort(key=lambda x: x['similarity'], reverse=True)

        print(f"[Deduplicator] Found {len(duplicates)} duplicate pairs")

        return duplicates

    def _compute_similarity(self, text1: str, text2: str, embeddings_engine) -> float:
        """Compute cosine similarity between two texts"""
        try:
            # Check if model is available
            if not embeddings_engine.model:
                # Fall back to simple text comparison
                return self._simple_text_similarity(text1, text2)

            # Generate embeddings
            embeddings = embeddings_engine.model.encode([text1, text2], convert_to_numpy=True)

            # Compute cosine similarity
            vec1 = embeddings[0]
            vec2 = embeddings[1]

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            return float(similarity)

        except Exception as e:
            print(f"[Deduplicator] Error computing similarity: {e}")
            return 0.0

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple word-based similarity for fallback"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def merge_duplicates(self, duplicates: List[Dict], auto_merge: bool = False) -> Dict:
        """
        Merge duplicate chunks.

        Args:
            duplicates: List of duplicates from find_duplicates()
            auto_merge: If True, auto-merge duplicates >0.95 similarity

        Returns:
            Merge statistics
        """
        merged_count = 0
        skipped_count = 0

        for dup in duplicates:
            similarity = dup['similarity']

            # Auto-merge only if >0.95 similarity and auto_merge enabled
            if similarity > 0.95 and auto_merge:
                # Keep the longer, more detailed version
                chunk1_len = len(dup['chunk1']['content'])
                chunk2_len = len(dup['chunk2']['content'])

                primary_conv_id = dup['chunk1']['conversation_id'] if chunk1_len > chunk2_len else dup['chunk2']['conversation_id']
                secondary_conv_id = dup['chunk2']['conversation_id'] if primary_conv_id == dup['chunk1']['conversation_id'] else dup['chunk1']['conversation_id']

                # Mark secondary as duplicate reference
                self._add_duplicate_reference(primary_conv_id, secondary_conv_id, similarity)

                merged_count += 1
            else:
                skipped_count += 1

        return {
            'merged': merged_count,
            'skipped': skipped_count,
            'total_duplicates': len(duplicates)
        }

    def _add_duplicate_reference(self, primary_conv_id: str, secondary_conv_id: str, similarity: float):
        """Add duplicate reference to primary conversation"""
        # Update conversation JSON with duplicate reference
        primary_file = self.conversations_path / f"{primary_conv_id}.json"

        if not primary_file.exists():
            return

        try:
            with open(primary_file, 'r', encoding='utf-8') as f:
                conv = json.load(f)

            if 'duplicates' not in conv.get('metadata', {}):
                conv.setdefault('metadata', {})['duplicates'] = []

            conv['metadata']['duplicates'].append({
                'conversation_id': secondary_conv_id,
                'similarity': similarity,
                'detected_at': datetime.now().isoformat()
            })

            with open(primary_file, 'w', encoding='utf-8') as f:
                json.dump(conv, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[Deduplicator] Error adding duplicate reference: {e}")


# Example usage
if __name__ == "__main__":
    dedup = MemoryDeduplicator()

    # Find duplicates
    print("Finding duplicates...")
    duplicates = dedup.find_duplicates(threshold=0.90)

    print(f"\nFound {len(duplicates)} duplicate pairs:")
    for i, dup in enumerate(duplicates[:10], 1):
        print(f"\n{i}. Similarity: {dup['similarity']}")
        print(f"   Chunk 1 (Conv: {dup['chunk1']['conversation_id']}): {dup['chunk1']['content'][:100]}")
        print(f"   Chunk 2 (Conv: {dup['chunk2']['conversation_id']}): {dup['chunk2']['content'][:100]}")
        print(f"   Suggestion: {dup['suggestion']}")

    # Merge duplicates
    if duplicates:
        print("\n\nMerging duplicates...")
        stats = dedup.merge_duplicates(duplicates, auto_merge=True)
        print(f"Merged: {stats['merged']}, Skipped: {stats['skipped']}")
