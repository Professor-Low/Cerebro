"""
Auto-Context Injector - Proactively retrieve and inject relevant context

AGENT 13: SMART AUTO-CONTEXT INJECTION
This is the INTELLIGENCE LAYER that makes Claude "remember" without being asked.
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import entity detector
from entity_detector import EntityDetector


class AutoContextInjector:
    """
    Automatically inject relevant context based on user prompts.

    Flow:
    1. Detect entities in user prompt
    2. Fast search for relevant memories
    3. Format top 2-3 results with confidence scores
    4. Return as additional context for Claude
    """

    def __init__(self):
        self.entity_detector = EntityDetector()
        from config import DATA_DIR
        self.base_path = DATA_DIR

        # Import existing services
        sys.path.insert(0, str(Path(__file__).parent))

        # Lazy load embeddings engine (only when needed)
        self.embeddings = None
        self._embeddings_failed = False

    def _get_embeddings_engine(self):
        """Lazy load embeddings engine to avoid startup delays"""
        if self.embeddings is not None:
            return self.embeddings

        if self._embeddings_failed:
            return None

        try:
            from ai_embeddings_engine import EmbeddingsEngine
            self.embeddings = EmbeddingsEngine(base_path=str(self.base_path))
            return self.embeddings
        except Exception as e:
            print(f"[AutoContextInjector] Warning: Could not load embeddings engine: {e}")
            self._embeddings_failed = True
            return None

    def inject_context(self, user_prompt: str, max_chunks: int = 3) -> Optional[str]:
        """
        Main entry point: Detect entities and inject relevant context.

        Args:
            user_prompt: The user's message
            max_chunks: Max context chunks to inject (default 3)

        Returns:
            Formatted context string or None if no context found
        """
        # Step 1: Detect entities
        entities = self.entity_detector.detect(user_prompt)

        # Step 2: Decide if we should inject
        if not self.entity_detector.should_inject_context(entities):
            return None

        # Step 3: Build search query from entities
        search_query = self._build_search_query(entities, user_prompt)

        # Step 4: Fast hybrid search
        try:
            embeddings = self._get_embeddings_engine()
            if embeddings is None:
                return None

            results = embeddings.hybrid_search(
                query=search_query,
                top_k=max_chunks,
                alpha=0.7  # Balance semantic + keyword
            )

            if not results:
                return None

            # Step 5: Format with confidence scores
            context = self._format_context(results, entities)
            return context

        except Exception as e:
            print(f"[AutoContextInjector] Error during search: {e}")
            return None

    def _build_search_query(self, entities: Dict, original_prompt: str) -> str:
        """
        Build optimized search query from detected entities.

        Strategy:
        - Prioritize specific entities (tools, projects, file paths)
        - Include original prompt for semantic matching
        - Boost with technical keywords
        """
        query_parts = []

        # Add specific entities (highest priority)
        if entities['tools']:
            query_parts.extend(entities['tools'][:3])  # Top 3 tools

        if entities['technologies']:
            query_parts.extend(entities['technologies'][:3])  # Top 3 technologies

        if entities['projects']:
            query_parts.extend(entities['projects'])

        if entities['file_paths']:
            # Extract just filename for better matching
            for path in entities['file_paths'][:2]:  # Top 2 file paths
                query_parts.append(Path(path).name)

        # Add networks/IPs
        if entities['networks']:
            query_parts.extend(entities['networks'][:2])

        # Add original prompt (for semantic)
        query_parts.append(original_prompt)

        # Add top keywords for boosting
        if entities['keywords']:
            query_parts.extend(entities['keywords'][:3])

        # Join into search query
        return " ".join(query_parts)

    def _format_context(self, results: List[Dict], entities: Dict) -> str:
        """
        Format search results as injected context with confidence scores.

        Format:
        [AUTO-CONTEXT] Based on detected entities: NAS, config

        📌 RELEVANT MEMORY (Confidence: HIGH - 94%)
        From conversation 2 weeks ago:
        "Your NAS IP is 10.0.0.100, configured with SMB timeout of 30s..."

        📌 RELEVANT MEMORY (Confidence: MEDIUM - 78%)
        From conversation 1 month ago:
        "When updating NAS config, remember to restart the SMB service..."
        """
        lines = []

        # Header - show what entities we detected
        detected = []
        if entities['tools']:
            detected.extend(entities['tools'][:3])
        if entities['technologies']:
            detected.extend(entities['technologies'][:2])
        if entities['projects']:
            detected.extend(entities['projects'][:2])
        if entities['file_paths']:
            # Just show filenames, not full paths
            detected.extend([Path(p).name for p in entities['file_paths'][:2]])

        if detected:
            header = f"[AUTO-CONTEXT] Detected: {', '.join(detected[:5])}"
        else:
            header = "[AUTO-CONTEXT] Relevant context found:"

        lines.append(header)
        lines.append("")

        # Format each result
        for i, result in enumerate(results[:3], 1):
            # Get similarity score (might be 'similarity' or 'score')
            similarity = result.get('similarity_score',
                                   result.get('similarity',
                                   result.get('score', 0.0)))

            confidence = self._score_to_confidence(similarity)

            # Get timestamp info
            result.get('conversation_id', 'unknown')
            timestamp = result.get('metadata', {}).get('timestamp', '')
            time_ago = self._format_time_ago(timestamp)

            # Get content and truncate if needed
            content = result.get('content', '')
            if len(content) > 250:
                content = content[:250] + "..."

            lines.append(f"[{i}] RELEVANT MEMORY (Confidence: {confidence} - {similarity*100:.0f}%)")
            lines.append(f"From conversation {time_ago}:")
            lines.append(f'"{content}"')
            lines.append("")

        return "\n".join(lines)

    def _score_to_confidence(self, similarity: float) -> str:
        """Convert similarity score to confidence label"""
        if similarity >= 0.85:
            return "HIGH"
        elif similarity >= 0.70:
            return "MEDIUM"
        else:
            return "LOW"

    def _format_time_ago(self, timestamp: str) -> str:
        """Format timestamp as 'X days/weeks ago'"""
        if not timestamp:
            return "unknown time"

        try:
            ts = datetime.fromisoformat(timestamp)
            now = datetime.now()
            delta = now - ts

            if delta.days < 1:
                hours = delta.seconds // 3600
                if hours < 1:
                    return "just now"
                return f"{hours} hours ago"
            elif delta.days < 7:
                return f"{delta.days} days ago"
            elif delta.days < 30:
                weeks = delta.days // 7
                return f"{weeks} weeks ago"
            else:
                months = delta.days // 30
                if months == 1:
                    return "1 month ago"
                return f"{months} months ago"
        except Exception as e:
            print(f"[AutoContextInjector] Error parsing timestamp: {e}")
            return "unknown time"


# Example usage / testing
if __name__ == "__main__":
    injector = AutoContextInjector()

    # Test cases
    test_prompts = [
        "What's my NAS IP address?",
        "Update the config at /data/memory/config.json",
        "How do I configure the MCP server?",
        "Tell me about the cerebral-interface project",
        "Hello, how are you?"  # Should return None
    ]

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        context = injector.inject_context(prompt, max_chunks=3)

        if context:
            print(context)
        else:
            print("[No context injected]")
