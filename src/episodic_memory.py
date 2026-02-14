"""
Episodic Memory Manager - Claude.Me v6.0
Stores events with full context: what happened, when, who was involved, emotional state.

Part of Phase 1: Episodic vs Semantic Separation
"""
import hashlib
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from memory_types import EpisodicMemory


class EpisodicMemoryManager:
    """
    Manage episodic memories - events with full context.

    Episodic memories answer: "What happened on date X?"
    They preserve the temporal, emotional, and causal context of events.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.episodic_path = self.base_path / "episodic"
        self.index_path = self.episodic_path / "_index.json"
        self._index_lock = threading.Lock()
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.episodic_path.mkdir(parents=True, exist_ok=True)

    def _generate_id(self, timestamp: str = None) -> str:
        """Generate unique episodic memory ID."""
        ts = timestamp or datetime.now().isoformat()
        date_part = ts[:10].replace("-", "")
        hash_part = hashlib.sha256(f"{ts}{datetime.now().timestamp()}".encode()).hexdigest()[:6]
        return f"ep_{date_part}_{hash_part}"

    def _load_index(self) -> Dict:
        """Load the episodic memory index."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"entries": [], "by_date": {}, "by_actor": {}, "by_emotion": {}}

    def _save_index(self, index: Dict):
        """Save the episodic memory index."""
        with self._index_lock:
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)

    def save_episode(self, episode: EpisodicMemory) -> str:
        """
        Save an episodic memory.

        Returns:
            The episode ID
        """
        # Save the episode file
        episode_file = self.episodic_path / f"{episode.id}.json"
        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump(episode.to_dict(), f, indent=2)

        # Update index
        index = self._load_index()

        entry = {
            "id": episode.id,
            "timestamp": episode.timestamp,
            "event_summary": episode.event[:200],
            "actors": episode.actors,
            "emotional_state": episode.emotional_state,
            "outcome": episode.outcome[:100] if episode.outcome else None
        }

        # Add to entries list
        index["entries"].append(entry)

        # Index by date
        date_key = episode.timestamp[:10]
        if date_key not in index["by_date"]:
            index["by_date"][date_key] = []
        index["by_date"][date_key].append(episode.id)

        # Index by actor
        for actor in episode.actors:
            actor_key = actor.lower()
            if actor_key not in index["by_actor"]:
                index["by_actor"][actor_key] = []
            index["by_actor"][actor_key].append(episode.id)

        # Index by emotion
        if episode.emotional_state:
            emotion_key = episode.emotional_state.lower()
            if emotion_key not in index["by_emotion"]:
                index["by_emotion"][emotion_key] = []
            index["by_emotion"][emotion_key].append(episode.id)

        self._save_index(index)
        return episode.id

    def get_episode(self, episode_id: str) -> Optional[EpisodicMemory]:
        """Retrieve an episodic memory by ID."""
        episode_file = self.episodic_path / f"{episode_id}.json"
        if not episode_file.exists():
            return None

        try:
            with open(episode_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return EpisodicMemory.from_dict(data)
        except Exception as e:
            print(f"[EpisodicMemory] Error loading {episode_id}: {e}")
            return None

    def query_by_date(self, date: str) -> List[EpisodicMemory]:
        """
        Get all episodes from a specific date.

        Args:
            date: Date string in YYYY-MM-DD format
        """
        index = self._load_index()
        episode_ids = index.get("by_date", {}).get(date, [])
        return [self.get_episode(eid) for eid in episode_ids if self.get_episode(eid)]

    def query_by_date_range(self, start_date: str, end_date: str) -> List[EpisodicMemory]:
        """Get all episodes within a date range."""
        index = self._load_index()
        episodes = []

        for date_key, episode_ids in index.get("by_date", {}).items():
            if start_date <= date_key <= end_date:
                for eid in episode_ids:
                    ep = self.get_episode(eid)
                    if ep:
                        episodes.append(ep)

        # Sort by timestamp
        episodes.sort(key=lambda e: e.timestamp)
        return episodes

    def query_by_actor(self, actor: str) -> List[EpisodicMemory]:
        """Get all episodes involving a specific actor."""
        index = self._load_index()
        episode_ids = index.get("by_actor", {}).get(actor.lower(), [])
        episodes = [self.get_episode(eid) for eid in episode_ids if self.get_episode(eid)]
        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        return episodes

    def query_by_emotion(self, emotion: str) -> List[EpisodicMemory]:
        """Get all episodes with a specific emotional context."""
        index = self._load_index()
        episode_ids = index.get("by_emotion", {}).get(emotion.lower(), [])
        episodes = [self.get_episode(eid) for eid in episode_ids if self.get_episode(eid)]
        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        return episodes

    def search_episodes(self, query: str, limit: int = 10) -> List[EpisodicMemory]:
        """
        Search episodic memories by keyword.

        Simple keyword search - for semantic search, use embeddings.
        """
        index = self._load_index()
        query_lower = query.lower()
        results = []

        for entry in index.get("entries", []):
            # Search in event summary
            if query_lower in entry.get("event_summary", "").lower():
                ep = self.get_episode(entry["id"])
                if ep:
                    results.append(ep)

        # Sort by timestamp (most recent first)
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def get_recent_episodes(self, days: int = 7, limit: int = 20) -> List[EpisodicMemory]:
        """Get recent episodic memories."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        index = self._load_index()
        episodes = []

        for date_key, episode_ids in index.get("by_date", {}).items():
            if date_key >= cutoff:
                for eid in episode_ids:
                    ep = self.get_episode(eid)
                    if ep:
                        episodes.append(ep)

        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        return episodes[:limit]

    def link_to_semantic(self, episode_id: str, semantic_id: str):
        """Link an episodic memory to a semantic memory it contributed to."""
        episode = self.get_episode(episode_id)
        if episode and semantic_id not in episode.linked_semantic_ids:
            episode.linked_semantic_ids.append(semantic_id)
            self.save_episode(episode)

    def create_from_conversation(
        self,
        conversation: Dict,
        event_summary: str,
        outcome: Optional[str] = None,
        emotional_state: Optional[str] = None
    ) -> EpisodicMemory:
        """
        Create an episodic memory from a conversation.

        Args:
            conversation: The conversation data
            event_summary: What happened (e.g., "Debugged NAS timeout issue")
            outcome: The result (e.g., "Fixed by increasing timeout to 60s")
            emotional_state: User's emotional journey (e.g., "frustrated -> relieved")
        """
        timestamp = conversation.get("timestamp", datetime.now().isoformat())
        conv_id = conversation.get("id")

        # Extract context
        context = {
            "conversation_id": conv_id,
            "session_type": conversation.get("metadata", {}).get("session_type"),
            "topics": conversation.get("metadata", {}).get("topics", []),
            "tags": conversation.get("metadata", {}).get("tags", [])
        }

        # Estimate duration from message count
        messages = conversation.get("messages", [])
        duration = len(messages) * 2  # Rough estimate: 2 min per message

        episode = EpisodicMemory(
            id=self._generate_id(timestamp),
            timestamp=timestamp,
            event=event_summary,
            context=context,
            actors=["User", "Claude"],
            outcome=outcome,
            emotional_state=emotional_state,
            duration_minutes=duration,
            conversation_id=conv_id
        )

        self.save_episode(episode)
        return episode

    def detect_emotional_state(self, messages: List[Dict]) -> str:
        """
        Detect emotional state from conversation messages.

        Returns emotional journey like "frustrated -> relieved" or single state.
        """
        emotions_detected = []

        # Emotion keywords
        frustration_words = ["stuck", "not working", "broken", "error", "failed", "ugh", "argh"]
        relief_words = ["finally", "it works", "fixed", "solved", "yes!", "perfect"]
        excitement_words = ["awesome", "amazing", "love it", "incredible", "wow"]
        confusion_words = ["confused", "don't understand", "what", "why", "how"]

        for msg in messages:
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "").lower()

            if any(w in content for w in frustration_words):
                if "frustrated" not in emotions_detected:
                    emotions_detected.append("frustrated")
            if any(w in content for w in relief_words):
                if "relieved" not in emotions_detected:
                    emotions_detected.append("relieved")
            if any(w in content for w in excitement_words):
                if "excited" not in emotions_detected:
                    emotions_detected.append("excited")
            if any(w in content for w in confusion_words):
                if "confused" not in emotions_detected:
                    emotions_detected.append("confused")

        if not emotions_detected:
            return "neutral"

        if len(emotions_detected) == 1:
            return emotions_detected[0]

        # Return journey
        return " -> ".join(emotions_detected)

    def get_stats(self) -> Dict:
        """Get statistics about episodic memory."""
        index = self._load_index()

        return {
            "total_episodes": len(index.get("entries", [])),
            "unique_dates": len(index.get("by_date", {})),
            "actors": list(index.get("by_actor", {}).keys()),
            "emotional_states": list(index.get("by_emotion", {}).keys()),
            "recent_7_days": len(self.get_recent_episodes(days=7, limit=1000))
        }
