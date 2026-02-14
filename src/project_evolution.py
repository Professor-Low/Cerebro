"""
Project Evolution Tracker
=========================
Tracks project versions and evolution over time.
Solves the problem of search returning outdated iterations.

KEY FEATURES:
1. Version tracking - links conversations to project versions
2. Supersession marking - marks old versions as superseded
3. Recency weighting - newer info ranks higher in search
4. Evolution timeline - shows how a project progressed

Usage:
    tracker = ProjectEvolutionTracker()
    tracker.record_evolution("cerebral-interface", conversation_id, "Added 3D brain visualization")
    tracker.mark_superseded("cerebral-interface", "v2", "v3", "Replaced 2D with 3D")
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ProjectEvolutionTracker:
    """
    Tracks how projects evolve over time to ensure search returns current info.

    Core concepts:
    - Evolution: A change in how something works (architecture, implementation)
    - Version: A distinct iteration that may supersede previous ones
    - Supersession: When new content makes old content obsolete
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.evolution_dir = self.base_path / "project_evolution"
        self.evolution_file = self.evolution_dir / "evolution_registry.json"
        self.superseded_file = self.evolution_dir / "superseded_content.json"

        # Ensure directory exists
        self.evolution_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.evolutions = self._load_evolutions()
        self.superseded = self._load_superseded()

    def record_evolution(self,
                        project_id: str,
                        conversation_id: str,
                        summary: str,
                        version: str = None,
                        supersedes_version: str = None,
                        keywords: List[str] = None) -> Dict:
        """
        Record a project evolution event.

        Args:
            project_id: Unique project identifier (e.g., "cerebral-interface")
            conversation_id: The conversation where this was discussed
            summary: What changed in this version
            version: Optional version label (auto-generated if not provided)
            supersedes_version: If this supersedes an older version
            keywords: Keywords for matching (e.g., ["visualization", "3D", "brain"])

        Returns:
            Evolution record
        """
        if project_id not in self.evolutions:
            self.evolutions[project_id] = {
                "project_id": project_id,
                "created_at": datetime.now().isoformat(),
                "versions": [],
                "current_version": None,
                "keywords": keywords or []
            }

        project = self.evolutions[project_id]

        # Auto-generate version if not provided
        if not version:
            version_num = len(project["versions"]) + 1
            version = f"v{version_num}"

        # Create evolution record
        evolution = {
            "version": version,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "supersedes": supersedes_version,
            "is_current": True,
            "keywords": keywords or []
        }

        # Mark previous versions as not current
        for v in project["versions"]:
            v["is_current"] = False

        project["versions"].append(evolution)
        project["current_version"] = version
        project["last_updated"] = datetime.now().isoformat()

        # Update keywords
        if keywords:
            existing_keywords = set(project.get("keywords", []))
            project["keywords"] = list(existing_keywords.union(set(keywords)))

        # If this supersedes another version, record it
        if supersedes_version:
            self.mark_superseded(
                project_id,
                supersedes_version,
                version,
                f"Superseded by {version}: {summary}"
            )

        self._save_evolutions()
        return evolution

    def mark_superseded(self,
                       project_id: str,
                       old_version: str,
                       new_version: str,
                       reason: str = None,
                       conversation_ids: List[str] = None) -> bool:
        """
        Mark content as superseded by newer content.

        Args:
            project_id: Project this belongs to
            old_version: The superseded version
            new_version: The version that supersedes it
            reason: Why it was superseded
            conversation_ids: Specific conversation IDs to mark

        Returns:
            True if successful
        """
        key = f"{project_id}:{old_version}"

        self.superseded[key] = {
            "project_id": project_id,
            "old_version": old_version,
            "superseded_by": new_version,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "conversation_ids": conversation_ids or []
        }

        self._save_superseded()
        return True

    def is_superseded(self, project_id: str, version: str = None,
                     conversation_id: str = None) -> Tuple[bool, Optional[str]]:
        """
        Check if content is superseded.

        Args:
            project_id: Project to check
            version: Version to check (optional)
            conversation_id: Conversation to check (optional)

        Returns:
            (is_superseded, superseded_by_version)
        """
        # Check by version
        if version:
            key = f"{project_id}:{version}"
            if key in self.superseded:
                return True, self.superseded[key].get("superseded_by")

        # Check by conversation ID
        if conversation_id:
            for entry in self.superseded.values():
                if conversation_id in entry.get("conversation_ids", []):
                    return True, entry.get("superseded_by")

        return False, None

    def get_project_timeline(self, project_id: str) -> List[Dict]:
        """
        Get the evolution timeline for a project.

        Returns:
            List of versions in chronological order with metadata
        """
        if project_id not in self.evolutions:
            return []

        project = self.evolutions[project_id]
        timeline = []

        for version in project["versions"]:
            is_superseded, superseded_by = self.is_superseded(
                project_id, version["version"]
            )

            timeline.append({
                "version": version["version"],
                "timestamp": version["timestamp"],
                "summary": version["summary"],
                "conversation_id": version["conversation_id"],
                "is_current": version["is_current"],
                "is_superseded": is_superseded,
                "superseded_by": superseded_by
            })

        return timeline

    def get_current_version(self, project_id: str) -> Optional[Dict]:
        """Get the current (most recent non-superseded) version of a project."""
        if project_id not in self.evolutions:
            return None

        project = self.evolutions[project_id]
        for version in reversed(project["versions"]):
            if version.get("is_current"):
                return version

        # Fallback to last version
        if project["versions"]:
            return project["versions"][-1]

        return None

    def calculate_recency_score(self,
                               timestamp: str,
                               decay_days: int = 30,
                               min_score: float = 0.3,
                               topic: str = None) -> float:
        """
        Calculate a recency multiplier for search results.
        Newer content gets higher scores.

        Args:
            timestamp: ISO timestamp of the content
            decay_days: Days until score reaches half-life
            min_score: Minimum multiplier (even old content gets some weight)
            topic: Optional topic for topic-specific decay rates

        Returns:
            Multiplier between min_score and 1.0
        """
        try:
            content_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            # Handle timezone-naive comparison
            if content_time.tzinfo:
                content_time = content_time.replace(tzinfo=None)

            now = datetime.now()
            age_days = (now - content_time).days

            # Phase 2.3: Topic-specific decay rates
            TOPIC_DECAY_RATES = {
                "debugging": 14,    # Debugging solutions become stale faster
                "error": 14,
                "fix": 14,
                "bug": 14,
                "personal": 90,     # Personal facts stay relevant longer
                "identity": 90,
                "preference": 90,
                "infrastructure": 60,  # Config facts stay moderately relevant
                "configuration": 60,
            }

            if topic:
                topic_lower = topic.lower()
                for topic_key, rate in TOPIC_DECAY_RATES.items():
                    if topic_key in topic_lower:
                        decay_days = rate
                        break

            # Phase 2.3: "Mentioned today" boost (2x weight)
            if age_days == 0:
                # Content from today gets 2x boost
                return min(1.0, 1.0 * 2.0)  # Cap at 1.0 but effectively doubles relevance

            # Exponential decay: score = e^(-age/decay_days)
            decay_factor = math.exp(-age_days / decay_days)

            # Scale to min_score..1.0 range
            return min_score + (1.0 - min_score) * decay_factor

        except (ValueError, TypeError):
            # If timestamp parsing fails, return neutral score
            return 0.7

    def get_search_penalty(self,
                          conversation_id: str,
                          timestamp: str,
                          project_hints: List[str] = None) -> Dict:
        """
        Calculate search score adjustments for a piece of content.

        Returns:
            {
                "recency_multiplier": float,
                "superseded_penalty": float,
                "final_multiplier": float,
                "is_superseded": bool,
                "superseded_by": str or None
            }
        """
        # Check if superseded by any project
        is_superseded = False
        superseded_by = None

        if project_hints:
            for project in project_hints:
                superseded, by_version = self.is_superseded(
                    project, conversation_id=conversation_id
                )
                if superseded:
                    is_superseded = True
                    superseded_by = f"{project}:{by_version}"
                    break

        # Calculate recency
        recency_multiplier = self.calculate_recency_score(timestamp)

        # Apply supersession penalty (0.2x for superseded content)
        superseded_penalty = 0.2 if is_superseded else 1.0

        final_multiplier = recency_multiplier * superseded_penalty

        return {
            "recency_multiplier": recency_multiplier,
            "superseded_penalty": superseded_penalty,
            "final_multiplier": final_multiplier,
            "is_superseded": is_superseded,
            "superseded_by": superseded_by
        }

    def find_project_for_query(self, query: str) -> Optional[str]:
        """
        Try to identify which project a query is about based on keywords.

        Args:
            query: Search query

        Returns:
            Project ID if found, None otherwise
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        best_match = None
        best_score = 0

        for project_id, project in self.evolutions.items():
            # Check project ID match
            if project_id.replace("-", " ").lower() in query_lower:
                return project_id

            # Check keywords
            keywords = set(k.lower() for k in project.get("keywords", []))
            keyword_matches = len(query_words.intersection(keywords))

            if keyword_matches > best_score:
                best_score = keyword_matches
                best_match = project_id

        # Only return if we have at least 2 keyword matches
        return best_match if best_score >= 2 else None

    def auto_detect_evolution(self,
                             conversation: Dict,
                             existing_projects: List[str] = None) -> Optional[Dict]:
        """
        Automatically detect if a conversation represents a project evolution.

        Args:
            conversation: Conversation data
            existing_projects: Known project IDs to check against

        Returns:
            Detected evolution info or None
        """
        content = ""
        for msg in conversation.get("messages", []):
            content += msg.get("content", "") + " "
        content_lower = content.lower()

        # Evolution indicators
        evolution_patterns = [
            "updated", "new version", "replaced", "refactored",
            "rewritten", "improved", "changed to", "now using",
            "switched from", "migrated", "upgraded"
        ]

        has_evolution = any(p in content_lower for p in evolution_patterns)

        if not has_evolution:
            return None

        # Try to match to existing project
        matched_project = None
        if existing_projects:
            for project_id in existing_projects:
                if project_id.replace("-", " ").lower() in content_lower:
                    matched_project = project_id
                    break

        if matched_project:
            return {
                "detected": True,
                "project_id": matched_project,
                "conversation_id": conversation.get("id"),
                "likely_evolution": True
            }

        return None

    def get_all_projects(self) -> List[Dict]:
        """Get summary of all tracked projects."""
        return [
            {
                "project_id": project_id,
                "current_version": project.get("current_version"),
                "version_count": len(project.get("versions", [])),
                "last_updated": project.get("last_updated"),
                "keywords": project.get("keywords", [])
            }
            for project_id, project in self.evolutions.items()
        ]

    def _load_evolutions(self) -> Dict:
        """Load evolution registry from disk."""
        if self.evolution_file.exists():
            try:
                with open(self.evolution_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_evolutions(self):
        """Save evolution registry to disk."""
        with open(self.evolution_file, 'w', encoding='utf-8') as f:
            json.dump(self.evolutions, f, indent=2)

    def _load_superseded(self) -> Dict:
        """Load superseded content registry."""
        if self.superseded_file.exists():
            try:
                with open(self.superseded_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_superseded(self):
        """Save superseded content registry."""
        with open(self.superseded_file, 'w', encoding='utf-8') as f:
            json.dump(self.superseded, f, indent=2)


# ============================================================================
# Recency-Weighted Search Integration
# ============================================================================

def apply_recency_weighting(results: List[Dict],
                           tracker: ProjectEvolutionTracker = None,
                           query: str = None,
                           decay_days: int = 30) -> List[Dict]:
    """
    Apply recency weighting to search results.
    Call this after semantic_search to rerank by recency.

    Args:
        results: Search results from semantic_search or hybrid_search
        tracker: ProjectEvolutionTracker instance
        query: Original query (for project matching)
        decay_days: Decay half-life in days

    Returns:
        Re-ranked results with recency adjustments
    """
    if not tracker:
        tracker = ProjectEvolutionTracker()

    # Try to find project context
    project_hints = []
    if query:
        matched_project = tracker.find_project_for_query(query)
        if matched_project:
            project_hints.append(matched_project)

    # Apply recency adjustments
    for result in results:
        timestamp = result.get("metadata", {}).get("timestamp")
        conv_id = result.get("conversation_id")

        if timestamp:
            penalty_info = tracker.get_search_penalty(
                conversation_id=conv_id,
                timestamp=timestamp,
                project_hints=project_hints
            )

            # Apply multiplier to score
            original_score = result.get("score", result.get("similarity", 0.5))
            result["original_score"] = original_score
            result["score"] = original_score * penalty_info["final_multiplier"]
            result["recency_multiplier"] = penalty_info["recency_multiplier"]
            result["is_superseded"] = penalty_info["is_superseded"]
            result["superseded_by"] = penalty_info["superseded_by"]
        else:
            # No timestamp, apply neutral recency
            result["recency_multiplier"] = 0.7

    # Re-sort by adjusted score
    results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    return results


if __name__ == "__main__":
    # Test the tracker
    print("Testing Project Evolution Tracker...")

    tracker = ProjectEvolutionTracker()

    # Record some test evolutions
    tracker.record_evolution(
        "cerebral-interface",
        "test-conv-1",
        "Initial 2D visualization with simple node graph",
        version="v1",
        keywords=["visualization", "2D", "brain", "nodes"]
    )

    tracker.record_evolution(
        "cerebral-interface",
        "test-conv-2",
        "Added 3D Force-Directed Graph with Three.js",
        version="v2",
        supersedes_version="v1",
        keywords=["visualization", "3D", "brain", "three.js", "force-directed"]
    )

    tracker.record_evolution(
        "cerebral-interface",
        "test-conv-3",
        "Real-time WebSocket updates with animated search ripples",
        version="v3",
        supersedes_version="v2",
        keywords=["visualization", "3D", "websocket", "real-time", "animation"]
    )

    # Show timeline
    print("\nProject Timeline:")
    for entry in tracker.get_project_timeline("cerebral-interface"):
        status = "CURRENT" if entry["is_current"] else "SUPERSEDED" if entry["is_superseded"] else ""
        print(f"  {entry['version']}: {entry['summary'][:50]}... [{status}]")

    # Test recency scoring
    print("\nRecency Scores:")
    for days_ago in [0, 7, 30, 90, 180]:
        test_time = (datetime.now() - timedelta(days=days_ago)).isoformat()
        score = tracker.calculate_recency_score(test_time)
        print(f"  {days_ago} days ago: {score:.3f}")

    # Test project matching
    print("\nProject Matching:")
    test_queries = [
        "How does the cerebral interface visualization work?",
        "What's the NAS setup?",
        "Show me the 3D brain graph"
    ]
    for query in test_queries:
        matched = tracker.find_project_for_query(query)
        print(f"  '{query[:40]}...' -> {matched or 'No match'}")

    print("\nTest complete!")
