"""
Abstractor - Claude.Me v6.0
Create abstractions from episodic memories.

Part of Phase 10: Active Memory Consolidation
"""
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


class Abstraction:
    """Represents an abstraction formed from multiple episodes."""

    def __init__(
        self,
        abstraction_id: str,
        pattern_name: str,
        description: str,
        source_episodes: List[str],
        commonalities: List[str],
        generalized_insight: str,
        confidence: float = 0.5,
        domain: str = "general"
    ):
        self.abstraction_id = abstraction_id
        self.pattern_name = pattern_name
        self.description = description
        self.source_episodes = source_episodes
        self.commonalities = commonalities
        self.generalized_insight = generalized_insight
        self.confidence = confidence
        self.domain = domain
        self.created_at = datetime.now().isoformat()
        self.access_count = 0

    def to_dict(self) -> Dict:
        return {
            "abstraction_id": self.abstraction_id,
            "pattern_name": self.pattern_name,
            "description": self.description,
            "source_episodes": self.source_episodes,
            "commonalities": self.commonalities,
            "generalized_insight": self.generalized_insight,
            "confidence": self.confidence,
            "domain": self.domain,
            "created_at": self.created_at,
            "access_count": self.access_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Abstraction":
        abstr = cls(
            abstraction_id=data["abstraction_id"],
            pattern_name=data["pattern_name"],
            description=data["description"],
            source_episodes=data.get("source_episodes", []),
            commonalities=data.get("commonalities", []),
            generalized_insight=data.get("generalized_insight", ""),
            confidence=data.get("confidence", 0.5),
            domain=data.get("domain", "general")
        )
        abstr.created_at = data.get("created_at", abstr.created_at)
        abstr.access_count = data.get("access_count", 0)
        return abstr


class Abstractor:
    """
    Create abstractions from episodes through pattern detection.

    Capabilities:
    - Cluster similar episodes
    - Extract common patterns
    - Generate generalized insights
    - Identify reusable solutions
    """

    # Common pattern types to detect
    PATTERN_CATEGORIES = {
        "debugging": ["bug", "fix", "error", "issue", "debug", "solved"],
        "configuration": ["config", "setting", "setup", "install", "path"],
        "performance": ["slow", "fast", "optimize", "performance", "timeout"],
        "integration": ["connect", "integrate", "api", "service", "endpoint"],
        "data": ["data", "database", "query", "schema", "migration"]
    }

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.episodic_path = self.base_path / "episodic"
        self.semantic_path = self.base_path / "semantic"
        self.abstractions_path = self.base_path / "consolidation" / "abstractions"
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.abstractions_path.mkdir(parents=True, exist_ok=True)

    def _generate_id(self) -> str:
        """Generate abstraction ID."""
        ts = datetime.now().isoformat()
        return f"abstr_{hashlib.sha256(ts.encode()).hexdigest()[:10]}"

    def _load_episodes(self, limit: int = 100) -> List[Dict]:
        """Load recent episodes for processing."""
        episodes = []

        for ep_file in sorted(self.episodic_path.glob("ep_*.json"), reverse=True)[:limit]:
            try:
                with open(ep_file, 'r', encoding='utf-8') as f:
                    episodes.append(json.load(f))
            except:
                continue

        return episodes

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in", "for", "on", "with"}
        words = set(re.findall(r'\b\w{3,}\b', text.lower()))
        return words - stopwords

    def _calculate_similarity(self, ep1: Dict, ep2: Dict) -> float:
        """Calculate similarity between two episodes."""
        text1 = f"{ep1.get('event', '')} {ep1.get('outcome', '')}"
        text2 = f"{ep2.get('event', '')} {ep2.get('outcome', '')}"

        words1 = self._extract_keywords(text1)
        words2 = self._extract_keywords(text2)

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def cluster_episodes(self, episodes: List[Dict], threshold: float = 0.3) -> List[List[Dict]]:
        """
        Cluster similar episodes together.
        Uses simple greedy clustering based on similarity threshold.
        """
        if not episodes:
            return []

        clusters = []
        used = set()

        for i, ep1 in enumerate(episodes):
            if i in used:
                continue

            cluster = [ep1]
            used.add(i)

            for j, ep2 in enumerate(episodes):
                if j in used or j <= i:
                    continue

                similarity = self._calculate_similarity(ep1, ep2)
                if similarity >= threshold:
                    cluster.append(ep2)
                    used.add(j)

            if len(cluster) >= 2:  # Only keep clusters with 2+ episodes
                clusters.append(cluster)

        return clusters

    def detect_pattern_category(self, cluster: List[Dict]) -> str:
        """Detect the category of a cluster."""
        combined_text = " ".join(
            f"{ep.get('event', '')} {ep.get('outcome', '')}"
            for ep in cluster
        ).lower()

        scores = {}
        for category, keywords in self.PATTERN_CATEGORIES.items():
            score = sum(1 for kw in keywords if kw in combined_text)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def extract_commonalities(self, cluster: List[Dict]) -> List[str]:
        """Extract common elements from a cluster of episodes."""
        if len(cluster) < 2:
            return []

        # Find common words
        word_sets = [
            self._extract_keywords(f"{ep.get('event', '')} {ep.get('outcome', '')}")
            for ep in cluster
        ]

        common_words = word_sets[0]
        for ws in word_sets[1:]:
            common_words = common_words & ws

        commonalities = []

        # Look for common outcomes
        outcomes = [ep.get("outcome", "") for ep in cluster if ep.get("outcome")]
        if outcomes:
            # Find common outcome phrases
            common_outcome_words = set(outcomes[0].lower().split())
            for outcome in outcomes[1:]:
                common_outcome_words &= set(outcome.lower().split())
            if common_outcome_words:
                commonalities.append(f"Common outcome elements: {', '.join(list(common_outcome_words)[:5])}")

        # Look for common actors
        actors_sets = [set(ep.get("actors", [])) for ep in cluster]
        if actors_sets:
            common_actors = actors_sets[0]
            for actor_set in actors_sets[1:]:
                common_actors &= actor_set
            if common_actors:
                commonalities.append(f"Common actors: {', '.join(common_actors)}")

        if common_words:
            commonalities.append(f"Common themes: {', '.join(list(common_words)[:5])}")

        return commonalities

    def generate_abstraction(self, cluster: List[Dict]) -> Optional[Abstraction]:
        """Generate an abstraction from a cluster of similar episodes."""
        if len(cluster) < 2:
            return None

        category = self.detect_pattern_category(cluster)
        commonalities = self.extract_commonalities(cluster)

        # Generate pattern name from common words
        all_events = " ".join(ep.get("event", "") for ep in cluster)
        keywords = list(self._extract_keywords(all_events))[:3]
        pattern_name = f"{category}_{'-'.join(keywords)}" if keywords else f"{category}_pattern"

        # Generate description
        description = f"Pattern observed across {len(cluster)} similar episodes"

        # Generate generalized insight
        outcomes = [ep.get("outcome", "") for ep in cluster if ep.get("outcome")]
        if outcomes:
            generalized_insight = f"When dealing with {category} issues: {outcomes[0][:100]}"
        else:
            generalized_insight = f"Pattern in {category} domain with common themes: {', '.join(keywords)}"

        source_episodes = [ep.get("id", "") for ep in cluster if ep.get("id")]

        abstraction = Abstraction(
            abstraction_id=self._generate_id(),
            pattern_name=pattern_name,
            description=description,
            source_episodes=source_episodes,
            commonalities=commonalities,
            generalized_insight=generalized_insight,
            confidence=min(0.9, 0.5 + (len(cluster) * 0.05)),
            domain=category
        )

        return abstraction

    def save_abstraction(self, abstraction: Abstraction) -> str:
        """Save an abstraction."""
        abstr_file = self.abstractions_path / f"{abstraction.abstraction_id}.json"
        with open(abstr_file, 'w', encoding='utf-8') as f:
            json.dump(abstraction.to_dict(), f, indent=2)
        return abstraction.abstraction_id

    def get_abstraction(self, abstraction_id: str) -> Optional[Abstraction]:
        """Get an abstraction by ID."""
        abstr_file = self.abstractions_path / f"{abstraction_id}.json"
        if not abstr_file.exists():
            return None

        try:
            with open(abstr_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Abstraction.from_dict(data)
        except:
            return None

    def run_abstraction_pass(self, episode_limit: int = 100) -> Dict:
        """
        Run a full abstraction pass over recent episodes.
        Returns statistics about abstractions created.
        """
        episodes = self._load_episodes(limit=episode_limit)
        clusters = self.cluster_episodes(episodes)

        abstractions_created = []
        for cluster in clusters:
            abstraction = self.generate_abstraction(cluster)
            if abstraction:
                self.save_abstraction(abstraction)
                abstractions_created.append({
                    "id": abstraction.abstraction_id,
                    "pattern": abstraction.pattern_name,
                    "source_count": len(abstraction.source_episodes)
                })

        return {
            "episodes_processed": len(episodes),
            "clusters_found": len(clusters),
            "abstractions_created": len(abstractions_created),
            "abstractions": abstractions_created
        }

    def get_all_abstractions(self, domain: str = None) -> List[Dict]:
        """Get all abstractions, optionally filtered by domain."""
        abstractions = []

        for abstr_file in self.abstractions_path.glob("abstr_*.json"):
            try:
                with open(abstr_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if domain and data.get("domain") != domain:
                    continue

                abstractions.append(data)
            except:
                continue

        return abstractions

    def get_stats(self) -> Dict:
        """Get abstraction statistics."""
        total = 0
        by_domain = defaultdict(int)

        for abstr_file in self.abstractions_path.glob("abstr_*.json"):
            try:
                with open(abstr_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                total += 1
                by_domain[data.get("domain", "general")] += 1
            except:
                continue

        return {
            "total_abstractions": total,
            "by_domain": dict(by_domain)
        }
