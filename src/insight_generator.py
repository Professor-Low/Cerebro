"""
Insight Generator - Cerebro AI Memory
Detect patterns and generate hypotheses from memory.

Part of Phase 4: Active Reasoning Over Memory
"""
import hashlib
import json
import re
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class InsightType:
    """Types of insights that can be generated."""
    PATTERN = "pattern"           # Recurring pattern detected
    CONNECTION = "connection"     # Connection between concepts
    GAP = "gap"                   # Missing knowledge detected
    ANOMALY = "anomaly"           # Something doesn't fit pattern
    INFERENCE = "inference"       # Logical deduction
    PREDICTION = "prediction"     # Predicted outcome


class Insight:
    """
    A generated insight - knowledge derived from reasoning over memories.

    Example:
    - Type: pattern
    - Content: "80% of MCP debugging happens 23:00-02:00"
    - Evidence: 25 memories analyzed, 20 match pattern
    - Novelty: 0.75 (fairly new insight)
    - Usefulness: 0.60 (moderately useful)
    """

    def __init__(
        self,
        insight_id: str,
        insight_type: str,
        content: str,
        evidence: Dict[str, Any] = None,
        novelty_score: float = 0.5,
        usefulness_score: float = 0.5,
        confidence: float = 0.6,
        related_memory_ids: List[str] = None
    ):
        self.insight_id = insight_id
        self.insight_type = insight_type
        self.content = content
        self.evidence = evidence or {}
        self.novelty_score = novelty_score
        self.usefulness_score = usefulness_score
        self.confidence = confidence
        self.related_memory_ids = related_memory_ids or []
        self.status = "hypothesis"  # hypothesis, validated, refuted
        self.validated_at = None
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "insight_id": self.insight_id,
            "type": self.insight_type,
            "content": self.content,
            "evidence": self.evidence,
            "novelty_score": self.novelty_score,
            "usefulness_score": self.usefulness_score,
            "confidence": self.confidence,
            "related_memory_ids": self.related_memory_ids,
            "status": self.status,
            "validated_at": self.validated_at,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Insight":
        insight = cls(
            insight_id=data["insight_id"],
            insight_type=data["type"],
            content=data["content"],
            evidence=data.get("evidence", {}),
            novelty_score=data.get("novelty_score", 0.5),
            usefulness_score=data.get("usefulness_score", 0.5),
            confidence=data.get("confidence", 0.6),
            related_memory_ids=data.get("related_memory_ids", [])
        )
        insight.status = data.get("status", "hypothesis")
        insight.validated_at = data.get("validated_at")
        insight.created_at = data.get("created_at", datetime.now().isoformat())
        return insight

    def validate(self):
        """Mark insight as validated."""
        self.status = "validated"
        self.validated_at = datetime.now().isoformat()
        self.confidence = min(1.0, self.confidence + 0.2)

    def refute(self):
        """Mark insight as refuted."""
        self.status = "refuted"
        self.confidence = 0.0


class InsightGenerator:
    """
    Generate insights by analyzing memories.

    Capabilities:
    - Detect recurring patterns across memories
    - Find non-obvious connections
    - Identify knowledge gaps
    - Detect anomalies
    - Generate inferences
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.insights_path = self.base_path / "insights"
        self._lock = threading.Lock()
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.insights_path.mkdir(parents=True, exist_ok=True)

    def _generate_id(self, content: str) -> str:
        """Generate insight ID."""
        ts = datetime.now().isoformat()
        return f"ins_{hashlib.sha256(f'{content}{ts}'.encode()).hexdigest()[:10]}"

    def save_insight(self, insight: Insight) -> str:
        """Save an insight to disk."""
        insight_file = self.insights_path / f"{insight.insight_id}.json"
        with open(insight_file, 'w', encoding='utf-8') as f:
            json.dump(insight.to_dict(), f, indent=2)
        return insight.insight_id

    def get_insight(self, insight_id: str) -> Optional[Insight]:
        """Get an insight by ID."""
        insight_file = self.insights_path / f"{insight_id}.json"
        if not insight_file.exists():
            return None

        try:
            with open(insight_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Insight.from_dict(data)
        except:
            return None

    def detect_temporal_patterns(self, conversations: List[Dict]) -> List[Insight]:
        """
        Detect time-based patterns in conversation data.

        Examples:
        - "Most debugging happens at night"
        - "MCP issues cluster on weekends"
        """
        insights = []
        hour_counts = defaultdict(int)
        day_counts = defaultdict(int)
        topic_times = defaultdict(list)

        for conv in conversations:
            timestamp = conv.get("timestamp")
            if not timestamp:
                continue

            try:
                dt = datetime.fromisoformat(timestamp)
                hour_counts[dt.hour] += 1
                day_counts[dt.weekday()] += 1

                # Track topic timing
                topics = conv.get("metadata", {}).get("topics", [])
                for topic in topics:
                    topic_times[topic].append(dt.hour)
            except:
                continue

        # Analyze hour distribution
        total_convs = sum(hour_counts.values())
        if total_convs >= 10:
            # Check for night owl pattern (22:00-03:00)
            night_hours = sum(hour_counts.get(h, 0) for h in [22, 23, 0, 1, 2, 3])
            if night_hours / total_convs >= 0.4:
                insight = Insight(
                    insight_id=self._generate_id("night_work"),
                    insight_type=InsightType.PATTERN,
                    content=f"User works late: {int(night_hours/total_convs*100)}% of sessions are between 22:00-03:00",
                    evidence={
                        "total_conversations": total_convs,
                        "night_conversations": night_hours,
                        "percentage": round(night_hours / total_convs * 100, 1)
                    },
                    novelty_score=0.6,
                    usefulness_score=0.5,
                    confidence=0.7 if total_convs >= 20 else 0.5
                )
                insights.append(insight)

        # Analyze topic timing patterns
        for topic, hours in topic_times.items():
            if len(hours) >= 5:
                avg_hour = sum(hours) / len(hours)
                if avg_hour >= 22 or avg_hour <= 3:
                    insight = Insight(
                        insight_id=self._generate_id(f"topic_time_{topic}"),
                        insight_type=InsightType.PATTERN,
                        content=f"'{topic}' work typically happens at night (avg hour: {int(avg_hour):02d}:00)",
                        evidence={
                            "topic": topic,
                            "sample_size": len(hours),
                            "average_hour": round(avg_hour, 1)
                        },
                        novelty_score=0.7,
                        usefulness_score=0.4,
                        confidence=0.6
                    )
                    insights.append(insight)

        return insights

    def detect_problem_patterns(self, conversations: List[Dict]) -> List[Insight]:
        """
        Detect patterns in problems/solutions.

        Examples:
        - "Timeout issues are usually fixed by increasing timeout"
        - "MCP problems often have the same root cause"
        """
        insights = []
        problem_solutions = defaultdict(list)

        for conv in conversations:
            problems = conv.get("extracted_data", {}).get("problems_solved", [])
            for prob in problems:
                problem_text = prob.get("problem", "").lower()
                solution_text = prob.get("solution", "")

                # Extract keywords from problem
                keywords = self._extract_keywords(problem_text)
                for kw in keywords:
                    if solution_text and solution_text != "Not explicitly stated":
                        problem_solutions[kw].append(solution_text)

        # Find recurring solution patterns
        for keyword, solutions in problem_solutions.items():
            if len(solutions) >= 3:
                # Check for common solution patterns
                solution_patterns = self._find_common_patterns(solutions)
                for pattern, count in solution_patterns.items():
                    if count >= 2:
                        insight = Insight(
                            insight_id=self._generate_id(f"solution_{keyword}"),
                            insight_type=InsightType.PATTERN,
                            content=f"'{keyword}' problems are often solved by: {pattern}",
                            evidence={
                                "keyword": keyword,
                                "solution_pattern": pattern,
                                "occurrences": count,
                                "total_problems": len(solutions)
                            },
                            novelty_score=0.7,
                            usefulness_score=0.8,
                            confidence=0.6 + (0.05 * min(count, 6))
                        )
                        insights.append(insight)

        return insights

    def find_connections(self, memory_a: Dict, memory_b: Dict) -> Optional[Insight]:
        """
        Find connections between two memories.

        Returns an insight if a meaningful connection is found.
        """
        # Extract keywords from both
        text_a = memory_a.get("content", "") or memory_a.get("fact", "")
        text_b = memory_b.get("content", "") or memory_b.get("fact", "")

        keywords_a = set(self._extract_keywords(text_a.lower()))
        keywords_b = set(self._extract_keywords(text_b.lower()))

        # Find shared keywords
        shared = keywords_a.intersection(keywords_b)

        if len(shared) >= 2:
            insight = Insight(
                insight_id=self._generate_id(f"connection_{memory_a.get('id', '')}_{memory_b.get('id', '')}"),
                insight_type=InsightType.CONNECTION,
                content=f"Connection found: both memories relate to {', '.join(list(shared)[:3])}",
                evidence={
                    "memory_a_id": memory_a.get("id"),
                    "memory_b_id": memory_b.get("id"),
                    "shared_concepts": list(shared)
                },
                novelty_score=0.5,
                usefulness_score=0.5,
                confidence=0.5 + (0.05 * min(len(shared), 5)),
                related_memory_ids=[memory_a.get("id"), memory_b.get("id")]
            )
            return insight

        return None

    def detect_knowledge_gaps(self, queries: List[str], memories: List[Dict]) -> List[Insight]:
        """
        Detect gaps in knowledge based on queries vs available memories.

        If we're asked about something frequently but have little memory of it,
        that's a knowledge gap.
        """
        insights = []

        # Count query topics
        query_topics = defaultdict(int)
        for query in queries:
            keywords = self._extract_keywords(query.lower())
            for kw in keywords:
                query_topics[kw] += 1

        # Count memory coverage
        memory_topics = defaultdict(int)
        for mem in memories:
            text = mem.get("content", "") or mem.get("fact", "")
            keywords = self._extract_keywords(text.lower())
            for kw in keywords:
                memory_topics[kw] += 1

        # Find gaps: frequently queried but rarely covered
        for topic, query_count in query_topics.items():
            if query_count >= 3:  # Asked about at least 3 times
                memory_count = memory_topics.get(topic, 0)
                if memory_count <= 1:  # But we have 1 or fewer memories
                    insight = Insight(
                        insight_id=self._generate_id(f"gap_{topic}"),
                        insight_type=InsightType.GAP,
                        content=f"Knowledge gap detected: '{topic}' queried {query_count} times but only {memory_count} memories exist",
                        evidence={
                            "topic": topic,
                            "query_count": query_count,
                            "memory_count": memory_count,
                            "gap_severity": query_count - memory_count
                        },
                        novelty_score=0.8,
                        usefulness_score=0.9,
                        confidence=0.7
                    )
                    insights.append(insight)

        return insights

    def generate_inference(self, premise_a: str, premise_b: str) -> Optional[Insight]:
        """
        Generate a logical inference from two premises.

        Simple pattern matching for now - could be enhanced with DGX LLM.
        """
        # Look for transitive patterns
        # If A causes B and B causes C, then A causes C
        patterns = [
            (r"(.+) causes (.+)", r"(.+) causes (.+)", "transitive_causation"),
            (r"(.+) requires (.+)", r"(.+) requires (.+)", "transitive_requirement"),
            (r"(.+) is (.+)", r"(.+) is (.+)", "transitive_property"),
        ]

        for pattern_a, pattern_b, inference_type in patterns:
            match_a = re.search(pattern_a, premise_a, re.IGNORECASE)
            match_b = re.search(pattern_b, premise_b, re.IGNORECASE)

            if match_a and match_b:
                # Check for transitive link
                if match_a.group(2).lower() == match_b.group(1).lower():
                    conclusion = f"{match_a.group(1)} {inference_type.split('_')[1]} {match_b.group(2)}"
                    insight = Insight(
                        insight_id=self._generate_id(f"inference_{conclusion}"),
                        insight_type=InsightType.INFERENCE,
                        content=f"Inferred: {conclusion}",
                        evidence={
                            "premise_a": premise_a[:100],
                            "premise_b": premise_b[:100],
                            "inference_type": inference_type
                        },
                        novelty_score=0.8,
                        usefulness_score=0.7,
                        confidence=0.5
                    )
                    return insight

        return None

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "that", "which", "who", "whom", "this", "these", "those", "it", "its",
            "and", "or", "but", "so", "yet", "not", "no"
        }

        words = re.findall(r'[a-zA-Z][a-zA-Z0-9_-]*', text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return keywords

    def _find_common_patterns(self, texts: List[str]) -> Dict[str, int]:
        """Find common patterns in a list of texts."""
        patterns = defaultdict(int)

        for text in texts:
            text_lower = text.lower()
            # Check for action patterns
            action_patterns = [
                r'increase[d]?\s+\w+',
                r'decrease[d]?\s+\w+',
                r'chang[ed]+\s+\w+',
                r'add[ed]?\s+\w+',
                r'remov[ed]+\s+\w+',
                r'restart[ed]?\s+\w+',
                r'updat[ed]+\s+\w+',
            ]
            for pattern in action_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    patterns[match.group(0)] += 1

        return dict(patterns)

    def get_recent_insights(self, days: int = 7, limit: int = 20) -> List[Insight]:
        """Get recently generated insights."""
        cutoff = datetime.now() - timedelta(days=days)
        insights = []

        for insight_file in self.insights_path.glob("ins_*.json"):
            try:
                with open(insight_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                created = data.get("created_at", "")
                if created:
                    created_dt = datetime.fromisoformat(created)
                    if created_dt >= cutoff:
                        insights.append(Insight.from_dict(data))
            except:
                continue

        insights.sort(key=lambda i: i.created_at, reverse=True)
        return insights[:limit]

    def get_high_value_insights(self, min_usefulness: float = 0.7) -> List[Insight]:
        """Get high-value insights based on usefulness score."""
        insights = []

        for insight_file in self.insights_path.glob("ins_*.json"):
            try:
                with open(insight_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if data.get("usefulness_score", 0) >= min_usefulness:
                    if data.get("status") != "refuted":
                        insights.append(Insight.from_dict(data))
            except:
                continue

        insights.sort(key=lambda i: i.usefulness_score, reverse=True)
        return insights

    def get_stats(self) -> Dict:
        """Get insight statistics."""
        total = 0
        by_type = defaultdict(int)
        by_status = defaultdict(int)
        avg_usefulness = 0
        avg_confidence = 0

        for insight_file in self.insights_path.glob("ins_*.json"):
            try:
                with open(insight_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                total += 1
                by_type[data.get("type", "unknown")] += 1
                by_status[data.get("status", "unknown")] += 1
                avg_usefulness += data.get("usefulness_score", 0)
                avg_confidence += data.get("confidence", 0)
            except:
                continue

        return {
            "total_insights": total,
            "by_type": dict(by_type),
            "by_status": dict(by_status),
            "avg_usefulness": round(avg_usefulness / max(total, 1), 2),
            "avg_confidence": round(avg_confidence / max(total, 1), 2)
        }
