"""
Meta Learner - Claude.Me v6.0
Track strategy effectiveness and learn which approaches work best.

Part of Phase 8: Meta-Learning
"""
import hashlib
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


class StrategyRecord:
    """Represents performance data for a search/retrieval strategy."""

    def __init__(
        self,
        strategy_id: str,
        total_queries: int = 0,
        successful_queries: int = 0,
        total_latency_ms: float = 0,
        query_types: Dict[str, int] = None,
        feedback_positive: int = 0,
        feedback_negative: int = 0
    ):
        self.strategy_id = strategy_id
        self.total_queries = total_queries
        self.successful_queries = successful_queries
        self.total_latency_ms = total_latency_ms
        self.query_types = query_types or {}
        self.feedback_positive = feedback_positive
        self.feedback_negative = feedback_negative
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries

    @property
    def avg_latency_ms(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries

    @property
    def feedback_score(self) -> float:
        total_feedback = self.feedback_positive + self.feedback_negative
        if total_feedback == 0:
            return 0.5  # Neutral
        return self.feedback_positive / total_feedback

    @property
    def best_for(self) -> List[str]:
        """Return query types this strategy works best for."""
        if not self.query_types:
            return []
        # Sort by count and return top types
        sorted_types = sorted(self.query_types.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_types[:3]]

    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "success_rate": round(self.success_rate, 3),
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "query_types": self.query_types,
            "feedback_positive": self.feedback_positive,
            "feedback_negative": self.feedback_negative,
            "feedback_score": round(self.feedback_score, 3),
            "best_for": self.best_for,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StrategyRecord":
        record = cls(
            strategy_id=data["strategy_id"],
            total_queries=data.get("total_queries", 0),
            successful_queries=data.get("successful_queries", 0),
            total_latency_ms=data.get("total_latency_ms", 0),
            query_types=data.get("query_types", {}),
            feedback_positive=data.get("feedback_positive", 0),
            feedback_negative=data.get("feedback_negative", 0)
        )
        record.created_at = data.get("created_at", record.created_at)
        record.updated_at = data.get("updated_at", record.updated_at)
        return record


class QueryLog:
    """Log entry for a single query."""

    def __init__(
        self,
        query_id: str,
        query: str,
        strategy_id: str,
        query_type: str,
        latency_ms: float,
        result_count: int,
        success: bool = None,
        feedback: str = None
    ):
        self.query_id = query_id
        self.query = query
        self.strategy_id = strategy_id
        self.query_type = query_type
        self.latency_ms = latency_ms
        self.result_count = result_count
        self.success = success
        self.feedback = feedback  # "positive", "negative", None
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "query": self.query[:100],  # Truncate for storage
            "strategy_id": self.strategy_id,
            "query_type": self.query_type,
            "latency_ms": self.latency_ms,
            "result_count": self.result_count,
            "success": self.success,
            "feedback": self.feedback,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QueryLog":
        log = cls(
            query_id=data["query_id"],
            query=data.get("query", ""),
            strategy_id=data["strategy_id"],
            query_type=data.get("query_type", "unknown"),
            latency_ms=data.get("latency_ms", 0),
            result_count=data.get("result_count", 0),
            success=data.get("success"),
            feedback=data.get("feedback")
        )
        log.timestamp = data.get("timestamp", log.timestamp)
        return log


class MetaLearner:
    """
    Track and learn from retrieval strategy performance.

    Capabilities:
    - Track query performance by strategy
    - Classify query types
    - Record user feedback
    - Recommend optimal strategies
    - Run A/B experiments
    """

    # Query type classification keywords
    QUERY_TYPE_PATTERNS = {
        "technical": ["error", "bug", "fix", "code", "function", "class", "method", "api"],
        "debugging": ["debug", "trace", "issue", "problem", "crash", "fail", "exception"],
        "conceptual": ["what is", "how does", "explain", "why", "concept", "understand"],
        "factual": ["when", "where", "who", "which", "list", "find"],
        "procedural": ["how to", "steps", "process", "guide", "tutorial"],
        "configuration": ["config", "setting", "setup", "install", "path", "port"]
    }

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.meta_path = self.base_path / "meta_learning"
        self.strategies_file = self.meta_path / "strategies.json"
        self.logs_path = self.meta_path / "query_logs"
        self._lock = threading.Lock()
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.meta_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

    def _generate_query_id(self, query: str) -> str:
        """Generate unique query ID."""
        ts = datetime.now().isoformat()
        return f"q_{hashlib.sha256(f'{query}{ts}'.encode()).hexdigest()[:10]}"

    def _load_strategies(self) -> Dict:
        """Load strategies data."""
        if self.strategies_file.exists():
            try:
                with open(self.strategies_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            "strategies": {},
            "current_default": "hybrid_alpha_0.7",
            "experiments": [],
            "updated_at": datetime.now().isoformat()
        }

    def _save_strategies(self, data: Dict):
        """Save strategies data."""
        data["updated_at"] = datetime.now().isoformat()
        with self._lock:
            with open(self.strategies_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

    def classify_query_type(self, query: str) -> str:
        """Classify a query into a type based on keywords."""
        query_lower = query.lower()

        scores = {}
        for qtype, keywords in self.QUERY_TYPE_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[qtype] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def record_query(
        self,
        query: str,
        strategy_id: str,
        latency_ms: float,
        result_count: int,
        success: bool = None
    ) -> str:
        """
        Record a query execution.

        Args:
            query: The search query
            strategy_id: Which strategy was used (e.g., "hybrid_alpha_0.7")
            latency_ms: Query execution time
            result_count: Number of results returned
            success: Whether retrieval was successful (optional, can be set later)

        Returns:
            Query ID for later feedback
        """
        query_type = self.classify_query_type(query)
        query_id = self._generate_query_id(query)

        log = QueryLog(
            query_id=query_id,
            query=query,
            strategy_id=strategy_id,
            query_type=query_type,
            latency_ms=latency_ms,
            result_count=result_count,
            success=success
        )

        # Save query log
        log_file = self.logs_path / f"{query_id}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log.to_dict(), f, indent=2)

        # Update strategy stats
        self._update_strategy_stats(strategy_id, query_type, latency_ms, success)

        return query_id

    def _update_strategy_stats(
        self,
        strategy_id: str,
        query_type: str,
        latency_ms: float,
        success: bool = None
    ):
        """Update strategy statistics."""
        data = self._load_strategies()

        if strategy_id not in data["strategies"]:
            data["strategies"][strategy_id] = StrategyRecord(strategy_id).to_dict()

        strat = data["strategies"][strategy_id]
        strat["total_queries"] = strat.get("total_queries", 0) + 1
        strat["total_latency_ms"] = strat.get("total_latency_ms", 0) + latency_ms

        if success is True:
            strat["successful_queries"] = strat.get("successful_queries", 0) + 1

        # Track query types
        if "query_types" not in strat:
            strat["query_types"] = {}
        strat["query_types"][query_type] = strat["query_types"].get(query_type, 0) + 1

        strat["updated_at"] = datetime.now().isoformat()

        # Recalculate derived fields
        if strat["total_queries"] > 0:
            strat["success_rate"] = round(strat.get("successful_queries", 0) / strat["total_queries"], 3)
            strat["avg_latency_ms"] = round(strat["total_latency_ms"] / strat["total_queries"], 2)

        self._save_strategies(data)

    def record_feedback(self, query_id: str, positive: bool) -> Dict:
        """
        Record user feedback for a query.

        Args:
            query_id: The query ID from record_query
            positive: True for positive feedback, False for negative
        """
        log_file = self.logs_path / f"{query_id}.json"
        if not log_file.exists():
            return {"error": f"Query {query_id} not found"}

        with open(log_file, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        log_data["feedback"] = "positive" if positive else "negative"
        log_data["success"] = positive

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)

        # Update strategy feedback stats
        strategy_id = log_data.get("strategy_id")
        if strategy_id:
            data = self._load_strategies()
            if strategy_id in data["strategies"]:
                strat = data["strategies"][strategy_id]
                if positive:
                    strat["feedback_positive"] = strat.get("feedback_positive", 0) + 1
                    strat["successful_queries"] = strat.get("successful_queries", 0) + 1
                else:
                    strat["feedback_negative"] = strat.get("feedback_negative", 0) + 1

                # Recalculate feedback score
                total_feedback = strat.get("feedback_positive", 0) + strat.get("feedback_negative", 0)
                if total_feedback > 0:
                    strat["feedback_score"] = round(strat.get("feedback_positive", 0) / total_feedback, 3)

                self._save_strategies(data)

        return {
            "query_id": query_id,
            "feedback": "positive" if positive else "negative",
            "recorded": True
        }

    def get_strategy_stats(self, strategy_id: str = None) -> Dict:
        """Get statistics for a strategy or all strategies."""
        data = self._load_strategies()

        if strategy_id:
            if strategy_id in data["strategies"]:
                return data["strategies"][strategy_id]
            return {"error": f"Strategy {strategy_id} not found"}

        return {
            "strategies": data["strategies"],
            "current_default": data.get("current_default"),
            "total_strategies": len(data["strategies"])
        }

    def recommend_strategy(self, query: str) -> Dict:
        """
        Recommend the best strategy for a given query.

        Returns strategy recommendation based on query type and historical performance.
        """
        query_type = self.classify_query_type(query)
        data = self._load_strategies()

        candidates = []
        for strat_id, strat in data.get("strategies", {}).items():
            # Calculate a composite score
            success_rate = strat.get("success_rate", 0.5)
            feedback_score = strat.get("feedback_score", 0.5)
            query_type_count = strat.get("query_types", {}).get(query_type, 0)

            # Weight by query type experience
            experience_bonus = min(0.2, query_type_count * 0.01)

            composite_score = (success_rate * 0.4) + (feedback_score * 0.4) + experience_bonus

            candidates.append({
                "strategy_id": strat_id,
                "score": round(composite_score, 3),
                "success_rate": success_rate,
                "feedback_score": feedback_score,
                "experience_for_type": query_type_count
            })

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        if candidates:
            return {
                "recommended": candidates[0]["strategy_id"],
                "query_type": query_type,
                "score": candidates[0]["score"],
                "alternatives": candidates[1:3] if len(candidates) > 1 else []
            }

        # Default recommendation
        return {
            "recommended": data.get("current_default", "hybrid_alpha_0.7"),
            "query_type": query_type,
            "score": 0.5,
            "reason": "No strategy data available"
        }

    def set_default_strategy(self, strategy_id: str) -> Dict:
        """Set the default strategy."""
        data = self._load_strategies()
        data["current_default"] = strategy_id
        self._save_strategies(data)
        return {"success": True, "default_strategy": strategy_id}

    def get_recent_queries(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        """Get recent query logs."""
        cutoff = datetime.now() - timedelta(hours=hours)
        queries = []

        for log_file in sorted(self.logs_path.glob("q_*.json"), reverse=True):
            if len(queries) >= limit:
                break

            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log = json.load(f)

                log_time = datetime.fromisoformat(log.get("timestamp", ""))
                if log_time >= cutoff:
                    queries.append(log)
            except:
                continue

        return queries

    def get_stats(self) -> Dict:
        """Get overall meta-learning statistics."""
        data = self._load_strategies()

        total_queries = sum(
            s.get("total_queries", 0)
            for s in data.get("strategies", {}).values()
        )

        avg_success = 0
        if data.get("strategies"):
            rates = [s.get("success_rate", 0) for s in data["strategies"].values()]
            avg_success = sum(rates) / len(rates)

        return {
            "total_strategies": len(data.get("strategies", {})),
            "total_queries_tracked": total_queries,
            "average_success_rate": round(avg_success, 3),
            "current_default": data.get("current_default"),
            "active_experiments": len(data.get("experiments", []))
        }
