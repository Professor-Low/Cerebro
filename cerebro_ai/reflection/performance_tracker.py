"""
Performance Tracker - Track concrete metrics over time.

Part of Phase 7 Enhancement in the All-Knowing Brain PRD.
Tracks:
- Correction rate (how often user corrects Claude)
- Suggestion acceptance (how often suggestions are used)
- Context relevance (was injected context useful)
- Task completion (tasks completed vs abandoned)
- Search efficiency (searches that found what was needed)
"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Metrics we track
TRACKED_METRICS = {
    "correction_rate": {
        "description": "How often user corrects Claude",
        "unit": "percent",
        "direction": "lower_is_better",
    },
    "suggestion_acceptance": {
        "description": "How often suggestions are accepted/used",
        "unit": "percent",
        "direction": "higher_is_better",
    },
    "context_relevance": {
        "description": "Was injected context actually used",
        "unit": "percent",
        "direction": "higher_is_better",
    },
    "task_completion": {
        "description": "Tasks completed vs abandoned",
        "unit": "percent",
        "direction": "higher_is_better",
    },
    "search_efficiency": {
        "description": "Searches that found what was needed",
        "unit": "percent",
        "direction": "higher_is_better",
    },
    "session_continuation_rate": {
        "description": "Sessions that were successfully continued",
        "unit": "percent",
        "direction": "higher_is_better",
    },
    "pattern_injection_count": {
        "description": "Patterns injected per session",
        "unit": "count",
        "direction": "neutral",
    },
    "response_helpfulness": {
        "description": "User satisfaction with responses (implicit)",
        "unit": "score",
        "direction": "higher_is_better",
    },
}


@dataclass
class MetricRecord:
    """A single metric measurement."""
    metric_name: str
    value: float
    timestamp: str
    context: Optional[str] = None
    session_id: Optional[str] = None
    tags: Optional[List[str]] = None


class PerformanceTracker:
    """
    Tracks performance metrics over time.
    Stores data in JSON files on NAS for persistence.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.metrics_path = self.base_path / "metrics"
        self.metrics_path.mkdir(parents=True, exist_ok=True)

        # Index file for quick lookups
        self.index_path = self.metrics_path / "metrics_index.json"

    def record_metric(self,
                      metric_name: str,
                      value: float,
                      context: str = None,
                      session_id: str = None,
                      tags: List[str] = None) -> bool:
        """
        Record a metric measurement.

        Args:
            metric_name: Name of the metric (from TRACKED_METRICS)
            value: The measurement value
            context: Optional context about the measurement
            session_id: Optional session identifier
            tags: Optional tags for categorization

        Returns:
            True if recorded successfully
        """
        if metric_name not in TRACKED_METRICS:
            # Allow custom metrics too
            pass

        record = MetricRecord(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now().isoformat(),
            context=context,
            session_id=session_id,
            tags=tags or []
        )

        # Store in daily file
        date_str = datetime.now().strftime("%Y-%m-%d")
        daily_file = self.metrics_path / f"metrics_{date_str}.json"

        try:
            # Load existing records for today
            records = []
            if daily_file.exists():
                with open(daily_file, "r", encoding="utf-8") as f:
                    records = json.load(f)

            # Add new record
            records.append(asdict(record))

            # Save
            with open(daily_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

            # Update index
            self._update_index(metric_name, date_str)

            return True

        except Exception as e:
            print(f"Error recording metric: {e}")
            return False

    def _update_index(self, metric_name: str, date_str: str) -> None:
        """Update the metrics index for quick lookups."""
        try:
            index = {}
            if self.index_path.exists():
                with open(self.index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)

            # Update metric entry
            if metric_name not in index:
                index[metric_name] = {
                    "first_recorded": date_str,
                    "last_recorded": date_str,
                    "dates": [date_str],
                    "total_records": 1
                }
            else:
                if date_str not in index[metric_name]["dates"]:
                    index[metric_name]["dates"].append(date_str)
                index[metric_name]["last_recorded"] = date_str
                index[metric_name]["total_records"] += 1

            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)

        except Exception:
            pass  # Index is optional

    def get_metric_records(self,
                           metric_name: str,
                           days: int = 30,
                           start_date: datetime = None,
                           end_date: datetime = None) -> List[MetricRecord]:
        """
        Get records for a specific metric.

        Args:
            metric_name: Name of the metric
            days: Number of days to look back (default 30)
            start_date: Optional start date (overrides days)
            end_date: Optional end date

        Returns:
            List of MetricRecords
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        if end_date is None:
            end_date = datetime.now()

        records = []

        # Iterate through daily files
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            daily_file = self.metrics_path / f"metrics_{date_str}.json"

            if daily_file.exists():
                try:
                    with open(daily_file, "r", encoding="utf-8") as f:
                        day_records = json.load(f)

                    for r in day_records:
                        if r.get("metric_name") == metric_name:
                            records.append(MetricRecord(**r))
                except Exception:
                    pass

            current += timedelta(days=1)

        return records

    def get_metric_trend(self,
                         metric_name: str,
                         days: int = 30) -> Dict[str, Any]:
        """
        Calculate trend for a metric.

        Args:
            metric_name: Name of the metric
            days: Number of days to analyze

        Returns:
            Dict with trend analysis
        """
        records = self.get_metric_records(metric_name, days=days)

        if not records:
            return {
                "metric": metric_name,
                "has_data": False,
                "message": f"No data for {metric_name} in last {days} days"
            }

        # Group by date
        daily_values = defaultdict(list)
        for record in records:
            date = record.timestamp[:10]  # YYYY-MM-DD
            daily_values[date].append(record.value)

        # Calculate daily averages
        daily_averages = {
            date: sum(vals) / len(vals)
            for date, vals in sorted(daily_values.items())
        }

        values = list(daily_averages.values())
        list(daily_averages.keys())

        # Calculate statistics
        avg = sum(values) / len(values) if values else 0
        min_val = min(values) if values else 0
        max_val = max(values) if values else 0

        # Calculate trend (simple linear regression slope)
        trend_direction = "stable"
        trend_slope = 0.0

        if len(values) >= 2:
            n = len(values)
            x_mean = (n - 1) / 2
            y_mean = avg

            numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            if denominator > 0:
                trend_slope = numerator / denominator

                # Determine direction based on significance
                if abs(trend_slope) > avg * 0.05:  # 5% of average
                    trend_direction = "improving" if trend_slope > 0 else "declining"
                    # Flip for metrics where lower is better
                    metric_info = TRACKED_METRICS.get(metric_name, {})
                    if metric_info.get("direction") == "lower_is_better":
                        trend_direction = "declining" if trend_slope > 0 else "improving"

        return {
            "metric": metric_name,
            "has_data": True,
            "period_days": days,
            "data_points": len(records),
            "unique_days": len(daily_averages),
            "statistics": {
                "average": round(avg, 2),
                "min": round(min_val, 2),
                "max": round(max_val, 2),
                "latest": round(values[-1], 2) if values else None,
            },
            "trend": {
                "direction": trend_direction,
                "slope": round(trend_slope, 4),
            },
            "daily_data": daily_averages,
            "description": TRACKED_METRICS.get(metric_name, {}).get("description", ""),
        }

    def get_all_metrics_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get summary of all tracked metrics.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with all metrics summaries
        """
        # Find all metrics with data
        metrics_with_data = set()

        # Check index first
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                metrics_with_data.update(index.keys())
            except:
                pass

        # Also scan recent files
        for i in range(min(days, 7)):  # Scan last week
            date_str = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_file = self.metrics_path / f"metrics_{date_str}.json"
            if daily_file.exists():
                try:
                    with open(daily_file, "r", encoding="utf-8") as f:
                        records = json.load(f)
                    for r in records:
                        metrics_with_data.add(r.get("metric_name", ""))
                except:
                    pass

        # Get trends for all metrics
        summaries = {}
        for metric_name in metrics_with_data:
            if metric_name:
                summaries[metric_name] = self.get_metric_trend(metric_name, days)

        # Add metrics without data
        for metric_name in TRACKED_METRICS:
            if metric_name not in summaries:
                summaries[metric_name] = {
                    "metric": metric_name,
                    "has_data": False,
                    "description": TRACKED_METRICS[metric_name].get("description", ""),
                }

        return {
            "period_days": days,
            "generated_at": datetime.now().isoformat(),
            "metrics_with_data": len([m for m in summaries.values() if m.get("has_data")]),
            "total_metrics_tracked": len(TRACKED_METRICS),
            "summaries": summaries,
        }

    def calculate_derived_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics derived from conversation data.
        These are computed from existing data rather than explicit recording.
        """
        derived = {}

        try:
            # Correction rate from corrections data
            corrections_path = self.base_path / "corrections.json"
            if corrections_path.exists():
                with open(corrections_path, "r", encoding="utf-8") as f:
                    corrections = json.load(f)
                total_corrections = len(corrections.get("corrections", []))

                # Count conversations
                conv_path = self.base_path / "conversations"
                total_convs = len(list(conv_path.glob("*.json"))) if conv_path.exists() else 0

                if total_convs > 0:
                    derived["correction_rate"] = {
                        "value": round((total_corrections / total_convs) * 100, 1),
                        "unit": "percent",
                        "description": f"{total_corrections} corrections across {total_convs} conversations",
                    }

            # Session continuation rate from session data
            sessions_analyzed = 0
            sessions_continued = 0

            conv_path = self.base_path / "conversations"
            if conv_path.exists():
                for conv_file in list(conv_path.glob("*.json"))[:100]:
                    try:
                        with open(conv_file, "r", encoding="utf-8") as f:
                            conv = json.load(f)
                        if conv.get("continued_from"):
                            sessions_continued += 1
                        sessions_analyzed += 1
                    except:
                        pass

            if sessions_analyzed > 0:
                derived["session_continuation_rate"] = {
                    "value": round((sessions_continued / sessions_analyzed) * 100, 1),
                    "unit": "percent",
                    "description": f"{sessions_continued} continued out of {sessions_analyzed} sessions",
                }

            # Learning system usage
            learnings_path = self.base_path / "learnings"
            if learnings_path.exists():
                solutions = list(learnings_path.glob("solution_*.json"))
                antipatterns = list(learnings_path.glob("antipattern_*.json"))
                derived["learning_system_usage"] = {
                    "solutions_recorded": len(solutions),
                    "antipatterns_recorded": len(antipatterns),
                    "total_learnings": len(solutions) + len(antipatterns),
                }

        except Exception as e:
            derived["error"] = str(e)

        return derived


# Convenience functions

def record_metric(metric_name: str,
                  value: float,
                  context: str = None,
                  session_id: str = None) -> bool:
    """Record a metric measurement."""
    tracker = PerformanceTracker()
    return tracker.record_metric(metric_name, value, context, session_id)


def get_metric_trend(metric_name: str, days: int = 30) -> Dict[str, Any]:
    """Get trend for a specific metric."""
    tracker = PerformanceTracker()
    return tracker.get_metric_trend(metric_name, days)


if __name__ == "__main__":
    # Test the tracker
    tracker = PerformanceTracker()

    print("=== Performance Tracker Test ===")

    # Record some test metrics
    print("\nRecording test metrics...")
    tracker.record_metric("suggestion_acceptance", 75.0, context="test session")
    tracker.record_metric("context_relevance", 80.0, context="test session")
    tracker.record_metric("task_completion", 90.0, context="test session")

    print("\n=== Derived Metrics ===")
    derived = tracker.calculate_derived_metrics()
    for key, value in derived.items():
        print(f"  {key}: {value}")

    print("\n=== All Metrics Summary ===")
    summary = tracker.get_all_metrics_summary(days=7)
    print(f"Metrics with data: {summary['metrics_with_data']}")

    for name, data in summary["summaries"].items():
        if data.get("has_data"):
            stats = data.get("statistics", {})
            trend = data.get("trend", {})
            print(f"  {name}: avg={stats.get('average')}, trend={trend.get('direction')}")
