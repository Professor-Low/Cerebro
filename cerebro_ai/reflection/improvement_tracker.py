"""
Improvement Tracker - Track before/after impact of improvements.

Part of Phase 7 Enhancement in the All-Knowing Brain PRD.
Tracks the impact of specific improvements over time.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ImprovementStatus(Enum):
    """Status of an improvement."""
    MEASURING = "measuring"      # Currently gathering data
    EFFECTIVE = "effective"      # Improvement worked
    INEFFECTIVE = "ineffective"  # Improvement didn't help
    NEUTRAL = "neutral"          # No significant change
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class Improvement:
    """An improvement being tracked."""
    name: str
    description: str
    metric: str
    baseline_value: float
    implemented_at: str
    status: str = "measuring"
    measurement_period_days: int = 7
    current_value: Optional[float] = None
    change_percent: Optional[float] = None
    measured_at: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class ImprovementTracker:
    """
    Tracks the impact of specific improvements.
    Compares before/after metrics to determine effectiveness.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.improvements_path = self.base_path / "improvements.json"

    def _load_improvements(self) -> List[Dict[str, Any]]:
        """Load all tracked improvements."""
        if not self.improvements_path.exists():
            return []

        try:
            with open(self.improvements_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("improvements", [])
        except Exception:
            return []

    def _save_improvements(self, improvements: List[Dict[str, Any]]) -> None:
        """Save improvements to file."""
        try:
            data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "improvements": improvements
            }
            with open(self.improvements_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving improvements: {e}")

    def record_improvement(self,
                           name: str,
                           description: str,
                           metric: str,
                           baseline_value: float,
                           measurement_period_days: int = 7,
                           tags: List[str] = None,
                           notes: str = None) -> Dict[str, Any]:
        """
        Record a new improvement for tracking.

        Args:
            name: Unique name for the improvement
            description: What was improved
            metric: The metric to track (from TRACKED_METRICS)
            baseline_value: The value before improvement
            measurement_period_days: How long to measure (default 7)
            tags: Optional tags for categorization
            notes: Optional notes

        Returns:
            The recorded improvement
        """
        improvement = Improvement(
            name=name,
            description=description,
            metric=metric,
            baseline_value=baseline_value,
            implemented_at=datetime.now().isoformat(),
            measurement_period_days=measurement_period_days,
            tags=tags or [],
            notes=notes
        )

        improvements = self._load_improvements()

        # Check if improvement with this name already exists
        for i, existing in enumerate(improvements):
            if existing.get("name") == name:
                # Update existing
                improvements[i] = asdict(improvement)
                self._save_improvements(improvements)
                return asdict(improvement)

        # Add new
        improvements.append(asdict(improvement))
        self._save_improvements(improvements)

        return asdict(improvement)

    def measure_improvement_impact(self,
                                    improvement_name: str,
                                    force_remeasure: bool = False) -> Dict[str, Any]:
        """
        Measure the impact of an improvement.

        Args:
            improvement_name: Name of the improvement to measure
            force_remeasure: Force remeasurement even if already measured

        Returns:
            Dict with improvement impact analysis
        """
        improvements = self._load_improvements()

        # Find the improvement
        improvement = None
        improvement_idx = None
        for i, imp in enumerate(improvements):
            if imp.get("name") == improvement_name:
                improvement = imp
                improvement_idx = i
                break

        if improvement is None:
            return {
                "error": f"Improvement '{improvement_name}' not found",
                "available_improvements": [i.get("name") for i in improvements]
            }

        # Check if already measured and not forcing
        if improvement.get("measured_at") and not force_remeasure:
            return {
                "improvement": improvement_name,
                "already_measured": True,
                "status": improvement.get("status"),
                "baseline": improvement.get("baseline_value"),
                "current": improvement.get("current_value"),
                "change_percent": improvement.get("change_percent"),
                "measured_at": improvement.get("measured_at"),
            }

        # Get current metric value
        try:
            from .performance_tracker import PerformanceTracker, get_metric_trend  # noqa: F401
        except ImportError:
            from performance_tracker import PerformanceTracker

        metric_name = improvement.get("metric")
        datetime.fromisoformat(improvement.get("implemented_at"))
        measurement_days = improvement.get("measurement_period_days", 7)

        # Get metric trend since implementation
        tracker = PerformanceTracker()
        trend = tracker.get_metric_trend(
            metric_name,
            days=measurement_days
        )

        if not trend.get("has_data"):
            # Try derived metrics
            derived = tracker.calculate_derived_metrics()
            if metric_name in derived:
                current_value = derived[metric_name].get("value", 0)
            else:
                return {
                    "improvement": improvement_name,
                    "status": ImprovementStatus.INSUFFICIENT_DATA.value,
                    "message": f"No data available for metric '{metric_name}'",
                }
        else:
            current_value = trend.get("statistics", {}).get("average", 0)

        baseline = improvement.get("baseline_value", 0)

        # Calculate change
        if baseline != 0:
            change_percent = ((current_value - baseline) / abs(baseline)) * 100
        else:
            change_percent = 100 if current_value > 0 else 0

        # Determine verdict
        # Assume higher is better unless metric indicates otherwise
        try:
            from .performance_tracker import TRACKED_METRICS
        except ImportError:
            from performance_tracker import TRACKED_METRICS
        metric_info = TRACKED_METRICS.get(metric_name, {})
        lower_is_better = metric_info.get("direction") == "lower_is_better"

        if lower_is_better:
            if change_percent < -10:
                status = ImprovementStatus.EFFECTIVE.value
            elif change_percent > 10:
                status = ImprovementStatus.INEFFECTIVE.value
            else:
                status = ImprovementStatus.NEUTRAL.value
        else:
            if change_percent > 10:
                status = ImprovementStatus.EFFECTIVE.value
            elif change_percent < -10:
                status = ImprovementStatus.INEFFECTIVE.value
            else:
                status = ImprovementStatus.NEUTRAL.value

        # Update improvement record
        improvement["current_value"] = round(current_value, 2)
        improvement["change_percent"] = round(change_percent, 2)
        improvement["status"] = status
        improvement["measured_at"] = datetime.now().isoformat()

        improvements[improvement_idx] = improvement
        self._save_improvements(improvements)

        return {
            "improvement": improvement_name,
            "description": improvement.get("description"),
            "metric": metric_name,
            "baseline": baseline,
            "current": round(current_value, 2),
            "change_percent": round(change_percent, 2),
            "status": status,
            "verdict": self._get_verdict_message(status, change_percent),
            "measured_at": improvement["measured_at"],
            "implemented_at": improvement.get("implemented_at"),
        }

    def _get_verdict_message(self, status: str, change_percent: float) -> str:
        """Generate a human-readable verdict message."""
        change_str = f"{'+' if change_percent > 0 else ''}{change_percent:.1f}%"

        if status == ImprovementStatus.EFFECTIVE.value:
            return f"Improvement is working ({change_str})"
        elif status == ImprovementStatus.INEFFECTIVE.value:
            return f"Improvement may have regressed performance ({change_str})"
        elif status == ImprovementStatus.NEUTRAL.value:
            return f"No significant change detected ({change_str})"
        else:
            return "Insufficient data to determine impact"

    def get_all_improvements(self,
                             status_filter: str = None) -> Dict[str, Any]:
        """
        Get all tracked improvements.

        Args:
            status_filter: Optional status to filter by

        Returns:
            Dict with all improvements
        """
        improvements = self._load_improvements()

        if status_filter:
            improvements = [
                i for i in improvements
                if i.get("status") == status_filter
            ]

        # Categorize
        by_status = {
            "measuring": [],
            "effective": [],
            "ineffective": [],
            "neutral": [],
            "insufficient_data": [],
        }

        for imp in improvements:
            status = imp.get("status", "measuring")
            if status in by_status:
                by_status[status].append(imp)

        return {
            "total_improvements": len(improvements),
            "by_status": {k: len(v) for k, v in by_status.items()},
            "improvements": improvements,
            "categories": by_status,
        }

    def generate_improvement_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive improvement report.

        Returns:
            Dict with improvement report
        """
        all_improvements = self.get_all_improvements()
        improvements = all_improvements.get("improvements", [])

        # Measure all improvements that need measuring
        measured = []
        for imp in improvements:
            if imp.get("status") == "measuring":
                result = self.measure_improvement_impact(imp.get("name"))
                measured.append(result)

        # Calculate overall effectiveness
        total = len(improvements)
        effective = len([i for i in improvements if i.get("status") == "effective"])
        ineffective = len([i for i in improvements if i.get("status") == "ineffective"])

        effectiveness_rate = (effective / total * 100) if total > 0 else 0

        # Find most impactful
        improvements_sorted = sorted(
            [i for i in improvements if i.get("change_percent") is not None],
            key=lambda x: abs(x.get("change_percent", 0)),
            reverse=True
        )

        return {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_improvements": total,
                "effective": effective,
                "ineffective": ineffective,
                "neutral": len([i for i in improvements if i.get("status") == "neutral"]),
                "measuring": len([i for i in improvements if i.get("status") == "measuring"]),
                "effectiveness_rate": round(effectiveness_rate, 1),
            },
            "most_impactful": improvements_sorted[:5] if improvements_sorted else [],
            "recently_measured": measured,
            "recommendations": self._generate_recommendations(improvements),
        }

    def _generate_recommendations(self, improvements: List[Dict]) -> List[str]:
        """Generate recommendations based on improvement data."""
        recommendations = []

        # Check for ineffective improvements
        ineffective = [i for i in improvements if i.get("status") == "ineffective"]
        if ineffective:
            recommendations.append(
                f"Review {len(ineffective)} ineffective improvement(s): "
                f"{', '.join(i.get('name', 'unknown') for i in ineffective[:3])}"
            )

        # Check for stale measurements
        now = datetime.now()
        stale = []
        for imp in improvements:
            measured_at = imp.get("measured_at")
            if measured_at:
                measured_dt = datetime.fromisoformat(measured_at)
                if (now - measured_dt).days > 14:
                    stale.append(imp.get("name"))

        if stale:
            recommendations.append(
                f"Consider re-measuring {len(stale)} stale improvement(s) for updated data"
            )

        # Suggest new metrics if few tracked
        if len(improvements) < 3:
            recommendations.append(
                "Consider tracking more improvements to build a comprehensive baseline"
            )

        if not recommendations:
            recommendations.append("System is performing well. Continue monitoring.")

        return recommendations


# Convenience functions

def record_improvement(name: str,
                       description: str,
                       metric: str,
                       baseline_value: float,
                       **kwargs) -> Dict[str, Any]:
    """Record a new improvement for tracking."""
    tracker = ImprovementTracker()
    return tracker.record_improvement(name, description, metric, baseline_value, **kwargs)


def measure_improvement_impact(improvement_name: str,
                                force_remeasure: bool = False) -> Dict[str, Any]:
    """Measure the impact of an improvement."""
    tracker = ImprovementTracker()
    return tracker.measure_improvement_impact(improvement_name, force_remeasure)


if __name__ == "__main__":
    # Test the tracker
    tracker = ImprovementTracker()

    print("=== Improvement Tracker Test ===")

    # Record a test improvement
    print("\nRecording test improvement...")
    result = tracker.record_improvement(
        name="phase4_pattern_validation",
        description="Added pattern validation for smarter context injection",
        metric="context_relevance",
        baseline_value=60.0,
        tags=["phase4", "patterns"],
        notes="Part of All-Knowing Brain PRD Phase 4"
    )
    print(f"Recorded: {result.get('name')}")

    # Get all improvements
    print("\n=== All Improvements ===")
    all_imps = tracker.get_all_improvements()
    print(f"Total: {all_imps['total_improvements']}")
    print(f"By status: {all_imps['by_status']}")

    # Generate report
    print("\n=== Improvement Report ===")
    report = tracker.generate_improvement_report()
    print(f"Summary: {report['summary']}")
    print(f"Recommendations: {report['recommendations']}")
