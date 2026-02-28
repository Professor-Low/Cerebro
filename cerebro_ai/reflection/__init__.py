"""
Reflection Module - Phase 7 of All-Knowing Brain PRD.

Provides:
- PerformanceTracker: Track metrics over time
- ImprovementTracker: Measure before/after impact of improvements
- self_report: Generate comprehensive self-improvement reports
"""

from .improvement_tracker import ImprovementTracker, measure_improvement_impact, record_improvement
from .performance_tracker import TRACKED_METRICS, PerformanceTracker, get_metric_trend, record_metric

__all__ = [
    # Performance tracking
    "PerformanceTracker",
    "record_metric",
    "get_metric_trend",
    "TRACKED_METRICS",
    # Improvement tracking
    "ImprovementTracker",
    "record_improvement",
    "measure_improvement_impact",
]
