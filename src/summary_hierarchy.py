"""
Summary Hierarchy - Roll up conversation summaries over time.

Implements daily -> weekly -> monthly summary compression to reduce storage
while preserving key facts and learnings.

Phase 3.4 of Brain Evolution.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List


class SummaryHierarchy:
    """
    Manages hierarchical summary rollups:
    - Daily: Full summaries for last 7 days
    - Weekly: Compressed summaries for last 4 weeks
    - Monthly: Minimal summaries for older data
    """

    def __init__(self, base_path: str = ""):
        if not base_path:
            from config import DATA_DIR
            base_path = str(DATA_DIR)
        self.base_path = Path(base_path)
        self.summaries_path = self.base_path / "temporal" / "session_summaries.jsonl"
        self.hierarchy_path = self.base_path / "temporal" / "summary_hierarchy.json"
        self.temporal_path = self.base_path / "temporal"

        self.temporal_path.mkdir(parents=True, exist_ok=True)

    def load_summaries(self) -> List[Dict]:
        """Load all session summaries."""
        summaries = []
        if self.summaries_path.exists():
            try:
                with open(self.summaries_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                summaries.append(json.loads(line))
                            except:
                                continue
            except Exception as e:
                print(f"Error loading summaries: {e}")
        return summaries

    def group_by_period(self, summaries: List[Dict]) -> Dict[str, List[Dict]]:
        """Group summaries by day."""
        by_day = defaultdict(list)

        for summary in summaries:
            date_str = summary.get("date", "")
            if date_str:
                by_day[date_str].append(summary)

        return dict(by_day)

    def compress_day(self, day_summaries: List[Dict]) -> Dict[str, Any]:
        """Compress a day's summaries into a single record."""
        if not day_summaries:
            return {}

        # Aggregate stats
        total_messages = sum(s.get("duration_messages", 0) for s in day_summaries)
        all_topics = []
        all_accomplishments = []
        all_files = []

        for s in day_summaries:
            all_topics.extend(s.get("detected_topics", []))
            all_accomplishments.extend(s.get("accomplishments", []))
            all_files.extend(s.get("files_worked", []))

        # Deduplicate
        unique_topics = list(set(all_topics))[:10]
        unique_files = list(set(all_files))[:15]

        # Take top accomplishments
        unique_accomplishments = []
        seen = set()
        for a in all_accomplishments:
            a_key = a[:50].lower()
            if a_key not in seen:
                seen.add(a_key)
                unique_accomplishments.append(a)
        unique_accomplishments = unique_accomplishments[:5]

        return {
            "period_type": "daily",
            "date": day_summaries[0].get("date"),
            "session_count": len(day_summaries),
            "total_messages": total_messages,
            "topics": unique_topics,
            "accomplishments": unique_accomplishments,
            "files_worked": unique_files,
            "generated_at": datetime.now().isoformat()
        }

    def compress_week(self, daily_summaries: List[Dict]) -> Dict[str, Any]:
        """Compress a week's daily summaries into a single record."""
        if not daily_summaries:
            return {}

        # Get date range
        dates = [s.get("date", "") for s in daily_summaries]
        dates.sort()

        # Aggregate
        total_sessions = sum(s.get("session_count", 1) for s in daily_summaries)
        total_messages = sum(s.get("total_messages", 0) for s in daily_summaries)

        all_topics = []
        all_accomplishments = []

        for s in daily_summaries:
            all_topics.extend(s.get("topics", []))
            all_accomplishments.extend(s.get("accomplishments", []))

        # Top topics by frequency
        topic_counts = defaultdict(int)
        for t in all_topics:
            topic_counts[t] += 1
        top_topics = sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:8]

        # Unique accomplishments
        unique_accomplishments = list(set(all_accomplishments))[:5]

        return {
            "period_type": "weekly",
            "start_date": dates[0] if dates else "",
            "end_date": dates[-1] if dates else "",
            "days_covered": len(dates),
            "session_count": total_sessions,
            "total_messages": total_messages,
            "top_topics": top_topics,
            "key_accomplishments": unique_accomplishments,
            "generated_at": datetime.now().isoformat()
        }

    def compress_month(self, weekly_summaries: List[Dict]) -> Dict[str, Any]:
        """Compress a month's weekly summaries into a single record."""
        if not weekly_summaries:
            return {}

        total_sessions = sum(s.get("session_count", 0) for s in weekly_summaries)
        total_messages = sum(s.get("total_messages", 0) for s in weekly_summaries)

        all_topics = []
        all_accomplishments = []

        for s in weekly_summaries:
            all_topics.extend(s.get("top_topics", []))
            all_accomplishments.extend(s.get("key_accomplishments", []))

        # Most frequent topics
        topic_counts = defaultdict(int)
        for t in all_topics:
            topic_counts[t] += 1
        top_topics = sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:5]

        return {
            "period_type": "monthly",
            "month": weekly_summaries[0].get("start_date", "")[:7],
            "weeks_covered": len(weekly_summaries),
            "session_count": total_sessions,
            "total_messages": total_messages,
            "main_topics": top_topics,
            "highlights": all_accomplishments[:3],
            "generated_at": datetime.now().isoformat()
        }

    def run_rollup(self, min_age_days: int = 60) -> Dict[str, Any]:
        """
        Run the hierarchical rollup process.

        Args:
            min_age_days: Only process summaries older than this

        Returns:
            Stats about what was processed
        """
        summaries = self.load_summaries()
        cutoff_date = (datetime.now() - timedelta(days=min_age_days)).strftime("%Y-%m-%d")

        # Filter to old summaries
        old_summaries = [s for s in summaries if s.get("date", "9999") < cutoff_date]

        if not old_summaries:
            return {"status": "no_old_summaries", "min_age_days": min_age_days}

        # Group by day
        by_day = self.group_by_period(old_summaries)

        # Compress each day
        daily_compressed = []
        for date, day_summaries in sorted(by_day.items()):
            compressed = self.compress_day(day_summaries)
            if compressed:
                daily_compressed.append(compressed)

        # Group into weeks and compress
        weekly_compressed = []
        week_buffer = []
        for i, daily in enumerate(daily_compressed):
            week_buffer.append(daily)
            if len(week_buffer) >= 7 or i == len(daily_compressed) - 1:
                if week_buffer:
                    weekly = self.compress_week(week_buffer)
                    if weekly:
                        weekly_compressed.append(weekly)
                    week_buffer = []

        # Group into months and compress
        monthly_compressed = []
        month_buffer = []
        current_month = ""
        for weekly in weekly_compressed:
            week_month = weekly.get("start_date", "")[:7]
            if week_month != current_month and month_buffer:
                monthly = self.compress_month(month_buffer)
                if monthly:
                    monthly_compressed.append(monthly)
                month_buffer = []
            current_month = week_month
            month_buffer.append(weekly)

        if month_buffer:
            monthly = self.compress_month(month_buffer)
            if monthly:
                monthly_compressed.append(monthly)

        # Save hierarchy
        hierarchy = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "cutoff_date": cutoff_date,
            "stats": {
                "daily_compressed": len(daily_compressed),
                "weekly_compressed": len(weekly_compressed),
                "monthly_compressed": len(monthly_compressed),
                "original_summaries": len(old_summaries)
            },
            "daily": daily_compressed[-30:],  # Keep last 30 days
            "weekly": weekly_compressed[-12:],  # Keep last 12 weeks
            "monthly": monthly_compressed  # Keep all months
        }

        with open(self.hierarchy_path, "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)

        return {
            "status": "success",
            "original_summaries": len(old_summaries),
            "daily_compressed": len(daily_compressed),
            "weekly_compressed": len(weekly_compressed),
            "monthly_compressed": len(monthly_compressed),
            "hierarchy_file": str(self.hierarchy_path)
        }

    def get_period_summary(self, period: str) -> List[Dict]:
        """Get summaries for a specific period (daily/weekly/monthly)."""
        if not self.hierarchy_path.exists():
            return []

        try:
            with open(self.hierarchy_path, "r", encoding="utf-8") as f:
                hierarchy = json.load(f)
            return hierarchy.get(period, [])
        except:
            return []


def run_summary_hierarchy(min_age_days: int = 60) -> Dict[str, Any]:
    """Convenience function to run hierarchical rollup."""
    engine = SummaryHierarchy()
    return engine.run_rollup(min_age_days=min_age_days)


if __name__ == "__main__":
    print("Running summary hierarchy rollup...")
    result = run_summary_hierarchy(min_age_days=60)
    print(f"Result: {json.dumps(result, indent=2)}")
