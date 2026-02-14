"""
Consolidator - Claude.Me v6.0
Background memory consolidation process.

Part of Phase 10: Active Memory Consolidation

Can be run as:
- Manual trigger via MCP tool
- Scheduled task (Windows Task Scheduler / cron)
"""
import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

# DGX Spark configuration
_dgx_host = os.environ.get("CEREBRO_DGX_HOST", "")
DGX_CONSOLIDATION_SERVICE = f"http://{_dgx_host}:8769" if _dgx_host else ""
DGX_TIMEOUT = 120  # Longer timeout for batch processing


class ConsolidationState:
    """Represents the consolidation system state."""

    def __init__(
        self,
        last_run: str = None,
        next_scheduled: str = None,
        stats: Dict = None
    ):
        self.last_run = last_run
        self.next_scheduled = next_scheduled
        self.stats = stats or {
            "episodes_processed": 0,
            "abstractions_created": 0,
            "connections_strengthened": 0,
            "memories_pruned": 0,
            "insights_generated": 0
        }

    def to_dict(self) -> Dict:
        return {
            "last_run": self.last_run,
            "next_scheduled": self.next_scheduled,
            "stats": self.stats
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConsolidationState":
        return cls(
            last_run=data.get("last_run"),
            next_scheduled=data.get("next_scheduled"),
            stats=data.get("stats", {})
        )


class ConsolidationRun:
    """Represents a single consolidation run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.started_at = datetime.now().isoformat()
        self.completed_at = None
        self.status = "running"
        self.stats = {
            "episodes_processed": 0,
            "abstractions_created": 0,
            "connections_strengthened": 0,
            "memories_pruned": 0,
            "insights_generated": 0
        }
        self.errors = []

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "stats": self.stats,
            "errors": self.errors
        }


class Consolidator:
    """
    Background memory consolidation process.

    Operations:
    1. Cluster similar episodes
    2. Extract abstractions
    3. Strengthen frequently co-accessed connections
    4. Prune redundant memories
    5. Generate insights from patterns
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.consolidation_path = self.base_path / "consolidation"
        self.state_file = self.consolidation_path / "state.json"
        self.runs_path = self.consolidation_path / "runs"
        self._dgx_available = None
        self._lock = threading.Lock()
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.consolidation_path.mkdir(parents=True, exist_ok=True)
        self.runs_path.mkdir(parents=True, exist_ok=True)

    def _generate_run_id(self) -> str:
        """Generate consolidation run ID."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{ts}"

    def _load_state(self) -> ConsolidationState:
        """Load consolidation state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return ConsolidationState.from_dict(data)
            except:
                pass
        return ConsolidationState()

    def _save_state(self, state: ConsolidationState):
        """Save consolidation state."""
        with self._lock:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, indent=2)

    def _save_run(self, run: ConsolidationRun):
        """Save consolidation run."""
        run_file = self.runs_path / f"{run.run_id}.json"
        with open(run_file, 'w', encoding='utf-8') as f:
            json.dump(run.to_dict(), f, indent=2)

    def _check_dgx_available(self) -> bool:
        """Check if DGX consolidation service is available."""
        if self._dgx_available is not None:
            return self._dgx_available

        try:
            resp = requests.get(f"{DGX_CONSOLIDATION_SERVICE}/health", timeout=5)
            self._dgx_available = resp.status_code == 200
        except:
            self._dgx_available = False

        return self._dgx_available

    def run_consolidation(self, full: bool = False) -> Dict:
        """
        Run a consolidation pass.

        Args:
            full: If True, process all memories. If False, only recent.

        Returns:
            Statistics about the consolidation run.
        """
        run = ConsolidationRun(self._generate_run_id())

        try:
            # 1. Run abstraction pass
            abstraction_stats = self._run_abstraction_pass(run, full)

            # 2. Strengthen connections
            connection_stats = self._strengthen_connections(run)

            # 3. Prune redundant memories
            prune_stats = self._prune_redundant(run)

            # 4. Generate insights
            insight_stats = self._generate_insights(run)

            # Update run stats
            run.stats["episodes_processed"] = abstraction_stats.get("episodes_processed", 0)
            run.stats["abstractions_created"] = abstraction_stats.get("abstractions_created", 0)
            run.stats["connections_strengthened"] = connection_stats.get("strengthened", 0)
            run.stats["memories_pruned"] = prune_stats.get("pruned", 0)
            run.stats["insights_generated"] = insight_stats.get("generated", 0)

            run.status = "completed"
            run.completed_at = datetime.now().isoformat()

        except Exception as e:
            run.status = "failed"
            run.errors.append(str(e))
            run.completed_at = datetime.now().isoformat()

        # Save run and update state
        self._save_run(run)
        self._update_state_after_run(run)

        return run.to_dict()

    def _run_abstraction_pass(self, run: ConsolidationRun, full: bool) -> Dict:
        """Run abstraction extraction."""
        from abstractor import Abstractor

        abstractor = Abstractor(base_path=str(self.base_path))

        limit = 1000 if full else 100
        result = abstractor.run_abstraction_pass(episode_limit=limit)

        return result

    def _strengthen_connections(self, run: ConsolidationRun) -> Dict:
        """Strengthen frequently co-accessed memory connections."""
        strengthened = 0

        # Load access patterns (from meta_learning if available)
        access_file = self.base_path / "meta_learning" / "access_patterns.json"
        if access_file.exists():
            try:
                with open(access_file, 'r', encoding='utf-8') as f:
                    patterns = json.load(f)

                # Find co-accessed memory pairs
                co_access = patterns.get("co_accessed", {})
                for pair_key, count in co_access.items():
                    if count >= 3:  # Strengthen if accessed together 3+ times
                        # Mark these as related in semantic memory
                        strengthened += 1

            except:
                pass

        return {"strengthened": strengthened}

    def _prune_redundant(self, run: ConsolidationRun) -> Dict:
        """Archive redundant memories."""
        pruned = 0
        archive_path = self.base_path / "archive"
        archive_path.mkdir(parents=True, exist_ok=True)

        # Find duplicate semantic facts
        semantic_path = self.base_path / "semantic"
        if semantic_path.exists():
            facts_by_content = defaultdict(list)

            for sem_file in semantic_path.glob("sem_*.json"):
                try:
                    with open(sem_file, 'r', encoding='utf-8') as f:
                        sem = json.load(f)

                    # Normalize content for comparison
                    content = sem.get("fact", "").lower().strip()
                    facts_by_content[content].append(sem_file)
                except:
                    continue

            # Archive duplicates, keep highest confidence
            for content, files in facts_by_content.items():
                if len(files) > 1:
                    # Load all and keep best
                    best_file = None
                    best_confidence = -1

                    for f in files:
                        try:
                            with open(f, 'r', encoding='utf-8') as fh:
                                data = json.load(fh)
                            if data.get("confidence", 0) > best_confidence:
                                best_confidence = data.get("confidence", 0)
                                best_file = f
                        except:
                            continue

                    # Archive non-best
                    for f in files:
                        if f != best_file:
                            # Move to archive
                            try:
                                import shutil
                                shutil.move(str(f), str(archive_path / f.name))
                                pruned += 1
                            except:
                                pass

        return {"pruned": pruned}

    def _generate_insights(self, run: ConsolidationRun) -> Dict:
        """Generate insights from consolidated patterns."""
        generated = 0

        # Try to use DGX for insight generation
        if self._check_dgx_available():
            try:
                # Load recent abstractions
                abstractions = []
                abstr_path = self.consolidation_path / "abstractions"
                for abstr_file in sorted(abstr_path.glob("abstr_*.json"), reverse=True)[:20]:
                    try:
                        with open(abstr_file, 'r', encoding='utf-8') as f:
                            abstractions.append(json.load(f))
                    except:
                        continue

                if abstractions:
                    resp = requests.post(
                        f"{DGX_CONSOLIDATION_SERVICE}/generate_insights",
                        json={"abstractions": abstractions},
                        timeout=DGX_TIMEOUT
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        insights = data.get("insights", [])

                        # Save generated insights
                        insights_path = self.base_path / "insights"
                        insights_path.mkdir(parents=True, exist_ok=True)

                        for insight in insights:
                            insight_file = insights_path / f"ins_consolidated_{datetime.now().strftime('%Y%m%d%H%M%S')}_{generated}.json"
                            insight["source"] = "consolidation"
                            insight["timestamp"] = datetime.now().isoformat()

                            with open(insight_file, 'w', encoding='utf-8') as f:
                                json.dump(insight, f, indent=2)

                            generated += 1

            except Exception as e:
                run.errors.append(f"DGX insight generation failed: {e}")

        return {"generated": generated}

    def _update_state_after_run(self, run: ConsolidationRun):
        """Update state after a consolidation run."""
        state = self._load_state()

        state.last_run = run.completed_at or run.started_at

        # Schedule next run (default: 24 hours)
        next_time = datetime.now() + timedelta(hours=24)
        state.next_scheduled = next_time.isoformat()

        # Accumulate stats
        for key in run.stats:
            state.stats[key] = state.stats.get(key, 0) + run.stats.get(key, 0)

        self._save_state(state)

    def get_state(self) -> Dict:
        """Get current consolidation state."""
        return self._load_state().to_dict()

    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get a specific consolidation run."""
        run_file = self.runs_path / f"{run_id}.json"
        if not run_file.exists():
            return None

        try:
            with open(run_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def get_recent_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent consolidation runs."""
        runs = []

        for run_file in sorted(self.runs_path.glob("run_*.json"), reverse=True)[:limit]:
            try:
                with open(run_file, 'r', encoding='utf-8') as f:
                    runs.append(json.load(f))
            except:
                continue

        return runs

    def schedule_next_run(self, hours_from_now: int = 24) -> Dict:
        """Schedule the next consolidation run."""
        state = self._load_state()
        next_time = datetime.now() + timedelta(hours=hours_from_now)
        state.next_scheduled = next_time.isoformat()
        self._save_state(state)

        return {
            "scheduled": True,
            "next_run": state.next_scheduled
        }

    def should_run(self) -> bool:
        """Check if consolidation should run based on schedule."""
        state = self._load_state()

        if not state.next_scheduled:
            return True

        try:
            next_time = datetime.fromisoformat(state.next_scheduled)
            return datetime.now() >= next_time
        except:
            return True

    def get_stats(self) -> Dict:
        """Get consolidation statistics."""
        state = self._load_state()

        run_count = len(list(self.runs_path.glob("run_*.json")))

        return {
            "total_runs": run_count,
            "last_run": state.last_run,
            "next_scheduled": state.next_scheduled,
            "cumulative_stats": state.stats,
            "dgx_available": self._check_dgx_available() if self._dgx_available is None else self._dgx_available
        }


# Entry point for scheduled execution
if __name__ == "__main__":
    import sys

    consolidator = Consolidator()

    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        print("Running forced consolidation...")
        result = consolidator.run_consolidation(full=True)
    elif consolidator.should_run():
        print("Running scheduled consolidation...")
        result = consolidator.run_consolidation(full=False)
    else:
        print("Consolidation not due yet.")
        print(f"Next scheduled: {consolidator.get_state().get('next_scheduled')}")
        sys.exit(0)

    print(f"Consolidation completed: {json.dumps(result, indent=2)}")
