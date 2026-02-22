"""
Storage Manager
Monitor storage sizes and trigger decay.

Features:
- Track storage size by directory
- Size-based decay triggers
- Cold storage archival
- Storage alerts and warnings

Author: Claude
Created: 2026-01-18
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Size thresholds (in bytes)
KB = 1024
MB = 1024 * KB
GB = 1024 * MB

# Default thresholds - Sized for 16TB NAS
# Philosophy: We have plenty of storage, so be generous
# Only decay when storage becomes genuinely concerning
DEFAULT_THRESHOLDS = {
    "conversations": {
        "warning": 10 * GB,       # Warn at 10GB
        "critical": 25 * GB,      # Critical at 25GB
        "max": 50 * GB,           # Trigger aggressive decay at 50GB
    },
    "facts": {
        "warning": 5 * GB,        # Warn at 5GB
        "critical": 10 * GB,      # Critical at 10GB
        "max": 25 * GB,           # Trigger aggressive decay at 25GB
    },
    "embeddings": {
        "warning": 25 * GB,       # Warn at 25GB (embeddings are large)
        "critical": 50 * GB,      # Critical at 50GB
        "max": 100 * GB,          # Trigger aggressive decay at 100GB
    },
    "total": {
        "warning": 100 * GB,      # Warn at 100GB
        "critical": 250 * GB,     # Critical at 250GB
        "max": 500 * GB,          # Trigger aggressive decay at 500GB (~3% of 16TB)
    }
}

# Cold storage path (relative to AI_MEMORY_BASE)
COLD_STORAGE_DIR = "cold_storage"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StorageMetrics:
    """Storage metrics for a directory."""
    path: str
    file_count: int
    total_bytes: int
    oldest_file: Optional[str]     # ISO timestamp
    newest_file: Optional[str]     # ISO timestamp
    measured_at: str               # ISO timestamp

    @property
    def total_mb(self) -> float:
        return round(self.total_bytes / MB, 2)

    @property
    def total_gb(self) -> float:
        return round(self.total_bytes / GB, 3)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["total_mb"] = self.total_mb
        d["total_gb"] = self.total_gb
        return d


@dataclass
class StorageAlert:
    """Storage alert or warning."""
    directory: str
    level: str                     # "warning", "critical", "max"
    current_bytes: int
    threshold_bytes: int
    message: str
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Storage Manager
# =============================================================================

class StorageManager:
    """
    Monitors storage usage and triggers decay when thresholds are exceeded.
    """

    def __init__(self, base_path: Path, thresholds: Optional[Dict] = None):
        """
        Initialize storage manager.

        Args:
            base_path: Base AI_MEMORY directory
            thresholds: Optional custom thresholds (overrides defaults)
        """
        self.base_path = Path(base_path)
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

        # Storage state file
        self.state_dir = self.base_path / "metadata"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "storage_state.json"

        # Cold storage directory
        self.cold_storage = self.base_path / COLD_STORAGE_DIR
        self.cold_storage.mkdir(parents=True, exist_ok=True)

        # Alert log
        self.alert_log = self.state_dir / "storage_alerts.jsonl"

    # -------------------------------------------------------------------------
    # Measurement
    # -------------------------------------------------------------------------

    def measure_directory(self, dir_path: Path) -> StorageMetrics:
        """
        Measure storage metrics for a directory.

        Args:
            dir_path: Directory to measure

        Returns:
            StorageMetrics for the directory
        """
        if not dir_path.exists():
            return StorageMetrics(
                path=str(dir_path),
                file_count=0,
                total_bytes=0,
                oldest_file=None,
                newest_file=None,
                measured_at=datetime.now().isoformat()
            )

        total_bytes = 0
        file_count = 0
        oldest_mtime = None
        newest_mtime = None

        try:
            for f in dir_path.rglob("*"):
                if f.is_file():
                    file_count += 1
                    try:
                        stat = f.stat()
                        total_bytes += stat.st_size
                        mtime = stat.st_mtime

                        if oldest_mtime is None or mtime < oldest_mtime:
                            oldest_mtime = mtime
                        if newest_mtime is None or mtime > newest_mtime:
                            newest_mtime = mtime
                    except (OSError, PermissionError):
                        continue
        except Exception as e:
            logger.error(f"Error measuring directory {dir_path}: {e}")

        return StorageMetrics(
            path=str(dir_path),
            file_count=file_count,
            total_bytes=total_bytes,
            oldest_file=datetime.fromtimestamp(oldest_mtime).isoformat() if oldest_mtime else None,
            newest_file=datetime.fromtimestamp(newest_mtime).isoformat() if newest_mtime else None,
            measured_at=datetime.now().isoformat()
        )

    def measure_all(self) -> Dict[str, StorageMetrics]:
        """
        Measure all tracked directories.

        Returns:
            Dict mapping directory name to metrics
        """
        directories = {
            "conversations": self.base_path / "conversations",
            "facts": self.base_path / "facts",
            "embeddings": self.base_path / "embeddings",
            "cache": self.base_path / "cache",
            "knowledge": self.base_path / "knowledge",
            "cold_storage": self.cold_storage,
        }

        metrics = {}
        for name, path in directories.items():
            metrics[name] = self.measure_directory(path)

        # Add total
        total_bytes = sum(m.total_bytes for m in metrics.values())
        total_files = sum(m.file_count for m in metrics.values())

        metrics["total"] = StorageMetrics(
            path=str(self.base_path),
            file_count=total_files,
            total_bytes=total_bytes,
            oldest_file=min(
                (m.oldest_file for m in metrics.values() if m.oldest_file),
                default=None
            ),
            newest_file=max(
                (m.newest_file for m in metrics.values() if m.newest_file),
                default=None
            ),
            measured_at=datetime.now().isoformat()
        )

        return metrics

    # -------------------------------------------------------------------------
    # Threshold Checking
    # -------------------------------------------------------------------------

    def check_thresholds(self, metrics: Optional[Dict[str, StorageMetrics]] = None) -> List[StorageAlert]:
        """
        Check storage against thresholds.

        Returns:
            List of alerts for directories exceeding thresholds
        """
        if metrics is None:
            metrics = self.measure_all()

        alerts = []

        for dir_name, thresholds in self.thresholds.items():
            if dir_name not in metrics:
                continue

            current = metrics[dir_name].total_bytes

            # Check each threshold level (most severe first)
            for level in ["max", "critical", "warning"]:
                threshold = thresholds.get(level, 0)
                if current >= threshold:
                    alert = StorageAlert(
                        directory=dir_name,
                        level=level,
                        current_bytes=current,
                        threshold_bytes=threshold,
                        message=self._format_alert_message(dir_name, level, current, threshold),
                        timestamp=datetime.now().isoformat()
                    )
                    alerts.append(alert)
                    self._log_alert(alert)
                    break  # Only log the highest severity alert

        return alerts

    def _format_alert_message(self, dir_name: str, level: str, current: int, threshold: int) -> str:
        """Format a human-readable alert message."""
        current_mb = round(current / MB, 1)
        threshold_mb = round(threshold / MB, 1)
        pct = round((current / threshold) * 100, 1)

        if level == "max":
            return f"CRITICAL: {dir_name} at {current_mb}MB ({pct}% of max). Aggressive decay recommended!"
        elif level == "critical":
            return f"WARNING: {dir_name} at {current_mb}MB ({pct}% of critical threshold). Decay recommended."
        else:
            return f"INFO: {dir_name} at {current_mb}MB approaching threshold ({threshold_mb}MB)."

    def _log_alert(self, alert: StorageAlert):
        """Log an alert to the alert log file."""
        try:
            with open(self.alert_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Error logging alert: {e}")

    def needs_decay(self, metrics: Optional[Dict[str, StorageMetrics]] = None) -> Tuple[bool, str]:
        """
        Check if decay is needed based on storage thresholds.

        Returns:
            (needs_decay, reason)
        """
        alerts = self.check_thresholds(metrics)

        # Check for max-level alerts (urgent)
        max_alerts = [a for a in alerts if a.level == "max"]
        if max_alerts:
            return True, f"Max threshold exceeded: {max_alerts[0].directory}"

        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.level == "critical"]
        if critical_alerts:
            return True, f"Critical threshold reached: {critical_alerts[0].directory}"

        return False, "Storage within limits"

    # -------------------------------------------------------------------------
    # Cold Storage
    # -------------------------------------------------------------------------

    def move_to_cold_storage(
        self,
        source_path: Path,
        preserve_structure: bool = True
    ) -> Optional[Path]:
        """
        Move a file to cold storage.

        Args:
            source_path: File to move
            preserve_structure: If True, preserve relative directory structure

        Returns:
            Path to the cold storage location, or None on failure
        """
        if not source_path.exists():
            return None

        try:
            if preserve_structure:
                # Preserve relative path from base
                try:
                    rel_path = source_path.relative_to(self.base_path)
                except ValueError:
                    rel_path = source_path.name
                dest = self.cold_storage / rel_path
            else:
                dest = self.cold_storage / source_path.name

            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(dest))

            logger.info(f"Moved to cold storage: {source_path} -> {dest}")
            return dest

        except Exception as e:
            logger.error(f"Error moving to cold storage: {e}")
            return None

    def restore_from_cold_storage(
        self,
        cold_path: Path,
        dest_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Restore a file from cold storage.

        Args:
            cold_path: Path in cold storage (relative or absolute)
            dest_path: Destination path, or None to restore to original location

        Returns:
            Path to the restored file, or None on failure
        """
        # Handle relative paths
        if not cold_path.is_absolute():
            cold_path = self.cold_storage / cold_path

        if not cold_path.exists():
            logger.error(f"Cold storage file not found: {cold_path}")
            return None

        try:
            if dest_path is None:
                # Restore to original location
                rel_path = cold_path.relative_to(self.cold_storage)
                dest_path = self.base_path / rel_path

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(cold_path), str(dest_path))

            logger.info(f"Restored from cold storage: {cold_path} -> {dest_path}")
            return dest_path

        except Exception as e:
            logger.error(f"Error restoring from cold storage: {e}")
            return None

    def get_cold_storage_contents(self) -> List[Dict]:
        """Get list of files in cold storage with metadata."""
        contents = []

        if not self.cold_storage.exists():
            return contents

        for f in self.cold_storage.rglob("*"):
            if f.is_file():
                try:
                    stat = f.stat()
                    contents.append({
                        "path": str(f.relative_to(self.cold_storage)),
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / MB, 3),
                        "moved_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except:
                    continue

        return sorted(contents, key=lambda x: x.get("moved_at", ""), reverse=True)

    # -------------------------------------------------------------------------
    # Statistics and Reporting
    # -------------------------------------------------------------------------

    def get_storage_report(self) -> Dict[str, Any]:
        """Generate a comprehensive storage report."""
        metrics = self.measure_all()
        alerts = self.check_thresholds(metrics)
        needs_decay, decay_reason = self.needs_decay(metrics)

        report = {
            "measured_at": datetime.now().isoformat(),
            "directories": {k: v.to_dict() for k, v in metrics.items()},
            "alerts": [a.to_dict() for a in alerts],
            "needs_decay": needs_decay,
            "decay_reason": decay_reason,
            "cold_storage": {
                "file_count": len(self.get_cold_storage_contents()),
                "total_mb": metrics.get("cold_storage", StorageMetrics(
                    path="", file_count=0, total_bytes=0,
                    oldest_file=None, newest_file=None, measured_at=""
                )).total_mb
            },
            "thresholds": {
                k: {level: f"{v/MB:.0f}MB" for level, v in thresholds.items()}
                for k, thresholds in self.thresholds.items()
            }
        }

        return report

    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent storage alerts."""
        alerts = []

        if not self.alert_log.exists():
            return alerts

        try:
            with open(self.alert_log, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in reversed(lines[-limit:]):
                try:
                    alerts.append(json.loads(line.strip()))
                except:
                    continue
        except Exception as e:
            logger.error(f"Error reading alert log: {e}")

        return alerts

    def save_state(self, metrics: Dict[str, StorageMetrics]):
        """Save current storage state for historical tracking."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {k: v.to_dict() for k, v in metrics.items()}
        }

        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving storage state: {e}")


# =============================================================================
# Module-level convenience functions
# =============================================================================

_manager: Optional[StorageManager] = None

def get_manager(base_path: Optional[Path] = None) -> StorageManager:
    """Get or create the global storage manager instance."""
    global _manager
    if _manager is None:
        if base_path is None:
            from config import AI_MEMORY_BASE
            base_path = AI_MEMORY_BASE
        _manager = StorageManager(base_path)
    return _manager


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    # Initialize with config
    from config import AI_MEMORY_BASE
    manager = StorageManager(AI_MEMORY_BASE)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "report":
            print("Storage Report:")
            report = manager.get_storage_report()
            print(json.dumps(report, indent=2))

        elif cmd == "check":
            metrics = manager.measure_all()
            alerts = manager.check_thresholds(metrics)
            if alerts:
                print("ALERTS:")
                for alert in alerts:
                    print(f"  [{alert.level.upper()}] {alert.message}")
            else:
                print("All storage within limits")

        elif cmd == "measure":
            print("Storage Measurements:")
            for name, metrics in manager.measure_all().items():
                print(f"  {name}: {metrics.total_mb}MB ({metrics.file_count} files)")

        elif cmd == "cold":
            print("Cold Storage Contents:")
            contents = manager.get_cold_storage_contents()
            for item in contents[:20]:
                print(f"  {item['path']}: {item['size_mb']}MB")

        elif cmd == "alerts":
            print("Recent Alerts:")
            for alert in manager.get_recent_alerts():
                print(f"  [{alert['level']}] {alert['message']}")

        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python storage_manager.py [report|check|measure|cold|alerts]")
    else:
        print("Storage Manager - Quick Check:")
        needs_decay, reason = manager.needs_decay()
        print(f"  Needs decay: {needs_decay}")
        print(f"  Reason: {reason}")
