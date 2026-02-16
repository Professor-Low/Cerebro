"""
Local Brain - Fast local storage for AI Memory operations.

Writes data to local disk first (instant), then syncs to NAS in background.
This decouples user experience from NAS performance.

INCLUDES:
- LocalBrain: Write-through cache for learnings/solutions
- LocalReadCache: Read cache for frequently-accessed NAS files

Usage:
    from local_brain import LocalBrain, LocalReadCache, get_local_brain, get_read_cache

    brain = get_local_brain()
    brain.save_learning(data)  # Returns immediately

    cache = get_read_cache()
    profile = cache.get("user_profile.json")  # Fast local read
"""

import json
import os
import shutil
import socket
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class LocalBrain:
    """Fast local storage that syncs to NAS in background."""

    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = str(Path.home() / ".cerebro" / "local_brain")

        self.base_path = Path(base_path)
        self.pending_path = self.base_path / "pending"
        self.failed_path = self.base_path / "failed"

        # Ensure directories exist
        self.pending_path.mkdir(parents=True, exist_ok=True)
        self.failed_path.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, prefix: str) -> str:
        """Generate a unique filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}_{timestamp}_{unique_id}.json"

    def _save_pending(self, data: Dict[str, Any], file_type: str, filename: str = None) -> Dict[str, Any]:
        """
        Save data to pending folder for later sync to NAS.
        Returns immediately - NAS sync happens in background.
        """
        # Add local brain metadata
        data["_local_brain_type"] = file_type
        data["_local_brain_timestamp"] = datetime.now().isoformat()
        data["_sync_retry_count"] = 0

        # Generate filename if not provided
        if filename is None:
            filename = self._generate_filename(file_type)

        # Save to pending folder
        file_path = self.pending_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "stored": "local",
            "filename": filename,
            "will_sync": True,
            "message": "Saved locally, will sync to NAS in background"
        }

    def save_learning(self, problem: str, solution: str, context: str = "",
                      tags: list = None, conversation_id: str = None,
                      **kwargs) -> Dict[str, Any]:
        """Save a learning/solution to local brain."""
        data = {
            "type": "learning",
            "problem": problem,
            "solution": solution,
            "context": context,
            "tags": tags or [],
            "conversation_id": conversation_id,
            "created_at": datetime.now().isoformat(),
            **kwargs
        }
        return self._save_pending(data, "learning")

    def save_solution(self, problem: str, solution: str, context: str = "",
                      tags: list = None, conversation_id: str = None,
                      supersedes: str = None, caused_by_failure: str = None) -> Dict[str, Any]:
        """Save a solution to local brain."""
        import hashlib

        # Generate solution ID
        solution_id = hashlib.md5((problem + solution).encode()).hexdigest()[:16]

        data = {
            "id": solution_id,
            "problem": problem,
            "problem_hash": hashlib.md5(problem.lower().encode()).hexdigest()[:12],
            "solution": solution,
            "context": context,
            "tags": tags or [],
            "current_version": 1,
            "version_history": [],
            "status": "active",
            "conversation_id": conversation_id,
            "supersedes": supersedes,
            "caused_by_failure": caused_by_failure,
            "failure_count": 0,
            "success_confirmations": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        filename = f"solution_{solution_id}.json"
        result = self._save_pending(data, "solution", filename)
        result["solution_id"] = solution_id
        return result

    def save_failure(self, solution_id: str, failure_description: str,
                     error_message: str = "", conversation_id: str = None) -> Dict[str, Any]:
        """Save a failure record to local brain."""
        import hashlib

        failure_id = hashlib.md5(
            (solution_id + failure_description + datetime.now().isoformat()).encode()
        ).hexdigest()[:16]

        data = {
            "type": "failure",
            "failure_id": failure_id,
            "solution_id": solution_id,
            "failure_description": failure_description,
            "error_message": error_message,
            "conversation_id": conversation_id,
            "created_at": datetime.now().isoformat()
        }

        result = self._save_pending(data, "failure")
        result["failure_id"] = failure_id
        return result

    def save_antipattern(self, what_not_to_do: str, why_it_failed: str,
                         error_details: str = "", original_problem: str = "",
                         conversation_id: str = None, source_solution_id: str = None,
                         tags: list = None) -> Dict[str, Any]:
        """Save an antipattern to local brain."""
        import hashlib

        antipattern_id = hashlib.md5(
            (what_not_to_do + why_it_failed).encode()
        ).hexdigest()[:16]

        data = {
            "id": antipattern_id,
            "type": "antipattern",
            "what_not_to_do": what_not_to_do,
            "why_it_failed": why_it_failed,
            "error_details": error_details,
            "original_problem": original_problem,
            "tags": tags or [],
            "source_solution_id": source_solution_id,
            "conversation_id": conversation_id,
            "created_at": datetime.now().isoformat(),
            "times_referenced": 0
        }

        filename = f"antipattern_{antipattern_id}.json"
        result = self._save_pending(data, "antipattern", filename)
        result["antipattern_id"] = antipattern_id
        return result

    def save_preference(self, category: str, preference: str,
                        positive: bool = True) -> Dict[str, Any]:
        """Save a preference update to local brain."""
        data = {
            "type": "preference_update",
            "category": category,
            "preference": preference,
            "positive": positive,
            "created_at": datetime.now().isoformat()
        }
        return self._save_pending(data, "preference")

    def save_correction(self, wrong_value: str, correct_value: str,
                        topic: str = "", context: str = "") -> Dict[str, Any]:
        """Save a correction to local brain."""
        import hashlib

        correction_id = hashlib.md5(
            (wrong_value + correct_value).encode()
        ).hexdigest()[:12]

        data = {
            "id": correction_id,
            "type": "correction",
            "wrong_value": wrong_value,
            "correct_value": correct_value,
            "topic": topic,
            "context": context,
            "created_at": datetime.now().isoformat()
        }

        filename = f"correction_{correction_id}.json"
        result = self._save_pending(data, "correction", filename)
        result["correction_id"] = correction_id
        return result

    def get_pending_count(self) -> int:
        """Get count of pending files waiting to sync."""
        return len(list(self.pending_path.glob("*.json")))

    def get_failed_count(self) -> int:
        """Get count of failed sync files."""
        return len(list(self.failed_path.glob("*.json")))

    def get_stats(self) -> Dict[str, Any]:
        """Get local brain statistics."""
        return {
            "pending_count": self.get_pending_count(),
            "failed_count": self.get_failed_count(),
            "pending_path": str(self.pending_path),
            "failed_path": str(self.failed_path)
        }


class LocalReadCache:
    """
    Local cache for frequently-read NAS files.

    Provides instant access to cached data when NAS is unavailable.
    Automatically refreshes cache when NAS is available.

    Usage:
        cache = LocalReadCache()
        data = cache.get("user_profile.json")
        if data:
            # Use cached data (has _from_cache: true flag)
            pass
        else:
            # Cache miss and NAS unavailable
            pass
    """

    NAS_IP = os.environ.get("CEREBRO_NAS_IP", "")
    NAS_SMB_PORT = 445

    def __init__(self):
        self.CACHE_PATH = Path.home() / ".cerebro" / "local_brain" / "cache"
        data_dir = os.environ.get("CEREBRO_DATA_DIR", str(Path.home() / ".cerebro" / "data"))
        self.CACHE_CONFIG = {
            "quick_facts.json": {
                "source": f"{data_dir}/quick_facts.json",
                "ttl_minutes": 5,
                "description": "Fast-access facts for instant recall"
            },
            "user_profile.json": {
                "source": f"{data_dir}/user/profile.json",
                "ttl_minutes": 60,
                "description": "User profile and preferences"
            },
            "corrections.json": {
                "source": f"{data_dir}/corrections/corrections_index.json",
                "ttl_minutes": 30,
                "description": "Known corrections to avoid mistakes"
            }
        }
        self.CACHE_PATH.mkdir(parents=True, exist_ok=True)

    def _is_nas_reachable(self, timeout: float = 2.0) -> bool:
        """Quick socket check for NAS availability."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((self.NAS_IP, self.NAS_SMB_PORT))
            sock.close()
            return result == 0
        except Exception:
            return False

    def get(self, key: str, refresh_if_stale: bool = True) -> Optional[Dict]:
        """
        Get data from cache.

        Args:
            key: Cache key (e.g., "user_profile.json")
            refresh_if_stale: If True and cache is stale and NAS available, refresh first

        Returns:
            Cached data with metadata, or None if not cached
        """
        cache_file = self.CACHE_PATH / key
        config = self.CACHE_CONFIG.get(key, {"ttl_minutes": 30})

        if not cache_file.exists():
            # Try to populate cache if NAS available
            if refresh_if_stale and self._is_nas_reachable():
                if self.refresh(key):
                    return self.get(key, refresh_if_stale=False)
            return None

        try:
            # Check age
            mtime = cache_file.stat().st_mtime
            age_minutes = (datetime.now() - datetime.fromtimestamp(mtime)).total_seconds() / 60
            ttl = config.get("ttl_minutes", 30)
            is_stale = age_minutes > ttl

            # If stale and NAS available, refresh first
            if is_stale and refresh_if_stale and self._is_nas_reachable():
                if self.refresh(key):
                    return self.get(key, refresh_if_stale=False)

            # Read cached data
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Add cache metadata
            data["_from_cache"] = True
            data["_cache_age_minutes"] = round(age_minutes, 1)
            data["_cache_stale"] = is_stale
            data["_cache_ttl_minutes"] = ttl

            return data

        except Exception as e:
            return {"_error": str(e), "_from_cache": True}

    def refresh(self, key: str) -> bool:
        """
        Refresh cache from NAS (call when NAS is available).

        Args:
            key: Cache key to refresh

        Returns:
            True if refresh succeeded, False otherwise
        """
        config = self.CACHE_CONFIG.get(key)
        if not config:
            return False

        try:
            source = Path(config["source"])
            if source.exists():
                shutil.copy(source, self.CACHE_PATH / key)
                return True
        except Exception:
            pass
        return False

    def refresh_all(self) -> Dict[str, bool]:
        """
        Refresh all configured cache files from NAS.

        Returns:
            Dict mapping key to success status
        """
        if not self._is_nas_reachable():
            return {key: False for key in self.CACHE_CONFIG}

        results = {}
        for key in self.CACHE_CONFIG:
            results[key] = self.refresh(key)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "cache_path": str(self.CACHE_PATH),
            "nas_reachable": self._is_nas_reachable(),
            "files": {}
        }

        for key, config in self.CACHE_CONFIG.items():
            cache_file = self.CACHE_PATH / key
            if cache_file.exists():
                mtime = cache_file.stat().st_mtime
                age_minutes = (datetime.now() - datetime.fromtimestamp(mtime)).total_seconds() / 60
                stats["files"][key] = {
                    "cached": True,
                    "age_minutes": round(age_minutes, 1),
                    "stale": age_minutes > config.get("ttl_minutes", 30),
                    "size_bytes": cache_file.stat().st_size
                }
            else:
                stats["files"][key] = {
                    "cached": False
                }

        return stats


# Singleton instances for easy access
_local_brain: Optional[LocalBrain] = None
_read_cache: Optional[LocalReadCache] = None


def get_local_brain() -> LocalBrain:
    """Get singleton LocalBrain instance."""
    global _local_brain
    if _local_brain is None:
        _local_brain = LocalBrain()
    return _local_brain


def get_read_cache() -> LocalReadCache:
    """Get singleton LocalReadCache instance."""
    global _read_cache
    if _read_cache is None:
        _read_cache = LocalReadCache()
    return _read_cache
