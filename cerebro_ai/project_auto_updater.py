"""
Project Auto-Updater - Automatically update project state based on activity.

Part of Phase 3 Enhancement in the All-Knowing Brain PRD.
Detects project activity from conversations and updates:
- Status transitions (active -> stale -> inactive)
- Current focus tracking
- Last worked timestamp
- Activity metrics
"""

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Status transition thresholds
STATUS_THRESHOLDS = {
    "active_to_stale_days": 7,     # No activity for 7 days -> stale
    "stale_to_inactive_days": 30,  # No activity for 30 days -> inactive
    "reactivation_mentions": 2,    # 2+ mentions to reactivate an inactive project
}

# Project detection patterns
PROJECT_INDICATORS = [
    r'/home/[^/]+/([a-zA-Z0-9_-]+)/',  # Home directory projects
    r'/projects?/([a-zA-Z0-9_-]+)/',   # Generic projects directory
    r'/repos?/([a-zA-Z0-9_-]+)/',      # Repository directory
    r'github\.com/[^/]+/([a-zA-Z0-9_-]+)',  # GitHub URLs
]

# Focus detection keywords
FOCUS_KEYWORDS = {
    "debugging": ["debug", "error", "fix", "bug", "issue", "traceback", "exception"],
    "implementing": ["implement", "add", "create", "build", "develop", "write"],
    "refactoring": ["refactor", "clean", "reorganize", "restructure", "improve"],
    "testing": ["test", "testing", "unittest", "pytest", "coverage"],
    "documenting": ["doc", "readme", "comment", "explain", "documentation"],
    "configuring": ["config", "setup", "install", "configure", "settings"],
    "deploying": ["deploy", "release", "publish", "production", "staging"],
    "researching": ["research", "investigate", "explore", "understand", "analyze"],
}


class ProjectAutoUpdater:
    """
    Automatically updates project state based on conversation activity.
    Integrates with ProjectTracker to keep project info current.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.projects_file = self.base_path / "projects" / "tracker.json"
        self.activity_log_file = self.base_path / "projects" / "activity_log.json"

        # Ensure directory exists
        self.projects_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_projects(self) -> Dict[str, Dict[str, Any]]:
        """Load projects from tracker file."""
        if self.projects_file.exists():
            try:
                with open(self.projects_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_projects(self, projects: Dict[str, Dict[str, Any]]) -> None:
        """Save projects to tracker file."""
        try:
            with open(self.projects_file, "w", encoding="utf-8") as f:
                json.dump(projects, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving projects: {e}")

    def _load_activity_log(self) -> List[Dict[str, Any]]:
        """Load activity log."""
        if self.activity_log_file.exists():
            try:
                with open(self.activity_log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_activity_log(self, log: List[Dict[str, Any]]) -> None:
        """Save activity log (keep last 500 entries)."""
        try:
            # Keep only last 500 entries
            log = log[-500:]
            with open(self.activity_log_file, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def detect_projects_from_conversation(self,
                                          messages: List[Dict[str, str]],
                                          cwd: str = None) -> List[Dict[str, Any]]:
        """
        Detect projects mentioned or worked on in a conversation.

        Args:
            messages: List of conversation messages
            cwd: Current working directory (if known)

        Returns:
            List of detected projects with confidence scores
        """
        detected = {}

        # Combine all message content
        all_content = "\n".join(m.get("content", "") for m in messages)

        # 1. Detect from file paths in content
        for pattern in PROJECT_INDICATORS:
            for match in re.finditer(pattern, all_content):
                project_name = match.group(1)
                if self._is_valid_project_name(project_name):
                    if project_name not in detected:
                        detected[project_name] = {
                            "name": project_name,
                            "confidence": 0,
                            "sources": [],
                            "files_mentioned": []
                        }
                    detected[project_name]["confidence"] += 0.3
                    detected[project_name]["sources"].append("path_mention")
                    # Extract full path for file tracking
                    full_path = match.group(0)
                    if full_path not in detected[project_name]["files_mentioned"]:
                        detected[project_name]["files_mentioned"].append(full_path)

        # 2. Detect from CWD
        if cwd:
            cwd_parts = Path(cwd).parts
            for i, part in enumerate(cwd_parts):
                if part in ["projects", "repos", "src", "home"] and i + 1 < len(cwd_parts):
                    project_name = cwd_parts[i + 1]
                    if self._is_valid_project_name(project_name):
                        if project_name not in detected:
                            detected[project_name] = {
                                "name": project_name,
                                "confidence": 0,
                                "sources": [],
                                "files_mentioned": []
                            }
                        detected[project_name]["confidence"] += 0.5
                        detected[project_name]["sources"].append("cwd")

        # 3. Check for project name mentions directly
        projects = self._load_projects()
        for project_id, project in projects.items():
            project_name = project.get("name", project_id)
            # Case-insensitive search for project name
            if re.search(rf'\b{re.escape(project_name)}\b', all_content, re.IGNORECASE):
                if project_name not in detected:
                    detected[project_name] = {
                        "name": project_name,
                        "confidence": 0,
                        "sources": [],
                        "files_mentioned": []
                    }
                detected[project_name]["confidence"] += 0.4
                detected[project_name]["sources"].append("direct_mention")

        # Cap confidence at 1.0 and filter low confidence
        results = []
        for name, info in detected.items():
            info["confidence"] = min(info["confidence"], 1.0)
            if info["confidence"] >= 0.3:  # Minimum threshold
                results.append(info)

        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def detect_focus_from_conversation(self,
                                       messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Detect what the user is focusing on in this conversation.

        Args:
            messages: List of conversation messages

        Returns:
            Dict with focus area and keywords
        """
        all_content = "\n".join(m.get("content", "").lower() for m in messages)

        focus_scores = defaultdict(int)
        matched_keywords = defaultdict(list)

        for focus_type, keywords in FOCUS_KEYWORDS.items():
            for keyword in keywords:
                count = all_content.count(keyword)
                if count > 0:
                    focus_scores[focus_type] += count
                    matched_keywords[focus_type].append(keyword)

        if not focus_scores:
            return {
                "focus": "general",
                "confidence": 0.3,
                "keywords": []
            }

        # Get top focus area
        top_focus = max(focus_scores.items(), key=lambda x: x[1])
        total_score = sum(focus_scores.values())

        return {
            "focus": top_focus[0],
            "confidence": min(top_focus[1] / max(total_score, 1), 1.0),
            "keywords": matched_keywords[top_focus[0]][:5],
            "all_focuses": dict(focus_scores)
        }

    def _is_valid_project_name(self, name: str) -> bool:
        """Check if a string is a valid project name."""
        # Filter out common non-project directories
        invalid_names = {
            "bin", "lib", "usr", "var", "etc", "tmp", "opt",
            "home", "root", "dev", "proc", "sys", "run",
            "node_modules", ".git", "__pycache__", ".cache",
            "venv", ".venv", "env", ".env", "site-packages",
            "dist", "build", ".next", ".nuxt"
        }

        if not name or len(name) < 2:
            return False
        if name.lower() in invalid_names:
            return False
        if name.startswith(".") or name.startswith("_"):
            return False
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
            return False

        return True

    def update_project_from_activity(self,
                                      project_name: str,
                                      session_id: str,
                                      focus: str = None,
                                      files: List[str] = None,
                                      confidence: float = 0.5) -> Dict[str, Any]:
        """
        Update a project's state based on detected activity.

        Args:
            project_name: Name of the project
            session_id: Session ID where activity was detected
            focus: Detected focus area
            files: Files that were mentioned/edited
            confidence: Detection confidence

        Returns:
            Updated project info
        """
        projects = self._load_projects()
        now = datetime.now()

        # Get or create project
        if project_name not in projects:
            projects[project_name] = {
                "project_id": project_name.lower().replace(" ", "-"),
                "name": project_name,
                "status": "active",
                "priority": "medium",
                "last_worked": now.isoformat(),
                "current_focus": focus or "",
                "files": [],
                "technologies": [],
                "blockers": [],
                "next_steps": [],
                "milestones": {"completed": [], "in_progress": []},
                "mention_count": 0,
                "created_at": now.isoformat(),
                "activity_history": []
            }

        project = projects[project_name]

        # Update timestamp
        project["last_worked"] = now.isoformat()

        # Update status to active if it was stale/inactive
        old_status = project.get("status", "active")
        if old_status in ["stale", "inactive"]:
            project["status"] = "active"

        # Update focus if provided and confident
        if focus and confidence >= 0.5:
            project["current_focus"] = focus

        # Add new files
        if files:
            existing_files = set(project.get("files", []))
            for f in files:
                if f and f not in existing_files:
                    existing_files.add(f)
            project["files"] = list(existing_files)[:50]  # Keep max 50 files

        # Increment mention count
        project["mention_count"] = project.get("mention_count", 0) + 1

        # Add to activity history (keep last 20)
        activity_history = project.get("activity_history", [])
        activity_history.append({
            "timestamp": now.isoformat(),
            "session_id": session_id,
            "focus": focus,
            "confidence": confidence
        })
        project["activity_history"] = activity_history[-20:]

        # Save
        self._save_projects(projects)

        # Log activity
        activity_log = self._load_activity_log()
        activity_log.append({
            "timestamp": now.isoformat(),
            "project": project_name,
            "session_id": session_id,
            "action": "activity_detected",
            "focus": focus,
            "status_change": f"{old_status} -> {project['status']}" if old_status != project['status'] else None
        })
        self._save_activity_log(activity_log)

        return {
            "project": project_name,
            "status": project["status"],
            "status_changed": old_status != project["status"],
            "old_status": old_status if old_status != project["status"] else None,
            "focus": focus,
            "last_worked": project["last_worked"],
            "mention_count": project["mention_count"]
        }

    def run_status_transitions(self) -> Dict[str, Any]:
        """
        Run automatic status transitions based on time thresholds.
        Should be called periodically (e.g., on SessionEnd).

        Returns:
            Dict with transition results
        """
        projects = self._load_projects()
        now = datetime.now()
        transitions = []

        active_to_stale_threshold = timedelta(days=STATUS_THRESHOLDS["active_to_stale_days"])
        stale_to_inactive_threshold = timedelta(days=STATUS_THRESHOLDS["stale_to_inactive_days"])

        for project_name, project in projects.items():
            last_worked_str = project.get("last_worked", "")
            if not last_worked_str:
                continue

            try:
                last_worked = datetime.fromisoformat(last_worked_str)
            except (ValueError, TypeError):
                continue

            time_since_activity = now - last_worked
            old_status = project.get("status", "active")
            new_status = old_status

            # Check for transitions
            if old_status == "active" and time_since_activity > active_to_stale_threshold:
                new_status = "stale"
            elif old_status == "stale" and time_since_activity > stale_to_inactive_threshold:
                new_status = "inactive"

            if new_status != old_status:
                project["status"] = new_status
                project["status_changed_at"] = now.isoformat()
                transitions.append({
                    "project": project_name,
                    "from": old_status,
                    "to": new_status,
                    "days_inactive": time_since_activity.days
                })

        if transitions:
            self._save_projects(projects)

            # Log transitions
            activity_log = self._load_activity_log()
            for t in transitions:
                activity_log.append({
                    "timestamp": now.isoformat(),
                    "project": t["project"],
                    "action": "status_transition",
                    "from_status": t["from"],
                    "to_status": t["to"],
                    "reason": f"Inactive for {t['days_inactive']} days"
                })
            self._save_activity_log(activity_log)

        return {
            "checked": len(projects),
            "transitions": transitions,
            "timestamp": now.isoformat()
        }

    def process_conversation(self,
                             messages: List[Dict[str, str]],
                             session_id: str,
                             cwd: str = None) -> Dict[str, Any]:
        """
        Process a conversation to auto-update project states.
        Main entry point for autosave integration.

        Args:
            messages: Conversation messages
            session_id: Session identifier
            cwd: Current working directory

        Returns:
            Processing results
        """
        # Detect projects
        detected_projects = self.detect_projects_from_conversation(messages, cwd)

        # Detect focus
        focus_info = self.detect_focus_from_conversation(messages)

        # Update each detected project
        updated_projects = []
        for project_info in detected_projects:
            result = self.update_project_from_activity(
                project_name=project_info["name"],
                session_id=session_id,
                focus=focus_info.get("focus"),
                files=project_info.get("files_mentioned", []),
                confidence=project_info.get("confidence", 0.5)
            )
            updated_projects.append(result)

        # Run periodic status transitions
        transitions = self.run_status_transitions()

        return {
            "detected_projects": len(detected_projects),
            "updated_projects": updated_projects,
            "focus": focus_info,
            "status_transitions": transitions.get("transitions", []),
            "session_id": session_id
        }

    def get_activity_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get a summary of project activity over the specified period.

        Args:
            days: Number of days to look back

        Returns:
            Activity summary
        """
        projects = self._load_projects()
        activity_log = self._load_activity_log()

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Filter recent activity
        recent_activity = [
            a for a in activity_log
            if a.get("timestamp", "") > cutoff
        ]

        # Count activity by project
        project_activity = defaultdict(int)
        for a in recent_activity:
            if a.get("action") == "activity_detected":
                project_activity[a.get("project", "")] += 1

        # Get active projects
        active_projects = [
            p for p in projects.values()
            if p.get("status") == "active"
        ]

        # Get stale/inactive counts
        stale_count = sum(1 for p in projects.values() if p.get("status") == "stale")
        inactive_count = sum(1 for p in projects.values() if p.get("status") == "inactive")

        return {
            "period_days": days,
            "total_projects": len(projects),
            "active_projects": len(active_projects),
            "stale_projects": stale_count,
            "inactive_projects": inactive_count,
            "total_activity_events": len(recent_activity),
            "most_active_projects": sorted(
                project_activity.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "recent_transitions": [
                a for a in recent_activity
                if a.get("action") == "status_transition"
            ][-10:]
        }


# Convenience functions

def auto_update_from_conversation(messages: List[Dict[str, str]],
                                   session_id: str,
                                   cwd: str = None) -> Dict[str, Any]:
    """Process a conversation and auto-update project states."""
    updater = ProjectAutoUpdater()
    return updater.process_conversation(messages, session_id, cwd)


def run_status_transitions() -> Dict[str, Any]:
    """Run automatic status transitions."""
    updater = ProjectAutoUpdater()
    return updater.run_status_transitions()


def get_activity_summary(days: int = 7) -> Dict[str, Any]:
    """Get project activity summary."""
    updater = ProjectAutoUpdater()
    return updater.get_activity_summary(days)


if __name__ == "__main__":
    # Test the auto-updater
    print("=== Project Auto-Updater Test ===\n")

    updater = ProjectAutoUpdater()

    # Test conversation
    test_messages = [
        {"role": "user", "content": "Let's continue working on ai-memory-mcp. I need to fix a bug."},
        {"role": "assistant", "content": "I'll help you debug the issue. Looking at /home/user/ai-memory-mcp/src/mcp_ultimate_memory.py"},
        {"role": "user", "content": "There's an error when calling the search function"},
        {"role": "assistant", "content": "I found the issue. Let me fix it."}
    ]

    # Test project detection
    print("1. Testing project detection...")
    detected = updater.detect_projects_from_conversation(test_messages)
    for p in detected:
        print(f"   - {p['name']}: confidence={p['confidence']:.2f}, sources={p['sources']}")

    # Test focus detection
    print("\n2. Testing focus detection...")
    focus = updater.detect_focus_from_conversation(test_messages)
    print(f"   Focus: {focus['focus']} (confidence={focus['confidence']:.2f})")
    print(f"   Keywords: {focus['keywords']}")

    # Test full processing
    print("\n3. Testing full conversation processing...")
    result = updater.process_conversation(test_messages, "test-session-123")
    print(f"   Detected projects: {result['detected_projects']}")
    for p in result['updated_projects']:
        print(f"   - {p['project']}: status={p['status']}, focus={p['focus']}")

    # Test activity summary
    print("\n4. Testing activity summary...")
    summary = updater.get_activity_summary(days=7)
    print(f"   Total projects: {summary['total_projects']}")
    print(f"   Active: {summary['active_projects']}, Stale: {summary['stale_projects']}, Inactive: {summary['inactive_projects']}")

    print("\n=== Test Complete ===")
