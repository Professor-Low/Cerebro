"""
Project State Tracking System
Maintains state of all active projects.

MULTI-AGENT NOTICE: This is Agent 3's exclusive domain.
Agent 4 (memory backend) and Agent 6 (MCP tools) will integrate this.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ProjectTracker:
    """Manages project state persistence and retrieval."""

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.projects_dir = self.base_path / "project_states"
        self.active_projects_file = self.projects_dir / "active_projects.json"
        self.history_file = self.projects_dir / "project_history.jsonl"

        # Ensure directory exists
        self.projects_dir.mkdir(parents=True, exist_ok=True)

        # Load active projects
        self.projects = self._load_active_projects()

    def update_project(self,
                      project_id: str,
                      name: str = None,
                      status: str = None,
                      current_focus: str = None,
                      priority: str = None,
                      add_blocker: str = None,
                      add_next_step: str = None,
                      add_file: str = None,
                      add_technology: str = None,
                      add_milestone_completed: str = None,
                      add_milestone_in_progress: str = None,
                      conversation_id: str = None) -> Dict:
        """
        Update project state.

        Returns: Updated project dict
        """
        # Get or create project
        if project_id not in self.projects:
            self.projects[project_id] = {
                "project_id": project_id,
                "name": name or project_id.replace("-", " ").title(),
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "last_worked": datetime.now().isoformat(),
                "current_focus": "",
                "priority": "medium",
                "blockers": [],
                "next_steps": [],
                "files": [],
                "technologies": [],
                "milestones": {
                    "completed": [],
                    "in_progress": [],
                    "planned": []
                },
                "conversations": []
            }

        project = self.projects[project_id]

        # Update fields
        if name:
            project["name"] = name
        if status:
            project["status"] = status
        if current_focus:
            project["current_focus"] = current_focus
        if priority:
            project["priority"] = priority

        project["last_worked"] = datetime.now().isoformat()

        # Add items
        if add_blocker and add_blocker not in project["blockers"]:
            project["blockers"].append(add_blocker)

        if add_next_step and add_next_step not in project["next_steps"]:
            project["next_steps"].append(add_next_step)

        if add_file and add_file not in project["files"]:
            project["files"].append(add_file)

        if add_technology and add_technology not in project["technologies"]:
            project["technologies"].append(add_technology)

        if add_milestone_completed and add_milestone_completed not in project["milestones"]["completed"]:
            # Remove from in_progress if there
            if add_milestone_completed in project["milestones"]["in_progress"]:
                project["milestones"]["in_progress"].remove(add_milestone_completed)
            project["milestones"]["completed"].append(add_milestone_completed)

        if add_milestone_in_progress and add_milestone_in_progress not in project["milestones"]["in_progress"]:
            project["milestones"]["in_progress"].append(add_milestone_in_progress)

        if conversation_id and conversation_id not in project["conversations"]:
            project["conversations"].append(conversation_id)

        # Save
        self._save_active_projects()

        # Log to history
        self._log_history(project_id, {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "updates": {
                "status": status,
                "current_focus": current_focus,
                "blocker_added": add_blocker,
                "next_step_added": add_next_step,
                "milestone_completed": add_milestone_completed
            }
        })

        return project

    def get_project(self, project_id: str) -> Optional[Dict]:
        """Get project by ID."""
        return self.projects.get(project_id)

    def get_active_projects(self, status: str = "active") -> List[Dict]:
        """Get all active projects."""
        return [
            p for p in self.projects.values()
            if p["status"] == status
        ]

    def search_project_by_path(self, file_path: str) -> Optional[Dict]:
        """Find project that contains this file path."""
        path_lower = file_path.lower()
        for project in self.projects.values():
            for project_file in project["files"]:
                if project_file.lower() in path_lower or path_lower in project_file.lower():
                    return project
        return None

    def get_project_timeline(self, project_id: str) -> List[Dict]:
        """Get project history timeline."""
        timeline = []
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("project_id") == project_id:
                            timeline.append(entry)
        return timeline

    def complete_next_step(self, project_id: str, step: str):
        """Mark a next step as completed."""
        if project_id in self.projects:
            project = self.projects[project_id]
            if step in project["next_steps"]:
                project["next_steps"].remove(step)
                # Add to completed milestones
                project["milestones"]["completed"].append(step)
                self._save_active_projects()

    def resolve_blocker(self, project_id: str, blocker: str):
        """Remove a blocker."""
        if project_id in self.projects:
            project = self.projects[project_id]
            if blocker in project["blockers"]:
                project["blockers"].remove(blocker)
                self._save_active_projects()

    def _load_active_projects(self) -> Dict:
        """Load active projects from file."""
        if self.active_projects_file.exists():
            try:
                with open(self.active_projects_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_active_projects(self):
        """Save active projects to file."""
        with open(self.active_projects_file, 'w', encoding='utf-8') as f:
            json.dump(self.projects, f, indent=2)

    def _log_history(self, project_id: str, entry: Dict):
        """Log project update to history."""
        entry["project_id"] = project_id
        with open(self.history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')

    def get_stats(self) -> Dict:
        """Get project statistics."""
        active = [p for p in self.projects.values() if p["status"] == "active"]
        return {
            "total_projects": len(self.projects),
            "active_projects": len(active),
            "total_blockers": sum(len(p["blockers"]) for p in active),
            "total_next_steps": sum(len(p["next_steps"]) for p in active),
            "most_recent_project": max(
                active,
                key=lambda p: p["last_worked"]
            )["name"] if active else None
        }
