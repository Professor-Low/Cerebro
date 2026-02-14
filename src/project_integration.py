"""
Helper for integrating project tracking into conversation flow.

MULTI-AGENT NOTICE: Agent 4 (ai_memory_ultimate) and Agent 6 (mcp_ultimate_memory) will use this.

Usage in ai_memory_ultimate.py:
    from project_integration import process_conversation_for_projects

    project_results = process_conversation_for_projects(
        messages=messages,
        conversation_id=conversation_id,
        file_paths=extracted_data.get("file_paths", [])
    )

Usage in mcp_ultimate_memory.py:
    from project_integration import get_project_context

    # Get context for current file
    project = get_project_context(file_path="/path/to/file.py")
"""

from typing import Dict, List, Optional

from project_state_detector import ProjectStateDetector
from project_tracker import ProjectTracker


def process_conversation_for_projects(messages: list,
                                      conversation_id: str,
                                      file_paths: list = None) -> Dict:
    """
    Process conversation to detect and update project states.

    Args:
        messages: List of conversation messages with role/content
        conversation_id: Unique conversation identifier
        file_paths: List of file paths mentioned (optional)

    Returns: {
        "projects_detected": [...],
        "projects_updated": [...],
        "state_changes": {...}
    }
    """
    detector = ProjectStateDetector()
    tracker = ProjectTracker()

    # Combine all message content
    full_text = " ".join([m.get("content", "") for m in messages if isinstance(m, dict)])

    # Detect projects mentioned
    projects_detected = detector.detect_project_mention(full_text)

    # Detect state changes
    state_changes = detector.detect_state_changes(full_text)

    # Detect file activity
    file_activity = {}
    if file_paths:
        # Extract just the path strings if they're dicts
        path_strings = []
        for fp in file_paths:
            if isinstance(fp, dict):
                path_strings.append(fp.get("path", ""))
            else:
                path_strings.append(str(fp))
        file_activity = detector.detect_file_activity(path_strings)

    # Infer priority
    priority = detector.infer_priority(full_text)

    projects_updated = []

    for project_info in projects_detected:
        project_id = project_info["project_id"]

        # Update project with detected changes
        updates = {"conversation_id": conversation_id}

        # Set priority if detected as high/critical
        if priority in ["high", "critical"]:
            updates["priority"] = priority

        # Update current focus from in_progress state
        if state_changes.get("in_progress"):
            updates["current_focus"] = state_changes["in_progress"][0]

        # Add blockers
        if state_changes.get("blocker"):
            for blocker in state_changes["blocker"]:
                tracker.update_project(project_id, add_blocker=blocker, conversation_id=conversation_id)

        # Add next steps
        if state_changes.get("next_step"):
            for step in state_changes["next_step"]:
                tracker.update_project(project_id, add_next_step=step, conversation_id=conversation_id)

        # Add completed milestones
        if state_changes.get("finished"):
            for milestone in state_changes["finished"]:
                tracker.update_project(project_id, add_milestone_completed=milestone, conversation_id=conversation_id)

        # Add started milestones
        if state_changes.get("started"):
            for milestone in state_changes["started"]:
                tracker.update_project(project_id, add_milestone_in_progress=milestone, conversation_id=conversation_id)

        # Add files
        if project_id in file_activity:
            for file_path in file_activity[project_id]:
                tracker.update_project(project_id, add_file=file_path, conversation_id=conversation_id)

        # Update with main changes
        tracker.update_project(project_id, **updates)

        projects_updated.append(project_id)

    return {
        "projects_detected": [p["project_id"] for p in projects_detected],
        "projects_updated": projects_updated,
        "state_changes": state_changes,
        "detection_details": projects_detected
    }


def get_project_context(file_path: str = None, project_id: str = None) -> Optional[Dict]:
    """
    Get project context for current work.

    Args:
        file_path: File path to search for (optional)
        project_id: Direct project ID lookup (optional)

    Returns: Project state dict or None
    """
    tracker = ProjectTracker()

    if project_id:
        return tracker.get_project(project_id)
    elif file_path:
        return tracker.search_project_by_path(file_path)

    return None


def get_all_active_projects() -> List[Dict]:
    """
    Get all active projects.

    Returns: List of project dicts
    """
    tracker = ProjectTracker()
    return tracker.get_active_projects()


def get_project_stats() -> Dict:
    """
    Get project statistics.

    Returns: {
        "total_projects": int,
        "active_projects": int,
        "total_blockers": int,
        "total_next_steps": int,
        "most_recent_project": str
    }
    """
    tracker = ProjectTracker()
    return tracker.get_stats()


def complete_project_step(project_id: str, step: str) -> bool:
    """
    Mark a next step as completed.

    Args:
        project_id: Project identifier
        step: Step description to complete

    Returns: True if successful
    """
    tracker = ProjectTracker()
    try:
        tracker.complete_next_step(project_id, step)
        return True
    except:
        return False


def resolve_project_blocker(project_id: str, blocker: str) -> bool:
    """
    Resolve a project blocker.

    Args:
        project_id: Project identifier
        blocker: Blocker description to resolve

    Returns: True if successful
    """
    tracker = ProjectTracker()
    try:
        tracker.resolve_blocker(project_id, blocker)
        return True
    except:
        return False
