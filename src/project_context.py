"""
Project Context Manager - Detect current project and provide relevant context.

Maps working directories to known projects and provides:
- Project state (active_work from quick_facts.json)
- Recent files worked on
- Project-specific settings
- Known blockers and next actions
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

# Auto-populated from user's project history
KNOWN_PROJECTS: Dict[str, str] = {}

# Project-specific context templates
PROJECT_CONTEXTS = {
    "ai_memory": {
        "name": "AI Memory MCP",
        "description": "Persistent memory system for Claude with semantic search",
        "key_files": [
            "src/mcp_ultimate_memory.py",
            "src/ai_memory_ultimate.py",
            "src/ai_embeddings_engine.py"
        ],
        "quick_tips": [
            "MCP timeout: 60s for embedding operations",
            "Pre-load embedding model at startup",
            "Test with: python -c 'from mcp_ultimate_memory import *; print(\"OK\")'"
        ],
        "common_issues": [
            "NAS connectivity - check Z: is mounted",
            "Embedding model loading timeout",
            "FAISS index corruption"
        ]
    },
    "cerebral_interface": {
        "name": "Cerebral Interface",
        "description": "3D brain visualization using Three.js",
        "key_files": [
            "src/App.js",
            "src/components/Brain.js",
            "server.js"
        ],
        "quick_tips": [
            "Port 8080 for main, 8081 for WebSocket",
            "Use 'npm run dev' for development",
            "Three.js OrbitControls for camera"
        ]
    },
    "lead-gen": {
        "name": "Lead Generation",
        "description": "B2B lead generation scrapers",
        "key_files": [
            "scrapers/",
            "enrichment/"
        ],
        "quick_tips": [
            "Check robots.txt before scraping",
            "Use rate limiting",
            "Store results in CSV format"
        ]
    }
}


class ProjectContextManager:
    """Manages project-aware context injection."""

    def __init__(self, quick_facts_path: str = ""):
        if not quick_facts_path:
            from config import DATA_DIR
            quick_facts_path = str(DATA_DIR / "quick_facts.json")
        self.quick_facts_path = Path(quick_facts_path)

    def detect_project(self, cwd: str) -> Optional[str]:
        """Detect project from current working directory."""
        if not cwd:
            return None

        # Normalize path
        cwd_normalized = cwd.replace("/", "\\")

        # Try exact match first
        if cwd_normalized in KNOWN_PROJECTS:
            return KNOWN_PROJECTS[cwd_normalized]

        # Try prefix matching (for subdirectories)
        for project_path, project_id in KNOWN_PROJECTS.items():
            if cwd_normalized.startswith(project_path):
                return project_id

        return None

    def get_active_work(self) -> Dict[str, Any]:
        """Get active_work section from quick_facts.json."""
        try:
            if self.quick_facts_path.exists():
                with open(self.quick_facts_path, "r", encoding="utf-8") as f:
                    facts = json.load(f)
                return facts.get("active_work", {})
        except Exception:
            pass
        return {}

    def get_project_context(self, project_id: str) -> Dict[str, Any]:
        """Get context for a specific project."""
        return PROJECT_CONTEXTS.get(project_id, {})

    def build_context_string(self, cwd: str) -> str:
        """Build a context string for injection based on CWD."""
        project_id = self.detect_project(cwd)

        if not project_id:
            return ""

        parts = []

        # Project info
        project_ctx = self.get_project_context(project_id)
        if project_ctx:
            parts.append(f"[Current Project: {project_ctx.get('name', project_id)}]")
            parts.append(f"Description: {project_ctx.get('description', '')}")

            if project_ctx.get("quick_tips"):
                parts.append("Quick Tips:")
                for tip in project_ctx["quick_tips"][:3]:
                    parts.append(f"  • {tip}")

        # Active work from quick_facts
        active_work = self.get_active_work()
        if active_work and active_work.get("project"):
            parts.append("\n[Active Work]")
            parts.append(f"Phase: {active_work.get('phase_name', active_work.get('current_phase', 'unknown'))}")
            if active_work.get("next_action"):
                parts.append(f"Next: {active_work.get('next_action')[:150]}")
            if active_work.get("last_completed"):
                parts.append(f"Last: {active_work.get('last_completed')[:100]}")

        return "\n".join(parts) if parts else ""


def get_project_context_for_cwd(cwd: str) -> str:
    """Convenience function for hooks."""
    manager = ProjectContextManager()
    return manager.build_context_string(cwd)


if __name__ == "__main__":
    # Test
    manager = ProjectContextManager()

    test_dirs = [
        "/home/user/project-a",
        "/home/user/project-a/src",
        "/home/user/project-b",
        "/home/user/unknown-dir"
    ]

    for d in test_dirs:
        project = manager.detect_project(d)
        print(f"{d}")
        print(f"  -> Project: {project}")
        print(f"  -> Context: {manager.build_context_string(d)[:100]}...")
        print()
