"""
CWD Context Detector - Detect project and load relevant context from CWD.

Part of Phase 1 Enhancement in the All-Knowing Brain PRD.
Detects project from current working directory and loads:
- Project name and type
- Recent conversations about this project
- Related solutions and antipatterns
- Uncommitted changes (git integration)
"""

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


class CWDDetector:
    """
    Detects project context from current working directory.
    Loads relevant past context for the detected project.
    """

    # Project indicator files and their types
    PROJECT_INDICATORS = {
        'pyproject.toml': 'python',
        'setup.py': 'python',
        'requirements.txt': 'python',
        'package.json': 'javascript',
        'tsconfig.json': 'typescript',
        'Cargo.toml': 'rust',
        'go.mod': 'go',
        'Makefile': 'make',
        'CMakeLists.txt': 'cpp',
        'pom.xml': 'java',
        'build.gradle': 'java',
        '.git': 'git',
    }

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.conversations_path = self.base_path / "conversations"
        self.projects_path = self.base_path / "projects"

    def detect_project(self, cwd: str) -> Dict[str, Any]:
        """
        Detect project from current working directory.

        Args:
            cwd: Current working directory path

        Returns:
            Dict with project info:
                - name: Project name
                - type: Project type (python, javascript, etc.)
                - path: Project root path
                - indicators: List of detected indicators
        """
        cwd_path = Path(cwd).resolve()

        result = {
            "name": None,
            "type": None,
            "path": str(cwd_path),
            "indicators": [],
            "git_info": None,
        }

        # Check current directory and parent for indicators
        for check_path in [cwd_path, cwd_path.parent]:
            for indicator, proj_type in self.PROJECT_INDICATORS.items():
                if (check_path / indicator).exists():
                    result["indicators"].append(indicator)
                    if not result["type"]:
                        result["type"] = proj_type
                    if not result["name"]:
                        result["name"] = check_path.name
                        result["path"] = str(check_path)

        # Get git info if available
        result["git_info"] = self._get_git_info(cwd_path)

        # Try to get project name from pyproject.toml or package.json
        if not result["name"]:
            result["name"] = self._extract_project_name(cwd_path)

        # Fall back to directory name
        if not result["name"]:
            result["name"] = cwd_path.name

        return result

    def _extract_project_name(self, path: Path) -> Optional[str]:
        """Extract project name from config files."""
        # Try pyproject.toml
        pyproject = path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                # Simple parse for name
                for line in content.split("\n"):
                    if line.strip().startswith("name"):
                        # name = "project-name"
                        if "=" in line and '"' in line:
                            return line.split('"')[1]
            except Exception:
                pass

        # Try package.json
        package_json = path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, "r") as f:
                    data = json.load(f)
                    return data.get("name")
            except Exception:
                pass

        # Try Cargo.toml
        cargo = path / "Cargo.toml"
        if cargo.exists():
            try:
                content = cargo.read_text()
                for line in content.split("\n"):
                    if line.strip().startswith("name"):
                        if "=" in line and '"' in line:
                            return line.split('"')[1]
            except Exception:
                pass

        return None

    def _get_git_info(self, path: Path) -> Optional[Dict[str, Any]]:
        """Get git repository info if available."""
        try:
            # Check if this is a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return None

            git_info = {
                "is_repo": True,
                "branch": None,
                "has_changes": False,
                "modified_files": [],
                "untracked_files": [],
            }

            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=5
            )
            if branch_result.returncode == 0:
                git_info["branch"] = branch_result.stdout.strip()

            # Get status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=5
            )
            if status_result.returncode == 0 and status_result.stdout.strip():
                for line in status_result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    status_code = line[:2]
                    filepath = line[3:].strip()
                    # Skip noise
                    if any(filepath.endswith(ext) for ext in ['.pyc', '.pyo', '__pycache__']):
                        continue
                    if status_code.startswith("?"):
                        git_info["untracked_files"].append(filepath)
                    else:
                        git_info["modified_files"].append(filepath)

                git_info["has_changes"] = bool(
                    git_info["modified_files"] or git_info["untracked_files"]
                )

            return git_info

        except Exception:
            return None

    def load_project_context(self,
                             project_name: str,
                             max_conversations: int = 5,
                             max_age_days: int = 30) -> Dict[str, Any]:
        """
        Load relevant context for a project from memory.

        Args:
            project_name: Name of the project
            max_conversations: Maximum conversations to load
            max_age_days: Maximum age of conversations to consider

        Returns:
            Dict with project context:
                - conversations: Recent project conversations
                - solutions: Known solutions for this project
                - pending_work: Any pending work detected
        """
        result = {
            "project_name": project_name,
            "conversations": [],
            "solutions": [],
            "pending_work": [],
            "total_found": 0,
        }

        if not project_name:
            return result

        project_lower = project_name.lower()
        cutoff = datetime.now() - timedelta(days=max_age_days)

        # Search conversations
        if self.conversations_path.exists():
            conversations = []
            for conv_file in self.conversations_path.glob("*.json"):
                try:
                    with open(conv_file, "r", encoding="utf-8") as f:
                        conv = json.load(f)

                    # Check for project match
                    conv_project = conv.get("metadata", {}).get("project", "")
                    if project_lower not in conv_project.lower():
                        # Check topics
                        topics = conv.get("metadata", {}).get("topics", [])
                        if not any(project_lower in t.lower() for t in topics):
                            continue

                    # Check timestamp
                    timestamp = conv.get("timestamp", "")
                    if timestamp:
                        try:
                            conv_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                            if hasattr(conv_time, 'tzinfo') and conv_time.tzinfo:
                                conv_time = conv_time.replace(tzinfo=None)
                            if conv_time < cutoff:
                                continue
                        except Exception:
                            pass

                    conversations.append({
                        "id": conv.get("id", conv_file.stem),
                        "summary": conv.get("metadata", {}).get("summary", "")[:200],
                        "timestamp": timestamp,
                        "topics": conv.get("metadata", {}).get("topics", [])[:5],
                    })

                except Exception:
                    continue

            # Sort by timestamp (most recent first)
            conversations.sort(
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )

            result["conversations"] = conversations[:max_conversations]
            result["total_found"] = len(conversations)

        # Load project state if available
        project_state = self._load_project_state(project_name)
        if project_state:
            result["pending_work"] = project_state.get("pending_work", [])
            result["solutions"] = project_state.get("solutions", [])

        return result

    def _load_project_state(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Load project state from projects directory."""
        if not self.projects_path.exists():
            return None

        # Try exact match first
        project_file = self.projects_path / f"{project_name}.json"
        if not project_file.exists():
            # Try case-insensitive search
            for f in self.projects_path.glob("*.json"):
                if f.stem.lower() == project_name.lower():
                    project_file = f
                    break

        if not project_file.exists():
            return None

        try:
            with open(project_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def get_full_context(self, cwd: str) -> Dict[str, Any]:
        """
        Get full context for a CWD including project detection and history.

        Args:
            cwd: Current working directory

        Returns:
            Complete context dict
        """
        # Detect project
        project_info = self.detect_project(cwd)

        # Load project context if we found a project
        project_context = {}
        if project_info.get("name"):
            project_context = self.load_project_context(project_info["name"])

        return {
            "project": project_info,
            "context": project_context,
            "detected_at": datetime.now().isoformat(),
        }


def detect_from_cwd(cwd: str) -> Dict[str, Any]:
    """
    Convenience function to detect project from CWD.

    Args:
        cwd: Current working directory

    Returns:
        Project detection result
    """
    detector = CWDDetector()
    return detector.detect_project(cwd)


def get_cwd_context(cwd: str) -> Dict[str, Any]:
    """
    Convenience function to get full CWD context.

    Args:
        cwd: Current working directory

    Returns:
        Full context including project history
    """
    detector = CWDDetector()
    return detector.get_full_context(cwd)


if __name__ == "__main__":
    import os

    # Test with current directory
    cwd = os.getcwd()
    detector = CWDDetector()

    print("=== CWD Detection Test ===")
    project = detector.detect_project(cwd)
    print(f"Project: {project['name']}")
    print(f"Type: {project['type']}")
    print(f"Path: {project['path']}")
    print(f"Indicators: {project['indicators']}")

    if project['git_info']:
        git = project['git_info']
        print(f"Git branch: {git['branch']}")
        print(f"Has changes: {git['has_changes']}")
        if git['modified_files']:
            print(f"Modified: {git['modified_files'][:5]}")

    print("\n=== Project Context ===")
    if project['name']:
        context = detector.load_project_context(project['name'])
        print(f"Conversations found: {context['total_found']}")
        for conv in context['conversations'][:3]:
            print(f"  - {conv['summary'][:60]}...")
