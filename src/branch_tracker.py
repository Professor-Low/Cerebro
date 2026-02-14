"""
Branch Tracker for AI Memory System
Tracks conversation branches (different exploration approaches)
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class BranchTracker:
    """
    Track conversation branches (different exploration approaches).
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.branches_path = self.base_path / "branches"
        self.branches_file = self.branches_path / "exploration_branches.json"

        # Create directory
        self.branches_path.mkdir(parents=True, exist_ok=True)

        # Load branches
        self.branches = self._load_branches()

    def _load_branches(self) -> Dict:
        """Load branches from file"""
        if self.branches_file.exists():
            with open(self.branches_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'branches': {}}

    def _save_branches(self):
        """Save branches to file"""
        with open(self.branches_file, 'w', encoding='utf-8') as f:
            json.dump(self.branches, f, indent=2, ensure_ascii=False)

    def create_branch(self, name: str, description: str = "",
                     parent_conversation_id: Optional[str] = None) -> str:
        """
        Create a new exploration branch.

        Args:
            name: Branch name
            description: What this approach explores
            parent_conversation_id: Optional parent conversation

        Returns:
            Branch ID
        """
        branch_id = f"branch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.branches['branches'][branch_id] = {
            'id': branch_id,
            'name': name,
            'description': description,
            'parent_conversation_id': parent_conversation_id,
            'created_at': datetime.now().isoformat(),
            'status': 'exploring',  # exploring, chosen, abandoned
            'reason': None,
            'conversation_ids': []
        }

        self._save_branches()
        return branch_id

    def add_conversation_to_branch(self, branch_id: str, conversation_id: str):
        """Add a conversation to a branch"""
        if branch_id in self.branches['branches']:
            if conversation_id not in self.branches['branches'][branch_id]['conversation_ids']:
                self.branches['branches'][branch_id]['conversation_ids'].append(conversation_id)
                self._save_branches()

    def mark_branch_chosen(self, branch_id: str, reason: str):
        """Mark a branch as the chosen approach"""
        if branch_id in self.branches['branches']:
            self.branches['branches'][branch_id]['status'] = 'chosen'
            self.branches['branches'][branch_id]['reason'] = reason
            self.branches['branches'][branch_id]['decided_at'] = datetime.now().isoformat()
            self._save_branches()

    def mark_branch_abandoned(self, branch_id: str, reason: str):
        """Mark a branch as abandoned"""
        if branch_id in self.branches['branches']:
            self.branches['branches'][branch_id]['status'] = 'abandoned'
            self.branches['branches'][branch_id]['reason'] = reason
            self.branches['branches'][branch_id]['decided_at'] = datetime.now().isoformat()
            self._save_branches()

    def get_branch(self, branch_id: str) -> Optional[Dict]:
        """Get branch details"""
        return self.branches['branches'].get(branch_id)

    def get_all_branches(self, status: Optional[str] = None) -> List[Dict]:
        """Get all branches, optionally filtered by status"""
        branches = list(self.branches['branches'].values())

        if status:
            branches = [b for b in branches if b['status'] == status]

        return sorted(branches, key=lambda x: x['created_at'], reverse=True)
