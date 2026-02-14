"""
Working Memory Manager - Claude.Me v6.0
Short-term storage for active reasoning, persists across compactions.

Part of Phase 2: Working Memory Integration
"""
import hashlib
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from reasoning_chain import ReasoningChain


class WorkingMemoryManager:
    """
    Manage working memory - active reasoning state that persists.

    Working memory holds:
    - Active reasoning chains (hypotheses being tested)
    - Scratch pad notes (temporary observations)
    - Active goal context
    - Key facts for current task

    It expires after a configurable period (default 4 hours).
    """

    DEFAULT_EXPIRY_HOURS = 4

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.working_memory_path = self.base_path / "working_memory"
        self._lock = threading.Lock()
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.working_memory_path.mkdir(parents=True, exist_ok=True)

    def _generate_session_id(self) -> str:
        """Generate a session ID based on current time."""
        return f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.working_memory_path / f"{session_id}.json"

    def create_session(
        self,
        session_id: str = None,
        active_goal: str = None,
        expiry_hours: int = DEFAULT_EXPIRY_HOURS
    ) -> Dict:
        """
        Create a new working memory session.

        Args:
            session_id: Optional session ID (auto-generated if not provided)
            active_goal: The current goal/task being worked on
            expiry_hours: Hours until this session expires

        Returns:
            The session data
        """
        if not session_id:
            session_id = self._generate_session_id()

        now = datetime.now()
        expires_at = now + timedelta(hours=expiry_hours)

        session = {
            "session_id": session_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "active_goal": active_goal,
            "reasoning_chains": [],
            "scratch_pad": {
                "notes": [],
                "temp_facts": [],
                "key_observations": []
            },
            "context": {},
            "archived": False
        }

        self._save_session(session)
        return session

    def _save_session(self, session: Dict):
        """Save a session to disk."""
        session["updated_at"] = datetime.now().isoformat()
        session_file = self.get_session_file(session["session_id"])

        with self._lock:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session, f, indent=2)

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a session by ID."""
        session_file = self.get_session_file(session_id)
        if not session_file.exists():
            return None

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WorkingMemory] Error loading session {session_id}: {e}")
            return None

    def get_active_session(self) -> Optional[Dict]:
        """Get the most recent active (non-expired) session."""
        now = datetime.now()
        sessions = []

        for session_file in self.working_memory_path.glob("sess_*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session = json.load(f)

                if session.get("archived"):
                    continue

                # Check expiry
                expires_at = datetime.fromisoformat(session["expires_at"])
                if expires_at > now:
                    sessions.append(session)
            except:
                continue

        if not sessions:
            return None

        # Return most recently updated
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions[0]

    def get_or_create_session(self, active_goal: str = None) -> Dict:
        """Get active session or create a new one."""
        session = self.get_active_session()
        if session:
            # Optionally update the goal
            if active_goal and active_goal != session.get("active_goal"):
                session["active_goal"] = active_goal
                self._save_session(session)
            return session
        return self.create_session(active_goal=active_goal)

    def add_reasoning_chain(
        self,
        session_id: str,
        hypothesis: str,
        parent_chain_id: str = None
    ) -> ReasoningChain:
        """
        Add a new reasoning chain to a session.

        Args:
            session_id: The session to add to
            hypothesis: The hypothesis being tested
            parent_chain_id: If this branches from another chain

        Returns:
            The created ReasoningChain
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        chain_id = f"rc_{hashlib.sha256(f'{session_id}{hypothesis}{datetime.now().timestamp()}'.encode()).hexdigest()[:8]}"

        chain = ReasoningChain(
            chain_id=chain_id,
            hypothesis=hypothesis,
            parent_chain_id=parent_chain_id
        )

        session["reasoning_chains"].append(chain.to_dict())
        self._save_session(session)

        return chain

    def update_reasoning_chain(
        self,
        session_id: str,
        chain_id: str,
        evidence: List[Dict] = None,
        evaluation: str = None,
        next_step: str = None,
        outcome: str = None
    ) -> Optional[ReasoningChain]:
        """Update an existing reasoning chain."""
        session = self.get_session(session_id)
        if not session:
            return None

        for i, chain_data in enumerate(session["reasoning_chains"]):
            if chain_data["chain_id"] == chain_id:
                chain = ReasoningChain.from_dict(chain_data)

                if evidence:
                    for ev in evidence:
                        chain.add_evidence(
                            evidence_type=ev.get("type", "observation"),
                            content=ev["content"],
                            confidence=ev.get("confidence", 0.8),
                            source=ev.get("source")
                        )

                if evaluation:
                    chain.evaluate(evaluation, next_step, outcome)

                session["reasoning_chains"][i] = chain.to_dict()
                self._save_session(session)
                return chain

        return None

    def get_reasoning_chain(self, session_id: str, chain_id: str) -> Optional[ReasoningChain]:
        """Get a specific reasoning chain."""
        session = self.get_session(session_id)
        if not session:
            return None

        for chain_data in session["reasoning_chains"]:
            if chain_data["chain_id"] == chain_id:
                return ReasoningChain.from_dict(chain_data)

        return None

    def get_active_chains(self, session_id: str) -> List[ReasoningChain]:
        """Get all active (not concluded) reasoning chains."""
        session = self.get_session(session_id)
        if not session:
            return []

        active = []
        for chain_data in session["reasoning_chains"]:
            chain = ReasoningChain.from_dict(chain_data)
            # Active = not yet concluded (no outcome)
            if not chain.outcome:
                active.append(chain)

        return active

    def add_scratch_note(self, session_id: str, note: str, category: str = "notes"):
        """Add a note to the scratch pad."""
        session = self.get_session(session_id)
        if not session:
            return

        if category not in session["scratch_pad"]:
            session["scratch_pad"][category] = []

        session["scratch_pad"][category].append({
            "content": note,
            "timestamp": datetime.now().isoformat()
        })

        self._save_session(session)

    def set_context(self, session_id: str, key: str, value: Any):
        """Set a context value."""
        session = self.get_session(session_id)
        if not session:
            return

        session["context"][key] = value
        self._save_session(session)

    def get_context(self, session_id: str, key: str = None) -> Any:
        """Get context value(s)."""
        session = self.get_session(session_id)
        if not session:
            return None

        if key:
            return session.get("context", {}).get(key)
        return session.get("context", {})

    def archive_session(self, session_id: str):
        """Mark a session as archived (won't show as active)."""
        session = self.get_session(session_id)
        if session:
            session["archived"] = True
            self._save_session(session)

    def extend_session(self, session_id: str, hours: int = DEFAULT_EXPIRY_HOURS):
        """Extend a session's expiry time."""
        session = self.get_session(session_id)
        if session:
            new_expiry = datetime.now() + timedelta(hours=hours)
            session["expires_at"] = new_expiry.isoformat()
            self._save_session(session)

    def get_summary(self, session_id: str) -> Dict:
        """Get a summary of a session's state."""
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        chains = [ReasoningChain.from_dict(c) for c in session.get("reasoning_chains", [])]
        active_chains = [c for c in chains if not c.outcome]
        completed_chains = [c for c in chains if c.outcome]

        return {
            "session_id": session_id,
            "active_goal": session.get("active_goal"),
            "created_at": session.get("created_at"),
            "expires_at": session.get("expires_at"),
            "total_chains": len(chains),
            "active_chains": len(active_chains),
            "completed_chains": len(completed_chains),
            "scratch_notes": len(session.get("scratch_pad", {}).get("notes", [])),
            "active_chain_summaries": [c.summarize() for c in active_chains],
            "context_keys": list(session.get("context", {}).keys())
        }

    def export_for_handoff(self, session_id: str) -> Dict:
        """
        Export session data for handoff (e.g., pre-compaction save).

        Returns a structured summary suitable for continuation.
        """
        session = self.get_session(session_id)
        if not session:
            return {}

        chains = [ReasoningChain.from_dict(c) for c in session.get("reasoning_chains", [])]
        active_chains = [c for c in chains if not c.outcome]

        return {
            "session_id": session_id,
            "active_goal": session.get("active_goal"),
            "incomplete_reasoning": [
                {
                    "chain_id": c.chain_id,
                    "hypothesis": c.hypothesis,
                    "status": c.evaluation,
                    "evidence_count": len(c.evidence),
                    "last_evidence": c.evidence[-1].content if c.evidence else None,
                    "next_step": c.next_step
                }
                for c in active_chains
            ],
            "key_context": session.get("context", {}),
            "scratch_pad": session.get("scratch_pad", {}),
            "exported_at": datetime.now().isoformat()
        }

    def import_from_handoff(self, handoff_data: Dict) -> str:
        """
        Import session from handoff data (e.g., post-compaction restore).

        Returns the new session ID.
        """
        # Create new session with transferred goal
        session = self.create_session(
            active_goal=handoff_data.get("active_goal"),
            expiry_hours=self.DEFAULT_EXPIRY_HOURS
        )
        session_id = session["session_id"]

        # Restore reasoning chains
        for incomplete in handoff_data.get("incomplete_reasoning", []):
            chain = self.add_reasoning_chain(
                session_id=session_id,
                hypothesis=incomplete.get("hypothesis", "Unknown hypothesis")
            )
            if incomplete.get("next_step"):
                self.update_reasoning_chain(
                    session_id=session_id,
                    chain_id=chain.chain_id,
                    evaluation=incomplete.get("status", "untested"),
                    next_step=incomplete.get("next_step")
                )

        # Restore context
        for key, value in handoff_data.get("key_context", {}).items():
            self.set_context(session_id, key, value)

        # Restore scratch pad notes
        scratch = handoff_data.get("scratch_pad", {})
        for note in scratch.get("notes", []):
            if isinstance(note, dict):
                self.add_scratch_note(session_id, note.get("content", str(note)))
            else:
                self.add_scratch_note(session_id, str(note))

        return session_id

    def get_stats(self) -> Dict:
        """Get working memory statistics."""
        now = datetime.now()
        total = 0
        active = 0
        archived = 0
        expired = 0

        for session_file in self.working_memory_path.glob("sess_*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session = json.load(f)
                total += 1

                if session.get("archived"):
                    archived += 1
                elif datetime.fromisoformat(session["expires_at"]) <= now:
                    expired += 1
                else:
                    active += 1
            except:
                continue

        return {
            "total_sessions": total,
            "active_sessions": active,
            "archived_sessions": archived,
            "expired_sessions": expired
        }
