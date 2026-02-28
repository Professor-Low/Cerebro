"""
Session Continuity Manager - Track and resume conversation threads
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class SessionContinuityManager:
    """
    Manage conversation threading and session continuity.

    Features:
    - Detect when user is continuing a previous conversation
    - Auto-inject last session summary
    - Track active sessions
    - Surface recent context
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.sessions_path = self.base_path / "session_states"
        self.conversations_path = self.base_path / "conversations"

    def get_active_sessions(self, days_back: int = 7) -> List[Dict]:
        """
        Get recently active sessions (last N days).

        Returns:
            [
                {
                    'session_id': 'proj_cerebral_20251220',
                    'last_active': '2025-12-20T15:30:00',
                    'message_count': 45,
                    'segment_count': 3,
                    'last_topic': 'WebSocket visualization updates'
                }
            ]
        """
        cutoff = datetime.now() - timedelta(days=days_back)
        active_sessions = []

        for session_file in self.sessions_path.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session = json.load(f)

                # Check if recently active
                last_active = session.get('updated_at') or session.get('created_at')
                if last_active:
                    timestamp = datetime.fromisoformat(last_active)
                    if timestamp > cutoff:
                        # Get last topic from most recent conversation
                        last_conv_id = session.get('last_conversation_id')
                        last_topic = self._get_conversation_topic(last_conv_id)

                        active_sessions.append({
                            'session_id': session['session_id'],
                            'last_active': last_active,
                            'message_count': session.get('message_count', 0),
                            'segment_count': session.get('segment_count', 0),
                            'last_topic': last_topic
                        })

            except Exception as e:
                print(f"[SessionContinuity] Error reading session {session_file}: {e}")
                continue

        # Sort by last active (most recent first)
        active_sessions.sort(key=lambda x: x['last_active'], reverse=True)
        return active_sessions

    def _get_conversation_topic(self, conversation_id: str) -> str:
        """Extract topic/summary from conversation"""
        if not conversation_id:
            return "Unknown"

        try:
            conv_file = self.conversations_path / f"{conversation_id}.json"
            if conv_file.exists():
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                # Try to get summary or first user message
                summary = conv.get('search_index', {}).get('summary', '')
                if summary:
                    return summary[:100]

                # Fallback: first user message
                for msg in conv.get('messages', []):
                    if msg.get('role') == 'user':
                        return msg.get('content', '')[:100]

        except Exception as e:
            print(f"[SessionContinuity] Error getting topic: {e}")

        return "Unknown topic"

    def get_session_summary(self, session_id: str, include_last_n: int = 1) -> Optional[str]:
        """
        Get summary of session for context injection.

        Args:
            session_id: Session to summarize
            include_last_n: Number of recent segments to include

        Returns:
            Formatted summary string for context injection
        """
        session_file = self.sessions_path / f"{session_id}.json"
        if not session_file.exists():
            return None

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session = json.load(f)

            # Get last N conversation IDs
            conv_ids = session.get('conversation_ids', [])
            recent_convs = conv_ids[-include_last_n:] if conv_ids else []

            if not recent_convs:
                return None

            # Build summary
            lines = []
            lines.append(f"[SESSION CONTINUITY] Resuming session: {session_id}")
            lines.append(f"Last active: {self._format_time_ago(session.get('updated_at'))}")
            lines.append(f"Total messages: {session.get('message_count', 0)}")
            lines.append("")

            # Include summaries of recent conversations
            for conv_id in recent_convs:
                conv_summary = self._get_conversation_summary(conv_id)
                if conv_summary:
                    lines.append("PREVIOUS CONTEXT:")
                    lines.append(conv_summary)
                    lines.append("")

            return "\n".join(lines)

        except Exception as e:
            print(f"[SessionContinuity] Error getting session summary: {e}")
            return None

    def _get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """Get formatted summary of a conversation"""
        try:
            conv_file = self.conversations_path / f"{conversation_id}.json"
            if not conv_file.exists():
                return None

            with open(conv_file, 'r', encoding='utf-8') as f:
                conv = json.load(f)

            # Extract key information
            summary = conv.get('search_index', {}).get('summary', '')
            key_takeaways = conv.get('search_index', {}).get('key_takeaways', [])
            decisions = conv.get('extracted_data', {}).get('decisions_made', [])
            problems = conv.get('extracted_data', {}).get('problems_solved', [])

            lines = []

            if summary:
                lines.append(f"Summary: {summary}")

            if key_takeaways:
                lines.append("Key points:")
                for takeaway in key_takeaways[:3]:
                    lines.append(f"  - {takeaway}")

            if decisions:
                lines.append("Decisions made:")
                for decision in decisions[:2]:
                    lines.append(f"  - {decision.get('decision', '')[:100]}")

            if problems:
                lines.append("Issues resolved:")
                for problem in problems[:2]:
                    lines.append(f"  - {problem.get('problem', '')[:100]}")

            return "\n".join(lines) if lines else None

        except Exception as e:
            print(f"[SessionContinuity] Error getting conversation summary: {e}")
            return None

    def _format_time_ago(self, timestamp: str) -> str:
        """Format timestamp as 'X hours/days ago'"""
        if not timestamp:
            return "unknown time"

        try:
            ts = datetime.fromisoformat(timestamp)
            now = datetime.now()
            delta = now - ts

            if delta.seconds < 3600:
                minutes = delta.seconds // 60
                return f"{minutes} minutes ago"
            elif delta.days < 1:
                hours = delta.seconds // 3600
                return f"{hours} hours ago"
            elif delta.days < 7:
                return f"{delta.days} days ago"
            elif delta.days < 30:
                weeks = delta.days // 7
                return f"{weeks} weeks ago"
            else:
                months = delta.days // 30
                return f"{months} months ago"
        except:
            return "unknown time"

    def detect_continuation(self, user_prompt: str, recent_sessions: List[Dict]) -> Optional[str]:
        """
        Detect if user is trying to continue a previous session.

        Indicators:
        - "continue working on..."
        - "last time we..."
        - "remember when we..."
        - Project/topic mentioned that matches recent session

        Returns:
            session_id to resume or None
        """
        prompt_lower = user_prompt.lower()

        # Check for explicit continuation keywords
        continuation_keywords = [
            'continue', 'resume', 'keep working on', 'go back to',
            'last time', 'remember when', 'earlier we', 'previously'
        ]

        is_continuation = any(kw in prompt_lower for kw in continuation_keywords)

        if is_continuation and recent_sessions:
            # Return most recent session
            return recent_sessions[0]['session_id']

        # Check if prompt mentions a recent session topic
        for session in recent_sessions[:3]:  # Check top 3 recent
            topic = session['last_topic'].lower()
            # Simple keyword matching (could be enhanced)
            topic_words = set(topic.split())
            prompt_words = set(prompt_lower.split())

            # If significant overlap, likely continuing that session
            overlap = topic_words.intersection(prompt_words)
            if len(overlap) >= 2:  # At least 2 words match
                return session['session_id']

        return None

    # ============================================================
    # PHASE 5: Enhanced Reasoning Continuity (v6.0)
    # ============================================================

    def save_handoff(self, handoff_data: Dict, reason: str = "compaction") -> str:
        """
        Save a session handoff for reasoning continuity.

        Args:
            handoff_data: Working memory export and other context
            reason: Why handoff is happening (compaction, session_end, etc.)

        Returns:
            Handoff ID
        """
        handoffs_path = self.base_path / "session_handoffs"
        handoffs_path.mkdir(parents=True, exist_ok=True)

        handoff_id = f"handoff_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        handoff_file = handoffs_path / f"{handoff_id}.json"

        handoff = {
            "id": handoff_id,
            "ended_at": datetime.now().isoformat(),
            "reason": reason,
            "incomplete_reasoning": handoff_data.get("incomplete_reasoning", []),
            "key_context": handoff_data.get("key_context", {}),
            "active_goal": handoff_data.get("active_goal"),
            "scratch_pad": handoff_data.get("scratch_pad", {}),
            "continuation_prompt": self._generate_continuation_prompt(handoff_data)
        }

        with open(handoff_file, 'w', encoding='utf-8') as f:
            json.dump(handoff, f, indent=2)

        return handoff_id

    def _generate_continuation_prompt(self, handoff_data: Dict) -> str:
        """Generate a prompt to help continue the session."""
        parts = []

        if handoff_data.get("active_goal"):
            parts.append(f"Goal: {handoff_data['active_goal']}")

        incomplete = handoff_data.get("incomplete_reasoning", [])
        if incomplete:
            parts.append("Incomplete reasoning:")
            for chain in incomplete[:3]:
                parts.append(f"  - {chain.get('hypothesis', 'Unknown')[:50]}... ({chain.get('status', 'unknown')})")
                if chain.get("next_step"):
                    parts.append(f"    Next: {chain['next_step'][:50]}")

        if not parts:
            return "No specific continuation context available."

        return "\n".join(parts)

    def get_latest_handoff(self) -> Optional[Dict]:
        """Get the most recent handoff."""
        handoffs_path = self.base_path / "session_handoffs"
        if not handoffs_path.exists():
            return None

        handoff_files = sorted(handoffs_path.glob("handoff_*.json"), reverse=True)
        if not handoff_files:
            return None

        try:
            with open(handoff_files[0], 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def get_recent_handoffs(self, hours: int = 48) -> List[Dict]:
        """Get recent handoffs within time window."""
        handoffs_path = self.base_path / "session_handoffs"
        if not handoffs_path.exists():
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        handoffs = []

        for handoff_file in handoffs_path.glob("handoff_*.json"):
            try:
                with open(handoff_file, 'r', encoding='utf-8') as f:
                    handoff = json.load(f)

                ended_at = handoff.get("ended_at")
                if ended_at:
                    ended_dt = datetime.fromisoformat(ended_at)
                    if ended_dt >= cutoff:
                        handoffs.append(handoff)
            except:
                continue

        handoffs.sort(key=lambda h: h.get("ended_at", ""), reverse=True)
        return handoffs

    def integrate_with_quick_facts(self, handoff_data: Dict):
        """
        Update quick_facts.json with handoff context.

        This ensures active_work is always up to date.
        """
        quick_facts_path = self.base_path / "quick_facts.json"

        try:
            if quick_facts_path.exists():
                with open(quick_facts_path, 'r', encoding='utf-8') as f:
                    quick_facts = json.load(f)
            else:
                quick_facts = {}

            # Update active_work
            active_work = quick_facts.get("active_work", {})

            if handoff_data.get("active_goal"):
                active_work["project"] = handoff_data.get("active_goal", "Unknown")

            if handoff_data.get("incomplete_reasoning"):
                # Use first incomplete chain's next_step as next_action
                for chain in handoff_data["incomplete_reasoning"]:
                    if chain.get("next_step"):
                        active_work["next_action"] = chain["next_step"]
                        break

            active_work["last_updated"] = datetime.now().isoformat()
            quick_facts["active_work"] = active_work

            with open(quick_facts_path, 'w', encoding='utf-8') as f:
                json.dump(quick_facts, f, indent=2)

        except Exception as e:
            print(f"[SessionContinuity] Error updating quick_facts: {e}")
