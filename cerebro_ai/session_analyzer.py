"""
Session Analyzer - Smart Detection of Continuable Work
Distinguishes between work-in-progress that should be resumed vs one-off tasks.
Supports cross-device awareness for multi-device workflows.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from device_registry import get_current_device_tag


class SessionAnalyzer:
    """
    Analyzes conversations to determine if they represent continuable work
    or one-off tasks that don't need follow-up.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.conversations_path = self.base_path / "conversations"

        # One-off task indicators (low continuation score)
        self.oneoff_patterns = [
            r"(?:what(?:'s| is) (?:my|the))",  # "what's my IP"
            r"(?:grab|get|show|list|display) (?:my|the)",  # "grab my IP"
            r"(?:how do (?:I|you))",  # Quick how-to questions
            r"(?:can you (?:tell|show|get))",  # Simple requests
            r"(?:remind me)",  # Memory lookups
            r"(?:what time|what date)",  # Time queries
            r"(?:thanks|thank you|perfect|great)$",  # Closing statements
            r"(?:^hi$|^hello$|^hey$)",  # Greetings only
        ]

        # Continuable work indicators (high continuation score)
        self.continuable_patterns = [
            r"(?:implement|build|create|develop|write)",  # Development work
            r"(?:fix|debug|resolve|troubleshoot)",  # Debugging
            r"(?:refactor|optimize|improve|enhance)",  # Improvements
            r"(?:add (?:a |the )?(?:feature|function|method|class))",  # Feature work
            r"(?:working on|continue|pick up|resume)",  # Continuation language
            r"(?:project|codebase|repository|repo)",  # Project work
            r"(?:error|bug|issue|problem)",  # Problem solving
            r"(?:test|verify|validate)",  # Testing work
            r"(?:configure|setup|install)",  # Setup work
            r"(?:TODO|FIXME|WIP)",  # Work markers
        ]

        # File modification indicators
        self.file_work_patterns = [
            r"\.py$|\.js$|\.ts$|\.go$|\.rs$|\.java$|\.cpp$|\.c$",  # Code files
            r"(?:created|modified|updated|edited|wrote)",  # File actions
            r"(?:saved to|wrote to|created file)",  # File creation
        ]

    def get_recent_sessions(self, hours: int = 48, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation sessions within time window.

        Optimized: pre-filters by file modification time to avoid reading
        every JSON file on NAS (which can be 300+ and very slow).
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        cutoff_ts = cutoff.timestamp()
        sessions = []

        # Pre-filter: only consider files modified within the time window
        # This avoids loading 300+ old conversation files from NAS
        try:
            candidates = []
            for conv_file in self.conversations_path.glob("*.json"):
                try:
                    mtime = conv_file.stat().st_mtime
                    if mtime >= cutoff_ts:
                        candidates.append((mtime, conv_file))
                except OSError:
                    continue

            # Sort by mtime descending, only read the most recent ones
            candidates.sort(key=lambda x: x[0], reverse=True)
            # Cap at 2x limit to avoid reading too many files
            candidates = candidates[:limit * 2]
        except OSError:
            return []

        for _mtime, conv_file in candidates:
            try:
                with open(conv_file, "r", encoding="utf-8") as f:
                    conv = json.load(f)

                # Parse timestamp
                timestamp_str = conv.get("timestamp", "")
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        if timestamp.tzinfo:
                            timestamp = timestamp.replace(tzinfo=None)
                    except:
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()

                if timestamp > cutoff:
                    sessions.append({
                        "id": conv.get("id", conv_file.stem),
                        "timestamp": timestamp,
                        "conversation": conv,
                        "file_path": str(conv_file)
                    })
            except Exception:
                continue

        # Sort by timestamp descending
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        return sessions[:limit]

    def analyze_session(self, conversation: Dict[str, Any],
                       current_device: str = None) -> Dict[str, Any]:
        """
        Analyze a conversation and return continuation analysis.

        Args:
            conversation: The conversation dict to analyze
            current_device: Current device tag for cross-device comparison

        Returns:
            Dict with:
            - is_continuable: bool - Should this be offered for continuation?
            - confidence: float - How confident (0.0-1.0)
            - reason: str - Why it is/isn't continuable
            - summary: str - Brief summary of what was being worked on
            - suggested_context: list - Files/topics to inject if continuing
            - cross_device_warning: bool - True if session is from different device
            - same_device: bool - True if session is from same device
        """
        messages = conversation.get("messages", [])
        metadata = conversation.get("metadata", {})
        extracted = conversation.get("extracted_data", {})

        # Get device info
        session_device = metadata.get("device_tag", "unknown")
        if current_device is None:
            try:
                current_device = get_current_device_tag()
            except:
                current_device = "unknown"

        same_device = (session_device == current_device) or session_device == "unknown"

        # Calculate various scores
        scores = {
            "message_count": self._score_message_count(messages),
            "content_depth": self._score_content_depth(messages),
            "work_indicators": self._score_work_indicators(messages),
            "file_work": self._score_file_work(messages, extracted),
            "oneoff_indicators": self._score_oneoff_indicators(messages),
            "has_pending_work": self._score_pending_work(messages, extracted),
            "session_duration": self._score_session_duration(conversation),
            "importance": self._score_importance(metadata),
        }

        # Calculate weighted continuation score
        weights = {
            "message_count": 0.10,
            "content_depth": 0.15,
            "work_indicators": 0.25,
            "file_work": 0.20,
            "oneoff_indicators": -0.30,  # Negative weight
            "has_pending_work": 0.25,
            "session_duration": 0.05,
            "importance": 0.10,
        }

        continuation_score = sum(
            scores[key] * weights[key]
            for key in weights
        )

        # Apply cross-device penalty (0.7x) if different device
        cross_device_warning = False
        if not same_device:
            cross_device_warning = True
            continuation_score *= 0.7
            scores["cross_device_penalty"] = 0.7

        # Normalize to 0-1
        continuation_score = max(0.0, min(1.0, continuation_score))

        # Determine if continuable (threshold: 0.4)
        is_continuable = continuation_score >= 0.4

        # Generate reason
        reason = self._generate_reason(scores, is_continuable)
        if cross_device_warning and is_continuable:
            reason += f" (from {session_device} device)"

        # Generate summary
        summary = self._generate_summary(conversation)

        # Suggest context to inject
        suggested_context = self._suggest_context(conversation, extracted)

        return {
            "is_continuable": is_continuable,
            "confidence": continuation_score,
            "reason": reason,
            "summary": summary,
            "suggested_context": suggested_context,
            "scores": scores,  # For debugging
            "session_id": conversation.get("id", "unknown"),
            "timestamp": conversation.get("timestamp", ""),
            "device_tag": session_device,
            "same_device": same_device,
            "cross_device_warning": cross_device_warning
        }

    def _score_message_count(self, messages: List[Dict]) -> float:
        """Score based on conversation length. Longer = more likely continuable."""
        count = len(messages)
        if count <= 2:
            return 0.1  # Very short, likely one-off
        elif count <= 4:
            return 0.3
        elif count <= 8:
            return 0.6
        elif count <= 15:
            return 0.8
        else:
            return 1.0  # Long conversation, likely substantial work

    def _score_content_depth(self, messages: List[Dict]) -> float:
        """Score based on total content length."""
        total_length = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            total_length += len(content)

        if total_length < 500:
            return 0.1
        elif total_length < 2000:
            return 0.4
        elif total_length < 5000:
            return 0.7
        else:
            return 1.0

    def _score_work_indicators(self, messages: List[Dict]) -> float:
        """Score based on presence of work-related patterns."""
        all_content = self._get_all_content(messages).lower()

        matches = 0
        for pattern in self.continuable_patterns:
            if re.search(pattern, all_content, re.IGNORECASE):
                matches += 1

        # Normalize: max expected is ~5 matches
        return min(1.0, matches / 5)

    def _score_file_work(self, messages: List[Dict], extracted: Dict) -> float:
        """
        Score based on file modifications, with EXTRA weight for file creation.

        File creation is the strongest signal of substantial work - when a session
        creates new files (like ALL_KNOWING_BRAIN_PRD.md), that should be the
        PRIMARY context for "where did we leave off?"
        """
        score = 0.0
        all_content = self._get_all_content(messages)

        # HIGHEST PRIORITY: Explicit file creation (strongest signal)
        file_creation_patterns = [
            r'Created\s+[`\"\']?(/[^\s`\"\']+\.(?:py|md|json|yaml|ts|js|go|rs))[`\"\']?',
            r'wrote\s+(?:to\s+)?[`\"\']?([^\s`\"\']+\.(?:py|md|json|yaml|ts|js|go|rs))[`\"\']?',
            r'saved\s+(?:to\s+)?[`\"\']?([^\s`\"\']+\.(?:py|md|json|yaml|ts|js|go|rs))[`\"\']?',
            r'new file[s]?:\s*([^\n]+)',
        ]

        files_created = []
        for pattern in file_creation_patterns:
            matches = re.findall(pattern, all_content, re.IGNORECASE)
            files_created.extend(matches)

        if files_created:
            # File creation is a VERY strong signal - give it maximum score
            score = 1.0
            # Store for later reference
            extracted["files_created"] = files_created[:5]
            return score

        # Check extracted file paths (second priority)
        file_paths = extracted.get("file_paths", [])
        if len(file_paths) >= 3:
            return 0.9
        elif len(file_paths) >= 1:
            score = max(score, 0.6)

        # Check for general file work in content (lowest priority)
        matches = 0
        for pattern in self.file_work_patterns:
            if re.search(pattern, all_content, re.IGNORECASE):
                matches += 1

        file_work_score = min(1.0, matches / 3) * 0.5  # Lower weight for general file work
        score = max(score, file_work_score)

        return score

    def _score_oneoff_indicators(self, messages: List[Dict]) -> float:
        """Score based on one-off task indicators. Higher = more likely one-off."""
        # Focus on user messages
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                user_content += " " + content

        user_content = user_content.lower().strip()

        # Very short user input is likely one-off
        if len(user_content) < 50:
            return 0.8

        matches = 0
        for pattern in self.oneoff_patterns:
            if re.search(pattern, user_content, re.IGNORECASE):
                matches += 1

        return min(1.0, matches / 3)

    def _score_pending_work(self, messages: List[Dict], extracted: Dict) -> float:
        """Score based on indicators of unfinished work."""
        all_content = self._get_all_content(messages).lower()

        # Check for explicit pending indicators
        pending_patterns = [
            r"(?:will|going to|need to|should) (?:continue|finish|complete)",
            r"(?:next step|next we|then we)",
            r"(?:todo|to-do|to do):",
            r"(?:pending|remaining|left to)",
            r"(?:part \d|step \d|phase \d)",
            r"(?:work in progress|wip|incomplete)",
        ]

        matches = 0
        for pattern in pending_patterns:
            if re.search(pattern, all_content):
                matches += 1

        # Check for problems without clear solutions
        problems = extracted.get("problems_solved", [])
        if problems:
            # If problems exist, likely ongoing work
            matches += 1

        return min(1.0, matches / 3)

    def _score_session_duration(self, conversation: Dict) -> float:
        """Score based on session duration (if available)."""
        # Try to estimate from message timestamps or metadata
        metadata = conversation.get("metadata", {})

        # Check if marked as long session
        if metadata.get("session_type") in ["feature_implementation", "debugging", "project_work"]:
            return 1.0

        return 0.5  # Default middle score

    def _score_importance(self, metadata: Dict) -> float:
        """Score based on conversation importance."""
        importance = metadata.get("importance", "medium")
        importance_map = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }
        return importance_map.get(importance, 0.5)

    def _get_all_content(self, messages: List[Dict]) -> str:
        """Extract all text content from messages."""
        content_parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            content_parts.append(content)
        return " ".join(content_parts)

    def _generate_reason(self, scores: Dict[str, float], is_continuable: bool) -> str:
        """Generate human-readable reason for the decision."""
        if is_continuable:
            reasons = []
            if scores["work_indicators"] > 0.5:
                reasons.append("development work detected")
            if scores["file_work"] > 0.5:
                reasons.append("files were modified")
            if scores["has_pending_work"] > 0.5:
                reasons.append("has pending tasks")
            if scores["message_count"] > 0.5:
                reasons.append("substantial conversation")

            if reasons:
                return "Continuable: " + ", ".join(reasons)
            return "Continuable: appears to be ongoing work"
        else:
            reasons = []
            if scores["oneoff_indicators"] > 0.5:
                reasons.append("looks like a quick task")
            if scores["message_count"] < 0.3:
                reasons.append("very short conversation")
            if scores["content_depth"] < 0.3:
                reasons.append("minimal content")

            if reasons:
                return "One-off: " + ", ".join(reasons)
            return "One-off: doesn't appear to need follow-up"

    def _generate_summary(self, conversation: Dict) -> str:
        """Generate a brief summary of what was being worked on."""
        metadata = conversation.get("metadata", {})
        conversation.get("extracted_data", {})
        messages = conversation.get("messages", [])

        # Try to get summary from metadata
        if metadata.get("summary"):
            return metadata["summary"][:200]

        # Try topics
        topics = metadata.get("topics", [])
        if topics:
            ", ".join(topics[:3])

        # Try to extract from first user message
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                # First substantial user message
                if len(content) > 20:
                    # Truncate intelligently
                    summary = content[:150]
                    if len(content) > 150:
                        summary = summary.rsplit(" ", 1)[0] + "..."
                    return summary

        # Fallback to tags
        tags = metadata.get("tags", [])
        if tags:
            return f"Session involving: {', '.join(tags[:5])}"

        return "Previous session"

    def _suggest_context(self, conversation: Dict, extracted: Dict) -> List[Dict[str, str]]:
        """Suggest files and context to inject if continuing."""
        context = []

        # Add file paths that were worked on
        file_paths = extracted.get("file_paths", [])
        for fp in file_paths[:5]:
            path = fp.get("path", "") if isinstance(fp, dict) else str(fp)
            if path:
                context.append({
                    "type": "file",
                    "path": path,
                    "reason": "Modified in this session"
                })

        # Add any problems that were being solved
        problems = extracted.get("problems_solved", [])
        for prob in problems[:3]:
            problem_text = prob.get("problem", "") if isinstance(prob, dict) else str(prob)
            if problem_text:
                context.append({
                    "type": "problem",
                    "content": problem_text[:200],
                    "reason": "Problem being solved"
                })

        # Add conversation ID for full context retrieval
        context.append({
            "type": "conversation",
            "id": conversation.get("id", ""),
            "reason": "Full conversation context"
        })

        return context

    def _apply_recency_boost(self, candidate: Dict[str, Any]) -> float:
        """
        Boost confidence for recent sessions - AGGRESSIVELY.

        Recent sessions should almost always win over older sessions unless
        they are clearly trivial. The old 1.3x boost was too weak - a session
        with 0.6 confidence * 1.3 = 0.78 still loses to 0.8 confidence old session.

        New aggressive boosts:
        - < 30 min: 3.0x (very recent work almost always wins)
        - < 1 hour: 2.5x
        - < 2 hours: 2.0x
        - < 4 hours: 1.6x
        - < 8 hours: 1.3x
        - < 24 hours: 1.1x
        - >= 24 hours: 1.0x

        Example: 0.5 confidence * 2.5x = 1.25, beats 0.8 confidence * 1.0 = 0.8
        """
        timestamp = candidate.get("timestamp", "")
        base_confidence = candidate.get("confidence", 0)

        if not timestamp:
            return base_confidence

        try:
            # Handle both string timestamps and datetime objects
            if isinstance(timestamp, str):
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                ts = timestamp

            if ts.tzinfo:
                ts = ts.replace(tzinfo=None)

            hours_ago = (datetime.now() - ts).total_seconds() / 3600

            # Aggressive recency boost - recent work should almost always win
            if hours_ago < 0.5:  # < 30 minutes
                recency_boost = 3.0
            elif hours_ago < 1:  # < 1 hour
                recency_boost = 2.5
            elif hours_ago < 2:  # < 2 hours
                recency_boost = 2.0
            elif hours_ago < 4:  # < 4 hours
                recency_boost = 1.6
            elif hours_ago < 8:  # < 8 hours
                recency_boost = 1.3
            elif hours_ago < 24:  # < 24 hours
                recency_boost = 1.1
            else:
                recency_boost = 1.0

            return base_confidence * recency_boost
        except Exception:
            return base_confidence

    def get_best_continuation_candidate(self, hours: int = 48,
                                        device_tag: str = None,
                                        prefer_same_device: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the best candidate for session continuation.

        Args:
            hours: How many hours back to look
            device_tag: Current device tag for cross-device comparison
            prefer_same_device: If True, prefer sessions from same device

        Returns the most recent continuable session, or None if no good candidates.
        Includes cross_device_warning if the best candidate is from a different device.
        """
        sessions = self.get_recent_sessions(hours=hours)

        # Get current device if not provided
        if device_tag is None:
            try:
                device_tag = get_current_device_tag()
            except:
                device_tag = "unknown"

        # Analyze all sessions with device awareness
        candidates = []
        for session in sessions:
            analysis = self.analyze_session(session["conversation"], current_device=device_tag)
            if analysis["is_continuable"]:
                candidates.append({
                    **analysis,
                    "file_path": session["file_path"]
                })

        if not candidates:
            return None

        # Sort candidates: same device first (if prefer_same_device),
        # then by recency-boosted confidence, then by raw timestamp as tiebreaker
        def sort_key(x):
            boosted_conf = self._apply_recency_boost(x)
            # Use timestamp as tiebreaker (ISO format sorts correctly)
            timestamp = x.get("timestamp", "")
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            return (x.get("same_device", False) if prefer_same_device else True,
                    boosted_conf,
                    timestamp)

        candidates.sort(key=sort_key, reverse=True)

        return candidates[0]


def get_startup_prompt() -> Dict[str, Any]:
    """
    Main function to get the startup prompt for Claude Code.

    Returns:
        Dict with:
        - has_continuation: bool
        - summary: str
        - confidence: float
        - suggested_context: list
        - session_id: str
        - cross_device_warning: bool (if continuing from different device)
        - same_device: bool
    """
    analyzer = SessionAnalyzer()

    # Get current device for cross-device comparison
    try:
        current_device = get_current_device_tag()
    except:
        current_device = "unknown"

    candidate = analyzer.get_best_continuation_candidate(hours=48, device_tag=current_device)

    if candidate and candidate["confidence"] >= 0.4:
        result = {
            "has_continuation": True,
            "summary": candidate["summary"],
            "confidence": candidate["confidence"],
            "reason": candidate["reason"],
            "suggested_context": candidate["suggested_context"],
            "session_id": candidate["session_id"],
            "timestamp": candidate["timestamp"],
            "device_tag": candidate.get("device_tag", "unknown"),
            "same_device": candidate.get("same_device", True),
            "cross_device_warning": candidate.get("cross_device_warning", False),
            "current_device": current_device
        }

        # Add warning message if cross-device
        if candidate.get("cross_device_warning"):
            session_device = candidate.get("device_tag", "unknown")
            result["warning_message"] = (
                f"Note: This session was from '{session_device}' device. "
                f"You're currently on '{current_device}'. "
                "Some paths or configurations may differ."
            )

        return result

    return {
        "has_continuation": False,
        "summary": "",
        "confidence": 0.0,
        "reason": "No recent continuable work found",
        "suggested_context": [],
        "session_id": "",
        "timestamp": "",
        "device_tag": "",
        "same_device": True,
        "cross_device_warning": False,
        "current_device": current_device
    }


if __name__ == "__main__":
    # Test the analyzer

    analyzer = SessionAnalyzer()
    print("=== Session Analyzer Test ===\n")

    # Get recent sessions
    sessions = analyzer.get_recent_sessions(hours=48, limit=5)
    print(f"Found {len(sessions)} recent sessions\n")

    for session in sessions:
        analysis = analyzer.analyze_session(session["conversation"])
        print(f"Session: {analysis['session_id']}")
        print(f"  Continuable: {analysis['is_continuable']}")
        print(f"  Confidence: {analysis['confidence']:.2f}")
        print(f"  Reason: {analysis['reason']}")
        print(f"  Summary: {analysis['summary'][:80]}...")
        print()

    # Get best continuation candidate
    print("=== Best Continuation Candidate ===")
    candidate = analyzer.get_best_continuation_candidate()
    if candidate:
        print(f"Session: {candidate['session_id']}")
        print(f"Summary: {candidate['summary']}")
        print(f"Confidence: {candidate['confidence']:.2f}")
    else:
        print("No continuable sessions found")
