"""
Context Injector - Prepares relevant context for session continuation.
Reads files and conversation history to inject into Claude's context.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


class ContextInjector:
    """
    Prepares and formats context for session continuation.
    Reads relevant files and conversation snippets to inject.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.conversations_path = self.base_path / "conversations"
        self.max_file_lines = 100  # Max lines to include from each file
        self.max_context_chars = 8000  # Max total context characters

    def prepare_continuation_context(self, session_id: str,
                                     suggested_context: List[Dict] = None) -> Dict[str, Any]:
        """
        Prepare context for continuing a session.

        Args:
            session_id: The conversation ID to continue
            suggested_context: List of suggested files/context to inject

        Returns:
            Dict with formatted context ready for injection
        """
        context = {
            "session_id": session_id,
            "conversation_summary": "",
            "key_points": [],
            "files_context": [],
            "pending_work": [],
            "formatted_prompt": ""
        }

        # Load the conversation
        conv_file = self.conversations_path / f"{session_id}.json"
        if not conv_file.exists():
            return context

        try:
            with open(conv_file, "r", encoding="utf-8") as f:
                conversation = json.load(f)
        except Exception:
            return context

        # Extract key information
        context["conversation_summary"] = self._summarize_conversation(conversation)
        context["key_points"] = self._extract_key_points(conversation)
        context["pending_work"] = self._extract_pending_work(conversation)

        # Load relevant files
        if suggested_context:
            context["files_context"] = self._load_file_context(suggested_context)

        # Format the complete prompt
        context["formatted_prompt"] = self._format_continuation_prompt(context)

        return context

    def _summarize_conversation(self, conversation: Dict) -> str:
        """Create a brief summary of the conversation."""
        messages = conversation.get("messages", [])
        metadata = conversation.get("metadata", {})
        conv_id = conversation.get("id", "")

        # Try metadata summary first
        if metadata.get("summary"):
            return metadata["summary"]

        # Check identity cache for distilled summary
        # Only use if this conversation is recent (distiller's last_session is for most recent)
        try:
            identity_file = self.base_path / "cache" / "identity_core.json"
            if identity_file.exists():
                import json
                with open(identity_file, "r", encoding="utf-8") as f:
                    identity = json.load(f)
                last_session = identity.get("last_session", "")
                last_session_id = identity.get("last_session_id", "")
                # Only use identity cache summary if it's for THIS conversation
                # or if no session ID recorded (backwards compatibility)
                if last_session and len(last_session) > 20:
                    if not last_session_id or last_session_id == conv_id:
                        return last_session
        except Exception:
            pass

        # Extract from messages
        summary_parts = []

        # Get the main topic from first REAL user message (not tool results)
        for msg in messages:
            if msg.get("role") == "user":
                content = self._get_content(msg)
                # Skip tool results and system messages
                if content.startswith("[Tool result:") or content.startswith("[System"):
                    continue
                # Lower threshold to 10 chars (catch short questions like "hey")
                if len(content) > 10:
                    summary_parts.append(f"Started with: {content[:200]}")
                    break

        # Get topics
        topics = metadata.get("topics", [])
        if topics:
            summary_parts.append(f"Topics: {', '.join(topics[:5])}")

        # Get last assistant action
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = self._get_content(msg)
                if len(content) > 50:
                    # Find first sentence or meaningful chunk
                    first_part = content[:300].split(". ")[0]
                    summary_parts.append(f"Last action: {first_part}")
                    break

        return " | ".join(summary_parts) if summary_parts else "Previous session"

    def _extract_key_points(self, conversation: Dict) -> List[str]:
        """Extract key points from the conversation."""
        key_points = []
        extracted = conversation.get("extracted_data", {})
        conversation.get("metadata", {})

        # Add decisions made
        decisions = extracted.get("decisions_made", [])
        for dec in decisions[:3]:
            decision_text = dec.get("decision", "") if isinstance(dec, dict) else str(dec)
            if decision_text:
                key_points.append(f"Decision: {decision_text[:100]}")

        # Add problems solved
        problems = extracted.get("problems_solved", [])
        for prob in problems[:3]:
            if isinstance(prob, dict):
                problem_text = prob.get("problem", "")
                solution_text = prob.get("solution", "")
                if problem_text and solution_text:
                    key_points.append(f"Solved: {problem_text[:50]} -> {solution_text[:50]}")

        # Add files worked on
        file_paths = extracted.get("file_paths", [])
        if file_paths:
            paths = [fp.get("path", "") if isinstance(fp, dict) else str(fp) for fp in file_paths[:5]]
            paths = [p for p in paths if p]
            if paths:
                key_points.append(f"Files: {', '.join(paths)}")

        return key_points

    def _extract_pending_work(self, conversation: Dict) -> List[str]:
        """Extract any pending/incomplete work."""
        pending = []
        messages = conversation.get("messages", [])

        # Look for TODO markers in assistant messages
        todo_pattern_words = ["todo", "next step", "need to", "should", "will then", "remaining"]

        for msg in reversed(messages[-5:]):  # Check last 5 messages
            if msg.get("role") == "assistant":
                content = self._get_content(msg).lower()
                for pattern in todo_pattern_words:
                    if pattern in content:
                        # Extract the sentence containing the pattern
                        sentences = content.split(".")
                        for sentence in sentences:
                            if pattern in sentence and len(sentence) > 20:
                                pending.append(sentence.strip()[:150])
                                break

        return pending[:5]  # Max 5 pending items

    def _load_file_context(self, suggested_context: List[Dict]) -> List[Dict[str, str]]:
        """Load content from suggested files."""
        files_context = []
        total_chars = 0

        for ctx in suggested_context:
            if ctx.get("type") != "file":
                continue

            file_path = ctx.get("path", "")
            if not file_path:
                continue

            try:
                path = Path(file_path)
                if not path.exists():
                    continue

                # Read file with limits
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()[:self.max_file_lines]
                    content = "".join(lines)

                # Check total context limit
                if total_chars + len(content) > self.max_context_chars:
                    remaining = self.max_context_chars - total_chars
                    if remaining > 500:
                        content = content[:remaining] + "\n... (truncated)"
                    else:
                        break

                total_chars += len(content)

                files_context.append({
                    "path": file_path,
                    "content": content,
                    "lines": len(lines),
                    "reason": ctx.get("reason", "Referenced in session")
                })

            except Exception:
                continue

        return files_context

    def _format_continuation_prompt(self, context: Dict) -> str:
        """Format the complete continuation prompt."""
        parts = []

        parts.append("# Session Continuation Context")
        parts.append(f"\n**Previous Session:** {context['session_id']}")
        parts.append(f"\n**Summary:** {context['conversation_summary']}")

        if context["key_points"]:
            parts.append("\n## Key Points from Last Session:")
            for point in context["key_points"]:
                parts.append(f"- {point}")

        if context["pending_work"]:
            parts.append("\n## Pending Work:")
            for work in context["pending_work"]:
                parts.append(f"- {work}")

        if context["files_context"]:
            parts.append("\n## Relevant Files:")
            for fc in context["files_context"]:
                parts.append(f"\n### {fc['path']}")
                parts.append(f"({fc['lines']} lines, {fc['reason']})")
                parts.append("```")
                parts.append(fc['content'][:2000])  # Limit per file
                parts.append("```")

        return "\n".join(parts)

    def _get_content(self, msg: Dict) -> str:
        """Extract text content from a message."""
        content = msg.get("content", "")
        if isinstance(content, list):
            return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        return content


def get_continuation_context(session_id: str) -> str:
    """
    Quick function to get formatted continuation context.

    Args:
        session_id: The session to continue

    Returns:
        Formatted context string for injection
    """
    injector = ContextInjector()

    # Get suggested context from session analyzer
    from session_analyzer import SessionAnalyzer
    analyzer = SessionAnalyzer()

    # Find the session
    sessions = analyzer.get_recent_sessions(hours=168)  # 1 week
    suggested_context = []

    for session in sessions:
        if session["id"] == session_id:
            analysis = analyzer.analyze_session(session["conversation"])
            suggested_context = analysis.get("suggested_context", [])
            break

    # Prepare context
    context = injector.prepare_continuation_context(session_id, suggested_context)

    return context["formatted_prompt"]


if __name__ == "__main__":
    # Test
    import sys

    if len(sys.argv) > 1:
        session_id = sys.argv[1]
    else:
        # Get most recent continuable session
        from session_analyzer import SessionAnalyzer
        analyzer = SessionAnalyzer()
        candidate = analyzer.get_best_continuation_candidate()
        if candidate:
            session_id = candidate["session_id"]
        else:
            print("No continuable sessions found")
            sys.exit(1)

    print(f"Loading context for: {session_id}")
    print("=" * 50)

    context = get_continuation_context(session_id)
    print(context)
