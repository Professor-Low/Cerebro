#!/usr/bin/env python3
"""
Startup Check - Claude Code Hook for Session Continuation
Runs on user prompt to check if there's work to continue.
Only outputs on the FIRST prompt of each Claude Code session.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Session tracking file
SESSION_TRACKER = Path("/tmp/claude_session_tracker.json")
DEBUG_LOG = Path("/tmp/claude_hook_debug.log")


def log_debug(msg):
    """Write debug info to log file."""
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except:
        pass


def is_first_prompt(claude_session_id: str) -> bool:
    """Check if this is the first prompt in the current Claude session."""
    tracker = {}
    if SESSION_TRACKER.exists():
        try:
            with open(SESSION_TRACKER, "r") as f:
                tracker = json.load(f)
        except:
            tracker = {}

    # Clean old sessions (older than 4 hours)
    cutoff = (datetime.now() - timedelta(hours=4)).isoformat()
    tracker = {k: v for k, v in tracker.items() if v.get("timestamp", "") > cutoff}

    if claude_session_id in tracker:
        log_debug(f"Session {claude_session_id} already seen, skipping")
        return False

    tracker[claude_session_id] = {"timestamp": datetime.now().isoformat()}

    try:
        with open(SESSION_TRACKER, "w") as f:
            json.dump(tracker, f)
    except:
        pass

    log_debug(f"First prompt for session {claude_session_id}")
    return True


def check_continuation():
    """Check for continuable work and return context if found."""
    try:
        from session_analyzer import SessionAnalyzer

        analyzer = SessionAnalyzer()
        candidate = analyzer.get_best_continuation_candidate(hours=48)

        if candidate and candidate["confidence"] >= 0.4:
            summary = candidate["summary"][:120]
            if len(candidate["summary"]) > 120:
                summary = summary.rsplit(" ", 1)[0] + "..."

            confidence_pct = int(candidate["confidence"] * 100)
            session_id = candidate["session_id"]

            log_debug(f"Found continuation: {session_id} ({confidence_pct}%)")

            return f"""[SESSION CONTINUATION AVAILABLE]
I found recent work you may want to continue:
- Session: {session_id}
- Summary: {summary}
- Confidence: {confidence_pct}%

Please ask the user: "I found you were working on '{summary[:60]}...' ({confidence_pct}% match). Would you like to pick up where you left off?"

If user says YES: Call get_continuation_context MCP tool with session_id="{session_id}"
If user says NO: Proceed with their current request normally."""

        else:
            log_debug("No continuable session found")

    except Exception as e:
        log_debug(f"Error: {e}")

    return None


def main():
    log_debug("Hook started")

    # Read stdin to get the hook input
    stdin_data = ""
    try:
        stdin_data = sys.stdin.read()
        log_debug(f"Received stdin: {stdin_data[:200]}...")
    except Exception as e:
        log_debug(f"Error reading stdin: {e}")

    # Parse the hook input to get Claude's session ID
    claude_session_id = "unknown"
    try:
        hook_input = json.loads(stdin_data) if stdin_data else {}
        claude_session_id = hook_input.get("session_id", str(os.getpid()))
        log_debug(f"Session ID: {claude_session_id}")
    except Exception as e:
        log_debug(f"Error parsing input: {e}")
        claude_session_id = str(os.getpid())

    # Only check on first prompt of this Claude session
    if is_first_prompt(claude_session_id):
        result = check_continuation()
        if result:
            log_debug("Outputting continuation prompt")
            print(result)
        else:
            log_debug("No output (no continuation)")
    else:
        log_debug("Not first prompt, no output")

    sys.exit(0)


if __name__ == "__main__":
    main()
