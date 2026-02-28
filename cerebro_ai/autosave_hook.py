#!/usr/bin/env python3
"""
Autosave Hook - Lightweight entry point for automatic conversation saving.
Fires on Stop and SessionEnd hooks, spawns background worker, exits immediately.
SILENT - no output to avoid cluttering Claude's context.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Log file for debugging
LOG_FILE = Path("/tmp/autosave_hook.log")

# Retry settings for transcript file availability
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 0.3  # seconds, will use exponential backoff

def log(msg: str):
    """Write debug info to log file."""
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except:
        pass


def main():
    """Read hook input, spawn background worker, exit immediately."""
    log("Hook triggered")

    # Read stdin to get hook input
    try:
        stdin_data = sys.stdin.read()
        hook_input = json.loads(stdin_data) if stdin_data else {}
        log(f"Input: {json.dumps(hook_input)[:500]}")
    except Exception as e:
        log(f"Error reading stdin: {e}")
        sys.exit(0)  # Exit silently on error

    # Extract key info
    session_id = hook_input.get("session_id", "")
    transcript_path = hook_input.get("transcript_path", "")
    hook_event = hook_input.get("hook_event_name", "Stop")

    if not session_id or not transcript_path:
        log("Missing session_id or transcript_path, skipping")
        sys.exit(0)

    # Retry logic for transcript file - may not exist yet due to race condition
    transcript_file = Path(transcript_path)
    file_found = False

    for attempt in range(RETRY_ATTEMPTS):
        if transcript_file.exists():
            # Also check file has content (not empty/still being written)
            try:
                size = transcript_file.stat().st_size
                if size > 0:
                    file_found = True
                    if attempt > 0:
                        log(f"Transcript found on attempt {attempt + 1} (size: {size} bytes)")
                    break
            except OSError:
                pass  # File might be locked, retry

        if attempt < RETRY_ATTEMPTS - 1:
            delay = RETRY_BASE_DELAY * (2 ** attempt)  # Exponential backoff: 0.3, 0.6, 1.2, 2.4s
            time.sleep(delay)

    if not file_found:
        log(f"Transcript file not found after {RETRY_ATTEMPTS} attempts: {transcript_path}")
        # Log additional context for debugging
        log(f"  Hook event: {hook_event}, Session: {session_id}")
        hook_reason = hook_input.get("reason", "unknown")
        log(f"  Exit reason: {hook_reason}")
        sys.exit(0)

    # Spawn background worker
    worker_script = Path(__file__).parent / "autosave_worker.py"
    if not worker_script.exists():
        log(f"Worker script not found: {worker_script}")
        sys.exit(0)

    try:
        # Use the same Python interpreter that's running this script
        python_exe = sys.executable

        # Spawn as detached process (won't block)
        env = os.environ.copy()
        env["AUTOSAVE_SESSION_ID"] = session_id
        env["AUTOSAVE_TRANSCRIPT_PATH"] = transcript_path
        env["AUTOSAVE_HOOK_EVENT"] = hook_event
        env["ENABLE_EMBEDDINGS"] = "1"  # Enhancement 1: Enable embedding generation

        # On Unix, use nohup-style process detachment
        subprocess.Popen(
            [python_exe, str(worker_script)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True  # Detach from parent process group
        )

        log(f"Spawned worker for session {session_id}, event {hook_event}")
    except Exception as e:
        log(f"Error spawning worker: {e}")

    # Exit immediately - no output (silent)
    sys.exit(0)


if __name__ == "__main__":
    main()
