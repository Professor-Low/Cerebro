#!/usr/bin/env python3
# Copyright (C) 2026 Michael Lopez (Professor-Low)
# Licensed under AGPL-3.0 — see LICENSE for details.
"""
Pre-compact hook for Cerebro AI Memory.

Runs just before Claude Code compacts (summarises) the conversation
transcript. Saves a checkpoint so session_start can later detect
resumable sessions.

Input (stdin JSON): {"session_id": "...", "transcript_path": "..."}
Output:             {"continue": true}
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_data_dir() -> Path:
    return Path(os.environ.get("CEREBRO_DATA_DIR", Path.home() / ".cerebro" / "data"))


def main() -> None:
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}

        session_id = hook_input.get("session_id", "unknown")
        transcript_path = hook_input.get("transcript_path", "")

        data_dir = get_data_dir()
        checkpoint_dir = data_dir / "cache" / "pre_compact"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transcript_path": transcript_path,
        }

        out_path = checkpoint_dir / f"{session_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        print(f"[pre_compact] Saved checkpoint for session {session_id}", file=sys.stderr)

    except Exception as exc:
        print(f"[pre_compact] Error (non-fatal): {exc}", file=sys.stderr)

    print(json.dumps({"continue": True}))


if __name__ == "__main__":
    main()
