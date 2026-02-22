#!/usr/bin/env python3
# Copyright (C) 2026 Michael Lopez (Professor-Low)
# Licensed under AGPL-3.0 — see LICENSE for details.
"""
Session start hook for Cerebro AI Memory.

Runs when a new Claude Code session begins. Loads active work context
from quick_facts.json and checks for session continuation markers so
the agent can resume where it left off.

Input:  (none — session_start receives no stdin payload)
Output: {"continue": true}  (stdout, always — never blocks Claude Code)
Context information is written to stderr.
"""

import json
import os
import sys
from pathlib import Path


def get_data_dir() -> Path:
    """Return the Cerebro data directory from env or default."""
    return Path(os.environ.get("CEREBRO_DATA_DIR", Path.home() / ".cerebro" / "data"))


def load_active_work(data_dir: Path) -> dict:
    """Load quick_facts.json and return the active_work section."""
    qf_path = data_dir / "quick_facts.json"
    if not qf_path.exists():
        return {}
    try:
        with open(qf_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("active_work", {})
    except (json.JSONDecodeError, OSError):
        return {}


def check_continuation_markers(data_dir: Path) -> list:
    """Look for recent pre-compact checkpoints that indicate resumable sessions."""
    cache_dir = data_dir / "cache" / "pre_compact"
    if not cache_dir.exists():
        return []
    markers = []
    try:
        for entry in sorted(cache_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if entry.suffix == ".json":
                with open(entry, "r", encoding="utf-8") as f:
                    markers.append(json.load(f))
                if len(markers) >= 3:
                    break
    except OSError:
        pass
    return markers


def main() -> None:
    try:
        data_dir = get_data_dir()

        active_work = load_active_work(data_dir)
        if active_work:
            print("[session_start] Active work context:", file=sys.stderr)
            for key, value in active_work.items():
                print(f"  {key}: {value}", file=sys.stderr)
        else:
            print("[session_start] No active work found.", file=sys.stderr)

        markers = check_continuation_markers(data_dir)
        if markers:
            print(f"[session_start] Found {len(markers)} recent session checkpoint(s).", file=sys.stderr)
            for m in markers:
                sid = m.get("session_id", "unknown")
                ts = m.get("timestamp", "unknown")
                print(f"  session={sid}  at={ts}", file=sys.stderr)

    except Exception as exc:
        print(f"[session_start] Error (non-fatal): {exc}", file=sys.stderr)

    # Always emit valid JSON to stdout — never block Claude Code.
    print(json.dumps({"continue": True}))


if __name__ == "__main__":
    main()
