#!/usr/bin/env python3
# Copyright (C) 2026 Michael Lopez (Professor-Low)
# Licensed under AGPL-3.0 — see LICENSE for details.
"""
Session end hook for Cerebro AI Memory.

Runs when a Claude Code session ends. Reads the JSONL transcript,
extracts user and assistant messages, computes lightweight metadata
(message count, keywords, timestamps), and saves the conversation
to the data directory for later indexing.

Input (stdin JSON): {"session_id": "...", "transcript_path": "...", "cwd": "..."}
Output:             {"continue": true}
"""

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Minimal stopwords — same compact set used by user_prompt_submit.
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will "
    "would shall should may might can could of in to for on with at by from "
    "as into through during before after above below between out off over "
    "under again further then once here there when where why how all each "
    "every both few more most other some such no nor not only own same so "
    "than too very s t just don didn doesn won wasn weren isn aren hasn "
    "haven hadn wouldn couldn shouldn mustn needn shan it its i me my we "
    "our you your he him his she her they them their what which who whom "
    "this that these those am about up if or and but because until while".split()
)


def get_data_dir() -> Path:
    return Path(os.environ.get("CEREBRO_DATA_DIR", Path.home() / ".cerebro" / "data"))


def extract_keywords(texts: list, max_keywords: int = 30) -> list:
    """Extract the most frequent meaningful keywords from a list of texts."""
    counter: Counter = Counter()
    for text in texts:
        words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        for w in words:
            if len(w) >= 3 and w not in _STOPWORDS:
                counter[w] += 1
    return [word for word, _ in counter.most_common(max_keywords)]


def read_transcript(transcript_path: str) -> tuple:
    """
    Read a JSONL transcript file and return (user_messages, assistant_messages).

    Each line is expected to be a JSON object. Messages with type "human" or
    "user" are user messages; type "assistant" are assistant messages. The
    actual text is in the "message" field (or nested under "content").
    """
    user_msgs: list = []
    assistant_msgs: list = []

    path = Path(transcript_path)
    if not path.exists():
        return user_msgs, assistant_msgs

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = obj.get("type", "")
                # Extract text content.
                text = ""
                message = obj.get("message", "")
                if isinstance(message, str):
                    text = message
                elif isinstance(message, dict):
                    # Some formats nest under message.content
                    content = message.get("content", "")
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        # content can be a list of blocks
                        parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            elif isinstance(block, str):
                                parts.append(block)
                        text = " ".join(parts)

                if not text:
                    continue

                if msg_type in ("human", "user"):
                    user_msgs.append(text)
                elif msg_type == "assistant":
                    assistant_msgs.append(text)
    except OSError:
        pass

    return user_msgs, assistant_msgs


def save_conversation(
    data_dir: Path,
    session_id: str,
    user_msgs: list,
    assistant_msgs: list,
    keywords: list,
    cwd: str,
) -> None:
    """Save the conversation JSON to {data_dir}/conversations/{YYYY-MM-DD}/."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conv_dir = data_dir / "conversations" / today
    conv_dir.mkdir(parents=True, exist_ok=True)

    messages = []
    # Interleave in order — user messages and assistant messages alternate,
    # but we only have them in two flat lists so store them labelled.
    for text in user_msgs:
        messages.append({"role": "user", "content": text})
    for text in assistant_msgs:
        messages.append({"role": "assistant", "content": text})

    conversation = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cwd": cwd,
        "message_count": len(user_msgs) + len(assistant_msgs),
        "user_message_count": len(user_msgs),
        "assistant_message_count": len(assistant_msgs),
        "keywords": keywords,
        "messages": messages,
    }

    out_path = conv_dir / f"{session_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)

    print(f"[session_end] Saved conversation ({len(messages)} msgs) to {out_path}", file=sys.stderr)


def main() -> None:
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}

        session_id = hook_input.get("session_id", "unknown")
        transcript_path = hook_input.get("transcript_path", "")
        cwd = hook_input.get("cwd", "")

        if not transcript_path:
            print("[session_end] No transcript_path provided, skipping.", file=sys.stderr)
        else:
            data_dir = get_data_dir()
            user_msgs, assistant_msgs = read_transcript(transcript_path)

            if not user_msgs and not assistant_msgs:
                print("[session_end] Transcript empty or unreadable, skipping save.", file=sys.stderr)
            else:
                keywords = extract_keywords(user_msgs)
                save_conversation(data_dir, session_id, user_msgs, assistant_msgs, keywords, cwd)

    except Exception as exc:
        print(f"[session_end] Error (non-fatal): {exc}", file=sys.stderr)

    print(json.dumps({"continue": True}))


if __name__ == "__main__":
    main()
