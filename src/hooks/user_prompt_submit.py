#!/usr/bin/env python3
# Copyright (C) 2026 Michael Lopez (Professor-Low)
# Licensed under AGPL-3.0 — see LICENSE for details.
"""
User prompt submit hook for Cerebro AI Memory.

Runs each time the user submits a prompt. Extracts keywords from the
message, then searches corrections and recent learnings for anything
relevant. Matching context is printed to stderr so the agent sees it;
stdout always emits {"continue": true}.

Input (stdin JSON): {"session_id": "...", "message": "..."}
Output:             {"continue": true}
"""

import json
import os
import re
import sys
from pathlib import Path

# Common English stopwords — kept small to avoid heavy imports.
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


def extract_keywords(text: str, max_keywords: int = 20) -> list:
    """Extract meaningful lowercase keywords from text."""
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    seen = set()
    keywords = []
    for w in words:
        if len(w) < 3 or w in _STOPWORDS or w in seen:
            continue
        seen.add(w)
        keywords.append(w)
        if len(keywords) >= max_keywords:
            break
    return keywords


def load_corrections(data_dir: Path, keywords: list, max_results: int = 5) -> list:
    """Load corrections that match any of the given keywords."""
    corr_path = data_dir / "corrections" / "corrections.jsonl"
    if not corr_path.exists():
        return []
    kw_set = set(keywords)
    matches = []
    try:
        with open(corr_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("content", "") or obj.get("correction", "")
                text_lower = text.lower()
                score = sum(1 for kw in kw_set if kw in text_lower)
                if score > 0:
                    matches.append((score, text))
        matches.sort(key=lambda x: x[0], reverse=True)
    except OSError:
        pass
    return [m[1] for m in matches[:max_results]]


def load_learnings(data_dir: Path, keywords: list, max_results: int = 5) -> list:
    """Load recent learnings that match any of the given keywords."""
    learn_dir = data_dir / "learnings"
    if not learn_dir.exists():
        return []
    kw_set = set(keywords)
    matches = []
    try:
        files = sorted(learn_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for fpath in files[:100]:  # scan at most 100 recent files
            if fpath.suffix != ".json":
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            # Learnings may have problem, solution, description, or tags fields.
            searchable = " ".join(
                str(obj.get(k, ""))
                for k in ("problem", "solution", "description", "tags", "context")
            ).lower()
            score = sum(1 for kw in kw_set if kw in searchable)
            if score > 0:
                summary = obj.get("problem", "") or obj.get("description", "")
                if obj.get("solution"):
                    summary += f" -> {obj['solution']}"
                matches.append((score, summary.strip()))
    except OSError:
        pass
    matches.sort(key=lambda x: x[0], reverse=True)
    return [m[1] for m in matches[:max_results]]


def main() -> None:
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
        message = hook_input.get("message", "")

        if message:
            keywords = extract_keywords(message)
            if keywords:
                data_dir = get_data_dir()
                corrections = load_corrections(data_dir, keywords)
                learnings = load_learnings(data_dir, keywords)

                if corrections:
                    print("[user_prompt_submit] Relevant corrections:", file=sys.stderr)
                    for c in corrections:
                        print(f"  - {c[:200]}", file=sys.stderr)

                if learnings:
                    print("[user_prompt_submit] Relevant learnings:", file=sys.stderr)
                    for l in learnings:
                        print(f"  - {l[:200]}", file=sys.stderr)

                if not corrections and not learnings:
                    print("[user_prompt_submit] No matching corrections or learnings.", file=sys.stderr)
            else:
                print("[user_prompt_submit] No keywords extracted from message.", file=sys.stderr)
        else:
            print("[user_prompt_submit] Empty message, skipping lookup.", file=sys.stderr)

    except Exception as exc:
        print(f"[user_prompt_submit] Error (non-fatal): {exc}", file=sys.stderr)

    print(json.dumps({"continue": True}))


if __name__ == "__main__":
    main()
