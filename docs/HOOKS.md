# Cerebro Hooks

> Lifecycle hooks that integrate Cerebro with Claude Code

## Overview

Cerebro uses Claude Code's hook system to maintain persistent context across sessions. Four hooks fire at key lifecycle events, enabling seamless memory without manual intervention.

## Install

```bash
cerebro hooks install
```

This copies hook scripts to `~/.claude/hooks/` and registers them in `~/.claude/settings.json`.

```bash
cerebro hooks status     # check what's installed
cerebro hooks uninstall  # remove all Cerebro hooks
```

## How Hooks Work

```
┌─────────────────────────────────────────────────────┐
│                  Claude Code Session                 │
│                                                     │
│  ┌──────────┐                                       │
│  │ Session   │──► Load active work from quick_facts  │
│  │ Start     │──► Check for resumable sessions       │
│  └──────────┘                                       │
│       │                                             │
│       ▼                                             │
│  ┌──────────┐                                       │
│  │ User      │──► Extract keywords from message      │
│  │ Prompt    │──► Match against corrections           │
│  │ Submit    │──► Match against past learnings        │
│  └──────────┘                                       │
│       │  (repeated for each message)                │
│       ▼                                             │
│  ┌──────────┐                                       │
│  │ Pre       │──► Save session checkpoint             │
│  │ Compact   │                                       │
│  └──────────┘                                       │
│       │                                             │
│       ▼                                             │
│  ┌──────────┐                                       │
│  │ Session   │──► Read transcript                     │
│  │ End       │──► Save conversation with metadata     │
│  └──────────┘                                       │
└─────────────────────────────────────────────────────┘
```

## Hook Reference

### SessionStart

**When**: New Claude Code session begins
**Purpose**: Load active work context and check for resumable sessions

- Reads `quick_facts.json` for active work summary
- Scans recent pre-compact checkpoints to detect continuable sessions
- Outputs context to stderr (visible in hook debug logs)
- Always returns `{"continue": true}` to stdout

### UserPromptSubmit

**When**: User sends a message
**Purpose**: Inject relevant corrections and learnings into context

- Extracts keywords from the user message (lightweight, no embeddings)
- Searches `corrections.jsonl` for matching past mistakes
- Searches recent learnings for relevant solutions
- Top 5 matches for each are printed to stderr
- Always returns `{"continue": true}` to stdout

### PreCompact

**When**: Before Claude Code compresses conversation history
**Purpose**: Save a checkpoint so session can be resumed

- Saves session ID, timestamp, and transcript path to a checkpoint file
- Checkpoint is used by SessionStart to detect resumable sessions
- Always returns `{"continue": true}` to stdout

### SessionEnd

**When**: Claude Code session ends
**Purpose**: Save the conversation for future retrieval

- Reads the JSONL transcript file
- Extracts user and assistant messages
- Computes metadata: message count, keywords, timestamp
- Saves to `{data_dir}/conversations/{YYYY-MM-DD}/{session_id}.json`
- Always returns `{"continue": true}` to stdout

## Design Principles

- **Self-contained**: Zero imports from the Cerebro package — only Python stdlib
- **Cross-platform**: No OS-specific code (no `fcntl`, no `signal.alarm`)
- **Non-blocking**: Errors are caught and logged to stderr; stdout always emits valid JSON
- **Lightweight**: No embedding models, no network calls — keyword matching only
- **Configurable**: Uses `CEREBRO_DATA_DIR` env var (default `~/.cerebro/data`)

## Troubleshooting

### Hook Not Firing

1. Run `cerebro hooks status` to verify installation
2. Check that Python is in your PATH
3. Test directly: `echo '{}' | python ~/.claude/hooks/cerebro_session_start.py`

### Hook Output Errors

- **Symptom**: Claude Code reports "invalid hook output"
- **Cause**: Hook script printed non-JSON text to stdout
- **Fix**: All Cerebro hooks send debug info to stderr only. If you modified a hook, ensure `print()` calls use `file=sys.stderr`.

### Slow Hook Execution

The shipped hooks are lightweight (keyword matching only). If you notice delays:
- Check that `CEREBRO_DATA_DIR` points to local storage (not a network mount)
- Reduce the number of learnings files in `{data_dir}/learnings/`
