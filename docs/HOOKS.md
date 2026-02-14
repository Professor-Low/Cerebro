# Cerebro Hooks

> Lifecycle hooks that integrate Cerebro with Claude Code

## Overview

Cerebro uses Claude Code's hook system to maintain persistent context across sessions. Four hooks fire at key lifecycle events, enabling seamless memory and learning without requiring manual intervention. Together, they create a closed loop: sessions resume where they left off, every message benefits from past experience, and every session leaves the system smarter.

## How Hooks Work

Claude Code supports lifecycle hooks: shell commands that execute in response to specific events. Cerebro registers hooks that call into the Memory Server, injecting context and persisting learnings automatically.

```
┌─────────────────────────────────────────────────────┐
│                  Claude Code Session                 │
│                                                     │
│  ┌──────────┐                                       │
│  │ Session   │──► Wake NAS                          │
│  │ Start     │──► Brain maintenance                 │
│  │           │──► Check continuable work             │
│  └──────────┘                                       │
│       │                                             │
│       ▼                                             │
│  ┌──────────┐    ┌──────────┐                       │
│  │ User      │──►│ Memory   │──► Context injection  │
│  │ Prompt    │   │ Search   │──► Correction check    │
│  │ Submit    │   │          │──► Breakthrough detect │
│  └──────────┘    └──────────┘                       │
│       │                                             │
│       ▼  (repeated for each message)                │
│                                                     │
│  ┌──────────┐                                       │
│  │ Pre       │──► Save reasoning chains              │
│  │ Compact   │──► Index RAG chunks                   │
│  │           │──► Extract learnings                   │
│  └──────────┘                                       │
│       │                                             │
│       ▼                                             │
│  ┌──────────┐                                       │
│  │ Session   │──► Save conversation                  │
│  │ End       │──► Extract learnings                  │
│  │           │──► Update active_work                 │
│  └──────────┘                                       │
└─────────────────────────────────────────────────────┘
```

## Hook Reference

### SessionStart

**When**: Every new Claude Code session begins
**Timing**: Before the user's first message is processed
**Purpose**: Initialize memory, check for continuable work, inject baseline context

#### Actions

| Step | Action | Description |
|------|--------|-------------|
| 1 | **Verify storage** | Check NAS/local storage connectivity; run `mount-nas.bat` if needed |
| 2 | **Brain maintenance** | Rebuild FAISS index if stale, promote patterns (3+ occurrences), archive unused patterns (90+ days), update `quick_facts.json` |
| 3 | **Session continuation** | Call `check_session_continuation` to find recent work-in-progress |
| 4 | **Load quick facts** | Read `quick_facts.json` for instant recall of config, active work, and corrections |

#### Output

The hook outputs a JSON system reminder containing:
- Active work context (if resuming a session)
- Top corrections to avoid
- Recent learnings summary
- System health status

#### Error Handling

If NAS is unreachable:
1. Attempt remount via `mount-nas.bat`
2. Fall back to local cache at `~/.cerebro/cache/`
3. Flag degraded mode in session context

---

### UserPromptSubmit

**When**: Each time the user sends a message
**Timing**: After the user presses Enter, before Claude processes the message
**Purpose**: Enrich the conversation with relevant memory and detect learning opportunities

#### Actions

| Step | Action | Description |
|------|--------|-------------|
| 1 | **Entity extraction** | Identify technologies, file paths, people, projects from user message |
| 2 | **Keyword extraction** | Extract search terms for memory retrieval |
| 3 | **Memory search** | Hybrid search (semantic + keyword) for relevant past context |
| 4 | **Correction check** | Call `get_corrections()` to load known mistakes related to the topic |
| 5 | **Breakthrough detection** | Scan for trigger phrases: "It works!", "Finally!", "Perfect!", "That fixed it!" |
| 6 | **Context injection** | Format results as `<system-reminder>` block for Claude |

#### Breakthrough Detection

When a breakthrough is detected, the hook automatically:
1. Extracts the problem-solution pair from conversation context
2. Calls `record_learning()` with type `solution`
3. Tags with relevant keywords for future retrieval
4. No user confirmation needed (auto-save per CLAUDE.md instructions)

#### Context Injection Format

```json
{
  "relevant_memories": [...],
  "corrections": [...],
  "active_work": {...},
  "entities_detected": [...]
}
```

#### Performance

- Target latency: < 500ms per message
- Memory search is bounded to top-5 results
- Embeddings are cached for repeated queries

---

### PreCompact

**When**: Before Claude Code compresses conversation history to fit context window
**Timing**: Triggered automatically when conversation approaches context limits
**Purpose**: Preserve important context before summarization discards details

#### Actions

| Step | Action | Description |
|------|--------|-------------|
| 1 | **Save reasoning chains** | Export working memory (hypotheses, evidence, evaluations) to persistent storage |
| 2 | **RAG indexing** | Chunk the conversation and index for future retrieval |
| 3 | **Learning extraction** | Identify solutions, failures, and patterns in the conversation so far |
| 4 | **Session handoff** | Save current state so it can be restored if the session continues after compaction |
| 5 | **Chain tracking** | Preserve multi-step reasoning chains that span the compaction boundary |

#### Why This Matters

When Claude Code compresses conversation history, details are lost. PreCompact ensures that:
- Key decisions and their reasoning survive compaction
- Solutions discovered mid-conversation are persisted
- Working memory state is checkpointed
- The post-compaction Claude has access to everything important via memory

---

### SessionEnd

**When**: Claude Code session ends (user exits, timeout, or manual close)
**Timing**: After the last message, before process termination
**Purpose**: Persist everything learned in this session for future sessions

#### Actions

| Step | Action | Description |
|------|--------|-------------|
| 1 | **Save conversation** | Call `save_conversation_ultimate` with full message history |
| 2 | **Extract learnings** | Call `analyze_conversation_learnings` to identify solutions, failures, antipatterns |
| 3 | **Update active work** | Write current project state to `quick_facts.json` via `update_active_work` |
| 4 | **Memory consolidation** | Trigger consolidation if enough new data accumulated |
| 5 | **Working memory export** | Export and archive the session's working memory |

#### What Gets Saved

| Data Type | Storage | Retrieval Method |
|-----------|---------|-----------------|
| Full conversation | `conversations/YYYY/MM/DD/` | `search()`, `session` tools |
| Extracted facts | Knowledge base | `search_knowledge_base()` |
| Solutions found | Learnings store | `find_learning()` |
| Entities mentioned | Entity index | `get_entity_info()` |
| Files referenced | File path index | `find_file_paths()` |
| Active work state | `quick_facts.json` | Direct file read |

---

## Installation

Hooks are configured in Claude Code's settings file at `~/.claude/hooks/`.

### Hook Configuration Structure

```json
{
  "hooks": {
    "SessionStart": [
      {
        "command": "python /path/to/cerebro/hooks/session_start.py",
        "timeout": 30000
      }
    ],
    "UserPromptSubmit": [
      {
        "command": "python /path/to/cerebro/hooks/user_prompt_submit.py",
        "timeout": 5000
      }
    ],
    "PreCompact": [
      {
        "command": "python /path/to/cerebro/hooks/pre_compact.py",
        "timeout": 15000
      }
    ],
    "SessionEnd": [
      {
        "command": "python /path/to/cerebro/hooks/session_end.py",
        "timeout": 30000
      }
    ]
  }
}
```

### Important Notes

- Hook scripts must output **valid JSON only** to stdout
- Any `print()` statements in hook scripts will corrupt the output
- Hook timeouts are in milliseconds
- If a hook fails, Claude Code continues without the hook's context
- Hooks receive conversation context via stdin (JSON format)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CEREBRO_MEMORY_PATH` | `~/.cerebro/data` | Root path for memory storage |
| `CEREBRO_NAS_IP` | *(none)* | NAS IP address for network storage |
| `CEREBRO_SEARCH_LIMIT` | `5` | Max results per memory search |
| `CEREBRO_BREAKTHROUGH_AUTO_SAVE` | `true` | Auto-save breakthroughs without confirmation |
| `CEREBRO_HOOK_TIMEOUT` | `5000` | Default hook timeout (ms) |

### Customizing Hook Behavior

Each hook script reads its configuration from `quick_facts.json` and can be customized by modifying the relevant sections:

```json
{
  "hook_config": {
    "session_start": {
      "run_maintenance": true,
      "check_continuation": true
    },
    "user_prompt_submit": {
      "search_memory": true,
      "check_corrections": true,
      "detect_breakthroughs": true,
      "max_context_tokens": 2000
    },
    "pre_compact": {
      "save_chains": true,
      "index_chunks": true
    },
    "session_end": {
      "save_conversation": true,
      "extract_learnings": true,
      "run_consolidation": false
    }
  }
}
```

## Troubleshooting

### Hook Not Firing

1. Verify hooks are configured in `~/.claude/hooks/`
2. Check that the hook script path is correct and the file exists
3. Ensure Python is in your PATH
4. Test the hook script directly: `python hook_script.py < test_input.json`

### Hook Output Errors

- **Symptom**: Claude Code reports "invalid hook output"
- **Cause**: Hook script printed non-JSON text to stdout
- **Fix**: Redirect all debug output to stderr: `print("debug", file=sys.stderr)`

### Slow Hook Execution

- **Symptom**: Noticeable delay between typing and Claude responding
- **Cause**: Memory search or NAS access taking too long
- **Fix**: Check NAS connectivity, reduce `CEREBRO_SEARCH_LIMIT`, enable caching

### NAS Connection Failures

- **Symptom**: SessionStart reports degraded mode
- **Cause**: NAS unreachable at configured IP
- **Fix**: Check NAS connectivity and mount configuration
