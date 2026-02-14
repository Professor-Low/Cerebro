# Cerebro MCP Tools Reference

> Complete reference for all 49 MCP tools available in Cerebro

## Overview

Cerebro exposes 49 tools via the Model Context Protocol (MCP), organized into 10 categories. These tools give Claude Code persistent memory, reasoning, and learning capabilities that survive across sessions, enabling continuous improvement and contextual awareness.

## Tool Categories

| Category | Tools | Purpose |
|----------|-------|---------|
| [Memory Core](#1-memory-core) | 5 | Save, search, and retrieve conversations and knowledge |
| [Knowledge Graph](#2-knowledge-graph) | 5 | Query entities, timelines, files, and user context |
| [3-Tier Memory](#3-3-tier-memory) | 6 | Episodic, semantic, and working memory management |
| [Reasoning & Intelligence](#4-reasoning--intelligence) | 5 | Causal reasoning, prediction, self-modeling |
| [Learning System](#5-learning-system) | 4 | Record and retrieve solutions, failures, corrections |
| [Session Continuity](#6-session-continuity) | 6 | Session handoff, continuation, and active work tracking |
| [User Intelligence](#7-user-intelligence) | 5 | Preferences, personality, goals, and suggestions |
| [Project Management](#8-project-management) | 2 | Project state tracking and version evolution |
| [Quality & Maintenance](#9-quality--maintenance) | 5 | Data quality, confidence, decay, and health monitoring |
| [Meta & Utilities](#10-meta--utilities) | 6 | Meta-learning, privacy, devices, branching, and images |

---

## 1. Memory Core

Core tools for saving and retrieving conversations and knowledge.

| Tool | Description | Key Parameters | Example |
|------|-------------|----------------|---------|
| `save_conversation_ultimate` | Save a conversation with comprehensive extraction of facts, entities, actions, decisions, and code snippets | `messages` (required), `session_id`, `metadata` | `save_conversation_ultimate(messages=[{role: "user", content: "How do I..."}])` |
| `search` | Primary search tool. Hybrid mode combines semantic + keyword search for best results | `query` (required), `mode` (hybrid/semantic/rag), `alpha` (0-1), `top_k`, `chunk_type` | `search(query="docker port conflict", mode="hybrid", top_k=5)` |
| `search_knowledge_base` | Search extracted facts with confidence scores. Filters out superseded facts by default | `query` (required), `fact_type`, `limit`, `include_superseded` | `search_knowledge_base(query="NAS IP address", fact_type="configuration")` |
| `search_by_device` | Search conversations filtered by originating device | `query` (required), `device_tag`, `include_untagged` | `search_by_device(query="GPU training", device_tag="dgx_spark")` |
| `get_chunk` | Retrieve a specific chunk by ID for context injection | `chunk_id` (required), `conversation_id` (required), `include_context` | `get_chunk(chunk_id="a1b2c3d4e5f6g7h8", conversation_id="conv_123")` |

### Search Modes Explained

| Mode | Best For | How It Works |
|------|----------|--------------|
| `hybrid` (default) | General queries | Combines semantic understanding + keyword matching |
| `semantic` | Conceptual queries | Pure vector similarity search |
| `rag` | Question answering | Retrieval-augmented generation optimized |

---

## 2. Knowledge Graph

Tools for querying the structured knowledge graph: entities, timelines, files, and user profiles.

| Tool | Description | Key Parameters | Example |
|------|-------------|----------------|---------|
| `get_entity_info` | Get information about a specific entity (tool, server, person, technology) | `entity_type` (required), `entity_name` (required) | `get_entity_info(entity_type="technologies", entity_name="Python")` |
| `get_timeline` | Get chronological history of actions and decisions for a specific month | `year_month` (required, YYYY-MM), `event_type` (action/decision/all) | `get_timeline(year_month="2026-02", event_type="decision")` |
| `find_file_paths` | Find all file paths mentioned in conversations with their purpose | `path_pattern` (required), `purpose` (configuration/script/log/data_storage/hook_or_command) | `find_file_paths(path_pattern="cerebro", purpose="configuration")` |
| `get_user_context` | Get comprehensive user context: goals, preferences, technical environment | `limit` (recent conversations to analyze) | `get_user_context(limit=10)` |
| `get_user_profile` | Get the user's personal profile: identity, relationships, projects, preferences | `category` (all/identity/relationships/projects/preferences/goals/technical_environment) | `get_user_profile(category="technical_environment")` |

### Entity Types

| Type | Examples |
|------|----------|
| `tools` | Docker, Git, Claude Code |
| `networks` | 10.0.0.0/24 |
| `people` | User, collaborators |
| `servers` | NAS, DGX Spark |
| `technologies` | Python, FastAPI, FAISS |

---

## 3. 3-Tier Memory

Direct access to the three memory tiers: episodic (events), semantic (facts), and working (active reasoning).

### memory_type

Unified tool for episodic and semantic memory operations.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `query_episodic` | Find events by date, actor, or emotion | `query`, `date`, `date_range`, `actor`, `emotion` | `memory_type(action="query_episodic", date="2026-02-10", actor="Claude")` |
| `query_semantic` | Find general facts by domain or keywords | `query`, `domain`, `keywords` | `memory_type(action="query_semantic", domain="infrastructure")` |
| `save_episodic` | Record a specific event with context | `event` (required), `outcome`, `emotional_state` | `memory_type(action="save_episodic", event="Migrated to FAISS", outcome="success")` |
| `save_semantic` | Store a general fact | `fact` (required) | `memory_type(action="save_semantic", fact="DGX Spark has 128GB RAM")` |
| `stats` | Get memory statistics | _(none)_ | `memory_type(action="stats")` |
| `link` | Link an episodic event to a semantic fact | `episode_id`, `semantic_id` | `memory_type(action="link", episode_id="ep_1", semantic_id="sem_1")` |

### working_memory

Manage active reasoning state that persists across context compactions.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `get_active` | Get current working memory session | `active_goal` | `working_memory(action="get_active")` |
| `create` | Start a new working memory session | `active_goal` | `working_memory(action="create", active_goal="Debug auth flow")` |
| `add_chain` | Add a reasoning hypothesis to test | `hypothesis` | `working_memory(action="add_chain", hypothesis="Token expiry causes 401")` |
| `update_chain` | Add evidence or evaluation to a hypothesis | `chain_id`, `evidence`, `evaluation` | `working_memory(action="update_chain", chain_id="c1", evaluation="supported")` |
| `add_note` | Add a scratch pad note | `note`, `note_category` | `working_memory(action="add_note", note="Check Redis TTL")` |
| `get_summary` | Get working memory summary | _(none)_ | `working_memory(action="get_summary")` |
| `export` | Export for session handoff | _(none)_ | `working_memory(action="export")` |
| `import` | Restore from handoff data | `handoff_data` | `working_memory(action="import", handoff_data={...})` |

### consolidate

Active memory consolidation: cluster episodes, create abstractions, strengthen connections.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `run` | Trigger consolidation | `full` (process all vs. recent only) | `consolidate(action="run", full=false)` |
| `get_state` | Get current consolidation state | _(none)_ | `consolidate(action="get_state")` |
| `get_abstractions` | Get generated abstractions | `domain`, `limit` | `consolidate(action="get_abstractions", domain="debugging")` |
| `schedule` | Schedule next consolidation | `hours` | `consolidate(action="schedule", hours=24)` |
| `stats` | Get consolidation statistics | _(none)_ | `consolidate(action="stats")` |

---

## 4. Reasoning & Intelligence

Tools for active reasoning, causal modeling, prediction, and self-awareness.

### reason

Active reasoning over memories to generate insights.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `analyze` | Full reasoning over provided memories | `memories` | `reason(action="analyze", memories=[...])` |
| `find_insights` | Find insights relevant to a query | `query` | `reason(action="find_insights", query="deployment failures")` |
| `proactive` | Surface insights during conversation | `user_message` | `reason(action="proactive", user_message="Setting up CI/CD")` |
| `validate` | Confirm or refute a generated insight | `insight_id`, `is_valid` | `reason(action="validate", insight_id="ins_1", is_valid=true)` |
| `goal` | Reason about achieving a specific goal | `goal` | `reason(action="goal", goal="Reduce response latency")` |

### causal

Manage cause-effect relationships for understanding WHY things happen.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `add_link` | Record a cause-effect relationship | `cause`, `effect`, `mechanism`, `interventions` | `causal(action="add_link", cause="High memory", effect="OOM crash", mechanism="No GC")` |
| `find_causes` | What causes a given effect? | `effect` | `causal(action="find_causes", effect="Connection timeout")` |
| `find_effects` | What effects does a cause produce? | `cause` | `causal(action="find_effects", cause="NAS disconnection")` |
| `get_interventions` | How to prevent an effect | `effect` | `causal(action="get_interventions", effect="Data loss")` |
| `what_if` | Simulate an intervention | `intervention`, `context` | `causal(action="what_if", intervention="Add retry logic")` |
| `search` | Search causal links | `query` | `causal(action="search", query="timeout")` |

### predict

Predictive simulation using causal model and history.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `from_causal` | Predict outcome of an action | `action_text`, `context` | `predict(action="from_causal", action_text="Deploy to prod")` |
| `anticipate_failures` | Get failure warnings for context | `context` | `predict(action="anticipate_failures", context="Docker migration")` |
| `check_pattern` | Check for specific failure pattern | `pattern_type` (timeout/encoding/path/network/permission) | `predict(action="check_pattern", pattern_type="timeout")` |
| `preventive_actions` | Get prevention suggestions | `context` | `predict(action="preventive_actions", context="Database upgrade")` |
| `verify` | Record whether a prediction was correct | `prediction_id`, `outcome` | `predict(action="verify", prediction_id="p_1", outcome="correct")` |

### self_model

Continuous self-modeling for real-time self-awareness.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `get_state` | Get current self-model state | _(none)_ | `self_model(action="get_state")` |
| `update_confidence` | Set confidence level for current task | `confidence` (0-1) | `self_model(action="update_confidence", confidence=0.8)` |
| `add_uncertainty` | Record an area of uncertainty | `topic` | `self_model(action="add_uncertainty", topic="Kubernetes networking")` |
| `add_limitation` | Record a known limitation | `topic` | `self_model(action="add_limitation", topic="Cannot access private repos")` |
| `add_strength` | Record a known strength | `topic` | `self_model(action="add_strength", topic="Python debugging")` |
| `update_quality` | Update response quality metrics | `hallucination_risk`, `evidence_sufficiency`, `task_clarity` | `self_model(action="update_quality", hallucination_risk=0.1)` |
| `introspect` | Deep self-analysis of text | `text`, `use_llm` | `self_model(action="introspect", text="My analysis of...")` |
| `hallucination_check` | Check text for hallucination risk | `text` | `self_model(action="hallucination_check", text="The API returns...")` |

### analyze

Pattern analysis, knowledge gaps, and skill development tracking.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `patterns` | Find recurring patterns across conversations | `threshold` (minimum occurrences) | `analyze(type="patterns", threshold=3)` |
| `knowledge_gaps` | Identify areas where knowledge is lacking | `threshold` | `analyze(type="knowledge_gaps", threshold=3)` |
| `skill` | Track development of a specific skill | `skill` | `analyze(type="skill", skill="Docker")` |
| `validated_patterns` | Get patterns that are confirmed and ready to apply | `max_per_type` | `analyze(type="validated_patterns", max_per_type=5)` |

---

## 5. Learning System

Tools for recording, retrieving, and managing learnings from experience.

| Tool | Description | Key Parameters | Example |
|------|-------------|----------------|---------|
| `record_learning` | Save a solution, failure, or antipattern | `type` (solution/failure/antipattern/confirm), `problem`, `solution`, `tags` | `record_learning(type="solution", problem="FAISS OOM", solution="Use batch indexing", tags=["faiss","memory"])` |
| `find_learning` | Search for proven solutions or antipatterns | `type` (solution/antipattern/chain/summary), `problem`, `tags` | `find_learning(type="solution", problem="Docker port conflict")` |
| `analyze_conversation_learnings` | Extract learnings from a past conversation | `conversation_id` (required), `save` | `analyze_conversation_learnings(conversation_id="conv_abc", save=true)` |
| `get_corrections` | Get corrections Claude learned from the user. Check BEFORE answering to avoid repeating mistakes | `query`, `topic`, `importance` | `get_corrections(topic="network", importance="high")` |

### Learning Types

| Type | Purpose | Example |
|------|---------|---------|
| `solution` | What worked | "Use `encoding='utf-8-sig'` for BOM files" |
| `failure` | What didn't work (linked to a solution) | "Plain utf-8 fails on AST parse" |
| `antipattern` | What to never do | "Don't use `git add -A` in shared repos" |
| `confirm` | Reinforce a known solution | "BOM encoding fix confirmed again" |

---

## 6. Session Continuity

Tools for maintaining context across Claude Code sessions.

| Tool | Description | Key Parameters | Example |
|------|-------------|----------------|---------|
| `check_session_continuation` | Check if there is recent work to continue. Call at session start | `hours` (how far back to look, default 48) | `check_session_continuation(hours=24)` |
| `get_continuation_context` | Get full context for resuming a previous session | `session_id` (required, from check above) | `get_continuation_context(session_id="sess_abc")` |
| `update_active_work` | Track current project state for session handoff | `project`, `current_phase`, `phase_name`, `last_completed`, `next_action`, `key_files` | `update_active_work(project="Cerebro", next_action="Implement auth")` |
| `session_handoff` | Save or restore working memory for handoff | `action` (save/get_latest/get_recent/restore), `handoff_data`, `reason` | `session_handoff(action="save", reason="session_end")` |
| `working_memory` | Manage active reasoning state (see 3-Tier Memory) | `action`, various | See 3-Tier Memory section |
| `session` | Session information and detection | `action` (thread/active/summary/detect), `session_id`, `user_prompt` | `session(action="detect", user_prompt="Continue from yesterday")` |

### Session Continuation Flow

```
Session Start
    │
    ▼
check_session_continuation(hours=48)
    │
    ├── Found continuable work
    │   ├── get_continuation_context(session_id)
    │   └── Resume with full context
    │
    └── No continuable work
        └── Start fresh session
```

---

## 7. User Intelligence

Tools for understanding and adapting to the user.

| Tool | Description | Key Parameters | Example |
|------|-------------|----------------|---------|
| `preferences` | Get or update user preferences (communication, workflow, technical) | `action` (get/update/evolved/decay/contradictions), `category`, `preference`, `positive` | `preferences(action="update", category="workflow", preference="Use bun over npm", positive=true)` |
| `personality` | Track personality evolution and communication style | `action` (traits/evolution/consistency/evolve/sync), `feedback_type`, `content` | `personality(action="evolve", feedback_type="correction", content="Be more concise")` |
| `goals` | Track and reason about user goals | `action` (detect/add/get/update/complete/list_active/proactive_context/find_relevant), `text`, `description` | `goals(action="detect", text="I want to deploy Cerebro as open source")` |
| `suggest_questions` | Get suggested questions to fill knowledge gaps in user profile | `importance` (critical/helpful/all), `limit` | `suggest_questions(importance="critical", limit=3)` |
| `get_suggestions` | Get proactive suggestions based on current context | `user_message` (required), `cwd`, `recent_tools` | `get_suggestions(user_message="Setting up CI pipeline")` |

### Preference Categories

| Category | Examples |
|----------|----------|
| `communication_style` | Direct, no pleasantries, solution-focused |
| `workflow` | Automation over manual, memory-first protocol |
| `technical` | Python 3.13, PowerShell, Docker, FAISS |

---

## 8. Project Management

Tools for tracking project state and version history.

### projects

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `state` | Get current state of a project | `project_id` or `file_path` | `projects(action="state", project_id="cerebro")` |
| `active` | List active projects | `status` (active/paused/completed/all) | `projects(action="active", status="active")` |
| `stale` | Find inactive projects | `stale_days` (default 14) | `projects(action="stale", stale_days=30)` |
| `auto_update` | Run automatic status transitions | _(none)_ | `projects(action="auto_update")` |
| `activity` | Get activity summary | `days` (default 7) | `projects(action="activity", days=30)` |

### project_evolution

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `record` | Record a new version milestone | `project_id`, `version`, `summary`, `keywords` | `project_evolution(action="record", project_id="cerebro", version="2.0.0", summary="Claude agents replace LLM")` |
| `timeline` | Get version history | `project_id` | `project_evolution(action="timeline", project_id="cerebro")` |
| `supersede` | Mark an old version as superseded | `old_version`, `new_version`, `reason` | `project_evolution(action="supersede", old_version="1.0", new_version="2.0", reason="Architecture rewrite")` |
| `list` | List all tracked projects | _(none)_ | `project_evolution(action="list")` |

---

## 9. Quality & Maintenance

Tools for maintaining data quality, confidence, and system health.

### quality

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `stats` | Get overall quality statistics | _(none)_ | `quality(action="stats")` |
| `score` | Get quality score | _(none)_ | `quality(action="score")` |
| `duplicates` | Find duplicate chunks | `threshold` (0-1, default 0.90) | `quality(action="duplicates", threshold=0.85)` |
| `merge` | Merge duplicate chunks | `auto_merge` (default false) | `quality(action="merge", auto_merge=true)` |
| `fact_duplicates` | Find duplicate facts | _(none)_ | `quality(action="fact_duplicates")` |
| `fact_merge` | Merge duplicate facts | _(none)_ | `quality(action="fact_merge")` |

### confidence

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `stats` | Confidence overview | _(none)_ | `confidence(action="stats")` |
| `get` | Get specific fact's confidence | `fact_id` | `confidence(action="get", fact_id="f_123")` |
| `reinforce` | Boost a fact's confidence | `fact_id`, `reason` | `confidence(action="reinforce", fact_id="f_123", reason="Confirmed by user")` |
| `low` | Find low-confidence facts | `threshold` (default 0.5) | `confidence(action="low", threshold=0.4)` |
| `quarantine` | Quarantine unreliable facts | `threshold` (default 0.4) | `confidence(action="quarantine")` |
| `detect_contradictions` | Find conflicting facts | `content` | `confidence(action="detect_contradictions", content="NAS IP is 10.0.0.100")` |
| `resolve_contradiction` | Resolve a specific contradiction | `fact_id_a`, `fact_id_b`, `status`, `resolution` | `confidence(action="resolve_contradiction", fact_id_a="f_1", fact_id_b="f_2", status="confirmed")` |

### decay

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `run` | Execute decay on stale data | `force`, `include_conversations`, `include_facts`, `include_summaries` | `decay(action="run")` |
| `preview` | Dry-run decay without changes | _(none)_ | `decay(action="preview")` |
| `stats` | Get decay statistics | _(none)_ | `decay(action="stats")` |
| `storage` | Get storage size report | _(none)_ | `decay(action="storage")` |
| `golden` | Manage protected (never-decay) items | `golden_action` (list/add/remove), `item_id` | `decay(action="golden", golden_action="add", item_id="f_123")` |

### self_report

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `metrics` | Get performance trends | `days` (default 30) | `self_report(action="metrics", days=7)` |
| `improvements` | List tracked improvements | _(none)_ | `self_report(action="improvements")` |
| `report` | Full self-improvement analysis | `days` | `self_report(action="report", days=30)` |
| `record_metric` | Record a performance metric | `metric_name`, `value` | `self_report(action="record_metric", metric_name="search_latency", value=120)` |
| `record_improvement` | Record a before/after improvement | `improvement_name`, `description`, `baseline_value` | `self_report(action="record_improvement", improvement_name="Search speed", baseline_value=500)` |

### system_health_check

Check health of all components: NAS, embeddings, indexes, database, MCP.

| Parameter | Description |
|-----------|-------------|
| _(none)_ | Returns overall status and detailed component diagnostics |

**Example**: `system_health_check()` returns `{ status: "healthy", components: { nas: "connected", faiss: "loaded", ... } }`

---

## 10. Meta & Utilities

Advanced tools for meta-learning, privacy, devices, branching, and media.

### meta_learn

Retrieval strategy optimization through experimentation.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `record_query` | Track search performance | `query`, `strategy_id`, `latency_ms`, `result_count`, `success` | `meta_learn(action="record_query", query="docker", latency_ms=45, success=true)` |
| `feedback` | Record user feedback on results | `query_id`, `positive` | `meta_learn(action="feedback", query_id="q_1", positive=true)` |
| `recommend` | Get recommended search strategy | `query` | `meta_learn(action="recommend", query="deployment errors")` |
| `create_experiment` | Create A/B test for strategies | `experiment_name`, `strategy_a`, `strategy_b` | `meta_learn(action="create_experiment", experiment_name="alpha_test", strategy_a="hybrid_0.7", strategy_b="hybrid_0.5")` |

### privacy

Secret detection and data protection.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `scan` | Scan text for secrets/credentials | `text` (required), `redact`, `min_confidence` | `privacy(action="scan", text="API_KEY=sk-abc123", redact=true)` |
| `stats` | Get redaction statistics | _(none)_ | `privacy(action="stats")` |
| `sensitive` | List sensitive conversations | `limit` | `privacy(action="sensitive", limit=10)` |

### device

Device management for multi-device memory.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `current` | Get current device info | _(none)_ | `device(action="current")` |
| `all` | List all registered devices | _(none)_ | `device(action="all")` |
| `register` | Register a new device | `friendly_name`, `description` | `device(action="register", friendly_name="DGX Spark", description="128GB NVIDIA server")` |

### branch

Exploration branch management for tracking alternative approaches.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `create` | Create a new exploration branch | `name`, `description`, `parent_conversation_id` | `branch(action="create", name="Try Redis caching", description="Test Redis vs file cache")` |
| `mark` | Mark branch as chosen or abandoned | `branch_id`, `status` (chosen/abandoned), `reason` | `branch(action="mark", branch_id="b_1", status="chosen", reason="40% faster")` |

### conversation

Manage conversation metadata.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `tag` | Add tags to a conversation | `conversation_id` (required), `tags` | `conversation(action="tag", conversation_id="conv_1", tags=["debugging", "docker"])` |
| `note` | Add a note to a conversation | `conversation_id` (required), `note` | `conversation(action="note", conversation_id="conv_1", note="Key breakthrough here")` |
| `relevance` | Set conversation relevance level | `conversation_id` (required), `relevance` | `conversation(action="relevance", conversation_id="conv_1", relevance="critical")` |

### images

Save and search screenshots and images.

| Action | Description | Key Parameters | Example |
|--------|-------------|----------------|---------|
| `save` | Save a screenshot with description | `image_data`, `description`, `conversation_id` | `images(action="save", image_data="/path/to/screenshot.png", description="Error dialog")` |
| `search` | Search saved images | `query`, `limit` | `images(action="search", query="error dialog", limit=5)` |

---

## Quick Reference: Most Used Tools

For everyday Claude Code usage, these are the tools used most frequently:

| Tool | When to Use |
|------|-------------|
| `search` | Find anything in memory (default: hybrid mode) |
| `get_corrections` | Before answering to avoid past mistakes |
| `find_learning` | Find proven solutions to known problems |
| `record_learning` | After solving a problem (auto-triggered by hooks) |
| `update_active_work` | Track project state for session handoff |
| `check_session_continuation` | At session start to resume previous work |
| `get_user_profile` | Understand user preferences and context |
| `system_health_check` | Verify all components are healthy |
