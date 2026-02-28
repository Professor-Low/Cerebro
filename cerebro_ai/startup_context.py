#!/usr/bin/env python3
"""
Startup Context - Claude Code Hook for Identity Core Injection
Runs on UserPromptSubmit to inject user context before Claude responds.
Only injects on the FIRST prompt of each Claude Code session.

Phase 1 Enhancement: Relevance scoring for smarter context injection.
Phase 4 Enhancement: Validated pattern injection.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
SRC_PATH = Path(__file__).parent
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from config import CONVERSATIONS_DIR, IDENTITY_CACHE, LOG_DIR, TEMP_DIR

# Session tracking file (separate from startup_check to avoid conflicts)
SESSION_TRACKER = TEMP_DIR / "claude_identity_injection_tracker.json"
DEBUG_LOG = LOG_DIR / "startup_context_debug.log"

# Problem detection patterns (Enhancement 3: Preflight antipattern check)
PROBLEM_KEYWORDS = [
    'problem', 'error', 'fix', 'issue', 'failing', 'broken', 'not working',
    'bug', 'crash', 'exception', 'fail', 'trouble', 'wrong', 'stuck'
]


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
        log_debug(f"Identity already injected for session {claude_session_id}")
        return False

    tracker[claude_session_id] = {"timestamp": datetime.now().isoformat()}

    try:
        with open(SESSION_TRACKER, "w") as f:
            json.dump(tracker, f)
    except:
        pass

    log_debug(f"First prompt for session {claude_session_id}, will inject identity")
    return True


def load_identity_core() -> dict:
    """Load the identity core from cache."""
    if not IDENTITY_CACHE.exists():
        log_debug(f"Identity cache not found at {IDENTITY_CACHE}")
        return None

    try:
        with open(IDENTITY_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_debug(f"Error loading identity cache: {e}")
        return None


def is_problem_prompt(user_message: str) -> bool:
    """
    Detect if user message describes a problem.
    Enhancement 3: Preflight antipattern check.
    """
    if not user_message:
        return False

    message_lower = user_message.lower()
    matches = sum(1 for kw in PROBLEM_KEYWORDS if kw in message_lower)
    return matches >= 1


def extract_problem_keywords(user_message: str) -> list:
    """Extract keywords from problem description for antipattern matching."""
    if not user_message:
        return []

    # Extract significant words (excluding stopwords)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'i', 'my', 'me',
                 'it', 'this', 'that', 'to', 'in', 'on', 'at', 'for', 'with', 'and'}

    words = re.findall(r'\b[a-z]{3,}\b', user_message.lower())
    keywords = [w for w in words if w not in stopwords]

    # Also extract technical terms (uppercase-containing, like NAS, MCP)
    tech_terms = re.findall(r'\b[A-Z][A-Za-z0-9_]+\b', user_message)
    keywords.extend([t.lower() for t in tech_terms])

    return list(set(keywords))[:10]  # Max 10 keywords


def load_suggestions(user_message: str, cwd: str = None) -> list:
    """
    Load proactive suggestions based on current context.
    Uses the Anticipation Engine (Phase 5).

    Returns list of formatted suggestions.
    """
    if not user_message or len(user_message) < 10:
        return []

    try:
        # Add src to path for imports
        src_path = Path(__file__).parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from anticipation_engine import AnticipationEngine
        engine = AnticipationEngine()

        result = engine.get_suggestions(
            user_message=user_message[:500],
            cwd=cwd or os.getcwd(),
            recent_tools=[],  # Not available at startup
            conversation_length=1,  # First message
            max_suggestions=2  # Keep it concise
        )

        suggestions = result.get("suggestions", [])

        if not suggestions:
            log_debug("No suggestions generated for this context")
            return []

        # Format suggestions
        formatted = []
        for s in suggestions:
            stype = s.get("type", "context")
            content = s.get("content", "")[:100]
            if content:
                formatted.append(f"- [{stype.upper()}] {content}")

        log_debug(f"Generated {len(formatted)} suggestions")
        return formatted

    except Exception as e:
        log_debug(f"Error loading suggestions: {e}")
        return []


def format_suggestions_block(suggestions: list) -> str:
    """Format suggestions as injection block."""
    if not suggestions:
        return ""

    lines = ["[SUGGESTIONS]"]
    lines.extend(suggestions)
    lines.append("[/SUGGESTIONS]")
    return "\n".join(lines)


def load_relevant_antipatterns(user_message: str) -> list:
    """
    Load antipatterns relevant to the user's problem.
    Enhancement 3: Preflight antipattern check.

    Returns list of formatted antipattern warnings.
    """
    if not is_problem_prompt(user_message):
        return []

    try:
        # Add src to path for imports
        src_path = Path(__file__).parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from solution_tracker import SolutionTracker
        tracker = SolutionTracker()

        # Extract keywords for tag-based matching
        keywords = extract_problem_keywords(user_message)

        # Search by problem text and tags
        antipatterns = tracker.find_antipatterns(
            problem=user_message[:200],  # Limit problem text length
            tags=keywords
        )

        if not antipatterns:
            log_debug(f"No antipatterns found for: {user_message[:50]}...")
            return []

        # Format top 3 antipatterns
        formatted = []
        for ap in antipatterns[:3]:
            what_not_to_do = ap.get('what_not_to_do', '')[:60]
            why_failed = ap.get('why_it_failed', '')[:60]
            if what_not_to_do:
                formatted.append(f"- DONT: {what_not_to_do} (BECAUSE: {why_failed})")

        log_debug(f"Found {len(formatted)} relevant antipatterns")
        return formatted

    except Exception as e:
        log_debug(f"Error loading antipatterns: {e}")
        return []


def format_antipattern_block(antipatterns: list) -> str:
    """Format antipatterns as injection block."""
    if not antipatterns:
        return ""

    lines = ["[AVOID - KNOWN FAILURES]"]
    lines.extend(antipatterns)
    lines.append("[/AVOID]")
    return "\n".join(lines)


def format_identity_context(identity: dict) -> str:
    """Format the identity core as injection text (max 300 tokens target)."""
    if not identity:
        return None

    lines = ["[MEMORY CONTEXT]"]

    # User tags (compact)
    user_tags = identity.get("user_tags", "")
    if user_tags:
        lines.append(f"USER: {user_tags}")

    # Active projects (compact)
    projects = identity.get("projects", [])
    if projects:
        proj_strs = []
        for p in projects[:3]:  # Max 3 projects
            name = p.get("name", "unknown")
            status = p.get("status", "")
            context = p.get("context", "")
            proj_strs.append(f"{name} ({status} - {context})")
        lines.append(f"PROJECTS: {'; '.join(proj_strs)}")

    # Corrections (most important)
    corrections = identity.get("corrections", [])
    if corrections:
        lines.append("CORRECTIONS:")
        for c in corrections[:5]:  # Max 5 corrections
            level = c.get("level", "")
            text = c.get("text", "")
            if text:
                lines.append(f"- [{level}] {text}")

    # Last session (brief)
    last_session = identity.get("last_session", "")
    if last_session:
        lines.append(f"LAST: {last_session[:150]}")

    # Recent decisions (Enhancement 4)
    recent_decisions = identity.get("recent_decisions", [])
    if recent_decisions:
        lines.append("RECENT DECISIONS:")
        for d in recent_decisions[:3]:  # Max 3 decisions
            decision = d.get("decision", "")[:50]
            reasoning = d.get("reasoning", "")[:40]
            if decision:
                if reasoning:
                    lines.append(f"- {decision} (because: {reasoning})")
                else:
                    lines.append(f"- {decision}")

    # Environment hints (compact)
    env = identity.get("environment", {})
    if env:
        device = env.get("device", "")
        if device:
            lines.append(f"DEVICE: {device}")

    lines.append("[/MEMORY CONTEXT]")

    return "\n".join(lines)


def get_git_status(cwd: str = None) -> dict:
    """
    Get git status for the current working directory.
    Returns uncommitted changes that represent current work-in-progress.

    This is CRITICAL for session continuation - uncommitted changes are
    often the most relevant context for "where did we leave off?"
    """
    if cwd is None:
        cwd = os.getcwd()

    result = {
        "is_git_repo": False,
        "modified_files": [],
        "untracked_files": [],
        "branch": "",
        "has_changes": False
    }

    try:
        # Check if this is a git repo
        check = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=cwd, capture_output=True, text=True, timeout=5
        )
        if check.returncode != 0:
            return result

        result["is_git_repo"] = True

        # Get current branch
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd, capture_output=True, text=True, timeout=5
        )
        if branch.returncode == 0:
            result["branch"] = branch.stdout.strip()

        # Get modified files (staged and unstaged)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=5
        )
        if status.returncode == 0 and status.stdout.strip():
            for line in status.stdout.strip().split("\n"):
                if not line:
                    continue
                status_code = line[:2]
                filepath = line[3:].strip()
                # Skip binary files and common noise
                if any(filepath.endswith(ext) for ext in ['.pyc', '.pyo', '__pycache__']):
                    continue
                if status_code.startswith("?"):  # Untracked
                    result["untracked_files"].append(filepath)
                else:  # Modified, Added, Deleted, etc.
                    result["modified_files"].append(filepath)

        result["has_changes"] = bool(result["modified_files"] or result["untracked_files"])
        log_debug(f"Git status: {len(result['modified_files'])} modified, {len(result['untracked_files'])} untracked")

    except subprocess.TimeoutExpired:
        log_debug("Git status timed out")
    except FileNotFoundError:
        log_debug("Git not found")
    except Exception as e:
        log_debug(f"Git status error: {e}")

    return result


def format_git_context(git_status: dict) -> str:
    """Format git status as injection block."""
    if not git_status.get("has_changes"):
        return ""

    lines = ["[GIT WORK-IN-PROGRESS]"]

    if git_status.get("branch"):
        lines.append(f"Branch: {git_status['branch']}")

    # Modified files are most important
    modified = git_status.get("modified_files", [])
    if modified:
        # Show up to 5 modified files
        lines.append(f"Modified ({len(modified)}): {', '.join(modified[:5])}")
        if len(modified) > 5:
            lines.append(f"  ...and {len(modified) - 5} more")

    # Untracked files (often new work)
    untracked = git_status.get("untracked_files", [])
    if untracked:
        # Show up to 3 untracked files
        lines.append(f"New files ({len(untracked)}): {', '.join(untracked[:3])}")
        if len(untracked) > 3:
            lines.append(f"  ...and {len(untracked) - 3} more")

    lines.append("[/GIT WORK-IN-PROGRESS]")

    return "\n".join(lines)


def detect_current_project(cwd: str = None) -> str:
    """
    Detect project name from current working directory.
    Looks for project indicators like pyproject.toml, package.json, etc.
    """
    if cwd is None:
        cwd = os.getcwd()

    cwd_path = Path(cwd)

    # Check for common project files
    project_files = ['pyproject.toml', 'package.json', 'Cargo.toml', 'go.mod', 'setup.py']
    for f in project_files:
        if (cwd_path / f).exists():
            return cwd_path.name

    # Fall back to directory name if it's a git repo
    if (cwd_path / '.git').exists():
        return cwd_path.name

    # Check parent directory (common for src/ subdirs)
    for f in project_files:
        if (cwd_path.parent / f).exists():
            return cwd_path.parent.name

    return None


def load_relevant_context(user_message: str, cwd: str = None, project: str = None) -> list:
    """
    Load relevant context using Phase 1 Enhancement scoring.

    Uses:
    - RelevanceScorer: Score past conversations/facts by relevance
    - TokenBudgetManager: Fit within token limits
    - CWDDetector: Get project context from CWD

    Returns list of formatted relevant context items.
    """
    try:
        from context import CWDDetector, RelevanceScorer, TokenBudget, TokenBudgetManager

        # Get current context
        current_context = {
            "query": user_message[:500] if user_message else "",
            "project": project,
            "topics": [],
            "cwd": cwd,
        }

        # Detect project from CWD if not provided
        if not project and cwd:
            detector = CWDDetector()
            project_info = detector.detect_project(cwd)
            current_context["project"] = project_info.get("name")

        # Extract topics from message
        topic_patterns = [
            r'\b(api|database|auth|cache|memory|session|config|deploy|test|debug)\b',
            r'\b(python|javascript|typescript|rust|go|java|sql)\b',
            r'\b(docker|kubernetes|aws|linux|git)\b',
            r'\b(error|bug|fix|issue|problem)\b',
        ]
        for pattern in topic_patterns:
            matches = re.findall(pattern, user_message.lower() if user_message else "")
            current_context["topics"].extend(matches)
        current_context["topics"] = list(set(current_context["topics"]))

        # Load conversations for scoring
        if not CONVERSATIONS_DIR.exists():
            log_debug("Conversations path not found")
            return []

        # Load recent conversations (last 30 days, max 50)
        items = []
        cutoff = datetime.now() - timedelta(days=30)

        for conv_file in list(CONVERSATIONS_DIR.glob("*.json"))[:100]:
            try:
                with open(conv_file, "r", encoding="utf-8") as f:
                    conv = json.load(f)

                # Check timestamp
                timestamp = conv.get("timestamp", "")
                if timestamp:
                    try:
                        conv_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        if hasattr(conv_time, 'tzinfo') and conv_time.tzinfo:
                            conv_time = conv_time.replace(tzinfo=None)
                        if conv_time < cutoff:
                            continue
                    except:
                        pass

                # Build summary from available fields
                metadata = conv.get("metadata", {})
                extracted = conv.get("extracted_data", {})

                # Get summary: try topic -> facts -> first user message
                summary = metadata.get("topic", "")
                if not summary:
                    facts = extracted.get("facts", [])
                    if facts and isinstance(facts[0], dict):
                        summary = facts[0].get("content", "")[:200]
                    elif facts:
                        summary = str(facts[0])[:200]

                # Fallback: use first user message
                if not summary:
                    messages = conv.get("messages", [])
                    for msg in messages:
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                            if content and len(content) > 10:
                                summary = content[:150]
                                break

                # Detect project from tags
                project = ""
                tags = metadata.get("tags", [])
                for tag in tags:
                    tag_lower = str(tag).lower()
                    if "memory" in tag_lower or "mcp" in tag_lower:
                        project = "ai-memory-mcp"
                        break

                items.append({
                    "id": conv.get("id", conv_file.stem),
                    "summary": summary,
                    "content": summary,  # For text similarity
                    "topics": metadata.get("topics", []),
                    "project": project,
                    "timestamp": timestamp,
                    "has_solution": bool(extracted.get("problems_solved")),
                })
            except Exception:
                continue

        if not items:
            log_debug("No recent conversations found")
            return []

        # Score items (lower threshold since natural scores are ~0.2-0.4)
        scorer = RelevanceScorer()
        scored = scorer.score_items(items, current_context, min_score=0.15)

        # Apply token budget (context portion only: 350 tokens)
        budget_manager = TokenBudgetManager(TokenBudget(context=350))

        # Format for budget manager
        context_items = [(item, score) for item, score in scored[:10]]

        fitted = budget_manager.fit_to_budget(context_items=context_items)

        # Format output
        formatted = []
        for item in fitted.get("context", []):
            summary = item.get("summary", "")[:100]
            score = item.get("relevance_score", 0)
            if summary:
                formatted.append(f"- {summary} ({score:.0%} match)")

        log_debug(f"Loaded {len(formatted)} relevant context items")
        return formatted

    except Exception as e:
        log_debug(f"Error loading relevant context: {e}")
        return []


def format_relevant_context_block(items: list) -> str:
    """Format relevant context as injection block."""
    if not items:
        return ""

    lines = ["[RELEVANT CONTEXT]"]
    lines.extend(items[:5])  # Max 5 items
    lines.append("[/RELEVANT CONTEXT]")
    return "\n".join(lines)


def load_validated_patterns() -> str:
    """
    Load validated patterns for injection.
    Phase 4 Enhancement: Pattern Validation.

    Uses:
    - PatternValidator: Validate patterns before acting
    - PatternApplier: Format for injection

    Returns formatted pattern block or empty string.
    """
    try:
        from patterns import PatternApplier, PatternValidator

        validator = PatternValidator()

        # First try to load from cache (faster)
        cached = validator.load_cached_patterns()
        if cached:
            log_debug("Using cached validated patterns")
            applier = PatternApplier(max_patterns=5, max_tokens=200)
            injection = applier.get_injection_context(cached)
            if injection:
                log_debug(f"Pattern injection from cache ({len(injection)} chars)")
                return injection

        # Cache miss or stale - validate fresh
        validated = validator.get_validated_patterns(max_per_type=3)

        # Save to cache for next time
        validator.save_validated_patterns(validated)

        # Format for injection
        applier = PatternApplier(max_patterns=5, max_tokens=200)
        injection = applier.get_injection_context(validated)

        if injection:
            stats = validated.get("stats", {})
            log_debug(f"Pattern injection: {stats.get('total_validated', 0)} patterns validated")
            return injection

        log_debug("No validated patterns to inject")
        return ""

    except Exception as e:
        log_debug(f"Error loading validated patterns: {e}")
        return ""


def reactivate_project_if_needed(identity: dict, git_status: dict) -> bool:
    """
    Re-activate a project if it's marked 'completed' but has uncommitted changes.

    This fixes the issue where a project is marked complete but new work starts,
    and the system still thinks it's completed.

    Returns True if a project was reactivated.
    """
    if not git_status.get("has_changes"):
        return False

    current_project = detect_current_project()
    if not current_project:
        return False

    projects = identity.get("projects", [])
    for project in projects:
        if project.get("name", "").lower() == current_project.lower():
            if project.get("status") == "completed":
                # Project was marked completed but has new uncommitted changes
                # This means new work has started - reactivate it
                log_debug(f"Reactivating project {current_project}: was 'completed' but has uncommitted changes")

                project["status"] = "active"
                old_context = project.get("context", "")
                project["context"] = old_context.replace("(complete)", "").replace("completed", "").strip()
                if "new work" not in project["context"].lower():
                    project["context"] = "new work in progress"

                # Save updated identity cache
                try:
                    with open(IDENTITY_CACHE, "w", encoding="utf-8") as f:
                        json.dump(identity, f, indent=2, ensure_ascii=False)
                    log_debug(f"Saved reactivated project status for {current_project}")
                    return True
                except Exception as e:
                    log_debug(f"Failed to save reactivated project: {e}")

    return False


def main():
    log_debug("Startup context hook started")

    # Read stdin to get the hook input
    stdin_data = ""
    try:
        stdin_data = sys.stdin.read()
        log_debug(f"Received stdin: {stdin_data[:200]}...")
    except Exception as e:
        log_debug(f"Error reading stdin: {e}")

    # Parse the hook input to get Claude's session ID and user message
    claude_session_id = "unknown"
    user_message = ""
    try:
        hook_input = json.loads(stdin_data) if stdin_data else {}
        claude_session_id = hook_input.get("session_id", str(os.getpid()))
        # Extract user message for antipattern matching (Enhancement 3)
        user_message = hook_input.get("prompt", "")
        log_debug(f"Session ID: {claude_session_id}, User message: {user_message[:50]}...")
    except Exception as e:
        log_debug(f"Error parsing input: {e}")
        claude_session_id = str(os.getpid())

    output_parts = []

    # Only inject identity on first prompt of this Claude session
    if is_first_prompt(claude_session_id):
        identity = load_identity_core()
        if identity:
            context = format_identity_context(identity)
            if context:
                output_parts.append(context)
                log_debug(f"Injecting identity context ({len(context)} chars)")
            else:
                log_debug("Failed to format identity context")
        else:
            log_debug("No identity cache loaded")

        # Git integration: Inject uncommitted changes as work-in-progress context
        # This is critical for "where did we leave off?" - uncommitted changes
        # are often the most relevant signal of current work
        git_status = get_git_status()
        if git_status.get("has_changes"):
            git_context = format_git_context(git_status)
            if git_context:
                output_parts.append(git_context)
                log_debug(f"Injecting git context: {len(git_status['modified_files'])} modified, {len(git_status['untracked_files'])} new")

            # Project re-activation: If a "completed" project has uncommitted changes,
            # it means new work has started - reactivate it
            if identity:
                reactivated = reactivate_project_if_needed(identity, git_status)
                if reactivated:
                    log_debug("Project reactivated due to uncommitted changes")

        # Phase 5: Anticipation Engine - Proactive suggestions
        # Only on first prompt with meaningful content
        if user_message and len(user_message) > 10:
            suggestions = load_suggestions(user_message, os.getcwd())
            if suggestions:
                suggestions_block = format_suggestions_block(suggestions)
                output_parts.append(suggestions_block)
                log_debug(f"Injecting suggestions ({len(suggestions)} items)")

        # Phase 1 Enhancement: Relevance-scored context
        # Load relevant past context based on current message and project
        if user_message and len(user_message) > 10:
            relevant_context = load_relevant_context(
                user_message=user_message,
                cwd=os.getcwd(),
                project=detect_current_project()
            )
            if relevant_context:
                context_block = format_relevant_context_block(relevant_context)
                output_parts.append(context_block)
                log_debug(f"Injecting relevant context ({len(relevant_context)} items)")

        # Phase 4 Enhancement: Validated patterns
        # Load and inject validated patterns (topics, problems, workflows)
        pattern_block = load_validated_patterns()
        if pattern_block:
            output_parts.append(pattern_block)
    else:
        log_debug("Not first prompt, skipping identity injection")

    # Enhancement 3: Check for relevant antipatterns on ALL prompts with problems
    if user_message and is_problem_prompt(user_message):
        antipatterns = load_relevant_antipatterns(user_message)
        if antipatterns:
            antipattern_block = format_antipattern_block(antipatterns)
            output_parts.append(antipattern_block)
            log_debug(f"Injecting antipattern warnings ({len(antipatterns)} items)")

    # Output all parts
    if output_parts:
        print("\n".join(output_parts))

    sys.exit(0)


if __name__ == "__main__":
    main()
