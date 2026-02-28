#!/usr/bin/env python3
"""
Distiller - Extracts compressed summaries and corrections from conversations.
Used by autosave_worker.py to update the identity core cache.

Operates with rule-based extraction (no LLM calls) for speed.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config import IDENTITY_CACHE, LOG_DIR, SUMMARIES_DIR
from content_filter import filter_messages_for_extraction, get_message_hash, is_system_generated_content

# Cache location (from config.py for cross-platform support)
DEBUG_LOG = LOG_DIR / "distiller_debug.log"

# Correction detection patterns (user correcting Claude)
CORRECTION_PATTERNS = [
    (r'\bno[,.]?\s+(?:that\'s|thats|it\'s|its)\s+(?:not|wrong)', "explicit_no"),
    (r'\bwrong\b', "wrong"),
    (r'\bdon\'t\s+do\s+that\b', "dont_do"),
    (r'\bactually[,.]?\s+(?:i|it|that)', "actually"),
    (r'\bi\s+meant\b', "i_meant"),
    (r'\bnot\s+what\s+i\s+(?:asked|wanted|meant)', "not_what_i"),
    (r'\bthat\'s\s+incorrect\b', "incorrect"),
    (r'\bno[,.]?\s+(?:i|we|you)\s+(?:should|need|want)', "no_should"),
    (r'\bstop\b', "stop"),
    (r'\bundo\b', "undo"),
]

# Topic extraction patterns
TOPIC_KEYWORDS = {
    "mcp": ["mcp", "model context protocol", "mcp server"],
    "memory": ["memory", "recall", "remember", "forget"],
    "automation": ["automate", "automation", "hook", "script"],
    "configuration": ["config", "settings", "setup"],
    "debugging": ["debug", "error", "fix", "bug", "issue"],
    "feature": ["add", "implement", "create", "build", "feature"],
}

# Decision and reasoning extraction patterns (Enhancement 4)
DECISION_PATTERNS = [
    # "decided to X because Y"
    (r'(?:i\s+)?decided\s+to\s+(.+?)\s+because\s+(.+?)(?:\.|$)', 'decided_because'),
    # "chose X over Y" / "chose X instead of Y"
    (r'chose\s+(.+?)\s+(?:over|instead\s+of)\s+(.+?)(?:\.|$)', 'chose_over'),
    # "instead of X, we/I Y"
    (r'instead\s+of\s+(.+?),\s+(?:we|i)\s+(.+?)(?:\.|$)', 'instead_of'),
    # "trade-off: X"
    (r'trade-?off[:\s]+(.+?)(?:\.|$)', 'tradeoff'),
    # "assuming X"
    (r'assuming\s+(.+?)(?:\.|$)', 'assumption'),
    # "the reason is X" / "reason: X"
    (r'(?:the\s+)?reason(?:\s+is)?[:\s]+(.+?)(?:\.|$)', 'reason'),
    # "went with X because Y"
    (r'went\s+with\s+(.+?)\s+because\s+(.+?)(?:\.|$)', 'went_with'),
    # "opted for X"
    (r'opted\s+for\s+(.+?)(?:\.|$)', 'opted_for'),
    # "this approach X rather than Y"
    (r'this\s+approach\s+(.+?)\s+rather\s+than\s+(.+?)(?:\.|$)', 'rather_than'),
]

# Project completion detection patterns
COMPLETION_PATTERNS = [
    (r'\ball\s+(?:done|finished|complete|implemented)\b', 0.9),
    (r'\b(?:it\'s|that\'s|we\'re|everything\'s)\s+(?:done|finished|complete|working)\b', 0.8),
    (r'\bfully\s+implemented\b', 0.9),
    (r'\beverything\s+(?:is\s+)?(?:working|done|complete)\b', 0.8),
    (r'\bshipped\b', 0.7),
    (r'\b(?:phase|step)\s+\d+.*complete.*final\b', 0.8),
    (r'\bwe\s+finished\b', 0.85),
    (r'\bimplementation\s+(?:is\s+)?complete\b', 0.9),
    (r'\bproject\s+(?:is\s+)?(?:done|complete|finished)\b', 0.9),
]

# User confirmation patterns (response to "is X done?")
CONFIRMATION_PATTERNS = [
    (r'^(?:yes|yep|yeah|yup|correct|right|exactly)[\s,\.!]*$', 0.7),
    (r'\byes\s+(?:it\'s|that\'s|we\'re)\s+done\b', 0.9),
    (r'\byep\s+(?:finished|done|complete)\b', 0.9),
]

PROJECT_STATUS_LOG = Path("/tmp/project_status.log")

# One-off patterns - sessions matching these are NOT substantial
ONEOFF_SESSION_PATTERNS = [
    r"where\s+did\s+we\s+leave\s+off",
    r"what(?:'s| is)\s+(?:my|the)",  # "what's my IP"
    r"^(?:hi|hello|hey)[\s,\.!]*$",  # Greetings only
    r"^(?:thanks|thank you|perfect|great)[\s,\.!]*$",  # Just thanks
    r"what\s+time\s+is\s+it",
    r"remind\s+me",
]

# Substantial work patterns - sessions matching these ARE substantial
SUBSTANTIAL_SESSION_PATTERNS = [
    r"implement|build|create|develop|write",
    r"fix|debug|resolve|troubleshoot",
    r"refactor|optimize|improve|enhance",
    r"add\s+(?:a\s+)?(?:feature|function|method|class)",
    r"configure|setup|install",
    r"error|bug|issue|problem",
    r"test|verify|validate",
]


def is_substantial_session(messages: List[Dict[str, str]]) -> bool:
    """
    Determine if a session represents substantial work that should
    update the last_session cache.

    A session is substantial if:
    1. Message count > 4 (not just a quick Q&A)
    2. Contains work indicators (implement, fix, debug, create, etc.)
    3. NOT primarily one-off patterns (what's my IP, where did we leave off, etc.)

    Returns True if substantial, False if trivial/one-off.
    """
    if not messages:
        return False

    # Check message count first - very short sessions are not substantial
    if len(messages) <= 4:
        return False

    # Build text from user messages (user intent matters most)
    user_text = ""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            user_text += " " + content

    user_text_lower = user_text.lower().strip()

    # Check if it's a one-off pattern (should NOT be substantial)
    for pattern in ONEOFF_SESSION_PATTERNS:
        if re.search(pattern, user_text_lower, re.IGNORECASE):
            # If the first user message is primarily a one-off query, not substantial
            first_user_msg = ""
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                    first_user_msg = content.lower()
                    break

            # If the first message is short and matches one-off pattern, not substantial
            if len(first_user_msg) < 100 and re.search(pattern, first_user_msg, re.IGNORECASE):
                log_debug(f"Session not substantial: matches one-off pattern '{pattern}'")
                return False

    # Check for work indicators
    work_matches = 0
    for pattern in SUBSTANTIAL_SESSION_PATTERNS:
        if re.search(pattern, user_text_lower, re.IGNORECASE):
            work_matches += 1

    # Need at least 1 work indicator to be substantial
    if work_matches >= 1:
        log_debug(f"Session is substantial: {work_matches} work indicators, {len(messages)} messages")
        return True

    # Default: if > 10 messages, consider substantial even without work indicators
    if len(messages) > 10:
        log_debug(f"Session is substantial: {len(messages)} messages (no specific work indicators)")
        return True

    log_debug(f"Session not substantial: {len(messages)} messages, {work_matches} work indicators")
    return False


def log_debug(msg: str):
    """Write debug info to log file."""
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except:
        pass


def log_project_status(msg: str):
    """Write project status changes to dedicated log."""
    try:
        with open(PROJECT_STATUS_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except:
        pass


def detect_project_completion(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Detect if any projects were marked as complete in the conversation.

    Returns list of projects with completion signals:
    [{"project_name": str, "confidence": float, "evidence": str}]
    """
    completed_projects = []

    # Load current projects from identity cache
    active_projects = []
    if IDENTITY_CACHE.exists():
        try:
            with open(IDENTITY_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)
                active_projects = [
                    p["name"] for p in cache.get("projects", [])
                    if p.get("status") == "active"
                ]
        except:
            pass

    if not active_projects:
        return []

    # Build conversation text
    all_text = " ".join(m.get("content", "") for m in messages).lower()

    # Check for completion signals
    for project_name in active_projects:
        project_lower = project_name.lower().replace("-", " ").replace("_", " ")

        # Check if project is mentioned
        project_mentioned = (
            project_lower in all_text or
            project_name.lower() in all_text
        )

        best_confidence = 0.0
        best_evidence = ""

        # Check completion patterns in conversation
        for msg in messages:
            content = msg.get("content", "")
            content_lower = content.lower()
            role = msg.get("role", "")

            for pattern, base_confidence in COMPLETION_PATTERNS:
                if re.search(pattern, content_lower):
                    # Boost confidence if project is mentioned nearby
                    confidence = base_confidence
                    if project_mentioned:
                        confidence = min(1.0, confidence + 0.1)

                    # Boost if user confirms
                    if role == "user":
                        confidence = min(1.0, confidence + 0.05)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_evidence = content[:100]

            # Check confirmation patterns (user saying "yes" after being asked)
            if role == "user":
                for pattern, base_confidence in CONFIRMATION_PATTERNS:
                    if re.search(pattern, content_lower):
                        # Only count if project was discussed in conversation
                        if project_mentioned:
                            confidence = base_confidence
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_evidence = f"User confirmed: {content[:50]}"

        # Only report if confidence is high enough
        if best_confidence >= 0.7:
            completed_projects.append({
                "project_name": project_name,
                "confidence": best_confidence,
                "evidence": best_evidence
            })
            log_project_status(
                f"DETECTED completion: {project_name} (conf={best_confidence:.2f}) - {best_evidence[:50]}"
            )

    return completed_projects


def generate_compressed_summary(messages: List[Dict[str, str]]) -> str:
    """
    Generate max 50-word summary of session.
    Format: "Accomplished: X. Created: Y. Topics: Z."

    IMPORTANT: Focus on what was ACCOMPLISHED, not what was asked.
    The first user message is often a trivial question, but the session
    may then proceed to do substantial work. Prioritize:
    1. Files created/modified
    2. Features implemented
    3. Bugs fixed
    4. Decisions made
    Rule-based extraction, no LLM.
    """
    if not messages:
        return ""

    summary_parts = []
    all_text = " ".join(m.get("content", "") for m in messages)

    # 1. HIGHEST PRIORITY: Files created (strongest signal of real work)
    files_created = []
    file_patterns = [
        r'Created\s+[`\"\']?(/[^\s`\"\']+|[^\s`\"\']+\.[a-z]+)[`\"\']?',
        r'wrote\s+(?:to\s+)?[`\"\']?(/[^\s`\"\']+|[^\s`\"\']+\.[a-z]+)[`\"\']?',
        r'saved\s+(?:to\s+)?[`\"\']?(/[^\s`\"\']+|[^\s`\"\']+\.[a-z]+)[`\"\']?',
    ]
    for pattern in file_patterns:
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        for m in matches:
            # Extract just the filename
            filename = m.split('/')[-1] if '/' in m else m
            if filename and filename not in files_created:
                files_created.append(filename)

    if files_created:
        summary_parts.append(f"Created: {', '.join(files_created[:3])}")

    # 2. HIGH PRIORITY: Look for completion indicators (from LATER messages, not first)
    accomplishments = []
    completion_patterns = [
        r'(?:Fixed|Implemented|Added|Built|Completed|Resolved|Finished)\s+([^\.!]{10,60})',
        r'(?:all\s+)?(?:\d+\s+)?(?:bugs?|issues?|errors?|tests?)\s+(?:fixed|passed|resolved)',
        r'(?:feature|functionality|system|module)\s+(?:is\s+)?(?:working|complete|ready)',
    ]

    # Look at later assistant messages (more likely to contain accomplishments)
    later_messages = messages[len(messages)//2:] if len(messages) > 4 else messages
    for msg in later_messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            for pattern in completion_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                accomplishments.extend(matches[:2])

    if accomplishments:
        acc_str = accomplishments[0] if isinstance(accomplishments[0], str) else str(accomplishments[0])
        summary_parts.append(f"Accomplished: {acc_str[:50]}")

    # 3. MEDIUM PRIORITY: Actions taken (prefer later actions over earlier)
    actions = []
    for msg in reversed(messages):  # Start from end
        if msg.get("role") == "assistant" and len(actions) < 3:
            content = msg.get("content", "")
            action_matches = re.findall(
                r'(?:I\'ve|I have|I\'ll|I will|Let me|I am going to|Done|Finished|Completed)\s+(\w+(?:\s+\w+){0,5})',
                content, re.IGNORECASE
            )
            actions.extend(action_matches[:2])

    if actions and not accomplishments:  # Only if no accomplishments found
        action_str = ", ".join(actions[:2])
        summary_parts.append(f"Did: {action_str[:40]}")

    # 4. LOW PRIORITY: Topics (always useful context)
    all_text_lower = all_text.lower()
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in all_text_lower for kw in keywords):
            topics.append(topic)

    if topics:
        summary_parts.append(f"Topics: {', '.join(topics[:4])}")

    # 5. FALLBACK: Only use first request if nothing else found
    if not summary_parts:
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")[:100]
                if len(content) > 15:
                    summary_parts.append(f"Discussed: {content.split('.')[0][:50]}")
                    break

    summary = ". ".join(summary_parts)

    # Enforce 50-word limit
    words = summary.split()
    if len(words) > 50:
        summary = " ".join(words[:50]) + "..."

    return summary if summary else "Session with no extractable summary"


def extract_corrections(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Find moments where Claude was corrected by the user.
    Returns list of correction dicts with context.

    Filters out system-generated content (session continuation, memory context, etc.)
    and uses content hashing for deduplication.
    """
    corrections = []
    seen_hashes = set()

    # Filter out system-generated messages before processing
    filtered_messages = filter_messages_for_extraction(messages)
    log_debug(f"Filtered {len(messages)} -> {len(filtered_messages)} messages (removed system content)")

    for i, msg in enumerate(filtered_messages):
        if msg.get("role") != "user":
            continue

        content = msg.get("content", "")

        # Skip if this is system-generated content that slipped through
        if is_system_generated_content(content):
            log_debug(f"Skipping system content: {content[:50]}...")
            continue

        content_lower = content.lower()

        for pattern, correction_type in CORRECTION_PATTERNS:
            if re.search(pattern, content_lower):
                # Generate content hash for deduplication
                content_hash = get_message_hash(content)

                # Skip if we've seen this exact content
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                # Found a potential correction
                correction = {
                    "type": correction_type,
                    "user_said": content[:200],
                    "message_index": i,
                    "timestamp": datetime.now().isoformat(),
                    "content_hash": content_hash,
                }

                # Get what Claude said before (context)
                if i > 0:
                    prev_msg = filtered_messages[i - 1]
                    if prev_msg.get("role") == "assistant":
                        correction["claude_said"] = prev_msg.get("content", "")[:200]

                # Get what Claude said after (the fix)
                if i + 1 < len(filtered_messages):
                    next_msg = filtered_messages[i + 1]
                    if next_msg.get("role") == "assistant":
                        correction["claude_corrected"] = next_msg.get("content", "")[:200]

                corrections.append(correction)
                break  # Only one correction per message

    log_debug(f"Extracted {len(corrections)} corrections (after filtering)")
    return corrections


def format_correction_for_cache(correction: Dict[str, Any]) -> Dict[str, str]:
    """
    Format a correction for the identity cache.
    Returns: {"level": "HIGH/MED/LOW", "text": "description"}
    """
    user_said = correction.get("user_said", "")
    correction_type = correction.get("type", "")

    # Determine severity
    level = "MED"
    if correction_type in ["explicit_no", "wrong", "stop", "undo"]:
        level = "HIGH"
    elif correction_type in ["actually", "i_meant"]:
        level = "MED"
    else:
        level = "LOW"

    # Create concise description
    text = user_said[:100]
    if len(user_said) > 100:
        text = text.rsplit(" ", 1)[0] + "..."

    return {"level": level, "text": text}


def extract_decisions_and_reasoning(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Extract decisions and reasoning from conversation.
    Enhancement 4: Capture WHY, not just WHAT.

    Returns list of decision dicts:
    [{"type": str, "decision": str, "reasoning": str, "context": str}]
    """
    decisions = []

    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", "")
        content_lower = content.lower()

        for pattern, decision_type in DECISION_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)

            for match in matches:
                decision_entry = {
                    "type": decision_type,
                    "timestamp": datetime.now().isoformat(),
                }

                # Handle different match formats
                if isinstance(match, tuple) and len(match) >= 2:
                    # Patterns with two capture groups (decision + reasoning)
                    decision_entry["decision"] = match[0].strip()[:100]
                    decision_entry["reasoning"] = match[1].strip()[:100]
                elif isinstance(match, tuple) and len(match) == 1:
                    decision_entry["decision"] = match[0].strip()[:100]
                    decision_entry["reasoning"] = ""
                else:
                    decision_entry["decision"] = str(match).strip()[:100]
                    decision_entry["reasoning"] = ""

                # Skip very short decisions (likely false positives)
                if len(decision_entry["decision"]) < 5:
                    continue

                decisions.append(decision_entry)

                # Limit to 5 decisions per message
                if len([d for d in decisions if d.get("type") == decision_type]) >= 5:
                    break

    # Deduplicate by decision text
    seen_decisions = set()
    unique_decisions = []
    for d in decisions:
        decision_key = d["decision"].lower()
        if decision_key not in seen_decisions:
            seen_decisions.add(decision_key)
            unique_decisions.append(d)

    log_debug(f"Extracted {len(unique_decisions)} decisions/reasoning from {len(messages)} messages")
    return unique_decisions[:10]  # Max 10 decisions per session


def _text_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.
    Returns a value between 0.0 (no overlap) and 1.0 (identical).
    """
    if not text1 or not text2:
        return 0.0

    # Tokenize by words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    if union == 0:
        return 0.0

    return intersection / union


def update_identity_cache(summary: str, corrections: List[Dict[str, Any]],
                         session_id: str = None,
                         completed_projects: List[Dict[str, Any]] = None,
                         decisions: List[Dict[str, Any]] = None,
                         messages: List[Dict[str, str]] = None) -> bool:
    """
    Update the Layer 1 identity cache with new session data.
    - Updates last_session with new summary (ONLY if session is substantial)
    - Adds new corrections (if HIGH importance, keeps max 5)
    - Updates project statuses if completions detected
    - Updates timestamp

    Uses content_hash AND text similarity for deduplication.

    The messages parameter is used to determine if the session is substantial.
    If messages are provided and the session is NOT substantial (e.g., "where did we leave off?"),
    the last_session cache will NOT be updated to preserve the previous meaningful work summary.
    """
    try:
        # Load existing cache
        cache = {}
        if IDENTITY_CACHE.exists():
            with open(IDENTITY_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)

        # Update project statuses if completions detected
        if completed_projects:
            projects = cache.get("projects", [])
            for completion in completed_projects:
                project_name = completion["project_name"]
                confidence = completion["confidence"]

                for project in projects:
                    if project["name"] == project_name and project.get("status") == "active":
                        # Update status to completed
                        project["status"] = "completed"
                        # Update context to reflect completion
                        old_context = project.get("context", "")
                        if "implementing" in old_context.lower():
                            project["context"] = old_context.replace("implementing", "completed")
                        elif "complete" not in old_context.lower():
                            project["context"] = f"{old_context} (complete)"

                        log_project_status(
                            f"UPDATED project status: {project_name} -> completed (conf={confidence:.2f})"
                        )

            cache["projects"] = projects

        # Update last session (ONLY if session is substantial)
        # This prevents trivial sessions like "where did we leave off?" from overwriting
        # meaningful work summaries
        if summary:
            should_update_last_session = True

            # If messages provided, check if session is substantial
            if messages is not None:
                if is_substantial_session(messages):
                    log_debug(f"Updating last_session (substantial session): {summary[:50]}...")
                else:
                    log_debug(f"NOT updating last_session (trivial session): {summary[:50]}...")
                    should_update_last_session = False

            if should_update_last_session:
                cache["last_session"] = summary
                # Track which session this summary is for (used by context_injector)
                if session_id:
                    cache["last_session_id"] = session_id

        # Process new corrections
        if corrections:
            existing_corrections = cache.get("corrections", [])
            existing_hashes = {c.get("content_hash") for c in existing_corrections if c.get("content_hash")}

            for corr in corrections:
                formatted = format_correction_for_cache(corr)
                content_hash = corr.get("content_hash", "")

                # Only keep HIGH corrections, limit to 5 total
                if formatted["level"] == "HIGH":
                    # Add content_hash to formatted correction
                    formatted["content_hash"] = content_hash

                    # Check for duplicates by hash first (exact match)
                    if content_hash and content_hash in existing_hashes:
                        log_debug(f"Skipping duplicate correction (hash match): {formatted['text'][:30]}...")
                        continue

                    # Check for duplicates by text similarity (> 0.8 = too similar)
                    is_similar = any(
                        _text_similarity(formatted["text"], existing["text"]) > 0.8
                        for existing in existing_corrections
                    )
                    if is_similar:
                        log_debug(f"Skipping duplicate correction (similar text): {formatted['text'][:30]}...")
                        continue

                    existing_corrections.insert(0, formatted)  # Newest first
                    if content_hash:
                        existing_hashes.add(content_hash)

            # Keep max 5 corrections
            cache["corrections"] = existing_corrections[:5]

        # Process new decisions (Enhancement 4)
        if decisions:
            existing_decisions = cache.get("recent_decisions", [])

            for decision in decisions:
                # Format decision for cache
                formatted_decision = {
                    "decision": decision.get("decision", "")[:80],
                    "reasoning": decision.get("reasoning", "")[:80],
                    "type": decision.get("type", "unknown"),
                    "timestamp": decision.get("timestamp", datetime.now().isoformat()),
                }

                # Skip if similar decision already exists
                is_similar = any(
                    _text_similarity(formatted_decision["decision"], existing["decision"]) > 0.7
                    for existing in existing_decisions
                )
                if not is_similar:
                    existing_decisions.insert(0, formatted_decision)  # Newest first

            # Keep max 5 recent decisions
            cache["recent_decisions"] = existing_decisions[:5]
            log_debug(f"Updated recent_decisions: {len(cache['recent_decisions'])} entries")

        # Update timestamp
        cache["updated_at"] = datetime.now().isoformat()

        # Estimate token count (rough: chars / 4)
        cache_str = json.dumps(cache)
        cache["token_count"] = len(cache_str) // 4

        # Ensure cache directory exists
        IDENTITY_CACHE.parent.mkdir(parents=True, exist_ok=True)

        # Save cache
        with open(IDENTITY_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

        log_debug(f"Updated identity cache (tokens: ~{cache['token_count']})")
        return True

    except Exception as e:
        log_debug(f"Error updating identity cache: {e}")
        return False


def save_session_summary(session_id: str, summary: str, corrections: List[Dict],
                        message_count: int) -> bool:
    """
    Save detailed session summary to separate file (for decay pipeline).
    """
    try:
        SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

        # Use date-based filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        summary_file = SUMMARIES_DIR / f"{date_str}.json"

        # Load existing summaries for today
        summaries = []
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                summaries = json.load(f)

        # Add new summary
        summaries.append({
            "session_id": session_id,
            "summary": summary,
            "corrections_count": len(corrections),
            "message_count": message_count,
            "timestamp": datetime.now().isoformat(),
            "word_count": len(summary.split())
        })

        # Save
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

        log_debug(f"Saved session summary for {session_id}")
        return True

    except Exception as e:
        log_debug(f"Error saving session summary: {e}")
        return False


def distill_session(messages: List[Dict[str, str]], session_id: str) -> Dict[str, Any]:
    """
    Main entry point: Generate compressed summary and extract corrections.
    Called by autosave_worker after saving.

    Returns:
        {
            "summary": str,
            "corrections": List[Dict],
            "completed_projects": List[Dict],
            "cache_updated": bool
        }
    """
    log_debug(f"Distilling session {session_id} ({len(messages)} messages)")

    # Generate summary
    summary = generate_compressed_summary(messages)

    # Extract corrections
    corrections = extract_corrections(messages)

    # Extract decisions and reasoning (Enhancement 4)
    decisions = extract_decisions_and_reasoning(messages)
    if decisions:
        log_debug(f"Extracted {len(decisions)} decisions/reasoning")

    # Detect project completions
    completed_projects = detect_project_completion(messages)
    if completed_projects:
        log_debug(f"Detected {len(completed_projects)} project completion(s)")

    # Update identity cache (including project status updates and decisions)
    # Pass messages so we can check if this is a substantial session before updating last_session
    cache_updated = update_identity_cache(summary, corrections, session_id, completed_projects, decisions, messages)

    # Save session summary for decay pipeline
    save_session_summary(session_id, summary, corrections, len(messages))

    result = {
        "summary": summary,
        "corrections": corrections,
        "decisions": decisions,  # Enhancement 4
        "completed_projects": completed_projects,
        "cache_updated": cache_updated
    }

    log_debug(f"Distillation complete: {len(summary)} char summary, {len(corrections)} corrections, "
              f"{len(decisions)} decisions, {len(completed_projects)} completions")
    return result


def compress_old_summaries(days_old: int = 7) -> int:
    """
    Compress summaries older than N days.
    Used by decay pipeline.

    Returns number of summaries compressed.
    """
    if not SUMMARIES_DIR.exists():
        return 0

    compressed_count = 0
    cutoff = datetime.now().timestamp() - (days_old * 24 * 60 * 60)

    for summary_file in SUMMARIES_DIR.glob("*.json"):
        try:
            # Check file age
            if summary_file.stat().st_mtime > cutoff:
                continue

            # Load and compress
            with open(summary_file, "r", encoding="utf-8") as f:
                summaries = json.load(f)

            for s in summaries:
                if s.get("word_count", 100) > 20:
                    # Compress summary to 20 words max
                    words = s.get("summary", "").split()
                    s["summary"] = " ".join(words[:20])
                    s["word_count"] = min(20, len(words))
                    s["compressed"] = True
                    compressed_count += 1

            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)

        except Exception as e:
            log_debug(f"Error compressing {summary_file}: {e}")

    log_debug(f"Compressed {compressed_count} old summaries")
    return compressed_count


if __name__ == "__main__":
    # Test with sample messages
    import sys

    if len(sys.argv) > 1:
        # Load conversation file
        conv_file = Path(sys.argv[1])
        if conv_file.exists():
            with open(conv_file, "r", encoding="utf-8") as f:
                conv = json.load(f)
            messages = conv.get("messages", [])
            session_id = conv_file.stem
        else:
            print(f"File not found: {conv_file}")
            sys.exit(1)
    else:
        # Sample test
        messages = [
            {"role": "user", "content": "Help me implement a memory system for Claude"},
            {"role": "assistant", "content": "I'll help you create a memory system. Let me start by creating the core module."},
            {"role": "user", "content": "No, that's wrong. I want it to use MCP protocol."},
            {"role": "assistant", "content": "I understand, let me implement it using MCP protocol instead."},
            {"role": "user", "content": "Thanks, that looks better."},
        ]
        session_id = "test-session"

    result = distill_session(messages, session_id)

    print(f"Summary: {result['summary']}")
    print(f"Corrections: {len(result['corrections'])}")
    for c in result['corrections']:
        print(f"  - [{c['type']}] {c['user_said'][:50]}...")
    print(f"Cache updated: {result['cache_updated']}")
