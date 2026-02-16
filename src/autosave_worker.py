#!/usr/bin/env python3
"""
Autosave Worker - Background worker that parses transcripts and saves to AI Memory.
Runs asynchronously from the hook to avoid blocking Claude's response.

Environment variables (set by autosave_hook.py):
- AUTOSAVE_SESSION_ID: Claude's session ID
- AUTOSAVE_TRANSCRIPT_PATH: Path to JSONL transcript file
- AUTOSAVE_HOOK_EVENT: "Stop" or "SessionEnd"
"""

import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress all warnings and redirect stderr to avoid any output
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, 'w')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import CONVERSATIONS_DIR, LOG_DIR, TEMP_DIR

# Configuration (cross-platform)
LOG_FILE = LOG_DIR / "autosave_worker.log"
STATE_FILE = TEMP_DIR / "autosave_state.json"

# Completion detection patterns
COMPLETION_USER_PATTERNS = [
    r'\b(thanks|thank you|bye|goodbye|that\'s all|done|perfect|great)\b',
    r'\b(cheers|appreciate it|got it|all set)\b',
]
COMPLETION_ASSISTANT_PATTERNS = [
    r'\b(let me know if|anything else|feel free to|happy to help)\b',
    r'\b(is there anything|need anything else|have any other)\b',
]

# Meta-conversation detection keywords (Enhancement 2)
META_SYSTEM_KEYWORDS = [
    'memory', 'mcp', 'distiller', 'embedding', 'hook', 'brain',
    'ai-memory', 'autosave', 'identity_core', 'startup_context',
    'vector index', 'faiss', 'semantic search', 'conversation save'
]


def log(msg: str):
    """Write debug info to log file."""
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except:
        pass


def load_state() -> Dict[str, Any]:
    """Load autosave state (tracks saves per session)."""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        log(f"Error loading state: {e}")
    return {}


def save_state(state: Dict[str, Any]):
    """Save autosave state."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log(f"Error saving state: {e}")


def parse_transcript(transcript_path: str) -> List[Dict[str, str]]:
    """
    Parse Claude Code's JSONL transcript into messages.

    The transcript contains various event types. We extract:
    - User messages (type: "user")
    - Assistant messages (type: "assistant")
    """
    messages = []
    transcript_file = Path(transcript_path)

    if not transcript_file.exists():
        log(f"Transcript file not found: {transcript_path}")
        return messages

    try:
        with open(transcript_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    msg_type = entry.get("type", "")

                    if msg_type == "user":
                        # User message
                        content = extract_content(entry.get("message", {}))
                        if content:
                            messages.append({
                                "role": "user",
                                "content": content
                            })

                    elif msg_type == "assistant":
                        # Assistant message
                        content = extract_content(entry.get("message", {}))
                        if content:
                            messages.append({
                                "role": "assistant",
                                "content": content
                            })

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        log(f"Error parsing transcript: {e}")

    log(f"Parsed {len(messages)} messages from transcript")
    return messages


def extract_content(message: Dict) -> str:
    """
    Extract text content from a message object.

    Claude Code messages have content as either:
    - A string directly
    - A list of content blocks (each with type: "text", "tool_use", etc.)
    """
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        # Extract text from content blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                # NOTE: Skip tool_result blocks entirely - they pollute summaries
                # and aren't useful for continuation context. The actual tool
                # outputs are available in the transcript if needed.
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts).strip()

    return ""


def detect_completion(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Detect if the conversation appears complete.

    Returns:
        {
            "likely_complete": bool,
            "reason": str,
            "confidence": float
        }
    """
    if not messages:
        return {"likely_complete": False, "reason": "no_messages", "confidence": 0.0}

    # Check last few messages for completion signals
    last_messages = messages[-3:]  # Last 3 messages

    for msg in reversed(last_messages):
        content = msg.get("content", "").lower()
        role = msg.get("role", "")

        if role == "user":
            for pattern in COMPLETION_USER_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    return {
                        "likely_complete": True,
                        "reason": "user_closing",
                        "confidence": 0.8
                    }

        elif role == "assistant":
            for pattern in COMPLETION_ASSISTANT_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    return {
                        "likely_complete": True,
                        "reason": "assistant_offering_more",
                        "confidence": 0.6
                    }

    return {"likely_complete": False, "reason": "ongoing", "confidence": 0.3}


def detect_conversation_type(messages: List[Dict[str, str]]) -> str:
    """
    Detect if this is a meta-system conversation or user work.

    Returns:
        "meta_system" - Discussion about the AI memory system itself
        "user_work" - Regular user work (default)
    """
    if not messages:
        return "user_work"

    # Combine all message content (first 5 messages are usually most indicative)
    sample_messages = messages[:5]
    all_text = " ".join(m.get("content", "").lower() for m in sample_messages)

    # Count meta-system keyword matches
    meta_matches = sum(1 for kw in META_SYSTEM_KEYWORDS if kw.lower() in all_text)

    # Threshold: 3+ meta keywords = meta_system conversation
    if meta_matches >= 3:
        return "meta_system"

    return "user_work"


def do_save(session_id: str, messages: List[Dict[str, str]], hook_event: str,
            is_incremental: bool, save_number: int, completion_status: Dict,
            last_message_count: int) -> Optional[str]:
    """
    Save conversation - single file per session, append new messages.

    Optimized for Claude's retrieval:
    - One file = full context in one read
    - Only extracts from new messages
    - Merges extractions with existing

    Returns conversation ID on success, None on failure.
    """
    try:

        from ai_memory_ultimate import UltimateMemoryService

        memory = UltimateMemoryService()
        conv_file = memory.conversations_path / f"{session_id}.json"

        # Build metadata for autosave
        conversation_type = detect_conversation_type(messages)
        metadata = {
            "autosave": True,
            "hook_event": hook_event,
            "autosave_number": save_number,
            "completion_status": completion_status,
            "saved_at": datetime.now().isoformat(),
            "conversation_type": conversation_type,  # Enhancement 2: meta_system or user_work
        }

        # Mark as complete if SessionEnd hook
        if hook_event == "SessionEnd":
            metadata["session_complete"] = True
            completion_status["likely_complete"] = True
            completion_status["reason"] = "session_end_hook"
            completion_status["confidence"] = 1.0

        if is_incremental and conv_file.exists():
            # APPEND MODE: Load existing, add new messages, merge extractions
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)

                # Get only new messages
                new_messages = messages[last_message_count:]

                # Append new messages to existing
                existing['messages'].extend(new_messages)

                # Update metadata
                existing['metadata']['autosave_number'] = save_number
                existing['metadata']['completion_status'] = completion_status
                existing['metadata']['saved_at'] = datetime.now().isoformat()
                existing['metadata']['message_count'] = len(existing['messages'])

                if hook_event == "SessionEnd":
                    existing['metadata']['session_complete'] = True

                # Save back to same file
                with open(conv_file, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, indent=2, ensure_ascii=False)

                log(f"Appended {len(new_messages)} messages to {session_id} (total: {len(existing['messages'])})")
                return session_id

            except Exception as e:
                log(f"Append failed, falling back to full save: {e}")
                # Fall through to full save

        # FULL SAVE: First save or fallback
        conv_id = memory.save_conversation(
            messages=messages,
            session_id=session_id,
            metadata=metadata,
            incremental=False  # Always use session_id as filename
        )

        log(f"Saved conversation: {conv_id} ({len(messages)} messages, save #{save_number})")
        return conv_id

    except Exception as e:
        log(f"Error saving conversation: {e}")
        import traceback
        log(traceback.format_exc())
        return None


def main():
    """Main worker entry point."""
    log("Worker started")

    # Get environment variables from hook
    session_id = os.environ.get("AUTOSAVE_SESSION_ID", "")
    transcript_path = os.environ.get("AUTOSAVE_TRANSCRIPT_PATH", "")
    hook_event = os.environ.get("AUTOSAVE_HOOK_EVENT", "Stop")

    if not session_id or not transcript_path:
        log("Missing required environment variables")
        return

    log(f"Processing session {session_id}, event {hook_event}")

    # Parse transcript
    messages = parse_transcript(transcript_path)
    if not messages:
        log("No messages to save")
        return

    # Load state to check for incremental save
    state = load_state()
    session_state = state.get(session_id, {
        "message_count": 0,
        "save_count": 0,
        "first_save_at": None,
        "last_save_at": None
    })

    last_message_count = session_state.get("message_count", 0)
    current_message_count = len(messages)

    # Check if there are new messages to save
    if current_message_count <= last_message_count and hook_event != "SessionEnd":
        log(f"No new messages (current={current_message_count}, last={last_message_count})")
        return

    # Determine if incremental
    is_incremental = session_state.get("save_count", 0) > 0
    save_number = session_state.get("save_count", 0) + 1

    # Detect completion
    completion_status = detect_completion(messages)

    # Do the save
    conv_id = do_save(
        session_id=session_id,
        messages=messages,
        hook_event=hook_event,
        is_incremental=is_incremental,
        save_number=save_number,
        completion_status=completion_status,
        last_message_count=last_message_count
    )

    if conv_id:
        # Update state
        session_state["message_count"] = current_message_count
        session_state["save_count"] = save_number
        session_state["last_save_at"] = datetime.now().isoformat()
        if not session_state.get("first_save_at"):
            session_state["first_save_at"] = datetime.now().isoformat()
        session_state["last_conv_id"] = conv_id

        state[session_id] = session_state

        # Clean old sessions from state (older than 24 hours)
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        state = {k: v for k, v in state.items()
                 if v.get("last_save_at", "") > cutoff or k == session_id}

        save_state(state)
        log(f"State updated: {save_number} saves for session {session_id}")

        # Run distillation on SessionEnd to update identity cache
        if hook_event == "SessionEnd":
            try:
                from distiller import distill_session
                distill_result = distill_session(messages, session_id)
                log(f"Distillation complete: {len(distill_result['summary'])} char summary, "
                    f"{len(distill_result['corrections'])} corrections")
            except Exception as e:
                log(f"Distillation error (non-fatal): {e}")

            # Enhancement 1: Generate embeddings for searchability
            try:
                # Check if embeddings are enabled (os already imported at module level)
                if os.environ.get('ENABLE_EMBEDDINGS', '0') == '1':

                    from ai_embeddings_engine import EmbeddingsEngine

                    # Load the saved conversation
                    conv_path = CONVERSATIONS_DIR / f"{session_id}.json"
                    if conv_path.exists():
                        with open(conv_path, 'r', encoding='utf-8') as f:
                            conversation = json.load(f)
                        conversation['id'] = session_id

                        # Process conversation to generate embeddings
                        engine = EmbeddingsEngine()
                        engine.process_conversation(conversation)
                        log(f"Generated embeddings for {session_id}")
                    else:
                        log(f"Conversation file not found for embedding: {conv_path}")
                else:
                    log("Embeddings disabled (set ENABLE_EMBEDDINGS=1 to enable)")
            except Exception as e:
                log(f"Embedding generation error (non-fatal): {e}")

            # Auto-trigger learning extraction on SessionEnd
            try:
                from learning_extractor import LearningExtractor
                from solution_tracker import SolutionTracker

                extractor = LearningExtractor()
                tracker = SolutionTracker()

                # Build conversation dict for analysis
                conversation = {
                    "id": session_id,
                    "messages": messages,
                    "timestamp": datetime.now().isoformat()
                }

                learnings = extractor.analyze_conversation(conversation)

                # Connect to solution tracker (top 3 problem/solution pairs)
                problems = learnings.get("problems_found", [])
                solutions = learnings.get("solutions_found", [])

                solutions_recorded = 0
                for problem, solution in zip(problems[:3], solutions[:3]):
                    problem_text = problem.get("text", "") if isinstance(problem, dict) else str(problem)
                    solution_text = solution.get("text", "") if isinstance(solution, dict) else str(solution)

                    if problem_text and solution_text:
                        # Extract tags from content
                        tags = []
                        if "error" in problem_text.lower():
                            tags.append("error")
                        if "config" in problem_text.lower():
                            tags.append("configuration")
                        if "path" in problem_text.lower() or "file" in problem_text.lower():
                            tags.append("file_system")

                        tracker.record_solution(
                            problem=problem_text,
                            solution=solution_text,
                            conversation_id=session_id,
                            tags=tags
                        )
                        solutions_recorded += 1

                # Save learnings file if we found anything
                if learnings.get("learnings") or learnings.get("problems_found"):
                    extractor.save_learnings(learnings)
                    log(f"Learning extraction: {len(learnings.get('learnings', []))} learnings, "
                        f"{solutions_recorded} solutions recorded")

            except Exception as e:
                log(f"Learning extraction error (non-fatal): {e}")

            # Run decay pipeline check (daily, non-blocking)
            try:
                from decay_pipeline import run_decay
                decay_result = run_decay(force=False)
                if not decay_result.get("skipped"):
                    log(f"Decay pipeline: {decay_result.get('compressed', 0)} compressed, "
                        f"{decay_result.get('archived', 0)} archived")
            except Exception as e:
                log(f"Decay error (non-fatal): {e}")

            # Phase 7: Record session metrics
            try:
                from reflection.performance_tracker import PerformanceTracker
                tracker = PerformanceTracker()

                # Record task completion (session completed vs abandoned)
                if completion_status.get("likely_complete"):
                    tracker.record_metric(
                        metric_name="task_completion",
                        value=100.0,  # Completed
                        context=f"Session {session_id} completed: {completion_status.get('reason')}",
                        session_id=session_id
                    )
                else:
                    tracker.record_metric(
                        metric_name="task_completion",
                        value=50.0,  # Potentially incomplete
                        context=f"Session {session_id} ended without clear completion",
                        session_id=session_id
                    )

                # Record message count as proxy for session depth
                tracker.record_metric(
                    metric_name="session_message_count",
                    value=float(current_message_count),
                    context=f"Session {session_id}",
                    session_id=session_id
                )

                log(f"Recorded session metrics for {session_id}")

            except Exception as e:
                log(f"Metrics recording error (non-fatal): {e}")

            # Phase 3: Auto-update project states based on conversation
            try:
                from project_auto_updater import ProjectAutoUpdater

                auto_updater = ProjectAutoUpdater()
                update_result = auto_updater.process_conversation(
                    messages=messages,
                    session_id=session_id,
                    cwd=os.environ.get("PWD", None)  # CWD from Claude Code
                )

                if update_result.get("updated_projects"):
                    projects_updated = [p["project"] for p in update_result["updated_projects"]]
                    log(f"Project auto-update: {len(projects_updated)} projects updated "
                        f"({', '.join(projects_updated[:3])})")

                    # Record any status transitions
                    transitions = update_result.get("status_transitions", [])
                    if transitions:
                        log(f"Status transitions: {len(transitions)} projects changed status")

            except Exception as e:
                log(f"Project auto-update error (non-fatal): {e}")

            # Phase 6: Process corrections for personality evolution
            try:
                from corrections_tracker import CorrectionsTracker
                from personality.evolution_engine import PersonalityEvolutionEngine

                evolution_engine = PersonalityEvolutionEngine()
                corrections_tracker = CorrectionsTracker()

                # Get recent corrections from this session
                recent_corrections = corrections_tracker.get_recent_corrections(days=1, limit=10)

                if recent_corrections:
                    # Process corrections for trait evolution
                    result = evolution_engine.process_corrections_batch(recent_corrections)
                    if result.get("total_changes", 0) > 0:
                        log(f"Personality evolution: {result['total_changes']} trait changes from "
                            f"{result['corrections_processed']} corrections")

                        # Sync traits to profile periodically
                        sync_result = evolution_engine.synchronize_with_profile()
                        if sync_result.get("synced"):
                            log(f"Synced {sync_result.get('traits_synced', 0)} traits to profile")

            except Exception as e:
                log(f"Personality evolution error (non-fatal): {e}")

    log("Worker finished")


if __name__ == "__main__":
    main()
