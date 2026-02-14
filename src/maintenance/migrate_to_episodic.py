"""
Migrate conversations to episodic memory format.
Creates episodic memories from existing conversations for consolidation.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def extract_episode_from_conversation(conv: Dict) -> Optional[Dict]:
    """Extract an episodic memory from a conversation."""
    messages = conv.get("messages", [])
    if not messages:
        return None

    # Get first user message as the "event" (what was happening)
    first_user_msg = None
    for msg in messages:
        if msg.get("role") == "user":
            first_user_msg = msg.get("content", "")
            break

    if not first_user_msg:
        return None

    # Get last assistant message as the "outcome"
    last_assistant_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant_msg = msg.get("content", "")
            break

    # Truncate long content
    event = first_user_msg[:500] if len(first_user_msg) > 500 else first_user_msg
    outcome = last_assistant_msg[:500] if last_assistant_msg and len(last_assistant_msg) > 500 else (last_assistant_msg or "")

    # Detect emotional state from content
    emotional_state = detect_emotion(first_user_msg)

    # Create episode
    conv_id = conv.get("id", "")
    timestamp = conv.get("timestamp", datetime.now().isoformat())

    episode_id = f"ep_{timestamp[:10].replace('-', '')}_{hashlib.sha256(conv_id.encode()).hexdigest()[:6]}"

    episode = {
        "id": episode_id,
        "conversation_id": conv_id,
        "event": event,
        "outcome": outcome,
        "actors": ["user", "claude"],
        "emotional_state": emotional_state,
        "timestamp": timestamp,
        "date": timestamp[:10],
        "source": "migration"
    }

    return episode


def detect_emotion(text: str) -> str:
    """Simple emotion detection from text content."""
    text_lower = text.lower()

    # Frustration indicators
    frustration_words = ["not working", "broken", "fail", "error", "issue", "problem", "bug", "wrong", "help", "stuck"]
    if any(word in text_lower for word in frustration_words):
        return "frustrated"

    # Success indicators
    success_words = ["works", "working", "great", "awesome", "perfect", "thanks", "fixed", "solved"]
    if any(word in text_lower for word in success_words):
        return "satisfied"

    # Question indicators
    if "?" in text or text_lower.startswith(("how", "what", "why", "where", "when", "can")):
        return "curious"

    return "neutral"


def migrate_conversations_to_episodic(
    conversations_path: str = None,
    episodic_path: str = None,
    batch_size: int = 100,
    dry_run: bool = False
) -> Dict:
    """
    Migrate conversations to episodic memory format.

    Args:
        conversations_path: Path to conversations folder
        episodic_path: Path to episodic folder
        batch_size: How many to process at once
        dry_run: If True, don't write files

    Returns:
        Statistics about the migration
    """
    if conversations_path is None:
        from config import DATA_DIR
        conversations_path = str(DATA_DIR / "conversations")
    if episodic_path is None:
        from config import DATA_DIR
        episodic_path = str(DATA_DIR / "episodic")

    conv_path = Path(conversations_path)
    ep_path = Path(episodic_path)

    if not dry_run:
        ep_path.mkdir(parents=True, exist_ok=True)

    # Get existing episodic IDs to avoid duplicates
    existing_conv_ids = set()
    for ep_file in ep_path.glob("ep_*.json"):
        try:
            with open(ep_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get("conversation_id"):
                existing_conv_ids.add(data["conversation_id"])
        except:
            continue

    # Process conversations
    stats = {
        "total_conversations": 0,
        "already_migrated": 0,
        "migrated": 0,
        "skipped": 0,
        "errors": []
    }

    conv_files = list(conv_path.glob("*.json"))
    stats["total_conversations"] = len(conv_files)

    for conv_file in conv_files:
        try:
            with open(conv_file, 'r', encoding='utf-8') as f:
                conv = json.load(f)

            conv_id = conv.get("id", "")

            # Skip if already migrated
            if conv_id in existing_conv_ids:
                stats["already_migrated"] += 1
                continue

            # Extract episode
            episode = extract_episode_from_conversation(conv)

            if not episode:
                stats["skipped"] += 1
                continue

            # Save episode
            if not dry_run:
                ep_file = ep_path / f"{episode['id']}.json"
                with open(ep_file, 'w', encoding='utf-8') as f:
                    json.dump(episode, f, indent=2)

            stats["migrated"] += 1

        except Exception as e:
            stats["errors"].append(f"{conv_file.name}: {str(e)}")
            continue

    # Update index
    if not dry_run and stats["migrated"] > 0:
        update_episodic_index(ep_path)

    return stats


def update_episodic_index(episodic_path: Path):
    """Update the episodic memory index."""
    index = {
        "total_episodes": 0,
        "by_date": {},
        "by_emotion": {},
        "last_updated": datetime.now().isoformat()
    }

    for ep_file in episodic_path.glob("ep_*.json"):
        try:
            with open(ep_file, 'r', encoding='utf-8') as f:
                ep = json.load(f)

            index["total_episodes"] += 1

            # Index by date
            date = ep.get("date", "unknown")
            if date not in index["by_date"]:
                index["by_date"][date] = []
            index["by_date"][date].append(ep.get("id"))

            # Index by emotion
            emotion = ep.get("emotional_state", "neutral")
            if emotion not in index["by_emotion"]:
                index["by_emotion"][emotion] = 0
            index["by_emotion"][emotion] += 1

        except:
            continue

    index_file = episodic_path / "_index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    import sys

    dry_run = "--dry-run" in sys.argv

    print("Migrating conversations to episodic memory format...")
    if dry_run:
        print("(DRY RUN - no files will be written)")

    stats = migrate_conversations_to_episodic(dry_run=dry_run)

    print("\nMigration complete:")
    print(f"  Total conversations: {stats['total_conversations']}")
    print(f"  Already migrated: {stats['already_migrated']}")
    print(f"  Newly migrated: {stats['migrated']}")
    print(f"  Skipped: {stats['skipped']}")

    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
        for err in stats["errors"][:5]:
            print(f"    - {err}")
