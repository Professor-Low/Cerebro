"""
Helper function for integrating corrections into conversation flow.

MULTI-AGENT NOTICE: Agent 4 and Agent 6 will use this.
Part of the NAS Cerebral Interface - Learning from Corrections System
"""

from corrections_detector import CorrectionDetector
from corrections_tracker import CorrectionsTracker


def process_conversation_for_corrections(messages: list, conversation_id: str) -> list:
    """
    Process conversation messages to detect and save corrections.

    Args:
        messages: List of {role, content} dicts
        conversation_id: ID of the conversation

    Returns:
        List of detected corrections
    """
    detector = CorrectionDetector()
    tracker = CorrectionsTracker()

    corrections_found = []

    for i in range(1, len(messages)):
        current = messages[i]
        previous = messages[i-1] if i > 0 else None

        if current["role"] == "user" and previous and previous["role"] == "assistant":
            # Check if user is correcting assistant
            detection = detector.detect_correction(
                current["content"],
                previous["content"]
            )

            if detection:
                # Extract entities
                entities = detector.extract_entities(current["content"])

                # Save correction
                correction_id = tracker.save_correction(
                    mistake=detection.get("mistake_text", ""),
                    correction=detection.get("correction_text", ""),
                    topic=detection["topic"],
                    conversation_id=conversation_id,
                    context=previous["content"][:500],
                    importance=detection["importance"],
                    entities=entities,
                    user_message=current["content"]
                )

                corrections_found.append({
                    "id": correction_id,
                    "topic": detection["topic"],
                    "importance": detection["importance"]
                })

    return corrections_found


def get_relevant_corrections(prompt: str, topic: str = None) -> list:
    """
    Get corrections relevant to current prompt.

    Args:
        prompt: User's current prompt
        topic: Optional topic filter

    Returns:
        List of relevant corrections
    """
    tracker = CorrectionsTracker()

    # Get corrections by topic if specified
    if topic:
        return tracker.get_corrections_by_topic(topic, limit=5)

    # Otherwise search by keywords in prompt
    return tracker.search_corrections(prompt, limit=5)
