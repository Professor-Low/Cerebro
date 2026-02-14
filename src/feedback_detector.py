"""
Feedback Detector - Auto-detect success/failure signals from conversations.

Part of Phase 7: Feedback Loops in the Brain Evolution Plan.

Detects:
- Success signals: "It worked!", "Perfect!", "That fixed it!", etc.
- Failure signals: "Still broken", "That didn't work", "Same error", etc.
- Links signals to most recent solution/suggestion
- Auto-records to solution_tracker for learning
"""

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FeedbackSignal:
    """A detected feedback signal from user message."""
    signal_type: str  # 'success' or 'failure'
    confidence: float  # 0.0 - 1.0
    matched_pattern: str
    user_message: str
    timestamp: str
    context_before: Optional[str] = None  # Assistant message before
    solution_id: Optional[str] = None  # Linked solution if found


class FeedbackDetector:
    """
    Detect success/failure feedback in user messages.

    This is the core of the feedback loop - when users express
    satisfaction or frustration, we capture it and link it to
    the solution that was just provided.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            from config import AI_MEMORY_BASE
            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.feedback_path = self.base_path / "feedback"
        self.feedback_path.mkdir(parents=True, exist_ok=True)

        # Log file for all detected feedback
        self.feedback_log = self.feedback_path / "feedback_log.jsonl"

        # Success patterns with confidence scores
        self.success_patterns = [
            # High confidence (0.95)
            (r"\bit\s+work(?:s|ed|ing)!?\b", 0.95, "it worked"),
            (r"\bthat\s+work(?:s|ed|ing)!?\b", 0.95, "that worked"),
            (r"\bthat\s+fix(?:es|ed)\s+it\b", 0.95, "that fixed it"),
            (r"\bperfect!+\b", 0.95, "perfect!"),
            (r"\bfinally!+\b", 0.90, "finally!"),
            (r"\byes!{1,}", 0.90, "yes!"),  # yes! with any exclamation marks
            (r"\bwe\s+did\s+it\b", 0.90, "we did it"),
            (r"\boh\s+my\s+god\b.*\bwork", 0.90, "omg it works"),
            (r"\bfinally\s+work", 0.90, "finally working"),  # Added for "Finally working!"

            # Medium-high confidence (0.85)
            (r"\bawesome!?\b", 0.85, "awesome"),
            (r"\bgreat!?\b", 0.80, "great"),
            (r"\bnice!?\b", 0.75, "nice"),
            (r"\bthanks!?\b.*\bwork", 0.85, "thanks + work"),
            (r"\bwork.*\bthanks!?\b", 0.85, "work + thanks"),
            (r"\bproblem\s+(?:is\s+)?(?:solved|fixed|resolved)\b", 0.90, "problem solved"),
            (r"\bissue\s+(?:is\s+)?(?:solved|fixed|resolved)\b", 0.90, "issue solved"),
            (r"\berror\s+(?:is\s+)?gone\b", 0.90, "error gone"),
            (r"\bno\s+(?:more\s+)?errors?\b", 0.85, "no more errors"),

            # Medium confidence (0.70)
            (r"\bnow\s+it\s+works\b", 0.80, "now it works"),
            (r"\bit'?s\s+working\s+now\b", 0.80, "it's working now"),
            (r"\bthat\s+did\s+(?:it|the\s+trick)\b", 0.85, "that did it"),
            (r"\bgot\s+it\s+working\b", 0.80, "got it working"),
            (r"\bruns?\s+(?:fine|correctly|properly)\b", 0.75, "runs fine"),
            (r"\bexactly\s+what\s+I\s+(?:needed|wanted)\b", 0.80, "exactly what I needed"),
            (r"\bwork(?:s|ed|ing)!+\b", 0.85, "working!"),  # "working!" with exclamation
        ]

        # Failure patterns with confidence scores
        self.failure_patterns = [
            # High confidence (0.95)
            (r"\bstill\s+(?:getting|seeing|having)\s+(?:the\s+)?(?:same\s+)?error\b", 0.95, "still getting error"),
            (r"\bstill\s+(?:broken|failing|doesn't\s+work)\b", 0.95, "still broken"),
            (r"\bsame\s+(?:error|issue|problem)\b", 0.95, "same error"),
            (r"\bthat\s+didn'?t\s+work\b", 0.95, "that didn't work"),
            (r"\bit\s+didn'?t\s+work\b", 0.95, "it didn't work"),
            (r"\bdoesn'?t\s+(?:work|help)\b", 0.90, "doesn't work"),

            # Medium-high confidence (0.85)
            (r"\bnope\b", 0.85, "nope"),
            (r"\bno\s+luck\b", 0.85, "no luck"),
            (r"\bnot\s+working\b", 0.90, "not working"),
            (r"\bstill\s+(?:the\s+)?same\b", 0.80, "still the same"),
            (r"\banother\s+error\b", 0.85, "another error"),
            (r"\bnew\s+error\b", 0.85, "new error"),
            (r"\bdifferent\s+error\b", 0.80, "different error"),
            (r"\bdidn'?t\s+(?:fix|solve|help)\b", 0.90, "didn't fix"),

            # Medium confidence (0.70)
            (r"\bfailed\b", 0.70, "failed"),
            (r"\bbroken\b", 0.70, "broken"),
            (r"\bcrash(?:es|ed|ing)?\b", 0.75, "crash"),
            (r"\bexception\b", 0.65, "exception"),  # Lower - might be discussing, not experiencing
            (r"\btry\s+(?:again|something\s+else)\b", 0.70, "try again"),
        ]

        # Excitement amplifiers (increase confidence)
        self.amplifiers = [
            (r"!{2,}", 0.10),  # Multiple exclamation marks
            (r"\b(?:omg|oh\s+my\s+god|holy)\b", 0.10),
            (r"\b(?:finally|at\s+last)\b", 0.10),
            (r"(?:^|\s)[A-Z]{3,}\b", 0.05),  # ALL CAPS words
        ]

    def detect_feedback(self,
                        user_message: str,
                        assistant_message_before: str = None,
                        conversation_context: Dict = None) -> Optional[FeedbackSignal]:
        """
        Detect feedback signal in a user message.

        Args:
            user_message: The user's message to analyze
            assistant_message_before: The assistant's previous message (for context)
            conversation_context: Optional conversation metadata

        Returns:
            FeedbackSignal if detected, None otherwise
        """
        if not user_message or len(user_message.strip()) < 2:
            return None

        message_lower = user_message.lower()

        # Check for success patterns
        best_success = self._find_best_match(message_lower, self.success_patterns)

        # Check for failure patterns
        best_failure = self._find_best_match(message_lower, self.failure_patterns)

        # Apply amplifiers
        amplifier_boost = self._calculate_amplifier_boost(user_message)

        if best_success:
            best_success = (best_success[0], min(1.0, best_success[1] + amplifier_boost), best_success[2])
        if best_failure:
            best_failure = (best_failure[0], min(1.0, best_failure[1] + amplifier_boost), best_failure[2])

        # Determine winner (if both detected, higher confidence wins)
        # But check for mixed signals first
        if best_success and best_failure:
            # Mixed signals - check if it's "didn't work but now works" pattern
            if self._is_turnaround_success(message_lower):
                best_failure = None  # It's actually a success
            elif best_success[1] > best_failure[1]:
                best_failure = None
            else:
                best_success = None

        if best_success and best_success[1] >= 0.65:  # Confidence threshold
            signal = FeedbackSignal(
                signal_type="success",
                confidence=best_success[1],
                matched_pattern=best_success[2],
                user_message=user_message[:500],
                timestamp=datetime.now().isoformat(),
                context_before=assistant_message_before[:500] if assistant_message_before else None
            )
            self._log_feedback(signal)
            return signal

        if best_failure and best_failure[1] >= 0.65:  # Confidence threshold
            signal = FeedbackSignal(
                signal_type="failure",
                confidence=best_failure[1],
                matched_pattern=best_failure[2],
                user_message=user_message[:500],
                timestamp=datetime.now().isoformat(),
                context_before=assistant_message_before[:500] if assistant_message_before else None
            )
            self._log_feedback(signal)
            return signal

        return None

    def _find_best_match(self, text: str, patterns: List[Tuple]) -> Optional[Tuple]:
        """Find the best matching pattern with highest confidence."""
        best = None

        for pattern, confidence, name in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if best is None or confidence > best[1]:
                    best = (pattern, confidence, name)

        return best

    def _calculate_amplifier_boost(self, text: str) -> float:
        """Calculate confidence boost from amplifiers."""
        boost = 0.0
        for pattern, amount in self.amplifiers:
            if re.search(pattern, text, re.IGNORECASE):
                boost += amount
        return min(boost, 0.20)  # Cap at 20% boost

    def _is_turnaround_success(self, text: str) -> bool:
        """Detect 'it wasn't working but now it works' pattern."""
        turnaround_patterns = [
            r"(?:wasn't|didn't|wasn't)\s+work.*(?:but|now).*work",
            r"before.*(?:fail|error|broken).*(?:but|now).*(?:work|fix)",
            r"(?:finally|now)\s+(?:it\s+)?work",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in turnaround_patterns)

    def _log_feedback(self, signal: FeedbackSignal) -> None:
        """Log detected feedback to JSONL file."""
        try:
            with open(self.feedback_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(signal), ensure_ascii=False) + '\n')
        except Exception:
            pass  # Logging failure shouldn't break detection

    def analyze_conversation_feedback(self,
                                       messages: List[Dict],
                                       conversation_id: str = None) -> Dict[str, Any]:
        """
        Analyze an entire conversation for feedback signals.

        Args:
            messages: List of message dicts with 'role' and 'content'
            conversation_id: Optional conversation ID for linking

        Returns:
            Analysis results with all detected signals
        """
        results = {
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'success_signals': [],
            'failure_signals': [],
            'net_sentiment': 0.0,  # Positive = more success, negative = more failure
            'feedback_timeline': []
        }

        last_assistant_msg = None

        for i, msg in enumerate(messages):
            role = msg.get('role', '')
            content = msg.get('content', '')

            if isinstance(content, list):
                content = ' '.join(c.get('text', '') for c in content if isinstance(c, dict))

            if role == 'assistant':
                last_assistant_msg = content

            elif role == 'user' and content:
                signal = self.detect_feedback(content, last_assistant_msg)

                if signal:
                    signal_dict = asdict(signal)
                    signal_dict['message_index'] = i

                    if signal.signal_type == 'success':
                        results['success_signals'].append(signal_dict)
                        results['net_sentiment'] += signal.confidence
                    else:
                        results['failure_signals'].append(signal_dict)
                        results['net_sentiment'] -= signal.confidence

                    results['feedback_timeline'].append({
                        'index': i,
                        'type': signal.signal_type,
                        'confidence': signal.confidence,
                        'pattern': signal.matched_pattern
                    })

        # Calculate summary
        total_signals = len(results['success_signals']) + len(results['failure_signals'])
        if total_signals > 0:
            results['success_rate'] = len(results['success_signals']) / total_signals
        else:
            results['success_rate'] = None

        # Determine overall outcome
        # Key insight: the LAST signal often indicates the final outcome
        last_signal = results['feedback_timeline'][-1] if results['feedback_timeline'] else None

        if results['success_signals'] and not results['failure_signals']:
            results['overall_outcome'] = 'success'
        elif results['failure_signals'] and not results['success_signals']:
            results['overall_outcome'] = 'failure'
        elif last_signal and last_signal['type'] == 'success' and last_signal['confidence'] >= 0.8:
            # If the conversation ends on a strong success signal, that's probably the outcome
            results['overall_outcome'] = 'success'
        elif last_signal and last_signal['type'] == 'failure' and last_signal['confidence'] >= 0.8:
            results['overall_outcome'] = 'failure'
        elif results['net_sentiment'] > 0:
            results['overall_outcome'] = 'mixed_positive'
        elif results['net_sentiment'] < 0:
            results['overall_outcome'] = 'mixed_negative'
        else:
            results['overall_outcome'] = 'neutral'

        return results

    def link_feedback_to_solution(self,
                                   signal: FeedbackSignal,
                                   solution_id: str) -> Dict[str, Any]:
        """
        Link a feedback signal to a specific solution and record in solution_tracker.

        Args:
            signal: The detected feedback signal
            solution_id: ID of the solution to link

        Returns:
            Result of recording the feedback
        """
        try:
            from solution_tracker import SolutionTracker
            tracker = SolutionTracker(str(self.base_path))

            if signal.signal_type == 'success':
                result = tracker.confirm_solution_works(
                    solution_id=solution_id,
                    conversation_id=None
                )
                return {
                    'action': 'confirmed',
                    'solution_id': solution_id,
                    'result': result
                }
            else:
                result = tracker.record_failure(
                    solution_id=solution_id,
                    failure_description=f"User indicated failure: {signal.matched_pattern}",
                    error_message=signal.user_message[:200],
                    conversation_id=None
                )
                return {
                    'action': 'failure_recorded',
                    'solution_id': solution_id,
                    'result': result
                }

        except Exception as e:
            return {
                'error': str(e),
                'solution_id': solution_id
            }

    def get_feedback_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get statistics on detected feedback over time.

        Args:
            days: Number of days to analyze

        Returns:
            Feedback statistics
        """
        stats = {
            'period_days': days,
            'total_success': 0,
            'total_failure': 0,
            'by_pattern': {},
            'by_day': {},
            'avg_confidence': {'success': 0.0, 'failure': 0.0}
        }

        if not self.feedback_log.exists():
            return stats

        datetime.now().isoformat()[:10]
        from datetime import timedelta
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        success_confidences = []
        failure_confidences = []

        try:
            with open(self.feedback_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        timestamp = record.get('timestamp', '')

                        if timestamp < cutoff_date:
                            continue

                        date = timestamp[:10]
                        signal_type = record.get('signal_type')
                        pattern = record.get('matched_pattern', 'unknown')
                        confidence = record.get('confidence', 0.0)

                        # Count by type
                        if signal_type == 'success':
                            stats['total_success'] += 1
                            success_confidences.append(confidence)
                        elif signal_type == 'failure':
                            stats['total_failure'] += 1
                            failure_confidences.append(confidence)

                        # Count by pattern
                        if pattern not in stats['by_pattern']:
                            stats['by_pattern'][pattern] = {'success': 0, 'failure': 0}
                        stats['by_pattern'][pattern][signal_type] = \
                            stats['by_pattern'][pattern].get(signal_type, 0) + 1

                        # Count by day
                        if date not in stats['by_day']:
                            stats['by_day'][date] = {'success': 0, 'failure': 0}
                        stats['by_day'][date][signal_type] = \
                            stats['by_day'][date].get(signal_type, 0) + 1

                    except json.JSONDecodeError:
                        continue

        except Exception:
            pass

        # Calculate averages
        if success_confidences:
            stats['avg_confidence']['success'] = round(
                sum(success_confidences) / len(success_confidences), 3
            )
        if failure_confidences:
            stats['avg_confidence']['failure'] = round(
                sum(failure_confidences) / len(failure_confidences), 3
            )

        # Calculate overall success rate
        total = stats['total_success'] + stats['total_failure']
        if total > 0:
            stats['success_rate'] = round(stats['total_success'] / total, 3)
        else:
            stats['success_rate'] = None

        return stats


# Convenience function for quick detection
def detect_feedback(user_message: str,
                    assistant_message: str = None) -> Optional[FeedbackSignal]:
    """Quick feedback detection without full initialization."""
    detector = FeedbackDetector()
    return detector.detect_feedback(user_message, assistant_message)


if __name__ == "__main__":
    # Test the feedback detector
    detector = FeedbackDetector()

    print("=== Feedback Detector Test ===\n")

    test_messages = [
        # Success signals
        ("It worked! Finally!", None),
        ("Perfect, that fixed it", "Try adding the --force flag"),
        ("YES!! No more errors", None),
        ("thanks that did the trick", "The solution is to restart the service"),
        ("awesome, problem solved", None),
        ("omg it works now!!!", None),

        # Failure signals
        ("Still getting the same error", "Try clearing the cache"),
        ("nope, didn't work", None),
        ("still broken", "Have you tried restarting?"),
        ("that didn't help, same issue", None),
        ("another error now", None),

        # Neutral/ambiguous
        ("ok let me try that", None),
        ("interesting", None),
        ("what about option B?", None),

        # Turnaround success
        ("it wasn't working before but now it works!", None),
    ]

    for msg, ctx in test_messages:
        signal = detector.detect_feedback(msg, ctx)
        if signal:
            print(f"[{signal.signal_type.upper():7}] {signal.confidence:.2f} | {msg[:50]}")
            print(f"          Pattern: {signal.matched_pattern}")
        else:
            print(f"[NONE   ]      | {msg[:50]}")
        print()

    # Test conversation analysis
    print("\n=== Conversation Analysis Test ===\n")

    test_conversation = [
        {'role': 'user', 'content': 'The build is failing with error code 1'},
        {'role': 'assistant', 'content': 'Try running npm install first'},
        {'role': 'user', 'content': 'still failing'},
        {'role': 'assistant', 'content': 'Clear node_modules and try again'},
        {'role': 'user', 'content': 'Same error'},
        {'role': 'assistant', 'content': 'Check if you have the right Node version'},
        {'role': 'user', 'content': 'That was it! Works now, thanks!'},
    ]

    analysis = detector.analyze_conversation_feedback(test_conversation, 'test_conv')
    print(f"Overall outcome: {analysis['overall_outcome']}")
    print(f"Success signals: {len(analysis['success_signals'])}")
    print(f"Failure signals: {len(analysis['failure_signals'])}")
    print(f"Net sentiment: {analysis['net_sentiment']:.2f}")
    print(f"Success rate: {analysis['success_rate']}")
