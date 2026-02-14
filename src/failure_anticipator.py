"""
Failure Anticipator - Claude.Me v6.0
Warn about likely failures before they happen.

Part of Phase 7: Predictive Simulation
"""
import json
from pathlib import Path
from typing import Dict, List, Optional


class FailureAnticipator:
    """
    Anticipate failures based on past experience.

    Capabilities:
    - Match current context to past failures
    - Warn about likely issues
    - Suggest preventive actions
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.failure_path = self.base_path / "failure_memory"

    def anticipate_failures(self, context: str) -> List[Dict]:
        """
        Anticipate potential failures based on context.

        Returns list of potential failures with warnings.
        """
        warnings = []
        context_lower = context.lower()
        context_words = set(context_lower.split())

        # Check failure index
        index_file = self.failure_path / "failures_index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    index = json.load(f)

                for failure in index.get("failures", []):
                    problem = failure.get("problem", "").lower()
                    problem_words = set(problem.split())

                    overlap = len(context_words.intersection(problem_words))
                    if overlap >= 2:
                        confidence = min(0.9, 0.4 + (overlap * 0.1))

                        warnings.append({
                            "warning": f"Potential issue: {failure.get('problem', '')}",
                            "what_failed_before": failure.get("what_didnt_work", ""),
                            "successful_fix": failure.get("what_worked", ""),
                            "confidence": confidence,
                            "match_score": overlap,
                            "frequency": failure.get("frequency", 1)
                        })
            except:
                pass

        # Check individual failure files
        for fail_file in self.failure_path.glob("fail_*.json"):
            try:
                with open(fail_file) as f:
                    failure = json.load(f)

                problem = failure.get("problem", "").lower()
                problem_words = set(problem.split())

                overlap = len(context_words.intersection(problem_words))
                if overlap >= 2:
                    # Check if already in warnings
                    already_warned = any(
                        w["warning"] == f"Potential issue: {failure.get('problem', '')}"
                        for w in warnings
                    )
                    if not already_warned:
                        warnings.append({
                            "warning": f"Potential issue: {failure.get('problem', '')}",
                            "what_failed_before": failure.get("what_didnt_work", ""),
                            "successful_fix": failure.get("what_worked", ""),
                            "confidence": min(0.9, 0.3 + (overlap * 0.1)),
                            "match_score": overlap
                        })
            except:
                continue

        # Sort by confidence
        warnings.sort(key=lambda w: w.get("confidence", 0), reverse=True)
        return warnings[:5]

    def check_specific_pattern(self, pattern_type: str, context: str) -> Optional[Dict]:
        """
        Check for a specific failure pattern.

        Args:
            pattern_type: Type of pattern (timeout, encoding, path, etc.)
            context: Current context

        Returns warning if pattern detected.
        """
        pattern_signatures = {
            "timeout": ["timeout", "timed out", "hang", "stuck", "waiting"],
            "encoding": ["encoding", "utf", "bytes", "decode", "encode"],
            "path": ["path", "file not found", "directory", "exists"],
            "network": ["connection", "socket", "network", "unreachable"],
            "permission": ["permission", "access denied", "unauthorized"]
        }

        context_lower = context.lower()
        signatures = pattern_signatures.get(pattern_type, [])

        if any(sig in context_lower for sig in signatures):
            warnings = self.anticipate_failures(context)
            for w in warnings:
                if any(sig in w.get("warning", "").lower() for sig in signatures):
                    return w

        return None

    def get_preventive_actions(self, context: str) -> List[str]:
        """Get preventive actions for current context."""
        actions = []
        warnings = self.anticipate_failures(context)

        for warning in warnings:
            fix = warning.get("successful_fix")
            if fix and fix != "Not explicitly stated":
                actions.append(f"Consider: {fix}")

        return list(set(actions))[:5]

    def get_stats(self) -> Dict:
        """Get failure anticipator statistics."""
        total_failures = 0
        by_frequency = {"high": 0, "medium": 0, "low": 0}

        index_file = self.failure_path / "failures_index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    index = json.load(f)

                for failure in index.get("failures", []):
                    total_failures += 1
                    freq = failure.get("frequency", 1)
                    if freq >= 3:
                        by_frequency["high"] += 1
                    elif freq >= 2:
                        by_frequency["medium"] += 1
                    else:
                        by_frequency["low"] += 1
            except:
                pass

        return {
            "total_failures_tracked": total_failures,
            "by_frequency": by_frequency
        }
