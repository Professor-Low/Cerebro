#!/usr/bin/env python3
"""
Learning Promoter - Auto-promotes recurring patterns to quick_facts.json

Detects:
1. Recurring problems (same topic appears 3+ times in learnings)
2. Solutions used repeatedly (same fix applied multiple times)
3. Correction patterns (same mistake made multiple times)

Promotes them to the `promoted_patterns` section of quick_facts.json
for immediate recall in future sessions.

Phase 5.3 of Brain Evolution.
"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

from config import DATA_DIR

# Paths
LEARNINGS_DIR = DATA_DIR / "learnings"
FAILURE_MEMORY_PATH = DATA_DIR / "failure_memory" / "failures_index.json"
CORRECTIONS_LOG_PATH = DATA_DIR / "corrections" / "auto_detected.json"
QUICK_FACTS_PATH = DATA_DIR / "quick_facts.json"


class LearningPromoter:
    """Detects recurring patterns and promotes them to quick_facts."""

    # Quality gate patterns - solutions containing these are corrupted/low-quality
    BAD_SOLUTION_PATTERNS = [
        "trigger entity extraction",
        ".agent-card",
        "border-left:",
        "let me check if hooks fire",
        "have tested instead of speculating",
        "let me check the css",
        "see context messages for details",
        "auto-detected breakthrough (problem details unavailable)",
        "exit code 127",
        "add the user message before the api call",
    ]

    def __init__(self):
        self.quick_facts = self._load_quick_facts()

    def is_quality_solution(self, solution: str) -> bool:
        """Filter out corrupted/low-quality solutions.

        Returns True if the solution passes quality checks.
        Returns False if the solution should be rejected.
        """
        if not solution:
            return False

        # Reject very short solutions (less than 50 chars means likely truncated/garbage)
        if len(solution) < 50:
            return False

        # Check for known bad patterns (case-insensitive)
        solution_lower = solution.lower()
        for bad_pattern in self.BAD_SOLUTION_PATTERNS:
            if bad_pattern in solution_lower:
                return False

        # Reject solutions that look like CSS (common corruption pattern)
        css_indicators = ['{', '}', 'var(--', 'px;', 'color:', 'border:']
        css_count = sum(1 for indicator in css_indicators if indicator in solution)
        if css_count >= 2:  # 2+ CSS indicators means it's likely CSS garbage
            return False

        # Reject solutions that are truncated (end with common truncation patterns)
        truncation_indicators = ['...', '|', ' f', ' (`']
        for indicator in truncation_indicators:
            if solution.rstrip().endswith(indicator):
                return False

        return True

    def _load_quick_facts(self) -> Dict:
        """Load quick_facts.json."""
        try:
            if QUICK_FACTS_PATH.exists():
                with open(QUICK_FACTS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading quick_facts: {e}")
        return {}

    def _save_quick_facts(self):
        """Save quick_facts.json."""
        try:
            self.quick_facts["_last_updated"] = datetime.now().isoformat()
            with open(QUICK_FACTS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.quick_facts, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving quick_facts: {e}")

    def _load_learnings(self) -> List[Dict]:
        """Load all learning files."""
        learnings = []
        try:
            if LEARNINGS_DIR.exists():
                for f in LEARNINGS_DIR.glob("*.json"):
                    try:
                        with open(f, "r", encoding="utf-8") as file:
                            learnings.append(json.load(file))
                    except:
                        continue
        except Exception as e:
            print(f"Error loading learnings: {e}")
        return learnings

    def _load_failures(self) -> List[Dict]:
        """Load failure memory."""
        try:
            if FAILURE_MEMORY_PATH.exists():
                with open(FAILURE_MEMORY_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("failures", [])
        except Exception as e:
            print(f"Error loading failures: {e}")
        return []

    def _load_corrections(self) -> List[Dict]:
        """Load auto-detected corrections."""
        try:
            if CORRECTIONS_LOG_PATH.exists():
                with open(CORRECTIONS_LOG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("corrections", [])
        except Exception as e:
            print(f"Error loading corrections: {e}")
        return []

    def detect_recurring_problems(self, min_occurrences: int = 3) -> List[Dict]:
        """Find problems that occur multiple times.

        Groups learnings by keyword and finds topics that appear 3+ times.
        """
        learnings = self._load_learnings()
        failures = self._load_failures()

        # Count keyword occurrences
        keyword_counts = defaultdict(list)

        for learning in learnings:
            keywords = learning.get("keywords", [])
            problem = learning.get("problem", "")
            for kw in keywords:
                kw_lower = kw.lower()
                keyword_counts[kw_lower].append({
                    "type": "learning",
                    "problem": problem[:100],
                    "solution": learning.get("solution", "")[:150],
                    "date": learning.get("date", ""),
                })

        for failure in failures:
            category = failure.get("category", "")
            keywords = failure.get("keywords", [])
            for kw in [category] + keywords:
                if kw:
                    kw_lower = kw.lower()
                    keyword_counts[kw_lower].append({
                        "type": "failure",
                        "problem": failure.get("problem", "")[:100],
                        "solution": failure.get("what_worked", "")[:150],
                        "date": failure.get("date", ""),
                    })

        # Filter to recurring (3+ occurrences)
        recurring = []
        for keyword, occurrences in keyword_counts.items():
            if len(occurrences) >= min_occurrences:
                # Get the most recent solution
                sorted_occ = sorted(occurrences, key=lambda x: x.get("date", ""), reverse=True)
                best_solution = next((o["solution"] for o in sorted_occ if o.get("solution")), "")

                recurring.append({
                    "keyword": keyword,
                    "occurrence_count": len(occurrences),
                    "best_solution": self._smart_truncate(best_solution, 500),
                    "problems": [o["problem"] for o in sorted_occ[:3]],
                })

        # Sort by occurrence count
        recurring.sort(key=lambda x: x["occurrence_count"], reverse=True)
        return recurring

    @staticmethod
    def _smart_truncate(text: str, max_len: int = 500) -> str:
        """Truncate text at sentence boundary, not mid-word/mid-sentence."""
        if not text or len(text) <= max_len:
            return text
        # Find the last sentence-ending punctuation before max_len
        truncated = text[:max_len]
        for end_char in ['. ', '.\n', '! ', '?\n']:
            last_period = truncated.rfind(end_char)
            if last_period > max_len * 0.5:  # Only if we keep at least half
                return truncated[:last_period + 1].strip()
        # Fallback: truncate at last space
        last_space = truncated.rfind(' ')
        if last_space > max_len * 0.5:
            return truncated[:last_space].strip()
        return truncated.strip()

    def detect_repeated_solutions(self, min_uses: int = 2) -> List[Dict]:
        """Find solutions that have been used multiple times."""
        learnings = self._load_learnings()
        failures = self._load_failures()

        # Group by solution similarity using full content hash (not prefix)
        solutions = defaultdict(list)

        for learning in learnings:
            solution = learning.get("solution", "").strip()
            if solution and len(solution) > 20:
                # Use MD5 hash of full lowercase solution for proper dedup
                key = hashlib.md5(solution.lower().encode()).hexdigest()[:16]
                solutions[key].append({
                    "full_solution": self._smart_truncate(solution, 500),
                    "problem": learning.get("problem", "")[:100],
                    "keywords": learning.get("keywords", []),
                    "date": learning.get("date", ""),
                })

        for failure in failures:
            solution = failure.get("what_worked", "").strip()
            if solution and len(solution) > 20:
                key = hashlib.md5(solution.lower().encode()).hexdigest()[:16]
                solutions[key].append({
                    "full_solution": self._smart_truncate(solution, 500),
                    "problem": failure.get("problem", "")[:100],
                    "keywords": failure.get("keywords", []),
                    "date": failure.get("date", ""),
                })

        # Filter to repeated (2+ uses)
        repeated = []
        for key, uses in solutions.items():
            if len(uses) >= min_uses:
                # Get unique keywords
                all_keywords = set()
                for use in uses:
                    all_keywords.update(use.get("keywords", []))

                repeated.append({
                    "solution": uses[0]["full_solution"],
                    "use_count": len(uses),
                    "keywords": list(all_keywords)[:10],
                    "problems_solved": [u["problem"] for u in uses[:3]],
                })

        repeated.sort(key=lambda x: x["use_count"], reverse=True)
        return repeated

    def promote_patterns(self, dry_run: bool = True) -> Dict[str, Any]:
        """Detect and promote recurring patterns to quick_facts.

        Args:
            dry_run: If True, only preview what would be promoted

        Returns:
            Summary of what was (or would be) promoted
        """
        recurring = self.detect_recurring_problems(min_occurrences=3)
        repeated_solutions = self.detect_repeated_solutions(min_uses=2)

        # Initialize promoted_patterns if needed
        if "promoted_patterns" not in self.quick_facts:
            self.quick_facts["promoted_patterns"] = {
                "_description": "Recurring problems and their proven solutions, auto-promoted from learning system",
                "patterns": []
            }

        existing_keywords = {
            p.get("keyword", "").lower()
            for p in self.quick_facts["promoted_patterns"].get("patterns", [])
        }

        new_patterns = []
        rejected_count = 0

        # Add recurring problems (with quality gate)
        for r in recurring[:10]:  # Top 10
            if r["keyword"].lower() not in existing_keywords:
                # QUALITY GATE: Check solution quality before adding
                if not self.is_quality_solution(r["best_solution"]):
                    rejected_count += 1
                    continue

                new_patterns.append({
                    "keyword": r["keyword"],
                    "type": "recurring_problem",
                    "occurrence_count": r["occurrence_count"],
                    "solution": r["best_solution"],
                    "example_problems": r["problems"][:2],
                    "promoted_at": datetime.now().isoformat(),
                    "last_used": None,  # Track when solution was applied
                    "use_count": 0,     # Track how many times it helped
                    "effectiveness": None  # "high", "medium", "low"
                })
                existing_keywords.add(r["keyword"].lower())

        # Add repeated solutions (with quality gate)
        for s in repeated_solutions[:5]:  # Top 5
            # Create a keyword from the solution
            solution_key = "_".join(s.get("keywords", ["solution"])[:2])
            if solution_key.lower() not in existing_keywords:
                # QUALITY GATE: Check solution quality before adding
                if not self.is_quality_solution(s["solution"]):
                    rejected_count += 1
                    continue

                new_patterns.append({
                    "keyword": solution_key,
                    "type": "proven_solution",
                    "use_count": s["use_count"],
                    "solution": s["solution"],
                    "problems_solved": s["problems_solved"][:2],
                    "promoted_at": datetime.now().isoformat(),
                    "last_used": None,
                    "effectiveness": None
                })
                existing_keywords.add(solution_key.lower())

        result = {
            "recurring_problems_found": len(recurring),
            "repeated_solutions_found": len(repeated_solutions),
            "new_patterns_to_promote": len(new_patterns),
            "rejected_low_quality": rejected_count,  # Patterns rejected by quality gate
            "preview": new_patterns[:5],  # Show preview
        }

        if not dry_run and new_patterns:
            self.quick_facts["promoted_patterns"]["patterns"].extend(new_patterns)
            self.quick_facts["promoted_patterns"]["_last_promoted"] = datetime.now().isoformat()
            self._save_quick_facts()
            result["promoted"] = True
            result["total_promoted_patterns"] = len(self.quick_facts["promoted_patterns"]["patterns"])

        return result

    def get_promoted_patterns(self) -> List[Dict]:
        """Get all promoted patterns."""
        return self.quick_facts.get("promoted_patterns", {}).get("patterns", [])

    def mark_pattern_used(self, keyword: str, effective: bool = True) -> bool:
        """Mark a pattern as used and update effectiveness tracking.

        Args:
            keyword: The pattern keyword to mark as used
            effective: Whether the solution was effective

        Returns:
            True if pattern was found and updated
        """
        patterns = self.quick_facts.get("promoted_patterns", {}).get("patterns", [])

        for p in patterns:
            if p.get("keyword", "").lower() == keyword.lower():
                # Update usage tracking
                p["last_used"] = datetime.now().isoformat()
                p["use_count"] = p.get("use_count", 0) + 1

                # Update effectiveness
                if effective:
                    # High if used 3+ times, medium if 2, low if 1
                    use_count = p["use_count"]
                    if use_count >= 3:
                        p["effectiveness"] = "high"
                    elif use_count >= 2:
                        p["effectiveness"] = "medium"
                    else:
                        p["effectiveness"] = "low"

                self._save_quick_facts()
                return True

        return False


def run_pattern_detection(promote: bool = False) -> Dict[str, Any]:
    """Run pattern detection and optionally promote.

    Args:
        promote: If True, actually promote patterns. Otherwise just preview.
    """
    promoter = LearningPromoter()
    return promoter.promote_patterns(dry_run=not promote)


if __name__ == "__main__":
    import sys

    promote = "--promote" in sys.argv

    print("Running pattern detection...")
    result = run_pattern_detection(promote=promote)

    print(f"\nRecurring problems found: {result['recurring_problems_found']}")
    print(f"Repeated solutions found: {result['repeated_solutions_found']}")
    print(f"New patterns to promote: {result['new_patterns_to_promote']}")

    if result.get("preview"):
        print("\nPreview of patterns:")
        for p in result["preview"]:
            print(f"  - {p['keyword']}: {p.get('solution', '')[:60]}...")

    if promote:
        print(f"\nPatterns promoted! Total: {result.get('total_promoted_patterns', 0)}")
    else:
        print("\nRun with --promote to actually promote these patterns")
