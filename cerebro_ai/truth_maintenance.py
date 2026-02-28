"""
Truth Maintenance System - Fact Invalidation via Corrections
When a correction is recorded, find and supersede contradicting facts.

This is the key to preventing "confidently wrong forever."

How it works:
1. User corrects Claude: "NAS IP is .100 not .1"
2. Correction is saved with mistake="10.0.0.1" and correction="10.0.0.100"
3. TruthMaintenance.propagate_correction() finds facts containing the mistake
4. Those facts get marked superseded_by: correction_id
5. Search filters out superseded facts by default

IMPORTANT: Facts are not DELETED, just SUPERSEDED. History is preserved.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class TruthMaintenance:
    """
    Maintains truth consistency by propagating corrections to facts.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            from config import AI_MEMORY_BASE
            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.facts_path = self.base_path / "facts"
        self.corrections_path = self.base_path / "corrections"
        self.supersession_log_path = self.base_path / "corrections" / "supersession_log.jsonl"

        # Ensure directories exist
        self.facts_path.mkdir(parents=True, exist_ok=True)
        self.corrections_path.mkdir(parents=True, exist_ok=True)

    def propagate_correction(self, correction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propagate a correction to all facts that contain the mistake.

        Args:
            correction: Dict with 'id', 'mistake', 'correction', 'topic', etc.

        Returns:
            Dict with:
                - facts_found: Number of facts containing the mistake
                - facts_superseded: Number of facts marked superseded
                - fact_ids: List of superseded fact IDs
        """
        mistake = correction.get("mistake", "")
        correction_id = correction.get("id", "")
        topic = correction.get("topic", "")

        if not mistake or not correction_id:
            return {
                "error": "Correction must have 'id' and 'mistake' fields",
                "facts_found": 0,
                "facts_superseded": 0,
                "fact_ids": []
            }

        # Find facts containing the mistake
        matching_facts = self._find_facts_with_content(mistake, topic)

        superseded_count = 0
        superseded_ids = []

        for fact_path in matching_facts:
            try:
                success = self._supersede_fact(fact_path, correction_id, correction.get("correction", ""))
                if success:
                    superseded_count += 1
                    superseded_ids.append(fact_path.stem)
            except Exception as e:
                print(f"Warning: Could not supersede fact {fact_path}: {e}")

        # Log the supersession event
        self._log_supersession(correction_id, superseded_ids, mistake, correction.get("correction", ""))

        return {
            "correction_id": correction_id,
            "mistake": mistake,
            "facts_found": len(matching_facts),
            "facts_superseded": superseded_count,
            "fact_ids": superseded_ids
        }

    def _find_facts_with_content(self, mistake: str, topic: str = "") -> List[Path]:
        """
        Find all fact files containing the mistake text.

        Uses case-insensitive matching and handles common variations.
        """
        matching_facts = []
        mistake_lower = mistake.lower().strip()

        # Also check for common patterns
        # e.g., "10.0.0.1" should match facts with "IP is 10.0.0.1" or "10.0.0.1"
        mistake_patterns = self._generate_match_patterns(mistake_lower)

        # Search individual fact JSON files
        for fact_file in self.facts_path.glob("*.json"):
            try:
                with open(fact_file, 'r', encoding='utf-8') as f:
                    fact = json.load(f)

                content = fact.get("content", "").lower()

                # Check if already superseded
                if fact.get("superseded_by"):
                    continue

                # Check for match
                if self._content_matches(content, mistake_patterns):
                    matching_facts.append(fact_file)

            except Exception:
                # Skip unreadable files
                continue

        # Also search facts.jsonl if it exists
        facts_jsonl = self.facts_path / "facts.jsonl"
        if facts_jsonl.exists():
            # Note: For JSONL, we can't easily update in place
            # We'll handle this separately if needed
            pass

        return matching_facts

    def _generate_match_patterns(self, mistake: str) -> List[str]:
        """
        Generate patterns to match the mistake in various contexts.

        Example: "10.0.0.1" generates:
        - "10.0.0.1" (exact)
        - "ip is 10.0.0.1"
        - "ip: 10.0.0.1"
        """
        patterns = [mistake]

        # For IP addresses, add common prefixes
        if re.match(r'\d+\.\d+\.\d+\.\d+', mistake):
            patterns.extend([
                f"ip is {mistake}",
                f"ip: {mistake}",
                f"address is {mistake}",
                f"address: {mistake}",
            ])

        # For port numbers
        if mistake.isdigit():
            patterns.extend([
                f"port {mistake}",
                f"port: {mistake}",
                f":{mistake}",
            ])

        return patterns

    def _content_matches(self, content: str, patterns: List[str]) -> bool:
        """Check if content matches any of the patterns."""
        for pattern in patterns:
            if pattern in content:
                return True
        return False

    def _supersede_fact(self, fact_path: Path, correction_id: str, correct_value: str) -> bool:
        """
        Mark a fact as superseded by a correction.

        Adds fields:
        - superseded_by: correction_id
        - superseded_at: timestamp
        - status: "superseded"
        - correct_value: what the correct value is
        """
        try:
            with open(fact_path, 'r', encoding='utf-8') as f:
                fact = json.load(f)

            # Add supersession metadata
            fact["superseded_by"] = correction_id
            fact["superseded_at"] = datetime.now().isoformat()
            fact["status"] = "superseded"
            fact["correct_value"] = correct_value

            # Preserve original status if tracking
            if "original_status" not in fact:
                fact["original_status"] = fact.get("status", "active")

            with open(fact_path, 'w', encoding='utf-8') as f:
                json.dump(fact, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Error superseding fact {fact_path}: {e}")
            return False

    def _log_supersession(self, correction_id: str, fact_ids: List[str],
                          mistake: str, correction: str):
        """Log supersession events for audit trail."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "correction_id": correction_id,
            "mistake": mistake,
            "correction": correction,
            "facts_superseded": fact_ids,
            "count": len(fact_ids)
        }

        try:
            with open(self.supersession_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not write supersession log: {e}")

    def filter_superseded(self, facts: List[Dict[str, Any]],
                          include_superseded: bool = False) -> List[Dict[str, Any]]:
        """
        Filter out superseded facts from a list.

        Args:
            facts: List of fact dicts
            include_superseded: If True, include superseded facts (for history)

        Returns:
            Filtered list of active facts
        """
        if include_superseded:
            return facts

        return [f for f in facts if f.get("status") != "superseded"
                and not f.get("superseded_by")]

    def get_supersession_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent supersession events."""
        history = []

        if self.supersession_log_path.exists():
            try:
                with open(self.supersession_log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            history.append(json.loads(line))
            except Exception as e:
                print(f"Error reading supersession log: {e}")

        # Return most recent first
        history.reverse()
        return history[:limit]

    def get_superseded_facts(self) -> List[Dict[str, Any]]:
        """Get all superseded facts (for review/debugging)."""
        superseded = []

        for fact_file in self.facts_path.glob("*.json"):
            try:
                with open(fact_file, 'r', encoding='utf-8') as f:
                    fact = json.load(f)

                if fact.get("superseded_by") or fact.get("status") == "superseded":
                    superseded.append(fact)

            except Exception:
                continue

        return superseded

    def undo_supersession(self, fact_id: str) -> Dict[str, Any]:
        """
        Undo a supersession (restore a fact to active).

        Use with caution - this reverses a correction.
        """
        fact_path = self.facts_path / f"{fact_id}.json"

        if not fact_path.exists():
            return {"error": f"Fact {fact_id} not found"}

        try:
            with open(fact_path, 'r', encoding='utf-8') as f:
                fact = json.load(f)

            # Remove supersession metadata
            old_correction = fact.pop("superseded_by", None)
            fact.pop("superseded_at", None)
            fact.pop("correct_value", None)
            fact["status"] = fact.pop("original_status", "active")

            with open(fact_path, 'w', encoding='utf-8') as f:
                json.dump(fact, f, indent=2, ensure_ascii=False)

            return {
                "success": True,
                "fact_id": fact_id,
                "restored_from": old_correction,
                "new_status": fact.get("status", "active")
            }

        except Exception as e:
            return {"error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get truth maintenance statistics."""
        total_facts = 0
        active_facts = 0
        superseded_facts = 0

        for fact_file in self.facts_path.glob("*.json"):
            try:
                with open(fact_file, 'r', encoding='utf-8') as f:
                    fact = json.load(f)

                total_facts += 1
                if fact.get("superseded_by") or fact.get("status") == "superseded":
                    superseded_facts += 1
                else:
                    active_facts += 1

            except Exception:
                continue

        # Count supersession events
        supersession_events = 0
        if self.supersession_log_path.exists():
            try:
                with open(self.supersession_log_path, 'r', encoding='utf-8') as f:
                    supersession_events = sum(1 for _ in f)
            except Exception:
                pass

        return {
            "total_facts": total_facts,
            "active_facts": active_facts,
            "superseded_facts": superseded_facts,
            "supersession_rate": round(superseded_facts / max(total_facts, 1) * 100, 1),
            "total_supersession_events": supersession_events
        }


# Convenience function for integration
def propagate_correction_to_facts(correction: Dict[str, Any],
                                   base_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to propagate a correction.

    Call this after saving a correction to automatically supersede
    contradicting facts.
    """
    maintainer = TruthMaintenance(base_path=base_path)
    return maintainer.propagate_correction(correction)


if __name__ == "__main__":
    # Test the truth maintenance system
    maintainer = TruthMaintenance()

    # Show current stats
    print("Current stats:")
    stats = maintainer.get_stats()
    print(json.dumps(stats, indent=2))

    # Test with a sample correction
    test_correction = {
        "id": "test_corr_001",
        "mistake": "10.0.0.1",
        "correction": "10.0.0.100",
        "topic": "network"
    }

    print("\nTesting propagation with:")
    print(json.dumps(test_correction, indent=2))

    # Dry run - just find matches
    matching = maintainer._find_facts_with_content(test_correction["mistake"], "")
    print(f"\nFound {len(matching)} facts containing the mistake")
    for m in matching[:5]:
        print(f"  - {m.stem}")

    # Uncomment to actually propagate:
    # result = maintainer.propagate_correction(test_correction)
    # print(json.dumps(result, indent=2))
