"""
Correction Tracking System
Stores and retrieves corrections made by user.

MULTI-AGENT NOTICE: This is Agent 2's exclusive domain.
Part of the NAS Cerebral Interface - Learning from Corrections System

TRUTH MAINTENANCE: When a correction is saved, it automatically
propagates to supersede contradicting facts. This prevents
"confidently wrong forever."
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Import truth maintenance for automatic fact supersession
try:
    from truth_maintenance import propagate_correction_to_facts
    TRUTH_MAINTENANCE_ENABLED = True
except ImportError:
    TRUTH_MAINTENANCE_ENABLED = False
    print("Warning: truth_maintenance not found. Corrections won't auto-supersede facts.")


class CorrectionsTracker:
    """Manages correction storage and retrieval."""

    # Characters that indicate UTF-8 corruption
    BAD_CHARS = ["â€", "Ã", "Â", "\ufffd"]

    # Common English words that should never be correction values
    COMMON_WORD_BLOCKLIST = {
        'think', 'know', 'want', 'need', 'use', 'try', 'say', 'make', 'go', 'get',
        'see', 'come', 'take', 'give', 'tell', 'ask', 'work', 'call', 'put', 'keep',
        'better', 'worse', 'good', 'bad', 'great', 'right', 'wrong', 'done',
        'the', 'a', 'an', 'it', 'this', 'that', 'to', 'in', 'for', 'of', 'on', 'at',
        'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'and', 'but', 'or', 'not', 'so', 'yet', 'just', 'also', 'very', 'really',
        'thing', 'way', 'time', 'bar', 'help', 'skill', 'worth', 'urgent', 'using',
        'know', 'what', 'where', 'when', 'why', 'how', 'here', 'there', 'create',
    }

    def _is_valid_correction(self, correction: Dict) -> bool:
        """Validate correction before storing.

        Real corrections involve technical values like paths, IPs, port numbers,
        config keys, or proper nouns - NOT common English words.
        """
        mistake = correction.get("mistake", "")
        corrected = correction.get("correction", "")

        # Reject if either is empty or too short (min 3 chars)
        if not mistake or len(mistake) < 3:
            return False
        if not corrected or len(corrected) < 3:
            return False

        # Reject if they're the same
        if mistake.lower().strip() == corrected.lower().strip():
            return False

        # Reject if either contains encoding artifacts
        if any(c in mistake or c in corrected for c in self.BAD_CHARS):
            return False

        # Reject common English words
        if mistake.lower().strip() in self.COMMON_WORD_BLOCKLIST:
            return False
        if corrected.lower().strip() in self.COMMON_WORD_BLOCKLIST:
            return False

        # Reject values starting with conjunctions/articles (sentence fragments)
        import re
        invalid_patterns = [
            r"^(i'?m|you'?re|he'?s|she'?s|it'?s|we'?re|they'?re)\s",
            r"^\s*(the|a|an|this|that|but|and|or|so)\s+\w+",
        ]
        for pattern in invalid_patterns:
            if re.match(pattern, mistake, re.IGNORECASE) or re.match(pattern, corrected, re.IGNORECASE):
                return False

        # At least one value must look technical (contains digits, dots, slashes, etc.)
        # or be a proper noun (starts with uppercase)
        def looks_technical(val):
            return (any(c in val for c in './:_-@0123456789')
                    or val[0].isupper()
                    or (val.isupper() and len(val) >= 3))

        if not looks_technical(mistake) and not looks_technical(corrected):
            return False

        return True

    def _validate_with_llm(self, mistake: str, correction: str, context: str, user_message: str) -> object:
        """Validate correction via DGX Spark LLM. Returns cleaned correction dict, None (rejected), or 'FALLBACK'."""
        import urllib.error
        import urllib.request

        dgx_host = os.environ.get("CEREBRO_DGX_HOST", "")
        dgx_port = os.environ.get("CEREBRO_DGX_OLLAMA_PORT", "11434")

        prompt = f"""Analyze if this is a REAL correction where the user is correcting a factual mistake the AI made.

User said: "{user_message[:300]}"
AI previously said: "{context[:300]}"
Detected mistake: "{mistake}"
Detected correction: "{correction}"

Rules:
- A REAL correction is when the user explicitly says something like "no, it's X not Y" or "the correct value is X"
- NOT a correction: casual conversation containing "not" (e.g., "it's not working", "I'm not sure")
- NOT a correction: the user asking questions or giving instructions
- NOT a correction: single common words as the "mistake" (the, a, is, from, an, use, etc.)

If this IS a real correction, respond: YES|cleaned_mistake|cleaned_correction
If this is NOT a real correction, respond: NO"""

        try:
            req_data = json.dumps({
                "model": os.environ.get("CEREBRO_LLM_MODEL", "qwen3:8b"),
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 100}
            }).encode()

            req = urllib.request.Request(
                f"http://{dgx_host}:{dgx_port}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())
                response_text = result.get("response", "").strip()

                if response_text.startswith("YES"):
                    parts = response_text.split("|")
                    if len(parts) >= 3:
                        return {"mistake": parts[1].strip(), "correction": parts[2].strip()}
                    return {"mistake": mistake, "correction": correction}
                return None  # LLM says not a real correction

        except Exception:
            return "FALLBACK"  # DGX unavailable, fall back to strict regex

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.corrections_dir = self.base_path / "corrections"
        self.corrections_file = self.corrections_dir / "corrections.jsonl"
        self.index_file = self.corrections_dir / "correction_index.json"

        # Ensure directory exists
        self.corrections_dir.mkdir(parents=True, exist_ok=True)

        # Load index
        self.index = self._load_index()

    def _generate_correction_hash(self, mistake: str, correction: str, topic: str) -> str:
        """Generate a hash for deduplication based on content."""
        content = f"{mistake.lower().strip()}|{correction.lower().strip()}|{topic.lower()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _find_by_hash(self, content_hash: str) -> Optional[Dict]:
        """Find an existing correction by its content hash."""
        all_corrections = self._load_all_corrections()
        for corr in all_corrections:
            # Check if hash is in the ID or if we need to regenerate
            if content_hash in corr.get("id", ""):
                return corr
            # Also check by regenerating hash from content
            existing_hash = self._generate_correction_hash(
                corr.get("mistake", ""),
                corr.get("correction", ""),
                corr.get("topic", "")
            )
            if existing_hash == content_hash:
                return corr
        return None

    def save_correction(self,
                       mistake: str,
                       correction: str,
                       topic: str,
                       conversation_id: str,
                       context: str = "",
                       importance: str = "medium",
                       entities: Dict = None,
                       user_message: str = "") -> Optional[str]:
        """
        Save a new correction.

        Returns: correction_id (existing ID if duplicate found), None if invalid
        """
        # Validate before saving
        validation_check = {
            "mistake": mistake,
            "correction": correction
        }
        if not self._is_valid_correction(validation_check):
            print(f"Rejected invalid correction: {mistake[:30]} -> {correction[:30]}")
            return None

        # LLM validation (if DGX available)
        llm_result = self._validate_with_llm(mistake, correction, context, user_message)
        llm_validated = False

        if llm_result is None:
            # LLM explicitly rejected
            print(f"LLM rejected correction: {mistake[:30]} -> {correction[:30]}")
            return None
        elif llm_result == "FALLBACK":
            # DGX unavailable — apply strict regex fallback
            if len(mistake.split()) < 2 or len(correction.split()) < 3:
                print(f"Strict regex rejected: {mistake[:30]} -> {correction[:30]}")
                return None
            llm_validated = False
        elif isinstance(llm_result, dict):
            # LLM confirmed + cleaned
            mistake = llm_result.get("mistake", mistake)
            correction = llm_result.get("correction", correction)
            llm_validated = True

        # Generate content hash for deduplication
        content_hash = self._generate_correction_hash(mistake, correction, topic)

        # Check if this correction already exists
        existing = self._find_by_hash(content_hash)
        if existing:
            # Increment applied count instead of creating duplicate
            self.increment_applied_count(existing['id'])
            return existing['id']

        # Generate ID with hash for future dedup lookups
        correction_id = f"corr_{content_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        correction_data = {
            "id": correction_id,
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "mistake": mistake,
            "correction": correction,
            "conversation_id": conversation_id,
            "context": context,
            "importance": importance,
            "entities": entities or {},
            "verified": llm_validated,
            "validation_method": "llm" if llm_validated else "regex_strict",
            "applied_count": 0  # How many times this correction was surfaced
        }

        # Append to JSONL file
        with open(self.corrections_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(correction_data) + '\n')

        # Update index
        self._update_index(correction_data)

        # TRUTH MAINTENANCE: Propagate correction to supersede contradicting facts
        if TRUTH_MAINTENANCE_ENABLED:
            try:
                propagation_result = propagate_correction_to_facts(
                    correction_data,
                    base_path=str(self.base_path)
                )
                # Store propagation results in index for tracking
                if "propagation_results" not in self.index:
                    self.index["propagation_results"] = {}
                self.index["propagation_results"][correction_id] = {
                    "facts_found": propagation_result.get("facts_found", 0),
                    "facts_superseded": propagation_result.get("facts_superseded", 0),
                    "fact_ids": propagation_result.get("fact_ids", []),
                    "propagated_at": datetime.now().isoformat()
                }
                self._save_index()

                if propagation_result.get("facts_superseded", 0) > 0:
                    print(f"[TruthMaintenance] Superseded {propagation_result['facts_superseded']} facts for correction {correction_id}")
            except Exception as e:
                print(f"Warning: Truth maintenance propagation failed: {e}")

        return correction_id

    def get_corrections_by_topic(self, topic: str, limit: int = 10) -> List[Dict]:
        """Get corrections for a specific topic."""
        if topic not in self.index.get("by_topic", {}):
            return []

        correction_ids = self.index["by_topic"][topic]
        corrections = self._load_corrections_by_ids(correction_ids)

        # Sort by importance and recency
        corrections.sort(key=lambda x: (
            {"high": 3, "medium": 2, "low": 1}.get(x["importance"], 0),
            x["timestamp"]
        ), reverse=True)

        return corrections[:limit]

    def search_corrections(self, query: str, limit: int = 10) -> List[Dict]:
        """Search corrections by keyword."""
        query_lower = query.lower()
        all_corrections = self._load_all_corrections()

        # Score each correction
        scored = []
        for corr in all_corrections:
            score = 0
            # Check mistake and correction text
            if query_lower in corr["mistake"].lower():
                score += 3
            if query_lower in corr["correction"].lower():
                score += 3
            if query_lower in corr["context"].lower():
                score += 1
            if query_lower in corr["topic"].lower():
                score += 2

            if score > 0:
                scored.append((score, corr))

        # Sort by score and return top results
        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in scored[:limit]]

    def get_recent_corrections(self, days: int = 30, limit: int = 10) -> List[Dict]:
        """Get corrections from last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        all_corrections = self._load_all_corrections()

        recent = [
            c for c in all_corrections
            if datetime.fromisoformat(c["timestamp"]) > cutoff
        ]

        recent.sort(key=lambda x: x["timestamp"], reverse=True)
        return recent[:limit]

    def get_high_importance_corrections(self, limit: int = 20) -> List[Dict]:
        """Get all high-importance corrections."""
        all_corrections = self._load_all_corrections()
        high_importance = [c for c in all_corrections if c["importance"] == "high"]
        high_importance.sort(key=lambda x: x["timestamp"], reverse=True)
        return high_importance[:limit]

    def increment_applied_count(self, correction_id: str):
        """Increment the count when a correction is surfaced to Claude."""
        # This would require rewriting the JSONL file
        # For now, we'll track in memory in the index
        if "applied_counts" not in self.index:
            self.index["applied_counts"] = {}
        self.index["applied_counts"][correction_id] = \
            self.index["applied_counts"].get(correction_id, 0) + 1
        self._save_index()

    def _load_index(self) -> Dict:
        """Load correction index."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"by_topic": {}, "by_date": {}, "applied_counts": {}}

    def _save_index(self):
        """Save correction index."""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2)

    def _update_index(self, correction: Dict):
        """Update index with new correction."""
        # Index by topic
        topic = correction["topic"]
        if topic not in self.index["by_topic"]:
            self.index["by_topic"][topic] = []
        self.index["by_topic"][topic].append(correction["id"])

        # Index by date
        date_key = correction["timestamp"][:10]  # YYYY-MM-DD
        if date_key not in self.index["by_date"]:
            self.index["by_date"][date_key] = []
        self.index["by_date"][date_key].append(correction["id"])

        self._save_index()

    def _load_all_corrections(self) -> List[Dict]:
        """Load all corrections from JSONL file."""
        corrections = []
        if self.corrections_file.exists():
            with open(self.corrections_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        corrections.append(json.loads(line))
        return corrections

    def _load_corrections_by_ids(self, correction_ids: List[str]) -> List[Dict]:
        """Load specific corrections by ID."""
        all_corrections = self._load_all_corrections()
        id_set = set(correction_ids)
        return [c for c in all_corrections if c["id"] in id_set]

    def get_stats(self) -> Dict:
        """Get correction statistics."""
        all_corrections = self._load_all_corrections()
        return {
            "total_corrections": len(all_corrections),
            "by_topic": {
                topic: len(ids)
                for topic, ids in self.index.get("by_topic", {}).items()
            },
            "by_importance": {
                imp: len([c for c in all_corrections if c["importance"] == imp])
                for imp in ["high", "medium", "low"]
            },
            "most_recent": all_corrections[-1]["timestamp"] if all_corrections else None
        }
