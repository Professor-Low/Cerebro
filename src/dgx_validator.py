#!/usr/bin/env python3
"""
DGX Spark Fact Validator - Validates facts using Ollama on DGX Spark.

Sends new facts to the DGX Spark's Ollama instance for validation.
Returns confidence adjustments and flags obvious errors.

Phase 4.3 of Brain Evolution.
"""

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from config import DATA_DIR

# DGX Spark Ollama configuration
_dgx_host = os.environ.get("CEREBRO_DGX_HOST", "")
DGX_OLLAMA_URL = f"http://{_dgx_host}:11434/api/generate" if _dgx_host else ""
DGX_OLLAMA_MODEL = "llama3:latest"  # Fast model for validation
DGX_TIMEOUT = 30  # 30 second timeout for LLM calls

# Validation results cache
VALIDATION_CACHE_PATH = DATA_DIR / "validation" / "fact_validations.json"


class DGXFactValidator:
    """Validates facts using DGX Spark's Ollama instance."""

    def __init__(self):
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load validation cache."""
        try:
            if VALIDATION_CACHE_PATH.exists():
                with open(VALIDATION_CACHE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except:
            pass
        return {"validations": {}, "_last_updated": None}

    def _save_cache(self):
        """Save validation cache."""
        try:
            VALIDATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.cache["_last_updated"] = datetime.now().isoformat()
            with open(VALIDATION_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Error saving validation cache: {e}")

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API on DGX Spark."""
        if not HAS_REQUESTS:
            return None

        try:
            response = requests.post(
                DGX_OLLAMA_URL,
                json={
                    "model": DGX_OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temp for consistent validation
                        "num_predict": 200,  # Short response
                    }
                },
                timeout=DGX_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"Ollama error: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            print("Ollama timeout")
            return None
        except Exception as e:
            print(f"Ollama error: {e}")
            return None

    def validate_fact(self, fact_content: str, fact_id: str = None) -> Dict[str, Any]:
        """Validate a single fact using Ollama.

        Returns:
            Dict with:
            - valid: bool - whether fact seems valid
            - confidence_adjustment: float - suggested adjustment (-0.3 to +0.2)
            - issues: list - any issues found
            - reasoning: str - explanation
        """
        # Check cache first
        cache_key = fact_content[:100]  # Use truncated content as key
        if cache_key in self.cache.get("validations", {}):
            cached = self.cache["validations"][cache_key]
            # Return cached if less than 7 days old
            if cached.get("validated_at"):
                try:
                    validated = datetime.fromisoformat(cached["validated_at"])
                    if (datetime.now() - validated).days < 7:
                        return cached
                except:
                    pass

        # Build validation prompt
        prompt = f"""Analyze this fact for accuracy and quality. Respond in JSON format only.

FACT: "{fact_content}"

Evaluate:
1. Is this a complete, meaningful statement (not a fragment)?
2. Does it make logical sense?
3. Could it be factually incorrect or contradictory?
4. Is it specific enough to be useful?

Respond with JSON:
{{"valid": true/false, "confidence_adjustment": -0.3 to +0.2, "issues": ["issue1", "issue2"], "reasoning": "explanation"}}

JSON response:"""

        response = self._call_ollama(prompt)

        if not response:
            return {
                "valid": True,
                "confidence_adjustment": 0.0,
                "issues": [],
                "reasoning": "Could not validate (DGX unavailable)",
                "validated_at": datetime.now().isoformat(),
            }

        # Parse response
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
                result["validated_at"] = datetime.now().isoformat()
                result["fact_id"] = fact_id

                # Cache the result
                self.cache["validations"][cache_key] = result
                self._save_cache()

                return result
            else:
                # Couldn't parse - assume valid
                return {
                    "valid": True,
                    "confidence_adjustment": 0.0,
                    "issues": [],
                    "reasoning": f"Validation parse error: {response[:100]}",
                    "validated_at": datetime.now().isoformat(),
                }
        except json.JSONDecodeError:
            return {
                "valid": True,
                "confidence_adjustment": 0.0,
                "issues": [],
                "reasoning": "JSON parse error",
                "validated_at": datetime.now().isoformat(),
            }

    def validate_facts_async(self, facts: List[Dict], callback=None):
        """Validate multiple facts asynchronously.

        Args:
            facts: List of fact dicts with 'content' and optional 'id' keys
            callback: Optional function to call with results
        """
        def _validate():
            results = []
            for fact in facts:
                content = fact.get("content", "")
                fact_id = fact.get("id", "")
                if content:
                    result = self.validate_fact(content, fact_id)
                    results.append(result)

            if callback:
                callback(results)
            return results

        thread = threading.Thread(target=_validate, daemon=True)
        thread.start()
        return thread

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get statistics about validations."""
        validations = self.cache.get("validations", {})
        total = len(validations)
        valid = sum(1 for v in validations.values() if v.get("valid", True))
        invalid = total - valid

        avg_adjustment = 0.0
        if total > 0:
            adjustments = [v.get("confidence_adjustment", 0) for v in validations.values()]
            avg_adjustment = sum(adjustments) / len(adjustments)

        return {
            "total_validated": total,
            "valid_count": valid,
            "invalid_count": invalid,
            "average_adjustment": round(avg_adjustment, 3),
            "last_updated": self.cache.get("_last_updated"),
        }


def validate_new_facts(facts: List[Dict]) -> List[Dict]:
    """Convenience function to validate a list of facts.

    Args:
        facts: List of dicts with 'content' key

    Returns:
        List of validation results
    """
    validator = DGXFactValidator()
    results = []

    for fact in facts:
        content = fact.get("content", "")
        if content:
            result = validator.validate_fact(content, fact.get("id"))
            results.append(result)

    return results


if __name__ == "__main__":
    # Test validation
    validator = DGXFactValidator()

    test_facts = [
        {"content": "User uses Windows 11 as their primary operating system", "id": "test1"},
        {"content": "The NAS IP address is 10.0.0.100", "id": "test2"},
        {"content": "Before I can", "id": "test3"},  # Fragment - should fail
    ]

    print("Testing DGX Fact Validator...")
    for fact in test_facts:
        print(f"\nValidating: {fact['content'][:50]}...")
        result = validator.validate_fact(fact["content"], fact["id"])
        print(f"  Valid: {result.get('valid')}")
        print(f"  Adjustment: {result.get('confidence_adjustment')}")
        print(f"  Issues: {result.get('issues')}")

    print("\n\nValidation Stats:")
    print(json.dumps(validator.get_validation_stats(), indent=2))
