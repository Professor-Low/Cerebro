"""
Causal Extractor - Claude.Me v6.0
Extracts cause-effect relationships from conversations.

Part of Phase 3: Causal Model Building
Uses DGX Spark for LLM-based extraction when available.
"""
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import requests

# DGX Spark configuration
_dgx_host = os.environ.get("CEREBRO_DGX_HOST", "")
DGX_CAUSAL_SERVICE = f"http://{_dgx_host}:8767" if _dgx_host else ""
DGX_TIMEOUT = 30


class CausalExtractor:
    """
    Extract causal relationships from text.

    Uses a combination of:
    1. Pattern matching (fast, local)
    2. DGX Spark LLM service (accurate, requires network)
    """

    # Causal language patterns
    CAUSAL_PATTERNS = [
        # Direct causation
        (r'(?P<cause>.+?)\s+(?:caused|causes|caused by)\s+(?P<effect>.+?)(?:\.|,|$)', 'caused'),
        (r'(?P<cause>.+?)\s+(?:led to|leads to|leading to)\s+(?P<effect>.+?)(?:\.|,|$)', 'led_to'),
        (r'(?P<cause>.+?)\s+(?:resulted in|results in|resulting in)\s+(?P<effect>.+?)(?:\.|,|$)', 'resulted'),

        # Because/due to (effect first)
        (r'(?P<effect>.+?)\s+because\s+(?:of\s+)?(?P<cause>.+?)(?:\.|,|$)', 'because'),
        (r'(?P<effect>.+?)\s+(?:due to|owing to)\s+(?P<cause>.+?)(?:\.|,|$)', 'due_to'),

        # Conditional
        (r'(?:if|when)\s+(?P<cause>.+?)\s+(?:then\s+)?(?P<effect>.+?)(?:\.|,|$)', 'conditional'),
        (r'(?P<cause>.+?)\s+(?:makes|made)\s+(?P<effect>.+?)(?:\.|,|$)', 'makes'),

        # Fix patterns (solution causes resolution)
        (r'(?:fixed|resolved|solved)\s+(?:by|with|using)\s+(?P<cause>.+?),?\s*(?:which\s+)?(?P<effect>.+?)(?:\.|$)', 'fixed_by'),
        (r'(?P<cause>.+?)\s+(?:fixed|resolved|solved)\s+(?:the\s+)?(?P<effect>.+?)(?:\.|,|$)', 'fixed'),

        # Prevention
        (r'(?P<cause>.+?)\s+(?:prevents?|prevented)\s+(?P<effect>.+?)(?:\.|,|$)', 'prevents'),
        (r'(?:to prevent|preventing)\s+(?P<effect>.+?),?\s*(?:we\s+)?(?P<cause>.+?)(?:\.|$)', 'prevent'),
    ]

    # Mechanism indicators (how the causation works)
    MECHANISM_PATTERNS = [
        r'(?:via|through|by means of)\s+(.+?)(?:\.|,|$)',
        r'(?:the mechanism is|works by)\s+(.+?)(?:\.|,|$)',
        r'(?:this happens because|this works because)\s+(.+?)(?:\.|,|$)',
    ]

    # Counterfactual patterns (what would happen otherwise)
    COUNTERFACTUAL_PATTERNS = [
        r'(?:if\s+)?(?:not|without)\s+(.+?),?\s*(?:would|could|might)\s+(.+?)(?:\.|$)',
        r'(?:otherwise|if not)\s+(.+?)(?:\.|$)',
    ]

    def __init__(self, use_dgx: bool = True):
        self.use_dgx = use_dgx
        self._dgx_available = None

    def _check_dgx_available(self) -> bool:
        """Check if DGX causal service is available."""
        if self._dgx_available is not None:
            return self._dgx_available

        try:
            resp = requests.get(f"{DGX_CAUSAL_SERVICE}/health", timeout=5)
            self._dgx_available = resp.status_code == 200
        except:
            self._dgx_available = False

        return self._dgx_available

    def extract_from_text(self, text: str, use_llm: bool = False) -> List[Dict]:
        """
        Extract causal relationships from text.

        Args:
            text: Text to analyze
            use_llm: Whether to use DGX LLM (slower but more accurate)

        Returns:
            List of causal relationships
        """
        results = []

        # Pattern-based extraction (always run)
        pattern_results = self._extract_with_patterns(text)
        results.extend(pattern_results)

        # LLM-based extraction (if enabled and available)
        if use_llm and self.use_dgx and self._check_dgx_available():
            llm_results = self._extract_with_llm(text)
            results.extend(llm_results)

        # Deduplicate
        seen = set()
        unique_results = []
        for r in results:
            key = f"{r['cause'][:50]}::{r['effect'][:50]}"
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return unique_results

    def _extract_with_patterns(self, text: str) -> List[Dict]:
        """Extract causal relationships using regex patterns."""
        results = []

        for pattern, pattern_type in self.CAUSAL_PATTERNS:
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    groups = match.groupdict()

                    cause = groups.get('cause', '').strip()
                    effect = groups.get('effect', '').strip()

                    # Skip if too short
                    if len(cause) < 5 or len(effect) < 5:
                        continue

                    # Skip if too long (likely wrong match)
                    if len(cause) > 200 or len(effect) > 200:
                        continue

                    result = {
                        'cause': cause,
                        'effect': effect,
                        'pattern_type': pattern_type,
                        'confidence': 0.6,  # Pattern-based confidence
                        'extraction_method': 'pattern',
                        'extracted_at': datetime.now().isoformat()
                    }

                    # Try to extract mechanism
                    mechanism = self._extract_mechanism(text, match.start(), match.end())
                    if mechanism:
                        result['mechanism'] = mechanism
                        result['confidence'] += 0.1

                    # Try to extract counterfactual
                    counterfactual = self._extract_counterfactual(text, match.start(), match.end())
                    if counterfactual:
                        result['counterfactual'] = counterfactual

                    results.append(result)
            except:
                continue

        return results

    def _extract_mechanism(self, text: str, start: int, end: int) -> Optional[str]:
        """Try to extract the mechanism from surrounding text."""
        # Look in nearby text (100 chars before and after)
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context = text[context_start:context_end]

        for pattern in self.MECHANISM_PATTERNS:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:200]

        return None

    def _extract_counterfactual(self, text: str, start: int, end: int) -> Optional[str]:
        """Try to extract counterfactual from surrounding text."""
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 150)
        context = text[context_start:context_end]

        for pattern in self.COUNTERFACTUAL_PATTERNS:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(0).strip()[:200]

        return None

    def _extract_with_llm(self, text: str) -> List[Dict]:
        """Extract causal relationships using DGX Spark LLM."""
        try:
            resp = requests.post(
                f"{DGX_CAUSAL_SERVICE}/extract",
                json={"text": text},
                timeout=DGX_TIMEOUT
            )

            if resp.status_code != 200:
                return []

            data = resp.json()
            results = []

            for item in data.get("causal_links", []):
                results.append({
                    'cause': item.get('cause', ''),
                    'effect': item.get('effect', ''),
                    'mechanism': item.get('mechanism'),
                    'counterfactual': item.get('counterfactual'),
                    'interventions': item.get('interventions', []),
                    'confidence': item.get('confidence', 0.7),
                    'extraction_method': 'llm',
                    'extracted_at': datetime.now().isoformat()
                })

            return results
        except Exception as e:
            print(f"[CausalExtractor] LLM extraction failed: {e}")
            return []

    def extract_from_conversation(self, messages: List[Dict]) -> List[Dict]:
        """Extract causal relationships from a conversation."""
        results = []

        # Combine all message content
        full_text = "\n".join([
            msg.get("content", "") for msg in messages
        ])

        # Extract from full text
        results = self.extract_from_text(full_text)

        # Also look for problem-solution patterns (these imply causality)
        for i, msg in enumerate(messages):
            content = msg.get("content", "")

            # Look for fix statements
            fix_patterns = [
                r'fixed (?:by|with|using) (.+?)(?:\.|,|\n)',
                r'the (?:fix|solution) was to (.+?)(?:\.|,|\n)',
                r'resolved (?:by|with) (.+?)(?:\.|,|\n)',
            ]

            for pattern in fix_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Look backward for the problem
                    problem = self._find_problem_before(messages, i)
                    if problem:
                        results.append({
                            'cause': match.group(1).strip()[:200],
                            'effect': f"Resolved: {problem[:100]}",
                            'pattern_type': 'fix',
                            'confidence': 0.75,
                            'extraction_method': 'conversation_pattern',
                            'extracted_at': datetime.now().isoformat()
                        })

        return results

    def _find_problem_before(self, messages: List[Dict], current_idx: int) -> Optional[str]:
        """Find problem mentioned before the current message."""
        problem_indicators = [
            r'(?:problem|issue|error|bug)(?:\s+is)?:?\s*(.+?)(?:\.|,|\n)',
            r'(?:not working|broken|fails?)(?:\s+with)?:?\s*(.+?)(?:\.|,|\n)',
            r"(?:can't|cannot|couldn't|won't)(.+?)(?:\.|,|\n)",
        ]

        # Look back up to 5 messages
        for i in range(max(0, current_idx - 5), current_idx):
            content = messages[i].get("content", "")
            for pattern in problem_indicators:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1).strip()[:200]

        return None


def extract_interventions_from_text(text: str) -> List[str]:
    """
    Extract potential interventions mentioned in text.

    Interventions are actions that could prevent/fix something.
    """
    interventions = []

    patterns = [
        r'(?:you should|try to|need to|can)\s+(.+?)(?:\.|,|to fix)',
        r'(?:the fix is to|the solution is to|we need to)\s+(.+?)(?:\.|,|\n)',
        r'(?:increase|decrease|change|modify|update|set)\s+(?:the\s+)?(.+?)(?:\.|,|\n)',
        r'(?:disable|enable)\s+(.+?)(?:\.|,|\n)',
        r'(?:add|remove|install|uninstall)\s+(.+?)(?:\.|,|\n)',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            intervention = match.group(1).strip()
            if 5 < len(intervention) < 150:
                interventions.append(intervention)

    return list(set(interventions))
