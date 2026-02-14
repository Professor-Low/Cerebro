"""
Introspection - Claude.Me v6.0
Monitor reasoning quality and detect issues in real-time.

Part of Phase 9: Continuous Self-Modeling
"""
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests

# DGX Spark configuration
_dgx_host = os.environ.get("CEREBRO_DGX_HOST", "")
DGX_SELF_EVAL_SERVICE = f"http://{_dgx_host}:8770" if _dgx_host else ""
DGX_TIMEOUT = 30


class IntrospectionResult:
    """Result of an introspection analysis."""

    def __init__(
        self,
        result_id: str,
        analysis_type: str,
        content_analyzed: str,
        findings: Dict,
        recommendations: List[str] = None,
        confidence: float = 0.5
    ):
        self.result_id = result_id
        self.analysis_type = analysis_type
        self.content_analyzed = content_analyzed[:200]  # Truncate for storage
        self.findings = findings
        self.recommendations = recommendations or []
        self.confidence = confidence
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "analysis_type": self.analysis_type,
            "content_analyzed": self.content_analyzed,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "IntrospectionResult":
        result = cls(
            result_id=data["result_id"],
            analysis_type=data["analysis_type"],
            content_analyzed=data.get("content_analyzed", ""),
            findings=data.get("findings", {}),
            recommendations=data.get("recommendations", []),
            confidence=data.get("confidence", 0.5)
        )
        result.timestamp = data.get("timestamp", result.timestamp)
        return result


class Introspector:
    """
    Monitor reasoning quality in real-time.

    Capabilities:
    - Analyze reasoning for logical consistency
    - Detect potential hallucinations
    - Evaluate evidence sufficiency
    - Monitor confidence calibration
    - Integrate with DGX for deep analysis
    """

    # Patterns indicating potential issues
    CONTRADICTION_INDICATORS = [
        (r"but (?:I|we) (?:also|just) said", "potential_contradiction"),
        (r"(?:actually|no,? wait)", "self_correction"),
        (r"I (?:think|believe).*but also", "hedging"),
        (r"(?:on one hand|on the other)", "ambivalence"),
    ]

    HALLUCINATION_PATTERNS = [
        (r"the exact (?:date|number|quote)", "fabrication_risk"),
        (r"(?:definitely|certainly|absolutely) (?:is|was|will)", "overconfidence"),
        (r"(?:according to|as stated in)(?! the (?:conversation|user|code))", "citation_risk"),
        (r"I remember (?:when|that|seeing)", "false_memory_risk"),
    ]

    EVIDENCE_PATTERNS = [
        (r"based on (?:the|this)", "has_evidence"),
        (r"from (?:the|this) (?:conversation|context|code|file)", "grounded"),
        (r"I (?:assume|guess|speculate)", "speculation"),
        (r"without (?:more|additional) (?:information|context)", "evidence_gap"),
    ]

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.introspection_path = self.base_path / "self_model" / "introspection"
        self._dgx_available = None
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.introspection_path.mkdir(parents=True, exist_ok=True)

    def _generate_id(self) -> str:
        """Generate result ID."""
        ts = datetime.now().isoformat()
        return f"intro_{hashlib.sha256(ts.encode()).hexdigest()[:10]}"

    def _check_dgx_available(self) -> bool:
        """Check if DGX self-eval service is available."""
        if self._dgx_available is not None:
            return self._dgx_available

        try:
            resp = requests.get(f"{DGX_SELF_EVAL_SERVICE}/health", timeout=5)
            self._dgx_available = resp.status_code == 200
        except:
            self._dgx_available = False

        return self._dgx_available

    def analyze_reasoning(self, text: str, context: str = None) -> IntrospectionResult:
        """
        Analyze reasoning text for quality issues.

        Args:
            text: The reasoning/response text to analyze
            context: Optional context (original question, etc.)
        """
        text_lower = text.lower()

        findings = {
            "contradictions": [],
            "hallucination_risks": [],
            "evidence_status": [],
            "quality_score": 0.7
        }

        # Check for contradictions
        for pattern, issue_type in self.CONTRADICTION_INDICATORS:
            matches = re.findall(pattern, text_lower)
            if matches:
                findings["contradictions"].append({
                    "type": issue_type,
                    "count": len(matches)
                })

        # Check for hallucination risks
        for pattern, risk_type in self.HALLUCINATION_PATTERNS:
            matches = re.findall(pattern, text_lower)
            if matches:
                findings["hallucination_risks"].append({
                    "type": risk_type,
                    "count": len(matches)
                })

        # Check evidence grounding
        for pattern, status in self.EVIDENCE_PATTERNS:
            matches = re.findall(pattern, text_lower)
            if matches:
                findings["evidence_status"].append({
                    "type": status,
                    "count": len(matches)
                })

        # Calculate quality score
        penalty = 0
        penalty += len(findings["contradictions"]) * 0.1
        penalty += len(findings["hallucination_risks"]) * 0.15
        penalty += sum(1 for e in findings["evidence_status"] if e["type"] in ["speculation", "evidence_gap"]) * 0.1

        # Bonus for grounded evidence
        grounded = sum(1 for e in findings["evidence_status"] if e["type"] in ["has_evidence", "grounded"])
        bonus = grounded * 0.05

        findings["quality_score"] = max(0.1, min(1.0, 0.7 - penalty + bonus))

        # Generate recommendations
        recommendations = []
        if findings["contradictions"]:
            recommendations.append("Review for logical consistency")
        if findings["hallucination_risks"]:
            recommendations.append("Verify factual claims before stating")
        if any(e["type"] == "speculation" for e in findings["evidence_status"]):
            recommendations.append("Seek additional evidence or acknowledge uncertainty")

        result = IntrospectionResult(
            result_id=self._generate_id(),
            analysis_type="reasoning",
            content_analyzed=text,
            findings=findings,
            recommendations=recommendations,
            confidence=findings["quality_score"]
        )

        self._save_result(result)
        return result

    def evaluate_with_llm(self, text: str, context: str = None) -> IntrospectionResult:
        """
        Use DGX LLM for deep reasoning evaluation.
        Falls back to local analysis if DGX unavailable.
        """
        if not self._check_dgx_available():
            return self.analyze_reasoning(text, context)

        try:
            resp = requests.post(
                f"{DGX_SELF_EVAL_SERVICE}/evaluate",
                json={
                    "text": text,
                    "context": context,
                    "evaluation_type": "reasoning_quality"
                },
                timeout=DGX_TIMEOUT
            )

            if resp.status_code != 200:
                return self.analyze_reasoning(text, context)

            data = resp.json()

            result = IntrospectionResult(
                result_id=self._generate_id(),
                analysis_type="llm_evaluation",
                content_analyzed=text,
                findings=data.get("findings", {}),
                recommendations=data.get("recommendations", []),
                confidence=data.get("confidence", 0.5)
            )

            self._save_result(result)
            return result

        except Exception as e:
            print(f"[Introspector] LLM evaluation failed: {e}")
            return self.analyze_reasoning(text, context)

    def detect_hallucination_risk(self, text: str) -> Dict:
        """
        Specifically analyze text for hallucination risk.
        Returns risk assessment and specific concerns.
        """
        text_lower = text.lower()

        risks = {
            "overall_risk": 0.0,
            "specific_risks": [],
            "high_risk_phrases": []
        }

        # Check each hallucination pattern
        for pattern, risk_type in self.HALLUCINATION_PATTERNS:
            matches = re.findall(pattern, text_lower)
            if matches:
                risks["specific_risks"].append({
                    "type": risk_type,
                    "severity": "high" if risk_type in ["fabrication_risk", "overconfidence"] else "medium"
                })

        # Check for specific fact-like claims
        fact_patterns = [
            r'\d{4}[-/]\d{2}[-/]\d{2}',  # Dates
            r'\d+%',  # Percentages
            r'\$[\d,]+',  # Prices
            r'"[^"]{20,}"',  # Quotes
        ]

        for pattern in fact_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                risks["high_risk_phrases"].append(match)

        # Calculate overall risk
        risk_score = len(risks["specific_risks"]) * 0.15
        risk_score += len(risks["high_risk_phrases"]) * 0.1
        risks["overall_risk"] = min(1.0, risk_score)

        return risks

    def check_confidence_calibration(
        self,
        stated_confidence: float,
        actual_outcome: bool
    ) -> Dict:
        """
        Check if stated confidence matches actual outcomes.
        Helps calibrate future confidence assessments.
        """
        # Load calibration history
        calibration_file = self.introspection_path / "calibration.json"
        history = []

        if calibration_file.exists():
            try:
                with open(calibration_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                pass

        # Add new data point
        history.append({
            "stated_confidence": stated_confidence,
            "actual_outcome": actual_outcome,
            "timestamp": datetime.now().isoformat()
        })

        # Keep last 100 data points
        history = history[-100:]

        # Save updated history
        with open(calibration_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

        # Calculate calibration metrics
        if len(history) < 5:
            return {
                "calibration_score": 0.5,
                "message": "Not enough data for calibration"
            }

        # Group by confidence buckets
        buckets = {"low": [], "medium": [], "high": []}
        for entry in history:
            conf = entry["stated_confidence"]
            if conf < 0.4:
                buckets["low"].append(entry["actual_outcome"])
            elif conf < 0.7:
                buckets["medium"].append(entry["actual_outcome"])
            else:
                buckets["high"].append(entry["actual_outcome"])

        # Calculate success rate per bucket
        bucket_rates = {}
        for bucket_name, outcomes in buckets.items():
            if outcomes:
                bucket_rates[bucket_name] = sum(outcomes) / len(outcomes)

        # Calculate calibration score (good calibration = high conf -> high success)
        calibration_score = 0.5
        if bucket_rates.get("low") is not None and bucket_rates.get("high") is not None:
            # Higher is better: high confidence should have higher success
            calibration_score = 0.5 + (bucket_rates.get("high", 0.5) - bucket_rates.get("low", 0.5)) * 0.5

        return {
            "calibration_score": round(calibration_score, 3),
            "bucket_rates": bucket_rates,
            "total_samples": len(history),
            "recommendation": "Well calibrated" if calibration_score > 0.6 else "Consider adjusting confidence"
        }

    def _save_result(self, result: IntrospectionResult):
        """Save introspection result."""
        result_file = self.introspection_path / f"{result.result_id}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)

    def get_recent_results(self, limit: int = 10, analysis_type: str = None) -> List[Dict]:
        """Get recent introspection results."""
        results = []

        for result_file in sorted(self.introspection_path.glob("intro_*.json"), reverse=True):
            if len(results) >= limit:
                break

            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if analysis_type and data.get("analysis_type") != analysis_type:
                    continue

                results.append(data)
            except:
                continue

        return results

    def get_stats(self) -> Dict:
        """Get introspection statistics."""
        total_analyses = 0
        avg_quality = 0
        by_type = {}

        for result_file in self.introspection_path.glob("intro_*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                total_analyses += 1
                avg_quality += data.get("confidence", 0.5)

                atype = data.get("analysis_type", "unknown")
                by_type[atype] = by_type.get(atype, 0) + 1
            except:
                continue

        if total_analyses > 0:
            avg_quality /= total_analyses

        return {
            "total_analyses": total_analyses,
            "average_quality_score": round(avg_quality, 3),
            "by_type": by_type,
            "dgx_available": self._check_dgx_available() if self._dgx_available is None else self._dgx_available
        }
