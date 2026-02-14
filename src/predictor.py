"""
Predictor - Claude.Me v6.0
Use causal model and history for predictions.

Part of Phase 7: Predictive Simulation
Uses DGX Spark for LLM-powered predictions when available.
"""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

# DGX Spark configuration
_dgx_host = os.environ.get("CEREBRO_DGX_HOST", "")
DGX_PREDICTION_SERVICE = f"http://{_dgx_host}:8771" if _dgx_host else ""
DGX_TIMEOUT = 45


class Prediction:
    """Represents a prediction about future outcomes."""

    def __init__(
        self,
        prediction_id: str,
        context: str,
        prediction: str,
        confidence: float = 0.5,
        reasoning: Dict = None,
        suggested_mitigation: str = None,
        outcome: str = None
    ):
        self.prediction_id = prediction_id
        self.context = context
        self.prediction = prediction
        self.confidence = confidence
        self.reasoning = reasoning or {}
        self.suggested_mitigation = suggested_mitigation
        self.outcome = outcome  # None, "correct", "incorrect", "partial"
        self.created_at = datetime.now().isoformat()
        self.verified_at = None

    def to_dict(self) -> Dict:
        return {
            "prediction_id": self.prediction_id,
            "context": self.context,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "suggested_mitigation": self.suggested_mitigation,
            "outcome": self.outcome,
            "created_at": self.created_at,
            "verified_at": self.verified_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Prediction":
        pred = cls(
            prediction_id=data["prediction_id"],
            context=data["context"],
            prediction=data["prediction"],
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", {}),
            suggested_mitigation=data.get("suggested_mitigation"),
            outcome=data.get("outcome")
        )
        pred.created_at = data.get("created_at", pred.created_at)
        pred.verified_at = data.get("verified_at")
        return pred


class Predictor:
    """
    Make predictions using causal model and history.

    Capabilities:
    - Predict outcomes based on causal model
    - Warn about likely failures
    - Suggest mitigations
    - Track prediction accuracy
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.predictions_path = self.base_path / "predictions"
        self._dgx_available = None
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.predictions_path.mkdir(parents=True, exist_ok=True)

    def _generate_id(self, context: str) -> str:
        """Generate prediction ID."""
        ts = datetime.now().isoformat()
        return f"pred_{hashlib.sha256(f'{context}{ts}'.encode()).hexdigest()[:10]}"

    def _check_dgx_available(self) -> bool:
        """Check if DGX prediction service is available."""
        if self._dgx_available is not None:
            return self._dgx_available

        try:
            resp = requests.get(f"{DGX_PREDICTION_SERVICE}/health", timeout=5)
            self._dgx_available = resp.status_code == 200
        except:
            self._dgx_available = False

        return self._dgx_available

    def save_prediction(self, pred: Prediction) -> str:
        """Save a prediction."""
        pred_file = self.predictions_path / f"{pred.prediction_id}.json"
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(pred.to_dict(), f, indent=2)
        return pred.prediction_id

    def get_prediction(self, pred_id: str) -> Optional[Prediction]:
        """Get a prediction by ID."""
        pred_file = self.predictions_path / f"{pred_id}.json"
        if not pred_file.exists():
            return None

        try:
            with open(pred_file) as f:
                data = json.load(f)
            return Prediction.from_dict(data)
        except:
            return None

    def predict_from_causal(self, action: str, context: str = None) -> Prediction:
        """
        Make a prediction based on causal model.

        Args:
            action: What action is being taken
            context: Additional context
        """
        from causal_model import CausalModelManager

        cm = CausalModelManager(base_path=str(self.base_path))

        # Find effects of this action
        effects = cm.find_effects(action, threshold=0.4)
        cm.find_causes(action, threshold=0.4)

        reasoning = {
            "causal_links_used": [],
            "similar_episodes": [],
            "failure_pattern_match": 0.0
        }

        prediction_text = ""
        confidence = 0.4
        mitigation = None

        if effects:
            # Action has known effects
            best_effect = effects[0]
            prediction_text = f"Based on causal model: {action} will likely result in '{best_effect.effect}'"
            confidence = best_effect.confidence
            reasoning["causal_links_used"].append(best_effect.link_id)

            if best_effect.interventions:
                mitigation = f"Consider: {best_effect.interventions[0]}"
        else:
            prediction_text = f"No direct causal links found for: {action}"

        # Check for similar past episodes
        similar = self._find_similar_episodes(action)
        if similar:
            reasoning["similar_episodes"] = similar[:3]
            confidence = min(1.0, confidence + 0.1)

        pred = Prediction(
            prediction_id=self._generate_id(action),
            context=f"{action} | {context or ''}",
            prediction=prediction_text,
            confidence=confidence,
            reasoning=reasoning,
            suggested_mitigation=mitigation
        )

        self.save_prediction(pred)
        return pred

    def _find_similar_episodes(self, action: str) -> List[str]:
        """Find similar past episodes."""
        similar = []
        action_lower = action.lower()
        action_words = set(action_lower.split())

        episodic_path = self.base_path / "episodic"
        if episodic_path.exists():
            for ep_file in episodic_path.glob("ep_*.json"):
                try:
                    with open(ep_file) as f:
                        ep = json.load(f)

                    event_words = set(ep.get("event", "").lower().split())
                    if len(action_words.intersection(event_words)) >= 2:
                        similar.append(ep.get("id", ""))
                except:
                    pass

        return similar[:5]

    def predict_with_llm(self, action: str, context: str = None) -> Prediction:
        """Make a prediction using DGX LLM."""
        if not self._check_dgx_available():
            return self.predict_from_causal(action, context)

        try:
            resp = requests.post(
                f"{DGX_PREDICTION_SERVICE}/predict",
                json={
                    "action": action,
                    "context": context,
                    "history_summary": self._get_relevant_history(action)
                },
                timeout=DGX_TIMEOUT
            )

            if resp.status_code != 200:
                return self.predict_from_causal(action, context)

            data = resp.json()

            pred = Prediction(
                prediction_id=self._generate_id(action),
                context=f"{action} | {context or ''}",
                prediction=data.get("prediction", ""),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", {}),
                suggested_mitigation=data.get("mitigation")
            )

            self.save_prediction(pred)
            return pred

        except Exception as e:
            print(f"[Predictor] LLM prediction failed: {e}")
            return self.predict_from_causal(action, context)

    def _get_relevant_history(self, action: str) -> str:
        """Get relevant history for prediction."""
        parts = []

        # Recent episodes
        episodic_path = self.base_path / "episodic"
        if episodic_path.exists():
            action_words = set(action.lower().split())
            for ep_file in sorted(episodic_path.glob("ep_*.json"), reverse=True)[:10]:
                try:
                    with open(ep_file) as f:
                        ep = json.load(f)

                    event_words = set(ep.get("event", "").lower().split())
                    if len(action_words.intersection(event_words)) >= 1:
                        parts.append(f"Past: {ep.get('event', '')[:100]} -> {ep.get('outcome', 'unknown')[:50]}")
                except:
                    pass

        return "\n".join(parts[:5])

    def verify_prediction(self, pred_id: str, outcome: str) -> Dict:
        """
        Verify a prediction's accuracy.

        Args:
            pred_id: Prediction ID
            outcome: "correct", "incorrect", "partial"
        """
        pred = self.get_prediction(pred_id)
        if not pred:
            return {"error": f"Prediction {pred_id} not found"}

        pred.outcome = outcome
        pred.verified_at = datetime.now().isoformat()
        self.save_prediction(pred)

        return {
            "prediction_id": pred_id,
            "outcome": outcome,
            "original_confidence": pred.confidence
        }

    def get_accuracy_stats(self) -> Dict:
        """Get prediction accuracy statistics."""
        total = 0
        correct = 0
        incorrect = 0
        partial = 0
        unverified = 0

        for pred_file in self.predictions_path.glob("pred_*.json"):
            try:
                with open(pred_file) as f:
                    pred = json.load(f)

                total += 1
                outcome = pred.get("outcome")

                if outcome == "correct":
                    correct += 1
                elif outcome == "incorrect":
                    incorrect += 1
                elif outcome == "partial":
                    partial += 1
                else:
                    unverified += 1
            except:
                continue

        verified = total - unverified
        accuracy = correct / verified if verified > 0 else 0

        return {
            "total_predictions": total,
            "verified": verified,
            "correct": correct,
            "incorrect": incorrect,
            "partial": partial,
            "unverified": unverified,
            "accuracy": round(accuracy, 2)
        }

    def get_stats(self) -> Dict:
        """Get predictor statistics."""
        return {
            "accuracy": self.get_accuracy_stats(),
            "dgx_available": self._check_dgx_available() if self._dgx_available is None else self._dgx_available
        }
