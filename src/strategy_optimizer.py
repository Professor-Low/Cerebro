"""
Strategy Optimizer - Claude.Me v6.0
Auto-tune parameters based on meta-learning data.

Part of Phase 8: Meta-Learning
"""
import hashlib
import json
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class Experiment:
    """Represents an A/B test experiment."""

    def __init__(
        self,
        experiment_id: str,
        name: str,
        strategy_a: str,
        strategy_b: str,
        traffic_split: float = 0.5,
        min_queries: int = 100,
        status: str = "active"
    ):
        self.experiment_id = experiment_id
        self.name = name
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.traffic_split = traffic_split  # Proportion going to strategy_a
        self.min_queries = min_queries
        self.status = status  # active, completed, cancelled
        self.created_at = datetime.now().isoformat()
        self.completed_at = None
        self.results = {
            "a_queries": 0,
            "a_successes": 0,
            "b_queries": 0,
            "b_successes": 0
        }
        self.winner = None

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "strategy_a": self.strategy_a,
            "strategy_b": self.strategy_b,
            "traffic_split": self.traffic_split,
            "min_queries": self.min_queries,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "results": self.results,
            "winner": self.winner
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Experiment":
        exp = cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            strategy_a=data["strategy_a"],
            strategy_b=data["strategy_b"],
            traffic_split=data.get("traffic_split", 0.5),
            min_queries=data.get("min_queries", 100),
            status=data.get("status", "active")
        )
        exp.created_at = data.get("created_at", exp.created_at)
        exp.completed_at = data.get("completed_at")
        exp.results = data.get("results", exp.results)
        exp.winner = data.get("winner")
        return exp


class ParameterTuning:
    """Represents a parameter tuning configuration."""

    def __init__(
        self,
        param_name: str,
        current_value: float,
        min_value: float,
        max_value: float,
        step_size: float,
        optimization_target: str = "success_rate"
    ):
        self.param_name = param_name
        self.current_value = current_value
        self.min_value = min_value
        self.max_value = max_value
        self.step_size = step_size
        self.optimization_target = optimization_target
        self.history = []  # List of (value, score) tuples

    def to_dict(self) -> Dict:
        return {
            "param_name": self.param_name,
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step_size": self.step_size,
            "optimization_target": self.optimization_target,
            "history": self.history
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ParameterTuning":
        pt = cls(
            param_name=data["param_name"],
            current_value=data["current_value"],
            min_value=data["min_value"],
            max_value=data["max_value"],
            step_size=data["step_size"],
            optimization_target=data.get("optimization_target", "success_rate")
        )
        pt.history = data.get("history", [])
        return pt


class StrategyOptimizer:
    """
    Auto-tune retrieval strategy parameters.

    Capabilities:
    - Run A/B experiments
    - Gradient-based parameter tuning
    - Automatic strategy selection
    - Performance monitoring
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.meta_path = self.base_path / "meta_learning"
        self.optimizer_file = self.meta_path / "optimizer_state.json"
        self._lock = threading.Lock()
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.meta_path.mkdir(parents=True, exist_ok=True)

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        ts = datetime.now().isoformat()
        return f"{prefix}_{hashlib.sha256(ts.encode()).hexdigest()[:8]}"

    def _load_state(self) -> Dict:
        """Load optimizer state."""
        if self.optimizer_file.exists():
            try:
                with open(self.optimizer_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            "experiments": [],
            "parameter_tunings": {},
            "auto_optimize": False,
            "optimization_history": [],
            "updated_at": datetime.now().isoformat()
        }

    def _save_state(self, data: Dict):
        """Save optimizer state."""
        data["updated_at"] = datetime.now().isoformat()
        with self._lock:
            with open(self.optimizer_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

    def create_experiment(
        self,
        name: str,
        strategy_a: str,
        strategy_b: str,
        traffic_split: float = 0.5,
        min_queries: int = 100
    ) -> Dict:
        """
        Create a new A/B experiment.

        Args:
            name: Experiment name
            strategy_a: First strategy to test
            strategy_b: Second strategy to test
            traffic_split: Proportion of traffic to strategy_a (0-1)
            min_queries: Minimum queries before concluding
        """
        exp = Experiment(
            experiment_id=self._generate_id("exp"),
            name=name,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            traffic_split=traffic_split,
            min_queries=min_queries
        )

        state = self._load_state()
        state["experiments"].append(exp.to_dict())
        self._save_state(state)

        return {
            "success": True,
            "experiment_id": exp.experiment_id,
            "name": name,
            "strategies": [strategy_a, strategy_b]
        }

    def get_experiment_strategy(self, experiment_id: str = None) -> Optional[str]:
        """
        Get which strategy to use for an experiment.
        Uses random assignment based on traffic split.
        """
        state = self._load_state()

        # Find active experiment
        active_exp = None
        for exp_data in state.get("experiments", []):
            if exp_data.get("status") == "active":
                if experiment_id is None or exp_data.get("experiment_id") == experiment_id:
                    active_exp = Experiment.from_dict(exp_data)
                    break

        if not active_exp:
            return None

        # Random assignment
        if random.random() < active_exp.traffic_split:
            return active_exp.strategy_a
        return active_exp.strategy_b

    def record_experiment_result(
        self,
        experiment_id: str,
        strategy_used: str,
        success: bool
    ) -> Dict:
        """Record a result for an experiment."""
        state = self._load_state()

        for i, exp_data in enumerate(state.get("experiments", [])):
            if exp_data.get("experiment_id") == experiment_id:
                if strategy_used == exp_data["strategy_a"]:
                    exp_data["results"]["a_queries"] += 1
                    if success:
                        exp_data["results"]["a_successes"] += 1
                elif strategy_used == exp_data["strategy_b"]:
                    exp_data["results"]["b_queries"] += 1
                    if success:
                        exp_data["results"]["b_successes"] += 1

                state["experiments"][i] = exp_data

                # Check if experiment should conclude
                total = exp_data["results"]["a_queries"] + exp_data["results"]["b_queries"]
                if total >= exp_data.get("min_queries", 100):
                    self._conclude_experiment(state, i)

                self._save_state(state)
                return {"success": True, "recorded": True}

        return {"error": f"Experiment {experiment_id} not found"}

    def _conclude_experiment(self, state: Dict, exp_index: int):
        """Conclude an experiment and determine winner."""
        exp_data = state["experiments"][exp_index]

        a_rate = 0
        if exp_data["results"]["a_queries"] > 0:
            a_rate = exp_data["results"]["a_successes"] / exp_data["results"]["a_queries"]

        b_rate = 0
        if exp_data["results"]["b_queries"] > 0:
            b_rate = exp_data["results"]["b_successes"] / exp_data["results"]["b_queries"]

        if a_rate > b_rate:
            exp_data["winner"] = exp_data["strategy_a"]
        elif b_rate > a_rate:
            exp_data["winner"] = exp_data["strategy_b"]
        else:
            exp_data["winner"] = "tie"

        exp_data["status"] = "completed"
        exp_data["completed_at"] = datetime.now().isoformat()

        state["experiments"][exp_index] = exp_data

        # Add to optimization history
        state["optimization_history"].append({
            "type": "experiment",
            "experiment_id": exp_data["experiment_id"],
            "winner": exp_data["winner"],
            "a_rate": round(a_rate, 3),
            "b_rate": round(b_rate, 3),
            "timestamp": datetime.now().isoformat()
        })

    def get_experiment(self, experiment_id: str) -> Dict:
        """Get experiment details."""
        state = self._load_state()

        for exp_data in state.get("experiments", []):
            if exp_data.get("experiment_id") == experiment_id:
                # Add calculated rates
                exp_data["a_success_rate"] = 0
                exp_data["b_success_rate"] = 0
                if exp_data["results"]["a_queries"] > 0:
                    exp_data["a_success_rate"] = round(
                        exp_data["results"]["a_successes"] / exp_data["results"]["a_queries"], 3
                    )
                if exp_data["results"]["b_queries"] > 0:
                    exp_data["b_success_rate"] = round(
                        exp_data["results"]["b_successes"] / exp_data["results"]["b_queries"], 3
                    )
                return exp_data

        return {"error": f"Experiment {experiment_id} not found"}

    def list_experiments(self, status: str = None) -> List[Dict]:
        """List experiments, optionally filtered by status."""
        state = self._load_state()
        experiments = state.get("experiments", [])

        if status:
            experiments = [e for e in experiments if e.get("status") == status]

        return experiments

    def cancel_experiment(self, experiment_id: str) -> Dict:
        """Cancel an active experiment."""
        state = self._load_state()

        for i, exp_data in enumerate(state.get("experiments", [])):
            if exp_data.get("experiment_id") == experiment_id:
                if exp_data.get("status") != "active":
                    return {"error": "Experiment is not active"}

                exp_data["status"] = "cancelled"
                exp_data["completed_at"] = datetime.now().isoformat()
                state["experiments"][i] = exp_data
                self._save_state(state)
                return {"success": True, "experiment_id": experiment_id}

        return {"error": f"Experiment {experiment_id} not found"}

    def setup_parameter_tuning(
        self,
        param_name: str,
        current_value: float,
        min_value: float,
        max_value: float,
        step_size: float
    ) -> Dict:
        """
        Setup automatic parameter tuning.

        Args:
            param_name: Parameter to tune (e.g., "hybrid_alpha")
            current_value: Current parameter value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            step_size: Step size for adjustments
        """
        tuning = ParameterTuning(
            param_name=param_name,
            current_value=current_value,
            min_value=min_value,
            max_value=max_value,
            step_size=step_size
        )

        state = self._load_state()
        state["parameter_tunings"][param_name] = tuning.to_dict()
        self._save_state(state)

        return {
            "success": True,
            "param_name": param_name,
            "current_value": current_value,
            "range": [min_value, max_value]
        }

    def record_parameter_performance(
        self,
        param_name: str,
        value: float,
        score: float
    ) -> Dict:
        """Record performance for a parameter value."""
        state = self._load_state()

        if param_name not in state.get("parameter_tunings", {}):
            return {"error": f"Parameter {param_name} not set up for tuning"}

        tuning = state["parameter_tunings"][param_name]
        tuning["history"].append({
            "value": value,
            "score": score,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last 100 records
        tuning["history"] = tuning["history"][-100:]

        state["parameter_tunings"][param_name] = tuning
        self._save_state(state)

        return {"success": True, "recorded": True}

    def suggest_parameter_value(self, param_name: str) -> Dict:
        """
        Suggest next parameter value to try based on history.
        Uses simple hill-climbing with exploration.
        """
        state = self._load_state()

        if param_name not in state.get("parameter_tunings", {}):
            return {"error": f"Parameter {param_name} not set up for tuning"}

        tuning = ParameterTuning.from_dict(state["parameter_tunings"][param_name])

        if len(tuning.history) < 3:
            # Not enough data - explore
            return {
                "suggested_value": tuning.current_value,
                "reason": "Not enough data for optimization",
                "exploration": True
            }

        # Find best value so far
        best_record = max(tuning.history, key=lambda x: x["score"])
        best_value = best_record["value"]
        best_score = best_record["score"]

        # Exploration vs exploitation
        if random.random() < 0.2:  # 20% exploration
            # Try a random value in range
            suggested = random.uniform(tuning.min_value, tuning.max_value)
            suggested = round(suggested / tuning.step_size) * tuning.step_size
            return {
                "suggested_value": suggested,
                "reason": "Exploration",
                "best_so_far": best_value,
                "best_score": best_score
            }

        # Exploitation - move towards best with small perturbation
        direction = random.choice([-1, 1])
        suggested = best_value + (direction * tuning.step_size)
        suggested = max(tuning.min_value, min(tuning.max_value, suggested))

        return {
            "suggested_value": round(suggested, 3),
            "reason": "Hill climbing from best",
            "best_so_far": best_value,
            "best_score": best_score
        }

    def enable_auto_optimize(self, enabled: bool = True) -> Dict:
        """Enable or disable automatic optimization."""
        state = self._load_state()
        state["auto_optimize"] = enabled
        self._save_state(state)
        return {"success": True, "auto_optimize": enabled}

    def get_optimization_history(self, limit: int = 20) -> List[Dict]:
        """Get recent optimization history."""
        state = self._load_state()
        history = state.get("optimization_history", [])
        return history[-limit:]

    def get_stats(self) -> Dict:
        """Get optimizer statistics."""
        state = self._load_state()

        active_experiments = [
            e for e in state.get("experiments", [])
            if e.get("status") == "active"
        ]

        completed_experiments = [
            e for e in state.get("experiments", [])
            if e.get("status") == "completed"
        ]

        return {
            "total_experiments": len(state.get("experiments", [])),
            "active_experiments": len(active_experiments),
            "completed_experiments": len(completed_experiments),
            "parameters_tuning": list(state.get("parameter_tunings", {}).keys()),
            "auto_optimize": state.get("auto_optimize", False),
            "optimization_events": len(state.get("optimization_history", []))
        }
