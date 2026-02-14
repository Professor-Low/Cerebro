"""
Active Reasoner - Claude.Me v6.0
Form new connections and generate insights without being prompted.

Part of Phase 4: Active Reasoning Over Memory
Uses DGX Spark for LLM-powered reasoning when available.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests

from insight_generator import Insight, InsightGenerator, InsightType

# DGX Spark configuration
_dgx_host = os.environ.get("CEREBRO_DGX_HOST", "")
DGX_REASONING_SERVICE = f"http://{_dgx_host}:8768" if _dgx_host else ""
DGX_TIMEOUT = 45


class ActiveReasoner:
    """
    Perform active reasoning over memories.

    Capabilities:
    - Find connections between unrelated memories
    - Identify patterns across conversations
    - Generate hypotheses for testing
    - Detect anomalies that don't fit patterns
    - Create inferences from multiple facts
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.insight_generator = InsightGenerator(base_path)
        self._dgx_available = None

    def _check_dgx_available(self) -> bool:
        """Check if DGX reasoning service is available."""
        if self._dgx_available is not None:
            return self._dgx_available

        try:
            resp = requests.get(f"{DGX_REASONING_SERVICE}/health", timeout=5)
            self._dgx_available = resp.status_code == 200
        except:
            self._dgx_available = False

        return self._dgx_available

    def reason_about_memories(
        self,
        memories: List[Dict],
        context: str = None,
        use_llm: bool = False
    ) -> Dict:
        """
        Perform active reasoning over a set of memories.

        Args:
            memories: List of memory dicts (conversations, facts, etc.)
            context: Optional context for focused reasoning
            use_llm: Whether to use DGX LLM for deeper reasoning

        Returns:
            Dict with insights, connections, and reasoning trace
        """
        results = {
            "insights": [],
            "connections": [],
            "inferences": [],
            "reasoning_trace": [],
            "timestamp": datetime.now().isoformat()
        }

        # Step 1: Detect patterns in the memories
        results["reasoning_trace"].append({
            "step": 1,
            "action": "pattern_detection",
            "description": "Analyzing memories for temporal and problem patterns"
        })

        temporal_insights = self.insight_generator.detect_temporal_patterns(memories)
        problem_insights = self.insight_generator.detect_problem_patterns(memories)

        for insight in temporal_insights + problem_insights:
            self.insight_generator.save_insight(insight)
            results["insights"].append(insight.to_dict())

        # Step 2: Find connections between memories
        results["reasoning_trace"].append({
            "step": 2,
            "action": "connection_finding",
            "description": f"Looking for connections between {len(memories)} memories"
        })

        # Sample pairs to check (limit for performance)
        max_pairs = 50
        pairs_checked = 0
        for i in range(len(memories)):
            if pairs_checked >= max_pairs:
                break
            for j in range(i + 1, len(memories)):
                if pairs_checked >= max_pairs:
                    break
                connection = self.insight_generator.find_connections(memories[i], memories[j])
                if connection:
                    self.insight_generator.save_insight(connection)
                    results["connections"].append(connection.to_dict())
                pairs_checked += 1

        # Step 3: Generate inferences
        results["reasoning_trace"].append({
            "step": 3,
            "action": "inference_generation",
            "description": "Generating logical inferences from facts"
        })

        # Extract facts for inference
        facts = []
        for mem in memories:
            if mem.get("type") == "semantic":
                facts.append(mem.get("fact", ""))
            elif mem.get("extracted_data", {}).get("facts"):
                for f in mem["extracted_data"]["facts"]:
                    facts.append(f.get("content", ""))

        # Try to generate inferences from fact pairs
        for i in range(len(facts)):
            for j in range(i + 1, min(i + 5, len(facts))):  # Limit pairs
                inference = self.insight_generator.generate_inference(facts[i], facts[j])
                if inference:
                    self.insight_generator.save_insight(inference)
                    results["inferences"].append(inference.to_dict())

        # Step 4: LLM-powered reasoning (if enabled)
        if use_llm and self._check_dgx_available():
            results["reasoning_trace"].append({
                "step": 4,
                "action": "llm_reasoning",
                "description": "Using LLM for deeper reasoning"
            })

            llm_insights = self._reason_with_llm(memories, context)
            for insight in llm_insights:
                self.insight_generator.save_insight(insight)
                results["insights"].append(insight.to_dict())

        # Summary
        results["summary"] = {
            "insights_generated": len(results["insights"]),
            "connections_found": len(results["connections"]),
            "inferences_made": len(results["inferences"]),
            "memories_analyzed": len(memories)
        }

        return results

    def _reason_with_llm(self, memories: List[Dict], context: str = None) -> List[Insight]:
        """Use DGX Spark LLM for deeper reasoning."""
        try:
            # Prepare memory summaries
            memory_summaries = []
            for mem in memories[:20]:  # Limit for context window
                if mem.get("type") == "semantic":
                    memory_summaries.append(f"Fact: {mem.get('fact', '')[:200]}")
                elif mem.get("type") == "episodic":
                    memory_summaries.append(f"Event: {mem.get('event', '')[:200]}")
                elif mem.get("search_index", {}).get("summary"):
                    memory_summaries.append(f"Conversation: {mem['search_index']['summary'][:200]}")

            resp = requests.post(
                f"{DGX_REASONING_SERVICE}/reason",
                json={
                    "memories": memory_summaries,
                    "context": context,
                    "request_types": ["patterns", "connections", "inferences"]
                },
                timeout=DGX_TIMEOUT
            )

            if resp.status_code != 200:
                return []

            data = resp.json()
            insights = []

            for item in data.get("insights", []):
                insight = Insight(
                    insight_id=self.insight_generator._generate_id(item.get("content", "")),
                    insight_type=item.get("type", InsightType.INFERENCE),
                    content=item.get("content", ""),
                    evidence=item.get("evidence", {}),
                    novelty_score=item.get("novelty", 0.7),
                    usefulness_score=item.get("usefulness", 0.6),
                    confidence=item.get("confidence", 0.6)
                )
                insights.append(insight)

            return insights

        except Exception as e:
            print(f"[ActiveReasoner] LLM reasoning failed: {e}")
            return []

    def find_relevant_insights(self, query: str, limit: int = 5) -> List[Insight]:
        """Find insights relevant to a query."""
        # Simple keyword matching for now
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_insights = []
        for insight_file in self.insight_generator.insights_path.glob("ins_*.json"):
            try:
                with open(insight_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if data.get("status") == "refuted":
                    continue

                content_lower = data.get("content", "").lower()
                content_words = set(content_lower.split())

                # Score by word overlap
                overlap = len(query_words.intersection(content_words))
                if overlap > 0:
                    insight = Insight.from_dict(data)
                    scored_insights.append((insight, overlap))
            except:
                continue

        # Sort by overlap score, then by usefulness
        scored_insights.sort(key=lambda x: (x[1], x[0].usefulness_score), reverse=True)
        return [i[0] for i in scored_insights[:limit]]

    def proactive_insights(self, user_message: str, recent_memories: List[Dict]) -> List[Dict]:
        """
        Generate proactive insights based on user's current context.

        Called during conversation to surface relevant insights.
        """
        relevant_insights = []

        # Find insights relevant to the user message
        insights = self.find_relevant_insights(user_message, limit=3)

        for insight in insights:
            if insight.usefulness_score >= 0.5 and insight.confidence >= 0.5:
                relevant_insights.append({
                    "insight": insight.content,
                    "type": insight.insight_type,
                    "confidence": insight.confidence,
                    "why_relevant": "Matches current context"
                })

        return relevant_insights

    def reason_about_goal(self, goal: str, available_knowledge: List[Dict]) -> Dict:
        """
        Reason about how to achieve a goal given available knowledge.

        Returns plan, potential blockers, and relevant knowledge.
        """
        results = {
            "goal": goal,
            "relevant_knowledge": [],
            "potential_blockers": [],
            "suggested_steps": [],
            "confidence": 0.5
        }

        # Find knowledge relevant to the goal
        goal_lower = goal.lower()
        goal_keywords = set(goal_lower.split())

        for item in available_knowledge:
            text = (
                item.get("content", "") or
                item.get("fact", "") or
                item.get("event", "") or
                ""
            ).lower()

            text_keywords = set(text.split())
            overlap = len(goal_keywords.intersection(text_keywords))

            if overlap >= 2:
                results["relevant_knowledge"].append({
                    "id": item.get("id"),
                    "content": text[:200],
                    "relevance_score": overlap / len(goal_keywords)
                })

        # Sort by relevance
        results["relevant_knowledge"].sort(key=lambda x: x["relevance_score"], reverse=True)
        results["relevant_knowledge"] = results["relevant_knowledge"][:5]

        # Calculate confidence based on available knowledge
        if len(results["relevant_knowledge"]) >= 3:
            results["confidence"] = 0.7
        elif len(results["relevant_knowledge"]) >= 1:
            results["confidence"] = 0.5
        else:
            results["confidence"] = 0.3
            results["potential_blockers"].append("Limited relevant knowledge available")

        return results

    def validate_insight(self, insight_id: str, is_valid: bool, feedback: str = None) -> Dict:
        """Validate or refute an insight based on user feedback."""
        insight = self.insight_generator.get_insight(insight_id)
        if not insight:
            return {"error": f"Insight {insight_id} not found"}

        if is_valid:
            insight.validate()
        else:
            insight.refute()

        self.insight_generator.save_insight(insight)

        return {
            "insight_id": insight_id,
            "new_status": insight.status,
            "new_confidence": insight.confidence
        }

    def get_stats(self) -> Dict:
        """Get reasoning statistics."""
        insight_stats = self.insight_generator.get_stats()

        return {
            "insights": insight_stats,
            "dgx_available": self._check_dgx_available() if self._dgx_available is None else self._dgx_available
        }
