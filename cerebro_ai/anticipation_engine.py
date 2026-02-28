"""
Anticipation Engine - Proactive Intelligence for AI Memory
Predicts what the user needs and generates contextual suggestions.

This is the core of Phase 5 in the All-Knowing Brain PRD.

ENHANCED (2026-01-17):
- Added semantic search via EmbeddingsEngine
- Expanded topic patterns with modern tech keywords
- Fixed Windows path detection
- Added recency weighting to solutions
- Improved confidence scoring with percentage-based calculation
- Added solutions index for fast lookups
"""

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import embeddings for semantic search
try:
    from ai_embeddings_engine import EmbeddingsEngine
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class AnticipationEngine:
    """
    Generates proactive suggestions based on:
    - Current context analysis
    - Historical pattern matching
    - Solution/antipattern library
    - User preferences
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            from config import AI_MEMORY_BASE
            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.conversations_path = self.base_path / "conversations"
        self.solutions_path = self.base_path / "solutions"
        self.antipatterns_path = self.base_path / "antipatterns"
        self.suggestions_log_path = self.base_path / "suggestions_log"
        self.solutions_index_path = self.base_path / "solutions" / "index.json"

        # Ensure directories exist
        self.suggestions_log_path.mkdir(parents=True, exist_ok=True)
        self.solutions_path.mkdir(parents=True, exist_ok=True)
        self.antipatterns_path.mkdir(parents=True, exist_ok=True)

        # Initialize embeddings engine for semantic search
        self._embeddings = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self._embeddings = EmbeddingsEngine(base_path=base_path)
            except Exception:
                pass  # Fall back to keyword matching

        # Load or build solutions index for fast lookups
        self._solutions_index = self._load_or_build_solutions_index()

    # ==========================================================================
    # SOLUTIONS INDEX - Fast lookups without full file scan
    # ==========================================================================

    def _load_or_build_solutions_index(self) -> Dict[str, Any]:
        """Load solutions index from file or build it from solution files."""
        if self.solutions_index_path.exists():
            try:
                with open(self.solutions_index_path, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                # Check if index is stale (older than 1 hour)
                last_updated = index.get("last_updated", "")
                if last_updated:
                    updated_time = datetime.fromisoformat(last_updated)
                    if datetime.now() - updated_time < timedelta(hours=1):
                        return index
            except Exception:
                pass

        return self._rebuild_solutions_index()

    def _rebuild_solutions_index(self) -> Dict[str, Any]:
        """Rebuild the solutions index from all solution files."""
        index = {
            "solutions": {},
            "tags_to_solutions": defaultdict(list),
            "keywords_to_solutions": defaultdict(list),
            "last_updated": datetime.now().isoformat(),
            "total_solutions": 0
        }

        if not self.solutions_path.exists():
            return index

        for sol_file in self.solutions_path.glob("*.json"):
            if sol_file.name == "index.json":
                continue
            try:
                with open(sol_file, 'r', encoding='utf-8') as f:
                    sol = json.load(f)

                sol_id = sol.get("id", sol_file.stem)

                # Extract keywords from problem text
                problem = sol.get("problem", "").lower()
                keywords = set(re.findall(r'\b\w{4,}\b', problem))

                # Calculate effectiveness score
                successes = sol.get("success_confirmations", 0)
                failures = sol.get("failure_count", 0)
                total = successes + failures
                effectiveness = (successes / total) if total > 0 else 0.5

                # Parse created_at for recency
                created_at = sol.get("created_at", "")

                # Build solution entry
                index["solutions"][sol_id] = {
                    "problem": problem[:200],
                    "solution_preview": sol.get("solution", "")[:150],
                    "tags": sol.get("tags", []),
                    "status": sol.get("status", "active"),
                    "effectiveness": round(effectiveness, 2),
                    "successes": successes,
                    "failures": failures,
                    "created_at": created_at,
                    "keywords": list(keywords)[:20]
                }

                # Build reverse indexes
                for tag in sol.get("tags", []):
                    index["tags_to_solutions"][tag.lower()].append(sol_id)

                for keyword in list(keywords)[:10]:
                    index["keywords_to_solutions"][keyword].append(sol_id)

                index["total_solutions"] += 1

            except Exception:
                continue

        # Convert defaultdicts to regular dicts for JSON
        index["tags_to_solutions"] = dict(index["tags_to_solutions"])
        index["keywords_to_solutions"] = dict(index["keywords_to_solutions"])

        # Save index
        try:
            with open(self.solutions_index_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return index

    def refresh_solutions_index(self) -> Dict[str, Any]:
        """Force refresh the solutions index."""
        self._solutions_index = self._rebuild_solutions_index()
        return {"success": True, "total_solutions": self._solutions_index.get("total_solutions", 0)}

    # ==========================================================================
    # SEMANTIC SEARCH - Use embeddings for similarity matching
    # ==========================================================================

    def _semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings.
        Falls back to keyword search if embeddings unavailable.
        """
        if not self._embeddings or not hasattr(self._embeddings, 'search'):
            return []

        try:
            results = self._embeddings.search(query, top_k=top_k)
            return results if results else []
        except Exception:
            return []

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using embeddings.
        Returns 0.0 if embeddings unavailable.
        """
        if not self._embeddings or not hasattr(self._embeddings, 'model'):
            return self._keyword_similarity(text1, text2)

        try:
            # Get embeddings for both texts
            emb1 = self._embeddings.model.encode([text1])[0]
            emb2 = self._embeddings.model.encode([text2])[0]

            # Compute cosine similarity
            import numpy as np
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception:
            return self._keyword_similarity(text1, text2)

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Fallback keyword-based similarity using Jaccard index."""
        words1 = set(re.findall(r'\b\w{3,}\b', text1.lower()))
        words2 = set(re.findall(r'\b\w{3,}\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    # ==========================================================================
    # CONTEXT ANALYZER - Understand current situation
    # ==========================================================================

    def analyze_context(self,
                       user_message: str,
                       cwd: str = None,
                       recent_tools: List[str] = None,
                       conversation_length: int = 0) -> Dict[str, Any]:
        """
        Analyze current context to understand what's happening.

        Args:
            user_message: The user's current message
            cwd: Current working directory (if available)
            recent_tools: List of recently used tools
            conversation_length: Number of messages in conversation

        Returns:
            Context analysis with task, stage, and detected entities
        """
        analysis = {
            "user_message": user_message[:500],
            "message_length": len(user_message),
            "cwd": cwd,
            "detected_project": self._detect_project(cwd, user_message),
            "detected_topics": self._extract_topics(user_message),
            "detected_entities": self._extract_entities(user_message),
            "task_stage": self._detect_stage(conversation_length, recent_tools or []),
            "intent": self._classify_intent(user_message),
            "urgency": self._detect_urgency(user_message),
            "analyzed_at": datetime.now().isoformat()
        }

        return analysis

    def _detect_project(self, cwd: str, message: str) -> Optional[str]:
        """Detect project from CWD or message content."""
        if cwd:
            cwd_path = Path(cwd)
            # Check for project indicators
            indicators = ['pyproject.toml', 'package.json', 'Cargo.toml', '.git', 'setup.py']
            for indicator in indicators:
                if (cwd_path / indicator).exists():
                    return cwd_path.name
            # Fall back to directory name if it looks like a project
            if '-' in cwd_path.name or '_' in cwd_path.name:
                return cwd_path.name

        # Try to extract from message
        project_patterns = [
            r'(?:project|repo|working on)\s+([a-zA-Z0-9_-]+)',
            r'in\s+([a-zA-Z0-9_-]+)\s+(?:directory|folder|repo)',
        ]
        for pattern in project_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_topics(self, message: str) -> List[str]:
        """Extract key topics from message."""
        # Technical terms that indicate topics - EXPANDED for modern tech
        topic_patterns = [
            # Core infrastructure
            r'\b(api|database|auth|authentication|cache|memory|session|config|deploy|test|debug)\b',
            # Programming languages
            r'\b(python|javascript|typescript|rust|go|java|sql|html|css|bash|powershell)\b',
            # Frameworks and libraries
            r'\b(react|vue|django|flask|fastapi|nextjs|node|express|pytorch|tensorflow)\b',
            # DevOps and cloud
            r'\b(docker|kubernetes|aws|gcp|azure|linux|git|github|ci/cd|pipeline|terraform)\b',
            # AI/ML and LLMs - NEW
            r'\b(mcp|claude|anthropic|openai|llm|gpt|embedding|vector|transformer|rag)\b',
            r'\b(llama|ollama|huggingface|sentence-transformer|faiss|chromadb)\b',
            # Problems and actions
            r'\b(error|bug|fix|issue|problem|crash|fail|broken|timeout|exception)\b',
            r'\b(feature|implement|add|create|build|update|refactor|optimize|migrate)\b',
            # Storage and networking - NEW
            r'\b(nas|nfs|cifs|smb|mount|storage|backup|sync|network|ssh|sftp)\b',
            # Claude Code specific - NEW
            r'\b(hook|hooks|compact|session|transcript|conversation|brain|anticipation)\b',
        ]

        topics = set()
        message_lower = message.lower()

        for pattern in topic_patterns:
            matches = re.findall(pattern, message_lower)
            topics.update(matches)

        return list(topics)[:15]

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """Extract named entities from message."""
        entities = {
            "files": [],
            "paths": [],
            "urls": [],
            "ips": [],
            "commands": []
        }

        # File patterns - expanded extensions
        file_matches = re.findall(
            r'\b[\w.-]+\.(py|js|ts|json|yaml|yml|md|txt|sh|sql|html|css|ps1|bat|exe|dll|log|cfg|ini|toml)\b',
            message
        )
        entities["files"] = list(set(file_matches))[:5]

        # Path patterns - FIXED: Now handles both Unix AND Windows paths
        unix_paths = re.findall(r'(?:/[\w.-]+)+/?', message)
        windows_paths = re.findall(r'[A-Za-z]:\\(?:[\w.-]+\\?)+', message)  # C:\path\to\file
        unc_paths = re.findall(r'\\\\[\w.-]+(?:\\[\w.-]+)+', message)  # \\server\share
        entities["paths"] = list(set(unix_paths + windows_paths + unc_paths))[:5]

        # URL patterns
        url_matches = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', message)
        entities["urls"] = list(set(url_matches))[:3]

        # IP patterns
        ip_matches = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', message)
        entities["ips"] = list(set(ip_matches))[:3]

        # Command patterns (backticks or common commands)
        cmd_matches = re.findall(r'`([^`]+)`', message)
        entities["commands"] = list(set(cmd_matches))[:5]

        return entities

    def _detect_stage(self, conversation_length: int, recent_tools: List[str]) -> str:
        """
        Detect what stage of work the user is in.

        Stages:
        - starting: Beginning of conversation
        - investigating: Reading files, searching, exploring
        - implementing: Writing/editing code
        - testing: Running tests, checking output
        - debugging: Fixing errors
        - finishing: Wrapping up, committing
        """
        if conversation_length < 2:
            return "starting"

        recent_tools_lower = [t.lower() for t in recent_tools[-5:]]

        # Check for patterns
        read_tools = ['read', 'grep', 'glob', 'search']
        write_tools = ['edit', 'write', 'notebookedit']

        read_count = sum(1 for t in recent_tools_lower if any(r in t for r in read_tools))
        write_count = sum(1 for t in recent_tools_lower if any(w in t for w in write_tools))
        bash_count = sum(1 for t in recent_tools_lower if 'bash' in t)

        if write_count > read_count:
            return "implementing"
        elif read_count > 2:
            return "investigating"
        elif bash_count > 1:
            # Check if testing based on common patterns
            return "testing"
        elif conversation_length > 10:
            return "working"

        return "investigating"

    def _classify_intent(self, message: str) -> str:
        """Classify the user's intent."""
        message_lower = message.lower()

        intent_patterns = {
            "debug": ["error", "bug", "fix", "broken", "not working", "fails", "crash", "issue"],
            "implement": ["add", "create", "implement", "build", "make", "write", "new"],
            "refactor": ["refactor", "clean up", "improve", "optimize", "simplify"],
            "understand": ["what", "how", "why", "explain", "show me", "where"],
            "test": ["test", "run", "check", "verify", "validate"],
            "deploy": ["deploy", "release", "publish", "push", "ship"],
            "configure": ["config", "setup", "install", "configure", "set up"],
        }

        for intent, keywords in intent_patterns.items():
            if any(kw in message_lower for kw in keywords):
                return intent

        return "general"

    def _detect_urgency(self, message: str) -> str:
        """Detect urgency level from message."""
        message_lower = message.lower()

        high_urgency = ["urgent", "asap", "critical", "emergency", "immediately", "broken", "down", "crash"]
        medium_urgency = ["soon", "quickly", "important", "need to"]

        if any(u in message_lower for u in high_urgency):
            return "high"
        elif any(u in message_lower for u in medium_urgency):
            return "medium"

        return "normal"

    # ==========================================================================
    # HISTORY MATCHER - Find similar past situations
    # ==========================================================================

    def find_similar_situations(self,
                                context: Dict[str, Any],
                                limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar past situations from conversation history.
        ENHANCED: Uses semantic search for better matching.

        Args:
            context: Context analysis from analyze_context()
            limit: Maximum matches to return

        Returns:
            List of similar past situations with outcomes
        """
        matches = []
        user_message = context.get("user_message", "")

        # Get topics and project from context
        topics = set(context.get("detected_topics", []))
        project = context.get("detected_project")
        intent = context.get("intent", "general")

        # 1. Try semantic search first for candidate conversations
        candidate_ids = set()
        if self._embeddings and user_message:
            try:
                semantic_results = self._semantic_search(user_message, top_k=15)
                for result in semantic_results:
                    source_id = result.get("source_id", "")
                    if source_id:
                        candidate_ids.add(source_id)
            except Exception:
                pass

        # 2. If no semantic candidates, fall back to full scan (limited)
        if not candidate_ids:
            # Get most recent conversations as candidates
            conv_files = sorted(
                self.conversations_path.glob("*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:50]  # Limit to 50 most recent
            candidate_ids = {f.stem for f in conv_files}

        # 3. Score each candidate
        for conv_id in list(candidate_ids)[:30]:  # Limit processing
            conv_file = self.conversations_path / f"{conv_id}.json"
            if not conv_file.exists():
                continue

            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                # Check for topic overlap
                conv_topics = set(conv.get("metadata", {}).get("topics", []))
                topic_overlap = topics & conv_topics

                # Check for project match
                conv_project = conv.get("metadata", {}).get("project")
                project_match = project and conv_project and project.lower() in conv_project.lower()

                # Calculate base similarity score
                similarity = 0.0
                if topic_overlap:
                    similarity += len(topic_overlap) / max(len(topics), 1) * 0.25

                if project_match:
                    similarity += 0.2

                # Semantic similarity boost
                conv_summary = conv.get("metadata", {}).get("summary", "")
                if user_message and conv_summary:
                    semantic_score = self._compute_semantic_similarity(user_message, conv_summary[:500])
                    similarity += semantic_score * 0.35

                # Check for solutions/outcomes
                extracted = conv.get("extracted_data", {})
                solutions = extracted.get("solutions", [])
                problems_solved = extracted.get("problems_solved", [])
                has_solution = len(solutions) > 0 or len(problems_solved) > 0

                if has_solution:
                    similarity += 0.15

                # Check for same intent
                if intent != "general" and intent in conv_summary.lower():
                    similarity += 0.05

                if similarity >= 0.25:
                    matches.append({
                        "conversation_id": conv.get("id", conv_id),
                        "similarity": round(similarity, 2),
                        "summary": conv_summary[:200] if conv_summary else "",
                        "topics": list(conv_topics)[:5],
                        "has_solution": has_solution,
                        "solutions": [s[:100] for s in solutions[:2]] if solutions else [],
                        "problems_solved": [p.get("problem", "")[:100] for p in problems_solved[:2]] if problems_solved else [],
                        "timestamp": conv.get("timestamp", ""),
                        "project": conv_project
                    })

            except Exception:
                continue

        # Sort by similarity
        matches.sort(key=lambda x: x["similarity"], reverse=True)

        return matches[:limit]

    def find_relevant_solutions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find solutions from the solution tracker that match current context.
        ENHANCED: Uses semantic search + index for fast lookups + recency weighting.
        """
        solutions = []
        topics = context.get("detected_topics", [])
        user_message = context.get("user_message", "")

        if not self.solutions_path.exists():
            return solutions

        # 1. Try semantic search first if available
        candidate_ids = set()

        if self._embeddings and user_message:
            try:
                # Search for semantically similar content
                semantic_results = self._semantic_search(user_message, top_k=10)
                for result in semantic_results:
                    # Extract solution IDs from semantic results
                    if "solution" in result.get("chunk_type", "").lower():
                        candidate_ids.add(result.get("source_id", ""))
            except Exception:
                pass

        # 2. Use index for keyword-based lookup (fast)
        if topics and self._solutions_index:
            index = self._solutions_index
            for topic in topics:
                topic_lower = topic.lower()
                # Check tag index
                for sol_id in index.get("tags_to_solutions", {}).get(topic_lower, []):
                    candidate_ids.add(sol_id)
                # Check keyword index
                for sol_id in index.get("keywords_to_solutions", {}).get(topic_lower, []):
                    candidate_ids.add(sol_id)

        # 3. If no candidates, fall back to full scan (last resort)
        if not candidate_ids:
            for sol_file in self.solutions_path.glob("*.json"):
                if sol_file.name == "index.json":
                    continue
                candidate_ids.add(sol_file.stem)

        # 4. Score each candidate solution
        now = datetime.now()
        for sol_id in list(candidate_ids)[:20]:  # Limit to prevent slowdown
            sol_file = self.solutions_path / f"{sol_id}.json"
            if not sol_file.exists():
                continue

            try:
                with open(sol_file, 'r', encoding='utf-8') as f:
                    sol = json.load(f)

                # Check if solution is active
                if sol.get("status") != "active":
                    continue

                # Calculate effectiveness score (percentage-based)
                successes = sol.get("success_confirmations", 0)
                failures = sol.get("failure_count", 0)
                total = successes + failures
                effectiveness = (successes / total) if total > 0 else 0.5

                # Calculate recency bonus (newer solutions get boost)
                recency_bonus = 0.0
                created_at = sol.get("created_at", "")
                if created_at:
                    try:
                        created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        days_old = (now - created_time.replace(tzinfo=None)).days
                        # Newer solutions get up to 0.2 bonus, decays over 30 days
                        recency_bonus = max(0, 0.2 * (1 - days_old / 30))
                    except Exception:
                        pass

                # Calculate semantic similarity if available
                semantic_score = 0.0
                if user_message and sol.get("problem"):
                    semantic_score = self._compute_semantic_similarity(user_message, sol.get("problem", ""))

                # Combined confidence score
                # 40% effectiveness + 30% semantic + 20% recency + 10% base
                confidence = (
                    effectiveness * 0.4 +
                    semantic_score * 0.3 +
                    recency_bonus +
                    0.1  # Base score for being in candidate set
                )

                solutions.append({
                    "solution_id": sol.get("id"),
                    "problem": sol.get("problem", "")[:150],
                    "solution": sol.get("solution", "")[:200],
                    "confidence": round(confidence, 2),
                    "effectiveness": round(effectiveness, 2),
                    "semantic_score": round(semantic_score, 2),
                    "tags": sol.get("tags", [])[:5]
                })

            except Exception:
                continue

        # Sort by combined confidence
        solutions.sort(key=lambda x: x["confidence"], reverse=True)

        return solutions[:5]  # Return more options

    def find_relevant_code_snippets(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find code vault snippets matching context."""
        try:
            vault_index_path = self.base_path / "code_vault" / "index.json"
            if not vault_index_path.exists():
                return []

            with open(vault_index_path, 'r', encoding='utf-8') as f:
                vault = json.load(f)

            topics = set(context.get("detected_topics", []))
            user_message = context.get("user_message", "").lower()
            snippets = vault.get("snippets", [])

            matches = []
            for s in snippets:
                # Match by keywords/tags
                snippet_keywords = set(kw.lower() for kw in s.get("keywords", []) + s.get("tags", []))
                overlap = topics & snippet_keywords

                # Also check if snippet problem_solved matches user message
                problem_solved = s.get("problem_solved", "").lower()
                message_match = any(
                    word in user_message
                    for word in problem_solved.split()
                    if len(word) > 4
                )

                if overlap or message_match:
                    confidence = len(overlap) / max(len(topics), 1) if topics else 0
                    if message_match:
                        confidence += 0.3

                    matches.append({
                        "id": s.get("id"),
                        "title": s.get("title", s.get("name", "Snippet")),
                        "problem_solved": s.get("problem_solved", ""),
                        "usage": s.get("usage", ""),
                        "confidence": min(1.0, confidence),
                        "source": "code_vault"
                    })

            return sorted(matches, key=lambda x: x["confidence"], reverse=True)[:3]
        except Exception:
            return []

    def find_relevant_antipatterns(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find antipatterns (things NOT to do) that match current context.
        ENHANCED: Uses semantic search for better matching.
        """
        antipatterns = []
        topics = context.get("detected_topics", [])
        user_message = context.get("user_message", "")

        if not self.antipatterns_path.exists():
            return antipatterns

        for ap_file in self.antipatterns_path.glob("*.json"):
            try:
                with open(ap_file, 'r', encoding='utf-8') as f:
                    ap = json.load(f)

                # Check tag/topic overlap
                ap_tags = set(ap.get("tags", []))
                topic_overlap = set(topics) & ap_tags

                # Calculate semantic similarity if available
                semantic_score = 0.0
                if user_message and ap.get("original_problem"):
                    semantic_score = self._compute_semantic_similarity(
                        user_message, ap.get("original_problem", "")
                    )

                # Calculate combined relevance score
                relevance = 0.0
                if topic_overlap:
                    relevance += len(topic_overlap) * 0.15
                if any(t in ap.get("original_problem", "").lower() for t in topics):
                    relevance += 0.2
                relevance += semantic_score * 0.5

                if relevance >= 0.2 or semantic_score >= 0.5:
                    antipatterns.append({
                        "antipattern_id": ap.get("id"),
                        "what_not_to_do": ap.get("what_not_to_do", "")[:150],
                        "why_it_failed": ap.get("why_it_failed", "")[:150],
                        "tags": list(ap_tags)[:5],
                        "relevance": round(relevance, 2),
                        "semantic_score": round(semantic_score, 2)
                    })

            except Exception:
                continue

        # Sort by relevance
        antipatterns.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return antipatterns[:3]

    # ==========================================================================
    # SUGGESTION GENERATOR - Create proactive suggestions
    # ==========================================================================

    def generate_suggestions(self,
                            context: Dict[str, Any],
                            similar_situations: List[Dict[str, Any]],
                            solutions: List[Dict[str, Any]],
                            antipatterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate proactive suggestions based on all available information.

        Args:
            context: Context analysis
            similar_situations: Similar past conversations
            solutions: Relevant solutions
            antipatterns: Relevant antipatterns

        Returns:
            List of suggestions with type, content, and confidence
        """
        suggestions = []

        # 1. Stage-based suggestions
        stage_suggestions = self._generate_stage_suggestions(context)
        suggestions.extend(stage_suggestions)

        # 2. History-based suggestions (from similar situations)
        for situation in similar_situations[:2]:
            if situation.get("has_solution") and situation.get("solutions"):
                suggestions.append({
                    "type": "context",
                    "content": f"Similar work found: {situation['summary'][:100]}",
                    "detail": f"Solution used: {situation['solutions'][0]}" if situation['solutions'] else None,
                    "confidence": situation["similarity"],
                    "source": "history",
                    "reference_id": situation["conversation_id"]
                })

        # 3. Solution-based suggestions
        for solution in solutions[:2]:
            if solution.get("confidence", 0) > 0:
                suggestions.append({
                    "type": "solution",
                    "content": f"Known solution for '{solution['problem'][:50]}...': {solution['solution'][:100]}",
                    "confidence": min(0.9, 0.5 + solution["confidence"] * 0.1),
                    "source": "solutions",
                    "reference_id": solution["solution_id"]
                })

        # 4. Warning suggestions (from antipatterns)
        for antipattern in antipatterns[:1]:
            suggestions.append({
                "type": "warning",
                "content": f"Avoid: {antipattern['what_not_to_do'][:80]}",
                "detail": f"Because: {antipattern['why_it_failed'][:80]}",
                "confidence": 0.7,
                "source": "antipatterns",
                "reference_id": antipattern["antipattern_id"]
            })

        # 5. Intent-based suggestions
        intent_suggestions = self._generate_intent_suggestions(context)
        suggestions.extend(intent_suggestions)

        # 6. Code vault snippets
        code_snippets = self.find_relevant_code_snippets(context)
        for snippet in code_snippets[:2]:
            if snippet.get("confidence", 0) >= 0.3:
                suggestions.append({
                    "type": "code_snippet",
                    "content": f"Code: {snippet['title']} - {snippet['problem_solved'][:80]}",
                    "reference": snippet["id"],
                    "usage": snippet.get("usage", "")[:100],
                    "confidence": snippet["confidence"],
                    "source": "code_vault"
                })

        return suggestions

    def _generate_stage_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions based on task stage."""
        stage = context.get("task_stage", "starting")
        suggestions = []

        if stage == "starting":
            if context.get("detected_project"):
                suggestions.append({
                    "type": "context",
                    "content": f"Working in project: {context['detected_project']}. Want me to load relevant past work?",
                    "confidence": 0.7,
                    "source": "stage"
                })

        elif stage == "investigating":
            suggestions.append({
                "type": "action",
                "content": "I can search for related past solutions while you investigate.",
                "confidence": 0.6,
                "source": "stage"
            })

        elif stage == "implementing":
            suggestions.append({
                "type": "action",
                "content": "Ready to run tests when implementation is complete.",
                "confidence": 0.7,
                "source": "stage"
            })

        elif stage == "testing":
            suggestions.append({
                "type": "reminder",
                "content": "Remember to commit changes after tests pass.",
                "confidence": 0.6,
                "source": "stage"
            })

        elif stage == "debugging":
            suggestions.append({
                "type": "action",
                "content": "Want me to search for similar errors in past conversations?",
                "confidence": 0.8,
                "source": "stage"
            })

        return suggestions

    def _generate_intent_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions based on detected intent."""
        intent = context.get("intent", "general")
        urgency = context.get("urgency", "normal")
        suggestions = []

        if intent == "debug" and urgency == "high":
            suggestions.append({
                "type": "action",
                "content": "High urgency detected. I'll focus on quick diagnosis.",
                "confidence": 0.8,
                "source": "intent"
            })

        elif intent == "implement":
            topics = context.get("detected_topics", [])
            if topics:
                suggestions.append({
                    "type": "context",
                    "content": f"Implementing with: {', '.join(topics[:3])}. I'll watch for common pitfalls.",
                    "confidence": 0.6,
                    "source": "intent"
                })

        elif intent == "deploy":
            suggestions.append({
                "type": "warning",
                "content": "Deployment detected. Double-check environment configurations.",
                "confidence": 0.7,
                "source": "intent"
            })

        return suggestions

    # ==========================================================================
    # SUGGESTION RANKER - Rank and filter suggestions
    # ==========================================================================

    def rank_suggestions(self,
                        suggestions: List[Dict[str, Any]],
                        max_count: int = 3) -> List[Dict[str, Any]]:
        """
        Rank and filter suggestions to return the most useful ones.

        Args:
            suggestions: Raw list of suggestions
            max_count: Maximum suggestions to return

        Returns:
            Ranked and filtered suggestions
        """
        if not suggestions:
            return []

        # Score each suggestion
        scored = []
        for suggestion in suggestions:
            score = suggestion.get("confidence", 0.5)

            # Boost certain types
            type_boosts = {
                "warning": 1.2,      # Warnings are high priority
                "solution": 1.1,     # Known solutions are valuable
                "context": 1.0,      # Context is helpful
                "action": 0.9,       # Actions need user buy-in
                "reminder": 0.8,     # Reminders are lower priority
            }
            score *= type_boosts.get(suggestion.get("type", "context"), 1.0)

            # Boost if has reference (grounded in data)
            if suggestion.get("reference_id"):
                score *= 1.1

            # Cap at 1.0
            score = min(1.0, score)

            scored.append({
                **suggestion,
                "score": round(score, 2)
            })

        # Filter low confidence
        filtered = [s for s in scored if s["score"] >= 0.5]

        # Deduplicate similar suggestions
        seen_content = set()
        deduped = []
        for suggestion in filtered:
            content_key = suggestion["content"][:50].lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduped.append(suggestion)

        # Sort by score
        deduped.sort(key=lambda x: x["score"], reverse=True)

        return deduped[:max_count]

    # ==========================================================================
    # MAIN API - Get suggestions
    # ==========================================================================

    def get_suggestions(self,
                       user_message: str,
                       cwd: str = None,
                       recent_tools: List[str] = None,
                       conversation_length: int = 0,
                       max_suggestions: int = 3) -> Dict[str, Any]:
        """
        Main API: Get proactive suggestions based on current context.

        Args:
            user_message: The user's current message
            cwd: Current working directory
            recent_tools: List of recently used tools
            conversation_length: Number of messages in conversation
            max_suggestions: Maximum suggestions to return

        Returns:
            Dict with context summary and ranked suggestions
        """
        # 1. Analyze context
        context = self.analyze_context(
            user_message=user_message,
            cwd=cwd,
            recent_tools=recent_tools,
            conversation_length=conversation_length
        )

        # 2. Find similar situations
        similar = self.find_similar_situations(context, limit=5)

        # 3. Find relevant solutions
        solutions = self.find_relevant_solutions(context)

        # 4. Find relevant antipatterns
        antipatterns = self.find_relevant_antipatterns(context)

        # 5. Generate suggestions
        raw_suggestions = self.generate_suggestions(
            context=context,
            similar_situations=similar,
            solutions=solutions,
            antipatterns=antipatterns
        )

        # 6. Rank and filter
        ranked = self.rank_suggestions(raw_suggestions, max_count=max_suggestions)

        # 7. Build result
        result = {
            "context_summary": {
                "project": context.get("detected_project"),
                "topics": context.get("detected_topics", [])[:5],
                "stage": context.get("task_stage"),
                "intent": context.get("intent"),
                "urgency": context.get("urgency")
            },
            "suggestions": ranked,
            "suggestion_count": len(ranked),
            "similar_situations_found": len(similar),
            "solutions_found": len(solutions),
            "antipatterns_found": len(antipatterns),
            "generated_at": datetime.now().isoformat()
        }

        # 8. Log for future analysis (optional)
        self._log_suggestion(result)

        return result

    def _log_suggestion(self, result: Dict[str, Any]) -> None:
        """Log suggestion for future analysis and improvement tracking."""
        try:
            log_file = self.suggestions_log_path / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception:
            pass  # Logging failure shouldn't break suggestions


# Convenience function for direct use
def get_suggestions(user_message: str, **kwargs) -> Dict[str, Any]:
    """Get suggestions using default engine instance."""
    engine = AnticipationEngine()
    return engine.get_suggestions(user_message, **kwargs)


if __name__ == "__main__":
    # Test the engine
    engine = AnticipationEngine()

    # Test context
    result = engine.get_suggestions(
        user_message="I need to fix the timeout error in the MCP server",
        cwd="/home/user/ai-memory-mcp",
        recent_tools=["Read", "Grep", "Bash"],
        conversation_length=5
    )

    print("=== Anticipation Engine Test ===")
    print(f"Context: {result['context_summary']}")
    print(f"\nSuggestions ({result['suggestion_count']}):")
    for s in result['suggestions']:
        print(f"  [{s['type']}] {s['content']} (score: {s['score']})")
