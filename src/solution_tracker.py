"""
Solution Tracker - Versioned Solution Management with Causal Linking
Tracks the evolution of solutions: what worked, what failed, and why.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class SolutionTracker:
    """
    Track solutions with versioning and causal relationships.

    Handles:
    - Solution versioning (v1 → v2 → v3)
    - Failure tracking (what went wrong and why)
    - Anti-patterns (what NOT to do)
    - Causal chains (Problem A → Solution B → Problem C → Solution D)
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.solutions_path = self.base_path / "solutions"
        self.antipatterns_path = self.base_path / "antipatterns"
        self.learnings_path = self.base_path / "learnings"

        # Ensure directories exist
        self.solutions_path.mkdir(parents=True, exist_ok=True)
        self.antipatterns_path.mkdir(parents=True, exist_ok=True)
        self.learnings_path.mkdir(parents=True, exist_ok=True)

    def record_solution(self,
                       problem: str,
                       solution: str,
                       context: str = "",
                       tags: List[str] = None,
                       conversation_id: str = None,
                       supersedes: str = None,
                       caused_by_failure: str = None) -> Dict[str, Any]:
        """
        Record a solution with full context and relationships.

        Args:
            problem: The problem being solved
            solution: The solution/fix applied
            context: Additional context (code, config, etc.)
            tags: Keywords for searching
            conversation_id: Source conversation
            supersedes: ID of previous solution this replaces
            caused_by_failure: ID of failure that led to this solution

        Returns:
            The created solution record
        """
        # Generate solution ID
        solution_id = self._generate_id(problem + solution)

        # Load existing solution file if it exists (for versioning)
        solution_file = self.solutions_path / f"{solution_id}.json"

        if solution_file.exists():
            with open(solution_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            version = existing.get('current_version', 0) + 1
            history = existing.get('version_history', [])
            # Archive current version to history
            history.append({
                'version': existing.get('current_version', 1),
                'solution': existing.get('solution'),
                'timestamp': existing.get('updated_at'),
                'status': 'superseded'
            })
        else:
            version = 1
            history = []

        record = {
            'id': solution_id,
            'problem': problem,
            'problem_hash': hashlib.md5(problem.lower().encode()).hexdigest()[:12],
            'solution': solution,
            'context': context,
            'tags': tags or [],
            'current_version': version,
            'version_history': history,
            'status': 'active',  # active, superseded, failed
            'conversation_id': conversation_id,
            'supersedes': supersedes,
            'caused_by_failure': caused_by_failure,
            'failure_count': 0,
            'success_confirmations': 0,
            'created_at': datetime.now().isoformat() if version == 1 else existing.get('created_at'),
            'updated_at': datetime.now().isoformat(),
            'causal_chain': self._build_causal_chain(caused_by_failure)
        }

        # Save solution
        with open(solution_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        # Update superseded solution if specified
        if supersedes:
            self._mark_superseded(supersedes, solution_id)

        return record

    def record_failure(self,
                      solution_id: str,
                      failure_description: str,
                      error_message: str = "",
                      conversation_id: str = None) -> Dict[str, Any]:
        """
        Record that a solution failed.

        This is KEY for learning - when something breaks, we need to know:
        1. What solution was being used
        2. How it failed
        3. Link it to the new solution that fixes it

        Args:
            solution_id: The solution that failed
            failure_description: What went wrong
            error_message: Actual error if available
            conversation_id: Where this was discovered

        Returns:
            Failure record
        """
        failure_id = self._generate_id(solution_id + failure_description + datetime.now().isoformat())

        # Load the solution that failed
        solution_file = self.solutions_path / f"{solution_id}.json"
        if solution_file.exists():
            with open(solution_file, 'r', encoding='utf-8') as f:
                solution = json.load(f)

            # Increment failure count
            solution['failure_count'] = solution.get('failure_count', 0) + 1
            solution['last_failure'] = datetime.now().isoformat()
            solution['status'] = 'failed' if solution['failure_count'] >= 2 else solution['status']

            # Add failure to history
            if 'failures' not in solution:
                solution['failures'] = []
            solution['failures'].append({
                'failure_id': failure_id,
                'description': failure_description,
                'error': error_message,
                'timestamp': datetime.now().isoformat(),
                'conversation_id': conversation_id
            })

            with open(solution_file, 'w', encoding='utf-8') as f:
                json.dump(solution, f, indent=2, ensure_ascii=False)
        else:
            solution = {'id': solution_id, 'status': 'not_found'}

        # Create anti-pattern from failure
        antipattern = self.record_antipattern(
            what_not_to_do=solution.get('solution', 'Unknown solution'),
            why_it_failed=failure_description,
            error_details=error_message,
            original_problem=solution.get('problem', ''),
            conversation_id=conversation_id,
            source_solution_id=solution_id
        )

        return {
            'failure_id': failure_id,
            'solution_id': solution_id,
            'solution_status': solution.get('status'),
            'antipattern_created': antipattern['id'],
            'message': 'Failure recorded and anti-pattern created'
        }

    def record_antipattern(self,
                          what_not_to_do: str,
                          why_it_failed: str,
                          error_details: str = "",
                          original_problem: str = "",
                          conversation_id: str = None,
                          source_solution_id: str = None,
                          tags: List[str] = None) -> Dict[str, Any]:
        """
        Record an anti-pattern - something that should NOT be done.

        Anti-patterns are crucial for learning. They tell us:
        - What approach to AVOID
        - Why it doesn't work
        - What error/problem it causes

        Args:
            what_not_to_do: The approach/solution to avoid
            why_it_failed: Explanation of the failure
            error_details: Specific error messages
            original_problem: What problem was being solved
            conversation_id: Source conversation
            source_solution_id: The failed solution this came from
            tags: Keywords for searching

        Returns:
            Anti-pattern record
        """
        antipattern_id = self._generate_id(what_not_to_do + why_it_failed)

        record = {
            'id': antipattern_id,
            'type': 'antipattern',
            'what_not_to_do': what_not_to_do,
            'why_it_failed': why_it_failed,
            'error_details': error_details,
            'original_problem': original_problem,
            'tags': tags or [],
            'source_solution_id': source_solution_id,
            'conversation_id': conversation_id,
            'created_at': datetime.now().isoformat(),
            'times_referenced': 0
        }

        # Save anti-pattern
        antipattern_file = self.antipatterns_path / f"{antipattern_id}.json"
        with open(antipattern_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        return record

    def confirm_solution_works(self, solution_id: str, conversation_id: str = None) -> Dict[str, Any]:
        """
        Confirm that a solution is working correctly.
        Increases confidence in the solution.
        """
        solution_file = self.solutions_path / f"{solution_id}.json"

        if not solution_file.exists():
            return {'error': f'Solution {solution_id} not found'}

        with open(solution_file, 'r', encoding='utf-8') as f:
            solution = json.load(f)

        solution['success_confirmations'] = solution.get('success_confirmations', 0) + 1
        solution['last_confirmed'] = datetime.now().isoformat()
        solution['status'] = 'active'  # Re-activate if it was marked failed

        if 'confirmations' not in solution:
            solution['confirmations'] = []
        solution['confirmations'].append({
            'timestamp': datetime.now().isoformat(),
            'conversation_id': conversation_id
        })

        with open(solution_file, 'w', encoding='utf-8') as f:
            json.dump(solution, f, indent=2, ensure_ascii=False)

        return {
            'solution_id': solution_id,
            'status': 'confirmed',
            'total_confirmations': solution['success_confirmations']
        }

    def find_solution(self, problem: str) -> List[Dict[str, Any]]:
        """
        Find solutions for a given problem.
        Returns solutions sorted by confidence (confirmations - failures).
        """
        problem_hash = hashlib.md5(problem.lower().encode()).hexdigest()[:12]
        results = []

        for solution_file in self.solutions_path.glob("*.json"):
            try:
                with open(solution_file, 'r', encoding='utf-8') as f:
                    solution = json.load(f)

                # Match by hash or keyword similarity
                if solution.get('problem_hash') == problem_hash:
                    results.append(solution)
                elif self._text_similarity(problem, solution.get('problem', '')) > 0.5:
                    results.append(solution)

            except Exception:
                continue

        # Sort by confidence (success_confirmations - failure_count)
        results.sort(
            key=lambda x: x.get('success_confirmations', 0) - x.get('failure_count', 0),
            reverse=True
        )

        return results

    def find_antipatterns(self, problem: str = None, tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        Find anti-patterns related to a problem or tags.
        Returns things NOT to do.
        """
        results = []

        for ap_file in self.antipatterns_path.glob("*.json"):
            try:
                with open(ap_file, 'r', encoding='utf-8') as f:
                    antipattern = json.load(f)

                # Match by problem similarity
                if problem and self._text_similarity(problem, antipattern.get('original_problem', '')) > 0.3:
                    results.append(antipattern)
                # Match by tags
                elif tags:
                    ap_tags = antipattern.get('tags', [])
                    if any(t in ap_tags for t in tags):
                        results.append(antipattern)

            except Exception:
                continue

        return results

    def get_solution_chain(self, solution_id: str) -> Dict[str, Any]:
        """
        Get the full evolution chain of a solution.
        Shows: Original → Failure → Fix → Failure → Fix → Current
        """
        chain = []
        current_id = solution_id

        # Walk backwards through supersedes chain
        while current_id:
            solution_file = self.solutions_path / f"{current_id}.json"
            if not solution_file.exists():
                break

            with open(solution_file, 'r', encoding='utf-8') as f:
                solution = json.load(f)

            chain.append({
                'id': solution['id'],
                'version': solution.get('current_version', 1),
                'solution': solution.get('solution', '')[:200],
                'status': solution.get('status'),
                'failures': len(solution.get('failures', [])),
                'confirmations': solution.get('success_confirmations', 0)
            })

            current_id = solution.get('supersedes')

        chain.reverse()  # Oldest first

        return {
            'solution_id': solution_id,
            'chain_length': len(chain),
            'evolution': chain
        }

    def get_learnings_summary(self) -> Dict[str, Any]:
        """Get summary of all learnings and patterns."""
        solutions = list(self.solutions_path.glob("*.json"))
        antipatterns = list(self.antipatterns_path.glob("*.json"))

        active_solutions = 0
        failed_solutions = 0
        total_failures = 0
        total_confirmations = 0

        for sf in solutions:
            try:
                with open(sf, 'r', encoding='utf-8') as f:
                    s = json.load(f)
                if s.get('status') == 'active':
                    active_solutions += 1
                elif s.get('status') == 'failed':
                    failed_solutions += 1
                total_failures += s.get('failure_count', 0)
                total_confirmations += s.get('success_confirmations', 0)
            except:
                continue

        return {
            'total_solutions': len(solutions),
            'active_solutions': active_solutions,
            'failed_solutions': failed_solutions,
            'total_antipatterns': len(antipatterns),
            'total_failures_recorded': total_failures,
            'total_confirmations': total_confirmations,
            'learning_ratio': round(total_confirmations / max(total_failures, 1), 2)
        }

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID from content."""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _mark_superseded(self, old_id: str, new_id: str):
        """Mark an old solution as superseded by a new one."""
        old_file = self.solutions_path / f"{old_id}.json"
        if old_file.exists():
            with open(old_file, 'r', encoding='utf-8') as f:
                old = json.load(f)
            old['status'] = 'superseded'
            old['superseded_by'] = new_id
            old['superseded_at'] = datetime.now().isoformat()
            with open(old_file, 'w', encoding='utf-8') as f:
                json.dump(old, f, indent=2, ensure_ascii=False)

    def _build_causal_chain(self, failure_id: str) -> List[str]:
        """Build causal chain from a failure back to original."""
        chain = []
        if failure_id:
            chain.append(f"caused_by:{failure_id}")
        return chain

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word-based similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)


if __name__ == "__main__":
    tracker = SolutionTracker()

    # Example: Record a solution
    sol1 = tracker.record_solution(
        problem="NAS mount fails at boot",
        solution="Add _netdev option to fstab",
        context="Mount tries before network is ready",
        tags=["nas", "fstab", "mount", "boot"]
    )
    print(f"Created solution: {sol1['id']}")

    # Example: Record a failure
    failure = tracker.record_failure(
        solution_id=sol1['id'],
        failure_description="Still fails because CIFS doesn't support reconnect option",
        error_message="cifs: Unknown parameter 'reconnect'"
    )
    print(f"Recorded failure, created antipattern: {failure['antipattern_created']}")

    # Example: Record the fix
    sol2 = tracker.record_solution(
        problem="NAS mount fails at boot",
        solution="Use x-systemd.automount without reconnect option",
        context="reconnect option not supported on this kernel",
        tags=["nas", "fstab", "mount", "systemd"],
        supersedes=sol1['id'],
        caused_by_failure=failure['failure_id']
    )
    print(f"Created fix solution: {sol2['id']} (v{sol2['current_version']})")

    # Get the chain
    chain = tracker.get_solution_chain(sol2['id'])
    print(f"\nSolution evolution: {chain}")
