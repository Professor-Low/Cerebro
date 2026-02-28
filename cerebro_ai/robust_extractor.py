"""
Robust Extraction Engine - Production Ready
Replaces weak regex-based extraction with comprehensive NLP-light approach
"""
import re
from datetime import datetime
from typing import Dict, List


class RobustExtractor:
    """
    Production-grade extraction using pattern matching + heuristics.
    Much more reliable than simple regex.
    """

    @staticmethod
    def extract_actions(content: str, messages: List[Dict]) -> List[Dict]:
        """Extract actions taken - what was actually done"""
        actions = []

        # Comprehensive action patterns
        action_verbs = [
            # File operations
            r'\b(created?|creat(ing|ed)|made|wrote|written|added|generated?)\s+(?:a\s+)?(?:new\s+)?(?:file|script|module|component|class|function)?\s*(?:named|called)?\s*["\']?([^\s,\."\']+)',
            r'\b(modifi(ed|es|ed)|updat(ed|es|ing)|chang(ed|es|ing)|edit(ed|ing)|revised?)\s+(?:the\s+)?(?:file|code|function|class|component)?\s*["\']?([^\s,\."\']+)',
            r'\b(delet(ed|es|ing)|remov(ed|es|ing))\s+(?:the\s+)?(?:file|code|function)?\s*["\']?([^\s,\."\']+)',

            # Installation/setup
            r'\b(install(ed|ing)?|add(ed|ing)?)\s+(?:the\s+)?(?:package|library|module|dependency)?\s*["\']?([^\s,\."\']+)',
            r'\b(configur(ed|ing)|set\s*up|setup)\s+([^\n]+?)(?:\.|,|\n)',

            # Execution
            r'\b(ran|run|execut(ed|ing)|start(ed|ing))\s+(?:the\s+)?(?:command|script|server)?\s*["\']?([^\s,\."\']+)',

            # Fixes
            r'\b(fix(ed|ing)?|resolv(ed|ing)?|patch(ed|ing)?)\s+(?:the\s+)?([^\n]+?)(?:\.|because|by)',

            # Implementation
            r'\b(implement(ed|ing)?|built|develop(ed|ing)?)\s+(?:a\s+)?(?:new\s+)?([^\n]+?)(?:\.|,|\n)',
        ]

        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue

            msg_content = msg.get("content", "")

            for pattern in action_verbs:
                matches = re.finditer(pattern, msg_content, re.IGNORECASE)
                for match in matches:
                    # Get the verb and target
                    groups = [g for g in match.groups() if g]
                    if len(groups) >= 2:
                        verb = groups[0].lower()
                        target = groups[-1].strip()

                        # Determine action type
                        action_type = "other"
                        if any(x in verb for x in ['creat', 'made', 'wrote', 'add', 'generat']):
                            action_type = "create"
                        elif any(x in verb for x in ['modif', 'updat', 'chang', 'edit']):
                            action_type = "modify"
                        elif any(x in verb for x in ['delet', 'remov']):
                            action_type = "delete"
                        elif any(x in verb for x in ['install', 'add']):
                            action_type = "install"
                        elif any(x in verb for x in ['config', 'setup']):
                            action_type = "configure"
                        elif any(x in verb for x in ['ran', 'run', 'execut', 'start']):
                            action_type = "execute"
                        elif any(x in verb for x in ['fix', 'resolv', 'patch']):
                            action_type = "fix"
                        elif any(x in verb for x in ['implement', 'built', 'develop']):
                            action_type = "implement"

                        actions.append({
                            "action_type": action_type,
                            "verb": verb,
                            "target": target[:200],  # Limit length
                            "description": match.group(0)[:300],
                            "message_index": i,
                            "timestamp": datetime.now().isoformat()
                        })

        return actions

    @staticmethod
    def extract_decisions(content: str, messages: List[Dict]) -> List[Dict]:
        """Extract decisions and choices made"""
        decisions = []

        decision_patterns = [
            # Direct decisions
            r'\b(decid(ed|ing)?|chose|chosen|opt(ed|ing)?|select(ed|ing)?)\s+to\s+([^\n.!?]+)',
            r"\b(we'?ll|I'll|let's)\s+(use|go\s+with|implement|build|create)\s+([^\n.!?]+)",
            r'\b(?:the\s+)?(best|better|recommended|preferred)\s+(?:option|approach|way|solution|method)\s+is\s+to\s+([^\n.!?]+)',
            r'\binstead\s+of\s+([^,]+),?\s+(?:we|I)(?:\'ll)?\s+([^\n.!?]+)',

            # Architectural choices
            r'\b(?:using|use|with|via)\s+([A-Z][A-Za-z0-9_-]+)\s+(?:for|as|to)\s+([^\n.!?]+)',
            r'\bgoing\s+to\s+use\s+([^\n.!?]+)',
        ]

        for pattern in decision_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                decision_text = match.group(0).strip()

                # Get context (surrounding text)
                start = max(0, match.start() - 150)
                end = min(len(content), match.end() + 150)
                context = content[start:end].strip()

                decisions.append({
                    "decision": decision_text[:300],
                    "context": context[:500],
                    "extracted_at": datetime.now().isoformat()
                })

        # Deduplicate similar decisions
        unique_decisions = []
        seen = set()
        for dec in decisions:
            # Simple dedup by first 50 chars
            key = dec['decision'][:50].lower()
            if key not in seen:
                seen.add(key)
                unique_decisions.append(dec)

        return unique_decisions

    @staticmethod
    def extract_problems_solutions(content: str, messages: List[Dict]) -> List[Dict]:
        """Extract problems and their solutions"""
        problems = []

        # Problem indicators
        problem_patterns = [
            r'\b(?:problem|issue|error|bug|broken|not\s+working|doesn\'?t\s+work|fails?|failing)\s*:?\s*([^\n.!?]+)',
            r'(?:getting|receiving|seeing)\s+(?:an?\s+)?(?:error|exception|issue)\s*:?\s*([^\n]+)',
            r'\b([^\n]+?)\s+(?:is\s+not\s+working|doesn\'?t\s+work|won\'?t\s+work|isn\'?t\s+working)',
        ]

        # Solution indicators
        solution_patterns = [
            r'\b(?:fix(?:ed)?|solv(?:ed)?|resolv(?:ed)?|patch(?:ed)?)\s+(?:by|with|using)\s+([^\n.!?]+)',
            r'\b(?:solution|fix|answer)\s*:?\s*([^\n]+)',
            r'to\s+fix\s+this[,:]?\s+([^\n]+)',
        ]

        # Extract problems with context
        for i, msg in enumerate(messages):
            msg_content = msg.get("content", "")

            # Look for problems
            for pattern in problem_patterns:
                matches = re.finditer(pattern, msg_content, re.IGNORECASE)
                for match in matches:
                    problem_desc = match.group(1) if match.lastindex >= 1 else match.group(0)
                    problem_desc = problem_desc.strip()

                    # Look for solution in next few messages
                    solution_desc = None
                    for j in range(i, min(i + 5, len(messages))):
                        sol_content = messages[j].get("content", "")
                        for sol_pattern in solution_patterns:
                            sol_match = re.search(sol_pattern, sol_content, re.IGNORECASE)
                            if sol_match:
                                solution_desc = sol_match.group(1) if sol_match.lastindex >= 1 else sol_match.group(0)
                                solution_desc = solution_desc.strip()
                                break
                        if solution_desc:
                            break

                    # Skip garbage: require minimum length for both problem and solution
                    if len(problem_desc) < 20 or len(problem_desc.split()) < 4:
                        continue
                    if solution_desc and solution_desc != "Not explicitly stated":
                        if len(solution_desc) < 20 or len(solution_desc.split()) < 4:
                            solution_desc = None

                    problems.append({
                        "problem": problem_desc[:300],
                        "solution": solution_desc[:300] if solution_desc else "Not explicitly stated",
                        "message_index": i,
                        "timestamp": datetime.now().isoformat()
                    })

        # Deduplicate
        unique_problems = []
        seen = set()
        for prob in problems:
            key = prob['problem'][:50].lower()
            if key not in seen:
                seen.add(key)
                unique_problems.append(prob)

        return unique_problems

    @staticmethod
    def extract_code_snippets_simple(content: str, messages: List[Dict]) -> List[Dict]:
        """Extract code blocks from markdown"""
        code_snippets = []

        # Match markdown code blocks
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            language = match.group(1) or "unknown"
            code = match.group(2).strip()

            if len(code) > 10:  # Ignore trivial snippets
                code_snippets.append({
                    "language": language.lower(),
                    "code": code[:2000],  # Limit size
                    "lines": len(code.split('\n')),
                    "extracted_at": datetime.now().isoformat()
                })

        return code_snippets

    @staticmethod
    def extract_goals(content: str, messages: List[Dict]) -> List[Dict]:
        """Extract user goals and intentions"""
        goals = []

        goal_patterns = [
            r'\b(?:I|we)\s+(?:want|need|would\s+like|\'d\s+like)\s+to\s+([^\n.!?]+)',
            r'\b(?:goal|objective|aim|purpose)\s+is\s+to\s+([^\n.!?]+)',
            r'\b(?:trying|attempting)\s+to\s+([^\n.!?]+)',
            r'\b(?:plan|planning|intend(?:ing)?)\s+to\s+([^\n.!?]+)',
        ]

        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue

            msg_content = msg.get("content", "")

            for pattern in goal_patterns:
                matches = re.finditer(pattern, msg_content, re.IGNORECASE)
                for match in matches:
                    goal_text = match.group(1).strip()

                    goals.append({
                        "goal": goal_text[:300],
                        "message_index": i,
                        "timestamp": datetime.now().isoformat()
                    })

        # Deduplicate
        unique_goals = []
        seen = set()
        for goal in goals:
            key = goal['goal'][:50].lower()
            if key not in seen:
                seen.add(key)
                unique_goals.append(goal)

        return unique_goals

    @staticmethod
    def extract_preferences(content: str, messages: List[Dict]) -> List[Dict]:
        """Extract user preferences"""
        preferences = []

        pref_patterns = [
            r'\bI\s+prefer\s+([^\n.!?]+)',
            r'\b(?:I|we)\s+(?:like|love|enjoy)\s+using\s+([^\n.!?]+)',
            r'\b(?:I|we)\s+(?:don\'?t|do\s+not)\s+like\s+([^\n.!?]+)',
            r'\b(?:I|we)\s+always\s+use\s+([^\n.!?]+)',
        ]

        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue

            msg_content = msg.get("content", "")

            for pattern in pref_patterns:
                matches = re.finditer(pattern, msg_content, re.IGNORECASE)
                for match in matches:
                    pref_text = match.group(1).strip()

                    # Determine if positive or negative
                    positive = "don't" not in match.group(0).lower() and "do not" not in match.group(0).lower()

                    preferences.append({
                        "preference": pref_text[:300],
                        "positive": positive,
                        "message_index": i,
                        "timestamp": datetime.now().isoformat()
                    })

        return preferences
