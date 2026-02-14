"""
Code Indexer - Extract and index code snippets from conversations
Part of Agent 9: Code Understanding & Pattern Detection
"""
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


class CodeIndexer:
    """
    Extract, classify, and index code snippets from conversations.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.code_index_path = self.base_path / "code_index"
        self.snippets_file = self.code_index_path / "snippets.jsonl"
        self.by_language_path = self.code_index_path / "by_language"
        self.by_project_path = self.code_index_path / "by_project"

        # Create directories
        self.code_index_path.mkdir(parents=True, exist_ok=True)
        self.by_language_path.mkdir(parents=True, exist_ok=True)
        self.by_project_path.mkdir(parents=True, exist_ok=True)

    def extract_code_from_conversation(self, conversation: Dict) -> List[Dict]:
        """
        Extract all code blocks from a conversation.

        Returns:
            List of code snippets with metadata
        """
        snippets = []

        for idx, message in enumerate(conversation.get('messages', [])):
            content = message.get('content', '')

            # Extract code blocks (markdown format)
            code_blocks = self._extract_code_blocks(content)

            for block in code_blocks:
                snippet = {
                    'snippet_id': f"{conversation['id']}_{idx}_{len(snippets)}",
                    'conversation_id': conversation['id'],
                    'message_index': idx,
                    'code': block['code'],
                    'language': block['language'],
                    'context': self._get_context(content, block['code']),
                    'timestamp': conversation['timestamp'],
                    'purpose': self._classify_purpose(content, block['code']),
                    'project': self._detect_project(conversation),
                    'tags': conversation.get('metadata', {}).get('tags', []),
                    'topics': conversation.get('metadata', {}).get('topics', [])
                }

                snippets.append(snippet)

        return snippets

    def _extract_code_blocks(self, text: str) -> List[Dict]:
        """Extract code blocks from markdown text"""
        # Pattern: ```language\ncode\n```
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        code_blocks = []
        for language, code in matches:
            if not language:
                language = self._detect_language(code)

            code_blocks.append({
                'language': language.lower() if language else 'unknown',
                'code': code.strip()
            })

        return code_blocks

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code"""
        # Simple heuristic-based detection
        code_lower = code.lower()

        # TypeScript - check before JavaScript (more specific patterns)
        if ('interface ' in code or ': string' in code or ': number' in code or
            ': boolean' in code or '<T>' in code or 'type ' in code and '=' in code):
            # Check for JSX/TSX (React TypeScript)
            if '<' in code and '/>' in code and ('useState' in code or 'return (' in code):
                return 'tsx'
            return 'typescript'

        # Python
        if 'def ' in code_lower or ('import ' in code_lower and 'from ' in code_lower) or 'class ' in code_lower:
            return 'python'

        # JavaScript/JSX
        if 'function ' in code_lower or 'const ' in code_lower or 'let ' in code_lower or '=>' in code:
            # Check for JSX (React)
            if '<' in code and '/>' in code and ('useState' in code or 'return (' in code):
                return 'jsx'
            return 'javascript'

        # Rust
        if 'fn ' in code or 'let mut ' in code or 'impl ' in code or '-> ' in code:
            return 'rust'

        # Go
        if 'func ' in code or 'package ' in code or 'import "' in code or ':= ' in code:
            return 'go'

        # YAML
        if (code.strip().startswith('-') or ': ' in code) and '{' not in code:
            lines = code.strip().split('\n')
            yaml_lines = sum(1 for line in lines if ': ' in line or line.strip().startswith('-'))
            if yaml_lines > len(lines) * 0.5:
                return 'yaml'

        # Dockerfile
        if any(cmd in code for cmd in ['FROM ', 'RUN ', 'COPY ', 'CMD ', 'ENTRYPOINT ', 'WORKDIR ']):
            return 'dockerfile'

        # PowerShell
        if ('$' in code and '-' in code) or 'Write-Host' in code or 'Get-' in code or 'Set-' in code:
            return 'powershell'

        # Markdown
        if code.startswith('#') and not code.startswith('#!'):
            md_indicators = ['##', '**', '- ', '* ', '```', '[', '](']
            if any(ind in code for ind in md_indicators):
                return 'markdown'

        # HTML
        if '<' in code and '>' in code and ('html' in code_lower or 'div' in code_lower or '<p>' in code_lower):
            return 'html'

        # CSS
        if '{' in code and '}' in code and ('color' in code_lower or 'font' in code_lower or 'margin' in code_lower):
            return 'css'

        # SQL
        if 'SELECT' in code or 'INSERT' in code or 'CREATE TABLE' in code:
            return 'sql'

        # C/C++
        if '#include' in code or 'int main' in code:
            return 'c'

        # Bash/Shell
        if '#!/bin/bash' in code or '#!/bin/sh' in code or ('echo' in code_lower and '$' in code):
            return 'bash'

        # JSON
        if code.strip().startswith('{') and code.strip().endswith('}') and '":' in code:
            return 'json'

        return 'unknown'

    def _get_context(self, full_text: str, code: str, context_chars: int = 200) -> str:
        """Get surrounding context for code block"""
        code_pos = full_text.find(code)
        if code_pos == -1:
            return ""

        start = max(0, code_pos - context_chars)
        end = min(len(full_text), code_pos + len(code) + context_chars)

        context = full_text[start:code_pos] + full_text[code_pos + len(code):end]
        return context.strip()

    def _classify_purpose(self, context: str, code: str) -> str:
        """Classify the purpose of code snippet"""
        context_lower = context.lower()

        if any(word in context_lower for word in ['fix', 'bug', 'error', 'issue', 'problem']):
            return 'bugfix'
        elif any(word in context_lower for word in ['add', 'new', 'implement', 'feature']):
            return 'feature'
        elif any(word in context_lower for word in ['setup', 'install', 'configure', 'init']):
            return 'setup'
        elif any(word in context_lower for word in ['refactor', 'improve', 'optimize', 'clean']):
            return 'refactor'
        elif any(word in context_lower for word in ['test', 'testing', 'spec', 'assert']):
            return 'test'
        elif any(word in context_lower for word in ['example', 'demo', 'sample']):
            return 'example'
        else:
            return 'general'

    def _detect_project(self, conversation: Dict) -> Optional[str]:
        """Detect project from conversation metadata or content"""
        # Check topics for project names
        topics = conversation.get('metadata', {}).get('topics', [])
        for topic in topics:
            if 'project' in topic.lower() or 'cerebral' in topic.lower():
                return topic

        # Check file paths
        file_paths = conversation.get('extracted_data', {}).get('file_paths', [])
        for fp in file_paths:
            path = fp.get('path', '')
            if 'cerebral-interface' in path.lower():
                return 'cerebral-interface'
            elif 'nas' in path.lower():
                return 'nas-projects'

        return None

    def save_snippets(self, snippets: List[Dict], conversation_id: str):
        """Save code snippets to index"""
        if not snippets:
            return

        # Append to main snippets file
        with open(self.snippets_file, 'a', encoding='utf-8') as f:
            for snippet in snippets:
                f.write(json.dumps(snippet, ensure_ascii=False) + '\n')

        # Index by language
        for snippet in snippets:
            lang = snippet['language']
            lang_file = self.by_language_path / f"{lang}.jsonl"
            with open(lang_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(snippet, ensure_ascii=False) + '\n')

        # Index by project
        for snippet in snippets:
            if snippet['project']:
                proj_file = self.by_project_path / f"{snippet['project']}.jsonl"
                with open(proj_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(snippet, ensure_ascii=False) + '\n')

    def search_code(self, query: str, language: Optional[str] = None,
                    project: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Search code snippets by query.

        Args:
            query: Search query (keyword or natural language)
            language: Filter by programming language
            project: Filter by project
            limit: Max results

        Returns:
            List of matching code snippets
        """
        results = []
        query_lower = query.lower()

        # Determine which file to search
        if language:
            search_file = self.by_language_path / f"{language}.jsonl"
        elif project:
            search_file = self.by_project_path / f"{project}.jsonl"
        else:
            search_file = self.snippets_file

        if not search_file.exists():
            return []

        # Search through file
        with open(search_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    snippet = json.loads(line)

                    # Check if query matches
                    if (query_lower in snippet['code'].lower() or
                        query_lower in snippet['context'].lower() or
                        any(query_lower in tag.lower() for tag in snippet.get('tags', []))):

                        results.append(snippet)

                        if len(results) >= limit:
                            break

                except json.JSONDecodeError:
                    continue

        return results

    def get_code_stats(self) -> Dict:
        """Get statistics about indexed code"""
        stats = {
            'total_snippets': 0,
            'by_language': defaultdict(int),
            'by_purpose': defaultdict(int),
            'by_project': defaultdict(int)
        }

        if not self.snippets_file.exists():
            return stats

        with open(self.snippets_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    snippet = json.loads(line)
                    stats['total_snippets'] += 1
                    stats['by_language'][snippet['language']] += 1
                    stats['by_purpose'][snippet['purpose']] += 1
                    if snippet['project']:
                        stats['by_project'][snippet['project']] += 1
                except json.JSONDecodeError:
                    continue

        return {
            'total_snippets': stats['total_snippets'],
            'by_language': dict(stats['by_language']),
            'by_purpose': dict(stats['by_purpose']),
            'by_project': dict(stats['by_project'])
        }
