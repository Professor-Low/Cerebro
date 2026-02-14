#!/usr/bin/env python3
"""
Health check system for AI Memory components.
Verifies all systems are operational.

OPTIMIZED: Includes FastHealthChecker for instant checks (<100ms)
"""
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class FastHealthChecker:
    """
    Instant health check - never blocks on NAS. Completes in <100ms.

    Uses socket-only NAS check and skips slow operations like:
    - Loading embedding model
    - Counting conversation files
    - Full directory traversals

    This is the recommended health checker for MCP operations.
    """

    NAS_IP = os.environ.get("CEREBRO_NAS_IP", "")
    NAS_SMB_PORT = 445
    LOCAL_BRAIN_PATH = Path(os.environ.get("CEREBRO_LOCAL_BRAIN_PATH", str(Path.home() / ".cerebro" / "local_brain")))

    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = os.environ.get("CEREBRO_DATA_DIR", str(Path.home() / ".cerebro" / "data"))
        self.base_path = Path(base_path)
        self.results: Dict[str, Dict[str, Any]] = {}

    def check_all(self) -> Dict[str, Any]:
        """Run all fast health checks. Completes in <100ms."""
        start_time = time.time()

        # Run checks in parallel using threads for maximum speed
        threads = []

        def run_nas_check():
            self.results['nas'] = self._check_nas_socket()

        def run_local_check():
            self.results['local_brain'] = self._check_local_brain()

        def run_cache_check():
            self.results['local_cache'] = self._check_local_cache()

        # Start threads
        for fn in [run_nas_check, run_local_check, run_cache_check]:
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            threads.append(t)

        # Wait for all with timeout
        for t in threads:
            t.join(timeout=2.0)

        # Add embeddings status without loading model
        self.results['embeddings'] = {
            'status': 'not_checked',
            'note': 'Skipped for fast health check - use full check to verify model'
        }

        # Compute overall status
        overall = self._compute_overall()

        check_time_ms = round((time.time() - start_time) * 1000)

        return {
            'overall': overall,
            'timestamp': datetime.now().isoformat(),
            'check_time_ms': check_time_ms,
            'check_type': 'fast',
            'summary': {
                'total_checks': len(self.results),
                'healthy': sum(1 for r in self.results.values() if r.get('status') == 'healthy'),
                'degraded': sum(1 for r in self.results.values() if r.get('status') == 'degraded'),
                'down': sum(1 for r in self.results.values() if r.get('status') == 'down')
            },
            'checks': self.results
        }

    def _check_nas_socket(self) -> Dict[str, Any]:
        """Socket-only NAS check with latency measurement."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            start = time.time()
            result = sock.connect_ex((self.NAS_IP, self.NAS_SMB_PORT))
            latency_ms = round((time.time() - start) * 1000)
            sock.close()

            if result == 0:
                return {
                    'status': 'healthy',
                    'latency_ms': latency_ms,
                    'ip': self.NAS_IP,
                    'port': self.NAS_SMB_PORT,
                    'note': 'Socket test only - filesystem not verified'
                }
            else:
                return {
                    'status': 'down',
                    'error': f'Socket connect failed (code {result})',
                    'ip': self.NAS_IP,
                    'port': self.NAS_SMB_PORT
                }
        except socket.timeout:
            return {
                'status': 'down',
                'error': 'Socket timeout (>2s)',
                'ip': self.NAS_IP
            }
        except Exception as e:
            return {
                'status': 'down',
                'error': str(e),
                'ip': self.NAS_IP
            }

    def _check_local_brain(self) -> Dict[str, Any]:
        """Check local brain sync status (pending/failed counts)."""
        try:
            pending_path = self.LOCAL_BRAIN_PATH / "pending"
            failed_path = self.LOCAL_BRAIN_PATH / "failed"

            pending_count = len(list(pending_path.glob("*.json"))) if pending_path.exists() else 0
            failed_count = len(list(failed_path.glob("*.json"))) if failed_path.exists() else 0

            if failed_count > 0:
                status = 'degraded'
            elif pending_count > 10:
                status = 'degraded'
            else:
                status = 'healthy'

            return {
                'status': status,
                'pending_sync': pending_count,
                'failed_sync': failed_count,
                'path': str(self.LOCAL_BRAIN_PATH)
            }
        except Exception as e:
            return {
                'status': 'unknown',
                'error': str(e)
            }

    def _check_local_cache(self) -> Dict[str, Any]:
        """Check if local cache exists and has recent data."""
        try:
            cache_path = self.LOCAL_BRAIN_PATH / "cache"
            if not cache_path.exists():
                return {
                    'status': 'degraded',
                    'note': 'Local cache not initialized',
                    'path': str(cache_path)
                }

            # Check for cached files
            cached_files = list(cache_path.glob("*.json"))

            return {
                'status': 'healthy' if cached_files else 'degraded',
                'cached_files': len(cached_files),
                'path': str(cache_path)
            }
        except Exception as e:
            return {
                'status': 'unknown',
                'error': str(e)
            }

    def _compute_overall(self) -> str:
        """Determine overall system status."""
        status_priority = {'down': 3, 'degraded': 2, 'healthy': 1, 'unknown': 2, 'not_checked': 1}

        max_priority = max(
            (status_priority.get(r.get('status', 'unknown'), 2)
             for r in self.results.values()),
            default=1
        )

        overall_map = {1: 'healthy', 2: 'degraded', 3: 'down'}
        return overall_map.get(max_priority, 'unknown')


class HealthChecker:
    """System health checker for AI Memory components."""

    def __init__(self, base_path: str = ""):
        if not base_path:
            from config import DATA_DIR
            base_path = str(DATA_DIR)
        self.base_path = Path(base_path)
        self.checks = []
        self.results: Dict[str, Dict[str, Any]] = {}

    def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        self.check_nas_connection()
        self.check_embedding_model()
        self.check_indexes()
        self.check_database_structure()
        self.check_mcp_components()
        return self.get_report()

    def check_nas_connection(self):
        """Verify NAS is accessible."""
        try:
            if not self.base_path.exists():
                self.results['nas'] = {
                    'status': 'down',
                    'error': f'NAS path not accessible: {self.base_path}'
                }
                return

            # Check key directories
            conversations_dir = self.base_path / "conversations"
            user_dir = self.base_path / "user"
            embeddings_dir = self.base_path / "embeddings"

            missing_dirs = []
            if not conversations_dir.exists():
                missing_dirs.append("conversations")
            if not user_dir.exists():
                missing_dirs.append("user")
            if not embeddings_dir.exists():
                missing_dirs.append("embeddings")

            if missing_dirs:
                self.results['nas'] = {
                    'status': 'degraded',
                    'warning': f'Missing directories: {", ".join(missing_dirs)}',
                    'path': str(self.base_path)
                }
            else:
                # Count conversations
                conv_count = len(list(conversations_dir.glob("*.json")))

                self.results['nas'] = {
                    'status': 'healthy',
                    'path': str(self.base_path),
                    'conversations_count': conv_count,
                    'accessible_dirs': ['conversations', 'user', 'embeddings']
                }

        except PermissionError as e:
            self.results['nas'] = {
                'status': 'down',
                'error': f'Permission denied: {str(e)}'
            }
        except Exception as e:
            self.results['nas'] = {
                'status': 'down',
                'error': str(e)
            }

    def check_embedding_model(self):
        """Verify embedding model loads."""
        try:
            # Try to import and initialize the embedding engine
            from ai_embeddings_engine import EmbeddingsEngine

            engine = EmbeddingsEngine()

            # Check if model is available
            model_available = hasattr(engine, 'model') and engine.model is not None

            if model_available:
                self.results['embeddings'] = {
                    'status': 'healthy',
                    'model': 'all-MiniLM-L6-v2',
                    'cached': engine._model_cache is not None
                }
            else:
                self.results['embeddings'] = {
                    'status': 'degraded',
                    'note': 'Model not loaded - keyword fallback active',
                    'reason': 'Embeddings disabled or model failed to load'
                }

        except ImportError:
            self.results['embeddings'] = {
                'status': 'down',
                'error': 'ai_embeddings_engine module not found'
            }
        except Exception as e:
            self.results['embeddings'] = {
                'status': 'degraded',
                'error': str(e),
                'fallback': 'keyword search active'
            }

    def check_indexes(self):
        """Verify FAISS indexes exist."""
        try:
            index_path = self.base_path / "embeddings" / "indexes"

            if not index_path.exists():
                self.results['indexes'] = {
                    'status': 'degraded',
                    'note': 'Index directory does not exist',
                    'path': str(index_path)
                }
                return

            # Count index files (check multiple extensions)
            faiss_indexes = list(index_path.glob("*.faiss")) + list(index_path.glob("*faiss*.bin"))
            pkl_indexes = list(index_path.glob("*.pkl")) + list(index_path.glob("*mapping*.json"))

            if len(faiss_indexes) > 0 or len(pkl_indexes) > 0:
                self.results['indexes'] = {
                    'status': 'healthy',
                    'faiss_count': len(faiss_indexes),
                    'pkl_count': len(pkl_indexes),
                    'path': str(index_path)
                }
            else:
                self.results['indexes'] = {
                    'status': 'degraded',
                    'note': 'No index files found - will be created on first search',
                    'path': str(index_path)
                }

        except Exception as e:
            self.results['indexes'] = {
                'status': 'down',
                'error': str(e)
            }

    def check_database_structure(self):
        """Check that required database files exist."""
        try:
            # Check directories instead of specific files for some components
            required_items = {
                'user_profile': (self.base_path / "user" / "profile.json", 'file'),
                'corrections': (self.base_path / "corrections" / "corrections.jsonl", 'file'),
                'sessions': (self.base_path / "session_states", 'dir')
            }

            existing = []
            missing = []

            for name, (path, item_type) in required_items.items():
                if path.exists():
                    existing.append(name)
                else:
                    missing.append(name)

            if not missing:
                self.results['database'] = {
                    'status': 'healthy',
                    'files_present': existing
                }
            else:
                self.results['database'] = {
                    'status': 'degraded',
                    'files_present': existing,
                    'files_missing': missing,
                    'note': 'Missing files will be created automatically'
                }

        except Exception as e:
            self.results['database'] = {
                'status': 'down',
                'error': str(e)
            }

    def check_mcp_components(self):
        """Verify MCP components are available."""
        try:
            # Check if MCP server module exists
            components_found = []
            components_missing = []

            required_components = [
                'mcp_ultimate_memory',
                'auto_context_injector',
                'corrections_tracker',
                'project_tracker',
                'session_continuity'
            ]

            for component in required_components:
                try:
                    __import__(component)
                    components_found.append(component)
                except ImportError:
                    components_missing.append(component)

            if not components_missing:
                self.results['mcp_components'] = {
                    'status': 'healthy',
                    'components': components_found
                }
            else:
                self.results['mcp_components'] = {
                    'status': 'degraded',
                    'components_found': components_found,
                    'components_missing': components_missing
                }

        except Exception as e:
            self.results['mcp_components'] = {
                'status': 'unknown',
                'error': str(e)
            }

    def get_report(self) -> Dict[str, Any]:
        """Generate health report."""
        # Determine overall status
        status_priority = {'down': 3, 'degraded': 2, 'healthy': 1, 'unknown': 2}
        overall_priority = max(
            (status_priority.get(r.get('status', 'unknown'), 2)
             for r in self.results.values()),
            default=1
        )

        overall_map = {1: 'healthy', 2: 'degraded', 3: 'down'}
        overall = overall_map[overall_priority]

        # Generate summary
        healthy_count = sum(1 for r in self.results.values() if r.get('status') == 'healthy')
        degraded_count = sum(1 for r in self.results.values() if r.get('status') == 'degraded')
        down_count = sum(1 for r in self.results.values() if r.get('status') == 'down')

        report = {
            'overall': overall,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_checks': len(self.results),
                'healthy': healthy_count,
                'degraded': degraded_count,
                'down': down_count
            },
            'checks': self.results
        }

        return report

    def print_report(self, report: Dict[str, Any] = None):
        """Print health report in human-readable format."""
        if report is None:
            report = self.get_report()

        print("=" * 60)
        print("AI MEMORY SYSTEM HEALTH CHECK")
        print("=" * 60)
        print(f"Overall Status: {report['overall'].upper()}")
        print(f"Timestamp: {report['timestamp']}")
        print()

        summary = report['summary']
        print(f"Summary: {summary['healthy']} healthy, {summary['degraded']} degraded, {summary['down']} down")
        print()

        for component, result in report['checks'].items():
            status = result.get('status', 'unknown')
            status_symbol = {
                'healthy': '✓',
                'degraded': '⚠',
                'down': '✗',
                'unknown': '?'
            }.get(status, '?')

            print(f"{status_symbol} {component.upper()}: {status}")

            if 'error' in result:
                print(f"  Error: {result['error']}")
            if 'warning' in result:
                print(f"  Warning: {result['warning']}")
            if 'note' in result:
                print(f"  Note: {result['note']}")

            # Print additional details
            for key, value in result.items():
                if key not in ['status', 'error', 'warning', 'note']:
                    if isinstance(value, (list, dict)):
                        print(f"  {key}: {json.dumps(value, ensure_ascii=False)}")
                    else:
                        print(f"  {key}: {value}")

            print()

        print("=" * 60)


def run_health_check(base_path: str = "", verbose: bool = True, fast: bool = False) -> Dict[str, Any]:
    """
    Run health check and return report.

    Args:
        base_path: Path to AI Memory base directory
        verbose: Print report to console
        fast: If True, use FastHealthChecker (instant, <100ms)

    Returns:
        Health report dictionary
    """
    if fast:
        checker = FastHealthChecker(base_path)
        report = checker.check_all()
        if verbose:
            print(f"[Fast Health Check] Overall: {report['overall']} ({report['check_time_ms']}ms)")
        return report

    checker = HealthChecker(base_path)
    report = checker.check_all()

    if verbose:
        checker.print_report(report)

    return report


def run_fast_health_check(base_path: str = "") -> Dict[str, Any]:
    """
    Run instant health check (<100ms). Use this for MCP operations.

    Returns:
        Health report dictionary with check_type: 'fast'
    """
    checker = FastHealthChecker(base_path)
    return checker.check_all()


if __name__ == "__main__":
    # Run health check
    report = run_health_check()

    # Exit with appropriate code
    if report['overall'] == 'down':
        sys.exit(1)
    elif report['overall'] == 'degraded':
        sys.exit(2)
    else:
        sys.exit(0)
