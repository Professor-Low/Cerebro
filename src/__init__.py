"""Cerebro AI - Cognitive memory system for AI agents."""

try:
    from importlib.metadata import version

    __version__ = version("cerebro-ai")
except Exception:
    __version__ = "2.0.0"
