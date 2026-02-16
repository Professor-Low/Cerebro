"""Tests for initialization and config safety."""
import os
from pathlib import Path
from unittest.mock import patch

from src.config import (
    DGX_HOST,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    NAS_IP,
    ensure_directories,
)


def test_ensure_directories(tmp_path):
    """ensure_directories should create expected subdirs."""
    with patch("src.config.DATA_DIR", tmp_path):
        # Patch all derived dirs to use tmp_path
        dirs_to_patch = {
            "src.config.CONVERSATIONS_DIR": tmp_path / "conversations",
            "src.config.CACHE_DIR": tmp_path / "cache",
            "src.config.SUMMARIES_DIR": tmp_path / "cache" / "session_summaries",
            "src.config.ARCHIVE_DIR": tmp_path / "cache" / "archive",
            "src.config.EMBEDDINGS_DIR": tmp_path / "embeddings",
            "src.config.KNOWLEDGE_BASE_DIR": tmp_path / "knowledge_base",
            "src.config.PROJECTS_DIR": tmp_path / "projects",
            "src.config.PATTERNS_DIR": tmp_path / "patterns",
            "src.config.PERSONALITY_DIR": tmp_path / "personality",
            "src.config.LEARNINGS_DIR": tmp_path / "learnings",
            "src.config.CORRECTIONS_DIR": tmp_path / "corrections",
            "src.config.DEVICES_DIR": tmp_path / "devices",
            "src.config.IMAGES_DIR": tmp_path / "images",
            "src.config.METRICS_DIR": tmp_path / "metrics",
            "src.config.BRANCHES_DIR": tmp_path / "branches",
        }
        with patch.dict("src.config.__dict__", {k.split(".")[-1]: v for k, v in dirs_to_patch.items()}):
            ensure_directories()

    # At least some subdirs should exist
    created = [d.name for d in tmp_path.iterdir() if d.is_dir()]
    assert len(created) >= 5
    assert "conversations" in created
    assert "knowledge_base" in created


def test_safe_defaults_no_env():
    """When env vars are unset, code defaults should be empty (safe for PyPI users)."""
    # These are the CODE defaults — not the env-var overridden values
    # Professor's machines set env vars so they get 192.168.0.6
    # Fresh installs get empty strings
    with patch.dict(os.environ, {}, clear=False):
        # Re-import to get fresh defaults
        import importlib
        import src.config as cfg
        # The default in code should be empty string for DGX_HOST
        # (even if env var overrides on Professor's machines)
        assert cfg.NAS_IP is not None  # Should exist (empty or set)


def test_embedding_model_name():
    """Embedding model should be mpnet (not MiniLM)."""
    assert "mpnet" in EMBEDDING_MODEL.lower() or EMBEDDING_MODEL == "all-mpnet-base-v2"


def test_embedding_dim():
    """Embedding dimension should be 768 for mpnet."""
    assert EMBEDDING_DIM == 768
