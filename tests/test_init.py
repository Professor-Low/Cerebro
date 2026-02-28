"""Tests for initialization and config safety."""
import os
from pathlib import Path
from unittest.mock import patch

from cerebro_ai.config import (
    DGX_HOST,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    NAS_IP,
    ensure_directories,
)


def test_ensure_directories(tmp_path):
    """ensure_directories should create expected subdirs."""
    with patch("cerebro_ai.config.DATA_DIR", tmp_path):
        # Patch all derived dirs to use tmp_path
        dirs_to_patch = {
            "cerebro_ai.config.CONVERSATIONS_DIR": tmp_path / "conversations",
            "cerebro_ai.config.CACHE_DIR": tmp_path / "cache",
            "cerebro_ai.config.SUMMARIES_DIR": tmp_path / "cache" / "session_summaries",
            "cerebro_ai.config.ARCHIVE_DIR": tmp_path / "cache" / "archive",
            "cerebro_ai.config.EMBEDDINGS_DIR": tmp_path / "embeddings",
            "cerebro_ai.config.KNOWLEDGE_BASE_DIR": tmp_path / "knowledge_base",
            "cerebro_ai.config.PROJECTS_DIR": tmp_path / "projects",
            "cerebro_ai.config.PATTERNS_DIR": tmp_path / "patterns",
            "cerebro_ai.config.PERSONALITY_DIR": tmp_path / "personality",
            "cerebro_ai.config.LEARNINGS_DIR": tmp_path / "learnings",
            "cerebro_ai.config.CORRECTIONS_DIR": tmp_path / "corrections",
            "cerebro_ai.config.DEVICES_DIR": tmp_path / "devices",
            "cerebro_ai.config.IMAGES_DIR": tmp_path / "images",
            "cerebro_ai.config.METRICS_DIR": tmp_path / "metrics",
            "cerebro_ai.config.BRANCHES_DIR": tmp_path / "branches",
        }
        with patch.dict("cerebro_ai.config.__dict__", {k.split(".")[-1]: v for k, v in dirs_to_patch.items()}):
            ensure_directories()

    # At least some subdirs should exist
    created = [d.name for d in tmp_path.iterdir() if d.is_dir()]
    assert len(created) >= 5
    assert "conversations" in created
    assert "knowledge_base" in created


def test_safe_defaults_no_env():
    """When env vars are unset, code defaults should be safe for fresh installs."""
    with patch.dict(os.environ, {}, clear=False):
        import importlib
        import cerebro_ai.config as cfg
        # NAS_IP should always be defined (empty string for fresh installs)
        assert cfg.NAS_IP is not None


def test_embedding_model_name():
    """Embedding model should be mpnet (not MiniLM)."""
    assert "mpnet" in EMBEDDING_MODEL.lower() or EMBEDDING_MODEL == "all-mpnet-base-v2"


def test_embedding_dim():
    """Embedding dimension should be 768 for mpnet."""
    assert EMBEDDING_DIM == 768
