"""Smoke tests — every core module should import without error."""
import importlib

import pytest

CORE_MODULES = [
    "src.config",
    "src.cli",
    "src.validators",
    "src.corrections_tracker",
    "src.device_registry",
    "src.solution_tracker",
    "src.confidence_tracker",
    "src.causal_model",
    "src.goal_tracker",
    "src.preference_manager",
    "src.session_continuity",
    "src.branch_tracker",
    "src.project_tracker",
    "src.secret_detector",
    "src.storage_manager",
    "src.decay_pipeline",
    "src.quality_scorer",
]

# These require optional deps or use bare imports needing sys.path setup
OPTIONAL_MODULES = [
    "src.ai_embeddings_engine",
    "src.embedding_service",
    "src.learning_promoter",
    "src.working_memory",
    "src.privacy_filter",
]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_import(module):
    """Each core module should import without error."""
    importlib.import_module(module)


@pytest.mark.parametrize("module", OPTIONAL_MODULES)
def test_optional_import(module):
    """Optional modules may raise ImportError if deps are missing — that's OK."""
    try:
        importlib.import_module(module)
    except ImportError:
        pytest.skip(f"{module} requires optional dependencies")
