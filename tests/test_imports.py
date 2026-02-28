"""Smoke tests — every core module should import without error."""
import importlib

import pytest

CORE_MODULES = [
    "cerebro_ai.config",
    "cerebro_ai.cli",
    "cerebro_ai.validators",
    "cerebro_ai.corrections_tracker",
    "cerebro_ai.device_registry",
    "cerebro_ai.solution_tracker",
    "cerebro_ai.confidence_tracker",
    "cerebro_ai.causal_model",
    "cerebro_ai.goal_tracker",
    "cerebro_ai.preference_manager",
    "cerebro_ai.session_continuity",
    "cerebro_ai.branch_tracker",
    "cerebro_ai.project_tracker",
    "cerebro_ai.secret_detector",
    "cerebro_ai.storage_manager",
    "cerebro_ai.decay_pipeline",
    "cerebro_ai.quality_scorer",
]

# These require optional deps or use bare imports needing sys.path setup
OPTIONAL_MODULES = [
    "cerebro_ai.ai_embeddings_engine",
    "cerebro_ai.embedding_service",
    "cerebro_ai.learning_promoter",
    "cerebro_ai.working_memory",
    "cerebro_ai.privacy_filter",
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
