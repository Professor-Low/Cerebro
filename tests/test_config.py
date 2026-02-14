"""Smoke tests for cerebro-memory configuration."""
from src.config import DATA_DIR, get_platform_info


def test_data_dir_is_set():
    """DATA_DIR should resolve to a path."""
    assert DATA_DIR is not None
    assert str(DATA_DIR) != ""


def test_platform_info():
    """get_platform_info should return a dict with expected keys."""
    info = get_platform_info()
    assert isinstance(info, dict)
    assert "platform" in info
