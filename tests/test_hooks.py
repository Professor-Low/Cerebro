"""Tests for Cerebro hook scripts and CLI hooks command."""
import subprocess
import sys
from pathlib import Path


def test_hook_source_files_exist():
    """All hook source files should exist in the package."""
    hooks_dir = Path(__file__).parent.parent / "cerebro_ai" / "hooks"
    assert hooks_dir.exists(), f"hooks directory not found at {hooks_dir}"

    expected = [
        "__init__.py",
        "session_start.py",
        "user_prompt_submit.py",
        "pre_compact.py",
        "session_end.py",
    ]
    for filename in expected:
        assert (hooks_dir / filename).exists(), f"Missing hook: {filename}"


def test_hooks_command_shows_usage():
    """'cerebro hooks' with no subcommand should show usage."""
    result = subprocess.run(
        [sys.executable, "-m", "cerebro_ai.cli", "hooks"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "install" in result.stdout
    assert "status" in result.stdout
    assert "uninstall" in result.stdout


def test_hooks_status():
    """'cerebro hooks status' should run without error."""
    result = subprocess.run(
        [sys.executable, "-m", "cerebro_ai.cli", "hooks", "status"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "Cerebro hooks" in result.stdout


def test_hooks_unknown_subcommand():
    """Unknown hooks subcommand should exit with code 1."""
    result = subprocess.run(
        [sys.executable, "-m", "cerebro_ai.cli", "hooks", "bogus"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 1
    assert "Unknown" in result.stdout
