"""Tests for the cerebro CLI entry point."""
import subprocess
import sys


def test_version():
    """--version should print version string and exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--version"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "cerebro" in result.stdout


def test_help():
    """--help should mention serve, init, doctor."""
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "serve" in result.stdout
    assert "init" in result.stdout
    assert "doctor" in result.stdout


def test_unknown_command():
    """Unknown command should exit with code 1."""
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "nonexistent"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 1
    assert "Unknown command" in result.stdout
