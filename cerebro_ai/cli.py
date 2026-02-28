"""
Cerebro Memory - CLI entry point

Copyright (C) 2026 Michael Lopez (Professor-Low)
Licensed under AGPL-3.0 — see LICENSE for details.
"""
import asyncio
import sys

from . import __version__


def _guard_stdio():
    """Ensure stdout/stderr are never None.

    The distlib exe launcher can set sys.stderr=None in Git Bash on Windows.
    Any write to a None stream crashes Python silently. Redirect to devnull
    as a safety net so errors at least don't cause silent exits.
    """
    import os
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")


def main():
    """Main CLI dispatcher"""
    _guard_stdio()
    args = sys.argv[1:]

    if not args:
        _print_help()
        return

    cmd = args[0]

    if cmd == "serve":
        _serve()
    elif cmd == "init":
        _init(args[1:])
    elif cmd in ("doctor", "status"):
        _doctor()
    elif cmd == "hooks":
        _hooks(args[1:])
    elif cmd in ("--help", "-h", "help"):
        _print_help()
    elif cmd in ("--version", "-v"):
        print(f"cerebro {__version__}")
    else:
        print(f"Unknown command: {cmd}")
        _print_help()
        sys.exit(1)


def _print_help():
    print(f"""cerebro v{__version__} - Cognitive memory for AI agents

Usage:
  cerebro serve            Start the MCP memory server (default)
  cerebro init             Initialize the local memory store
  cerebro hooks install    Install Claude Code lifecycle hooks
  cerebro hooks status     Show installed hook status
  cerebro hooks uninstall  Remove Cerebro hooks
  cerebro doctor           Run a health check
  cerebro status           Alias for doctor
  cerebro --version        Show version
  cerebro --help           Show this help

MCP Config (~/.claude/mcp.json):
  {{
    "mcpServers": {{
      "cerebro": {{
        "command": "cerebro",
        "args": ["serve"]
      }}
    }}
  }}

  Alternative (works everywhere, no PATH needed):
  {{
    "mcpServers": {{
      "cerebro": {{
        "command": "python",
        "args": ["-m", "cerebro_ai", "serve"]
      }}
    }}
  }}

  Run 'cerebro init' to auto-detect the correct path for your system.

Environment:
  CEREBRO_DATA_DIR    Base data directory (default: ~/.cerebro/data)
  CEREBRO_LOG_LEVEL   Log level (default: INFO)

Docs: https://github.com/Professor-Low/Cerebro""")


def _hooks(args):
    """Manage Claude Code lifecycle hooks."""
    if not args:
        print("Usage: cerebro hooks <install|status|uninstall> [--force]")
        return

    subcmd = args[0]
    force = "--force" in args

    if subcmd == "install":
        _hooks_install(force)
    elif subcmd == "status":
        _hooks_status()
    elif subcmd == "uninstall":
        _hooks_uninstall()
    else:
        print(f"Unknown hooks subcommand: {subcmd}")
        print("Usage: cerebro hooks <install|status|uninstall>")
        sys.exit(1)


# Hook name → (source filename, Claude Code event name)
_HOOK_DEFS = {
    "session_start": ("session_start.py", "SessionStart"),
    "user_prompt_submit": ("user_prompt_submit.py", "UserPromptSubmit"),
    "pre_compact": ("pre_compact.py", "PreCompact"),
    "session_end": ("session_end.py", "SessionEnd"),
}


def _hooks_install(force=False):
    """Install Cerebro hooks into Claude Code settings."""
    import json
    import shutil
    from pathlib import Path

    hooks_src = Path(__file__).parent / "hooks"
    claude_dir = Path.home() / ".claude"
    hooks_dest = claude_dir / "hooks"
    settings_path = claude_dir / "settings.json"

    if not hooks_src.exists():
        print("Error: Hook source files not found in package.")
        sys.exit(1)

    hooks_dest.mkdir(parents=True, exist_ok=True)

    installed = []
    skipped = []

    for name, (filename, _event) in _HOOK_DEFS.items():
        src_file = hooks_src / filename
        dest_file = hooks_dest / f"cerebro_{filename}"

        if not src_file.exists():
            print(f"  Warning: {filename} not found in package, skipping")
            continue

        if dest_file.exists() and not force:
            skipped.append(name)
            continue

        shutil.copy2(src_file, dest_file)
        installed.append(name)

    # Update settings.json
    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    hooks_config = settings.get("hooks", {})

    for name, (filename, event) in _HOOK_DEFS.items():
        dest_file = hooks_dest / f"cerebro_{filename}"
        if not dest_file.exists():
            continue

        command = f'"{sys.executable}" "{dest_file}"'
        entry = {"type": "command", "command": command}

        # Get or create the event's hook list
        event_hooks = hooks_config.get(event, [])

        # Check if we already have a Cerebro hook for this event
        already_exists = False
        for rule in event_hooks:
            for hook in rule.get("hooks", []):
                if "cerebro_" in hook.get("command", ""):
                    if force:
                        hook["command"] = command
                    already_exists = True
                    break

        if not already_exists:
            event_hooks.append({"hooks": [entry]})

        hooks_config[event] = event_hooks

    settings["hooks"] = hooks_config
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    print(f"Cerebro hooks — v{__version__}")
    if installed:
        print(f"  Installed: {', '.join(installed)}")
    if skipped:
        print(f"  Skipped (use --force to overwrite): {', '.join(skipped)}")
    if not installed and not skipped:
        print("  No hooks to install.")
    else:
        print(f"  Settings:  {settings_path}")
        print()
        print("Hooks will activate on next Claude Code session.")


def _hooks_status():
    """Show status of installed Cerebro hooks."""
    import json
    from pathlib import Path

    claude_dir = Path.home() / ".claude"
    hooks_dir = claude_dir / "hooks"
    settings_path = claude_dir / "settings.json"

    print(f"Cerebro hooks — v{__version__}")
    print()

    # Check hook files
    for name, (filename, event) in _HOOK_DEFS.items():
        dest_file = hooks_dir / f"cerebro_{filename}"
        status = "installed" if dest_file.exists() else "not installed"
        print(f"  {name:<25} {status}")

    # Check settings.json registration
    print()
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            hooks_config = settings.get("hooks", {})
            registered = []
            for _name, (_filename, event) in _HOOK_DEFS.items():
                for rule in hooks_config.get(event, []):
                    for hook in rule.get("hooks", []):
                        if "cerebro_" in hook.get("command", ""):
                            registered.append(event)
                            break
            if registered:
                print(f"  Registered in settings.json: {', '.join(registered)}")
            else:
                print("  Not registered in settings.json")
        except (json.JSONDecodeError, OSError):
            print("  Could not read settings.json")
    else:
        print("  settings.json not found")


def _hooks_uninstall():
    """Remove Cerebro hooks from Claude Code."""
    import json
    from pathlib import Path

    claude_dir = Path.home() / ".claude"
    hooks_dir = claude_dir / "hooks"
    settings_path = claude_dir / "settings.json"

    removed = []

    # Remove hook files
    for name, (filename, _event) in _HOOK_DEFS.items():
        dest_file = hooks_dir / f"cerebro_{filename}"
        if dest_file.exists():
            dest_file.unlink()
            removed.append(name)

    # Clean settings.json
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            hooks_config = settings.get("hooks", {})

            for _name, (_filename, event) in _HOOK_DEFS.items():
                if event in hooks_config:
                    # Remove rules containing cerebro_ hooks
                    hooks_config[event] = [
                        rule for rule in hooks_config[event]
                        if not any("cerebro_" in h.get("command", "") for h in rule.get("hooks", []))
                    ]
                    # Remove empty event lists
                    if not hooks_config[event]:
                        del hooks_config[event]

            settings["hooks"] = hooks_config
            if not hooks_config:
                del settings["hooks"]

            settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        except (json.JSONDecodeError, OSError):
            pass

    print(f"Cerebro hooks — v{__version__}")
    if removed:
        print(f"  Removed: {', '.join(removed)}")
        print("  Cleaned settings.json")
    else:
        print("  No Cerebro hooks found to remove.")


def _serve():
    from . import mcp_ultimate_memory
    asyncio.run(mcp_ultimate_memory.main())


def _init(args):
    import os
    from pathlib import Path

    storage = None
    # Allow --storage override
    for i, arg in enumerate(args):
        if arg == "--storage" and i + 1 < len(args):
            storage = Path(args[i + 1]).expanduser().resolve()
            break

    if storage is not None:
        # Set env var so config module picks up the custom path
        os.environ["CEREBRO_DATA_DIR"] = str(storage)

    # Import config AFTER potentially setting env var
    from .config import DATA_DIR, ensure_directories

    print(f"Cerebro Memory v{__version__}")
    print(f"Initializing memory store at: {DATA_DIR}")
    ensure_directories()
    print()
    print("  Created directory structure:")
    print(f"    {DATA_DIR}/conversations/")
    print(f"    {DATA_DIR}/knowledge_base/")
    print(f"    {DATA_DIR}/learnings/")
    print(f"    {DATA_DIR}/embeddings/")
    print(f"    {DATA_DIR}/cache/")
    print("    ... and 10 more")
    print()

    # Detect the best command for MCP config
    cerebro_cmd = _resolve_cerebro_command()
    # Escape backslashes for JSON on Windows
    cmd_json = cerebro_cmd.replace("\\", "\\\\")

    print("Next steps:")
    print('  1. Add to your MCP config (~/.claude/mcp.json):')
    print(f'     {{ "mcpServers": {{ "cerebro": {{ "command": "{cmd_json}", "args": ["serve"] }} }} }}')
    print("  2. Restart Claude Code")
    print("  3. Run /mcp to verify 49 tools are loaded")
    if storage is not None:
        print()
        print(f"  Note: Set CEREBRO_DATA_DIR={storage} in your MCP config env")
        print("  so the server uses this custom path.")
    print()
    print("Done!")


def _resolve_cerebro_command():
    """Find the best command path for MCP config.

    Returns the bare 'cerebro' if it's in PATH, otherwise the absolute path
    to the executable so MCP servers work regardless of PATH configuration.
    """
    import os
    import shutil
    import sysconfig
    from pathlib import Path

    # If 'cerebro' is in PATH, use the bare name (simplest, portable)
    if shutil.which("cerebro"):
        return "cerebro"

    # Not in PATH — search known script directories
    exe_name = "cerebro.exe" if sys.platform == "win32" else "cerebro"

    # Check user scripts directory (pip install --user, or default on Windows)
    try:
        user_scripts = sysconfig.get_path("scripts", f"{os.name}_user")
        candidate = Path(user_scripts) / exe_name
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass

    # Check system scripts directory
    try:
        sys_scripts = sysconfig.get_path("scripts")
        candidate = Path(sys_scripts) / exe_name
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass

    # Fallback: bare name and let the user figure it out
    return "cerebro"


def _doctor():
    from .config import DATA_DIR, get_platform_info

    info = get_platform_info()
    print(f"Cerebro Memory v{__version__} - Health Check")
    print()
    print(f"  Platform:        {info['platform']}")
    print(f"  Data directory:  {info['base_path']}")
    print(f"  Embedding model: {info['embedding_model']}")
    print(f"  LLM endpoint:    {info['llm_url']}")
    print(f"  NAS storage:     {info['nas_path']}")
    if info.get('nas_connected'):
        print("  NAS status:      Connected")
    elif info['nas_path'] != "(not configured)":
        print("  NAS status:      Disconnected")
    print()

    # Check data directory
    if DATA_DIR.exists():
        subdirs = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
        print(f"  Storage:         OK ({len(subdirs)} directories)")
        # Count conversations
        conv_dir = DATA_DIR / "conversations"
        if conv_dir.exists():
            convs = list(conv_dir.glob("*.json"))
            print(f"  Conversations:   {len(convs)}")
        # Count learnings
        learn_dir = DATA_DIR / "learnings"
        if learn_dir.exists():
            learns = list(learn_dir.glob("*.json"))
            print(f"  Learnings:       {len(learns)}")
        # Check embeddings / FAISS index
        emb_dir = DATA_DIR / "embeddings"
        if emb_dir.exists():
            # Check all known FAISS index locations and extensions
            faiss_candidates = [
                emb_dir / "indexes" / "faiss_index.bin",
                emb_dir / "indexes" / "faiss_index.faiss",
                *list(emb_dir.glob("*.faiss")),
                *list(emb_dir.glob("**/*.bin")),
            ]
            faiss_found = [f for f in faiss_candidates if f.exists() and f.stat().st_size > 0]
            if faiss_found:
                largest = max(faiss_found, key=lambda f: f.stat().st_size)
                size_mb = largest.stat().st_size / (1024 * 1024)
                print(f"  FAISS index:     OK ({size_mb:.1f} MB)")
            else:
                print("  FAISS index:     Not found (run search to build)")
    else:
        print("  Storage:         NOT INITIALIZED")
        print("  Run 'cerebro init' to set up the memory store.")

    print()
    print("All checks passed." if DATA_DIR.exists() else "Run 'cerebro init' first.")


if __name__ == "__main__":
    main()
