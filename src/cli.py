"""Cerebro Memory - CLI entry point"""
import asyncio
import sys

__version__ = "0.1.0"


def main():
    """Main CLI dispatcher"""
    args = sys.argv[1:]

    if not args:
        _print_help()
        return

    cmd = args[0]

    if cmd == "serve":
        _serve()
    elif cmd == "init":
        _init(args[1:])
    elif cmd == "doctor":
        _doctor()
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
  cerebro serve       Start the MCP memory server (default)
  cerebro init        Initialize the local memory store
  cerebro doctor      Run a health check
  cerebro --version   Show version
  cerebro --help      Show this help

MCP Config (~/.claude/mcp.json):
  {{
    "mcpServers": {{
      "cerebro": {{
        "command": "cerebro",
        "args": ["serve"]
      }}
    }}
  }}

Environment:
  CEREBRO_DATA_DIR    Base data directory (default: ~/.cerebro/data)
  CEREBRO_LOG_LEVEL   Log level (default: INFO)

Docs: https://github.com/Professor-Low/Cerebro""")


def _serve():
    from . import mcp_ultimate_memory
    asyncio.run(mcp_ultimate_memory.main())


def _init(args):
    from pathlib import Path

    from .config import DATA_DIR, ensure_directories

    storage = DATA_DIR
    # Allow --storage override
    for i, arg in enumerate(args):
        if arg == "--storage" and i + 1 < len(args):
            storage = Path(args[i + 1]).expanduser().resolve()
            break

    print(f"Cerebro Memory v{__version__}")
    print(f"Initializing memory store at: {storage}")
    ensure_directories()
    print()
    print("  Created directory structure:")
    print(f"    {storage}/conversations/")
    print(f"    {storage}/knowledge_base/")
    print(f"    {storage}/learnings/")
    print(f"    {storage}/embeddings/")
    print(f"    {storage}/cache/")
    print("    ... and 10 more")
    print()
    print("Next steps:")
    print('  1. Add to your MCP config (~/.claude/mcp.json):')
    print('     { "mcpServers": { "cerebro": { "command": "cerebro", "args": ["serve"] } } }')
    print("  2. Restart Claude Code")
    print("  3. Run /mcp to verify 49 tools are loaded")
    print()
    print("Done!")


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
