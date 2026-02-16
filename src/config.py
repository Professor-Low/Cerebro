"""
Cerebro Configuration

Cross-platform, environment-variable driven configuration.
All paths default to ~/.cerebro/data and can be overridden.

Environment Variables:
    CEREBRO_DATA_DIR: Base directory for all Cerebro data (default: ~/.cerebro/data)
    CEREBRO_NAS_PATH: NAS mount path (optional, for NAS-backed deployments)
    CEREBRO_EMBEDDING_MODEL: Sentence transformer model (default: all-mpnet-base-v2)
    CEREBRO_EMBEDDING_DIM: Embedding dimensions (default: 768)
    CEREBRO_DEVICE: Embedding compute device (auto/cuda/cpu, default: auto)
    CEREBRO_LOG_LEVEL: Logging level (default: INFO)
    CEREBRO_LLM_URL: Optional LLM endpoint for reasoning features
    CEREBRO_LLM_MODEL: Optional LLM model name
    CEREBRO_PORT: Server port (default: 8400)
    CEREBRO_HOST: Server bind host (default: 0.0.0.0)
    CEREBRO_NAS_IP: NAS IP address (optional)
"""
import os
import platform
import sys
from pathlib import Path

# ============== PLATFORM DETECTION ==============
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MAC = platform.system() == "Darwin"

# ============== BASE DATA DIRECTORY ==============
DATA_DIR = Path(os.environ.get("CEREBRO_DATA_DIR", str(Path.home() / ".cerebro" / "data")))

# Backward-compatible alias: many modules import AI_MEMORY_BASE
AI_MEMORY_BASE = DATA_DIR

# ============== NAS / NETWORK STORAGE (OPTIONAL) ==============
NAS_PATH = Path(os.environ.get("CEREBRO_NAS_PATH", ""))
NAS_IP = os.environ.get("CEREBRO_NAS_IP", "")

# ============== DERIVED PATHS ==============
CONVERSATIONS_DIR = DATA_DIR / "conversations"
CACHE_DIR = DATA_DIR / "cache"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
PROJECTS_DIR = DATA_DIR / "projects"
PATTERNS_DIR = DATA_DIR / "patterns"
PERSONALITY_DIR = DATA_DIR / "personality"
LEARNINGS_DIR = DATA_DIR / "learnings"
CORRECTIONS_DIR = DATA_DIR / "corrections"
DEVICES_DIR = DATA_DIR / "devices"
IMAGES_DIR = DATA_DIR / "images"
METRICS_DIR = DATA_DIR / "metrics"
BRANCHES_DIR = DATA_DIR / "branches"
SUMMARIES_DIR = CACHE_DIR / "session_summaries"
ARCHIVE_DIR = CACHE_DIR / "archive"
IDENTITY_CACHE = CACHE_DIR / "identity_core.json"
DECAY_STATE_FILE = CACHE_DIR / "decay_state.json"

# ============== EMBEDDING CONFIGURATION ==============
EMBEDDING_MODEL = os.environ.get("CEREBRO_EMBEDDING_MODEL", "all-mpnet-base-v2")
EMBEDDING_DIM = int(os.environ.get("CEREBRO_EMBEDDING_DIM", "768"))
EMBEDDING_DEVICE = os.environ.get("CEREBRO_DEVICE", "auto")
# Valid: "auto" (GPU if available, else CPU), "cuda" (force GPU), "cpu" (force CPU)

# ============== DGX / GPU SERVER CONFIGURATION ==============
DGX_HOST = os.environ.get("CEREBRO_DGX_HOST", "")
DGX_OLLAMA_PORT = int(os.environ.get("CEREBRO_DGX_OLLAMA_PORT", "11434"))
DGX_EMBEDDING_PORT = int(os.environ.get("CEREBRO_DGX_EMBEDDING_PORT", "8781"))
DGX_SEARCH_PORT = int(os.environ.get("CEREBRO_DGX_SEARCH_PORT", "8765"))

# ============== LLM CONFIGURATION (OPTIONAL) ==============
LLM_URL = os.environ.get("CEREBRO_LLM_URL", "")
LLM_MODEL = os.environ.get("CEREBRO_LLM_MODEL", "")

# ============== SERVER CONFIGURATION ==============
PORT = int(os.environ.get("CEREBRO_PORT", "8400"))
HOST = os.environ.get("CEREBRO_HOST", "0.0.0.0")
LOG_LEVEL = os.environ.get("CEREBRO_LOG_LEVEL", "INFO")

# ============== PLATFORM-SPECIFIC PATHS ==============
if IS_WINDOWS:
    TEMP_DIR = Path(os.environ.get("TEMP", "C:/Temp"))
    CLAUDE_CONFIG_DIR = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".claude"
else:
    TEMP_DIR = Path("/tmp")
    CLAUDE_CONFIG_DIR = Path.home() / ".claude"

LOG_DIR = TEMP_DIR / "cerebro-logs"
AUTOSAVE_LOG = LOG_DIR / "autosave_hook.log"
WORKER_LOG = LOG_DIR / "autosave_worker.log"

# Optional: venv python path (for subprocess calls)
VENV_PYTHON = Path(os.environ.get("CEREBRO_VENV_PYTHON", sys.executable))


# ============== HELPER FUNCTIONS ==============
def ensure_directories():
    """Create all required directories if they don't exist."""
    dirs = [
        DATA_DIR, CONVERSATIONS_DIR, CACHE_DIR, SUMMARIES_DIR, ARCHIVE_DIR,
        EMBEDDINGS_DIR, KNOWLEDGE_BASE_DIR, PROJECTS_DIR, PATTERNS_DIR,
        PERSONALITY_DIR, LEARNINGS_DIR, CORRECTIONS_DIR, DEVICES_DIR,
        IMAGES_DIR, METRICS_DIR, BRANCHES_DIR, LOG_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_base_path() -> str:
    """Get the base path as string (for backward compatibility)."""
    return str(DATA_DIR)


def validate_nas_connection() -> bool:
    """Check if NAS is accessible."""
    if not NAS_PATH or str(NAS_PATH) == ".":
        return False
    try:
        return NAS_PATH.exists() and DATA_DIR.exists()
    except (OSError, PermissionError):
        return False


def get_platform_info() -> dict:
    """Get current platform configuration."""
    # Detect NAS from either explicit NAS_PATH or DATA_DIR on a network mount
    nas_display = "(not configured)"
    nas_connected = validate_nas_connection()
    if str(NAS_PATH) != "." and str(NAS_PATH):
        nas_display = str(NAS_PATH)
    elif _is_network_path(DATA_DIR):
        nas_display = str(DATA_DIR)
        nas_connected = DATA_DIR.exists()
    return {
        "platform": platform.system(),
        "is_windows": IS_WINDOWS,
        "is_linux": IS_LINUX,
        "base_path": str(DATA_DIR),
        "embedding_model": EMBEDDING_MODEL,
        "llm_url": LLM_URL or "(not configured)",
        "port": PORT,
        "nas_path": nas_display,
        "nas_connected": nas_connected,
    }


def _is_network_path(path: Path) -> bool:
    """Detect if a path is on a network mount (NAS, CIFS, etc.)."""
    s = str(path)
    # Windows: mapped drives (Z:, etc.) or UNC paths
    if IS_WINDOWS and len(s) >= 2 and s[0].isalpha() and s[1] == ":":
        drive = s[0].upper()
        if drive not in ("C", "D"):  # Non-local drives are likely network mounts
            return True
    if s.startswith("\\\\") or s.startswith("//"):
        return True
    # Linux: /mnt/ or /media/ paths (common mount points)
    if s.startswith("/mnt/") or s.startswith("/media/"):
        return True
    return False


# ============== STARTUP VALIDATION ==============
if __name__ == "__main__":
    import json
    ensure_directories()
    print("Cerebro Configuration:")
    print(json.dumps(get_platform_info(), indent=2))

    print("\nDirectory Structure:")
    for name, path in [
        ("Conversations", CONVERSATIONS_DIR),
        ("Cache", CACHE_DIR),
        ("Embeddings", EMBEDDINGS_DIR),
        ("Knowledge Base", KNOWLEDGE_BASE_DIR),
        ("Projects", PROJECTS_DIR),
    ]:
        exists = "OK" if path.exists() else "MISSING"
        print(f"  {name}: {path} [{exists}]")
