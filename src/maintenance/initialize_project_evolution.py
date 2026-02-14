"""
Initialize Project Evolution Tracking
=====================================
Sets up the cerebral-interface project evolution timeline based on
conversations and known project history.

Run this once to establish the project evolution baseline.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from project_evolution import ProjectEvolutionTracker


def initialize_cerebral_interface_evolution():
    """
    Initialize the cerebral-interface project with its known evolution history.
    This helps the search system understand which content is current vs superseded.
    """
    print("=" * 60)
    print("Initializing Project Evolution Tracking")
    print("=" * 60)

    tracker = ProjectEvolutionTracker()

    # Check if already initialized
    existing = tracker.get_project_timeline("cerebral-interface")
    if existing:
        print(f"\ncerebral-interface already has {len(existing)} version(s) tracked.")
        print("Existing versions:")
        for v in existing:
            print(f"  - {v['version']}: {v['summary'][:50]}...")
        response = input("\nReset and reinitialize? (y/N): ")
        if response.lower() != 'y':
            print("Skipping cerebral-interface initialization.")
            return

    print("\nRecording cerebral-interface evolution history...")

    # Record the evolution of cerebral-interface
    evolutions = [
        {
            "version": "v1",
            "summary": "Initial 2D node-based memory visualization with basic graph layout",
            "keywords": ["visualization", "2D", "graph", "nodes", "memory", "basic"],
            "conversation_id": "initial-concept"
        },
        {
            "version": "v2",
            "summary": "Upgraded to 3D Force-Directed Graph with Three.js and OrbitControls",
            "keywords": ["visualization", "3D", "Three.js", "force-directed", "WebGL", "brain"],
            "supersedes_version": "v1",
            "conversation_id": "3d-upgrade"
        },
        {
            "version": "v3",
            "summary": "Current version: Real-time WebSocket updates, animated search ripples, brain-shaped force layout, connection strength visualization",
            "keywords": ["visualization", "3D", "WebSocket", "real-time", "animation", "brain", "ripple", "connection"],
            "supersedes_version": "v2",
            "conversation_id": "realtime-upgrade"
        }
    ]

    for evo in evolutions:
        tracker.record_evolution(
            project_id="cerebral-interface",
            conversation_id=evo["conversation_id"],
            summary=evo["summary"],
            version=evo["version"],
            supersedes_version=evo.get("supersedes_version"),
            keywords=evo["keywords"]
        )
        print(f"  Recorded: {evo['version']} - {evo['summary'][:40]}...")

    print("\n" + "=" * 60)
    print("AI Memory Project Evolution")
    print("=" * 60)

    # Also initialize AI Memory system evolution
    ai_memory_evolutions = [
        {
            "version": "v1",
            "summary": "Basic conversation storage with facts extraction",
            "keywords": ["ai-memory", "conversation", "facts", "storage"],
            "conversation_id": "initial-memory"
        },
        {
            "version": "v2",
            "summary": "Added embeddings, FAISS vector search, and MCP integration",
            "keywords": ["ai-memory", "embeddings", "FAISS", "vector", "MCP", "semantic-search"],
            "supersedes_version": "v1",
            "conversation_id": "embeddings-upgrade"
        },
        {
            "version": "v3",
            "summary": "Current: Recency boosting, project evolution tracking, supersession detection, auto-maintenance",
            "keywords": ["ai-memory", "recency", "evolution", "maintenance", "quality-scoring"],
            "supersedes_version": "v2",
            "conversation_id": "evolution-upgrade"
        }
    ]

    print("\nRecording ai-memory evolution history...")
    for evo in ai_memory_evolutions:
        tracker.record_evolution(
            project_id="ai-memory",
            conversation_id=evo["conversation_id"],
            summary=evo["summary"],
            version=evo["version"],
            supersedes_version=evo.get("supersedes_version"),
            keywords=evo["keywords"]
        )
        print(f"  Recorded: {evo['version']} - {evo['summary'][:40]}...")

    # Show final summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_projects = tracker.get_all_projects()
    print(f"\nTotal projects tracked: {len(all_projects)}")
    for proj in all_projects:
        print(f"\n  {proj['project_id']}:")
        print(f"    Current version: {proj['current_version']}")
        print(f"    Total versions: {proj['version_count']}")
        print(f"    Keywords: {', '.join(proj['keywords'][:5])}...")

    print("\n" + "=" * 60)
    print("Project evolution tracking initialized!")
    print("Search results will now prioritize current content.")
    print("=" * 60)


if __name__ == "__main__":
    initialize_cerebral_interface_evolution()
