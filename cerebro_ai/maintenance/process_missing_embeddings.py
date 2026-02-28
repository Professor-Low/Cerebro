"""
Process Missing Embeddings
==========================
Finds conversations without corresponding chunk/embedding files
and processes them with full embedding generation.

This fixes the issue where 62+ conversations were missing from
the semantic search index.
"""

import json
import os
import sys
import time
from pathlib import Path

# ENABLE embeddings (the opposite of what rebuild_all_chunks.py does)
os.environ['ENABLE_EMBEDDINGS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from ai_embeddings_engine import EmbeddingsEngine


def find_missing_embeddings(base_path: Path) -> list:
    """Find conversations that don't have corresponding chunk files"""
    conversations_path = base_path / "conversations"
    chunks_path = base_path / "embeddings" / "chunks"

    # Get all conversation IDs
    conv_files = list(conversations_path.glob("*.json"))
    conv_ids = set()
    for f in conv_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                conv_ids.add((f, data.get('id', f.stem)))
        except:
            conv_ids.add((f, f.stem))

    # Get all chunk file IDs
    chunk_files = list(chunks_path.glob("*.jsonl"))
    chunk_ids = {f.stem for f in chunk_files}

    # Find missing
    missing = [(f, cid) for f, cid in conv_ids if cid not in chunk_ids]

    return missing


def process_missing_embeddings(dry_run: bool = False):
    """Process all conversations missing embeddings"""

    from config import DATA_DIR
    base_path = DATA_DIR

    print("=" * 60)
    print("AI Memory - Process Missing Embeddings")
    print("=" * 60)
    print(f"Base path: {base_path}")
    print()

    # Find missing
    missing = find_missing_embeddings(base_path)
    print(f"Found {len(missing)} conversations missing embeddings")

    if not missing:
        print("All conversations have embeddings!")
        return

    if dry_run:
        print("\n[DRY RUN] Would process:")
        for f, cid in missing[:10]:
            print(f"  - {cid}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        return

    # Initialize engine with embeddings enabled
    print("\nInitializing embeddings engine (loading model)...")
    engine = EmbeddingsEngine(base_path=str(base_path))

    # Process each missing conversation
    success_count = 0
    error_count = 0

    start_time = time.time()

    for i, (conv_file, conv_id) in enumerate(missing, 1):
        try:
            # Load conversation
            with open(conv_file, "r", encoding="utf-8") as f:
                conversation = json.load(f)

            # Process with full embedding pipeline
            # This chunks, embeds, saves chunks, saves vectors
            result_id = engine.process_conversation(conversation)

            if result_id:
                success_count += 1
                print(f"[{i}/{len(missing)}] Processed: {conv_id}")
            else:
                print(f"[{i}/{len(missing)}] No result for: {conv_id}")

        except Exception as e:
            error_count += 1
            print(f"[{i}/{len(missing)}] ERROR: {conv_id} - {e}")

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Conversations processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print()

    # Verify
    chunks_path = base_path / "embeddings" / "chunks"
    chunk_files = list(chunks_path.glob("*.jsonl"))
    print(f"Total chunk files now: {len(chunk_files)}")

    return {"success": success_count, "errors": error_count}


if __name__ == "__main__":
    import sys

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("[DRY RUN MODE - No files will be modified]\n")

    process_missing_embeddings(dry_run=dry_run)
