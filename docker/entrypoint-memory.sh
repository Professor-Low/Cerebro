#!/bin/sh
# Cerebro Memory - Data Volume Initializer
# Creates the required directory structure on the shared data volume.
# Runs once and exits cleanly (exit 0).

DATA_DIR="${AI_MEMORY_PATH:-/data/memory}"

echo "Cerebro Memory: Initializing data directories at $DATA_DIR"

mkdir -p \
  "$DATA_DIR/conversations" \
  "$DATA_DIR/knowledge_base" \
  "$DATA_DIR/learnings" \
  "$DATA_DIR/embeddings/chunks" \
  "$DATA_DIR/cache" \
  "$DATA_DIR/entities" \
  "$DATA_DIR/timeline" \
  "$DATA_DIR/devices" \
  "$DATA_DIR/agents" \
  "$DATA_DIR/agent_contexts" \
  "$DATA_DIR/summaries" \
  "$DATA_DIR/goals" \
  "$DATA_DIR/episodic" \
  "$DATA_DIR/semantic" \
  "$DATA_DIR/causal" \
  "$DATA_DIR/working_memory" \
  "$DATA_DIR/images"

echo "Cerebro Memory: Data directories ready. Exiting."
exit 0
