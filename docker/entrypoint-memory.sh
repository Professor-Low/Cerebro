#!/bin/sh
# Cerebro Memory - Data Volume Initializer
# Creates the required directory structure on the shared data volume.
# Runs once and exits cleanly (exit 0).

DATA_DIR="${CEREBRO_DATA_DIR:-/data}"

echo "Cerebro Memory: Initializing data directories at $DATA_DIR"

mkdir -p \
  "$DATA_DIR/conversations" \
  "$DATA_DIR/knowledge_base" \
  "$DATA_DIR/learnings" \
  "$DATA_DIR/embeddings/chunks" \
  "$DATA_DIR/cache" \
  "$DATA_DIR/cache/session_summaries" \
  "$DATA_DIR/cache/archive" \
  "$DATA_DIR/devices" \
  "$DATA_DIR/images" \
  "$DATA_DIR/projects" \
  "$DATA_DIR/patterns" \
  "$DATA_DIR/personality" \
  "$DATA_DIR/corrections" \
  "$DATA_DIR/metrics" \
  "$DATA_DIR/branches"

echo "Cerebro Memory: Data directories ready. Exiting."
exit 0
