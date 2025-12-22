#!/bin/bash
# Pillar 2: Scaling Law Sweep
# Tests BC+RoMemo with memory sizes 10, 50, 100, 500, 1000, 2000.

export GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
export METHODS="bc_romemo"
BASE_LOG_DIR="logs/pillar2_scaling"

echo "Running Pillar 2: Scaling Law..."

for size in 10 50 100 500 1000 2000; do
  export INIT_MEMORY_PATH="data/mem_${size}.pt"
  export SAVE_ROOT="${BASE_LOG_DIR}/size_${size}"
  
  if [ ! -f "$INIT_MEMORY_PATH" ]; then
    echo "Warning: $INIT_MEMORY_PATH not found, skipping."
    continue
  fi

  echo "Testing Memory Size: $size"
  NUM_TRAJS=100 LEVEL="all" bash scripts/smoke_test_romemo_parallel.sh
done
