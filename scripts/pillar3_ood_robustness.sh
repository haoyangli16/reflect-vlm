#!/bin/bash
# Pillar 3: OOD Robustness Sweep
# Tests BC+RoMemo with mixed In-Domain/OOD memory (N=100 fixed).

export GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
export METHODS="bc_romemo"
BASE_LOG_DIR="logs/pillar3_ood"

echo "Running Pillar 3: OOD Robustness..."

for ratio in 100 80 60 40 20; do
  export INIT_MEMORY_PATH="data/mem_mix_${ratio}pct.pt"
  export SAVE_ROOT="${BASE_LOG_DIR}/ratio_${ratio}pct"
  
  if [ ! -f "$INIT_MEMORY_PATH" ]; then
    echo "Warning: $INIT_MEMORY_PATH not found, skipping."
    continue
  fi

  echo "Testing In-Domain Ratio: ${ratio}%"
  NUM_TRAJS=100 LEVEL="all" bash scripts/smoke_test_romemo_parallel.sh
done
