#!/bin/bash
# Pillar 1: Universality Sweep
# Compares BC and Reflect with and without RoMemo (N=100 Memory).
#
# Env overrides:
#   GPUS="0,1,2,3,4,5,6,7"
#   RETRIEVAL_MODE="visual" (or "symbolic" or "hybrid")
#   SYMBOLIC_WEIGHT=0.5 (for hybrid mode)

export GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
export INIT_MEMORY_PATH="data/three_pillars/memory_subsets/mem_100.pt"

# NEW: Retrieval mode for state-query based retrieval
export RETRIEVAL_MODE="${RETRIEVAL_MODE:-visual}"
export SYMBOLIC_WEIGHT="${SYMBOLIC_WEIGHT:-0.5}"

# Include retrieval mode in output dir
export SAVE_ROOT="${SAVE_ROOT:-logs/pillar1_universality_${RETRIEVAL_MODE}}"

# Check if memory exists
if [ ! -f "$INIT_MEMORY_PATH" ]; then
    echo "Error: $INIT_MEMORY_PATH not found. Run step1_make_memory_subsets.py first."
    exit 1
fi

echo "Running Pillar 1: Universality..."
echo "Retrieval mode: ${RETRIEVAL_MODE}"
export METHODS="bc,bc_romemo,bc_romemo_wb,reflect,reflect_romemo,reflect_romemo_wb"

# Run 100 evaluation episodes
NUM_TRAJS=100 LEVEL="all" bash scripts/smoke_test_romemo_parallel.sh