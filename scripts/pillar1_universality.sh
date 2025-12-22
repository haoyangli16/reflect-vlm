#!/bin/bash
# Pillar 1: Universality Sweep
# Compares BC, Reflect, MCTS with and without RoMemo (N=100 Memory).

export GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
export INIT_MEMORY_PATH="data/mem_100.pt" 
export SAVE_ROOT="logs/pillar1_universality"

# Check if memory exists
if [ ! -f "$INIT_MEMORY_PATH" ]; then
    echo "Error: $INIT_MEMORY_PATH not found. Run step1_make_memory_subsets.py first."
    exit 1
fi

echo "Running Pillar 1: Universality..."
export METHODS="bc,bc_romemo,reflect,reflect_romemo,mcts,mcts_romemo"

# Run 100 evaluation episodes
NUM_TRAJS=100 LEVEL="all" bash scripts/smoke_test_romemo_parallel.sh