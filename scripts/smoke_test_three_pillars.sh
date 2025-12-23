#!/bin/bash
# Smoke Test for the Three Pillars Experiment
#
# This script runs a quick validation of all three pillars with minimal data:
#   - 5 expert episodes for memory generation
#   - 3 test episodes per method
#   - Subset of methods and memory sizes
#
# Use this to verify the pipeline works before running the full experiment.
#
# Usage:
#   bash scripts/smoke_test_three_pillars.sh
#
# Env overrides:
#   GPUS="0,1,2,3,4,5,6,7"
#   SAVE_ROOT="logs/smoke_three_pillars"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TRITON_PTXAS_PATH=""
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=117
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${HF_HOME:-/share/project/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

set -euo pipefail

REFLECT_VLM_ROOT="${REFLECT_VLM_ROOT:-/share/project/lhy/thirdparty/reflect-vlm}"
cd "$REFLECT_VLM_ROOT"

# Local model paths (default to your local directories)
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-base}"
POST_MODEL_PATH="${POST_MODEL_PATH:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-post-trained}"

# Configuration (minimal for smoke test)
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
SAVE_ROOT=${SAVE_ROOT:-"logs/smoke_three_pillars"}
DATA_ROOT="$SAVE_ROOT/data"

# Minimal sizes for smoke test
SMOKE_IN_DOMAIN_TRAJS=10
SMOKE_OOD_TRAJS=10
SMOKE_NUM_TRAJS=3
SMOKE_LEVEL="medium"
SMOKE_MAX_STEPS=30
SMOKE_SCALING_SIZES="0,5,10"
SMOKE_OOD_RATIOS="100,50"

echo "=========================================="
echo "Three Pillars Smoke Test"
echo "=========================================="
echo "This is a quick validation run with minimal data."
echo "GPUs:   $GPUS"
echo "Output: $SAVE_ROOT"
echo "=========================================="

mkdir -p "$SAVE_ROOT" "$DATA_ROOT"

# ============================================
# Step 0: Generate minimal expert data
# ============================================
echo ""
echo "=========================================="
echo "Step 0: Generating minimal expert data..."
echo "=========================================="

IN_DOMAIN_MEMORY="$DATA_ROOT/expert_in_domain_full.pt"
OOD_MEMORY="$DATA_ROOT/expert_ood_full.pt"

if [[ ! -f "$IN_DOMAIN_MEMORY" ]]; then
    echo "Generating in-domain expert data ($SMOKE_IN_DOMAIN_TRAJS episodes)..."
    CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python run-rom.py \
        --seed=1000000 \
        --reset_seed_start=0 \
        --n_trajs=$SMOKE_IN_DOMAIN_TRAJS \
        --save_dir="$DATA_ROOT/expert_in_domain" \
        --start_traj_id=0 \
        --start_board_id=1000000 \
        --logging.online=False \
        --agent_type="expert_romemo_wb" \
        --level="$SMOKE_LEVEL" \
        --oracle_prob=1.0 \
        --record=False \
        --max_steps=$SMOKE_MAX_STEPS \
        --model_path="$POST_MODEL_PATH" \
        --load_4bit=True \
        --romemo_save_memory_path="$IN_DOMAIN_MEMORY"
fi

if [[ ! -f "$OOD_MEMORY" ]]; then
    echo "Generating OOD expert data ($SMOKE_OOD_TRAJS episodes)..."
    CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python run-rom.py \
        --seed=9000000 \
        --reset_seed_start=0 \
        --n_trajs=$SMOKE_OOD_TRAJS \
        --save_dir="$DATA_ROOT/expert_ood" \
        --start_traj_id=0 \
        --start_board_id=9000000 \
        --logging.online=False \
        --agent_type="expert_romemo_wb" \
        --level="$SMOKE_LEVEL" \
        --oracle_prob=1.0 \
        --record=False \
        --max_steps=$SMOKE_MAX_STEPS \
        --model_path="$POST_MODEL_PATH" \
        --load_4bit=True \
        --romemo_save_memory_path="$OOD_MEMORY"
fi

echo "✓ Expert data ready"

# ============================================
# Step 1: Create memory subsets
# ============================================
echo ""
echo "=========================================="
echo "Step 1: Creating memory subsets..."
echo "=========================================="

MEMORY_SUBSET_DIR="$DATA_ROOT/memory_subsets"
mkdir -p "$MEMORY_SUBSET_DIR"

python scripts/step1_make_memory_subsets.py \
    --in_domain "$IN_DOMAIN_MEMORY" \
    --ood "$OOD_MEMORY" \
    --out_dir "$MEMORY_SUBSET_DIR" \
    --scaling_sizes "$SMOKE_SCALING_SIZES" \
    --ood_ratios "$SMOKE_OOD_RATIOS" \
    --ood_total_size 10

echo "✓ Memory subsets ready"

# ============================================
# Pillar 1: Universality (minimal)
# ============================================
echo ""
echo "=========================================="
echo "Pillar 1: Universality (smoke test)..."
echo "=========================================="

# Only test bc and bc_romemo for smoke test
INIT_MEMORY_PATH="$MEMORY_SUBSET_DIR/mem_10.pt" \
SAVE_ROOT="$SAVE_ROOT/pillar1" \
NUM_TRAJS=$SMOKE_NUM_TRAJS \
LEVEL="$SMOKE_LEVEL" \
TEST_SEED=2000000 \
MAX_STEPS=$SMOKE_MAX_STEPS \
METHODS="bc,bc_romemo" \
    bash scripts/smoke_test_romemo_parallel.sh

echo "✓ Pillar 1 smoke test complete"

# ============================================
# Pillar 2: Scaling (minimal)
# ============================================
echo ""
echo "=========================================="
echo "Pillar 2: Scaling (smoke test)..."
echo "=========================================="

MEMORY_SUBSET_DIR="$MEMORY_SUBSET_DIR" \
SAVE_ROOT="$SAVE_ROOT/pillar2" \
NUM_TRAJS=$SMOKE_NUM_TRAJS \
LEVEL="$SMOKE_LEVEL" \
TEST_SEED=2000000 \
MAX_STEPS=$SMOKE_MAX_STEPS \
SCALING_SIZES="$SMOKE_SCALING_SIZES" \
AGENT_SEEDS="0" \
    bash scripts/pillar2_scaling.sh

echo "✓ Pillar 2 smoke test complete"

# ============================================
# Pillar 3: Robustness (minimal)
# ============================================
echo ""
echo "=========================================="
echo "Pillar 3: Robustness (smoke test)..."
echo "=========================================="

MEMORY_SUBSET_DIR="$MEMORY_SUBSET_DIR" \
SAVE_ROOT="$SAVE_ROOT/pillar3" \
NUM_TRAJS=$SMOKE_NUM_TRAJS \
LEVEL="$SMOKE_LEVEL" \
TEST_SEED=2000000 \
MAX_STEPS=$SMOKE_MAX_STEPS \
OOD_RATIOS="$SMOKE_OOD_RATIOS" \
AGENT_SEEDS="0" \
    bash scripts/pillar3_robustness.sh

echo "✓ Pillar 3 smoke test complete"

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "✅ Three Pillars Smoke Test Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Pillar 1: $SAVE_ROOT/pillar1/_aggregate/results.csv"
echo "  Pillar 2: $SAVE_ROOT/pillar2/_aggregate/scaling_results.csv"
echo "  Pillar 3: $SAVE_ROOT/pillar3/_aggregate/robustness_results.csv"
echo ""
echo "Check outputs look reasonable, then run full experiment:"
echo ""
echo "  # Full experiment (24-48 GPU hours)"
echo "  bash scripts/run_three_pillars_full.sh"
