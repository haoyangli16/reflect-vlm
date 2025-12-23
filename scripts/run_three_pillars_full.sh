#!/bin/bash
# Run the Full Three Pillars Experiment
#
# This script orchestrates the complete experiment:
#   1. Step 0: Generate expert data (2500 in-domain + 2000 OOD)
#   2. Step 1: Create memory subsets (scaling + OOD mixtures)
#   3. Pillar 1: Universality (6 methods × 100 tasks)
#   4. Pillar 2: Scaling (7 sizes × 100 tasks)
#   5. Pillar 3: Robustness (5 OOD ratios × 100 tasks)
#
# Total estimated time: 24-48 GPU hours on 8× A100
#
# Usage:
#   bash scripts/run_three_pillars_full.sh
#
# Env overrides:
#   GPUS="0,1,2,3,4,5,6,7"
#   DATA_ROOT="data/three_pillars"
#   LOGS_ROOT="logs/three_pillars_full"

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
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-base}"
export POST_MODEL_PATH="${POST_MODEL_PATH:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-post-trained}"

# Configuration
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
DATA_ROOT=${DATA_ROOT:-"data/three_pillars"}
LOGS_ROOT=${LOGS_ROOT:-"logs/three_pillars_full"}

# Full experiment parameters
NUM_TRAJS=${NUM_TRAJS:-100}
LEVEL=${LEVEL:-"all"}
MAX_STEPS=${MAX_STEPS:-50}
AGENT_SEEDS=${AGENT_SEEDS:-"0"}
TEST_SEED=${TEST_SEED:-2000000}

# Data generation parameters
IN_DOMAIN_TRAJS=${IN_DOMAIN_TRAJS:-2500}
OOD_TRAJS=${OOD_TRAJS:-2000}
IN_DOMAIN_SEED=${IN_DOMAIN_SEED:-1000000}
OOD_SEED=${OOD_SEED:-9000000}

# Scaling parameters
SCALING_SIZES=${SCALING_SIZES:-"0,10,50,100,500,1000,2000"}

# OOD parameters
OOD_RATIOS=${OOD_RATIOS:-"100,80,60,40,20"}
OOD_TOTAL_SIZE=${OOD_TOTAL_SIZE:-100}

echo "=========================================="
echo "Three Pillars Full Experiment"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  GPUs:           $GPUS"
echo "  Data root:      $DATA_ROOT"
echo "  Logs root:      $LOGS_ROOT"
echo "  Test tasks:     $NUM_TRAJS (seed=$TEST_SEED, level=$LEVEL)"
echo "  Agent seeds:    $AGENT_SEEDS"
echo ""
echo "Data Generation:"
echo "  In-domain:      $IN_DOMAIN_TRAJS episodes (seed=$IN_DOMAIN_SEED)"
echo "  OOD:            $OOD_TRAJS episodes (seed=$OOD_SEED)"
echo ""
echo "Pillar 2 (Scaling):"
echo "  Sizes:          $SCALING_SIZES"
echo ""
echo "Pillar 3 (Robustness):"
echo "  OOD Ratios:     $OOD_RATIOS (total=$OOD_TOTAL_SIZE)"
echo "=========================================="

mkdir -p "$DATA_ROOT" "$LOGS_ROOT"

# Record start time
START_TIME=$(date +%s)

# ============================================
# Step 0: Generate Expert Data
# ============================================
echo ""
echo "=========================================="
echo "Step 0: Generating Expert Data..."
echo "=========================================="

DATA_ROOT="$DATA_ROOT" \
IN_DOMAIN_TRAJS=$IN_DOMAIN_TRAJS \
OOD_TRAJS=$OOD_TRAJS \
IN_DOMAIN_SEED=$IN_DOMAIN_SEED \
OOD_SEED=$OOD_SEED \
LEVEL="$LEVEL" \
MAX_STEPS=$MAX_STEPS \
    bash scripts/step0_generate_expert_data.sh

echo "✓ Step 0 complete"

# ============================================
# Step 1: Create Memory Subsets
# ============================================
echo ""
echo "=========================================="
echo "Step 1: Creating Memory Subsets..."
echo "=========================================="

MEMORY_SUBSET_DIR="$DATA_ROOT/memory_subsets"
mkdir -p "$MEMORY_SUBSET_DIR"

python scripts/step1_make_memory_subsets.py \
    --in_domain "$DATA_ROOT/expert_in_domain_full.pt" \
    --ood "$DATA_ROOT/expert_ood_full.pt" \
    --out_dir "$MEMORY_SUBSET_DIR" \
    --scaling_sizes "$SCALING_SIZES" \
    --ood_ratios "$OOD_RATIOS" \
    --ood_total_size $OOD_TOTAL_SIZE

echo "✓ Step 1 complete"

# ============================================
# Pillar 1: Universality
# ============================================
echo ""
echo "=========================================="
echo "Pillar 1: Universality..."
echo "=========================================="

INIT_MEMORY_PATH="$MEMORY_SUBSET_DIR/mem_100.pt" \
SAVE_ROOT="$LOGS_ROOT/pillar1_universality" \
NUM_TRAJS=$NUM_TRAJS \
LEVEL="$LEVEL" \
TEST_SEED=$TEST_SEED \
MAX_STEPS=$MAX_STEPS \
AGENT_SEEDS="$AGENT_SEEDS" \
    bash scripts/pillar1_universality.sh

echo "✓ Pillar 1 complete"

# ============================================
# Pillar 2: Scaling Law
# ============================================
echo ""
echo "=========================================="
echo "Pillar 2: Scaling Law..."
echo "=========================================="

MEMORY_SUBSET_DIR="$MEMORY_SUBSET_DIR" \
SAVE_ROOT="$LOGS_ROOT/pillar2_scaling" \
NUM_TRAJS=$NUM_TRAJS \
LEVEL="$LEVEL" \
TEST_SEED=$TEST_SEED \
MAX_STEPS=$MAX_STEPS \
SCALING_SIZES="$SCALING_SIZES" \
AGENT_SEEDS="$AGENT_SEEDS" \
    bash scripts/pillar2_scaling.sh

echo "✓ Pillar 2 complete"

# ============================================
# Pillar 3: Robustness
# ============================================
echo ""
echo "=========================================="
echo "Pillar 3: Robustness / OOD Mixture..."
echo "=========================================="

MEMORY_SUBSET_DIR="$MEMORY_SUBSET_DIR" \
SAVE_ROOT="$LOGS_ROOT/pillar3_robustness" \
NUM_TRAJS=$NUM_TRAJS \
LEVEL="$LEVEL" \
TEST_SEED=$TEST_SEED \
MAX_STEPS=$MAX_STEPS \
OOD_RATIOS="$OOD_RATIOS" \
AGENT_SEEDS="$AGENT_SEEDS" \
    bash scripts/pillar3_robustness.sh

echo "✓ Pillar 3 complete"

# ============================================
# Summary
# ============================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo "✅ Three Pillars Experiment Complete!"
echo "=========================================="
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results:"
echo ""
echo "  Pillar 1 (Universality):"
echo "    $LOGS_ROOT/pillar1_universality/_aggregate/results.csv"
echo "    $LOGS_ROOT/pillar1_universality/_aggregate/plots/"
echo ""
echo "  Pillar 2 (Scaling Law):"
echo "    $LOGS_ROOT/pillar2_scaling/_aggregate/scaling_results.csv"
echo "    $LOGS_ROOT/pillar2_scaling/_aggregate/plots/scaling_law.png"
echo ""
echo "  Pillar 3 (Robustness):"
echo "    $LOGS_ROOT/pillar3_robustness/_aggregate/robustness_results.csv"
echo "    $LOGS_ROOT/pillar3_robustness/_aggregate/plots/ood_robustness.png"
echo ""
echo "=========================================="
echo "Paper Claims Supported:"
echo "=========================================="
echo ""
echo "  1. UNIVERSALITY: 'RoMemo improves BC and Reflect at test time'"
echo "     → Check: *_romemo methods have higher SR than baselines"
echo ""
echo "  2. SCALING: 'Memory data scales predictably with log(N)'"
echo "     → Check: scaling_law.png shows diminishing returns curve"
echo ""
echo "  3. ROBUSTNESS: 'Visual retrieval degrades gracefully with OOD noise'"
echo "     → Check: SR drops slowly even at 20% in-domain"
echo ""
echo "=========================================="
