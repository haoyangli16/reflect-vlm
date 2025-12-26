#!/bin/bash
# =============================================================================
# Neuro-Symbolic Memory System Comparison Experiment
# =============================================================================
# This script runs the baseline and memory experiments for comparison.
#
# Usage:
#   ./run_comparison.sh                  # Run with defaults (100 episodes)
#   ./run_comparison.sh 50               # Run with 50 episodes
#   ./run_comparison.sh 100 kimi         # Run with 100 episodes using Kimi VLM
# =============================================================================

set -e

# =============================================================================
# Environment Setup (CRITICAL for headless rendering)
# =============================================================================

# MuJoCo EGL rendering (for headless servers without display)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Fix for LLVM command-line option conflict between triton and bitsandbytes
export TRITON_PTXAS_PATH=""

# CUDA setup
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# Force bitsandbytes to use CUDA 11.7 libraries (matching PyTorch cu117)
export BNB_CUDA_VERSION=117

# Performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

# HuggingFace cache settings
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME="${HF_HOME:-/share/project/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" 2>/dev/null || true

# Offline mode to prevent download issues
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Python settings
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true

# =============================================================================
# Configuration
# =============================================================================
N_EPISODES=${1:-100}
VLM_PROVIDER=${2:-rule}  # "rule", "kimi", "openai", "gemini", "qwen"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Model paths
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-./ReflectVLM-llava-v1.5-13b-base}"
export POST_MODEL_PATH="${POST_MODEL_PATH:-./ReflectVLM-llava-v1.5-13b-post-trained}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="logs/neuro_memory_comparison_${TIMESTAMP}"

echo "============================================================"
echo "NEURO-SYMBOLIC MEMORY COMPARISON EXPERIMENT"
echo "============================================================"
echo "Episodes: $N_EPISODES"
echo "VLM Provider: $VLM_PROVIDER"
echo "Save Dir: $SAVE_DIR"
echo "MUJOCO_GL: $MUJOCO_GL"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# Check for API keys if using VLM
if [[ "$VLM_PROVIDER" == "kimi" ]]; then
    if [[ -z "$MOONSHOT_API_KEY" ]]; then
        echo "ERROR: MOONSHOT_API_KEY not set"
        echo "Run: export MOONSHOT_API_KEY='your_key'"
        exit 1
    fi
    echo "Kimi API Key: Set ✓"
elif [[ "$VLM_PROVIDER" == "openai" ]]; then
    if [[ -z "$OPENAI_API_KEY" ]]; then
        echo "ERROR: OPENAI_API_KEY not set"
        exit 1
    fi
elif [[ "$VLM_PROVIDER" == "gemini" ]]; then
    if [[ -z "$GOOGLE_API_KEY" ]]; then
        echo "ERROR: GOOGLE_API_KEY not set"
        exit 1
    fi
elif [[ "$VLM_PROVIDER" == "qwen" ]]; then
    if [[ -z "$DASHSCOPE_API_KEY" ]]; then
        echo "ERROR: DASHSCOPE_API_KEY not set"
        exit 1
    fi
fi

# Create output directory
mkdir -p "$SAVE_DIR"

# ===========================================================================
# Step 1: Run Baseline (No Memory)
# ===========================================================================
echo ""
echo "[1/2] Running BASELINE (No Memory)..."
echo "------------------------------------------------------------"

python experiments/neuro_memory/run_memory_experiment.py \
    --mode baseline \
    --n_episodes $N_EPISODES \
    --save_dir "$SAVE_DIR" \
    --name baseline \
    --base_model "$BASE_MODEL_PATH" \
    --verbose 2>&1 | tee "$SAVE_DIR/baseline.log"

echo ""
echo "Baseline complete. Results: $SAVE_DIR/baseline_baseline_*/"

# ===========================================================================
# Step 2: Run Memory System
# ===========================================================================
echo ""
echo "[2/2] Running MEMORY SYSTEM (VLM: $VLM_PROVIDER)..."
echo "------------------------------------------------------------"

python experiments/neuro_memory/run_memory_experiment.py \
    --mode memory \
    --provider $VLM_PROVIDER \
    --n_episodes $N_EPISODES \
    --save_dir "$SAVE_DIR" \
    --name memory_$VLM_PROVIDER \
    --base_model "$BASE_MODEL_PATH" \
    --verbose 2>&1 | tee "$SAVE_DIR/memory.log"

echo ""
echo "Memory complete. Results: $SAVE_DIR/memory_${VLM_PROVIDER}_memory_*/"

# ===========================================================================
# Step 3: Compare Results
# ===========================================================================
echo ""
echo "============================================================"
echo "COMPARISON RESULTS"
echo "============================================================"

# Extract success rates from results.json files
BASELINE_SR=$(cat "$SAVE_DIR"/baseline_baseline_*/results.json 2>/dev/null | jq -r '.success_rate' || echo "N/A")
MEMORY_SR=$(cat "$SAVE_DIR"/memory_${VLM_PROVIDER}_memory_*/results.json 2>/dev/null | jq -r '.success_rate' || echo "N/A")

echo ""
echo "Baseline Success Rate: $BASELINE_SR"
echo "Memory Success Rate:   $MEMORY_SR"
echo ""

# Calculate improvement if both are numbers
if [[ "$BASELINE_SR" != "N/A" && "$MEMORY_SR" != "N/A" ]]; then
    IMPROVEMENT=$(python3 -c "print(f'{(float($MEMORY_SR) - float($BASELINE_SR)) * 100:.1f}%')" 2>/dev/null || echo "N/A")
    echo "Improvement: $IMPROVEMENT"
fi

echo ""
echo "Full results saved to: $SAVE_DIR"
echo ""

# Summary report
cat > "$SAVE_DIR/comparison_summary.txt" << EOF
Neuro-Symbolic Memory Comparison
================================
Date: $(date)
Episodes: $N_EPISODES
VLM Provider: $VLM_PROVIDER

Results
-------
Baseline Success Rate: $BASELINE_SR
Memory Success Rate:   $MEMORY_SR
Improvement:           ${IMPROVEMENT:-N/A}

Files
-----
Baseline Logs: $SAVE_DIR/baseline.log
Memory Logs:   $SAVE_DIR/memory.log
EOF

echo "Summary saved to: $SAVE_DIR/comparison_summary.txt"
echo "============================================================"
