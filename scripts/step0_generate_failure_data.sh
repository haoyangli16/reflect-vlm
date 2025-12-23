#!/bin/bash
# Step 0: Generate Failure Data for Constraint Learning
# =====================================================
# This script runs a "clumsy" agent (BC without memory) to collect failures.
# The Oracle diagnoses WHY each failure occurred (physics/logic constraints).
#
# Output: data/raw_failure_constraints.pt
#
# Philosophy: "The Simulator is the Compiler. The Oracle is the Linter."

export MUJOCO_GL=egl 
export PYOPENGL_PLATFORM=egl

# Fix for LLVM command-line option conflict between triton and bitsandbytes
export TRITON_PTXAS_PATH=""
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Critical: Add NVIDIA library path so PyTorch can find CUDA
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Force bitsandbytes to use CUDA 11.7 libraries (matching PyTorch cu117)
export BNB_CUDA_VERSION=117

# Performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_VERBOSITY=error  # Reduce transformers logging
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings

# Use HuggingFace mirror for China (better than unstable proxy)
export HF_ENDPOINT=https://hf-mirror.com

# Use a shared HuggingFace cache by default to avoid re-downloading on different machines.
# Override by exporting HF_HOME/TRANSFORMERS_CACHE before running.
export HF_HOME="${HF_HOME:-/share/project/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Configuration
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

# Create data directory
mkdir -p "${DATA_DIR}"

# ==========================================
# Configuration Variables (override with env vars)
# ==========================================
N_TRAJS="${N_TRAJS:-1000}"  # Number of trajectories (more = more failures)
LEVEL="${LEVEL:-all}"       # Task difficulty: medium, hard, all
SEED="${SEED:-0}"           # Random seed
AGENT_TYPE="${AGENT_TYPE:-bc_romemo_wb}"  # Agent to generate failures
MODEL_PATH="${MODEL_PATH:-yunhaif/ReflectVLM-llava-v1.5-13b-post-trained}"
LOAD_4BIT="${LOAD_4BIT:-True}"
SAVE_DIR="${SAVE_DIR:-${DATA_DIR}/failure_collection}"
OUTPUT_PT="${OUTPUT_PT:-${DATA_DIR}/raw_failure_constraints.pt}"

echo "=========================================="
echo "Failure Data Generation"
echo "=========================================="
echo "Trajectories: ${N_TRAJS}"
echo "Level: ${LEVEL}"
echo "Agent: ${AGENT_TYPE}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_PT}"
echo "=========================================="

# ==========================================
# Run Failure Collection
# ==========================================
python -u "${PROJECT_ROOT}/run-rom-fail.py" \
    --agent_type="${AGENT_TYPE}" \
    --seed="${SEED}" \
    --n_trajs="${N_TRAJS}" \
    --level="${LEVEL}" \
    --max_steps=50 \
    --oracle_prob=0.0 \
    --model_path="${MODEL_PATH}" \
    --load_4bit="${LOAD_4BIT}" \
    --save_dir="${SAVE_DIR}" \
    --romemo_save_memory_path="${OUTPUT_PT}" \
    --trace_jsonl=True \
    --save_images=False \
    --record=False \
    --write_on_success=False \
    --logging.use_wandb=False

echo ""
echo "=========================================="
echo "Failure Data Generation Complete!"
echo "=========================================="
echo "Failure memory saved to: ${OUTPUT_PT}"
echo ""
echo "Traces saved to: ${SAVE_DIR}/traces/"
echo "  - step_traces.jsonl: All steps"
echo "  - failure_traces.jsonl: Only failures (with diagnosis)"
echo "  - episode_traces.jsonl: Episode summaries"
echo ""
echo "Next: Run scripts/pillar4_constraint_learning.sh to test constraint learning"
