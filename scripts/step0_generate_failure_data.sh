#!/bin/bash
# Step 0: Generate Failure Data for Constraint Learning (Parallelized)
# =====================================================================
# This script runs a "clumsy" agent (BC without memory) to collect failures.
# The Oracle diagnoses WHY each failure occurred (physics/logic constraints).
#
# Output: data/raw_failure_constraints.pt
#
# Philosophy: "The Simulator is the Compiler. The Oracle is the Linter."

# ==========================================
# Environment Setup (CRITICAL for headless GPU servers)
# ==========================================
export MUJOCO_GL=egl 
export PYOPENGL_PLATFORM=egl

# Fix for LLVM command-line option conflict between triton and bitsandbytes
export TRITON_PTXAS_PATH=""

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

# Use a shared HuggingFace cache
export HF_HOME="${HF_HOME:-/share/project/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" 2>/dev/null || true

# ==========================================
# Configuration
# ==========================================
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

# Create data directory
mkdir -p "${DATA_DIR}"

# ==========================================
# Parallel Configuration
# ==========================================
# Set the number of parallel jobs.
# - If you have 8 GPUs, set N_JOBS=8.
# - If you have 1 A100 (80GB), set N_JOBS=4 (each 13B model takes ~10-12GB in 4bit).
N_JOBS="${N_JOBS:-8}"

# GPU IDs to use (comma-separated)
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# ==========================================
# Experiment Configuration (override with env vars)
# ==========================================
N_TRAJS="${N_TRAJS:-1000}"  # Number of trajectories (more = more failures)
LEVEL="${LEVEL:-all}"       # Task difficulty: medium, hard, all
SEED="${SEED:-0}"           # Random seed
AGENT_TYPE="${AGENT_TYPE:-bc_romemo_wb}"  # Agent to generate failures
MODEL_PATH="${MODEL_PATH:-yunhaif/ReflectVLM-llava-v1.5-13b-base}"
LOAD_4BIT="${LOAD_4BIT:-True}"
OUTPUT_PT="${OUTPUT_PT:-${DATA_DIR}/raw_failure_constraints.pt}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-${DATA_DIR}/failure_collection}"

# Failure collection settings
STOP_ON_FAILURE="${STOP_ON_FAILURE:-True}"  # Stop after first failure
WRITE_ON_SUCCESS="${WRITE_ON_SUCCESS:-False}"  # Only record failures

echo "=========================================="
echo "Failure Data Generation (Parallelized)"
echo "=========================================="
echo "Total Trajectories: ${N_TRAJS}"
echo "Parallel Jobs: ${N_JOBS}"
echo "GPUs: ${GPUS}"
echo "Level: ${LEVEL}"
echo "Agent: ${AGENT_TYPE}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_PT}"
echo "Stop on failure: ${STOP_ON_FAILURE}"
echo "Write on success: ${WRITE_ON_SUCCESS}"
echo "=========================================="

# ==========================================
# Run Parallel Failure Collection
# ==========================================
cd "${PROJECT_ROOT}"

python scripts/run_parallel_failure.py \
    --n_jobs="${N_JOBS}" \
    --gpus="${GPUS}" \
    --total_trajs="${N_TRAJS}" \
    --agent_type="${AGENT_TYPE}" \
    --model_path="${MODEL_PATH}" \
    --output_pt="${OUTPUT_PT}" \
    --level="${LEVEL}" \
    --seed_start="${SEED}" \
    --output_dir_base="${OUTPUT_DIR_BASE}" \
    --load_4bit="${LOAD_4BIT}" \
    --logging_online=False \
    --stop_on_failure="${STOP_ON_FAILURE}" \
    --write_on_success="${WRITE_ON_SUCCESS}"

echo ""
echo "=========================================="
echo "Failure Data Generation Complete!"
echo "=========================================="
echo "Failure memory saved to: ${OUTPUT_PT}"
echo ""
echo "Job logs saved to: ${OUTPUT_DIR_BASE}/job_*/run.log"
echo "Traces saved to: ${OUTPUT_DIR_BASE}/job_*/traces/"
echo ""
echo "Next: Run scripts/pillar4_constraint_learning.sh to test constraint learning"
