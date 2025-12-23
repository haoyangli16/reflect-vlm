#!/bin/bash
# Step 0: Generate Expert Data (Parallelized)
# We need ~2500 In-Domain and ~2000 OOD trajectories.

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

# ==========================================
# OFFLINE MODE: Use local cached models (no network)
# ==========================================
# Set HF_HUB_OFFLINE=1 to force using local cache only
# This prevents the stuck "Downloading shards" issue
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"


# Set the number of parallel jobs.
# Adjust this based on your GPU count and VRAM.
# - If you have 8 GPUs, set N_JOBS=8.
# - If you have 1 A100 (80GB), set N_JOBS=4 (each 13B model takes ~10-12GB in 4bit).
N_JOBS=8

# Detect available GPUs (optional, comma separated)
GPUS="0,1,2,3,4,5,6,7"
# GPUS="0"
# Leave empty to auto-detect from CUDA_VISIBLE_DEVICES
# GPUS=""

echo "Generating In-Domain Expert Data (2500 trajs) with ${N_JOBS} jobs..."
python scripts/run_parallel_expert.py \
  --n_jobs=${N_JOBS} \
  --gpus="${GPUS}" \
  --total_trajs=2500 \
  --agent_type="expert_romemo_wb" \
  --output_pt="data/raw_expert_indomain.pt" \
  --level="all" \
  --seed_start=1000001 \
  --output_dir_base="datasets/parallel_indomain" \
  --model_path='yunhaif/ReflectVLM-llava-v1.5-13b-post-trained' \
  --load_4bit=True \
  --logging_online=False

echo "Generating OOD Expert Data (2000 trajs) with ${N_JOBS} jobs..."
# Note: large seed offset to ensure OOD
python scripts/run_parallel_expert.py \
  --n_jobs=${N_JOBS} \
  --gpus="${GPUS}" \
  --total_trajs=2000 \
  --agent_type="expert_romemo_wb" \
  --output_pt="data/raw_expert_ood.pt" \
  --level="all" \
  --seed_start=2000001 \
  --output_dir_base="datasets/parallel_ood" \
  --model_path='yunhaif/ReflectVLM-llava-v1.5-13b-post-trained' \
  --load_4bit=True \
  --logging_online=False

echo "Done. Data saved to data/raw_expert_indomain.pt and data/raw_expert_ood.pt"
