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

# Local model paths (default to your local directories)
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-base}"
POST_MODEL_PATH="${POST_MODEL_PATH:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-post-trained}"

# Optional overrides used by run_three_pillars_full.sh
DATA_ROOT="${DATA_ROOT:-}"
IN_DOMAIN_TRAJS="${IN_DOMAIN_TRAJS:-2500}"
OOD_TRAJS="${OOD_TRAJS:-2000}"
LEVEL="${LEVEL:-all}"
IN_DOMAIN_SEED="${IN_DOMAIN_SEED:-1000001}"
OOD_SEED="${OOD_SEED:-2000001}"

if [[ -n "$DATA_ROOT" ]]; then
  mkdir -p "$DATA_ROOT"
  OUT_IN_DOMAIN="${DATA_ROOT}/expert_in_domain_full.pt"
  OUT_OOD="${DATA_ROOT}/expert_ood_full.pt"
  OUTDIR_IN_DOMAIN="${DATA_ROOT}/parallel_indomain"
  OUTDIR_OOD="${DATA_ROOT}/parallel_ood"
else
  OUT_IN_DOMAIN="data/raw_expert_indomain.pt"
  OUT_OOD="data/raw_expert_ood.pt"
  OUTDIR_IN_DOMAIN="datasets/parallel_indomain"
  OUTDIR_OOD="datasets/parallel_ood"
fi

echo "Generating In-Domain Expert Data (${IN_DOMAIN_TRAJS} trajs) with ${N_JOBS} jobs..."
python scripts/run_parallel_expert.py \
  --n_jobs=${N_JOBS} \
  --gpus="${GPUS}" \
  --total_trajs="${IN_DOMAIN_TRAJS}" \
  --agent_type="expert_romemo_wb" \
  --output_pt="${OUT_IN_DOMAIN}" \
  --level="${LEVEL}" \
  --seed_start="${IN_DOMAIN_SEED}" \
  --output_dir_base="${OUTDIR_IN_DOMAIN}" \
  --model_path="${POST_MODEL_PATH}" \
  --load_4bit=True \
  --logging_online=False

echo "Generating OOD Expert Data (${OOD_TRAJS} trajs) with ${N_JOBS} jobs..."
# Note: large seed offset to ensure OOD
python scripts/run_parallel_expert.py \
  --n_jobs=${N_JOBS} \
  --gpus="${GPUS}" \
  --total_trajs="${OOD_TRAJS}" \
  --agent_type="expert_romemo_wb" \
  --output_pt="${OUT_OOD}" \
  --level="${LEVEL}" \
  --seed_start="${OOD_SEED}" \
  --output_dir_base="${OUTDIR_OOD}" \
  --model_path="${POST_MODEL_PATH}" \
  --load_4bit=True \
  --logging_online=False

echo "Done. Data saved to ${OUT_IN_DOMAIN} and ${OUT_OOD}"
