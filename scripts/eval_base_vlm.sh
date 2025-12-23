#!/bin/bash

# MuJoCo EGL rendering (for headless servers without display)
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

# Disable WandB (too slow)
export WANDB_MODE=disabled

# Optional: Uncomment if you want to use proxy instead of mirror
# export https_proxy=104.250.52.76:2080
# export http_proxy=104.250.52.76:2080

exp_name="eval_base_vlm"
# Local model paths (default to your local directories)
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-base}"
model_path="$BASE_MODEL_PATH"

python run-rom.py \
    --seed=1000000 \
    --reset_seed_start=0 \
    --n_trajs=100 \
    --save_dir="logs/$exp_name" \
    --start_traj_id=0 \
    --start_board_id=1000000 \
    --logging.online=False \
    --logging.group=$exp_name \
    --logging.prefix='eval' \
    --agent_type="llava" \
    --level='hard' \
    --oracle_prob=0 \
    --model_path=$model_path \
    --load_8bit=True \
    --record=True
