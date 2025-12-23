#!/bin/bash
# Step 0: Generate Expert Data (Run this first!)
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

# 1. Generate In-Domain (Peg Insertion / Assembly)
# We use agent_type="expert_romemo_wb" to record successful trajectories to memory.
# Note: This takes a long time. Run on parallel GPUs if possible.
echo "Generating In-Domain Expert Data (2500 trajs)..."
python run-rom.py \
  --agent_type="expert_romemo_wb" \
  --n_trajs=2500 \
  --romemo_save_memory_path="data/raw_expert_indomain.pt" \
  --level="all" \
  --record=False \
  --logging.online=False \
  --model_path='yunhaif/ReflectVLM-llava-v1.5-13b-post-trained' \
  --load_4bit=True

# 2. Generate OOD Data (Distractors)
# HACK: To simulate OOD, we can either:
# a) Run a different task (if available)
# b) Run the SAME task but with a different --level or visual randomized textures/camera.
# For Reflect-VLM simple setup, let's assume we run "Hard" level as OOD for "Medium", 
# or just generate more data with different seeds that we will label as 'noise'.
# BETTER HACK: Run with a randomized agent that occasionally succeeds, or just 
# different seeds.
# For this script, let's just generate MORE data and treat it as the OOD pool 
# (simulating 'other episodes' that are not relevant to the current specific 100 test seeds).
echo "Generating OOD Expert Data (2000 trajs)..."
python run-rom.py \
  --agent_type="expert_romemo_wb" \
  --n_trajs=2000 \
  --seed=99999 \
  --romemo_save_memory_path="data/raw_expert_ood.pt" \
  --level="all" \
  --record=False \
  --logging.online=False \
  --model_path='yunhaif/ReflectVLM-llava-v1.5-13b-post-trained' \
  --load_4bit=True