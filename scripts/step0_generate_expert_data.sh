#!/bin/bash
# Step 0: Generate Expert Data (Parallelized)
# We need ~2500 In-Domain and ~2000 OOD trajectories.

# Set the number of parallel jobs.
# Adjust this based on your GPU count and VRAM.
# - If you have 8 GPUs, set N_JOBS=8.
# - If you have 1 A100 (80GB), set N_JOBS=4 (each 13B model takes ~10-12GB in 4bit).
N_JOBS=4

# Detect available GPUs (optional, comma separated)
# GPUS="0,1,2,3,4,5,6,7"
# GPUS="0"
# Leave empty to auto-detect from CUDA_VISIBLE_DEVICES
GPUS=""

echo "Generating In-Domain Expert Data (2500 trajs) with ${N_JOBS} jobs..."
python scripts/run_parallel_expert.py \
  --n_jobs=${N_JOBS} \
  --gpus="${GPUS}" \
  --total_trajs=2500 \
  --agent_type="expert_romemo_wb" \
  --output_pt="data/raw_expert_indomain.pt" \
  --level="all" \
  --seed_start=0 \
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
  --seed_start=100000 \
  --output_dir_base="datasets/parallel_ood" \
  --model_path='yunhaif/ReflectVLM-llava-v1.5-13b-post-trained' \
  --load_4bit=True \
  --logging_online=False

echo "Done. Data saved to data/raw_expert_indomain.pt and data/raw_expert_ood.pt"
