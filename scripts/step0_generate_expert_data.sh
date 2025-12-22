#!/bin/bash
# Step 0: Generate Expert Data (Run this first!)
# We need ~2500 In-Domain and ~2000 OOD trajectories.

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
  --logging.online=False

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
  --logging.online=False