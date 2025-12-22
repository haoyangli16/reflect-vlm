#!/bin/bash
set -e  # Exit on error

# Step 1: Create subsets from the raw expert data generated in Step 0
echo "Step 1: Creating memory subsets..."
python scripts/step1_make_memory_subsets.py \
  --in_domain_path="data/raw_expert_indomain.pt" \
  --ood_path="data/raw_expert_ood.pt" \
  --out_dir="data/three_pillars/memory_subsets"

# Pillar 1: Run the Universality experiment (Comparison across agents)
# We use a medium-sized memory bank (100 items) as the standard baseline.
echo "Pillar 1: Running Universality Experiment..."
INIT_MEMORY_PATH="data/three_pillars/memory_subsets/mem_100.pt" \
  bash scripts/pillar1_universality.sh

echo "Pipeline complete. Check logs and wandb for results."
