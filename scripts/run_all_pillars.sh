#!/bin/bash
# Run ALL pillars in sequence.
# Usage: bash scripts/run_all_pillars.sh

set -e

# 1. Generate Data (Comment out if already done)
# bash scripts/step0_generate_expert_data.sh

# 2. Process Subsets (Comment out if already done)
# python scripts/step1_make_memory_subsets.py \
#   --in_domain_path "data/raw_expert_indomain.pt" \
#   --ood_path "data/raw_expert_ood.pt" \
#   --out_dir "data"

# 3. Experiments
bash scripts/pillar1_universality.sh
bash scripts/pillar2_scaling_law.sh
bash scripts/pillar3_ood_robustness.sh

echo "All Pillars Completed!"
