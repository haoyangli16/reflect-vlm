#!/bin/bash

# RoMemo plug-in gain experiment runner for reflect-vlm.
# Runs base vs base+RoMemo(+WB) on identical seeds/tasks.
#
# Usage:
#   bash scripts/eval_romemo_plugin.sh
#
# Notes:
# - This script defaults to the same 100-task protocol as the provided eval scripts.
# - You can override NUM_TRAJS, SAVE_ROOT, etc by exporting env vars.

set -euo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true

SEED=${SEED:-1000000}
RESET_SEED_START=${RESET_SEED_START:-0}
NUM_TRAJS=${NUM_TRAJS:-100}
LEVEL=${LEVEL:-hard}
SAVE_ROOT=${SAVE_ROOT:-logs/eval_romemo_plugin}
AGENT_SEEDS=${AGENT_SEEDS:-0,1,2}

# BC model == their base VLM checkpoint (behavior cloning style)
BC_MODEL_PATH=${BC_MODEL_PATH:-yunhaif/ReflectVLM-llava-v1.5-13b-base}

# Build an initial RoMemo memory (expert rollouts) once.
ROMEMO_MEM_PATH=${ROMEMO_MEM_PATH:-${SAVE_ROOT}/romemo_init_memory.pt}
MEM_COLLECT_TRAJS=${MEM_COLLECT_TRAJS:-30}
if [ ! -f "${ROMEMO_MEM_PATH}" ]; then
  echo "=== Building initial RoMemo memory via expert (${MEM_COLLECT_TRAJS} trajs) ==="
  python -u run.py \
    --seed="${SEED}" \
    --reset_seed_start="${RESET_SEED_START}" \
    --n_trajs="${MEM_COLLECT_TRAJS}" \
    --save_dir="${SAVE_ROOT}/_collect_memory_expert" \
    --start_traj_id=0 \
    --start_board_id="${SEED}" \
    --logging.online=False \
    --logging.group="eval_romemo_plugin" \
    --logging.prefix="collect" \
    --agent_type="expert_romemo_wb" \
    --agent_seed=0 \
    --level="${LEVEL}" \
    --oracle_prob=0 \
    --record=False \
    --trace_jsonl=False \
    --romemo_save_memory_path="${ROMEMO_MEM_PATH}"
fi

declare -a METHODS=(
  "mcts"
  "mcts_romemo"
  "mcts_romemo_wb"
  "bc"
  "bc_romemo"
  "bc_romemo_wb"
)

IFS=',' read -ra SEEDS_ARR <<< "${AGENT_SEEDS}"
for agent_seed in "${SEEDS_ARR[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "=== Running method=${method} agent_seed=${agent_seed} ==="
    python -u run.py \
      --seed="${SEED}" \
      --reset_seed_start="${RESET_SEED_START}" \
      --n_trajs="${NUM_TRAJS}" \
      --save_dir="${SAVE_ROOT}/${method}/seed_${agent_seed}" \
      --start_traj_id=0 \
      --start_board_id="${SEED}" \
      --logging.online=False \
      --logging.group="eval_romemo_plugin" \
      --logging.prefix="eval" \
      --agent_type="${method}" \
      --agent_seed="${agent_seed}" \
      --level="${LEVEL}" \
      --oracle_prob=0 \
      --model_path="${BC_MODEL_PATH}" \
      --load_4bit=True \
      --record=False \
      --trace_jsonl=True \
      --romemo_init_memory_path="${ROMEMO_MEM_PATH}"
  done
done

echo "Done. Logs at: ${SAVE_ROOT}"


