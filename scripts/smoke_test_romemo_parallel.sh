#!/bin/bash
# Run the RoMemo smoke test in parallel across multiple GPUs.
#
# This script:
#  1) builds the initial RoMemo memory once (expert runs)
#  2) fans out (method, agent_seed) jobs across GPUs
#  3) aggregates + plots once at the end
#
# Env overrides:
#   GPUS="0,1,2,3,4,5,6,7"   (default)
#   INIT_GPU=0                (GPU for init memory build)
#   SEED, NUM_TRAJS, LEVEL, AGENT_SEEDS, MEM_COLLECT_TRAJS, SAVE_ROOT, MAX_STEPS
#   METHODS="bc,bc_romemo,..." (optional; default is 6 methods)
#   FORCE_REBUILD_INIT=0/1

set -euo pipefail

REFLECT_VLM_ROOT="${REFLECT_VLM_ROOT:-/share/project/lhy/thirdparty/reflect-vlm}"
cd "$REFLECT_VLM_ROOT"

GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
INIT_GPU=${INIT_GPU:-0}
FORCE_REBUILD_INIT=${FORCE_REBUILD_INIT:-0}

SAVE_ROOT=${SAVE_ROOT:-"logs/smoke_test_romemo"}
INIT_MEMORY_PATH=${INIT_MEMORY_PATH:-"$SAVE_ROOT/romemo_init_memory/romemo_memory.pt"}

# Build init memory once.
if [[ "$FORCE_REBUILD_INIT" == "1" || ! -f "$INIT_MEMORY_PATH" ]]; then
  echo "[Parallel] Building init RoMemo memory on GPU ${INIT_GPU}..."
  CUDA_VISIBLE_DEVICES="$INIT_GPU" \
    DO_INIT_MEMORY=1 DO_RUN=0 DO_AGG=0 DO_PLOT=0 \
    INIT_MEMORY_PATH="$INIT_MEMORY_PATH" \
    bash scripts/smoke_test_romemo.sh
else
  echo "[Parallel] Using existing init memory: $INIT_MEMORY_PATH"
fi

# Methods to run.
if [[ -n "${METHODS:-}" ]]; then
  IFS=',' read -ra METHODS_ARR <<< "${METHODS}"
else
  METHODS_ARR=(
    "bc" "bc_romemo" "bc_romemo_wb"
    "reflect" "reflect_romemo" "reflect_romemo_wb"
    "mcts" "mcts_romemo" "mcts_romemo_wb"
  )
fi

# Agent seeds.
AGENT_SEEDS=${AGENT_SEEDS:-"0"}
IFS=',' read -ra SEEDS_ARR <<< "${AGENT_SEEDS// /}"  # strip spaces

# GPU list.
IFS=',' read -ra GPU_ARR <<< "$GPUS"
if [[ ${#GPU_ARR[@]} -eq 0 ]]; then
  echo "GPUS is empty" >&2
  exit 2
fi

jobs=()
for seed in "${SEEDS_ARR[@]}"; do
  [[ -z "$seed" ]] && continue
  for method in "${METHODS_ARR[@]}"; do
    method="${method// /}"
    [[ -z "$method" ]] && continue
    jobs+=("${method}|${seed}")
  done
done

run_one() {
  local gpu="$1"
  local method="$2"
  local seed="$3"
  echo "[Parallel] Running method=${method} seed=${seed} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    DO_INIT_MEMORY=0 DO_AGG=0 DO_PLOT=0 \
    ONLY_METHOD="$method" AGENT_SEEDS="$seed" \
    INIT_MEMORY_PATH="$INIT_MEMORY_PATH" \
    bash scripts/smoke_test_romemo.sh
}

# One worker per GPU, each processes a disjoint slice of the job list.
pids=()
labels=()
fail=0
ngpu=${#GPU_ARR[@]}
njobs=${#jobs[@]}

for gi in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$gi]}"
  (
    idx="$gi"
    while [[ "$idx" -lt "$njobs" ]]; do
      IFS='|' read -r method seed <<< "${jobs[$idx]}"
      run_one "$gpu" "$method" "$seed"
      idx=$((idx + ngpu))
    done
  ) &
  pids+=("$!")
  labels+=("gpu=${gpu}")
done

for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[Parallel] FAILED worker: ${labels[$i]}" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[Parallel] One or more jobs failed; skipping aggregate/plot." >&2
  exit 1
fi

echo "[Parallel] All jobs done. Aggregating + plotting..."
DO_INIT_MEMORY=0 DO_RUN=0 DO_AGG=1 DO_PLOT=1 \
  INIT_MEMORY_PATH="$INIT_MEMORY_PATH" \
  bash scripts/smoke_test_romemo.sh

echo "[Parallel] Done. Results at: $SAVE_ROOT/_aggregate/results.csv"
