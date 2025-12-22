#!/bin/bash
# Smoke test: runs a small RoMemo plug-in experiment.
#
# Supports parallelization by letting callers run individual methods/seeds on
# separate GPUs, and then aggregating once.
#
# Useful env overrides:
#   SEED, NUM_TRAJS, LEVEL, AGENT_SEEDS, MEM_COLLECT_TRAJS, SAVE_ROOT, MAX_STEPS
#   METHODS="bc,bc_romemo" or ONLY_METHOD="mcts"
#   DO_INIT_MEMORY=0/1, DO_RUN=0/1, DO_AGG=0/1, DO_PLOT=0/1
#   INIT_MEMORY_PATH=/path/to/romemo_memory.pt
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

# Disable WandB (too slow)
# export WANDB_MODE=disabled

set -euo pipefail

echo "=========================================="
echo "RoMemo Plugin Smoke Test"
echo "Running 3 tasks × 9 methods × 1 seed"
echo "=========================================="

# Configuration (override via env vars)
SEED=${SEED:-1000000}
NUM_TRAJS=${NUM_TRAJS:-3}             # smoke default
LEVEL=${LEVEL:-medium}
AGENT_SEEDS=${AGENT_SEEDS:-"0"}       # comma-separated or space-separated
MEM_COLLECT_TRAJS=${MEM_COLLECT_TRAJS:-3}
SAVE_ROOT=${SAVE_ROOT:-"logs/smoke_test_romemo"}
MAX_STEPS=${MAX_STEPS:-50}

# MCTS speed controls (override via env vars)
MCTS_SIMS=${MCTS_SIMS:-50}
MCTS_PROPOSAL_OBS=${MCTS_PROPOSAL_OBS:-root}

DO_INIT_MEMORY=${DO_INIT_MEMORY:-1}
DO_RUN=${DO_RUN:-1}
DO_AGG=${DO_AGG:-1}
DO_PLOT=${DO_PLOT:-1}

# GPU settings
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Required paths
REFLECT_VLM_ROOT="/share/project/lhy/thirdparty/reflect-vlm"
ROMEMO_ROOT="/share/project/lhy/thirdparty/RoMemo"

cd "$REFLECT_VLM_ROOT"

INIT_MEMORY_PATH=${INIT_MEMORY_PATH:-"$SAVE_ROOT/romemo_init_memory/romemo_memory.pt"}

if [[ "$DO_INIT_MEMORY" != "0" ]]; then
  echo ""
  echo "Step 0: Building initial RoMemo memory (${MEM_COLLECT_TRAJS} expert trajs)..."
  mkdir -p "$SAVE_ROOT/romemo_init_memory"
  python run-rom.py \
    --seed=$SEED \
    --reset_seed_start=0 \
    --n_trajs=$MEM_COLLECT_TRAJS \
    --save_dir="$SAVE_ROOT/romemo_init_memory" \
    --start_traj_id=0 \
    --start_board_id=1000000 \
    --logging.online=False \
    --agent_type="expert_romemo_wb" \
    --level="$LEVEL" \
    --oracle_prob=1.0 \
    --imagine_future_steps=5 \
    --record=False \
    --max_steps=$MAX_STEPS \
    --romemo_save_memory_path="$INIT_MEMORY_PATH" \
    --model_path='yunhaif/ReflectVLM-llava-v1.5-13b-post-trained' \
    --load_4bit=True
fi

if [[ "$DO_RUN" != "0" ]]; then
  echo ""
  echo "Step 1: Running methods..."

# Define methods array (override via ONLY_METHOD or METHODS)
methods=()
if [[ -n "${ONLY_METHOD:-}" ]]; then
  methods+=("$ONLY_METHOD")
elif [[ -n "${METHODS:-}" ]]; then
  IFS=',' read -ra _m <<< "${METHODS}"
  for x in "${_m[@]}"; do
    x="${x// /}"
    [[ -n "$x" ]] && methods+=("$x")
  done
else
  methods+=("bc" "bc_romemo" "bc_romemo_wb" "reflect" "reflect_romemo" "reflect_romemo_wb" "mcts" "mcts_romemo" "mcts_romemo_wb")
fi

for agent_seed in ${AGENT_SEEDS//,/ }; do
  for method in "${methods[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Running: $method (agent_seed=$agent_seed)"
    echo "----------------------------------------"
    
    RUN_DIR="$SAVE_ROOT/${method}/seed_${agent_seed}"
    
    # Build command
    CMD="python run-rom.py \
      --seed=$SEED \
      --reset_seed_start=0 \
      --n_trajs=$NUM_TRAJS \
      --save_dir=\"$RUN_DIR\" \
      --start_traj_id=0 \
      --start_board_id=1000000 \
      --logging.online=False \
      --agent_type=\"$method\" \
      --level=\"$LEVEL\" \
      --oracle_prob=0 \
      --record=False \
      --max_steps=$MAX_STEPS \
      --agent_seed=$agent_seed"
    
    # Add RoMemo args if needed
    if [[ $method == *"romemo"* ]]; then
      CMD="$CMD \
        --romemo_init_memory_path=\"$INIT_MEMORY_PATH\" \
        --romemo_save_memory_path=\"$RUN_DIR/romemo_memory.pt\""
    fi
    
    # Add model args for BC/VLM (and MCTS, which now uses VLM proposals)
    if [[ $method == bc* || $method == mcts* || $method == reflect* ]]; then
      # Use post-trained model (better than base model)
      CMD="$CMD \
        --model_path='yunhaif/ReflectVLM-llava-v1.5-13b-post-trained' \
        --load_4bit=True"
    fi

    # MCTS tuning (speed vs quality)
    if [[ $method == mcts* ]]; then
      CMD="$CMD \
        --mcts_sims=$MCTS_SIMS \
        --mcts_proposal_observation=$MCTS_PROPOSAL_OBS"
    fi

    # Reflect policies use sim-based reflection (run-rom.py reflect* agents)
    if [[ $method == reflect* ]]; then
      CMD="$CMD \
        --imagine_future_steps=5"
    fi
    
    # Execute
    eval $CMD
  done
done
fi

if [[ "$DO_AGG" != "0" ]]; then
  echo ""
  echo "Step 2: Aggregating results..."
  python scripts/aggregate_romemo_plugin_results.py \
    --root "$SAVE_ROOT" \
    --out "$SAVE_ROOT/_aggregate"
fi

if [[ "$DO_PLOT" != "0" ]]; then
  echo ""
  echo "Step 3: Generating plots..."
  python scripts/plot_romemo_plugin_results.py \
    --csv "$SAVE_ROOT/_aggregate/results.csv" \
    --out "$SAVE_ROOT/_aggregate/plots"
fi

echo ""
echo "=========================================="
echo "✅ Smoke test complete!"
echo "=========================================="
echo "Results: $SAVE_ROOT/_aggregate/results.csv"
echo "Plots:   $SAVE_ROOT/_aggregate/plots/"
echo ""
echo "Check for issues, then run full experiment with:"
echo "  NUM_TRAJS=100 AGENT_SEEDS=0,1,2 MEM_COLLECT_TRAJS=30 bash scripts/eval_romemo_plugin.sh"
