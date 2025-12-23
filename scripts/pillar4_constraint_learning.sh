#!/bin/bash
# Pillar 4: Constraint Learning Experiment (Parallelized)
# =========================================================
# This experiment tests whether adding "Failure Memory" (Constraints) 
# prevents the VLM from repeating known physics/logic errors.
#
# Hypothesis:
#   Adding failure memories should REDUCE:
#   - Looping rate (repeating same failed action)
#   - Repeated failure rate (re-trying a failed action)
#   - Constraint violation rate (specific error types)
#
# Setup:
#   Fix total memory size N (e.g. 1000), and vary the ratio of:
#   - "good" memory (success memory, expert)
#   - "failure" memory (constraints)
#   Ratios are expressed as % good memory in the mixed bank.
#
# Mechanism:
#   Failure memories apply a NEGATIVE score penalty to actions
#   that previously failed in similar contexts.
#   Additionally, oracle_action is BOOSTED to suggest corrections.
#
# Env overrides:
#   GPUS="0,1,2,3,4,5,6,7"
#   TOTAL_MEMORY_SIZE=1000
#   GOOD_RATIOS="20,40,60,80,100"
#   METHODS="bc_romemo,bc_romemo_wb,reflect_romemo,reflect_romemo_wb"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TRITON_PTXAS_PATH=""
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${HF_HOME:-/share/project/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

set -euo pipefail

REFLECT_VLM_ROOT="${REFLECT_VLM_ROOT:-/share/project/lhy/thirdparty/reflect-vlm}"
cd "$REFLECT_VLM_ROOT"

# Configuration
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"
# Include retrieval mode in output dir so different modes are kept separate
RETRIEVAL_MODE_FOR_PATH="${RETRIEVAL_MODE:-visual}"
LOGS_DIR="${LOGS_DIR:-${PROJECT_ROOT}/logs/pillar4_constraints_ratio_${RETRIEVAL_MODE_FOR_PATH}}"

# Create directories
mkdir -p "${LOGS_DIR}"

# ==========================================
# Configuration Variables (override with env vars)
# ==========================================
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
NUM_TRAJS="${NUM_TRAJS:-100}"           # Number of test trajectories
LEVEL="${LEVEL:-all}"                    # Task difficulty
TEST_SEED="${TEST_SEED:-1000001}"        # Random seed for test environments
AGENT_SEEDS="${AGENT_SEEDS:-0}"          # Agent seeds (comma-separated)
MAX_STEPS="${MAX_STEPS:-50}"
LOAD_4BIT="${LOAD_4BIT:-True}"

# Local model paths (default to your local directories)
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-base}"
POST_MODEL_PATH="${POST_MODEL_PATH:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-post-trained}"

# Memory paths
SUCCESS_MEMORY="${SUCCESS_MEMORY:-${DATA_DIR}/raw_expert_indomain.pt}"
FAILURE_MEMORY="${FAILURE_MEMORY:-${DATA_DIR}/raw_failure_constraints.pt}"

# Fixed total memory size and good-memory ratios
TOTAL_MEMORY_SIZE="${TOTAL_MEMORY_SIZE:-1000}"
GOOD_RATIOS="${GOOD_RATIOS:-20,40,60,80,100}"  # % good (success) memory; remainder is failure memory

# Methods to run (all use RoMemo memory input)
METHODS="${METHODS:-bc_romemo,bc_romemo_wb,reflect_romemo,reflect_romemo_wb}"

# NEW: Retrieval mode for state-query based retrieval
# Options: "visual" (default, original behavior), "symbolic" (state-query), "hybrid"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-visual}"
SYMBOLIC_WEIGHT="${SYMBOLIC_WEIGHT:-0.5}"  # For hybrid mode

IFS=',' read -ra GPU_ARR <<< "$GPUS"
IFS=',' read -ra METHODS_ARR <<< "$METHODS"
IFS=',' read -ra RATIOS_ARR <<< "$GOOD_RATIOS"
IFS=',' read -ra SEEDS_ARR <<< "${AGENT_SEEDS// /}"
NGPU=${#GPU_ARR[@]}

echo "=========================================="
echo "Pillar 4: Constraint Learning (Parallelized)"
echo "=========================================="
echo "Question: Does failure memory reduce constraint violations?"
echo ""
echo "GPUs:             $GPUS ($NGPU total)"
echo "Test trajectories: ${NUM_TRAJS} (seed=${TEST_SEED}, level=${LEVEL})"
echo "Agent seeds:      ${AGENT_SEEDS}"
echo "Total memory size: ${TOTAL_MEMORY_SIZE}"
echo "Good ratios (%):   ${GOOD_RATIOS}"
echo "Methods:          ${METHODS}"
echo "Retrieval mode:   ${RETRIEVAL_MODE}"  # NEW
echo "Symbolic weight:  ${SYMBOLIC_WEIGHT}"  # NEW
echo "Success memory:   ${SUCCESS_MEMORY}"
echo "Failure memory:   ${FAILURE_MEMORY}"
echo "Output:           ${LOGS_DIR}"
echo "=========================================="

# ==========================================
# Step 1: Pre-build all mixed memory files
# ==========================================
echo ""
echo "=========================================="
echo "Step 1: Building mixed memory banks..."
echo "=========================================="

MIXED_MEM_DIR="${DATA_DIR}/mixed_memories"
mkdir -p "${MIXED_MEM_DIR}"

for ratio in "${RATIOS_ARR[@]}"; do
    ratio="${ratio// /}"
    [[ -z "$ratio" ]] && continue
    
    MIXED_MEMORY="${MIXED_MEM_DIR}/mixed_${TOTAL_MEMORY_SIZE}_good${ratio}.pt"
    
    if [[ -f "${MIXED_MEMORY}" ]]; then
        echo "  [SKIP] ${MIXED_MEMORY} (already exists)"
        continue
    fi
    
    echo "  Creating: total=${TOTAL_MEMORY_SIZE}, good_ratio=${ratio}%"
    
    python - <<PY
import os
import random
from pathlib import Path

success_path = Path("${SUCCESS_MEMORY}")
failure_path = Path("${FAILURE_MEMORY}")
output_path = Path("${MIXED_MEMORY}")
total_n = ${TOTAL_MEMORY_SIZE}
good_ratio = ${ratio}
seed = ${TEST_SEED}

from romemo.memory.schema import MemoryBank

random.seed(seed)

mem_good = MemoryBank.load_pt(str(success_path))
mem_fail = MemoryBank.load_pt(str(failure_path))

good_pool = list(mem_good.experiences)
fail_pool = list(mem_fail.experiences)

random.shuffle(good_pool)
random.shuffle(fail_pool)

n_good = int(round(total_n * (good_ratio / 100.0)))
n_fail = max(0, total_n - n_good)

if n_good > len(good_pool):
    print(f"[WARN] Not enough good experiences: need {n_good}, have {len(good_pool)}; taking all.")
    n_good = len(good_pool)
if n_fail > len(fail_pool):
    print(f"[WARN] Not enough failure experiences: need {n_fail}, have {len(fail_pool)}; taking all.")
    n_fail = len(fail_pool)

mixed = good_pool[:n_good] + fail_pool[:n_fail]
random.shuffle(mixed)

out = MemoryBank(name=f"mixed_{total_n}_good{good_ratio}")
for e in mixed:
    out.add(e)

output_path.parent.mkdir(parents=True, exist_ok=True)
out.save_pt(str(output_path))
print(f"  [OK] Saved: {output_path} (good={n_good}, fail={n_fail})")
PY
done

echo "✓ All mixed memory banks ready"

# ==========================================
# Step 2: Build job list and run in parallel
# ==========================================
echo ""
echo "=========================================="
echo "Step 2: Running experiments in parallel..."
echo "=========================================="

# Build job list: (method, ratio, seed)
jobs=()
for seed in "${SEEDS_ARR[@]}"; do
    [[ -z "$seed" ]] && continue
    for ratio in "${RATIOS_ARR[@]}"; do
        ratio="${ratio// /}"
        [[ -z "$ratio" ]] && continue
        for method in "${METHODS_ARR[@]}"; do
            method="${method// /}"
            [[ -z "$method" ]] && continue
            jobs+=("${method}|${ratio}|${seed}")
        done
    done
done

njobs=${#jobs[@]}
echo "Total jobs: $njobs"

run_one() {
    local gpu="$1"
    local method="$2"
    local ratio="$3"
    local seed="$4"
    
    local RUN_DIR="${LOGS_DIR}/${method}/good_${ratio}/seed_${seed}"
    mkdir -p "$RUN_DIR"
    
    local MIXED_MEMORY="${MIXED_MEM_DIR}/mixed_${TOTAL_MEMORY_SIZE}_good${ratio}.pt"
    
    if [[ ! -f "$MIXED_MEMORY" ]]; then
        echo "[ERROR] Memory not found: $MIXED_MEMORY"
        return 1
    fi
    
    echo "[GPU ${gpu}] Running: method=${method} good_ratio=${ratio}% seed=${seed}"
    
    # Select model:
    # - bc* uses BASE model
    # - reflect* uses POST-TRAINED model
    local model_path
    if [[ "$method" == reflect* ]]; then
        model_path="$POST_MODEL_PATH"
    else
        model_path="$BASE_MODEL_PATH"
    fi
    
    # Build quantization flag (use --load_4bit for true, --noload_4bit for false)
    local quant_flag=""
    if [[ "$LOAD_4BIT" == "True" || "$LOAD_4BIT" == "true" || "$LOAD_4BIT" == "1" ]]; then
        quant_flag="--load_4bit"
    else
        quant_flag="--noload_4bit --noload_8bit"
    fi
    
    CUDA_VISIBLE_DEVICES=$gpu python run-rom.py \
        --seed=$TEST_SEED \
        --reset_seed_start=0 \
        --n_trajs=$NUM_TRAJS \
        --save_dir="$RUN_DIR" \
        --start_traj_id=0 \
        --start_board_id=$TEST_SEED \
        --logging.online=False \
        --agent_type="$method" \
        --level="$LEVEL" \
        --oracle_prob=0 \
        --save_images=False \
        --record=False \
        --max_steps=$MAX_STEPS \
        --agent_seed=$seed \
        --model_path="$model_path" \
        $quant_flag \
        --romemo_init_memory_path="$MIXED_MEMORY" \
        --romemo_save_memory_path="$RUN_DIR/romemo_memory.pt" \
        --romemo_retrieval_mode="$RETRIEVAL_MODE" \
        --romemo_symbolic_weight=$SYMBOLIC_WEIGHT \
        --trace_jsonl=True \
        2>&1 | tee "$RUN_DIR/run.log"
}

# Parallel execution across GPUs
pids=()
labels=()
fail=0

for gi in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$gi]}"
    (
        idx="$gi"
        while [[ "$idx" -lt "$njobs" ]]; do
            IFS='|' read -r method ratio seed <<< "${jobs[$idx]}"
            run_one "$gpu" "$method" "$ratio" "$seed"
            idx=$((idx + NGPU))
        done
    ) &
    pids+=("$!")
    labels+=("gpu=${gpu}")
done

echo "Launched ${#pids[@]} GPU workers. Waiting..."

for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "[Parallel] FAILED worker: ${labels[$i]}" >&2
        fail=1
    fi
done

if [[ "$fail" -ne 0 ]]; then
    echo "WARNING: Some jobs failed. Aggregating what we have..." >&2
fi

# ==========================================
# Step 3: Aggregate Results
# ==========================================
echo ""
echo "=========================================="
echo "Step 3: Aggregating Results..."
echo "=========================================="

python -c "
import json
from pathlib import Path
import pandas as pd

logs_dir = Path('${LOGS_DIR}')

results = []
for method_dir in sorted(logs_dir.iterdir()):
    if not method_dir.is_dir():
        continue
    method = method_dir.name

    for ratio_dir in sorted(method_dir.iterdir()):
        if not ratio_dir.is_dir() or not ratio_dir.name.startswith('good_'):
            continue
        good_ratio = int(ratio_dir.name.replace('good_', ''))

        for seed_dir in sorted(ratio_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
                continue
            agent_seed = int(seed_dir.name.replace('seed_', ''))

            ep_trace = seed_dir / 'traces' / 'episode_traces.jsonl'
            if not ep_trace.exists():
                print(f'  WARN: No traces at {ep_trace}')
                continue

            episodes = [json.loads(ln) for ln in ep_trace.read_text().splitlines() if ln.strip()]
            if not episodes:
                continue

            n_eps = len(episodes)
            success_rate = sum(1 for e in episodes if e.get('success', False)) / n_eps
            mean_steps = sum(e.get('steps', 0) for e in episodes) / n_eps
            looping_rate = sum(e.get('looping_rate', 0) for e in episodes) / n_eps
            repeated_failure_rate = sum(e.get('repeated_failure_rate', 0) for e in episodes) / n_eps

            results.append({
                'method': method,
                'good_ratio': good_ratio,
                'agent_seed': agent_seed,
                'total_memory_size': int('${TOTAL_MEMORY_SIZE}'),
                'n_episodes': n_eps,
                'success_rate': round(success_rate, 4),
                'mean_steps': round(mean_steps, 2),
                'looping_rate': round(looping_rate, 4),
                'repeated_failure_rate': round(repeated_failure_rate, 4),
            })

df = pd.DataFrame(results)
out_dir = logs_dir / '_aggregate'
out_dir.mkdir(exist_ok=True)

df.to_csv(out_dir / 'constraint_results.csv', index=False)
print(f'Saved: {out_dir}/constraint_results.csv')

# Summary by method and good_ratio
if len(df) > 0:
    summary = df.groupby(['method', 'good_ratio']).agg({
        'success_rate': ['mean', 'std'],
        'mean_steps': ['mean', 'std'],
        'looping_rate': ['mean', 'std'],
        'repeated_failure_rate': ['mean', 'std'],
    }).round(4)
    print()
    print('Constraint Learning Summary:')
    print(summary)
"

# ==========================================
# Step 4: Generate Plot
# ==========================================
echo ""
echo "Generating Constraint Learning Plot..."

python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

root = Path('${LOGS_DIR}')
csv_path = root / '_aggregate' / 'constraint_results.csv'
if not csv_path.exists():
    print('No results to plot')
    exit(0)

df = pd.read_csv(csv_path)
if len(df) == 0:
    print('Empty results')
    exit(0)

# Aggregate by method and good_ratio
agg = df.groupby(['method', 'good_ratio']).agg({
    'success_rate': ['mean', 'std'],
    'looping_rate': ['mean', 'std'],
    'repeated_failure_rate': ['mean', 'std'],
}).reset_index()
agg.columns = ['method', 'good_ratio', 'sr_mean', 'sr_std', 'loop_mean', 'loop_std', 'rep_fail_mean', 'rep_fail_std']

methods = agg['method'].unique()
colors = plt.cm.tab10.colors

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Success Rate vs Good Ratio
ax1 = axes[0]
for i, method in enumerate(methods):
    sub = agg[agg['method'] == method]
    ax1.errorbar(sub['good_ratio'], sub['sr_mean'], yerr=sub['sr_std'],
                 marker='o', capsize=3, linewidth=2, markersize=6,
                 color=colors[i % len(colors)], label=method)
ax1.set_xlabel('Good Memory Ratio (%)', fontsize=11)
ax1.set_ylabel('Success Rate', fontsize=11)
ax1.set_title('Success Rate vs Good/Failure Mix', fontsize=12)
ax1.set_xlim(15, 105)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

# Plot 2: Looping Rate vs Good Ratio
ax2 = axes[1]
for i, method in enumerate(methods):
    sub = agg[agg['method'] == method]
    ax2.errorbar(sub['good_ratio'], sub['loop_mean'], yerr=sub['loop_std'],
                 marker='s', capsize=3, linewidth=2, markersize=6,
                 color=colors[i % len(colors)], label=method)
ax2.set_xlabel('Good Memory Ratio (%)', fontsize=11)
ax2.set_ylabel('Looping Rate', fontsize=11)
ax2.set_title('Looping Rate vs Good/Failure Mix', fontsize=12)
ax2.set_xlim(15, 105)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

# Plot 3: Repeated Failure Rate vs Good Ratio
ax3 = axes[2]
for i, method in enumerate(methods):
    sub = agg[agg['method'] == method]
    ax3.errorbar(sub['good_ratio'], sub['rep_fail_mean'], yerr=sub['rep_fail_std'],
                 marker='^', capsize=3, linewidth=2, markersize=6,
                 color=colors[i % len(colors)], label=method)
ax3.set_xlabel('Good Memory Ratio (%)', fontsize=11)
ax3.set_ylabel('Repeated Failure Rate', fontsize=11)
ax3.set_title('Repeated Failures vs Good/Failure Mix', fontsize=12)
ax3.set_xlim(15, 105)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8)

plt.tight_layout()

out_dir = root / '_aggregate' / 'plots'
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'constraint_learning.png', dpi=150, bbox_inches='tight')
print(f'Saved: {out_dir}/constraint_learning.png')
"

echo ""
echo "=========================================="
echo "✅ Pillar 4: Constraint Learning Complete!"
echo "=========================================="
echo "Results:  ${LOGS_DIR}/_aggregate/constraint_results.csv"
echo "Plots:    ${LOGS_DIR}/_aggregate/plots/constraint_learning.png"
echo ""
echo "Key hypothesis:"
echo "  - As good_ratio ↓ (more failure memory), looping_rate should ↓"
echo "  - Repeated_failure_rate should also ↓ with more failure memory"
echo "  - Success_rate may INCREASE or stay stable if failure memory helps avoid constraint violations"
