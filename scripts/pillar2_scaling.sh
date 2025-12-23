#!/bin/bash
# Pillar 2: The Scaling Law - Is memory data the new compute?
#
# Setup: Fix Base Policy (BC or Reflect). Vary Memory Size.
# Methods (6 total):
#   - bc, bc_romemo, bc_romemo_wb
#   - reflect, reflect_romemo, reflect_romemo_wb
# Memory Sizes: [0, 10, 50, 100, 500, 1000, 2000]
#
# Hypothesis: Logarithmic growth. At N=2000, SR should plateau or slowly rise.
# Critical: Memory subsets are nested (mem_10 ⊂ mem_50 ⊂ ... ⊂ mem_2000).
#
# Env overrides:
#   GPUS="0,1,2,3,4,5,6,7"
#   MEMORY_SUBSET_DIR="data/three_pillars/memory_subsets"
#   SAVE_ROOT="logs/pillar2_scaling"
#   NUM_TRAJS=100
#   SCALING_SIZES="0,10,50,100,500,1000,2000"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TRITON_PTXAS_PATH=""
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=117
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
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
MEMORY_SUBSET_DIR=${MEMORY_SUBSET_DIR:-"data/three_pillars/memory_subsets"}
SAVE_ROOT=${SAVE_ROOT:-"logs/pillar2_scaling"}
NUM_TRAJS=${NUM_TRAJS:-100}
LEVEL=${LEVEL:-"all"}
TEST_SEED=${TEST_SEED:-1000001}
MAX_STEPS=${MAX_STEPS:-50}
AGENT_SEEDS=${AGENT_SEEDS:-"0"}

# Scaling sizes (0 = no memory baseline)
SCALING_SIZES=${SCALING_SIZES:-"0,10,50,100,500,1000,2000"}

# Local model paths (default to your local directories)
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-base}"
POST_MODEL_PATH="${POST_MODEL_PATH:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-post-trained}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
IFS=',' read -ra SIZES_ARR <<< "$SCALING_SIZES"
NGPU=${#GPU_ARR[@]}

echo "=========================================="
echo "Pillar 2: The Scaling Law"
echo "=========================================="
echo "Question: Is memory data the new compute?"
echo ""
echo "Methods:  bc/bc_romemo/bc_romemo_wb + reflect/reflect_romemo/reflect_romemo_wb"
echo "Sizes:    ${SIZES_ARR[*]}"
echo "Test:     $NUM_TRAJS tasks (seed=$TEST_SEED, level=$LEVEL)"
echo "GPUs:     $GPUS ($NGPU total)"
echo "Output:   $SAVE_ROOT"
echo "=========================================="

mkdir -p "$SAVE_ROOT"

# Build job list: (method, size, seed)
# Convention:
# - size=0 only runs baselines (bc, reflect)
# - size>0 runs memory methods (bc_romemo, bc_romemo_wb, reflect_romemo, reflect_romemo_wb)
jobs=()
IFS=',' read -ra SEEDS_ARR <<< "${AGENT_SEEDS// /}"
for seed in "${SEEDS_ARR[@]}"; do
    [[ -z "$seed" ]] && continue
    for size in "${SIZES_ARR[@]}"; do
        if [[ "$size" == "0" ]]; then
            jobs+=("bc|${size}|${seed}")
            jobs+=("reflect|${size}|${seed}")
        else
            jobs+=("bc_romemo|${size}|${seed}")
            jobs+=("bc_romemo_wb|${size}|${seed}")
            jobs+=("reflect_romemo|${size}|${seed}")
            jobs+=("reflect_romemo_wb|${size}|${seed}")
        fi
    done
done

run_one() {
    local gpu="$1"
    local method="$2"
    local size="$3"
    local seed="$4"

    RUN_DIR="$SAVE_ROOT/${method}/size_${size}/seed_${seed}"
    mkdir -p "$RUN_DIR"
    
    echo "[GPU ${gpu}] Running: method=${method} size=${size} (seed=${seed})..."

    mem_path=""
    if [[ "$size" != "0" ]]; then
        mem_path="$MEMORY_SUBSET_DIR/mem_${size}.pt"
        if [[ ! -f "$mem_path" ]]; then
            echo "ERROR: Memory file not found: $mem_path"
            return 1
        fi
    fi

    # Select model:
    # - bc* uses BASE model
    # - reflect* uses POST-TRAINED model
    if [[ "$method" == reflect* ]]; then
        model_path="$POST_MODEL_PATH"
    else
        model_path="$BASE_MODEL_PATH"
    fi
    
    CMD="CUDA_VISIBLE_DEVICES=$gpu python run-rom.py \
        --seed=$TEST_SEED \
        --reset_seed_start=0 \
        --n_trajs=$NUM_TRAJS \
        --save_dir=\"$RUN_DIR\" \
        --start_traj_id=0 \
        --start_board_id=$TEST_SEED \
        --logging.online=False \
        --agent_type=\"$method\" \
        --level=\"$LEVEL\" \
        --oracle_prob=0 \
        --save_images=False \
        --record=False \
        --max_steps=$MAX_STEPS \
        --agent_seed=$seed \
        --model_path=\"$model_path\" \
        --load_4bit"
    
    # Add RoMemo args if using memory
    if [[ "$size" != "0" ]]; then
        CMD="$CMD \
            --romemo_init_memory_path=\"$mem_path\" \
            --romemo_save_memory_path=\"$RUN_DIR/romemo_memory.pt\""
    fi
    
    eval $CMD
}

# Parallel execution
pids=()
labels=()
fail=0
njobs=${#jobs[@]}

for gi in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$gi]}"
    (
        idx="$gi"
        while [[ "$idx" -lt "$njobs" ]]; do
            IFS='|' read -r method size seed <<< "${jobs[$idx]}"
            run_one "$gpu" "$method" "$size" "$seed"
            idx=$((idx + NGPU))
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
    echo "ERROR: Some jobs failed." >&2
    exit 1
fi

# Aggregate results (custom for scaling)
echo ""
echo "=========================================="
echo "Aggregating Scaling Results..."
echo "=========================================="

python -c "
import json
import pandas as pd
from pathlib import Path

root = Path('$SAVE_ROOT')
rows = []

for method_dir in sorted(root.iterdir()):
    if not method_dir.is_dir():
        continue
    method = method_dir.name

    for size_dir in sorted(method_dir.iterdir()):
        if not size_dir.is_dir() or not size_dir.name.startswith('size_'):
            continue
        size = int(size_dir.name.replace('size_', ''))

        for seed_dir in sorted(size_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
                continue
            agent_seed = int(seed_dir.name.replace('seed_', ''))

            ep_path = seed_dir / 'traces' / 'episode_traces.jsonl'
            if not ep_path.exists():
                print(f'  WARN: No traces at {ep_path}')
                continue

            eps = [json.loads(ln) for ln in ep_path.read_text().splitlines() if ln.strip()]
            if len(eps) == 0:
                continue

            sr = sum(1 for e in eps if e.get('success')) / len(eps)
            mean_steps = sum(e.get('steps', 0) for e in eps) / len(eps)
            looping = sum(e.get('looping_rate', 0) for e in eps) / len(eps)
            repeat_fail = sum(e.get('repeated_failure_rate', 0) for e in eps) / len(eps)

            rows.append({
                'method': method,
                'memory_size': size,
                'agent_seed': agent_seed,
                'num_episodes': len(eps),
                'success_rate': sr,
                'mean_steps': mean_steps,
                'looping_rate': looping,
                'repeated_failure_rate': repeat_fail,
            })

df = pd.DataFrame(rows)
out_dir = root / '_aggregate'
out_dir.mkdir(exist_ok=True)

df.to_csv(out_dir / 'scaling_results.csv', index=False)
print(f'Saved: {out_dir}/scaling_results.csv')

# Summary by (method, size)
if len(df) > 0:
    summary = df.groupby(['method','memory_size']).agg({
        'success_rate': ['mean', 'std'],
        'mean_steps': ['mean', 'std'],
        'looping_rate': ['mean', 'std'],
    }).round(4)
    print()
    print('Scaling Summary:')
    print(summary)
"

# Generate scaling plot
echo ""
echo "Generating Scaling Plot..."

python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

root = Path('$SAVE_ROOT')
csv_path = root / '_aggregate' / 'scaling_results.csv'
if not csv_path.exists():
    print('No results to plot')
    exit(0)

df = pd.read_csv(csv_path)
if len(df) == 0:
    print('Empty results')
    exit(0)

# Aggregate by (method, memory_size)
agg = df.groupby(['method','memory_size']).agg({
    'success_rate': ['mean', 'std'],
    'mean_steps': ['mean', 'std'],
}).reset_index()
agg.columns = ['method','size', 'sr_mean', 'sr_std', 'steps_mean', 'steps_std']

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Success Rate vs Memory Size (one curve per method)
for method in sorted(agg['method'].unique()):
    sub = agg[agg['method']==method].sort_values('size')
    ax1.errorbar(sub['size'], sub['sr_mean'], yerr=sub['sr_std'],
                 marker='o', capsize=3, linewidth=2, markersize=6, label=method)
ax1.set_xlabel('Memory Size (# experiences)', fontsize=12)
ax1.set_ylabel('Success Rate', fontsize=12)
ax1.set_title('Pillar 2: Scaling Law', fontsize=14)
ax1.set_xscale('symlog', linthresh=10)  # log scale but handles 0
ax1.grid(True, alpha=0.3)
ax1.legend()

# Steps vs Memory Size
for method in sorted(agg['method'].unique()):
    sub = agg[agg['method']==method].sort_values('size')
    ax2.errorbar(sub['size'], sub['steps_mean'], yerr=sub['steps_std'],
                 marker='s', capsize=3, linewidth=2, markersize=6, label=method)
ax2.set_xlabel('Memory Size (# experiences)', fontsize=12)
ax2.set_ylabel('Mean Steps to Completion', fontsize=12)
ax2.set_title('Efficiency vs Memory Size', fontsize=14)
ax2.set_xscale('symlog', linthresh=10)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()

out_dir = root / '_aggregate' / 'plots'
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'scaling_law.png', dpi=150, bbox_inches='tight')
print(f'Saved: {out_dir}/scaling_law.png')
"

echo ""
echo "=========================================="
echo "✅ Pillar 2: Scaling Law Complete!"
echo "=========================================="
echo "Results:  $SAVE_ROOT/_aggregate/scaling_results.csv"
echo "Plots:    $SAVE_ROOT/_aggregate/plots/scaling_law.png"
echo ""
echo "Expected claim: 'SR grows logarithmically with memory size, plateauing at N=2000.'"
