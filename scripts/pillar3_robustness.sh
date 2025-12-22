#!/bin/bash
# Pillar 3: Robustness / OOD Mixture - Can the retriever filter out noise?
#
# Setup: Fix Memory Size (N=100 total). Fix Base Policy (bc_romemo).
# Variable: In-Domain Ratio.
#   - 100% In-Domain (100 relevant memories)
#   - 80% In-Domain (80 relevant, 20 junk/OOD)
#   - 60%, 40%, 20% In-Domain
#
# Hypothesis: Visual Retrieval should handle this well (SR drops slowly).
#             This proves the retrieval mechanism is semantically robust.
#
# Env overrides:
#   GPUS="0,1,2,3,4,5,6,7"
#   MEMORY_SUBSET_DIR="data/three_pillars/memory_subsets"
#   SAVE_ROOT="logs/pillar3_robustness"
#   NUM_TRAJS=100
#   OOD_RATIOS="100,80,60,40,20"

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
SAVE_ROOT=${SAVE_ROOT:-"logs/pillar3_robustness"}
NUM_TRAJS=${NUM_TRAJS:-100}
LEVEL=${LEVEL:-"all"}
TEST_SEED=${TEST_SEED:-2000000}
MAX_STEPS=${MAX_STEPS:-50}
AGENT_SEEDS=${AGENT_SEEDS:-"0"}

# OOD ratios (percentage of in-domain data)
OOD_RATIOS=${OOD_RATIOS:-"100,80,60,40,20"}

# Fixed method for OOD experiment
METHOD="bc_romemo"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
IFS=',' read -ra RATIOS_ARR <<< "$OOD_RATIOS"
NGPU=${#GPU_ARR[@]}

echo "=========================================="
echo "Pillar 3: Robustness / OOD Mixture"
echo "=========================================="
echo "Question: Can the retriever filter out noise?"
echo ""
echo "Method:     $METHOD"
echo "OOD Ratios: ${RATIOS_ARR[*]} (% in-domain)"
echo "Test:       $NUM_TRAJS tasks (seed=$TEST_SEED, level=$LEVEL)"
echo "GPUs:       $GPUS ($NGPU total)"
echo "Output:     $SAVE_ROOT"
echo "=========================================="

mkdir -p "$SAVE_ROOT"

# Build job list: (ratio, seed)
jobs=()
IFS=',' read -ra SEEDS_ARR <<< "${AGENT_SEEDS// /}"
for seed in "${SEEDS_ARR[@]}"; do
    [[ -z "$seed" ]] && continue
    for ratio in "${RATIOS_ARR[@]}"; do
        jobs+=("${ratio}|${seed}")
    done
done

run_one() {
    local gpu="$1"
    local ratio="$2"
    local seed="$3"
    
    RUN_DIR="$SAVE_ROOT/ratio_${ratio}/seed_${seed}"
    mkdir -p "$RUN_DIR"
    
    local mem_path="$MEMORY_SUBSET_DIR/mem_mix_${ratio}pct.pt"
    
    if [[ ! -f "$mem_path" ]]; then
        echo "ERROR: Memory file not found: $mem_path"
        return 1
    fi
    
    echo "[GPU ${gpu}] Running: ratio=${ratio}% in-domain (seed=${seed})..."
    
    CUDA_VISIBLE_DEVICES=$gpu python run-rom.py \
        --seed=$TEST_SEED \
        --reset_seed_start=0 \
        --n_trajs=$NUM_TRAJS \
        --save_dir="$RUN_DIR" \
        --start_traj_id=0 \
        --start_board_id=$TEST_SEED \
        --logging.online=False \
        --agent_type="$METHOD" \
        --level="$LEVEL" \
        --oracle_prob=0 \
        --record=False \
        --max_steps=$MAX_STEPS \
        --agent_seed=$seed \
        --model_path='yunhaif/ReflectVLM-llava-v1.5-13b-post-trained' \
        --load_4bit=True \
        --romemo_init_memory_path="$mem_path" \
        --romemo_save_memory_path="$RUN_DIR/romemo_memory.pt"
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
            IFS='|' read -r ratio seed <<< "${jobs[$idx]}"
            run_one "$gpu" "$ratio" "$seed"
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

# Aggregate results (custom for OOD)
echo ""
echo "=========================================="
echo "Aggregating OOD Robustness Results..."
echo "=========================================="

python -c "
import json
import pandas as pd
from pathlib import Path

root = Path('$SAVE_ROOT')
rows = []

for ratio_dir in sorted(root.iterdir()):
    if not ratio_dir.is_dir() or not ratio_dir.name.startswith('ratio_'):
        continue
    ratio = int(ratio_dir.name.replace('ratio_', ''))
    
    for seed_dir in sorted(ratio_dir.iterdir()):
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
            'in_domain_ratio': ratio,
            'ood_ratio': 100 - ratio,
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

df.to_csv(out_dir / 'robustness_results.csv', index=False)
print(f'Saved: {out_dir}/robustness_results.csv')

# Summary by ratio
if len(df) > 0:
    summary = df.groupby('in_domain_ratio').agg({
        'success_rate': ['mean', 'std'],
        'mean_steps': ['mean', 'std'],
        'looping_rate': ['mean', 'std'],
    }).round(4)
    print()
    print('OOD Robustness Summary:')
    print(summary)
"

# Generate robustness plot
echo ""
echo "Generating Robustness Plot..."

python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

root = Path('$SAVE_ROOT')
csv_path = root / '_aggregate' / 'robustness_results.csv'
if not csv_path.exists():
    print('No results to plot')
    exit(0)

df = pd.read_csv(csv_path)
if len(df) == 0:
    print('Empty results')
    exit(0)

# Aggregate by in_domain_ratio
agg = df.groupby('in_domain_ratio').agg({
    'success_rate': ['mean', 'std'],
    'mean_steps': ['mean', 'std'],
    'looping_rate': ['mean', 'std'],
}).reset_index()
agg.columns = ['ratio', 'sr_mean', 'sr_std', 'steps_mean', 'steps_std', 'loop_mean', 'loop_std']

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Success Rate vs In-Domain Ratio
ax1.errorbar(agg['ratio'], agg['sr_mean'], yerr=agg['sr_std'], 
             marker='o', capsize=5, capthick=2, linewidth=2, markersize=8, color='blue')
ax1.set_xlabel('In-Domain Ratio (%)', fontsize=12)
ax1.set_ylabel('Success Rate', fontsize=12)
ax1.set_title('Pillar 3: OOD Robustness', fontsize=14)
ax1.set_xlim(0, 105)
ax1.set_xticks([20, 40, 60, 80, 100])
ax1.grid(True, alpha=0.3)

# Add reference line at 100% in-domain
baseline_sr = agg[agg['ratio']==100]['sr_mean'].values[0] if 100 in agg['ratio'].values else 0
ax1.axhline(baseline_sr, color='green', linestyle='--', alpha=0.5, label=f'100% in-domain: {baseline_sr:.2f}')
ax1.legend()

# Looping Rate vs In-Domain Ratio
ax2.errorbar(agg['ratio'], agg['loop_mean'], yerr=agg['loop_std'],
             marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='red')
ax2.set_xlabel('In-Domain Ratio (%)', fontsize=12)
ax2.set_ylabel('Looping Rate', fontsize=12)
ax2.set_title('Noise Tolerance', fontsize=14)
ax2.set_xlim(0, 105)
ax2.set_xticks([20, 40, 60, 80, 100])
ax2.grid(True, alpha=0.3)

plt.tight_layout()

out_dir = root / '_aggregate' / 'plots'
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'ood_robustness.png', dpi=150, bbox_inches='tight')
print(f'Saved: {out_dir}/ood_robustness.png')
"

echo ""
echo "=========================================="
echo "✅ Pillar 3: Robustness Complete!"
echo "=========================================="
echo "Results:  $SAVE_ROOT/_aggregate/robustness_results.csv"
echo "Plots:    $SAVE_ROOT/_aggregate/plots/ood_robustness.png"
echo ""
echo "Expected claim: 'Visual retrieval degrades gracefully with OOD contamination.'"
echo "               'Even at 20% in-domain, SR drops only modestly (not to zero).'"
