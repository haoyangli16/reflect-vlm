#!/bin/bash
# Pillar 4: Constraint Learning Experiment
# =========================================
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

set -e

# Configuration
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"
LOGS_DIR="${PROJECT_ROOT}/logs/pillar4_constraints_ratio"

# Create directories
mkdir -p "${LOGS_DIR}"

# ==========================================
# Configuration Variables (override with env vars)
# ==========================================
N_TRAJS="${N_TRAJS:-100}"           # Number of test trajectories
LEVEL="${LEVEL:-all}"                # Task difficulty
SEED="${SEED:-0}"                    # Random seed
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

echo "=========================================="
echo "Pillar 4: Constraint Learning Experiment"
echo "=========================================="
echo "Test trajectories: ${N_TRAJS}"
echo "Level: ${LEVEL}"
echo "Total memory size: ${TOTAL_MEMORY_SIZE}"
echo "Good ratios (%):   ${GOOD_RATIOS}"
echo "Success memory: ${SUCCESS_MEMORY}"
echo "Failure memory: ${FAILURE_MEMORY}"
echo "Methods: ${METHODS}"
echo "=========================================="

# ==========================================
# Helper: Build a fixed-size mixed memory bank
# ==========================================
make_mixed_memory() {
    local success_pt="$1"
    local failure_pt="$2"
    local total_n="$3"
    local good_ratio="$4"
    local seed="$5"
    local output_pt="$6"

    python - <<'PY'
import os
import random
from pathlib import Path

success_path = Path(os.environ["SUCCESS_PT"])
failure_path = Path(os.environ["FAILURE_PT"])
output_path = Path(os.environ["OUTPUT_PT"])
total_n = int(os.environ["TOTAL_N"])
good_ratio = int(os.environ["GOOD_RATIO"])
seed = int(os.environ["MIX_SEED"])

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
print(f"[OK] Saved mixed memory: {output_path}")
print(f"     total={len(mixed)} good={n_good} fail={n_fail} (good_ratio={good_ratio}%)")
PY
}

# ==========================================
# Run Experiments
# ==========================================

# Parse methods
IFS=',' read -ra METHODS_ARRAY <<< "${METHODS}"
IFS=',' read -ra RATIOS_ARRAY <<< "${GOOD_RATIOS}"

for ratio in "${RATIOS_ARRAY[@]}"; do
    ratio="${ratio// /}"
    [[ -z "$ratio" ]] && continue

    MIXED_MEMORY="${DATA_DIR}/mixed_memories/mixed_${TOTAL_MEMORY_SIZE}_good${ratio}.pt"
    if [[ ! -f "${MIXED_MEMORY}" ]]; then
        echo ""
        echo "Creating mixed memory: total=${TOTAL_MEMORY_SIZE}, good_ratio=${ratio}%"
        SUCCESS_PT="${SUCCESS_MEMORY}" FAILURE_PT="${FAILURE_MEMORY}" TOTAL_N="${TOTAL_MEMORY_SIZE}" GOOD_RATIO="${ratio}" MIX_SEED="${SEED}" OUTPUT_PT="${MIXED_MEMORY}" \
            make_mixed_memory "${SUCCESS_MEMORY}" "${FAILURE_MEMORY}" "${TOTAL_MEMORY_SIZE}" "${ratio}" "${SEED}" "${MIXED_MEMORY}"
    fi

    for method in "${METHODS_ARRAY[@]}"; do
        method="${method// /}"
        [[ -z "$method" ]] && continue

        echo ""
        echo "=========================================="
        echo "Running: ${method} | good_ratio=${ratio}% | total=${TOTAL_MEMORY_SIZE}"
        echo "=========================================="

        METHOD_DIR="${LOGS_DIR}/${method}/good_${ratio}"
        mkdir -p "${METHOD_DIR}"

        AGENT_TYPE="${method}"

        # Pick model by method family:
        # - bc* uses BASE model
        # - reflect* uses POST-TRAINED model
        if [[ "${AGENT_TYPE}" == reflect* ]]; then
            MODEL_PATH="${MODEL_PATH:-$POST_MODEL_PATH}"
        else
            MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH}"
        fi

        python -u "${PROJECT_ROOT}/run-rom.py" \
            --agent_type="${AGENT_TYPE}" \
            --seed="${SEED}" \
            --n_trajs="${N_TRAJS}" \
            --level="${LEVEL}" \
            --max_steps=50 \
            --model_path="${MODEL_PATH}" \
            --load_4bit="${LOAD_4BIT}" \
            --save_dir="${METHOD_DIR}" \
            --romemo_init_memory_path="${MIXED_MEMORY}" \
            --trace_jsonl=True \
            --save_images=False \
            --record=False \
            --logging.use_wandb=False \
            2>&1 | tee "${METHOD_DIR}/run.log"

        echo "Completed: ${method} @ good_ratio=${ratio}%"
        echo "Results saved to: ${METHOD_DIR}"
    done
done

# ==========================================
# Aggregate Results
# ==========================================
echo ""
echo "=========================================="
echo "Aggregating Results..."
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

        ep_trace = ratio_dir / 'traces' / 'episode_traces.jsonl'
        if not ep_trace.exists():
            print(f'Warning: No traces found for {method} {ratio_dir.name}')
            continue

        episodes = [json.loads(ln) for ln in ep_trace.read_text().splitlines() if ln.strip()]
        if not episodes:
            print(f'Warning: Empty traces for {method} {ratio_dir.name}')
            continue

        n_eps = len(episodes)
        success_rate = sum(1 for e in episodes if e.get('success', False)) / n_eps
        mean_steps = sum(e.get('steps', 0) for e in episodes) / n_eps
        looping_rate = sum(e.get('looping_rate', 0) for e in episodes) / n_eps
        repeated_failure_rate = sum(e.get('repeated_failure_rate', 0) for e in episodes) / n_eps

        results.append({
            'method': method,
            'good_ratio': good_ratio,
            'total_memory_size': int('${TOTAL_MEMORY_SIZE}'),
            'n_episodes': n_eps,
            'success_rate': round(success_rate, 4),
            'mean_steps': round(mean_steps, 2),
            'looping_rate': round(looping_rate, 4),
            'repeated_failure_rate': round(repeated_failure_rate, 4),
        })

        print(f'{method:20s} | good={good_ratio:3d}% | SR={success_rate:.2%} | Steps={mean_steps:.1f} | Loop={looping_rate:.2%} | RepFail={repeated_failure_rate:.2%}')

# Save results
if results:
    df = pd.DataFrame(results)
    csv_path = logs_dir / 'results.csv'
    df.to_csv(csv_path, index=False)
    print(f'\\nResults saved to: {csv_path}')
"

echo ""
echo "=========================================="
echo "Pillar 4 Experiment Complete!"
echo "=========================================="
echo "Results: ${LOGS_DIR}/results.csv"
echo ""
echo "Key metrics to compare:"
echo "  - looping_rate: Should DECREASE with failure memory"
echo "  - repeated_failure_rate: Should DECREASE with failure memory"
echo "  - success_rate: May INCREASE if fewer constraint violations"
