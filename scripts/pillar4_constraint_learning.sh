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
#   1. Baseline: BC/Reflect with Success Memory ONLY
#   2. Ours: BC/Reflect with Success Memory + Failure Memory
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
LOGS_DIR="${PROJECT_ROOT}/logs/pillar4_constraints"

# Create directories
mkdir -p "${LOGS_DIR}"

# ==========================================
# Configuration Variables (override with env vars)
# ==========================================
N_TRAJS="${N_TRAJS:-100}"           # Number of test trajectories
LEVEL="${LEVEL:-all}"                # Task difficulty
SEED="${SEED:-0}"                    # Random seed
MODEL_PATH="${MODEL_PATH:-yunhaif/ReflectVLM-llava-v1.5-13b-post-trained}"
LOAD_4BIT="${LOAD_4BIT:-True}"

# Memory paths
SUCCESS_MEMORY="${SUCCESS_MEMORY:-${DATA_DIR}/raw_expert_indomain.pt}"
FAILURE_MEMORY="${FAILURE_MEMORY:-${DATA_DIR}/raw_failure_constraints.pt}"

# Methods to run
# Baseline: BC with success memory only
# Ours: BC with success + failure memory
METHODS="${METHODS:-bc_romemo,bc_romemo_fail}"

echo "=========================================="
echo "Pillar 4: Constraint Learning Experiment"
echo "=========================================="
echo "Test trajectories: ${N_TRAJS}"
echo "Level: ${LEVEL}"
echo "Success memory: ${SUCCESS_MEMORY}"
echo "Failure memory: ${FAILURE_MEMORY}"
echo "Methods: ${METHODS}"
echo "=========================================="

# ==========================================
# Helper: Merge two memory banks
# ==========================================
merge_memories() {
    local success_pt="$1"
    local failure_pt="$2"
    local output_pt="$3"
    
    python -c "
import torch
from pathlib import Path

success_path = Path('${success_pt}')
failure_path = Path('${failure_pt}')
output_path = Path('${output_pt}')

print(f'Merging memories:')
print(f'  Success: {success_path}')
print(f'  Failure: {failure_path}')

all_experiences = []

# Load success memory
if success_path.exists():
    data = torch.load(success_path, map_location='cpu')
    exps = data.get('experiences', data.get('memory', []))
    print(f'  Loaded {len(exps)} success experiences')
    all_experiences.extend(exps)
else:
    print(f'  Warning: Success memory not found at {success_path}')

# Load failure memory
if failure_path.exists():
    data = torch.load(failure_path, map_location='cpu')
    exps = data.get('experiences', data.get('memory', []))
    print(f'  Loaded {len(exps)} failure experiences')
    all_experiences.extend(exps)
else:
    print(f'  Warning: Failure memory not found at {failure_path}')

print(f'  Total merged: {len(all_experiences)} experiences')

# Save merged
merged = {
    'experiences': all_experiences,
    'memory': all_experiences,
    'task': 'assembly',
    'metadata': {
        'source': 'merged_success_failure',
        'num_success': len(data.get('experiences', data.get('memory', []))) if success_path.exists() else 0,
        'num_failure': len(data.get('experiences', data.get('memory', []))) if failure_path.exists() else 0,
    }
}

output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(merged, output_path)
print(f'  Saved to {output_path}')
"
}

# ==========================================
# Run Experiments
# ==========================================

# Create merged memory (success + failure)
MERGED_MEMORY="${DATA_DIR}/merged_success_failure.pt"
if [ ! -f "${MERGED_MEMORY}" ]; then
    echo ""
    echo "Creating merged memory (Success + Failure)..."
    merge_memories "${SUCCESS_MEMORY}" "${FAILURE_MEMORY}" "${MERGED_MEMORY}"
fi

# Parse methods
IFS=',' read -ra METHODS_ARRAY <<< "${METHODS}"

for method in "${METHODS_ARRAY[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: ${method}"
    echo "=========================================="
    
    METHOD_DIR="${LOGS_DIR}/${method}"
    mkdir -p "${METHOD_DIR}"
    
    # Select memory based on method
    if [[ "${method}" == *"_fail"* ]]; then
        # Use merged memory (success + failure)
        INIT_MEMORY="${MERGED_MEMORY}"
        BASE_METHOD="${method%_fail}"  # Remove _fail suffix
        echo "Using MERGED memory (Success + Failure): ${INIT_MEMORY}"
    else
        # Use success memory only
        INIT_MEMORY="${SUCCESS_MEMORY}"
        BASE_METHOD="${method}"
        echo "Using SUCCESS-only memory: ${INIT_MEMORY}"
    fi
    
    # Map method name to actual agent_type
    AGENT_TYPE="${BASE_METHOD}"
    
    # Run experiment
    python -u "${PROJECT_ROOT}/run-rom.py" \
        --agent_type="${AGENT_TYPE}" \
        --seed="${SEED}" \
        --n_trajs="${N_TRAJS}" \
        --level="${LEVEL}" \
        --max_steps=50 \
        --model_path="${MODEL_PATH}" \
        --load_4bit="${LOAD_4BIT}" \
        --save_dir="${METHOD_DIR}" \
        --romemo_init_memory_path="${INIT_MEMORY}" \
        --trace_jsonl=True \
        --save_images=False \
        --record=False \
        --logging.use_wandb=False \
        2>&1 | tee "${METHOD_DIR}/run.log"
    
    echo "Completed: ${method}"
    echo "Results saved to: ${METHOD_DIR}"
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
    
    ep_trace = method_dir / 'traces' / 'episode_traces.jsonl'
    if not ep_trace.exists():
        print(f'Warning: No traces found for {method}')
        continue
    
    # Load episode traces
    episodes = [json.loads(ln) for ln in ep_trace.read_text().splitlines() if ln.strip()]
    
    if not episodes:
        print(f'Warning: Empty traces for {method}')
        continue
    
    # Calculate metrics
    n_eps = len(episodes)
    success_rate = sum(1 for e in episodes if e.get('success', False)) / n_eps
    mean_steps = sum(e.get('steps', 0) for e in episodes) / n_eps
    looping_rate = sum(e.get('looping_rate', 0) for e in episodes) / n_eps
    repeated_failure_rate = sum(e.get('repeated_failure_rate', 0) for e in episodes) / n_eps
    
    results.append({
        'method': method,
        'n_episodes': n_eps,
        'success_rate': round(success_rate, 4),
        'mean_steps': round(mean_steps, 2),
        'looping_rate': round(looping_rate, 4),
        'repeated_failure_rate': round(repeated_failure_rate, 4),
    })
    
    print(f'{method:25s} | SR={success_rate:.2%} | Steps={mean_steps:.1f} | Loop={looping_rate:.2%} | RepFail={repeated_failure_rate:.2%}')

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
