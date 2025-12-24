#!/bin/bash
# ============================================================================
# Pillar 5: Principle-Based Learning Experiments
# ============================================================================
#
# This script evaluates the principle extraction and usage system:
#   1. Baseline (no principles)
#   2. Rule-based reflector (no API calls)
#   3. VLM-based reflector (with API calls)
#
# Metrics:
#   - Success rate improvement over baseline
#   - Number of principles learned
#   - Principle quality (confidence, usage frequency)
#   - Failure reduction by fail_tag
#
# Usage:
#   bash scripts/pillar5_principle_learning.sh
#
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Experiment settings
N_ENVS="${N_ENVS:-50}"
N_TRAJS="${N_TRAJS:-200}"
MAX_STEPS="${MAX_STEPS:-30}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-symbolic}"
SYMBOLIC_WEIGHT="${SYMBOLIC_WEIGHT:-0.5}"

# Memory settings
MEMORY_PATH="${MEMORY_PATH:-data/memory/mixed_memory.json}"
PRINCIPLE_SAVE_DIR="${PRINCIPLE_SAVE_DIR:-data/principles}"

# Output settings
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGS_DIR="${LOGS_DIR:-logs/pillar5_principles_${TIMESTAMP}}"
mkdir -p "$LOGS_DIR"
mkdir -p "$PRINCIPLE_SAVE_DIR"

# Model paths
BASE_MODEL="${BASE_MODEL:-/share/project/lhy/ReflectVLM-llava-v1.5-13b-base}"
POST_MODEL="${POST_MODEL:-/share/project/lhy/ReflectVLM-llava-v1.5-13b-post-trained}"

# VLM reflector settings (for VLM-based experiments)
REFLECTOR_PROVIDER="${REFLECTOR_PROVIDER:-}"  # "openai", "gemini", "qwen", or empty for rule-based
REFLECTOR_MODEL="${REFLECTOR_MODEL:-}"

echo "============================================================"
echo "Pillar 5: Principle-Based Learning Experiments"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "N_ENVS: $N_ENVS"
echo "N_TRAJS: $N_TRAJS"
echo "Memory: $MEMORY_PATH"
echo "Retrieval Mode: $RETRIEVAL_MODE"
echo "Logs: $LOGS_DIR"
echo "Reflector Provider: ${REFLECTOR_PROVIDER:-rule-based}"
echo "============================================================"

# Check if memory exists
if [ ! -f "$MEMORY_PATH" ]; then
    echo "ERROR: Memory file not found: $MEMORY_PATH"
    echo "Please generate memory first using step0_generate_expert_data.sh or step0_generate_failure_data.sh"
    exit 1
fi

# ============================================================================
# Experiment 1: Baseline (No Principles)
# ============================================================================
run_baseline() {
    local agent_type=$1
    local model_path=$2
    local output_name=$3
    
    echo ""
    echo "------------------------------------------------------------"
    echo "Running: Baseline ($output_name) - No Principles"
    echo "------------------------------------------------------------"
    
    python run-rom.py \
        --agent_type="$agent_type" \
        --model_path="$model_path" \
        --n_envs="$N_ENVS" \
        --n_trajs="$N_TRAJS" \
        --max_episode_steps="$MAX_STEPS" \
        --romemo_memory_path="$MEMORY_PATH" \
        --romemo_retrieval_mode="$RETRIEVAL_MODE" \
        --romemo_symbolic_weight="$SYMBOLIC_WEIGHT" \
        --romemo_use_principles=false \
        --romemo_writeback=false \
        --output_dir="$LOGS_DIR/${output_name}_baseline" \
        2>&1 | tee "$LOGS_DIR/${output_name}_baseline.log"
}

# ============================================================================
# Experiment 2: With Principles (Rule-Based Reflector)
# ============================================================================
run_with_principles_rulebased() {
    local agent_type=$1
    local model_path=$2
    local output_name=$3
    
    echo ""
    echo "------------------------------------------------------------"
    echo "Running: With Principles - Rule-Based ($output_name)"
    echo "------------------------------------------------------------"
    
    local principle_path="$PRINCIPLE_SAVE_DIR/${output_name}_rulebased_principles.json"
    
    python run-rom.py \
        --agent_type="$agent_type" \
        --model_path="$model_path" \
        --n_envs="$N_ENVS" \
        --n_trajs="$N_TRAJS" \
        --max_episode_steps="$MAX_STEPS" \
        --romemo_memory_path="$MEMORY_PATH" \
        --romemo_retrieval_mode="$RETRIEVAL_MODE" \
        --romemo_symbolic_weight="$SYMBOLIC_WEIGHT" \
        --romemo_use_principles=true \
        --romemo_reflector_provider="" \
        --romemo_principle_store_path="$principle_path" \
        --romemo_writeback=true \
        --output_dir="$LOGS_DIR/${output_name}_principles_rulebased" \
        2>&1 | tee "$LOGS_DIR/${output_name}_principles_rulebased.log"
    
    echo "Principles saved to: $principle_path"
}

# ============================================================================
# Experiment 3: With Principles (VLM-Based Reflector)
# ============================================================================
run_with_principles_vlm() {
    local agent_type=$1
    local model_path=$2
    local output_name=$3
    local provider=$4
    local model=$5
    
    if [ -z "$provider" ]; then
        echo "Skipping VLM-based experiment (no provider specified)"
        return
    fi
    
    echo ""
    echo "------------------------------------------------------------"
    echo "Running: With Principles - VLM ($provider) ($output_name)"
    echo "------------------------------------------------------------"
    
    local principle_path="$PRINCIPLE_SAVE_DIR/${output_name}_${provider}_principles.json"
    
    python run-rom.py \
        --agent_type="$agent_type" \
        --model_path="$model_path" \
        --n_envs="$N_ENVS" \
        --n_trajs="$N_TRAJS" \
        --max_episode_steps="$MAX_STEPS" \
        --romemo_memory_path="$MEMORY_PATH" \
        --romemo_retrieval_mode="$RETRIEVAL_MODE" \
        --romemo_symbolic_weight="$SYMBOLIC_WEIGHT" \
        --romemo_use_principles=true \
        --romemo_reflector_provider="$provider" \
        --romemo_reflector_model="$model" \
        --romemo_principle_store_path="$principle_path" \
        --romemo_writeback=true \
        --output_dir="$LOGS_DIR/${output_name}_principles_${provider}" \
        2>&1 | tee "$LOGS_DIR/${output_name}_principles_${provider}.log"
    
    echo "Principles saved to: $principle_path"
}

# ============================================================================
# Run All Experiments
# ============================================================================

echo ""
echo "============================================================"
echo "Starting Experiments with BC (Base) Policy"
echo "============================================================"

# BC Policy experiments
run_baseline "bc" "$BASE_MODEL" "bc"
run_with_principles_rulebased "bc" "$BASE_MODEL" "bc"
run_with_principles_vlm "bc" "$BASE_MODEL" "bc" "$REFLECTOR_PROVIDER" "$REFLECTOR_MODEL"

echo ""
echo "============================================================"
echo "Starting Experiments with Reflect (Post-Trained) Policy"
echo "============================================================"

# Reflect Policy experiments
run_baseline "bc" "$POST_MODEL" "reflect"
run_with_principles_rulebased "bc" "$POST_MODEL" "reflect"
run_with_principles_vlm "bc" "$POST_MODEL" "reflect" "$REFLECTOR_PROVIDER" "$REFLECTOR_MODEL"

# ============================================================================
# Aggregate Results
# ============================================================================
echo ""
echo "============================================================"
echo "Aggregating Results"
echo "============================================================"

python scripts/analyze_principle_experiments.py \
    --logs_dir="$LOGS_DIR" \
    --output_csv="$LOGS_DIR/results_summary.csv" \
    --output_plot="$LOGS_DIR/results_comparison.png" \
    2>&1 | tee "$LOGS_DIR/analysis.log"

echo ""
echo "============================================================"
echo "Experiment Complete!"
echo "============================================================"
echo "Results saved to: $LOGS_DIR"
echo "Summary CSV: $LOGS_DIR/results_summary.csv"
echo "Comparison Plot: $LOGS_DIR/results_comparison.png"
echo "============================================================"
