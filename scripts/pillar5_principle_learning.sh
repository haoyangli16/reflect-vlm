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

# Experiment settings (note: run-rom.py doesn't use n_envs)
N_TRAJS="${N_TRAJS:-200}"
MAX_STEPS="${MAX_STEPS:-30}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-symbolic}"
SYMBOLIC_WEIGHT="${SYMBOLIC_WEIGHT:-0.5}"

# Memory settings - auto-detect if not set
if [ -z "$MEMORY_PATH" ]; then
    # Try to find memory file in common locations
    if [ -f "data/mixed_memories/mixed_1000_good100.pt" ]; then
        MEMORY_PATH="data/mixed_memories/mixed_1000_good100.pt"
    elif [ -f "data/raw_expert_indomain.pt" ]; then
        MEMORY_PATH="data/raw_expert_indomain.pt"
    elif [ -f "datasets/parallel_indomain/memory_part_0.pt" ]; then
        MEMORY_PATH="datasets/parallel_indomain/memory_part_0.pt"
    else
        echo "ERROR: No memory file found. Please set MEMORY_PATH or generate memory first."
        echo "Available .pt files:"
        find data datasets -name "*.pt" 2>/dev/null | head -10
        exit 1
    fi
fi
PRINCIPLE_SAVE_DIR="${PRINCIPLE_SAVE_DIR:-data/principles}"

# Output settings
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGS_DIR="${LOGS_DIR:-logs/pillar5_principles_${TIMESTAMP}}"
mkdir -p "$LOGS_DIR"
mkdir -p "$PRINCIPLE_SAVE_DIR"

# Model paths - auto-detect if in project directory
if [ -d "./ReflectVLM-llava-v1.5-13b-base" ]; then
    BASE_MODEL="${BASE_MODEL:-./ReflectVLM-llava-v1.5-13b-base}"
else
    BASE_MODEL="${BASE_MODEL:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-base}"
fi

if [ -d "./ReflectVLM-llava-v1.5-13b-post-trained" ]; then
    POST_MODEL="${POST_MODEL:-./ReflectVLM-llava-v1.5-13b-post-trained}"
else
    POST_MODEL="${POST_MODEL:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-post-trained}"
fi

# VLM reflector settings (for VLM-based experiments)
REFLECTOR_PROVIDER="${REFLECTOR_PROVIDER:-}"  # "openai", "gemini", "qwen", or empty for rule-based
REFLECTOR_MODEL="${REFLECTOR_MODEL:-}"

echo "============================================================"
echo "Pillar 5: Principle-Based Learning Experiments"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "N_TRAJS: $N_TRAJS"
echo "Memory: $MEMORY_PATH"
echo "Retrieval Mode: $RETRIEVAL_MODE"
echo "Base Model: $BASE_MODEL"
echo "Post Model: $POST_MODEL"
echo "Logs: $LOGS_DIR"
echo "Reflector Provider: ${REFLECTOR_PROVIDER:-rule-based}"
echo "============================================================"

# Check if memory exists
if [ ! -f "$MEMORY_PATH" ]; then
    echo "ERROR: Memory file not found: $MEMORY_PATH"
    echo "Please generate memory first or set MEMORY_PATH environment variable."
    echo "Example: MEMORY_PATH=data/mixed_memories/mixed_1000_good100.pt bash scripts/pillar5_principle_learning.sh"
    exit 1
fi

# Check if models exist
if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: Base model not found: $BASE_MODEL"
    echo "Please set BASE_MODEL environment variable to the correct path."
    exit 1
fi

if [ ! -d "$POST_MODEL" ]; then
    echo "ERROR: Post model not found: $POST_MODEL"
    echo "Please set POST_MODEL environment variable to the correct path."
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
        --n_trajs="$N_TRAJS" \
        --max_steps="$MAX_STEPS" \
        --romemo_init_memory_path="$MEMORY_PATH" \
        --romemo_retrieval_mode="$RETRIEVAL_MODE" \
        --romemo_symbolic_weight="$SYMBOLIC_WEIGHT" \
        --romemo_use_principles=false \
        --save_dir="$LOGS_DIR/${output_name}_baseline" \
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
    
    # Use _wb suffix for writeback agent
    local wb_agent="${agent_type}_wb"
    
    python run-rom.py \
        --agent_type="$wb_agent" \
        --model_path="$model_path" \
        --n_trajs="$N_TRAJS" \
        --max_steps="$MAX_STEPS" \
        --romemo_init_memory_path="$MEMORY_PATH" \
        --romemo_retrieval_mode="$RETRIEVAL_MODE" \
        --romemo_symbolic_weight="$SYMBOLIC_WEIGHT" \
        --romemo_use_principles=true \
        --romemo_principle_store_path="$principle_path" \
        --save_dir="$LOGS_DIR/${output_name}_principles_rulebased" \
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
    
    # Use _wb suffix for writeback agent
    local wb_agent="${agent_type}_wb"
    
    python run-rom.py \
        --agent_type="$wb_agent" \
        --model_path="$model_path" \
        --n_trajs="$N_TRAJS" \
        --max_steps="$MAX_STEPS" \
        --romemo_init_memory_path="$MEMORY_PATH" \
        --romemo_retrieval_mode="$RETRIEVAL_MODE" \
        --romemo_symbolic_weight="$SYMBOLIC_WEIGHT" \
        --romemo_use_principles=true \
        --romemo_reflector_provider="$provider" \
        --romemo_reflector_model="$model" \
        --romemo_principle_store_path="$principle_path" \
        --save_dir="$LOGS_DIR/${output_name}_principles_${provider}" \
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
# Use bc_romemo for baseline (no writeback)
run_baseline "bc_romemo" "$BASE_MODEL" "bc"
run_with_principles_rulebased "bc_romemo" "$BASE_MODEL" "bc"
run_with_principles_vlm "bc_romemo" "$BASE_MODEL" "bc" "$REFLECTOR_PROVIDER" "$REFLECTOR_MODEL"

echo ""
echo "============================================================"
echo "Starting Experiments with Reflect (Post-Trained) Policy"
echo "============================================================"

# Reflect Policy experiments
# Use bc_romemo for baseline (no writeback)
run_baseline "bc_romemo" "$POST_MODEL" "reflect"
run_with_principles_rulebased "bc_romemo" "$POST_MODEL" "reflect"
run_with_principles_vlm "bc_romemo" "$POST_MODEL" "reflect" "$REFLECTOR_PROVIDER" "$REFLECTOR_MODEL"

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
