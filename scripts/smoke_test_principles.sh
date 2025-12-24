#!/bin/bash
# ============================================================================
# Smoke Test: Principle-Based Learning
# ============================================================================
#
# Quick test to verify principle system works end-to-end.
# Uses small number of environments and trajectories.
#
# Usage:
#   bash scripts/smoke_test_principles.sh
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Small test settings
N_TRAJS=10
MAX_STEPS=15

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGS_DIR="logs/smoke_principles_${TIMESTAMP}"
PRINCIPLE_DIR="data/principles"
mkdir -p "$LOGS_DIR"
mkdir -p "$PRINCIPLE_DIR"

# Memory (use existing if available - try multiple locations)
# Priority: .pt files in data/mixed_memories, then data/*.pt, then datasets/
MEMORY_PATH=""
if [ -f "data/mixed_memories/mixed_1000_good100.pt" ]; then
    MEMORY_PATH="data/mixed_memories/mixed_1000_good100.pt"
elif [ -f "data/raw_expert_indomain.pt" ]; then
    MEMORY_PATH="data/raw_expert_indomain.pt"
elif [ -f "datasets/parallel_indomain/memory_part_0.pt" ]; then
    MEMORY_PATH="datasets/parallel_indomain/memory_part_0.pt"
fi

# Model paths - use relative paths if in project directory, else use absolute
if [ -d "./ReflectVLM-llava-v1.5-13b-base" ]; then
    BASE_MODEL="${BASE_MODEL:-./ReflectVLM-llava-v1.5-13b-base}"
else
    BASE_MODEL="${BASE_MODEL:-/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-base}"
fi

echo "============================================================"
echo "Smoke Test: Principle-Based Learning"
echo "============================================================"
echo "N_ENVS: $N_ENVS"
echo "N_TRAJS: $N_TRAJS"
echo "Logs: $LOGS_DIR"
echo "============================================================"

# Check if we can run with memory
USE_MEMORY=false
if [ -n "$MEMORY_PATH" ] && [ -f "$MEMORY_PATH" ]; then
    USE_MEMORY=true
    echo "Found memory at: $MEMORY_PATH"
else
    echo "No memory found, will test principle system without pre-loaded memory"
    echo "To use memory, generate it first or set MEMORY_PATH environment variable"
fi

# Check model exists
if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: Model not found at: $BASE_MODEL"
    echo "Please set BASE_MODEL to the correct path"
    exit 1
fi
echo "Using model: $BASE_MODEL"

# ============================================================================
# Test 1: Rule-based Reflector (No VLM API calls)
# ============================================================================
echo ""
echo "------------------------------------------------------------"
echo "Test 1: Rule-Based Reflector"
echo "------------------------------------------------------------"

PRINCIPLE_FILE="$PRINCIPLE_DIR/smoke_test_rulebased.json"

# Build command (note: run-rom.py doesn't use n_envs flag)
CMD="python run-rom.py \
    --agent_type=bc_romemo_wb \
    --model_path=$BASE_MODEL \
    --n_trajs=$N_TRAJS \
    --max_steps=$MAX_STEPS \
    --romemo_retrieval_mode=symbolic \
    --romemo_use_principles=true \
    --romemo_principle_store_path=$PRINCIPLE_FILE \
    --save_dir=$LOGS_DIR/rulebased"

if [ "$USE_MEMORY" = true ]; then
    CMD="$CMD --romemo_init_memory_path=$MEMORY_PATH"
fi

echo "Running: $CMD"
eval "$CMD" 2>&1 | tee "$LOGS_DIR/test1_rulebased.log"

# Check if principles were saved
if [ -f "$PRINCIPLE_FILE" ]; then
    echo "✅ Principles saved successfully"
    echo "Principle file content preview:"
    head -c 500 "$PRINCIPLE_FILE"
    echo ""
else
    echo "⚠️  No principles file created (may be normal if no failures occurred)"
fi

# ============================================================================
# Test 2: Verify Principle Loading
# ============================================================================
echo ""
echo "------------------------------------------------------------"
echo "Test 2: Load and Use Existing Principles"
echo "------------------------------------------------------------"

if [ -f "$PRINCIPLE_FILE" ]; then
    CMD="python run-rom.py \
        --agent_type=bc_romemo \
        --model_path=$BASE_MODEL \
        --n_trajs=5 \
        --max_steps=10 \
        --romemo_retrieval_mode=symbolic \
        --romemo_use_principles=true \
        --romemo_principle_store_path=$PRINCIPLE_FILE \
        --save_dir=$LOGS_DIR/load_test"
    
    if [ "$USE_MEMORY" = true ]; then
        CMD="$CMD --romemo_init_memory_path=$MEMORY_PATH"
    fi
    
    echo "Running: $CMD"
    eval "$CMD" 2>&1 | tee "$LOGS_DIR/test2_load.log"
    echo "✅ Principle loading test complete"
else
    echo "Skipping load test (no principles to load)"
fi

# ============================================================================
# Test 3: VLM Reflector (if API key available)
# ============================================================================
echo ""
echo "------------------------------------------------------------"
echo "Test 3: VLM Reflector (Optional)"
echo "------------------------------------------------------------"

# Check for API keys
VLM_PROVIDER=""
VLM_MODEL=""

if [ -n "$OPENAI_API_KEY" ]; then
    VLM_PROVIDER="openai"
    VLM_MODEL="gpt-4o"
    echo "Found OpenAI API key"
elif [ -n "$GOOGLE_API_KEY" ]; then
    VLM_PROVIDER="gemini"
    VLM_MODEL="gemini-3-pro-preview"
    echo "Found Google API key"
elif [ -n "$DASHSCOPE_API_KEY" ] || [ -n "$QWEN_API_KEY" ]; then
    VLM_PROVIDER="qwen"
    VLM_MODEL="qwen3-vl-235b-a22b-instruct"
    echo "Found DashScope/Qwen API key"
fi

if [ -n "$VLM_PROVIDER" ]; then
    PRINCIPLE_FILE_VLM="$PRINCIPLE_DIR/smoke_test_${VLM_PROVIDER}.json"
    
    CMD="python run-rom.py \
        --agent_type=bc_romemo_wb \
        --model_path=$BASE_MODEL \
        --n_trajs=5 \
        --max_steps=10 \
        --romemo_retrieval_mode=symbolic \
        --romemo_use_principles=true \
        --romemo_reflector_provider=$VLM_PROVIDER \
        --romemo_reflector_model=$VLM_MODEL \
        --romemo_principle_store_path=$PRINCIPLE_FILE_VLM \
        --save_dir=$LOGS_DIR/vlm_${VLM_PROVIDER}"
    
    if [ "$USE_MEMORY" = true ]; then
        CMD="$CMD --romemo_init_memory_path=$MEMORY_PATH"
    fi
    
    echo "Running VLM reflector test with $VLM_PROVIDER..."
    eval "$CMD" 2>&1 | tee "$LOGS_DIR/test3_vlm_${VLM_PROVIDER}.log"
    echo "✅ VLM reflector test complete"
else
    echo "⏭️  Skipping VLM test (no API keys found)"
    echo "   Set OPENAI_API_KEY, GOOGLE_API_KEY, or DASHSCOPE_API_KEY to test VLM reflector"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "Smoke Test Complete!"
echo "============================================================"
echo "Logs saved to: $LOGS_DIR"
echo "Principles saved to: $PRINCIPLE_DIR"
echo ""
echo "Check logs for any errors:"
echo "  grep -i 'error\|exception\|failed' $LOGS_DIR/*.log"
echo "============================================================"
