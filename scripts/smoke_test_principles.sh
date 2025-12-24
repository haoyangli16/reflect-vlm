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
N_ENVS=3
N_TRAJS=10
MAX_STEPS=15

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGS_DIR="logs/smoke_principles_${TIMESTAMP}"
PRINCIPLE_DIR="data/principles"
mkdir -p "$LOGS_DIR"
mkdir -p "$PRINCIPLE_DIR"

# Memory (use existing if available, else skip memory-dependent tests)
MEMORY_PATH="data/memory/mixed_memory.json"

# Model paths
BASE_MODEL="${BASE_MODEL:-/share/project/lhy/ReflectVLM-llava-v1.5-13b-base}"

echo "============================================================"
echo "Smoke Test: Principle-Based Learning"
echo "============================================================"
echo "N_ENVS: $N_ENVS"
echo "N_TRAJS: $N_TRAJS"
echo "Logs: $LOGS_DIR"
echo "============================================================"

# Check if we can run with memory
USE_MEMORY=false
if [ -f "$MEMORY_PATH" ]; then
    USE_MEMORY=true
    echo "Found memory at: $MEMORY_PATH"
else
    echo "No memory found, will test principle system without pre-loaded memory"
fi

# ============================================================================
# Test 1: Rule-based Reflector (No VLM API calls)
# ============================================================================
echo ""
echo "------------------------------------------------------------"
echo "Test 1: Rule-Based Reflector"
echo "------------------------------------------------------------"

PRINCIPLE_FILE="$PRINCIPLE_DIR/smoke_test_rulebased.json"

# Build command
CMD="python run-rom.py \
    --agent_type=bc_romemo_wb \
    --model_path=$BASE_MODEL \
    --n_envs=$N_ENVS \
    --n_trajs=$N_TRAJS \
    --max_steps=$MAX_STEPS \
    --romemo_retrieval_mode=symbolic \
    --romemo_use_principles=true \
    --romemo_reflector_provider= \
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
        --n_envs=2 \
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
        --n_envs=2 \
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
