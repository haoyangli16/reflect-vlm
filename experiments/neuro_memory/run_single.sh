#!/bin/bash
# =============================================================================
# Single Run Script for Neuro-Symbolic Memory Experiment
# =============================================================================
# This script runs a single experiment with flexible configuration.
# It wraps run_memory_experiment.py and handles environment setup.
#
# Usage:
#   ./run_single.sh [flags]
#
# Flags:
#   --mode <mode>               Experiment mode: 'baseline' or 'memory' (default: memory)
#   --episodes <n>              Number of episodes (default: 50)
#   --seed <n>                  Starting seed (default: 1000001)
#   --name <name>               Experiment name (default: exp)
#   --output <dir>              Output directory (default: logs/neuro_memory_exp)
#
# Policy (Action Agent):
#   --policy-type <type>        'bc' (LLaVA) or 'vlm' (External) (default: bc)
#   --policy-provider <name>    Provider for VLM policy (openai, gemini, qwen, kimi)
#   --policy-model <name>       Specific model name for VLM policy
#
# BC Policy Options (LLaVA):
#   --base-model <path>         Path to base LLaVA model
#   --post-model <path>         Path to post-trained LLaVA model
#   --use-post-train            Flag to use post-trained model instead of base
#
# Reflection (Memory System):
#   --reflection-provider <name> Provider for reflection (rule, kimi, openai, gemini, qwen) (default: rule)
#   --reflection-model <name>    Specific model name for reflection
#
# Examples:
#   ./run_single.sh --mode baseline --episodes 20
#   ./run_single.sh --mode memory --use-post-train --reflection-provider kimi
#   ./run_single.sh --policy-type vlm --policy-provider openai --reflection-provider kimi
# =============================================================================

set -e

# =============================================================================
# Default Configuration
# =============================================================================
MODE="memory"
N_EPISODES=50
SEED=1000001
EXP_NAME="exp"
SAVE_DIR="logs/neuro_memory_exp"

# Policy Defaults
POLICY_TYPE="bc"
POLICY_PROVIDER=""
POLICY_MODEL=""

# BC Model Defaults
BASE_MODEL_PATH="./ReflectVLM-llava-v1.5-13b-base"
POST_MODEL_PATH="./ReflectVLM-llava-v1.5-13b-post-trained"
USE_POST_TRAINED=false

# Reflection Defaults
REFLECTION_PROVIDER="rule"
REFLECTION_MODEL=""

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift ;;
        --episodes) N_EPISODES="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --name) EXP_NAME="$2"; shift ;;
        --output) SAVE_DIR="$2"; shift ;;
        
        --policy-type) POLICY_TYPE="$2"; shift ;;
        --policy-provider) POLICY_PROVIDER="$2"; shift ;;
        --policy-model) POLICY_MODEL="$2"; shift ;;
        
        --base-model) BASE_MODEL_PATH="$2"; shift ;;
        --post-model) POST_MODEL_PATH="$2"; shift ;;
        --use-post-train) USE_POST_TRAINED=true ;;
        
        --reflection-provider) REFLECTION_PROVIDER="$2"; shift ;;
        --reflection-model) REFLECTION_MODEL="$2"; shift ;;
        
        -h|--help)
            # Print usage from the header comments
            sed -n '2,27p' "$0"
            exit 0
            ;; 
        *) 
            echo "Unknown parameter: $1"
            exit 1 
            ;; 
    esac
    shift
done

# =============================================================================
# Environment Setup
# =============================================================================

# MuJoCo EGL rendering (for headless servers without display)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Fix for LLVM command-line option conflict
export TRITON_PTXAS_PATH=""

# CUDA setup
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export BNB_CUDA_VERSION=117

# Performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

# Python settings
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true

# =============================================================================
# API Key Checks
# =============================================================================
check_api_key() {
    local provider=$1
    case "$provider" in
        kimi)
            if [[ -z "$MOONSHOT_API_KEY" ]]; then
                echo "ERROR: MOONSHOT_API_KEY not set for $provider"
                return 1
            fi
            ;; 
        openai)
            if [[ -z "$OPENAI_API_KEY" ]]; then
                echo "ERROR: OPENAI_API_KEY not set for $provider"
                return 1
            fi
            ;; 
        gemini)
            if [[ -z "$GOOGLE_API_KEY" ]]; then
                echo "ERROR: GOOGLE_API_KEY not set for $provider"
                return 1
            fi
            ;; 
        qwen)
            if [[ -z "$DASHSCOPE_API_KEY" ]]; then
                echo "ERROR: DASHSCOPE_API_KEY not set for $provider"
                return 1
            fi
            ;; 
    esac
    return 0
}

# Check Reflection API Key
if [[ "$MODE" == "memory" && "$REFLECTION_PROVIDER" != "rule" ]]; then
    if ! check_api_key "$REFLECTION_PROVIDER"; then
        echo "Please export the required API key."
        exit 1
    fi
fi

# Check Policy API Key
if [[ "$POLICY_TYPE" == "vlm" ]]; then
    if [[ -z "$POLICY_PROVIDER" ]]; then
        echo "ERROR: --policy-provider required when --policy-type is 'vlm'"
        exit 1
    fi
    if ! check_api_key "$POLICY_PROVIDER"; then
        echo "Please export the required API key."
        exit 1
    fi
fi

# =============================================================================
# Construct Command
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_SAVE_DIR="${SAVE_DIR}_${MODE}_${TIMESTAMP}"
mkdir -p "$FULL_SAVE_DIR"

CMD="python experiments/neuro_memory/run_memory_experiment.py \
    --mode $MODE \
    --n_episodes $N_EPISODES \
    --seed_start $SEED \
    --name $EXP_NAME \
    --save_dir $SAVE_DIR \
    --policy_type $POLICY_TYPE \
    --base_model $BASE_MODEL_PATH \
    --post_model $POST_MODEL_PATH \
    --provider $REFLECTION_PROVIDER \
    --verbose \
    --show_memory \
    --memory_interval 5"

if [ "$USE_POST_TRAINED" = true ]; then
    CMD="$CMD --use_post_trained"
fi

if [ -n "$POLICY_PROVIDER" ]; then
    CMD="$CMD --policy_provider $POLICY_PROVIDER"
fi

if [ -n "$POLICY_MODEL" ]; then
    CMD="$CMD --policy_model $POLICY_MODEL"
fi

if [ -n "$REFLECTION_MODEL" ]; then
    CMD="$CMD --model $REFLECTION_MODEL"
fi

# =============================================================================
# Run
# =============================================================================
echo "============================================================"
echo "Starting Experiment: $EXP_NAME"
echo "============================================================"
echo "Mode:           $MODE"
echo "Episodes:       $N_EPISODES"
echo "Policy:         $POLICY_TYPE"
if [ "$POLICY_TYPE" == "bc" ]; then
    echo "  Base Model:   $BASE_MODEL_PATH"
    echo "  Post Model:   $POST_MODEL_PATH"
    echo "  Using Post:   $USE_POST_TRAINED"
else
    echo "  Provider:     $POLICY_PROVIDER"
    echo "  Model:        $POLICY_MODEL"
fi
echo "Reflection:     $REFLECTION_PROVIDER"
if [ -n "$REFLECTION_MODEL" ]; then
    echo "  Model:        $REFLECTION_MODEL"
fi
echo "Log Dir:        $FULL_SAVE_DIR"
echo "============================================================"
echo "Command:"
echo "$CMD"
echo "============================================================"

# Execute
$CMD 2>&1 | tee "$FULL_SAVE_DIR/run.log"