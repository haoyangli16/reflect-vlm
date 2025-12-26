#!/bin/bash
# =============================================================================
# Single Run Script for Neuro-Symbolic Memory Experiment
# =============================================================================
# This script runs a single experiment with all required environment setup.
#
# Usage (BC policy - LLaVA):
#   ./run_single.sh baseline 50           # Baseline mode, 50 episodes
#   ./run_single.sh memory 100 kimi       # Memory mode, 100 episodes, Kimi VLM
#   ./run_single.sh memory 100 rule       # Memory mode, rule-based reflection
#
# Usage (VLM policy - external VLMs):
#   ./run_single.sh memory 50 kimi vlm openai    # VLM policy (GPT-4V) + Kimi reflection
#   ./run_single.sh memory 50 kimi vlm gemini    # VLM policy (Gemini) + Kimi reflection
#   ./run_single.sh memory 50 rule vlm qwen     # VLM policy (Qwen) + rule reflection
# =============================================================================

set -e

# =============================================================================
# Environment Setup (CRITICAL for headless rendering)
# =============================================================================

# MuJoCo EGL rendering (for headless servers without display)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Fix for LLVM command-line option conflict
export TRITON_PTXAS_PATH=""

# CUDA setup
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
# export BNB_CUDA_VERSION=117

# Performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

# HuggingFace settings
# export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# export HF_HOME="${HF_HOME:-/share/project/hf_cache}"
# export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
# mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" 2>/dev/null || true
# export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Python settings
export PYTHONUNBUFFERED=1
export WANDB_SILENT=true

# =============================================================================
# Parse Arguments
# =============================================================================
MODE=${1:-memory}
N_EPISODES=${2:-50}
VLM_PROVIDER=${3:-rule}
POLICY_TYPE=${4:-bc}          # "bc" (LLaVA) or "vlm" (external VLM)
POLICY_PROVIDER=${5:-}        # For VLM policy: openai, gemini, qwen, kimi

if [[ "$MODE" != "baseline" && "$MODE" != "memory" ]]; then
    echo "Usage: $0 <baseline|memory> [n_episodes] [reflection_provider] [policy_type] [policy_provider]"
    echo ""
    echo "Examples (BC policy - uses LLaVA for actions):"
    echo "  $0 baseline 50             # Baseline for 50 episodes"
    echo "  $0 memory 100 kimi         # Memory with Kimi reflection"
    echo "  $0 memory 100 rule         # Memory with rule-based reflection"
    echo ""
    echo "Examples (VLM policy - uses external VLM for actions):"
    echo "  $0 memory 50 kimi vlm openai   # GPT-4V policy + Kimi reflection"
    echo "  $0 memory 50 kimi vlm gemini   # Gemini policy + Kimi reflection"
    echo "  $0 memory 50 rule vlm qwen     # Qwen policy + rule reflection"
    exit 1
fi

# =============================================================================
# Setup
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Model paths
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-./ReflectVLM-llava-v1.5-13b-base}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="logs/neuro_memory_${MODE}_${TIMESTAMP}"

echo "============================================================"
echo "NEURO-SYMBOLIC MEMORY EXPERIMENT"
echo "============================================================"
echo "Mode:           $MODE"
echo "Episodes:       $N_EPISODES"
echo "Policy:         $POLICY_TYPE $(if [[ -n "$POLICY_PROVIDER" ]]; then echo "($POLICY_PROVIDER)"; else echo "(LLaVA)"; fi)"
echo "Reflection:     $VLM_PROVIDER"
echo "MUJOCO_GL:      $MUJOCO_GL"
echo "GPU:            $CUDA_VISIBLE_DEVICES"
echo "Save Dir:       $SAVE_DIR"
echo "============================================================"

# Function to check API key
check_api_key() {
    local provider=$1
    case "$provider" in
        kimi)
            if [[ -z "$MOONSHOT_API_KEY" ]]; then
                echo "ERROR: MOONSHOT_API_KEY not set for $provider"
                echo "Run: export MOONSHOT_API_KEY='your_key'"
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

# Check API key for reflection VLM
if [[ "$MODE" == "memory" && "$VLM_PROVIDER" != "rule" ]]; then
    if ! check_api_key "$VLM_PROVIDER"; then
        exit 1
    fi
    echo "Reflection API Key ($VLM_PROVIDER): Set ✓"
fi

# Check API key for policy VLM
if [[ "$POLICY_TYPE" == "vlm" ]]; then
    if [[ -z "$POLICY_PROVIDER" ]]; then
        echo "ERROR: --policy_provider required when policy_type=vlm"
        exit 1
    fi
    if ! check_api_key "$POLICY_PROVIDER"; then
        exit 1
    fi
    echo "Policy API Key ($POLICY_PROVIDER): Set ✓"
fi

# =============================================================================
# Run Experiment
# =============================================================================
mkdir -p "$SAVE_DIR"

CMD="python experiments/neuro_memory/run_memory_experiment.py \
    --mode $MODE \
    --n_episodes $N_EPISODES \
    --save_dir $SAVE_DIR \
    --name exp \
    --base_model $BASE_MODEL_PATH \
    --verbose \
    --show_memory \
    --memory_interval 5"

# Add reflection provider for memory mode
if [[ "$MODE" == "memory" ]]; then
    CMD="$CMD --provider $VLM_PROVIDER"
fi

# Add policy type
CMD="$CMD --policy_type $POLICY_TYPE"

# Add policy provider for VLM policy
if [[ "$POLICY_TYPE" == "vlm" && -n "$POLICY_PROVIDER" ]]; then
    CMD="$CMD --policy_provider $POLICY_PROVIDER"
fi

echo ""
echo "Running: $CMD"
echo ""

$CMD 2>&1 | tee "$SAVE_DIR/run.log"

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Results saved to: $SAVE_DIR"
echo "============================================================"
