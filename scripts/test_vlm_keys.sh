#!/bin/bash

# ============================================================================
# VLM API Key Testing Script
# ============================================================================
# Instructions:
# 1. Fill in your API keys in the exports below.
# 2. Run this script from the project root: ./scripts/test_vlm_keys.sh
# ============================================================================

# --- OpenAI (GPT-4o, GPT-5.1) ---
# export OPENAI_API_KEY="your_openai_key_here"

# # --- Google Gemini (Gemini 1.5 Pro, Gemini 2.0 Flash) ---
# export GOOGLE_API_KEY="your_google_key_here"

# # --- Alibaba Qwen (DashScope) ---
# export DASHSCOPE_API_KEY="your_dashscope_key_here"

# # --- Moonshot AI (Kimi) ---
# export MOONSHOT_API_KEY="your_moonshot_key_here"


# ============================================================================
# Test Execution Logic
# ============================================================================

# Ensure the python path includes the current directory so modules resolve correctly
export PYTHONPATH=$PYTHONPATH:.

echo "============================================================================"
echo "Starting VLM API Connectivity Tests"
echo "============================================================================"
echo "Date: $(date)"
echo "Working Directory: $(pwd)"
echo "----------------------------------------------------------------"

# Helper function to run the python test
run_test() {
    local provider=$1
    echo ">> Testing Provider: $provider"
    
    # Run the test_vlm function inside roboworld/agent/vlm_api.py
    # We pass the provider name as the first argument
    python3 roboworld/agent/vlm_api.py "$provider"
    
    echo "----------------------------------------------------------------"
}

# Run tests for each provider
# The python script itself handles missing keys by catching the error and printing it.

# Uncomment to test OpenAI (requires OPENAI_API_KEY)
# run_test "openai"
# run_test "gemini"
# run_test "qwen"
# run_test "kimi"
run_test "huggingface"

# Uncomment to test HuggingFace (requires HF_TOKEN)
# run_test "huggingface"

echo "All tests completed."
