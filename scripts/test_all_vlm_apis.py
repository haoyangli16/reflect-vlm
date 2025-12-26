#!/usr/bin/env python3
"""
Test script to verify all VLM APIs defined in roboworld/agent/vlm_api.py.
This script iterates through all supported providers and models, checking for:
1. API Key presence
2. Text-only generation
3. Vision (Image) generation capabilities

Usage:
    python scripts/test_all_vlm_apis.py
"""

import os
import sys
import numpy as np
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PIL import Image
except ImportError:
    print("Pillow (PIL) not found. Please install it: pip install Pillow")
    sys.exit(1)

from roboworld.agent.vlm_api import UnifiedVLM, get_available_models

# ANSI Colors for nicer output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def create_dummy_image():
    """Create a simple 100x100 red square image for testing vision capabilities."""
    img_data = np.zeros((100, 100, 3), dtype=np.uint8)
    img_data[:] = [255, 0, 0]  # Red
    return Image.fromarray(img_data)


def check_api_keys(provider):
    """Check if appropriate API keys are set in environment variables."""
    api_key_map = {
        "openai": ["OPENAI_API_KEY"],
        "gpt": ["OPENAI_API_KEY"],
        "gemini": ["GOOGLE_API_KEY"],
        "google": ["GOOGLE_API_KEY"],
        "qwen": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
        "alibaba": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
        "dashscope": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
        "kimi": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
        "moonshot": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
    }

    required_keys = api_key_map.get(provider, [])
    # Return True if provider not in map (assuming no key needed or custom), else check keys
    if not required_keys:
        return True, []

    found = [k for k in required_keys if os.environ.get(k)]
    return len(found) > 0, required_keys


def test_model(provider, model_name):
    print(f"\n{'-' * 60}")
    print(f"Testing {GREEN}{provider}{RESET} model: {GREEN}{model_name}{RESET}")

    # 1. Check API Key
    has_key, required_keys = check_api_keys(provider)
    if not has_key:
        print(f"{YELLOW}⚠️  SKIPPED: Missing API Key.{RESET}")
        print(
            f"   Please set one of the following environment variables: {', '.join(required_keys)}"
        )
        return "SKIPPED_NO_KEY"

    vlm = None
    try:
        # 2. Instantiation
        print("   Initializing...", end=" ", flush=True)
        vlm = UnifiedVLM(provider=provider, model=model_name)
        print(f"{GREEN}OK{RESET}")

        # 3. Text Generation Test
        print("   Testing Text Generation...", end=" ", flush=True)
        text_prompt = "Reply with exactly the word 'Pong'."
        response_text = vlm.generate(text_prompt, max_tokens=10)

        if response_text and "Pong" in response_text:
            print(f"{GREEN}OK{RESET} (Response: {response_text.strip()})")
        else:
            print(f"{RED}FAILED{RESET}")
            print(f"     Received: '{response_text}'")
            return "FAILED_TEXT"

        # 4. Vision Generation Test
        print("   Testing Vision Generation...", end=" ", flush=True)
        img = create_dummy_image()
        # Prompt asking for color to verify image was actually processed
        vision_prompt = "What represents the dominant color in this image? Reply with just the color name (e.g. Red, Blue)."
        response_vision = vlm.generate(vision_prompt, images=[img], max_tokens=20)

        lower_response = response_vision.lower() if response_vision else ""
        if "red" in lower_response:
            print(f"{GREEN}OK{RESET} (Response: {response_vision.strip()})")
        else:
            print(f"{RED}FAILED{RESET}")
            print(f"     Received: '{response_vision}'")
            # Note: We don't fail the whole test if vision fails but text works,
            # as some models might be text-only or have specific vision issues,
            # but we flag it.
            return "FAILED_VISION"

        return "PASSED"

    except Exception as e:
        print(f"{RED}ERROR{RESET}")
        print(f"   Exception: {str(e)}")
        # print(traceback.format_exc()) # Uncomment for full stack trace
        return "ERROR"


def main():
    print(f"{'=' * 60}")
    print("VLM API Comprehensive Test Suite")
    print(f"{'=' * 60}")

    models_map = get_available_models()

    results = {"PASSED": 0, "FAILED_TEXT": 0, "FAILED_VISION": 0, "ERROR": 0, "SKIPPED_NO_KEY": 0}

    total_models = sum(len(models) for models in models_map.values())
    print(f"Found {len(models_map)} providers with {total_models} total models to test.\n")

    for provider, models in models_map.items():
        print(f"[{provider.upper()}] Found {len(models)} models.")
        for model in models:
            status = test_model(provider, model)
            results[status] = results.get(status, 0) + 1

    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    print(f"Total Models Checked: {total_models}")
    print(f"{GREEN}Passed:             {results['PASSED']}{RESET}")
    print(f"{RED}Failed (Text):      {results['FAILED_TEXT']}{RESET}")
    print(f"{RED}Failed (Vision):    {results['FAILED_VISION']}{RESET}")
    print(f"{RED}Errors:             {results['ERROR']}{RESET}")
    print(f"{YELLOW}Skipped (No Key):   {results['SKIPPED_NO_KEY']}{RESET}")

    if results["PASSED"] == 0 and results["SKIPPED_NO_KEY"] > 0:
        print(
            f"\n{YELLOW}Hint: You need to set environment variables (OPENAI_API_KEY, GOOGLE_API_KEY, etc.) to run these tests.{RESET}"
        )


if __name__ == "__main__":
    main()
