#!/usr/bin/env python3
"""
Test script for VLM-based Reflector.

Tests real VLM APIs (OpenAI, Gemini, Qwen) for principle extraction.

Usage:
    # Test all available providers
    python scripts/test_vlm_reflector.py

    # Test specific provider
    python scripts/test_vlm_reflector.py openai
    python scripts/test_vlm_reflector.py gemini
    python scripts/test_vlm_reflector.py qwen

    # Test with specific model
    python scripts/test_vlm_reflector.py openai gpt-4o

Environment Variables:
    OPENAI_API_KEY - for OpenAI GPT models
    GOOGLE_API_KEY - for Google Gemini models
    DASHSCOPE_API_KEY or QWEN_API_KEY - for Alibaba Qwen models
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
WORLDMEMORY_ROOT = PROJECT_ROOT.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(WORLDMEMORY_ROOT))

# Import modules
from roboworld.agent.reflector import (
    Reflector,
    ReflectionInput,
    ReflectionOutput,
    FAIL_TAG_DESCRIPTIONS,
)
from roboworld.agent.vlm_api import (
    UnifiedVLM,
    create_vlm,
    get_available_models,
)
from romemo.memory.principle import Principle, PrincipleStore


def check_api_keys():
    """Check which API keys are available."""
    keys = {
        "openai": os.environ.get("OPENAI_API_KEY"),
        "gemini": os.environ.get("GOOGLE_API_KEY"),
        "qwen": os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY"),
    }
    available = {k: bool(v) for k, v in keys.items()}
    return available


def create_test_input(fail_tag: str = "BLOCKED_BY_PREDECESSOR") -> ReflectionInput:
    """Create a test ReflectionInput."""
    test_cases = {
        "BLOCKED_BY_PREDECESSOR": ReflectionInput(
            failed_action="insert red",
            fail_tag="BLOCKED_BY_PREDECESSOR",
            oracle_action="insert blue",
            symbolic_state={
                "action_type": "insert",
                "holding_piece": "red",
                "inserted_pieces": [],
                "remaining_pieces": ["red", "blue", "green", "yellow"],
                "progress": 0.0,
            },
            action_history=["pick up red"],
            experience_id="test_001",
        ),
        "NEEDS_REORIENT": ReflectionInput(
            failed_action="insert green",
            fail_tag="NEEDS_REORIENT",
            oracle_action="reorient green",
            symbolic_state={
                "action_type": "insert",
                "holding_piece": "green",
                "inserted_pieces": ["blue"],
                "remaining_pieces": ["green", "red"],
                "progress": 0.25,
            },
            action_history=["pick up blue", "insert blue", "pick up green"],
            experience_id="test_002",
        ),
        "HAND_FULL": ReflectionInput(
            failed_action="pick up yellow",
            fail_tag="HAND_FULL",
            oracle_action="insert red",
            symbolic_state={
                "action_type": "pick",
                "holding_piece": "red",
                "inserted_pieces": ["blue", "green"],
                "remaining_pieces": ["red", "yellow"],
                "progress": 0.5,
            },
            action_history=[
                "pick up blue",
                "insert blue",
                "pick up green",
                "insert green",
                "pick up red",
            ],
            experience_id="test_003",
        ),
    }
    return test_cases.get(fail_tag, test_cases["BLOCKED_BY_PREDECESSOR"])


def test_vlm_api_basic(provider: str, model: str = None):
    """Test basic VLM API connectivity."""
    print(f"\n{'=' * 60}")
    print(f"Testing {provider.upper()} VLM API")
    print(f"{'=' * 60}")

    try:
        vlm = create_vlm(provider=provider, model=model)
        print(f"✅ Created VLM: {vlm}")

        # Simple text test
        response = vlm.generate(
            "What is 2 + 2? Answer with just the number.",
            max_tokens=10,
            temperature=0.0,
        )
        print(f"✅ Text generation: '{response}'")

        return vlm

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_reflector_with_vlm(provider: str, model: str = None, verbose: bool = True):
    """Test Reflector with real VLM backend."""
    print(f"\n{'=' * 60}")
    print(f"Testing Reflector with {provider.upper()}")
    print(f"{'=' * 60}")

    try:
        # Create reflector
        reflector = Reflector.create(
            provider=provider,
            model=model,
            mode="oracle_guided",
            verbose=verbose,
        )
        print(f"✅ Created Reflector with {provider}")

        # Test with different failure types
        test_cases = ["BLOCKED_BY_PREDECESSOR", "NEEDS_REORIENT", "HAND_FULL"]
        results = []

        for fail_tag in test_cases:
            print(f"\n--- Testing {fail_tag} ---")
            input_data = create_test_input(fail_tag)

            print(f"  Failed: {input_data.failed_action}")
            print(f"  Oracle: {input_data.oracle_action}")

            output = reflector.reflect(input_data)

            print(f"\n  📝 Root Cause: {output.root_cause[:100]}...")
            print(f"  📋 Principle: {output.general_principle[:100]}...")
            print(f"  🏷️  Action Types: {output.action_types}")
            print(f"  🎯 Addresses: {output.addresses_fail_tags}")

            results.append(
                {
                    "fail_tag": fail_tag,
                    "success": bool(output.general_principle),
                    "principle": output.general_principle,
                    "root_cause": output.root_cause,
                }
            )

        # Print stats
        stats = reflector.get_stats()
        print(f"\n📊 Reflector Stats:")
        print(f"   VLM calls: {stats['vlm_calls']}")
        print(f"   Fallback calls: {stats['fallback_calls']}")
        print(f"   Cache size: {stats['cache_size']}")

        # Summary
        success_count = sum(1 for r in results if r["success"])
        print(f"\n✅ {success_count}/{len(results)} tests passed")

        return results

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_principle_integration(provider: str, model: str = None):
    """Test full integration: Reflector -> PrincipleStore."""
    print(f"\n{'=' * 60}")
    print(f"Testing Full Integration with {provider.upper()}")
    print(f"{'=' * 60}")

    try:
        # Create components
        reflector = Reflector.create(provider=provider, model=model, verbose=True)
        store = PrincipleStore(name=f"test_{provider}")

        print("✅ Created Reflector and PrincipleStore")

        # Simulate multiple failures
        failures = [
            ("BLOCKED_BY_PREDECESSOR", "insert red", "insert blue"),
            ("BLOCKED_BY_PREDECESSOR", "insert green", "insert yellow"),  # Same type
            ("NEEDS_REORIENT", "insert pink", "reorient pink"),
        ]

        for fail_tag, failed_action, oracle_action in failures:
            input_data = ReflectionInput(
                failed_action=failed_action,
                fail_tag=fail_tag,
                oracle_action=oracle_action,
                symbolic_state={
                    "action_type": "insert" if "insert" in failed_action else "pick",
                    "holding_piece": failed_action.split()[-1],
                    "inserted_pieces": [],
                    "remaining_pieces": [failed_action.split()[-1]],
                    "progress": 0.0,
                },
                action_history=[],
                experience_id=f"int_{fail_tag}_{failed_action.replace(' ', '_')}",
            )

            print(f"\n  Processing: {failed_action} -> {fail_tag}")

            # Generate reflection
            output = reflector.reflect(input_data)
            print(f"    Principle: {output.general_principle[:60]}...")

            # Update principle store
            pid = store.update_from_reflection(output.to_dict(), input_data.experience_id)
            print(f"    Updated principle: {pid}")

        # Check store state
        print(f"\n📊 Final Store State:")
        stats = store.get_stats()
        print(f"   Total principles: {stats['total']}")
        print(f"   By action type: {stats.get('by_action_type', {})}")
        print(f"   By fail tag: {stats.get('by_fail_tag', {})}")

        # Test retrieval
        print(f"\n🔍 Retrieval Test:")
        insert_principles = store.retrieve(action_type="insert")
        print(f"   Principles for 'insert': {len(insert_principles)}")
        for p in insert_principles[:3]:
            print(f"     - [{p.confidence:.2f}] {p.content[:50]}...")

        print(f"\n✅ Integration test passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def compare_providers():
    """Compare reflection quality across providers."""
    print(f"\n{'=' * 60}")
    print("Comparing VLM Providers")
    print(f"{'=' * 60}")

    available = check_api_keys()
    active_providers = [p for p, available in available.items() if available]

    if not active_providers:
        print("❌ No API keys available. Set at least one of:")
        print("   OPENAI_API_KEY, GOOGLE_API_KEY, DASHSCOPE_API_KEY")
        return

    print(f"Available providers: {active_providers}")

    # Same test case for all providers
    input_data = create_test_input("BLOCKED_BY_PREDECESSOR")
    results = {}

    for provider in active_providers:
        print(f"\n--- {provider.upper()} ---")
        try:
            reflector = Reflector.create(provider=provider, verbose=False)
            output = reflector.reflect(input_data)

            results[provider] = {
                "principle": output.general_principle,
                "root_cause": output.root_cause,
                "action_types": output.action_types,
            }

            print(f"  Principle: {output.general_principle[:80]}...")

        except Exception as e:
            results[provider] = {"error": str(e)}
            print(f"  Error: {e}")

    # Summary comparison
    print(f"\n{'=' * 60}")
    print("Comparison Summary")
    print(f"{'=' * 60}")

    for provider, result in results.items():
        if "error" in result:
            print(f"\n{provider.upper()}: ❌ Error")
        else:
            print(f"\n{provider.upper()}:")
            print(f"  {result['principle']}")


def main():
    """Main test function."""
    print("=" * 70)
    print("VLM Reflector Test Suite")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")

    # Check API keys
    print("\n📋 API Key Status:")
    available = check_api_keys()
    for provider, has_key in available.items():
        status = "✅ Available" if has_key else "❌ Not set"
        print(f"   {provider.upper()}: {status}")

    # Parse command line
    provider = sys.argv[1] if len(sys.argv) > 1 else None
    model = sys.argv[2] if len(sys.argv) > 2 else None

    if provider:
        # Test specific provider
        if not available.get(provider):
            print(f"\n❌ API key not set for {provider}")
            print(f"   Set {provider.upper()}_API_KEY environment variable")
            sys.exit(1)

        test_vlm_api_basic(provider, model)
        test_reflector_with_vlm(provider, model)
        test_principle_integration(provider, model)

    else:
        # Test all available providers
        active_providers = [p for p, has_key in available.items() if has_key]

        if not active_providers:
            print("\n⚠️  No API keys available. Testing rule-based fallback only.")

            # Test rule-based
            print("\n--- Rule-Based Fallback ---")
            reflector = Reflector(verbose=True)
            input_data = create_test_input("BLOCKED_BY_PREDECESSOR")
            output = reflector.reflect(input_data)
            print(f"  Principle: {output.general_principle}")
            print("\n✅ Rule-based fallback works!")

        else:
            # Test each available provider
            for provider in active_providers:
                test_vlm_api_basic(provider)
                test_reflector_with_vlm(provider)

            # Compare all
            if len(active_providers) > 1:
                compare_providers()

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
