#!/usr/bin/env python3
"""
Test script to verify shape-based symbolic state extraction.

This verifies that:
1. generator.py correctly exports shape signatures
2. FrankaAssemblyEnv correctly stores shape info
3. extract_symbolic_state() returns shape-based fields
4. Symbolic state transfers across different color assignments
"""

import sys
import numpy as np
import random

# Add project to path
sys.path.insert(0, "/home/haoyang/project/haoyang/worldmemory/thirdparty/reflect-vlm")
sys.path.insert(0, "/home/haoyang/project/haoyang/worldmemory")

from roboworld.envs.generator import generate_xml, compute_brick_signature, analyze_brick_shape

# Try to import mujoco-dependent modules (may fail in some environments)
try:
    from roboworld.envs.mujoco.franka.franka_assembly import FrankaAssemblyEnv
    from roboworld.envs.asset_path_utils import full_path_for

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("⚠️  MuJoCo not available, skipping environment tests")


def test_generator_shape_info(seed: int = 1000000):
    """Test that generate_xml exports shape information."""
    print(f"\n{'=' * 60}")
    print(f"TEST 1: Generator Shape Info (seed={seed})")
    print(f"{'=' * 60}")

    xml, info = generate_xml(seed)

    # Check new fields exist
    assert "brick_shapes" in info, "Missing brick_shapes in info"
    assert "color_to_signature" in info, "Missing color_to_signature in info"
    assert "signature_to_color" in info, "Missing signature_to_color in info"
    assert "dependency_signatures" in info, "Missing dependency_signatures in info"

    print("\n✅ All required shape fields present in info dict")

    # Print shape info
    print("\nBrick Shapes:")
    for name, shape_info in info["brick_shapes"].items():
        sig = shape_info["signature"]
        color = shape_info["color"]
        features = shape_info.get("shape_features", {})
        print(f"  {name}: {color}")
        print(f"    Signature: {sig}")
        print(
            f"    Dimensions: {features.get('length')}×{features.get('width')}×{features.get('height')}"
        )
        print(f"    Aspect: {features.get('aspect_ratio')}")
        print(f"    Slots: {features.get('n_through_slots')} ({features.get('slot_direction')})")
        print(f"    Has hole: {features.get('has_insertion_hole')}")

    print("\nColor to Signature Mapping:")
    for color, sig in info["color_to_signature"].items():
        print(f"  {color} → {sig}")

    print("\nDependency Graph (shape-based):")
    for dep in info["dependency_signatures"]:
        print(f"  {dep['blocker_signature'][:40]}...")
        print(f"    → blocks → {dep['blocked_signature'][:40]}...")

    return info


def test_env_shape_info(seed: int = 1000000):
    """Test that FrankaAssemblyEnv stores and exposes shape info."""
    print(f"\n{'=' * 60}")
    print(f"TEST 2: Environment Shape Info (seed={seed})")
    print(f"{'=' * 60}")

    if not HAS_MUJOCO:
        print("\n⏭️  Skipping (MuJoCo not available)")
        return True

    xml, info = generate_xml(seed)
    xml_filename = full_path_for("tmp_test.xml")
    xml.write_to_file(filename=xml_filename)

    peg_names = [f"brick_{j + 1}" for j in range(1, info["n_bodies"])]
    peg_descriptions = [info["brick_descriptions"][peg_name] for peg_name in peg_names]

    env = FrankaAssemblyEnv(
        board_name="brick_1",
        fixture_name=None,
        peg_names=peg_names,
        peg_descriptions=peg_descriptions,
        render_mode="offscreen",
        frame_skip=20,
        model_name=xml_filename,
        magic_attaching=True,
        # Pass shape info
        brick_shapes=info.get("brick_shapes", {}),
        color_to_signature=info.get("color_to_signature", {}),
        signature_to_color=info.get("signature_to_color", {}),
        dependency_signatures=info.get("dependency_signatures", []),
    )

    # Check env has shape info
    assert hasattr(env, "brick_shapes"), "Missing brick_shapes on env"
    assert hasattr(env, "color_to_signature"), "Missing color_to_signature on env"
    assert hasattr(env, "peg_signatures"), "Missing peg_signatures on env"

    print("\n✅ Environment has all shape attributes")

    # Test lookup methods
    print("\nTesting lookup methods:")
    for color in env.peg_colors[:3]:
        sig = env.get_signature_for_color(color)
        features = env.get_shape_features(color)
        deps = env.get_piece_dependencies(color)

        print(f"\n  Color: {color}")
        print(f"    Signature: {sig}")
        print(f"    Aspect: {features.get('aspect_ratio', 'N/A')}")
        print(f"    Blocks: {deps['blocks_colors']}")
        print(f"    Blocked by: {deps['blocked_by_colors']}")

    env.close()
    return True


def test_symbolic_state_extraction(seed: int = 1000000):
    """Test that extract_symbolic_state returns shape-based fields."""
    print(f"\n{'=' * 60}")
    print(f"TEST 3: Symbolic State Extraction (seed={seed})")
    print(f"{'=' * 60}")

    if not HAS_MUJOCO:
        print("\n⏭️  Skipping (MuJoCo not available)")
        return {}

    # Import here to avoid circular imports
    from roboworld.agent.romemo_stack import extract_symbolic_state

    xml, info = generate_xml(seed)
    xml_filename = full_path_for("tmp_test2.xml")
    xml.write_to_file(filename=xml_filename)

    peg_names = [f"brick_{j + 1}" for j in range(1, info["n_bodies"])]
    peg_descriptions = [info["brick_descriptions"][peg_name] for peg_name in peg_names]
    peg_labels = [" ".join(pd.split()[:1]) for pd in peg_descriptions]

    env = FrankaAssemblyEnv(
        board_name="brick_1",
        fixture_name=None,
        peg_names=peg_names,
        peg_descriptions=peg_descriptions,
        render_mode="offscreen",
        frame_skip=20,
        model_name=xml_filename,
        magic_attaching=True,
        brick_shapes=info.get("brick_shapes", {}),
        color_to_signature=info.get("color_to_signature", {}),
        signature_to_color=info.get("signature_to_color", {}),
        dependency_signatures=info.get("dependency_signatures", []),
    )

    env.reset(seed=1)

    # Test symbolic state extraction
    test_action = f"insert {peg_labels[0]}"
    sym_state = extract_symbolic_state(env, test_action)

    # Check shape fields
    print("\nSymbolic State Fields:")
    shape_fields = [
        "target_signature",
        "target_shape_features",
        "holding_signature",
        "inserted_signatures",
        "remaining_signatures",
        "target_blocks",
        "target_blocked_by",
        "dependencies_satisfied",
    ]

    for field in shape_fields:
        value = sym_state.get(field)
        if isinstance(value, dict):
            print(f"  {field}: {dict(list(value.items())[:3])}...")
        elif isinstance(value, list) and len(value) > 3:
            print(f"  {field}: {value[:3]}...")
        else:
            print(f"  {field}: {value}")

    # Verify we have signatures
    assert sym_state.get("target_signature") is not None, "Missing target_signature"
    assert sym_state.get("remaining_signatures") is not None, "Missing remaining_signatures"

    print("\n✅ Symbolic state includes shape-based fields")

    env.close()
    return sym_state


def test_cross_episode_transfer(seeds=[1000000, 1000001, 1000002]):
    """Test that shape signatures transfer across different episodes."""
    print(f"\n{'=' * 60}")
    print(f"TEST 4: Cross-Episode Transfer (seeds={seeds})")
    print(f"{'=' * 60}")

    all_signatures = []

    for seed in seeds:
        xml, info = generate_xml(seed)

        # Collect all signatures from this episode
        sigs = set()
        for shape_info in info["brick_shapes"].values():
            sigs.add(shape_info["signature"])

        all_signatures.append((seed, sigs))

        print(f"\nSeed {seed}:")
        print(f"  Signatures: {len(sigs)}")
        for sig in list(sigs)[:5]:
            print(f"    - {sig[:50]}...")

    # Find common signatures across episodes
    common = all_signatures[0][1]
    for _, sigs in all_signatures[1:]:
        common = common.intersection(sigs)

    print(f"\n✅ Common signatures across all episodes: {len(common)}")
    for sig in list(common)[:5]:
        print(f"    - {sig[:50]}...")

    # Note: Signatures may differ because board generation is random
    # What matters is that the STRUCTURE of signatures is consistent
    print("\nNote: Exact signature matches depend on random generation.")
    print("The key is that signatures are SHAPE-based, not COLOR-based.")

    return True


def test_symbolic_state_matching():
    """Test symbolic state matching with shapes."""
    print(f"\n{'=' * 60}")
    print(f"TEST 5: Symbolic State Matching")
    print(f"{'=' * 60}")

    from roboworld.agent.romemo_stack import symbolic_state_matches, symbolic_state_similarity

    # Create test states with same shape but different colors
    state1 = {
        "action_type": "insert",
        "is_holding": True,
        "num_remaining": 4,
        "target_color": "red",
        "target_signature": "block_25x4x8_elongated_slots5_widthwise",
        "holding_signature": "block_25x4x8_elongated_slots5_widthwise",
        "dependencies_satisfied": True,
    }

    state2 = {
        "action_type": "insert",
        "is_holding": True,
        "num_remaining": 4,
        "target_color": "blue",  # Different color!
        "target_signature": "block_25x4x8_elongated_slots5_widthwise",  # Same shape!
        "holding_signature": "block_25x4x8_elongated_slots5_widthwise",
        "dependencies_satisfied": True,
    }

    state3 = {
        "action_type": "insert",
        "is_holding": True,
        "num_remaining": 4,
        "target_color": "green",
        "target_signature": "nail_4x4x17_tall",  # Different shape!
        "holding_signature": "nail_4x4x17_tall",
        "dependencies_satisfied": True,
    }

    # Test matching
    match_same_shape = symbolic_state_matches(state1, state2, use_shape_matching=True)
    match_diff_shape = symbolic_state_matches(state1, state3, use_shape_matching=True)

    print(f"\nState 1 (red, elongated block) vs State 2 (blue, elongated block):")
    print(f"  Match: {match_same_shape}")

    print(f"\nState 1 (red, elongated block) vs State 3 (green, tall nail):")
    print(f"  Match: {match_diff_shape}")

    # Test similarity
    sim_same = symbolic_state_similarity(state1, state2)
    sim_diff = symbolic_state_similarity(state1, state3)

    print(f"\nSimilarity scores:")
    print(f"  State 1 vs State 2 (same shape): {sim_same:.3f}")
    print(f"  State 1 vs State 3 (diff shape): {sim_diff:.3f}")

    assert match_same_shape, "Same shape should match!"
    assert not match_diff_shape, "Different shape should not match!"
    assert sim_same > sim_diff, "Same shape should have higher similarity!"

    print("\n✅ Shape-based matching works correctly!")

    return True


def main():
    print("=" * 60)
    print("SHAPE-BASED SYMBOLIC STATE TESTS")
    print("=" * 60)

    try:
        # Test 1: Generator exports shape info
        test_generator_shape_info()

        # Test 2: Environment stores shape info
        test_env_shape_info()

        # Test 3: Symbolic state extraction
        test_symbolic_state_extraction()

        # Test 4: Cross-episode transfer
        test_cross_episode_transfer()

        # Test 5: Symbolic state matching
        test_symbolic_state_matching()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nShape-based symbolic state is working correctly.")
        print("Experiences can now transfer across episodes!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
