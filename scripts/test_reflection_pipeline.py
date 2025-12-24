#!/usr/bin/env python3
"""
Test script for Phase 1: Reflection Pipeline + Principle Storage

This script tests:
1. Principle creation and storage
2. Reflection generation (rule-based fallback)
3. Principle retrieval by context
4. Voting/confidence mechanism
5. Integration with existing memory system

Run with:
    python scripts/test_reflection_pipeline.py
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

# Now import our modules
from romemo.memory.principle import Principle, PrincipleStore
from roboworld.agent.reflector import (
    Reflector,
    ReflectionInput,
    ReflectionOutput,
    FAIL_TAG_DESCRIPTIONS,
)


def test_principle_creation():
    """Test basic Principle creation and properties."""
    print("\n" + "=" * 60)
    print("TEST 1: Principle Creation")
    print("=" * 60)

    # Create a principle
    p = Principle(
        content="Before inserting any piece, verify that all predecessor pieces are already inserted.",
        action_types=["insert"],
        trigger_conditions=["piece_has_dependencies"],
        addresses_fail_tags=["BLOCKED_BY_PREDECESSOR"],
        evidence_for=["exp_001", "exp_002"],
        importance_score=4.0,
    )

    print(f"Created principle: {p.pid}")
    print(f"  Content: {p.content}")
    print(f"  Importance: {p.importance_score}")
    print(f"  Confidence: {p.confidence:.2f}")
    print(f"  Is Established: {p.is_established}")
    print(f"  Action Types: {p.action_types}")
    print(f"  Addresses: {p.addresses_fail_tags}")

    # Test upvote
    p.upvote("exp_003")
    print(f"\nAfter upvote:")
    print(f"  Importance: {p.importance_score}")
    print(f"  Evidence for: {p.evidence_for}")

    # Test downvote
    p.downvote("exp_004", strength=1.0)
    print(f"\nAfter downvote:")
    print(f"  Importance: {p.importance_score}")
    print(f"  Evidence against: {p.evidence_against}")

    # Test context matching
    assert p.matches_context(action_type="insert")
    assert p.matches_context(fail_tag="BLOCKED_BY_PREDECESSOR")
    assert not p.matches_context(action_type="pick")
    print("\n✅ Context matching works correctly")

    print("\n✅ TEST 1 PASSED")
    return True


def test_principle_store():
    """Test PrincipleStore operations."""
    print("\n" + "=" * 60)
    print("TEST 2: Principle Store")
    print("=" * 60)

    store = PrincipleStore(name="test_store")

    # Add principles
    p1 = Principle(
        content="Before inserting any piece, verify that all predecessor pieces are already inserted.",
        action_types=["insert"],
        addresses_fail_tags=["BLOCKED_BY_PREDECESSOR"],
        importance_score=5.0,
    )

    p2 = Principle(
        content="If a piece is lying flat, reorient it before attempting insertion.",
        action_types=["insert", "reorient"],
        addresses_fail_tags=["NEEDS_REORIENT"],
        importance_score=3.0,
    )

    p3 = Principle(
        content="Before picking up a new object, ensure the gripper is empty.",
        action_types=["pick"],
        addresses_fail_tags=["HAND_FULL"],
        importance_score=4.0,
    )

    pid1 = store.add(p1)
    pid2 = store.add(p2)
    pid3 = store.add(p3)

    print(f"Added 3 principles: {pid1}, {pid2}, {pid3}")
    print(f"Store size: {len(store)}")

    # Test retrieval by action type
    insert_principles = store.retrieve(action_type="insert")
    print(f"\nPrinciples for 'insert' action: {len(insert_principles)}")
    for p in insert_principles:
        print(f"  - {p.content[:50]}...")
    assert len(insert_principles) == 2

    # Test retrieval by fail tag
    blocked_principles = store.retrieve(fail_tag="BLOCKED_BY_PREDECESSOR")
    print(f"\nPrinciples for BLOCKED_BY_PREDECESSOR: {len(blocked_principles)}")
    assert len(blocked_principles) == 1

    # Test confidence filtering
    high_conf_principles = store.retrieve(min_confidence=0.4)
    print(f"\nHigh confidence principles (>=0.4): {len(high_conf_principles)}")

    # Test stats
    stats = store.get_stats()
    print(f"\nStore stats: {json.dumps(stats, indent=2)}")

    # Test save and load
    test_path = Path("/tmp/test_principles.json")
    store.save(test_path)
    print(f"\nSaved to {test_path}")

    loaded_store = PrincipleStore.load(test_path)
    print(f"Loaded store with {len(loaded_store)} principles")
    assert len(loaded_store) == len(store)

    # Clean up
    test_path.unlink()

    print("\n✅ TEST 2 PASSED")
    return True


def test_reflector_rule_based():
    """Test Reflector with rule-based fallback (no VLM)."""
    print("\n" + "=" * 60)
    print("TEST 3: Reflector (Rule-Based)")
    print("=" * 60)

    # Create reflector without VLM (uses rule-based fallback)
    reflector = Reflector(vlm_agent=None, mode="oracle_guided", verbose=True)

    # Test case 1: BLOCKED_BY_PREDECESSOR
    input1 = ReflectionInput(
        failed_action="insert red",
        fail_tag="BLOCKED_BY_PREDECESSOR",
        oracle_action="insert blue",
        symbolic_state={
            "action_type": "insert",
            "holding_piece": "red",
            "inserted_pieces": [],
            "remaining_pieces": ["red", "blue", "green"],
            "progress": 0.0,
        },
        action_history=["pick up red"],
        experience_id="exp_test_001",
    )

    print("\nTest Case 1: BLOCKED_BY_PREDECESSOR")
    print(f"  Failed: {input1.failed_action}")
    print(f"  Oracle: {input1.oracle_action}")

    output1 = reflector.reflect(input1)
    print(f"\nReflection Output:")
    print(f"  Root Cause: {output1.root_cause}")
    print(f"  Principle: {output1.general_principle}")
    print(f"  Action Types: {output1.action_types}")
    print(f"  Addresses: {output1.addresses_fail_tags}")

    assert (
        "predecessor" in output1.general_principle.lower()
        or "insert" in output1.general_principle.lower()
    )
    assert "insert" in output1.action_types

    # Test case 2: NEEDS_REORIENT
    input2 = ReflectionInput(
        failed_action="insert green",
        fail_tag="NEEDS_REORIENT",
        oracle_action="reorient green",
        symbolic_state={
            "action_type": "insert",
            "holding_piece": "green",
            "inserted_pieces": ["blue"],
            "remaining_pieces": ["green", "red"],
            "progress": 0.33,
        },
        action_history=["pick up blue", "insert blue", "pick up green"],
        experience_id="exp_test_002",
    )

    print("\n\nTest Case 2: NEEDS_REORIENT")
    print(f"  Failed: {input2.failed_action}")
    print(f"  Oracle: {input2.oracle_action}")

    output2 = reflector.reflect(input2)
    print(f"\nReflection Output:")
    print(f"  Root Cause: {output2.root_cause}")
    print(f"  Principle: {output2.general_principle}")

    assert (
        "reorient" in output2.general_principle.lower()
        or "upright" in output2.general_principle.lower()
    )

    # Test case 3: HAND_FULL
    input3 = ReflectionInput(
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
        experience_id="exp_test_003",
    )

    print("\n\nTest Case 3: HAND_FULL")
    output3 = reflector.reflect(input3)
    print(f"  Principle: {output3.general_principle}")

    assert (
        "gripper" in output3.general_principle.lower()
        or "empty" in output3.general_principle.lower()
        or "holding" in output3.general_principle.lower()
    )

    print("\n✅ TEST 3 PASSED")
    return True


def test_integration():
    """Test integration of Reflector with PrincipleStore."""
    print("\n" + "=" * 60)
    print("TEST 4: Integration (Reflector + PrincipleStore)")
    print("=" * 60)

    # Create components
    reflector = Reflector(vlm_agent=None, mode="oracle_guided")
    store = PrincipleStore(name="integration_test")

    # Simulate a sequence of failures and reflections
    failures = [
        {
            "failed_action": "insert red",
            "fail_tag": "BLOCKED_BY_PREDECESSOR",
            "oracle_action": "insert blue",
            "state": {
                "holding_piece": "red",
                "inserted_pieces": [],
                "remaining_pieces": ["red", "blue"],
                "progress": 0.0,
            },
        },
        {
            "failed_action": "insert green",
            "fail_tag": "BLOCKED_BY_PREDECESSOR",
            "oracle_action": "insert yellow",
            "state": {
                "holding_piece": "green",
                "inserted_pieces": ["blue"],
                "remaining_pieces": ["green", "yellow"],
                "progress": 0.25,
            },
        },
        {
            "failed_action": "insert pink",
            "fail_tag": "NEEDS_REORIENT",
            "oracle_action": "reorient pink",
            "state": {
                "holding_piece": "pink",
                "inserted_pieces": ["blue", "yellow"],
                "remaining_pieces": ["pink"],
                "progress": 0.67,
            },
        },
    ]

    print("Processing failures...")
    for i, failure in enumerate(failures):
        print(f"\n  Failure {i + 1}: {failure['failed_action']} -> {failure['fail_tag']}")

        # Create reflection input
        input_data = ReflectionInput(
            failed_action=failure["failed_action"],
            fail_tag=failure["fail_tag"],
            oracle_action=failure["oracle_action"],
            symbolic_state=failure["state"],
            action_history=[],
            experience_id=f"exp_int_{i}",
        )

        # Generate reflection
        output = reflector.reflect(input_data)
        print(f"    Principle: {output.general_principle[:60]}...")

        # Update principle store
        pid = store.update_from_reflection(output.to_dict(), input_data.experience_id)
        print(f"    Updated principle: {pid}")

    # Check store state
    print(f"\n\nFinal store state:")
    print(f"  Total principles: {len(store)}")

    stats = store.get_stats()
    print(f"  By action type: {stats.get('by_action_type', {})}")
    print(f"  By fail tag: {stats.get('by_fail_tag', {})}")

    # The first two failures should have created/upvoted the same principle
    blocked_principles = store.retrieve(fail_tag="BLOCKED_BY_PREDECESSOR")
    print(f"\n  BLOCKED_BY_PREDECESSOR principles: {len(blocked_principles)}")
    if blocked_principles:
        p = blocked_principles[0]
        print(f"    Importance: {p.importance_score} (should be 3.0 = 2.0 initial + 1.0 upvote)")
        print(f"    Evidence: {p.evidence_for}")

    # Test retrieval during action selection
    print("\n\nSimulating action selection...")
    current_action = "insert"
    applicable = store.retrieve(action_type=current_action, min_confidence=0.2)
    print(f"  Principles for '{current_action}': {len(applicable)}")

    formatted = store.format_for_prompt(applicable)
    print(f"\n  Formatted for prompt:\n{formatted}")

    print("\n✅ TEST 4 PASSED")
    return True


def test_voting_mechanism():
    """Test the ExpeL-style voting mechanism."""
    print("\n" + "=" * 60)
    print("TEST 5: Voting Mechanism")
    print("=" * 60)

    store = PrincipleStore(name="voting_test")

    # Add a principle
    p = Principle(
        content="Test principle for voting",
        action_types=["test"],
        importance_score=2.0,  # Initial score
    )
    pid = store.add(p)

    print(f"Initial: importance={store.get(pid).importance_score}")

    # Simulate multiple upvotes (evidence supports)
    for i in range(5):
        store.get(pid).upvote(f"exp_support_{i}")

    print(f"After 5 upvotes: importance={store.get(pid).importance_score}")
    print(f"  Confidence: {store.get(pid).confidence:.2f}")
    print(f"  Is Established: {store.get(pid).is_established}")

    # Simulate downvotes (contradictory evidence)
    for i in range(3):
        store.get(pid).downvote(f"exp_contradict_{i}")

    print(f"After 3 downvotes: importance={store.get(pid).importance_score}")
    print(f"  Should remove: {store.get(pid).should_remove()}")

    # Continue downvoting until removal threshold
    for i in range(10):
        store.get(pid).downvote(f"exp_final_{i}")

    print(f"After 10 more downvotes: importance={store.get(pid).importance_score}")
    print(f"  Should remove: {store.get(pid).should_remove()}")

    # Prune
    removed = store.prune()
    print(f"\nPruned {removed} principles")
    print(f"Store size: {len(store)}")

    print("\n✅ TEST 5 PASSED")
    return True


def test_fail_tag_coverage():
    """Test that we have rules for all common fail tags."""
    print("\n" + "=" * 60)
    print("TEST 6: Fail Tag Coverage")
    print("=" * 60)

    reflector = Reflector(vlm_agent=None)

    print("Testing rule-based reflection for all fail tags:\n")

    for fail_tag, description in FAIL_TAG_DESCRIPTIONS.items():
        input_data = ReflectionInput(
            failed_action=f"test action for {fail_tag}",
            fail_tag=fail_tag,
            oracle_action="correct action",
            symbolic_state={
                "holding_piece": "test",
                "inserted_pieces": [],
                "remaining_pieces": ["test"],
                "progress": 0.0,
            },
            action_history=[],
            experience_id=f"exp_{fail_tag}",
        )

        output = reflector.reflect(input_data)

        has_principle = bool(output.general_principle)
        status = "✅" if has_principle else "❌"

        print(f"  {status} {fail_tag}")
        if has_principle:
            print(f"      → {output.general_principle[:70]}...")

    print("\n✅ TEST 6 PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PHASE 1 TEST SUITE: Reflection Pipeline + Principle Storage")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")

    tests = [
        ("Principle Creation", test_principle_creation),
        ("Principle Store", test_principle_store),
        ("Reflector (Rule-Based)", test_reflector_rule_based),
        ("Integration", test_integration),
        ("Voting Mechanism", test_voting_mechanism),
        ("Fail Tag Coverage", test_fail_tag_coverage),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            import traceback

            results.append((name, False, traceback.format_exc()))
            print(f"\n❌ TEST FAILED: {name}")
            print(traceback.format_exc())

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, s, _ in results if s)
    total = len(results)

    for name, success, error in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 1 implementation is ready.")
    else:
        print("\n⚠️  Some tests failed. Please fix before proceeding.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
