#!/usr/bin/env python3
"""
Test script for Phase 2: Principle Integration into Action Selection

Tests:
1. Principle retrieval during action selection
2. Principle-based penalties and suggestions
3. Reflection on failure
4. End-to-end integration

Run with:
    python scripts/test_phase2_integration.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Add paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
WORLDMEMORY_ROOT = PROJECT_ROOT.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(WORLDMEMORY_ROOT))

# Import modules
import numpy as np
from romemo.memory.principle import Principle, PrincipleStore
from romemo.memory.schema import Experience, MemoryBank


def create_mock_env():
    """Create a mock environment for testing."""
    env = MagicMock()
    env.peg_colors = ["red", "blue", "green", "yellow"]
    env.peg_names = ["peg_red", "peg_blue", "peg_green", "peg_yellow"]
    env.is_success.return_value = False
    env.get_object_in_hand.return_value = None
    env.object_is_success.side_effect = lambda x: False
    env.get_env_state.return_value = {
        "joint": (np.zeros(9), np.zeros(9)),
        "mocap": (np.zeros(3), np.zeros(4)),
    }
    return env


def create_mock_agent():
    """Create a mock base agent for testing."""
    agent = MagicMock()
    agent.act.return_value = "insert red"
    agent.encode_image.return_value = np.random.randn(1024).astype(np.float32)
    return agent


def test_principle_retrieval():
    """Test that principles are retrieved during action selection."""
    print("\n" + "=" * 60)
    print("TEST 1: Principle Retrieval")
    print("=" * 60)

    # Create principle store with test principles
    store = PrincipleStore(name="test")

    p1 = Principle(
        content="Before inserting any piece, verify predecessors are inserted.",
        action_types=["insert"],
        addresses_fail_tags=["BLOCKED_BY_PREDECESSOR"],
        importance_score=5.0,
    )
    p2 = Principle(
        content="If a piece is lying flat, reorient before insertion.",
        action_types=["insert", "reorient"],
        addresses_fail_tags=["NEEDS_REORIENT"],
        importance_score=4.0,
    )
    p3 = Principle(
        content="Before picking up, ensure gripper is empty.",
        action_types=["pick"],
        addresses_fail_tags=["HAND_FULL"],
        importance_score=3.0,
    )

    store.add(p1)
    store.add(p2)
    store.add(p3)

    print(f"Created store with {len(store)} principles")

    # Test retrieval by action type
    insert_principles = store.retrieve(action_type="insert", min_confidence=0.2)
    print(f"\nPrinciples for 'insert': {len(insert_principles)}")
    for p in insert_principles:
        print(f"  - [{p.confidence:.2f}] {p.content[:50]}...")

    assert len(insert_principles) == 2, "Should retrieve 2 insert principles"

    # Test retrieval by fail tag
    blocked_principles = store.retrieve(fail_tag="BLOCKED_BY_PREDECESSOR")
    print(f"\nPrinciples for BLOCKED_BY_PREDECESSOR: {len(blocked_principles)}")
    assert len(blocked_principles) == 1

    print("\n✅ TEST 1 PASSED")
    return True


def test_principle_violation_check():
    """Test principle violation checking logic."""
    print("\n" + "=" * 60)
    print("TEST 2: Principle Violation Check")
    print("=" * 60)

    from roboworld.agent.romemo_stack import RoMemoDiscreteConfig, RoMemoDiscreteAgent

    # Create minimal agent for testing
    mock_env = create_mock_env()
    mock_base = create_mock_agent()

    cfg = RoMemoDiscreteConfig(
        use_principles=True,
        reflector_provider=None,  # Rule-based
    )

    # Create agent (will initialize principle system)
    agent = RoMemoDiscreteAgent(
        base_agent=mock_base,
        env=mock_env,
        cfg=cfg,
        writeback=False,
    )

    # Add test principles
    p1 = Principle(
        content="Before picking up, ensure gripper is empty.",
        action_types=["pick"],
        trigger_conditions=["gripper_occupied"],
        addresses_fail_tags=["HAND_FULL"],
        importance_score=5.0,
    )
    agent.principle_store.add(p1)

    # Test violation check
    symbolic_state = {
        "action_type": "pick",
        "is_holding": True,  # Gripper occupied
        "holding_piece": "red",
    }

    penalty, suggested = agent._check_principle_violations("pick up blue", [p1], symbolic_state)

    print(f"Proposed action: pick up blue")
    print(f"Symbolic state: holding=red")
    print(f"Penalty: {penalty}")
    print(f"Suggested: {suggested}")

    assert penalty > 0, "Should have penalty for violation"

    print("\n✅ TEST 2 PASSED")
    return True


def test_reflection_on_failure():
    """Test reflection generation on failure."""
    print("\n" + "=" * 60)
    print("TEST 3: Reflection on Failure")
    print("=" * 60)

    from roboworld.agent.romemo_stack import RoMemoDiscreteConfig, RoMemoDiscreteAgent
    from roboworld.agent.reflector import Reflector, ReflectionInput

    # Create reflector (rule-based)
    reflector = Reflector(mode="oracle_guided", verbose=True)

    # Create reflection input
    input_data = ReflectionInput(
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
        experience_id="test_001",
    )

    # Generate reflection
    output = reflector.reflect(input_data)

    print(f"Failed action: {input_data.failed_action}")
    print(f"Fail tag: {input_data.fail_tag}")
    print(f"Oracle action: {input_data.oracle_action}")
    print(f"\nReflection output:")
    print(f"  Root cause: {output.root_cause[:60]}...")
    print(f"  Principle: {output.general_principle[:60]}...")
    print(f"  Action types: {output.action_types}")

    assert output.general_principle, "Should extract principle"
    assert "insert" in output.action_types, "Should have insert action type"

    # Test updating principle store
    store = PrincipleStore(name="test")
    pid = store.update_from_reflection(output.to_dict(), input_data.experience_id)

    print(f"\nUpdated principle store: {pid}")
    print(f"Store size: {len(store)}")

    assert len(store) == 1, "Should have 1 principle"

    print("\n✅ TEST 3 PASSED")
    return True


def test_act_with_principles():
    """Test act() method with principles enabled."""
    print("\n" + "=" * 60)
    print("TEST 4: Act with Principles")
    print("=" * 60)

    from roboworld.agent.romemo_stack import RoMemoDiscreteConfig, RoMemoDiscreteAgent

    # Create mock environment and agent
    mock_env = create_mock_env()
    mock_base = create_mock_agent()
    mock_base.act.return_value = "insert red"

    cfg = RoMemoDiscreteConfig(
        use_principles=True,
        reflector_provider=None,  # Rule-based
        retrieval_mode="visual",
    )

    agent = RoMemoDiscreteAgent(
        base_agent=mock_base,
        env=mock_env,
        cfg=cfg,
        writeback=False,
        use_vision_retrieval=False,  # Use state vectors
    )

    # Add a principle that should influence decision
    p1 = Principle(
        content="Before inserting, verify predecessors are inserted.",
        action_types=["insert"],
        trigger_conditions=["piece_has_dependencies"],
        addresses_fail_tags=["BLOCKED_BY_PREDECESSOR"],
        importance_score=5.0,
    )
    agent.principle_store.add(p1)

    print(f"Principle store size: {len(agent.principle_store)}")
    print(f"Principles enabled: {agent.use_principles}")

    # Create test image
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_goal = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Call act
    action = agent.act(test_img, test_goal, "What action?")

    print(f"\nBase proposed: insert red")
    print(f"Chosen action: {action}")

    # Check trace for principles info
    trace = agent.last_trace
    print(f"\nTrace - principles enabled: {trace.get('principles', {}).get('enabled')}")
    print(f"Trace - principles count: {trace.get('principles', {}).get('count')}")
    print(f"Trace - principle store size: {trace.get('principles', {}).get('store_size')}")

    assert "principles" in trace, "Trace should have principles info"
    assert trace["principles"]["enabled"], "Principles should be enabled"

    print("\n✅ TEST 4 PASSED")
    return True


def test_update_with_reflection():
    """Test update() method triggering reflection on failure."""
    print("\n" + "=" * 60)
    print("TEST 5: Update with Reflection")
    print("=" * 60)

    from roboworld.agent.romemo_stack import RoMemoDiscreteConfig, RoMemoDiscreteAgent

    # Create mock environment and agent
    mock_env = create_mock_env()
    mock_base = create_mock_agent()

    cfg = RoMemoDiscreteConfig(
        use_principles=True,
        reflector_provider=None,  # Rule-based
        write_on_failure=True,
    )

    agent = RoMemoDiscreteAgent(
        base_agent=mock_base,
        env=mock_env,
        cfg=cfg,
        writeback=True,
        use_vision_retrieval=False,
    )

    print(f"Initial principle store size: {len(agent.principle_store)}")

    # Call act first to create pending experience
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_goal = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    action = agent.act(test_img, test_goal, "What action?")

    print(f"Action taken: {action}")

    # Simulate failure
    result = agent.update(
        executed_action=action,
        err_code=1,
        episode_id=1,
        step_id=1,
        fail_tag="BLOCKED_BY_PREDECESSOR",
        oracle_action="insert blue",
    )

    print(f"\nUpdate result: {result}")
    print(f"Learned principle: {result.get('learned_principle')}")
    print(f"Principle store size after: {len(agent.principle_store)}")

    # Check if principle was learned
    assert len(agent.principle_store) > 0, "Should have learned a principle"

    # Get stats
    stats = agent.get_principle_stats()
    print(f"\nPrinciple stats: {stats}")

    print("\n✅ TEST 5 PASSED")
    return True


def run_all_tests():
    """Run all Phase 2 tests."""
    print("=" * 70)
    print("PHASE 2 TEST SUITE: Principle Integration into Action Selection")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")

    tests = [
        ("Principle Retrieval", test_principle_retrieval),
        ("Principle Violation Check", test_principle_violation_check),
        ("Reflection on Failure", test_reflection_on_failure),
        ("Act with Principles", test_act_with_principles),
        ("Update with Reflection", test_update_with_reflection),
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
        print("\n🎉 ALL TESTS PASSED! Phase 2 implementation is ready.")
    else:
        print("\n⚠️  Some tests failed. Please fix before proceeding.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
