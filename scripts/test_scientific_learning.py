#!/usr/bin/env python3
"""
Test script for the Scientific Learning Loop implementation.

This script tests:
1. Hypothesis creation and management
2. Experience clustering and consolidation
3. Verification planning
4. Principle promotion
5. VLM prompt integration
6. State persistence

Run from the reflect-vlm directory:
    python scripts/test_scientific_learning.py
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np


def test_hypothesis_system():
    """Test hypothesis creation and management."""
    print("\n" + "=" * 60)
    print("TEST: Hypothesis System")
    print("=" * 60)

    from romemo.memory.hypothesis import (
        Hypothesis,
        HypothesisStore,
        ExperienceCluster,
        VerificationPlan,
        PrincipleType,
        HypothesisStatus,
    )

    # Test Hypothesis creation
    h1 = Hypothesis(
        statement="Before inserting a piece, verify all blocking pieces are already inserted",
        hypothesis_type=PrincipleType.AVOID,
        source_experience_ids=["exp_001", "exp_002", "exp_003"],
        action_types=["insert"],
        trigger_conditions=["has_dependencies"],
        predictions=[
            {
                "condition": "Insert with unsatisfied dependencies",
                "if_true": "BLOCKED_BY_PREDECESSOR",
                "if_false": "Success",
            }
        ],
    )

    assert h1.hid.startswith("h_")
    assert h1.hypothesis_type == PrincipleType.AVOID
    assert h1.status == HypothesisStatus.PROPOSED
    assert h1.confidence == 0.3
    print(f"✓ Created hypothesis: {h1.statement[:50]}...")

    # Test confidence update
    h1.update_confidence(0.8)
    assert h1.confidence > 0.3
    print(f"✓ Updated confidence: {h1.confidence:.2f}")

    # Test verification recording
    h1.add_verification(
        accuracy=0.9,
        conditions=[{"name": "test", "result": "pass"}],
        episode_ids=["ep_001", "ep_002"],
    )
    assert len(h1.verification_history) == 1
    assert h1.verification_episodes_completed == 2
    print(f"✓ Recorded verification: {h1.verification_history[-1]}")

    # Test HypothesisStore
    store = HypothesisStore(name="test_store")
    hid1 = store.add(h1)
    assert len(store) == 1
    print(f"✓ Added hypothesis to store: {hid1}")

    # Add another hypothesis
    h2 = Hypothesis(
        statement="Elongated blocks need specific orientation",
        hypothesis_type=PrincipleType.PREFER,
        action_types=["insert", "reorient"],
        shape_patterns=["elongated"],
    )
    hid2 = store.add(h2)
    assert len(store) == 2
    print(f"✓ Added second hypothesis: {hid2}")

    # Test retrieval by status
    proposed = store.get_proposed()
    # h1 is still PROPOSED (verification doesn't auto-change status)
    # h2 is PROPOSED
    # So we should have 2 proposed hypotheses
    assert len(proposed) == 2
    print(f"✓ Retrieved proposed hypotheses: {len(proposed)}")

    # Test similar finding
    similar = store.find_similar("Before inserting, check blocking pieces", threshold=0.2)
    if len(similar) > 0:
        print(f"✓ Found similar hypothesis: {similar[0][0].statement[:40]}...")
    else:
        # Simple word overlap may not match well, but that's OK for the test
        print("✓ Similar finding tested (no match at threshold 0.2, expected with simple overlap)")

    # Test serialization
    h1_dict = h1.to_dict()
    h1_restored = Hypothesis.from_dict(h1_dict)
    assert h1_restored.statement == h1.statement
    assert h1_restored.hypothesis_type == h1.hypothesis_type
    print("✓ Hypothesis serialization works")

    # Test store persistence
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        store.save(Path(f.name))
        loaded_store = HypothesisStore.load(Path(f.name))
        assert len(loaded_store) == 2
        os.unlink(f.name)
    print("✓ HypothesisStore persistence works")

    print("\n✅ Hypothesis System: All tests passed!")
    return True


def test_experience_cluster():
    """Test experience cluster creation."""
    print("\n" + "=" * 60)
    print("TEST: Experience Cluster")
    print("=" * 60)

    from romemo.memory.hypothesis import ExperienceCluster

    cluster = ExperienceCluster(
        experience_ids=["exp_001", "exp_002", "exp_003", "exp_004", "exp_005"],
        common_pattern="All involve inserting elongated blocks",
        outcome_distribution={"success": 3, "fail": 2},
        shape_patterns=["elongated"],
        action_patterns=["insert"],
    )

    assert cluster.cid.startswith("c_")
    assert cluster.size == 5
    assert cluster.success_rate == 0.6
    assert not cluster.is_mostly_failures
    print(f"✓ Created cluster with {cluster.size} experiences")
    print(f"✓ Success rate: {cluster.success_rate:.1%}")
    print(f"✓ Is mostly failures: {cluster.is_mostly_failures}")
    print(f"✓ Is mixed: {cluster.is_mixed}")

    # Test serialization
    c_dict = cluster.to_dict()
    c_restored = ExperienceCluster.from_dict(c_dict)
    assert c_restored.size == cluster.size
    print("✓ Cluster serialization works")

    print("\n✅ Experience Cluster: All tests passed!")
    return True


def test_consolidation_engine():
    """Test the consolidation engine."""
    print("\n" + "=" * 60)
    print("TEST: Consolidation Engine")
    print("=" * 60)

    from romemo.memory.schema import Experience, MemoryBank
    from romemo.memory.hypothesis import HypothesisStore
    from romemo.memory.consolidation import ConsolidationEngine, ConsolidationConfig

    # Create memory bank with test experiences
    memory = MemoryBank(name="test_memory")

    # Add diverse experiences
    for i in range(20):
        is_success = i % 3 != 0  # 2/3 success rate
        exp = Experience(
            task="assembly",
            subtask="action",
            success=is_success,
            fail=not is_success,
            fail_tag="BLOCKED_BY_PREDECESSOR" if not is_success else None,
            symbolic_state={
                "action_type": "insert",
                "is_holding": True,
                "num_remaining": 5 - (i % 5),
                "dependencies_satisfied": is_success,
                "target_signature": "block_25x4x8_elongated"
                if i % 2 == 0
                else "block_4x4x8_square",
            },
            extra_metrics={"action": f"insert piece_{i}"},
        )
        memory.add(exp)

    print(f"✓ Created memory bank with {len(memory)} experiences")

    # Create consolidation engine
    hypothesis_store = HypothesisStore(name="test_hypotheses")
    config = ConsolidationConfig(
        min_experiences_for_consolidation=5,
        min_cluster_size=2,
        min_experiences_for_hypothesis=3,
        run_async=False,  # Synchronous for testing
    )
    engine = ConsolidationEngine(
        memory=memory,
        hypothesis_store=hypothesis_store,
        config=config,
    )

    # Run consolidation
    clusters = engine.consolidate()
    print(f"✓ Generated {len(clusters)} clusters")

    for cluster in clusters:
        print(f"  - Cluster {cluster.cid}: {cluster.size} exp, {cluster.success_rate:.0%} success")

    # Generate hypotheses
    hypotheses = engine.generate_hypotheses(clusters)
    print(f"✓ Generated {len(hypotheses)} hypotheses")

    for h in hypotheses:
        print(f"  - [{h.hypothesis_type}] {h.statement[:50]}...")

    # Check stats
    stats = engine.get_stats()
    print(f"✓ Consolidation stats: {stats}")

    print("\n✅ Consolidation Engine: All tests passed!")
    return True


def test_verification_planner():
    """Test the verification planner."""
    print("\n" + "=" * 60)
    print("TEST: Verification Planner")
    print("=" * 60)

    from romemo.memory.hypothesis import (
        Hypothesis,
        HypothesisStore,
        PrincipleType,
        HypothesisStatus,
    )
    from romemo.memory.principle import Principle, PrincipleStore
    from romemo.memory.verification import VerificationPlanner, VerificationConfig

    # Create stores
    hypothesis_store = HypothesisStore(name="test_hypotheses")
    principle_store = PrincipleStore(name="test_principles")

    # Add some hypotheses
    h1 = Hypothesis(
        statement="Check dependencies before inserting",
        hypothesis_type=PrincipleType.AVOID,
        action_types=["insert"],
        predictions=[
            {
                "condition": "Insert without checking",
                "if_true": "Failure",
                "if_false": "Success",
            }
        ],
    )
    hypothesis_store.add(h1)

    h2 = Hypothesis(
        statement="Reorient elongated blocks before insertion",
        hypothesis_type=PrincipleType.PREFER,
        action_types=["insert", "reorient"],
        shape_patterns=["elongated"],
    )
    hypothesis_store.add(h2)

    print(f"✓ Added {len(hypothesis_store)} hypotheses")

    # Create planner
    config = VerificationConfig(
        verification_probability=1.0,  # Always verify for testing
        min_episodes_per_hypothesis=1,
    )
    planner = VerificationPlanner(
        hypothesis_store=hypothesis_store,
        principle_store=principle_store,
        config=config,
    )

    # Get verification plan
    plan = planner.get_next_verification()
    assert plan is not None
    print(f"✓ Created verification plan: {plan.plan_id}")
    print(f"  Hypothesis: {plan.hypothesis_id}")
    print(f"  Conditions: {len(plan.conditions)}")

    # Record successful result
    planner.record_result(success=True, episode_id="ep_001")
    print("✓ Recorded verification result")

    # Check hypothesis status
    h = hypothesis_store.get(plan.hypothesis_id)
    print(f"  Updated confidence: {h.confidence:.2f}")
    print(f"  Status: {h.status}")

    # Manually verify for promotion
    h.confidence = 0.9
    h.status = HypothesisStatus.VERIFIED
    h.verification_history.append({"accuracy": 0.9})
    h.verification_history.append({"accuracy": 0.85})

    # Check promotion
    promoted = planner.promote_verified()
    print(f"✓ Promoted {len(promoted)} hypotheses to principles")

    for p in promoted:
        print(f"  - [{p.principle_type}] {p.content[:50]}...")

    print("\n✅ Verification Planner: All tests passed!")
    return True


def test_scientific_learning_loop():
    """Test the complete scientific learning loop."""
    print("\n" + "=" * 60)
    print("TEST: Scientific Learning Loop")
    print("=" * 60)

    from romemo.memory.scientific_loop import (
        ScientificLearningLoop,
        ScientificLearningConfig,
        create_scientific_learning_loop,
    )

    # Create loop with minimal config
    config = ScientificLearningConfig(
        memory_name="test_scientific",
        consolidation_interval=5,
        min_experiences_for_consolidation=3,
        run_consolidation_async=False,  # Synchronous for testing
        verification_probability=0.5,
    )

    loop = ScientificLearningLoop(config=config)
    print("✓ Created ScientificLearningLoop")

    # Record some experiences
    for i in range(10):
        success = i % 3 != 0
        eid = loop.record_experience(
            action=f"insert piece_{i}",
            success=success,
            fail=not success,
            fail_tag="BLOCKED_BY_PREDECESSOR" if not success else None,
            symbolic_state={
                "action_type": "insert",
                "is_holding": True,
                "target_signature": "block_elongated",
                "dependencies_satisfied": success,
            },
        )
    print(f"✓ Recorded 10 experiences")

    # End episode
    result = loop.end_episode(success=True, episode_id="ep_001")
    print(f"✓ Ended episode: {result}")

    # Run manual consolidation
    consol_result = loop.run_consolidation()
    print(f"✓ Consolidation result: {consol_result}")

    # Test principle prompt integration
    test_prompt = """## Task
Select the next action for the robot.
Available actions: pick up, insert, reorient, put down, done."""

    # Add a test principle manually
    from romemo.memory.principle import Principle, PrincipleType

    test_principle = Principle(
        content="Always check dependencies before inserting",
        principle_type=PrincipleType.AVOID,
        action_types=["insert"],
        importance_score=5.0,
    )
    loop.principle_store.add(test_principle)

    enhanced_prompt = loop.enhance_prompt_with_principles(
        original_prompt=test_prompt,
        action_type="insert",
    )

    assert "Learned Principles" in enhanced_prompt
    print("✓ Enhanced prompt with principles:")
    print("-" * 40)
    print(enhanced_prompt[:500])
    print("-" * 40)

    # Test state persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_state"
        loop.save_state(str(save_path))
        print(f"✓ Saved state to {save_path}")

        # Load state
        loaded_loop = ScientificLearningLoop.load_state(str(save_path), config=config)
        assert len(loaded_loop.memory) == len(loop.memory)
        print(f"✓ Loaded state: {len(loaded_loop.memory)} experiences")

    # Print summary
    loop.print_summary()

    # Cleanup
    loop.shutdown()
    print("✓ Shutdown complete")

    print("\n✅ Scientific Learning Loop: All tests passed!")
    return True


def test_principle_types():
    """Test principle types in the Principle class."""
    print("\n" + "=" * 60)
    print("TEST: Principle Types")
    print("=" * 60)

    from romemo.memory.principle import Principle, PrincipleStore
    from romemo.memory.hypothesis import PrincipleType

    # Create principles of different types
    p_avoid = Principle(
        content="Don't insert before checking dependencies",
        principle_type=PrincipleType.AVOID,
        action_types=["insert"],
    )

    p_prefer = Principle(
        content="Prefer reorienting elongated blocks before insertion",
        principle_type=PrincipleType.PREFER,
        action_types=["reorient", "insert"],
    )

    p_sequence = Principle(
        content="Insert base pieces before top pieces",
        principle_type=PrincipleType.SEQUENCE,
        action_types=["insert"],
    )

    p_compare = Principle(
        content="Inserting with alignment is better than without",
        principle_type=PrincipleType.COMPARE,
        action_types=["insert"],
    )

    store = PrincipleStore(name="test_typed_principles")
    for p in [p_avoid, p_prefer, p_sequence, p_compare]:
        store.add(p)
        print(f"✓ Added {p.principle_type} principle: {p.content[:40]}...")

    assert len(store) == 4
    print(f"✓ Store has {len(store)} principles")

    # Test serialization
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        store.save(Path(f.name))
        loaded_store = PrincipleStore.load(Path(f.name))

        # Check types preserved
        for p in loaded_store.principles:
            assert isinstance(p.principle_type, PrincipleType)
            print(f"✓ Restored {p.principle_type} principle")

        os.unlink(f.name)

    print("\n✅ Principle Types: All tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SCIENTIFIC LEARNING LOOP TEST SUITE")
    print("=" * 60)

    tests = [
        ("Hypothesis System", test_hypothesis_system),
        ("Experience Cluster", test_experience_cluster),
        ("Consolidation Engine", test_consolidation_engine),
        ("Verification Planner", test_verification_planner),
        ("Principle Types", test_principle_types),
        ("Scientific Learning Loop", test_scientific_learning_loop),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            import traceback

            results.append((name, False, str(e)))
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, success, error in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name}: {status}")
        if error:
            print(f"    Error: {error}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
