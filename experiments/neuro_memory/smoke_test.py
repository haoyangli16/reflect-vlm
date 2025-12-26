#!/usr/bin/env python3
"""
Smoke test for the Neuro-Symbolic Memory System.

This script tests the core components WITHOUT running the full LLaVA model,
making it fast to verify the setup is correct.

Usage:
    cd /path/to/reflect-vlm
    python experiments/neuro_memory/smoke_test.py
"""

import sys
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent.parent))

import json
import tempfile


def test_imports():
    """Test that all required modules can be imported."""
    print("\n[1/5] Testing imports...")

    try:
        from romemo.memory import (
            ScientificLearningLoop,
            ScientificLearningConfig,
            Experience,
            MemoryBank,
            Principle,
            PrincipleStore,
            Hypothesis,
            HypothesisStore,
            PrincipleType,
        )

        print("  ✓ romemo.memory imports OK")
    except ImportError as e:
        print(f"  ✗ romemo.memory import failed: {e}")
        return False

    try:
        from roboworld.envs.generator import generate_xml

        print("  ✓ roboworld.envs.generator imports OK")
    except ImportError as e:
        print(f"  ✗ roboworld.envs.generator import failed: {e}")
        return False

    try:
        from roboworld.agent.vlm_api import UnifiedVLM, SUPPORTED_PROVIDERS

        print(f"  ✓ vlm_api imports OK (providers: {list(SUPPORTED_PROVIDERS.keys())})")
    except ImportError as e:
        print(f"  ✗ vlm_api import failed: {e}")
        return False

    return True


def test_scientific_learning_loop():
    """Test the Scientific Learning Loop without VLM."""
    print("\n[2/5] Testing ScientificLearningLoop...")

    from romemo.memory import ScientificLearningLoop, ScientificLearningConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ScientificLearningConfig(
            memory_name="smoke_test",
            consolidation_interval=5,
            min_experiences_for_consolidation=3,
            run_consolidation_async=False,  # Synchronous for testing
            save_path=tmpdir,
        )

        loop = ScientificLearningLoop(config=config)
        print("  ✓ Created ScientificLearningLoop")

        # Record some experiences
        for i in range(10):
            success = i % 3 != 0
            eid, is_surprising = loop.record_experience(
                action=f"insert piece_{i}",
                success=success,
                fail=not success,
                symbolic_state={
                    "action_type": "insert",
                    "is_holding": True,
                    "target_signature": "block_elongated",
                },
            )
        print(f"  ✓ Recorded 10 experiences")

        # End episode
        result = loop.end_episode(success=True, episode_id="smoke_test_ep")
        print(f"  ✓ Ended episode: {result['total_experiences']} experiences stored")

        # Run consolidation
        consol = loop.run_consolidation()
        print(
            f"  ✓ Consolidation: {consol['clusters_created']} clusters, {consol['hypotheses_generated']} hypotheses"
        )

        # Check memory trace was written
        trace_path = Path(tmpdir) / "memory_trace.jsonl"
        if trace_path.exists():
            with open(trace_path) as f:
                lines = f.readlines()
            print(f"  ✓ Memory trace written: {len(lines)} entries")
        else:
            print("  ⚠ Memory trace not written (expected)")

        # Cleanup
        loop.shutdown()
        print("  ✓ Shutdown complete")

    return True


def test_resonance_check():
    """Test the resonance check (surprise detection)."""
    print("\n[3/5] Testing Resonance Check...")

    from romemo.memory import (
        ScientificLearningLoop,
        ScientificLearningConfig,
        Principle,
        PrincipleType,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ScientificLearningConfig(
            memory_name="resonance_test",
            run_consolidation_async=False,
            save_path=tmpdir,
        )

        loop = ScientificLearningLoop(config=config)

        # Add a test principle
        test_principle = Principle(
            content="Prefer inserting elongated blocks first",
            principle_type=PrincipleType.PREFER,
            action_types=["insert"],
        )
        loop.principle_store.add(test_principle)
        print(f"  ✓ Added test principle")

        # Record experience WITH active principles
        eid, is_surprising = loop.record_experience(
            action="insert block",
            success=True,  # Success as predicted by PREFER principle
            fail=False,
            active_principles=[test_principle],
            symbolic_state={"action_type": "insert"},
        )
        print(f"  ✓ Recorded expected experience: surprising={is_surprising}")
        assert not is_surprising, "Expected success should NOT be surprising"

        # Record surprising failure
        eid2, is_surprising2 = loop.record_experience(
            action="insert block",
            success=False,  # Failure NOT predicted by PREFER principle
            fail=True,
            active_principles=[test_principle],
            symbolic_state={"action_type": "insert"},
        )
        print(f"  ✓ Recorded unexpected failure: surprising={is_surprising2}")
        assert is_surprising2, "Unexpected failure SHOULD be surprising"

        # Check principle reinforcement
        p = loop.principle_store.principles[0]
        print(f"  ✓ Principle reinforcement_count: {p.reinforcement_count}")
        print(f"  ✓ Principle prediction_errors: {p.prediction_errors}")

        loop.shutdown()

    return True


def test_garbage_collection():
    """Test the garbage collection (active forgetting)."""
    print("\n[4/5] Testing Garbage Collection...")

    from romemo.memory import ScientificLearningLoop, ScientificLearningConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ScientificLearningConfig(
            memory_name="gc_test",
            run_consolidation_async=False,
            save_path=tmpdir,
        )

        loop = ScientificLearningLoop(config=config)

        # Record experiences and mark some as folded
        for i in range(5):
            eid, _ = loop.record_experience(
                action=f"action_{i}",
                success=True,
                fail=False,
                symbolic_state={"action_type": "insert"},
            )

        # Manually mark some experiences as folded and old
        for exp in loop.memory.experiences[:2]:
            exp.memory_status = "folded"
            exp.last_accessed_episode = 0  # Very old

        # Run GC with low threshold
        loop._episode_count = 200  # Simulate many episodes passed
        gc_result = loop.run_garbage_collection(folded_ttl=50)
        print(
            f"  ✓ GC result: {gc_result['experiences_pruned']} pruned, {gc_result['principles_archived']} archived"
        )

        loop.shutdown()

    return True


def test_environment_generation():
    """Test that environments can be generated."""
    print("\n[5/5] Testing Environment Generation...")

    try:
        from roboworld.envs.generator import generate_xml

        # generate_xml returns (xml, info) tuple
        xml, info = generate_xml(seed=1000001)
        print(f"  ✓ Generated board: {info.get('board_name', 'N/A')}")
        print(f"  ✓ Pieces: {len(info.get('peg_names', []))}")
        print(f"  ✓ Shape info available: {'brick_shapes' in info}")

        if "brick_shapes" in info:
            shapes = info["brick_shapes"]
            print(f"  ✓ Shape signatures: {len(shapes)} pieces")
            for name, shape in list(shapes.items())[:2]:
                print(f"    - {name}: {shape.get('signature', 'N/A')[:50]}...")

        return True
    except Exception as e:
        print(f"  ✗ Environment generation failed: {e}")
        return False


def main():
    print("=" * 60)
    print("NEURO-SYMBOLIC MEMORY SYSTEM SMOKE TEST")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("ScientificLearningLoop", test_scientific_learning_loop),
        ("Resonance Check", test_resonance_check),
        ("Garbage Collection", test_garbage_collection),
        ("Environment Generation", test_environment_generation),
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

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, s, _ in results if s)
    failed = len(results) - passed

    for name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
        if error:
            print(f"    Error: {error}")

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All smoke tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
