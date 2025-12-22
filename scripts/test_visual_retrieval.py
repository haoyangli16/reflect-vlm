#!/usr/bin/env python3
"""
Quick test to verify visual-based retrieval works correctly.
Run this before the full experiment to catch any bugs.
"""

import sys
import numpy as np
from PIL import Image


def test_llava_encode():
    """Test that LlavaAgent.encode_image() works."""
    print("=" * 60)
    print("TEST 1: LlavaAgent.encode_image()")
    print("=" * 60)

    try:
        from roboworld.agent.llava import LlavaAgent
    except ImportError as e:
        print(f"✗ Failed to import LlavaAgent: {e}")
        return False

    try:
        # Load lightweight model for testing
        print("Loading LlavaAgent (this may take 30s)...")
        agent = LlavaAgent(
            model_path="yunhaif/ReflectVLM-llava-v1.5-13b-base",
            load_4bit=True,
        )
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    try:
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Encode
        print("Encoding image...")
        feat = agent.encode_image(dummy_img)

        # Validate
        assert isinstance(feat, np.ndarray), f"Expected np.ndarray, got {type(feat)}"
        assert feat.dtype == np.float32, f"Expected float32, got {feat.dtype}"
        assert feat.ndim == 1, f"Expected 1D vector, got shape {feat.shape}"
        assert 512 <= feat.shape[0] <= 4096, f"Unexpected dimension: {feat.shape[0]}"

        # Check normalization
        norm = float(np.linalg.norm(feat))
        assert 0.99 < norm < 1.01, f"Feature not L2-normalized: ||feat|| = {norm}"

        print(f"✓ encode_image() works! Shape: {feat.shape}, ||feat|| = {norm:.4f}")
        return True

    except Exception as e:
        print(f"✗ encode_image() failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_romemo_wrapper():
    """Test that RoMemoDiscreteAgent uses visual retrieval."""
    print("\n" + "=" * 60)
    print("TEST 2: RoMemoDiscreteAgent with visual retrieval")
    print("=" * 60)

    try:
        from roboworld.agent.romemo_stack import RoMemoDiscreteAgent, RoMemoDiscreteConfig
        from roboworld.agent.llava import LlavaAgent
        from roboworld.envs.generator import generate_xml
        from roboworld.envs.mujoco.franka.franka_assembly import FrankaAssemblyEnv
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    try:
        # Build mini env
        print("Building test environment...")
        xml, info = generate_xml(seed=1000000)
        xml_path = "/tmp/test_romemo_visual.xml"
        xml.write_to_file(filename=xml_path)

        env = FrankaAssemblyEnv(
            board_name="brick_1",
            fixture_name=None,
            peg_names=[f"brick_{j + 1}" for j in range(1, info["n_bodies"])],
            peg_descriptions=[
                info["brick_descriptions"][f"brick_{j + 1}"] for j in range(1, info["n_bodies"])
            ],
            render_mode="offscreen",
            frame_skip=20,
            model_name=xml_path,
            max_episode_length=50000,
            magic_attaching=True,
        )
        env.reset(seed=1)
        print("✓ Env created")

        # Load base agent
        print("Loading base VLM agent...")
        base = LlavaAgent(
            model_path="yunhaif/ReflectVLM-llava-v1.5-13b-base",
            load_4bit=True,
        )
        print("✓ Base agent loaded")

        # Wrap with RoMemo
        print("Creating RoMemo wrapper...")
        agent = RoMemoDiscreteAgent(
            base_agent=base,
            env=env,
            task="assembly",
            cfg=RoMemoDiscreteConfig(k=5),
            writeback=True,
            seed=0,
        )
        print("✓ Wrapper created")

        # Check that use_vision is True
        if not agent.use_vision:
            print("✗ use_vision should be True for LlavaAgent")
            return False
        print("✓ use_vision = True (as expected)")

        # Test act()
        print("Testing act() with visual retrieval...")
        img = env.read_pixels(camera_name="table_back")
        goal_img = env.goal_images.get("table_back")
        inp = "What action should be taken next?"

        action = agent.act(img, goal_img, inp)
        print(f"✓ act() returned: {action}")

        # Verify trace was populated
        assert "context_hash" in agent.last_trace, "Missing context_hash in trace"
        assert "memory_size" in agent.last_trace, "Missing memory_size in trace"
        print(f"✓ Trace populated: context_hash={agent.last_trace['context_hash'][:8]}...")

        # Test update (writeback)
        print("Testing update() (writeback)...")
        agent.update(action, err_code=0, episode_id=0, step_id=0)
        print(f"✓ Memory size after writeback: {len(agent.store.memory)}")

        # Cleanup
        env.close()
        import os

        os.remove(xml_path)

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("RoMemo Visual Retrieval Validation")
    print("=" * 60)

    success = True
    success &= test_llava_encode()
    success &= test_romemo_wrapper()

    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nYou can now run the full experiment:")
        print("  bash scripts/eval_romemo_plugin.sh")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\nFix the errors above before running the full experiment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
