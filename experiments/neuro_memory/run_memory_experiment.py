#!/usr/bin/env python3
"""
Neuro-Symbolic Memory Experiment Runner.

This is a NEW, decoupled script for running experiments with the Scientific Learning Loop.
It does NOT depend on the old run-rom.py - everything is self-contained.

Key differences from old system:
1. Memory is collected ONLINE (not pre-loaded from .pt files)
2. Hypotheses and Principles are generated through consolidation
3. Surprise-driven learning: only unexpected experiences trigger learning
4. Active forgetting: old experiences are pruned automatically

Usage:
    # Baseline (no memory)
    python run_memory_experiment.py --mode baseline --n_episodes 100

    # With memory system (using Kimi VLM for reflection)
    python run_memory_experiment.py --mode memory --provider kimi --n_episodes 100

    # With memory system (rule-based reflection, no VLM API needed)
    python run_memory_experiment.py --mode memory --provider rule --n_episodes 100
"""

from __future__ import annotations

import fix_triton_import

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent.parent))  # worldmemory root

# Imports from reflect-vlm
try:
    from roboworld.envs.generator import generate_xml
    from roboworld.envs.mujoco.franka.franka_assembly import FrankaAssemblyEnv, AssemblyOracle
    from roboworld.agent.llava import LlavaAgent
    from roboworld.agent.utils import get_prompt  # Original prompt format!
    from roboworld.agent.romemo_stack import (
        RoMemoDiscreteAgent,
        RoMemoDiscreteConfig,
        RoMemoStore,
        extract_symbolic_state,
    )
except ImportError as e:
    print(f"ERROR: Failed to import roboworld modules: {e}")
    print("Make sure you're running from the reflect-vlm directory.")
    sys.exit(1)

# Imports from romemo (worldmemory)
try:
    from romemo.memory import (
        ScientificLearningLoop,
        ScientificLearningConfig,
        Experience,
        MemoryBank,
        Principle,
        PrincipleStore,
    )
except ImportError as e:
    print(f"ERROR: Failed to import romemo modules: {e}")
    print("Make sure worldmemory is installed: pip install -e /path/to/worldmemory")
    sys.exit(1)


# ============================================================================
# Debug Logger
# ============================================================================


class DebugLogger:
    """
    Comprehensive debug logger for memory experiments.

    Logs detailed information about:
    - Symbolic states
    - Actions and their outcomes
    - Hypotheses (proposed, verified, refuted)
    - Principles (active, applied)
    - Consolidation events
    """

    def __init__(self, save_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create log files
        self.symbolic_log = self.save_dir / "symbolic_states.jsonl"
        self.hypothesis_log = self.save_dir / "hypotheses.jsonl"
        self.principle_log = self.save_dir / "principles.jsonl"
        self.action_log = self.save_dir / "actions.jsonl"
        self.consolidation_log = self.save_dir / "consolidation.jsonl"

        # Clear old logs
        for log_file in [
            self.symbolic_log,
            self.hypothesis_log,
            self.principle_log,
            self.action_log,
            self.consolidation_log,
        ]:
            if log_file.exists():
                log_file.unlink()

    def _write(self, log_file: Path, data: Dict):
        """Write a JSON line to a log file."""
        if not self.enabled:
            return
        with open(log_file, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def log_symbolic_state(
        self,
        episode_id: str,
        step: int,
        action: str,
        symbolic_state: Optional[Dict],
        action_success: bool,
        fail_tag: Optional[str] = None,
    ):
        """Log a symbolic state observation."""
        self._write(
            self.symbolic_log,
            {
                "timestamp": datetime.now().isoformat(),
                "episode_id": episode_id,
                "step": step,
                "action": action,
                "success": action_success,
                "fail_tag": fail_tag,
                "symbolic_state": symbolic_state,
            },
        )
        # Also print key info if verbose
        if self.enabled and symbolic_state:
            target = symbolic_state.get("target_signature") or "N/A"
            holding = symbolic_state.get("holding_signature") or "N/A"
            deps = symbolic_state.get("dependencies_satisfied", True)
            print(
                f"    [DEBUG] State: target={target}, holding={holding}, "
                f"deps_ok={deps}, success={action_success}"
            )

    def log_action(
        self,
        episode_id: str,
        step: int,
        prompt: str,
        action: str,
        enhanced_prompt: Optional[str] = None,
        principles_used: Optional[List] = None,
    ):
        """Log an action decision."""
        self._write(
            self.action_log,
            {
                "timestamp": datetime.now().isoformat(),
                "episode_id": episode_id,
                "step": step,
                "action": action,
                "prompt_length": len(prompt),
                "enhanced_prompt_length": len(enhanced_prompt) if enhanced_prompt else 0,
                "n_principles_used": len(principles_used) if principles_used else 0,
                "principles": (
                    [p.content[:50] + "..." for p in principles_used] if principles_used else []
                ),
            },
        )

    def log_hypothesis(
        self,
        hypothesis,
        event: str = "created",
        episode_id: Optional[str] = None,
    ):
        """Log a hypothesis event (created, verified, refuted)."""
        self._write(
            self.hypothesis_log,
            {
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "episode_id": episode_id,
                "hid": getattr(hypothesis, "hid", "unknown"),
                "statement": hypothesis.statement,
                "type": str(hypothesis.hypothesis_type),
                "status": str(hypothesis.status),
                "confidence": getattr(hypothesis, "confidence", 0.0),
            },
        )
        print(f"    [HYPOTHESIS {event.upper()}] {hypothesis.statement[:80]}...")

    def log_principle(
        self,
        principle,
        event: str = "active",
        episode_id: Optional[str] = None,
    ):
        """Log a principle event (created, applied, decayed)."""
        self._write(
            self.principle_log,
            {
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "episode_id": episode_id,
                "pid": getattr(principle, "pid", "unknown"),
                "content": principle.content,
                "confidence": principle.confidence,
                "status": getattr(principle, "status", "active"),
                "reinforcement_count": getattr(principle, "reinforcement_count", 0),
                "prediction_errors": getattr(principle, "prediction_errors", 0),
            },
        )

    def log_consolidation(
        self,
        n_experiences: int,
        n_clusters: int,
        n_hypotheses_before: int,
        n_hypotheses_after: int,
        n_principles: int,
        details: Optional[Dict] = None,
    ):
        """Log a consolidation event."""
        self._write(
            self.consolidation_log,
            {
                "timestamp": datetime.now().isoformat(),
                "n_experiences": n_experiences,
                "n_clusters": n_clusters,
                "hypotheses_before": n_hypotheses_before,
                "hypotheses_after": n_hypotheses_after,
                "new_hypotheses": n_hypotheses_after - n_hypotheses_before,
                "n_principles": n_principles,
                "details": details,
            },
        )
        print(
            f"    [CONSOLIDATION] {n_experiences} exp -> {n_clusters} clusters -> "
            f"+{n_hypotheses_after - n_hypotheses_before} hypotheses"
        )

    def log_episode_summary(
        self,
        episode_id: str,
        success: bool,
        steps: int,
        n_hypotheses: int,
        n_principles: int,
        active_principles: Optional[List] = None,
    ):
        """Log episode summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "episode_id": episode_id,
            "success": success,
            "steps": steps,
            "n_hypotheses": n_hypotheses,
            "n_principles": n_principles,
            "active_principles": (
                [p.content[:50] for p in active_principles] if active_principles else []
            ),
        }
        # Write to a separate summary file
        summary_file = self.save_dir / "episode_summaries.jsonl"
        self._write(summary_file, summary)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for memory experiments."""

    # Experiment identity
    name: str = "memory_exp"
    mode: str = "baseline"  # "baseline" or "memory"

    # Environment
    seed_start: int = 1000001
    n_episodes: int = 100
    max_steps_per_episode: int = 50
    camera_name: str = "table_back"  # Camera for rendering (matches run-rom.py default)

    # Model paths (relative to project root, or absolute)
    base_model_path: str = "./ReflectVLM-llava-v1.5-13b-base"
    post_model_path: str = "./ReflectVLM-llava-v1.5-13b-post-trained"
    use_post_trained: bool = False  # If True, use post-trained model

    # Memory system
    vlm_provider: Optional[str] = None  # "kimi", "openai", "gemini", "qwen", or None for rule-based
    vlm_model: Optional[str] = None

    # Scientific Learning Loop settings
    consolidation_interval: int = 10  # Episodes between consolidation runs
    min_experiences_for_consolidation: int = 5
    run_consolidation_async: bool = True

    # Output
    save_dir: str = "logs/neuro_memory_exp"
    save_memory: bool = True
    verbose: bool = True
    debug: bool = True  # Enable detailed debug logging

    def get_model_path(self) -> str:
        """Get the appropriate model path."""
        path = self.post_model_path if self.use_post_trained else self.base_model_path
        # Handle relative paths
        if not os.path.isabs(path) and not os.path.exists(path):
            abs_path = PROJECT_ROOT / path
            if abs_path.exists():
                return str(abs_path)
        return path


# ============================================================================
# Environment Creation
# ============================================================================


def create_environment(seed: int, render_mode: str = "offscreen") -> Tuple[FrankaAssemblyEnv, Dict]:
    """
    Create a FrankaAssemblyEnv for a given seed.

    This follows the same pattern as run-rom.py to ensure compatibility.

    Returns:
        Tuple of (environment, env_info_dict)
    """
    import uuid
    from roboworld.envs.asset_path_utils import full_path_for

    # Generate the board - returns (xml, info) tuple
    xml, info = generate_xml(seed=seed)

    # Write XML to assets directory (where panda.xml and other includes are located)
    # This is critical - MuJoCo XML files use relative includes
    xml_filename = full_path_for(f"tmp_neuro_{uuid.uuid4()}.xml")
    xml.write_to_file(filename=xml_filename)

    # Construct peg_ids, peg_names, peg_descriptions (matching run-rom.py)
    board_name = "brick_1"
    fixture_name = None
    peg_ids = [j + 1 for j in range(1, info["n_bodies"])]
    peg_names = [f"brick_{j + 1}" for j in range(1, info["n_bodies"])]
    peg_descriptions = [info["brick_descriptions"][peg_name] for peg_name in peg_names]

    # Extract shape information
    brick_shapes = info.get("brick_shapes", {})
    color_to_signature = info.get("color_to_signature", {})
    signature_to_color = info.get("signature_to_color", {})
    dependency_signatures = info.get("dependency_signatures", [])

    # Create environment
    env = FrankaAssemblyEnv(
        board_name=board_name,
        fixture_name=fixture_name,
        peg_names=peg_names,
        peg_descriptions=peg_descriptions,
        render_mode=render_mode,
        frame_skip=20,
        model_name=xml_filename,
        max_episode_length=50000,
        magic_attaching=True,
        # Shape information for symbolic state
        brick_shapes=brick_shapes,
        color_to_signature=color_to_signature,
        signature_to_color=signature_to_color,
        dependency_signatures=dependency_signatures,
    )

    # Extract color labels (matching run-rom.py)
    peg_labels = [" ".join(pd.split()[:1]) for pd in peg_descriptions]

    # Build env_info dict (matching run-rom.py structure)
    env_info = {
        "peg_ids": peg_ids,
        "peg_names": peg_names,
        "peg_descriptions": peg_descriptions,
        "peg_labels": peg_labels,  # Color labels for prompts
        "brick_shapes": brick_shapes,
        "color_to_signature": color_to_signature,
        "signature_to_color": signature_to_color,
        "dependency_signatures": dependency_signatures,
        "dependencies": info["dependencies"],
        "brick_descriptions": info["brick_descriptions"],
        # Store xml_filename for cleanup
        "_xml_filename": xml_filename,
    }

    return env, env_info


# ============================================================================
# Episode Runner
# ============================================================================


class EpisodeRunner:
    """
    Runs a single episode with the memory system.

    This encapsulates the episode loop logic and integrates with
    the ScientificLearningLoop for experience recording.
    """

    def __init__(
        self,
        base_agent: LlavaAgent,
        oracle: Optional[AssemblyOracle],
        learning_loop: Optional[ScientificLearningLoop],
        config: ExperimentConfig,
        debug_logger: Optional[DebugLogger] = None,
    ):
        self.base_agent = base_agent
        self.oracle = oracle
        self.learning_loop = learning_loop
        self.config = config
        self.debug_logger = debug_logger

        # Track active principles for resonance checking
        self._active_principles: List[Principle] = []

    def run_episode(
        self,
        env: FrankaAssemblyEnv,
        info: Dict,
        episode_id: str,
    ) -> Dict[str, Any]:
        """
        Run a single episode.

        Returns:
            Episode result dictionary
        """
        # Reset environment with seed (extracted from episode_id)
        seed = int(episode_id.split("_")[-1]) if "_" in episode_id else 0
        env.reset(seed=seed)

        done = False
        step = 0
        action_history = []
        success = False

        # Camera name for rendering (matching run-rom.py default)
        camera_name = self.config.camera_name

        # Get goal image (stored in env after reset)
        # This is the target assembly state that the agent should achieve
        goal_img = env.goal_images.get(camera_name, None)
        if goal_img is None:
            raise RuntimeError(f"Goal image not found for camera '{camera_name}'")

        while not done and step < self.config.max_steps_per_episode:
            # Get current observation using read_pixels (matching run-rom.py)
            img = env.read_pixels(camera_name=camera_name)

            # Generate action prompt
            prompt = self._build_action_prompt(env, info, action_history)

            # Get action from base agent
            if self.learning_loop and self.config.mode == "memory":
                # Enhance prompt with learned principles
                enhanced_prompt = self.learning_loop.enhance_prompt_with_principles(
                    original_prompt=prompt,
                    action_type=self._get_expected_action_type(env),
                )
                # Get active principles for resonance tracking
                self._active_principles, _ = self.learning_loop.get_principles_for_context(
                    action_type=self._get_expected_action_type(env),
                )
            else:
                enhanced_prompt = prompt

            # Get action from agent
            action = self.base_agent.act(img, goal_img, enhanced_prompt)
            action = str(action).strip()
            action_history.append(action)

            # Check for done action
            if action.lower().strip() == "done":
                success = env.is_success()
                done = True
                action_success = success
                action_fail = not success
                fail_tag = None if success else "incomplete"
            else:
                # Execute action using env.act_txt
                err = env.act_txt(action)
                action_fail = err != 0
                action_success = err == 0
                fail_tag = f"err_{err}" if err != 0 else None

            # Get oracle action (for learning from mistakes)
            oracle_action = None
            if self.oracle and action_fail:
                try:
                    oracle_action = self.oracle.act()
                except Exception:
                    pass

            # Record experience with the learning loop
            if self.learning_loop and self.config.mode == "memory":
                # extract_symbolic_state requires: env, proposed_action, last_action_success, last_fail_tag
                symbolic_state = extract_symbolic_state(
                    env,
                    action,
                    last_action_success=action_success,
                    last_fail_tag=fail_tag,
                )

                # Debug log the symbolic state
                if self.debug_logger:
                    self.debug_logger.log_symbolic_state(
                        episode_id=episode_id,
                        step=step,
                        action=action,
                        symbolic_state=symbolic_state,
                        action_success=action_success,
                        fail_tag=fail_tag,
                    )
                    self.debug_logger.log_action(
                        episode_id=episode_id,
                        step=step,
                        prompt=prompt,
                        action=action,
                        enhanced_prompt=enhanced_prompt if enhanced_prompt != prompt else None,
                        principles_used=self._active_principles,
                    )

                self.learning_loop.record_experience(
                    action=action,
                    success=action_success,
                    fail=action_fail,
                    fail_tag=fail_tag,
                    symbolic_state=symbolic_state,
                    oracle_action=oracle_action,
                    active_principles=self._active_principles,
                    extra_metrics={
                        "episode_id": episode_id,
                        "step": step,
                    },
                )

            step += 1

        # End episode in learning loop
        if self.learning_loop and self.config.mode == "memory":
            self.learning_loop.end_episode(success=success, episode_id=episode_id)

            # Log episode summary
            if self.debug_logger:
                stats = self.learning_loop.get_stats()
                self.debug_logger.log_episode_summary(
                    episode_id=episode_id,
                    success=success,
                    steps=step,
                    n_hypotheses=stats["hypotheses"]["total"],
                    n_principles=stats["principles"]["total"],
                    active_principles=self._active_principles,
                )

        return {
            "episode_id": episode_id,
            "success": success,
            "steps": step,
            "actions": action_history,
        }

    def _build_action_prompt(
        self,
        env: FrankaAssemblyEnv,
        info: Dict,
        action_history: List[str],
    ) -> str:
        """
        Build the action prompt for the agent.

        CRITICAL: Uses the ORIGINAL get_prompt() format that the LLaVA model
        was trained on. Using a different format will break model performance!
        """
        # Get color labels (same as run-rom.py)
        peg_labels = info.get("peg_labels", [])
        if not peg_labels:
            # Fallback: extract from peg_descriptions
            peg_descriptions = info.get("peg_descriptions", [])
            peg_labels = [" ".join(pd.split()[:1]) for pd in peg_descriptions]

        # Use the ORIGINAL prompt format from roboworld.agent.utils
        prompt = get_prompt(
            version="propose",
            history=action_history,
            obj_labels=peg_labels,
        )
        return prompt

    def _get_expected_action_type(self, env: FrankaAssemblyEnv) -> Optional[str]:
        """Get the expected action type based on current state."""
        try:
            body = env.get_object_in_hand() if hasattr(env, "get_object_in_hand") else None
            if body is not None:
                return "insert"  # Holding something, probably want to insert
            else:
                return "pick"  # Not holding, probably want to pick
        except Exception:
            return None


# ============================================================================
# Main Experiment
# ============================================================================


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the full experiment.

    Returns:
        Experiment results summary
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config.save_dir) / f"{config.name}_{config.mode}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"NEURO-SYMBOLIC MEMORY EXPERIMENT")
    print("=" * 60)
    print(f"Mode: {config.mode}")
    print(f"Episodes: {config.n_episodes}")
    print(f"Save Dir: {save_dir}")
    print(f"VLM Provider: {config.vlm_provider or 'rule-based'}")
    print("=" * 60)

    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # Initialize base agent
    print("\n[1/4] Loading base agent...")
    model_path = config.get_model_path()
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    base_agent = LlavaAgent(
        model_path=model_path,
        load_4bit=True,
    )
    print(f"  Loaded: {model_path}")

    # Initialize learning loop (if memory mode)
    learning_loop = None
    if config.mode == "memory":
        print("\n[2/4] Initializing Scientific Learning Loop...")

        loop_config = ScientificLearningConfig(
            memory_name=config.name,
            consolidation_interval=config.consolidation_interval,
            min_experiences_for_consolidation=config.min_experiences_for_consolidation,
            run_consolidation_async=config.run_consolidation_async,
            vlm_provider=config.vlm_provider if config.vlm_provider != "rule" else None,
            vlm_model=config.vlm_model,
            save_path=str(save_dir),
            auto_save_interval=20,
        )

        learning_loop = ScientificLearningLoop(config=loop_config)
        print(f"  Consolidation interval: {config.consolidation_interval} episodes")
        print(f"  VLM for reflection: {config.vlm_provider or 'rule-based'}")
    else:
        print("\n[2/4] Baseline mode - no learning loop")

    # Create debug logger
    debug_logger = None
    if config.debug:
        debug_logger = DebugLogger(save_dir=save_dir, enabled=True)
        print(f"  Debug logging enabled: {save_dir}")

    # Create episode runner
    runner = EpisodeRunner(
        base_agent=base_agent,
        oracle=None,  # We'll create per-episode
        learning_loop=learning_loop,
        config=config,
        debug_logger=debug_logger,
    )

    # Run episodes
    print("\n[3/4] Running episodes...")
    results = []
    successes = 0

    for i in range(config.n_episodes):
        seed = config.seed_start + i
        episode_id = f"ep_{seed}"

        if config.verbose and i % 10 == 0:
            print(f"\n  Episode {i + 1}/{config.n_episodes} (seed={seed})")

        try:
            # Create environment
            env, info = create_environment(seed)

            # Create oracle for this episode
            # AssemblyOracle needs: brick_ids, brick_descriptions, dependencies, env
            oracle = AssemblyOracle(
                brick_ids=info["peg_ids"],
                brick_descriptions=info["peg_descriptions"],
                dependencies=info["dependencies"],
                env=env,
            )
            runner.oracle = oracle

            # Run episode
            result = runner.run_episode(env, info, episode_id)
            results.append(result)

            if result["success"]:
                successes += 1

            if config.verbose:
                status = "✓" if result["success"] else "✗"
                print(f"    {status} {episode_id}: {result['steps']} steps")

            # Close environment and cleanup XML file
            env.close()
            xml_file = info.get("_xml_filename")
            if xml_file and os.path.exists(xml_file):
                try:
                    os.remove(xml_file)
                except Exception:
                    pass

        except Exception as e:
            print(f"    ERROR in episode {episode_id}: {e}")
            results.append(
                {
                    "episode_id": episode_id,
                    "success": False,
                    "error": str(e),
                }
            )
            # Cleanup on error too
            if "info" in dir() and info and info.get("_xml_filename"):
                try:
                    os.remove(info["_xml_filename"])
                except Exception:
                    pass

    # Calculate final metrics
    success_rate = successes / config.n_episodes if config.n_episodes > 0 else 0.0

    print("\n[4/4] Saving results...")

    # Save results
    summary = {
        "mode": config.mode,
        "n_episodes": config.n_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "timestamp": timestamp,
    }

    if learning_loop:
        summary["memory_stats"] = learning_loop.get_stats()
        learning_loop.save_state()
        learning_loop.shutdown()

    with open(save_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(save_dir / "episode_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Success Rate: {success_rate:.1%} ({successes}/{config.n_episodes})")
    print(f"Results saved to: {save_dir}")
    print("=" * 60)

    return summary


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Symbolic Memory Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Baseline (no memory)
    python run_memory_experiment.py --mode baseline --n_episodes 50

    # Memory with Kimi VLM
    export MOONSHOT_API_KEY="your_key"
    python run_memory_experiment.py --mode memory --provider kimi --n_episodes 100

    # Memory with rule-based reflection (no API needed)
    python run_memory_experiment.py --mode memory --provider rule --n_episodes 100
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["baseline", "memory"],
        help="Experiment mode",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=100,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=1000001,
        help="Starting seed for episodes",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="rule",
        choices=["rule", "kimi", "openai", "gemini", "qwen"],
        help="VLM provider for reflection",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific VLM model name",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="./ReflectVLM-llava-v1.5-13b-base",
        help="Path to base LLaVA model",
    )
    parser.add_argument(
        "--use_post_trained",
        action="store_true",
        help="Use post-trained model instead of base",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs/neuro_memory_exp",
        help="Directory to save results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Experiment name",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # Build config
    config = ExperimentConfig(
        name=args.name,
        mode=args.mode,
        seed_start=args.seed_start,
        n_episodes=args.n_episodes,
        base_model_path=args.base_model,
        use_post_trained=args.use_post_trained,
        vlm_provider=args.provider if args.provider != "rule" else None,
        vlm_model=args.model,
        save_dir=args.save_dir,
        verbose=args.verbose,
    )

    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
