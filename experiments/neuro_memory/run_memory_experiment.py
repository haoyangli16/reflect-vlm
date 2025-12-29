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

import roboworld.fix_triton_import

import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

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
    from roboworld.agent.vlm_api import UnifiedVLM  # For VLM-based policies
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
# IK Failure Detection
# ============================================================================


class IKFailureError(Exception):
    """
    Exception raised when IK (Inverse Kinematics) failure is detected.

    This happens when the robot cannot reach a target pose within the
    maximum allowed steps. The environment prints "Max steps exceeded for `goto`"
    or similar messages when this occurs.
    """

    pass


# Patterns that indicate IK failure (robot cannot reach target pose)
IK_FAILURE_PATTERNS = [
    "Max steps exceeded for `goto`"
    # "Max steps exceeded for aligning",
]


def execute_action_with_ik_check(env, action: str) -> Tuple[int, bool]:
    """
    Execute an action and check for IK failures.

    Args:
        env: The FrankaAssemblyEnv environment
        action: The action text to execute (e.g., "pick up red")

    Returns:
        Tuple of (error_code, ik_failure_detected)
        - error_code: The error code from env.act_txt (0 = success)
        - ik_failure_detected: True if IK failure was detected
    """
    # Capture stdout to detect IK failure messages
    captured_output = io.StringIO()

    # We need to capture stdout while still allowing it to print
    # So we use a Tee-like approach
    original_stdout = sys.stdout

    class TeeOutput:
        def __init__(self, original, capture):
            self.original = original
            self.capture = capture

        def write(self, text):
            self.original.write(text)
            self.capture.write(text)

        def flush(self):
            self.original.flush()
            self.capture.flush()

    sys.stdout = TeeOutput(original_stdout, captured_output)

    try:
        err = env.act_txt(action)
    finally:
        sys.stdout = original_stdout

    # Check if any IK failure patterns were detected
    output_text = captured_output.getvalue()
    ik_failure = any(pattern in output_text for pattern in IK_FAILURE_PATTERNS)

    return err, ik_failure


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
        # Only print on failures or first step
        if self.enabled and symbolic_state and not action_success:
            target = symbolic_state.get("target_signature") or "N/A"
            holding = symbolic_state.get("holding_signature") or "N/A"
            deps = symbolic_state.get("dependencies_satisfied", True)
            # Truncate long signatures
            target = target[:30] + "..." if len(target) > 30 else target
            holding = holding[:30] + "..." if len(holding) > 30 else holding
            print(f"    [STATE] target={target}, holding={holding}, deps={deps}")

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

        # Also save the full prompt for debugging
        prompt_log = self.save_dir / "prompts.jsonl"
        self._write(
            prompt_log,
            {
                "timestamp": datetime.now().isoformat(),
                "episode_id": episode_id,
                "step": step,
                "action": action,
                "enhanced_prompt": enhanced_prompt or prompt,
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

    def save_hypotheses_snapshot(
        self,
        learning_loop,
        episode: int,
    ):
        """Save current hypotheses to a JSON file for inspection."""
        if not self.enabled or not learning_loop:
            return

        hypotheses = learning_loop.hypothesis_store.hypotheses
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "episode": episode,
            "total": len(hypotheses),
            "hypotheses": [],
        }

        for h in hypotheses:
            snapshot["hypotheses"].append(
                {
                    "hid": h.hid,
                    "statement": h.statement,
                    "type": str(h.hypothesis_type),
                    "status": str(h.status),
                    "confidence": h.confidence,
                    "supporting_episodes": getattr(h, "supporting_episodes", []),
                    "contradicting_episodes": getattr(h, "contradicting_episodes", []),
                    "created_at": getattr(h, "created_at", ""),
                }
            )

        # Write to snapshot file
        snapshot_file = self.save_dir / f"hypotheses_ep{episode}.json"
        with open(snapshot_file, "w") as f:
            json.dump(snapshot, f, indent=2)

        # Also update the "latest" file
        latest_file = self.save_dir / "hypotheses_latest.json"
        with open(latest_file, "w") as f:
            json.dump(snapshot, f, indent=2)

    def save_principles_snapshot(
        self,
        learning_loop,
        episode: int,
    ):
        """Save current principles to a JSON file for inspection."""
        if not self.enabled or not learning_loop:
            return

        principles = learning_loop.principle_store.principles
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "episode": episode,
            "total": len(principles),
            "principles": [],
        }

        for p in principles:
            ptype = getattr(p, "principle_type", "GENERAL")
            if hasattr(ptype, "name"):
                ptype = ptype.name
            snapshot["principles"].append(
                {
                    "pid": getattr(p, "pid", "unknown"),
                    "content": p.content,
                    "type": ptype,
                    "confidence": p.confidence,
                    "status": getattr(p, "status", "active"),
                    "reinforcement_count": getattr(p, "reinforcement_count", 0),
                    "prediction_errors": getattr(p, "prediction_errors", 0),
                }
            )

        # Write to snapshot file
        snapshot_file = self.save_dir / f"principles_ep{episode}.json"
        with open(snapshot_file, "w") as f:
            json.dump(snapshot, f, indent=2)

        # Also update the "latest" file
        latest_file = self.save_dir / "principles_latest.json"
        with open(latest_file, "w") as f:
            json.dump(snapshot, f, indent=2)

    def display_working_memory(
        self,
        learning_loop,
        episode: int,
    ):
        """
        Display current working memory state (hypotheses and principles).

        This provides real-time visibility into what the memory system has learned.
        """
        if not self.enabled or not learning_loop:
            return

        # Save snapshots to files
        self.save_hypotheses_snapshot(learning_loop, episode)
        self.save_principles_snapshot(learning_loop, episode)

        stats = learning_loop.get_stats()

        print("\n" + "=" * 60)
        print(f"📊 WORKING MEMORY STATUS (Episode {episode})")
        print("=" * 60)

        # Principles
        principles = learning_loop.principle_store.principles
        active_principles = [p for p in principles if getattr(p, "status", "active") == "active"]
        print(f"\n🏆 PRINCIPLES ({len(active_principles)} active / {len(principles)} total):")
        if active_principles:
            for i, p in enumerate(active_principles[:5], 1):
                conf = int(p.confidence * 100)
                ptype = getattr(p, "principle_type", "GENERAL")
                if hasattr(ptype, "name"):
                    ptype = ptype.name
                print(f"  {i}. [{conf}%] [{ptype}] {p.content[:60]}...")
        else:
            print("  (none yet)")

        # Hypotheses
        hypotheses = learning_loop.hypothesis_store.hypotheses
        proposed = [h for h in hypotheses if h.status.value == "proposed"]
        testing = [h for h in hypotheses if h.status.value == "testing"]
        verified = [h for h in hypotheses if h.status.value == "verified"]
        refuted = [h for h in hypotheses if h.status.value == "refuted"]

        print(f"\n💡 HYPOTHESES ({len(hypotheses)} total):")
        print(
            f"  📝 Proposed: {len(proposed)}, 🧪 Testing: {len(testing)}, ✅ Verified: {len(verified)}, ❌ Refuted: {len(refuted)}"
        )

        if testing:
            print("\n  Currently Testing:")
            for h in testing[:3]:
                conf = int(h.confidence * 100)
                print(f"    - [{conf}%] {h.statement[:55]}...")

        if proposed:
            print("\n  Proposed (awaiting test):")
            for h in proposed[:3]:
                print(f"    - {h.statement[:55]}...")

        # Memory stats
        print(
            f"\n📦 MEMORY: {stats['memory']['total']} experiences, {stats['folding']['total_folded']} folded"
        )
        print("=" * 60 + "\n")


# ============================================================================
# VLM Policy Agent (for using external VLMs as action policy)
# ============================================================================


class VLMPolicyAgent:
    """
    Agent that uses an external VLM (GPT, Gemini, Qwen, Kimi) for action selection.

    This provides an alternative to the LLaVA BC policy, allowing experimentation
    with more capable VLMs that can better understand error feedback and principles.
    """

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """
        Initialize VLM policy agent.

        Args:
            provider: VLM provider ("openai", "gemini", "qwen", "kimi")
            model: Specific model name (or None for provider default)
            temperature: Sampling temperature
        """
        self.vlm = UnifiedVLM(provider=provider, model=model)
        self.temperature = temperature
        self.provider = provider
        self.model = self.vlm.model  # Get the actual model name from UnifiedVLM

    def act(
        self,
        current_image: np.ndarray,
        goal_image: np.ndarray,
        prompt: str,
    ) -> str:
        """
        Get action from VLM given current state, goal, and prompt.

        Args:
            current_image: Current observation (RGB numpy array)
            goal_image: Goal state image (RGB numpy array)
            prompt: Text prompt including principles, hypotheses, etc.

        Returns:
            Action string (e.g., "pick up blue", "insert red", "done")
        """
        # Build VLM prompt
        vlm_prompt = f"""{prompt}

Based on the goal image and current image, output ONLY the next action.
Valid actions: pick up [color], put down [color], reorient [color], insert [color], done

Your response should be ONLY the action, nothing else. Example: "pick up blue"
"""

        try:
            # Call VLM with both images
            response = self.vlm.generate(
                prompt=vlm_prompt,
                images=[goal_image, current_image],
            )

            # Parse action from response
            action = self._parse_action(response)
            return action

        except Exception as e:
            print(f"    [VLM Policy Error] {e}")
            # Fallback to a safe action
            return "done"

    def _parse_action(self, response: str) -> str:
        """Parse action from VLM response."""
        response = response.strip().lower()

        # Common action prefixes
        valid_prefixes = ["pick up", "put down", "reorient", "insert", "done"]

        # Try to find a valid action in the response
        for line in response.split("\n"):
            line = line.strip()
            for prefix in valid_prefixes:
                if line.startswith(prefix):
                    # Extract the full action (prefix + color if applicable)
                    if prefix == "done":
                        return "done"
                    # Get the color after the prefix
                    parts = line.split()
                    if len(parts) >= 3:  # e.g., "pick up blue"
                        return f"{parts[0]} {parts[1]} {parts[2]}"
                    elif len(parts) == 2:  # e.g., "insert blue"
                        return f"{parts[0]} {parts[1]}"

        # If no valid action found, return the raw response (may fail in env)
        return response.split("\n")[0].strip()


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
    # Reset seed controls the initial physical state of bricks (on table vs inserted)
    # A fixed reset_seed=1 gives consistent "easy" initial states (like interact.py)
    # Set to -1 to use the episode seed for varied difficulty
    reset_seed: int = 1

    # ==========================================================================
    # POLICY CONFIGURATION
    # ==========================================================================
    # Policy type: "bc" (LLaVA behavior cloning) or "vlm" (external VLM)
    policy_type: str = "bc"  # "bc" or "vlm"

    # For BC policy (LLaVA)
    base_model_path: str = "./ReflectVLM-llava-v1.5-13b-base"
    post_model_path: str = "./ReflectVLM-llava-v1.5-13b-post-trained"
    use_post_trained: bool = False  # If True, use post-trained model

    # For VLM policy (external APIs)
    policy_provider: Optional[str] = None  # "openai", "gemini", "qwen", "kimi"
    policy_model: Optional[str] = None  # Specific model name

    # Memory system (for consolidation/reflection, separate from policy)
    vlm_provider: Optional[str] = None  # "kimi", "openai", "gemini", "qwen", or None for rule-based
    vlm_model: Optional[str] = None

    # Scientific Learning Loop settings
    consolidation_interval: int = 10  # Episodes between consolidation runs
    min_experiences_for_consolidation: int = 5
    run_consolidation_async: bool = True

    # Working memory display
    show_working_memory: bool = True  # Show hypotheses/principles during run
    working_memory_interval: int = 5  # Show every N episodes

    # Output
    save_dir: str = "logs/neuro_memory_exp"
    save_memory: bool = True
    verbose: bool = True
    debug: bool = True  # Enable detailed debug logging
    show_prompts: bool = False  # Print prompts to terminal (very verbose)
    save_images: bool = False  # Save progress images per step for visualization

    # Resume functionality
    resume: bool = False  # If True, resume from previous run
    resume_dir: Optional[str] = None  # Directory to resume from (if not specified, uses latest)

    # Rate limiting (to avoid API throttling)
    api_delay: float = 0.0  # Delay between API calls (seconds)
    step_delay: float = 0.0  # Delay between steps (seconds)

    # Ablation settings (for Experiment 5)
    disable_hypotheses: bool = False  # Disable hypothesis injection
    disable_principles: bool = False  # Disable principle injection
    disable_forgetting: bool = False  # Disable memory forgetting
    disable_folding: bool = False  # Disable experience folding
    disable_resonance: bool = False  # Disable resonance filtering

    # Difficulty filtering (for Experiment 2)
    min_n_body: int = 2  # Minimum number of bodies
    max_n_body: int = 5  # Maximum number of bodies (default)

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


def create_environment(
    seed: int,
    render_mode: str = "offscreen",
    min_n_body: int = 2,
    max_n_body: int = 5,
) -> Tuple[FrankaAssemblyEnv, Dict]:
    """
    Create a FrankaAssemblyEnv for a given seed.

    This follows the same pattern as run-rom.py to ensure compatibility.

    Args:
        seed: Random seed for environment generation
        render_mode: Rendering mode for the environment
        min_n_body: Minimum number of bodies (for difficulty filtering)
        max_n_body: Maximum number of bodies (for difficulty filtering)

    Returns:
        Tuple of (environment, env_info_dict)
    """
    import uuid
    from roboworld.envs.asset_path_utils import full_path_for

    # Generate the board - returns (xml, info) tuple
    xml, info = generate_xml(seed=seed)

    # Difficulty filtering
    n_body = info["n_bodies"]
    if n_body < min_n_body or n_body > max_n_body:
        reason = "too simple" if n_body < min_n_body else "too complex"
        print(f"Environment {reason} (n_body={n_body}), skipping episode {seed}")
        return None, None

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

        # =====================================================================
        # PHASE 1: Tier 1 Reflex - Step Error Buffer
        # =====================================================================
        # Track recent errors for immediate feedback in prompts
        self._error_buffer: List[Dict[str, Any]] = []  # Last N failed steps
        self._error_buffer_size = 5  # Keep last 5 errors for Tier 1 reflex

        # Track active hypotheses for attribution
        self._active_hypotheses: List[Any] = []

        # Track episode action summary for attribution
        self._episode_action_summary: List[Dict[str, Any]] = []

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
        # Reset environment with reset_seed
        # reset_seed controls the initial physical state (bricks on table vs inserted)
        # Use config.reset_seed for consistent difficulty, or -1 to use episode seed for varied difficulty
        if self.config.reset_seed >= 0:
            reset_seed = self.config.reset_seed
        else:
            reset_seed = int(episode_id.split("_")[-1]) if "_" in episode_id else 0
        env.reset(seed=reset_seed)

        done = False
        step = 0
        action_history = []
        success = False

        # =====================================================================
        # PHASE 1 & 2: Reset per-episode tracking buffers
        # =====================================================================
        self._error_buffer = []  # Clear Tier 1 reflex buffer
        self._episode_action_summary = []  # Clear for attribution

        # PHASE 2: Get active hypotheses at episode start
        if self.learning_loop and self.config.mode == "memory":
            self._active_hypotheses = self.learning_loop.get_hypotheses_for_prompt()
        else:
            self._active_hypotheses = []

        # Track previous state for progress detection
        prev_holding = None
        repeated_action_count = 0
        last_action = None
        stuck_counter = 0  # Count consecutive stuck/no-progress steps
        max_stuck_steps = 10  # Early terminate if stuck for too long

        # Camera name for rendering (matching run-rom.py default)
        camera_name = self.config.camera_name

        # Get goal image (stored in env after reset)
        # This is the target assembly state that the agent should achieve
        goal_img = env.goal_images.get(camera_name, None)
        if goal_img is None:
            raise RuntimeError(f"Goal image not found for camera '{camera_name}'")

        # =====================================================================
        # IMAGE SAVING: Create episode directory and save goal image
        # =====================================================================
        episode_img_dir = None
        if self.config.save_images:
            episode_img_dir = Path(self.config.save_dir) / "images" / episode_id
            episode_img_dir.mkdir(parents=True, exist_ok=True)
            # Save goal image once at episode start
            goal_pil = Image.fromarray(goal_img)
            goal_pil.save(episode_img_dir / "goal.png")
            print(f"  [IMG] Saving images to: {episode_img_dir}")

        while not done and step < self.config.max_steps_per_episode:
            # Get current observation using read_pixels (matching run-rom.py)
            img = env.read_pixels(camera_name=camera_name)

            # Save current state image (before action)
            if episode_img_dir is not None:
                step_pil = Image.fromarray(img)
                step_pil.save(episode_img_dir / f"step_{step:03d}_before.png")

            # Generate action prompt
            prompt = self._build_action_prompt(env, info, action_history)

            # Get action from base agent
            if self.learning_loop and self.config.mode == "memory":
                # =====================================================================
                # THREE-TIER PROMPT ENHANCEMENT
                # =====================================================================
                action_type = self._get_expected_action_type(env)

                # Tier 3: Get principles (most reliable, highest priority)
                self._active_principles, _ = self.learning_loop.get_principles_for_context(
                    action_type=action_type,
                )

                # Tier 2: Get active hypotheses (under testing)
                active_hypotheses = self._active_hypotheses

                # Tier 1: Get recent errors (reflex feedback)
                recent_errors = self._error_buffer

                # Get current gripper state (CRITICAL for avoiding wrong actions)
                obj_in_hand = env.get_object_in_hand()
                holding_color = None
                if obj_in_hand:
                    # Convert brick_X to color name using peg_labels
                    holding_color = self._brick_to_color(obj_in_hand, info)

                # Build enhanced prompt with all tiers
                enhanced_prompt = self._build_tiered_prompt(
                    base_prompt=prompt,
                    principles=self._active_principles,
                    hypotheses=active_hypotheses,
                    recent_errors=recent_errors,
                    holding_object=holding_color,
                )
            else:
                enhanced_prompt = prompt

            # Optionally show prompt
            if self.config.show_prompts and step == 0:
                print("\n" + "=" * 40 + " PROMPT " + "=" * 40)
                print(enhanced_prompt[:1000] + ("..." if len(enhanced_prompt) > 1000 else ""))
                print("=" * 88 + "\n")

            # Get action from agent
            action = self.base_agent.act(img, goal_img, enhanced_prompt)
            action = str(action).strip()

            # Check for done action
            if action.lower().strip() == "done":
                success = env.is_success()
                done = True
                action_success = success
                action_fail = not success
                fail_tag = None if success else "incomplete"
                action_history.append(action)
            else:
                # Execute action using env.act_txt with IK failure detection
                err, ik_failure = execute_action_with_ik_check(env, action)

                # If IK failure detected, raise exception to skip this episode
                if ik_failure:
                    raise IKFailureError(
                        f"IK failure during '{action}' at step {step}. "
                        f"This is an environment bug, not a policy error."
                    )

                action_fail = err != 0
                action_success = err == 0
                fail_tag = f"err_{err}" if err != 0 else None

                # =====================================================================
                # CRITICAL FIX: Include error feedback in action history
                # This helps the model understand what failed
                # =====================================================================
                if action_fail:
                    # Add action with error feedback to history
                    action_history.append(f"{action} [FAILED]")
                else:
                    action_history.append(action)

                # =====================================================================
                # PART (ii): Check success after EVERY action execution
                # Don't wait for VLM to output "done" - the simulator is ground truth
                # This significantly increases measured success rate
                # =====================================================================
                if action_success and env.is_success():
                    print(
                        f"    [AUTO-DONE] Goal achieved after '{action}' - ending episode as success"
                    )
                    success = True
                    done = True

            # =====================================================================
            # PHASE 1: Add errors to Tier 1 reflex buffer + stuck detection
            # =====================================================================
            if action_fail and fail_tag:
                error_entry = {
                    "step": step,
                    "action": action,
                    "error": fail_tag,
                    "target_signature": None,  # Will be filled from symbolic_state
                }
                self._error_buffer.append(error_entry)
                # Keep only last N errors
                if len(self._error_buffer) > self._error_buffer_size:
                    self._error_buffer.pop(0)

                # Stuck detection: if same action failed 3+ times, try to break loop
                recent_failed = [e["action"] for e in self._error_buffer[-3:]]
                if len(recent_failed) >= 3 and len(set(recent_failed)) == 1:
                    # Only print first stuck detection
                    if stuck_counter == 0:
                        print(f"    [STUCK] Repeated: {action}")
                    # Add stronger hint to action history
                    action_history[-1] = f"{action} [FAILED - DO NOT REPEAT THIS ACTION]"

            # =====================================================================
            # NO-PROGRESS DETECTION: Catch "successful" actions that don't change state
            # =====================================================================
            # Check current holding state
            try:
                current_holding = (
                    env.get_object_in_hand() if hasattr(env, "get_object_in_hand") else None
                )
            except Exception:
                current_holding = None

            # Detect repeated same action
            if action == last_action:
                repeated_action_count += 1
            else:
                repeated_action_count = 1
                last_action = action

            # If action "succeeded" but state didn't change for insert/pick actions
            is_stuck = False
            blocking_info = ""  # Track what's blocking for VLM feedback
            if action_success and repeated_action_count >= 3:
                action_type = action.split()[0].lower() if action else ""
                if action_type == "insert" and current_holding is not None:
                    # Insert "succeeded" but still holding = slot is blocked!
                    # Extract symbolic state to find what's blocking
                    try:
                        temp_symbolic = extract_symbolic_state(env, action)
                        blocking_colors = temp_symbolic.get("target_blocked_by_colors", [])
                        # Filter out already-inserted pieces
                        remaining = temp_symbolic.get("remaining_pieces", [])
                        blocking_remaining = [c for c in blocking_colors if c in remaining]
                        if blocking_remaining:
                            blocking_info = f" Slot blocked by: {', '.join(blocking_remaining)}. Remove blocking brick(s) first!"
                            print(f"    [NO-PROGRESS] Insert failed - slot blocked by {blocking_remaining}")
                        else:
                            blocking_info = " Slot may be physically blocked. Try a different target."
                            print(f"    [NO-PROGRESS] Insert succeeded but still holding object")
                    except Exception:
                        print(f"    [NO-PROGRESS] Insert succeeded but still holding object")

                    action_history[-1] = f"{action} [FAILED - SLOT OCCUPIED!{blocking_info}]"
                    action_success = False  # Mark as failure for learning
                    fail_tag = "no_progress"
                    is_stuck = True
                elif action_type == "pick" and current_holding == prev_holding:
                    # Pick "succeeded" but holding state unchanged = no progress
                    print(f"    [NO-PROGRESS] Pick succeeded but not holding new object")
                    action_history[-1] = f"{action} [NO PROGRESS - TRY DIFFERENT ACTION]"
                    action_success = False
                    fail_tag = "no_progress"
                    is_stuck = True

            # Update stuck counter
            if action_fail or is_stuck:
                stuck_counter += 1
            else:
                stuck_counter = 0  # Reset on successful progress

            # Early termination if stuck for too long
            if stuck_counter >= max_stuck_steps:
                print(
                    f"    [EARLY-TERM] Agent stuck for {stuck_counter} steps, terminating episode"
                )
                done = True
                success = False

            prev_holding = current_holding

            # =====================================================================
            # PHASE 3: Track action for attribution
            # =====================================================================
            self._episode_action_summary.append(
                {
                    "step": step,
                    "action": action,
                    "success": action_success,
                    "fail_tag": fail_tag,
                }
            )

            # Get oracle action (for learning from mistakes)
            # Also get oracle action when stuck, not just when action_fail
            oracle_action = None
            if self.oracle and (action_fail or is_stuck):
                try:
                    oracle_action = self.oracle.act()
                    if is_stuck and oracle_action:
                        print(f"    [ORACLE] Correct action would be: {oracle_action}")
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

        # =====================================================================
        # IMAGE SAVING: Save final state and action log
        # =====================================================================
        if episode_img_dir is not None:
            # Save final state image
            final_img = env.read_pixels(camera_name=camera_name)
            final_pil = Image.fromarray(final_img)
            final_pil.save(episode_img_dir / "final.png")

            # Save action log as JSON
            action_log = {
                "episode_id": episode_id,
                "success": success,
                "total_steps": step,
                "actions": action_history,
            }
            with open(episode_img_dir / "actions.json", "w") as f:
                json.dump(action_log, f, indent=2)

        # End episode in learning loop with attribution (PHASE 3)
        if self.learning_loop and self.config.mode == "memory":
            self.learning_loop.end_episode(
                success=success,
                episode_id=episode_id,
                action_summary=self._episode_action_summary,  # For attribution
                active_hypotheses=self._active_hypotheses,  # For attribution
            )

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

    def _build_tiered_prompt(
        self,
        base_prompt: str,
        principles: List[Principle],
        hypotheses: List[Any],
        recent_errors: List[Dict[str, Any]],
        holding_object: Optional[str] = None,
    ) -> str:
        """
        Build prompt with proper ordering for human-like reading.

        CORRECT Order (Task → Actions → Strategy → State → Memory → Reflex → Query):
        1. TASK DESCRIPTION (base_prompt) - What the robot needs to do
        2. ACTION DEFINITIONS - Clear explanation of each action
        3. STRATEGIC GUIDANCE - How to plan and decide
        4. CURRENT STATE - What you're holding (critical!)
        5. LONG-TERM MEMORY - Principles (high confidence rules)
        6. WORKING MEMORY - Hypotheses (testing)
        7. REFLEX MEMORY - Recent errors (avoid repeating)
        """
        sections = []

        # =====================================================================
        # 1. TASK DESCRIPTION FIRST (extracted from base_prompt)
        # =====================================================================
        sections.append(base_prompt)

        # =====================================================================
        # 2. ACTION DEFINITIONS - Clear explanation of each action
        # =====================================================================
        action_defs = [
            "",
            "---",
            "## 📋 ACTION DEFINITIONS",
            "",
            "**pick up [color]**: Pick up a brick. Can pick up bricks from the TABLE or bricks already INSERTED in the board.",
            "",
            "**put down [color]**: Put down the brick you're holding ONTO THE TABLE (not onto the board).",
            "",
            "**reorient [color]**: Rotate the brick you're holding to align with the board slot. WARNING: The resulting orientation may not be perfect - after reorienting, check if the brick can be correctly inserted. If not, you may need to reorient again.",
            "",
            "**insert [color]**: Insert the brick you're holding INTO THE BOARD. This will only succeed if: (1) you're holding the brick, (2) it's properly oriented, and (3) the target slot is not blocked by other bricks.",
            "",
            "**done**: Declare the task complete. Use this ONLY when the current observation matches the goal image - all required bricks are correctly inserted in their target positions.",
            "",
            "---",
            "## ⚡ CRITICAL: BEFORE EVERY ACTION, FIRST CHECK IF DONE",
            "",
            "**STEP 0 (MANDATORY)**: Compare the current observation with the goal image:",
            "- If the current state MATCHES the goal state (all bricks are in their correct positions as shown in the goal image), output `done` IMMEDIATELY.",
            "- Do NOT pick up, remove, or re-insert bricks after the goal is achieved.",
            "- Unnecessary actions after goal completion waste steps and may break the solution.",
            "",
            "**What 'done' means**: The task is COMPLETE. The current observation matches the goal image. Any further action is unnecessary because we already achieved the goal.",
            "",
            "**Common mistake to avoid**: After successfully inserting the last brick, do NOT pick it back up to 'verify' or 're-insert'. If the goal is achieved, just output `done`.",
            "",
        ]
        sections.append("\n".join(action_defs))

        # =====================================================================
        # 3. STRATEGIC GUIDANCE - How to plan and decide
        # =====================================================================
        strategy_lines = [
            "---",
            "## 🎯 STRATEGIC PLANNING",
            "",
            "Before each action, think through these questions:",
            "",
            "1. **Check for blockers**: Are there bricks currently on the board that will BLOCK me from inserting the remaining bricks (on the table) to reach the goal state?",
            "",
            "2. **If blockers exist**: Which brick should I REMOVE FIRST? Pick it up from the board and put it down on the table.",
            "",
            "3. **If no blockers**: Which brick on the table should I INSERT FIRST to make progress toward the goal? Consider the assembly order - some bricks must be inserted before others.",
            "",
            "4. **Handle blocked insertions**: If you try to insert a brick but fail because another brick is blocking the slot, you must:",
            "   - Put down the brick you're holding onto the table",
            "   - Pick up the blocking brick from the board",
            "   - Put down the blocking brick onto the table",
            "   - Then decide your next move",
            "",
        ]
        sections.append("\n".join(strategy_lines))

        # =====================================================================
        # 4. CURRENT STATE (CRITICAL - prevents invalid actions!)
        # =====================================================================
        if holding_object:
            state_lines = [
                "",
                "---",
                "## 🤖 CURRENT STATE",
                "",
                f"⚠️ You are currently HOLDING: **{holding_object}**",
                "",
                "IMPORTANT RULES:",
                f"- You CANNOT 'pick up' another object while holding {holding_object}",
                f"- You CAN ONLY: 'insert {holding_object}' or 'put down {holding_object}' or 'reorient {holding_object}'",
                "",
            ]
            sections.append("\n".join(state_lines))
        else:
            state_lines = [
                "",
                "---",
                "## 🤖 CURRENT STATE",
                "",
                "Your gripper is EMPTY. You can 'pick up' any object.",
                "",
            ]
            sections.append("\n".join(state_lines))

        # =====================================================================
        # 5. LONG-TERM MEMORY: Verified Principles
        # =====================================================================
        if principles:
            principle_lines = ["## 🏆 LEARNED PRINCIPLES (Apply these!)"]
            principle_lines.append("")
            for i, p in enumerate(principles[:5], 1):  # Max 5 principles
                conf_pct = int(p.confidence * 100)
                type_label = getattr(p, "principle_type", "GENERAL")
                if hasattr(type_label, "name"):
                    type_label = type_label.name
                principle_lines.append(f"{i}. [{conf_pct}%] [{type_label}] {p.content}")
            principle_lines.append("")
            sections.append("\n".join(principle_lines))

        # =====================================================================
        # 6. WORKING MEMORY: Active Hypotheses (under testing)
        # =====================================================================
        if hypotheses:
            hypo_lines = ["## 💡 HYPOTHESES (Consider but verify)"]
            hypo_lines.append("")
            for i, h in enumerate(hypotheses[:3], 1):  # Max 3 hypotheses
                h_type = getattr(h, "hypothesis_type", "GENERAL")
                if hasattr(h_type, "name"):
                    h_type = h_type.name
                statement = getattr(h, "statement", str(h))
                hypo_lines.append(f"{i}. [{h_type}] {statement}")
            hypo_lines.append("")
            sections.append("\n".join(hypo_lines))

        # =====================================================================
        # 7. REFLEX MEMORY: Recent Errors
        # =====================================================================
        if recent_errors:
            error_lines = ["## ⚠️ RECENT ERRORS (Avoid repeating!)"]
            error_lines.append("")
            for err in recent_errors[-3:]:  # Show last 3 errors
                error_lines.append(
                    f"- Step {err['step']}: '{err['action']}' failed ({err['error']})"
                )
            error_lines.append("")
            sections.append("\n".join(error_lines))

        # Combine all sections
        return "\n".join(sections)

    def _brick_to_color(self, brick_name: str, info: Dict[str, Any]) -> str:
        """
        Convert internal brick name (e.g., 'brick_2') to color name (e.g., 'yellow').

        This grounding is critical because:
        - The VLM sees colors in images and expects color-based actions
        - The internal simulator uses brick_X naming
        - Actions are specified as 'pick up yellow', not 'pick up brick_2'
        """
        try:
            # Extract brick ID (e.g., 'brick_2' -> 2)
            brick_id = int(brick_name.split("_")[-1])

            # Get peg_names and peg_labels from info
            peg_names = info.get("peg_names", [])
            peg_labels = info.get("peg_labels", [])

            # Find the index of this brick in peg_names
            for i, peg_name in enumerate(peg_names):
                if peg_name == brick_name and i < len(peg_labels):
                    return peg_labels[i]  # Return the color

            # Fallback: try using brick_id directly as index
            peg_ids = info.get("peg_ids", [])
            if brick_id in peg_ids:
                idx = peg_ids.index(brick_id)
                if idx < len(peg_labels):
                    return peg_labels[idx]

            # Last resort: return the original name
            return brick_name

        except (ValueError, IndexError, AttributeError):
            return brick_name

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


def find_latest_run_dir(base_dir: str, name: str, mode: str) -> Optional[Path]:
    """Find the latest run directory matching the pattern."""
    base = Path(base_dir)
    if not base.exists():
        return None

    # Look for directories matching pattern: {name}_{mode}_*
    pattern = f"{name}_{mode}_*"
    matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the full experiment.

    Supports resuming from previous runs if --resume is specified.

    Returns:
        Experiment results summary
    """
    # =========================================================================
    # Handle Resume Logic
    # =========================================================================
    resume_state = None
    start_episode = 0
    previous_results = []

    if config.resume:
        # Find directory to resume from
        if config.resume_dir:
            resume_path = Path(config.resume_dir)
        else:
            resume_path = find_latest_run_dir(config.save_dir, config.name, config.mode)

        if resume_path and resume_path.exists():
            print(f"[RESUME] Found previous run at: {resume_path}")

            # Load previous results
            results_file = resume_path / "episode_results.jsonl"
            if results_file.exists():
                with open(results_file) as f:
                    previous_results = [json.loads(line) for line in f if line.strip()]
                start_episode = len(previous_results)
                print(f"[RESUME] Loaded {start_episode} previous episodes")

            # Load metadata for seed offset
            metadata_file = resume_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    resume_state = json.load(f)
                print(f"[RESUME] Loaded metadata from {metadata_file}")

            # Use same save_dir for continuity
            save_dir = resume_path
        else:
            print(f"[RESUME] No previous run found, starting fresh")
            config.resume = False

    if not config.resume:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(config.save_dir) / f"{config.name}_{config.mode}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"NEURO-SYMBOLIC MEMORY EXPERIMENT")
    print("=" * 60)
    print(f"Mode: {config.mode}")
    print(f"Episodes: {config.n_episodes}" + (f" (resuming from {start_episode})" if config.resume else ""))
    print(f"Save Dir: {save_dir}")
    print(f"Difficulty: n_body in [{config.min_n_body}, {config.max_n_body}]")
    print(f"Policy: {config.policy_type.upper()}", end="")
    if config.policy_type == "vlm":
        print(f" ({config.policy_provider}/{config.policy_model or 'default'})")
    else:
        print(f" (LLaVA)")
    print(f"Reflection VLM: {config.vlm_provider or 'rule-based'}")
    if config.api_delay > 0:
        print(f"API Delay: {config.api_delay}s")
    if config.resume:
        print(f"[RESUME MODE] Continuing from episode {start_episode}")
    print("=" * 60)

    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # =========================================================================
    # Initialize Policy Agent (BC or VLM)
    # =========================================================================
    print("\n[1/4] Loading policy agent...")

    if config.policy_type == "vlm":
        # Use external VLM as policy
        if not config.policy_provider:
            print("ERROR: --policy_provider required when --policy_type=vlm")
            sys.exit(1)

        base_agent = VLMPolicyAgent(
            provider=config.policy_provider,
            model=config.policy_model,
            temperature=0.1,
        )
        print(f"  VLM Policy: {config.policy_provider}/{base_agent.model}")
    else:
        # Use LLaVA BC policy (default)
        model_path = config.get_model_path()
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            sys.exit(1)

        base_agent = LlavaAgent(
            model_path=model_path,
            load_4bit=True,
        )
        print(f"  BC Policy (LLaVA): {model_path}")

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

        # Load existing state if resuming
        if config.resume and save_dir.exists() and (save_dir / "memory.pt").exists():
            print(f"  [RESUME] Loading learning loop state from {save_dir}")
            learning_loop = ScientificLearningLoop.load_state(str(save_dir), config=loop_config)
            print(f"  [RESUME] Loaded {learning_loop._episode_count} episodes of state")
        else:
            learning_loop = ScientificLearningLoop(config=loop_config)

        print(f"  Consolidation interval: {config.consolidation_interval} episodes")
        print(f"  VLM for reflection: {config.vlm_provider or 'rule-based'}")

        # Apply ablation settings
        if config.disable_forgetting:
            learning_loop.config.max_memory_size = 1000000  # Effectively disable
            print(f"  [ABLATION] Forgetting DISABLED")
        if config.disable_hypotheses:
            print(f"  [ABLATION] Hypothesis injection DISABLED")
        if config.disable_principles:
            print(f"  [ABLATION] Principle injection DISABLED")
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
    # We continue until we have n_episodes successful (non-IK-failure) episodes
    print("\n[3/4] Running episodes...")
    results = previous_results.copy() if config.resume else []
    successes = sum(1 for r in results if r.get("success", False))
    completed_episodes = start_episode  # Start from resume point
    ik_failures = 0  # Episodes skipped due to IK failure
    skipped_seeds = []  # Seeds that had IK failures

    seed_offset = 0  # Offset to get next seed when skipping

    # Calculate starting seed offset for resume
    if config.resume and resume_state:
        # Use the seed offset from previous run if available
        seed_offset = resume_state.get("seed_offset", 0)
        skipped_seeds = resume_state.get("skipped_seeds", [])

    while completed_episodes < config.n_episodes:
        seed = config.seed_start + completed_episodes + seed_offset
        episode_id = f"ep_{seed}"

        if config.verbose and completed_episodes % 10 == 0:
            print(f"\n  Episode {completed_episodes + 1}/{config.n_episodes} (seed={seed})")

        # =====================================================================
        # Display working memory periodically
        # =====================================================================
        if (
            config.show_working_memory
            and debug_logger
            and learning_loop
            and completed_episodes > 0
            and completed_episodes % config.working_memory_interval == 0
        ):
            debug_logger.display_working_memory(learning_loop, completed_episodes)

        env = None
        info = None
        try:
            # Create environment with difficulty filtering
            env, info = create_environment(
                seed,
                min_n_body=config.min_n_body,
                max_n_body=config.max_n_body,
            )

            if env is None:
                seed_offset += 1
                skipped_seeds.append(seed)
                continue

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
            completed_episodes += 1

            if result["success"]:
                successes += 1

            if config.verbose:
                status = "✓" if result["success"] else "✗"
                print(f"    {status} {episode_id}: {result['steps']} steps")

            # Close environment and cleanup XML file
            env.close()
            runner.oracle = None
            xml_file = info.get("_xml_filename")
            if xml_file and os.path.exists(xml_file):
                try:
                    os.remove(xml_file)
                except Exception:
                    pass

        except IKFailureError as e:
            # IK failure is an environment bug, skip this episode
            ik_failures += 1
            seed_offset += 1  # Shift to next seed
            skipped_seeds.append(seed)
            print(f"    [IK BUG] Skipping {episode_id}: {e}")
            print(f"    (IK failures so far: {ik_failures}, will try seed {seed + 1} next)")

            # Cleanup on IK failure
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            runner.oracle = None
            if info and info.get("_xml_filename"):
                try:
                    os.remove(info["_xml_filename"])
                except Exception:
                    pass

        except Exception as e:
            # Other errors still count as completed episodes (just failed)
            print(f"    ERROR in episode {episode_id}: {e}")
            results.append(
                {
                    "episode_id": episode_id,
                    "success": False,
                    "error": str(e),
                }
            )
            completed_episodes += 1

            # Cleanup on error too
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            runner.oracle = None
            if info and info.get("_xml_filename"):
                try:
                    os.remove(info["_xml_filename"])
                except Exception:
                    pass

    # Calculate final metrics (only counting completed episodes, not IK failures)
    success_rate = successes / completed_episodes if completed_episodes > 0 else 0.0

    # Report IK failures
    if ik_failures > 0:
        print(f"\n  [IK BUG SUMMARY] Skipped {ik_failures} episodes due to IK failures")
        print(f"  Skipped seeds: {skipped_seeds[:10]}{'...' if len(skipped_seeds) > 10 else ''}")

    print("\n[4/4] Saving results...")

    # Save results
    summary = {
        "mode": config.mode,
        "n_episodes": config.n_episodes,
        "completed_episodes": completed_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "ik_failures": ik_failures,
        "skipped_seeds": skipped_seeds,
        "seed_offset": seed_offset,  # For resume
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "difficulty": {"min_n_body": config.min_n_body, "max_n_body": config.max_n_body},
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

    # Save final hypothesis/principle snapshots
    if debug_logger and learning_loop:
        debug_logger.save_hypotheses_snapshot(learning_loop, completed_episodes)
        debug_logger.save_principles_snapshot(learning_loop, completed_episodes)
        debug_logger.display_working_memory(learning_loop, completed_episodes)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Success Rate: {success_rate:.1%} ({successes}/{completed_episodes})")
    if ik_failures > 0:
        print(f"IK Failures (skipped): {ik_failures} episodes")
    print(f"Results saved to: {save_dir}")
    print(f"Hypotheses: {save_dir}/hypotheses_latest.json")
    print(f"Principles: {save_dir}/principles_latest.json")
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
    # Baseline with LLaVA BC policy (no memory)
    python run_memory_experiment.py --mode baseline --n_episodes 50

    # Memory with LLaVA BC policy + Kimi reflection
    python run_memory_experiment.py --mode memory --provider kimi --n_episodes 100

    # Memory with GPT-4 Vision as policy + Kimi reflection
    python run_memory_experiment.py --mode memory --policy_type vlm --policy_provider openai \\
        --provider kimi --n_episodes 50

    # Memory with Gemini as policy (no separate reflection VLM)
    python run_memory_experiment.py --mode memory --policy_type vlm --policy_provider gemini \\
        --provider rule --n_episodes 50
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
        "--reset_seed",
        type=int,
        default=1,
        help="Reset seed for initial physical state (1=easy like interact.py, -1=use episode seed for varied difficulty)",
    )

    # ==========================================================================
    # POLICY OPTIONS (what generates actions)
    # ==========================================================================
    parser.add_argument(
        "--policy_type",
        type=str,
        default="bc",
        choices=["bc", "vlm"],
        help="Policy type: 'bc' (LLaVA behavior cloning) or 'vlm' (external VLM)",
    )
    parser.add_argument(
        "--policy_provider",
        type=str,
        default=None,
        choices=["openai", "gemini", "qwen", "kimi"],
        help="VLM provider for policy (required if policy_type=vlm)",
    )
    parser.add_argument(
        "--policy_model",
        type=str,
        default=None,
        help="Specific VLM model for policy (e.g., 'gpt-4-vision-preview')",
    )

    # ==========================================================================
    # REFLECTION OPTIONS (what generates hypotheses/principles)
    # ==========================================================================
    parser.add_argument(
        "--provider",
        type=str,
        default="rule",
        choices=["rule", "kimi", "openai", "gemini", "qwen"],
        help="VLM provider for reflection/consolidation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific VLM model for reflection",
    )

    # ==========================================================================
    # BC POLICY OPTIONS (LLaVA)
    # ==========================================================================
    parser.add_argument(
        "--base_model",
        type=str,
        default="./ReflectVLM-llava-v1.5-13b-base",
        help="Path to base LLaVA model (for BC policy)",
    )
    parser.add_argument(
        "--post_model",
        type=str,
        default="./ReflectVLM-llava-v1.5-13b-post-trained",
        help="Path to post-trained LLaVA model (for BC policy)",
    )
    parser.add_argument(
        "--use_post_trained",
        action="store_true",
        help="Use post-trained model instead of base (for BC policy)",
    )

    # ==========================================================================
    # OUTPUT OPTIONS
    # ==========================================================================
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
    parser.add_argument(
        "--show_memory",
        action="store_true",
        default=True,
        help="Show working memory status during run",
    )
    parser.add_argument(
        "--memory_interval",
        type=int,
        default=5,
        help="Show working memory every N episodes",
    )
    parser.add_argument(
        "--show_prompts",
        action="store_true",
        help="Print prompts to terminal (first step of each episode)",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save progress images per step for visualization",
    )

    # ==========================================================================
    # RESUME OPTIONS
    # ==========================================================================
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Directory to resume from (if not specified, finds latest)",
    )

    # ==========================================================================
    # RATE LIMITING OPTIONS
    # ==========================================================================
    parser.add_argument(
        "--api_delay",
        type=float,
        default=0.0,
        help="Delay between API calls in seconds (for rate limiting)",
    )
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.0,
        help="Delay between steps in seconds",
    )

    # ==========================================================================
    # ABLATION OPTIONS (for Experiment 5)
    # ==========================================================================
    parser.add_argument(
        "--disable_hypotheses",
        action="store_true",
        help="Disable hypothesis injection in prompts",
    )
    parser.add_argument(
        "--disable_principles",
        action="store_true",
        help="Disable principle injection in prompts",
    )
    parser.add_argument(
        "--disable_forgetting",
        action="store_true",
        help="Disable memory forgetting mechanism",
    )
    parser.add_argument(
        "--disable_folding",
        action="store_true",
        help="Disable experience folding mechanism",
    )
    parser.add_argument(
        "--disable_resonance",
        action="store_true",
        help="Disable resonance filtering",
    )

    # ==========================================================================
    # DIFFICULTY OPTIONS (for Experiment 2)
    # ==========================================================================
    parser.add_argument(
        "--min_n_body",
        type=int,
        default=2,
        help="Minimum number of bodies for difficulty filtering",
    )
    parser.add_argument(
        "--max_n_body",
        type=int,
        default=5,
        help="Maximum number of bodies for difficulty filtering",
    )

    args = parser.parse_args()

    # Build config
    config = ExperimentConfig(
        name=args.name,
        mode=args.mode,
        seed_start=args.seed_start,
        reset_seed=args.reset_seed,
        n_episodes=args.n_episodes,
        # Policy options
        policy_type=args.policy_type,
        policy_provider=args.policy_provider,
        policy_model=args.policy_model,
        # BC policy options
        base_model_path=args.base_model,
        post_model_path=args.post_model,
        use_post_trained=args.use_post_trained,
        # Reflection options
        vlm_provider=args.provider if args.provider != "rule" else None,
        vlm_model=args.model,
        # Output options
        save_dir=args.save_dir,
        verbose=args.verbose,
        show_working_memory=args.show_memory,
        working_memory_interval=args.memory_interval,
        show_prompts=args.show_prompts,
        save_images=args.save_images,
        # Resume options
        resume=args.resume,
        resume_dir=args.resume_dir,
        # Rate limiting
        api_delay=args.api_delay,
        step_delay=args.step_delay,
        # Ablation options
        disable_hypotheses=args.disable_hypotheses,
        disable_principles=args.disable_principles,
        disable_forgetting=args.disable_forgetting,
        disable_folding=args.disable_folding,
        disable_resonance=args.disable_resonance,
        # Difficulty options
        min_n_body=args.min_n_body,
        max_n_body=args.max_n_body,
    )

    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
