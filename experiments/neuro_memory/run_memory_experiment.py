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

# import fix_triton_import

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
        # Reset environment with seed (extracted from episode_id)
        seed = int(episode_id.split("_")[-1]) if "_" in episode_id else 0
        env.reset(seed=seed)

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

        while not done and step < self.config.max_steps_per_episode:
            # Get current observation using read_pixels (matching run-rom.py)
            img = env.read_pixels(camera_name=camera_name)

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

                # Build enhanced prompt with all tiers
                enhanced_prompt = self._build_tiered_prompt(
                    base_prompt=prompt,
                    principles=self._active_principles,
                    hypotheses=active_hypotheses,
                    recent_errors=recent_errors,
                )
            else:
                enhanced_prompt = prompt

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
                # Execute action using env.act_txt
                err = env.act_txt(action)
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
                    print(f"    [STUCK] Detected repeated failure: {action}")
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
            if action_success and repeated_action_count >= 3:
                action_type = action.split()[0].lower() if action else ""
                if action_type == "insert" and current_holding is not None:
                    # Insert "succeeded" but still holding = no progress
                    print(f"    [NO-PROGRESS] Insert succeeded but still holding object")
                    action_history[-1] = f"{action} [NO PROGRESS - TRY DIFFERENT ACTION]"
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
    ) -> str:
        """
        Build prompt with three-tier memory injection.

        Tier Structure (in prompt order):
        1. Tier 3: VERIFIED PRINCIPLES (highest confidence, prominent display)
        2. Tier 2: ACTIVE HYPOTHESES (under testing, advisory)
        3. Tier 1: RECENT ERRORS (immediate reflex feedback)
        4. Base prompt (original action request)

        This ordering ensures the agent sees the most reliable guidance first.
        """
        sections = []

        # =====================================================================
        # TIER 3: Verified Principles (PHASE 5 - Prominent Display)
        # =====================================================================
        if principles:
            principle_lines = ["## 🏆 LEARNED PRINCIPLES (Apply these first!)"]
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
        # TIER 2: Active Hypotheses (PHASE 2 - Injection)
        # =====================================================================
        if hypotheses:
            hypo_lines = ["## 💡 HYPOTHESES (Under Testing - Consider but verify)"]
            hypo_lines.append("")
            for i, h in enumerate(hypotheses[:3], 1):  # Max 3 hypotheses
                h_type = getattr(h, "hypothesis_type", "GENERAL")
                if hasattr(h_type, "name"):
                    h_type = h_type.name
                statement = getattr(h, "statement", str(h))
                hypo_lines.append(f"{i}. [TESTING] [{h_type}] {statement}")
            hypo_lines.append("")
            sections.append("\n".join(hypo_lines))

        # =====================================================================
        # TIER 1: Recent Errors (PHASE 1 - Reflex)
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

        # Combine all sections with base prompt
        if sections:
            memory_context = "\n".join(sections)
            # Inject memory context before the base prompt
            return f"{memory_context}\n---\n\n{base_prompt}"
        else:
            return base_prompt

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
    print(f"Policy: {config.policy_type.upper()}", end="")
    if config.policy_type == "vlm":
        print(f" ({config.policy_provider}/{config.policy_model or 'default'})")
    else:
        print(f" (LLaVA)")
    print(f"Reflection VLM: {config.vlm_provider or 'rule-based'}")
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

        # =====================================================================
        # Display working memory periodically
        # =====================================================================
        if (
            config.show_working_memory
            and debug_logger
            and learning_loop
            and i > 0
            and i % config.working_memory_interval == 0
        ):
            debug_logger.display_working_memory(learning_loop, i)

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

    # Save final hypothesis/principle snapshots
    if debug_logger and learning_loop:
        debug_logger.save_hypotheses_snapshot(learning_loop, config.n_episodes)
        debug_logger.save_principles_snapshot(learning_loop, config.n_episodes)
        debug_logger.display_working_memory(learning_loop, config.n_episodes)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Success Rate: {success_rate:.1%} ({successes}/{config.n_episodes})")
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

    args = parser.parse_args()

    # Build config
    config = ExperimentConfig(
        name=args.name,
        mode=args.mode,
        seed_start=args.seed_start,
        n_episodes=args.n_episodes,
        # Policy options
        policy_type=args.policy_type,
        policy_provider=args.policy_provider,
        policy_model=args.policy_model,
        # BC policy options
        base_model_path=args.base_model,
        use_post_trained=args.use_post_trained,
        # Reflection options
        vlm_provider=args.provider if args.provider != "rule" else None,
        vlm_model=args.model,
        # Output options
        save_dir=args.save_dir,
        verbose=args.verbose,
        show_working_memory=args.show_memory,
        working_memory_interval=args.memory_interval,
    )

    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
