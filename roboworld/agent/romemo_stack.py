"""
RoMemo wrapper that *reuses* the worldmemory/romemo codebase.

This is the aligned implementation (vs the earlier lightweight standalone wrapper):
- Uses romemo.memory.schema.{Experience, MemoryBank}
- Uses romemo.memory.retrieve.Retriever (FAISS if available, numpy fallback otherwise)
- Uses romemo.memory.writeback.WritebackPolicy for optional writeback

We adapt RoMemo's \"option\" concept to reflect-vlm's discrete action strings:
  action := \"<primitive> <color>\" or \"done\"

We treat each decision as an \"option_start\" experience keyed by a deterministic
numeric state embedding from env.get_env_state().

NEW: Supports state-query based retrieval for better task relevance.
- Visual retrieval: original behavior, uses image embeddings
- Symbolic retrieval: filters by discrete task state before ranking
- Hybrid retrieval: combines both approaches

NEW (Phase 2): Principle-based learning from failures.
- On failure: generates reflection and extracts reusable principles
- At action time: retrieves applicable principles to guide decisions
- Principles are abstract rules that transfer across task instances
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

from roboworld.envs.generator import COLORS as _COLORS

try:
    from romemo.memory.schema import Experience, MemoryBank
    from romemo.memory.retrieve import Retriever, RetrievalResult
    from romemo.memory.writeback import WritebackConfig, WritebackPolicy
    from romemo.memory.principle import Principle, PrincipleStore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import romemo. In your reflectvlm env run:\n"
        "  pip install -e /home/haoyang/project/haoyang/worldmemory\n"
        "so `import romemo` works.\n"
        f"Original error: {e}"
    )

# Try to import Reflector for principle extraction
try:
    from roboworld.agent.reflector import Reflector, ReflectionInput, FAIL_TAG_DESCRIPTIONS

    HAS_REFLECTOR = True
except ImportError:
    HAS_REFLECTOR = False
    Reflector = None
    ReflectionInput = None
    FAIL_TAG_DESCRIPTIONS = {}


# =========================================================================
# Symbolic State Extraction for State-Query Retrieval
# =========================================================================


def _get_signature_for_color(env, color: str) -> str:
    """Helper to get shape signature for a color from env."""
    if hasattr(env, "get_signature_for_color"):
        return env.get_signature_for_color(color)
    elif hasattr(env, "color_signature_map"):
        return env.color_signature_map.get(color, f"unknown_{color}")
    return f"unknown_{color}"


def _get_shape_features_for_color(env, color: str) -> Dict[str, Any]:
    """Helper to get shape features for a color from env."""
    if hasattr(env, "get_shape_features"):
        return env.get_shape_features(color)
    return {}


def _get_piece_dependencies(env, color: str) -> Dict[str, Any]:
    """Helper to get dependency info for a piece."""
    if hasattr(env, "get_piece_dependencies"):
        return env.get_piece_dependencies(color)
    return {"blocks": [], "blocked_by": [], "blocks_colors": [], "blocked_by_colors": []}


def extract_symbolic_state(
    env,
    proposed_action: str,
    last_action_success: Optional[bool] = None,
    last_fail_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract discrete symbolic state from the environment.

    This captures the task-relevant state that determines the correct action:
    - What pieces are inserted (progress)
    - What the robot is holding
    - What action type is being attempted
    - Recent failure context

    NEW: Also includes shape-based information that transfers across episodes:
    - Shape signatures (color-independent identifiers)
    - Shape features (dimensions, slots, holes)
    - Dependency information (which pieces block which)

    Args:
        env: FrankaAssemblyEnv instance
        proposed_action: The action being considered (e.g., "insert red")
        last_action_success: Whether the previous action succeeded
        last_fail_tag: Failure tag from previous action (if any)

    Returns:
        Dict with symbolic state fields (both color-based and shape-based)
    """
    # Parse action type from proposed action
    action_type = "unknown"
    target_color = None
    action_str = str(proposed_action).strip().lower()

    if action_str == "done":
        action_type = "done"
    elif action_str.startswith("pick up"):
        action_type = "pick"
        parts = action_str.split()
        if len(parts) >= 3:
            target_color = parts[2]
    elif action_str.startswith("insert"):
        action_type = "insert"
        parts = action_str.split()
        if len(parts) >= 2:
            target_color = parts[1]
    elif action_str.startswith("reorient"):
        action_type = "reorient"
        parts = action_str.split()
        if len(parts) >= 2:
            target_color = parts[1]
    elif action_str.startswith("put down"):
        action_type = "putdown"
        parts = action_str.split()
        if len(parts) >= 3:
            target_color = parts[2]

    # Get holding state
    is_holding = False
    holding_piece = None
    holding_signature = None
    holding_shape_features = {}
    try:
        body = env.get_object_in_hand() if hasattr(env, "get_object_in_hand") else None
        if body is not None:
            is_holding = True
            # Map body name to color
            colors = list(getattr(env, "peg_colors", []))
            names = list(getattr(env, "peg_names", []))
            try:
                idx = names.index(body)
                holding_piece = str(colors[idx])
                # NEW: Get shape signature for held piece
                holding_signature = _get_signature_for_color(env, holding_piece)
                holding_shape_features = _get_shape_features_for_color(env, holding_piece)
            except (ValueError, IndexError):
                holding_piece = body
    except Exception:
        pass

    # Get progress: which pieces are inserted
    inserted_pieces = []
    remaining_pieces = []
    inserted_signatures = []
    remaining_signatures = []
    try:
        colors = list(getattr(env, "peg_colors", []))
        names = list(getattr(env, "peg_names", []))
        for c, n in zip(colors, names):
            try:
                sig = _get_signature_for_color(env, c)
                if env.object_is_success(n):
                    inserted_pieces.append(str(c))
                    inserted_signatures.append(sig)
                else:
                    remaining_pieces.append(str(c))
                    remaining_signatures.append(sig)
            except Exception:
                remaining_pieces.append(str(c))
                remaining_signatures.append(f"unknown_{c}")
    except Exception:
        pass

    total_pieces = len(inserted_pieces) + len(remaining_pieces)
    progress = len(inserted_pieces) / total_pieces if total_pieces > 0 else 0.0

    # NEW: Get target piece information
    target_signature = None
    target_shape_features = {}
    target_dependencies = {
        "blocks": [],
        "blocked_by": [],
        "blocks_colors": [],
        "blocked_by_colors": [],
    }
    if target_color:
        target_signature = _get_signature_for_color(env, target_color)
        target_shape_features = _get_shape_features_for_color(env, target_color)
        target_dependencies = _get_piece_dependencies(env, target_color)

    # NEW: Check if target's dependencies are satisfied
    dependencies_satisfied = True
    unsatisfied_dependencies = []
    if target_dependencies.get("blocked_by"):
        for blocker_sig in target_dependencies["blocked_by"]:
            if blocker_sig not in inserted_signatures:
                dependencies_satisfied = False
                unsatisfied_dependencies.append(blocker_sig)

    return {
        # === ACTION INFO ===
        "action_type": action_type,
        # === COLOR-BASED (for backward compatibility and visual grounding) ===
        "target_color": target_color,
        "holding_piece": holding_piece,  # color of held piece
        "inserted_pieces": inserted_pieces,  # colors
        "remaining_pieces": remaining_pieces,  # colors
        # === SHAPE-BASED (transferable across episodes) ===
        "target_signature": target_signature,
        "target_shape_features": target_shape_features,
        "holding_signature": holding_signature,
        "holding_shape_features": holding_shape_features,
        "inserted_signatures": inserted_signatures,
        "remaining_signatures": remaining_signatures,
        # === DEPENDENCY INFO (the key to correct ordering!) ===
        "target_blocks": target_dependencies.get("blocks", []),  # signatures this piece blocks
        "target_blocked_by": target_dependencies.get(
            "blocked_by", []
        ),  # signatures that block this
        "target_blocks_colors": target_dependencies.get("blocks_colors", []),
        "target_blocked_by_colors": target_dependencies.get("blocked_by_colors", []),
        "dependencies_satisfied": dependencies_satisfied,
        "unsatisfied_dependencies": unsatisfied_dependencies,
        # === PROGRESS ===
        "is_holding": is_holding,
        "progress": progress,
        "num_remaining": len(remaining_pieces),
        "num_inserted": len(inserted_pieces),
        # === FAILURE CONTEXT ===
        "last_action_success": last_action_success,
        "last_fail_tag": last_fail_tag,
    }


def symbolic_state_matches(
    query: Dict[str, Any],
    stored: Dict[str, Any],
    strict_action_type: bool = True,
    strict_holding: bool = True,
    use_shape_matching: bool = True,
) -> bool:
    """
    Check if two symbolic states are compatible for retrieval.

    Args:
        query: The current symbolic state
        stored: The stored experience's symbolic state
        strict_action_type: Require exact action type match
        strict_holding: Require exact holding state match
        use_shape_matching: Use shape signatures instead of colors (recommended)

    Returns:
        True if states are compatible
    """
    if stored is None:
        return False

    # Must match action type (most important filter)
    if strict_action_type:
        if query.get("action_type") != stored.get("action_type"):
            return False

    # Must match holding state
    if strict_holding:
        if query.get("is_holding") != stored.get("is_holding"):
            return False

    # Progress should be somewhat similar (within ±2 pieces)
    q_remaining = query.get("num_remaining", 0)
    s_remaining = stored.get("num_remaining", 0)
    if abs(q_remaining - s_remaining) > 2:
        return False

    # NEW: Shape-based matching (transfers across episodes!)
    if use_shape_matching:
        # Match by target shape signature if available
        q_target_sig = query.get("target_signature")
        s_target_sig = stored.get("target_signature")

        if q_target_sig and s_target_sig:
            # Extract shape category from signature (e.g., "block_25x4x8_elongated" -> "elongated")
            q_aspect = _extract_aspect_from_signature(q_target_sig)
            s_aspect = _extract_aspect_from_signature(s_target_sig)

            # Match on aspect ratio (elongated, tall, square, etc.)
            if q_aspect and s_aspect and q_aspect != s_aspect:
                return False

        # Match on holding shape signature if holding
        if query.get("is_holding") and stored.get("is_holding"):
            q_hold_sig = query.get("holding_signature")
            s_hold_sig = stored.get("holding_signature")

            if q_hold_sig and s_hold_sig:
                q_hold_aspect = _extract_aspect_from_signature(q_hold_sig)
                s_hold_aspect = _extract_aspect_from_signature(s_hold_sig)

                if q_hold_aspect and s_hold_aspect and q_hold_aspect != s_hold_aspect:
                    return False

        # Match on dependency satisfaction status
        q_deps_sat = query.get("dependencies_satisfied")
        s_deps_sat = stored.get("dependencies_satisfied")

        if q_deps_sat is not None and s_deps_sat is not None:
            if q_deps_sat != s_deps_sat:
                return False

    return True


def _extract_aspect_from_signature(signature: str) -> Optional[str]:
    """Extract aspect ratio from shape signature."""
    if not signature:
        return None

    aspects = ["elongated", "tall", "square", "flat", "rectangular"]
    for aspect in aspects:
        if aspect in signature:
            return aspect
    return None


def symbolic_state_similarity(
    query: Dict[str, Any],
    stored: Dict[str, Any],
) -> float:
    """
    Compute similarity score between two symbolic states.

    Returns a score from 0.0 to 1.0 where higher is more similar.
    This is useful for ranking when multiple experiences match.
    """
    if stored is None:
        return 0.0

    score = 0.0
    max_score = 0.0

    # Action type match (weight: 3)
    max_score += 3.0
    if query.get("action_type") == stored.get("action_type"):
        score += 3.0

    # Holding state match (weight: 2)
    max_score += 2.0
    if query.get("is_holding") == stored.get("is_holding"):
        score += 2.0

    # Progress similarity (weight: 2)
    max_score += 2.0
    q_remaining = query.get("num_remaining", 0)
    s_remaining = stored.get("num_remaining", 0)
    progress_diff = abs(q_remaining - s_remaining)
    if progress_diff == 0:
        score += 2.0
    elif progress_diff == 1:
        score += 1.5
    elif progress_diff == 2:
        score += 1.0

    # Shape signature match (weight: 3) - THE KEY FOR TRANSFER!
    max_score += 3.0
    q_target_sig = query.get("target_signature", "")
    s_target_sig = stored.get("target_signature", "")

    if q_target_sig and s_target_sig:
        # Full signature match
        if q_target_sig == s_target_sig:
            score += 3.0
        else:
            # Partial match (same aspect ratio)
            q_aspect = _extract_aspect_from_signature(q_target_sig)
            s_aspect = _extract_aspect_from_signature(s_target_sig)
            if q_aspect and s_aspect and q_aspect == s_aspect:
                score += 1.5

    # Dependency satisfaction match (weight: 2)
    max_score += 2.0
    if query.get("dependencies_satisfied") == stored.get("dependencies_satisfied"):
        score += 2.0

    # Holding signature match (weight: 1)
    max_score += 1.0
    q_hold_sig = query.get("holding_signature", "")
    s_hold_sig = stored.get("holding_signature", "")
    if q_hold_sig and s_hold_sig and q_hold_sig == s_hold_sig:
        score += 1.0

    return score / max_score if max_score > 0 else 0.0


def extract_env_state_vec(env) -> np.ndarray:
    """
    Deterministic numeric context embedding using env.get_env_state().
    Note: Retriever uses L2 distance; we normalize vectors so L2~cosine.

    IMPORTANT: The raw MuJoCo state (qpos/qvel/eq arrays) changes length across different
    boards (different numbers of pegs/geoms/constraints). RoMemo's FAISS/numpy index requires
    a fixed embedding dimension. So we build a fixed-size vector from:
      - robot joint qpos/qvel (first N DoFs)
      - mocap pose
      - previous action + path length
      - per-color object pose/status for a canonical color palette
    """

    canonical_colors = tuple(_COLORS.keys())

    robot_dof = int(len(getattr(env, "robot_init_qpos", [])) or 9)
    action_dim = int(len(getattr(env, "prev_action", np.zeros((8,), dtype=np.float32))) or 8)

    # Defaults (fixed size)
    robot_qpos = np.zeros((robot_dof,), dtype=np.float32)
    robot_qvel = np.zeros((robot_dof,), dtype=np.float32)
    mocap_pos = np.zeros((3,), dtype=np.float32)
    mocap_quat = np.zeros((4,), dtype=np.float32)
    prev_action = np.zeros((action_dim,), dtype=np.float32)
    curr_path_length = np.zeros((1,), dtype=np.float32)

    # Read env state (best-effort)
    try:
        st = env.get_env_state()
    except Exception:
        st = {}
    qpos, qvel = st.get("joint", (None, None))
    _mpos, _mquat = st.get("mocap", (None, None))

    if qpos is not None:
        v = np.asarray(qpos, dtype=np.float32).reshape(-1)
        robot_qpos[: min(robot_dof, v.shape[0])] = v[:robot_dof]
    if qvel is not None:
        v = np.asarray(qvel, dtype=np.float32).reshape(-1)
        robot_qvel[: min(robot_dof, v.shape[0])] = v[:robot_dof]
    if _mpos is not None:
        v = np.asarray(_mpos, dtype=np.float32).reshape(-1)
        mocap_pos[: min(3, v.shape[0])] = v[:3]
    if _mquat is not None:
        v = np.asarray(_mquat, dtype=np.float32).reshape(-1)
        mocap_quat[: min(4, v.shape[0])] = v[:4]
    if "prev_action" in st and st["prev_action"] is not None:
        v = np.asarray(st["prev_action"], dtype=np.float32).reshape(-1)
        prev_action[: min(action_dim, v.shape[0])] = v[:action_dim]
    if "curr_path_length" in st and st["curr_path_length"] is not None:
        curr_path_length[0] = float(st["curr_path_length"])

    # Per-color object features (fixed palette)
    obj_feat_parts: List[np.ndarray] = []
    peg_colors = list(getattr(env, "peg_colors", []) or [])
    peg_names = list(getattr(env, "peg_names", []) or [])
    color_to_body: Dict[str, str] = {}
    for c, n in zip(peg_colors, peg_names):
        if isinstance(c, str) and isinstance(n, str):
            color_to_body[c] = n

    for color in canonical_colors:
        body = color_to_body.get(str(color))
        if body is None:
            obj_feat_parts.append(np.zeros((11,), dtype=np.float32))
            continue

        present = 1.0
        in_hand = 0.0
        upright = 0.0
        success = 0.0
        pos = np.zeros((3,), dtype=np.float32)
        quat = np.zeros((4,), dtype=np.float32)

        try:
            if hasattr(env, "object_is_in_hand"):
                in_hand = 1.0 if bool(env.object_is_in_hand(body)) else 0.0
            if hasattr(env, "object_is_upright"):
                upright = 1.0 if bool(env.object_is_upright(body)) else 0.0
            if hasattr(env, "object_is_success"):
                success = 1.0 if bool(env.object_is_success(body)) else 0.0
            if hasattr(env, "get_body_pos"):
                p = np.asarray(env.get_body_pos(body), dtype=np.float32).reshape(-1)
                pos[: min(3, p.shape[0])] = p[:3]
            if hasattr(env, "get_body_quat"):
                q = np.asarray(env.get_body_quat(body), dtype=np.float32).reshape(-1)
                quat[: min(4, q.shape[0])] = q[:4]
        except Exception:
            # Keep defaults; this is best-effort and must not crash rollout.
            pass

        obj_feat_parts.append(
            np.concatenate(
                [
                    np.asarray([present, in_hand, upright, success], dtype=np.float32),
                    pos,
                    quat,
                ],
                axis=0,
            )
        )

    return np.concatenate(
        [robot_qpos, robot_qvel, mocap_pos, mocap_quat, prev_action, curr_path_length]
        + obj_feat_parts,
        axis=0,
    ).astype(np.float32, copy=False)


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32, copy=False)


def state_hash_from_vec(state_vec: np.ndarray, quant: float = 1e-3) -> str:
    sv = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    q = np.round(sv / float(quant)).astype(np.int32)
    return hashlib.md5(q.tobytes()).hexdigest()[:16]


def _dist_to_sim(dist: float, eps: float = 1e-6) -> float:
    # Retriever returns L2 distances. Convert to a similarity-like weight.
    s = 1.0 / (float(dist) + float(eps))
    # Prevent extreme dominance when dist==0 (identical state).
    return float(min(100.0, s))


@dataclass(frozen=True)
class RoMemoDiscreteConfig:
    k: int = 20
    alpha: float = 0.7
    lambda_fail: float = 1.5
    # When a retrieved memory is a failure and contains an oracle correction action,
    # boost that oracle action by this multiplier (averaged over retrieved items).
    lambda_correct: float = 1.0
    # Optionally treat certain failure tags as "hard constraints" and penalize
    # the taken action more aggressively.
    lambda_fail_hard: float = 2.5
    beta_failrate: float = 0.5
    beta_repeat: float = 0.2
    max_mem_candidates: int = 5
    quant_for_hash: float = 1e-3
    write_on_success: bool = True
    write_on_failure: bool = True
    deduplicate: bool = False  # keep repeated evidence; we also keep explicit repeat counters

    # NEW: Retrieval mode for state-query based retrieval
    # Options:
    #   "visual" - original behavior, uses image/state embeddings only
    #   "symbolic" - filters by symbolic state first, then ranks by embedding
    #   "hybrid" - combines both approaches with weighting
    retrieval_mode: str = "visual"  # "visual", "symbolic", or "hybrid"

    # For hybrid mode: weight for symbolic filtering (0=visual only, 1=symbolic only)
    symbolic_weight: float = 0.5

    # Minimum candidates before falling back to visual retrieval
    min_symbolic_candidates: int = 5

    # =========================================================================
    # NEW (Phase 2): Principle-based learning configuration
    # =========================================================================

    # Enable principle extraction and usage
    use_principles: bool = False

    # Reflector VLM provider for generating reflections ("openai", "gemini", "qwen", None)
    # If None, uses rule-based fallback (no API calls)
    reflector_provider: Optional[str] = None
    reflector_model: Optional[str] = None

    # Principle retrieval settings
    max_principles_in_prompt: int = 3  # Max principles to include in action prompt
    min_principle_confidence: float = 0.3  # Minimum confidence for principle retrieval

    # Principle penalty settings
    # When a proposed action violates a known principle, apply this penalty
    principle_violation_penalty: float = 2.0

    # When a principle suggests an alternative action, boost it by this amount
    principle_boost: float = 1.0

    # Path to save/load principle store (optional)
    principle_store_path: Optional[str] = None


class RoMemoDiscreteAgent:
    """
    Base-agent wrapper: retrieval-conditioned rerank + optional writeback.

    base_agent.act(img, goal_img, inp, ...) -> action_str

    NEW: Supports multiple retrieval modes:
    - "visual": original behavior, uses image embeddings
    - "symbolic": filters by discrete task state before ranking
    - "hybrid": combines both approaches

    NEW (Phase 2): Principle-based learning
    - On failure: reflects and extracts reusable principles
    - At action time: retrieves applicable principles to guide decisions
    """

    def __init__(
        self,
        base_agent,
        env,
        task: str = "assembly",
        cfg: Optional[RoMemoDiscreteConfig] = None,
        writeback: bool = False,
        seed: int = 0,
        shared_store: Optional["RoMemoStore"] = None,
        use_vision_retrieval: bool = True,
    ):
        self.base_agent = base_agent
        self.env = env
        self.task = str(task)
        self.cfg = cfg or RoMemoDiscreteConfig()
        self.writeback_enabled = bool(writeback)
        self.use_vision = use_vision_retrieval

        # Retrieval mode (from config)
        self.retrieval_mode = self.cfg.retrieval_mode

        # Log retrieval mode
        mode_str = self.retrieval_mode
        if mode_str == "visual":
            print("[RoMemo] Visual-based retrieval enabled.")
        elif mode_str == "symbolic":
            print("[RoMemo] Symbolic state-query retrieval enabled.")
        elif mode_str == "hybrid":
            print(
                f"[RoMemo] Hybrid retrieval enabled (symbolic_weight={self.cfg.symbolic_weight})."
            )
        else:
            print(f"[RoMemo] Unknown retrieval mode '{mode_str}', defaulting to visual.")
            self.retrieval_mode = "visual"

        self.store = shared_store or RoMemoStore(
            task=self.task,
            cfg=self.cfg,
            writeback=bool(self.writeback_enabled),
            seed=int(seed),
        )

        self._pending: Optional[Experience] = None
        self._pending_pred: Optional[bool] = None
        self._last_context_hash: Optional[str] = None
        self._pending_symbolic_state: Optional[Dict[str, Any]] = None

        # Track last action outcome for symbolic state context
        self._last_action_success: Optional[bool] = None
        self._last_fail_tag: Optional[str] = None

        # trace payload compatible with run.py logging
        self.last_trace: Dict[str, Any] = {}

        # =========================================================================
        # NEW (Phase 2): Principle-based learning
        # =========================================================================
        self.use_principles = bool(self.cfg.use_principles)
        self.principle_store: Optional[PrincipleStore] = None
        self.reflector: Optional[Reflector] = None
        self._action_history: List[str] = []  # Track action history for reflection
        self._last_image: Optional[np.ndarray] = None  # Cache last image for reflection

        if self.use_principles:
            self._init_principle_system()

    def _init_principle_system(self) -> None:
        """Initialize principle store and reflector."""
        # Initialize principle store
        self.principle_store = PrincipleStore(name=f"principles_{self.task}")

        # Try to load existing principles
        if self.cfg.principle_store_path:
            path = Path(self.cfg.principle_store_path)
            if path.exists():
                try:
                    if path.suffix == ".pt":
                        self.principle_store = PrincipleStore.load_pt(path)
                    else:
                        self.principle_store = PrincipleStore.load(path)
                    print(f"[RoMemo] Loaded {len(self.principle_store)} principles from {path}")
                except Exception as e:
                    print(f"[RoMemo] Warning: Failed to load principles: {e}")

        # Initialize reflector
        if HAS_REFLECTOR:
            try:
                if self.cfg.reflector_provider:
                    # Use VLM-based reflector
                    self.reflector = Reflector.create(
                        provider=self.cfg.reflector_provider,
                        model=self.cfg.reflector_model,
                        mode="oracle_guided",
                        verbose=False,
                    )
                    print(f"[RoMemo] VLM Reflector initialized ({self.cfg.reflector_provider})")
                else:
                    # Use rule-based reflector (no API calls)
                    self.reflector = Reflector(mode="oracle_guided", verbose=False)
                    print("[RoMemo] Rule-based Reflector initialized")
            except Exception as e:
                print(f"[RoMemo] Warning: Failed to initialize reflector: {e}")
                self.reflector = Reflector(mode="oracle_guided", verbose=False)
        else:
            print("[RoMemo] Warning: Reflector not available, principles disabled")
            self.use_principles = False

        print(
            f"[RoMemo] Principle-based learning {'enabled' if self.use_principles else 'disabled'}"
        )

    def _get_applicable_principles(
        self,
        action_type: str,
        symbolic_state: Optional[Dict[str, Any]] = None,
    ) -> List[Principle]:
        """
        Retrieve principles applicable to the current context.

        Args:
            action_type: Type of action being considered ("pick", "insert", etc.)
            symbolic_state: Current symbolic state

        Returns:
            List of applicable principles sorted by confidence
        """
        if not self.use_principles or self.principle_store is None:
            return []

        # Retrieve by action type
        principles = self.principle_store.retrieve(
            action_type=action_type,
            min_confidence=float(self.cfg.min_principle_confidence),
            top_k=int(self.cfg.max_principles_in_prompt),
        )

        return principles

    def _format_principles_for_prompt(self, principles: List[Principle]) -> str:
        """Format principles as text for inclusion in VLM prompt."""
        if not principles:
            return ""

        lines = []
        for i, p in enumerate(principles, 1):
            conf_str = "HIGH" if p.confidence > 0.7 else "MEDIUM" if p.confidence > 0.4 else "LOW"
            lines.append(f"{i}. [{conf_str}] {p.content}")

        return "\n".join(lines)

    def _check_principle_violations(
        self,
        proposed_action: str,
        principles: List[Principle],
        symbolic_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Optional[str]]:
        """
        Check if the proposed action violates any known principles.

        Returns:
            Tuple of (penalty, suggested_alternative_action)
        """
        if not principles:
            return 0.0, None

        penalty = 0.0
        suggested_action = None

        action_type = self._extract_action_type(proposed_action)

        for principle in principles:
            # Check for action type match
            if action_type not in principle.action_types:
                continue

            # Check for trigger condition matches
            if symbolic_state and principle.trigger_conditions:
                # Check specific conditions
                triggers = principle.trigger_conditions

                # Example: "piece_has_dependencies" and trying to insert
                if "piece_has_dependencies" in triggers and action_type == "insert":
                    # This principle warns about dependency issues
                    # Check if we might be violating it
                    if symbolic_state.get("num_remaining", 0) > 1:
                        # Multiple pieces remaining - dependency might matter
                        penalty += float(self.cfg.principle_violation_penalty) * 0.5

                # "gripper_occupied" and trying to pick
                if "gripper_occupied" in triggers and action_type == "pick":
                    if symbolic_state.get("is_holding", False):
                        penalty += float(self.cfg.principle_violation_penalty)

                # "piece_not_upright" and trying to insert
                if "piece_not_upright" in triggers and action_type == "insert":
                    # This principle says to reorient first
                    # Suggest reorient as alternative
                    target = proposed_action.split()[-1] if " " in proposed_action else None
                    if target:
                        suggested_action = f"reorient {target}"
                        penalty += float(self.cfg.principle_violation_penalty)

        return penalty, suggested_action

    def _extract_action_type(self, action: str) -> str:
        """Extract action type from action string."""
        action = str(action).strip().lower()
        if action == "done":
            return "done"
        elif action.startswith("pick up"):
            return "pick"
        elif action.startswith("insert"):
            return "insert"
        elif action.startswith("reorient"):
            return "reorient"
        elif action.startswith("put down"):
            return "putdown"
        return "unknown"

    def _reflect_on_failure(
        self,
        failed_action: str,
        fail_tag: str,
        oracle_action: Optional[str],
        symbolic_state: Optional[Dict[str, Any]],
        image: Optional[np.ndarray] = None,
    ) -> Optional[Principle]:
        """
        Generate reflection on failure and extract principle.

        Args:
            failed_action: The action that failed
            fail_tag: Failure type tag
            oracle_action: The correct action (if known)
            symbolic_state: Current symbolic state
            image: Current observation image

        Returns:
            New or updated Principle, or None if reflection failed
        """
        if not self.use_principles or self.reflector is None or self.principle_store is None:
            return None

        try:
            # Build reflection input
            input_data = ReflectionInput(
                failed_action=failed_action,
                fail_tag=fail_tag,
                oracle_action=oracle_action,
                symbolic_state=symbolic_state or {},
                action_history=self._action_history[-10:],
                experience_id=f"exp_{self._last_context_hash or 'unknown'}",
                image=image,
            )

            # Generate reflection
            output = self.reflector.reflect(input_data)

            if output and output.general_principle:
                # Update principle store
                pid = self.principle_store.update_from_reflection(
                    output.to_dict(),
                    input_data.experience_id,
                )

                # Return the updated principle
                principle = self.principle_store.get(pid)
                if principle:
                    print(f"[RoMemo] Learned principle: {principle.content[:60]}...")
                    return principle

        except Exception as e:
            print(f"[RoMemo] Warning: Reflection failed: {e}")

        return None

    def save_principles(self, path: Optional[str] = None) -> None:
        """Save principle store to file."""
        if self.principle_store is None:
            return

        save_path = Path(path or self.cfg.principle_store_path or "principles.json")
        try:
            if save_path.suffix == ".pt":
                self.principle_store.save_pt(save_path)
            else:
                self.principle_store.save(save_path)
            print(f"[RoMemo] Saved {len(self.principle_store)} principles to {save_path}")
        except Exception as e:
            print(f"[RoMemo] Warning: Failed to save principles: {e}")

    def get_principle_stats(self) -> Dict[str, Any]:
        """Get statistics about learned principles."""
        if self.principle_store is None:
            return {"enabled": False}

        stats = self.principle_store.get_stats()
        stats["enabled"] = True
        if self.reflector:
            stats["reflector_stats"] = self.reflector.get_stats()
        return stats

    def set_env(self, env) -> None:
        self.env = env

    def set_base_agent(self, base_agent) -> None:
        self.base_agent = base_agent

    def _mem_action_scores(
        self, ret: RetrievalResult
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        # Note: we compute a per-action *average* similarity score (not a raw sum).
        # Raw sums can explode with duplicate/near-duplicate memories and dominate
        # the base policy even after repeated failures, causing unrecoverable loops.
        scores: Dict[str, float] = {}
        stats: Dict[str, Dict[str, Any]] = {}
        # Separate accumulator for oracle-correction boosts.
        corr_stats: Dict[str, Dict[str, Any]] = {}

        # Failure tags that should be treated as hard constraints (stronger veto).
        hard_tags = {
            "BLOCKED_BY_PREDECESSOR",
            "BLOCKED_BY_SUCCESSOR",
            "BAD_PLACEMENT",
            "NEEDS_REORIENT",
            "HAND_FULL",
            "GRASP_FAILED",
            "INSERT_TIMEOUT",
        }

        for exp, dist in zip(ret.experiences, ret.distances.tolist()):
            dec = (exp.extra_metrics or {}).get("decision", {})
            action = None
            if isinstance(dec, dict):
                action = dec.get("action", None)
            if action is None:
                action = (exp.extra_metrics or {}).get("action", None)
            if action is None:
                continue

            sim = _dist_to_sim(float(dist))
            # Determine fail severity via fail_tag when available.
            tag = getattr(exp, "fail_tag", None)
            if tag is None:
                tag = (exp.extra_metrics or {}).get("fail_tag", None)

            if bool(exp.success):
                w = 1.0
            else:
                w = (
                    -float(self.cfg.lambda_fail_hard)
                    if str(tag) in hard_tags
                    else -float(self.cfg.lambda_fail)
                )

            st = stats.get(str(action))
            if st is None:
                st = {"n": 0, "n_succ": 0, "n_fail": 0, "sim_sum": 0.0, "signed_sim_sum": 0.0}
                stats[str(action)] = st
            st["n"] += 1
            st["sim_sum"] += float(sim)
            st["signed_sim_sum"] += float(sim) * float(w)
            if bool(exp.success):
                st["n_succ"] += 1
            if bool(exp.fail):
                st["n_fail"] += 1

            # NEW: "Correction" signal from failure memories.
            # If memory says action_taken failed, and provides oracle_action (what to do instead),
            # boost that oracle action so it can be selected even if base policy didn't propose it.
            if bool(exp.fail):
                oracle_action = None
                if exp.extra_metrics:
                    oracle_action = exp.extra_metrics.get("oracle_action", None)
                if oracle_action is not None:
                    corr = str(oracle_action).strip()
                    if corr:
                        cst = corr_stats.get(corr)
                        if cst is None:
                            cst = {"n_oracle": 0, "sim_sum": 0.0}
                            corr_stats[corr] = cst
                        cst["n_oracle"] += 1
                        cst["sim_sum"] += float(sim)

        for a, st in stats.items():
            n = max(1, int(st["n"]))
            st["fail_rate"] = float(int(st["n_fail"]) / n)
            scores[a] = float(st.get("signed_sim_sum", 0.0)) / float(n)

        # Apply oracle-correction boosts (average, not sum).
        for a, cst in corr_stats.items():
            n_or = max(1, int(cst.get("n_oracle", 0)))
            bonus = float(self.cfg.lambda_correct) * float(cst.get("sim_sum", 0.0)) / float(n_or)
            scores[a] = float(scores.get(a, 0.0)) + float(bonus)
            # Attach correction stats for debugging (and so downstream can see it's boosted).
            st = stats.get(a)
            if st is None:
                st = {"n": 0, "n_succ": 0, "n_fail": 0, "sim_sum": 0.0, "signed_sim_sum": 0.0}
                stats[a] = st
            st["n_oracle"] = int(cst.get("n_oracle", 0))
            # For actions that only appear as oracle corrections, define fail_rate=0.
            if "fail_rate" not in st:
                st["fail_rate"] = float(0.0)
        return scores, stats

    def _feasible_actions(self) -> Optional[set[str]]:
        """Best-effort legality filter to prevent memory from proposing invalid primitives.

        This matches the discrete action semantics used by the oracle/policies:
        - If not holding: only "pick up <color>" for not-yet-done colors (+ "done")
        - If holding: allow {insert,reorient,put down} for the held color (+ "done")
        """
        try:
            colors = list(getattr(self.env, "peg_colors", []))
            names = list(getattr(self.env, "peg_names", []))
            done_mask: Dict[str, bool] = {}
            for c, name in zip(colors, names):
                try:
                    done_mask[str(c)] = bool(self.env.object_is_success(name))
                except Exception:
                    done_mask[str(c)] = False

            body = (
                self.env.get_object_in_hand() if hasattr(self.env, "get_object_in_hand") else None
            )
            held: Optional[str] = None
            if body is not None:
                try:
                    i = names.index(body)
                    held = str(colors[i])
                except Exception:
                    held = None

            feasible: set[str] = {"done"}
            if held is None:
                for c in colors:
                    if not done_mask.get(str(c), False):
                        feasible.add(f"pick up {c}")
            else:
                feasible.add(f"insert {held}")
                feasible.add(f"reorient {held}")
                feasible.add(f"put down {held}")
            return feasible
        except Exception:
            return None

    def _choose(
        self,
        base_action: str,
        scores: Dict[str, float],
        stats: Dict[str, Dict[str, Any]],
        ctx_hash: str,
        principles: Optional[List[Principle]] = None,
        symbolic_state: Optional[Dict[str, Any]] = None,
    ):
        cand = {str(base_action)}
        top_mem = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            : int(self.cfg.max_mem_candidates)
        ]
        for a, _ in top_mem:
            cand.add(str(a))

        feasible = self._feasible_actions()
        if feasible is not None:
            cand = {a for a in cand if a in feasible or a == str(base_action)}
            if not cand:
                cand = {str(base_action)}

        # Normalize memory scores to [0, 1] to prevent overpowering base_prior
        raw_mem_scores = {a: float(scores.get(a, 0.0)) for a in cand}
        max_mem = max(raw_mem_scores.values()) if raw_mem_scores else 1.0
        if max_mem < 1e-6:
            max_mem = 1.0

        # NEW: Calculate principle-based adjustments
        principle_penalties: Dict[str, float] = {}
        principle_suggested: Optional[str] = None

        if principles and self.use_principles:
            for a in cand:
                penalty, suggested = self._check_principle_violations(a, principles, symbolic_state)
                principle_penalties[a] = penalty
                if suggested and suggested not in cand:
                    # Add principle-suggested alternative to candidates
                    if feasible is None or suggested in feasible:
                        cand.add(suggested)
                        raw_mem_scores[suggested] = 0.0  # No memory evidence
                        principle_suggested = suggested

        ranked: List[Tuple[str, float]] = []
        for a in cand:
            base_prior = 1.0 if a == str(base_action) else 0.0
            mem = raw_mem_scores.get(a, 0.0) / max_mem
            fail_rate = float(stats.get(a, {}).get("fail_rate", 0.0))
            rep = int(self.store.repeat_fail_counts.get((ctx_hash, a), 0))
            risk = float(self.cfg.beta_failrate) * fail_rate + float(self.cfg.beta_repeat) * float(
                rep
            )

            # NEW: Apply principle-based penalties
            principle_risk = float(principle_penalties.get(a, 0.0))

            # NEW: Boost principle-suggested actions
            principle_bonus = 0.0
            if a == principle_suggested:
                principle_bonus = float(self.cfg.principle_boost)

            s = (
                float(self.cfg.alpha) * base_prior
                + float(1.0 - self.cfg.alpha) * mem
                - risk
                - principle_risk
                + principle_bonus
            )

            # HACK: Penalize "done" if base agent didn't propose it
            if a == "done" and str(base_action) != "done":
                s -= 10  # massive penalty
            ranked.append((a, s))
        ranked.sort(key=lambda x: x[1], reverse=True)
        chosen = ranked[0][0] if ranked else str(base_action)
        return chosen, ranked, top_mem, principle_suggested

    def act(self, img, goal_img, inp, next_image=None):
        # Cache image for potential reflection later
        if isinstance(img, np.ndarray):
            self._last_image = img.copy()
        else:
            self._last_image = None

        # context embedding: use vision if available, else fallback to state
        if self.use_vision:
            try:
                # Use VLM's visual encoder
                raw = self.base_agent.encode_image(img)
                state_vec = _l2_normalize(raw)  # safe to normalize again
                ctx_hash = state_hash_from_vec(raw, quant=self.cfg.quant_for_hash)
            except Exception as e:
                print(f"[RoMemo] Warning: encode_image failed ({e}), fallback to state-based")
                raw = extract_env_state_vec(self.env)
                state_vec = _l2_normalize(raw)
                ctx_hash = state_hash_from_vec(raw, quant=self.cfg.quant_for_hash)
        else:
            # Fallback to state vectors
            raw = extract_env_state_vec(self.env)
            state_vec = _l2_normalize(raw)
            ctx_hash = state_hash_from_vec(raw, quant=self.cfg.quant_for_hash)

        self._last_context_hash = ctx_hash

        # base proposal
        base_action = str(self.base_agent.act(img, goal_img, inp, next_image=next_image)).strip()

        # NEW: Extract symbolic state for state-query retrieval
        symbolic_state = None
        if self.retrieval_mode in ("symbolic", "hybrid") or self.use_principles:
            symbolic_state = extract_symbolic_state(
                self.env,
                base_action,
                last_action_success=self._last_action_success,
                last_fail_tag=self._last_fail_tag,
            )
        self._pending_symbolic_state = symbolic_state

        # NEW (Phase 2): Retrieve applicable principles
        principles: List[Principle] = []
        if self.use_principles:
            action_type = self._extract_action_type(base_action)
            principles = self._get_applicable_principles(action_type, symbolic_state)

        # Retrieve based on mode
        if self.retrieval_mode == "symbolic":
            # Symbolic: filter by discrete state first, then rank by embedding
            ret = self.store.retriever.retrieve_filtered(
                query=state_vec,
                symbolic_state=symbolic_state,
                k=int(self.cfg.k),
                fallback_to_visual=True,
                min_candidates=int(self.cfg.min_symbolic_candidates),
            )
        elif self.retrieval_mode == "hybrid":
            # Hybrid: combine symbolic filtering with visual similarity
            ret = self.store.retriever.retrieve_hybrid(
                query=state_vec,
                symbolic_state=symbolic_state,
                k=int(self.cfg.k),
                symbolic_weight=float(self.cfg.symbolic_weight),
            )
        else:
            # Visual: original behavior
            ret = self.store.retriever.retrieve(query=state_vec, k=int(self.cfg.k))

        scores, stats = self._mem_action_scores(ret)

        # NEW: Pass principles to _choose for constraint checking
        chosen, ranked, top_mem, principle_suggested = self._choose(
            base_action,
            scores,
            stats,
            ctx_hash,
            principles=principles,
            symbolic_state=symbolic_state,
        )

        # Track action history for reflection
        self._action_history.append(str(chosen))
        if len(self._action_history) > 20:
            self._action_history = self._action_history[-20:]

        # create pending experience (aligned with RoMemo pending/update pattern)
        self._pending = Experience(
            task=self.task,
            subtask="discrete_action",
            env_id="reflect-vlm",
            state_vec=state_vec.astype(np.float32, copy=False),
            symbolic_state=symbolic_state,  # NEW: store symbolic state
            strategy_id="romemo_wrapper",
            success=False,
            fail=False,
            steps=0,
            reward=0.0,
            extra_metrics={
                "step_type": "option_start",
                "context_hash": ctx_hash,
                "decision": {
                    "action": str(chosen),
                    "base_action": str(base_action),
                },
                "retrieval": {
                    "k": int(self.cfg.k),
                    "n": int(len(ret.experiences)),
                    "mode": self.retrieval_mode,  # NEW: log retrieval mode
                },
                # NEW: Track principle usage
                "principles_used": len(principles) if principles else 0,
                "principle_suggested": principle_suggested,
            },
        )

        # prediction proxy: whether chosen action looks good from memory
        predicted = None
        if ranked:
            predicted = bool(ranked[0][1] > 0.0)
        self._pending_pred = predicted

        # trace for logging
        self.last_trace = {
            "context_hash": ctx_hash,
            "base_action": str(base_action),
            "chosen_action": str(chosen),
            "retrieval_k": int(self.cfg.k),
            "retrieval_mode": self.retrieval_mode,  # NEW
            "memory_size": int(len(self.store.memory)),
            "symbolic_state": symbolic_state,  # NEW: for debugging
            # NEW (Phase 2): Principle info
            "principles": {
                "enabled": self.use_principles,
                "count": len(principles) if principles else 0,
                "suggested_action": principle_suggested,
                "store_size": len(self.principle_store) if self.principle_store else 0,
                "applied": [
                    {
                        "content": p.content[:80] + "..." if len(p.content) > 80 else p.content,
                        "confidence": p.confidence,
                        "action_types": p.action_types,
                    }
                    for p in (principles or [])[:3]
                ],
            },
            "retrieved": [
                {
                    "dist": float(d),
                    "action": (e.extra_metrics or {}).get("decision", {}).get("action", None)
                    if isinstance((e.extra_metrics or {}).get("decision", {}), dict)
                    else (e.extra_metrics or {}).get("action", None),
                    "success": bool(e.success),
                    "fail": bool(e.fail),
                    "fail_tag": getattr(e, "fail_tag", None),
                    "oracle_action": (e.extra_metrics or {}).get("oracle_action", None),
                    # NEW: Include stored symbolic state for debugging
                    "symbolic_match": bool(
                        symbolic_state_matches(symbolic_state, e.symbolic_state)
                        if symbolic_state and e.symbolic_state
                        else None
                    ),
                }
                for e, d in zip(ret.experiences, ret.distances.tolist())
            ][:10],
            "mem_action_scores": [
                {
                    "action": a,
                    "mem_score": float(scores.get(a, 0.0)),
                    "fail_rate": float(stats.get(a, {}).get("fail_rate", 0.0)),
                }
                for a, _ in top_mem
            ],
            "rerank": {
                "candidates": [{"action": a, "score": float(s)} for a, s in ranked[:10]],
            },
        }

        return str(chosen)

    def update(
        self,
        executed_action: str,
        err_code: int,
        episode_id: Optional[int] = None,
        step_id: Optional[int] = None,
        fail_tag: Optional[str] = None,
        history: Optional[List[str]] = None,
        oracle_state_context: Optional[str] = None,
        oracle_action: Optional[str] = None,
    ):
        """
        Align with RoMemo update_on_failure: write an Experience after outcome is observed.
        We treat err_code!=0 or (done but not success) as failure.

        Enhanced for Failure Memory Collection:
        - fail_tag: explicit constraint tag from Oracle diagnosis (e.g., "BLOCKED_BY_PREDECESSOR")
        - history: full action history for this episode (for context)
        - oracle_state_context: raw Oracle state string for debugging
        - oracle_action: the CORRECT action the agent should have taken (for learning corrections)
        """
        if not self.writeback_enabled or self.store.writeback is None:
            return {}
        if self._pending is None:
            return {}

        a = str(executed_action).strip()
        ctx_hash = self._last_context_hash or "unknown"

        is_done = a == "done"
        env_success = bool(self.env.is_success()) if hasattr(self.env, "is_success") else False

        # Determine failure status
        fail = False
        inferred_fail_tag = None

        # Use explicit fail_tag if provided (from Oracle diagnosis)
        if fail_tag is not None:
            fail = True
            inferred_fail_tag = fail_tag
        elif int(err_code) != 0:
            fail = True
            inferred_fail_tag = "invalid_action"
        elif is_done and (not env_success):
            fail = True
            inferred_fail_tag = "bad_done"

        self._pending.success = bool((not fail) and (not is_done))
        self._pending.fail = bool(fail)
        self._pending.fail_tag = inferred_fail_tag
        self._pending.steps = 1

        # Enhanced extra_metrics with full context
        if self._pending.extra_metrics is not None:
            self._pending.extra_metrics["episode_id"] = (
                int(episode_id) if episode_id is not None else None
            )
            self._pending.extra_metrics["step_id"] = int(step_id) if step_id is not None else None
            self._pending.extra_metrics["err_code"] = int(err_code)
            self._pending.extra_metrics["fail_tag"] = inferred_fail_tag

            # New fields for Failure Memory
            if history is not None:
                self._pending.extra_metrics["history"] = list(history)
            if oracle_state_context is not None:
                self._pending.extra_metrics["oracle_state_context"] = str(oracle_state_context)
            if oracle_action is not None:
                self._pending.extra_metrics["oracle_action"] = str(oracle_action)

        if fail:
            self.store.repeat_fail_counts[(ctx_hash, a)] = int(
                self.store.repeat_fail_counts.get((ctx_hash, a), 0) + 1
            )

        if self.store.writeback.should_write(self._pending, predicted_success=self._pending_pred):
            if fail:
                self.store.writeback.write_failure(
                    self._pending,
                    correction={
                        "avoid_action": a,
                        "context_hash": ctx_hash,
                        "fail_tag": inferred_fail_tag,
                        "oracle_state_context": oracle_state_context,
                        "correct_action": oracle_action,  # What the agent SHOULD have done
                    },
                )
            else:
                self.store.writeback.write(self._pending)

        # NEW (Phase 2): Reflect on failure and extract principle
        learned_principle = None
        if fail and self.use_principles and inferred_fail_tag:
            learned_principle = self._reflect_on_failure(
                failed_action=a,
                fail_tag=inferred_fail_tag,
                oracle_action=oracle_action,
                symbolic_state=self._pending_symbolic_state,
                image=self._last_image,
            )

        # NEW: Track last action outcome for next symbolic state extraction
        self._last_action_success = not fail
        self._last_fail_tag = inferred_fail_tag if fail else None

        # clear pending
        out = {
            "fail": bool(fail),
            "fail_tag": inferred_fail_tag,
            # NEW: Include learned principle info
            "learned_principle": learned_principle.content[:80] if learned_principle else None,
        }
        self._pending = None
        self._pending_pred = None
        self._pending_symbolic_state = None
        return out


class RoMemoStore:
    """
    Shareable RoMemo memory stack across many episodes/boards.
    """

    def __init__(
        self,
        task: str,
        cfg: RoMemoDiscreteConfig,
        writeback: bool,
        seed: int = 0,
        init_memory_path: Optional[str] = None,
    ):
        self.task = str(task)
        self.cfg = cfg
        if init_memory_path:
            try:
                self.memory = MemoryBank.load_pt(init_memory_path)  # type: ignore[arg-type]
            except Exception:
                # fall back to empty if load fails
                self.memory = MemoryBank(name=f"romemo_reflectvlm_{self.task}")
        else:
            self.memory = MemoryBank(name=f"romemo_reflectvlm_{self.task}")
        self.retriever = Retriever(self.memory, use_gpu=False, seed=int(seed))
        self.retriever.build_index(task=self.task, subtask=None, step_type="option_start")

        self.writeback = None
        if writeback:
            self.writeback = WritebackPolicy(
                memory=self.memory,
                retriever=self.retriever,
                config=WritebackConfig(
                    on_failure=bool(cfg.write_on_failure),
                    on_success=bool(cfg.write_on_success),
                    on_surprise=False,
                    deduplicate=bool(cfg.deduplicate),
                    novelty_threshold=0.0,
                    max_memory_size=10000,
                    recency_bias=True,
                ),
            )

        self.repeat_fail_counts: Dict[Tuple[str, str], int] = {}

    def save_pt(self, path: str) -> None:
        self.memory.save_pt(path)  # type: ignore[arg-type]
