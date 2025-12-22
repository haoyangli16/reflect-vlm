"""
RoMemo wrapper for reflect-vlm discrete-action environment.

Goal: show RoMemo is an *orthogonal* test-time memory stack that can improve
any fixed base policy/planner by retrieval-conditioned reranking + optional
corrective writeback.

This wrapper is intentionally lightweight:
- Context embedding: deterministic numeric state vector from env.get_env_state()
- Retrieval: brute-force cosine similarity (fast enough for small memories)
- Deliberation: rerank base action with memory-derived scores
- Writeback: store success/failure items keyed by context hash
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboworld.envs.generator import COLORS as _COLORS


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype(np.float32, copy=False).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32, copy=False)


def state_hash_from_vec(state_vec: np.ndarray, quant: float = 1e-3) -> str:
    """
    Stable hash of a numeric state vector (for \"repeatable failure\" tracking).
    We quantize to reduce floating noise, then md5.
    """
    sv = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    q = np.round(sv / float(quant)).astype(np.int32)
    h = hashlib.md5(q.tobytes()).hexdigest()
    return h[:16]


def extract_env_state_vec(env) -> np.ndarray:
    """
    Deterministic numeric context embedding using env.get_env_state().
    This is available in this benchmark and is more stable than raw pixels.

    IMPORTANT: MuJoCo qpos/qvel/eq arrays change length across different boards.
    We therefore build a fixed-size embedding.
    """

    canonical_colors = tuple(_COLORS.keys())
    robot_dof = int(len(getattr(env, "robot_init_qpos", [])) or 9)
    action_dim = int(len(getattr(env, "prev_action", np.zeros((8,), dtype=np.float32))) or 8)

    robot_qpos = np.zeros((robot_dof,), dtype=np.float32)
    robot_qvel = np.zeros((robot_dof,), dtype=np.float32)
    mocap_pos = np.zeros((3,), dtype=np.float32)
    mocap_quat = np.zeros((4,), dtype=np.float32)
    prev_action = np.zeros((action_dim,), dtype=np.float32)
    curr_path_length = np.zeros((1,), dtype=np.float32)

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


@dataclass(frozen=True)
class RoMemoConfig:
    k: int = 20
    alpha: float = 0.7  # trust base policy
    lambda_fail: float = 1.5  # penalty weight for failures in memory evidence
    beta_failrate: float = 0.5  # additional risk penalty proportional to fail-rate in retrieved neighbors
    beta_repeat: float = 0.2  # penalty per repeat failure count for same (context_hash, action)
    max_mem_candidates: int = 5  # add top-N memory actions to candidate set
    min_sim_threshold: float = -1.0  # keep all by default
    quant_for_hash: float = 1e-3
    max_memory_items: Optional[int] = None  # optional cap (FIFO)


@dataclass
class MemoryItem:
    q: np.ndarray  # normalized vector
    action: str
    outcome: int  # +1 success, -1 fail
    fail_tag: Optional[str] = None
    episode_id: Optional[int] = None
    step_id: Optional[int] = None
    context_hash: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=lambda: time.time())


class RoMemoMemory:
    def __init__(self, cfg: RoMemoConfig):
        self.cfg = cfg
        self.items: List[MemoryItem] = []
        self._mat: Optional[np.ndarray] = None  # (N, D) normalized vectors

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        # Optional cap (FIFO)
        if self.cfg.max_memory_items is not None and len(self.items) > int(self.cfg.max_memory_items):
            overflow = len(self.items) - int(self.cfg.max_memory_items)
            if overflow > 0:
                self.items = self.items[overflow:]
        self._mat = None  # lazy rebuild

    def _ensure_mat(self) -> None:
        if self._mat is not None:
            return
        if not self.items:
            self._mat = None
            return
        self._mat = np.stack([it.q for it in self.items], axis=0).astype(np.float32, copy=False)

    def retrieve(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, List[MemoryItem]]:
        if not self.items:
            return np.zeros((0,), dtype=np.float32), []
        self._ensure_mat()
        assert self._mat is not None
        qn = _l2_normalize(q)
        sims = (self._mat @ qn.reshape(-1, 1)).reshape(-1)  # cosine
        if k <= 0:
            return np.zeros((0,), dtype=np.float32), []
        k = min(int(k), sims.shape[0])
        # argpartition for speed
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        # sort by similarity descending
        idx = idx[np.argsort(-sims[idx])]
        out_items = [self.items[int(i)] for i in idx.tolist()]
        out_sims = sims[idx].astype(np.float32, copy=False)
        return out_sims, out_items


class RoMemoAgent:
    """
    Wrapper agent around any base_agent that outputs an action string.

    - base_agent: must implement act(img, goal_img, inp, **kwargs?) -> str
    - env: used only for context embedding and optional progress heuristic
    """

    def __init__(
        self,
        base_agent,
        env,
        cfg: Optional[RoMemoConfig] = None,
        writeback: bool = False,
        candidate_actions: Optional[Sequence[str]] = None,
    ):
        self.base_agent = base_agent
        self.env = env
        self.cfg = cfg or RoMemoConfig()
        self.writeback = bool(writeback)
        self.memory = RoMemoMemory(self.cfg)
        self.candidate_actions = list(candidate_actions) if candidate_actions is not None else None

        # repeatable failure counter: (context_hash, action) -> count
        self._repeat_fail_counts: Dict[Tuple[str, str], int] = {}

        # last-step caches for logging/writeback
        self.last_trace: Dict[str, Any] = {}
        self._last_qvec: Optional[np.ndarray] = None
        self._last_context_hash: Optional[str] = None
        self._last_base_action: Optional[str] = None

        # simple progress proxy (optional)
        self._last_progress: Optional[float] = None

    def _progress(self) -> float:
        # Fraction of bricks successfully inserted
        try:
            done = 0
            for name in getattr(self.env, "peg_names", []):
                if bool(self.env.object_is_success(name)):
                    done += 1
            total = max(1, len(getattr(self.env, "peg_names", [])))
            return float(done / total)
        except Exception:
            return 0.0

    def _validate_action(self, a: str) -> str:
        a = (a or "").strip()
        if not a:
            return "done"
        if self.candidate_actions is None:
            return a
        # accept if in whitelist; otherwise fall back to base action selection
        if a in self.candidate_actions:
            return a
        return a

    def _mem_scores(
        self, sims: np.ndarray, items: List[MemoryItem]
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        """
        Group evidence by action and compute RoMemo scores.
        Returns:
          - mem_score[action] = sum(sim * weight(outcome))
          - stats[action] with counts and fail_rate for risk penalty
        """
        # Per-action average score (not a raw sum) to avoid runaway dominance from
        # repeated near-duplicate memories.
        scores: Dict[str, float] = {}
        stats: Dict[str, Dict[str, Any]] = {}
        for sim, it in zip(sims.tolist(), items):
            if sim < self.cfg.min_sim_threshold:
                continue
            a = it.action
            st = stats.get(a)
            if st is None:
                st = {"n": 0, "n_succ": 0, "n_fail": 0, "sim_sum": 0.0, "signed_sim_sum": 0.0}
                stats[a] = st
            st["n"] += 1
            st["sim_sum"] += float(sim)
            w = 1.0 if it.outcome > 0 else -float(self.cfg.lambda_fail)
            st["signed_sim_sum"] += float(sim) * float(w)
            if it.outcome > 0:
                st["n_succ"] += 1
            else:
                st["n_fail"] += 1
        for a, st in stats.items():
            n = max(1, int(st["n"]))
            st["fail_rate"] = float(int(st["n_fail"]) / n)
            scores[a] = float(st.get("signed_sim_sum", 0.0)) / float(n)
        return scores, stats

    def _feasible_actions(self) -> Optional[set[str]]:
        try:
            colors = list(getattr(self.env, "peg_colors", []))
            names = list(getattr(self.env, "peg_names", []))
            done_mask: Dict[str, bool] = {}
            for c, name in zip(colors, names):
                try:
                    done_mask[str(c)] = bool(self.env.object_is_success(name))
                except Exception:
                    done_mask[str(c)] = False

            body = self.env.get_object_in_hand() if hasattr(self.env, "get_object_in_hand") else None
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

    def _choose_action(
        self,
        base_action: str,
        mem_scores: Dict[str, float],
        mem_stats: Dict[str, Dict[str, Any]],
        context_hash: str,
    ) -> Tuple[str, Dict[str, Any]]:
        # candidate set: base action + top memory actions
        cand = {base_action}
        top_mem = sorted(mem_scores.items(), key=lambda x: x[1], reverse=True)[: int(self.cfg.max_mem_candidates)]
        for a, _s in top_mem:
            cand.add(a)

        feasible = self._feasible_actions()
        if feasible is not None:
            cand = {a for a in cand if a in feasible}
            if not cand:
                cand = {base_action}

        scored: List[Tuple[str, float]] = []
        for a in cand:
            base_prior = 1.0 if a == base_action else 0.0
            ms = float(mem_scores.get(a, 0.0))
            fail_rate = float(mem_stats.get(a, {}).get("fail_rate", 0.0))
            rep = int(self._repeat_fail_counts.get((context_hash, a), 0))
            risk = float(self.cfg.beta_failrate) * fail_rate + float(self.cfg.beta_repeat) * float(rep)
            s = float(self.cfg.alpha) * base_prior + float(1.0 - self.cfg.alpha) * ms - risk
            scored.append((a, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        chosen = scored[0][0] if scored else base_action
        debug = {
            "candidates": [{"action": a, "score": float(s)} for a, s in scored],
            "top_mem": [{"action": a, "mem_score": float(s)} for a, s in top_mem],
        }
        return chosen, debug

    def act(self, img, goal_img, inp, next_image=None):
        # context embedding
        state_vec = extract_env_state_vec(self.env)
        q = _l2_normalize(state_vec)
        h = state_hash_from_vec(state_vec, quant=self.cfg.quant_for_hash)
        self._last_qvec = q
        self._last_context_hash = h
        self._last_progress = self._progress()

        # base action proposal
        base_action = self.base_agent.act(img, goal_img, inp, next_image=next_image)
        base_action = self._validate_action(str(base_action))
        self._last_base_action = base_action

        sims, items = self.memory.retrieve(q, k=int(self.cfg.k))
        mem_scores, mem_stats = self._mem_scores(sims, items)
        chosen, dbg = self._choose_action(base_action, mem_scores, mem_stats, h)

        # trace payload (for step_traces.jsonl)
        self.last_trace = {
            "context_hash": h,
            "base_action": base_action,
            "chosen_action": chosen,
            "retrieval_k": int(self.cfg.k),
            "memory_size": len(self.memory),
            "retrieved": [
                {
                    "sim": float(s),
                    "action": it.action,
                    "outcome": int(it.outcome),
                    "fail_tag": it.fail_tag,
                    "episode_id": it.episode_id,
                    "step_id": it.step_id,
                    "context_hash": it.context_hash,
                }
                for s, it in zip(sims.tolist(), items)
            ][:10],
            "mem_action_scores": [
                {
                    "action": a,
                    "mem_score": float(mem_scores[a]),
                    "fail_rate": float(mem_stats.get(a, {}).get("fail_rate", 0.0)),
                }
                for a in sorted(mem_scores.keys(), key=lambda x: mem_scores[x], reverse=True)[:10]
            ],
            "rerank": dbg,
        }
        return chosen

    def update(
        self,
        action: str,
        err_code: int,
        episode_id: Optional[int] = None,
        step_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Writeback: store outcome for the last context-action.
        Returns a small dict for tracing (fail_tag, outcome).
        """
        if not self.writeback:
            return {}

        action = str(action).strip()
        # decide outcome
        prog_after = self._progress()
        prog_before = float(self._last_progress) if self._last_progress is not None else None

        is_done = (action == "done")
        ep_success = bool(self.env.is_success()) if hasattr(self.env, "is_success") else False

        fail_tag: Optional[str] = None
        if err_code != 0:
            outcome = -1
            fail_tag = "invalid_action"
        elif is_done and (not ep_success):
            outcome = -1
            fail_tag = "bad_done"
        else:
            # treat as success by default if no explicit error
            outcome = +1
            # optionally mark suspicious regressions
            if prog_before is not None and prog_after + 1e-6 < prog_before:
                outcome = -1
                fail_tag = "progress_regress"

        # update repeat counter on failures
        h = self._last_context_hash or "unknown"
        if outcome < 0:
            key = (h, action)
            self._repeat_fail_counts[key] = int(self._repeat_fail_counts.get(key, 0) + 1)

        q = self._last_qvec
        if q is None:
            # fallback: recompute
            q = _l2_normalize(extract_env_state_vec(self.env))

        item = MemoryItem(
            q=q,
            action=action,
            outcome=int(outcome),
            fail_tag=fail_tag,
            episode_id=episode_id,
            step_id=step_id,
            context_hash=h,
            meta={
                "err_code": int(err_code),
                "prog_before": float(prog_before) if prog_before is not None else None,
                "prog_after": float(prog_after),
            },
        )
        self.memory.add(item)
        return {"outcome": int(outcome), "fail_tag": fail_tag, "prog_after": float(prog_after)}


