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
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from roboworld.envs.generator import COLORS as _COLORS

try:
    from romemo.memory.schema import Experience, MemoryBank
    from romemo.memory.retrieve import Retriever, RetrievalResult
    from romemo.memory.writeback import WritebackConfig, WritebackPolicy
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import romemo. In your reflectvlm env run:\n"
        "  pip install -e /home/haoyang/project/haoyang/worldmemory\n"
        "so `import romemo` works.\n"
        f"Original error: {e}"
    )


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
    return float(min(1000.0, s))


@dataclass(frozen=True)
class RoMemoDiscreteConfig:
    k: int = 20
    alpha: float = 0.7
    lambda_fail: float = 1.5
    beta_failrate: float = 0.5
    beta_repeat: float = 0.2
    max_mem_candidates: int = 5
    quant_for_hash: float = 1e-3
    write_on_success: bool = True
    write_on_failure: bool = True
    deduplicate: bool = False  # keep repeated evidence; we also keep explicit repeat counters


class RoMemoDiscreteAgent:
    """
    Base-agent wrapper: retrieval-conditioned rerank + optional writeback.

    base_agent.act(img, goal_img, inp, ...) -> action_str
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
        # Check if base agent supports visual encoding
        # self.use_vision = hasattr(base_agent, "encode_image") and callable(
        #     getattr(base_agent, "encode_image", None)
        # )

        if self.use_vision:
            print("[RoMemo] Vision-based retrieval enabled (using VLM encoder).")
        else:
            print("[RoMemo] Fallback to state-based retrieval.")
            exit(0)
        self.store = shared_store or RoMemoStore(
            task=self.task,
            cfg=self.cfg,
            writeback=bool(self.writeback_enabled),
            seed=int(seed),
        )

        self._pending: Optional[Experience] = None
        self._pending_pred: Optional[bool] = None
        self._last_context_hash: Optional[str] = None
        # local alias for convenience (repeat counts live in the store for sharing across episodes)

        # trace payload compatible with run.py logging
        self.last_trace: Dict[str, Any] = {}

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
            w = 1.0 if bool(exp.success) else -float(self.cfg.lambda_fail)
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

        for a, st in stats.items():
            n = max(1, int(st["n"]))
            st["fail_rate"] = float(int(st["n_fail"]) / n)
            scores[a] = float(st.get("signed_sim_sum", 0.0)) / float(n)
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

        ranked: List[Tuple[str, float]] = []
        for a in cand:
            base_prior = 1.0 if a == str(base_action) else 0.0
            mem = float(scores.get(a, 0.0))
            fail_rate = float(stats.get(a, {}).get("fail_rate", 0.0))
            rep = int(self.store.repeat_fail_counts.get((ctx_hash, a), 0))
            risk = float(self.cfg.beta_failrate) * fail_rate + float(self.cfg.beta_repeat) * float(
                rep
            )
            s = float(self.cfg.alpha) * base_prior + float(1.0 - self.cfg.alpha) * mem - risk

            # HACK: Penalize "done" if base agent didn't propose it
            if a == "done" and str(base_action) != "done":
                s-=10 # massive penalty
            ranked.append((a, s))
        ranked.sort(key=lambda x: x[1], reverse=True)
        chosen = ranked[0][0] if ranked else str(base_action)
        return chosen, ranked, top_mem

    def act(self, img, goal_img, inp, next_image=None):
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

        # retrieve
        ret = self.store.retriever.retrieve(query=state_vec, k=int(self.cfg.k))
        scores, stats = self._mem_action_scores(ret)
        chosen, ranked, top_mem = self._choose(base_action, scores, stats, ctx_hash)

        # create pending experience (aligned with RoMemo pending/update pattern)
        self._pending = Experience(
            task=self.task,
            subtask="discrete_action",
            env_id="reflect-vlm",
            state_vec=state_vec.astype(np.float32, copy=False),
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
                },
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
            "memory_size": int(len(self.store.memory)),
            "retrieved": [
                {
                    "dist": float(d),
                    "action": (e.extra_metrics or {}).get("decision", {}).get("action", None)
                    if isinstance((e.extra_metrics or {}).get("decision", {}), dict)
                    else (e.extra_metrics or {}).get("action", None),
                    "success": bool(e.success),
                    "fail": bool(e.fail),
                    "fail_tag": getattr(e, "fail_tag", None),
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
    ):
        """
        Align with RoMemo update_on_failure: write an Experience after outcome is observed.
        We treat err_code!=0 or (done but not success) as failure.
        """
        if not self.writeback_enabled or self.store.writeback is None:
            return {}
        if self._pending is None:
            return {}

        a = str(executed_action).strip()
        ctx_hash = self._last_context_hash or "unknown"

        is_done = a == "done"
        env_success = bool(self.env.is_success()) if hasattr(self.env, "is_success") else False

        fail = False
        fail_tag = None
        if int(err_code) != 0:
            fail = True
            fail_tag = "invalid_action"
        elif is_done and (not env_success):
            fail = True
            fail_tag = "bad_done"

        self._pending.success = bool((not fail) and (not is_done))
        self._pending.fail = bool(fail)
        self._pending.fail_tag = fail_tag
        self._pending.steps = 1
        if self._pending.extra_metrics is not None:
            self._pending.extra_metrics["episode_id"] = (
                int(episode_id) if episode_id is not None else None
            )
            self._pending.extra_metrics["step_id"] = int(step_id) if step_id is not None else None
            self._pending.extra_metrics["err_code"] = int(err_code)

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
                    },
                )
            else:
                self.store.writeback.write(self._pending)

        # clear pending
        out = {"fail": bool(fail), "fail_tag": fail_tag}
        self._pending = None
        self._pending_pred = None
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
