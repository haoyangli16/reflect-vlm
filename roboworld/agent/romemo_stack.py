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
    """
    st = env.get_env_state()
    qpos, qvel = st.get("joint", (None, None))
    mocap_pos, mocap_quat = st.get("mocap", (None, None))
    eq_active, eq_data = st.get("eq", (None, None))

    parts: List[np.ndarray] = []
    if qpos is not None:
        parts.append(np.asarray(qpos, dtype=np.float32).reshape(-1))
    if qvel is not None:
        parts.append(np.asarray(qvel, dtype=np.float32).reshape(-1))
    if mocap_pos is not None:
        parts.append(np.asarray(mocap_pos, dtype=np.float32).reshape(-1))
    if mocap_quat is not None:
        parts.append(np.asarray(mocap_quat, dtype=np.float32).reshape(-1))
    if eq_active is not None:
        parts.append(np.asarray(eq_active, dtype=np.float32).reshape(-1))
    if eq_data is not None:
        parts.append(np.asarray(eq_data, dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


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
    ):
        self.base_agent = base_agent
        self.env = env
        self.task = str(task)
        self.cfg = cfg or RoMemoDiscreteConfig()
        self.writeback_enabled = bool(writeback)

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
            scores[str(action)] = float(scores.get(str(action), 0.0) + sim * w)

            st = stats.get(str(action))
            if st is None:
                st = {"n": 0, "n_succ": 0, "n_fail": 0, "sim_sum": 0.0}
                stats[str(action)] = st
            st["n"] += 1
            st["sim_sum"] += float(sim)
            if bool(exp.success):
                st["n_succ"] += 1
            if bool(exp.fail):
                st["n_fail"] += 1

        for a, st in stats.items():
            n = max(1, int(st["n"]))
            st["fail_rate"] = float(int(st["n_fail"]) / n)
        return scores, stats

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
            ranked.append((a, s))
        ranked.sort(key=lambda x: x[1], reverse=True)
        chosen = ranked[0][0] if ranked else str(base_action)
        return chosen, ranked, top_mem

    def act(self, img, goal_img, inp, next_image=None):
        # context
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
