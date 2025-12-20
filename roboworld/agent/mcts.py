"""
Lightweight MCTS baseline for reflect-vlm.

Important: this is a *minimal, runnable* MCTS in the discrete high-level action space:
  action := \"<primitive> <color>\" or \"done\"

It uses environment state cloning (env.__getstate__/__setstate__) to simulate outcomes.
This is intentionally small-budget to keep evaluation feasible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _progress(env) -> float:
    try:
        done = 0
        for name in getattr(env, "peg_names", []):
            if bool(env.object_is_success(name)):
                done += 1
        total = max(1, len(getattr(env, "peg_names", [])))
        return float(done / total)
    except Exception:
        return 0.0


def _in_hand_color(env) -> Optional[str]:
    body = env.get_object_in_hand() if hasattr(env, "get_object_in_hand") else None
    if body is None:
        return None
    try:
        # env.peg_names aligns with env.peg_colors
        i = env.peg_names.index(body)
        return str(env.peg_colors[i])
    except Exception:
        return None


def _candidate_actions(env) -> List[str]:
    """
    Cheap legality heuristic without oracle:
    - if holding: {insert, reorient, put down} for held color + done
    - else: pick up for colors not already done + done
    """
    primitives = ["pick up", "put down", "insert", "reorient"]
    colors = list(getattr(env, "peg_colors", []))
    done_mask: Dict[str, bool] = {}
    for c, name in zip(colors, getattr(env, "peg_names", [])):
        try:
            done_mask[str(c)] = bool(env.object_is_success(name))
        except Exception:
            done_mask[str(c)] = False

    held = _in_hand_color(env)
    acts: List[str] = []
    if held is None:
        for c in colors:
            if not done_mask.get(str(c), False):
                acts.append(f"pick up {c}")
        acts.append("done")
    else:
        # allow a small set while holding
        acts.append(f"insert {held}")
        acts.append(f"reorient {held}")
        acts.append(f"put down {held}")
        acts.append("done")
    # de-dup while preserving order
    seen = set()
    out: List[str] = []
    for a in acts:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


@dataclass
class _Node:
    N: int = 0
    W: float = 0.0
    children: Dict[str, "_Node"] = field(default_factory=dict)
    untried: List[str] = field(default_factory=list)

    @property
    def Q(self) -> float:
        return self.W / max(1, self.N)


class MCTSAgent:
    def __init__(
        self,
        env,
        seed: int = 0,
        num_simulations: int = 30,
        max_depth: int = 2,
        rollout_depth: int = 3,
        c_uct: float = 1.4,
        invalid_action_penalty: float = 0.2,
    ):
        self.env = env
        self.rng = np.random.default_rng(int(seed))
        self.num_simulations = int(num_simulations)
        self.max_depth = int(max_depth)
        self.rollout_depth = int(rollout_depth)
        self.c_uct = float(c_uct)
        self.invalid_action_penalty = float(invalid_action_penalty)

        self.last_debug: Dict = {}

    def _simulate_action(self, action: str) -> Tuple[int, float]:
        """
        Apply action to env. Returns (err_code, immediate_value_proxy).
        """
        if action == "done":
            # don't change state, just score based on success
            return 0, (1.0 if bool(self.env.is_success()) else _progress(self.env))
        try:
            err = int(self.env.act_txt(action))
        except Exception:
            return -999, _progress(self.env) - self.invalid_action_penalty
        val = _progress(self.env)
        if bool(self.env.is_success()):
            val = 1.0
        if err != 0:
            val -= self.invalid_action_penalty
        return err, float(val)

    def _rollout(self) -> float:
        """
        Random rollout for a few steps to estimate downstream value.
        """
        for _ in range(max(0, self.rollout_depth)):
            if bool(self.env.is_success()):
                return 1.0
            acts = _candidate_actions(self.env)
            if not acts:
                break
            a = acts[int(self.rng.integers(0, len(acts)))]
            self._simulate_action(a)
        return float(_progress(self.env))

    def _uct_select(self, node: _Node) -> str:
        assert node.children, "UCT select requires children"
        logN = np.log(max(1, node.N))
        best_a = None
        best = -1e9
        for a, ch in node.children.items():
            u = ch.Q + self.c_uct * np.sqrt(logN / max(1, ch.N))
            if u > best:
                best = float(u)
                best_a = a
        assert best_a is not None
        return best_a

    def _mcts(self) -> str:
        # disable recording if present
        env_recording = getattr(self.env, "is_recording", False)
        if env_recording and hasattr(self.env, "_record"):
            prev_record = bool(self.env._record)
            self.env._record = False
        else:
            prev_record = None

        root_state = self.env.__getstate__()
        root = _Node(untried=_candidate_actions(self.env))
        root_progress = _progress(self.env)

        for _sim in range(self.num_simulations):
            # restore to root
            self.env.__setstate__(root_state)
            node = root
            depth = 0

            # selection
            path: List[Tuple[_Node, str]] = []
            while (not node.untried) and node.children and depth < self.max_depth:
                a = self._uct_select(node)
                path.append((node, a))
                node = node.children[a]
                self._simulate_action(a)
                depth += 1

            # expansion
            if node.untried and depth < self.max_depth:
                a = node.untried.pop(int(self.rng.integers(0, len(node.untried))))
                # apply
                self._simulate_action(a)
                child = _Node(untried=_candidate_actions(self.env))
                node.children[a] = child
                path.append((node, a))
                node = child
                depth += 1

            # rollout
            if bool(self.env.is_success()):
                value = 1.0
            else:
                value = self._rollout()

            # backprop
            root.N += 1
            root.W += float(value)
            for parent, a in path:
                ch = parent.children.get(a)
                if ch is None:
                    continue
                ch.N += 1
                ch.W += float(value)

        # final selection: best child by visit count / Q
        if not root.children:
            return "done"
        ranked = sorted(
            [(a, ch.N, ch.Q) for a, ch in root.children.items()],
            key=lambda x: (x[1], x[2]),
            reverse=True,
        )
        best_action = ranked[0][0]
        self.last_debug = {
            "num_simulations": int(self.num_simulations),
            "max_depth": int(self.max_depth),
            "rollout_depth": int(self.rollout_depth),
            "root_progress": float(root_progress),
            "children": [{"action": a, "N": int(n), "Q": float(q)} for a, n, q in ranked[:10]],
        }

        # restore recording flag
        self.env.__setstate__(root_state)
        if prev_record is not None:
            self.env._record = prev_record
        return best_action

    def act(self, image=None, goal_image=None, inp=None, next_image=None):
        # ignores images/prompts; uses env dynamics + MCTS
        return self._mcts()


