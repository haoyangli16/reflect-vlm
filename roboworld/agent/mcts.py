"""roboworld.agent.mcts

Appendix-style MCTS in a discrete high-level action space:

- Action is one of: "pick up <color>", "put down <color>", "reorient <color>",
    "insert <color>", or "done".
- Selection uses UCT: maximize Q + U.
- Expansion uses top-K VLM proposals (deduped + normalized).
- Value uses an expert (oracle) rollout: V = exp(-0.1 * S), where S is the number
    of steps until success (including the first simulated action). If success is not
    reached within the rollout cap, V = 0.

Environment simulation relies on state cloning (env.__getstate__/__setstate__).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import contextlib
import io

import numpy as np

from roboworld.agent.utils import get_prompt


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


_PRIMITIVES: Tuple[str, ...] = ("pick up", "put down", "insert", "reorient")


def _extract_history_from_prompt(inp: Optional[str]) -> List[str]:
    if not inp:
        return []
    marker = "The most recently executed actions are:"
    i = inp.find(marker)
    if i < 0:
        return []
    j = inp.find("[", i)
    k = inp.find("]", j + 1)
    if j < 0 or k < 0:
        return []
    inner = inp[j + 1 : k].strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(",")]
    return [p for p in parts if p]


def _normalize_action(action: str, obj_labels: Sequence[str]) -> Optional[str]:
    if not action:
        return None
    s = str(action).strip().lower().splitlines()[0].strip()
    if not s:
        return None
    # Strip common leading numbering (e.g., "1. ", "2) ")
    if len(s) >= 3 and s[0].isdigit() and s[1] in {".", ")"}:
        s = s[2:].strip()
        if s.startswith("-"):
            s = s[1:].strip()

    if "done" in s:
        return "done"

    primitive = None
    for p in _PRIMITIVES:
        if p in s:
            primitive = p
            break
    if primitive is None:
        return None

    labels = [str(x).lower() for x in obj_labels]
    tokens = s.replace(".", " ").replace(",", " ").split()
    obj = None
    for tok in reversed(tokens):
        if tok in labels:
            obj = tok
            break
    if obj is None:
        return None
    return f"{primitive} {obj}"


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
        proposer_agent: Optional[Any] = None,
        oracle_agent: Optional[Any] = None,
        obj_labels: Optional[Sequence[str]] = None,
        camera_name: str = "camera0",
        proposal_k: int = 5,
        proposal_observation: str = "root",
        seed: int = 0,
        num_simulations: int = 50,
        max_depth: int = 2,
        rollout_depth: int = 20,
        c_uct: float = 0.5,
        suppress_sim_output: bool = True,
    ):
        self.env = env
        self.proposer_agent = proposer_agent
        self.oracle_agent = oracle_agent
        self.obj_labels = list(obj_labels) if obj_labels is not None else list(getattr(env, "peg_colors", []))
        self.camera_name = str(camera_name)
        self.proposal_k = int(proposal_k)
        # "root" (fast): always propose using the root observation image.
        # "node" (slow): render an image from each simulated node state.
        self.proposal_observation = str(proposal_observation)
        self.rng = np.random.default_rng(int(seed))
        self.num_simulations = int(num_simulations)
        self.max_depth = int(max_depth)
        # Reused as an expert rollout cap.
        self.rollout_depth = int(rollout_depth)
        self.c_uct = float(c_uct)
        self.suppress_sim_output = bool(suppress_sim_output)

        self.last_debug: Dict = {}
        self._proposal_cache: Dict[str, List[str]] = {}
        self._vlm_calls: int = 0
        self._render_calls: int = 0

    def _simulate_action(self, action: str) -> int:
        if action == "done":
            return 0
        try:
            return int(self.env.act_txt(action))
        except Exception:
            return -999

    def _propose_actions(self, image, goal_image, history: List[str]) -> List[str]:
        # Prefer VLM proposals (top-K); fall back to cheap legality heuristics.
        if self.proposer_agent is None:
            return _candidate_actions(self.env)

        prompt = get_prompt(version="propose", history=history, obj_labels=self.obj_labels)
        cached = self._proposal_cache.get(prompt)
        if cached is not None:
            return list(cached)
        try:
            self._vlm_calls += 1
            proposed = self.proposer_agent.act(
                image,
                goal_image,
                prompt,
                next_image=None,
                num_propose_actions=int(self.proposal_k),
                return_score=True,
                temperature=0,
            )
        except Exception:
            return _candidate_actions(self.env)

        # `proposed` is either [(text, score), ...] or a single (text, score)
        if isinstance(proposed, tuple) and len(proposed) == 2:
            proposed_list = [proposed]
        else:
            proposed_list = list(proposed)

        out: List[str] = []
        seen = set()
        for a, _s in proposed_list:
            na = _normalize_action(a, self.obj_labels)
            if na is None:
                continue
            if na in seen:
                continue
            out.append(na)
            seen.add(na)

        if not out:
            out = _candidate_actions(self.env)
        # Cache for the duration of this MCTS step.
        self._proposal_cache[prompt] = list(out)
        return out

    def _expert_value(self, already_spent_steps: int = 0) -> float:
        if bool(self.env.is_success()):
            return 1.0
        if self.oracle_agent is None:
            # Heuristic fallback if oracle not provided
            return float(_progress(self.env))

        cap = max(0, int(self.rollout_depth))
        steps = 0
        for _ in range(cap):
            if bool(self.env.is_success()):
                total = already_spent_steps + steps
                return float(np.exp(-0.1 * float(total)))
            a = self.oracle_agent.act()
            a = _normalize_action(a, self.obj_labels) or ("done" if "done" in str(a).lower() else None)
            if a is None or a == "done":
                break
            self._simulate_action(a)
            steps += 1
        # Not solved within cap
        return 0.0

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

    def _mcts(self, image, goal_image, inp: Optional[str]) -> str:
        # disable recording if present
        env_recording = getattr(self.env, "is_recording", False)
        if env_recording and hasattr(self.env, "_record"):
            prev_record = bool(self.env._record)
            self.env._record = False
        else:
            prev_record = None

        root_state = self.env.__getstate__()
        root = _Node(untried=[])
        root_progress = _progress(self.env)

        # Per-step caches/counters
        self._proposal_cache = {}
        self._vlm_calls = 0
        self._render_calls = 0

        root_history = _extract_history_from_prompt(inp)

        sim_ctx = (
            contextlib.redirect_stdout(io.StringIO())
            if self.suppress_sim_output
            else contextlib.nullcontext()
        )
        err_ctx = (
            contextlib.redirect_stderr(io.StringIO())
            if self.suppress_sim_output
            else contextlib.nullcontext()
        )

        with sim_ctx, err_ctx:
            for _sim in range(self.num_simulations):
                # restore to root
                self.env.__setstate__(root_state)
                node = root
                depth = 0
                history_sim = list(root_history)

                # selection
                path: List[_Node] = [root]
                while node.children and not node.untried and depth < self.max_depth:
                    a = self._uct_select(node)
                    self._simulate_action(a)
                    history_sim.append(a)
                    node = node.children[a]
                    path.append(node)
                    depth += 1

                # expansion
                spent = 0
                if depth < self.max_depth:
                    if not node.untried:
                        # Proposing from simulated node images is extremely slow (MuJoCo render per simulation).
                        # Default to using the root observation for proposals.
                        sim_img = image
                        if self.proposal_observation == "node":
                            try:
                                self._render_calls += 1
                                sim_img = self.env.read_pixels(camera_name=self.camera_name)
                            except Exception:
                                sim_img = image
                        node.untried = self._propose_actions(sim_img, goal_image, history_sim)

                    if node.untried:
                        a = node.untried.pop(0)
                        spent = 1
                        self._simulate_action(a)
                        history_sim.append(a)
                        child = node.children.get(a)
                        if child is None:
                            child = _Node(untried=[])
                            node.children[a] = child
                        node = child
                        path.append(node)
                        depth += 1

                # value (expert rollout)
                if bool(self.env.is_success()):
                    value = 1.0
                else:
                    value = self._expert_value(already_spent_steps=spent)

                # backup
                for n in path:
                    n.N += 1
                    n.W += float(value)

        # final selection: best child by Q (tie-break by visit count)
        if not root.children:
            return "done"
        ranked = sorted(
            [(a, ch.N, ch.Q) for a, ch in root.children.items()],
            key=lambda x: (x[2], x[1]),
            reverse=True,
        )
        best_action = ranked[0][0]
        self.last_debug = {
            "num_simulations": int(self.num_simulations),
            "max_depth": int(self.max_depth),
            "rollout_depth": int(self.rollout_depth),
            "root_progress": float(root_progress),
            "proposal_k": int(self.proposal_k),
            "proposal_observation": str(self.proposal_observation),
            "vlm_calls": int(self._vlm_calls),
            "render_calls": int(self._render_calls),
            "children": [{"action": a, "N": int(n), "Q": float(q)} for a, n, q in ranked[:10]],
        }

        # restore recording flag
        self.env.__setstate__(root_state)
        if prev_record is not None:
            self.env._record = prev_record
        return best_action

    def act(self, image=None, goal_image=None, inp=None, next_image=None):
        return self._mcts(image=image, goal_image=goal_image, inp=inp)


