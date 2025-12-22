from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from roboworld.agent.utils import get_prompt, parse_act_txt


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


@dataclass
class ReflectWrapperConfig:
    imagine_future_steps: int = 5
    camera_name: str = "table_back"


class ReflectWrapperAgent:
    """Wrap a base VLM policy with the ReflectVLM-style reflection mechanism.

    - Propose an action from the current observation.
    - Simulate/imagine K future steps under the base policy (using sim dynamics).
    - Ask the base policy to reflect and revise only the *next* action.

    This wrapper is designed to be composable with RoMemo (wrap this agent with
    RoMemoDiscreteAgent to compare reflect vs reflect+romemo).
    """

    def __init__(
        self,
        *,
        env,
        base_agent,
        obj_labels: Sequence[str],
        cfg: Optional[ReflectWrapperConfig] = None,
    ):
        self.env = env
        self.base_agent = base_agent
        self.obj_labels = list(obj_labels)
        self.cfg = cfg or ReflectWrapperConfig()
        self.candidate_act_list = ["pick up", "put down", "insert", "reorient"]

    def _imagine_with_sim(
        self,
        *,
        first_action: str,
        goal_img,
        history: List[str],
    ) -> Tuple[List[str], Optional[object]]:
        env_recording = getattr(self.env, "is_recording", False)
        if env_recording and hasattr(self.env, "_record"):
            prev_record = bool(self.env._record)
            self.env._record = False
        else:
            prev_record = None

        _env_state = self.env.__getstate__()
        plan = [first_action]

        future_img = None
        for i in range(max(0, int(self.cfg.imagine_future_steps))):
            try:
                _ = self.env.act_txt(plan[-1])
            except Exception:
                break

            try:
                future_img = self.env.read_pixels(camera_name=self.cfg.camera_name)
            except Exception:
                future_img = None

            if i + 1 == int(self.cfg.imagine_future_steps) or bool(self.env.is_success()):
                break

            try:
                sim_img = self.env.read_pixels(camera_name=self.cfg.camera_name)
            except Exception:
                sim_img = None

            next_prompt = get_prompt(
                version="propose",
                history=history + plan,
                obj_labels=self.obj_labels,
            )
            a = self.base_agent.act(sim_img, goal_img, inp=next_prompt)
            if a == "done":
                break
            plan.append(a)

        # Restore env
        self.env.__setstate__(_env_state)
        if prev_record is not None:
            self.env._record = prev_record

        return plan, future_img

    def _validate_action(self, action: str) -> str:
        try:
            assert len(action.strip().split(" ")) <= 3, "Bad output format."
            p, o = parse_act_txt(action)
            if p == "done":
                return "done"
            assert p in self.candidate_act_list and o in self.obj_labels, "Bad output format."
            return action
        except Exception:
            return "done"

    def act(self, image=None, goal_image=None, inp=None, next_image=None):
        history = _extract_history_from_prompt(inp)

        # Propose
        propose_prompt = get_prompt(
            version="propose",
            history=history,
            obj_labels=self.obj_labels,
        )
        first_action = self.base_agent.act(image, goal_image, inp=propose_prompt)
        first_action = self._validate_action(first_action)
        if first_action == "done" or int(self.cfg.imagine_future_steps) <= 0:
            return first_action

        # Imagine with sim dynamics
        plan, future_img = self._imagine_with_sim(
            first_action=first_action,
            goal_img=goal_image,
            history=history,
        )

        # Reflect
        reflect_prompt = get_prompt(
            version="reflect",
            history=history,
            obj_labels=self.obj_labels,
            initial_plan=plan,
        )
        revised = self.base_agent.act(image, goal_image, inp=reflect_prompt, next_image=future_img)
        revised = self._validate_action(revised)

        # Fallback to first action if the model fails to produce a valid revision
        if revised == "done":
            return first_action
        return revised
