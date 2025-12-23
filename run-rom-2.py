# Fix LLVM command-line option conflict between triton and bitsandbytes
# This MUST be imported first, before any other imports
import fix_triton_import  # noqa: F401

import os
import time
import traceback
import pandas as pd
from PIL import Image

try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore

from absl import app, flags
import random
import uuid
import copy
from typing import Optional
import numpy as np

# mujoco is used transitively by env; keep import for early failure clarity
import mujoco  # type: ignore  # noqa: F401

from roboworld.utils.config import define_flags, get_user_flags
from roboworld.utils.logger import WandBLogger, set_random_seed
from roboworld.envs.generator import generate_xml
from roboworld.envs.asset_path_utils import full_path_for
from roboworld.envs.mujoco.franka.franka_assembly import FrankaAssemblyEnv, AssemblyOracle
from roboworld.agent.oracle import OracleAgent
from roboworld.agent.utils import parse_act_txt, get_prompt

FLAGS = flags.FLAGS

FLAGS_DEF = define_flags(
    agent_type=(
        "expert",
        "string",
        "Type of agent (llava/bc, mcts, expert, random, *_romemo, *_romemo_wb)",
    ),
    agent_seed=(0, "integer", "Seed for agent randomness (MCTS/RoMemo), separate from env seed."),
    camera_name=("table_back", "string", "Camera name."),
    model_path=("/path/to/model", "string", "Path to model"),
    model_base=(None, "string", "Path to base model"),
    load_8bit=(False, "bool", "Load model in 8bit mode"),
    load_4bit=(False, "bool", "Load model in 4bit mode"),
    revise_action=(False, "bool", "Revise an initially proposed action"),
    dummy_revised_action=(
        False,
        "bool",
        "If True, use original action as revised action. "
        "This is for data collection when the agent is not trained to reflect yet "
        "(usually the first iteration of DAgger).",
    ),
    imagine_future_steps=(0, "integer", "Generate/simulate future image."),
    diffuser_pretrained_model=(None, "string", "Path to pretrained diffuser"),
    diffuser_unet_dir=(None, "string", "Path to unet model of diffuser"),
    diffuser_vae_dir=(None, "string", "Path to vae model of diffuser"),
    level=("all", "string", "Level of difficulty (medium, hard, or all)"),
    seed=(42, "integer", "Random seed."),
    reset_seed_start=(0, "integer", "first seed to reset env"),
    max_steps=(50, "integer", "Max number of decision steps in a trajectory"),
    n_trajs=(100, "integer", "Number of trajectories."),
    repeat_per_env=(1, "integer", "Number of trajectories for each env/board."),
    save_dir=("datasets/data_v9", "string", "Directory for saving data."),
    start_traj_id=(0, "integer", "Starting trajectory index"),
    start_board_id=(0, "integer", "Starting board index"),
    oracle_prob=(0.5, "float", "Probability of executing oracle action at each timestep"),
    record=(True, "bool", "Record video."),
    record_frame_skip=(5, "integer", "Skip between recorded frames."),
    # RoMemo wrapper
    romemo_k=(20, "integer", "RoMemo: top-k retrieved items."),
    romemo_alpha=(0.7, "float", "RoMemo: alpha weight on base action prior."),
    romemo_lambda_fail=(1.5, "float", "RoMemo: penalty weight for failures."),
    romemo_beta_failrate=(0.5, "float", "RoMemo: risk penalty for fail-rate in neighbors."),
    romemo_beta_repeat=(0.2, "float", "RoMemo: penalty per repeat-failure count."),
    romemo_max_mem_candidates=(5, "integer", "RoMemo: add top-N memory actions to candidate set."),
    romemo_max_memory_items=(
        None,
        "string",
        "RoMemo: optional cap on stored memory items (int or None).",
    ),
    romemo_init_memory_path=(
        None,
        "string",
        "RoMemo: optional MemoryBank .pt to preload (read-only unless *_wb).",
    ),
    romemo_save_memory_path=(
        None,
        "string",
        "RoMemo: optional path to save MemoryBank .pt at end.",
    ),
    trace_jsonl=(True, "bool", "Write step/episode traces as JSONL."),
    # MCTS baseline
    mcts_sims=(50, "integer", "MCTS: number of simulations per step."),
    mcts_depth=(2, "integer", "MCTS: max tree depth."),
    mcts_rollout_depth=(20, "integer", "MCTS: expert rollout cap (steps)."),
    mcts_c_uct=(0.5, "float", "MCTS: UCT exploration constant."),
    mcts_proposal_observation=(
        "root",
        "string",
        "MCTS: proposal observation source: 'root' (fast) or 'node' (slow, renders per sim).",
    ),
    logging=WandBLogger.get_default_config(),
)


def _safe_int_or_none(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    ss = str(s).strip().lower()
    if ss in {"none", ""}:
        return None
    return int(ss)


def _jsonl_append(path, obj):
    import json

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _looping_metrics(actions, fails):
    """
    actions: list[str]
    fails: list[bool] aligned to actions (True when that step failed)
    """
    n = len(actions)
    if n <= 1:
        return {
            "looping_rate": 0.0,
            "repeated_failure_rate": 0.0,
            "num_retries": 0,
            "num_repeat_actions": 0,
            "num_repeated_failures": 0,
        }
    repeat_actions = 0
    repeated_failures = 0
    num_retries = 0
    for i in range(1, n):
        if actions[i] == actions[i - 1]:
            repeat_actions += 1
        if bool(fails[i]) and bool(fails[i - 1]) and actions[i] == actions[i - 1]:
            repeated_failures += 1
        if bool(fails[i - 1]) and actions[i] == actions[i - 1]:
            num_retries += 1
    denom = max(1, n - 1)
    return {
        "looping_rate": float(repeat_actions / denom),
        "repeated_failure_rate": float(repeated_failures / denom),
        "num_retries": int(num_retries),
        "num_repeat_actions": int(repeat_actions),
        "num_repeated_failures": int(repeated_failures),
    }


def imagine_with_sim(env, agent, first_action, goal_img, history, obj_labels, traj_dir, t):
    env_recording = env.is_recording
    if env_recording:
        env._record = False
    _env_state = env.__getstate__()
    _plan = [first_action]
    for i in range(FLAGS.imagine_future_steps):
        try:
            _err = env.act_txt(_plan[-1])
            Image.fromarray(env.read_pixels(camera_name=FLAGS.camera_name)).save(
                os.path.join(traj_dir, f"sim-{t}-{'-'.join(_plan).replace(' ', '_')}.png")
            )
        except Exception as e:
            print(f"Error during imagining into future: {e}")
            break

        if i + 1 == FLAGS.imagine_future_steps or env.is_success():
            break

        _img = env.read_pixels(camera_name=FLAGS.camera_name)
        _a = agent.act(
            _img,
            goal_img,
            inp=get_prompt(version="propose", history=history + _plan, obj_labels=obj_labels),
        )
        if _a == "done":
            break
        _plan.append(_a)

    future_img = env.read_pixels(camera_name=FLAGS.camera_name)
    env.__setstate__(_env_state)
    if env_recording:
        env._record = True

    return _plan, future_img


def imagine_with_diffusion(
    diffusion_sim, agent, first_action, img, goal_img, history, obj_labels, traj_dir, t
):
    _plan = [first_action]
    _img = img
    for i in range(FLAGS.imagine_future_steps):
        try:
            next_img = diffusion_sim.forward(curr_image=_img, act_text=_plan[-1])
            Image.fromarray(next_img).save(
                os.path.join(traj_dir, f"gen-{t}-{'-'.join(_plan).replace(' ', '_')}.png")
            )
        except Exception as e:
            print(f"Error during imagining into future with diffusion: {e}")
            break

        _img = next_img
        if i + 1 == FLAGS.imagine_future_steps:
            break

        _a = agent.act(
            _img,
            goal_img,
            inp=get_prompt(version="propose", history=history + _plan, obj_labels=obj_labels),
        )
        if _a == "done":
            break

        _plan.append(_a)

    future_img = _img

    return _plan, future_img


def build_env(env_seed, xml_filename, render_mode="offscreen"):
    xml, info = generate_xml(seed=env_seed)
    if (FLAGS.level == "medium" and info["n_bodies"] > 5) or (
        FLAGS.level == "hard" and info["n_bodies"] <= 5
    ):
        return None, None
    xml.write_to_file(filename=xml_filename)

    board_name = "brick_1"
    fixture_name = None  # "fixture"
    peg_ids = [j + 1 for j in range(1, info["n_bodies"])]
    peg_names = [f"brick_{j + 1}" for j in range(1, info["n_bodies"])]
    peg_descriptions = [info["brick_descriptions"][peg_name] for peg_name in peg_names]

    peg_labels = [" ".join(pd.split()[:1]) for pd in peg_descriptions]
    peg_labels_shuffled = peg_labels.copy()
    random.shuffle(peg_labels_shuffled)

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
    )
    env_info = {
        "peg_ids": peg_ids,
        "peg_names": peg_names,
        "peg_descriptions": peg_descriptions,
        "peg_labels": peg_labels,
        "peg_labels_shuffled": peg_labels_shuffled,
        "dependencies": info["dependencies"],
    }

    return env, env_info


def main(_):
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    set_random_seed(FLAGS.seed)
    env_seed = FLAGS.seed
    render_mode = "offscreen"
    candidate_act_list = ["pick up", "put down", "insert", "reorient"]
    xml_filename = full_path_for(f"tmp_{uuid.uuid4()}.xml")

    # check flags
    supported = {
        "llava",
        "bc",
        "mcts",
        "reflect",
        "reflect_romemo",
        "reflect_romemo_wb",
        "expert",
        "random",
        "expert_romemo_wb",
        "llava_romemo",
        "llava_romemo_wb",
        "bc_romemo",
        "bc_romemo_wb",
        "mcts_romemo",
        "mcts_romemo_wb",
    }
    assert FLAGS.agent_type in supported, f"Unknown agent type `{FLAGS.agent_type}`"
    assert FLAGS.level in {"medium", "hard", "all"}, f"Unknown assembly level `{FLAGS.level}`"
    if FLAGS.revise_action and not str(FLAGS.agent_type).startswith("reflect"):
        from roboworld.agent.diffuser import DiffusionSim

        assert FLAGS.agent_type in {
            "llava",
            "bc",
            "llava_romemo",
            "llava_romemo_wb",
            "bc_romemo",
            "bc_romemo_wb",
        }
        if FLAGS.diffuser_pretrained_model is not None:
            diffusion_sim = DiffusionSim(
                pretrained_model=FLAGS.diffuser_pretrained_model,
                unet_dir=FLAGS.diffuser_unet_dir,
                vae_dir=FLAGS.diffuser_vae_dir,
            )
        else:
            diffusion_sim = None

    # initialize agents that should persist across envs (e.g., heavy VLM)
    base_llava_agent = None
    if FLAGS.agent_type in {
        "llava",
        "bc",
        "mcts",
        "reflect",
        "reflect_romemo",
        "reflect_romemo_wb",
        "mcts_romemo",
        "mcts_romemo_wb",
        "llava_romemo",
        "llava_romemo_wb",
        "bc_romemo",
        "bc_romemo_wb",
        "expert_romemo_wb",  # Added: Expert needs vision encoder for memory generation
    }:
        from roboworld.agent.llava import LlavaAgent

        base_llava_agent = LlavaAgent(
            model_path=FLAGS.model_path,
            model_base=FLAGS.model_base,
            load_8bit=FLAGS.load_8bit,
            load_4bit=FLAGS.load_4bit,
        )

    # Shared RoMemo store across boards (test-time adaptation across tasks)
    romemo_cfg = None
    romemo_store = None
    if FLAGS.agent_type in {
        "llava_romemo",
        "llava_romemo_wb",
        "bc_romemo",
        "bc_romemo_wb",
        "mcts_romemo",
        "mcts_romemo_wb",
        "reflect_romemo",
        "reflect_romemo_wb",
        "expert_romemo_wb",
    }:
        from roboworld.agent.romemo_stack import RoMemoDiscreteConfig, RoMemoStore

        romemo_cfg = RoMemoDiscreteConfig(
            k=int(FLAGS.romemo_k),
            alpha=float(FLAGS.romemo_alpha),
            lambda_fail=float(FLAGS.romemo_lambda_fail),
            beta_failrate=float(FLAGS.romemo_beta_failrate),
            beta_repeat=float(FLAGS.romemo_beta_repeat),
            max_mem_candidates=int(FLAGS.romemo_max_mem_candidates),
        )
        romemo_store = RoMemoStore(
            task="assembly",
            cfg=romemo_cfg,
            writeback=bool(str(FLAGS.agent_type).endswith("_wb")),
            seed=int(FLAGS.agent_seed),
            init_memory_path=FLAGS.romemo_init_memory_path,
        )

    board_cnt, traj_cnt, succ_cnt = 0, 0, 0
    data = []
    traj_id = FLAGS.start_traj_id
    board_id = FLAGS.start_board_id

    while traj_id < FLAGS.start_traj_id + FLAGS.n_trajs:
        env, env_info = build_env(env_seed, xml_filename, render_mode)
        if env is None:
            env_seed += 1
            continue
        oracle = AssemblyOracle(
            env=env,
            brick_ids=env_info["peg_ids"],
            brick_descriptions=env_info["peg_descriptions"],
            dependencies=env_info["dependencies"],
        )
        oracle_agent = OracleAgent(oracle)

        # Per-board agent construction (env-specific wrappers / MCTS)
        from roboworld.agent.mcts import MCTSAgent
        from roboworld.agent.romemo_stack import RoMemoDiscreteAgent

        def build_agent_for_env():
            if FLAGS.agent_type in {"llava", "bc"}:
                return base_llava_agent
            if FLAGS.agent_type == "reflect":
                from roboworld.agent.reflect_wrapper import (
                    ReflectWrapperAgent,
                    ReflectWrapperConfig,
                )

                cfg = ReflectWrapperConfig(
                    imagine_future_steps=int(FLAGS.imagine_future_steps)
                    if int(FLAGS.imagine_future_steps) > 0
                    else 5,
                    camera_name=str(FLAGS.camera_name),
                )
                return ReflectWrapperAgent(
                    env=env,
                    base_agent=base_llava_agent,
                    obj_labels=env_info["peg_labels_shuffled"],
                    cfg=cfg,
                )
            if FLAGS.agent_type == "mcts":
                return MCTSAgent(
                    env=env,
                    proposer_agent=base_llava_agent,
                    oracle_agent=oracle_agent,
                    obj_labels=env_info["peg_labels_shuffled"],
                    camera_name=str(FLAGS.camera_name),
                    proposal_k=5,
                    proposal_observation=str(FLAGS.mcts_proposal_observation),
                    seed=int(FLAGS.agent_seed),
                    num_simulations=int(FLAGS.mcts_sims),
                    max_depth=int(FLAGS.mcts_depth),
                    rollout_depth=int(FLAGS.mcts_rollout_depth),
                    c_uct=float(FLAGS.mcts_c_uct),
                )
            if FLAGS.agent_type == "expert_romemo_wb":
                # Wrap oracle policy so it fits base_agent.act(...) interface
                class _OracleBase:
                    def __init__(self, oa, encoder_agent):
                        self.oa = oa
                        self.encoder_agent = encoder_agent

                    def act(self, img, goal_img, inp, next_image=None):
                        return self.oa.act(img, goal_img, inp)

                    def encode_image(self, image):
                        if self.encoder_agent and hasattr(self.encoder_agent, "encode_image"):
                            return self.encoder_agent.encode_image(image)
                        raise NotImplementedError("Encoder agent missing or does not support encode_image")

                base = _OracleBase(oracle_agent, base_llava_agent)
                return RoMemoDiscreteAgent(
                    base_agent=base,
                    env=env,
                    task="assembly",
                    cfg=romemo_cfg,
                    writeback=True,
                    seed=int(FLAGS.agent_seed),
                    shared_store=romemo_store,
                )
            if FLAGS.agent_type in {"llava_romemo", "bc_romemo"}:
                return RoMemoDiscreteAgent(
                    base_agent=base_llava_agent,
                    env=env,
                    task="assembly",
                    cfg=romemo_cfg,
                    writeback=False,
                    seed=int(FLAGS.agent_seed),
                    shared_store=romemo_store,
                )
            if FLAGS.agent_type == "reflect_romemo":
                from roboworld.agent.reflect_wrapper import (
                    ReflectWrapperAgent,
                    ReflectWrapperConfig,
                )

                cfg = ReflectWrapperConfig(
                    imagine_future_steps=int(FLAGS.imagine_future_steps)
                    if int(FLAGS.imagine_future_steps) > 0
                    else 5,
                    camera_name=str(FLAGS.camera_name),
                )
                base = ReflectWrapperAgent(
                    env=env,
                    base_agent=base_llava_agent,
                    obj_labels=env_info["peg_labels_shuffled"],
                    cfg=cfg,
                )
                return RoMemoDiscreteAgent(
                    base_agent=base,
                    env=env,
                    task="assembly",
                    cfg=romemo_cfg,
                    writeback=False,
                    seed=int(FLAGS.agent_seed),
                    shared_store=romemo_store,
                )
            if FLAGS.agent_type in {"llava_romemo_wb", "bc_romemo_wb"}:
                return RoMemoDiscreteAgent(
                    base_agent=base_llava_agent,
                    env=env,
                    task="assembly",
                    cfg=romemo_cfg,
                    writeback=True,
                    seed=int(FLAGS.agent_seed),
                    shared_store=romemo_store,
                )
            if FLAGS.agent_type == "reflect_romemo_wb":
                from roboworld.agent.reflect_wrapper import (
                    ReflectWrapperAgent,
                    ReflectWrapperConfig,
                )

                cfg = ReflectWrapperConfig(
                    imagine_future_steps=int(FLAGS.imagine_future_steps)
                    if int(FLAGS.imagine_future_steps) > 0
                    else 5,
                    camera_name=str(FLAGS.camera_name),
                )
                base = ReflectWrapperAgent(
                    env=env,
                    base_agent=base_llava_agent,
                    obj_labels=env_info["peg_labels_shuffled"],
                    cfg=cfg,
                )
                return RoMemoDiscreteAgent(
                    base_agent=base,
                    env=env,
                    task="assembly",
                    cfg=romemo_cfg,
                    writeback=True,
                    seed=int(FLAGS.agent_seed),
                    shared_store=romemo_store,
                )
            if FLAGS.agent_type == "mcts_romemo":
                base = MCTSAgent(
                    env=env,
                    proposer_agent=base_llava_agent,
                    oracle_agent=oracle_agent,
                    obj_labels=env_info["peg_labels_shuffled"],
                    camera_name=str(FLAGS.camera_name),
                    proposal_k=5,
                    proposal_observation=str(FLAGS.mcts_proposal_observation),
                    seed=int(FLAGS.agent_seed),
                    num_simulations=int(FLAGS.mcts_sims),
                    max_depth=int(FLAGS.mcts_depth),
                    rollout_depth=int(FLAGS.mcts_rollout_depth),
                    c_uct=float(FLAGS.mcts_c_uct),
                )
                return RoMemoDiscreteAgent(
                    base_agent=base,
                    env=env,
                    task="assembly",
                    cfg=romemo_cfg,
                    writeback=False,
                    seed=int(FLAGS.agent_seed),
                    shared_store=romemo_store,
                )
            if FLAGS.agent_type == "mcts_romemo_wb":
                base = MCTSAgent(
                    env=env,
                    proposer_agent=base_llava_agent,
                    oracle_agent=oracle_agent,
                    obj_labels=env_info["peg_labels_shuffled"],
                    camera_name=str(FLAGS.camera_name),
                    proposal_k=5,
                    proposal_observation=str(FLAGS.mcts_proposal_observation),
                    seed=int(FLAGS.agent_seed),
                    num_simulations=int(FLAGS.mcts_sims),
                    max_depth=int(FLAGS.mcts_depth),
                    rollout_depth=int(FLAGS.mcts_rollout_depth),
                    c_uct=float(FLAGS.mcts_c_uct),
                )
                return RoMemoDiscreteAgent(
                    base_agent=base,
                    env=env,
                    task="assembly",
                    cfg=romemo_cfg,
                    writeback=True,
                    seed=int(FLAGS.agent_seed),
                    shared_store=romemo_store,
                )
            if FLAGS.agent_type == "expert":
                return None  # handled inline
            if FLAGS.agent_type == "random":
                return None  # handled inline
            raise ValueError(f"Unsupported agent_type: {FLAGS.agent_type}")

        agent = build_agent_for_env()

        idx_in_env = 0
        reset_seed = FLAGS.reset_seed_start
        while idx_in_env < FLAGS.repeat_per_env:
            print(">" * 10, f"Trajectory {traj_id}", ">" * 10)
            traj_dir = os.path.join(FLAGS.save_dir, str(traj_id))
            os.makedirs(traj_dir, exist_ok=True)
            reset_seed += 1
            env.reset(seed=reset_seed)
            goal_img = env.goal_images.get(FLAGS.camera_name, None)
            assert goal_img is not None

            # initialize data
            traj_succ = False
            exec_act_list = [None] * FLAGS.max_steps
            oracle_act_list = [None] * FLAGS.max_steps
            agent_act_list = [None] * FLAGS.max_steps
            base_act_list = [None] * FLAGS.max_steps
            agent_plan_list = [None] * FLAGS.max_steps
            agent_act_revised_list = [None] * FLAGS.max_steps
            step_fail_list = [False] * FLAGS.max_steps
            traj_key_frames = []
            question_list = []
            history = []
            history_list = []
            total_time, inference_time, rollout_time = 0.0, 0.0, 0.0

            if FLAGS.record:
                env.record_on(record_frame_skip=FLAGS.record_frame_skip)
                env.render()

            trace_dir = os.path.join(FLAGS.save_dir, "traces")
            if FLAGS.trace_jsonl:
                os.makedirs(trace_dir, exist_ok=True)
                step_trace_path = os.path.join(trace_dir, "step_traces.jsonl")
                ep_trace_path = os.path.join(trace_dir, "episode_traces.jsonl")

            # start rollout
            total_time_t0 = time.time()
            for t in range(FLAGS.max_steps):
                print(f"[Step {t}]")
                oracle_action = oracle_agent.act()
                oracle_act_list[t] = oracle_action
                oracle_action_primitive, oracle_brick_color = parse_act_txt(oracle_action)
                print("Oracle action:", oracle_action)

                img = env.read_pixels(camera_name=FLAGS.camera_name)
                traj_key_frames.append(img)

                inp = get_prompt(
                    version="propose", history=history, obj_labels=env_info["peg_labels_shuffled"]
                )
                print("Q:", inp)
                question_list.append(inp)
                history_list.append(copy.deepcopy(history))

                inference_t0, rollout_t0, rollout_t = None, None, None
                agent_action_revised = None

                try:
                    # get action with agent
                    if FLAGS.agent_type == "expert":
                        agent_action = oracle_action
                    elif FLAGS.agent_type == "random":
                        agent_action = f"{np.random.choice(candidate_act_list)} {np.random.choice(env_info['peg_labels'])}"
                    else:
                        inference_t0 = time.time()
                        agent_action = agent.act(img, goal_img, inp)
                        # RoMemo wrapper exposes base_action via last_trace if applicable
                        if getattr(agent, "last_trace", None) and isinstance(
                            agent.last_trace, dict
                        ):
                            base_act_list[t] = agent.last_trace.get("base_action", agent_action)
                        else:
                            base_act_list[t] = agent_action

                        # reflect and revise action (handled internally for reflect* agents)
                        if FLAGS.revise_action and not str(FLAGS.agent_type).startswith("reflect"):
                            assert FLAGS.imagine_future_steps > 0

                            if diffusion_sim is None:
                                _plan, future_img = imagine_with_sim(
                                    env=env,
                                    agent=agent,
                                    first_action=agent_action,
                                    goal_img=goal_img,
                                    history=history,
                                    obj_labels=env_info["peg_labels_shuffled"],
                                    traj_dir=traj_dir,
                                    t=t,
                                )
                            else:
                                _plan, future_img = imagine_with_diffusion(
                                    diffusion_sim=diffusion_sim,
                                    agent=agent,
                                    first_action=agent_action,
                                    img=img,
                                    goal_img=goal_img,
                                    history=history,
                                    obj_labels=env_info["peg_labels_shuffled"],
                                    traj_dir=traj_dir,
                                    t=t,
                                )
                            agent_plan_list[t] = copy.deepcopy(_plan)

                            if FLAGS.dummy_revised_action:
                                agent_action_revised = agent_action
                            else:
                                inp2 = get_prompt(
                                    version="reflect",
                                    history=history,
                                    obj_labels=env_info["peg_labels_shuffled"],
                                    initial_plan=_plan,
                                )
                                print("Q2:", inp2)
                                agent_action_revised = agent.act(
                                    img, goal_img, inp2, next_image=future_img
                                )

                            print("*" * 20, f"Step {t} reflection summary", "*" * 20)
                            print(
                                f"Initial plan: {_plan}\n=> Revised action: {agent_action_revised}"
                            )
                            print("*" * 20, "End of reflection summary", "*" * 20)

                        inference_time += time.time() - inference_t0

                    agent_act_list[t] = agent_action
                    print("A:", agent_action)

                    # process revised action
                    if not str(FLAGS.agent_type).startswith("reflect"):
                        assert FLAGS.revise_action == (agent_action_revised is not None)
                    if agent_action_revised is not None:
                        try:
                            assert len(agent_action_revised.strip().split(" ")) <= 3, (
                                "Bad output format."
                            )
                            _p, _o = parse_act_txt(agent_action_revised)
                            assert _p in candidate_act_list and _o in env_info["peg_labels"], (
                                "Bad output format."
                            )
                        except Exception as e:
                            print(
                                f"Error during parsing `agent_action_revised`({agent_action_revised}): {e}"
                            )
                            agent_action_revised = agent_action
                        agent_act_revised_list[t] = agent_action_revised
                        agent_action = agent_action_revised  # Use the revised action as the final action to execute
                    print("A2:", agent_action)

                    # parse and choose action
                    agent_action_primitive, agent_brick_color = parse_act_txt(agent_action)
                    if random.uniform(0, 1) < FLAGS.oracle_prob:
                        exec_action_primitive = oracle_action_primitive
                        exec_brick_color = oracle_brick_color
                    else:
                        exec_action_primitive = agent_action_primitive
                        exec_brick_color = agent_brick_color
                    exec_action = (
                        "done"
                        if exec_action_primitive == "done"
                        else " ".join([exec_action_primitive, exec_brick_color])
                    )
                    exec_act_list[t] = exec_action

                    # add action to history
                    history.append(exec_action)

                    # execute
                    print("Executed action:", exec_action)
                    if exec_brick_color not in env.peg_colors:
                        print(f"Unknown object '{exec_brick_color}'")
                        step_fail_list[t] = True
                        continue
                    if exec_action == "done":
                        rollout_t = 0
                        break
                    else:
                        rollout_t0 = time.time()
                        err = env.act_txt(exec_action)
                        if err != 0:
                            print(f"Error {err} when executing `{exec_action}`")
                            step_fail_list[t] = True
                        rollout_t = time.time() - rollout_t0

                    # RoMemo writeback (if supported)
                    if hasattr(agent, "update"):
                        try:
                            agent.update(
                                exec_action,
                                int(err if exec_action != "done" else 0),
                                episode_id=traj_id,
                                step_id=t,
                            )
                        except TypeError:
                            # tolerate different signature
                            try:
                                agent.update(exec_action, int(err if exec_action != "done" else 0))
                            except Exception:
                                pass

                    # Step trace logging
                    if FLAGS.trace_jsonl:
                        st = {
                            "traj_id": int(traj_id),
                            "env_seed": int(env_seed),
                            "reset_seed": int(reset_seed),
                            "step_id": int(t),
                            "agent_type": str(FLAGS.agent_type),
                            "agent_seed": int(FLAGS.agent_seed),
                            "oracle_action": str(oracle_action),
                            "agent_action": str(agent_act_list[t]),
                            "base_action": str(base_act_list[t])
                            if base_act_list[t] is not None
                            else str(agent_act_list[t]),
                            "exec_action": str(exec_action),
                            "err_code": int(err if exec_action != "done" else 0),
                            "step_fail": bool(step_fail_list[t]),
                            "is_success_after": bool(env.is_success()),
                        }
                        if getattr(agent, "last_trace", None) and isinstance(
                            agent.last_trace, dict
                        ):
                            st["romemo"] = agent.last_trace
                        if hasattr(agent, "last_debug") and isinstance(
                            getattr(agent, "last_debug"), dict
                        ):
                            st["mcts"] = getattr(agent, "last_debug")
                        _jsonl_append(step_trace_path, st)

                except Exception as e:
                    print(traceback.format_exc())
                    if rollout_t is None:
                        if rollout_t0 is None:
                            rollout_t = 0
                        else:
                            rollout_t = time.time() - rollout_t0
                    break  # fail

                rollout_time += rollout_t

                # check success
                if env.is_success():
                    traj_succ = True
                    succ_cnt += 1
                    break

            last_img = env.read_pixels(camera_name=FLAGS.camera_name)
            total_time = time.time() - total_time_t0

            if env.is_recording:
                env.record_off()

            traj_key_frames.append(last_img)
            last_img_path = os.path.join(traj_dir, f"{len(question_list)}.png")
            Image.fromarray(last_img).save(last_img_path)
            goal_img_path = os.path.join(traj_dir, "goal.png")
            Image.fromarray(goal_img).save(goal_img_path)

            # log data
            print("Success:", traj_succ)
            print(f"Inference time: {inference_time}, Rollout time: {rollout_time}")
            wandb_logger.log(
                {
                    "success": int(traj_succ),
                    "accumulated_success_cnt": succ_cnt,
                    "accumulated_success_rate": succ_cnt / (traj_cnt + 1),
                    "total_inference_time": inference_time,
                    "total_rollout_time": rollout_time,
                    "average_inference_time": inference_time / len(question_list),
                    "average_rollout_time": rollout_time / len(question_list),
                    "total_steps": len(question_list),
                    "total_time": total_time,
                },
                step=traj_cnt,
            )

            if FLAGS.record:
                wandb_logger.log_video(
                    np.stack(env.frames).transpose((0, 3, 1, 2)),
                    fps=60,
                    caption=f"traj{traj_id}_seed{env_seed}_{reset_seed}_{['fail', 'succ'][traj_succ]}",
                )

            for i, question in enumerate(question_list):
                img_path = os.path.join(traj_dir, f"{i}.png")
                Image.fromarray(traj_key_frames[i]).save(img_path)

                entry = {
                    "trajectory_id": traj_id,
                    "board_id": board_id,
                    "env_seed": env_seed,
                    "reset_seed": reset_seed,
                    "step_id": i,
                    "action_description": exec_act_list[i],
                    "oracle_action": oracle_act_list[i],
                    "agent_action": agent_act_list[i],
                    "history": history_list[i],
                    "image": f"{traj_id}/{i}.png",
                    "next_image": f"{traj_id}/{i + 1}.png",
                    "final_goal_image": f"{traj_id}/goal.png",
                    "traj_success": int(traj_succ),
                    "traj_total_steps": len(question_list),
                    "object_descriptions": copy.deepcopy(env_info["peg_descriptions"]),
                    "object_dependencies": copy.deepcopy(env_info["dependencies"]),
                }
                if FLAGS.revise_action:
                    entry["agent_action_revised"] = agent_act_revised_list[i]
                    entry["agent_plan"] = copy.deepcopy(agent_plan_list[i])

                data.append(entry)

            # Episode trace
            if FLAGS.trace_jsonl:
                acts = [a for a in exec_act_list[: len(question_list)] if a is not None]
                fails = [bool(x) for x in step_fail_list[: len(question_list)]]
                loop = _looping_metrics(acts, fails)
                ep = {
                    "traj_id": int(traj_id),
                    "board_id": int(board_id),
                    "env_seed": int(env_seed),
                    "reset_seed": int(reset_seed),
                    "agent_type": str(FLAGS.agent_type),
                    "agent_seed": int(FLAGS.agent_seed),
                    "success": bool(traj_succ),
                    "steps": int(len(question_list)),
                    **loop,
                }
                _jsonl_append(ep_trace_path, ep)

            idx_in_env += 1
            traj_id += 1
            traj_cnt += 1

        env.close()
        env_seed += 1
        board_id += 1
        board_cnt += 1

    # save data
    df = pd.DataFrame(data)
    meta_path = os.path.join(FLAGS.save_dir, "meta.csv")
    df.to_csv(meta_path)

    # log summary
    print("=" * 20)
    print("Total # boards:", board_cnt)
    print("Total # trajectories:", traj_cnt)
    print("Total # samples(steps):", len(data))
    succ_rate = succ_cnt / traj_cnt
    print(f"Success rate: {succ_rate} ({succ_cnt}/{traj_cnt})")
    wandb_logger.log({"success_rate": succ_rate})

    # Save RoMemo memory snapshot for reproducibility / reuse by *_romemo (no wb) runs
    if romemo_store is not None and FLAGS.romemo_save_memory_path is not None:
        try:
            romemo_store.save_pt(str(FLAGS.romemo_save_memory_path))
            print(f"[romemo] saved memory to {FLAGS.romemo_save_memory_path}")
        except Exception as e:
            print(f"[romemo] failed to save memory: {e}")

    os.remove(xml_filename)


if __name__ == "__main__":
    app.run(main)
