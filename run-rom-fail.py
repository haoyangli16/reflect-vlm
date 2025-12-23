# run-rom-fail.py: Failure Memory Collection Runner
# =================================================
# This script is specifically designed to collect FAILURE data with Oracle diagnosis.
# Philosophy: "The Simulator is the Compiler. The Oracle is the Linter."
# We record WHY actions failed to learn Constraints.
#
# Key differences from run-rom.py:
# 1. Uses Oracle to diagnose failures (BLOCKED_P, BLOCKED_S, BAD_D, BAD_B)
# 2. Captures full action history for each failure
# 3. Writes ONLY failures to memory (success memory is separate)
# 4. Adds explicit fail_tag to each Experience

# Fix LLVM command-line option conflict between triton and bitsandbytes
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
from typing import Optional, Dict, Any, List
import numpy as np

import mujoco  # type: ignore  # noqa: F401

from roboworld.utils.config import define_flags, get_user_flags
from roboworld.utils.logger import WandBLogger, set_random_seed
from roboworld.envs.generator import generate_xml
from roboworld.envs.asset_path_utils import full_path_for
from roboworld.envs.mujoco.franka.franka_assembly import FrankaAssemblyEnv, AssemblyOracle, State
from roboworld.agent.oracle import OracleAgent
from roboworld.agent.utils import parse_act_txt, get_prompt

FLAGS = flags.FLAGS

FLAGS_DEF = define_flags(
    agent_type=(
        "bc",
        "string",
        "Type of agent to generate failures (bc, llava, random, bc_romemo_wb)",
    ),
    agent_seed=(0, "integer", "Seed for agent randomness."),
    camera_name=("table_back", "string", "Camera name."),
    model_path=("/path/to/model", "string", "Path to model"),
    model_base=(None, "string", "Path to base model"),
    load_8bit=(False, "bool", "Load model in 8bit mode"),
    load_4bit=(False, "bool", "Load model in 4bit mode"),
    level=("all", "string", "Level of difficulty (medium, hard, or all)"),
    seed=(42, "integer", "Random seed."),
    reset_seed_start=(0, "integer", "first seed to reset env"),
    max_steps=(50, "integer", "Max number of decision steps in a trajectory"),
    n_trajs=(100, "integer", "Number of trajectories."),
    repeat_per_env=(1, "integer", "Number of trajectories for each env/board."),
    save_dir=("datasets/failure_data", "string", "Directory for saving data."),
    start_traj_id=(0, "integer", "Starting trajectory index"),
    start_board_id=(0, "integer", "Starting board index"),
    oracle_prob=(
        0.0,
        "float",
        "Probability of executing oracle action (0.0 for pure agent failures)",
    ),
    record=(False, "bool", "Record video."),
    record_frame_skip=(5, "integer", "Skip between recorded frames."),
    # RoMemo wrapper for failure collection
    romemo_k=(20, "integer", "RoMemo: top-k retrieved items."),
    romemo_alpha=(0.2, "float", "RoMemo: alpha weight on base action prior."),
    romemo_lambda_fail=(1.5, "float", "RoMemo: penalty weight for failures."),
    romemo_beta_failrate=(0.5, "float", "RoMemo: risk penalty for fail-rate in neighbors."),
    romemo_beta_repeat=(0.2, "float", "RoMemo: penalty per repeat-failure count."),
    romemo_max_mem_candidates=(5, "integer", "RoMemo: add top-N memory actions to candidate set."),
    romemo_init_memory_path=(
        None,
        "string",
        "RoMemo: optional MemoryBank .pt to preload.",
    ),
    romemo_save_memory_path=(
        None,
        "string",
        "RoMemo: path to save FailureMemory .pt at end.",
    ),
    # NEW: Retrieval mode for state-query based retrieval
    romemo_retrieval_mode=(
        "visual",
        "string",
        "RoMemo: retrieval mode - 'visual' (default), 'symbolic', or 'hybrid'.",
    ),
    romemo_symbolic_weight=(
        0.5,
        "float",
        "RoMemo: weight for symbolic filtering in hybrid mode (0=visual only, 1=symbolic only).",
    ),
    romemo_min_symbolic_candidates=(
        5,
        "integer",
        "RoMemo: minimum candidates from symbolic filter before fallback to visual.",
    ),
    trace_jsonl=(True, "bool", "Write step/episode traces as JSONL."),
    save_images=(False, "bool", "Save trajectory images to disk."),
    # Focus on failure collection
    write_on_success=(
        False,
        "bool",
        "Also write success experiences (default: False for pure failure memory)",
    ),
    stop_on_failure=(
        True,
        "bool",
        "Stop episode after first failure and move to next environment (default: True)",
    ),
    logging=WandBLogger.get_default_config(),
)


def _jsonl_append(path, obj):
    import json

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def diagnose_failure(
    env,
    oracle: AssemblyOracle,
    exec_action: str,
    err_code: int,
    history: List[str],
) -> Dict[str, Any]:
    """
    Diagnose WHY an action failed using Oracle knowledge.

    Returns:
        dict with keys:
            - fail_tag: str (the constraint type)
            - oracle_state_context: str (raw oracle state for debugging)
            - inferred_constraint: str (human-readable explanation)
    """
    exec_action = str(exec_action).strip()

    # Parse action
    try:
        prim, color = parse_act_txt(exec_action)
    except Exception:
        return {
            "fail_tag": "PARSE_ERROR",
            "oracle_state_context": None,
            "inferred_constraint": "Could not parse action format",
        }

    if exec_action == "done" or prim == "done":
        if not env.is_success():
            return {
                "fail_tag": "BAD_DONE",
                "oracle_state_context": "NOT_SUCCESS",
                "inferred_constraint": "Called done but task not complete",
            }
        return {"fail_tag": None, "oracle_state_context": "SUCCESS", "inferred_constraint": None}

    # Find target object
    try:
        idx = env.peg_colors.index(color)
        body = env.peg_names[idx]
        obj_id = int(body.split("_")[-1])
    except Exception:
        return {
            "fail_tag": "UNKNOWN_OBJECT",
            "oracle_state_context": None,
            "inferred_constraint": f"Object '{color}' not found in environment",
        }

    # Get oracle state for the target object
    try:
        obj_state = oracle.state.get(obj_id, None)
        obj_state_name = State(obj_state).name if obj_state is not None else "UNKNOWN"
    except Exception:
        obj_state_name = "UNKNOWN"

    # ========================================
    # Explicit Error Codes (Logic & Preconditions)
    # ========================================
    if int(err_code) != 0:
        if prim == "pick up":
            if err_code == -1:
                return {
                    "fail_tag": "HAND_FULL",
                    "oracle_state_context": obj_state_name,
                    "inferred_constraint": "Cannot pick up: another object is in hand",
                }
            elif err_code == -2:
                return {
                    "fail_tag": "REDUNDANT_PICKUP",
                    "oracle_state_context": obj_state_name,
                    "inferred_constraint": "Object is already in hand",
                }
        elif prim == "put down":
            if err_code == -1:
                return {
                    "fail_tag": "EMPTY_HAND",
                    "oracle_state_context": obj_state_name,
                    "inferred_constraint": "Cannot put down: hand is empty",
                }
        elif prim == "insert":
            if err_code == -1:
                return {
                    "fail_tag": "NOT_HOLDING_FOR_INSERT",
                    "oracle_state_context": obj_state_name,
                    "inferred_constraint": "Cannot insert: object not in hand",
                }
        elif prim == "reorient":
            if err_code == -1:
                return {
                    "fail_tag": "NOT_HOLDING_FOR_REORIENT",
                    "oracle_state_context": obj_state_name,
                    "inferred_constraint": "Cannot reorient: object not in hand",
                }

        # Generic error code
        return {
            "fail_tag": f"ERR_CODE_{err_code}",
            "oracle_state_context": obj_state_name,
            "inferred_constraint": f"Action returned error code {err_code}",
        }

    # ========================================
    # Logic State Constraints (Oracle Knowledge)
    # ========================================
    # These are semantic failures even if err_code == 0

    if obj_state == State.BLOCKED_P:
        return {
            "fail_tag": "BLOCKED_BY_PREDECESSOR",
            "oracle_state_context": obj_state_name,
            "inferred_constraint": "Cannot proceed: a predecessor object must be inserted first",
        }

    if obj_state == State.BLOCKED_S:
        return {
            "fail_tag": "BLOCKED_BY_SUCCESSOR",
            "oracle_state_context": obj_state_name,
            "inferred_constraint": "Cannot insert: a successor object is blocking the hole",
        }

    if obj_state == State.BAD_B:
        return {
            "fail_tag": "BAD_PLACEMENT",
            "oracle_state_context": obj_state_name,
            "inferred_constraint": "Object was placed incorrectly, blocking others",
        }

    if obj_state == State.BAD_D:
        if prim == "insert":
            return {
                "fail_tag": "NEEDS_REORIENT",
                "oracle_state_context": obj_state_name,
                "inferred_constraint": "Object is lying flat; must reorient before insert",
            }

    # ========================================
    # Check for implicit physics failures
    # ========================================
    # These would require checking env state changes, which we approximate

    if prim == "pick up":
        # Check if pickup actually succeeded
        try:
            in_hand = env.get_object_in_hand()
            if in_hand is None or in_hand != body:
                return {
                    "fail_tag": "GRASP_FAILED",
                    "oracle_state_context": obj_state_name,
                    "inferred_constraint": "Grasp failed: object not graspable (geometry/reach issue)",
                }
        except Exception:
            pass

    if prim == "insert":
        # Check if insert actually succeeded
        try:
            if not env.object_is_success(body):
                return {
                    "fail_tag": "INSERT_TIMEOUT",
                    "oracle_state_context": obj_state_name,
                    "inferred_constraint": "Insert failed: alignment timeout or precision issue",
                }
        except Exception:
            pass

    # No failure detected
    return {
        "fail_tag": None,
        "oracle_state_context": obj_state_name,
        "inferred_constraint": None,
    }


def build_env(env_seed, xml_filename, render_mode="offscreen"):
    xml, info = generate_xml(seed=env_seed)
    if (FLAGS.level == "medium" and info["n_bodies"] > 5) or (
        FLAGS.level == "hard" and info["n_bodies"] <= 5
    ):
        return None, None
    xml.write_to_file(filename=xml_filename)

    board_name = "brick_1"
    fixture_name = None
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

    # Check flags
    supported = {"llava", "bc", "random", "bc_romemo_wb"}
    assert FLAGS.agent_type in supported, (
        f"Unknown agent type `{FLAGS.agent_type}` for failure collection"
    )
    assert FLAGS.level in {"medium", "hard", "all"}, f"Unknown assembly level `{FLAGS.level}`"

    # Initialize VLM agent (for bc/llava)
    base_llava_agent = None
    if FLAGS.agent_type in {"llava", "bc", "bc_romemo_wb"}:
        from roboworld.agent.llava import LlavaAgent

        base_llava_agent = LlavaAgent(
            model_path=FLAGS.model_path,
            model_base=FLAGS.model_base,
            load_8bit=FLAGS.load_8bit,
            load_4bit=FLAGS.load_4bit,
        )

    # Shared RoMemo store for failure collection
    romemo_cfg = None
    romemo_store = None
    if FLAGS.agent_type in {"bc_romemo_wb"}:
        from roboworld.agent.romemo_stack import RoMemoDiscreteConfig, RoMemoStore

        romemo_cfg = RoMemoDiscreteConfig(
            k=int(FLAGS.romemo_k),
            alpha=float(FLAGS.romemo_alpha),
            lambda_fail=float(FLAGS.romemo_lambda_fail),
            beta_failrate=float(FLAGS.romemo_beta_failrate),
            beta_repeat=float(FLAGS.romemo_beta_repeat),
            max_mem_candidates=int(FLAGS.romemo_max_mem_candidates),
            # Key: Write failures only (unless explicitly enabled)
            write_on_failure=True,
            write_on_success=bool(FLAGS.write_on_success),
            # NEW: Retrieval mode for state-query based retrieval
            retrieval_mode=str(FLAGS.romemo_retrieval_mode),
            symbolic_weight=float(FLAGS.romemo_symbolic_weight),
            min_symbolic_candidates=int(FLAGS.romemo_min_symbolic_candidates),
        )
        romemo_store = RoMemoStore(
            task="assembly",
            cfg=romemo_cfg,
            writeback=True,
            seed=int(FLAGS.agent_seed),
            init_memory_path=FLAGS.romemo_init_memory_path,
        )

    board_cnt, traj_cnt, succ_cnt = 0, 0, 0
    failure_cnt = 0  # Track total failures collected
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

        # Build agent for this env
        from roboworld.agent.romemo_stack import RoMemoDiscreteAgent

        def build_agent_for_env():
            if FLAGS.agent_type in {"llava", "bc"}:
                return base_llava_agent
            if FLAGS.agent_type == "random":
                return None  # handled inline
            if FLAGS.agent_type == "bc_romemo_wb":
                return RoMemoDiscreteAgent(
                    base_agent=base_llava_agent,
                    env=env,
                    task="assembly",
                    cfg=romemo_cfg,
                    writeback=True,
                    seed=int(FLAGS.agent_seed),
                    shared_store=romemo_store,
                )
            raise ValueError(f"Unsupported agent_type: {FLAGS.agent_type}")

        agent = build_agent_for_env()

        idx_in_env = 0
        reset_seed = FLAGS.reset_seed_start
        while idx_in_env < FLAGS.repeat_per_env:
            print(">" * 10, f"Trajectory {traj_id} (Failure Collection)", ">" * 10)
            traj_dir = os.path.join(FLAGS.save_dir, str(traj_id))
            os.makedirs(traj_dir, exist_ok=True)
            reset_seed += 1
            env.reset(seed=reset_seed)
            goal_img = env.goal_images.get(FLAGS.camera_name, None)
            assert goal_img is not None

            # Initialize data
            traj_succ = False
            exec_act_list = [None] * FLAGS.max_steps
            oracle_act_list = [None] * FLAGS.max_steps
            agent_act_list = [None] * FLAGS.max_steps
            step_fail_list = [False] * FLAGS.max_steps
            fail_diagnosis_list = [None] * FLAGS.max_steps
            traj_key_frames = []
            question_list = []
            history = []  # Full action history for this episode
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
                failure_trace_path = os.path.join(trace_dir, "failure_traces.jsonl")

            # Start rollout
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
                err = 0

                try:
                    # Get action from agent
                    if FLAGS.agent_type == "random":
                        agent_action = f"{np.random.choice(candidate_act_list)} {np.random.choice(env_info['peg_labels'])}"
                    else:
                        inference_t0 = time.time()
                        agent_action = agent.act(img, goal_img, inp)
                        inference_time += time.time() - inference_t0

                    agent_act_list[t] = agent_action
                    print("A:", agent_action)

                    # Choose action (with optional oracle mixing)
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

                    # Add action to history BEFORE execution
                    history.append(exec_action)

                    # Execute
                    print("Executed action:", exec_action)
                    if exec_brick_color not in env.peg_colors and exec_action != "done":
                        print(f"Unknown object '{exec_brick_color}'")
                        step_fail_list[t] = True
                        err = -99  # Custom code for unknown object
                    elif exec_action == "done":
                        rollout_t = 0
                    else:
                        rollout_t0 = time.time()
                        err = env.act_txt(exec_action)
                        if err != 0:
                            print(f"Error {err} when executing `{exec_action}`")
                            step_fail_list[t] = True
                        rollout_t = time.time() - rollout_t0

                    # ========================================
                    # DIAGNOSIS: Query Oracle to understand WHY
                    # ========================================
                    diagnosis = diagnose_failure(
                        env=env,
                        oracle=oracle,
                        exec_action=exec_action,
                        err_code=err,
                        history=history,
                    )
                    fail_diagnosis_list[t] = diagnosis

                    # Mark as failure if diagnosis found a fail_tag
                    if diagnosis["fail_tag"] is not None:
                        step_fail_list[t] = True
                        failure_cnt += 1
                        print(f"  >> FAILURE DIAGNOSED: {diagnosis['fail_tag']}")
                        print(f"     Constraint: {diagnosis['inferred_constraint']}")

                        # Stop episode after first failure if configured
                        if FLAGS.stop_on_failure:
                            print("  >> Stopping episode (stop_on_failure=True)")
                            # Still need to do writeback before breaking
                            if hasattr(agent, "update"):
                                try:
                                    agent.update(
                                        exec_action,
                                        int(err if exec_action != "done" else 0),
                                        episode_id=traj_id,
                                        step_id=t,
                                        fail_tag=diagnosis["fail_tag"],
                                        history=copy.deepcopy(history),
                                        oracle_state_context=diagnosis["oracle_state_context"],
                                        oracle_action=oracle_action,  # Correct action!
                                    )
                                except TypeError:
                                    try:
                                        agent.update(
                                            exec_action, int(err if exec_action != "done" else 0)
                                        )
                                    except Exception:
                                        pass
                            # Log failure trace before breaking
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
                                    "exec_action": str(exec_action),
                                    "err_code": int(err if exec_action != "done" else 0),
                                    "step_fail": True,
                                    "fail_tag": diagnosis["fail_tag"],
                                    "oracle_state_context": diagnosis["oracle_state_context"],
                                    "inferred_constraint": diagnosis["inferred_constraint"],
                                    "history": copy.deepcopy(history),
                                    "is_success_after": bool(env.is_success()),
                                    "stopped_on_failure": True,
                                }
                                _jsonl_append(step_trace_path, st)
                                _jsonl_append(failure_trace_path, st)
                            break  # Exit the step loop, move to next episode

                    # ========================================
                    # WRITEBACK: Store failure in memory
                    # (Skip if already handled in stop_on_failure block above)
                    # ========================================
                    if diagnosis["fail_tag"] is None or not FLAGS.stop_on_failure:
                        if hasattr(agent, "update"):
                            try:
                                agent.update(
                                    exec_action,
                                    int(err if exec_action != "done" else 0),
                                    episode_id=traj_id,
                                    step_id=t,
                                    fail_tag=diagnosis["fail_tag"],
                                    history=copy.deepcopy(history),
                                    oracle_state_context=diagnosis["oracle_state_context"],
                                    oracle_action=oracle_action,  # Correct action!
                                )
                            except TypeError:
                                # Fallback for older signature
                                try:
                                    agent.update(
                                        exec_action, int(err if exec_action != "done" else 0)
                                    )
                                except Exception:
                                    pass

                    # Trace logging (skip if already handled in stop_on_failure block)
                    if FLAGS.trace_jsonl and not (
                        diagnosis["fail_tag"] is not None and FLAGS.stop_on_failure
                    ):
                        st = {
                            "traj_id": int(traj_id),
                            "env_seed": int(env_seed),
                            "reset_seed": int(reset_seed),
                            "step_id": int(t),
                            "agent_type": str(FLAGS.agent_type),
                            "agent_seed": int(FLAGS.agent_seed),
                            "oracle_action": str(oracle_action),
                            "agent_action": str(agent_act_list[t]),
                            "exec_action": str(exec_action),
                            "err_code": int(err if exec_action != "done" else 0),
                            "step_fail": bool(step_fail_list[t]),
                            "fail_tag": diagnosis["fail_tag"],
                            "oracle_state_context": diagnosis["oracle_state_context"],
                            "inferred_constraint": diagnosis["inferred_constraint"],
                            "history": copy.deepcopy(history),
                            "is_success_after": bool(env.is_success()),
                        }
                        _jsonl_append(step_trace_path, st)

                        # Also write to dedicated failure trace if this was a failure
                        if diagnosis["fail_tag"] is not None:
                            _jsonl_append(failure_trace_path, st)

                    # Check for done
                    if exec_action == "done":
                        if env.is_success():
                            traj_succ = True
                            succ_cnt += 1
                        break

                except Exception as e:
                    print(traceback.format_exc())
                    if rollout_t is None:
                        if rollout_t0 is None:
                            rollout_t = 0
                        else:
                            rollout_t = time.time() - rollout_t0
                    break

                if rollout_t is not None:
                    rollout_time += rollout_t

                # Check success
                if env.is_success():
                    traj_succ = True
                    succ_cnt += 1
                    break

            last_img = env.read_pixels(camera_name=FLAGS.camera_name)
            total_time = time.time() - total_time_t0

            if env.is_recording:
                env.record_off()

            traj_key_frames.append(last_img)
            if FLAGS.save_images:
                last_img_path = os.path.join(traj_dir, f"{len(question_list)}.png")
                goal_img_path = os.path.join(traj_dir, "goal.png")
                Image.fromarray(last_img).save(last_img_path)
                Image.fromarray(goal_img).save(goal_img_path)

            # Log data
            print("Success:", traj_succ)
            print(f"Failures this episode: {sum(step_fail_list[: len(question_list)])}")
            print(f"Total failures collected: {failure_cnt}")

            # Episode trace
            if FLAGS.trace_jsonl:
                fails_this_ep = [bool(x) for x in step_fail_list[: len(question_list)]]
                tags_this_ep = [
                    d["fail_tag"] if d else None for d in fail_diagnosis_list[: len(question_list)]
                ]
                ep = {
                    "traj_id": int(traj_id),
                    "board_id": int(board_id),
                    "env_seed": int(env_seed),
                    "reset_seed": int(reset_seed),
                    "agent_type": str(FLAGS.agent_type),
                    "agent_seed": int(FLAGS.agent_seed),
                    "success": bool(traj_succ),
                    "steps": int(len(question_list)),
                    "num_failures": sum(fails_this_ep),
                    "failure_tags": [t for t in tags_this_ep if t is not None],
                }
                _jsonl_append(ep_trace_path, ep)

            idx_in_env += 1
            traj_id += 1
            traj_cnt += 1

        env.close()
        env_seed += 1
        board_id += 1
        board_cnt += 1

    # Summary
    print("=" * 40)
    print("FAILURE COLLECTION COMPLETE")
    print("=" * 40)
    print(f"Total trajectories: {traj_cnt}")
    print(f"Total failures collected: {failure_cnt}")
    print(f"Success rate: {succ_cnt}/{traj_cnt} = {succ_cnt / max(1, traj_cnt):.2%}")

    # Save RoMemo memory (failure memory bank)
    if romemo_store is not None and FLAGS.romemo_save_memory_path is not None:
        try:
            romemo_store.save_pt(str(FLAGS.romemo_save_memory_path))
            print(f"[romemo] saved failure memory to {FLAGS.romemo_save_memory_path}")
            print(f"[romemo] memory size: {len(romemo_store.memory)} experiences")
        except Exception as e:
            print(f"[romemo] failed to save memory: {e}")

    os.remove(xml_filename)


if __name__ == "__main__":
    app.run(main)
