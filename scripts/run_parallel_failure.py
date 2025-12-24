#!/usr/bin/env python3
"""
Parallel failure data generation.
Similar to run_parallel_expert.py but uses run-rom-fail.py for failure collection.
"""

import argparse
import subprocess
import os
import sys
import math
import shutil

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def merge_memory_banks(part_paths, output_path):
    """
    Loads multiple MemoryBank .pt files and merges them into one.
    """
    print(f"Merging {len(part_paths)} memory banks into {output_path}...")
    try:
        from romemo.memory.schema import MemoryBank
    except ImportError:
        print("Error: Could not import romemo.memory.schema.MemoryBank")
        return

    if not part_paths:
        print("No parts to merge.")
        return

    # Load the first one as the base
    merged_bank = MemoryBank.load_pt(part_paths[0])
    print(f"  Loaded part 0: {len(merged_bank.experiences)} items")

    for i, path in enumerate(part_paths[1:], 1):
        try:
            bank = MemoryBank.load_pt(path)
            print(f"  Loaded part {i} ({path}): {len(bank.experiences)} items")
            merged_bank.experiences.extend(bank.experiences)
        except Exception as e:
            print(f"  Error loading part {path}: {e}")

    # Count failure types
    fail_tags = {}
    for exp in merged_bank.experiences:
        tag = getattr(exp, "fail_tag", None)
        if tag is None and exp.extra_metrics:
            tag = exp.extra_metrics.get("fail_tag", "UNKNOWN")
        if tag:
            fail_tags[tag] = fail_tags.get(tag, 0) + 1

    print("\nFailure tag distribution:")
    for tag, count in sorted(fail_tags.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")

    # Save
    print(f"\nSaving merged bank with {len(merged_bank.experiences)} total failure experiences...")
    merged_bank.save_pt(output_path)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Run failure data generation in parallel")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs (e.g. '0,1,2,3'). If None, uses all visible.",
    )
    parser.add_argument(
        "--total_trajs", type=int, required=True, help="Total trajectories to generate"
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="bc_romemo_wb",
        help="Agent type for failure collection (bc_romemo_wb, bc, llava)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path. If not provided: bc* uses BASE_MODEL_PATH, reflect* uses POST_MODEL_PATH.",
    )
    parser.add_argument("--output_pt", type=str, required=True, help="Final merged memory path")
    parser.add_argument("--level", type=str, default="all")
    parser.add_argument("--seed_start", type=int, default=0, help="Base seed")
    parser.add_argument(
        "--output_dir_base",
        type=str,
        default="datasets/parallel_failure",
        help="Base dir for temp output",
    )

    # Pass-through args
    parser.add_argument("--load_4bit", type=str, default="True")
    parser.add_argument("--logging_online", type=str, default="False")
    parser.add_argument(
        "--stop_on_failure",
        type=str,
        default="True",
        help="Stop episode after first failure (default: True)",
    )
    parser.add_argument(
        "--write_on_success",
        type=str,
        default="False",
        help="Also record success experiences (default: False for pure failure memory)",
    )

    args = parser.parse_args()

    # Determine GPUs
    if args.gpus:
        gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
    else:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        gpu_ids = [x.strip() for x in cvd.split(",") if x.strip()]

    print(f"Using GPUs: {gpu_ids}")
    print(f"Agent type: {args.agent_type}")
    print(f"Stop on failure: {args.stop_on_failure}")
    print(f"Write on success: {args.write_on_success}")

    base_model = os.environ.get(
        "BASE_MODEL_PATH",
        "/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-base",
    )
    post_model = os.environ.get(
        "POST_MODEL_PATH",
        "/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-post-trained",
    )
    if args.model_path is None:
        if str(args.agent_type).startswith("reflect"):
            args.model_path = post_model
        else:
            args.model_path = base_model
    print(f"Model path: {args.model_path}")

    # Prepare directories
    if os.path.exists(args.output_dir_base):
        print(f"Warning: {args.output_dir_base} exists. Cleaning up...")
        shutil.rmtree(args.output_dir_base)
    os.makedirs(args.output_dir_base, exist_ok=True)

    # Also ensure output directory for final .pt exists
    output_dir = os.path.dirname(args.output_pt)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Calculate splits
    trajs_per_job = math.ceil(args.total_trajs / args.n_jobs)

    processes = []
    part_pt_files = []

    for i in range(args.n_jobs):
        job_dir = os.path.join(args.output_dir_base, f"job_{i}")
        part_pt = os.path.join(args.output_dir_base, f"memory_part_{i}.pt")
        part_pt_files.append(part_pt)

        gpu_id = gpu_ids[i % len(gpu_ids)]

        # Calculate start indices to ensure no overlap
        start_traj = i * trajs_per_job
        start_board = i * trajs_per_job
        job_seed = args.seed_start + (i * trajs_per_job)

        # Use run-rom-fail.py for failure collection
        cmd = [
            "python",
            "-u",
            "run-rom-fail.py",
            f"--agent_type={args.agent_type}",
            f"--n_trajs={trajs_per_job}",
            f"--romemo_save_memory_path={part_pt}",
            f"--level={args.level}",
            "--record=False",
            "--oracle_prob=0.0",  # No oracle help - let agent make mistakes
            "--save_images=False",
            f"--logging.online={args.logging_online}",
            "--load_4bit" if args.load_4bit.lower() in ("true", "1", "yes") else "--noload_4bit",
            f"--model_path={args.model_path}",
            f"--save_dir={job_dir}",
            f"--start_traj_id={start_traj}",
            f"--start_board_id={start_board}",
            f"--seed={job_seed}",
            f"--reset_seed_start={job_seed}",
            f"--romemo_retrieval_mode=symbolic",
            f"--stop_on_failure={args.stop_on_failure}",
            f"--write_on_success={args.write_on_success}",
            "--trace_jsonl=True",
        ]

        env_vars = os.environ.copy()
        env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id

        print(f"Launching Job {i} on GPU {gpu_id}: {trajs_per_job} trajs, seed={job_seed}")

        log_file = os.path.join(job_dir, "run.log")
        os.makedirs(job_dir, exist_ok=True)
        f_log = open(log_file, "w")

        p = subprocess.Popen(cmd, env=env_vars, stdout=f_log, stderr=subprocess.STDOUT)
        processes.append((p, f_log, i))

    # Wait for completion
    print(f"\nWaiting for {args.n_jobs} jobs to finish...")
    print("(logs redirected to job folders)")

    failed_jobs = []
    for p, f_log, job_idx in processes:
        code = p.wait()
        f_log.close()
        if code != 0:
            print(
                f"Job {job_idx} failed with exit code {code}. Check {args.output_dir_base}/job_{job_idx}/run.log"
            )
            failed_jobs.append(job_idx)
        else:
            print(f"Job {job_idx} finished successfully.")

    if failed_jobs:
        print(f"\n{len(failed_jobs)} jobs failed. Merging what we have...")

    # Merge
    valid_parts = [p for p in part_pt_files if os.path.exists(p)]
    if valid_parts:
        merge_memory_banks(valid_parts, args.output_pt)
    else:
        print("No output files found to merge.")

    # Summary
    print("\n" + "=" * 50)
    print("FAILURE DATA GENERATION COMPLETE")
    print("=" * 50)
    print(f"Output: {args.output_pt}")
    print(f"Jobs completed: {args.n_jobs - len(failed_jobs)}/{args.n_jobs}")
    if failed_jobs:
        print(f"Failed jobs: {failed_jobs}")


if __name__ == "__main__":
    main()
