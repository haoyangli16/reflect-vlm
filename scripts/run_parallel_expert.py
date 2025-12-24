import argparse
import subprocess
import os
import sys
import math
import time
import shutil

# Add parent directory to path to find roboworld and romemo if needed
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

    # Save
    print(f"Saving merged bank with {len(merged_bank.experiences)} total items...")
    merged_bank.save_pt(output_path)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Run expert generation in parallel")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g. '0,1,2,3'). If None, uses all visible.",
    )
    parser.add_argument(
        "--total_trajs", type=int, required=True, help="Total trajectories to generate"
    )
    parser.add_argument("--agent_type", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path. If not provided, uses POST_MODEL_PATH env (default: local post-trained).",
    )
    parser.add_argument("--output_pt", type=str, required=True, help="Final merged memory path")
    parser.add_argument("--level", type=str, default="all")
    parser.add_argument("--seed_start", type=int, default=1000000, help="Base seed")
    parser.add_argument(
        "--output_dir_base",
        type=str,
        default="datasets/parallel_run",
        help="Base dir for temp output",
    )

    # Pass-through args that might be needed
    parser.add_argument("--load_4bit", type=str, default="True")
    parser.add_argument("--logging_online", type=str, default="False")

    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = os.environ.get(
            "POST_MODEL_PATH",
            "/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-post-trained",
        )
    print(f"Model path: {args.model_path}")

    # Determine GPUs
    if args.gpus:
        gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
    else:
        # Default to CUDA_VISIBLE_DEVICES or just one
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        gpu_ids = [x.strip() for x in cvd.split(",") if x.strip()]

    print(f"Using GPUs: {gpu_ids}")

    # Prepare directories
    if os.path.exists(args.output_dir_base):
        print(f"Warning: {args.output_dir_base} exists. Cleaning up...")
        shutil.rmtree(args.output_dir_base)
    os.makedirs(args.output_dir_base, exist_ok=True)

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
        # Each job gets 'trajs_per_job' trajectories.
        # We shift start_traj_id and start_board_id
        # Assuming roughly 1 traj per board or we can just shift board_id by a large number?
        # run-rom.py logic: board_id increments when env needs reset.
        # Safest is to shift both significantly or just rely on 'n_trajs' per job and distinct seeds.

        start_traj = i * trajs_per_job
        # We assume 1 traj per board roughly, or just let them pick new boards.
        # The script uses 'start_board_id'. Let's shift it by trajs_per_job too to be safe.
        start_board = i * trajs_per_job

        # Also shift seed slightly if needed, but run-rom uses seed for env generation.
        # If we want diverse environments, we should increment seed range.
        # run-rom sets `env_seed = FLAGS.seed` then increments it.
        # So job i should start with seed = base_seed + i * (trajs_per_job * 2) (safety margin)
        job_seed = args.seed_start + (i * trajs_per_job)

        cmd = [
            "python",
            "run-rom.py",
            f"--agent_type={args.agent_type}",
            f"--n_trajs={trajs_per_job}",
            f"--romemo_save_memory_path={part_pt}",
            f"--level={args.level}",
            f"--record=False",
            f"--oracle_prob=1.0",
            f"--imagine_future_steps=5",
            f"--save_images=True",
            f"--logging.online={args.logging_online}",
            "--load_4bit" if args.load_4bit.lower() in ("true", "1", "yes") else "--noload_4bit",
            f"--model_path={args.model_path}",
            f"--save_dir={job_dir}",
            f"--start_traj_id={start_traj}",
            f"--start_board_id={start_board}",
            f"--seed={job_seed}",
            f"--romemo_retrieval_mode=symbolic",
            f"--reset_seed_start={job_seed}",  # Also shift reset seeds
        ]

        env_vars = os.environ.copy()
        env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id

        print(f"Launching Job {i} on GPU {gpu_id}: {trajs_per_job} trajs, seed={job_seed}")
        # print(" ".join(cmd))

        log_file = os.path.join(job_dir, "run.log")
        os.makedirs(job_dir, exist_ok=True)
        f_log = open(log_file, "w")

        p = subprocess.Popen(cmd, env=env_vars, stdout=f_log, stderr=subprocess.STDOUT)
        processes.append((p, f_log))

    # Wait for completion
    print("Waiting for jobs to finish... (logs redirected to job folders)")
    failed = False
    for i, (p, f_log) in enumerate(processes):
        code = p.wait()
        f_log.close()
        if code != 0:
            print(f"Job {i} failed with exit code {code}. Check logs.")
            failed = True
        else:
            print(f"Job {i} finished successfully.")

    if failed:
        print("Some jobs failed. Merging what we have...")

    # Merge
    valid_parts = [p for p in part_pt_files if os.path.exists(p)]
    if valid_parts:
        merge_memory_banks(valid_parts, args.output_pt)
    else:
        print("No output files found to merge.")


if __name__ == "__main__":
    main()
