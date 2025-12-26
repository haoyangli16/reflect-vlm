#!/usr/bin/env python3
"""
Benchmark Runner for Neuro-Symbolic Memory System.

This script runs standardized ablation studies to validate the memory system.
It wraps `run-rom.py` with preset configurations for consistent comparison.

Ablations:
1. Baseline (No Memory)
2. Memory (With Kimi VLM)
3. Memory (With other providers)

Usage:
    python scripts/run_benchmark_memory.py --mode baseline --n_trajs 50
    python scripts/run_benchmark_memory.py --mode memory --provider kimi --n_trajs 50
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Configuration Presets
SEEDS = {
    "start": 1000001,  # Standard test set start
}

MODELS = {
    "bc": "/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-base",
    "reflect": "/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-post-trained",
}


def run_command(cmd, log_file):
    """Run shell command and stream output to log."""
    print(f"Running: {' '.join(cmd)}")
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=str(Path(__file__).parent.parent),  # Run from project root
        )

        # Stream output to console and file
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)

        process.wait()
        return process.returncode


def main():
    parser = argparse.ArgumentParser(description="Neuro-Symbolic Memory Benchmark")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["baseline", "memory"], help="Experiment mode"
    )
    parser.add_argument(
        "--policy", type=str, default="bc", choices=["bc", "reflect"], help="Base policy type"
    )
    parser.add_argument(
        "--provider", type=str, default="kimi", help="VLM provider for memory (kimi, openai, etc)"
    )
    parser.add_argument("--n_trajs", type=int, default=100, help="Number of episodes")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--name", type=str, default=None, help="Custom experiment name")

    args = parser.parse_args()

    # 1. Setup Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.name or f"bench_{args.mode}_{args.policy}"
    if args.mode == "memory":
        exp_name += f"_{args.provider}"

    save_dir = Path(f"logs/benchmark_neuro_symbolic/{exp_name}_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"==================================================")
    print(f"Benchmark: {exp_name}")
    print(f"Output: {save_dir}")
    print(f"==================================================")

    # 2. Build Command
    cmd = [
        "python",
        "run-rom.py",
        f"--seed={SEEDS['start']}",
        f"--n_trajs={args.n_trajs}",
        f"--save_dir={save_dir}",
        f"--model_path={MODELS[args.policy]}",
        f"--load_4bit=True",
        f"--max_steps=50",
        f"--record=False",  # Disable video to save space/time
        f"--trace_jsonl=True",  # Enable structured logging
    ]

    # Mode-specific config
    if args.mode == "baseline":
        # Pure BC/Reflect agent without memory wrapper
        cmd.append(f"--agent_type={args.policy}")

    elif args.mode == "memory":
        # Agent with RoMemo wrapper and Write-Back (wb) enabled
        # Note: We use _wb suffix to enable learning (writing to memory)
        cmd.append(f"--agent_type={args.policy}_romemo_wb")

        # Memory Settings
        cmd.extend(
            [
                "--romemo_use_principles=True",
                f"--romemo_reflector_provider={args.provider}",
                "--romemo_retrieval_mode=symbolic",  # Use new hybrid/symbolic retrieval
                "--romemo_symbolic_weight=0.5",
                "--romemo_principle_store_path=" + str(save_dir / "learned_principles.json"),
                "--romemo_save_memory_path=" + str(save_dir / "final_memory.pt"),
            ]
        )

    # GPU
    full_cmd = ["env", f"CUDA_VISIBLE_DEVICES={args.gpu}"] + cmd

    # 3. Run
    log_file = save_dir / "console.log"
    ret = run_command(full_cmd, log_file)

    if ret == 0:
        print(f"\n✅ Experiment completed successfully.")
        print(f"Results saved to: {save_dir}")
        print(f"Memory Trace: {save_dir}/traces/memory_trace.jsonl")
    else:
        print(f"\n❌ Experiment failed with code {ret}")


if __name__ == "__main__":
    main()
