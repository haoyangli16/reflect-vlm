#!/usr/bin/env python3
"""
Run Principle Learning Comparison Experiments

This script runs controlled experiments comparing:
1. Baseline (no principles)
2. With principles (rule-based reflector)
3. With principles (VLM-based reflector) - if API key available

It handles:
- Parallel execution (optional)
- Result aggregation
- Statistical comparison
- Plot generation

Usage:
    python scripts/run_principle_comparison.py \
        --memory_path=data/memory/mixed_memory.json \
        --n_envs=50 \
        --n_trajs=200 \
        --output_dir=logs/principle_comparison

    # With VLM reflector
    python scripts/run_principle_comparison.py \
        --memory_path=data/memory/mixed_memory.json \
        --vlm_provider=openai \
        --vlm_model=gpt-4o
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Try imports
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    agent_type: str
    model_path: str
    n_envs: int
    n_trajs: int
    max_steps: int
    memory_path: Optional[str]
    retrieval_mode: str
    use_principles: bool
    reflector_provider: Optional[str] = None
    reflector_model: Optional[str] = None
    principle_store_path: Optional[str] = None
    output_dir: str = ""

    def to_command(self) -> List[str]:
        """Convert to command-line arguments."""
        cmd = [
            "python",
            "run-rom.py",
            f"--agent_type={self.agent_type}",
            f"--model_path={self.model_path}",
            f"--n_envs={self.n_envs}",
            f"--n_trajs={self.n_trajs}",
            f"--max_steps={self.max_steps}",
            f"--romemo_retrieval_mode={self.retrieval_mode}",
            f"--romemo_use_principles={str(self.use_principles).lower()}",
            f"--save_dir={self.output_dir}",
        ]

        if self.memory_path:
            cmd.append(f"--romemo_init_memory_path={self.memory_path}")

        if self.reflector_provider:
            cmd.append(f"--romemo_reflector_provider={self.reflector_provider}")

        if self.reflector_model:
            cmd.append(f"--romemo_reflector_model={self.reflector_model}")

        if self.principle_store_path:
            cmd.append(f"--romemo_principle_store_path={self.principle_store_path}")

        return cmd


def run_experiment(config: ExperimentConfig, log_dir: Path) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    print(f"\n{'=' * 60}")
    print(f"Running: {config.name}")
    print(f"{'=' * 60}")

    log_file = log_dir / f"{config.name}.log"
    cmd = config.to_command()

    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()

    try:
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent.parent),
                timeout=3600 * 4,  # 4 hour timeout
            )

        elapsed = time.time() - start_time

        # Parse results from output directory
        results = parse_experiment_results(Path(config.output_dir))
        results["name"] = config.name
        results["elapsed_seconds"] = elapsed
        results["return_code"] = result.returncode

        if result.returncode == 0:
            print(f"✅ {config.name} completed in {elapsed:.1f}s")
        else:
            print(f"❌ {config.name} failed with code {result.returncode}")

        return results

    except subprocess.TimeoutExpired:
        print(f"⏱️ {config.name} timed out")
        return {"name": config.name, "error": "timeout"}
    except Exception as e:
        print(f"❌ {config.name} failed: {e}")
        return {"name": config.name, "error": str(e)}


def parse_experiment_results(output_dir: Path) -> Dict[str, Any]:
    """Parse results from experiment output directory."""
    results = {
        "success_rate": 0.0,
        "avg_steps": 0.0,
        "n_episodes": 0,
    }

    # Try to find results
    for results_file in output_dir.glob("**/results*.json"):
        with open(results_file, "r") as f:
            data = json.load(f)
            results.update(data)
            break

    # Try CSV
    for csv_file in output_dir.glob("**/results*.csv"):
        if HAS_PANDAS:
            df = pd.read_csv(csv_file)
            results["success_rate"] = df["success"].mean() if "success" in df else 0.0
            results["avg_steps"] = df["steps"].mean() if "steps" in df else 0.0
            results["n_episodes"] = len(df)
        break

    return results


def check_vlm_availability() -> Dict[str, bool]:
    """Check which VLM providers are available."""
    available = {}
    available["openai"] = bool(os.environ.get("OPENAI_API_KEY"))
    available["gemini"] = bool(os.environ.get("GOOGLE_API_KEY"))
    available["qwen"] = bool(os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY"))
    return available


def main():
    parser = argparse.ArgumentParser(description="Run Principle Learning Comparison")
    parser.add_argument("--memory_path", type=str, required=True, help="Path to memory file")
    parser.add_argument("--n_envs", type=int, default=50, help="Number of environments")
    parser.add_argument("--n_trajs", type=int, default=200, help="Number of trajectories")
    parser.add_argument("--max_steps", type=int, default=30, help="Max steps per episode")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--retrieval_mode", type=str, default="symbolic", help="Retrieval mode")
    parser.add_argument("--vlm_provider", type=str, default=None, help="VLM provider for reflector")
    parser.add_argument("--vlm_model", type=str, default=None, help="VLM model for reflector")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/share/project/lhy/ReflectVLM-llava-v1.5-13b-base",
        help="Base model path",
    )
    parser.add_argument(
        "--post_model",
        type=str,
        default="/share/project/lhy/ReflectVLM-llava-v1.5-13b-post-trained",
        help="Post-trained model path",
    )
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel")
    parser.add_argument("--skip_vlm", action="store_true", help="Skip VLM-based experiments")

    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"logs/principle_comparison_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    principle_dir = output_dir / "principles"
    principle_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Check VLM availability
    vlm_available = check_vlm_availability()
    print(f"VLM availability: {vlm_available}")

    # Define experiments
    experiments = []

    # Common settings
    common = {
        "n_envs": args.n_envs,
        "n_trajs": args.n_trajs,
        "max_steps": args.max_steps,
        "memory_path": args.memory_path,
        "retrieval_mode": args.retrieval_mode,
    }

    # BC Policy experiments
    for policy_name, model_path, agent_suffix in [
        ("bc", args.base_model, "bc_romemo"),
        ("reflect", args.post_model, "bc_romemo"),
    ]:
        # Baseline (no principles)
        experiments.append(
            ExperimentConfig(
                name=f"{policy_name}_baseline",
                agent_type=agent_suffix,
                model_path=model_path,
                use_principles=False,
                output_dir=str(output_dir / f"{policy_name}_baseline"),
                **common,
            )
        )

        # Rule-based reflector
        experiments.append(
            ExperimentConfig(
                name=f"{policy_name}_principles_rulebased",
                agent_type=f"{agent_suffix}_wb",
                model_path=model_path,
                use_principles=True,
                reflector_provider=None,
                principle_store_path=str(principle_dir / f"{policy_name}_rulebased.json"),
                output_dir=str(output_dir / f"{policy_name}_principles_rulebased"),
                **common,
            )
        )

        # VLM-based reflector (if available and not skipped)
        if not args.skip_vlm:
            provider = args.vlm_provider
            model = args.vlm_model

            if not provider:
                # Auto-detect
                for p in ["openai", "gemini", "qwen"]:
                    if vlm_available.get(p):
                        provider = p
                        model = {
                            "openai": "gpt-4o",
                            "gemini": "gemini-3-pro-preview",
                            "qwen": "qwen3-vl-235b-a22b-instruct",
                        }.get(p)
                        break

            if provider and vlm_available.get(provider):
                experiments.append(
                    ExperimentConfig(
                        name=f"{policy_name}_principles_{provider}",
                        agent_type=f"{agent_suffix}_wb",
                        model_path=model_path,
                        use_principles=True,
                        reflector_provider=provider,
                        reflector_model=model,
                        principle_store_path=str(principle_dir / f"{policy_name}_{provider}.json"),
                        output_dir=str(output_dir / f"{policy_name}_principles_{provider}"),
                        **common,
                    )
                )

    print(f"\nPlanned experiments: {len(experiments)}")
    for exp in experiments:
        print(f"  - {exp.name}")

    # Run experiments
    all_results = []

    if args.parallel and len(experiments) > 1:
        print("\nRunning experiments in parallel...")
        with ProcessPoolExecutor(max_workers=min(4, len(experiments))) as executor:
            futures = {executor.submit(run_experiment, exp, output_dir): exp for exp in experiments}
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
    else:
        print("\nRunning experiments sequentially...")
        for exp in experiments:
            result = run_experiment(exp, output_dir)
            all_results.append(result)

    # Save all results
    results_file = output_dir / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    for r in all_results:
        name = r.get("name", "unknown")
        sr = r.get("success_rate", 0.0)
        elapsed = r.get("elapsed_seconds", 0)
        error = r.get("error")

        if error:
            print(f"  ❌ {name}: ERROR - {error}")
        else:
            print(f"  ✅ {name}: SR={sr:.3f}, time={elapsed:.0f}s")

    # Generate comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    for policy in ["bc", "reflect"]:
        print(f"\n--- {policy.upper()} Policy ---")

        baseline = next((r for r in all_results if r.get("name") == f"{policy}_baseline"), None)
        rulebased = next(
            (r for r in all_results if r.get("name") == f"{policy}_principles_rulebased"), None
        )

        if baseline:
            base_sr = baseline.get("success_rate", 0.0)
            print(f"Baseline: {base_sr:.3f}")

            if rulebased:
                rb_sr = rulebased.get("success_rate", 0.0)
                improvement = (rb_sr - base_sr) / max(0.001, base_sr) * 100
                print(f"Rule-based: {rb_sr:.3f} ({improvement:+.1f}%)")

            # Check for VLM results
            for r in all_results:
                name = r.get("name", "")
                if name.startswith(f"{policy}_principles_") and "rulebased" not in name:
                    vlm_sr = r.get("success_rate", 0.0)
                    provider = name.split("_")[-1]
                    improvement = (vlm_sr - base_sr) / max(0.001, base_sr) * 100
                    print(f"VLM ({provider}): {vlm_sr:.3f} ({improvement:+.1f}%)")

    print("\n" + "=" * 70)
    print(f"All results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
