#!/usr/bin/env python3
"""
Analyze Principle Learning Experiments (Pillar 5)

This script analyzes the results from pillar5_principle_learning.sh:
1. Compare success rates across conditions
2. Analyze principle quality and usage
3. Measure failure reduction by fail_tag
4. Generate comparison plots

Usage:
    python scripts/analyze_principle_experiments.py --logs_dir=logs/pillar5_xxx
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime

import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, using basic CSV output")


def load_results(results_dir: Path) -> Optional[Dict[str, Any]]:
    """Load results from a single experiment directory."""
    results_file = results_dir / "results.json"
    if not results_file.exists():
        # Try CSV format
        results_csv = results_dir / "results.csv"
        if results_csv.exists() and HAS_PANDAS:
            df = pd.read_csv(results_csv)
            return {
                "success_rate": df["success"].mean() if "success" in df else 0.0,
                "avg_steps": df["steps"].mean() if "steps" in df else 0.0,
                "n_episodes": len(df),
            }
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def load_principles(principle_path: Path) -> Optional[Dict[str, Any]]:
    """Load principle store from file (.json or .pt)."""
    if not principle_path.exists():
        return None

    try:
        if principle_path.suffix == ".pt":
            import torch

            data = torch.load(principle_path, map_location="cpu")
            return data
        else:
            with open(principle_path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {principle_path}: {e}")
        return None


def load_step_traces(results_dir: Path) -> List[Dict[str, Any]]:
    """Load step-level traces for detailed analysis."""
    traces_file = results_dir / "step_traces.jsonl"
    if not traces_file.exists():
        return []

    traces = []
    with open(traces_file, "r") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces


def analyze_principles(principle_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze principle quality and statistics."""
    if not principle_data or "principles" not in principle_data:
        return {"count": 0}

    principles = principle_data["principles"]

    stats = {
        "count": len(principles),
        "avg_confidence": 0.0,
        "by_action_type": defaultdict(int),
        "by_fail_tag": defaultdict(int),
        "high_confidence": 0,
        "established": 0,
    }

    if not principles:
        return stats

    confidences = []
    for p in principles:
        conf = p.get("confidence", p.get("importance_score", 2.0) / 10.0)
        confidences.append(conf)

        if conf > 0.7:
            stats["high_confidence"] += 1
        if conf > 0.5:
            stats["established"] += 1

        for at in p.get("action_types", []):
            stats["by_action_type"][at] += 1

        for ft in p.get("addresses_fail_tags", []):
            stats["by_fail_tag"][ft] += 1

    stats["avg_confidence"] = np.mean(confidences) if confidences else 0.0
    stats["by_action_type"] = dict(stats["by_action_type"])
    stats["by_fail_tag"] = dict(stats["by_fail_tag"])

    return stats


def analyze_failure_reduction(
    baseline_traces: List[Dict],
    experiment_traces: List[Dict],
) -> Dict[str, float]:
    """Compare failure rates by fail_tag between baseline and experiment."""

    def count_failures(traces):
        fail_counts = defaultdict(int)
        total = 0
        for t in traces:
            if t.get("fail") or t.get("fail_tag"):
                tag = t.get("fail_tag", "unknown")
                fail_counts[tag] += 1
            total += 1
        return fail_counts, total

    base_fails, base_total = count_failures(baseline_traces)
    exp_fails, exp_total = count_failures(experiment_traces)

    # Calculate reduction for each fail_tag
    all_tags = set(base_fails.keys()) | set(exp_fails.keys())
    reduction = {}

    for tag in all_tags:
        base_rate = base_fails.get(tag, 0) / max(1, base_total)
        exp_rate = exp_fails.get(tag, 0) / max(1, exp_total)

        if base_rate > 0:
            reduction[tag] = (base_rate - exp_rate) / base_rate * 100
        else:
            reduction[tag] = 0.0

    return reduction


def aggregate_results(logs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Aggregate results from all experiment directories."""
    results = {}

    for subdir in logs_dir.iterdir():
        if not subdir.is_dir():
            continue

        name = subdir.name

        # Load main results
        exp_results = load_results(subdir)
        if exp_results is None:
            print(f"Warning: No results found in {subdir}")
            continue

        results[name] = {
            "success_rate": exp_results.get("success_rate", 0.0),
            "avg_steps": exp_results.get("avg_steps", 0.0),
            "n_episodes": exp_results.get("n_episodes", 0),
        }

        # Check for principle file
        for principle_file in logs_dir.parent.glob(f"data/principles/*{name}*.json"):
            principle_data = load_principles(principle_file)
            if principle_data:
                results[name]["principles"] = analyze_principles(principle_data)
                break

        # Load step traces for failure analysis
        traces = load_step_traces(subdir)
        results[name]["n_steps"] = len(traces)
        results[name]["traces_loaded"] = len(traces) > 0

    return results


def compare_conditions(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compare baseline vs principle-based conditions."""

    comparison = {
        "bc": {"baseline": None, "rulebased": None, "vlm": {}},
        "reflect": {"baseline": None, "rulebased": None, "vlm": {}},
    }

    for name, data in results.items():
        if name.startswith("bc_"):
            policy = "bc"
        elif name.startswith("reflect_"):
            policy = "reflect"
        else:
            continue

        if "baseline" in name:
            comparison[policy]["baseline"] = data
        elif "rulebased" in name:
            comparison[policy]["rulebased"] = data
        elif "principles_" in name:
            # VLM-based (extract provider)
            parts = name.split("_")
            provider = parts[-1] if len(parts) > 2 else "unknown"
            comparison[policy]["vlm"][provider] = data

    return comparison


def generate_summary_csv(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Generate CSV summary of results."""

    rows = []
    headers = [
        "Experiment",
        "Success Rate",
        "Avg Steps",
        "N Episodes",
        "N Principles",
        "Avg Confidence",
        "High Conf Principles",
    ]

    for name, data in sorted(results.items()):
        row = [
            name,
            f"{data.get('success_rate', 0.0):.3f}",
            f"{data.get('avg_steps', 0.0):.1f}",
            str(data.get("n_episodes", 0)),
        ]

        principles = data.get("principles", {})
        row.extend(
            [
                str(principles.get("count", 0)),
                f"{principles.get('avg_confidence', 0.0):.2f}",
                str(principles.get("high_confidence", 0)),
            ]
        )

        rows.append(row)

    with open(output_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")

    print(f"Summary saved to: {output_path}")


def generate_comparison_plot(
    comparison: Dict[str, Any],
    output_path: Path,
):
    """Generate comparison bar plot."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not available)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, policy in enumerate(["bc", "reflect"]):
        ax = axes[idx]
        data = comparison[policy]

        conditions = []
        success_rates = []

        if data["baseline"]:
            conditions.append("Baseline")
            success_rates.append(data["baseline"].get("success_rate", 0.0))

        if data["rulebased"]:
            conditions.append("Rule-based")
            success_rates.append(data["rulebased"].get("success_rate", 0.0))

        for provider, pdata in data["vlm"].items():
            conditions.append(f"VLM ({provider})")
            success_rates.append(pdata.get("success_rate", 0.0))

        if conditions:
            bars = ax.bar(
                conditions,
                success_rates,
                color=["gray", "blue", "green", "orange"][: len(conditions)],
            )
            ax.set_ylabel("Success Rate")
            ax.set_title(f"{policy.upper()} Policy")
            ax.set_ylim(0, 1.0)

            # Add value labels
            for bar, rate in zip(bars, success_rates):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{rate:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    plt.suptitle("Principle Learning: Success Rate Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")


def generate_principle_analysis_plot(
    results: Dict[str, Dict[str, Any]],
    output_path: Path,
):
    """Generate principle analysis plots."""
    if not HAS_MATPLOTLIB:
        return

    # Collect principle data
    experiments = []
    for name, data in results.items():
        if "principles" in data and data["principles"].get("count", 0) > 0:
            experiments.append((name, data["principles"]))

    if not experiments:
        print("No principle data found for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Principles by action type
    ax1 = axes[0]
    all_action_types = set()
    for _, pdata in experiments:
        all_action_types.update(pdata.get("by_action_type", {}).keys())

    if all_action_types:
        x = np.arange(len(all_action_types))
        width = 0.8 / len(experiments)

        for i, (name, pdata) in enumerate(experiments):
            counts = [pdata.get("by_action_type", {}).get(at, 0) for at in all_action_types]
            ax1.bar(x + i * width, counts, width, label=name[:20])

        ax1.set_xticks(x + width * len(experiments) / 2)
        ax1.set_xticklabels(list(all_action_types), rotation=45, ha="right")
        ax1.set_ylabel("Count")
        ax1.set_title("Principles by Action Type")
        ax1.legend(fontsize=8)

    # Plot 2: Principles by fail tag
    ax2 = axes[1]
    all_fail_tags = set()
    for _, pdata in experiments:
        all_fail_tags.update(pdata.get("by_fail_tag", {}).keys())

    if all_fail_tags:
        x = np.arange(len(all_fail_tags))
        width = 0.8 / len(experiments)

        for i, (name, pdata) in enumerate(experiments):
            counts = [pdata.get("by_fail_tag", {}).get(ft, 0) for ft in all_fail_tags]
            ax2.bar(x + i * width, counts, width, label=name[:20])

        ax2.set_xticks(x + width * len(experiments) / 2)
        ax2.set_xticklabels(list(all_fail_tags), rotation=45, ha="right")
        ax2.set_ylabel("Count")
        ax2.set_title("Principles by Fail Tag")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    output_analysis = output_path.parent / "principle_analysis.png"
    plt.savefig(output_analysis, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Principle analysis plot saved to: {output_analysis}")


def print_summary(results: Dict[str, Dict[str, Any]], comparison: Dict[str, Any]):
    """Print human-readable summary."""

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    for policy in ["bc", "reflect"]:
        print(f"\n--- {policy.upper()} Policy ---")
        data = comparison[policy]

        baseline_sr = data["baseline"]["success_rate"] if data["baseline"] else 0.0
        print(f"Baseline Success Rate: {baseline_sr:.3f}")

        if data["rulebased"]:
            rb_sr = data["rulebased"]["success_rate"]
            improvement = (rb_sr - baseline_sr) / max(0.001, baseline_sr) * 100
            print(f"Rule-based Success Rate: {rb_sr:.3f} ({improvement:+.1f}%)")

            if "principles" in data["rulebased"]:
                p = data["rulebased"]["principles"]
                print(f"  - Principles learned: {p.get('count', 0)}")
                print(f"  - Avg confidence: {p.get('avg_confidence', 0.0):.2f}")

        for provider, pdata in data["vlm"].items():
            vlm_sr = pdata["success_rate"]
            improvement = (vlm_sr - baseline_sr) / max(0.001, baseline_sr) * 100
            print(f"VLM ({provider}) Success Rate: {vlm_sr:.3f} ({improvement:+.1f}%)")

            if "principles" in pdata:
                p = pdata["principles"]
                print(f"  - Principles learned: {p.get('count', 0)}")
                print(f"  - Avg confidence: {p.get('avg_confidence', 0.0):.2f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze Principle Learning Experiments")
    parser.add_argument(
        "--logs_dir", type=str, required=True, help="Directory containing experiment logs"
    )
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--output_plot", type=str, default=None, help="Output plot path")

    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        sys.exit(1)

    print(f"Analyzing experiments in: {logs_dir}")

    # Aggregate results
    results = aggregate_results(logs_dir)

    if not results:
        print("No results found!")
        sys.exit(1)

    print(f"Found {len(results)} experiment results")

    # Compare conditions
    comparison = compare_conditions(results)

    # Generate outputs
    output_csv = Path(args.output_csv) if args.output_csv else logs_dir / "results_summary.csv"
    generate_summary_csv(results, output_csv)

    output_plot = (
        Path(args.output_plot) if args.output_plot else logs_dir / "results_comparison.png"
    )
    generate_comparison_plot(comparison, output_plot)
    generate_principle_analysis_plot(results, output_plot)

    # Print summary
    print_summary(results, comparison)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
