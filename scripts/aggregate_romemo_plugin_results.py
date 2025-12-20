#!/usr/bin/env python3
"""
Aggregate RoMemo plug-in experiment outputs produced by run.py into:
- results.csv (method-level aggregates)
- episode_traces.jsonl (merged)
- step_traces.jsonl (merged)

Assumes directory structure created by scripts/eval_romemo_plugin.sh:
  logs/eval_romemo_plugin/<method>/traces/{episode_traces.jsonl,step_traces.jsonl}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            yield json.loads(ln)
        except Exception:
            continue


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="logs/eval_romemo_plugin", help="root directory")
    p.add_argument(
        "--out", type=str, default="logs/eval_romemo_plugin/_aggregate", help="output directory"
    )
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    merged_ep = out / "episode_traces.jsonl"
    merged_st = out / "step_traces.jsonl"
    if merged_ep.exists():
        merged_ep.unlink()
    if merged_st.exists():
        merged_st.unlink()

    rows = []

    # Structure:
    #   root/<method>/seed_<agent_seed>/traces/...
    for method_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        method = method_dir.name
        for seed_dir in sorted(
            [p for p in method_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]
        ):
            agent_seed = seed_dir.name.replace("seed_", "")
            trace_dir = seed_dir / "traces"
            ep_path = trace_dir / "episode_traces.jsonl"
            st_path = trace_dir / "step_traces.jsonl"

            eps = list(_iter_jsonl(ep_path))
            if not eps:
                continue

            # merge jsonl
            with merged_ep.open("a", encoding="utf-8") as f:
                for r in eps:
                    f.write(json.dumps({"method": method, "agent_seed": agent_seed, **r}) + "\n")
            with merged_st.open("a", encoding="utf-8") as f:
                for r in _iter_jsonl(st_path):
                    f.write(json.dumps({"method": method, "agent_seed": agent_seed, **r}) + "\n")

            sr = _mean([1.0 if bool(r.get("success", False)) else 0.0 for r in eps])
            mean_steps = _mean([float(r.get("steps", 0)) for r in eps])
            looping = _mean([float(r.get("looping_rate", 0.0)) for r in eps])
            rep_fail = _mean([float(r.get("repeated_failure_rate", 0.0)) for r in eps])
            retries = _mean([float(r.get("num_retries", 0)) for r in eps])

            rows.append(
                {
                    "method": method,
                    "agent_seed": agent_seed,
                    "num_episodes": len(eps),
                    "success_rate": sr,
                    "mean_steps": mean_steps,
                    "looping_rate": looping,
                    "repeated_failure_rate": rep_fail,
                    "mean_num_retries": retries,
                }
            )

    # results.csv
    out_csv = out / "results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "agent_seed",
                "num_episodes",
                "success_rate",
                "mean_steps",
                "looping_rate",
                "repeated_failure_rate",
                "mean_num_retries",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote: {out_csv}")
    print(f"Merged episode traces: {merged_ep}")
    print(f"Merged step traces: {merged_st}")


if __name__ == "__main__":
    main()
