#!/usr/bin/env python3
"""
Plot SR + looping metrics from aggregate results.csv.
Outputs two pngs:
- sr_bar.png
- looping_bar.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="logs/eval_romemo_plugin/_aggregate/results.csv")
    p.add_argument("--out", type=str, default="logs/eval_romemo_plugin/_aggregate/plots")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(csv_path)
    rows.sort(key=lambda r: r["method"])
    methods = [r["method"] for r in rows]
    sr = [float(r["success_rate"]) for r in rows]
    looping = [float(r["looping_rate"]) for r in rows]
    rep_fail = [float(r["repeated_failure_rate"]) for r in rows]

    import matplotlib.pyplot as plt

    # SR bar
    plt.figure(figsize=(10, 4))
    plt.bar(methods, sr)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Success Rate")
    plt.title("ReflectVLM benchmark: Base vs Base+RoMemo(+WB)")
    plt.tight_layout()
    sr_path = out / "sr_bar.png"
    plt.savefig(sr_path, dpi=200)
    plt.close()

    # Looping / repeated failure bar (side-by-side)
    x = list(range(len(methods)))
    w = 0.4
    plt.figure(figsize=(10, 4))
    plt.bar([i - w / 2 for i in x], looping, width=w, label="looping_rate")
    plt.bar([i + w / 2 for i in x], rep_fail, width=w, label="repeated_failure_rate")
    plt.xticks(x, methods, rotation=30, ha="right")
    plt.ylabel("Rate")
    plt.title("Looping / repeated failures reduction")
    plt.legend()
    plt.tight_layout()
    lp_path = out / "looping_bar.png"
    plt.savefig(lp_path, dpi=200)
    plt.close()

    print(f"Wrote: {sr_path}")
    print(f"Wrote: {lp_path}")


if __name__ == "__main__":
    main()


