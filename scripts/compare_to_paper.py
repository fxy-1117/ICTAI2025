from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


PAPER_STSB_TAU_070 = {
    ("overall", "accuracy"): 0.69,
    ("ent", "precision"): 0.69,
    ("ent", "recall"): 0.72,
    ("ent", "f1"): 0.70,
    ("noent", "precision"): 0.68,
    ("noent", "recall"): 0.70,
    ("noent", "f1"): 0.69,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a metrics CSV with selected paper values.")
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics_csv)
    if "threshold" in metrics:
        metrics = metrics[metrics["threshold"].round(2) == round(args.threshold, 2)]

    rows = []
    for (class_label, metric), target in PAPER_STSB_TAU_070.items():
        match = metrics[(metrics["class_label"] == class_label) & (metrics["metric"] == metric)]
        actual = float(match["score"].iloc[0]) if not match.empty else None
        rows.append(
            {
                "class_label": class_label,
                "metric": metric,
                "paper_target": target,
                "actual": actual,
                "delta": None if actual is None else actual - target,
            }
        )
    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
