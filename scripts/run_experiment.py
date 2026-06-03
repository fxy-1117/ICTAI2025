from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argumentation.config import DEFAULT_THRESHOLDS, ExperimentConfig
from argumentation.experiments import run_parameter_analysis


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run neuro-symbolic argumentation experiments.")
    parser.add_argument("--dataset", choices=("stsb", "sick"), required=True)
    parser.add_argument("--analysis", choices=("parameter",), default="parameter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-label", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--dataset-threshold", type=float, default=0.8)
    parser.add_argument("--amr-batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--thresholds", type=parse_float_list, default=DEFAULT_THRESHOLDS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        dataset=args.dataset,
        seed=args.seed,
        dataset_threshold=args.dataset_threshold,
        max_sentence_length=args.max_length,
        min_sentence_length=args.min_length,
        sample_per_label=args.sample_per_label,
        amr_batch_size=args.amr_batch_size,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )

    run_parameter_analysis(config, args.thresholds)


if __name__ == "__main__":
    main()
