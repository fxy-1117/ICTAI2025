from __future__ import annotations

import contextlib
import csv
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from . import core
from .config import ExperimentConfig
from .data import PairExample, balanced_sample, examples_for_length_bin, load_examples
from .runtime import ArgumentationRuntime


def _sample_label_order(dataset_key: str) -> list[str]:
    if dataset_key == "stsb":
        return ["ent", "noent"]
    if dataset_key == "sick":
        return ["ent", "neu", "con"]
    return []


def _cache_name(config: ExperimentConfig, examples: Sequence[PairExample], suffix: str) -> Path:
    return config.cache_dir / f"{config.dataset_key}_{len(examples)}_{config.seed}_{suffix}.pkl"


def build_logic_cache(
    runtime: ArgumentationRuntime,
    examples: Sequence[PairExample],
    cache_path: Path,
    log_path: Path,
) -> Dict[str, object]:
    """Create or load sentence -> transformed AMR logic cache."""

    unique_sentences = list(dict.fromkeys([e.sentence1 for e in examples] + [e.sentence2 for e in examples]))
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            logic_cache = pickle.load(handle)
    else:
        logic_cache = {}

    missing = [sentence for sentence in unique_sentences if sentence not in logic_cache]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        for start in range(0, len(missing), runtime.config.amr_batch_size):
            batch = missing[start : start + runtime.config.amr_batch_size]
            print(f"AMR {len(logic_cache) + len(batch)}/{len(unique_sentences)}", flush=True)
            try:
                with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                    transformed = [core.transform_logic(x) for x in core.generate_logic(batch)[-2]]
                logic_cache.update(dict(zip(batch, transformed)))
            except Exception as exc:
                print(f"AMR batch failed, falling back to individual sentences: {exc!r}", flush=True)
                for sentence in batch:
                    try:
                        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                            logic_cache[sentence] = core.transform_logic(core.generate_logic([sentence])[-2][0])
                    except Exception as sentence_exc:
                        logic_cache[sentence] = None
                        print(f"AMR failed for sentence: {sentence!r} -> {sentence_exc!r}", file=log_file)

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as handle:
                pickle.dump(logic_cache, handle)

    return logic_cache


def predict_examples(
    examples: Sequence[PairExample],
    threshold: float,
    logic_cache: Dict[str, object],
    log_path: Path,
    dataset_key: str,
) -> pd.DataFrame:
    rows = []
    with log_path.open("a", encoding="utf-8") as log_file:
        for item_id, example in enumerate(examples):
            try:
                left_logic = logic_cache.get(example.sentence1)
                right_logic = logic_cache.get(example.sentence2)
                if left_logic is None or right_logic is None:
                    raise ValueError("missing logic cache entry")

                with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                    raw_label = core.prove([left_logic, right_logic], threshold)[0]
                if dataset_key == "stsb":
                    prediction = "ent" if raw_label == "ent" else "noent"
                else:
                    prediction = raw_label
            except Exception as exc:
                raw_label = ""
                prediction = "error"
                rows.append(
                    {
                        "item_id": item_id,
                        "sentence1": example.sentence1,
                        "sentence2": example.sentence2,
                        "gold": example.gold,
                        "score": example.score,
                        "threshold": threshold,
                        "raw_label": raw_label,
                        "prediction": prediction,
                        "error": repr(exc),
                    }
                )
                continue

            rows.append(
                {
                    "item_id": item_id,
                    "sentence1": example.sentence1,
                    "sentence2": example.sentence2,
                    "gold": example.gold,
                    "score": example.score,
                    "threshold": threshold,
                    "raw_label": raw_label,
                    "prediction": prediction,
                    "error": "",
                }
            )
            if (item_id + 1) % 50 == 0 or item_id + 1 == len(examples):
                print(f"tau={threshold:.2f} predicted {item_id + 1}/{len(examples)}", flush=True)
    return pd.DataFrame(rows)


def metric_table(predictions: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    scored = predictions[predictions["prediction"] != "error"]
    precision, recall, f1, support = precision_recall_fscore_support(
        scored["gold"],
        scored["prediction"],
        labels=labels,
        zero_division=0,
    )
    rows = [
        {
            "metric": "accuracy",
            "class_label": "overall",
            "score": accuracy_score(scored["gold"], scored["prediction"]),
            "support": len(scored),
        }
    ]
    for label, p_value, r_value, f_value, s_value in zip(labels, precision, recall, f1, support):
        rows.extend(
            [
                {"metric": "precision", "class_label": label, "score": p_value, "support": s_value},
                {"metric": "recall", "class_label": label, "score": r_value, "support": s_value},
                {"metric": "f1", "class_label": label, "score": f_value, "support": s_value},
            ]
        )
    return pd.DataFrame(rows)


def run_parameter_analysis(config: ExperimentConfig, thresholds: Sequence[float]) -> pd.DataFrame:
    runtime = ArgumentationRuntime(config)
    examples = balanced_sample(
        load_examples(config),
        config.sample_per_label,
        config.seed,
        _sample_label_order(config.dataset_key),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([e.__dict__ for e in examples]).to_csv(config.output_dir / f"{config.dataset_key}_sample_seed{config.seed}.csv", index=False)

    log_path = config.output_dir / f"{config.dataset_key}_pipeline_seed{config.seed}.log"
    logic_cache = build_logic_cache(runtime, examples, _cache_name(config, examples, "logic"), log_path)
    labels = ["ent", "noent"] if config.dataset_key == "stsb" else ["con", "ent", "neu"]

    all_metrics = []
    for threshold in thresholds:
        predictions = predict_examples(examples, threshold, logic_cache, log_path, config.dataset_key)
        predictions.to_csv(config.output_dir / f"{config.dataset_key}_predictions_tau{threshold:.2f}_seed{config.seed}.csv", index=False)
        metrics = metric_table(predictions, labels)
        metrics.insert(0, "threshold", threshold)
        metrics.insert(0, "dataset", config.dataset_key)
        all_metrics.append(metrics)

        report = classification_report(
            predictions[predictions["prediction"] != "error"]["gold"],
            predictions[predictions["prediction"] != "error"]["prediction"],
            labels=labels,
            zero_division=0,
        )
        (config.output_dir / f"{config.dataset_key}_report_tau{threshold:.2f}_seed{config.seed}.txt").write_text(report, encoding="utf-8")
        runtime.save()

    result = pd.concat(all_metrics, ignore_index=True)
    result.to_csv(config.output_dir / f"{config.dataset_key}_parameter_metrics_seed{config.seed}.csv", index=False)
    runtime.save()
    return result


def run_length_analysis(
    config: ExperimentConfig,
    threshold: float,
    bins: Sequence[tuple[int, int]],
) -> pd.DataFrame:
    runtime = ArgumentationRuntime(config)
    labels = ["ent", "noent"] if config.dataset_key == "stsb" else ["con", "ent", "neu"]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for lower, upper in bins:
        pool = examples_for_length_bin(config, lower, upper)
        examples = balanced_sample(pool, config.sample_per_label, config.seed, _sample_label_order(config.dataset_key))
        label = f"{lower}-{upper}"
        pd.DataFrame([e.__dict__ for e in examples]).to_csv(config.output_dir / f"{config.dataset_key}_length_{label}_sample_seed{config.seed}.csv", index=False)
        log_path = config.output_dir / f"{config.dataset_key}_length_{label}_seed{config.seed}.log"
        logic_cache = build_logic_cache(runtime, examples, _cache_name(config, examples, f"logic_length_{label}"), log_path)
        predictions = predict_examples(examples, threshold, logic_cache, log_path, config.dataset_key)
        predictions.to_csv(config.output_dir / f"{config.dataset_key}_length_{label}_predictions_seed{config.seed}.csv", index=False)
        metrics = metric_table(predictions, labels)
        metrics.insert(0, "threshold", threshold)
        metrics.insert(0, "length_bin", label)
        metrics.insert(0, "dataset", config.dataset_key)
        all_metrics.append(metrics)
        runtime.save()

    result = pd.concat(all_metrics, ignore_index=True)
    result.to_csv(config.output_dir / f"{config.dataset_key}_length_metrics_seed{config.seed}.csv", index=False)
    runtime.save()
    return result
