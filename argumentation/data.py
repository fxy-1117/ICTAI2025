from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from datasets import load_dataset
from nltk.tokenize import word_tokenize

from .config import ExperimentConfig


@dataclass(frozen=True)
class PairExample:
    sentence1: str
    sentence2: str
    gold: str
    score: float | None = None


def _within_length(example: PairExample, min_len: int, max_len: int) -> bool:
    len1 = len(word_tokenize(example.sentence1))
    len2 = len(word_tokenize(example.sentence2))
    return min_len < len1 <= max_len and min_len < len2 <= max_len


def load_stsb_examples(config: ExperimentConfig) -> List[PairExample]:
    examples: List[PairExample] = []
    for split in ("train", "test", "validation"):
        dataset = load_dataset("sentence-transformers/stsb", split=split)
        for row in dataset:
            label = "ent" if row["score"] >= config.dataset_threshold else "noent"
            example = PairExample(row["sentence1"], row["sentence2"], label, float(row["score"]))
            if _within_length(example, config.min_sentence_length, config.max_sentence_length):
                examples.append(example)
    return examples


def load_sick_examples(config: ExperimentConfig) -> List[PairExample]:
    examples: List[PairExample] = []
    for split in ("train", "test", "validation"):
        dataset = load_dataset("sick", split=split)
        for row in dataset:
            sentence_a = row["sentence_A"]
            sentence_b = row["sentence_B"]
            relation = row["entailment_AB"]
            label = int(row["label"])

            if label == 0:
                gold = "ent"
                pair = (sentence_a, sentence_b) if relation == " A_entails_B" else (sentence_b, sentence_a)
            elif label == 1:
                gold = "neu"
                pair = (sentence_a, sentence_b) if relation == " A_neutral_B" else (sentence_b, sentence_a)
            else:
                gold = "con"
                pair = (sentence_a, sentence_b) if relation == " A_contradicts_B" else (sentence_b, sentence_a)

            example = PairExample(pair[0], pair[1], gold, None)
            if _within_length(example, config.min_sentence_length, config.max_sentence_length):
                examples.append(example)
    return examples


def load_examples(config: ExperimentConfig) -> List[PairExample]:
    if config.dataset_key == "stsb":
        return load_stsb_examples(config)
    if config.dataset_key == "sick":
        return load_sick_examples(config)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def balanced_sample(
    examples: Sequence[PairExample],
    per_label: int,
    seed: int,
    label_order: Sequence[str] | None = None,
) -> List[PairExample]:
    # Use NumPy's seeded choice so sampled examples are reproducible.
    np.random.seed(seed)
    by_label: Dict[str, List[PairExample]] = {}
    for example in examples:
        by_label.setdefault(example.gold, []).append(example)

    sampled: List[PairExample] = []
    labels = list(label_order) if label_order is not None else list(by_label)
    for label in labels:
        if label not in by_label:
            continue
        pool = by_label[label]
        if per_label > len(pool):
            raise ValueError(f"Requested {per_label} {label} examples, but only {len(pool)} are available.")
        indices = np.random.choice(len(pool), per_label, replace=False)
        sampled.extend(pool[int(i)] for i in indices)
    return sampled
