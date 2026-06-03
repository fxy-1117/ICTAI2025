from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
DEFAULT_LENGTH_BINS = ((0, 10), (10, 15), (15, 20), (20, 100))


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    seed: int = 42
    dataset_threshold: float = 0.8
    max_sentence_length: int = 20
    min_sentence_length: int = 0
    sample_per_label: int = 500
    amr_batch_size: int = 32
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("results")
    sentence_model_name: str = "BAAI/bge-small-en-v1.5"
    amr_model_name: str = "AMR3-structbart-L"

    @property
    def dataset_key(self) -> str:
        return self.dataset.lower()
