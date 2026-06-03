from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

from amr_logic_converter import AmrLogicConverter
from sentence_transformers import SentenceTransformer, util
from transition_amr_parser.parse import AMRParser

from . import core
from .config import ExperimentConfig


class NeuralScoreCache:
    """Persistent cache for phrase-pair similarity scores used inside ``prove``."""

    def __init__(self, model: SentenceTransformer, path: Path):
        self.model = model
        self.path = path
        self.embeddings = {}
        self.scores: Dict[Tuple[str, str], float] = self._load()

    def _load(self) -> Dict[Tuple[str, str], float]:
        if self.path.exists():
            with self.path.open("rb") as handle:
                return pickle.load(handle)
        return {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as handle:
            pickle.dump(self.scores, handle)

    def score(self, left: str, right: str) -> float:
        key = (str(left), str(right))
        if key not in self.scores:
            if key[0] not in self.embeddings:
                self.embeddings[key[0]] = self.model.encode(key[0], convert_to_tensor=True, show_progress_bar=False)
            if key[1] not in self.embeddings:
                self.embeddings[key[1]] = self.model.encode(key[1], convert_to_tensor=True, show_progress_bar=False)
            emb_left = self.embeddings[key[0]]
            emb_right = self.embeddings[key[1]]
            self.scores[key] = float(util.pytorch_cos_sim(emb_left, emb_right)[0][0])
        return self.scores[key]


class ArgumentationRuntime:
    """Loads AMR/BGE models and installs them into the core pipeline."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.parser = AMRParser.from_pretrained(config.amr_model_name)
        self.converter = AmrLogicConverter(
            existentially_quantify_instances=False,
            invert_relations=True,
        )
        self.sentence_model = SentenceTransformer(config.sentence_model_name)
        self.neural_cache = NeuralScoreCache(
            self.sentence_model,
            config.cache_dir / f"neural_scores_{config.sentence_model_name.replace('/', '_')}.pkl",
        )
        self.install()

    def install(self) -> None:
        core.parser = self.parser
        core.converter = self.converter
        core.model = self.sentence_model
        core.util = util
        core.score = self.neural_cache.score

    def save(self) -> None:
        self.neural_cache.save()
