# Argumentation Neuro-Symbolic Experiments

This repository contains the Python implementation of the neuro-symbolic
argumentation experiments for:

- STSB parameter analysis
- SICK parameter analysis

The pipeline converts sentence pairs into AMR-based logical representations and
uses approximate propositional reasoning to predict entailment, contradiction,
or neutrality.

## Repository Layout

```text
argumentation/          Core data loading, runtime, caching, and experiment code
scripts/run_experiment.py
                        CLI for STSB/SICK parameter analysis
cache/                  Local runtime cache, ignored by git
results/                Local experiment outputs, ignored by git
```

## Setup

Use Python 3.9 or newer.  Install the package dependencies with:

```powershell
pip install -r requirements.txt
```

The AMR parser requires the project-specific AMR dependencies used by the
pipeline:

- `transition_amr_parser`
- `amr_logic_converter`

If these are installed from local checkouts, install them before running the
scripts.

## Run Parameter Analysis

STSB, paper-style sample size:

```powershell
python scripts/run_experiment.py --dataset stsb --sample-per-label 500 --seed 42 --output-dir results/stsb_parameter --cache-dir cache/stsb_parameter
```

SICK:

```powershell
python scripts/run_experiment.py --dataset sick --sample-per-label 500 --seed 42 --output-dir results/sick_parameter --cache-dir cache/sick_parameter
```

Dataset loading can still be length-filtered with `--min-length` and
`--max-length`.  For example, this runs SICK parameter analysis on examples where
both sentences have more than 5 and at most 20 tokens:

```powershell
python scripts/run_experiment.py --dataset sick --min-length 5 --max-length 20 --sample-per-label 500 --seed 42
```

By default the thresholds are:

```text
0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00
```

## Caching

Two caches are used:

- `cache/*logic*.pkl`: sentence to transformed AMR logic
- `cache/neural_scores_*.pkl`: phrase-pair neural similarity scores used by `prove`

These caches make repeated runs across thresholds much faster.  They can be
deleted safely if you want a fresh run.

## Reproducibility Note

The scripts use a fixed seed by default (`--seed 42`) so experiments are
repeatable.  Reported scores can vary slightly with the sampled examples and
available AMR/parser model versions.

For the default seed-42 STSB parameter analysis, `tau=0.70` gives:

```text
accuracy: 0.711
ent precision/recall/f1: 0.704062 / 0.728000 / 0.715831
noent precision/recall/f1: 0.718427 / 0.694000 / 0.706002
```

## Paper Reference

```bibtex
@inproceedings{DBLP:conf/ictai/FengH25,
  author       = {Xuyao Feng and
                  Anthony Hunter},
  title        = {Formalizing Simple Natural Language Arguments Using Abstract Meaning
                  Representation and Approximate Propositional Reasoning},
  booktitle    = {37th {IEEE} International Conference on Tools with Artificial Intelligence,
                  {ICTAI} 2025, Athens, Greece, November 3-5, 2025},
  pages        = {238--245},
  publisher    = {{IEEE}},
  year         = {2025},
  url          = {https://doi.org/10.1109/ICTAI66417.2025.00038},
  doi          = {10.1109/ICTAI66417.2025.00038},
  timestamp    = {Tue, 24 Mar 2026 08:39:28 +0100},
  biburl       = {https://dblp.org/rec/conf/ictai/FengH25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
