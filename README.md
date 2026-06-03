# Argumentation Neuro-Symbolic Experiments

This repository contains the refactored Python implementation of the argumentation
experiments from the original notebooks.  It keeps the AMR-to-logic and SAT-based
reasoning pipeline, and focuses on reproducing the paper experiments for:

- STSB parameter analysis
- SICK parameter analysis
- STSB length analysis
- SICK length analysis

The LLM prompt-ablation notebooks were removed from the experiment path so this
codebase is centered on paper reproduction.

## Repository Layout

```text
argumentation/          Core data loading, runtime, caching, and experiment code
scripts/run_experiment.py
                        CLI for STSB/SICK parameter and length analyses
scripts/compare_to_paper.py
                        Small STSB tau=0.70 comparison helper
cache/                  Local runtime cache, ignored by git
results/                Local experiment outputs, ignored by git
```

## Setup

Use the same environment that can run the original notebook.  In the local
workspace this was:

```powershell
C:\Users\fxy19\anaconda3\envs\eesnli\python.exe
```

Install the package dependencies if needed:

```powershell
pip install -r requirements.txt
```

The AMR parser may also need the same local `transition-amr-parser` setup used by
the notebook.
Two project-specific imports are expected from that setup:

- `transition_amr_parser`
- `amr_logic_converter`

If these are installed from local checkouts, install them before running the
scripts.

## Run Parameter Analysis

STSB, paper-style sample size:

```powershell
python scripts/run_experiment.py --dataset stsb --analysis parameter --sample-per-label 500 --seed 42 --output-dir results/stsb_parameter --cache-dir cache/stsb_parameter
```

SICK:

```powershell
python scripts/run_experiment.py --dataset sick --analysis parameter --sample-per-label 500 --seed 42 --output-dir results/sick_parameter --cache-dir cache/sick_parameter
```

By default the thresholds are:

```text
0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00
```

## Run Length Analysis

STSB:

```powershell
python scripts/run_experiment.py --dataset stsb --analysis length --threshold 0.70 --length-bins 0-10,10-15,15-100 --sample-per-label 150 --seed 42 --output-dir results/stsb_length --cache-dir cache/stsb_length
```

SICK:

```powershell
python scripts/run_experiment.py --dataset sick --analysis length --threshold 0.75 --length-bins 10-15,15-20,20-25,25-100 --sample-per-label 1 --seed 42 --output-dir results/sick_length --cache-dir cache/sick_length
```

The SICK length-analysis command uses one example per label to match the original
notebook-style smoke run.  The longer SICK bins are sparse, so a larger fixed
sample size fails for some bins unless the bins are changed.

## Caching

Two caches are used:

- `cache/*logic*.pkl`: sentence to transformed AMR logic
- `cache/neural_scores_*.pkl`: phrase-pair neural similarity scores used by `prove`

These caches make repeated runs across thresholds much faster.  They can be
deleted safely if you want a clean rerun.

## Reproducibility Note

The original notebooks use random sampling through NumPy without a fixed seed.
This refactor fixes the seed by default (`--seed 42`) so experiments are
repeatable.  The exact paper numbers may differ slightly unless the original
random sample is restored, but the code path matches the notebook core.

In the seed-42 rerun after refactoring, STSB at `tau=0.70` exactly matched the
fixed-seed reference run:

```text
accuracy: 0.711
ent precision/recall/f1: 0.704062 / 0.728000 / 0.715831
noent precision/recall/f1: 0.718427 / 0.694000 / 0.706002
```

## Compare With Paper STSB tau=0.70

```powershell
python scripts/compare_to_paper.py results/stsb_parameter/stsb_parameter_metrics_seed42.csv --threshold 0.70
```
