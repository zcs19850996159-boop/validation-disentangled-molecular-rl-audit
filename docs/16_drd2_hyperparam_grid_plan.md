# DRD2/QED/MW Overnight Robustness Queue

This queue fixes the trained RF reward model and SVM validation model. It does
not retrain QSAR models and does not modify REINVENT4 core source files.

## Phase 1: Main Result Repeat Extension

Goal: extend the deterministic DRD2/QED/MW paired real-vs-random validation
result from 5 repeats to 10 repeats.

Existing repeats:

- seed17: `deterministic_db002_seed17`
- seed23, seed101, seed202, seed303: `deterministic_mg_seed*`

New queued repeats:

- seed404
- seed505
- seed606
- seed707
- seed808

Fixed controller settings:

- mode: deterministic validation-response controller
- epsilon/deadband: 0.02
- action step alpha: 0.15
- validation batch size: 64
- validation cadence: every 20 REINVENT steps

## Phase 2: One-Factor Hyperparameter Robustness Grid

Each cell is run as paired real-validation vs shuffled-validation control with
two repeats: seed17 and seed23.

| Label | epsilon | alpha | validation batch | validation cadence |
|---|---:|---:|---:|---:|
| eps001 | 0.01 | 0.15 | 64 | 20 |
| eps004 | 0.04 | 0.15 | 64 | 20 |
| alpha005 | 0.02 | 0.05 | 64 | 20 |
| alpha025 | 0.02 | 0.25 | 64 | 20 |
| vbatch32 | 0.02 | 0.15 | 32 | 20 |
| vbatch128 | 0.02 | 0.15 | 128 | 20 |
| cadence10 | 0.02 | 0.15 | 64 | 10 |
| cadence30 | 0.02 | 0.15 | 64 | 30 |

Primary metric:

- `last50_mean_validation_activity` real-minus-random paired delta.

Output files:

- `qsar_models_DRD2_drd2_hyperparam_grid_posthoc.json`
- `drd2_hyperparam_grid_pair_summary.json`
- `independent_source_inventory_DRD2.txt`

Remote launcher:

- `scripts/remote_drd2_hyperparam_grid.py`

