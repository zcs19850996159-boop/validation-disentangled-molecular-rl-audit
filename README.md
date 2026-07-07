# Validation-Disentangled Reward Weighting

Reproducibility package for a validation-disentangled diagnostic framework for adaptive reward weighting in molecular generation.

The project audits whether an adaptive reward-weight controller in REINVENT4 responds to an independent validation signal, rather than to the same scores used to train the molecular generator. The repository contains the controller wrapper, local analysis scripts, configuration files, aggregate results, and manuscript figures used for the diagnostic study.

## Repository Contents

- `03_dynamic_weight_controller.py` - REINVENT4 wrapper/controller implementation.
- `configs/` - TOML configuration files for baseline, dynamic, random-validation, and activity/SA experiments.
- `scripts/` - local analysis, validation, docking-calibration, figure-generation, positive-control, and hindsight scripts.
- `results/` - lightweight aggregate result files used in the manuscript.
- `figures/` - editable SVG and PDF manuscript figures.
- `docs/` - experiment plans and diagnostic reports.
- `manuscript/` - current LaTeX draft.

Server-specific remote execution helpers, raw checkpoints, dependency caches, and raw large generation logs are intentionally excluded.

## Main Results Included

The included aggregate results support the diagnostic conclusion that the tested controllers did not show a reliable validation-specific advantage:

- DRD2/QED/MW ten-repeat paired real-vs-random result:
  - mean delta: `-0.0016`
  - 95% paired-bootstrap CI: `[-0.0102, +0.0063]`
  - positive repeats: `5/10`
- DRD2 activity/SA fast-gate result:
  - mean delta: `-0.0007`
  - 95% paired-bootstrap CI: `[-0.0038, +0.0022]`
- Hyperparameter scan:
  - included as exploratory only, with two paired repeats per setting.

## Environment

The experiments were run with REINVENT4 and RDKit. A typical analysis environment needs:

```bash
python >= 3.10
rdkit
numpy
scipy
scikit-learn
pandas
matplotlib
seaborn
tomlkit
```

Install the Python-side analysis dependencies with:

```bash
pip install -r requirements.txt
```

REINVENT4 itself should be installed following the official REINVENT4 instructions. This repository does not vendor REINVENT4 or modify REINVENT4 core source files.

## Reproducing Analyses

Examples:

```bash
python scripts/controller_effect_statistics.py
python scripts/positive_control_sensitivity_ladder.py
python scripts/qsar_signal_independence.py
python scripts/qsar_vina_triangulation.py
python scripts/make_manuscript_figures.py
```

The exact REINVENT4 training runs require trained QSAR models and the corresponding REINVENT4 prior/model files. Large generated molecule CSVs and checkpoints are not included in this lightweight GitHub package; the aggregate outputs needed to reproduce manuscript-level tables and figures are included under `results/`.

## Safety and Scope

This repository is a reproducibility artifact, not a production drug-discovery pipeline. Docking scripts are included for protocol transparency, but the manuscript concludes that the current DRD2 docking calibration was insufficient as a reliable controller oracle.

## Citation

Please cite the accompanying manuscript when available. A provisional `CITATION.cff` is included and should be updated with final author, DOI, and publication metadata before release.
