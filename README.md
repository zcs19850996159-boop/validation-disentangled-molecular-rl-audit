# Validation-Disentangled Molecular RL Audit

Reproducibility materials for a diagnostic framework that audits
validation-driven adaptive reward weighting in molecular generation.

## Current Release

The submission-aligned package is frozen at
[`release/jcheminformatics-2026/`](release/jcheminformatics-2026/README.md).
It contains the code snapshots, locked protocols, audit outputs, source data,
final figures, and row-level molecular supplement supporting the revised study.

The confirmatory design includes:

- temporal-preserving circular-shift placebos and phase-matched cross-seed replay;
- validation-compute and random-number-generator-matched frozen controls;
- separate RF training reward, SVM controller feedback, and untouched MPN evaluation;
- equal and development-selected static scalarizations;
- a locked alpha=0.25 experiment with 10 paired seeds and 70 runs; and
- an end-to-end semi-synthetic positive control.

The tested controller did not demonstrate a practically meaningful advantage at
the prespecified margin. This repository supports a diagnostic-framework claim,
not a general claim that adaptive weighting improves or cannot improve molecular
generation.

## Repository Layout

- `release/jcheminformatics-2026/code/`: frozen controller and analysis code.
- `release/jcheminformatics-2026/protocols/`: locked experimental protocols.
- `release/jcheminformatics-2026/paper_data/`: machine-readable table data.
- `release/jcheminformatics-2026/figure_source_data/`: source data for Figures 2-5.
- `release/jcheminformatics-2026/figures/`: final PDF figures.
- `release/jcheminformatics-2026/audit_outputs/`: placebo and RNG qualification records.
- `release/jcheminformatics-2026/additional_file_3/`: 224,000 row generated-molecule supplement and metadata.

The manuscript source is intentionally excluded from this public repository.

## Integrity Check

On Linux, verify every frozen artifact with:

```bash
cd release/jcheminformatics-2026
sha256sum --check MANIFEST.sha256
gzip --test additional_file_3/Additional_file_3_generated_molecule_records.csv.gz
```

The compressed molecular supplement has SHA-256
`78098e09d0dfae705aa53f7ff8a6d3baa6dbdbb785e9d40ace704c276f54c388`.

## Historical Material

The repository state preceding submission cleanup is preserved on the
`legacy/pre-jcheminformatics-cleanup` branch. It contains exploratory analyses,
obsolete shuffle-history controls, an earlier manuscript draft, and unrelated
planning notes. Those materials are retained only for provenance and must not be
used as the current analysis.

## Environment

Analysis scripts require Python 3.10 or newer. Common dependencies are listed in
`requirements.txt`. Install them with `pip install -r requirements.txt` in an
isolated environment.

The frozen tables, figures, and reported contrasts can be checked from the
included machine-readable data. Re-running molecular RL training additionally
requires REINVENT4 model files, trained oracle weights, source configurations,
and the original software environment; these large or restricted assets are not
redistributed here. Server launch scripts are retained as provenance snapshots
and contain the paths used during the study, which must be adapted in another
environment.

## License And Citation

Code is released under the MIT License. Data provenance and source restrictions
are described in the protocols and accompanying manuscript. Citation metadata is
provided in `CITATION.cff` and should be updated with the article DOI after
publication.
