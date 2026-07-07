# DRD2/QED/MW n=10 Interim Result

Generated while the hyperparameter grid continued running on the server.

## Setup

- Training reward model: fixed DRD2 RandomForest QSAR reward model.
- Validation model: fixed DRD2 SVM QSAR validation model.
- Controller: deterministic validation-response controller.
- Comparison: real validation feedback vs shuffled-validation control.
- Primary metric: `last50_mean_validation_activity`.
- Repeats: 10 paired repeats.

## Main Result

For the primary last-50 validation activity metric:

- Mean real-minus-random delta: `-0.00159`
- Paired bootstrap 95% CI: `[-0.01023, 0.00625]`
- Positive paired repeats: `5/10`
- Repeat-level SD: `0.01411`

This no longer shows the weak positive pattern seen in the n=5 result. After
expanding to n=10, the direction is split evenly and the confidence interval
overlaps zero.

## Secondary Metrics

| Metric | Mean delta | Bootstrap 95% CI | Positive repeats | SD |
|---|---:|---:|---:|---:|
| all | -0.00210 | [-0.00770, 0.00363] | 5/10 | 0.00969 |
| final | 0.01463 | [-0.00282, 0.03070] | 7/10 | 0.02862 |
| last50 | -0.00159 | [-0.01023, 0.00625] | 5/10 | 0.01411 |
| top10 | -0.00333 | [-0.01079, 0.00382] | 4/10 | 0.01245 |

The final-step metric remains directionally positive but noisy, with a wide CI
that still overlaps zero. The more stable last-50 and all-row metrics are near
zero or slightly negative.

## Interpretation

The n=10 extension strengthens the negative conclusion: the tested
validation-driven deterministic controller does not show a reliable
validation-specific advantage over shuffled-validation control in the
DRD2/QED/MW setting.

This is stronger evidence than the earlier n=5 result because the apparent
small positive last-50 effect did not persist after adding five paired repeats.

## Independent-Source Inventory

The current remote inventory found ChEMBL validation files, Vina calibration
files, and existing run outputs, but no obvious ExCAPE independent-source file.
Therefore, a truly independent ExCAPE-vs-ChEMBL source analysis still requires
fetching or mounting an external ExCAPE/assay-source dataset before it can be
run.

Local files:

- `drd2_base_n10_interim_pair_summary.json`
- `qsar_models_DRD2_drd2_base_n10_interim_posthoc.json`
- `independent_source_inventory_DRD2.txt`

