# DRD2/QED/MW frozen static-scalarization development protocol

## Scope and lock

This is a development-only selection exercise.  It is not a test of the
dynamic controller and its runs must not be included in the confirmatory
analysis.  The protocol is locked before launching any development run.

All arms use the `frozen_weight` controller wrapper.  They therefore perform
the same validation sampling, SVM scoring, validation cadence, RNG
capture/restore, and controller-history logging as a controller arm.  The
only intended difference is the fixed weight vector supplied both to the
REINVENT scoring configuration and the frozen wrapper.

## Development seeds

The development seeds are `1103`, `2203`, and `3303`.  They are disjoint from
the historical seeds `17, 23, 101, 202, 303, 404, 505, 606, 707, 808` and are
reserved for this static-grid exercise only.

## Fixed execution settings

| Setting | Value |
| --- | --- |
| Training reward | DRD2 RF, QED, molecular-weight reward exactly as in `11_drd2_qsar_dynamic_fixed.toml` |
| Validation model | `qsar_models/DRD2/validation_svm.pkl` |
| Wrapper mode | `frozen_weight` |
| Initial/fixed weights | The configuration-specific vector below |
| Validation cadence | Every 20 REINVENT steps |
| Validation batch size | 64 molecules |
| Training budget | The unchanged 300-step source configuration |
| RNG control | `REINVENT_STRICT_DETERMINISM=1`, validation capture/restore enabled |
| Outputs | Per-run training trace, validation history, final checkpoint, and post-hoc SVM analysis |

## Predeclared static grid

| Label | Activity | QED | MW |
| --- | ---: | ---: | ---: |
| equal | 0.34 | 0.33 | 0.33 |
| activity_heavy_1 | 0.50 | 0.25 | 0.25 |
| activity_heavy_2 | 0.60 | 0.20 | 0.20 |
| qed_heavy | 0.25 | 0.50 | 0.25 |
| mw_heavy | 0.25 | 0.25 | 0.50 |
| activity_qed | 0.45 | 0.40 | 0.15 |
| activity_mw | 0.45 | 0.15 | 0.40 |

This is 21 runs in total.  No weights will be added, removed, or retuned after
their development outcomes are available.

## Selection rule

For each configuration, aggregate per-run post-hoc summaries over the three
development seeds.  The initial ranking metric is the mean
`last50_mean_validation_activity` produced by the fixed validation SVM.  This
is a development surrogate only; it is not an untouched final-oracle claim.

A configuration is eligible only if both of these mean quality summaries are
at least the corresponding equal-weight mean minus `0.015`:

- `final_step_mean_qed`;
- `final_step_mean_mw_score`.

Among eligible configurations, any that lie within `0.005` validation-activity
units of the largest mean are tied.  Resolve a tie by larger mean final-step
QED, then larger mean final-step molecular-weight score, then the fixed label
order in the table above.  If no configuration passes the quality gate, select
`equal` and record that the tuning exercise did not identify an admissible
alternative.

The selected vector is the sole tuned-static comparator for the later,
non-overlapping confirmatory experiment.  The confirmatory primary analysis
will not be changed in response to this development result.
