# Paper Data Audit Report

## Frozen evidence available

- The sole confirmatory experiment contains 10 paired seeds and 70 verified
  arms: 10 real, 30 circular, 10 cyclic cross-seed, 10 frozen equal, and 10
  frozen tuned runs.
- Final evaluation uses steps 251-300 and contains 217,423 unique valid
  molecules across all arms. Seed-level pairing, not molecule rows, is the
  inference unit.
- The final result JSON, Markdown summary, arm CSV, and paired-difference CSV
  match the server SHA256 manifest.
- All three frozen circular shifts and the cyclic target-to-donor mapping are
  available with source-history hashes.
- The untouched BindingDB MPN, static-grid selection, frozen-weight RNG test,
  concurrent-worker test, and V5 end-to-end positive control all have
  machine-readable reports.

## Central confirmatory readouts

| Contrast | Endpoint | Mean | 95% bootstrap CI | Positive seeds | Sign p | >=0.015 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Real - circular | SVM activity | +0.002362 | [-0.003273, +0.009457] | 7/10 | 0.343750 | No |
| Real - circular | MPN, all valid | +0.007881 | [+0.002235, +0.013335] | 7/10 | 0.343750 | No |
| Real - circular | MPN, AD covered | -0.000316 | [-0.012115, +0.013406] | 4/10 | 0.753906 | No |
| Real - cross-seed | SVM activity | -0.004195 | [-0.014808, +0.006595] | 4/10 | 0.753906 | No |
| Real - equal static | SVM activity | -0.000606 | [-0.007235, +0.006216] | 4/10 | 0.753906 | No |
| Real - tuned static | SVM activity | +0.013096 | [+0.004692, +0.022348] | 7/10 | 0.343750 | No |
| Real - tuned static | MPN, AD covered | -0.004341 | [-0.023968, +0.020128] | 2/10 | 0.109375 | No |
| Real - tuned static | QED | -0.018959 | [-0.023295, -0.014848] | 0/10 | 0.001953 | No |

The main conclusion is absence of a practically meaningful utilization effect
under the frozen +0.015 criterion and absence of an AD-supported downstream
gain. The all-valid MPN contrast must not be presented alone.

## Temporal placebo qualification

On the same 10 historical trajectories used for offline placebo development,
the frozen three-shift circular mean is materially closer to the real dynamics
than the legacy shuffle:

| Metric | Legacy shuffle | Circular mean |
| --- | ---: | ---: |
| Mean absolute lag-1 ACF gap | 0.386 | 0.140 |
| Mean delta KS distance | 0.231 | 0.071 |
| Mean absolute decline-count gap | 1.567 | 0.544 |
| Component-action gap | 2.3 | 1.1 |
| Final-weight L1 gap | 0.177 | 0.076 |
| QED decline events, real/placebo | 2.5 / 4.6 | 2.5 / 3.27 |

For the exact shifts frozen for the confirmatory experiment, replicate indices
0, 1, and 2 have ACF gaps 0.131, 0.150, and 0.135 and delta-KS distances 0.067,
0.069, and 0.071. Their within-seed mean has an ACF gap of 0.139 and delta-KS
of 0.069. Thus the selected shifts do not hide a poor replicate behind an
all-shift average.

The cyclic cross-seed mapping is one-to-one with no self-donor. Target-donor
matching has overall checkpoint MAE 0.0295, mean absolute score-mean difference
0.0073, ACF gap 0.2578, delta-KS 0.2381, and mean absolute decline-count
difference 1.2667. Cross-seed replay preserves each donor trajectory exactly,
but donor-to-target mismatch remains a reported limitation and robustness
feature rather than a claim of exact matching.

## Qualification evidence

- Frozen-weight validation produced six validation events with unchanged RNG
  hashes, fixed weights, identical training traces, identical final agent state,
  and identical canonical network tensors.
- Concurrent-worker smoke tests passed for five worker slots against the same
  canonical agent and network hashes.
- The BindingDB MPN passed its frozen gate: test AUROC 0.8264, AP 0.9093, Brier
  0.2462. Its test set has only 25 molecules and ECE is 0.2453.
- Mean applicability-domain coverage for real confirmatory molecules is 0.0439.
  This low coverage is why the AD-covered MPN result is the decisive utility
  readout.
- V5 passed every end-to-end gate. All five paired QED gains were positive and
  the mean gain was +0.12579, while validity and uniqueness remained within the
  frozen noninferiority limits.

## Data placement decision

The main text should contain the compact temporal qualification, oracle/RNG/V5
qualification, and central paired confirmatory tables. Exact shifts, donor
hashes, all 32 contrasts, all per-seed differences, static-grid candidates,
and detailed qualification hashes belong in the supplement. Earlier pilots,
activity/SA, GSK3B, docking, and controller-input sensitivity data remain
exploratory and cannot support the primary claim.

## Remaining presentation work

- Render the forest plot from `table_confirmatory_all_contrasts.csv`.
- Render central seed-level paired differences from
  `figure_confirmatory_seed_differences.csv`.
- Render V5 paired QED gains from `figure_v5_paired_qed.csv`.
- Convert the compact CSV tables to LaTeX after the revised Results structure is
  fixed.
