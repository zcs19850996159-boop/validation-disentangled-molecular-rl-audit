# Manuscript Data Presentation Plan

## Evidence hierarchy

The revised manuscript treats the ten-seed alpha=0.25 experiment as the sole
confirmatory efficacy/utilization analysis. Earlier hyperparameter sweeps and
the two-seed alpha=0.25 signal are explicitly exploratory. The analysis unit is
the paired seed; generated molecule rows are not independent replicates.

## Main-text tables

1. **Study design and frozen comparisons.** Three data streams, seed sets,
   controller settings, placebo construction, static weights, endpoints, and
   predeclared practical threshold.
2. **Temporal-placebo qualification.** Legacy shuffle, the three frozen
   circular replicate rules and their within-seed mean, block permutation, and
   cyclic cross-seed replay. Report ACF, delta-KS, decline-event, action, and
   final-weight gaps.
3. **Qualification gates.** Frozen-weight RNG matching, untouched BindingDB MPN,
   concurrent-worker determinism, and the V5 end-to-end positive control.
4. **Confirmatory paired results.** Real minus circular, cross-seed, frozen
   equal, and frozen tuned for SVM, untouched MPN, AD-covered MPN, and molecular
   quality endpoints.

## Main-text figures

1. Three-stream audit design and the role of each control.
2. Temporal-dynamics comparison of legacy shuffle and temporal-preserving
   placebos.
3. Forest plot of paired confirmatory effects with zero and +0.015 reference
   lines.
4. Seed-level paired-difference plot for the three central readouts.
5. V5 end-to-end qualification paired QED gains.

## Supplementary data

- Exact circular shifts and target-to-donor mapping for every seed.
- Per-seed temporal metrics and controller actions.
- All 32 comparator-endpoint confirmatory contrasts.
- Static-grid candidate results and frozen selection rule.
- RNG hashes, canonical network hashes, and concurrent-worker checks.
- MPN data audit, scaffold exclusions, calibration, correlations, and
  applicability-domain definition.
- Earlier controller-input sensitivity ladder, alpha=0.25 two-seed pilot,
  activity/SA, GSK3B, and docking analyses with exploratory labels.
- Failed V3/V4 positive-control development attempts and all exclusions or
  interrupted runs.

## Required limitations in the main text

- The untouched MPN test set contains 25 compounds and has ECE 0.245.
- Applicability-domain coverage among confirmatory generated molecules is only
  about 4-5%, so the all-valid MPN result cannot establish practical downstream
  utility.
- The tuned static arm was selected using three disjoint development seeds.
- Circular shifts preserve most adjacent dynamics but introduce a wrap
  boundary; cross-seed replay therefore remains a mechanism-distinct robustness
  placebo.
- The confirmatory analysis script was implemented after execution, while the
  endpoints, seed-level inference, bootstrap settings, and threshold were
  frozen in the pre-run protocol. This chronology must be stated plainly.

## Interpretation lock

The paper may conclude that the diagnostic pipeline is qualified and that the
locked controller did not demonstrate a practically meaningful validation-
signal utilization effect or applicability-domain-supported downstream gain.
It must not interpret small all-valid MPN differences as practical molecular
utility or relabel exploratory pilots as confirmatory evidence.
