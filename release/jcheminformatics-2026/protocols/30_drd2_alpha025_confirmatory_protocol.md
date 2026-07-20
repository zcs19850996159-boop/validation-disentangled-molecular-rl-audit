# DRD2 alpha=0.25 confirmatory protocol

## Authorization and lock

This is the sole confirmatory follow-up to the earlier two-seed alpha=0.25
pilot. It begins only after V5 end-to-end qualification passed all frozen gates.
No endpoint, seed, arm, placebo shift, static weight, or exclusion rule may be
changed after the first confirmatory run starts.

## New paired seeds and fixed training settings

The ten new seeds are `6101, 6102, 6103, 6104, 6105, 6106, 6107, 6108,
6109, 6110`. They were not used in controller development, static-grid
selection, positive controls, or earlier pilots.

All arms use the unchanged 300-step DRD2 RF/QED/MW training task, batch size
64, strict deterministic CUDA, cadence 20, validation batch 64, deadband 0.02,
and initial weights `0.34,0.33,0.33`. Dynamic and placebo controllers use the
locked action step `alpha=0.25` and never decrease weight on improvement.

## Arms

For every seed run:

1. real SVM validation controller;
2. three circular-shift replays, replicate indices 0, 1, 2;
3. cross-seed phase-matched replay with sorted seed `i -> i+1 (mod 10)`;
4. frozen equal static `0.34,0.33,0.33`;
5. frozen tuned static `0.25,0.50,0.25`, selected as `qed_heavy` solely on the
   disjoint static development seeds 1103, 2203, 3303.

Circular shifts use schedule seed `51001` and only the middle-half candidate
set `ceil(T/4) <= shift <= floor(3T/4)`. The three replicas are averaged within
seed before any across-seed contrast. After all ten real histories complete,
`audit_temporal_placebo.py` creates and hashes the exact shifts and cyclic donor
mapping before any placebo run begins. No similarity or outcome selects a shift
or donor.

Static arms execute the same wrapper, validation sampling, SVM calls, cadence,
batch size, history logging, and RNG capture/restore, but never act or
renormalize weights.

## Evaluation and inference

The controller-feedback endpoint is last-50-step validation-SVM activity,
rescored from generated SMILES. The untouched downstream endpoint is the
BindingDB external MPN probability from the passed, frozen checkpoint; report
both all-valid and applicability-domain-covered means, with coverage defined as
maximum ECFP4 similarity at least 0.30 to its external training set. QED, MW
target score, validity, uniqueness, and scaffold diversity are quality
endpoints.

The primary utilization contrast is real minus within-seed mean circular SVM.
The untouched-oracle real-minus-circular contrast is the key generalization
endpoint. Cross-seed replay is robustness; equal and tuned static are fixed
comparators. Report ten paired differences, mean, median, sample SD, positive
count, exact sign test, and a 100,000-resample paired bootstrap 95% interval
with seed 51001. The pre-existing practical threshold is `+0.015`; molecule
rows are never treated as independent replicates. No failed or unfavorable
seed may be excluded. Early termination before 300 steps invalidates the arm
and halts the queue rather than silently shortening histories.

## Storage rule

CSV, controller history, replay state, generated config, run manifest, and
checkpoint are retained. Each checkpoint is SHA-256 hashed. TensorBoard files
are written under the explicit temporary root
`/tmp/drd2_alpha025_confirmatory_tb` and removed only after that run completes
and its manifest is written; they are not analysis inputs. This predeclared
rule prevents the 70-run queue from exhausting the data disk.
