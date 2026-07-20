# Frozen-Weight Compute/RNG-Matched Control

Status: implemented and passed a two-step DRD2 QSAR integration test on the
server. This was a diagnostic test only; no confirmatory experiment was
started.

## Runtime Behavior

Set `REINVENT_CONTROLLER_MODE=frozen_weight` to use the normal controller
wrapper, validation cadence, validation batch sampling, SVM command, controller
history, and training trace path while keeping all objective weights fixed.

Every frozen validation history record requires:

- `policy_updated = false`
- `action_applied = false`
- all component actions equal to zero
- `old_weights == new_weights`
- `rng_state_hash_before_validation == rng_state_hash_after_validation`

The wrapper snapshots and restores Python, NumPy, Torch CPU, and initialized
Torch CUDA RNG streams around the full validation sampling and oracle call. It
logs their hashes for every validation event.

`scripts/run_frozen_weight_control.sh` is the reusable launcher. Its five
arguments are the paired real arm's config, output prefix, REINVENT seed,
validation cadence, and validation batch size. It refuses to overwrite an
existing frozen history.

## Required Integration Test

The server ran two identical two-step DRD2 QSAR jobs with REINVENT seed `1701`,
fixed weights `0.34,0.33,0.33`, identical wrapper/scoring paths, and batch size
`64`:

- ordinary frozen wrapper, with validation cadence beyond the two-step run;
- frozen wrapper with cadence `1`, validation batch `8`, and the normal DRD2
  validation SVM called at both checkpoints.

The strict reproducibility test sets `REINVENT_STRICT_DETERMINISM=1`, which
enables deterministic CUDA algorithms and disables cuDNN benchmarking. This is
necessary for a bitwise parameter comparison: without it, CUDA embedding
backward produced an unrelated maximum `9.3e-10` numerical difference despite
identical sampled training batches and RNG hashes.

All required checks passed:

| Check | Result |
| --- | --- |
| Training SMILES and states, step 1-2 | Exact match |
| Training total scores and component scores | Exact match |
| Training RNG hashes after scoring and after updates | Exact match |
| Agent update order and post-update parameter hash | Exact match |
| Frozen validation events | 2 |
| Validation RNG hashes before/after | Unchanged for both events |
| Controller actions and weight drift | All zero / fixed |
| Final checkpoint `network` tensors | Exact match |

The raw checkpoint file hashes differ only because REINVENT stores timestamped
metadata. The canonical hash of the actual `network` state is identical in both
arms: `06293a493f96a25bdf69cb8b89352786c613662c18853233434e14cb261a3a86`.

The machine-readable result is `frozen_weight_rng_match_summary.json`.
