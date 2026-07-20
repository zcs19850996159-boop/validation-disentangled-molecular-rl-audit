# Frozen Temporal-Preserving Placebo Plan

Status: implemented, frozen, and checked against the existing ten DRD2 QSAR
controller histories. No REINVENT training or confirmatory run was started.

## Frozen Primary Placebo

The server implementation is in `/root/autodl-tmp/four_project/scripts/`:

- `temporal_placebo_common.py`: the shared, deterministic eligibility and
  schedule rule.
- `validate_temporal_placebo.py`: runtime replay command.
- `audit_temporal_placebo.py`: offline preflight and frozen-manifest generator.

Circular time-shift is the primary placebo. It replays an existing real score
trajectory to the controller while severing any relation to the current
validation molecules. The replay state is atomically advanced and records the
history checksum, selected shift, schedule seed, and replay index.

For a source trajectory of length `T`, only shifts in the frozen middle-half
candidate set are allowed:

```
ceil(T / 4) <= shift <= floor(3T / 4)
```

For the present `T=15` histories, valid shifts are `4..11`; near-zero shifts
such as `1` and `14` are rejected even when explicitly supplied. The frozen
schedule seed is `51001`. Formal circular replicas are indices `0`, `1`, and
`2`; each selects a distinct member of the deterministic candidate permutation.
Their outcomes must be averaged within seed before any cross-seed analysis.

Example runtime command template:

```bash
export REINVENT_VALIDATION_COMMAND="/root/miniconda3/envs/reinvent4/bin/python \
  scripts/validate_temporal_placebo.py \
  --input {input} --output {output} \
  --history controller_history_REAL_PREFIX.jsonl \
  --component-names 'DRD2_activity,QED,Molecular weight' \
  --mode circular_shift \
  --replay-state placebo_state_CIRCULAR_PREFIX.json \
  --frozen-manifest temporal_placebo_frozen_manifest.json \
  --target-seed TARGET_SEED \
  --schedule-seed 51001 --shift auto --shift-replicate-index 0"
```

Run three distinct control arms with replica indices `0`, `1`, and `2`; use an
independent replay-state file for each arm. The code rejects reuse of a state
file with mismatched history, schedule, or replica metadata.

## Frozen Cross-Seed Robustness Placebo

Cross-seed phase-matched replay uses donor record `i` at target validation index
`i`; it never wraps at donor exhaustion. The donor mapping is pre-specified as
the sorted supplied seed order shifted cyclically: `i -> i + 1 (mod n)`. It is
one-to-one and forbids self-donation or similarity/outcome-based donor choice.

For the current ten-seed audit, this is:

```
17 -> 23 -> 101 -> 202 -> 303 -> 404 -> 505 -> 606 -> 707 -> 808 -> 17
```

Use the mapped donor history with:

```bash
--mode cross_seed_phase_matched \
--history controller_history_MAPPED_DONOR.jsonl \
--replay-state placebo_state_CROSS_PREFIX.json \
--frozen-manifest temporal_placebo_frozen_manifest.json \
--target-seed TARGET_SEED
```

## Frozen Artifacts

- `temporal_placebo_frozen_manifest.json` pins the schedule seed, eligible
  range, exact seed-to-shift mapping, history checksums, and donor mapping.
- `temporal_placebo_frozen_audit.md` reports each formal circular replica and
  their within-seed mean, plus the target-donor matching diagnostics.
- `temporal_placebo_frozen_audit.json` retains all per-seed and per-component
  values, including signed target-minus-donor trends and early/middle/late
  validation phases.

The legacy independent `shuffle_history` control is retained only to reproduce
historical results and is not an admissible sole placebo for new experiments.
