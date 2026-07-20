# Frozen Temporal Placebo Audit

The formal circular placebo uses only replicate indices 0, 1, and 2. The schedule seed is `51001` and the eligible shift rule is `ceil(T/4) <= s <= floor(3T/4)`; with T=15, that is 4 through 11.

## Frozen Circular Schedule

| Target seed | Circular replicate 0 | Circular replicate 1 | Circular replicate 2 |
| --- | ---: | ---: | ---: |
| 17 | 8 | 10 | 11 |
| 23 | 9 | 8 | 10 |
| 101 | 9 | 7 | 4 |
| 202 | 4 | 7 | 6 |
| 303 | 4 | 5 | 9 |
| 404 | 9 | 5 | 10 |
| 505 | 5 | 7 | 10 |
| 606 | 6 | 5 | 4 |
| 707 | 9 | 8 | 5 |
| 808 | 8 | 5 | 7 |

The three values in every row are distinct. They are selected before outcome analysis from the middle-half candidate set, not from the full set of non-zero offsets.

## Replay-Source Dynamics

All gaps in this table compare a replay with the trajectory it replays. For cross-seed replay, this is the donor, so zero confirms replay fidelity rather than target-donor phase matching.

| Placebo | Seeds | Lag-1 ACF gap | Delta KS | Decline-count gap | Action-checkpoint gap | Final-weight L1 gap | Score-corr gap | Delta-corr gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy_shuffle_history | 10 | 0.3858 | 0.2310 | 1.5667 | 1.4000 | 0.1769 | 0.1629 | 0.3322 |
| circular_shift_replicate_0 | 10 | 0.1541 | 0.0714 | 0.5000 | 0.2000 | 0.0773 | 0.0000 | 0.1950 |
| circular_shift_replicate_1 | 10 | 0.1352 | 0.0690 | 0.6000 | 0.6000 | 0.0797 | 0.0000 | 0.1421 |
| circular_shift_replicate_2 | 10 | 0.1292 | 0.0714 | 0.5333 | 0.3000 | 0.0698 | 0.0000 | 0.1530 |
| circular_shift_frozen_mean | 10 | 0.1395 | 0.0706 | 0.5444 | 0.3667 | 0.0756 | 0.0000 | 0.1633 |
| cross_seed_phase_matched | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| block_permutation | 10 | 0.1305 | 0.0690 | 0.6000 | 0.3000 | 0.0829 | 0.0000 | 0.1566 |

| Placebo | Action checkpoints (source -> placebo) | Component actions (source -> placebo) | DRD2_activity declines (source -> placebo) | QED declines (source -> placebo) | Molecular weight declines (source -> placebo) | Weight L1 drift (source -> placebo) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy_shuffle_history | 8.30 -> 8.30 | 10.30 -> 12.20 | 4.20 -> 3.60 | 2.50 -> 4.60 | 3.60 -> 4.00 | 0.1462 -> 0.1288 |
| circular_shift_replicate_0 | 8.30 -> 8.50 | 10.30 -> 11.20 | 4.20 -> 4.20 | 2.50 -> 3.30 | 3.60 -> 3.70 | 0.1462 -> 0.1328 |
| circular_shift_replicate_1 | 8.30 -> 8.90 | 10.30 -> 11.70 | 4.20 -> 4.30 | 2.50 -> 3.30 | 3.60 -> 4.10 | 0.1462 -> 0.1563 |
| circular_shift_replicate_2 | 8.30 -> 8.60 | 10.30 -> 11.30 | 4.20 -> 4.10 | 2.50 -> 3.20 | 3.60 -> 4.00 | 0.1462 -> 0.1334 |
| circular_shift_frozen_mean | 8.30 -> 8.67 | 10.30 -> 11.40 | 4.20 -> 4.20 | 2.50 -> 3.27 | 3.60 -> 3.93 | 0.1462 -> 0.1408 |
| cross_seed_phase_matched | 8.30 -> 8.30 | 10.30 -> 10.30 | 4.20 -> 4.20 | 2.50 -> 2.50 | 3.60 -> 3.60 | 0.1462 -> 0.1462 |
| block_permutation | 8.30 -> 8.40 | 10.30 -> 11.10 | 4.20 -> 4.00 | 2.50 -> 3.20 | 3.60 -> 3.90 | 0.1462 -> 0.1382 |

`circular_shift_frozen_mean` is the formal per-seed mean over replicate indices 0, 1, and 2. The individual replica rows are retained so a favorable pooled result cannot mask a poor selected shift.

## Cross-Seed Target-Donor Preflight

Mapping is frozen as the supplied seed order circularly shifted by one: target i uses donor i+1 (mod n). It is one-to-one, never self-replays, and does not use any outcome or similarity-based donor selection.

| Target seed | Donor seed | Mean score difference | Checkpoint MAE | ACF gap | Delta KS | Decline-count gap | Slope gap | Phase-mean gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 23 | 0.0099 | 0.0311 | 0.2070 | 0.1667 | 1.0000 | 0.0015 | 0.0123 |
| 23 | 101 | 0.0133 | 0.0312 | 0.0960 | 0.2143 | 1.0000 | 0.0046 | 0.0224 |
| 101 | 202 | 0.0090 | 0.0310 | 0.3757 | 0.2857 | 1.6667 | 0.0029 | 0.0184 |
| 202 | 303 | 0.0072 | 0.0296 | 0.4504 | 0.3095 | 2.6667 | 0.0017 | 0.0134 |
| 303 | 404 | 0.0107 | 0.0332 | 0.3271 | 0.2143 | 2.3333 | 0.0023 | 0.0139 |
| 404 | 505 | 0.0108 | 0.0292 | 0.3628 | 0.2381 | 2.0000 | 0.0024 | 0.0135 |
| 505 | 606 | 0.0058 | 0.0308 | 0.3478 | 0.1667 | 1.0000 | 0.0023 | 0.0144 |
| 606 | 707 | 0.0078 | 0.0279 | 0.4437 | 0.2143 | 1.6667 | 0.0016 | 0.0105 |
| 707 | 808 | 0.0062 | 0.0290 | 0.2619 | 0.2143 | 0.3333 | 0.0004 | 0.0091 |
| 808 | 17 | 0.0124 | 0.0293 | 0.1921 | 0.1905 | 0.3333 | 0.0021 | 0.0147 |

Across the frozen mapping: mean score difference `0.0093`, checkpoint MAE `0.0302`, ACF gap `0.3065`, Delta KS `0.2214`, decline-count gap `1.4000`, slope gap `0.0022`, and phase-mean gap `0.0143`.

| Component | Mean score difference | Checkpoint MAE | ACF gap | Delta KS | Decline-count gap | Slope gap | Phase-mean gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DRD2_activity | 0.0048 | 0.0302 | 0.2513 | 0.2357 | 1.2000 | 0.0026 | 0.0152 |
| QED | 0.0100 | 0.0261 | 0.3807 | 0.2286 | 1.4000 | 0.0014 | 0.0118 |
| Molecular weight | 0.0131 | 0.0344 | 0.2874 | 0.2000 | 1.6000 | 0.0026 | 0.0158 |

Target-minus-donor signed component means, per-checkpoint values, and early/middle/late phase profiles are retained in the JSON output. Lower is better for all absolute-gap and distance columns in this report.
