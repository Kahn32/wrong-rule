# Measured acquisition results

These are preliminary controlled-task measurements, not evidence of novelty or a general advantage.

| Method | Final held-out NMSE | Teacher bootstrap 95% interval | New labels |
|---|---:|---:|---:|
| random_id | 0.316967 | [0.316967, 0.316967] | 8 |
| random_pairs | 0.138409 | [0.138409, 0.138409] | 8 |
| diversity | 0.156182 | [0.156182, 0.156182] | 8 |
| seed_pairs | 0.176011 | [0.176011, 0.176011] | 8 |
| path_points | 0.132213 | [0.132213, 0.132213] | 8 |
| path_pairs | 0.151064 | [0.151064, 0.151064] | 8 |
| oracle_pairs | 0.151064 | [0.151064, 0.151064] | 8 |

Intervals resample independent teacher instances after averaging nested training seeds. With few teachers, intervals are descriptive and unreliable for broad inference.
All arms train both committees; ID validation selects the initial predictor decay, then freezes it across rounds. The oracle arm, if present, uses extra privileged labels and is not a fair budget-matched baseline.

## Direct paired comparison

Path-pair minus random-pair NMSE: 0.012655, teacher-bootstrap interval [0.012655, 0.012655]. Negative favors path-pair acquisition.

## Initial score quality

| Signal | Mean AUROC |
|---|---:|
| path_pairs | 0.557292 |
| seed_pairs | 0.776042 |
| path_points | 0.380208 |

See `results.json` for every run, per-cell errors, selected checkpoints, rank diagnostics, score metrics, and compute accounting.
The nonlinear rank measurements are descriptive; no causal conclusion about low rank is implied.
