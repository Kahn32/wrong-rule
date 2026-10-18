# Measured acquisition results

These are preliminary controlled-task measurements, not evidence of novelty or a general advantage.

| Method | Final held-out NMSE | Teacher bootstrap 95% interval | New labels |
|---|---:|---:|---:|
| random_id | 0.072085 | [0.048878, 0.086868] | 128 |
| random_pairs | 0.009734 | [0.005930, 0.011885] | 128 |
| diversity | 0.008066 | [0.006183, 0.009290] | 128 |
| seed_pairs | 0.008990 | [0.006150, 0.011543] | 128 |
| path_points | 0.008922 | [0.006067, 0.012072] | 128 |
| path_pairs | 0.007418 | [0.003949, 0.010567] | 128 |
| oracle_pairs | 0.005761 | [0.002327, 0.011762] | 128 |

Intervals resample independent teacher instances after averaging nested training seeds. With few teachers, intervals are descriptive and unreliable for broad inference.
All arms train both committees; ID validation selects the initial predictor decay, then freezes it across rounds. The oracle arm, if present, uses extra privileged labels and is not a fair budget-matched baseline.

## Direct paired comparison

Path-pair minus random-pair NMSE: -0.002316, teacher-bootstrap interval [-0.004148, -0.000820]. Negative favors path-pair acquisition.

## Initial score quality

| Signal | Mean AUROC |
|---|---:|
| path_pairs | 0.705566 |
| seed_pairs | 0.844591 |
| path_points | 0.611762 |

See `results.json` for every run, per-cell errors, selected checkpoints, rank diagnostics, score metrics, and compute accounting.
The nonlinear rank measurements are descriptive; no causal conclusion about low rank is implied.
