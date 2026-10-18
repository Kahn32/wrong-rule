# Implementation and measured-results report

Completed 17 October 2026. This deliverable is an independent PyTorch research prototype with executed experiments. It is not an official Poggio Lab implementation or a completed scientific validation of the proposal.

## Delivered and executed

- A Python package and CLI, with JSON configurations and no external data/API requirement.
- The primary nonlinear teacher benchmark, a separate held-out-combination acquisition study, all seven baseline arms, and the analytical ridge check.
- 13 scientific correctness tests, all passed on CPU Linux; test output is in `results/validation.log`.
- A complete smoke run, all seven methods, to verify integration.
- 36 nonlinear sweep training runs: three teacher instances, four coverage settings, three decay settings.
- 315 acquisition training runs: three teacher instances, seven arms, three predeclared rounds, five distinct committee training runs per arm/round.
- Thus 351 model-training runs in the two main pilots, each with 1,000 optimizer steps. These are not 351 independent scientific replicates: the study has only three teacher instances and one initialization seed per teacher.
- Checkpoints, training traces, raw JSON/CSV metrics, selected endpoints, committee scores, source hashes, and plotted results.
- Successful package build/install and successful loading/inference from a saved checkpoint.

## Acquisition outcome

Every nonprivileged arm starts from 512 training labels and receives 128 additional endpoint labels over two acquisition rounds. ID validation has 512 separately labeled, clean simulated examples. All arms have the same predictor/committee training budget.

| Method | Mean final held-out normalized MSE |
|---|---:|
| random_id | 0.072085 |
| random_pairs | 0.009734 |
| diversity | 0.008066 |
| seed_pairs | 0.008990 |
| path_points | 0.008922 |
| path_pairs | 0.007418 |
| oracle_pairs | 0.005761 |

Lower is better. The path-pair method's mean error was **23.8% lower than random-pair acquisition** in this particular three-teacher pilot. Its error was also lower on average than diversity-only and the seed committee, but the ranking varied by teacher and round. The oracle arm has additional privileged label access and is not a fair budget-matched competitor.

This is promising development evidence, not a general method claim. There are only three closely related teacher instances, a single training seed, one task topology, and a short training budget. The bootstrap intervals in the generated report are descriptive, not reliable evidence of population-wide significance. The confirmation configuration has **not** been run.

## An important mixed result

Initial failure-detection AUROC was 0.705566 for path-pair disagreement, 0.844591 for seed-pair disagreement, and 0.611762 for path-point disagreement. **The ordinary seed committee was the better initial failure detector.** A claim that the proposed audit universally detects errors better would contradict these results. Acquisition performance also depends on diversity, exploration, and how useful selected labels are; error detection and label efficiency are different endpoints.

## Coverage/regularization outcome

With zero independent-factor coverage and no decay, mean ID MSE was approximately 0.000171 while independent-factor test MSE was 0.052274. With complete coverage, those values were approximately 0.000235 and 0.000234. This illustrates the importance of distinguishing evidence, but the zero-coverage case is intentionally unidentifiable.

At coverage 0.02, stronger AdamW decay (0.1 rather than 0) raised mean OOD MSE from approximately 0.003867 to 0.004821. At zero coverage it lowered OOD MSE. The effect is regime-dependent, and this pilot does not establish the stronger claim that decay improves ordinary validation while harming composition. It also does not identify low rank as the cause. All sweep outcomes are retained in `results/pilot_sweep/RESULTS.md`.

## Relation to the literature

Five core papers were checked against the lab publication list and the original papers. Source URLs, inspected sections, repository provenance, and implementation boundaries are documented in `SOURCES.md`. The initialization study uses Keras/TensorFlow; this package independently uses PyTorch. Huang et al.'s multiplicative norm constraint is not equated with this benchmark's ordinary weight decay.

Verified repositories include actual lab research-automation software, PoggioAI/PoggioAI_MSc, plus the official modular-hyperteacher and compositional-ood-generalization repositories from adjacent research groups. The adjacent papers are explicitly identified as non-Poggio papers. No official repository was verified for the five core paper implementations; that does not establish that none exists. No external source code was copied or silently ported.

## Still research work, not completed implementation

The primary controlled study is runnable. A visual benchmark, a port/extension of the public modular-hyperteacher experiments, causal rank-removal/rescue experiments, an exhaustive novelty review, and a nonlinear theorem remain future work. A larger fresh-seed confirmation run is provided as a configuration, not presented as an executed result.

The next scientific decision is whether path-dependent responses add a stable, interpretable advantage beyond well-tuned standard active learning. Establish that before adding architectural complexity or pitching the project as a novel research contribution.
