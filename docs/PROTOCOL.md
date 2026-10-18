# Experimental protocol

## Research question

Can a simplicity preference suppress a functionally important distinction when evidence separating two inputs is weak? Can changes in predicted responses across regularization settings identify useful new experiments better than ordinary uncertainty or seed disagreement?

The linear ridge calculation is standard. Any scientific contribution would have to come from the nonlinear evidence, a defensible quantitative explanation, or a validated improvement in experimental design.

## Teacher and observations

`TreeTeacher` is a seeded sparse DAG of bounded two-input nonlinearities. Its first leaf is contrast-sensitive, `tanh(1.2*x0 - 1.2*x1)`, or sum-sensitive in the negative control. Remaining constituent coefficients vary with teacher seed. A common-direction skip preserves ID signal even when the contrast has zero training variation. Graph size is linear in the input dimension. All implemented teachers share one topology; varying coefficients does not test transfer across arbitrary task families.

Learners receive observed input coordinates and scalar final-output labels. They never receive intermediate teacher values or teacher weights. Input edits are explicitly available. This edit access is extra structure, and is shared by all pair-acquisition baselines. It does not solve intervention discovery for entangled real observations.

Training labels have reproducible additive Gaussian measurement noise. ID validation and evaluation use clean simulated outputs; that idealization is declared and shared by all arms. The fixed validation set is additional supervision, recorded separately from acquired training labels. No reference-set labels are used to normalize learner targets.

## Two different OOD experiments

### Coverage sweep

Draw `(x0,x1)=(U,U)` with probability `1-p`, and independent uniforms `(U,V)` with probability `p`. Other coordinates are independent uniforms. Each factor's marginal remains uniform. The independent-factor reference test is fixed for a teacher across decay/coverage settings.

For `p>0`, reference-test combinations may be rare rather than mathematically absent. This experiment must not be described as strictly zero-shot composition testing. At `p=0`, distinct extensions can be observationally indistinguishable; failure is an information limit, not necessarily a training defect.

### Acquisition experiment

Discretize x0 and x1 into eight bins and define residue `(bin(x1)-bin(x0)) mod 8`.

| Role | Residues |
|---|---|
| Training and ID validation | 0, 1, 2, 3 |
| Acquisition pool, both endpoints | 4, 5 |
| Locked final evaluation, including audit pairs | 6, 7 |

Each residue occupies one cell in every row and column, preserving uniform factor marginals within each partition. The training distribution mixes the exact diagonal with a uniform draw over its permitted cells. A pair changes exactly one observed coordinate, using rejection sampling to keep both endpoints permitted. Final-test residues are never queried, used for checkpoints, or used in acquisition scoring.

Evaluation is logged at predeclared rounds to form learning curves. Evaluation labels used to assess score quality never enter the scorer or selection function. The entire three-teacher run is exploratory development evidence, not a blinded confirmation study.

## Committees and checkpoint rules

The path committee varies kernel weight decay with fixed initialization and batch-order seed. The seed committee fixes decay and batch-order seed but varies initialization. Its first member is the selected target predictor.

Select the target's decay once, on initial ID validation, and freeze that choice across rounds. At each round, every model is trained from scratch for the same optimizer-step budget. Checkpoints use only ID validation loss; the full loss trace and last training loss are retained. No OOD-driven stopping is allowed.

Every acquisition arm trains both committees, including baselines that do not use them. With three decays, that is five unique training runs per round: three path models plus two additional seed models. This equalizes committee-training budgets. Inference and selection overheads are recorded separately and can still differ.

These are AdamW/SGD experiments, not an implementation of Huang et al.'s multiplicative Frobenius norm constraint. In PyTorch, coupled SGD `weight_decay=d` contributes `d*W` to the gradient; this convention differs by a factor of two from an objective written as `loss + lambda*||W||^2`. Biases are excluded from decay here.

## Audit and policies

For pair `(x,x')`, model j predicts `Delta_j=f_j(x')-f_j(x)`. The pair score is population variance of Delta across models. The point score averages predictive variance at the two endpoints. Targets never enter either score.

Scored policies reserve 20% of each pair budget, rounded upward, for random exploration. Remaining selection uses a top-5k shortlist followed by greedy diversity on concatenated raw endpoint coordinates. Seed and path policies share this exact procedure. Diversity-only and random-pair arms isolate those components. With a tiny pool, shortlist/diversity can dominate and different signals can select identical batches; the smoke run is only a correctness check.

Every selected pair contributes two unique endpoint labels in this generated pool. The ledger caches repeated points, rejects a batch exceeding its budget before mutating state, and keys measurement noise to endpoints so acquisition arms receive the same label for the same point. Selected pairs are removed from later rounds. The ordinary-ID baseline obtains the same number of new labels from its original distribution.

The oracle reference scores true response error using all pool labels. Its extra scoring-label evaluations are recorded separately. They are repeated evaluations, not deduplicated unique-label costs. The oracle is privileged and cannot be called a fair budget-matched competitor or a guaranteed performance upper bound.

## Metrics and inference

- ID and held-out MSE and NMSE; NMSE divides by the evaluation target variance.
- Worst observed final-test cell MSE, with per-cell error logs.
- Mean squared error of controlled response changes.
- Spearman association, AUROC, and precision for the highest-error quartile of audit pairs.
- Layer stable ranks, kernel norms, and centered final-hidden activation stable rank.
- Oracle endpoints, training runs, optimizer steps, training duration, and selection duration.

Average training seeds within teacher instances, then average teachers. Bootstrap teacher instances for descriptive intervals. Three teachers are insufficient for strong inferential claims. Initial audit comparisons use only one method's identical round-zero models, avoiding duplicate pseudo-observations across arms.

Layer rescaling and nonlinear representation changes can confound rank interpretations. Rank correlations do not identify a computation graph. The implementation does not perform causal removal/rescue interventions or claim to do so.

## Confirmation and stop criteria

The provided confirmation configuration uses fresh teacher seeds 100-109 and training seeds 10-11. It has not been run. Before using it, freeze a method using development evidence and register the intended comparisons. Check that conclusions survive convergence checks, teacher-family variations, sum controls, noise changes, and ordinary active-learning baselines. If path disagreement adds no value, report that outcome and reconsider the acquisition claim rather than searching the final test for a favorable setting.
