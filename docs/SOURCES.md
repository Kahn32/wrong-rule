# Verified papers, repositories, and implementation boundaries

Checked 17 October 2026. The lab's own [publication list](https://poggio-lab.mit.edu/publications) identifies the five core papers below. Original paper text was retrieved and read; document URLs and SHA-256 hashes are recorded in `paper_verification.json`. Full copyrighted papers are not redistributed in this project.

## Actual Poggio Lab papers

| Paper and authors | Original source | Relevant material inspected | Verified author repository |
|---|---|---|---|
| **Position: A Theory of Deep Learning Must Include Compositional Sparsity** - David A. Danhofer, Davide D'Ascenzo, Rafael Dubach, Tomaso Poggio | [ICML 2025](https://proceedings.mlr.press/v267/danhofer25a.html), [full text](https://arxiv.org/html/2507.02550v1) | Definition of compositional sparsity; representation versus learnability; open questions in section 4 | Not located in the checked paper or targeted search |
| **Learning Sparse Compositional Functions with Norm-Constrained Neural Networks** - Shuo Huang, Lorenzo Fiorito, Lorenzo Rosasco, Tomaso Poggio | [May 2026 preprint, full text](https://arxiv.org/html/2605.25608v1) | Section 3.1's multiplicative Frobenius measure, statistical analysis, and section 4.1's constrained-ERM/regularization discussion | Not located in the checked paper or targeted search |
| **SGD with Weight Decay Secretly Minimizes the Ranks of Your Neural Networks** - Tomer Galanti, Zachary S. Siegel, Aparna Gupte, Tomaso Poggio | [CPAL 2025](https://proceedings.mlr.press/v280/galanti25a.html), [original PDF](https://raw.githubusercontent.com/mlresearch/v280/main/assets/galanti25a/galanti25a.pdf) | Objective and training setup; recent-update/forgetting mechanism; rank/generalization findings; appendix experiments | Not located in the checked PDF or targeted search |
| **Do Deep Networks Forget Initialization? A Forgetting-Time View of Practical Inductive Bias** - Mohua Das, Pierfrancesco Beneventano, Shibshankar Dey, Gareth H. McKinley, Tomaso Poggio | [May 2026 preprint, full text](https://arxiv.org/html/2605.29152v1) | Definition of initialization memory, optimizer comparisons, limitations, Appendix A's Keras 3/TensorFlow setup and checkpointing | Not located in the checked paper/author page or targeted search |
| **System Identification of Neural Systems: If We Got It Right, Would We Know?** - Yena Han, Tomaso Poggio, Brian Cheung | [ICML 2023](https://proceedings.mlr.press/v202/han23d.html), [original PDF](https://proceedings.mlr.press/v202/han23d/han23d.pdf) | Known-target evaluation, encoding/CKA comparison, architecture-identification limitations | Not located; the cited `pytorch-image-models` repository is a dependency, not this paper's official implementation |

Not located does not mean a repository does not exist. arXiv's generic GitHub conversion-issue links are not paper-code repositories. Author/title spelling differs in some metadata for the initialization paper; the PDF's McKinley spelling is used here.

## What these papers contribute to this code

- **Compositionality position paper:** motivates the teacher's sparse graph. No theorem from the paper is claimed to guarantee this learner's success.
- **Norm-constrained paper:** motivates studying structure and complexity jointly. Its actual norm constraint is not enforced by this code; ordinary weight decay is a separate treatment.
- **SGD/rank paper:** motivates varying decay and recording spectra. The original recent-update mechanism and reported image-classification experiments are not reproduced here.
- **Initialization paper:** motivates treating the complete training procedure as an experimental variable. Original experiments use Keras/TensorFlow; this benchmark uses independently written PyTorch MLPs, not its ResNet/CIFAR setup.
- **System-identification paper:** motivates using a known teacher and evaluating behavior without interpreting a similarity score as structural proof.

## Repositories actually verified

| Repository | Provenance and role | Framework/status |
|---|---|---|
| [PoggioAI/PoggioAI_MSc](https://github.com/PoggioAI/PoggioAI_MSc) | Actual Poggio Lab research-automation repository, linked from coauthor Pierfrancesco Beneventano's [page](https://pierbeneventano.github.io/), associated with [pAI/MSc](https://arxiv.org/abs/2604.20622) | Separate system; not used as a learner or a dependency |
| [smonsays/modular-hyperteacher](https://github.com/smonsays/modular-hyperteacher) | Official code linked by Schug et al., [Discovering Modular Solutions That Generalize Compositionally](https://arxiv.org/html/2312.15001v2), ICLR 2024 | JAX/metax; adjacent research, **not a Poggio Lab paper**; not ported or run here |
| [brendel-group/compositional-ood-generalization](https://github.com/brendel-group/compositional-ood-generalization) | Official code for Wiedemer et al., [Compositional Generalization from First Principles](https://arxiv.org/html/2307.05596v1), NeurIPS 2023 | Adjacent research, **not a Poggio Lab paper**; not copied or run here |

Repository branches, inspected README hashes, and commit IDs are recorded in `repository_verification.json` when retrievable. No third-party source code was copied into the implementation.

## Novelty assessment

The general ideas are not new: compositional support conditions, module identification, compression-related failures, and disagreement-based active learning all have prior work. Importantly, Schug et al. already evaluate weight decay and module alignment in Appendix B.1. Repeating that observation cannot establish a new contribution.

The candidate contribution is a quantitative relationship between distinguishing evidence and regularization-dependent response changes, plus a validated advantage for choosing interventions. The delivered pilot tests components of that hypothesis. It does not prove priority or establish such an advantage. An exhaustive novelty review and fresh confirmation study remain necessary.
