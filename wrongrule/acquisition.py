from __future__ import annotations

import math

import torch
from torch import Tensor

from .data import PairPool, rng
from .metrics import predict

def audit(committee, pairs: PairPool) -> tuple[Tensor, Tensor]:

    if len(committee) < 2:
        raise ValueError("an audit requires at least two committee members")
    px = torch.stack([predict(m, pairs.x) for m in committee])
    pe = torch.stack([predict(m, pairs.edited) for m in committee])
    pair_scores = (pe - px).var(0, unbiased=False).mean(1)
    point_scores = .5 * (px.var(0, unbiased=False).mean(1) + pe.var(0, unbiased=False).mean(1))
    return pair_scores, point_scores

def farthest(features: Tensor, candidates: Tensor, k: int, centers: Tensor | None = None) -> Tensor:

    if not k:
        return torch.empty(0, dtype=torch.long)
    if k > len(candidates):
        raise ValueError("not enough candidates")
    z = features[candidates]
    if centers is None or not len(centers):
        distances = (z - z.mean(0)).square().sum(1)
    else:
        distances = torch.cdist(z, centers).square().min(1).values
    chosen = []
    for _ in range(k):
        index = int(distances.argmax())
        chosen.append(index)
        distances = torch.minimum(distances, (z - z[index]).square().sum(1))
        distances[torch.tensor(chosen)] = -1.
    return candidates[torch.tensor(chosen)]

def select_pairs(method: str, pairs: PairPool, available: Tensor, k: int,
                 scores: Tensor | None, seed: int, train_x: Tensor,
                 exploration: float = .2) -> Tensor:

    if not 0 <= exploration <= 1 or k < 0:
        raise ValueError("invalid acquisition configuration")
    candidates = torch.where(available)[0]
    if k > len(candidates):
        raise ValueError("candidate pool exhausted")
    g = rng(seed)
    shuffled = candidates[torch.randperm(len(candidates), generator=g)]
    if method == "random_pairs":
        return shuffled[:k]
    features = torch.cat([pairs.x, pairs.edited], dim=1)

    centers = torch.cat([train_x[:256], train_x[:256]], dim=1)
    if method == "diversity":
        return farthest(features, shuffled, k, centers)
    if scores is None or len(scores) != len(pairs) or not torch.isfinite(scores).all():
        raise ValueError("scored selection requires finite pool scores")
    n_random = min(k, math.ceil(k * exploration))
    random_part = shuffled[:n_random]
    remainder = shuffled[n_random:]
    n_scored = k - n_random
    if not n_scored:
        return random_part

    ordered = remainder[torch.argsort(scores[remainder], descending=True, stable=True)]
    shortlist = ordered[:min(len(ordered), 5 * n_scored)]
    if len(random_part):
        centers = torch.cat([centers, features[random_part]], dim=0)
    chosen = farthest(features, shortlist, n_scored, centers)
    return torch.cat([random_part, chosen])
