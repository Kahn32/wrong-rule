from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from torch import Tensor

from .data import PairPool, Support

@torch.no_grad()
def predict(model, x: Tensor, batch_size: int = 2048) -> Tensor:
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([model(batch.to(device)).cpu() for batch in x.split(batch_size)])

def evaluate(model, x: Tensor, y: Tensor, support: Support | None = None) -> dict:
    pred = predict(model, x)
    errors = (pred - y).square().mean(1)
    mse = float(errors.mean())
    variance = float((y - y.mean(0)).square().mean())
    result = {"mse": mse, "nmse": mse / max(variance, 1e-12),
              "target_variance": variance, "examples": len(x)}
    if support is not None:
        bins = (((x[:, :2] + 1) * support.bins / 2).floor().long()).clamp(0, support.bins - 1)
        cells = bins[:, 0] * support.bins + bins[:, 1]
        cell_errors = {str(int(i)): float(errors[cells == i].mean()) for i in cells.unique()}
        result.update({"cell_mse": cell_errors, "worst_cell_mse": max(cell_errors.values()),
                       "represented_cells": len(cell_errors)})
    return result

@torch.no_grad()
def response_error(model, pairs: PairPool, y: Tensor, edited_y: Tensor) -> Tensor:
    delta = predict(model, pairs.edited) - predict(model, pairs.x)
    return (delta - (edited_y - y)).square().mean(1)

def score_quality(scores: Tensor, error: Tensor) -> dict:
    s, e = scores.double().numpy(), error.double().numpy()
    top = max(1, len(e) // 4)
    positives = np.zeros(len(e), dtype=bool)
    positives[np.argsort(e, kind="stable")[-top:]] = True
    if np.ptp(s) < 1e-15:
        correlation, auroc = None, .5
        precision = float(positives.mean())
    else:
        correlation = float(spearmanr(s, e).statistic) if np.ptp(e) else None
        n_pos, n_neg = int(positives.sum()), int((~positives).sum())
        ranks = rankdata(s, method="average")
        auroc = float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
        precision = float(positives[np.argsort(s, kind="stable")[-top:]].mean())
    return {"spearman": correlation, "auroc_top_quartile_error": auroc,
            "precision_at_25_percent": precision, "n": len(e)}
