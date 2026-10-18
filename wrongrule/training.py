from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, replace

import torch
from torch import Tensor

from .data import rng
from .models import MLP

@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    batch_size: int = 128
    lr: float = .003
    weight_decay: float = .001
    optimizer: str = "adamw"
    momentum: float = .9
    width: int = 64
    depth: int = 3
    activation: str = "tanh"
    init_scale: float = 1.
    eval_every: int = 100
    checkpoint: str = "best_id"

    def validate(self):
        if self.steps < 1 or self.batch_size < 1 or self.width < 1 or self.depth < 0:
            raise ValueError("invalid training dimensions or budget")
        if self.lr <= 0 or self.weight_decay < 0 or self.eval_every < 1 or self.init_scale <= 0:
            raise ValueError("invalid learning rate, decay, evaluation cadence or initialization")
        if self.optimizer not in {"sgd", "momentum", "adamw"}:
            raise ValueError("optimizer must be sgd, momentum or adamw")
        if self.checkpoint not in {"best_id", "last"}:
            raise ValueError("checkpoint must be best_id or last")

@dataclass
class Trained:
    model: MLP
    metadata: dict
    trace: list[dict]

def fit(x: Tensor, y: Tensor, val_x: Tensor, val_y: Tensor, config: TrainConfig,
        init_seed: int, order_seed: int, device: str = "cpu") -> Trained:

    config.validate()
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("invalid training data")
    start = time.perf_counter()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        model = MLP(x.shape[1], config.width, config.depth,
                    config.activation, config.init_scale).to(device)
    xd, yd, vx, vy = (a.to(device) for a in (x, y, val_x, val_y))

    groups = [
        {"params": [p for n, p in model.named_parameters() if n.endswith("weight")],
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n.endswith("bias")],
         "weight_decay": 0.},
    ]
    if config.optimizer == "adamw":
        opt = torch.optim.AdamW(groups, lr=config.lr)
    else:
        opt = torch.optim.SGD(groups, lr=config.lr,
                              momentum=config.momentum if config.optimizer == "momentum" else 0.)
    generator = rng(order_seed)
    best, best_state, best_step = float("inf"), None, 0
    trace = []
    for step in range(1, config.steps + 1):
        model.train()
        ids = torch.randint(len(x), (min(config.batch_size, len(x)),), generator=generator).to(device)
        opt.zero_grad(set_to_none=True)
        loss = (model(xd[ids]) - yd[ids]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"training diverged at step {step}")
        loss.backward()
        opt.step()
        if step % config.eval_every == 0 or step == 1 or step == config.steps:
            model.eval()
            with torch.no_grad():
                val_loss = float((model(vx) - vy).square().mean().cpu())
                train_loss = float((model(xd) - yd).square().mean().cpu())
            trace.append({"step": step, "train_mse": train_loss, "id_val_mse": val_loss})
            if val_loss < best:
                best, best_step = val_loss, step
                best_state = copy.deepcopy(model.state_dict())
    last_train_mse = trace[-1]["train_mse"]
    if config.checkpoint == "best_id":
        model.load_state_dict(best_state)
    model.eval()
    return Trained(model, {
        "config": asdict(config), "init_seed": init_seed, "order_seed": order_seed,
        "selected_step": best_step if config.checkpoint == "best_id" else config.steps,
        "best_id_val_mse": best, "last_train_mse": last_train_mse,
        "optimizer_steps": config.steps,
        "train_examples": len(x), "seconds": time.perf_counter() - start,
    }, trace)

def committees(x: Tensor, y: Tensor, vx: Tensor, vy: Tensor, config: TrainConfig,
               decays: list[float], init_seed: int, order_seed: int,
               device: str = "cpu", selected_index: int | None = None):

    path = [fit(x, y, vx, vy, replace(config, weight_decay=d), init_seed,
                order_seed, device) for d in decays]
    if selected_index is None:
        selected_index = min(range(len(path)), key=lambda j: path[j].metadata["best_id_val_mse"])
    target = path[selected_index]
    seeds = [target] + [fit(x, y, vx, vy, replace(config, weight_decay=decays[selected_index]),
                           init_seed + 1009 * j, order_seed, device) for j in range(1, len(decays))]
    unique_runs = path + seeds[1:]
    accounting = {
        "training_runs": len(unique_runs),
        "optimizer_steps": sum(r.metadata["optimizer_steps"] for r in unique_runs),
        "seconds": sum(r.metadata["seconds"] for r in unique_runs),
        "selected_decay": decays[selected_index],
    }
    return path, seeds, target, selected_index, accounting
