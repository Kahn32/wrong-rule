from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .acquisition import audit, select_pairs
from .data import LabelLedger, Support, TreeTeacher, candidate_pairs, point_key, rng, sample_inputs
from .io import complete_output, prepare_output, write_json
from .metrics import evaluate, predict, response_error, score_quality
from .models import spectra
from .training import TrainConfig, committees, fit

METHODS = ("random_id", "random_pairs", "diversity", "seed_pairs", "path_points", "path_pairs", "oracle_pairs")

def config_train(config: dict) -> TrainConfig:
    return TrainConfig(**config.get("train", {}))

def save_trained(out: Path, run, extra: dict):
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "training.json", {**run.metadata, **extra, "trace": run.trace})
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in run.model.state_dict().items()},
                "dim": run.model.features[0].in_features if len(run.model.features) else run.model.head.in_features,
                "train_config": run.metadata["config"], **extra}, out / "checkpoint.pt")

def save_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})

def sweep(config: dict, output: str) -> Path:
    train = config_train(config)
    train.validate()
    dim, device = config.get("dim", 8), config.get("device", "cpu")
    out = prepare_output(output, "sweep", config)
    rows = []
    for ts in config.get("teacher_seeds", [0]):
        teacher = TreeTeacher(dim, ts, config.get("teacher_mode", "contrast"))
        test_x = sample_inputs(config.get("n_test", 2048), dim, 1., 81001 + ts)
        test_y = teacher(test_x)
        response_pairs = candidate_pairs(256, dim, 82001 + ts)
        for seed in config.get("training_seeds", [0]):
            base_seed = 100000 * ts + 1000 * seed + 17
            for p in config.get("coverages", [0., .02, .2, 1.]):
                x = sample_inputs(config.get("n_train", 512), dim, p, base_seed)
                clean_y = teacher(x)
                y = clean_y + config.get("noise", .01) * torch.randn(clean_y.shape, generator=rng(base_seed + 1))
                vx = sample_inputs(config.get("n_val", 512), dim, p, base_seed + 2)
                vy = teacher(vx)
                for decay in config.get("decays", [0., .001, .03]):
                    cfg = replace(train, weight_decay=decay)
                    run = fit(x, y, vx, vy, cfg, seed + 42, base_seed + 3, device)
                    row = {"teacher_seed": ts, "training_seed": seed, "coverage": p,
                           "weight_decay": decay, "optimizer": cfg.optimizer,
                           "train_clean": evaluate(run.model, x, clean_y),
                           "id": evaluate(run.model, vx, vy),
                           "ood": evaluate(run.model, test_x, test_y),
                           "response_mse": float(response_error(run.model, response_pairs,
                               teacher(response_pairs.x), teacher(response_pairs.edited)).mean()),
                           "spectra": spectra(run.model, vx.to(device)), "training": run.metadata}
                    tag = f"t{ts}_s{seed}_p{p:g}_wd{decay:g}"
                    save_trained(out / "runs" / tag, run, {"teacher_seed": ts, "training_seed": seed})
                    rows.append(row)
                    write_json(out / "results.json", rows)
                    print(f"sweep {tag} ID={row['id']['mse']:.6f} OOD={row['ood']['mse']:.6f}", flush=True)
    flat = [{**r, "id_mse": r["id"]["mse"], "ood_mse": r["ood"]["mse"],
             "ood_nmse": r["ood"]["nmse"], "selected_step": r["training"]["selected_step"]} for r in rows]
    save_csv(out / "results.csv", flat, ["teacher_seed", "training_seed", "coverage", "weight_decay",
             "optimizer", "id_mse", "ood_mse", "ood_nmse", "response_mse", "selected_step"])
    complete_output(out)
    return out

def _validate_active(config):
    decays = config.get("decays", [0., .003, .1])
    if len(decays) < 2 or len(set(decays)) != len(decays) or min(decays) < 0:
        raise ValueError("need >=2 distinct nonnegative committee decays")
    methods = config.get("methods", list(METHODS[:-1]))
    if len(set(methods)) != len(methods) or not set(methods) <= set(METHODS) or not methods:
        raise ValueError(f"methods must be distinct and drawn from {METHODS}")
    labels, rounds = config.get("labels_per_round", 64), config.get("rounds", 2)
    if labels < 2 or labels % 2 or rounds < 1:
        raise ValueError("labels_per_round must be positive/even; rounds >=1")
    if labels // 2 * rounds > config.get("pool_pairs", 512):
        raise ValueError("pool too small for label budget")
    if config.get("audit_pairs", 256) < 8:
        raise ValueError("audit_pairs must be >=8 for score-quality metrics")
    return decays, methods, labels, rounds

def active(config: dict, output: str) -> Path:
    decays, methods, labels_per_round, rounds = _validate_active(config)
    cfg = config_train(config); cfg.validate()
    support = Support(**{k: tuple(v) if isinstance(v, list) else v for k, v in config.get("support", {}).items()})
    dim, device = config.get("dim", 8), config.get("device", "cpu")
    n_train, n_val = config.get("n_train", 512), config.get("n_val", 512)
    out = prepare_output(output, "active", config)
    write_json(out / "support.json", asdict(support))
    rows = []
    for ts in config.get("teacher_seeds", [0]):
        teacher = TreeTeacher(dim, ts, config.get("teacher_mode", "contrast"))
        for seed in config.get("training_seeds", [0]):
            base = 100000 * ts + 1000 * seed + 313
            x0 = sample_inputs(n_train, dim, config.get("coverage", .02), base, support)
            vx = sample_inputs(n_val, dim, config.get("coverage", .02), base + 1, support)
            vy = teacher(vx)
            tx = sample_inputs(config.get("n_test", 2048), dim, 1., base + 2, support, "test")
            ty = teacher(tx)
            pool = candidate_pairs(config.get("pool_pairs", 512), dim, base + 3, support)
            audit_pool = candidate_pairs(config.get("audit_pairs", 256), dim, base + 5, support, "test")

            ay, ae = teacher(audit_pool.x), teacher(audit_pool.edited)
            assert support.contains(x0, "train").all()
            assert support.contains(pool.x, "acquisition").all() and support.contains(pool.edited, "acquisition").all()
            assert support.contains(tx, "test").all()

            keys = [point_key(row) for row in torch.cat([pool.x, pool.edited])]
            if len(set(keys)) != len(keys):
                raise RuntimeError("pool endpoints unexpectedly collide")
            pair_path = out / "data" / f"t{ts}_s{seed}"
            pair_path.mkdir(parents=True, exist_ok=True)
            torch.save({"initial_train_x": x0, "val_x": vx, "test_x": tx,
                        "pool_x": pool.x, "pool_edited": pool.edited,
                        "pool_coordinate": pool.coordinate, "teacher_state": teacher.state_dict(),
                        "teacher_mode": teacher.mode, "teacher_seed": ts}, pair_path / "problem.pt")
            for method in methods:
                ledger = LabelLedger(teacher, n_train + labels_per_round * rounds,
                                     config.get("noise", .01), base + 9)
                x, y = x0.clone(), ledger.query(x0)
                initial_calls = ledger.calls
                available = torch.ones(len(pool), dtype=torch.bool)
                selected_index = None
                cumulative = {"training_runs": 0, "optimizer_steps": 0, "seconds": 0.}
                oracle_score_labels = 0
                method_dir = out / "runs" / f"t{ts}_s{seed}" / method
                for round_id in range(rounds + 1):
                    path, seeds, target, selected_index, accounting = committees(
                        x, y, vx, vy, cfg, decays, seed + 42, base + 11, device, selected_index)
                    for key in cumulative:
                        cumulative[key] += accounting[key]
                    start_audit = time.perf_counter()
                    audit_path, audit_point = audit([r.model for r in path], audit_pool)
                    audit_seed, _ = audit([r.model for r in seeds], audit_pool)
                    true_errors = response_error(target.model, audit_pool, ay, ae)
                    score_eval = {name: score_quality(score, true_errors) for name, score in
                                  [("path_pairs", audit_path), ("seed_pairs", audit_seed), ("path_points", audit_point)]}
                    row = {"teacher_seed": ts, "training_seed": seed, "method": method,
                           "round": round_id, "initial_labels": initial_calls,
                           "acquired_labels": ledger.calls - initial_calls,
                           "oracle_scoring_labels": oracle_score_labels,
                           "id": evaluate(target.model, vx, vy, support),
                           "ood": evaluate(target.model, tx, ty, support),
                           "response_mse": float(true_errors.mean()), "score_quality": score_eval,
                           "selected_decay": decays[selected_index], "compute": dict(cumulative),
                           "spectra": spectra(target.model, vx.to(device)),
                           "selected_step": target.metadata["selected_step"],
                           "audit_eval_seconds": time.perf_counter() - start_audit}
                    rows.append(row)
                    run_dir = method_dir / f"round_{round_id}"
                    save_trained(run_dir, target, {"teacher_seed": ts, "training_seed": seed,
                                                 "method": method, "round": round_id})

                    for family, members in [("path", path), ("seeds", seeds)]:
                        for j, member in enumerate(members):
                            if family == "seeds" and j == 0:
                                continue
                            save_trained(run_dir / f"{family}_{j}", member, {"teacher_seed": ts, "round": round_id})
                    np.savez_compressed(run_dir / "audit_scores.npz", path_pairs=audit_path.numpy(),
                                        seed_pairs=audit_seed.numpy(), path_points=audit_point.numpy(),
                                        true_response_errors=true_errors.numpy())
                    write_json(out / "results.json", rows)
                    print(f"active t{ts}/s{seed} {method} round={round_id} labels={row['acquired_labels']} "
                          f"ID={row['id']['mse']:.6f} heldout={row['ood']['mse']:.6f}", flush=True)
                    if round_id == rounds:
                        continue
                    selection_start = time.perf_counter()
                    before = ledger.calls
                    if method == "random_id":
                        new_x = sample_inputs(labels_per_round, dim, config.get("coverage", .02),
                                              base + 100 + round_id, support)
                        selected = torch.empty(0, dtype=torch.long)
                        new_y = ledger.query(new_x)
                    else:

                        pool_path, pool_points = audit([r.model for r in path], pool)
                        pool_seed, _ = audit([r.model for r in seeds], pool)
                        choices = {"path_pairs": pool_path, "seed_pairs": pool_seed, "path_points": pool_points}
                        if method == "oracle_pairs":
                            choices[method] = response_error(target.model, pool, teacher(pool.x), teacher(pool.edited))
                            oracle_score_labels += 2 * len(pool)
                        score = choices.get(method)
                        selected = select_pairs(method, pool, available, labels_per_round // 2,
                                                score, base + 200 + round_id, x,
                                                config.get("exploration", .2))
                        available[selected] = False
                        new_x = torch.cat([pool.x[selected], pool.edited[selected]])
                        assert not support.contains(new_x, "test").any()
                        new_y = ledger.query(new_x)
                    actual_cost = ledger.calls - before
                    if actual_cost != labels_per_round:
                        raise RuntimeError(f"label count mismatch: {actual_cost} vs {labels_per_round}")
                    write_json(run_dir / "acquisition.json", {
                        "selected_pair_indices": selected.tolist(), "new_labels": actual_cost,
                        "selection_seconds": time.perf_counter() - selection_start,
                        "oracle_scoring_labels_cumulative": oracle_score_labels,
                    })
                    x, y = torch.cat([x, new_x]), torch.cat([y, new_y])
    flat = [{**r, "id_mse": r["id"]["mse"], "ood_mse": r["ood"]["mse"],
             "ood_nmse": r["ood"]["nmse"], "worst_cell_mse": r["ood"]["worst_cell_mse"],
             "training_runs": r["compute"]["training_runs"]} for r in rows]
    save_csv(out / "results.csv", flat, ["teacher_seed", "training_seed", "method", "round", "acquired_labels",
              "id_mse", "ood_mse", "ood_nmse", "worst_cell_mse", "response_mse", "selected_decay",
              "training_runs", "selected_step", "oracle_scoring_labels"])
    complete_output(out)
    return out
