from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from wrongrule.acquisition import audit, select_pairs
from wrongrule.analysis import ridge_errors
from wrongrule.data import LabelLedger, PairPool, Support, TreeTeacher, candidate_pairs, rng, sample_inputs
from wrongrule.experiments import active
from wrongrule.metrics import predict, score_quality
from wrongrule.models import stable_rank
from wrongrule.training import TrainConfig, fit

torch.set_num_threads(1)

class SamplingTests(unittest.TestCase):
    def test_partitions_are_exhaustive_and_disjoint(self):
        support = Support()
        for group in ["train", "acquisition", "test"]:
            x = sample_inputs(1000, 8, .02, 123, support, group)
            self.assertTrue(support.contains(x, group).all())
            for other in ["train", "acquisition", "test"]:
                if group != other:
                    self.assertFalse(support.contains(x, other).any())
        with self.assertRaises(ValueError):
            Support(train=(0, 1), acquisition=(1, 2), test=(3, 4, 5, 6, 7))

    def test_marginals_remain_uniform_when_joint_dependence_changes(self):
        support = Support()
        for group in ["train", "acquisition", "test"]:
            for coverage in [0., .05, 1.]:
                x = sample_inputs(40000, 4, coverage, 92, support, group)
                for coordinate in [0, 1]:
                    bins = (((x[:, coordinate] + 1) * 4).floor().long()).clamp(0, 7)
                    probabilities = torch.bincount(bins, minlength=8).float() / len(x)
                    self.assertLess(float((probabilities - .125).abs().max()), .009)

    def test_zero_coverage_is_exactly_unidentifiable(self):
        x = sample_inputs(128, 8, 0., 1)
        self.assertTrue(torch.equal(x[:, 0], x[:, 1]))
        a = x[:, 0] + x[:, 1]
        b = a + 7 * (x[:, 0] - x[:, 1])
        self.assertTrue(torch.equal(a, b))

    def test_pairs_change_one_coordinate_and_never_leave_their_partition(self):
        support = Support()
        for group in ["acquisition", "test"]:
            p = candidate_pairs(1000, 8, 73, support, group)
            self.assertTrue(((p.x != p.edited).sum(1) == 1).all())
            self.assertTrue(support.contains(p.x, group).all())
            self.assertTrue(support.contains(p.edited, group).all())

    def test_sum_control_is_invariant_to_contrast_at_fixed_sum(self):
        x = sample_inputs(40, 8, 1., 5)
        edited = x.clone()
        edited[:, 0], edited[:, 1] = x[:, 1], x[:, 0]
        teacher = TreeTeacher(8, 3, "sum")
        self.assertTrue(torch.allclose(teacher(x), teacher(edited), atol=1e-7))
        contrast = TreeTeacher(8, 3, "contrast")
        self.assertGreater(float((contrast(x) - contrast(edited)).abs().mean()), .01)

class OracleTests(unittest.TestCase):
    def test_cached_endpoints_cost_once_and_rejected_batch_has_no_side_effect(self):
        t = TreeTeacher(4, 0)
        x = sample_inputs(3, 4, 1., 4)
        ledger = LabelLedger(t, 2, noise=.1, seed=1)
        ys = ledger.query(torch.stack([x[0], x[0], x[1]]))
        self.assertEqual(ledger.calls, 2)
        self.assertTrue(torch.equal(ys[0], ys[1]))
        with self.assertRaises(ValueError):
            ledger.query(x[2:])
        self.assertEqual(ledger.calls, 2)
        other = LabelLedger(t, 2, noise=.1, seed=1)
        self.assertTrue(torch.equal(other.query(x[:2]), ys[[0, 2]]))

class MathematicalTests(unittest.TestCase):
    def test_ridge_formula_matches_an_independently_constructed_design_matrix(self):

        for rho in [0., .0001, .2, 1.]:
            x = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]], dtype=torch.float64)
            x *= torch.tensor([np.sqrt(2 - rho), np.sqrt(rho)], dtype=torch.float64)
            y = x @ torch.ones(2, dtype=torch.float64)
            for penalty in [1e-4, .01, .3]:
                w = torch.linalg.solve(x.T @ x / 4 + penalty * torch.eye(2, dtype=torch.float64), x.T @ y / 4)
                result = ridge_errors(rho, penalty)
                self.assertAlmostEqual(float((w - 1).square().sum()), result["analytic_test_mse"], places=12)
                self.assertAlmostEqual(float((x @ w - y).square().mean()), result["train_distribution_mse"], places=12)

    def test_stable_rank_known_spectra(self):
        self.assertAlmostEqual(stable_rank(torch.eye(5)), 5.)
        self.assertAlmostEqual(stable_rank(torch.ones(5, 3)), 1.)

    def test_response_disagreement_differs_from_point_uncertainty(self):
        class Linear(torch.nn.Module):
            def __init__(self, weight, bias):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([[weight]])); self.bias = bias
            def forward(self, x):
                return x @ self.weight + self.bias
        p = PairPool(torch.tensor([[0.], [1.]]), torch.tensor([[1.], [3.]]), torch.zeros(2).long())
        pair, point = audit([Linear(1., 0.), Linear(1., 10.)], p)
        self.assertTrue(torch.equal(pair, torch.zeros(2)))
        self.assertTrue((point > 0).all())
        pair, _ = audit([Linear(1., 0.), Linear(2., 0.)], p)
        self.assertAlmostEqual(float(pair[1] / pair[0]), 4.)

    def test_constant_scores_are_chance_and_perfect_scores_detect_error(self):
        error = torch.arange(100).float()
        self.assertEqual(score_quality(torch.zeros(100), error)["auroc_top_quartile_error"], .5)
        self.assertEqual(score_quality(error, error)["auroc_top_quartile_error"], 1.)

class TrainingTests(unittest.TestCase):
    def test_reproducibility_learning_and_id_checkpoint(self):
        x = sample_inputs(128, 4, 1., 0)
        y = .7 * x[:, :1] - .3 * x[:, 1:2]
        config = TrainConfig(steps=150, width=16, depth=1, eval_every=15, lr=.01)
        run = fit(x, y, x, y, config, 7, 9)
        again = fit(x, y, x, y, config, 7, 9)
        self.assertTrue(torch.equal(predict(run.model, x), predict(again.model, x)))
        self.assertLess(run.trace[-1]["train_mse"], run.trace[0]["train_mse"] / 10)
        expected = min(run.trace, key=lambda row: row["id_val_mse"])["step"]
        self.assertEqual(run.metadata["selected_step"], expected)

    def test_baselines_have_identical_starting_points_and_exact_budgets(self):
        config = {
            "dim": 4, "teacher_seeds": [0], "training_seeds": [0],
            "n_train": 32, "n_val": 32, "n_test": 64, "pool_pairs": 16, "audit_pairs": 16,
            "coverage": .02, "rounds": 1, "labels_per_round": 4, "decays": [0., .1],
            "methods": ["random_id", "random_pairs", "seed_pairs", "path_pairs", "oracle_pairs"],
            "train": {"steps": 5, "width": 8, "depth": 1, "eval_every": 5},
        }
        with tempfile.TemporaryDirectory() as folder:
            out = active(config, folder)
            rows = json.loads((out / "results.json").read_text())
            initial = [r for r in rows if r["round"] == 0]
            self.assertEqual(len(set(r["ood"]["mse"] for r in initial)), 1)
            final = [r for r in rows if r["round"] == 1]
            self.assertTrue(all(r["acquired_labels"] == 4 for r in final))
            self.assertEqual(len(set(r["compute"]["optimizer_steps"] for r in final)), 1)
            self.assertTrue(all(r["selected_decay"] == initial[0]["selected_decay"] for r in final))
            oracle = next(r for r in final if r["method"] == "oracle_pairs")
            self.assertEqual(oracle["oracle_scoring_labels"], 32)
            self.assertEqual(json.loads((out / "manifest.json").read_text())["status"], "complete")

    def test_selection_never_reuses_consumed_pairs(self):
        pairs = candidate_pairs(20, 4, 3, Support())
        available = torch.ones(20, dtype=torch.bool)
        available[:8] = False
        for method in ["random_pairs", "diversity", "seed_pairs", "path_points", "path_pairs"]:
            ids = select_pairs(method, pairs, available, 8, torch.arange(20).float(), 7, pairs.x)
            self.assertEqual(len(ids.unique()), 8)
            self.assertTrue((ids >= 8).all())

if __name__ == "__main__":
    unittest.main()
