"""Numerical, isolation, accounting and protocol regressions (no full benchmark)."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from baselines.research_mlp import ResearchMLP, normalize
from baselines.research_replay import ReservoirBuffer, ReplayMLP
from baselines.ewc import EWC
from baselines.derpp import DERPP
from benchmarks.research_split_mnist import task_indices, sequence_hash
from experiments.research_memory import faststore_memory, memory_report
from experiments.research_state import CheckpointInference, evaluate, state_hash
from experiments.research_results import metrics, atomic_json, seal_result, valid_result
from experiments.run_research_benchmark import parse_args
from mnema.model import MNEMA
from mnema.store import FastStore


def ce(model, x, y):
    _, z = model.logits(x)
    z = z.astype(float)[0]
    maximum = z.max()
    return float(maximum + np.log(np.exp(z - maximum).sum()) - z[y])


def numerical_gradient(model, objective, name, index, epsilon=1e-3):
    parameter = getattr(model, name)
    original = float(parameter[index])
    parameter[index] = original + epsilon
    plus = objective()
    parameter[index] = original - epsilon
    minus = objective()
    parameter[index] = original
    return (plus - minus) / (2 * epsilon)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)

    def test_mlp_ce_gradient(self):
        model = ResearchMLP(n_in=4, n_hidden=3)
        x = np.array([0.3, 0.8, 0.1, 0.6], dtype=np.float32)
        gradient, _ = model.ce_gradients(x, [2])
        for name, index in (("W1", (1, 0)), ("b1", (0,)), ("W2", (0, 2)), ("b2", (2,))):
            finite = numerical_gradient(model, lambda: ce(model, x, 2), name, index)
            self.assertAlmostEqual(float(gradient[name][index]), finite, delta=0.001)

    def test_ewc_fisher_and_penalty(self):
        model = EWC(n_in=4, n_hidden=3, ewc_lambda=7)
        images = np.array([[40, 90, 20, 140], [90, 10, 200, 40]], dtype=np.uint8)
        labels = np.array([1, 4])
        expected = np.mean([numerical_gradient(model, lambda i=i: ce(model, normalize(images[i]), labels[i]),
                                              "b2", (1,)) ** 2 for i in range(2)])
        original = {k: p.copy() for k, p in model.parameters().items()}
        model.consolidate(images, labels, [0, 1])
        self.assertAlmostEqual(float(model.fishers[0]["b2"][1]), expected, delta=0.001)
        for name, parameter in model.parameters().items():
            np.testing.assert_array_equal(parameter, original[name])
            self.assertTrue((model.fishers[0][name] >= 0).all())
        model.b2[1] += 0.2
        objective = lambda: 7 / 2 * sum(float(np.sum(model.fishers[0][k] * (p - model.optima[0][k]) ** 2))
                                      for k, p in model.parameters().items())
        finite = numerical_gradient(model, objective, "b2", (1,))
        self.assertAlmostEqual(float(model.penalty_gradients()["b2"][1]), finite, delta=0.001)
        memory = memory_report(model)
        self.assertEqual(memory["fisher_bytes"], memory["model_adaptive_bytes"])
        self.assertEqual(memory["parameter_snapshot_bytes"], memory["model_adaptive_bytes"])

    def test_derpp_combined_loss_and_original_logits(self):
        model = DERPP(n_in=4, n_hidden=3, seed=9)
        first = np.array([30, 70, 10, 190], dtype=np.uint8)
        _, observed = model.logits(normalize(first))
        model.observe(first, 2)
        np.testing.assert_array_equal(model.buffer.rows[0]["logits"], observed[0])
        current = np.array([140, 20, 100, 80], dtype=np.uint8)
        past = model.buffer.rows[0]
        def objective():
            _, z = model.logits(normalize(past["image"]))
            return (ce(model, normalize(current), 4)
                    + model.alpha * float(np.mean((z - past["logits"]) ** 2))
                    + model.beta * ce(model, normalize(past["image"]), int(past["label"])))
        expected = numerical_gradient(model, objective, "b2", (2,))
        before = float(model.b2[2])
        model.observe(current, 4)
        self.assertAlmostEqual((before - float(model.b2[2])) / model.lr, expected, delta=0.002)
        self.assertEqual(model.replay_draws, 2)
        self.assertGreater(model.buffer.memory()["logit_bytes"], 0)

    def test_strict_buffer_budget_and_reservoir_coverage(self):
        buffer = ReservoirBuffer(budget_bytes=65536, seed=4)
        for i in range(1000):
            buffer.add(np.full(784, i % 256, dtype=np.uint8), i % 10)
            self.assertLessEqual(buffer.memory_bytes, 65536)
        self.assertEqual(buffer.size, 83)
        self.assertEqual(buffer.memory_bytes, 65171)
        self.assertEqual(buffer.rows["image"].dtype, np.uint8)
        # Across independent reservoirs early and late stream items should be retained.
        counts = np.zeros(30, dtype=int)
        for seed in range(400):
            small = ReservoirBuffer(capacity=5, n_in=1, seed=seed)
            for i in range(30):
                small.add(np.array([i], dtype=np.uint8), i % 10)
            counts[small.rows["image"][:, 0]] += 1
        self.assertGreater(counts.min(), 35)
        self.assertLess(counts.max(), 100)

    def test_shared_mlp_initialization(self):
        models = []
        for factory in (ResearchMLP, ReplayMLP, EWC, DERPP):
            np.random.seed(7)
            models.append(factory())
        for model in models[1:]:
            for name, parameter in models[0].parameters().items():
                np.testing.assert_array_equal(parameter, getattr(model, name))


class ProtocolTests(unittest.TestCase):
    def test_negative_forgetting_excludes_final_and_prelearning(self):
        matrix = np.full((5, 5), 0.99)
        for j in range(4):
            matrix[j:4, j] = 0.4
        matrix[4] = 0.6
        self.assertAlmostEqual(metrics(matrix)["forgetting"], -20)
        self.assertAlmostEqual(metrics(matrix)["final_acc"], 60)

    def test_full_stream_has_no_truncation(self):
        data = {"train_lbl": np.tile(np.arange(10), 6000), "test_lbl": np.tile(np.arange(10), 1000)}
        trains, tests, _ = task_indices(data, 3)
        self.assertEqual(sum(map(len, trains)), 60000)
        self.assertEqual(sum(map(len, tests)), 10000)
        np.testing.assert_array_equal(np.sort(np.concatenate(trains)), np.arange(60000))
        again, _, _ = task_indices(data, 3)
        self.assertEqual([sequence_hash(i) for i in trains], [sequence_hash(i) for i in again])
        self.assertNotEqual(sequence_hash(trains[0]), sequence_hash(task_indices(data, 4)[0][0]))
        self.assertNotIn("task_id", inspect.signature(evaluate).parameters)
        self.assertNotIn("labels", inspect.signature(evaluate).parameters)

    def test_mode_defaults(self):
        self.assertEqual(parse_args(["--quick"]).seeds, 1)
        self.assertEqual(parse_args(["--mode", "full"]).seeds, 10)

    def test_atomic_resume_validation(self):
        result = seal_result({"status": "complete", "config_id": "abc", "seed": 1, "method": "Naive MLP",
                              "stream_hashes": ["x"], "retention_matrix": np.eye(5).tolist(),
                              "metrics": metrics(np.eye(5)), "memory_by_task": [{}] * 5,
                              "evaluation_checks": [{"state_equal": True}]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            atomic_json(path, result)
            loaded = json.loads(path.read_text())
            self.assertTrue(valid_result(loaded, "abc", 1, "Naive MLP", ["x"]))
            self.assertFalse(valid_result(loaded, "different", 1, "Naive MLP", ["x"]))
            loaded["retention_matrix"][0][0] = 0
            self.assertFalse(valid_result(loaded, "abc", 1, "Naive MLP", ["x"]))

    def test_atomic_save_retries_temporary_windows_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            atomic_json(path, {"old": True})
            replace = os.replace
            attempts = []
            def transient_lock(source, destination):
                attempts.append(1)
                if len(attempts) == 1:
                    self.assertEqual(json.loads(path.read_text()), {"old": True})
                    raise PermissionError("Temporary Windows sharing violation")
                replace(source, destination)
            with patch("experiments.research_results.os.replace", side_effect=transient_lock):
                atomic_json(path, {"new": True})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(json.loads(path.read_text()), {"new": True})


class MNEMAIsolationTests(unittest.TestCase):
    def test_all_state_restored_and_order_independent(self):
        np.random.seed(8)
        model = MNEMA()
        images = np.random.default_rng(8).integers(0, 256, (5, 784), dtype=np.uint8)
        for x in images[:3]:
            model.step(normalize(x), y=2)
        before = state_hash(model)
        predictions, checks = evaluate(model, images, np.arange(5), True, reverse_check=True)
        self.assertEqual(before, state_hash(model))
        self.assertTrue(checks["state_equal"])
        self.assertTrue(checks["reverse_order_checked"])
        self.assertTrue(np.isin(predictions, np.arange(10)).all())
        self.assertLess(checks["snapshot_array_bytes"], 200000)
        live_step = model.step
        def failing_step(*args, **kwargs):
            live_step(*args, **kwargs)
            raise RuntimeError("Injected prediction failure")
        checkpoint = CheckpointInference(model)
        with patch.object(model, "step", side_effect=failing_step):
            with self.assertRaises(RuntimeError):
                checkpoint.predict(images[0])
        self.assertEqual(before, state_hash(model))

    def test_frozen_faststore_accounting_bug_is_flagged(self):
        store = FastStore()
        store.write(np.arange(2700), 0)
        report = faststore_memory(store)
        self.assertEqual(report["implementation_row_bytes"], 25)
        self.assertEqual(report["actual_row_bytes"], 28)
        self.assertLessEqual(report["implementation_occupancy_bytes"], 65536)
        self.assertGreater(report["actual_active_payload_bytes"], 65536)
        self.assertTrue(report["actual_exceeds_configured_budget"])


if __name__ == "__main__":
    unittest.main()
