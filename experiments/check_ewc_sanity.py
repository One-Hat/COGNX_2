"""Inspect the saved quick EWC configuration without tuning or changing results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Sets a single BLAS thread before NumPy is imported.
from experiments.run_research_benchmark import build_model, source_manifest
import numpy as np
from baselines.research_mlp import normalize
from benchmarks.research_split_mnist import load_dataset, sequence_hash
from experiments.research_results import atomic_json, metrics, valid_result
from experiments.research_state import evaluate


def array_hash(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def norm(arrays):
    return float(np.sqrt(sum(np.sum(value.astype(np.float64) ** 2) for value in arrays.values())))


def penalty_loss(model):
    # Independent scalar objective, accumulated in float64 for diagnostic precision.
    total = 0.0
    for fisher, optimum in zip(model.fishers, model.optima):
        for name, parameter in model.parameters().items():
            displacement = parameter.astype(np.float64) - optimum[name]
            total += float(np.sum(fisher[name] * displacement ** 2))
    return model.ewc_lambda * total / 2


def inspect_update(model, raw, label):
    """Verify gradients actually passed to SGD and the resulting fp32 parameters."""
    before = {k: v.copy() for k, v in model.parameters().items()}
    ce, _ = model.ce_gradients(normalize(raw), [label])
    penalty = model.penalty_gradients()
    expected_gradient = {k: ce[k] + penalty[k] for k in ce}
    loss_before = penalty_loss(model)
    original_apply = model.apply_gradients
    calls = []

    def checked_apply(actual):
        for name in actual:
            # A penalty omitted from observe() fails this check once it is nonzero.
            np.testing.assert_array_equal(actual[name], expected_gradient[name])
        calls.append(True)
        original_apply(actual)

    with patch.object(model, "apply_gradients", side_effect=checked_apply):
        model.observe(raw, label)
    assert len(calls) == 1
    changed_vs_ce_only = 0
    for name, parameter in model.parameters().items():
        expected = before[name] - model.lr * expected_gradient[name]
        np.testing.assert_array_equal(parameter, expected)
        ce_only = before[name] - model.lr * ce[name]
        changed_vs_ce_only += int(np.count_nonzero(parameter != ce_only))
    ce_norm, penalty_norm = norm(ce), norm(penalty)
    assert np.isfinite([loss_before, ce_norm, penalty_norm]).all()
    return {"penalty_loss_before": loss_before, "ce_gradient_l2": ce_norm,
            "penalty_gradient_l2": penalty_norm,
            "penalty_to_ce_gradient_ratio": penalty_norm / ce_norm if ce_norm else None,
            "parameters_different_from_ce_only_update": changed_vs_ce_only,
            "actual_gradient_matches_ce_plus_penalty": True,
            "actual_parameters_match_expected_update": True}


def fisher_statistics(fisher):
    result = {}
    for name, value in fisher.items():
        assert np.isfinite(value).all() and (value >= 0).all()
        result[name] = {"elements": value.size, "nonzero": int(np.count_nonzero(value)),
                        "mean": float(value.mean(dtype=np.float64)),
                        "min": float(value.min()), "max": float(value.max())}
    assert sum(item["nonzero"] for item in result.values()) > 0
    return result


def run_sanity(directory, seed=0):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config["mode"] != "quick":
        raise ValueError("This diagnostic only runs quick-mode data")
    assert config["source_sha256"] == source_manifest(), "Benchmark source changed since saved run"
    existing = json.loads((directory / "raw" / f"seed_{seed}_ewc.json").read_text(encoding="utf-8"))
    indices = json.loads((directory / "raw" / f"seed_{seed}_indices.json").read_text(encoding="utf-8"))
    trains = [np.array(x, dtype=np.int64) for x in indices["train"]]
    tests = [np.array(x, dtype=np.int64) for x in indices["test"]]
    assert valid_result(existing, config["config_id"], seed, "EWC", [sequence_hash(x) for x in trains])
    assert max(map(len, trains)) <= 500 and max(map(len, tests)) <= 200
    data = load_dataset(ROOT / "data")
    assert data["hashes"] == config["dataset_sha256"]
    model = build_model("EWC", seed, config)
    assert model.ewc_lambda == config["hyperparameters"]["ewc_lambda"]
    assert np.isfinite(model.ewc_lambda) and model.ewc_lambda > 0
    snapshots, fisher_hashes = [], []
    steps, tasks, predictions, retention = [], [], [], []
    for task, order in enumerate(trains):
        print(f"EWC diagnostic: task {task + 1}/5, {len(order)} samples, lambda={model.ewc_lambda}", flush=True)
        for i, index in enumerate(order):
            step = inspect_update(model, data["train_img"][index], int(data["train_lbl"][index]))
            step.update(task=task + 1, sample=i + 1)
            steps.append(step)
        # Earlier snapshots and Fishers must survive future training unchanged.
        assert [array_hash(x) for x in model.optima] == snapshots
        assert [array_hash(x) for x in model.fishers] == fisher_hashes
        record = {"task": task + 1, "old_snapshots_and_fishers_unchanged": True}
        if task < 4:
            parameters_before = array_hash(model.parameters())
            limit = config["hyperparameters"]["fisher_samples"]
            selected = order if limit == 0 else order[:limit]
            model.consolidate(data["train_img"], data["train_lbl"], selected)
            assert array_hash(model.parameters()) == parameters_before
            assert array_hash(model.optima[-1]) == parameters_before
            assert all(not np.shares_memory(value, model.optima[-1][name])
                       for name, value in model.parameters().items())
            assert all(not np.shares_memory(value, model.fishers[-1][name])
                       for name, value in model.parameters().items())
            assert all(not np.shares_memory(value, old[name])
                       for name, value in model.optima[-1].items() for old in model.optima[:-1])
            snapshots.append(array_hash(model.optima[-1]))
            fisher_hashes.append(array_hash(model.fishers[-1]))
            record.update(fisher=fisher_statistics(model.fishers[-1]), fisher_samples=len(selected),
                          snapshot_equals_task_end_parameters=True, snapshot_is_independent_copy=True,
                          fisher_does_not_update_parameters=True)
            total_fisher = {name: sum(f[name] for f in model.fishers) for name in model.param_names}
            record["max_lr_lambda_accumulated_fisher"] = float(
                model.lr * model.ewc_lambda * max(f.max() for f in total_fisher.values()))
            record["penalty_loss_after_consolidation"] = penalty_loss(model)
        tasks.append(record)
        predicted_tasks, row = [], []
        for evaluation_indices in tests:
            predicted, _ = evaluate(model, data["test_img"], evaluation_indices)
            predicted_tasks.append(predicted.tolist())
            row.append(float(np.mean(predicted == data["test_lbl"][evaluation_indices])))
        predictions.append(predicted_tasks)
        retention.append(row)
    assert snapshots == [array_hash(x) for x in model.optima]
    assert fisher_hashes == [array_hash(x) for x in model.fishers]
    assert predictions == existing["predictions"]
    assert retention == existing["retention_matrix"]
    assert metrics(retention) == existing["metrics"]
    # At the first old-task optimum, zero penalty is mathematically correct.
    assert tasks[0]["penalty_loss_after_consolidation"] == 0
    task2 = [s for s in steps if s["task"] == 2]
    assert task2[0]["penalty_gradient_l2"] == 0
    assert task2[1]["penalty_loss_before"] > 0 and task2[1]["penalty_gradient_l2"] > 0
    assert any(s["parameters_different_from_ce_only_update"] > 0 for s in task2)
    ratios = [s["penalty_to_ce_gradient_ratio"] for s in steps if s["task"] > 1
              and s["penalty_to_ce_gradient_ratio"] is not None]
    output = {"mode": "quick/debug", "seed": seed, "config_id": config["config_id"],
              "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "ewc_lambda": model.ewc_lambda, "learning_rate": model.lr,
              "lambda_selection": "Unchanged saved configuration; no sweep or tuning",
              "all_checks_passed": True, "original_predictions_and_metrics_reproduced": True,
              "metrics": metrics(retention), "tasks": tasks, "steps": steps,
              "penalty_to_ce_gradient_ratio": {"median": float(np.median(ratios)),
                                                "mean": float(np.mean(ratios)), "max": float(np.max(ratios))},
              "steps_with_update_different_from_ce_only": sum(s["parameters_different_from_ce_only_update"] > 0 for s in steps),
              "snapshots_retained": len(snapshots), "snapshot_hashes": snapshots, "fisher_hashes": fisher_hashes,
              "scope": "Functional checks, not proof that this lambda or empirical Fisher is optimal. No full experiment executed."}
    atomic_json(directory / "ewc_sanity.json", output)
    first_fisher = tasks[0]["fisher"]
    nonzero = sum(s["nonzero"] for s in first_fisher.values())
    total = sum(s["elements"] for s in first_fisher.values())
    report = f"""# EWC sanity check — DEBUG ONLY

All functional checks passed using the existing quick seed {seed}, unchanged lambda={model.ewc_lambda} and learning rate={model.lr}. No tuning or full benchmark was performed. The original final accuracy **{output['metrics']['final_acc']:.2f}%** and all checkpoint predictions were reproduced exactly; the existing benchmark results were not replaced.

| Check | Evidence |
|---|---|
| Nonzero Fisher | Task 1: {nonzero:,}/{total:,} entries nonzero; all tasks finite and nonnegative. Per-layer statistics are in the JSON. |
| Penalty activates | Zero at the exact Task 1 optimum, correctly; before Task 2 sample 2 the loss is {task2[1]['penalty_loss_before']:.8g} and gradient norm is {task2[1]['penalty_gradient_l2']:.8g}. |
| Correct snapshots | Each snapshot equals the task-end parameters, shares no parameter memory, and stays unchanged through subsequent training. Four snapshots are retained. |
| Lambda is used | Actual model lambda={model.ewc_lambda}, matching saved configuration; finite and positive. No candidate values were tested. |
| Penalty reaches SGD | All {len(steps)} actual gradient/update checks passed; {output['steps_with_update_different_from_ce_only']} updates differ from a CE-only update at the same parameters in actual fp32 storage. |
| Relative strength | Across tasks 2–5, median penalty/CE gradient norm = {output['penalty_to_ce_gradient_ratio']['median']:.6g}; maximum = {output['penalty_to_ce_gradient_ratio']['max']:.6g}. This is a global norm, not a per-parameter strength estimate. |

Lambda alone has no universal useful scale: its effect also depends on Fisher magnitude, loss normalization and parameter displacement. The effective diagonal factor lr × lambda × accumulated Fisher is recorded per task, and no non-finite gradients or parameters occurred. These checks establish that regularization is active and correctly applied, not that lambda=100 is optimal or that EWC should perform well on this Class-IL task. Keep the bad quick result.

Reproduce: `python experiments/check_ewc_sanity.py results/research_benchmark/quick`.
Detailed evidence: [ewc_sanity.json](ewc_sanity.json).
"""
    (directory / "EWC_SANITY_CHECK.md").write_text(report, encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("all_checks_passed", "metrics", "penalty_to_ce_gradient_ratio", "steps_with_update_different_from_ce_only")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "results/research_benchmark/quick")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_sanity(args.directory, args.seed)
