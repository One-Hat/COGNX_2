"""Validate saved artifacts; --reproduce reruns only quick results, sequentially."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Runner sets BLAS threads before importing NumPy.
from experiments.run_research_benchmark import run_one, source_manifest
import numpy as np
from benchmarks.research_split_mnist import load_dataset, sequence_hash
from experiments.research_results import atomic_json, valid_result, metrics, aggregate


def validate(directory, reproduce=False):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if reproduce and config["mode"] != "quick":
        raise ValueError("Automatic reproducibility reruns are restricted to quick mode")
    assert config["source_sha256"] == source_manifest(), "Source changed since this run"
    data = load_dataset(ROOT / "data")
    assert data["hashes"] == config["dataset_sha256"]
    runs = json.loads((directory / "raw/per_seed_results.json").read_text(encoding="utf-8"))
    assert len(runs) == len(config["seeds"]) * len(config["methods"])
    validation = {"mode": config["mode"], "config_id": config["config_id"],
                  "full_run_executed_by_validator": False, "checks": [], "reproducibility": {}}
    expected_pairs = {(seed, method) for seed in config["seeds"] for method in config["methods"]}
    assert {(r["seed"], r["method"]) for r in runs} == expected_pairs
    for run in runs:
        indices = json.loads((directory / "raw" / f"seed_{run['seed']}_indices.json").read_text(encoding="utf-8"))
        trains = [np.array(x, dtype=np.int64) for x in indices["train"]]
        tests = [np.array(x, dtype=np.int64) for x in indices["test"]]
        hashes = [sequence_hash(i) for i in trains]
        assert valid_result(run, config["config_id"], run["seed"], run["method"], hashes)
        assert run["test_hashes"] == [sequence_hash(i) for i in tests]
        assert [len(i) for i in trains] == [s["train"] for s in config["task_sizes"]]
        assert [len(i) for i in tests] == [s["test"] for s in config["task_sizes"]]
        matrix = []
        for checkpoint_predictions in run["predictions"]:
            row = []
            for j, predictions in enumerate(checkpoint_predictions):
                assert len(predictions) == len(tests[j])
                assert np.isin(predictions, np.arange(10)).all()
                row.append(float(np.mean(np.asarray(predictions) == data["test_lbl"][tests[j]])))
            matrix.append(row)
        assert matrix == run["retention_matrix"]
        assert metrics(matrix) == run["metrics"]
        assert all(c["state_before"] == c["state_after"] for c in run["evaluation_checks"])
        if run["method"] == "MNEMA":
            assert all(c["reverse_order_checked"] for c in run["evaluation_checks"][-5:])
            for memory in run["memory_by_task"]:
                actual = memory["active_rows"] * memory["actual_row_bytes"]
                assert actual == memory["actual_active_payload_bytes"]
                assert memory["actual_exceeds_configured_budget"] == (actual > memory["configured_budget_bytes"])
        if run["method"] == "Replay-64KiB":
            assert all(m["buffer_bytes"] <= 65536 for m in run["memory_by_task"])
        validation["checks"].append({"seed": run["seed"], "method": run["method"], "saved_artifacts_valid": True})
        if reproduce:
            repeated = run_one(data, trains, tests, run["seed"], run["method"], config)
            # Wall time and its enclosing digest are intentionally non-deterministic.
            ignored = {"wall_time_s", "result_digest"}
            assert {k: v for k, v in repeated.items() if k not in ignored} == {k: v for k, v in run.items() if k not in ignored}
            validation["reproducibility"][f"{run['seed']}:{run['method']}"] = "exact match excluding wall time and result digest"
    saved_summary = json.loads((directory / "summaries/summary.json").read_text(encoding="utf-8"))
    assert saved_summary == aggregate(config, runs)
    validation["all_passed"] = True
    atomic_json(directory / "validation.json", validation)
    print(f"Validated {len(runs)} saved runs. Quick reproducibility repeated: {reproduce}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reproduce", action="store_true")
    args = parser.parse_args()
    validate(args.directory, args.reproduce)
