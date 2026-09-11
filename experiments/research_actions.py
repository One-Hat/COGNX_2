"""Prepare, execute and collect independent GitHub Actions benchmark jobs."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Import runner first so its single-thread settings precede NumPy imports.
from experiments.run_research_benchmark import (
    METHODS, build_config, finish_run, parse_args, run_one, source_manifest,
)
import numpy as np
import matplotlib
import yaml
from benchmarks.research_split_mnist import load_dataset, task_indices, sequence_hash
from experiments.research_results import atomic_json, aggregate, digest, valid_result
from experiments.validate_research_benchmark import validate


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_inputs(directory):
    config = read_json(directory / "config.json")
    payload = {k: v for k, v in config.items()
               if k not in ("config_id", "created_utc", "git_commit")}
    if digest(payload) != config["config_id"]:
        raise ValueError("Configuration digest mismatch")
    if config["source_sha256"] != source_manifest():
        raise ValueError("Source differs from prepared benchmark")
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__}
    if any(config["software"][key] != value for key, value in versions.items()):
        raise ValueError("Software versions differ from prepared benchmark")
    data = load_dataset(ROOT / "data")
    if data["hashes"] != config["dataset_sha256"]:
        raise ValueError("Dataset differs from prepared benchmark")
    return config, data


def prepare(directory, mode):
    if directory.exists():
        raise ValueError("Prepare requires a fresh output directory")
    args = parse_args(["--mode", mode])
    data = load_dataset(ROOT / "data")
    config = build_config(args, data)
    atomic_json(directory / "config.json", config)
    for seed in config["seeds"]:
        trains, tests, _ = task_indices(data, seed, mode == "quick")
        atomic_json(directory / "raw" / f"seed_{seed}_indices.json", {
            "config_id": config["config_id"], "train": [i.tolist() for i in trains],
            "test": [i.tolist() for i in tests],
        })
    matrix = {"seed": config["seeds"], "method": list(METHODS.values())}
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            handle.write("matrix=" + json.dumps(matrix) + "\n")
    print(json.dumps(matrix), flush=True)


def indices(directory, config, data, seed):
    saved = read_json(directory / "raw" / f"seed_{seed}_indices.json")
    trains = [np.asarray(i, dtype=np.int64) for i in saved["train"]]
    tests = [np.asarray(i, dtype=np.int64) for i in saved["test"]]
    expected_train, expected_test, _ = task_indices(data, seed, config["mode"] == "quick")
    if (saved["config_id"] != config["config_id"]
            or [sequence_hash(i) for i in trains + tests]
            != [sequence_hash(i) for i in expected_train + expected_test]):
        raise ValueError(f"Prepared indices differ for seed {seed}")
    return trains, tests


def run_shard(directory, output, seed, slug):
    config, data = load_inputs(directory)
    if seed not in config["seeds"]:
        raise ValueError("Seed is outside the prepared experiment")
    method = next(name for name, value in METHODS.items() if value == slug)
    trains, tests = indices(directory, config, data, seed)
    result = run_one(data, trains, tests, seed, method, config)
    atomic_json(output / "raw" / f"seed_{seed}_{slug}.json", result)
    atomic_json(output / "workers" / f"seed_{seed}_{slug}.json", {
        "config_id": config["config_id"], "seed": seed, "method": method,
        "platform": platform.platform(), "processor": platform.processor(),
        "machine": platform.machine(), "software": config["software"],
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "runner_image": os.environ.get("ImageVersion"),
    })


def collect(directory, shards):
    config, data = load_inputs(directory)
    expected = {f"seed_{seed}_{slug}.json": (seed, method)
                for seed in config["seeds"] for method, slug in METHODS.items()}
    found = {}
    for path in shards.glob("*/raw/seed_*.json"):
        if path.name not in expected or path.name in found:
            raise ValueError(f"Unexpected or duplicate result: {path}")
        found[path.name] = path
    if set(found) != set(expected):
        raise ValueError(f"Missing results: {sorted(set(expected) - set(found))}")
    runs = []
    for name, (seed, method) in expected.items():
        trains, tests = indices(directory, config, data, seed)
        result = read_json(found[name])
        if (not valid_result(result, config["config_id"], seed, method,
                             [sequence_hash(i) for i in trains])
                or result["test_hashes"] != [sequence_hash(i) for i in tests]):
            raise ValueError(f"Invalid result: {name}")
        worker = read_json(found[name].parent.parent / "workers" / name)
        if (worker["config_id"], worker["seed"], worker["method"]) != (
                config["config_id"], seed, method):
            raise ValueError(f"Worker provenance mismatch: {name}")
        atomic_json(directory / "raw" / name, result)
        atomic_json(directory / "workers" / name, worker)
        runs.append(result)
    # Validate predictions and metrics before publishing a complete report/status.
    atomic_json(directory / "raw/per_seed_results.json", runs)
    atomic_json(directory / "summaries/summary.json", aggregate(config, runs))
    validate(directory)
    finish_run(directory, config, runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--mode", choices=("full", "quick"), default="full")
    run = sub.add_parser("run")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--method", choices=list(METHODS.values()), required=True)
    run.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("collect")
    merge.add_argument("--shards", type=Path, required=True)
    for command in (prep, run, merge):
        command.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory, args.mode)
    elif args.command == "run":
        run_shard(args.directory, args.output, args.seed, args.method)
    else:
        collect(args.directory, args.shards)


if __name__ == "__main__":
    main()
