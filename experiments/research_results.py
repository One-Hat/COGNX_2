"""Metrics, atomic result checkpoints and strict resume validation."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    # Windows readers/indexers can briefly open the destination without delete
    # sharing. Retry the SAME atomic replacement; never delete the valid old file.
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.025 * 2 ** attempt)


def metrics(retention):
    r = np.asarray(retention, dtype=float)
    assert r.shape == (5, 5) and np.isfinite(r).all()
    assert ((r >= 0) & (r <= 1)).all()
    # Zero-based: F_j = max_{t=j,...,3} R[t,j] - R[4,j], j=0,...,3.
    # The final checkpoint and pre-learning checkpoints are excluded; negatives stay.
    forgetting = [float(r[j:4, j].max() - r[4, j]) for j in range(4)]
    return {"final_acc": float(r[4].mean() * 100),
            "forgetting": float(np.mean(forgetting) * 100),
            "forgetting_per_task": [f * 100 for f in forgetting],
            "seen_accuracy": [float(r[t, :t + 1].mean() * 100) for t in range(5)]}


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "min": float(values.min()), "max": float(values.max()), "n": len(values)}


def seal_result(result):
    result = {k: v for k, v in result.items() if k != "result_digest"}
    return {**result, "result_digest": digest(result)}


def valid_result(result, config_id, seed, method, stream_hashes):
    try:
        payload = {k: v for k, v in result.items() if k != "result_digest"}
        return (result["result_digest"] == digest(payload)
                and result["status"] == "complete" and result["config_id"] == config_id
                and result["seed"] == seed and result["method"] == method
                and result["stream_hashes"] == stream_hashes
                and result["metrics"] == metrics(result["retention_matrix"])
                and len(result["memory_by_task"]) == 5
                and all(item["state_equal"] for item in result["evaluation_checks"]))
    except (KeyError, TypeError, ValueError, AssertionError):
        return False


def aggregate(config, results):
    summary = {}
    for method in config["methods"]:
        runs = [r for r in results if r["method"] == method]
        if not runs:
            continue
        entry = {key: statistics([r["metrics"][key] for r in runs])
                 for key in ("final_acc", "forgetting")}
        entry["retention_mean"] = np.mean([r["retention_matrix"] for r in runs], axis=0).tolist()
        curves = np.array([r["metrics"]["seen_accuracy"] for r in runs])
        entry["seen_accuracy_mean"] = curves.mean(axis=0).tolist()
        entry["seen_accuracy_std"] = curves.std(axis=0, ddof=1).tolist() if len(runs) > 1 else None
        entry["memory"] = {}
        for key, value in runs[0]["memory_final"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                entry["memory"][key] = statistics([r["memory_final"][key] for r in runs])
        entry["wall_time_s"] = statistics([r["wall_time_s"] for r in runs])
        entry["inference_projected_uj_per_sample"] = statistics(
            [r["energy"]["inference_projected_uj_per_sample"] for r in runs])
        if method == "MNEMA":
            entry["faststore_peak_payload_bytes"] = statistics([r["faststore_monitor"]["max_actual_active_payload_bytes"] for r in runs])
            entry["faststore_over_budget_observations"] = sum(r["faststore_monitor"]["over_budget_observations"] for r in runs)
        summary[method] = entry
    expected = len(config["seeds"]) * len(config["methods"])
    return {"mode": config["mode"], "label": config["label"], "config_id": config["config_id"],
            "complete": len(results) == expected, "completed_runs": len(results),
            "expected_runs": expected, "research_complete": config["mode"] == "full"
            and len(config["seeds"]) == 10 and len(results) == expected, "methods": summary}
