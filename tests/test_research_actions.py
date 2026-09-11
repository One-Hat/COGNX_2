"""Exercise distributed artifact integrity with tiny in-memory data."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments import research_actions as actions
import numpy as np


class DistributedBenchmarkTests(unittest.TestCase):
    def test_shards_match_local_and_collection_rejects_bad_inputs(self):
        rng = np.random.default_rng(9)
        data = {"train_img": rng.integers(0, 256, (20, 784), dtype=np.uint8),
                "train_lbl": np.tile(np.arange(10, dtype=np.uint8), 2),
                "test_img": rng.integers(0, 256, (10, 784), dtype=np.uint8),
                "test_lbl": np.arange(10, dtype=np.uint8), "hashes": {"fixture": "tiny"}}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(actions, "load_dataset", return_value=data), \
                patch("experiments.validate_research_benchmark.load_dataset", return_value=data), \
                patch.dict(actions.os.environ, {"GITHUB_OUTPUT": ""}):
            base = Path(temporary)
            prepared, shards = base / "prepared", base / "shards"
            actions.prepare(prepared, "quick")
            config = actions.read_json(prepared / "config.json")
            with self.assertRaisesRegex(ValueError, "Missing results"):
                actions.collect(prepared, shards)
            with self.assertRaisesRegex(ValueError, "Seed is outside"):
                actions.run_shard(prepared, shards / "invalid", 9, "naive")
            for slug in actions.METHODS.values():
                actions.run_shard(prepared, shards / slug, 0, slug)

            # The distributed wrapper preserves the original runner's numerics.
            trains, tests = actions.indices(prepared, config, data, 0)
            local = actions.run_one(data, trains, tests, 0, "Naive MLP", config)
            remote = actions.read_json(shards / "naive/raw/seed_0_naive.json")
            ignored = {"wall_time_s", "result_digest"}
            self.assertEqual({k: v for k, v in local.items() if k not in ignored},
                             {k: v for k, v in remote.items() if k not in ignored})

            duplicate = shards / "duplicate/raw"
            duplicate.mkdir(parents=True)
            duplicate_file = duplicate / "seed_0_naive.json"
            shutil.copyfile(shards / "naive/raw/seed_0_naive.json", duplicate_file)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                actions.collect(prepared, shards)
            duplicate_file.unlink()

            result_path = shards / "naive/raw/seed_0_naive.json"
            original = result_path.read_text(encoding="utf-8")
            corrupt = json.loads(original)
            corrupt["seed"] = 1
            actions.atomic_json(result_path, corrupt)
            with self.assertRaisesRegex(ValueError, "Invalid result"):
                actions.collect(prepared, shards)
            result_path.write_text(original, encoding="utf-8")

            indices_path = prepared / "raw/seed_0_indices.json"
            original = indices_path.read_text(encoding="utf-8")
            corrupt = json.loads(original)
            corrupt["train"][0].reverse()
            actions.atomic_json(indices_path, corrupt)
            with self.assertRaisesRegex(ValueError, "indices differ"):
                actions.collect(prepared, shards)
            indices_path.write_text(original, encoding="utf-8")

            with patch.object(actions, "source_manifest", return_value={}):
                with self.assertRaisesRegex(ValueError, "Source differs"):
                    actions.collect(prepared, shards)
            actions.collect(prepared, shards)
            self.assertTrue(actions.read_json(prepared / "validation.json")["all_passed"])
            summary = actions.read_json(prepared / "summaries/summary.json")
            self.assertEqual(summary["completed_runs"], 6)
            self.assertFalse(summary["research_complete"])
            self.assertTrue((prepared / "BENCHMARK_REPORT.md").is_file())
            self.assertTrue((prepared / "plots/final_accuracy.png").is_file())


if __name__ == "__main__":
    unittest.main()
