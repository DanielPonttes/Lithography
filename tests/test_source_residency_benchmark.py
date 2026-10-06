import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import scripts.benchmark_source_residency as benchmark
from scripts.benchmark_source_residency import (
    compare_tree, execution_order_counts, planned_execution_order_counts,
    record_failure, summarize_ratios,
)


class SourceResidencyBenchmarkTests(unittest.TestCase):
    def test_continuous_values_use_declared_tolerance_but_hard_counts_are_exact(self):
        left = {
            "weights": [0.2, 0.8],
            "metrics": {"continuous_fidelity_mse": 0.12345670, "L2_pixels": 12.0},
        }
        within = {
            "weights": [0.20000001, 0.79999999],
            "metrics": {"continuous_fidelity_mse": 0.12345671, "L2_pixels": 12.0},
        }
        outside = {
            "weights": [0.2, 0.8],
            "metrics": {"continuous_fidelity_mse": 0.12345670, "L2_pixels": 13.0},
        }
        self.assertTrue(compare_tree(left, within)["ok"])
        failed = compare_tree(left, outside)
        self.assertFalse(failed["ok"])
        self.assertTrue(any("hard pixel count" in item["reason"] for item in failed["examples"]))

    def test_paired_speedup_summary_reports_median_and_range(self):
        runs = [
            {"seed": 17, "cpu": {"preparation_seconds": 4, "optimization_seconds": 8, "full_fit_wall_seconds": 15},
             "device": {"preparation_seconds": 2, "optimization_seconds": 4, "full_fit_wall_seconds": 10}},
            {"seed": 17, "cpu": {"preparation_seconds": 6, "optimization_seconds": 10, "full_fit_wall_seconds": 18},
             "device": {"preparation_seconds": 3, "optimization_seconds": 5, "full_fit_wall_seconds": 12}},
        ]
        result = summarize_ratios(runs)
        self.assertEqual(result["preparation_seconds"]["paired_median"], 2.0)
        self.assertEqual(result["optimization_seconds"]["paired_min"], 2.0)
        self.assertEqual(result["full_fit_wall_seconds"]["paired_max"], 1.5)
        self.assertEqual(result["full_fit_wall_seconds"]["by_seed"]["17"]["paired_count"], 2)

    def test_nonfinite_missing_none_and_boolean_integer_values_fail_parity(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                result = compare_tree({"metric": value}, {"metric": value})
                self.assertFalse(result["ok"])
                self.assertFalse(result["bitwise_equal"])
                self.assertIn("non-finite", result["examples"][0]["reason"])
                self.assertEqual(result["max_abs_difference"], 0.0)

        missing = compare_tree({"metric": 1.0}, {})
        self.assertFalse(missing["ok"])
        self.assertIn("missing_from_right", missing["examples"][0]["reason"])
        self.assertFalse(compare_tree({"metric": None}, {"metric": 0.0})["ok"])
        self.assertFalse(compare_tree({"metric": True}, {"metric": 1})["ok"])
        self.assertFalse(compare_tree({"metric": 1.0}, {"metric": 1.00000001})["bitwise_equal"])
        self.assertTrue(compare_tree({"metric": 1.0}, {"metric": 1.0})["bitwise_equal"])

    def test_nan_failure_artifact_preserves_failed_status_and_marks_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            cpu_payload = {"weights": [float("nan"), float("inf"), float("-inf")]}
            device_payload = {"weights": [0.0, 0.0, 0.0]}
            parity = compare_tree(cpu_payload, device_payload)
            self.assertFalse(parity["ok"])
            report = {"status": "running", "runs": [{
                "cpu_payload": cpu_payload, "parity": parity,
            }]}
            record_failure(path, report, AssertionError("CPU/device fit parity failed: non-finite values"))
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "failed")
        self.assertIn("CPU/device fit parity failed", saved["failure"]["message"])
        self.assertEqual(saved["runs"][0]["cpu_payload"]["weights"], [
            {"__nonfinite_float__": "NaN"},
            {"__nonfinite_float__": "+Infinity"},
            {"__nonfinite_float__": "-Infinity"},
        ])

    def test_missing_git_provenance_fails_closed_with_source_hashes_in_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            report = {"status": "running"}
            with mock.patch.object(benchmark, "git_provenance", side_effect=OSError("git unavailable")):
                with self.assertRaisesRegex(RuntimeError, "Git provenance is required"):
                    benchmark.initialize_provenance(path, report)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["provenance"]["git_status"], "unavailable")
        self.assertTrue(saved["provenance"]["source_sha256"])
        self.assertIn("git unavailable", saved["provenance"]["git_error"]["message"])

    def test_actual_default_order_schedule_reports_eight_and_seven(self):
        runs = []
        for seed_index in range(3):
            for repetition in range(5):
                cpu_first = (repetition + seed_index) % 2 == 0
                order = ("cpu", "device") if cpu_first else ("device", "cpu")
                runs.append({"execution_order": list(order)})
        self.assertEqual(planned_execution_order_counts((17, 29, 43), 5), {
            "cpu_first": 8, "device_first": 7, "run_count": 15,
        })
        self.assertEqual(execution_order_counts(runs), {
            "cpu_first": 8, "device_first": 7, "run_count": 15,
        })


if __name__ == "__main__":
    unittest.main()
