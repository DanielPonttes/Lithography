import unittest
import copy
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import source_event_search as events
from scripts import diagnose_source_events as diagnostic
import source_coverage as coverage


def one_hot(index):
    weights = np.zeros(events.SOURCE_COUNT, dtype=np.float64)
    weights[index] = 1.0
    return weights


class ThresholdEventTests(unittest.TestCase):
    def test_endpoint_rounding_keeps_exact_vectors(self):
        start, end = one_hot(0), one_hot(1)
        result = events.interpolate_weights(start, end, 0.0)
        self.assertTrue(np.array_equal(result, start))
        self.assertTrue(np.array_equal(events.interpolate_weights(start, end, 1.0), end))

    def test_constant_pair_has_only_endpoints_and_midpoint(self):
        result = events.threshold_event_alphas(
            np.full((2, 3), 0.225), np.full((2, 3), 0.225), doses=(1.0,), threshold=0.225,
        )
        self.assertEqual(result["knots"], [0.0, 1.0])
        self.assertEqual(result["midpoints"], [0.5])
        self.assertEqual(result["proposal_alphas"], [0.0, 0.5, 1.0])
        self.assertEqual(result["isolated_crossing_count"], 0)
        self.assertEqual(result["constant_pixels_on_threshold_count"], 6)

    def test_duplicate_crossing_events_are_deduplicated_deterministically(self):
        result = events.threshold_event_alphas(
            np.array([0.0, 0.0, 0.3]), np.array([0.45, 0.9, 0.3]),
            doses=(1.0,), threshold=0.225,
        )
        self.assertEqual(result["knots"], [0.0, 0.25, 0.5, 1.0])
        self.assertEqual(result["proposal_alphas"], [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0])
        self.assertEqual(result["isolated_crossing_count"], 2)

    def test_segment_generation_preserves_reference_and_endpoint(self):
        start, end = one_hot(0), one_hot(1)
        start_intensity = {"fit": np.array([0.0, 0.3])}
        end_intensity = {"fit": np.array([0.45, 0.3])}
        candidates, audit = events.segment_event_candidates(
            start, end, start_intensity, end_intensity, segment_id="synthetic", doses=(1.0,),
        )
        self.assertTrue(np.array_equal(candidates[0]["weights"], start))
        endpoint = next(row for row in candidates if row["alpha"] == 1.0)
        self.assertTrue(np.array_equal(endpoint["weights"], end))
        self.assertEqual(audit["candidate_count"], len(candidates))
        self.assertEqual(candidates[0]["candidate_id"], "synthetic:000000")

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            events.validate_weights(np.full(events.SOURCE_COUNT, np.nan))
        with self.assertRaises(ValueError):
            events.validate_weights(np.zeros(events.SOURCE_COUNT))
        with self.assertRaises(ValueError):
            events.threshold_event_alphas([0.0], [np.inf])
        with self.assertRaises(ValueError):
            events.threshold_event_alphas([0.0], [1.0], doses=(0.0,))
        negative = one_hot(0)
        negative[1] = -1e-12
        negative[0] += 1e-12
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            events.validate_weights(negative)
        with self.assertRaisesRegex(ValueError, "source_count"):
            events.validate_weights(one_hot(0), source_count=49.5)
        with self.assertRaisesRegex(ValueError, "alpha_tolerance"):
            events.threshold_event_alphas([0.0], [1.0], alpha_tolerance=0.5)

    def test_identity_round_trips_file_and_named_vector_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.bin").write_bytes(b"frozen")
            vector = one_hot(7)
            identity = events.canonical_identity(
                code_paths=(), input_paths=(root / "input.bin",),
                named_vectors={"best": vector},
            )
            events.validate_identity(identity, base_dir=root, named_vectors={"best": vector})
            changed = one_hot(8)
            with self.assertRaisesRegex(ValueError, "source vector mismatch"):
                events.validate_identity(identity, base_dir=root, named_vectors={"best": changed})

    def test_wall_clock_budget_does_not_starve_protected_incumbents(self):
        reference, initial, best, event = (one_hot(i) for i in range(4))
        candidates = events.incumbent_candidates(
            reference_weights=reference, initial_incumbent_weights=initial,
            best_known_incumbent_weights=best,
            segment_candidates=[{"candidate_id": "event", "weights": event,
                                 "candidate_order": 0, "roles": ["threshold_event"]}],
        )
        basis = {key: np.ones((events.SOURCE_COUNT, 1, 1), dtype=np.float32)
                 for key in ("original", "new")}
        targets = {key: np.ones((1, 1), dtype=bool) for key in basis}
        real_hard_metrics = coverage.hard_metrics

        def slow_hard_metrics(*args, **kwargs):
            import time
            time.sleep(0.01)
            return real_hard_metrics(*args, **kwargs)

        with mock.patch.object(coverage, "hard_metrics", side_effect=slow_hard_metrics):
            report = events.score_candidate_set(
                candidates, basis, targets, original_layout_ids=("original",),
                new_layout_ids=("new",), wall_clock_budget_seconds=0.001,
            )
        self.assertEqual(report["attempted_candidates"], 3)
        self.assertTrue(all(row["status"] == "attempted" for row in report["candidate_records"][:3]))
        self.assertEqual(report["candidate_records"][3]["reason"], "wall_clock_budget")
        self.assertGreater(report["protected_incumbent_elapsed_seconds"], 0)

    def test_fixed_weights_adapter_preserves_all_49_values_including_zeros(self):
        weights = one_hot(16)
        source = diagnostic.FixedWeightsSource(weights)
        coordinates, observed = source.distribution()
        self.assertEqual(coordinates.shape[0], 49)
        self.assertEqual(observed.numel(), 49)
        self.assertTrue(torch.equal(observed.cpu(), torch.as_tensor(weights, dtype=torch.float32)))
        self.assertEqual(int((observed == 0).sum()), 48)
        self.assertTrue(callable(diagnostic.constrained.sha256_tensor))

    def test_existing_direct_verify_output_refuses_before_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "already-there.json"
            output.write_text("preserve", encoding="utf-8")
            with mock.patch.object(diagnostic, "_preflight") as preflight:
                with self.assertRaises(FileExistsError):
                    diagnostic.direct_verify(Path("plan"), "a" * 64, "b" * 64,
                                             Path("weights"), output, "cuda")
                preflight.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

    def test_exclusive_result_writer_never_clobbers_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                diagnostic._write_new_json(output, {"replacement": True})
            self.assertEqual(output.read_text(encoding="utf-8"), "original")

    def test_cli_returns_three_when_parity_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mock-report.json"

            def fake_direct_verify(*args, **kwargs):
                output.write_text(json.dumps({
                    "metrics": {
                        "direct_vs_weighted_basis": {"passed": False},
                        "candidate": {"new_fit_mean": {"band_pixels": 1}},
                        "reference": {"new_fit_mean": {"band_pixels": 2}},
                    },
                    "historical_report_hard_metric_parity": {"verified": False},
                    "verification": {"direct_source_gain_confirmed": False},
                    "source_report_provenance": {"verified": False},
                    "new_fit_candidate_minus_reference_mean": {"band_pixels": -1},
                }), encoding="utf-8")
                return output

            argv = ["--mode", "direct-verify", "--coverage-plan-file", "plan.json",
                    "--expected-coverage-plan-sha256", "a" * 64,
                    "--expected-previous-sha256", "b" * 64,
                    "--weights-file", "weights.json", "--output-file", str(output)]
            with mock.patch.object(diagnostic, "direct_verify", side_effect=fake_direct_verify):
                self.assertEqual(diagnostic.main(argv), 3)

    def test_aggregate_gates_reject_any_old_regression_new_tradeoff_or_blank(self):
        reference = {
            "original_fit_mean": {"band_pixels": 20.0, "L2_pixels": 0.0,
                                  "L2_worst_dose_pixels": 40.0},
            "new_fit_mean": {"band_pixels": 12.0, "L2_pixels": 3.0,
                             "L2_worst_dose_pixels": 8.0},
            "no_blank_positive_target_any_dose": True,
        }
        candidate = copy.deepcopy(reference)
        candidate["new_fit_mean"]["band_pixels"] = 11.0
        gates = diagnostic._aggregate_hard_gates(candidate, reference)
        self.assertTrue(all(gates.values()))

        mutations = (
            ("original_fit_mean", "band_pixels", 21.0),
            ("original_fit_mean", "L2_pixels", 1.0),
            ("original_fit_mean", "L2_worst_dose_pixels", 41.0),
            ("new_fit_mean", "band_pixels", 12.0),
            ("new_fit_mean", "L2_pixels", 4.0),
            ("new_fit_mean", "L2_worst_dose_pixels", 9.0),
        )
        for group, key, value in mutations:
            with self.subTest(group=group, key=key):
                regressed = copy.deepcopy(candidate)
                regressed[group][key] = value
                self.assertFalse(all(diagnostic._aggregate_hard_gates(
                    regressed, reference).values()))
        blank = copy.deepcopy(candidate)
        blank["no_blank_positive_target_any_dose"] = False
        self.assertFalse(all(diagnostic._aggregate_hard_gates(blank, reference).values()))

    def test_gain_confirmation_requires_historical_parity_when_supplied(self):
        gates = {"old_and_new_hard_gates": True}
        self.assertTrue(diagnostic._direct_gain_confirmed(True, gates, None))
        self.assertTrue(diagnostic._direct_gain_confirmed(True, gates, {"verified": True}))
        self.assertFalse(diagnostic._direct_gain_confirmed(True, gates, {"verified": False}))
        self.assertFalse(diagnostic._direct_gain_confirmed(False, gates, {"verified": True}))

    def test_cli_returns_three_on_historical_mismatch_even_when_routes_match(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "historical-mismatch.json"

            def fake_direct_verify(*args, **kwargs):
                output.write_text(json.dumps({
                    "metrics": {
                        "direct_vs_weighted_basis": {"passed": True},
                        "candidate": {"new_fit_mean": {"band_pixels": 1}},
                        "reference": {"new_fit_mean": {"band_pixels": 2}},
                    },
                    "historical_report_hard_metric_parity": {"verified": False},
                    "verification": {"direct_source_gain_confirmed": False},
                    "source_report_provenance": {"verified": True},
                    "new_fit_candidate_minus_reference_mean": {"band_pixels": -1},
                }), encoding="utf-8")
                return output

            argv = ["--mode", "direct-verify", "--coverage-plan-file", "plan.json",
                    "--expected-coverage-plan-sha256", "a" * 64,
                    "--expected-previous-sha256", "b" * 64,
                    "--weights-file", "weights.json", "--output-file", str(output)]
            with mock.patch.object(diagnostic, "direct_verify", side_effect=fake_direct_verify):
                self.assertEqual(diagnostic.main(argv), 3)

    def test_gpu_hard_metric_helper_matches_canonical_cpu_counts(self):
        aerial = torch.tensor([[0.10, 0.20], [0.25, 0.30]], dtype=torch.float32)
        target = np.array([[False, True], [True, True]], dtype=bool)
        from_gpu_route = diagnostic._hard_metrics_from_aerial(
            aerial, target, events.THRESHOLD, events.STEEPNESS,
        )
        from_cpu_route = coverage.hard_metrics(
            aerial.numpy()[None, ...], target, np.ones(1, dtype=np.float64),
            threshold=events.THRESHOLD, steepness=events.STEEPNESS,
        )
        self.assertEqual(from_gpu_route, from_cpu_route)

    def test_incumbents_survive_duplicate_event_vectors(self):
        reference, initial, best = one_hot(0), one_hot(1), one_hot(2)
        event_rows = [{"candidate_id": "same-as-best", "weights": best,
                       "proposal_role": "threshold_event", "alphas": [0.4],
                       "weights_f64_sha256": events.array_sha256(best, "<f8"),
                       "weights_f32_sha256": events.array_sha256(best, "<f4")}]
        candidates = events.incumbent_candidates(
            reference_weights=reference, initial_incumbent_weights=initial,
            best_known_incumbent_weights=best, segment_candidates=event_rows,
        )
        self.assertEqual([row["candidate_id"] for row in candidates[:3]],
                         ["reference", "initial_incumbent", "best_known_incumbent"])
        self.assertEqual(len(candidates), 3)
        self.assertIn("same-as-best", candidates[2]["aliases"])
        self.assertIn("best_known_incumbent", candidates[2]["roles"])
        self.assertIn("threshold_event", candidates[2]["roles"])

    def test_ranking_uses_only_actual_hard_qualified_records(self):
        def row(candidate_id, order, band, status="attempted", qualified=True):
            return {"candidate_id": candidate_id, "candidate_order": order,
                    "status": status, "hard_qualified": qualified,
                    "actual_hard_metrics": {
                        "new_fit_mean": {"band_pixels": band, "L2_worst_dose_pixels": 10, "L2_pixels": 5},
                        "original_fit_mean": {"band_pixels": 20, "L2_worst_dose_pixels": 11, "L2_pixels": 6},
                    }}
        ranked = events.rank_qualified_candidates([
            row("unattempted", 0, 1, status="not_attempted"),
            row("failed", 1, 0, qualified=False),
            row("worse", 2, 4), row("best", 3, 3),
        ])
        self.assertEqual([item["candidate_id"] for item in ranked], ["best", "worse"])

    def test_candidate_budget_audits_protected_incumbents_first(self):
        reference, initial, best, event = (one_hot(i) for i in range(4))
        candidates = events.incumbent_candidates(
            reference_weights=reference, initial_incumbent_weights=initial,
            best_known_incumbent_weights=best,
            segment_candidates=[{"candidate_id": "event", "weights": event,
                                 "candidate_order": 0, "roles": ["threshold_event"],
                                 "weights_f64_sha256": events.array_sha256(event, "<f8"),
                                 "weights_f32_sha256": events.array_sha256(event, "<f4")}],
        )
        basis = {"original": np.ones((events.SOURCE_COUNT, 1, 1), dtype=np.float32),
                 "new": np.ones((events.SOURCE_COUNT, 1, 1), dtype=np.float32)}
        targets = {key: np.ones((1, 1), dtype=bool) for key in basis}
        report = events.score_candidate_set(
            candidates, basis, targets, original_layout_ids=("original",),
            new_layout_ids=("new",), max_candidates=3,
        )
        self.assertEqual(report["status"], "incomplete_budget")
        self.assertEqual(report["attempted_candidates"], 3)
        self.assertEqual(report["candidate_records"][0]["roles"], ["reference"])
        self.assertEqual(report["candidate_records"][1]["roles"], ["initial_incumbent"])
        self.assertIn("best_known_incumbent", report["candidate_records"][2]["roles"])
        self.assertEqual(report["candidate_records"][3]["status"], "not_attempted")
        self.assertEqual(report["ranked_qualified_candidate_ids"], [])

    def test_fair_protocol_pins_shared_budget_and_scope(self):
        protocol = events.fair_quality_runtime_protocol(
            workload_sha256="a" * 64, candidate_evaluation_budget=5140,
            wall_clock_budget_seconds=600,
        )
        self.assertEqual(protocol["candidate_evaluation_budget_per_method"], 5140)
        self.assertTrue(protocol["quality_comparison"]["same_frozen_fit_masks_targets_basis_and_source_grid"])
        self.assertEqual(protocol["runtime_comparison"]["cross_method_score_cache"], "disabled")


if __name__ == "__main__":
    unittest.main()
