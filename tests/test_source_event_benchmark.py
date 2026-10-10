"""Synthetic-only tests for the prospective source-event experiment harness."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import source_event_search as event_search
import source_coverage as coverage
from scripts import benchmark_source_events as benchmark


def one_hot(index):
    weights = np.zeros(event_search.SOURCE_COUNT, dtype=np.float64)
    weights[index] = 1.0
    return weights


def vector_record(weights):
    return {"weights": weights.tolist(), "source_count": event_search.SOURCE_COUNT,
            "weights_f64_sha256": event_search.array_sha256(weights, "<f8"),
            "weights_f32_sha256": event_search.array_sha256(weights, "<f4")}


def valid_event_plan():
    vectors = {
        "reference": vector_record(one_hot(0)),
        "lp_anchor": vector_record(one_hot(1)),
        "best_known_slot_4755_seed_101": vector_record(one_hot(2)),
        "seed_endpoints": {},
    }
    for seed_index, seed in enumerate(benchmark.SEED_ORDER):
        vectors["seed_endpoints"][str(seed)] = {
            name: vector_record(one_hot((seed_index * len(benchmark.SEGMENT_NAMES)
                                        + segment_index + 3) % event_search.SOURCE_COUNT))
            for segment_index, name in enumerate(benchmark.SEGMENT_NAMES)
        }
    return {
        "schema_version": benchmark.SCHEMA_VERSION,
        "objective_id": benchmark.OBJECTIVE_ID,
        "status": "frozen_prospective_quality_time_plan",
        "fixed_fit_layout_hashes": coverage.PINNED_FIXED_FIT_LAYOUT_HASHES,
        "lineage_artifact_count": 14,
        "protocol": {
            "candidate_cap_per_arm_seed_including_protected": benchmark.ATTEMPT_CAP_PER_ARM_SEED,
            "wall_clock_budget_seconds_per_arm_seed": benchmark.WALL_BUDGET_SECONDS,
            "event_candidates": "not an exhaustive threshold partition",
            "thread_count_stability": "captured original host thread count",
        },
        "code_sha256": {}, "input_sha256": {}, "source_vectors": vectors,
    }


class EventBenchmarkContractTests(unittest.TestCase):
    def test_event_audit_compacts_large_knot_lists_without_losing_counts_or_hash(self):
        plan = valid_event_plan()
        basis64 = {layout_id: np.zeros((event_search.SOURCE_COUNT, 1, 1), dtype=np.float64)
                   for layout_id in coverage.ORIGINAL_FIT_LAYOUT_IDS + coverage.LAYOUT_IDS}
        rows = [{"candidate_id": "proposal-%d" % index, "weights": one_hot(0)}
                for index in range(258)]
        full_audit = {
            "segment_id": "synthetic",
            "unique_knot_count": 3,
            "proposal_count_before_weight_dedup": 4,
            "candidate_count": 4,
            "deduplicated_proposal_count": 0,
            "candidate_generation": "synthetic",
            "layout_events": [{
                "layout_id": "layout-a", "knots": [0.0, 0.5, 1.0],
                "proposal_alphas": [0.0, 0.25, 0.5, 1.0],
                "isolated_crossing_count": 1,
                "constant_pixels_on_threshold_count": 0,
            }],
        }
        with mock.patch.object(event_search, "segment_event_candidates",
                               side_effect=lambda *_args, **_kwargs:
                               (rows, copy.deepcopy(full_audit))):
            candidates, audits = benchmark._segment_events(17, plan, basis64)
        self.assertEqual(len(candidates), 4 * 257)
        self.assertTrue(all(audit["truncated"] for audit in audits))
        self.assertTrue(all("layout_events" not in audit for audit in audits))
        layout_summary = audits[0]["layout_event_summaries"][0]
        self.assertNotIn("knots", layout_summary)
        self.assertEqual(layout_summary["unique_knot_count"], 3)
        self.assertEqual(layout_summary["proposal_alpha_count"], 4)
        self.assertEqual(len(layout_summary["knot_and_proposal_alphas_sha256"]), 64)

    def test_grid_contains_all_four_frozen_257_point_segments(self):
        plan = {"source_vectors": {"reference": vector_record(one_hot(0)),
                                   "seed_endpoints": valid_event_plan()["source_vectors"]["seed_endpoints"]}}
        rows = benchmark._grid_candidates(17, plan)
        self.assertEqual(len(rows), benchmark.GRID_PROPOSALS_PER_SEED)
        self.assertEqual(rows[0]["alpha"], 0.0)
        self.assertEqual(rows[benchmark.GRID_DENOMINATOR]["alpha"], 1.0)
        self.assertTrue(np.array_equal(rows[0]["weights"], one_hot(0)))
        self.assertTrue(np.array_equal(
            rows[benchmark.GRID_DENOMINATOR]["weights"],
            np.asarray(plan["source_vectors"]["seed_endpoints"]["17"][
                benchmark.SEGMENT_NAMES[0]]["weights"], dtype=np.float64),
        ))

    def test_frozen_event_plan_validates_named_vector_hashes_and_rejects_mutation(self):
        plan = valid_event_plan()
        benchmark._validate_event_plan(plan)
        plan["source_vectors"]["seed_endpoints"]["43"][benchmark.SEGMENT_NAMES[0]][
            "weights_f64_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "endpoint identity mismatch"):
            benchmark._validate_event_plan(plan)

    def test_pareto_hard_pv_improvement_does_not_hide_a_metric_tradeoff(self):
        baseline = {"new_fit_mean": {"band_pixels": 10, "L2_worst_dose_pixels": 3,
                                     "L2_pixels": 2}}
        same = {"new_fit_mean": dict(baseline["new_fit_mean"])}
        tradeoff = {"new_fit_mean": {"band_pixels": 9, "L2_worst_dose_pixels": 4,
                                     "L2_pixels": 2}}
        strict = {"new_fit_mean": {"band_pixels": 9, "L2_worst_dose_pixels": 3,
                                   "L2_pixels": 2}}
        self.assertFalse(benchmark._hard_pv_improves(same, baseline))
        self.assertFalse(benchmark._hard_pv_improves(tradeoff, baseline))
        self.assertTrue(benchmark._hard_pv_improves(strict, baseline))

    def test_rank_only_improvement_snapshot_never_claims_hard_pv_gain(self):
        baseline = {"new_fit_mean": {"band_pixels": 10, "L2_worst_dose_pixels": 3,
                                     "L2_pixels": 2}}
        qualified = {"candidate_id": "soft-only", "actual_hard_metrics": baseline}
        snapshot = benchmark._quality_snapshot(qualified, (10, 3, 2, 0.0), baseline)
        self.assertEqual(snapshot["best_qualified_candidate_id"], "soft-only")
        self.assertFalse(snapshot["hard_pv_improved_preserved_best"])
        self.assertEqual(snapshot["hard_pv_counts"], {
            "band_pixels": 10, "L2_worst_dose_pixels": 3, "L2_pixels": 2,
        })

    def test_parity_records_require_all_expected_layouts_and_finite_errors(self):
        expected = coverage.ORIGINAL_FIT_LAYOUT_IDS
        good = [{"layout_id": layout_id, "max_abs_error": 0.0} for layout_id in expected]
        benchmark._validate_direct_basis_parity(good, expected, "synthetic")
        with self.assertRaisesRegex(ValueError, "layout/order mismatch"):
            benchmark._validate_direct_basis_parity(good[:-1], expected, "synthetic")
        bad = [*good[:-1], {"layout_id": expected[-1], "max_abs_error": float("nan")}]
        with self.assertRaisesRegex(ValueError, "invalid max_abs_error"):
            benchmark._validate_direct_basis_parity(bad, expected, "synthetic")

    def test_golden_thread_audit_records_mismatch_and_restores_original_threads(self):
        original_threads = benchmark.torch.get_num_threads()
        primary = {"per_layout": [{"layout_id": "fit", "band_pixels": 1}],
                   "original_fit_mean": {"band_pixels": 1},
                   "new_fit_mean": {"band_pixels": 1},
                   "no_blank_positive_target_any_dose": True}
        golden = {"per_layout": [{"layout_id": "fit", "band_pixels": 2}],
                  "original_fit_mean": {"band_pixels": 2},
                  "new_fit_mean": {"band_pixels": 1},
                  "no_blank_positive_target_any_dose": True}
        real_set_num_threads = benchmark.torch.set_num_threads
        calls = []

        def tracked_set_num_threads(count):
            calls.append(count)
            return real_set_num_threads(count)

        with mock.patch.object(benchmark, "_metrics", return_value=golden), \
                mock.patch.object(benchmark.torch, "set_num_threads",
                                  side_effect=tracked_set_num_threads):
            audit = benchmark._golden_thread_metric_audit(
                [], {}, one_hot(0), {}, primary, 24,
            )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["per_layout_mismatches"][0]["layout_id"], "fit")
        self.assertEqual(calls, [24, original_threads])
        self.assertEqual(benchmark.torch.get_num_threads(), original_threads)

    def test_runner_scores_protected_incumbents_before_exhausting_candidate_cap(self):
        reference, initial, best, proposal = (one_hot(index) for index in range(4))
        plan = valid_event_plan()
        plan["source_vectors"]["reference"] = vector_record(reference)
        plan["source_vectors"]["best_known_slot_4755_seed_101"] = vector_record(best)
        plan["source_vectors"]["seed_endpoints"]["17"][
            "schema8_initialization"] = vector_record(initial)
        old_ids = list(coverage.ORIGINAL_FIT_LAYOUT_IDS)
        new_ids = list(coverage.LAYOUT_IDS)

        def actual_metrics(weights):
            source_index = int(np.argmax(weights))
            new_band = 11 if source_index in (0, 1) else 10
            per_layout = []
            for layout_id in old_ids:
                per_layout.append({"layout_id": layout_id, "band_pixels": 100,
                                   "L2_pixels": 0.0, "L2_worst_dose_pixels": 100,
                                   "per_dose_L2_pixels": [0, 0, 0],
                                   "positive_target_pixels": 1,
                                   "no_blank_positive_target_any_dose": True})
            for layout_id in new_ids:
                per_layout.append({"layout_id": layout_id, "band_pixels": new_band,
                                   "L2_pixels": 2.0, "L2_worst_dose_pixels": 3,
                                   "per_dose_L2_pixels": [3, 2, 3],
                                   "positive_target_pixels": 1,
                                   "no_blank_positive_target_any_dose": True})
            return {"per_layout": per_layout,
                    "original_fit_mean": {"band_pixels": 100, "L2_pixels": 0.0,
                                           "L2_worst_dose_pixels": 100.0},
                    "new_fit_mean": {"band_pixels": new_band, "L2_pixels": 2.0,
                                     "L2_worst_dose_pixels": 3.0},
                    "no_blank_positive_target_any_dose": True}

        reference_metrics = actual_metrics(reference)
        best_metrics = actual_metrics(best)
        plan["reference_new_fit_metrics"] = {
            "per_layout": reference_metrics["per_layout"][len(old_ids):]}
        plan["best_known_historical_fit_metrics"] = best_metrics["per_layout"]
        plan["physical"] = {}
        all_ids = old_ids + new_ids
        basis32 = {layout_id: np.zeros((event_search.SOURCE_COUNT, 1, 1), dtype=np.float32)
                   for layout_id in all_ids}
        basis64 = {layout_id: np.zeros((event_search.SOURCE_COUNT, 1, 1), dtype=np.float64)
                   for layout_id in all_ids}
        targets_by_id = {layout_id: np.zeros((1, 1), dtype=bool) for layout_id in all_ids}
        domain = mock.Mock()
        domain.verify.return_value = {"passed": True}
        fake_proposal = {"candidate_id": "synthetic-proposal", "weights": proposal,
                         "roles": ["grid_proposal"], "weights_f64_sha256": event_search.array_sha256(proposal, "<f8"),
                         "weights_f32_sha256": event_search.array_sha256(proposal, "<f4")}

        def rank_record(weights, actual, _basis_torch, _targets_torch, _anchor, order):
            return {"new_fit_mean": actual["new_fit_mean"],
                    "selection_soft_objective": 0.0 if int(np.argmax(weights)) == 3 else 1.0,
                    "l1_to_lp_anchor": 0.0, "checkpoint_order": order}

        with mock.patch.object(benchmark, "_metrics", side_effect=lambda _rows, _basis, weights, _physical: actual_metrics(weights)), \
                mock.patch.object(benchmark, "_grid_candidates", return_value=[fake_proposal]), \
                mock.patch.object(benchmark, "_rank_record", side_effect=rank_record), \
                mock.patch.object(coverage, "verify_original_nominal_polytope",
                                  return_value={"passed": True}), \
                mock.patch.object(coverage, "audit_float32_critical_guards",
                                  return_value={"passed": True}), \
                mock.patch.object(benchmark.torch, "get_num_threads", return_value=1), \
                mock.patch.object(benchmark.torch, "set_num_threads"):
            with mock.patch.object(benchmark, "ATTEMPT_CAP_PER_ARM_SEED", 3):
                incomplete = benchmark._arm(
                    "grid", 17, 0, plan, [], basis32, basis64, {}, {},
                    one_hot(5), reference, best, domain, targets_by_id, lambda _event: 0.0, 4,
                )
            self.assertEqual(incomplete["status"], "incomplete_budget")
            self.assertEqual(incomplete["attempted_candidates"], 3)
            self.assertEqual([row["roles"][0] for row in incomplete["candidate_records"]],
                             ["reference", "initial_incumbent", "best_known_incumbent"])
            self.assertEqual(incomplete["not_attempted_candidates"], 1)
            self.assertEqual(incomplete["not_attempted_interpretation"],
                             "not evidence of no qualified point")

            journal_records = []

            def record_journal(event):
                if event.get("type") == "candidate_audit":
                    journal_records.append(event["candidate_audit_record"])
                return 0.0

            with mock.patch.object(benchmark, "ATTEMPT_CAP_PER_ARM_SEED", 4):
                completed = benchmark._arm(
                    "grid", 17, 0, plan, [], basis32, basis64, {}, {},
                    one_hot(5), reference, best, domain, targets_by_id, record_journal, 4,
                    retain_candidate_records=False,
                )
            self.assertEqual(completed["status"], "complete")
            self.assertNotIn("candidate_records", completed)
            self.assertEqual(completed["candidate_audit_journal_records"], 4)
            self.assertEqual(len(journal_records), 4)
            self.assertTrue(all(record["method"] == "grid"
                                and record["seed"] == 17
                                and record["paired_repeat"] == 0
                                for record in journal_records))
            self.assertEqual(completed["checkpoint_rank_improvement_candidate_id"],
                             "synthetic-proposal")
            self.assertIsNotNone(completed[
                "time_to_checkpoint_rank_improve_preserved_best_seconds"])
            self.assertIsNone(completed["time_to_hard_pv_improve_preserved_best_seconds"])
            self.assertFalse(completed["matched_candidate_attempt_checkpoints"]["3"][
                "hard_pv_improved_preserved_best"])

            audits = [{"passed": True, "elapsed_seconds": 0.0},
                      {"passed": True, "elapsed_seconds": 0.0},
                      {"passed": False, "elapsed_seconds": 0.0,
                       "per_layout_mismatches": [{"layout_id": "synthetic"}]}]
            with mock.patch.object(benchmark, "_golden_thread_metric_audit",
                                   side_effect=audits):
                rejected = benchmark._arm(
                    "grid", 17, 0, plan, [], basis32, basis64, {}, {},
                    one_hot(5), reference, best, domain, targets_by_id,
                    lambda _event: 0.0, 4,
                )
            proposal_record = rejected["candidate_records"][-1]
            self.assertFalse(proposal_record["hard_qualified"])
            self.assertFalse(proposal_record["hard_qualification_gates"][
                "cpu_thread_golden_metric_parity_passed"])
            self.assertEqual(proposal_record["thread_count_audit"]["per_layout_mismatches"][0][
                "layout_id"], "synthetic")

    def test_end_pin_recheck_path_is_called_after_marker_then_fit_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            plan = {"expected_previous_sha256": "b" * 64,
                    "coverage_plan_sha256": "c" * 64,
                    "coverage_plan_file": str(root / "coverage-plan.json"),
                    "segment_plan_sha256": "d" * 64,
                    "segment_report_sha256": "e" * 64,
                    "selected_weights_file_sha256": "f" * 64,
                    "code_sha256": {}, "input_sha256": {},
                    "source_identity": {}, "lineage_artifact_count": 14,
                    "protocol": {}}
            order = []
            parent_plan = {"prerequisite_manifest": str(manifest)}

            def marker(_manifest_path, _event_plan_sha):
                order.append("marker")
                return root / "event-marker.json"

            def load_fit(*_args, **_kwargs):
                order.append("fit_loader")
                raise RuntimeError("synthetic stop immediately after marker")

            with mock.patch.object(benchmark, "_load_json_with_sha", return_value=plan), \
                    mock.patch.object(benchmark, "_check_pins",
                                      return_value=(parent_plan, {})) as check_pins, \
                    mock.patch.object(event_search, "create_event_attempt_marker",
                                      side_effect=marker), \
                    mock.patch.object(benchmark.parent, "_prepare_original_controls",
                                      side_effect=load_fit):
                report_path = benchmark.run(root / "plan.json", "a" * 64,
                                            "b" * 64, root / "runs")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(order, ["marker", "fit_loader"])
            self.assertEqual(check_pins.call_count, 2)
            self.assertEqual(report["status"], "error")
            self.assertTrue(report["event_attempt_consumed_before_fit_load_or_optics"])
            self.assertTrue(report["end_pin_recheck_passed"])
            self.assertEqual(report["final3_status"], "never indexed or evaluated")


if __name__ == "__main__":
    unittest.main()
