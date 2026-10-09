import ast
import copy
import unittest
import json
import tempfile
from contextlib import ExitStack
from unittest import mock
from pathlib import Path

import numpy as np
import torch

import source_coverage as coverage
import source_segment_sweep as sweep
from scripts import diagnose_source_segments as runner


def endpoint_fixture(seed):
    ref = np.zeros(49, dtype=np.float64)
    ref[0] = 1.0
    start = np.zeros(49, dtype=np.float64)
    start[1] = 1.0
    s6 = np.zeros(49, dtype=np.float64)
    s6[2] = 1.0
    s7 = np.zeros(49, dtype=np.float64)
    s7[3] = 1.0
    s8 = np.zeros(49, dtype=np.float64)
    s8[4] = 1.0
    return {
        "initialization": start.tolist(), "initialization_sha256": sweep.weight_hash64(start),
        "schema6_endpoint": s6.tolist(), "schema6_endpoint_sha256": sweep.weight_hash64(s6),
        "schema6_endpoint_float32_sha256": sweep.weight_hash32(s6),
        "schema7_endpoint": s7.tolist(), "schema7_endpoint_sha256": sweep.weight_hash64(s7),
        "schema7_endpoint_float32_sha256": sweep.weight_hash32(s7),
        "schema8_endpoint": s8.tolist(), "schema8_endpoint_sha256": sweep.weight_hash64(s8),
        "schema8_endpoint_float32_sha256": sweep.weight_hash32(s8),
    }


def toy_plan():
    endpoints = {str(seed): endpoint_fixture(seed) for seed in sweep.SEEDS}
    return {"endpoints": endpoints}


def failed_report_fixture(seed=17, schema=4, objective=coverage.OBJECTIVE_ID_V8,
                          plan_sha=sweep.PARENT_SCHEMA8_PLAN_SHA256):
    rows = []
    for seed_value in sweep.SEEDS:
        weights = np.zeros(49, dtype=np.float64)
        weights[0] = 1.0
        initialization = {"status": "feasible_start_found", "fallback": False,
                          "used_jitter": False, "weights": weights.tolist(),
                          "weights_sha256": sweep.weight_hash64(weights)}
        checkpoints = []
        for order in range(11):
            row_weights = weights.copy()
            checkpoints.append({"checkpoint_order": 204 if order == 10 else order,
                                "qualified": False, "weights": row_weights.tolist(),
                                "weights_sha256": sweep.weight_hash64(row_weights)})
        rows.append({"seed": seed_value, "status": "complete", "steps_completed": 204,
                     "qualified_checkpoint_count": 0, "selected": None,
                     "initialization": initialization, "checkpoints": checkpoints})
    return {"schema_version": schema, "objective_id": objective,
            "status": "no_fit_qualified_checkpoint", "plan_sha256": plan_sha,
            "coverage_attempt_consumed": True, "calibration_status": "closed",
            "final3_status": "never indexed or evaluated", "seeds": rows,
            # The validator must ignore arbitrary non-FIT payload fields.
            "calibration_payload": {"must_not_be_used": [1, 2, 3]}}


class SegmentGridTests(unittest.TestCase):
    def test_grid_is_exactly_ordered_and_unprojected(self):
        reference = np.zeros(49, dtype=np.float64)
        reference[0] = 1.0
        slots = list(sweep.iter_slots(toy_plan(), reference))
        self.assertEqual(len(slots), 5140)
        self.assertEqual(slots[0]["slot_index"], 0)
        self.assertEqual(slots[0]["seed"], 17)
        self.assertEqual(slots[0]["segment_id"], "schema8_initialization")
        self.assertEqual(slots[0]["alpha_numerator"], 0)
        self.assertEqual(slots[256]["alpha_numerator"], 256)
        self.assertEqual(slots[257]["segment_id"], "schema6_endpoint")
        self.assertEqual(slots[-1]["seed"], 101)
        self.assertEqual(slots[-1]["segment_id"], "schema8_endpoint")
        self.assertEqual(slots[-1]["alpha_numerator"], 256)
        np.testing.assert_array_equal(slots[0]["weights"], reference)
        np.testing.assert_array_equal(slots[256]["weights"],
                                      np.asarray(toy_plan()["endpoints"]["17"]["initialization"]))
        middle = slots[128]["weights"]
        expected = 0.5 * reference + 0.5 * np.asarray(toy_plan()["endpoints"]["17"]["initialization"])
        np.testing.assert_array_equal(middle, expected)
        self.assertAlmostEqual(float(middle.sum()), 1.0)
        self.assertEqual(slots[0]["weights_f64_sha256"], sweep.weight_hash64(reference))
        self.assertEqual(slots[0]["weights_f32_sha256"], sweep.weight_hash32(reference))

    def test_duplicate_slots_are_retained_as_distinct_provenance_records(self):
        reference = np.zeros(49, dtype=np.float64)
        reference[0] = 1.0
        slots = list(sweep.iter_slots(toy_plan(), reference))
        zero_slots = [row for row in slots if row["alpha_numerator"] == 0]
        self.assertEqual(len(zero_slots), 20)
        self.assertEqual(len({row["slot_index"] for row in zero_slots}), 20)
        self.assertEqual({row["weights_f64_sha256"] for row in zero_slots},
                         {sweep.weight_hash64(reference)})
        first = runner._record_unattempted(slots[0], "synthetic_timeout")
        second = runner._record_unattempted(slots[257], "synthetic_timeout")
        self.assertNotEqual(first["slot_index"], second["slot_index"])
        self.assertEqual(first["weights_f64_sha256"], second["weights_f64_sha256"])
        self.assertEqual(first["reason"], "synthetic_timeout")

    def test_bad_endpoint_vector_is_rejected_before_grid_generation(self):
        plan = toy_plan()
        plan["endpoints"]["17"]["schema8_endpoint"] = [0.0] * 48
        # iter_slots validates vector shape before yielding that segment's first candidate.
        reference = np.zeros(49, dtype=np.float64)
        reference[0] = 1.0
        slots = sweep.iter_slots(plan, reference)
        for _ in range(sweep.SLOTS_PER_SEGMENT * 3):
            next(slots)
        with self.assertRaises(ValueError):
            next(slots)


class FrozenReportTests(unittest.TestCase):
    def test_frozen_report_extraction_uses_only_closed_fit_state(self):
        report = failed_report_fixture()
        extracted = sweep._endpoint_vectors(report, "schema8", 4,
                                            coverage.OBJECTIVE_ID_V8,
                                            sweep.PARENT_SCHEMA8_PLAN_SHA256)
        self.assertEqual(set(extracted), set(sweep.SEEDS))
        self.assertEqual(extracted[17]["endpoint_sha256"], sweep.weight_hash64(extracted[17]["endpoint"]))

    def test_frozen_report_rejects_mutated_weight_payload(self):
        report = failed_report_fixture()
        report["seeds"][0]["checkpoints"][-1]["weights"][1] = 0.25
        with self.assertRaisesRegex(ValueError, "terminal vector hash"):
            sweep._endpoint_vectors(report, "schema8", 4,
                                    coverage.OBJECTIVE_ID_V8,
                                    sweep.PARENT_SCHEMA8_PLAN_SHA256)

    def test_frozen_report_rejects_open_calibration_or_final_boundary(self):
        for field, value in (("calibration_status", "opened"),
                             ("final3_status", "indexed")):
            report = failed_report_fixture()
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                sweep._endpoint_vectors(report, "schema8", 4,
                                        coverage.OBJECTIVE_ID_V8,
                                        sweep.PARENT_SCHEMA8_PLAN_SHA256)


class FeasibilityAndBoundaryTests(unittest.TestCase):
    def test_float64_audit_checks_simplex_full_nominal_and_guard_domain(self):
        basis = np.ones((49, 1, 2), dtype=np.float64)
        basis[:, 0, 0] = 0.5
        basis[:, 0, 1] = 0.1
        target = np.array([[True, False]])
        reference = np.full(49, 1.0 / 49, dtype=np.float64)
        anchor = reference.copy()
        domain, _ = coverage.build_guarded_domain(
            [basis], [target], ["old"], [], [], [], anchor, reference,
        )
        result = sweep.audit_float64_candidate(reference, reference, anchor,
                                               [basis], [target], domain)
        self.assertTrue(result["passed"])
        self.assertTrue(result["original_nominal_polytope"]["passed"])
        self.assertTrue(result["guard_domain"]["passed"])
        invalid = np.zeros(49, dtype=np.float64)
        rejected = sweep.audit_float64_candidate(invalid, reference, anchor,
                                                 [basis], [target], domain)
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["simplex"]["passed"])

    def test_runner_has_no_optimizer_or_calibration_execution_path(self):
        tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        run_node = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "run")
        called = {node.func.attr for node in ast.walk(run_node)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertNotIn("run", called)
        self.assertNotIn("_load_calibration", called)
        self.assertNotIn("_evaluate_calibration", called)
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn('"calibration_status": "closed by design"', source)
        self.assertIn('"final3_status": "never indexed or evaluated"', source)


class PlanAndCacheTests(unittest.TestCase):
    def prospective_plan(self):
        with mock.patch.object(sweep, "collect_frozen_endpoints", return_value={
                seed: endpoint_fixture(seed) for seed in sweep.SEEDS}):
            return sweep.make_candidate_plan(Path(runner.ROOT), Path("unused"),
                                             sweep.PARENT_ROOT + "/source_family_manifest.json")

    def test_strict_plan_rejects_protocol_physics_and_dependency_mutations(self):
        plan = self.prospective_plan()
        pin = sweep.PINNED_PLAN_SHA256
        sweep.validate_plan(plan, pin, pin)
        for key in ("protocol", "physical", "budgets", "source_identity", "lineage"):
            changed = copy.deepcopy(plan)
            changed[key]["unexpected"] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                sweep.validate_plan(changed, pin, pin)
        for path in ("light_source.py", "source_robustness.py", "scripts/optimize_source_coverage.py"):
            self.assertIn(path, plan["source_identity"]["files"])
        changed = copy.deepcopy(plan)
        changed["source_identity"]["files"].pop("light_source.py")
        with self.assertRaises(ValueError):
            sweep.validate_plan(changed, pin, pin)

    def test_source_hash_normalizes_line_endings_and_detects_dependency_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in sweep.SOURCE_FILES:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                content = ('PINNED_PLAN_SHA256 = "' + 'a' * 64 + '"\n'
                           if relative == "source_segment_sweep.py" else "# dependency\n")
                target.write_bytes(content.encode())
            first = sweep.source_file_hashes(root)
            for relative in sweep.SOURCE_FILES:
                path = root / relative
                path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            self.assertEqual(first, sweep.source_file_hashes(root))
            (root / "light_source.py").write_bytes(b"# changed physics\n")
            self.assertNotEqual(first, sweep.source_file_hashes(root))

    def test_float32_cache_preserves_own_float64_objective_rank_and_audit(self):
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            basis = np.ones((49, 1, 2), dtype=np.float32)
            basis[:, 0, 0] = .5
            basis[:, 0, 1] = .1
            basis[1, 0, 0] = .52
            target = np.array([[True, False]])
            rows = [{"layout_id": str(i), "target": target} for i in range(7)]
            reference = np.zeros(49)
            reference[:2] = .5
            variant = reference.copy()
            variant[0] += 1e-10
            variant[1] -= 1e-10
            domain, _ = coverage.build_guarded_domain(
                [basis.astype(np.float64)] * 4, [target] * 4, [str(i) for i in range(4)],
                [basis.astype(np.float64)] * 3, [target] * 3, [str(i) for i in range(4, 7)],
                reference, reference,
            )
            context = {"all_rows": rows, "original_rows": rows[:4], "new_rows": rows[4:],
                       "basis32": {str(i): basis for i in range(7)},
                       "basis64": {str(i): basis.astype(np.float64) for i in range(7)},
                       "basis_torch": {str(i): torch.from_numpy(basis.astype(np.float64)) for i in range(7)},
                       "targets_torch": {str(i): torch.from_numpy(target) for i in range(7)},
                       "physical": sweep.PHYSICAL}
            def slot(weights, index):
                return {"weights": weights, "slot_index": index,
                        "weights_f32_sha256": sweep.weight_hash32(weights)}
            cache = {}
            poly = coverage.verify_original_nominal_polytope([basis] * 4, [target] * 4, reference)
            means = {"band_pixels": 0., "L2_pixels": 0., "L2_worst_dose_pixels": 0.}
            with mock.patch.object(parent := runner.parent, "_checkpoint_metrics", wraps=parent._checkpoint_metrics) as physical:
                first = runner._candidate_score(slot(reference, 0), context, means, reference, reference, poly, domain, cache)
                second = runner._candidate_score(slot(variant, 1), context, means, reference, reference, poly, domain, cache)
            self.assertEqual(physical.call_count, 1)
            self.assertTrue(second["physical_cache_hit"])
            self.assertFalse(first["physical_cache_hit"])
            self.assertEqual(second["weights"], variant.tolist())
            self.assertGreater(second["l1_to_lp_anchor"], first["l1_to_lp_anchor"])
            self.assertEqual(second["selection_soft_objective"], parent.critical_corner_softcount_value(
                variant, context["basis_torch"], context["targets_torch"], 800.))
            self.assertTrue(second["float64_candidate_audit"]["passed"])
            self.assertEqual(coverage.checkpoint_rank(second)[-1], 1)
            first["all_fit"]["mean"]["band_pixels"] = -1
            self.assertEqual(second["all_fit"]["mean"]["band_pixels"], 0.)
        finally:
            torch.set_num_threads(old_threads)

    def test_golden_score_restores_screen_threads_on_exception(self):
        original = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with mock.patch.object(runner, "_candidate_score", side_effect=RuntimeError("golden failed")):
                with self.assertRaises(RuntimeError):
                    runner._golden_score(None, None, None, None, None, None, None, original)
            self.assertEqual(torch.get_num_threads(), 1)
        finally:
            torch.set_num_threads(original)

    def test_preflight_never_deserializes_data_generates_optics_or_consumes_marker(self):
        with mock.patch.object(runner, "_load_plan", return_value=({"source_identity": {"files": {}}}, "sha")), \
             mock.patch.object(sweep, "validate_source_identity", return_value={}), \
             mock.patch.object(runner, "_validate_frozen_reports", return_value=({}, {})), \
             mock.patch.object(runner, "_parent_identity", return_value=({}, {}, "parent")), \
             mock.patch.object(coverage, "create_attempt_marker", side_effect=AssertionError("marker")), \
             mock.patch.object(runner.parent, "_prepare_original_controls", side_effect=AssertionError("dataset")), \
             mock.patch.object(runner.parent, "_prepare_basis", side_effect=AssertionError("optics")):
            runner._preflight(Path("unused"), "sha")

    def test_recheck_includes_full_parent_artifacts_and_endpoint_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            path.write_bytes(b"same")
            digest = coverage.sha256_file(path)
            plan = {"source_identity": {"files": {}}, "input_hashes": {
                "dataset_sha256": digest, "diagnostic_sha256": digest, "source_manifest_sha256": digest},
                "lineage": {"dataset_file": str(path), "diagnostic_file": str(path), "source_manifest_path": str(path)}}
            with mock.patch.object(sweep, "validate_source_identity"), \
                 mock.patch.object(runner, "_parent_identity") as parent_check, \
                 mock.patch.object(runner, "_validate_frozen_reports") as reports:
                runner._recheck(path, digest, plan, path, digest)
                parent_check.assert_called_once_with(plan)
                reports.assert_called_once_with(plan)
                reports.side_effect = ValueError("report changed")
                with self.assertRaisesRegex(ValueError, "report changed"):
                    runner._recheck(path, digest, plan, path, digest)


class RunnerLifecycleTests(unittest.TestCase):
    def run_fixture(self, tmp, *, fail_at=None, qualified=False, golden_pass=True,
                    per_seed_limit=600., total_limit=3600., setup_limit=300., reference_mismatch=False):
        root = Path(tmp)
        reference = np.zeros(49)
        reference[0] = 1.
        rows = [{"layout_id": str(i), "target": np.array([[True]])} for i in range(7)]
        slots = []
        for index, seed in enumerate((17, 17, 29, 29)):
            slots.append({"slot_index": index, "seed": seed, "segment_id": "schema8_endpoint",
                          "segment_order": index % 2, "alpha_numerator": 0, "alpha_denominator": 256,
                          "alpha": 0., "weights": reference, "weights_f64_sha256": sweep.weight_hash64(reference),
                          "weights_f32_sha256": sweep.weight_hash32(reference)})
        plan = {"lineage": {"schema8_parent_plan_sha256": "a" * 64,
                            "source_manifest_path": str(root / "manifest.json")},
                "physical": sweep.PHYSICAL, "fixed_fit_layout_hashes": [], "protocol": {}, "budgets": {}}
        original = {"rows": rows[:4], "reference": reference, "anchor": reference,
                    "device": "cpu", "basis32": {}, "basis64": {}, "basis_torch": {}, "targets_torch": {}}
        evaluated = {"mean": {"band_pixels": 1., "L2_pixels": 0., "L2_worst_dose_pixels": 1.}, "per_layout": []}
        metric = {"qualified": qualified, "new_fit_mean": evaluated["mean"], "selection_soft_objective": .1,
                  "l1_to_lp_anchor": 0., "checkpoint_order": 0, "float64_candidate_audit": {"passed": True}}
        domain = mock.Mock()
        domain.verify.return_value = {"passed": True}
        events = []
        real_marker = coverage.create_attempt_marker
        def consume(*args):
            events.append("marker")
            return real_marker(*args)
        def prepare(*args):
            events.append("prepare")
            return original
        def score(slot, *args):
            events.append("score")
            if slot["slot_index"] == fail_at:
                raise RuntimeError("synthetic slot failure")
            value = copy.deepcopy(metric)
            value["checkpoint_order"] = slot["slot_index"]
            return value
        def golden(slot, *args):
            events.append("golden")
            value = copy.deepcopy(metric)
            value["qualified"] = golden_pass
            value["checkpoint_order"] = slot["slot_index"]
            return value
        with ExitStack() as stack:
            patches = [
                mock.patch.object(sweep, "SEEDS", (17, 29)), mock.patch.object(sweep, "TOTAL_SLOTS", 4),
                mock.patch.object(sweep, "SLOTS_PER_SEED", 2), mock.patch.object(sweep, "iter_slots", return_value=iter(slots)),
                mock.patch.object(sweep, "PER_SEED_LIMIT_SECONDS", per_seed_limit),
                mock.patch.object(sweep, "TOTAL_LIMIT_SECONDS", total_limit), mock.patch.object(sweep, "SETUP_LIMIT_SECONDS", setup_limit),
                mock.patch.object(runner, "_preflight", return_value=(plan, "b" * 64,
                    {"new_fit_generation": {}}, {}, "parent-plan", {"source_identity": {}, "frozen_reports": {"schema8": {"reference_new_fit": evaluated}}})),
                mock.patch.object(runner, "_recheck"), mock.patch.object(coverage, "create_attempt_marker", side_effect=consume),
                mock.patch.object(runner.parent, "_prepare_original_controls", side_effect=prepare),
                mock.patch.object(coverage, "make_coverage_masks", return_value=[]),
                mock.patch.object(runner.parent, "_generate_new_targets", return_value=rows[4:]),
                mock.patch.object(runner.parent, "_novelty_check", return_value=[]),
                mock.patch.object(runner.parent, "_prepare_basis", return_value=(None, {}, {}, {}, {}, [])),
                mock.patch.object(coverage, "build_guarded_domain", return_value=(domain, {})),
                mock.patch.object(coverage, "verify_original_nominal_polytope", return_value={"passed": True}),
                mock.patch.object(runner.parent, "_evaluate_rows", return_value=evaluated),
                mock.patch.object(runner, "_verify_reference_fit"),
                mock.patch.object(coverage, "audit_float32_critical_guards", return_value={"passed": True}),
                mock.patch.object(runner, "_candidate_score", side_effect=score),
                mock.patch.object(runner, "_golden_score", side_effect=golden),
            ]
            for patch in patches:
                stack.enter_context(patch)
            for row in rows:
                original["basis64"][row["layout_id"]] = np.ones((49, 1, 1))
            if reference_mismatch:
                runner.parent._evaluate_rows.side_effect = [evaluated, {"different": True}]
            before = torch.get_num_threads()
            path = runner.run(root / "plan", "b" * 64, root / "reports", "cpu")
            self.assertEqual(torch.get_num_threads(), before)
            report = json.loads(path.read_text())
            self.assertFalse(report["calibration_opened"])
            self.assertEqual(report["final3_status"], "never indexed or evaluated")
            self.assertEqual(events[:2], ["marker", "prepare"])
            self.assertTrue((root / ("coverage_attempt_" + "b" * 64 + ".consumed")).is_file())
            return report, events

    def test_complete_grid_retains_every_duplicate_slot_and_five_seed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, events = self.run_fixture(tmp)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["attempted_slots"], 4)
        self.assertEqual(len(report["slot_records"]), 4)
        self.assertEqual(report["slot_records"][-1]["duplicate_of_slot_index_f32"], 0)
        self.assertNotIn("golden", events)

    def test_screen_pass_requires_golden_pass_and_preserves_best_weights(self):
        for passes in (False, True):
            with self.subTest(passes=passes), tempfile.TemporaryDirectory() as tmp:
                report, events = self.run_fixture(tmp, qualified=True, golden_pass=passes)
                self.assertEqual(events.count("golden"), 4)
                self.assertEqual(report["five_seed_fit_qualification"], passes)
                if passes:
                    self.assertEqual(len(report["selected_qualified_fit_point"]["weights"]), 49)
                else:
                    self.assertIsNone(report["selected_qualified_fit_point"])

    def test_error_retains_incumbent_and_marks_remaining_slots_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = self.run_fixture(tmp, qualified=True, fail_at=1)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["attempted_slots"], 2)
        self.assertEqual(len(report["slot_records"]), 4)
        self.assertIsNotNone(report["selected_qualified_fit_point"])
        self.assertEqual(report["seeds"][-1]["status"], "not_attempted")
        self.assertFalse(report["five_seed_fit_qualification"])

    def test_per_seed_total_and_setup_timeout_are_incomplete(self):
        for limits in ({"per_seed_limit": 0.}, {"total_limit": 0.}, {"setup_limit": 0.}):
            with self.subTest(limits=limits), tempfile.TemporaryDirectory() as tmp:
                report, _ = self.run_fixture(tmp, **limits)
                self.assertEqual(report["status"], "timeout")
                self.assertEqual(report["attempted_slots"], 0)
                self.assertEqual(len(report["slot_records"]), 4)
                self.assertEqual(len(report["seeds"]), 2)
                self.assertFalse(report["five_seed_fit_qualification"])

    def test_reference_thread_mismatch_fails_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, events = self.run_fixture(tmp, reference_mismatch=True)
        self.assertEqual(report["status"], "error")
        self.assertNotIn("score", events)

    def test_cli_exit_nonzero_for_error_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            for status, expected in (("complete", 0), ("timeout", 3), ("error", 3)):
                path.write_text(json.dumps({"status": status}))
                with self.subTest(status=status), mock.patch.object(runner, "run", return_value=path):
                    self.assertEqual(runner.main(["--mode", "run", "--plan-file", "plan",
                                                 "--expected-plan-sha256", "sha", "--output-root", tmp]), expected)


if __name__ == "__main__":
    unittest.main()
