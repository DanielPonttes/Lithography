import ast
import copy
import hashlib
import json
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import source_coverage as coverage
from scripts import optimize_source_coverage as runner


def toy_domain():
    old_b = np.array([[[0.5, 0.1]], [[0.3, 0.1]], [[0.1, 0.1]]], dtype=np.float64)
    old_t = np.array([[True, False]])
    anchor = np.array([1.0, 0.0, 0.0])
    reference = np.array([0.0, 1.0, 0.0])
    new_b = np.array([[[0.1, 0.1]], [[0.5, 0.1]], [[0.3, 0.1]]], dtype=np.float64)
    new_t = np.array([[True, False]])
    return old_b, old_t, new_b, new_t, anchor, reference


def diagnostic_contract_fixture():
    input_data = {
        "dataset_sha256": "1fb6555fbf1dc4b4748f05f37d557977df5bbfd3b5b8abf5853c68a04716d1d0",
    }
    for field, ids in (("fit_masks", coverage.ORIGINAL_FIT_LAYOUT_IDS),
                       ("fit_targets", coverage.ORIGINAL_FIT_LAYOUT_IDS),
                       ("calibration_masks", coverage.CALIBRATION_LAYOUT_IDS),
                       ("calibration_targets", coverage.CALIBRATION_LAYOUT_IDS)):
        input_data[field] = [{"layout_id": name, "sha256": "a" * 64} for name in ids]
    input_data["bases"] = [
        {"layout_id": name, "split": split, "sha256": "b" * 64,
         "shape": [1, 49, 128, 128], "dtype": "torch.float32"}
        for split, ids in (("fit", coverage.ORIGINAL_FIT_LAYOUT_IDS),
                           ("calibration", coverage.CALIBRATION_LAYOUT_IDS))
        for name in ids
    ]
    return {"input": input_data}


def full_run_fixture(root: Path):
    manifest = root / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    weights = np.zeros(49, dtype=np.float64)
    weights[0] = 1.0
    old_rows = [{"layout_id": "original_fit_toy", "target": np.zeros((1, 1), dtype=bool)}]
    new_rows = [{"layout_id": name, "target": np.zeros((1, 1), dtype=bool)}
                for name in coverage.LAYOUT_IDS]
    all_ids = [row["layout_id"] for row in old_rows + new_rows]
    basis32 = {name: np.zeros((49, 1, 1), dtype=np.float32) for name in all_ids}
    basis64 = {name: value.astype(np.float64) for name, value in basis32.items()}
    basis_torch = {name: torch.zeros((49, 1, 1), dtype=torch.float64) for name in all_ids}
    targets_torch = {row["layout_id"]: torch.zeros((1, 1), dtype=torch.bool)
                     for row in old_rows + new_rows}
    identity = {
        "source_identity": {"head": "toy-clean-head"}, "lineage_artifacts": [],
        "diagnostic": {"input": {}},
        "previous_weights": {"canonical_basis_parity": coverage.PINNED_BASIS_PARITY},
    }
    context = {
        "identity": identity, "fresh_fit_parity": [], "lineage_artifacts": [],
        "rows": old_rows, "device": "cpu", "original_guard_summary": {},
        "reference_fit": {"mean": {"band_pixels": 0.0}},
        "reference_poly": {"passed": True}, "reference_audit": {"passed": True},
        "basis32": {name: basis32[name] for name in [row["layout_id"] for row in old_rows]},
        "basis64": {name: basis64[name] for name in [row["layout_id"] for row in old_rows]},
        "basis_torch": {name: basis_torch[name] for name in [row["layout_id"] for row in old_rows]},
        "targets_torch": {name: targets_torch[name] for name in [row["layout_id"] for row in old_rows]},
        "anchor": weights.copy(), "reference": weights.copy(),
    }
    plan = {
        "prerequisite_manifest": str(manifest), "input_hashes": {},
        "physical": {"raster": 128}, "new_fit_generation": {},
    }
    args = SimpleNamespace(
        output_root=str(root / "output"), plan_file=str(root / "frozen-plan.json"),
        expected_previous_sha256=coverage.PINNED_PREVIOUS_REPORT_SHA256,
        started_at=time.time(),
    )
    return {
        "root": root, "manifest": manifest, "plan": plan, "plan_sha": "a" * 64,
        "identity": identity, "context": context, "args": args,
        "old_rows": old_rows, "new_rows": new_rows, "basis32": basis32,
        "basis64": basis64, "basis_torch": basis_torch,
        "targets_torch": targets_torch, "weights": weights,
    }


class FullRunDomain:
    def verify(self, weights):
        return {"passed": bool(np.isfinite(weights).all()), "maximum_row_violation": 0.0}


def complete_seed_result(harness, seed, qualified=True):
    weights = harness["weights"].tolist()
    selected = None if not qualified else {
        "qualified": True, "weights": weights, "weights_sha256": "c" * 64,
        "checkpoint_order": 25,
        "new_fit_mean": {"band_pixels": 10.0, "L2_worst_dose_pixels": 1.0,
                         "L2_pixels": 0.0},
        "selection_soft_objective": 0.125, "l1_to_lp_anchor": 0.0,
    }
    return {
        "seed": seed, "status": "complete", "steps_completed": 204,
        "history": [{"step": 1, "weights_after_sha256": "c" * 64}],
        "checkpoints": [{"checkpoint_order": 25, "qualified": bool(qualified)}],
        "last_incumbent": {"seed": seed, "steps_completed": 204, "weights": weights},
        "selected": selected,
    }


def install_full_run_mocks(harness, stack: ExitStack, *, seed_side_effect=None,
                           identity_side_effect=None, calibration_side_effect=None):
    identity = harness["identity"]
    context = harness["context"]
    new_rows = harness["new_rows"]
    new_ids = [row["layout_id"] for row in new_rows]
    stack.enter_context(patch.object(runner, "_preflight",
                                     return_value=(harness["plan"], harness["plan_sha"], identity)))
    stack.enter_context(patch.object(runner, "_prepare_original_controls", return_value=context))
    stack.enter_context(patch.object(coverage, "make_coverage_masks", return_value=("toy-mask-only",)))
    stack.enter_context(patch.object(runner, "_generate_new_targets", return_value=new_rows))
    stack.enter_context(patch.object(runner, "_novelty_check", return_value=[{"layout_id": name}
                                                                            for name in new_ids]))
    stack.enter_context(patch.object(
        runner, "_prepare_basis",
        return_value=(None,
                      {name: context["basis32"].get(name, harness["basis32"][name])
                       for name in new_ids},
                      {name: harness["basis64"][name] for name in new_ids},
                      {name: harness["basis_torch"][name] for name in new_ids},
                      {name: harness["targets_torch"][name] for name in new_ids}, []),
    ))
    stack.enter_context(patch.object(runner, "_assert_identity_unchanged",
                                     side_effect=identity_side_effect or (lambda *_: identity)))
    stack.enter_context(patch.object(runner, "_start_seed_clock",
                                     side_effect=lambda _check: (time.monotonic(), 0.0)))
    stack.enter_context(patch.object(coverage, "build_guarded_domain",
                                     return_value=(FullRunDomain(), {"toy": True})))
    stack.enter_context(patch.object(coverage, "verify_original_nominal_polytope",
                                     return_value={"passed": True, "row_count": 1}))
    stack.enter_context(patch.object(runner, "_evaluate_rows", return_value={
        "mean": {"band_pixels": 10.0, "L2_pixels": 0.0, "L2_worst_dose_pixels": 1.0},
        "per_layout": [], "no_blank_positive_target_any_dose": True,
    }))
    stack.enter_context(patch.object(coverage, "audit_float32_critical_guards", return_value={
        "passed": True, "groups": [], "pixel_count": 0, "correct_count": 0, "failed_count": 0,
    }))
    if seed_side_effect is None:
        seed_side_effect = lambda seed, *_args: complete_seed_result(harness, seed)
    stack.enter_context(patch.object(runner, "_run_seed", side_effect=seed_side_effect))
    if calibration_side_effect is None:
        calibration_side_effect = lambda *_args: {"passed": True, "toy": True}
    calibration_mock = stack.enter_context(patch.object(
        runner, "_calibration_metrics", side_effect=calibration_side_effect
    ))
    return calibration_mock


class CoverageTests(unittest.TestCase):
    def test_reference_feasible_when_lp_anchor_misses_new_augmented_guard(self):
        ob, ot, nb, nt, anchor, ref = toy_domain()
        domain, counts = coverage.build_guarded_domain([ob], [ot], ["old"], [nb], [nt], ["new"], anchor, ref)
        self.assertTrue(domain.verify(ref)["passed"])
        self.assertFalse(domain.verify(anchor)["passed"])
        self.assertTrue(counts["anchor_augmented_feasibility_is_informational"])

    def test_full_original_polytope_and_exact_nominal_pruning_certificates(self):
        ob, ot, nb, nt, anchor, ref = toy_domain()
        domain, counts = coverage.build_guarded_domain([ob], [ot], ["old"], [nb], [nt], ["new"], anchor, ref)
        self.assertTrue(coverage.verify_original_nominal_polytope([ob], [ot], anchor)["passed"])
        self.assertEqual(counts["original_nominal_full_rows_verified"], 2)
        self.assertGreater(counts["original_nominal_rows_pruned_by_anchor_critical_dominance"], 0)
        self.assertEqual(counts["original_nominal_pruning_certificates"]["critical_floor"], coverage.EPSILON)
        self.assertTrue(domain.verify(ref)["passed"])

    def test_sub_epsilon_floor_keeps_reference_feasible(self):
        old_b = np.array([[[0.5, 0.1]], [[0.0, 0.5]], [[0.3, 0.1]]], dtype=np.float64)
        old_t = np.array([[True, False]])
        anchor = np.array([1.0, 0.0, 0.0])
        value = (coverage.THRESHOLD + 0.5e-6) / coverage.LOW_DOSE
        new_b = np.array([[[value, 0.5]], [[0.0, 0.5]], [[0.3, 0.1]]], dtype=np.float64)
        new_t = np.array([[True, False]])
        domain, counts = coverage.build_guarded_domain([old_b], [old_t], ["old"], [new_b], [new_t], ["new"], anchor, anchor)
        margin = coverage.critical_signed_margins(new_b, new_t, anchor)[0, 0]
        self.assertGreater(margin, 0.0)
        self.assertLess(margin, coverage.EPSILON)
        self.assertTrue(domain.verify(anchor)["passed"])
        self.assertEqual(counts["reference_guards_new"][0]["candidate_rows"], 1)

    def test_float32_guard_audit_is_pixelwise_and_detects_swaps(self):
        basis = np.array([[[0.5, 0.1]], [[0.1, 0.5]], [[0.3, 0.3]]], dtype=np.float32)
        target = np.array([[True, False]])
        anchor = np.array([1.0, 0.0, 0.0])
        audit = coverage.audit_float32_critical_guards({"old": basis}, {"old": target}, anchor, anchor,
                                                        [0.0, 1.0, 0.0], ["old"], [])
        self.assertFalse(audit["passed"])
        self.assertGreater(audit["pixel_count"], 0)
        self.assertEqual(audit["correct_count"] + audit["failed_count"], audit["pixel_count"])

    def test_guard_indices_repeat_in_row_major_class_balanced_order(self):
        target = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool)
        indices = coverage.deterministic_guard_indices(target, per_class=2)
        self.assertEqual(indices.tolist(), sorted(indices.tolist()))
        np.testing.assert_array_equal(indices, coverage.deterministic_guard_indices(target, per_class=2))
        labels = target.reshape(-1)[indices]
        self.assertEqual(int(labels.sum()), int((~labels).sum()), 2)

    def test_four_calibration_rows_validate_hashes_without_optics(self):
        masks = torch.zeros((4, 1, 128, 128), dtype=torch.float32)
        targets = torch.zeros_like(masks)
        for i in range(4):
            masks[i, 0, 10 + i:20 + i, 10:20] = 1
            targets[i, 0, 12 + i:18 + i, 12:18] = 1
        def descriptors(tensor):
            return [{"layout_id": name, "sha256": coverage.sha256_array(tensor[i, 0].numpy())}
                    for i, name in enumerate(coverage.CALIBRATION_LAYOUT_IDS)]
        diag = {"input": {"calibration_masks": descriptors(masks), "calibration_targets": descriptors(targets)}}
        ds = SimpleNamespace(layout_ids=coverage.CALIBRATION_LAYOUT_IDS, pixel_size_nm=4.0,
                             masks=masks, targets=targets)
        rows = runner._calibration_rows_from_dataset(ds, diag)
        self.assertEqual(tuple(row["layout_id"] for row in rows), coverage.CALIBRATION_LAYOUT_IDS)
        bad = copy.deepcopy(diag)
        bad["input"]["calibration_targets"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "target hash mismatch"):
            runner._calibration_rows_from_dataset(ds, bad)

    def test_contacts_have_eight_noncollinear_separate_components(self):
        mask = coverage.make_irregular_contacts_mask().astype(bool)
        seen = np.zeros_like(mask)
        components = []
        for y, x in zip(*np.nonzero(mask)):
            if seen[y, x]:
                continue
            stack, points = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < 128 and 0 <= nx < 128 and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            components.append(points)
        self.assertEqual(len(components), 8)
        centers = np.array([(np.mean([p[1] for p in c]), np.mean([p[0] for p in c])) for c in components])
        self.assertGreater(np.linalg.matrix_rank(centers - centers.mean(axis=0)), 1)
        boxes = []
        for component in components:
            xs = [point[1] for point in component]
            ys = [point[0] for point in component]
            boxes.append((min(xs), max(xs) + 1, min(ys), max(ys) + 1))
        self.assertEqual(sorted(boxes, key=lambda b: (b[2], b[0])), [
            (70, 90, 8, 28), (8, 28, 10, 30), (36, 58, 10, 32),
            (100, 122, 11, 33), (37, 57, 77, 97), (5, 27, 78, 100),
            (102, 122, 78, 98), (67, 91, 80, 104),
        ])

    def test_geometry_ids_order_shapes_and_uniqueness(self):
        layouts = coverage.make_coverage_masks()
        self.assertEqual(tuple(row.layout_id for row in layouts), coverage.LAYOUT_IDS)
        self.assertEqual(len({coverage.sha256_array(row.mask) for row in layouts}), 3)
        self.assertTrue(all(row.mask.shape == (128, 128) and np.isfinite(row.mask).all() for row in layouts))

    def test_ribbon_and_line_end_rectangles_match_the_frozen_geometry(self):
        expected_ribbons = np.zeros((128, 128), dtype=np.float32)
        for y0, y1, x0, x1 in (
            (9, 21, 8, 112), (29, 43, 21, 119), (58, 71, 4, 94),
            (81, 97, 28, 124), (106, 118, 14, 87),
            (21, 54, 16, 28), (52, 78, 93, 107),
        ):
            expected_ribbons[y0:y1, x0:x1] = 1
        expected_ends = np.zeros((128, 128), dtype=np.float32)
        for x0, width, y0, y1 in (
            (12, 12, 10, 48), (40, 14, 24, 75),
            (76, 13, 8, 56), (103, 12, 48, 113),
        ):
            expected_ends[y0:y1, x0:x0 + width] = 1
        for y0, y1, x0, x1 in ((61, 75, 40, 67), (61, 88, 53, 67), (91, 105, 80, 112)):
            expected_ends[y0:y1, x0:x1] = 1
        np.testing.assert_array_equal(coverage.make_finite_ribbons_mask(), expected_ribbons)
        np.testing.assert_array_equal(coverage.make_asymmetric_line_ends_mask(), expected_ends)

    def test_previous_report_seed17_per_layout_metrics_exact(self):
        weights = [1.0 / 49] * 49
        per_layout = [
            {"layout_id": "train_vls_pitch24_width9", "band_pixels": 512, "L2_pixels": 0, "L2_worst_dose_pixels": 256},
            {"layout_id": "train_hls_pitch28_width10", "band_pixels": 256, "L2_pixels": 0, "L2_worst_dose_pixels": 256},
            {"layout_id": "train_l_contours", "band_pixels": 111, "L2_pixels": 0, "L2_worst_dose_pixels": 59},
            {"layout_id": "train_t_junctions", "band_pixels": 59, "L2_pixels": 0, "L2_worst_dose_pixels": 30},
        ]
        seeds = []
        for seed in coverage.SEEDS:
            fit = {"passed": True, "weights": weights}
            if seed == 17:
                fit["fit_metrics"] = {"per_layout": per_layout}
            seeds.append({"seed": seed, "solver": {"optimal_zero_gap": True}, "fit": fit})
        report = {"objective_id": "target_aware_pareto_protected_buffered_count_v1", "status": "complete",
                  "fit_improvement_over_anchor": True,
                  "fit_selection": {"selected_seed": 17, "all_five_seed_weights_frozen": True},
                  "seeds": seeds, "anchor": {"source_weights_supported_float64": weights},
                  "preflight": {"fit_layout_ids": list(coverage.ORIGINAL_FIT_LAYOUT_IDS),
                                "basis_parity": copy.deepcopy(coverage.PINNED_BASIS_PARITY)}}
        self.assertEqual(coverage.validate_previous_report(report)["selected_seed"], 17)
        bad = copy.deepcopy(report)
        bad["seeds"][0]["fit"]["fit_metrics"]["per_layout"][0]["band_pixels"] += 1
        with self.assertRaisesRegex(ValueError, "seed17 FIT per-layout"):
            coverage.validate_previous_report(bad)

    def test_canonical_lineage_parity_is_pinned_separately_from_fresh_parity(self):
        historical = copy.deepcopy(coverage.PINNED_BASIS_PARITY)
        fresh = [{"layout_id": name, "max_abs_error": 5e-8}
                 for name in coverage.ORIGINAL_FIT_LAYOUT_IDS]
        coverage.validate_canonical_basis_parity(historical)
        self.assertEqual(len(fresh), 4)
        self.assertNotEqual(fresh, historical)
        jittered = copy.deepcopy(historical)
        jittered[0]["max_abs_error"] = 4e-8
        with self.assertRaisesRegex(ValueError, "canonical FIT basis parity"):
            coverage.validate_canonical_basis_parity(jittered)

    def test_diagnostic_basis_descriptors_include_fit_and_calibration_split(self):
        diag = diagnostic_contract_fixture()
        runner._validate_diagnostic_contract(diag)
        changed = copy.deepcopy(diag)
        changed["input"]["bases"][0]["split"] = "calibration"
        with self.assertRaisesRegex(ValueError, "basis descriptor"):
            runner._validate_diagnostic_contract(changed)

    def test_strict_plan_protocol_nested_mutations_fail(self):
        # A schema-complete plan fixture copies the frozen literals from the
        # validator; no external plan, dataset, teacher, or optics is opened.
        plan = {"schema_version": 5, "status": "prospective_candidate_plan",
                "created_utc": "fixture", "objective_id": coverage.OBJECTIVE_ID,
                "base_commit": "7483c1e7c8003f3e2f61caa6324a30d0662a8121",
                "dataset_file": "dataset.pt", "diagnostic_file": "diag.json", "prerequisite_manifest": "manifest.json",
                "previous_report": {"path": "prev.json", "sha256": coverage.PINNED_PREVIOUS_REPORT_SHA256,
                    "selected_seed": 17, "selection_role": "previously frozen FIT-only ranking; no calibration selection"},
                "previous_plan": {"path": "oldplan.json", "sha256": coverage.PINNED_PREVIOUS_PLAN_SHA256},
                "input_hashes": {"dataset_sha256": "1fb6555fbf1dc4b4748f05f37d557977df5bbfd3b5b8abf5853c68a04716d1d0",
                    "diagnostic_sha256": "cd236f09d6c0ad32398d3ec638f1d2433a2a8cc6b5060016c181c15358d8901b",
                    "source_manifest_sha256": coverage.PINNED_SOURCE_MANIFEST_SHA256},
                "scope": {"source_only": True, "fixed_masks": True, "number_of_source_weights": 49,
                    "calibration_role": "reused development data, never optimizer or tie-break input",
                    "final3_access": "never indexed or evaluated"},
                "physical": {"source_grid": 9, "sigma_inner": .3, "sigma_outer": .9, "NA": 1.35,
                    "wavelength_nm": 193, "pixel_nm": 4, "raster": 128, "doses": [.98, 1, 1.02],
                    "focus": 0, "threshold": .225, "steepness": 50},
                "protocol": {"rho": .01, "lp_margin": coverage.LP_MARGIN, "epsilon": 2e-6,
                    "solver_tolerance": 1e-9, "seeds": [17,29,43,71,101], "betas": [200.,400.,800.],
                    "steps_per_beta": 68, "total_steps_per_seed": 204, "checkpoint_interval": 25,
                    "checkpoint_at_beta_transition_and_final": True, "initial_simplex_jitter": 1e-4,
                    "fallback_if_guard_infeasible": "exact previous FIT-only seed17 reference; record fallback",
                    "per_seed_time_limit_seconds": 360., "total_time_limit_seconds": 1800.,
                    "original_guards": "full original nominal q plus original LP-anchor critical protections epsilon; preserve original float32 hard qualification",
                    "reference_guards": "all originally FIT-only seed17 reference-correct critical pixels, floor min(epsilon,0.5*positive float64 reference margin); original protections unchanged",
                    "new_guards": "deterministic row-major sampling, at most128 target-positive and128 target-negative indices per new layout; include only positive reference critical margin; floor min(epsilon,0.5*margin)",
                    "objective": "critical softcount mean with static equal weight per each of7FITlayouts; no dynamic/hotspot weights; existing target-aware critical dose softcount",
                    "fit_selection": "qualified checkpoints only; new3 mean hard PV, new3 mean worst-dose L2, new3 mean nominal L2, soft objective, L1 to original LP anchor, registered checkpoint order; never calibration",
                    "new_fit_gate": "strict reduction of new3 mean hard PV versus fixed seed17 reference, mean nominal and worst-dose L2 no greater than same reference, no positive-target blank at any dose",
                    "calibration_boundary": "once only after all5 complete qualified FIT-only source vectors freeze and identities rechecked; own FW eligibility, never fabricate MILP optimal certificates; failures after opening remain opened_then_failed no_retry",
                    "retries": "one exclusive consumed marker beside plan with same parent as original prerequisite manifest; no silent retry or budget reset; new attempt requires new prospective plan",
                    "interpretation": "nonconvex surrogate experiment; computational restarts, not independent generalization; no global optimum claim; final3 closed",
                    "original_fit_gate": {"mean_band_pixels_max":234.5, "mean_nominal_l2_pixels_max":0, "mean_worst_dose_l2_pixels_max":150.25},
                    "calibration_gate": {"mean_band_pixels_max":239.4, "mean_nominal_l2_pixels_max":56.175,
                        "mean_worst_dose_l2_pixels_max":157.2375, "each_seed_band_pixels_strictly_less_than":268,
                        "no_blank_positive_target_any_dose":True, "all_five_complete_fit_qualified":True}},
                "new_fit_generation": {"teacher":"same frozen original synthetic teacher and physics, targets fixed independently of candidate and LP anchor before optimization",
                    "policy":"three independent functions; no legacy make_layouts or heldout generation/access; generator source pinned by clean reviewed committed code before execution; no adaptive geometry correction based on scores",
                    "layout_ids":list(coverage.LAYOUT_IDS),
                    "novelty":"duplicate mask/target hashes checked only against permitted original FIT metadata and new3; no calibration/final3 data access",
                    "positive_fraction_min":.01, "positive_fraction_max":.99},
                "completion":"all5 complete204steps or registered convergence accounting; deadline preserves incumbent and closes calibration; same immutable control gates"}
        coverage.validate_plan_payload(plan)
        changed = copy.deepcopy(plan)
        changed["protocol"]["calibration_gate"]["mean_band_pixels_max"] = 239.5
        with self.assertRaisesRegex(ValueError, "calibration gate changed"):
            coverage.validate_plan_payload(changed)

    def test_hash_uses_same_bytes_and_lineage_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lineage.json"
            raw = b'{"status":"same-bytes"}'
            path.write_bytes(raw)
            parsed, actual_bytes, digest = runner._read_hashed_json(path)
            self.assertEqual(parsed, {"status":"same-bytes"})
            self.assertEqual(actual_bytes, raw)
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            identity = {"lineage_artifacts": [{"path": str(path), "sha256": digest}]}
            runner._check_lineage_artifacts(identity)
            path.write_text('{"status":"mutated"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lineage artifact changed"):
                runner._check_lineage_artifacts(identity)

    def test_all_five_fit_vectors_required_before_calibration(self):
        selected = {"qualified": True, "weights": [1/49] * 49}
        records = [{"seed": seed, "status":"complete", "steps_completed":204, "selected":selected}
                   for seed in coverage.SEEDS]
        self.assertTrue(coverage.fit_freeze_eligibility(records)["passed"])
        records[-1] = {**records[-1], "status":"timeout"}
        self.assertFalse(coverage.fit_freeze_eligibility(records)["passed"])

    def test_negative_fw_gap_is_a_failure_not_stationarity(self):
        self.assertEqual(coverage.classify_fw_gap(-1.1e-9), "negative_gap_failure")
        self.assertEqual(coverage.classify_fw_gap(-.5e-9), "stationary_tolerance")
        self.assertEqual(coverage.classify_fw_gap(1e-4), "continue")

    def test_attempt_marker_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            marker = coverage.create_attempt_marker(manifest, "a" * 64)
            self.assertTrue(marker.is_file())
            with self.assertRaises(FileExistsError):
                coverage.create_attempt_marker(manifest, "a" * 64)

    def test_original_audit_failure_leaves_attempt_marker_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            plan = {"prerequisite_manifest": str(manifest), "input_hashes": {}}
            identity = {"source_identity": {"head": "fixture"}}
            args = SimpleNamespace(output_root=str(root / "out"))
            with patch.object(runner, "_preflight", return_value=(plan, "a" * 64, identity)), \
                 patch.object(runner, "_prepare_original_controls",
                              side_effect=ValueError("original audit failed")), \
                 patch.object(coverage, "create_attempt_marker") as create_marker:
                with self.assertRaisesRegex(ValueError, "original audit failed"):
                    runner.run(args)
            create_marker.assert_not_called()
            marker = root / ("coverage_attempt_" + "a" * 64 + ".consumed")
            self.assertFalse(marker.exists())
            report_path = next((root / "out").rglob("coverage_report.json"))
            report = __import__("json").loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["coverage_attempt_consumed"])
            self.assertIsNone(report["attempt_marker"])
            self.assertEqual(report["calibration_status"], "closed")

    def test_new_mask_generation_occurs_only_after_marker_consumption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            plan = {"prerequisite_manifest": str(manifest), "input_hashes": {},
                    "physical": {"raster": 128}}
            identity = {"source_identity": {"head": "fixture"}, "lineage_artifacts": [],
                        "previous_weights": {"canonical_basis_parity": coverage.PINNED_BASIS_PARITY}}
            context = {"identity": identity, "fresh_fit_parity": [], "rows": [],
                       "device": "cpu", "original_guard_summary": {},
                       "reference_fit": {}, "reference_poly": {}, "reference_audit": {}}
            args = SimpleNamespace(output_root=str(root / "out"))

            def stop_after_marker(_size):
                marker = root / ("coverage_attempt_" + "a" * 64 + ".consumed")
                self.assertTrue(marker.is_file())
                raise ValueError("stop at masked generation boundary")

            with patch.object(runner, "_preflight", return_value=(plan, "a" * 64, identity)), \
                 patch.object(runner, "_prepare_original_controls", return_value=context), \
                 patch.object(coverage, "make_coverage_masks", side_effect=stop_after_marker):
                with self.assertRaisesRegex(ValueError, "masked generation boundary"):
                    runner.run(args)
            report_path = next((root / "out").rglob("coverage_report.json"))
            report = __import__("json").loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["coverage_attempt_consumed"])
            self.assertEqual(report["calibration_status"], "closed")

    def test_run_full_incomplete_seed_keeps_seed_report_and_calibration_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = full_run_fixture(Path(tmp))

            def incomplete(seed, *_args):
                return {"seed": seed, "status": "timeout", "steps_completed": 3,
                        "history": [{"step": 1}, {"step": 2}, {"step": 3}],
                        "checkpoints": [], "last_incumbent": {"steps_completed": 3},
                        "selected": None}

            with ExitStack() as stack:
                calibration = install_full_run_mocks(harness, stack, seed_side_effect=incomplete)
                report_path = runner.run(harness["args"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "timeout")
            self.assertEqual(report["calibration_status"], "closed")
            self.assertEqual(len(report["seeds"]), 1)
            self.assertEqual(report["seeds"][0]["status"], "timeout")
            self.assertEqual(len(report["seeds"][0]["history"]), 3)
            self.assertEqual(report["seeds"][0]["last_incumbent"]["steps_completed"], 3)
            calibration.assert_not_called()

    def test_run_full_complete_seed_without_qualified_checkpoint_closes_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = full_run_fixture(Path(tmp))

            def one_seed_unqualified(seed, *_args):
                return complete_seed_result(harness, seed, qualified=(seed != coverage.SEEDS[0]))

            with ExitStack() as stack:
                calibration = install_full_run_mocks(
                    harness, stack, seed_side_effect=one_seed_unqualified
                )
                report_path = runner.run(harness["args"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "no_fit_qualified_checkpoint")
            self.assertEqual(report["calibration_status"], "closed")
            self.assertFalse(report["fit_frozen"])
            self.assertEqual(len(report["seeds"]), len(coverage.SEEDS))
            self.assertTrue(all(row["status"] == "complete" for row in report["seeds"]))
            self.assertIsNone(report["seeds"][0]["selected"])
            calibration.assert_not_called()

    def test_run_full_freeze_is_persisted_before_calibration_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = full_run_fixture(Path(tmp))

            def fail_after_open(*_args):
                report_path = next((harness["root"] / "output").rglob("coverage_report.json"))
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertTrue(report["fit_frozen"])
                self.assertTrue(report["fit_frozen_sha256"])
                self.assertEqual(report["calibration_status"], "opened_once_in_progress")
                self.assertTrue(report["coverage_attempt_consumed"])
                raise RuntimeError("toy calibration failure after opening")

            with ExitStack() as stack:
                calibration = install_full_run_mocks(
                    harness, stack, calibration_side_effect=fail_after_open
                )
                with self.assertRaisesRegex(RuntimeError, "toy calibration failure"):
                    runner.run(harness["args"])
            calibration.assert_called_once()
            report_path = next((harness["root"] / "output").rglob("coverage_report.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["calibration_status"], "opened_then_failed_no_retry")
            self.assertTrue(report["coverage_attempt_consumed"])
            self.assertTrue(report["fit_frozen"])
            self.assertEqual(len(report["seeds"]), len(coverage.SEEDS))
            self.assertTrue(report["error"]["traceback"])

    def test_run_full_post_seed_identity_drift_keeps_seed_and_closes_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = full_run_fixture(Path(tmp))
            calls = {"count": 0}

            def drift_after_first_seed(*_args):
                calls["count"] += 1
                if calls["count"] == 2:  # first check after new FIT bases; second after seed 17
                    raise RuntimeError("toy post-seed identity drift")
                return harness["identity"]

            with ExitStack() as stack:
                calibration = install_full_run_mocks(
                    harness, stack, identity_side_effect=drift_after_first_seed
                )
                with self.assertRaisesRegex(RuntimeError, "post-seed identity drift"):
                    runner.run(harness["args"])
            calibration.assert_not_called()
            report_path = next((harness["root"] / "output").rglob("coverage_report.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["calibration_status"], "closed")
            self.assertEqual(len(report["seeds"]), 1)
            self.assertEqual(report["seeds"][0]["seed"], coverage.SEEDS[0])
            self.assertEqual(report["seeds"][0]["status"], "complete")
            self.assertTrue(report["seeds"][0]["selected"]["qualified"])
            self.assertIn("identity drift", report["error"]["message"])

    def test_run_full_precalibration_identity_drift_keeps_frozen_fit_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = full_run_fixture(Path(tmp))
            calls = {"count": 0}

            def drift_before_calibration(*_args):
                calls["count"] += 1
                if calls["count"] == 7:  # post-basis + five post-seed checks + pre-calibration
                    raise RuntimeError("toy pre-calibration identity drift")
                return harness["identity"]

            with ExitStack() as stack:
                calibration = install_full_run_mocks(
                    harness, stack, identity_side_effect=drift_before_calibration
                )
                with self.assertRaisesRegex(RuntimeError, "pre-calibration identity drift"):
                    runner.run(harness["args"])
            calibration.assert_not_called()
            report_path = next((harness["root"] / "output").rglob("coverage_report.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["calibration_status"], "closed")
            self.assertEqual(report["status"], "error")
            self.assertTrue(report["fit_frozen"])
            self.assertTrue(report["fit_frozen_sha256"])
            self.assertEqual(len(report["seeds"]), len(coverage.SEEDS))

    def test_run_full_midtrajectory_timeout_preserves_partial_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = full_run_fixture(Path(tmp))

            def timeout_after_one_step(seed, *_args):
                history = [{"phase": 0, "local_step": 0, "status": "accepted"}]
                incumbent = {"seed": seed, "steps_completed": 1,
                             "weights": harness["weights"].tolist(), "event": "accepted"}
                return {"seed": seed, "status": "timeout", "steps_completed": 1,
                        "history": history, "checkpoints": [],
                        "last_incumbent": incumbent, "selected": None}

            with ExitStack() as stack:
                calibration = install_full_run_mocks(
                    harness, stack, seed_side_effect=timeout_after_one_step
                )
                report_path = runner.run(harness["args"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            seed = report["seeds"][0]
            self.assertEqual(report["status"], "timeout")
            self.assertEqual(report["calibration_status"], "closed")
            self.assertEqual(seed["status"], "timeout")
            self.assertGreaterEqual(seed["steps_completed"], 1)
            self.assertGreaterEqual(len(seed["history"]), 1)
            self.assertEqual(seed["last_incumbent"]["steps_completed"], 1)
            self.assertNotEqual(seed["status"], "complete")
            calibration.assert_not_called()

    def test_run_full_solver_exception_persists_partial_progress_without_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = full_run_fixture(Path(tmp))

            def raise_after_progress(seed, *args):
                checkpoint_callback, iteration_callback = args[-2], args[-1]
                history = [{"phase": 0, "local_step": 0, "status": "accepted"}]
                checkpoint = {"checkpoint_order": 1, "beta": coverage.BETAS[0],
                              "weights": harness["weights"].tolist(),
                              "weights_sha256": "c" * 64, "qualified": False}
                checkpoints = [checkpoint]
                incumbent = {"seed": seed, "steps_completed": 1,
                             "weights": harness["weights"].tolist(), "event": "accepted"}
                iteration_callback(incumbent, history, checkpoints)
                checkpoint_callback(seed, checkpoint, history, checkpoints, incumbent)
                raise RuntimeError("toy checkpoint metrics failure")

            with ExitStack() as stack:
                calibration = install_full_run_mocks(
                    harness, stack, seed_side_effect=raise_after_progress
                )
                with self.assertRaisesRegex(RuntimeError, "toy checkpoint metrics failure"):
                    runner.run(harness["args"])
            calibration.assert_not_called()
            report_path = next((harness["root"] / "output").rglob("coverage_report.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["calibration_status"], "closed")
            self.assertEqual(report["status"], "error")
            self.assertEqual(len(report["seeds"]), 1)
            partial = report["seeds"][0]
            self.assertEqual(partial["status"], "solver_exception")
            self.assertEqual(partial["steps_completed"], 1)
            self.assertEqual(len(partial["history"]), 1)
            self.assertEqual(len(partial["checkpoints"]), 1)
            self.assertEqual(partial["last_incumbent"]["steps_completed"], 1)
            self.assertIsNone(partial["selected"])
            self.assertIn("checkpoint metrics failure", partial["failure"]["traceback"])

    def test_run_full_solver_exception_before_callback_does_not_reuse_prior_seed_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = full_run_fixture(Path(tmp))

            def fail_on_second_seed(seed, *_args):
                if seed == coverage.SEEDS[0]:
                    return complete_seed_result(harness, seed)
                raise RuntimeError("toy failure before seed callback")

            with ExitStack() as stack:
                calibration = install_full_run_mocks(
                    harness, stack, seed_side_effect=fail_on_second_seed
                )
                with self.assertRaisesRegex(RuntimeError, "before seed callback"):
                    runner.run(harness["args"])
            calibration.assert_not_called()
            report_path = next((harness["root"] / "output").rglob("coverage_report.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["calibration_status"], "closed")
            self.assertEqual(len(report["seeds"]), 2)
            self.assertEqual(report["seeds"][0]["seed"], coverage.SEEDS[0])
            partial = report["seeds"][1]
            self.assertEqual(partial["seed"], coverage.SEEDS[1])
            self.assertEqual(partial["status"], "solver_exception")
            self.assertEqual(partial["steps_completed"], 0)
            self.assertEqual(partial["history"], [])
            self.assertIsNone(partial["last_incumbent"])
            self.assertIsNone(partial["selected"])

    def test_run_full_five_seed_freeze_and_calibration_happy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = full_run_fixture(Path(tmp))
            with ExitStack() as stack:
                calibration = install_full_run_mocks(harness, stack)
                report_path = runner.run(harness["args"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            calibration.assert_called_once()
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["calibration_status"], "scored_once_after_fit_freeze")
            self.assertTrue(report["coverage_attempt_consumed"])
            self.assertTrue(report["fit_frozen"])
            self.assertTrue(report["fit_frozen_sha256"])
            self.assertEqual(len(report["seeds"]), len(coverage.SEEDS))
            self.assertEqual(len(report["fit_selection"]), len(coverage.SEEDS))
            self.assertTrue(report["calibration_gate_passed"])

    def test_seed_clock_starts_after_preseed_identity_audit(self):
        ticks = iter((10.0, 18.0, 20.0))
        started, audit_seconds = runner._start_seed_clock(
            lambda: None, monotonic=lambda: next(ticks)
        )
        self.assertEqual(audit_seconds, 8.0)
        self.assertEqual(started, 20.0)

    def test_jitter_infeasible_uses_exact_reference_fallback(self):
        reference = np.zeros(49, dtype=np.float64)
        reference[0] = 1.0
        rejected = np.zeros(49, dtype=np.float64)
        rejected[1] = 1.0

        class Domain:
            def verify(self, weights):
                passed = bool(np.array_equal(weights, reference))
                return {"passed": passed, "reason": "fixture"}

        with patch.object(coverage, "simplex_jitter", return_value=rejected):
            weights, record = runner._initial_weights(reference, 17, Domain())
        np.testing.assert_array_equal(weights, reference)
        self.assertTrue(record["fallback"])
        self.assertFalse(record["used_jitter"])

    def test_lmo_residual_failure_preserves_rejected_vector_and_residual(self):
        class Domain:
            source_count = 2
            a_ub = b_ub = None

            def verify(self, weights):
                return {"passed": False, "maximum_row_violation": 0.25,
                        "weights": np.asarray(weights).tolist()}

        fake = SimpleNamespace(status=0, message="optimal", success=True,
                               nit=1, x=np.array([0.4, 0.6]))
        with patch.object(runner, "linprog", return_value=fake):
            result = runner._linprog_lmo(Domain(), np.array([1.0, 2.0]), 1e9)
        self.assertEqual(result["status"], "residual_failure")
        np.testing.assert_allclose(result["rejected_x"], [0.4, 0.6])
        self.assertEqual(result["residual"]["maximum_row_violation"], 0.25)

    def test_run_seed_records_all_iteration_telemetry_and_eleven_checkpoints(self):
        reference = np.zeros(49, dtype=np.float64)
        reference[0] = 1.0

        class Domain:
            def verify(self, weights):
                return {"passed": True, "maximum_row_violation": 0.0}

        rows = [{"layout_id": key, "target": np.zeros((1, 1), dtype=bool)}
                for key in ("old", "new")]
        arrays32 = {key: np.zeros((49, 1, 1), dtype=np.float32) for key in ("old", "new")}
        arrays64 = {key: value.astype(np.float64) for key, value in arrays32.items()}
        tensors = {key: torch.zeros((49, 1, 1), dtype=torch.float64) for key in ("old", "new")}
        targets = {key: torch.zeros((1, 1), dtype=torch.bool) for key in ("old", "new")}
        checkpoints = []

        def fake_lmo(_domain, _gradient, _deadline):
            return {"status": "optimal_verified", "weights": reference.copy(),
                    "objective_value": 0.0, "record": {"status_code": 0}}

        def fake_checkpoint(_rows, _old, _new, _basis, _weights, _bases, _targets,
                            _beta, _anchor, _reference, _physical, order):
            return {"checkpoint_order": int(order)}

        with patch.object(runner.time, "monotonic", return_value=1000.0), \
             patch.object(runner, "critical_corner_softcount_value_gradient",
                          return_value=(0.0, np.zeros(49))), \
             patch.object(runner, "_linprog_lmo", side_effect=fake_lmo), \
             patch.object(runner, "_checkpoint_metrics", side_effect=fake_checkpoint), \
             patch.object(runner, "_fit_checkpoint_qualified",
                          side_effect=lambda row, *_: {**row, "qualified": False}), \
             patch.object(coverage, "verify_original_nominal_polytope",
                          return_value={"passed": True}):
            result = runner._run_seed(
                17, reference, reference, Domain(), rows, rows[:1], rows[1:],
                arrays32, arrays64, tensors, targets, {}, {}, 1000.0, 2000.0,
                lambda _seed, row, _history, _checkpoints, _incumbent:
                    checkpoints.append(row["checkpoint_order"]),
                lambda *_args: None,
            )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["steps_completed"], 204)
        self.assertEqual(checkpoints, [25, 50, 68, 75, 100, 125, 136, 150, 175, 200, 204])
        row = result["history"][0]
        for field in ("gradient", "lmo_vertex", "lmo_objective", "weights_after",
                      "weights_after_sha256"):
            self.assertIn(field, row)

    def test_run_seed_deadline_preserves_warm_incumbent_and_no_checkpoint(self):
        reference = np.zeros(49, dtype=np.float64)
        reference[0] = 1.0

        class Domain:
            def verify(self, weights):
                return {"passed": True}

        rows = [{"layout_id": key, "target": np.zeros((1, 1), dtype=bool)}
                for key in ("old", "new")]
        basis64 = {key: np.zeros((49, 1, 1), dtype=np.float64) for key in ("old", "new")}
        tensors = {key: torch.zeros((49, 1, 1), dtype=torch.float64) for key in ("old", "new")}
        targets = {key: torch.zeros((1, 1), dtype=torch.bool) for key in ("old", "new")}
        snapshots = []
        with patch.object(runner.time, "monotonic", return_value=100.0), \
             patch.object(coverage, "verify_original_nominal_polytope",
                          return_value={"passed": True}):
            result = runner._run_seed(
                17, reference, reference, Domain(), rows, rows[:1], rows[1:],
                {}, basis64, tensors, targets, {}, {}, 100.0, 99.0,
                lambda *_: self.fail("deadline must precede checkpoint"),
                lambda snapshot, *_: snapshots.append(snapshot),
            )
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["steps_completed"], 0)
        self.assertEqual(result["last_incumbent"]["event"], "deadline_before_iteration")
        self.assertTrue(snapshots)

    def test_calibration_failure_state_records_open_or_closed_boundary(self):
        closed = {"coverage_attempt_consumed": False}
        opened = {"coverage_attempt_consumed": True}
        runner._record_run_failure(closed, ValueError("precal"), calibration_opened=False)
        runner._record_run_failure(opened, ValueError("postcal"), calibration_opened=True)
        self.assertEqual(closed["calibration_status"], "closed")
        self.assertEqual(opened["calibration_status"], "opened_then_failed_no_retry")

    def test_fit_only_loader_and_mask_module_do_not_open_heldout_data(self):
        tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_fit_only")
        keys = [node.slice.value for node in ast.walk(function)
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)]
        self.assertEqual(keys, ["fit"])
        source = Path(coverage.__file__).read_text(encoding="utf-8")
        self.assertNotIn("make_layouts(", source)
        self.assertNotIn("generate_calibration(", source)
        self.assertNotIn("FINAL_IDS", source)


if __name__ == "__main__":
    unittest.main()
