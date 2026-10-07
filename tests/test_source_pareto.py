"""CPU-only toy and fail-closed contract checks for the protected-error MILP."""
import hashlib
import ast
import inspect
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from scipy.optimize import linprog

from source_pareto import (
    EPSILON, HIGH_DOSE, LP_MARGIN, LOW_DOSE, RHO,
    build_sparse_milp, check_model_residual, critical_signed_margins,
    five_seed_eligibility, solve_highs, validate_solver_record,
)
from scripts import diagnose_source_pareto as runner
from scripts import optimize_source_constrained as constrained


def _constrained_toy():
    """Three anchor errors; one is hard-repaired without its epsilon buffer."""
    t_max = (0.225 - EPSILON - HIGH_DOSE * 0.2) / (HIGH_DOSE * 0.8)
    buffered_i = (0.225 + EPSILON + 1e-7) / LOW_DOSE
    half_i = (0.225 + EPSILON / 2.0) / LOW_DOSE
    p0_source1 = (half_i - 0.226 * (1.0 - t_max)) / t_max
    g_source1 = (buffered_i - 0.225003 * (1.0 - t_max)) / t_max
    basis = np.asarray([
        [[0.226, 0.225003, 0.225003, 0.2]],
        [[p0_source1, 0.5, g_source1, 1.0]],
    ], dtype=np.float64)
    target = np.asarray([[1, 1, 1, 0]], dtype=bool)
    anchor = np.asarray([1.0, 0.0], dtype=np.float64)
    poly = constrained.build_polytope([basis], [target], LP_MARGIN, rho=RHO)
    model = build_sparse_milp([basis], [target], anchor, poly,
                              lp_margin=LP_MARGIN, rho=RHO, epsilon=EPSILON,
                              layout_ids=["toy"])
    return basis, target, anchor, poly, model


def _enumerated_minimum_binary_count(model):
    n = model.source_count
    e = model.binary_count
    A = model.matrix.toarray()
    minimum = None
    for bits in itertools.product((0.0, 1.0), repeat=e):
        z = np.asarray(bits, dtype=np.float64)
        adjusted = A[:, n:] @ z if e else np.zeros(model.row_upper.size)
        lower = model.row_lower - adjusted
        upper = model.row_upper - adjusted
        equality = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
        upper_only = np.isfinite(upper) & ~equality
        lower_only = np.isfinite(lower) & ~equality
        A_ub = np.concatenate((A[upper_only, :n], -A[lower_only, :n]), axis=0)
        b_ub = np.concatenate((upper[upper_only], -lower[lower_only]))
        result = linprog(
            np.zeros(n), A_ub=A_ub if b_ub.size else None,
            b_ub=b_ub if b_ub.size else None,
            A_eq=A[equality, :n] if equality.any() else None,
            b_eq=upper[equality] if equality.any() else None,
            bounds=[(0.0, 1.0)] * n, method="highs",
        )
        if result.success:
            count = int(sum(bits))
            minimum = count if minimum is None else min(minimum, count)
    return minimum


def _toy_solver_subprocess(*, initialize_scheduler_with_linprog=False):
    code = '''
import json
from scipy.optimize import linprog
from tests.test_source_pareto import _constrained_toy
from source_pareto import solve_highs
if INITIALIZE:
    linprog([1.0], bounds=[(0.0, 1.0)], method="highs",
            options={"threads": 1})
_, _, _, _, model = _constrained_toy()
row = solve_highs(model, 17, 5.0, require_pinned=False)
print(json.dumps({key: row.get(key) for key in (
    "model_status", "run_status", "objective", "dual_bound", "mip_gap",
    "optimal_zero_gap", "residual", "values", "post_run_error",
    "pass_model_status", "warm_start_status", "options_passed",
    "external_wall_seconds", "solver_wall_seconds")}))
'''.replace("INITIALIZE", repr(bool(initialize_scheduler_with_linprog)))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", code], cwd=project_root, env=env,
                            check=True, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_sparse_milp_matches_exhaustive_binary_enumeration_and_not_exact_pv():
    basis, target, anchor, poly, model = _constrained_toy()
    # The exhaustive LP enumeration may initialize SciPy's process-global
    # HiGHS scheduler. Isolate the MILP toy in a child process so this test
    # worker's earlier LP cannot affect the pinned threads=4 run.
    assert _enumerated_minimum_binary_count(model) == 1
    solution = _toy_solver_subprocess()
    assert solution["model_status"] == "kOptimal"
    assert solution["optimal_zero_gap"] is True
    assert solution["residual"]["passed"] is True
    assert solution["objective"] == 1.0

    weights = np.asarray(solution["values"][:2], dtype=np.float64)
    margins = critical_signed_margins(basis, target, weights).reshape(target.shape)
    # The first original error now prints correctly, but its margin remains
    # below epsilon; its binary is conservatively counted as unrepaired.
    assert 0.0 < margins[0, 0] < EPSILON
    assert int(round(sum(solution["values"][2:]))) == 1
    assert poly.verify(weights, tol=1e-9)["passed"]

    torch_basis = SimpleNamespace(intensities=torch.as_tensor(basis[None], dtype=torch.float32))
    metrics = constrained.diagnostic._metrics(
        {"layout_id": "toy", "split": "fit", "target": torch.as_tensor(target)},
        torch_basis, weights, [0.98, 1.0, 1.02],
    )
    assert metrics["band_pixels"] == 0
    assert model.objective @ np.asarray(solution["values"]) == 1.0
    assert metrics["band_pixels"] != model.objective @ np.asarray(solution["values"])


def test_linprog_scheduler_conflict_is_reproduced_and_isolated_in_child_process():
    conflict = _toy_solver_subprocess(initialize_scheduler_with_linprog=True)
    assert conflict["run_status"] == "kError"
    assert conflict["model_status"] == "kNotset"
    clean = _toy_solver_subprocess()
    assert clean["run_status"] == "kOk"
    assert clean["model_status"] == "kOptimal"


def _smoke_attempt(tmp_path, *, fail_seed=None, calibration_failure=False):
    _, _, _, _, model = _constrained_toy()
    toy_solution = _toy_solver_subprocess()
    report = {"seeds": [], "calibration_status": "closed",
              "attempt_marker": {"path": "marker", "consumed": False}}
    calls = {"solver": [], "identity": [], "claim": 0, "open": 0, "score": 0}
    good = {"model_status": "kOptimal", "run_status": "kOk",
            "pass_model_status": "kOk", "warm_start_status": "kOk",
            "options_passed": True, "objective": toy_solution["objective"],
            "dual_bound": toy_solution["dual_bound"], "mip_gap": 0.0,
            "external_wall_seconds": 0.001, "solver_wall_seconds": 0.001,
            "optimal_zero_gap": True, "incumbent_present": True,
            "values": toy_solution["values"],
            "residual": {"passed": True, "maximum_integrality_violation": 0.0}}

    def solver(solver_model, seed, time_limit, require_pinned):
        calls["solver"].append(seed)
        assert solver_model is model and time_limit <= 360.0 and require_pinned is True
        row = dict(good, seed=seed)
        if fail_seed == seed:
            row.update(model_status="kTimeLimit", optimal_zero_gap=False)
        return row

    def qualifier(row):
        return {"passed": True, "weights": row["values"][:model.source_count],
                "buffered_count_objective": row["objective"],
                "fit_metrics": {"mean": {"band_pixels": 0,
                                              "L2_worst_dose_pixels": 0,
                                              "L2_pixels": 0}}}

    def claim():
        calls["claim"] += 1
        report["attempt_marker"]["consumed"] = True

    def open_calibration(frozen):
        calls["open"] += 1
        assert len(frozen) == 5 and report["fit_selection"]["all_five_seed_weights_frozen"]
        if calibration_failure:
            raise RuntimeError("stub calibration generation failure")
        return {"stub": True}

    def score_calibration(frozen, context):
        calls["score"] += 1
        assert context == {"stub": True} and len(frozen) == 5
        return {"gate": {"passed": False}, "per_seed": []}

    result_path = tmp_path / "attempt.json"
    result = runner._run_registered_attempt(
        report, result_path, __import__("time").monotonic(), model=model,
        fit_rows=[], anchor_fit_metrics={"mean": {"band_pixels": 2}},
        identity_check=lambda stage: calls["identity"].append(stage),
        claim_attempt=claim, solver=solver, qualifier=qualifier,
        open_calibration=open_calibration, score_calibration=score_calibration,
    )
    return result, report, calls, result_path


def test_smoke_runner_freezes_five_fit_candidates_before_calibration():
    with tempfile.TemporaryDirectory() as temporary:
        path, report, calls, result_path = _smoke_attempt(Path(temporary))
        assert path == result_path
        assert calls["solver"] == [17, 29, 43, 71, 101]
        assert calls["claim"] == 1 and calls["open"] == 1 and calls["score"] == 1
        assert len(report["seeds"]) == 5 and report["status"] == "complete"
        assert report["calibration_status"] == "scored_once_after_fit_freeze; no_retry"
        vector = report["seeds"][0]["incumbent_vector"]
        values = np.asarray(vector["values"], dtype="<f8")
        assert vector["sha256_float64_le"] == hashlib.sha256(values.tobytes()).hexdigest()
        assert vector["audit"]["passed"]


def test_smoke_runner_solver_failure_before_five_never_opens_calibration():
    with tempfile.TemporaryDirectory() as temporary:
        _, report, calls, _ = _smoke_attempt(Path(temporary), fail_seed=29)
        assert calls["solver"] == [17, 29]
        assert calls["open"] == calls["score"] == 0
        assert report["calibration_status"] == "closed"
        assert report["fit_eligibility"]["passed"] is False


def test_post_solver_identity_drift_preserves_status_time_and_full_incumbent():
    with tempfile.TemporaryDirectory() as temporary:
        _, _, _, _, model = _constrained_toy()
        toy_solution = _toy_solver_subprocess()
        report = {"seeds": [], "calibration_status": "closed",
                  "attempt_marker": {"path": "marker", "consumed": False}}
        result_path = Path(temporary) / "attempt.json"
        now = __import__("time").monotonic()
        solver_row = {
            "seed": 17, "model_status": "kOptimal", "run_status": "kOk",
            "pass_model_status": "kOk", "warm_start_status": "kOk",
            "options_passed": True, "objective": toy_solution["objective"],
            "dual_bound": toy_solution["dual_bound"], "mip_gap": 0.0,
            "external_wall_seconds": 0.25, "solver_wall_seconds": 0.2,
            "optimal_zero_gap": True, "incumbent_present": True,
            "values": toy_solution["values"],
        }

        def identity_check(stage):
            if stage == "after solver seed 17":
                raise RuntimeError("registered input drifted after solver")

        def fail_calibration(*_args):
            raise AssertionError("calibration must remain closed")

        with unittest.TestCase().assertRaisesRegex(RuntimeError, "drifted after solver"):
            runner._run_registered_attempt(
                report, result_path, now, model=model, fit_rows=[],
                anchor_fit_metrics={"mean": {"band_pixels": 1}},
                identity_check=identity_check,
                claim_attempt=lambda: report["attempt_marker"].update(consumed=True),
                solver=lambda *_args, **_kwargs: solver_row,
                qualifier=lambda row: {
                    "passed": True, "weights": row["values"][:model.source_count],
                    "buffered_count_objective": row["objective"],
                    "fit_metrics": {"mean": {"band_pixels": 0,
                                               "L2_worst_dose_pixels": 0,
                                               "L2_pixels": 0}},
                },
                open_calibration=fail_calibration,
                score_calibration=fail_calibration,
            )
        runner._record_attempt_failure(report, result_path, now,
                                       RuntimeError("registered input drifted after solver"))
        persisted = json.loads(result_path.read_text(encoding="utf-8"))
        row = persisted["seeds"][0]
        vector = row["incumbent_vector"]
        values = np.asarray(vector["values"], dtype="<f8")
        assert row["solver"]["model_status"] == "kOptimal"
        assert persisted["solver_time_spent_seconds"] == 0.25
        assert vector["source_count"] == model.source_count
        assert vector["binary_count"] == model.binary_count
        assert values.size == model.source_count + model.binary_count
        assert vector["sha256_float64_le"] == hashlib.sha256(values.tobytes()).hexdigest()
        assert vector["audit"]["status"] == "pending"
        assert persisted["calibration_status"] == "closed"


def test_qualifier_exception_keeps_persisted_solver_row_vector_and_budget():
    with tempfile.TemporaryDirectory() as temporary:
        _, _, _, _, model = _constrained_toy()
        toy_solution = _toy_solver_subprocess()
        report = {"seeds": [], "calibration_status": "closed",
                  "attempt_marker": {"path": "marker", "consumed": False}}
        result_path = Path(temporary) / "attempt.json"
        started = __import__("time").monotonic()
        solver_row = {
            "seed": 17, "model_status": "kOptimal", "run_status": "kOk",
            "pass_model_status": "kOk", "warm_start_status": "kOk",
            "options_passed": True, "objective": toy_solution["objective"],
            "dual_bound": toy_solution["dual_bound"], "mip_gap": 0.0,
            "external_wall_seconds": 0.4, "solver_wall_seconds": 0.35,
            "optimal_zero_gap": True, "incumbent_present": True,
            "values": toy_solution["values"],
        }

        def qualifier(_row):
            persisted = json.loads(result_path.read_text(encoding="utf-8"))
            saved = persisted["seeds"][0]
            assert saved["solver"]["model_status"] == "kOptimal"
            assert persisted["solver_time_spent_seconds"] == 0.4
            assert saved["incumbent_vector"]["audit"]["passed"]
            assert persisted["attempt_marker"]["consumed"] is True
            raise RuntimeError("stub GPU qualifier failure")

        def fail_calibration(*_args):
            raise AssertionError("calibration must stay closed")

        with unittest.TestCase().assertRaisesRegex(RuntimeError, "stub GPU qualifier"):
            runner._run_registered_attempt(
                report, result_path, started, model=model, fit_rows=[],
                anchor_fit_metrics={"mean": {"band_pixels": 1}},
                identity_check=lambda _stage: None,
                claim_attempt=lambda: report["attempt_marker"].update(consumed=True),
                solver=lambda *_args, **_kwargs: solver_row,
                qualifier=qualifier, open_calibration=fail_calibration,
                score_calibration=fail_calibration,
            )
        runner._record_attempt_failure(report, result_path, started,
                                      RuntimeError("stub GPU qualifier failure"))
        persisted = json.loads(result_path.read_text(encoding="utf-8"))
        saved = persisted["seeds"][0]
        vector = saved["incumbent_vector"]
        values = np.asarray(vector["values"], dtype="<f8")
        assert saved["fit"]["status"] == "qualification_pending"
        assert vector["sha256_float64_le"] == hashlib.sha256(values.tobytes()).hexdigest()
        assert vector["audit"]["passed"]
        assert persisted["status"] == "failed; calibration_closed"
        assert persisted["calibration_status"] == "closed"
        assert persisted["attempt_marker"]["consumed"] is True


def test_run_registered_attempt_call_binds_to_helper_signature():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    run_node = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "run")
    calls = [node for node in ast.walk(run_node)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "_run_registered_attempt"]
    assert len(calls) == 1
    keyword_names = {item.arg for item in calls[0].keywords if item.arg is not None}
    inspect.signature(runner._run_registered_attempt).bind(
        object(), object(), object(), **{name: object() for name in keyword_names}
    )


def test_calibration_failures_keep_opened_no_retry_and_preopen_failures_stay_closed():
    with tempfile.TemporaryDirectory() as temporary:
        report = {"seeds": [], "calibration_status": "closed",
                  "attempt_marker": {"consumed": False}}
        result_path = Path(temporary) / "before.json"
        runner._record_attempt_failure(report, result_path, __import__("time").monotonic(),
                                       ValueError("preflight"))
        assert report["calibration_status"] == "closed"

        report = {"seeds": [], "calibration_status": "opened_no_retry; in_progress",
                  "calibration_attempt_consumed": True}
        result_path = Path(temporary) / "after.json"
        runner._record_attempt_failure(report, result_path, __import__("time").monotonic(),
                                       RuntimeError("calibration failed"))
        assert report["calibration_status"] == "opened_then_failed; no_retry"
        assert report["status"] == "failed; opened_then_failed; no_retry"


def test_plan_sha_attempt_marker_is_exclusive_and_independent_of_output_root():
    with tempfile.TemporaryDirectory() as temporary:
        plan_path = Path(temporary) / "frozen.json"
        plan_path.write_bytes(b"plan bytes")
        plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        marker = runner.claim_solver_attempt(plan_path, plan_sha)
        marker_path = Path(marker["path"])
        assert marker_path == runner.attempt_marker_path(plan_path, plan_sha)
        assert json.loads(marker_path.read_text())["candidate_plan_sha256"] == plan_sha
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "already consumed"):
            runner.claim_solver_attempt(plan_path, plan_sha)
        assert marker_path.is_file()
        different_sha = "f" * 64
        assert runner.ensure_attempt_available(plan_path, different_sha).exists() is False


def test_softcount_requires_exact_hinge_manifest_and_legacy_hinge_rows_use_header_head():
    with tempfile.TemporaryDirectory() as temporary:
        hinge_path = Path(temporary) / "hinge.json"
        hinge_path.write_text("{}", encoding="utf-8")
        hinge_sha = hashlib.sha256(hinge_path.read_bytes()).hexdigest()
        assert runner._validate_softcount_predecessor(
            {"prerequisite_manifest": {"path": str(hinge_path.resolve()), "sha256": hinge_sha}},
            hinge_path, hinge_sha,
        ) == {"path": str(hinge_path.resolve()), "sha256": hinge_sha}
        with unittest.TestCase().assertRaisesRegex(ValueError, "does not match"):
            runner._validate_softcount_predecessor(
                {"prerequisite_manifest": {"path": str(hinge_path.resolve()), "sha256": "bad"}},
                hinge_path, hinge_sha,
            )

        legacy = {"code_commit": "hinge-head", "runs": []}
        outer = []
        for index in range(3):
            run_hash = str(index)
            legacy["runs"].append({
                "candidate_index": index, "rho": (0.5, 0.05, 0.01)[index],
                "run_directory": "run-%d" % index,
                "protocol_sha256": "p" + run_hash, "run_state_sha256": "s" + run_hash,
                "identity_sha256": "i" + run_hash, "status": "complete",
                "gate_passed": False, "seeds": [{"seed": s} for s in (17, 29, 43, 71, 101)],
            })
            outer.append({**legacy["runs"][-1], "kind": "hinge",
                          "git_head": "hinge-head"})
        runner._validate_outer_hinge_entries(outer, legacy)
        assert all("git_head" not in row for row in legacy["runs"])


def test_identity_recheck_detects_file_drift_after_snapshot():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = root / "plan.json"
        dataset = root / "dataset.pt"
        diagnostic = root / "diagnostic.json"
        artifact = root / "run_state.json"
        plan.write_text(json.dumps({"base_commit": "dummy"}), encoding="utf-8")
        dataset.write_bytes(b"dataset")
        dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
        diagnostic.write_text(json.dumps({"input": {"dataset_sha256": dataset_sha}}),
                              encoding="utf-8")
        artifact.write_text('{"state": true}', encoding="utf-8")
        plan_sha_bytes = plan.read_bytes()
        provenance = {"head": "fixed", "clean_tree": True, "source_sha256": {}}
        prerequisite = {"artifact_paths": [{"path": str(artifact),
                                              "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]}
        plan_object = {"base_commit": "dummy"}
        diagnostic_object, diagnostic_sha, _ = runner._read_hashed_json(diagnostic)
        dataset.write_bytes(b"newer dataset content")
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "registered preflight SHA256"):
            runner._identity_snapshot(
                plan, plan_sha_bytes, plan_object, dataset, dataset_sha, diagnostic,
                diagnostic_object, diagnostic_sha, provenance, prerequisite,
            )
        dataset.write_bytes(b"dataset")
        snapshot = runner._identity_snapshot(
            plan, plan_sha_bytes, plan_object, dataset, dataset_sha, diagnostic,
            diagnostic_object, diagnostic_sha, provenance, prerequisite,
        )
        with mock.patch.object(runner, "git_provenance", return_value=provenance):
            runner._recheck_identity(snapshot, plan, plan_object, dataset, diagnostic,
                                     prerequisite, "before test")
            diagnostic.write_bytes(b"changed")
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "identity drifted"):
                runner._recheck_identity(snapshot, plan, plan_object, dataset, diagnostic,
                                         prerequisite, "after test")


def test_identity_snapshot_covers_retry_and_ancestor_lineage_artifacts():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        lineage = []
        for name in ("listed", "same-rho-retry", "origin"):
            run_dir = root / name
            run_dir.mkdir()
            protocol = run_dir / "protocol.json"
            state = run_dir / "run_state.json"
            protocol.write_text(json.dumps({"run": name}), encoding="utf-8")
            state.write_text(json.dumps({"state": name}), encoding="utf-8")
            lineage.append({
                "path": str(run_dir.resolve()),
                "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                "run_state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
            })
        artifacts = runner._lineage_artifact_paths(lineage)
        assert len(artifacts) == 6
        assert {Path(item["path"]).parent.name for item in artifacts} == {
            "listed", "same-rho-retry", "origin",
        }
        with unittest.TestCase().assertRaisesRegex(ValueError, "conflicting SHA256"):
            runner._merge_artifact_paths([
                artifacts[0], {**artifacts[0], "sha256": "f" * 64},
            ])

        plan = root / "plan.json"
        dataset = root / "dataset.pt"
        diagnostic = root / "diagnostic.json"
        plan_object = {"base_commit": "dummy"}
        plan.write_text(json.dumps(plan_object), encoding="utf-8")
        dataset.write_bytes(b"dataset")
        dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
        diagnostic_object = {"input": {"dataset_sha256": dataset_sha}}
        diagnostic.write_text(json.dumps(diagnostic_object), encoding="utf-8")
        diagnostic_parsed, diagnostic_sha, _ = runner._read_hashed_json(diagnostic)
        provenance = {"head": "fixed", "clean_tree": True, "source_sha256": {}}
        prerequisite = {"artifact_paths": artifacts}
        snapshot = runner._identity_snapshot(
            plan, plan.read_bytes(), plan_object, dataset, dataset_sha, diagnostic,
            diagnostic_parsed, diagnostic_sha, provenance, prerequisite,
        )
        retry_state = root / "same-rho-retry" / "run_state.json"
        with mock.patch.object(runner, "git_provenance", return_value=provenance):
            runner._recheck_identity(snapshot, plan, plan_object, dataset, diagnostic,
                                     prerequisite, "before retry drift")
            retry_state.write_text('{"state": "modified"}', encoding="utf-8")
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "identity drifted"):
                runner._recheck_identity(snapshot, plan, plan_object, dataset, diagnostic,
                                         prerequisite, "after retry drift")


def test_tight_big_m_and_protected_row_pruning_use_only_certified_bounds():
    target = np.asarray([[1, 1, 0]], dtype=bool)
    basis = np.asarray([[[0.5, 0.226, 0.1]], [[0.6, 0.5, 0.2]]], dtype=np.float64)
    anchor = np.asarray([0.0, 1.0])
    poly = constrained.build_polytope([basis], [target], LP_MARGIN, rho=RHO)
    model = build_sparse_milp([basis], [target], anchor, poly,
                              lp_margin=LP_MARGIN, rho=RHO, epsilon=EPSILON,
                              layout_ids=["safe-prune"])
    assert model.binary_count == 0
    assert model.protected_pruned_simplex == 2
    assert model.protected_pruned_nominal == 0
    assert model.protected_count == 1
    assert model.nominal_original_row_count == 3
    assert model.nominal_pruned_protected_count == 3
    assert model.nominal_row_count == 0
    assert check_model_residual(model, model.anchor_start)["passed"]

    error_target = np.asarray([[1]], dtype=bool)
    error_basis = np.asarray([[[0.226]], [[0.28]]], dtype=np.float64)
    error_poly = constrained.build_polytope([error_basis], [error_target], LP_MARGIN, rho=RHO)
    error_model = build_sparse_milp([error_basis], [error_target], [1.0, 0.0], error_poly,
                                    lp_margin=LP_MARGIN, rho=RHO, epsilon=EPSILON,
                                    layout_ids=["tight-M"])
    simplex_lb = LOW_DOSE * min(0.226, 0.28) - 0.225
    nominal_lb = LOW_DOSE * (RHO * LP_MARGIN) - (1.0 - LOW_DOSE) * 0.225
    assert np.isclose(error_model.binary_big_m[0], EPSILON - max(simplex_lb, nominal_lb),
                      rtol=0.0, atol=1e-15)
    assert error_model.anchor_start[-1] == 1.0
    assert check_model_residual(error_model, error_model.anchor_start)["passed"]


def test_nominal_rows_are_pruned_only_after_exact_polytope_partition_check():
    basis, target, anchor, poly, model = _constrained_toy()
    assert model.nominal_original_row_count == int(target.size)
    assert model.nominal_row_count == model.binary_count == 3
    assert model.nominal_pruned_protected_count == 1
    assert model.nominal_pixel_indices == model.binary_pixel_indices
    assert len(model.protected_pixel_indices) + len(
        model.protected_pruned_simplex_pixel_indices
    ) + len(model.protected_pruned_nominal_pixel_indices) == 1

    feasible_samples = 0
    for first_weight in np.linspace(0.0, 1.0, 1001):
        weights = np.asarray([first_weight, 1.0 - first_weight])
        values = np.concatenate((weights, np.ones(model.binary_count)))
        non_nominal_activity = model.matrix[model.nominal_row_count:-1] @ values
        non_nominal_upper = model.row_upper[model.nominal_row_count:-1]
        if np.all(non_nominal_activity <= non_nominal_upper + 1e-12):
            feasible_samples += 1
            assert poly.verify(weights, tol=1e-12)["passed"]
    assert feasible_samples > 0

    altered = SimpleNamespace(
        matrix=poly.matrix.copy(), labels=poly.labels.copy(),
        margin_floor=poly.margin_floor, aub=poly.aub.copy(), bub=poly.bub.copy(),
        aeq=poly.aeq.copy(), beq=poly.beq.copy(), bounds=poly.bounds,
    )
    altered.labels[0] = ~altered.labels[0]
    with unittest.TestCase().assertRaisesRegex(ValueError, "matrix, labels, order, or RHS"):
        build_sparse_milp([basis], [target], anchor, altered,
                          lp_margin=LP_MARGIN, rho=RHO, epsilon=EPSILON,
                          layout_ids=["toy"])


def _good_solver_record():
    return {
        "model_status": "kOptimal", "run_status": "kOk",
        "pass_model_status": "kOk", "warm_start_status": "kOk",
        "objective": 1.0, "dual_bound": 1.0, "mip_gap": 0.0,
        "options_passed": True,
        "residual": {"passed": True, "maximum_integrality_violation": 0.0},
    }


def test_solver_incumbent_or_bad_solver_record_fails_closed():
    changes_list = [
        {"model_status": "kTimeLimit"}, {"dual_bound": 0.0}, {"mip_gap": 0.01},
        {"options_passed": False}, {"run_status": "kError"},
        {"residual": {"passed": False, "maximum_integrality_violation": 0.0}},
        {"residual": {"passed": True, "maximum_integrality_violation": 1e-4}},
    ]
    for changes in changes_list:
        record = {**_good_solver_record(), **changes}
        assert validate_solver_record(record)["passed"] is False


def test_solver_record_and_all_five_seed_budget_are_required_for_calibration():
    good = _good_solver_record()
    assert validate_solver_record(good)["passed"]
    rows = [{"seed": seed, "fit_qualified": True, "solver": dict(good)}
            for seed in (17, 29, 43, 71, 101)]
    assert five_seed_eligibility(rows)["passed"]
    assert five_seed_eligibility(rows[:-1])["calibration_status"] == "closed"
    rows[-1] = {**rows[-1], "seed": 99}
    assert five_seed_eligibility(rows)["passed"] is False


def test_calibration_generator_is_unreachable_before_five_seed_freeze():
    with mock.patch.object(runner.experiment, "generate_calibration") as generate:
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "before all five FIT"):
            runner._generate_calibration_after_fit_freeze(
                SimpleNamespace(type="cuda"), {"input": {"dataset_sha256": "x"}}, "x", []
            )
    generate.assert_not_called()


def test_fit_loader_indexes_only_fit_payload_and_never_final3():
    accessed = []

    class Payload(dict):
        def __getitem__(self, key):
            accessed.append(key)
            if key == "final3":
                raise AssertionError("final3 must remain closed")
            return super().__getitem__(key)

    payload = Payload(fit={
        "masks": torch.zeros((1, 1, 2, 2)),
        "targets": torch.zeros((1, 1, 2, 2)),
        "layout_ids": ["fit_only"],
        "pixel_size_nm": 4.0,
    }, final3=object(), calibration=object())
    with mock.patch.object(constrained.torch, "load", return_value=payload):
        loaded = constrained.load_fit("unused.pt")
    assert loaded.layout_ids == ("fit_only",)
    assert accessed == ["fit"]


def _frozen_plan(tmp_path):
    dataset = tmp_path / "data.pt"
    diagnostic = tmp_path / "diagnostic.json"
    prereq = tmp_path / "source_family_manifest.json"
    plan_path = tmp_path / "plan.json"
    dataset.write_bytes(b"dataset")
    diagnostic.write_text("{}", encoding="utf-8")
    prereq.write_text("{}", encoding="utf-8")
    plan = {
        "schema_version": 4,
        "status": "prospective_candidate_plan",
        "objective": runner.OBJECTIVE_ID,
        "base_commit": "a71d3ed465464b173231f5be2023250e489df809",
        "dataset": str(dataset.resolve()),
        "diagnostic": str(diagnostic.resolve()),
        "prerequisite_manifest": str(prereq.resolve()),
        "protocol": runner.EXPECTED_PROTOCOL,
        "scope": runner.EXPECTED_SCOPE,
        **runner.EXPECTED_TEXT,
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan, plan_path, dataset.resolve(), diagnostic.resolve()


def test_candidate_plan_protocol_and_runtime_identity_are_pinned():
    with tempfile.TemporaryDirectory() as temporary:
        plan, plan_path, dataset, diagnostic = _frozen_plan(Path(temporary))
        original_pin = runner.PINNED_PLAN_SHA256
        try:
            runner.PINNED_PLAN_SHA256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            result = runner.validate_candidate_plan(plan, plan_path, dataset, diagnostic)
            assert result["sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()

            copied_directory = Path(temporary) / "copied"
            copied_directory.mkdir()
            copied_plan = copied_directory / plan_path.name
            copied_plan.write_bytes(plan_path.read_bytes())
            with unittest.TestCase().assertRaisesRegex(ValueError, "share a directory"):
                runner.validate_candidate_plan(
                    plan, copied_plan, dataset, diagnostic,
                    plan_bytes=copied_plan.read_bytes(),
                )

            drift = json.loads(json.dumps(plan))
            drift["protocol"]["epsilon"] = 1e-8
            plan_path.write_text(json.dumps(drift), encoding="utf-8")
            runner.PINNED_PLAN_SHA256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            with unittest.TestCase().assertRaisesRegex(ValueError, "fixed solver, seed, margin, or gate"):
                runner.validate_candidate_plan(drift, plan_path, dataset, diagnostic)
        finally:
            runner.PINNED_PLAN_SHA256 = original_pin


def test_runtime_gate_constants_must_match_frozen_expected_gates():
    assert runner._validate_runtime_gate_constants() == runner.EXPECTED_GATES
    changed_gate = dict(constrained.GATE)
    changed_gate["L2_pixels"] += 0.01
    with mock.patch.object(constrained, "GATE", changed_gate):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "differ from the frozen"):
            runner._validate_runtime_gate_constants()
    changed_lp_cal = dict(constrained.LP_CAL)
    changed_lp_cal["band_pixels"] += 1.0
    with mock.patch.object(constrained, "LP_CAL", changed_lp_cal):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "differ from the frozen"):
            runner._validate_runtime_gate_constants()


def test_manifest_hash_and_data_identity_tampering_fail_before_run_validation():
    with tempfile.TemporaryDirectory() as temporary:
        manifest_path = Path(temporary) / "manifest.json"
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "status": "completed_source_families_gate_failed",
            "dataset_sha256": "dataset-hash",
            "diagnostic_file_sha256": "diagnostic-hash",
        }), encoding="utf-8")
        expected_inputs = {"dataset_sha256": "different", "diagnostic_file_sha256": "diagnostic-hash"}
        with unittest.TestCase().assertRaisesRegex(ValueError, "SHA256 differs"):
            runner.validate_prerequisite_manifest(manifest_path, "bad-hash", expected_inputs)
        with unittest.TestCase().assertRaisesRegex(ValueError, "header/data identity"):
            runner.validate_prerequisite_manifest(manifest_path, None, expected_inputs)


def test_manifest_run_entry_rejects_missing_artifact_and_identity():
    with tempfile.TemporaryDirectory() as temporary:
        entry = {
            "kind": "softcount", "candidate_index": 0, "rho": 0.5,
            "status": "complete", "gate_passed": False,
            "run_directory": str(Path(temporary) / "run"),
        }
        with unittest.TestCase().assertRaisesRegex(ValueError, "missing protocol"):
            runner._validate_family_run(entry, "softcount", 0, 0.5, {}, "plan-sha")


def test_registered_runtime_pin_refuses_local_version_drift_without_fallback():
    import scipy
    if scipy.__version__ != "1.17.1":
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "requires scipy 1.17.1"):
            __import__("source_pareto").pinned_solver_backend(require_pinned=True)


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            suite.addTest(unittest.FunctionTestCase(function, description=name))
    return suite
