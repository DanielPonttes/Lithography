"""CPU-only invariant checks for the constrained source optimizer.

Run with unittest discovery. These tests use tiny synthetic bases; they never
load the registered dataset, train on the fit split, or access final3.
"""
import time
import unittest
from pathlib import Path
import sys
import tempfile
import json
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from unittest import mock

import numpy as np
import torch

from scripts import optimize_source_constrained as opt


def test_fit_loader_never_indexes_other_split_keys():
    touched = []

    class Guard(dict):
        def __getitem__(self, key):
            touched.append(key)
            if key == "final_test":
                raise AssertionError("final3 key was accessed")
            return super().__getitem__(key)

    fit = {
        "masks": torch.tensor([[[[1.0, 0.0]]]]),
        "targets": torch.tensor([[[[1.0, 0.0]]]]),
        "layout_ids": ["fit_only"],
        "pixel_size_nm": 4.0,
    }
    payload = Guard(fit=fit, final_test=object())
    with mock.patch.object(opt.torch, "load", return_value=payload):
        loaded = opt.load_fit("unused.pt")
    assert loaded.layout_ids == ("fit_only",)
    assert touched == ["fit"]


def test_constraints_preserve_flux_nonnegativity_margin_and_convex_mixes():
    bases = np.array([[[0.8, 0.1]], [[0.7, 0.15]]], dtype=np.float64)
    target = np.array([[1, 0]], dtype=np.uint8)
    matrix, labels = opt.fit_matrix([bases], [target])
    anchor = np.array([0.5, 0.5], dtype=np.float64)
    lp_margin = opt.signed_margin(matrix, labels, anchor)
    poly = opt.build_polytope([bases], [target], lp_margin, rho=0.5)
    left = np.array([1.0, 0.0])
    right = np.array([0.0, 1.0])
    assert poly.verify(anchor)["passed"]
    assert poly.verify(left)["passed"]
    assert poly.verify(right)["passed"]
    mixed = 0.63 * left + 0.37 * right
    check = poly.verify(mixed)
    assert check["passed"]
    assert abs(float(mixed.sum()) - 1.0) < 1e-12
    assert mixed.min() >= 0
    assert poly.verify(np.array([1.1, -0.1]))["passed"] is False


def test_edge_contrast_pairs_are_oriented_and_indexed_per_layout():
    first = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    second = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    bases = np.array([
        [[0.7, 0.2], [0.3, 0.1]],
        [[0.5, 0.1], [0.4, 0.2]],
    ], dtype=np.float64)
    matrix, _ = opt.fit_matrix([bases, bases], [first, second])
    contrasts = opt.edge_contrast_matrix(matrix, [first, second])
    assert contrasts.ndim == 2 and contrasts.shape[1] == 2
    assert np.isfinite(contrasts).all()
    # The first horizontal edge is foreground-minus-background for pixel 0→1.
    np.testing.assert_allclose(contrasts[0], [0.5, 0.4], atol=1e-12)
    assert len(contrasts) >= 4


def test_edge_lp_solves_auxiliary_minimum_contrast_and_stays_feasible():
    bases = np.array([[[0.8, 0.1]], [[0.7, 0.2]]], dtype=np.float64)
    target = np.array([[1, 0]], dtype=np.uint8)
    matrix, labels = opt.fit_matrix([bases], [target])
    anchor = np.array([0.5, 0.5], dtype=np.float64)
    margin = opt.signed_margin(matrix, labels, anchor)
    poly = opt.build_polytope([bases], [target], margin)
    contrasts = opt.edge_contrast_matrix(matrix, [target])
    result = opt.solve_edge_lp(poly, contrasts, time.monotonic() + 10.0, 5.0)
    assert result["status"] == "optimal_verified"
    assert poly.verify(result["weights"])["passed"]
    actual = float(np.min(contrasts @ result["weights"]))
    assert abs(float(result["objective_value"]) + actual) < 2e-8
    assert np.all(contrasts @ result["weights"] >= actual - 1e-10)


def test_smooth_objective_gradient_matches_finite_difference():
    basis = torch.tensor([
        [[0.22, 0.205], [0.24, 0.19]],
        [[0.23, 0.195], [0.21, 0.215]],
    ], dtype=torch.float64)
    source = {"tiny": basis}
    weights = np.array([0.4, 0.6], dtype=np.float64)
    value, grad = opt.smooth_gradient(weights, source, beta=200.0)
    assert np.isfinite(value) and np.isfinite(grad).all()
    eps = 1e-6
    for index in range(2):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical = (opt.smooth_value(plus, source, 200.0)
                     - opt.smooth_value(minus, source, 200.0)) / (2 * eps)
        assert abs(float(grad[index]) - numerical) < 2e-5


def test_lmo_returns_independently_verified_source():
    bases = np.array([[[0.8, 0.1]], [[0.7, 0.15]]], dtype=np.float64)
    target = np.array([[1, 0]], dtype=np.uint8)
    matrix, labels = opt.fit_matrix([bases], [target])
    anchor = np.array([0.5, 0.5])
    poly = opt.build_polytope([bases], [target],
                              opt.signed_margin(matrix, labels, anchor))
    result = opt.solve_lmo(poly, np.array([1.0, -1.0]),
                           time.monotonic() + 10.0, time_limit=5.0)
    assert result["status"] == "optimal_verified"
    assert poly.verify(result["weights"])["passed"]
    assert result["weights"].min() >= 0
    assert abs(float(result["weights"].sum()) - 1.0) < 2e-8


def test_fw_callback_handles_event_field_and_emits_block_end_candidate():
    bases = np.array([[[0.8, 0.1]], [[0.7, 0.15]]], dtype=np.float64)
    target = np.array([[1, 0]], dtype=np.uint8)
    matrix, labels = opt.fit_matrix([bases], [target])
    anchor = np.array([0.5, 0.5], dtype=np.float64)
    poly = opt.build_polytope([bases], [target], opt.signed_margin(matrix, labels, anchor))
    fit_rows = [{"layout_id": "tiny", "split": "fit", "target": torch.tensor(target)}]
    basis32 = {"tiny": type("Basis", (), {
        "intensities": torch.tensor(bases[None], dtype=torch.float32)
    })()}
    basis_gpu = {"tiny": torch.tensor(bases, dtype=torch.float64)}
    calls = [0]

    def fake_lmo(_poly, _objective, *_args, **_kwargs):
        weights = np.array([1.0, 0.0]) if calls[0] == 0 else np.array([0.0, 1.0])
        calls[0] += 1
        return {"status": "optimal_verified", "weights": weights, "attempts": []}

    def fake_candidate(weights, order, _poly, *_args):
        return {"weights": np.asarray(weights).copy(), "checkpoint_order": order,
                "fit_qualified": True, "smooth_beta800": 1.0,
                "polytope": _poly.verify(weights),
                "fit_metrics": {"mean": {"band_pixels": 0.0,
                                             "L2_worst_dose_pixels": 0.0},
                                "per_layout": [{"L2_pixels": 0}]}}

    with mock.patch.object(opt, "solve_lmo", side_effect=fake_lmo), \
         mock.patch.object(opt, "candidate_row", side_effect=fake_candidate), \
         mock.patch.object(opt, "smooth_value", return_value=1.0), \
         mock.patch.object(opt, "smooth_gradient", return_value=(1.0, np.array([1.0, -1.0]))), \
         mock.patch.object(opt, "line_search", return_value={
             "accepted": True, "gamma": 0.5, "value": 0.9, "evaluations": 1
         }):
        events = []
        def progress(detail):
            opt.report_fw_checkpoint(
                lambda event, **fields: events.append((event, fields)), 17, detail
            )
        record, best = opt.fw_seed(
            17, anchor, poly, fit_rows, basis32, basis_gpu,
            time.monotonic() + 10, 5.0, 1, 1, progress,
        )
    assert best is not None and record["status"] == "complete"
    assert events and all(event == "frank_wolfe_checkpoint" for event, _ in events)
    assert any(fields.get("checkpoint") == "block_end" for _, fields in events)


def test_source_grid_rejects_flux_or_out_of_support_weights():
    support = opt.support_mask()
    grid = np.zeros(support.shape, dtype=np.float64)
    grid[support] = 1.0 / int(support.sum())
    np.testing.assert_allclose(opt.compress(grid, support).sum(), 1.0)
    bad = grid.copy()
    bad[0, 0] = 1.0
    if not support[0, 0]:
        try:
            opt.compress(bad, support)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported source weight must be rejected")
    negative_with_unit_flux = grid.copy()
    positive_index, negative_index = np.flatnonzero(support)[:2]
    negative_with_unit_flux[tuple(np.argwhere(support)[0])] = -1e-12
    negative_with_unit_flux[tuple(np.argwhere(support)[1])] += 1e-12
    try:
        opt.compress(negative_with_unit_flux, support)
    except ValueError:
        pass
    else:
        raise AssertionError("negative source weight must be rejected")
    with_flux_error = grid * 0.5
    try:
        opt.compress(with_flux_error, support)
    except ValueError:
        pass
    else:
        raise AssertionError("non-unit source flux must be rejected")


def test_fw_restores_previous_point_when_float32_nominal_gate_fails():
    lp_bases = np.array([[[0.8, 0.1]], [[0.7, 0.15]]], dtype=np.float64)
    target = np.array([[1, 0]], dtype=np.uint8)
    matrix, labels = opt.fit_matrix([lp_bases], [target])
    anchor = np.array([0.5, 0.5], dtype=np.float64)
    poly = opt.build_polytope([lp_bases], [target], opt.signed_margin(matrix, labels, anchor))
    fit_rows = [{"layout_id": "tiny", "split": "fit", "target": torch.tensor(target)}]
    # In this adversarial precision check, the second float32 source sample
    # creates a false positive although the synthetic LP oracle remains feasible.
    hard_basis = np.array([[[0.8, 0.1]], [[0.5, 0.34]]], dtype=np.float32)
    basis32 = {"tiny": type("Basis", (), {
        "intensities": torch.tensor(hard_basis[None], dtype=torch.float32)
    })()}
    basis_gpu = {"tiny": torch.tensor(lp_bases, dtype=torch.float64)}
    calls, candidate_weights = [0], []

    def fake_lmo(_poly, _objective, *_args, **_kwargs):
        weights = np.array([1.0, 0.0]) if calls[0] == 0 else np.array([0.0, 1.0])
        calls[0] += 1
        return {"status": "optimal_verified", "weights": weights, "attempts": []}

    def fake_candidate(weights, order, _poly, *_args):
        candidate_weights.append(np.asarray(weights).copy())
        return {"weights": np.asarray(weights).copy(), "checkpoint_order": order,
                "fit_qualified": True, "smooth_beta800": 1.0,
                "polytope": _poly.verify(weights),
                "fit_metrics": {"mean": {"band_pixels": 0.0,
                                             "L2_worst_dose_pixels": 0.0},
                                "per_layout": [{"L2_pixels": 0}]}}

    with mock.patch.object(opt, "solve_lmo", side_effect=fake_lmo), \
         mock.patch.object(opt, "candidate_row", side_effect=fake_candidate), \
         mock.patch.object(opt, "smooth_value", return_value=1.0), \
         mock.patch.object(opt, "smooth_gradient", return_value=(1.0, np.array([1.0, -1.0]))), \
         mock.patch.object(opt, "line_search", return_value={
             "accepted": True, "gamma": 0.5, "value": 0.9, "evaluations": 1
         }):
        record, _best = opt.fw_seed(
            17, anchor, poly, fit_rows, basis32, basis_gpu,
            time.monotonic() + 10, 5.0, 1, 1, lambda _detail: None,
        )
    assert record["status"] == "numerical_error"
    expected_previous = 0.95 * anchor + 0.05 * np.array([1.0, 0.0])
    np.testing.assert_allclose(candidate_weights[-1], expected_previous, atol=1e-12)
    assert not record["blocks"][0]["steps"][0]["float32_nominal_fit"]["passed"]


def test_post_creation_error_is_written_to_result_and_progress():
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        opt.atomic_json(out / "results.json", {"status": "running", "preserved": True})
        opt.atomic_json(out / "progress.json", {"status": "running"})
        opt.persist_run_error(out, RuntimeError("synthetic failure"))
        result = json.loads((out / "results.json").read_text(encoding="utf-8"))
        progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
        assert result["status"] == progress["status"] == "error"
        assert result["preserved"] is True
        assert result["error"]["message"] == "synthetic failure"


def test_fixed_control_normalizes_only_float32_flux_roundoff():
    support = opt.support_mask()
    raw = np.zeros(support.shape, dtype=np.float32)
    raw[support] = np.float32(1.0 / int(support.sum()))
    raw_sum = float(raw[support].astype(np.float64).sum())
    assert abs(raw_sum - 1.0) < 8 * np.finfo(np.float32).eps
    weights, metadata = opt.compress_fixed_control(raw, support)
    assert metadata["raw_float32_grid_flux_sum"] == raw_sum
    assert metadata["normalization_applied"]
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    np.testing.assert_allclose(weights, raw[support].astype(np.float64) / raw_sum, rtol=0, atol=1e-12)
    with np.testing.assert_raises(ValueError):
        opt.compress(raw, support)  # LP and candidate tolerance remains strict.
    with np.testing.assert_raises(ValueError):
        opt.compress_fixed_control(raw * 0.5, support)
    outside = raw.astype(np.float64)
    outside[0, 0] = 1e-9
    if not support[0, 0]:
        with np.testing.assert_raises(ValueError):
            opt.compress_fixed_control(outside, support)


def test_calibration_gate_is_preregistered_and_checks_each_seed():
    anchor = {"mean": {"band_pixels": 268.0}}
    passing = {
        "mean": {"band_pixels": 230.0, "L2_pixels": 54.0,
                 "L2_worst_dose_pixels": 150.0},
        "per_seed_band_pixels": [228.0, 232.0],
    }
    assert opt._gate(passing, anchor, True)["passed"]
    one_bad_seed = dict(passing, per_seed_band_pixels=[228.0, 268.0])
    assert not opt._gate(one_bad_seed, anchor, True)["passed"]
    failed_fidelity = dict(passing, mean=dict(passing["mean"], L2_pixels=56.176))
    assert not opt._gate(failed_fidelity, anchor, True)["passed"]


def _resume_fw_fixture():
    bases = np.array([[[0.8, 0.1]], [[0.7, 0.2]]], dtype=np.float64)
    target = np.array([[1, 0]], dtype=np.uint8)
    matrix, labels = opt.fit_matrix([bases], [target])
    anchor = np.array([0.5, 0.5], dtype=np.float64)
    poly = opt.build_polytope([bases], [target], opt.signed_margin(matrix, labels, anchor))
    fit_rows = [{"layout_id": "tiny", "split": "fit", "target": torch.tensor(target)}]
    basis32 = {"tiny": type("Basis", (), {
        "intensities": torch.tensor(bases[None], dtype=torch.float32)
    })()}
    basis_gpu = {"tiny": torch.tensor(bases, dtype=torch.float64)}
    support = np.ones((1, 2), dtype=bool)

    def fake_candidate(weights, order, candidate_poly, *_args):
        weights = np.asarray(weights, dtype=np.float64).copy()
        return {"weights": weights, "checkpoint_order": order,
                "fit_qualified": True, "smooth_beta800": float(weights[1]),
                "polytope": candidate_poly.verify(weights),
                "fit_metrics": {"mean": {"band_pixels": float(100 * weights[0]),
                                             "L2_pixels": 0.0,
                                             "L2_worst_dose_pixels": float(50 * weights[0])},
                                "per_layout": [{"L2_pixels": 0}]}}

    mocks = [
        mock.patch.object(opt, "solve_lmo", return_value={
            "status": "optimal_verified", "weights": np.array([0.0, 1.0]), "attempts": []}),
        mock.patch.object(opt, "candidate_row", side_effect=fake_candidate),
        mock.patch.object(opt, "smooth_value", return_value=1.0),
        mock.patch.object(opt, "smooth_gradient", return_value=(1.0, np.array([1.0, -1.0]))),
        mock.patch.object(opt, "line_search", return_value={
            "accepted": True, "gamma": 0.2, "value": 0.8, "evaluations": 1}),
        mock.patch.object(opt, "nominal_fit_check", return_value={"passed": True}),
    ]
    return anchor, poly, fit_rows, basis32, basis_gpu, support, mocks


def _run_resume_fw_case(resume_state=None, state_callback=None, interval=1):
    anchor, poly, fit_rows, basis32, basis_gpu, _support, mocks = _resume_fw_fixture()
    with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5]:
        return opt.fw_seed(17, anchor, poly, fit_rows, basis32, basis_gpu,
                           time.monotonic() + 20, 5.0, 2, interval, lambda _event: None,
                           resume_state=resume_state, state_callback=state_callback)


def test_fw_resume_after_periodic_and_block_end_interrupts_matches_uninterrupted():
    expected_record, expected_best = _run_resume_fw_case(interval=1)
    for interrupt_kind in ("periodic", "block_end"):
        snapshots = []

        class Interrupted(RuntimeError):
            pass

        def checkpoint(snapshot):
            snapshots.append(json.loads(json.dumps(snapshot)))
            if snapshot.get("checkpoint_kind") == interrupt_kind:
                raise Interrupted(interrupt_kind)

        try:
            _run_resume_fw_case(state_callback=checkpoint, interval=1)
        except Interrupted:
            pass
        else:
            raise AssertionError("synthetic interruption did not fire")
        restored = snapshots[-1]
        record, best = _run_resume_fw_case(resume_state=restored, interval=1)
        assert [x["status"] for x in record["blocks"]] == [x["status"] for x in expected_record["blocks"]]
        assert [x["label"] for x in record["checkpoint_history"]] == [
            x["label"] for x in expected_record["checkpoint_history"]]
        assert best["label"] == expected_best["label"]
        np.testing.assert_allclose(best["weights"], expected_best["weights"], rtol=0, atol=0)
        np.testing.assert_allclose(record["selected_weights"], expected_record["selected_weights"],
                                   rtol=0, atol=0)
        assert sum(len(block["steps"]) for block in record["blocks"]) == 3 * 2


def test_fw_timeout_midblock_restores_trajectory_without_promoting_candidate():
    expected_record, expected_best = _run_resume_fw_case(interval=2)
    snapshots = []
    anchor, poly, fit_rows, basis32, basis_gpu, _support, mocks = _resume_fw_fixture()
    with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], \
            mock.patch.object(opt.time, "monotonic", side_effect=[0.0, 2.0]):
        record, _best = opt.fw_seed(
            17, anchor, poly, fit_rows, basis32, basis_gpu, 1.0, 5.0, 2, 2,
            lambda _event: None, state_callback=lambda state: snapshots.append(
                json.loads(json.dumps(state))))
    assert record["status"] == "timeout"
    state = snapshots[-1]
    assert state["checkpoint_kind"] == "timeout_current_trajectory"
    assert state["phase_index"] == 0 and state["next_step"] == 2
    assert len(state["candidates"]) == 2  # anchor and initial mix; current point is not eligible yet
    assert state["record"]["blocks"][0]["steps"][0]["status"] == "accepted"
    resumed_record, resumed_best = _run_resume_fw_case(resume_state=state, interval=2)
    assert [x["label"] for x in resumed_record["checkpoint_history"]] == [
        x["label"] for x in expected_record["checkpoint_history"]]
    np.testing.assert_allclose(resumed_best["weights"], expected_best["weights"], rtol=0, atol=0)


def test_fw_real_cpu_objective_resumes_after_accepted_periodic_checkpoint():
    bases = np.array([[[0.8, 0.1]], [[0.7, 0.15]]], dtype=np.float64)
    target = np.array([[1, 0]], dtype=np.uint8)
    matrix, labels = opt.fit_matrix([bases], [target])
    anchor = np.array([0.5, 0.5], dtype=np.float64)
    poly = opt.build_polytope([bases], [target], opt.signed_margin(matrix, labels, anchor))
    fit_rows = [{"layout_id": "tiny", "split": "fit", "target": torch.tensor(target)}]
    basis32 = {"tiny": type("Basis", (), {
        "intensities": torch.tensor(bases[None], dtype=torch.float32)
    })()}
    basis_gpu = {"tiny": torch.tensor(bases, dtype=torch.float64)}

    def execute(resume_state=None, callback=None):
        return opt.fw_seed(17, anchor, poly, fit_rows, basis32, basis_gpu,
                           time.monotonic() + 20, 5.0, 2, 1, lambda _detail: None,
                           resume_state=resume_state, state_callback=callback)

    expected, expected_best = execute()
    snapshots = []

    class StopAtFirstPeriodic(RuntimeError):
        pass

    def interrupt(snapshot):
        if snapshot.get("checkpoint_kind") == "periodic":
            snapshots.append(json.loads(json.dumps(snapshot)))
            raise StopAtFirstPeriodic()

    with np.testing.assert_raises(StopAtFirstPeriodic):
        execute(callback=interrupt)
    saved = snapshots[0]
    assert saved["phase_index"] == 0 and saved["next_step"] == 2
    assert saved["record"]["iterations_completed"] == 1
    assert saved["record"]["blocks"][0]["steps"][0]["status"] == "accepted"
    assert len(saved["candidates"]) == 3
    assert not np.array_equal(opt.smooth_gradient(anchor, basis_gpu, 200.0)[1],
                              opt.smooth_gradient(saved["current_weights"], basis_gpu, 200.0)[1])
    with mock.patch.object(opt, "nominal_fit_check", wraps=opt.nominal_fit_check):
        record, best = execute(resume_state=saved)
    assert record["status"] == expected["status"] == "complete"
    assert record["iterations_completed"] == expected["iterations_completed"] == 1
    assert [x["label"] for x in record["checkpoint_history"]] == [
        x["label"] for x in expected["checkpoint_history"]]
    assert best["label"] == expected_best["label"]
    np.testing.assert_allclose(record["selected_weights"], expected["selected_weights"], rtol=0, atol=0)
    assert record["selected_training_metrics"] == expected["selected_training_metrics"]


def test_fw_terminal_failure_snapshot_restores_without_another_lmo():
    anchor, poly, fit_rows, basis32, basis_gpu, _support, mocks = _resume_fw_fixture()
    calls = [0]

    def fail_second_lmo(*_args, **_kwargs):
        calls[0] += 1
        if calls[0] == 1:
            return {"status": "optimal_verified", "weights": np.array([0.0, 1.0]), "attempts": []}
        return {"status": "solver_failure", "weights": None, "attempts": []}

    snapshots = []
    class StopAfterTerminal(RuntimeError):
        pass
    def interrupt(snapshot):
        if snapshot.get("stage") == "terminal_seed":
            snapshots.append(json.loads(json.dumps(snapshot)))
            raise StopAfterTerminal()

    with mock.patch.object(opt, "solve_lmo", side_effect=fail_second_lmo), mocks[1], mocks[2], \
            mocks[3], mocks[4], mocks[5]:
        with np.testing.assert_raises(StopAfterTerminal):
            opt.fw_seed(17, anchor, poly, fit_rows, basis32, basis_gpu,
                        time.monotonic() + 20, 5.0, 2, 1, lambda _event: None,
                        state_callback=interrupt)
    state = snapshots[0]
    with mock.patch.object(opt, "nominal_fit_check", return_value={"passed": True}):
        opt._validate_fw_resume_state(state, 17, anchor, poly, fit_rows, basis32,
                                      np.ones((1, 2), dtype=bool), 2)
        with mock.patch.object(opt, "solve_lmo", side_effect=AssertionError("terminal seed revived")), \
                mocks[1], mocks[2], mocks[3], mocks[4], mocks[5]:
            record, _best = opt.fw_seed(17, anchor, poly, fit_rows, basis32, basis_gpu,
                                        time.monotonic() + 20, 5.0, 2, 1,
                                        lambda _event: None, resume_state=state)
    assert record["status"] == "solver_failure"
    assert calls[0] == 2
    assert record["iterations_completed"] == 0


def test_emergency_commit_uses_last_canonical_snapshot_only():
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        identity = {"dataset_sha256": "abc"}
        coherent = opt._new_run_state(identity, {"status": "frank_wolfe_running",
                                                  "frank_wolfe": [], "heldout_status": "not_indexed_or_evaluated",
                                                  "final3_status": "closed"}, phase="frank_wolfe")
        coherent["current_seed"] = 17
        coherent["fw_state"] = {"stage": "seed_start", "seed": 17, "schema_version": 1}
        opt._seal_run_state(coherent)
        args = SimpleNamespace(_last_canonical_state=coherent,
                               _active_output=out,
                               _run_results={"frank_wolfe": [{"seed": 17, "status": "complete"}]})
        opt._persist_emergency_state(args, "interrupted")
        state = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
        assert state["results"]["status"] == "interrupted"
        assert state["results"]["frank_wolfe"] == []
        assert state["current_seed"] == 17 and state["fw_state"]["stage"] == "seed_start"

        # The durable file wins if a signal landed between os.replace and the
        # in-memory snapshot copy; the emergency path must not roll it back.
        older = json.loads(json.dumps(coherent))
        newer = json.loads(json.dumps(coherent))
        newer["results"]["durable_generation"] = "newer"
        newer["progress"] = {"event": "newer_generation"}
        opt._seal_run_state(newer)
        opt.atomic_json(out / "run_state.json", newer)
        args._last_canonical_state = older
        opt._persist_emergency_state(args, "interrupted")
        state = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
        assert state["results"]["durable_generation"] == "newer"
        assert state["results"]["status"] == "interrupted"

        complete = json.loads(json.dumps(newer))
        complete["phase"] = "complete"
        complete["results"]["status"] = "complete"
        opt._seal_run_state(complete)
        opt.atomic_json(out / "run_state.json", complete)
        complete_bytes = (out / "run_state.json").read_bytes()
        opt._persist_emergency_state(args, "error", RuntimeError("late sidecar failure"))
        assert (out / "run_state.json").read_bytes() == complete_bytes
        state = json.loads(complete_bytes)
        assert state["phase"] == state["results"]["status"] == "complete"


def test_resume_state_identity_corruption_and_run_lock_fail_closed():
    identity = {"dataset_sha256": "abc", "solver": {"iterations": 2}}
    results = {"status": "timeout_during_frank_wolfe", "heldout_status": "not_indexed_or_evaluated",
               "final3_status": "closed"}
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        state = opt._new_run_state(identity, results, phase="frank_wolfe")
        opt.atomic_json(out / "run_state.json", state)
        assert opt._load_run_state(out, identity)["phase"] == "frank_wolfe"
        with np.testing.assert_raises_regex(ValueError, "identity mismatch"):
            opt._load_run_state(out, {"dataset_sha256": "changed"})
        lock1 = opt.RunLock(out).acquire()
        before = (out / ".run.lock").read_text(encoding="utf-8")
        with np.testing.assert_raises_regex(RuntimeError, "run directory is locked"):
            opt.RunLock(out).acquire()
        assert (out / ".run.lock").read_text(encoding="utf-8") == before
        lock1.release()
        (out / "run_state.json").write_text("{broken", encoding="utf-8")
        with np.testing.assert_raises_regex(ValueError, "corrupt"):
            opt._load_run_state(out, identity)


def test_resume_validator_rejects_corrupt_next_step_and_skips_completed_seeds():
    snapshots = []
    class Interrupted(RuntimeError):
        pass
    def stop_after_periodic(snapshot):
        snapshots.append(json.loads(json.dumps(snapshot)))
        if snapshot.get("checkpoint_kind") == "periodic":
            raise Interrupted()
    try:
        _run_resume_fw_case(state_callback=stop_after_periodic, interval=1)
    except Interrupted:
        pass
    state = snapshots[-1]
    anchor, poly, fit_rows, basis32, _basis_gpu, support, _mocks = _resume_fw_fixture()
    with mock.patch.object(opt, "nominal_fit_check", return_value={"passed": True}):
        opt._validate_fw_resume_state(state, 17, anchor, poly, fit_rows, basis32,
                                      support, iterations=2)
        state["next_step"] = 1
        with np.testing.assert_raises_regex(ValueError, "next step"):
            opt._validate_fw_resume_state(state, 17, anchor, poly, fit_rows, basis32,
                                          support, iterations=2)
    records = [{"seed": 17, "status": "complete"}, {"seed": 29, "status": "complete"}]
    assert opt._pending_fw_seeds(records) == [43, 71, 101]


def test_run_resume_canonical_state_skips_completed_seed_and_keeps_calibration_closed():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        dataset_path, diagnostic_path = root / "dataset.pt", root / "diagnostic.json"
        dataset_path.write_bytes(b"synthetic fit-only archive")
        fit_ids = ["fit0", "fit1", "fit2", "fit3"]
        cal_ids = ["cal0", "cal1", "cal2", "cal3"]
        diagnostic_payload = {
            "input": {"dataset_sha256": "dataset-hash",
                      "fit_masks": [], "fit_targets": [],
                      "calibration_masks": [], "calibration_targets": [],
                      "bases": [{"layout_id": name, "sha256": "basis"}
                                for name in fit_ids + cal_ids]},
            "scenarios": {"fit_only.nominal": {
                "status": "positive_margin_feasible", "lp_optimal_margin": 0.5,
                "source_weights_full_grid": [[0.5, 0.5]]}},
            "baselines": {"A0_initial_annulus_no_jitter": {
                "source_weights_full_grid": [[0.25, 0.75]]}},
        }
        diagnostic_path.write_text(json.dumps(diagnostic_payload), encoding="utf-8")
        fit = SimpleNamespace(masks=torch.zeros((4, 1, 1, 2)),
                              targets=torch.tensor([[[[1, 0]]]] * 4),
                              layout_ids=tuple(fit_ids), pixel_size_nm=4.0)
        cal = SimpleNamespace(masks=torch.zeros((4, 1, 128, 128)),
                              targets=torch.zeros((4, 1, 128, 128)),
                              layout_ids=tuple(cal_ids), pixel_size_nm=4.0)
        basis = {name: SimpleNamespace(intensities=torch.zeros((1, 2, 1, 2)))
                 for name in fit_ids + cal_ids}
        support = np.ones((1, 2), dtype=bool)
        fake_poly = SimpleNamespace(margin_floor=0.1,
                                    verify=lambda _weights: {"passed": True})
        metric_obj = {"mean": {"band_pixels": 100.0, "L2_pixels": 10.0,
                                "L2_worst_dose_pixels": 20.0},
                      "per_layout": [{"target_positive_pixels": 1,
                                      "per_corner": [{"predicted_positive_pixels": 1}]}]}
        fit_metric_obj = {"mean": {"band_pixels": 80.0, "L2_pixels": 10.0,
                                    "L2_worst_dose_pixels": 12.0},
                          "per_layout": [{"target_positive_pixels": 1,
                                          "per_corner": [{"predicted_positive_pixels": 1}]}]}
        fit_metrics = {"mean": {"band_pixels": 80.0, "L2_pixels": 0.0,
                                "L2_worst_dose_pixels": 10.0},
                       "per_layout": [{"L2_pixels": 0}]}
        fw_calls = []
        resume_mode = [False]

        def fake_fw(seed, anchor, _poly, _fit_rows, _basis32, _basis_gpu,
                    _deadline, _solver_limit, _iterations, _interval, _progress,
                    resume_state=None, state_callback=None):
            fw_calls.append(seed)
            if seed == 29 and not resume_mode[0]:
                state_callback({"schema_version": 1, "seed": seed, "stage": "seed_start"})
                return {"seed": seed, "status": "timeout"}, None
            record = {"seed": seed, "status": "complete", "selected_weights": anchor.tolist(),
                      "selected_training_candidate": "LP anchor"}
            best = {"weights": anchor.copy(), "label": "LP anchor", "fit_metrics": fit_metrics}
            return record, best

        def install_mocks(stack, fw):
            stack.enter_context(mock.patch.object(opt.torch.cuda, "is_available", return_value=True))
            stack.enter_context(mock.patch.object(opt.torch.cuda, "get_device_name", return_value="Mock GPU"))
            stack.enter_context(mock.patch.object(opt.torch.cuda, "synchronize"))
            stack.enter_context(mock.patch.object(opt.diagnostic, "sha256_file", return_value="dataset-hash"))
            stack.enter_context(mock.patch.object(opt, "load_fit", return_value=fit))
            stack.enter_context(mock.patch.object(opt.experiment, "generate_calibration", return_value=(None, cal, None)))
            stack.enter_context(mock.patch.object(opt, "check_hashes"))
            stack.enter_context(mock.patch.object(opt, "prepare_bases", return_value=(basis, basis, basis, {"passed": True})))
            stack.enter_context(mock.patch.object(opt, "support_mask", return_value=support))
            stack.enter_context(mock.patch.object(opt, "fit_matrix", return_value=(np.ones((1, 2)), np.array([1]))))
            stack.enter_context(mock.patch.object(opt, "signed_margin", return_value=0.5))
            stack.enter_context(mock.patch.object(opt, "build_polytope", return_value=fake_poly))
            stack.enter_context(mock.patch.object(opt, "edge_contrast_matrix", return_value=np.ones((1, 2))))
            def fake_metrics(rows, *_args):
                value = fit_metric_obj if rows[0]["split"] == "fit" else metric_obj
                return json.loads(json.dumps(value))
            metrics_mock = stack.enter_context(mock.patch.object(opt, "metrics", side_effect=fake_metrics))
            stack.enter_context(mock.patch.object(opt, "_control_check"))
            stack.enter_context(mock.patch.object(opt, "save_weights"))
            stack.enter_context(mock.patch.object(opt, "solve_edge_lp", return_value={
                "status": "optimal_verified", "weights": np.array([0.5, 0.5]), "attempts": []}))
            stack.enter_context(mock.patch.object(opt, "candidate_row", return_value={
                "weights": np.array([0.5, 0.5]), "checkpoint_order": 0,
                "fit_qualified": True, "smooth_beta800": 1.0,
                "polytope": {"passed": True}, "fit_metrics": fit_metrics}))
            stack.enter_context(mock.patch.object(opt, "nominal_fit_check", return_value={"passed": True}))
            stack.enter_context(mock.patch.object(opt, "fw_seed", side_effect=fw))
            return metrics_mock

        def cli_args(*extra):
            return ["--dataset-file", str(dataset_path), "--diagnostic-file", str(diagnostic_path),
                    "--output-root", str(root / "runs"), "--expected-gpu", "Mock GPU",
                    "--timeout-seconds", "60", *extra]

        invocation_end_interrupt = [True]
        real_atomic_json = opt.atomic_json
        def interrupt_at_invocation_end(path, payload):
            if (Path(path).name == "run_state.json" and invocation_end_interrupt[0]
                    and payload.get("progress", {}).get("event") == "invocation_end"):
                invocation_end_interrupt[0] = False
                raise KeyboardInterrupt()
            return real_atomic_json(path, payload)
        with ExitStack() as stack:
            install_mocks(stack, fake_fw)
            with mock.patch.object(opt, "atomic_json", side_effect=interrupt_at_invocation_end):
                with np.testing.assert_raises(KeyboardInterrupt):
                    opt.run(opt.parse_args(cli_args()))
        run_dirs = list((root / "runs").iterdir())
        assert len(run_dirs) == 1
        out = run_dirs[0]
        first = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
        assert first["phase"] == "frank_wolfe"
        assert first["results"]["status"] == "interrupted"
        assert [record["seed"] for record in first["results"]["frank_wolfe"]] == [17]
        assert first["current_seed"] == 29 and first["fw_state"]["stage"] == "seed_start"
        assert first["results"]["edge_contrast_lp"]["calibration"]["status"] == \
            "not_evaluated_incomplete_seed_set"
        assert first["results"]["comparison"]["status"] == "calibration_closed_incomplete_seed_set"
        assert not (out / ".run.lock").exists()

        # Model an abrupt process stop after the last canonical checkpoint.
        # The phase status itself is valid and remains resumable after lock recovery.
        first["results"]["status"] = "frank_wolfe_running"
        opt._seal_run_state(first)
        opt.atomic_json(out / "run_state.json", first)

        resume_mode[0] = True
        canonical_at_failure = []
        real_atomic_json = opt.atomic_json
        fail_progress_once = [True]
        def fail_progress_sidecar(path, payload):
            if Path(path).name == "progress.json" and fail_progress_once[0]:
                fail_progress_once[0] = False
                canonical_at_failure.append((out / "run_state.json").read_bytes())
                raise PermissionError("synthetic progress-sidecar denial")
            return real_atomic_json(path, payload)
        with ExitStack() as stack:
            install_mocks(stack, fake_fw)
            with mock.patch.object(opt, "atomic_json", side_effect=fail_progress_sidecar):
                with np.testing.assert_raises(opt.SidecarWriteError):
                    opt.run(opt.parse_args(resume_args := [
                        "--dataset-file", str(dataset_path), "--diagnostic-file", str(diagnostic_path),
                        "--resume-run", str(out), "--expected-gpu", "Mock GPU",
                        "--timeout-seconds", "60"]))
        active = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
        assert (out / "run_state.json").read_bytes() == canonical_at_failure[0]
        assert active["phase"] == "frank_wolfe"
        assert active["results"]["status"] == "frank_wolfe_running"
        assert not (out / ".run.lock").exists()

        fw_calls.clear()
        with ExitStack() as stack:
            install_mocks(stack, fake_fw)
            resumed = opt.run(opt.parse_args(resume_args))
        assert resumed == out
        assert fw_calls == [29, 43, 71, 101]  # seed 17 was loaded from canonical results
        final_state = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
        assert final_state["phase"] == "complete"
        assert final_state["results"]["comparison"]["status"] == "ready"
        assert final_state["results"]["comparison"]["arms"][0]["fit"]["mean"]["band_pixels"] == 80.0
        assert final_state["results"]["comparison"]["arms"][0]["calibration"]["mean"]["band_pixels"] == 100.0
        assert not (out / ".run.lock").exists()

        # Resume a canonical partial-calibration snapshot. The cached edge and
        # first seed are retained; only the four missing FW calibration rows run.
        saved_files = {name: (out / name).read_bytes()
                       for name in ("run_state.json", "results.json", "progress.json")}
        partial = json.loads(saved_files["run_state.json"])
        partial["phase"] = "calibration_gate"
        partial["results"]["status"] = "timeout_during_calibration_gate"
        partial["results"].pop("frank_wolfe_arm", None)
        partial["results"]["selection"] = {"status": "closed_until_training_frozen"}
        for record in partial["results"]["frank_wolfe"]:
            if record["seed"] != 17:
                record["calibration"] = {"status": "not_evaluated_timeout"}
        partial["results"]["comparison"] = opt._build_comparison(
            partial["results"], True, False)
        partial["current_seed"], partial["fw_state"] = None, None
        partial["progress"].update({"status": "timeout_during_calibration_gate",
                                    "phase": "calibration_gate", "calibration_seed": 17})
        opt._seal_run_state(partial)
        opt.atomic_json(out / "run_state.json", partial)
        fw_calls.clear()
        with ExitStack() as stack:
            metrics_mock = install_mocks(stack, fake_fw)
            opt.run(opt.parse_args(resume_args))
            assert metrics_mock.call_count == 4
        assert fw_calls == []
        partial_resume = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
        assert partial_resume["phase"] == "complete"
        assert partial_resume["current_seed"] is None
        assert partial_resume["results"]["comparison"]["arms"][-1]["calibration"]["status"] == "measured"
        for name, data in saved_files.items():
            (out / name).write_bytes(data)

        # A terminal seed failure stays settled when a later seed is interrupted.
        failed = json.loads(saved_files["run_state.json"])
        failed["phase"] = "frank_wolfe"
        failed["results"]["status"] = "frank_wolfe_running"
        failed_records = failed["results"]["frank_wolfe"][:3]
        failed_records[2]["status"] = "solver_failure"
        for record in failed_records:
            record["calibration"] = {"status": "not_evaluated_incomplete_seed_set"}
        failed["results"]["frank_wolfe"] = failed_records
        failed["results"].pop("frank_wolfe_arm", None)
        failed["results"]["selection"] = {"status": "closed_until_training_frozen"}
        failed["results"]["edge_contrast_lp"]["calibration"] = {
            "status": "not_evaluated_incomplete_seed_set"}
        failed["results"]["edge_contrast_lp"].pop("gate", None)
        failed["results"]["edge_contrast_lp"].pop("no_positive_target_blank_any_dose", None)
        failed["results"]["comparison"] = opt._build_comparison(failed["results"], False)
        failed_record = {"seed": 71, "status": "running", "iterations_completed": 1,
                         "checkpoint_history": [], "blocks": [{"beta": 200.0,
                             "status": "running", "steps": [{"step": 1, "status": "accepted"}]}]}
        active_state = {"schema_version": 1, "seed": 71, "stage": "iterations",
                        "phase_index": 0, "next_step": 2, "current_weights": [0.5, 0.5],
                        "next_checkpoint_order": 2, "record": failed_record,
                        "candidates": [{"weights": [0.5, 0.5], "checkpoint_order": 0,
                                        "label": "LP anchor"},
                                       {"weights": [0.5, 0.5], "checkpoint_order": 1,
                                        "label": "initial 5% feasible-vertex mix"}],
                        "checkpoint_kind": "periodic"}
        failed["current_seed"], failed["fw_state"] = 71, active_state
        failed["progress"].update({"status": "frank_wolfe_running", "phase": "frank_wolfe",
                                    "current_seed": 71})
        opt._seal_run_state(failed)
        opt.atomic_json(out / "run_state.json", failed)
        fw_calls.clear()
        with ExitStack() as stack:
            install_mocks(stack, fake_fw)
            opt.run(opt.parse_args(resume_args))
        assert fw_calls == [71, 101]
        failed_final = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
        assert failed_final["results"]["frank_wolfe"][2]["status"] == "solver_failure"
        assert failed_final["results"]["edge_contrast_lp"]["calibration"]["mean"]["band_pixels"] == 100.0
        assert failed_final["results"]["frank_wolfe_arm"]["status"] == "incomplete_seed_set"
        assert failed_final["results"]["comparison"]["status"] == "ready_fw_incomplete"

        # Restore the original complete canonical generation for no-op checks below.
        for name, data in saved_files.items():
            (out / name).write_bytes(data)

        state_bytes = (out / "run_state.json").read_bytes()
        # On a fresh live run, a results sidecar failure after run_complete
        # must leave the complete canonical generation terminal and repairable.
        live_root = root / "live_complete_runs"
        fail_live_results_once = [True]
        live_canonical = []
        real_atomic_json = opt.atomic_json
        def fail_live_complete_results(path, payload):
            if (Path(path).name == "results.json" and payload.get("status") == "complete"
                    and fail_live_results_once[0]):
                fail_live_results_once[0] = False
                live_dirs_now = list(live_root.iterdir()) if live_root.exists() else []
                if live_dirs_now:
                    live_canonical.append((live_dirs_now[0] / "run_state.json").read_bytes())
                raise PermissionError("synthetic live complete sidecar denial")
            return real_atomic_json(path, payload)
        fw_calls.clear()
        with ExitStack() as stack:
            metrics_mock = install_mocks(stack, fake_fw)
            with mock.patch.object(opt, "atomic_json", side_effect=fail_live_complete_results):
                with np.testing.assert_raises(opt.SidecarWriteError):
                    opt.run(opt.parse_args(cli_args("--output-root", str(live_root))))
        assert len(live_canonical) == 1
        live_out = Path(json.loads(live_canonical[0])["results"]["result_directory"])
        assert (live_out / "run_state.json").read_bytes() == live_canonical[0]
        live_complete = json.loads(live_canonical[0])
        assert live_complete["phase"] == live_complete["results"]["status"] == "complete"
        assert not (live_out / ".run.lock").exists()
        live_resume_args = ["--dataset-file", str(dataset_path), "--diagnostic-file", str(diagnostic_path),
                            "--resume-run", str(live_out), "--expected-gpu", "Mock GPU",
                            "--timeout-seconds", "60"]
        fw_calls.clear()
        with ExitStack() as stack:
            metrics_mock = install_mocks(stack, fake_fw)
            opt.run(opt.parse_args(live_resume_args))
            assert metrics_mock.call_count == 0
        assert fw_calls == []

        # A sidecar PermissionError cannot downgrade a completed canonical run.
        # A later complete no-op resume repairs the sidecars from that run state.
        real_atomic_json = opt.atomic_json
        fail_results_once = [True]
        canonical_at_failure = []
        def fail_results_sidecar(path, payload):
            if Path(path).name == "results.json" and fail_results_once[0]:
                fail_results_once[0] = False
                canonical_at_failure.append((out / "run_state.json").read_bytes())
                raise PermissionError("synthetic results-sidecar denial")
            return real_atomic_json(path, payload)
        with ExitStack() as stack:
            install_mocks(stack, fake_fw)
            with mock.patch.object(opt, "atomic_json", side_effect=fail_results_sidecar):
                with np.testing.assert_raises(opt.SidecarWriteError):
                    opt.run(opt.parse_args(resume_args))
        assert canonical_at_failure[0] == state_bytes
        assert (out / "run_state.json").read_bytes() == state_bytes
        assert json.loads(state_bytes)["phase"] == "complete"
        assert not (out / ".run.lock").exists()
        fw_calls.clear()
        with ExitStack() as stack:
            metrics_mock = install_mocks(stack, fake_fw)
            opt.run(opt.parse_args(resume_args))
            assert metrics_mock.call_count == 0
        assert fw_calls == []

        (out / "results.json").write_text('{"status":"stale"}', encoding="utf-8")
        (out / "progress.json").write_text('{"status":"stale"}', encoding="utf-8")
        fw_calls.clear()
        with ExitStack() as stack:
            metrics_mock = install_mocks(stack, fake_fw)
            opt.run(opt.parse_args(resume_args))
            assert metrics_mock.call_count == 0
        assert fw_calls == []
        assert (out / "run_state.json").read_bytes() == state_bytes
        assert json.loads((out / "results.json").read_text(encoding="utf-8")) == final_state["results"]
        assert json.loads((out / "progress.json").read_text(encoding="utf-8")) == final_state["progress"]

        names = ("run_state.json", "results.json", "progress.json")

        def rejected_without_writes(state_bytes_to_test):
            (out / "run_state.json").write_bytes(state_bytes_to_test)
            before_invalid = {name: (out / name).read_bytes() for name in names}
            with ExitStack() as stack:
                install_mocks(stack, fake_fw)
                capture_out, capture_err = io.StringIO(), io.StringIO()
                with redirect_stdout(capture_out), redirect_stderr(capture_err):
                    code = opt.main(resume_args)
            assert code == 2
            assert {name: (out / name).read_bytes() for name in names} == before_invalid

        original_canonical = json.loads(state_bytes)
        checksum_mutations = []
        for field_path in (
                ("controls", "fixed_annulus", "calibration", "mean", "band_pixels"),
                ("edge_contrast_lp", "weights_full_grid"),
                ("frank_wolfe_arm", "calibration", "mean", "band_pixels")):
            mutated = json.loads(state_bytes)
            cursor = mutated["results"]
            for key in field_path[:-1]:
                cursor = cursor[key]
            leaf = field_path[-1]
            if leaf == "weights_full_grid":
                cursor[leaf][0][0] += 0.25
            else:
                cursor[leaf] += 0.25
            checksum_mutations.append(json.dumps(mutated, ensure_ascii=False, indent=2,
                                                sort_keys=True).encode("utf-8"))
        for invalid_state in checksum_mutations:
            rejected_without_writes(invalid_state)

        # A valid checksum still cannot make an impossible active pointer valid.
        semantic = json.loads(state_bytes)
        semantic["current_seed"], semantic["fw_state"] = 17, None
        opt._seal_run_state(semantic)
        rejected_without_writes(json.dumps(semantic, ensure_ascii=False, indent=2,
                                           sort_keys=True).encode("utf-8"))
        (out / "run_state.json").write_bytes(state_bytes)

        before = {name: (out / name).read_bytes() for name in names}
        owner = opt.RunLock(out).acquire()
        lock_before = (out / ".run.lock").read_bytes()
        capture_out, capture_err = io.StringIO(), io.StringIO()
        with redirect_stdout(capture_out), redirect_stderr(capture_err):
            code = opt.main(resume_args)
        assert code == 2
        assert {name: (out / name).read_bytes() for name in names} == before
        assert (out / ".run.lock").read_bytes() == lock_before
        owner.release()

        diagnostic_payload["harmless_identity_change"] = True
        diagnostic_path.write_text(json.dumps(diagnostic_payload), encoding="utf-8")
        with ExitStack() as stack:
            install_mocks(stack, fake_fw)
            capture_out, capture_err = io.StringIO(), io.StringIO()
            with redirect_stdout(capture_out), redirect_stderr(capture_err):
                code = opt.main(resume_args)
        assert code == 2
        assert {name: (out / name).read_bytes() for name in names} == before
        assert not (out / ".run.lock").exists()


def _function_suite():
    return unittest.TestSuite(
        unittest.FunctionTestCase(value, description=name)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    )


def load_tests(loader, standard_tests, pattern):
    return _function_suite()


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(_function_suite())
    raise SystemExit(0 if result.wasSuccessful() else 1)
