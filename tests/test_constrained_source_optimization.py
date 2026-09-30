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
