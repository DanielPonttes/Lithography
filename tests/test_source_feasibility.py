from unittest import mock

import numpy as np
import torch

from light_source import DifferentiableAbbeLitho, PixelatedLightSource
from source_training import SourceDataset

from scripts import diagnose_source_feasibility as diagnostic


def _one_source_layout(values, target):
    return {
        "basis": np.asarray(values, dtype=np.float64).reshape(1, 1, -1),
        "target": np.asarray(target, dtype=np.uint8).reshape(1, -1),
    }


def test_known_feasible_solution_has_normalized_source_and_small_direct_residual():
    layout = {
        "basis": np.asarray([[[0.4, 0.05]], [[0.1, 0.2]]], dtype=np.float64),
        "target": np.asarray([[1, 0]], dtype=np.uint8),
    }
    result = diagnostic.solve_max_margin([layout])
    assert result["solver"]["success"]
    assert result["lp_margin"] > 0
    assert result["weights"].min() >= 0
    assert abs(result["weights"].sum() - 1) < 1e-12
    evaluated = diagnostic._metrics(
        {"layout_id": "known", "split": "fit", "target": layout["target"]},
        type("Basis", (), {"intensities": __import__("torch").as_tensor(layout["basis"][None])})(),
        result["weights"], [1.0],
    )
    records, direct_margin, residual = diagnostic._evaluate(
        [{"layout_id": "known", "split": "fit", "target": layout["target"]}],
        {"known": type("Basis", (), {"intensities": __import__("torch").as_tensor(layout["basis"][None])})()},
        result["weights"], [1.0], result["lp_margin"],
    )
    assert evaluated["minimum_direct_margin"] > 0
    assert residual["passed"]
    assert abs(direct_margin - result["lp_margin"]) < diagnostic.RESIDUAL_TOL
    assert records[0]["per_corner"][0]["intensity_vs_resist_hardprint_disagreements"] == 0


def test_identical_foreground_background_rows_have_zero_boundary_margin():
    basis = np.asarray([[[0.3, 0.3]], [[0.1, 0.1]]], dtype=np.float64)
    target = np.asarray([[1, 0]], dtype=np.uint8)
    result = diagnostic.solve_max_margin([{"basis": basis, "target": target}])
    assert result["solver"]["success"]
    assert abs(result["lp_margin"]) <= diagnostic.MARGIN_TOL
    assert diagnostic._status(result["lp_margin"]) == "indeterminate_boundary"


def test_nominal_can_be_feasible_while_robust_doses_are_infeasible():
    layout = _one_source_layout([0.226, 0.224], [1, 0])
    nominal = diagnostic.solve_max_margin([layout], doses=(1.0,))
    robust = diagnostic.solve_max_margin([layout], doses=(0.98, 1.0, 1.02))
    assert nominal["lp_margin"] > 0
    assert robust["lp_margin"] < -diagnostic.MARGIN_TOL
    assert diagnostic._status(robust["lp_margin"]) == "infeasible_under_strict_full_pixel_constraints"


def test_two_layouts_can_be_individually_feasible_but_share_no_positive_margin():
    first = {
        "basis": np.asarray([[[0.4, 0.1]], [[0.1, 0.4]]], dtype=np.float64),
        "target": np.asarray([[1, 0]], dtype=np.uint8),
    }
    second = {
        "basis": np.asarray([[[0.1, 0.4]], [[0.4, 0.1]]], dtype=np.float64),
        "target": np.asarray([[1, 0]], dtype=np.uint8),
    }
    first_result = diagnostic.solve_max_margin([first])
    second_result = diagnostic.solve_max_margin([second])
    shared = diagnostic.solve_max_margin([first, second])
    assert first_result["lp_margin"] > 0
    assert second_result["lp_margin"] > 0
    assert shared["lp_margin"] < -diagnostic.MARGIN_TOL


def test_fit_loader_does_not_index_final_test_field():
    accessed = []

    class GuardPayload(dict):
        def __getitem__(self, key):
            accessed.append(key)
            if key == "final_test":
                raise AssertionError("final_test must remain closed")
            return super().__getitem__(key)

    fit = {
        "masks": __import__("torch").zeros((1, 1, 2, 2)),
        "targets": __import__("torch").zeros((1, 1, 2, 2)),
        "layout_ids": ["fit_only"],
        "pixel_size_nm": 4.0,
    }
    payload = GuardPayload(fit=fit, final_test=object())
    with mock.patch.object(diagnostic.torch, "load", return_value=payload) as load:
        ds = diagnostic.load_fit_dataset("unread-by-mock.pt")
    assert accessed == ["fit"]
    assert ds.layout_ids == ("fit_only",)
    assert load.call_args.kwargs["weights_only"] is True



def test_source_dataset_layout_rows_use_2d_target_and_keep_cpu_basis_numpy_readable():
    dataset = SourceDataset(
        masks=torch.tensor([[[[1.0, 0.0]]]]),
        targets=torch.tensor([[[[1.0, 0.0]]]]),
        layout_ids=("shape_regression",),
        pixel_size_nm=4.0,
    )
    layout = diagnostic._layout_rows(dataset, "fit")[0]
    assert tuple(layout["target"].shape) == (1, 2)
    assert tuple(layout["mask"].shape) == (1, 1, 2)

    source = PixelatedLightSource(9, sigma_inner=0.3, sigma_outer=0.9)
    simulator = DifferentiableAbbeLitho(
        source, numerical_aperture=1.35, wavelength_nm=193.0,
        pixel_size_nm=4.0, source_chunk_size=8, cache_max_bytes=0,
    )
    basis = diagnostic._prepare_float32_basis(
        simulator, layout["mask"][None], verify_parity=True,
    )
    assert basis.intensities.dtype == torch.float32
    assert basis.intensities.device.type == "cpu"
    basis_numpy = basis.intensities[0].numpy()
    assert basis_numpy.shape == (49, 1, 2)

    lp_input = {"basis": basis_numpy, "target": layout["target"].numpy()}
    result = diagnostic.solve_max_margin([lp_input])
    assert result["solver"]["success"]
    records, margin, residual = diagnostic._evaluate(
        [layout], {layout["layout_id"]: basis}, result["weights"],
        [1.0], result["lp_margin"],
    )
    assert records[0]["per_corner"][0]["target_positive_pixels"] == 1
    assert residual["passed"]
    assert abs(margin - result["lp_margin"]) < diagnostic.RESIDUAL_TOL



def test_numerical_residual_failure_retries_ipm_without_presolve_and_keeps_both_results():
    basis_tensor = torch.tensor([[[[0.4, 0.05]]]], dtype=torch.float32)
    basis_obj = type("Basis", (), {"intensities": basis_tensor})()
    layout = {
        "layout_id": "retry",
        "split": "fit",
        "target": torch.tensor([[1, 0]], dtype=torch.float32),
    }
    bases = {"retry": basis_obj}
    basis_array = basis_tensor[0].numpy().astype(np.float64)
    direct_margin = min(
        float(basis_array[0, 0, 0] - diagnostic.experiment.THRESHOLD),
        float(diagnostic.experiment.THRESHOLD - basis_array[0, 0, 1]),
    )
    initial = {
        "solver": {"success": True, "method": "scipy highs", "presolve": True},
        "weights": np.array([1.0]), "lp_margin": direct_margin + 1e-4,
    }
    retry = {
        "solver": {"success": True, "method": "scipy highs-ipm", "presolve": False},
        "weights": np.array([1.0]), "lp_margin": direct_margin,
    }
    lp_layouts = [{"basis": basis_array, "target": layout["target"].numpy()}]
    with mock.patch.object(
        diagnostic, "solve_max_margin", side_effect=[initial, retry]
    ) as solve:
        resolved = diagnostic._solve_verified_lp(
            lp_layouts, [layout], bases, [1.0], 60.0, lambda: 30.0,
        )
    assert solve.call_count == 2
    assert solve.call_args_list[0].kwargs["method"] == "highs"
    assert solve.call_args_list[0].kwargs["presolve"] is True
    assert solve.call_args_list[1].kwargs["method"] == "highs-ipm"
    assert solve.call_args_list[1].kwargs["presolve"] is False
    assert solve.call_args_list[1].args[3] == 30.0
    assert resolved["solver_call_count"] == 2
    assert resolved["result"] is retry
    assert resolved["residual"]["passed"]
    assert resolved["numerical_retry"]["reason"] == "initial_numerical_verification_failed"
    assert resolved["numerical_retry"]["initial"]["lp_optimal_margin"] == initial["lp_margin"]
    assert resolved["numerical_retry"]["initial"]["direct_residual"]["passed"] is False
    assert resolved["numerical_retry"]["selected"] is True
