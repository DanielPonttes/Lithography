import math
import contextlib
from contextlib import ExitStack
import builtins
import copy
import dis
import io
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from source_robustness import (
    explicit_worst_corner_squared_hinge_from_aerial,
    fit_objective_gradient_diagnostics,
    robust_corner_squared_hinge,
    smooth_pv_from_aerial,
    smooth_pv_value,
    smooth_pv_value_gradient,
    signed_margin_quantiles,
    validate_nonnegative_bases,
    validate_rho,
    worst_corner_squared_hinge_from_aerial,
)
from scripts import optimize_source_constrained as constrained
from scripts import optimize_source_robust_corners as runner


class RobustSourceObjectiveTests(unittest.TestCase):
    def test_smooth_pv_aerial_matches_registered_dose_difference_formula(self):
        aerial = torch.tensor([[0.19, 0.225], [0.26, 0.31]], dtype=torch.float64)
        beta = 400.0
        actual = smooth_pv_from_aerial(aerial, beta)
        expected = (
            torch.sigmoid(beta * (1.02 * aerial - 0.225))
            - torch.sigmoid(beta * (0.98 * aerial - 0.225))
        ).mean()
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-15)
        self.assertGreaterEqual(float(actual), 0.0)
        self.assertLessEqual(float(actual), 1.0)

    def test_smooth_pv_source_gradient_matches_finite_difference_and_value(self):
        bases = {
            "fit_a": torch.tensor(
                [[[0.18, 0.22], [0.27, 0.31]],
                 [[0.29, 0.24], [0.19, 0.33]],
                 [[0.21, 0.30], [0.26, 0.17]]], dtype=torch.float64,
            ),
            "fit_b": torch.tensor(
                [[[0.25, 0.16], [0.32, 0.23]],
                 [[0.18, 0.35], [0.21, 0.28]],
                 [[0.30, 0.22], [0.17, 0.26]]], dtype=torch.float64,
            ),
        }
        weights = np.array([0.2, 0.35, 0.45], dtype=np.float64)
        value, gradient = smooth_pv_value_gradient(weights, bases, beta=800.0)
        self.assertAlmostEqual(value, smooth_pv_value(weights, bases, beta=800.0), places=14)
        self.assertEqual(gradient.shape, weights.shape)
        self.assertTrue(np.isfinite(gradient).all())
        direction = np.array([1.0, -0.25, -0.75], dtype=np.float64)
        epsilon = 1e-6
        plus = smooth_pv_value(weights + epsilon * direction, bases, beta=800.0)
        minus = smooth_pv_value(weights - epsilon * direction, bases, beta=800.0)
        finite_difference = (plus - minus) / (2.0 * epsilon)
        self.assertAlmostEqual(float(gradient @ direction), finite_difference, places=7)

    def test_smooth_pv_rejects_invalid_temperature_doses_and_basis_shapes(self):
        aerial = torch.ones((1, 2), dtype=torch.float64)
        for beta in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                smooth_pv_from_aerial(aerial, beta)
        with self.assertRaises(ValueError):
            smooth_pv_from_aerial(aerial, 800.0, low_dose=1.02, high_dose=0.98)
        with self.assertRaises(ValueError):
            smooth_pv_value_gradient(
                np.array([0.5, 0.5]), {"fit": torch.ones((3, 1, 2), dtype=torch.float64)}, 800.0
            )

    def test_analytic_worst_corner_matches_min_dose_reference(self):
        aerial = torch.tensor(
            [[0.0, 0.12, 0.23], [0.31, 0.45, 0.7]],
            dtype=torch.float64,
        )
        target = torch.tensor(
            [[True, False, True], [False, True, False]],
        )
        analytic = worst_corner_squared_hinge_from_aerial(
            aerial, target, lp_margin=0.04, rho=0.5,
        )
        reference = explicit_worst_corner_squared_hinge_from_aerial(
            aerial, target, lp_margin=0.04, rho=0.5,
        )
        torch.testing.assert_close(analytic, reference, rtol=0.0, atol=1e-15)

    def test_one_sided_feasible_derivative_at_zero_intensity(self):
        lp_margin, rho = 1.0, 0.5
        epsilon = 1e-7
        for target_value in (True, False):
            intensity = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            target = torch.tensor([[target_value]])
            value = worst_corner_squared_hinge_from_aerial(
                intensity.reshape(1, 1), target, lp_margin, rho,
            )
            gradient, = torch.autograd.grad(value, intensity)
            forward = worst_corner_squared_hinge_from_aerial(
                torch.tensor([[epsilon]], dtype=torch.float64),
                target, lp_margin, rho,
            )
            finite_difference = (forward - value.detach()) / epsilon
            self.assertTrue(torch.isfinite(gradient))
            self.assertAlmostEqual(
                float(gradient.item()), float(finite_difference.item()), places=5
            )

    def test_objective_autograd_gradient_is_finite_and_matches_finite_difference(self):
        basis = torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.4, 0.2], [0.1, 0.3]],
            ],
            dtype=torch.float64,
        )
        bases = {"fit0": basis}
        targets = {"fit0": torch.tensor([[True, False], [True, False]])}
        weights = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
        value = robust_corner_squared_hinge(
            weights, bases, targets, lp_margin=0.05, rho=0.5
        )
        gradient, = torch.autograd.grad(value, weights)
        epsilon = 1e-7
        plus = robust_corner_squared_hinge(
            torch.tensor([0.4 + epsilon, 0.6 - epsilon], dtype=torch.float64),
            bases, targets, lp_margin=0.05, rho=0.5,
        )
        minus = robust_corner_squared_hinge(
            torch.tensor([0.4 - epsilon, 0.6 + epsilon], dtype=torch.float64),
            bases, targets, lp_margin=0.05, rho=0.5,
        )
        numerical = (plus - minus) / (2 * epsilon)
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertAlmostEqual(
            float((gradient[0] - gradient[1]).item()),
            float(numerical.item()), places=5,
        )

    def test_objective_is_convex_on_source_simplex(self):
        generator = torch.Generator().manual_seed(71)
        basis = torch.rand((3, 4, 5), generator=generator, dtype=torch.float64)
        bases = {"fit0": basis}
        targets = {"fit0": torch.rand((4, 5), generator=generator) > 0.5}
        left = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        right = torch.tensor([0.6, 0.1, 0.3], dtype=torch.float64)
        alpha = 0.37
        middle = alpha * left + (1.0 - alpha) * right
        f_left = robust_corner_squared_hinge(left, bases, targets, 0.03, 0.5)
        f_right = robust_corner_squared_hinge(right, bases, targets, 0.03, 0.5)
        f_middle = robust_corner_squared_hinge(middle, bases, targets, 0.03, 0.5)
        self.assertLessEqual(
            float(f_middle.item()),
            alpha * float(f_left.item()) + (1.0 - alpha) * float(f_right.item()) + 1e-14,
        )

    def test_fit_gradient_diagnostics_and_margin_quantiles(self):
        bases = {
            "fit0": torch.tensor(
                [[[0.1, 0.3], [0.2, 0.4]], [[0.4, 0.2], [0.3, 0.1]]],
                dtype=torch.float64,
            )
        }
        targets = {"fit0": torch.tensor([[True, False], [True, False]])}
        weights = np.array([0.5, 0.5], dtype=np.float64)
        diagnostics = fit_objective_gradient_diagnostics(
            weights, bases, targets, lp_margin=0.05, rho=0.5
        )
        self.assertIn("robust_corner_squared_hinge", diagnostics["gradient_l2_norms"])
        self.assertIn(
            "nominal_fidelity_mse__smooth_pv_beta800",
            diagnostics["pairwise_gradient_cosines"],
        )
        self.assertTrue(all(math.isfinite(v) for v in diagnostics["objective_values"].values()))
        margins = signed_margin_quantiles(
            weights, bases["fit0"], targets["fit0"]
        )
        self.assertEqual(set(margins), {"d0.98", "nominal", "d1.02"})
        self.assertEqual(margins["nominal"]["pixel_count"], 4)
        self.assertTrue(all(math.isfinite(v) for v in diagnostics[
            "simplex_tangent_gradient_l2_norms"
        ].values()))

    def test_nonbinary_targets_are_rejected_and_tangent_projection_is_zero_mean(self):
        aerial = torch.ones((1, 2), dtype=torch.float64)
        for target in (torch.tensor([[0.0, 0.5]]),
                       torch.tensor([[0.0, float("nan")]])):
            with self.assertRaises(ValueError):
                worst_corner_squared_hinge_from_aerial(aerial, target, 0.1, 0.5)
        bases = {"fit0": torch.tensor(
            [[[0.1, 0.3]], [[0.4, 0.2]]], dtype=torch.float64
        )}
        targets = {"fit0": torch.tensor([[True, False]])}
        diagnostics = fit_objective_gradient_diagnostics(
            np.array([0.4, 0.6]), bases, targets, 0.05, 0.5
        )
        gradients = diagnostics["simplex_tangent_pairwise_gradient_cosines"]
        self.assertTrue(all(math.isfinite(value) for value in gradients.values()
                            if value is not None))

    def test_nonfinite_negative_basis_and_invalid_rho_rejected(self):
        with self.assertRaises(ValueError):
            validate_rho(0.0)
        with self.assertRaises(ValueError):
            validate_rho(float("nan"))
        with self.assertRaises(ValueError):
            validate_nonnegative_bases(
                {"fit0": torch.tensor([[[-1e-3]]], dtype=torch.float64)}
            )
        with self.assertRaises(ValueError):
            validate_nonnegative_bases(
                {"fit0": torch.tensor([[[-1e-14]]], dtype=torch.float64)}
            )
        with self.assertRaises(ValueError):
            validate_nonnegative_bases(
                {"fit0": torch.tensor([[[float("nan")]]], dtype=torch.float64)}
            )

    def test_rho_cli_default_and_explicit_value(self):
        common = [
            "--dataset-file", "dataset.pt",
            "--diagnostic-file", "diagnostic.json",
            "--candidate-plan", "candidate_plan.json",
            "--output-root", str(Path.cwd() / "out"),
        ]
        plan = {
            "schema_version": 2, "status": "prospective_candidate_plan",
            "objective": runner.OBJECTIVE_ID, "source_only": True,
            "fixed_masks": True, "seeds": list(runner.SEEDS),
            "initial_iterations_per_seed": 204,
            "prospective_rho_candidates": [0.5, 0.05, 0.01],
            "initial_rho": 0.5,
            "final3_status": "closed_not_indexed_or_evaluated",
            "acceptance": {
                "mean_cal_PV_max": 239.4,
                "mean_cal_nominal_L2_max": 56.175,
                "mean_cal_worst_dose_L2_max": 157.2375,
                "each_seed_PV_below": 268.0,
                "all_five_seeds_complete": True,
                "no_positive_target_blank": True,
                "fit_nominal_L2_zero": True,
            },
            "numerical_stopping": {
                "fw_gap_absolute_tolerance": 1e-10,
                "fw_gap_relative_tolerance": 1e-6,
            },
        }
        default_args = runner._parse_args(common)
        runner._resolve_options(default_args, plan)
        self.assertEqual(default_args.rho, 0.5)
        explicit_args = runner._parse_args(
            common + ["--candidate-index", "1", "--rho", "0.05"]
        )
        runner._resolve_options(explicit_args, plan)
        self.assertEqual(explicit_args.rho, 0.05)
        for invalid in ("0", "1.01", "nan"):
            with contextlib.redirect_stderr(io.StringIO()):
                parsed = runner._parse_args(common + ["--rho", invalid])
            with self.assertRaises(ValueError):
                runner._resolve_options(parsed, plan)

    def test_protocol_keeps_rho_and_existing_development_gate_immutable(self):
        source_path = Path(__file__).resolve()
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            plan = {
                "schema_version": 2, "status": "prospective_candidate_plan",
                "objective": runner.OBJECTIVE_ID, "source_only": True,
                "fixed_masks": True, "seeds": list(runner.SEEDS),
                "initial_iterations_per_seed": 204,
                "prospective_rho_candidates": [0.5, 0.05, 0.01],
                "initial_rho": 0.5,
                "final3_status": "closed_not_indexed_or_evaluated",
                "acceptance": {
                    "mean_cal_PV_max": 239.4,
                    "mean_cal_nominal_L2_max": 56.175,
                    "mean_cal_worst_dose_L2_max": 157.2375,
                    "each_seed_PV_below": 268.0,
                    "all_five_seeds_complete": True,
                    "no_positive_target_blank": True,
                    "fit_nominal_L2_zero": True,
                },
                "numerical_stopping": {
                    "fw_gap_absolute_tolerance": 1e-10,
                    "fw_gap_relative_tolerance": 1e-6,
                },
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            args = SimpleNamespace(
                rho=0.05, candidate_index=1, iterations=204,
                checkpoint_interval=25, timeout_seconds=3600.0,
                solver_time_limit=60.0,
            )
            diag = {
                "input": {
                    "dataset_sha256": "dataset-sha",
                    "fit_masks": [{"layout_id": "fit0"}],
                    "calibration_masks": [{"layout_id": "cal0"}],
                }
            }
            identity = runner._make_identity(
                diag, source_path, source_path,
                {"head": "test-head", "source_sha256": {},
                 "runtime": {"python": "test"}},
                torch.device("cpu"), "CPU test device", plan_path,
                runner.sha256_file(plan_path), 1, 0.05, 204, 25, 60.0,
                {"fit0": 0.0}, None,
            )
            protocol = runner._build_protocol(
                args, diag, source_path, source_path, 0.02, 0.001,
                {"fit0": 0.0}, torch.device("cpu"), "CPU test device",
                plan, plan_path, identity, None,
            )
        self.assertEqual(protocol["protocol"]["rho"], 0.05)
        self.assertEqual(protocol["protocol"]["lp_margin"], 0.02)
        self.assertEqual(protocol["calibration_gate"]["mean_band_pixels_max"], 239.4)
        self.assertEqual(protocol["calibration_gate"]["mean_nominal_l2_pixels_max"], 56.175)
        self.assertEqual(protocol["calibration_gate"]["mean_worst_dose_l2_pixels_max"], 157.2375)
        self.assertEqual(protocol["calibration_gate"]["each_seed_band_pixels_strictly_less_than"], 268.0)
        self.assertEqual(protocol["final3_status"], "closed; not indexed or evaluated")
        self.assertEqual(constrained.GATE["band_pixels"], 239.4)

    def test_runner_globals_resolve_for_every_function(self):
        def walk(code):
            yield code
            for item in code.co_consts:
                if inspect.iscode(item):
                    yield from walk(item)

        missing = set()
        for value in vars(runner).values():
            if inspect.isfunction(value) and value.__module__ == runner.__name__:
                for code in walk(value.__code__):
                    for instruction in dis.get_instructions(code):
                        if (instruction.opname == "LOAD_GLOBAL"
                                and instruction.argval not in runner.__dict__
                                and not hasattr(builtins, instruction.argval)):
                            missing.add((value.__name__, instruction.argval))
        self.assertEqual(missing, set())

    def test_run_state_rejections_do_not_modify_any_run_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            identity = {"fixture": "identity"}
            state = {
                "schema_version": runner.RUN_STATE_SCHEMA_VERSION,
                "identity": identity,
                "identity_sha256": runner._canonical_hash(identity),
                "phase": "fit_training",
                "results": {"status": "training", "seeds": []},
                "progress": {"status": "training"},
            }
            runner._seal_run_state(state)
            (run_dir / "run_state.json").write_text(json.dumps(state), encoding="utf-8")
            for name, payload in (("protocol.json", b"protocol"),
                                  ("results.json", b"results"),
                                  ("progress.json", b"progress"),
                                  ("weights.pt", b"weights")):
                (run_dir / name).write_bytes(payload)
            before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
            with self.assertRaises(ValueError):
                runner._load_run_state(run_dir, {"fixture": "other"})
            self.assertEqual(before, {path.name: path.read_bytes()
                                      for path in run_dir.iterdir()})
            broken_checksum = dict(state, payload_sha256="0" * 64)
            (run_dir / "run_state.json").write_text(
                json.dumps(broken_checksum), encoding="utf-8"
            )
            before_checksum = {path.name: path.read_bytes()
                               for path in run_dir.iterdir()}
            with self.assertRaises(ValueError):
                runner._load_run_state(run_dir, identity)
            self.assertEqual(before_checksum, {path.name: path.read_bytes()
                                               for path in run_dir.iterdir()})
            invalid_phase = dict(state, phase="unexpected")
            runner._seal_run_state(invalid_phase)
            (run_dir / "run_state.json").write_text(
                json.dumps(invalid_phase), encoding="utf-8"
            )
            before_phase = {path.name: path.read_bytes() for path in run_dir.iterdir()}
            with self.assertRaises(ValueError):
                runner._load_run_state(run_dir, identity)
            self.assertEqual(before_phase, {path.name: path.read_bytes()
                                            for path in run_dir.iterdir()})

    def test_main_rejected_resume_leaves_existing_run_files_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan = {
                "schema_version": 2, "status": "prospective_candidate_plan",
                "objective": runner.OBJECTIVE_ID, "source_only": True,
                "fixed_masks": True, "seeds": list(runner.SEEDS),
                "initial_iterations_per_seed": 204,
                "prospective_rho_candidates": [0.5, 0.05, 0.01],
                "initial_rho": 0.5,
                "final3_status": "closed_not_indexed_or_evaluated",
                "acceptance": {
                    "mean_cal_PV_max": 239.4,
                    "mean_cal_nominal_L2_max": 56.175,
                    "mean_cal_worst_dose_L2_max": 157.2375,
                    "each_seed_PV_below": 268.0,
                    "all_five_seeds_complete": True,
                    "no_positive_target_blank": True,
                    "fit_nominal_L2_zero": True,
                },
                "numerical_stopping": {
                    "fw_gap_absolute_tolerance": 1e-10,
                    "fw_gap_relative_tolerance": 1e-6,
                },
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            run_dir = root / "run"
            run_dir.mkdir()
            protocol = {"candidate_plan": {"path": str(root / "different-plan.json")}}
            (run_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            for name, payload in (("run_state.json", b"sealed state"),
                                  ("results.json", b"old results"),
                                  ("progress.json", b"old progress")):
                (run_dir / name).write_bytes(payload)
            before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = runner.main([
                    "--dataset-file", str(root / "missing-dataset.pt"),
                    "--diagnostic-file", str(root / "missing-diagnostic.json"),
                    "--candidate-plan", str(plan_path),
                    "--resume-run", str(run_dir),
                ])
            self.assertEqual(code, 2)
            self.assertEqual(before, {path.name: path.read_bytes()
                                      for path in run_dir.iterdir()})

    def test_hard_fit_selector_does_not_follow_hinge_loss(self):
        anchor = {
            "label": "LP anchor", "checkpoint_order": 0, "fit_qualified": True,
            "fit_metrics": {"mean": {"band_pixels": 332.75,
                                      "L2_worst_dose_pixels": 215.75}},
            "smooth_beta800": 1.0, "hinge": 2.90,
        }
        lower_hinge_worse_hard_pv = {
            "label": "lower hinge", "checkpoint_order": 1, "fit_qualified": True,
            "fit_metrics": {"mean": {"band_pixels": 365.0,
                                      "L2_worst_dose_pixels": 280.0}},
            "smooth_beta800": 0.1, "hinge": 2.05,
        }
        selected = runner._select_fit_checkpoint([anchor, lower_hinge_worse_hard_pv])
        self.assertEqual(selected["label"], "LP anchor")

    def test_registered_fw_gap_and_armijo_no_progress_rule(self):
        self.assertAlmostEqual(runner._gap_tolerance(0.0), 1.0001e-6, places=14)
        with mock.patch.object(runner, "robust_corner_value", return_value=1.0):
            result = runner._armijo_line_search(
                np.array([0.5, 0.5]), np.array([-0.5, 0.5]),
                loss=1.0, gap=1.0, bases_gpu={}, targets={},
                lp_margin=0.1, rho=0.5, deadline=math.inf,
            )
        self.assertFalse(result["accepted"])
        self.assertFalse(result["timed_out"])
        self.assertGreater(result["evaluations"], 0)

    def test_interrupted_fw_resumes_to_same_terminal_fit_state(self):
        class StopAfterAcceptedStep(Exception):
            pass

        class FeasiblePolytope:
            @staticmethod
            def verify(weights):
                return {"passed": True}

        def fake_candidate(weights, order, label, *args, **kwargs):
            vector = np.asarray(weights, dtype=np.float64).copy()
            return {
                "weights": vector, "checkpoint_order": int(order),
                "label": label, "fit_qualified": True,
                "fit_metrics": {"mean": {
                    "band_pixels": 10.0 + float(vector[0]),
                    "L2_worst_dose_pixels": 20.0 + float(vector[0]),
                }},
                "smooth_beta800": float(vector[0]),
                "weights_file": "mock-checkpoint-%d" % order,
            }

        def install_lmo(resumed=False):
            calls = {"n": 0}

            def solve(*args, **kwargs):
                vertex = (np.array([1.0, 0.0])
                          if calls["n"] == 0 and not resumed
                          else np.array([0.0, 1.0]))
                calls["n"] += 1
                return {"weights": vertex, "status": "optimal_verified", "attempts": []}

            return solve

        anchor = np.array([0.5, 0.5])
        fit_rows = []
        basis32 = {}
        basis_gpu = {}
        poly = FeasiblePolytope()

        def run_uninterrupted():
            callbacks = []
            with mock.patch.object(runner, "_candidate", side_effect=fake_candidate), \
                 mock.patch.object(runner.constrained, "solve_lmo", side_effect=install_lmo()), \
                 mock.patch.object(runner.constrained, "nominal_fit_check",
                                   return_value={"passed": True}), \
                 mock.patch.object(runner, "robust_corner_value_gradient",
                                   side_effect=lambda w, *a: (float(w[0]), np.array([1.0, -1.0]))), \
                 mock.patch.object(runner, "robust_corner_value",
                                   side_effect=lambda w, *a: float(np.asarray(w)[0])):
                result = runner._fw_seed(
                    17, anchor, poly, fit_rows, basis32, basis_gpu,
                    0.1, 0.5, "protocol", Path("unused"), math.inf, 10.0,
                    2, 25, lambda state, event, status: callbacks.append(state),
                )
            return result

        captured = {}

        def interrupt_callback(state, event, status):
            if event == "accepted_step":
                captured["cursor"] = state
                raise StopAfterAcceptedStep()

        with mock.patch.object(runner, "_candidate", side_effect=fake_candidate), \
             mock.patch.object(runner.constrained, "solve_lmo", side_effect=install_lmo()), \
             mock.patch.object(runner.constrained, "nominal_fit_check",
                               return_value={"passed": True}), \
             mock.patch.object(runner, "robust_corner_value_gradient",
                               side_effect=lambda w, *a: (float(w[0]), np.array([1.0, -1.0]))), \
             mock.patch.object(runner, "robust_corner_value",
                               side_effect=lambda w, *a: float(np.asarray(w)[0])):
            with self.assertRaises(StopAfterAcceptedStep):
                runner._fw_seed(
                    17, anchor, poly, fit_rows, basis32, basis_gpu,
                    0.1, 0.5, "protocol", Path("unused"), math.inf, 10.0,
                    2, 25, interrupt_callback,
                )

        def resumed_callback(state, event, status):
            captured["resumed_cursor"] = state

        with mock.patch.object(runner, "_candidate", side_effect=fake_candidate), \
             mock.patch.object(runner.constrained, "solve_lmo", side_effect=install_lmo(resumed=True)), \
             mock.patch.object(runner.constrained, "nominal_fit_check",
                               return_value={"passed": True}), \
             mock.patch.object(runner, "robust_corner_value_gradient",
                               side_effect=lambda w, *a: (float(w[0]), np.array([1.0, -1.0]))), \
             mock.patch.object(runner, "robust_corner_value",
                               side_effect=lambda w, *a: float(np.asarray(w)[0])):
            resumed = runner._fw_seed(
                17, anchor, poly, fit_rows, basis32, basis_gpu,
                0.1, 0.5, "protocol", Path("unused"), math.inf, 10.0,
                2, 25, resumed_callback, resume_state=captured["cursor"],
            )
        uninterrupted = run_uninterrupted()
        resumed_record, resumed_selected, cursor = resumed
        full_record, full_selected, full_cursor = uninterrupted
        self.assertIsNone(cursor)
        self.assertIsNone(full_cursor)
        self.assertEqual(resumed_record["status"], "complete_stationary")
        self.assertEqual(resumed_record["steps"], full_record["steps"])
        self.assertEqual(resumed_record["checkpoints"], full_record["checkpoints"])
        self.assertTrue(np.array_equal(resumed_selected["weights"], full_selected["weights"]))
        self.assertEqual(resumed_record["checkpoints"][-1]["label"], "terminal_step1")

    def test_calibration_requires_all_five_registered_fit_selections(self):
        partial = {"seeds": [
            {"seed": seed, "complete": True, "fit_qualified": True}
            for seed in runner.SEEDS[:4]
        ]}
        complete = {"seeds": [
            {"seed": seed, "complete": True, "fit_qualified": True}
            for seed in runner.SEEDS
        ]}
        self.assertFalse(runner._all_seeds_fit_qualified(partial))
        self.assertTrue(runner._all_seeds_fit_qualified(complete))

    def test_run_smoke_timeout_resume_complete_and_completed_resume_is_read_only(self):
        class FeasiblePolytope:
            def __init__(self, margin_floor):
                self.margin_floor = margin_floor

            @staticmethod
            def verify(weights):
                return {"passed": True, "residual_max": 0.0}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, diagnostic_path = root / "dataset.pt", root / "diagnostic.json"
            dataset_path.write_bytes(b"mock fit and calibration source dataset")
            diagnostic_path.write_text("{}", encoding="utf-8")
            run_output_root = root / "runs"
            fit_ids = ["fit%d" % index for index in range(4)]
            cal_ids = ["cal%d" % index for index in range(4)]
            all_ids = fit_ids + cal_ids
            candidate_plan = {
                "schema_version": 2, "status": "prospective_candidate_plan",
                "objective": runner.OBJECTIVE_ID, "source_only": True,
                "fixed_masks": True, "seeds": list(runner.SEEDS),
                "initial_iterations_per_seed": 204,
                "prospective_rho_candidates": [0.5, 0.05, 0.01],
                "initial_rho": 0.5, "base_commit": "test-base",
                "dataset": str(dataset_path.resolve()),
                "diagnostic": str(diagnostic_path.resolve()),
                "final3_status": "closed_not_indexed_or_evaluated",
                "acceptance": {
                    "mean_cal_PV_max": 239.4,
                    "mean_cal_nominal_L2_max": 56.175,
                    "mean_cal_worst_dose_L2_max": 157.2375,
                    "each_seed_PV_below": 268.0,
                    "all_five_seeds_complete": True,
                    "no_positive_target_blank": True,
                    "fit_nominal_L2_zero": True,
                },
                "numerical_stopping": {
                    "fw_gap_absolute_tolerance": 1e-10,
                    "fw_gap_relative_tolerance": 1e-6,
                },
            }
            plan_path = root / "candidate_plan.json"
            plan_path.write_text(json.dumps(candidate_plan), encoding="utf-8")
            diag = {
                "input": {
                    "dataset_sha256": runner.sha256_file(dataset_path),
                    "bases": [{"layout_id": name} for name in all_ids],
                    "fit_masks": [{"layout_id": name} for name in fit_ids],
                    "calibration_masks": [{"layout_id": name} for name in cal_ids],
                },
                "scenarios": {"fit_only.nominal": {
                    "status": "positive_margin_feasible",
                    "source_weights_full_grid": [0.5, 0.5],
                    "lp_optimal_margin": 0.2,
                }},
            }
            fit = SimpleNamespace(
                pixel_size_nm=4.0,
                masks=torch.zeros((4, 1, 128, 128), dtype=torch.float32),
            )
            cal = SimpleNamespace(
                pixel_size_nm=4.0,
                masks=torch.zeros((4, 1, 128, 128), dtype=torch.float32),
            )
            fit_rows = [{"layout_id": name, "target": torch.ones((1, 1), dtype=torch.bool)}
                        for name in fit_ids]
            cal_rows = [{"layout_id": name, "target": torch.ones((1, 1), dtype=torch.bool)}
                        for name in cal_ids]
            basis32 = {
                name: SimpleNamespace(intensities=[torch.ones((1, 1), dtype=torch.float32)])
                for name in all_ids
            }
            basis_gpu = {
                name: torch.ones((2, 1, 1), dtype=torch.float64)
                for name in all_ids
            }
            fake_poly = FeasiblePolytope(0.1)
            fw_mode = {"value": "interrupt_at_seed_begin"}
            metric_mode = {"interrupt": False}

            def fake_fw(seed, anchor, poly, rows, b32, bgpu, lp_margin, rho,
                        protocol_sha256, out_dir, deadline, solver_time_limit,
                        iterations, checkpoint_interval, callback, resume_state=None):
                weights = np.asarray(anchor, dtype=np.float64).copy()
                order, label = 0, "LP anchor"
                path = runner._expected_checkpoint_path(out_dir, seed, order)
                candidate = {
                    "weights": weights, "checkpoint_order": order, "label": label,
                    "fit_qualified": True,
                    "fit_metrics": {"mean": {
                        "band_pixels": 100.0, "L2_worst_dose_pixels": 50.0,
                    }},
                    "smooth_beta800": 0.0,
                    "weights_file": str(path),
                    "polytope": {"passed": True},
                    "nominal_fit_check": {"passed": True},
                }
                mode = fw_mode["value"]
                if mode == "interrupt_at_seed_begin":
                    raise KeyboardInterrupt()
                if mode == "timeout_once" and seed == runner.SEEDS[0]:
                    cursor = runner._fw_state_snapshot(
                        seed, weights, 1, 1,
                        {"seed": seed, "status": "running", "iterations_completed": 0,
                         "steps": [], "checkpoints": []},
                        [candidate], stage="seed_start",
                    )
                    callback(cursor, "seed_started", "training")
                    return {"seed": seed, "status": "timeout", "iterations_completed": 0}, None, cursor
                if resume_state is None:
                    cursor = runner._fw_state_snapshot(
                        seed, weights, 1, 1,
                        {"seed": seed, "status": "running", "iterations_completed": 0,
                         "steps": [], "checkpoints": []},
                        [candidate], stage="seed_start",
                    )
                    callback(cursor, "seed_started", "training")
                runner._save_checkpoint_file(
                    path, weights, seed, order, label, rho, lp_margin, protocol_sha256,
                )
                candidate["weights_sha256"] = runner.sha256_file(path)
                summary = runner._candidate_summary(candidate)
                record = {
                    "seed": seed, "status": "complete", "iterations_completed": 1,
                    "steps": [], "checkpoints": [summary],
                    "candidate_weights": [weights.tolist()],
                    "selected_training_candidate": label,
                    "selected_training_metrics": candidate["fit_metrics"]["mean"],
                    "selected_weights": weights.tolist(),
                    "selected_checkpoint_order": order,
                    "fit_qualified": True,
                }
                return record, candidate, None

            calibration_metrics = {
                "per_layout": [],
                "mean": {"band_pixels": 200.0, "L2_pixels": 0.0,
                         "L2_worst_dose_pixels": 140.0},
            }

            def metrics(*args, **kwargs):
                if metric_mode["interrupt"]:
                    metric_mode["interrupt"] = False
                    raise KeyboardInterrupt()
                return copy.deepcopy(calibration_metrics)

            provenance = {
                "head": "test-head", "clean_tree": True,
                "candidate_plan_base_commit": "test-base",
                "source_sha256": {"scripts/runner.py": "code-hash"},
                "runtime": {"python": "test", "numpy": np.__version__,
                            "scipy": "test", "torch": str(torch.__version__),
                            "torch_cuda": None},
            }
            patches = [
                mock.patch.object(runner, "_git_provenance", return_value=provenance),
                mock.patch.object(torch.cuda, "is_available", return_value=True),
                mock.patch.object(torch.cuda, "get_device_name", return_value="test GPU"),
                mock.patch.object(torch.cuda, "synchronize"),
                mock.patch.object(constrained, "load_inputs",
                                  return_value=(diag, fit, cal, fit_rows, cal_rows)),
                mock.patch.object(constrained, "prepare_bases",
                                  return_value=(basis32, {}, basis_gpu, {"max_abs_error": 0.0})),
                mock.patch.object(constrained, "support_mask",
                                  return_value=np.array([True, True])),
                mock.patch.object(constrained, "fit_matrix",
                                  return_value=(np.array([[1.0, 0.0]]), np.array([True]))),
                mock.patch.object(constrained, "signed_margin", return_value=0.2),
                mock.patch.object(constrained, "build_polytope", return_value=fake_poly),
                mock.patch.object(constrained, "nominal_fit_check",
                                  return_value={"passed": True, "mean": {"L2_pixels": 0.0},
                                                "per_layout": [{"L2_pixels": 0.0}] * 4}),
                mock.patch.object(runner, "_metric_bundle", return_value={"mean": {}}),
                mock.patch.object(runner, "_fw_seed", side_effect=fake_fw),
                mock.patch.object(constrained, "metrics", side_effect=metrics),
                mock.patch.object(constrained, "_control_check"),
                mock.patch.object(constrained, "_aggregate_cal",
                                  return_value={"mean": calibration_metrics["mean"]}),
                mock.patch.object(constrained, "_no_blank_print", return_value=True),
                mock.patch.object(constrained, "_gate", return_value={"passed": False}),
            ]
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)

                def call_run(resume=None):
                    argv = [
                        "--dataset-file", str(dataset_path),
                        "--diagnostic-file", str(diagnostic_path),
                        "--candidate-plan", str(plan_path),
                        "--device", "cuda", "--expected-gpu", "test GPU",
                        "--timeout-seconds", "3600",
                    ]
                    if resume is None:
                        argv += ["--output-root", str(run_output_root)]
                    else:
                        argv += ["--resume-run", str(resume)]
                    return runner.run(runner._parse_args(argv))

                with self.assertRaises(KeyboardInterrupt):
                    call_run()
                run_dir = next(run_output_root.iterdir())
                state = runner._read_json(run_dir / "run_state.json")
                self.assertEqual(state["results"]["status"], "interrupted")
                self.assertIsNone(state["current_seed"])

                fw_mode["value"] = "timeout_once"
                self.assertEqual(call_run(run_dir), run_dir)
                state = runner._read_json(run_dir / "run_state.json")
                self.assertEqual(state["results"]["status"], "timeout_during_training")
                self.assertEqual(state["current_seed"], runner.SEEDS[0])
                self.assertIsNotNone(state["fw_state"])
                shared_identity = runner._read_json(
                    run_dir / "protocol.json"
                )["identity"]["shared"]
                with self.assertRaisesRegex(ValueError, "resume the active or timed-out"):
                    runner._validate_prior_run(
                        run_dir, runner.sha256_file(plan_path), shared_identity, 0, 0.5,
                    )
                canonical = run_dir / "run_state.json"
                good_state_bytes = canonical.read_bytes()
                bad_state = runner._read_json(canonical)
                bad_state["fw_state"]["candidates"][0]["weights_file"] = str(
                    root / "outside-checkpoint.pt"
                )
                runner._seal_run_state(bad_state)
                canonical.write_text(json.dumps(bad_state), encoding="utf-8")
                before_rejection = {
                    str(path.relative_to(run_dir)): path.read_bytes()
                    for path in run_dir.rglob("*") if path.is_file()
                }
                with self.assertRaisesRegex(ValueError, "checkpoint path escapes"):
                    call_run(run_dir)
                after_rejection = {
                    str(path.relative_to(run_dir)): path.read_bytes()
                    for path in run_dir.rglob("*") if path.is_file()
                }
                self.assertEqual(before_rejection, after_rejection)
                canonical.write_bytes(good_state_bytes)

                fw_mode["value"] = "interrupt_at_seed_complete"
                original_mark_runtime = runner._mark_runtime

                def interrupt_before_seed_completion_commit(state, *args):
                    if state.get("progress", {}).get("event") == "fit_seed_complete":
                        raise KeyboardInterrupt()
                    return original_mark_runtime(state, *args)

                with mock.patch.object(runner, "_mark_runtime",
                                       side_effect=interrupt_before_seed_completion_commit):
                    with self.assertRaises(KeyboardInterrupt):
                        call_run(run_dir)
                state = runner._read_json(run_dir / "run_state.json")
                self.assertEqual(state["progress"]["event"], "keyboard_interrupt")
                self.assertEqual(state["current_seed"], runner.SEEDS[0])
                self.assertIsNotNone(state["fw_state"])
                self.assertEqual(state["results"]["seeds"], [])

                fw_mode["value"] = "complete"
                metric_mode["interrupt"] = True
                with self.assertRaises(KeyboardInterrupt):
                    call_run(run_dir)
                state = runner._read_json(run_dir / "run_state.json")
                self.assertEqual(state["phase"], "calibration_gate")
                self.assertEqual(state["results"]["status"], "interrupted")
                self.assertTrue(runner._all_seeds_fit_qualified(state["results"]))

                self.assertEqual(call_run(run_dir), run_dir)
                final_state_path = run_dir / "run_state.json"
                self.assertEqual(runner._read_json(final_state_path)["phase"], "complete")
                self.assertEqual(runner._read_json(final_state_path)["results"]["status"], "complete")
                before_complete_resume = {
                    str(path.relative_to(run_dir)): path.read_bytes()
                    for path in run_dir.rglob("*") if path.is_file()
                }
                self.assertEqual(call_run(run_dir), run_dir)
                after_complete_resume = {
                    str(path.relative_to(run_dir)): path.read_bytes()
                    for path in run_dir.rglob("*") if path.is_file()
                }
                self.assertEqual(before_complete_resume, after_complete_resume)
                (run_dir / "results.json").write_bytes(b"stale results sidecar")
                (run_dir / "progress.json").write_bytes(b"stale progress sidecar")
                before_main_resume = {
                    str(path.relative_to(run_dir)): path.read_bytes()
                    for path in run_dir.rglob("*") if path.is_file()
                }
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    main_result = runner.main([
                        "--dataset-file", str(dataset_path),
                        "--diagnostic-file", str(diagnostic_path),
                        "--candidate-plan", str(plan_path),
                        "--device", "cuda", "--expected-gpu", "test GPU",
                        "--timeout-seconds", "3600",
                        "--resume-run", str(run_dir),
                    ])
                self.assertEqual(main_result, 0)
                self.assertIn('"status": "complete"', stdout.getvalue())
                after_main_resume = {
                    str(path.relative_to(run_dir)): path.read_bytes()
                    for path in run_dir.rglob("*") if path.is_file()
                }
                self.assertEqual(before_main_resume, after_main_resume)
                shared = runner._read_json(run_dir / "protocol.json")["identity"]["shared"]
                prior = runner._validate_prior_run(
                    run_dir, runner.sha256_file(plan_path), shared, 1, 0.05,
                )
                self.assertEqual(prior["run_state_sha256"], runner.sha256_file(final_state_path))
                with self.assertRaisesRegex(ValueError, "terminal pre-calibration"):
                    runner._validate_prior_run(
                        run_dir, runner.sha256_file(plan_path), shared, 0, 0.5,
                    )


if __name__ == "__main__":
    unittest.main()
