import copy
from contextlib import ExitStack
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch

from scripts import optimize_source_constrained as constrained
from scripts import optimize_source_robust_corners as runner
from source_robustness import (
    SOFTCOUNT_BETA_SCHEDULE,
    SOFTCOUNT_OBJECTIVE_ID,
    critical_corner_softcount_from_aerial,
    critical_corner_softcount_value_gradient,
    target_aware_critical_corner_softcount,
)


BASE_COMMIT = "300232577f93c2162de54b3cb8efa83ac5cd670d"
HINGE_PLAN_SHA = "bd2bfa6156da48a88e00d4417ed54bf26a9dac045adc95ad42143b81dd4f8ec3"
OLD_OBJECTIVE = "target_aware_worst_dose_squared_hinge_v1"


def _gate_keys():
    return {
        "mean_band_pixels_max": constrained.GATE["band_pixels"],
        "mean_nominal_l2_pixels_max": constrained.GATE["L2_pixels"],
        "mean_worst_dose_l2_pixels_max": constrained.GATE["worst_L2_pixels"],
        "each_seed_band_pixels_strictly_less_than": constrained.LP_CAL["band_pixels"],
        "no_blank_positive_target_any_dose": True,
    }


def _write_hinge_prerequisites(directory, dataset_sha256, expected_inputs=None,
                               retry_index=None):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if expected_inputs is None:
        expected_inputs = {
            "dataset_sha256": dataset_sha256,
            "dataset_file_sha256": "b" * 64,
            "diagnostic_file_sha256": "c" * 64,
            "diagnostic_input": {"dataset_sha256": dataset_sha256,
                                 "bases": [], "fit_masks": [],
                                 "calibration_masks": []},
            "basis_parity": {"max_abs_error": 0.0},
        }
    entries = []
    shared_identity = {
        "inputs": copy.deepcopy(expected_inputs),
        "source_provenance": {"head": BASE_COMMIT},
        "fixed_protocol": {"objective_id": OLD_OBJECTIVE},
    }
    parent_info = None
    for index, rho in enumerate((0.5, 0.05, 0.01)):
        if retry_index == index:
            failed_dir = root / ("hinge_%d_attempt_1" % index)
            failed_summary = _write_hinge_attempt(
                failed_dir, dataset_sha256, expected_inputs, shared_identity,
                index, rho, 1, parent_info, status="training_failed",
            )
            parent_info = failed_summary
            attempt_number = 2
        else:
            attempt_number = 1
        run_dir = root / ("hinge_%d" % index)
        summary = _write_hinge_attempt(
            run_dir, dataset_sha256, expected_inputs, shared_identity,
            index, rho, attempt_number, parent_info, status="complete",
        )
        entries.append({
            "candidate_index": index, "rho": rho,
            "run_directory": summary["path"], "status": "complete",
            "gate_passed": False,
            "seeds": [
                {"seed": seed, "complete": True, "fit_qualified": True}
                for seed in runner.SEEDS
            ],
            "protocol_sha256": summary["protocol_sha256"],
            "run_state_sha256": summary["run_state_sha256"],
            "identity_sha256": summary["identity_sha256"],
        })
        parent_info = {
            "path": summary["path"], "candidate_index": index, "rho": rho,
            "attempt_number": attempt_number, "status": "complete",
            "protocol_sha256": summary["protocol_sha256"],
            "run_state_sha256": summary["run_state_sha256"],
        }
    manifest = {
        "schema_version": 1,
        "status": "completed_hinge_family_gate_failed",
        "created_utc": "test-only",
        "objective_id": OLD_OBJECTIVE,
        "candidate_plan_sha256": HINGE_PLAN_SHA,
        "code_commit": BASE_COMMIT,
        "dataset_sha256": dataset_sha256,
        "runs": entries,
    }
    path = root / "hinge_family_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_hinge_attempt(run_dir, dataset_sha256, expected_inputs,
                         shared_identity, index, rho, attempt_number,
                         parent_info, status):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "shared": copy.deepcopy(shared_identity),
        "candidate": {"index": index, "rho": rho},
        "attempt_number": attempt_number,
        "prior_run": copy.deepcopy(parent_info),
    }
    identity_hash = runner._canonical_hash(identity)
    protocol = {
        "schema_version": 2, "objective_id": OLD_OBJECTIVE,
        "identity": identity, "identity_sha256": identity_hash,
        "git_head": BASE_COMMIT,
        "candidate_plan": {"sha256": HINGE_PLAN_SHA},
        "candidate": {"index": index, "rho": rho,
                      "attempt_number": attempt_number,
                      "prior_run": copy.deepcopy(parent_info)},
        "protocol": {"candidate_index": index, "rho": rho},
        "dataset_sha256": dataset_sha256,
        "dataset_file_sha256": expected_inputs["dataset_file_sha256"],
        "diagnostic_file_sha256": expected_inputs["diagnostic_file_sha256"],
        "calibration_gate": _gate_keys(),
    }
    protocol_path = run_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = runner.sha256_file(protocol_path)
    complete = status == "complete"
    results = {
        "status": "complete" if complete else "training_failed",
        "objective_id": OLD_OBJECTIVE,
        "protocol_sha256": protocol_sha,
        "final3_status": "closed; not indexed or evaluated",
        "heldout_status": "not indexed or evaluated",
        "seeds": ([{"seed": seed, "complete": True, "fit_qualified": True}
                   for seed in runner.SEEDS] if complete else [
            {"seed": runner.SEEDS[0], "status": "no_progress",
             "fit_qualified": False}
        ]),
        "calibration": {"gate": {"passed": False}} if complete else {},
    }
    state = {
        "schema_version": 1, "identity": identity,
        "identity_sha256": identity_hash,
        "phase": "complete" if complete else "fit_training",
        "results": results,
        "progress": {"status": results["status"], "phase": state_phase(status)},
    }
    runner._seal_run_state(state)
    state_path = run_dir / "run_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return {
        "path": str(run_dir.resolve()),
        "candidate_index": index, "rho": rho,
        "attempt_number": attempt_number, "status": results["status"],
        "protocol_sha256": protocol_sha,
        "run_state_sha256": runner.sha256_file(state_path),
        "identity_sha256": identity_hash,
    }


def state_phase(status):
    return "complete" if status == "complete" else "fit_training"


def _soft_plan(dataset_path, diagnostic_path, manifest_path):
    return {
        "schema_version": 3, "status": "prospective_candidate_plan",
        "objective": SOFTCOUNT_OBJECTIVE_ID,
        "objective_spec": runner._softcount_objective_spec(),
        "source_only": True, "fixed_masks": True,
        "seeds": list(runner.SEEDS), "initial_iterations_per_seed": 204,
        "prospective_rho_candidates": [0.5, 0.05, 0.01], "initial_rho": 0.5,
        "base_commit": BASE_COMMIT,
        "dataset": str(Path(dataset_path).resolve()),
        "diagnostic": str(Path(diagnostic_path).resolve()),
        "candidate_sequence": (
            "First rho0.5 only after all three hinge candidates completed all five seeds "
            "and failed their frozen development gate. Advance within this new family only "
            "after previous complete all-five candidate failed. Timeout resumes same attempt; "
            "terminal numerical failure may retry same rho in preserved distinct attempt. "
            "Never weaken acceptance gates."
        ),
        "acceptance": {
            "mean_cal_PV_max": 239.4, "mean_cal_nominal_L2_max": 56.175,
            "mean_cal_worst_dose_L2_max": 157.2375,
            "each_seed_PV_below": 268, "all_five_seeds_complete": True,
            "no_positive_target_blank": True, "fit_nominal_L2_zero": True,
        },
        "final3_status": "closed_not_indexed_or_evaluated",
        "checkpoint_interval_per_block": 25, "solver_time_limit_seconds": 60,
        "numerical_stopping": {
            "fw_gap_absolute_tolerance": 1e-10,
            "fw_gap_relative_tolerance": 1e-6,
            "definition": "abs+rel*max(1,abs(objective)); nonconvex stationarity only",
        },
        "prerequisite_manifest": str(Path(manifest_path).resolve()),
        "prerequisite_family": {
            "objective": OLD_OBJECTIVE, "candidate_plan_sha256": HINGE_PLAN_SHA,
            "code_commit": BASE_COMMIT, "required_rhos": [0.5, 0.05, 0.01],
            "all_five_seeds_complete": True, "each_frozen_gate_passed": False,
        },
    }


def _write_prior_run(directory, shared_identity, plan_sha, index, rho,
                     attempt_number, parent_info=None, status="training_failed"):
    run_dir = Path(directory)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "shared": copy.deepcopy(shared_identity),
        "candidate": {"index": index, "rho": rho},
        "attempt_number": attempt_number,
        "prior_run": copy.deepcopy(parent_info),
    }
    protocol = {
        "objective_id": SOFTCOUNT_OBJECTIVE_ID,
        "objective_spec": runner._softcount_objective_spec(),
        "objective_spec_sha256": runner._canonical_hash(runner._softcount_objective_spec()),
        "identity": identity, "identity_sha256": runner._canonical_hash(identity),
        "candidate_plan": {"sha256": plan_sha},
        "protocol": {"candidate_index": index, "rho": rho},
        "candidate": {"index": index, "rho": rho,
                      "attempt_number": attempt_number,
                      "prior_run": copy.deepcopy(parent_info)},
    }
    protocol_path = run_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = runner.sha256_file(protocol_path)
    if status == "complete":
        seeds = [{"seed": seed, "complete": True, "fit_qualified": True}
                 for seed in runner.SEEDS]
        phase = "complete"
        calibration = {"gate": {"passed": False}}
    else:
        seeds = [{"seed": runner.SEEDS[0], "status": "no_progress",
                  "fit_qualified": False}]
        phase = "fit_training"
        calibration = {}
    state = {
        "schema_version": 1, "identity": identity,
        "identity_sha256": runner._canonical_hash(identity), "phase": phase,
        "results": {
            "status": status, "protocol_sha256": protocol_sha,
            "seeds": seeds, "calibration": calibration,
        },
    }
    runner._seal_run_state(state)
    (run_dir / "run_state.json").write_text(json.dumps(state), encoding="utf-8")
    return run_dir


class SoftcountMathTests(unittest.TestCase):
    def test_exact_critical_corner_formula_and_layout_mean(self):
        aerial0 = torch.tensor([[0.19, 0.225, 0.27]], dtype=torch.float64)
        target0 = torch.tensor([[1, 0, 1]], dtype=torch.int64)
        beta = 200.0
        margins = torch.tensor([
            0.98 * 0.19 - 0.225,
            0.225 - 1.02 * 0.225,
            0.98 * 0.27 - 0.225,
        ], dtype=torch.float64)
        expected = torch.sigmoid(-beta * margins).mean()
        actual = critical_corner_softcount_from_aerial(aerial0, target0, beta)
        self.assertAlmostEqual(float(actual), float(expected), places=15)
        bases = {"a": torch.stack((aerial0, aerial0 + 0.01)),
                 "b": torch.stack((aerial0 + 0.02, aerial0))}
        targets = {"a": target0, "b": 1 - target0}
        weights = torch.tensor([0.4, 0.6], dtype=torch.float64)
        per_layout = []
        for name in bases:
            aerial = torch.einsum("n,nhw->hw", weights, bases[name])
            per_layout.append(critical_corner_softcount_from_aerial(
                aerial, targets[name], beta
            ))
        loss = target_aware_critical_corner_softcount(
            weights, bases, targets, beta
        )
        self.assertAlmostEqual(float(loss), float(torch.stack(per_layout).mean()), places=15)

    def test_gradient_matches_central_difference_and_corner_identity(self):
        bases = {
            "fit0": torch.tensor(
                [[[0.15, 0.25], [0.20, 0.28]],
                 [[0.26, 0.19], [0.31, 0.17]]], dtype=torch.float64
            ),
            "fit1": torch.tensor(
                [[[0.18, 0.30], [0.24, 0.21]],
                 [[0.29, 0.16], [0.20, 0.34]]], dtype=torch.float64
            ),
        }
        targets = {
            "fit0": torch.tensor([[True, False], [True, False]]),
            "fit1": torch.tensor([[False, True], [False, True]]),
        }
        weights = np.array([0.45, 0.55], dtype=np.float64)
        value, gradient = critical_corner_softcount_value_gradient(
            weights, bases, targets, 400.0
        )
        self.assertTrue(math.isfinite(value))
        step = 1e-6
        for index in range(2):
            plus, minus = weights.copy(), weights.copy()
            plus[index] += step
            minus[index] -= step
            f_plus = float(target_aware_critical_corner_softcount(
                torch.tensor(plus), bases, targets, 400.0
            ))
            f_minus = float(target_aware_critical_corner_softcount(
                torch.tensor(minus), bases, targets, 400.0
            ))
            self.assertAlmostEqual(gradient[index], (f_plus - f_minus) / (2 * step),
                                   delta=2e-6)
        for intensity in (0.001, 0.1, 0.225, 0.7, 1.0):
            for positive in (False, True):
                critical = (0.98 * intensity - 0.225 if positive
                            else 0.225 - 1.02 * intensity)
                corners = [
                    (dose * intensity - 0.225) * (1 if positive else -1)
                    for dose in (0.98, 1.02)
                ]
                self.assertAlmostEqual(critical, min(corners), places=15)

    def test_invalid_target_beta_and_nonfinite_input_are_rejected(self):
        aerial = torch.ones((1, 2), dtype=torch.float64)
        for target in (torch.tensor([[0.0, 0.5]]),
                       torch.tensor([[0.0, float("nan")]])):
            with self.assertRaises(ValueError):
                critical_corner_softcount_from_aerial(aerial, target, 200)
        for beta in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                critical_corner_softcount_from_aerial(aerial, torch.ones_like(aerial), beta)
        with self.assertRaises(ValueError):
            critical_corner_softcount_from_aerial(
                torch.tensor([[float("inf")]]), torch.ones((1, 1)), 200
            )


class SoftcountPlanAndManifestTests(unittest.TestCase):
    def test_plan_spec_is_exact_and_cross_family_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = _soft_plan(root / "dataset.pt", root / "diag.json", root / "manifest.json")
            self.assertEqual(runner._validate_target_plan(plan), [0.5, 0.05, 0.01])
            changed = copy.deepcopy(plan)
            changed["objective_spec"]["margin_offset"] = 0.01
            with self.assertRaisesRegex(ValueError, "objective_spec"):
                runner._validate_target_plan(changed)
            with self.assertRaises(ValueError):
                runner._validate_target_plan({
                    **plan, "schema_version": 3, "objective": OLD_OBJECTIVE,
                })
            args = runner._parse_args([
                "--dataset-file", str((root / "dataset.pt").resolve()),
                "--diagnostic-file", str((root / "diag.json").resolve()),
                "--candidate-plan", str((root / "plan.json").resolve()),
                "--resume-run", str((root / "run").resolve()),
            ])
            spec = runner._softcount_objective_spec()
            spec_sha = runner._canonical_hash(spec)
            wrong_registered_candidate = {
                "objective_id": SOFTCOUNT_OBJECTIVE_ID,
                "objective_spec": spec, "objective_spec_sha256": spec_sha,
                "candidate": {"index": 1, "rho": 0.01},
                "protocol": {"max_iterations_per_seed": 204,
                             "checkpoint_interval": 25,
                             "solver_time_limit_seconds": 60.0,
                             "objective_spec_sha256": spec_sha},
            }
            with self.assertRaisesRegex(ValueError, "rho does not match"):
                runner._resolve_options(args, plan, wrong_registered_candidate)
            old_hinge_protocol = copy.deepcopy(wrong_registered_candidate)
            old_hinge_protocol["objective_id"] = OLD_OBJECTIVE
            with self.assertRaisesRegex(ValueError, "resume objective"):
                runner._resolve_options(args, plan, old_hinge_protocol)

    def test_manifest_checks_real_protocol_state_hashes_data_gates_and_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_sha = "a" * 64
            expected_inputs = {
                "dataset_sha256": dataset_sha,
                "dataset_file_sha256": "b" * 64,
                "diagnostic_file_sha256": "c" * 64,
                "diagnostic_input": {"dataset_sha256": dataset_sha,
                                     "bases": [{"layout_id": "fit0", "sha256": "d" * 64}],
                                     "fit_masks": [{"layout_id": "fit0"}],
                                     "calibration_masks": [{"layout_id": "cal0"}]},
                "basis_parity": {"layout_id": "fit0", "max_abs_error": 0.0},
            }
            manifest_path = _write_hinge_prerequisites(root, dataset_sha, expected_inputs)
            plan = _soft_plan(root / "data.pt", root / "diag.json", manifest_path)
            verified = runner._validate_hinge_family_manifest(plan, expected_inputs)
            self.assertEqual(verified["runs"], 3)
            self.assertEqual(verified["sha256"], runner.sha256_file(manifest_path))
            first_run = Path(runner._read_json(manifest_path)["runs"][0]["run_directory"])
            for artifact_name in ("protocol.json", "run_state.json"):
                artifact_path = first_run / artifact_name
                original = artifact_path.read_bytes()
                artifact_path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "artifact hashes"):
                    runner._validate_hinge_family_manifest(plan, expected_inputs)
                artifact_path.write_bytes(original)
            relative_plan = copy.deepcopy(plan)
            relative_plan["prerequisite_manifest"] = "relative/manifest.json"
            with self.assertRaisesRegex(ValueError, "path must be absolute"):
                runner._validate_hinge_family_manifest(relative_plan, expected_inputs)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            data["runs"][0], data["runs"][1] = data["runs"][1], data["runs"][0]
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate order"):
                runner._validate_hinge_family_manifest(plan, expected_inputs)
            data["runs"][0], data["runs"][1] = data["runs"][1], data["runs"][0]
            data["runs"][1]["gate_passed"] = True
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gate result"):
                runner._validate_hinge_family_manifest(plan, expected_inputs)
            data["runs"][1]["gate_passed"] = False
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            protocol_path = Path(data["runs"][0]["run_directory"]) / "protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["diagnostic_file_sha256"] = "e" * 64
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            data["runs"][0]["protocol_sha256"] = runner.sha256_file(protocol_path)
            state_path = Path(data["runs"][0]["run_directory"]) / "run_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["results"]["protocol_sha256"] = runner.sha256_file(protocol_path)
            runner._seal_run_state(state)
            state_path.write_text(json.dumps(state), encoding="utf-8")
            data["runs"][0]["run_state_sha256"] = runner.sha256_file(state_path)
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data identity"):
                runner._validate_hinge_family_manifest(plan, expected_inputs)
            protocol["diagnostic_file_sha256"] = expected_inputs["diagnostic_file_sha256"]
            protocol["git_head"] = "0" * 40
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            protocol_sha = runner.sha256_file(protocol_path)
            state["results"]["protocol_sha256"] = protocol_sha
            runner._seal_run_state(state)
            state_path.write_text(json.dumps(state), encoding="utf-8")
            data["runs"][0]["protocol_sha256"] = protocol_sha
            data["runs"][0]["run_state_sha256"] = runner.sha256_file(state_path)
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source, candidate"):
                runner._validate_hinge_family_manifest(plan, expected_inputs)

    def test_hinge_manifest_validates_retry_lineage_and_rejects_broken_chains(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_sha = "a" * 64
            expected_inputs = {
                "dataset_sha256": dataset_sha,
                "dataset_file_sha256": "b" * 64,
                "diagnostic_file_sha256": "c" * 64,
                "diagnostic_input": {"dataset_sha256": dataset_sha,
                                     "bases": [{"layout_id": "fit0", "sha256": "d" * 64}],
                                     "fit_masks": [{"layout_id": "fit0"}],
                                     "calibration_masks": [{"layout_id": "cal0"}]},
                "basis_parity": {"layout_id": "fit0", "max_abs_error": 0.0},
            }
            manifest_path = _write_hinge_prerequisites(
                root / "retry", dataset_sha, expected_inputs, retry_index=1,
            )
            plan = _soft_plan(root / "data.pt", root / "diag.json", manifest_path)
            runner._validate_hinge_family_manifest(plan, expected_inputs)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            retry_run = Path(data["runs"][1]["run_directory"])

            def resign(run_dir, manifest_entry):
                protocol_path = run_dir / "protocol.json"
                state_path = run_dir / "run_state.json"
                protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
                protocol["identity_sha256"] = runner._canonical_hash(protocol["identity"])
                protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
                protocol_sha = runner.sha256_file(protocol_path)
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["identity"] = copy.deepcopy(protocol["identity"])
                state["identity_sha256"] = protocol["identity_sha256"]
                state["results"]["protocol_sha256"] = protocol_sha
                runner._seal_run_state(state)
                state_path.write_text(json.dumps(state), encoding="utf-8")
                manifest_entry["protocol_sha256"] = protocol_sha
                manifest_entry["run_state_sha256"] = runner.sha256_file(state_path)
                manifest_entry["identity_sha256"] = protocol["identity_sha256"]

            # The listed rho=.05 attempt succeeds after a preserved terminal
            # same-rho failure; missing its prior pointer must not validate.
            protocol_path = retry_run / "protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["candidate"]["prior_run"] = None
            protocol["identity"]["prior_run"] = None
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            resign(retry_run, data["runs"][1])
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing an earlier registered attempt"):
                runner._validate_hinge_family_manifest(plan, expected_inputs)

            # A self-consistent protocol with stale parent hashes is rejected
            # while validating the same-rho retry edge.
            manifest_path = _write_hinge_prerequisites(
                root / "stale", dataset_sha, expected_inputs, retry_index=1,
            )
            plan["prerequisite_manifest"] = str(manifest_path.resolve())
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            retry_run = Path(data["runs"][1]["run_directory"])
            protocol_path = retry_run / "protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["candidate"]["prior_run"]["protocol_sha256"] = "0" * 64
            protocol["identity"]["prior_run"]["protocol_sha256"] = "0" * 64
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            resign(retry_run, data["runs"][1])
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata or artifact hash is stale"):
                runner._validate_hinge_family_manifest(plan, expected_inputs)

            manifest_path = _write_hinge_prerequisites(
                root / "duplicate", dataset_sha, expected_inputs,
            )
            plan["prerequisite_manifest"] = str(manifest_path.resolve())
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            data["runs"][1]["run_directory"] = data["runs"][0]["run_directory"]
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repeats a run directory"):
                runner._validate_hinge_family_manifest(plan, expected_inputs)

            # Exercise the explicit recursion-cycle guard on a real recorded
            # path without changing its bytes.
            first_run = Path(data["runs"][0]["run_directory"])
            with self.assertRaisesRegex(ValueError, "path cycle"):
                runner._validate_prior_run(
                    str(first_run), HINGE_PLAN_SHA,
                    json.loads((first_run / "protocol.json").read_text())["identity"]["shared"],
                    1, 0.05, _seen={first_run.resolve()},
                )

    def test_hinge_candidate_must_descend_from_the_listed_predecessor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_sha = "a" * 64
            expected_inputs = {
                "dataset_sha256": dataset_sha,
                "dataset_file_sha256": "b" * 64,
                "diagnostic_file_sha256": "c" * 64,
                "diagnostic_input": {"dataset_sha256": dataset_sha,
                                     "bases": [{"layout_id": "fit0", "sha256": "d" * 64}],
                                     "fit_masks": [{"layout_id": "fit0"}],
                                     "calibration_masks": [{"layout_id": "cal0"}]},
                "basis_parity": {"layout_id": "fit0", "max_abs_error": 0.0},
            }
            manifest_path = _write_hinge_prerequisites(
                root / "listed", dataset_sha, expected_inputs,
            )
            plan = _soft_plan(root / "data.pt", root / "diag.json", manifest_path)
            runner._validate_hinge_family_manifest(plan, expected_inputs)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            shared = json.loads(Path(manifest["runs"][0]["run_directory"])
                                .joinpath("protocol.json").read_text(encoding="utf-8"))[
                                    "identity"]["shared"]

            # Create a second, individually valid rho=.5 attempt-one run. It
            # has real protocol/state files and a distinct artifact lineage.
            decoy_dir = root / "valid_but_unlisted_rho05"
            decoy = _write_hinge_attempt(
                decoy_dir, dataset_sha, expected_inputs, shared,
                0, 0.5, 1, None, status="complete",
            )
            decoy_protocol_path = decoy_dir / "protocol.json"
            decoy_protocol = json.loads(decoy_protocol_path.read_text(encoding="utf-8"))
            decoy_protocol["run_instance"] = "second-independent-rho05-run"
            decoy_protocol_path.write_text(json.dumps(decoy_protocol), encoding="utf-8")
            decoy_protocol_sha = runner.sha256_file(decoy_protocol_path)
            decoy_state_path = decoy_dir / "run_state.json"
            decoy_state = json.loads(decoy_state_path.read_text(encoding="utf-8"))
            decoy_state["results"]["protocol_sha256"] = decoy_protocol_sha
            runner._seal_run_state(decoy_state)
            decoy_state_path.write_text(json.dumps(decoy_state), encoding="utf-8")
            decoy = {
                **decoy,
                "protocol_sha256": decoy_protocol_sha,
                "run_state_sha256": runner.sha256_file(decoy_state_path),
            }

            def point_to(run_dir, parent_info):
                run_dir = Path(run_dir)
                protocol_path = run_dir / "protocol.json"
                protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
                protocol["candidate"]["prior_run"] = copy.deepcopy(parent_info)
                protocol["identity"]["prior_run"] = copy.deepcopy(parent_info)
                protocol["identity_sha256"] = runner._canonical_hash(protocol["identity"])
                protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
                protocol_sha = runner.sha256_file(protocol_path)
                state_path = run_dir / "run_state.json"
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["identity"] = copy.deepcopy(protocol["identity"])
                state["identity_sha256"] = protocol["identity_sha256"]
                state["results"]["protocol_sha256"] = protocol_sha
                runner._seal_run_state(state)
                state_path.write_text(json.dumps(state), encoding="utf-8")
                return {
                    "path": str(run_dir.resolve()),
                    "candidate_index": protocol["candidate"]["index"],
                    "rho": protocol["candidate"]["rho"],
                    "attempt_number": protocol["candidate"]["attempt_number"],
                    "status": state["results"]["status"],
                    "protocol_sha256": protocol_sha,
                    "run_state_sha256": runner.sha256_file(state_path),
                    "identity_sha256": protocol["identity_sha256"],
                }

            b_dir = Path(manifest["runs"][1]["run_directory"])
            b_summary = point_to(b_dir, decoy)
            manifest["runs"][1].update(b_summary)
            c_dir = Path(manifest["runs"][2]["run_directory"])
            c_summary = point_to(c_dir, b_summary)
            manifest["runs"][2].update(c_summary)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            # Every artifact, identity, gate, and recursively recorded pointer
            # validates, but rho=.05 descends from the decoy, not listed A.
            for listed_index, following_rho in ((0, 0.05), (1, 0.01), (2, 0.01)):
                entry = manifest["runs"][listed_index]
                actual = runner._validate_prior_run(
                    entry["run_directory"], HINGE_PLAN_SHA, shared,
                    listed_index + 1, following_rho,
                )
                self.assertEqual(actual["protocol_sha256"], entry["protocol_sha256"])
                self.assertEqual(actual["run_state_sha256"], entry["run_state_sha256"])
            with self.assertRaisesRegex(ValueError, "listed predecessor"):
                runner._validate_hinge_family_manifest(plan, expected_inputs)

    def test_dirty_git_provenance_is_rejected(self):
        class Completed:
            def __init__(self, stdout="", returncode=0):
                self.stdout = stdout
                self.returncode = returncode
        responses = [
            Completed("300232577f93c2162de54b3cb8efa83ac5cd670d\n"),
            Completed(" M source_robustness.py\n"),
            Completed("", 0),
        ]
        with mock.patch.object(runner.subprocess, "run", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "clean committed source tree"):
                runner._git_provenance({"base_commit": BASE_COMMIT})


class SoftcountRealSolverContractTests(unittest.TestCase):
    def test_initial_lmo_timeout_can_retry_without_losing_call_history(self):
        basis = np.array([[[0.35, 0.10]], [[0.10, 0.35]]], dtype=np.float64)
        target = np.array([[True, False]])
        poly = constrained.build_polytope([basis], [target], 0.01, rho=0.5)
        anchor = np.array([0.8, 0.2], dtype=np.float64)
        fit_rows = [{"layout_id": "fit0", "target": torch.tensor(target)}]
        basis_gpu = {"fit0": torch.tensor(basis, dtype=torch.float64)}
        basis32 = {"fit0": SimpleNamespace(intensities=torch.tensor(basis, dtype=torch.float32))}
        attempts = [{"method": "highs", "presolve": True, "success": True,
                     "status_code": 0, "message": "mock optimal", "iterations": 1,
                     "time_limit_seconds": 60.0}]
        with tempfile.TemporaryDirectory() as temp:
            candidate = lambda weights, order, label, *args, **kwargs: {
                "weights": np.asarray(weights, dtype=np.float64).copy(),
                "checkpoint_order": order, "label": label, "fit_qualified": True,
                "fit_metrics": {"mean": {"band_pixels": 10.0,
                                           "L2_worst_dose_pixels": 1.0}},
                "smooth_beta800": 0.0,
                "weights_file": str(Path(temp) / ("%03d.pt" % order)),
            }
            with mock.patch.object(runner, "_candidate", side_effect=candidate), \
                 mock.patch.object(constrained, "nominal_fit_check",
                                   return_value={"passed": True}), \
                 mock.patch.object(constrained, "solve_lmo", return_value={
                     "weights": None, "status": "deadline", "attempts": [],
                 }):
                first, _, cursor = runner._softcount_seed(
                    17, anchor, poly, fit_rows, basis32, basis_gpu, 0.01, 0.5,
                    "protocol", "spec", Path(temp), math.inf, 60.0, 25,
                    lambda *args: None,
                )
            self.assertEqual(first["status"], "timeout")
            self.assertEqual(cursor["outer_lmo_invocations"], 1)
            with mock.patch.object(constrained, "nominal_fit_check",
                                   return_value={"passed": True}):
                runner._validate_softcount_state(
                    cursor, 17, anchor, poly, fit_rows, basis32,
                    runner._canonical_hash(runner._softcount_objective_spec()),
                )
            budget_cursor = copy.deepcopy(cursor)
            template_event = budget_cursor["outer_lmo_events"][0]
            budget_cursor["outer_lmo_events"] = []
            for invocation in range(1, runner.SOFTCOUNT_MAX_OUTER_LMO_CALLS + 1):
                event = copy.deepcopy(template_event)
                event["invocation"] = invocation
                budget_cursor["outer_lmo_events"].append(event)
            budget_cursor["outer_lmo_invocations"] = runner.SOFTCOUNT_MAX_OUTER_LMO_CALLS
            budget_cursor["record"]["outer_lmo_invocations"] = runner.SOFTCOUNT_MAX_OUTER_LMO_CALLS
            budget_cursor["record"]["outer_lmo_events"] = copy.deepcopy(
                budget_cursor["outer_lmo_events"]
            )
            with mock.patch.object(constrained, "nominal_fit_check",
                                   return_value={"passed": True}):
                runner._validate_softcount_state(
                    budget_cursor, 17, anchor, poly, fit_rows, basis32,
                    runner._canonical_hash(runner._softcount_objective_spec()),
                )
            with mock.patch.object(constrained, "solve_lmo",
                                   side_effect=AssertionError("205-call budget was exceeded")):
                exhausted, _, exhausted_cursor = runner._softcount_seed(
                    17, anchor, poly, fit_rows, basis32, basis_gpu, 0.01, 0.5,
                    "protocol", runner._canonical_hash(runner._softcount_objective_spec()),
                    Path(temp), math.inf, 60.0, 25, lambda *args: None,
                    resume_state=budget_cursor,
                )
            self.assertEqual(exhausted["status"], "outer_lmo_budget_exhausted")
            self.assertEqual(exhausted["outer_lmo_invocations"],
                             runner.SOFTCOUNT_MAX_OUTER_LMO_CALLS)
            self.assertIsNone(exhausted_cursor)
            calls = iter([
                {"weights": np.array([1.0, 0.0]), "status": "optimal_verified",
                 "attempts": copy.deepcopy(attempts)},
                {"weights": np.array([1.0, 0.0]), "status": "optimal_verified",
                 "attempts": copy.deepcopy(attempts)},
            ])
            with mock.patch.object(runner, "_candidate", side_effect=candidate), \
                 mock.patch.object(constrained, "nominal_fit_check",
                                   return_value={"passed": True}), \
                 mock.patch.object(constrained, "solve_lmo", side_effect=lambda *args: next(calls)), \
                 mock.patch.object(runner, "critical_corner_softcount_value_gradient",
                                   return_value=(0.5, np.array([-1.0, 1.0]))), \
                 mock.patch.object(runner, "_softcount_line_search", return_value={
                     "accepted": False, "timed_out": False, "gamma": 0.0,
                     "value": 0.5, "evaluations": 1,
                 }):
                resumed, _, final_cursor = runner._softcount_seed(
                    17, anchor, poly, fit_rows, basis32, basis_gpu, 0.01, 0.5,
                    "protocol", runner._canonical_hash(runner._softcount_objective_spec()),
                    Path(temp), math.inf, 60.0, 25, lambda *args: None,
                    resume_state=cursor,
                )
            self.assertEqual(resumed["status"], "no_progress")
            self.assertIsNone(final_cursor)
            self.assertEqual(resumed["outer_lmo_invocations"], 3)
            self.assertEqual([event["kind"] for event in resumed["outer_lmo_events"]], [
                "initial_seeded_vertex", "initial_seeded_vertex", "frank_wolfe_step",
            ])
            self.assertEqual([event["outcome"] for event in resumed["outer_lmo_events"]], [
                "timeout_retry", "completed", "completed",
            ])

    def test_inprogress_lmo_is_normalized_to_interrupted_retry_on_resume(self):
        basis = np.array([[[0.35, 0.10]], [[0.10, 0.35]]], dtype=np.float64)
        target = np.array([[True, False]])
        poly = constrained.build_polytope([basis], [target], 0.01, rho=0.5)
        anchor = np.array([0.8, 0.2], dtype=np.float64)
        fit_rows = [{"layout_id": "fit0", "target": torch.tensor(target)}]
        basis_gpu = {"fit0": torch.tensor(basis, dtype=torch.float64)}
        basis32 = {"fit0": SimpleNamespace(intensities=torch.tensor(basis, dtype=torch.float32))}
        attempts = [{"method": "highs", "presolve": True, "success": True,
                     "status_code": 0, "message": "mock optimal", "iterations": 1,
                     "time_limit_seconds": 60.0}]
        with tempfile.TemporaryDirectory() as temp:
            candidate = lambda weights, order, label, *args, **kwargs: {
                "weights": np.asarray(weights, dtype=np.float64).copy(),
                "checkpoint_order": order, "label": label, "fit_qualified": True,
                "fit_metrics": {"mean": {"band_pixels": 10.0,
                                           "L2_worst_dose_pixels": 1.0}},
                "smooth_beta800": 0.0,
                "weights_file": str(Path(temp) / ("%03d.pt" % order)),
            }
            saved = {}

            def interrupt(snapshot, event, status):
                if event == "outer_lmo_started":
                    saved["cursor"] = copy.deepcopy(snapshot)
                    raise RuntimeError("synthetic process interruption")

            with mock.patch.object(runner, "_candidate", side_effect=candidate), \
                 mock.patch.object(constrained, "nominal_fit_check",
                                   return_value={"passed": True}), \
                 mock.patch.object(constrained, "solve_lmo", return_value={
                     "weights": np.array([1.0, 0.0]), "status": "optimal_verified",
                     "attempts": copy.deepcopy(attempts),
                 }), \
                 mock.patch.object(runner, "critical_corner_softcount_value_gradient",
                                   return_value=(0.5, np.zeros(2, dtype=np.float64))):
                with self.assertRaisesRegex(RuntimeError, "synthetic process interruption"):
                    runner._softcount_seed(
                        17, anchor, poly, fit_rows, basis32, basis_gpu, 0.01, 0.5,
                        "protocol", runner._canonical_hash(runner._softcount_objective_spec()),
                        Path(temp), math.inf, 60.0, 25, interrupt,
                    )
                cursor = saved["cursor"]
                self.assertEqual(cursor["outer_lmo_events"][-1]["outcome"], "in_progress")
                runner._validate_softcount_state(
                    cursor, 17, anchor, poly, fit_rows, basis32,
                    runner._canonical_hash(runner._softcount_objective_spec()),
                )
                resumed_events = []
                record, _, final_cursor = runner._softcount_seed(
                    17, anchor, poly, fit_rows, basis32, basis_gpu, 0.01, 0.5,
                    "protocol", runner._canonical_hash(runner._softcount_objective_spec()),
                    Path(temp), math.inf, 60.0, 25,
                    lambda snapshot, event, status: resumed_events.append(
                        (copy.deepcopy(snapshot), event, status)
                    ), resume_state=cursor,
                )
            self.assertEqual(record["status"], "complete_stationary")
            self.assertIsNone(final_cursor)
            self.assertEqual(record["unobserved_lmo_calls"], 1)
            events = record["outer_lmo_events"]
            self.assertEqual(events[1]["outcome"], "interrupted_retry")
            self.assertIsNone(events[1]["solver_subattempts"])
            self.assertFalse(events[1]["subattempts_known"])
            self.assertEqual(events[2]["outcome"], "completed")
            self.assertTrue(any(event == "interrupted_lmo_retry"
                                for _, event, _ in resumed_events))

    def test_seed_uses_real_highs_attempt_lists_for_initial_and_fw_lmo(self):
        basis = np.array([[[0.35, 0.10]], [[0.10, 0.35]]], dtype=np.float64)
        target = np.array([[True, False]])
        poly = constrained.build_polytope([basis], [target], 0.01, rho=0.5)
        anchor = np.array([0.8, 0.2], dtype=np.float64)
        fit_rows = [{"layout_id": "fit0", "target": torch.tensor(target)}]
        basis_gpu = {"fit0": torch.tensor(basis, dtype=torch.float64)}
        basis32 = {"fit0": SimpleNamespace(intensities=torch.tensor(basis, dtype=torch.float32))}
        with tempfile.TemporaryDirectory() as temp:
            def fake_candidate(weights, order, label, *args, **kwargs):
                return {
                    "weights": np.asarray(weights, dtype=np.float64).copy(),
                    "checkpoint_order": order, "label": label,
                    "fit_qualified": True,
                    "fit_metrics": {"mean": {"band_pixels": 10.0,
                                              "L2_worst_dose_pixels": 1.0}},
                    "smooth_beta800": 0.0,
                    "weights_file": str(Path(temp) / ("%03d.pt" % order)),
                }

            callback_events = []
            with mock.patch.object(runner, "_candidate", side_effect=fake_candidate), \
                 mock.patch.object(constrained, "nominal_fit_check",
                                   return_value={"passed": True}), \
                 mock.patch.object(runner, "critical_corner_softcount_value_gradient",
                                   return_value=(0.5, np.array([1.0, -1.0]))), \
                 mock.patch.object(runner, "_softcount_line_search", return_value={
                     "accepted": False, "timed_out": False, "gamma": 0.0,
                     "value": 0.5, "evaluations": 1,
                 }):
                record, _, cursor = runner._softcount_seed(
                    17, anchor, poly, fit_rows, basis32, basis_gpu, 0.01, 0.5,
                    "protocol", "spec", Path(temp), math.inf, 60.0, 25,
                    lambda snapshot, event, status: callback_events.append(event),
                )
            self.assertEqual(record["status"], "no_progress")
            self.assertIsNone(cursor)
            events = record["outer_lmo_events"]
            self.assertEqual([event["kind"] for event in events], [
                "initial_seeded_vertex", "frank_wolfe_step",
            ])
            self.assertTrue(all(isinstance(event["solver_details"]["attempts"], list)
                                for event in events))
            self.assertEqual(record["solver_subattempts"], sum(
                len(event["solver_details"]["attempts"]) for event in events
            ))
            self.assertGreaterEqual(record["solver_subattempts"], 2)
            self.assertEqual(runner._lmo_attempts({"attempts": []}), [])
            with self.assertRaisesRegex(ValueError, "attempts must be a list"):
                runner._lmo_attempts({"attempts": 1})

    def test_resuming_completed_pending_lmo_does_not_call_solver_twice(self):
        basis = np.array([[[0.35, 0.10]], [[0.10, 0.35]]], dtype=np.float64)
        target = np.array([[True, False]])
        poly = constrained.build_polytope([basis], [target], 0.01, rho=0.5)
        anchor = np.array([0.8, 0.2], dtype=np.float64)
        fit_rows = [{"layout_id": "fit0", "target": torch.tensor(target)}]
        basis_gpu = {"fit0": torch.tensor(basis, dtype=torch.float64)}
        basis32 = {"fit0": SimpleNamespace(intensities=torch.tensor(basis, dtype=torch.float32))}
        attempts = [{"method": "highs", "presolve": True, "success": True,
                     "status_code": 0, "message": "mock optimal", "iterations": 1,
                     "time_limit_seconds": 60.0}]
        lmo_results = [
            {"weights": np.array([1.0, 0.0]), "status": "optimal_verified",
             "attempts": copy.deepcopy(attempts)},
            {"weights": np.array([0.54, 0.46]), "status": "optimal_verified",
             "attempts": copy.deepcopy(attempts)},
        ]
        saved = {}

        def interrupt_after_pending(snapshot, event, status):
            if event == "outer_lmo_complete":
                saved["cursor"] = copy.deepcopy(snapshot)
                raise RuntimeError("synthetic interruption after LMO commit")

        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(runner, "_candidate", side_effect=lambda weights, order,
                               label, *args, **kwargs: {
                                   "weights": np.asarray(weights, dtype=np.float64).copy(),
                                   "checkpoint_order": order, "label": label,
                                   "fit_qualified": True,
                                   "fit_metrics": {"mean": {"band_pixels": 10.0,
                                                             "L2_worst_dose_pixels": 1.0}},
                                   "smooth_beta800": 0.0,
                                   "weights_file": str(Path(temp) / ("%03d.pt" % order)),
                               }), \
             mock.patch.object(constrained, "nominal_fit_check",
                               return_value={"passed": True}), \
             mock.patch.object(constrained, "solve_lmo", side_effect=lmo_results), \
             mock.patch.object(runner, "critical_corner_softcount_value_gradient",
                               return_value=(0.5, np.array([1.0, -1.0]))):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                runner._softcount_seed(
                    17, anchor, poly, fit_rows, basis32, basis_gpu, 0.01, 0.5,
                    "protocol", "spec", Path(temp), math.inf, 60.0, 25,
                    interrupt_after_pending,
                )
            cursor = saved["cursor"]
            self.assertIsNotNone(cursor["pending_lmo"])
            runner._validate_softcount_state(
                cursor, 17, anchor, poly, fit_rows, basis32,
                runner._canonical_hash(runner._softcount_objective_spec()),
            )
            with mock.patch.object(constrained, "solve_lmo",
                                   side_effect=AssertionError("pending LMO was repeated")), \
                 mock.patch.object(runner, "_softcount_line_search", return_value={
                     "accepted": False, "timed_out": False, "gamma": 0.0,
                     "value": 0.5, "evaluations": 1,
                 }):
                record, _, resumed_cursor = runner._softcount_seed(
                    17, anchor, poly, fit_rows, basis32, basis_gpu, 0.01, 0.5,
                    "protocol", runner._canonical_hash(runner._softcount_objective_spec()),
                    Path(temp), math.inf, 60.0, 25, lambda *args: None,
                    resume_state=cursor,
                )
            self.assertEqual(record["status"], "no_progress")
            self.assertIsNone(resumed_cursor)

    def test_softcount_prior_chain_allows_preserved_same_rho_retry_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            shared = {"fixed_protocol": {"objective_id": SOFTCOUNT_OBJECTIVE_ID},
                      "inputs": {"dataset_sha256": "a" * 64}}
            plan_sha = "f" * 64
            first = _write_prior_run(Path(temp) / "rho05_attempt1", shared,
                                     plan_sha, 0, 0.5, 1)
            first_summary = runner._validate_prior_run(
                first, plan_sha, shared, candidate_index=0, rho=0.5,
            )
            second = _write_prior_run(Path(temp) / "rho05_attempt2", shared,
                                      plan_sha, 0, 0.5, 2, first_summary,
                                      status="complete")
            summary = runner._validate_prior_run(
                second, plan_sha, shared, candidate_index=1, rho=0.05,
            )
            self.assertEqual(summary["attempt_number"], 2)
            self.assertEqual(summary["status"], "complete")

            bad_parent = copy.deepcopy(first_summary)
            bad_parent["protocol_sha256"] = "0" * 64
            stale = _write_prior_run(Path(temp) / "rho05_stale_parent", shared,
                                     plan_sha, 0, 0.5, 2, bad_parent,
                                     status="complete")
            with self.assertRaisesRegex(ValueError, "metadata or artifact hash"):
                runner._validate_prior_run(
                    stale, plan_sha, shared, candidate_index=1, rho=0.05,
                )

            relative_parent = copy.deepcopy(first_summary)
            relative_parent["path"] = "relative/prior"
            relative = _write_prior_run(Path(temp) / "rho05_relative_parent", shared,
                                       plan_sha, 0, 0.5, 2, relative_parent,
                                       status="complete")
            with self.assertRaisesRegex(ValueError, "lineage path must be absolute"):
                runner._validate_prior_run(
                    relative, plan_sha, shared, candidate_index=1, rho=0.05,
                )


class SoftcountRunnerSmokeTests(unittest.TestCase):
    def test_schedule_checkpoint_budget_and_no_progress_is_terminal_failure(self):
        class FeasiblePolytope:
            @staticmethod
            def verify(weights):
                value = np.asarray(weights, dtype=np.float64)
                passed = (value.shape == (2,) and np.isfinite(value).all()
                          and np.min(value) >= -1e-12
                          and abs(float(np.sum(value)) - 1.0) <= 1e-10)
                return {"passed": passed, "residual_max": 0.0 if passed else 1.0}

        poly = FeasiblePolytope()
        rows = [{"layout_id": "fit0", "target": torch.ones((1, 1), dtype=torch.bool)}]
        bases = {"fit0": torch.tensor([[[0.2]], [[0.25]]], dtype=torch.float64)}
        anchor = np.array([0.5, 0.5])
        with tempfile.TemporaryDirectory() as temp:
            def fake_candidate(weights, order, label, *args, **kwargs):
                return {
                    "weights": np.asarray(weights, dtype=np.float64).copy(),
                    "checkpoint_order": order, "label": label,
                    "fit_qualified": True,
                    "fit_metrics": {"mean": {
                        "band_pixels": 100.0, "L2_worst_dose_pixels": 10.0,
                        "L2_pixels": 0.0,
                    }},
                    "smooth_beta800": 0.0,
                    "weights_file": str(Path(temp) / ("%03d.pt" % order)),
                }

            callbacks = []

            def callback(snapshot, event, status):
                callbacks.append((copy.deepcopy(snapshot), event, status))

            with mock.patch.object(runner, "_candidate", side_effect=fake_candidate), \
                 mock.patch.object(constrained, "nominal_fit_check",
                                   return_value={"passed": True}), \
                 mock.patch.object(constrained, "solve_lmo", return_value={
                     "weights": np.array([1.0, 0.0]),
                     "status": "optimal_verified", "attempts": [{"method": "highs", "presolve": True, "success": True, "status_code": 0, "message": "mock optimal", "iterations": 1, "time_limit_seconds": 60.0}],
                 }), \
                 mock.patch.object(runner, "critical_corner_softcount_value_gradient",
                                   return_value=(1.0, np.array([-100.0, 100.0]))), \
                 mock.patch.object(runner, "_softcount_line_search", return_value={
                     "accepted": True, "timed_out": False, "gamma": 0.001,
                     "value": 0.99, "evaluations": 1,
                 }):
                record, selected, cursor = runner._softcount_seed(
                    17, anchor, poly, rows, {}, bases, 0.01, 0.5,
                    "protocol", "spec", Path(temp), math.inf, 60.0,
                    25, callback,
                )
            self.assertEqual(record["status"], "complete")
            self.assertIsNone(cursor)
            self.assertEqual(record["accepted_step_counts"], [68, 68, 68], json.dumps({
                "status": record["status"],
                "calls": record["outer_lmo_invocations"],
                "counts": record["accepted_step_counts"],
                "last_step": record["steps"][-1:] }))
            self.assertEqual(record["outer_lmo_invocations"], 205)
            self.assertEqual(record["solver_subattempts"], 205)
            self.assertEqual(len(record["checkpoints"]), 11)
            self.assertEqual(
                [step for step in (25, 50, 68) for _ in SOFTCOUNT_BETA_SCHEDULE],
                [25, 25, 25, 50, 50, 50, 68, 68, 68],
            )
            checkpoint_labels = [row["label"] for row in record["checkpoints"]]
            self.assertEqual(
                checkpoint_labels[2:],
                ["beta%d_step%d" % (int(beta), step)
                 for beta in SOFTCOUNT_BETA_SCHEDULE for step in (25, 50, 68)],
            )
            self.assertEqual(selected["label"], "LP anchor")

            with mock.patch.object(runner, "_candidate", side_effect=fake_candidate), \
                 mock.patch.object(constrained, "nominal_fit_check",
                                   return_value={"passed": True}), \
                 mock.patch.object(constrained, "solve_lmo", return_value={
                     "weights": np.array([1.0, 0.0]),
                     "status": "optimal_verified", "attempts": [{"method": "highs", "presolve": True, "success": True, "status_code": 0, "message": "mock optimal", "iterations": 1, "time_limit_seconds": 60.0}],
                 }), \
                 mock.patch.object(runner, "critical_corner_softcount_value_gradient",
                                   return_value=(1.0, np.array([-1.0, 1.0]))), \
                 mock.patch.object(runner, "_softcount_line_search", return_value={
                     "accepted": False, "timed_out": False, "gamma": 0.0,
                     "value": 1.0, "evaluations": 1,
                 }):
                failed, failed_candidate, failed_cursor = runner._softcount_seed(
                    17, anchor, poly, rows, {}, bases, 0.01, 0.5,
                    "protocol", "spec", Path(temp), math.inf, 60.0,
                    25, callback,
                )
            self.assertEqual(failed["status"], "no_progress")
            self.assertEqual(failed["failure_stage"], "armijo_line_search")
            self.assertIsNone(failed_cursor)
            self.assertIsNotNone(failed_candidate)
            self.assertNotIn(failed["status"], {"complete", "complete_stationary"})

    def test_run_timeout_resume_complete_and_calibration_waits_for_all_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_path, diagnostic_path = root / "dataset.pt", root / "diagnostic.json"
            dataset_path.write_bytes(b"synthetic test-only source inputs")
            diagnostic_path.write_text("{}", encoding="utf-8")
            fit_ids = ["fit%d" % index for index in range(4)]
            cal_ids = ["cal%d" % index for index in range(4)]
            all_ids = fit_ids + cal_ids
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
                    "lp_optimal_margin": 0.1,
                }},
            }
            prereq_inputs = {
                "dataset_sha256": diag["input"]["dataset_sha256"],
                "dataset_file_sha256": runner.sha256_file(dataset_path),
                "diagnostic_file_sha256": runner.sha256_file(diagnostic_path),
                "diagnostic_input": diag["input"],
                "basis_parity": {"max_abs_error": 0.0},
            }
            manifest_path = _write_hinge_prerequisites(
                root / "prerequisites", diag["input"]["dataset_sha256"],
                prereq_inputs,
            )
            plan = _soft_plan(dataset_path, diagnostic_path, manifest_path)
            plan_path = root / "softcount_plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            output_root = root / "runs"
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
                name: SimpleNamespace(intensities=[
                    torch.tensor([[0.225]], dtype=torch.float32)
                ]) for name in all_ids
            }
            basis_gpu = {
                name: torch.tensor([[[0.20]], [[0.25]]], dtype=torch.float64)
                for name in fit_ids
            }

            class FeasiblePolytope:
                @staticmethod
                def verify(weights):
                    value = np.asarray(weights, dtype=np.float64)
                    passed = (value.shape == (2,) and np.isfinite(value).all()
                              and np.min(value) >= -1e-12
                              and abs(float(np.sum(value)) - 1.0) <= 1e-10)
                    return {"passed": passed, "residual_max": 0.0 if passed else 1.0}

            fake_poly = FeasiblePolytope()
            provenance = {
                "head": "softcount-test-head", "clean_tree": True,
                "candidate_plan_base_commit": BASE_COMMIT,
                "source_sha256": {"scripts/optimize_source_robust_corners.py": "test"},
                "runtime": {"python": "test", "numpy": np.__version__,
                            "scipy": "test", "torch": str(torch.__version__),
                            "torch_cuda": None},
            }
            metric_calls = []
            run_dir_holder = {"path": None}
            candidate_fit_metrics = {
                "mean": {"band_pixels": 100.0, "L2_pixels": 0.0,
                         "L2_worst_dose_pixels": 100.0},
                "per_layout": [{"L2_pixels": 0} for _ in fit_rows],
            }
            anchor_cal = {
                "mean": copy.deepcopy(constrained.LP_CAL),
                "per_layout": [{
                    "target_positive_pixels": 1,
                    "per_corner": [{"predicted_positive_pixels": 1} for _ in range(3)],
                } for _ in cal_rows],
            }
            candidate_cal = {
                "mean": {"band_pixels": 200.0, "L2_pixels": 0.0,
                         "L2_worst_dose_pixels": 140.0},
                "per_layout": [{
                    "target_positive_pixels": 1,
                    "per_corner": [{"predicted_positive_pixels": 1} for _ in range(3)],
                } for _ in cal_rows],
            }

            def fake_metrics(*args, **kwargs):
                metric_calls.append(True)
                state_path = run_dir_holder["path"] / "run_state.json"
                state = runner._read_json(state_path)
                if len(metric_calls) == 1:
                    self.assertEqual(state["phase"], "calibration_gate")
                    self.assertTrue(runner._all_seeds_fit_qualified(state["results"]))
                    return copy.deepcopy(anchor_cal)
                return copy.deepcopy(candidate_cal)

            lmo_mode = {"phase": "fresh", "calls": 0}
            virtual_time = {"now": 0.0}

            def success_lmo(weights):
                return {"weights": np.asarray(weights, dtype=np.float64),
                        "status": "optimal_verified", "attempts": [{
                            "method": "highs", "presolve": True, "success": True,
                            "status_code": 0, "message": "mock optimal",
                            "iterations": 1, "time_limit_seconds": 60.0,
                        }]}

            def fake_lmo(poly, objective, deadline, solver_time_limit):
                lmo_mode["calls"] += 1
                if lmo_mode["phase"] == "fresh":
                    if lmo_mode["calls"] == 1:
                        return success_lmo([1.0, 0.0])
                    virtual_time["now"] = 3601.0
                    return {"weights": None, "status": "solver_failure", "attempts": [{
                        "method": "highs", "presolve": True, "success": False,
                        "status_code": 1, "message": "time limit reached",
                        "iterations": 1, "time_limit_seconds": 60.0,
                    }]}
                if lmo_mode["phase"] == "retry_timeout":
                    return {"weights": None, "status": "deadline", "attempts": []}
                call = lmo_mode["calls"]
                return success_lmo([1.0, 0.0] if call == 1 else [0.525, 0.475])

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
                mock.patch.object(constrained, "signed_margin", return_value=0.1),
                mock.patch.object(constrained, "build_polytope", return_value=fake_poly),
                mock.patch.object(constrained, "nominal_fit_check",
                                  return_value={"passed": True, "mean": {"L2_pixels": 0.0},
                                                "per_layout_L2_pixels": {name: 0 for name in fit_ids}}),
                mock.patch.object(runner, "_metric_bundle", return_value=candidate_fit_metrics),
                mock.patch.object(constrained, "smooth_value", return_value=0.0),
                mock.patch.object(runner, "critical_corner_softcount_value_gradient",
                                  return_value=(1.0, np.zeros(2, dtype=np.float64))),
                mock.patch.object(constrained, "solve_lmo", side_effect=fake_lmo),
                mock.patch.object(constrained, "metrics", side_effect=fake_metrics),
                mock.patch.object(runner.time, "monotonic",
                                  side_effect=lambda: virtual_time["now"]),
            ]
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)

                def invoke(resume_path=None):
                    virtual_time["now"] = 0.0
                    argv = [
                        "--dataset-file", str(dataset_path),
                        "--diagnostic-file", str(diagnostic_path),
                        "--candidate-plan", str(plan_path),
                        "--device", "cuda", "--expected-gpu", "test GPU",
                        "--timeout-seconds", "3600",
                    ]
                    if resume_path is None:
                        argv += ["--output-root", str(output_root)]
                    else:
                        argv += ["--resume-run", str(resume_path)]
                    return runner.run(runner._parse_args(argv))

                first_run = invoke()
                run_dir = next(output_root.iterdir())
                run_dir_holder["path"] = run_dir
                state = runner._read_json(run_dir / "run_state.json")
                self.assertEqual(first_run, run_dir)
                self.assertEqual(state["results"]["status"], "timeout_during_training")
                self.assertEqual(state["phase"], "fit_training")
                self.assertEqual(state["results"]["seeds"], [])
                self.assertEqual(metric_calls, [])
                cursor = state["softcount_state"]
                self.assertEqual(cursor["phase_index"], 0)
                self.assertEqual(cursor["local_next_step"], 1)
                self.assertEqual(cursor["outer_lmo_invocations"], 2)
                self.assertEqual(
                    [event["outcome"] for event in cursor["outer_lmo_events"]],
                    ["completed", "timeout_retry"],
                )
                self.assertEqual(cursor["outer_lmo_events"][-1]["solver_status"],
                                 "solver_failure")
                runner._validate_softcount_state(
                    cursor, 17, np.array([0.5, 0.5]), fake_poly, fit_rows,
                    basis32, runner._canonical_hash(runner._softcount_objective_spec()),
                )
                tampered_cursor = copy.deepcopy(cursor)
                tampered_cursor["record"]["accepted_step_counts"] = [1, 0, 0]
                with self.assertRaisesRegex(ValueError, "block counts"):
                    runner._validate_softcount_state(
                        tampered_cursor, 17, np.array([0.5, 0.5]), fake_poly,
                        fit_rows, basis32,
                        runner._canonical_hash(runner._softcount_objective_spec()),
                    )
                tampered_cursor = copy.deepcopy(cursor)
                tampered_cursor["outer_lmo_events"][-1]["solver_subattempts"] = 2
                with self.assertRaisesRegex(ValueError, "attempt details"):
                    runner._validate_softcount_state(
                        tampered_cursor, 17, np.array([0.5, 0.5]), fake_poly,
                        fit_rows, basis32,
                        runner._canonical_hash(runner._softcount_objective_spec()),
                    )
                lmo_mode["phase"], lmo_mode["calls"] = "retry_timeout", 0
                self.assertEqual(invoke(run_dir), run_dir)
                state = runner._read_json(run_dir / "run_state.json")
                cursor = state["softcount_state"]
                self.assertEqual(cursor["outer_lmo_invocations"], 3)
                self.assertEqual(
                    [event["outcome"] for event in cursor["outer_lmo_events"]],
                    ["completed", "timeout_retry", "timeout_retry"],
                )
                self.assertEqual(cursor["outer_lmo_events"][-1]["solver_status"], "deadline")
                runner._validate_softcount_state(
                    cursor, 17, np.array([0.5, 0.5]), fake_poly, fit_rows,
                    basis32, runner._canonical_hash(runner._softcount_objective_spec()),
                )

                lmo_mode["phase"], lmo_mode["calls"] = "resume", 0
                self.assertEqual(invoke(run_dir), run_dir)
                state = runner._read_json(run_dir / "run_state.json")
                self.assertEqual(state["phase"], "complete", json.dumps({
                    "status": state["results"].get("status"),
                    "progress": state.get("progress"),
                    "seed_count": len(state["results"].get("seeds", [])),
                    "last_seed": state["results"].get("seeds", [])[-1:]
                }))
                self.assertEqual(state["results"]["status"], "complete")
                self.assertEqual(len(state["results"]["seeds"]), 5)
                self.assertTrue(all(
                    row["status"] == "complete_stationary"
                    and [step["beta"] for step in row["steps"]] == [200.0, 400.0, 800.0]
                    and row["outer_lmo_invocations"] == (6 if row["seed"] == 17 else 4)
                    for row in state["results"]["seeds"]
                ))
                self.assertEqual(len(metric_calls), 6)
                self.assertTrue(state["results"]["calibration"]["gate"]["passed"])
                self.assertEqual(
                    state["results"]["selection"]["selected_arm"],
                    "critical_corner_softcount",
                )
                self.assertEqual(state["results"]["final3_status"],
                                 "closed; not indexed or evaluated")
                saved_identity = state["identity"]
                incompatible = copy.deepcopy(saved_identity)
                incompatible["shared"]["fixed_protocol"]["objective_id"] = OLD_OBJECTIVE
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    runner._load_run_state(run_dir, incompatible)
                resume_args = runner._parse_args([
                    "--dataset-file", str(dataset_path),
                    "--diagnostic-file", str(diagnostic_path),
                    "--candidate-plan", str(plan_path),
                    "--resume-run", str(run_dir),
                ])
                wrong_objective_protocol = runner._read_json(run_dir / "protocol.json")
                wrong_objective_protocol["objective_id"] = OLD_OBJECTIVE
                with self.assertRaisesRegex(ValueError, "resume objective"):
                    runner._resolve_options(resume_args, plan, wrong_objective_protocol)


if __name__ == "__main__":
    unittest.main()
