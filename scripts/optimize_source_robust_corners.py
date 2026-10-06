"""Preregistered source-only robust-corner quality experiment; final3 stays closed."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import torch

from scripts import optimize_source_constrained as constrained
from source_robustness import (
    BASIS_NONNEGATIVE_TOL, HIGH_DOSE, LOW_DOSE, NOMINAL_DOSE, SMOOTH_PV_BETA,
    THRESHOLD, fit_objective_gradient_diagnostics, robust_corner_value,
    robust_corner_value_gradient, signed_margin_quantiles, validate_nonnegative_bases,
    validate_rho,
)

SEEDS = (17, 29, 43, 71, 101)
DEFAULT_ITERATIONS = 204
DEFAULT_CHECKPOINT_INTERVAL = 25
ARMIJO_C1 = 1e-4
MIN_LINE_SEARCH_GAMMA = 2.0 ** -24
FW_GAP_ABS_TOLERANCE = 1e-10
FW_GAP_REL_TOLERANCE = 1e-6
OBJECTIVE_ID = "target_aware_worst_dose_squared_hinge_v1"
RUN_STATE_SCHEMA_VERSION = 1
RESUMABLE_STATUSES = {
    "preregistered", "training", "timeout_during_training", "interrupted",
    "calibration_gate_running", "timeout_during_calibration",
}
RUN_PHASES = {"fit_training", "calibration_gate", "complete"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_json(value):
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(x) for x in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _persist(out_dir, results, progress):
    constrained.atomic_json(out_dir / "results.json", _finite_json(results))
    constrained.atomic_json(out_dir / "progress.json", _finite_json(progress))


def _require_output_outside_source_tree(path):
    output = Path(path).resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        return output
    raise ValueError("output root must be outside the source tree")


def _mark_runtime(state, invocation_started, base_seconds, base_invocations,
                  timeout_seconds):
    elapsed = max(0.0, time.monotonic() - invocation_started)
    runtime = state["results"].setdefault("runtime", {})
    runtime.update({
        "cumulative_seconds": float(base_seconds + elapsed),
        "invocation_count": int(base_invocations + 1),
        "last_invocation_seconds": elapsed,
        "last_invocation_timeout_seconds": float(timeout_seconds),
    })


def _canonical_hash(payload):
    return constrained._canonical_hash(_finite_json(payload))


def _seal_run_state(state):
    state.pop("payload_sha256", None)
    state["payload_sha256"] = _canonical_hash(state)
    return state


def _verify_run_state_checksum(state):
    recorded = state.get("payload_sha256")
    payload = copy.deepcopy(state)
    payload.pop("payload_sha256", None)
    if not isinstance(recorded, str) or recorded != _canonical_hash(payload):
        raise ValueError("canonical run_state.json checksum mismatch")


def _write_canonical_state(out_dir, state):
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _seal_run_state(state)
    constrained.atomic_json(out_dir / "run_state.json", _finite_json(state))
    try:
        constrained._atomic_sidecar(out_dir / "progress.json", _finite_json(state["progress"]))
        constrained._atomic_sidecar(out_dir / "results.json", _finite_json(state["results"]))
    except constrained.SidecarWriteError:
        raise


def _load_run_state(out_dir, identity):
    path = Path(out_dir) / "run_state.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("run has no canonical resumable state") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("canonical run state is corrupt; refusing sidecar inference") from exc
    if state.get("schema_version") != RUN_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported canonical run-state schema")
    _verify_run_state_checksum(state)
    if state.get("identity") != identity or state.get("identity_sha256") != _canonical_hash(identity):
        raise ValueError("resume identity mismatch (code, data, runtime, plan, or configuration changed)")
    if state.get("phase") not in RUN_PHASES or not isinstance(state.get("results"), dict):
        raise ValueError("canonical run state has an invalid phase or results payload")
    status = state["results"].get("status")
    if status not in RESUMABLE_STATUSES | {"complete"}:
        raise ValueError("run status %r is not resumable" % status)
    if (state["phase"] == "complete") != (status == "complete"):
        raise ValueError("canonical complete status and phase disagree")
    return state


def _recover_sidecars(out_dir, state):
    """Canonical JSON is authoritative; display sidecars are repairable."""
    constrained._atomic_sidecar(out_dir / "results.json", _finite_json(state["results"]))
    constrained._atomic_sidecar(out_dir / "progress.json", _finite_json(state["progress"]))


def _validate_target_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema_version") != 2:
        raise ValueError("candidate plan must use schema_version 2")
    if plan.get("status") != "prospective_candidate_plan":
        raise ValueError("candidate plan is not prospective")
    if plan.get("objective") != OBJECTIVE_ID:
        raise ValueError("candidate plan objective ID does not match this runner")
    if plan.get("source_only") is not True or plan.get("fixed_masks") is not True:
        raise ValueError("candidate plan must freeze masks and use source-only variables")
    if tuple(plan.get("seeds", ())) != SEEDS:
        raise ValueError("candidate plan seed list differs from the registered five seeds")
    if plan.get("initial_iterations_per_seed") != DEFAULT_ITERATIONS:
        raise ValueError("candidate plan iteration budget differs from this runner")
    rhos = plan.get("prospective_rho_candidates")
    if not isinstance(rhos, list) or not rhos or any(
        not math.isfinite(float(x)) or not 0.0 < float(x) <= 1.0 for x in rhos
    ):
        raise ValueError("candidate plan has an invalid prospective rho list")
    if float(plan.get("initial_rho", math.nan)) != float(rhos[0]):
        raise ValueError("candidate plan initial rho must be its first candidate")
    if plan.get("final3_status") != "closed_not_indexed_or_evaluated":
        raise ValueError("candidate plan does not keep final3 closed")
    expected_acceptance = {
        "mean_cal_PV_max": constrained.GATE["band_pixels"],
        "mean_cal_nominal_L2_max": constrained.GATE["L2_pixels"],
        "mean_cal_worst_dose_L2_max": constrained.GATE["worst_L2_pixels"],
        "each_seed_PV_below": constrained.LP_CAL["band_pixels"],
        "all_five_seeds_complete": True,
        "no_positive_target_blank": True,
        "fit_nominal_L2_zero": True,
    }
    if plan.get("acceptance") != expected_acceptance:
        raise ValueError("candidate plan acceptance criteria differ from frozen gates")
    stopping = plan.get("numerical_stopping")
    if not isinstance(stopping, dict) or (
        float(stopping.get("fw_gap_absolute_tolerance", math.nan)) != FW_GAP_ABS_TOLERANCE
        or float(stopping.get("fw_gap_relative_tolerance", math.nan)) != FW_GAP_REL_TOLERANCE
    ):
        raise ValueError("candidate plan stopping tolerances differ from this runner")
    return [float(x) for x in rhos]


def _candidate_plan_selection(plan, candidate_index, rho, iterations):
    rhos = _validate_target_plan(plan)
    if not isinstance(candidate_index, int) or not 0 <= candidate_index < len(rhos):
        raise ValueError("candidate index is outside the registered rho sequence")
    if float(rho) != rhos[candidate_index]:
        raise ValueError("rho does not match candidate index in the plan")
    if iterations != int(plan["initial_iterations_per_seed"]):
        raise ValueError("iteration budget differs from the candidate plan")
    return {"candidate_index": candidate_index, "rho": float(rho), "rho_candidates": rhos}


def _metric_bundle(rows, basis32, basis_gpu, weights, lp_margin, rho):
    metrics = constrained.metrics(rows, basis32, weights)
    by_id = {row["layout_id"]: row for row in metrics["per_layout"]}
    for row in rows:
        by_id[row["layout_id"]]["signed_margin_quantiles"] = signed_margin_quantiles(
            weights, basis32[row["layout_id"]].intensities[0], row["target"],
        )
    metrics["objective_diagnostics"] = fit_objective_gradient_diagnostics(
        weights,
        basis_gpu,
        {row["layout_id"]: row["target"] for row in rows},
        lp_margin, rho,
    )
    metrics["robust_corner_hinge"] = robust_corner_value(
        weights, basis_gpu, {row["layout_id"]: row["target"] for row in rows},
        lp_margin, rho,
    )
    return metrics


def _candidate(weights, order, label, poly, fit_rows, basis32, basis_gpu,
               lp_margin, rho, out_dir, seed, protocol_sha256):
    source_weights = np.asarray(weights, dtype=np.float64).copy()
    hard_fit = constrained.nominal_fit_check(fit_rows, basis32, source_weights)
    fit_metrics = _metric_bundle(
        fit_rows, basis32, basis_gpu, source_weights, lp_margin, rho
    )
    poly_check = poly.verify(source_weights)
    qualified = bool(
        poly_check["passed"] and hard_fit["passed"]
        and fit_metrics["mean"]["L2_pixels"] == 0.0
        and all(x["L2_pixels"] == 0 for x in fit_metrics["per_layout"])
    )
    weight_path = out_dir / "weights" / ("seed_%d_checkpoint_%03d.pt" % (seed, order))
    _save_checkpoint_file(
        weight_path, source_weights, seed, order, label, rho, lp_margin,
        protocol_sha256,
    )
    weight_sha256 = sha256_file(weight_path)
    return {
        "weights": source_weights,
        "checkpoint_order": int(order),
        "label": label,
        "weights_file": str(weight_path),
        "weights_sha256": weight_sha256,
        "polytope": poly_check,
        "nominal_fit_check": hard_fit,
        "fit_metrics": fit_metrics,
        "smooth_beta800": constrained.smooth_value(
            source_weights, basis_gpu, SMOOTH_PV_BETA
        ),
        "fit_qualified": qualified,
    }


def _save_checkpoint_file(path, weights, seed, order, label, rho, lp_margin,
                          protocol_sha256):
    source_weights = np.asarray(weights, dtype=np.float64)
    constrained.atomic_torch(
        path,
        {
            "weights_supported_float64": torch.tensor(source_weights, dtype=torch.float64),
            "weights_full_grid_float64": torch.tensor(
                constrained.expand(source_weights, constrained.support_mask()),
                dtype=torch.float64,
            ),
            "metadata": {
                "objective_id": OBJECTIVE_ID, "seed": int(seed),
                "checkpoint_order": int(order), "label": str(label),
                "rho": float(rho), "lp_margin": float(lp_margin),
                "protocol_sha256": protocol_sha256, "fit_only": True,
            },
        },
    )


def _expected_checkpoint_path(out_dir, seed, order):
    return Path(out_dir) / "weights" / ("seed_%d_checkpoint_%03d.pt" % (seed, order))


def _verify_or_repair_checkpoint(out_dir, candidate, seed, rho, lp_margin,
                                 protocol_sha256):
    weights = np.asarray(candidate["weights"], dtype=np.float64)
    order = int(candidate["checkpoint_order"])
    path = _expected_checkpoint_path(out_dir, seed, order)
    if str(Path(candidate["weights_file"]).resolve()) != str(path.resolve()):
        raise ValueError("checkpoint path escapes or disagrees with the run directory")
    expected_metadata = {
        "objective_id": OBJECTIVE_ID, "seed": int(seed),
        "checkpoint_order": order, "label": candidate["label"],
        "rho": float(rho), "lp_margin": float(lp_margin),
        "protocol_sha256": protocol_sha256, "fit_only": True,
    }
    valid = False
    if path.exists():
        try:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            saved_weights = saved["weights_supported_float64"].detach().cpu().numpy()
            valid = (
                np.array_equal(saved_weights, weights)
                and saved.get("metadata") == expected_metadata
            )
        except Exception:
            valid = False
    if not valid:
        _save_checkpoint_file(
            path, weights, seed, order, candidate["label"], rho, lp_margin,
            protocol_sha256,
        )
    candidate["weights_file"] = str(path)
    candidate["weights_sha256"] = sha256_file(path)
    return candidate


def _candidate_summary(row):
    return {
        key: value for key, value in row.items()
        if key != "weights"
    }


def _select_fit_checkpoint(candidates):
    eligible = [x for x in candidates if x["fit_qualified"]]
    if not eligible:
        return None
    return min(eligible, key=lambda x: (
        x["fit_metrics"]["mean"]["band_pixels"],
        x["fit_metrics"]["mean"]["L2_worst_dose_pixels"],
        x["smooth_beta800"],
        x["checkpoint_order"],
    ))


def _aerial_values(weights, bases_gpu):
    device = next(iter(bases_gpu.values())).device
    w = torch.as_tensor(weights, dtype=torch.float64, device=device)
    return {
        name: torch.einsum("n,nhw->hw", w, basis)
        for name, basis in bases_gpu.items()
    }


def _gap_tolerance(loss):
    return FW_GAP_ABS_TOLERANCE + FW_GAP_REL_TOLERANCE * max(1.0, abs(float(loss)))


def _armijo_line_search(weights, direction, loss, gap, bases_gpu, targets,
                         lp_margin, rho, deadline=math.inf):
    if not math.isfinite(gap) or gap <= 0.0:
        return {"accepted": False, "timed_out": False, "gamma": 0.0,
                "value": loss, "evaluations": 0}
    gamma, evaluations = 1.0, 0
    while gamma >= MIN_LINE_SEARCH_GAMMA:
        if time.monotonic() >= deadline:
            return {"accepted": False, "timed_out": True, "gamma": 0.0,
                    "value": loss, "evaluations": evaluations}
        evaluations += 1
        proposal = weights + gamma * direction
        trial_value = robust_corner_value(
            proposal, bases_gpu, targets, lp_margin, rho
        )
        if trial_value <= loss - ARMIJO_C1 * gamma * gap:
            return {"accepted": True, "timed_out": False, "gamma": gamma,
                    "value": trial_value, "evaluations": evaluations}
        gamma *= 0.5
    return {"accepted": False, "timed_out": False, "gamma": 0.0,
            "value": loss, "evaluations": evaluations}


def _fw_state_snapshot(seed, current, next_step, next_order, record, candidates, stage="iterations"):
    return {
        "schema_version": 1,
        "seed": int(seed),
        "stage": stage,
        "next_step": int(next_step),
        "next_checkpoint_order": int(next_order),
        "current_weights": np.asarray(current, dtype=np.float64).tolist(),
        "record": _finite_json(record),
        "candidates": [
            {**_candidate_summary(candidate),
             "weights": np.asarray(candidate["weights"], dtype=np.float64).tolist()}
            for candidate in candidates
        ],
    }


def _restore_fw_state(fw_state):
    record = copy.deepcopy(fw_state["record"])
    candidates = []
    for descriptor in fw_state["candidates"]:
        candidate = dict(descriptor)
        candidate["weights"] = np.asarray(candidate["weights"], dtype=np.float64)
        candidates.append(candidate)
    current = np.asarray(fw_state["current_weights"], dtype=np.float64)
    return record, candidates, current, int(fw_state["next_step"]), int(fw_state["next_checkpoint_order"])


def _fw_seed(seed, anchor, poly, fit_rows, basis32, basis_gpu, lp_margin,
             rho, protocol_sha256, out_dir, deadline, solver_time_limit, iterations,
             checkpoint_interval, state_callback, resume_state=None):
    targets = {row["layout_id"]: row["target"] for row in fit_rows}
    if resume_state is None:
        record = {
            "seed": int(seed), "status": "running", "iterations_completed": 0,
            "steps": [], "checkpoints": [],
        }
        current = np.asarray(anchor, dtype=np.float64).copy()
        candidates = [_candidate(
            current, 0, "LP anchor", poly, fit_rows, basis32, basis_gpu,
            lp_margin, rho, out_dir, seed, protocol_sha256,
        )]
        next_order, next_step = 1, 1
        state_callback(_fw_state_snapshot(
            seed, current, next_step, next_order, record, candidates, stage="seed_start"
        ), "seed_started", "training")
    else:
        record, candidates, current, next_step, next_order = _restore_fw_state(resume_state)
        record["status"] = "running"
        if resume_state.get("stage") == "seed_start":
            # Seed RNG is reset to its registered seed; the initial LMO is repeated.
            pass

    if resume_state is None or resume_state.get("stage") == "seed_start":
        rng = np.random.default_rng(int(seed))
        random_lmo = constrained.solve_lmo(
            poly, rng.normal(size=len(anchor)), deadline, solver_time_limit
        )
        if random_lmo["weights"] is None:
            timed_out = time.monotonic() >= deadline or random_lmo.get("status") == "deadline"
            state_callback(_fw_state_snapshot(
                seed, current, next_step, next_order, record, candidates, stage="seed_start"
            ), "initial_lmo_timeout" if timed_out else "initial_lmo_failure",
               "timeout_during_training" if timed_out else "training")
            if not timed_out:
                record["status"] = "solver_failure"
                record["failure_stage"] = "initial_random_lmo"
                record["checkpoints"] = [_candidate_summary(x) for x in candidates]
                record["random_vertex_solver"] = {
                    key: value for key, value in random_lmo.items() if key != "weights"
                }
                record["candidate_weights"] = [x["weights"].tolist() for x in candidates]
                selected = _select_fit_checkpoint(candidates)
                record["selected_candidate_label"] = selected["label"] if selected else None
                record["iterations_requested"] = int(iterations)
                return record, selected, None
            return {"seed": int(seed), "status": "timeout", "iterations_completed": 0,
                    "failure_stage": "initial_random_lmo"}, None, _fw_state_snapshot(
                        seed, current, next_step, next_order, record, candidates, stage="seed_start")
        record["random_vertex_solver"] = {
            key: value for key, value in random_lmo.items() if key != "weights"
        }
        trial_current = 0.95 * np.asarray(anchor, dtype=np.float64) + 0.05 * np.asarray(
            random_lmo["weights"], dtype=np.float64
        )
        if (not poly.verify(trial_current)["passed"]
                or not constrained.nominal_fit_check(fit_rows, basis32, trial_current)["passed"]):
            record["status"] = "verification_failed"
            record["failure_stage"] = "initial_mix"
        else:
            current = trial_current
            candidate = _candidate(
                current, next_order, "initial 5% feasible-vertex mix", poly,
                fit_rows, basis32, basis_gpu, lp_margin, rho, out_dir, seed,
                protocol_sha256,
            )
            candidates.append(candidate)
            record["checkpoints"] = [_candidate_summary(x) for x in candidates]
            next_order += 1
            next_step = 1
            state_callback(_fw_state_snapshot(
                seed, current, next_step, next_order, record, candidates
            ), "seed_initialized", "training")

    status = record.get("status", "running")
    for step in (range(next_step, iterations + 1) if status == "running" else ()):
        if time.monotonic() >= deadline:
            status = "timeout"
            break
        try:
            loss, gradient = robust_corner_value_gradient(
                current, basis_gpu, targets, lp_margin, rho
            )
        except Exception as exc:
            status = "numerical_error"
            record["failure_stage"] = "gradient"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            break
        lmo = constrained.solve_lmo(
            poly, gradient, deadline, solver_time_limit
        )
        if lmo["weights"] is None:
            timed_out = time.monotonic() >= deadline or lmo.get("status") == "deadline"
            if timed_out:
                status = "timeout"
                break
            status = "solver_failure"
            record["failure_stage"] = "linear_minimization_oracle"
            record["error"] = {"lmo_status": lmo.get("status"), "attempts": lmo.get("attempts")}
            break
        gap = float(gradient @ (current - lmo["weights"]))
        gap_tolerance = _gap_tolerance(loss)
        step_record = {
            "step": step, "loss": loss,
            "gradient_l2_norm": float(np.linalg.norm(gradient)),
            "fw_gap": gap, "fw_gap_tolerance": gap_tolerance,
            "fw_gap_interpretation": (
                "floating-point numerical stopping diagnostic for the continuous "
                "objective over the nominal fit polytope; not a rigorous certificate "
                "and not a hard-PV or calibration guarantee"
            ),
            "lmo_status": lmo["status"], "lmo_attempts": lmo["attempts"],
        }
        if not math.isfinite(gap) or gap < -1e-8:
            status = "numerical_error"
            record["failure_stage"] = "invalid_fw_gap"
            step_record["status"] = "invalid_gap"
            record["steps"].append(step_record)
            break
        if gap <= gap_tolerance:
            status = "complete_stationary"
            step_record["status"] = "stopping_tolerance"
            record["steps"].append(step_record)
            break
        try:
            line_search = _armijo_line_search(
                current, lmo["weights"] - current, loss, gap,
                basis_gpu, targets, lp_margin, rho, deadline,
            )
        except Exception as exc:
            status = "numerical_error"
            record["failure_stage"] = "line_search"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            break
        if line_search.get("timed_out"):
            status = "timeout"
            break
        step_record["line_search"] = line_search
        if not line_search["accepted"]:
            # A failed Armijo step above the registered gap criterion is a failure.
            status = "no_progress"
            record["failure_stage"] = "armijo_line_search"
            step_record["status"] = "no_progress_above_gap_tolerance"
            record["steps"].append(step_record)
            break
        proposal = current + line_search["gamma"] * (lmo["weights"] - current)
        poly_check = poly.verify(proposal)
        hard_check = constrained.nominal_fit_check(fit_rows, basis32, proposal)
        step_record["polytope_check"] = poly_check
        step_record["float32_nominal_fit_check"] = hard_check
        if not poly_check["passed"] or not hard_check["passed"]:
            status = "verification_failed"
            record["failure_stage"] = "accepted_step_verification"
            step_record["status"] = "verification_failed"
            record["steps"].append(step_record)
            break
        current = proposal
        step_record["status"] = "accepted"
        record["steps"].append(step_record)
        record["iterations_completed"] = step
        next_step = step + 1
        if step % checkpoint_interval == 0 or step == iterations:
            candidate = _candidate(
                current, next_order, "step%d" % step, poly, fit_rows,
                basis32, basis_gpu, lp_margin, rho, out_dir, seed, protocol_sha256,
            )
            candidates.append(candidate)
            record["checkpoints"] = [_candidate_summary(x) for x in candidates]
            next_order += 1
        state_callback(_fw_state_snapshot(
            seed, current, next_step, next_order, record, candidates
        ), "accepted_step", "training")
    if status == "timeout":
        record["status"] = "running"
        record["candidate_weights"] = [x["weights"].tolist() for x in candidates]
        selected = _select_fit_checkpoint(candidates)
        return {"seed": int(seed), "status": "timeout",
                "iterations_completed": int(record["iterations_completed"]),
                "failure_stage": "deadline"}, selected, _fw_state_snapshot(
                    seed, current, next_step, next_order, record, candidates,
                    stage="seed_start" if resume_state and resume_state.get("stage") == "seed_start"
                    and len(candidates) == 1 else "iterations")

    if status == "running":
        status = "complete"
    # Preserve the last accepted point for every non-timeout terminal outcome,
    # including a converged point and fail-closed numerical/solver exits.
    if not candidates or not np.array_equal(candidates[-1]["weights"], current):
        label = "terminal_step%d" % int(record["iterations_completed"])
        candidate = _candidate(
            current, next_order, label, poly, fit_rows, basis32, basis_gpu,
            lp_margin, rho, out_dir, seed, protocol_sha256,
        )
        candidates.append(candidate)
        next_order += 1
        record["checkpoints"] = [_candidate_summary(x) for x in candidates]
    record["status"] = status
    record["selected_training_candidate"] = None
    selected = _select_fit_checkpoint(candidates)
    record["selected_training_candidate"] = selected["label"] if selected else None
    record["selected_training_metrics"] = selected["fit_metrics"]["mean"] if selected else None
    record["selected_weights"] = selected["weights"].tolist() if selected else None
    record["selected_checkpoint_order"] = selected["checkpoint_order"] if selected else None
    record["candidate_weights"] = [x["weights"].tolist() for x in candidates]
    record["selection_rule"] = (
        "fit hard PV band, then fit worst-dose L2, common beta=800 score, "
        "earliest; zero nominal fit L2 and nominal LP region required"
    )
    record["fit_qualified"] = selected is not None
    record["iterations_requested"] = int(iterations)
    return record, selected, None


def _calibration_aggregate(records):
    return constrained._aggregate_cal(records)


def _no_blank(metrics):
    return constrained._no_blank_print(metrics)


def _git_provenance(candidate_plan):
    commands = {
        "head": ["git", "rev-parse", "HEAD"],
        "status": ["git", "status", "--porcelain", "--untracked-files=all"],
        "base_ancestor": ["git", "merge-base", "--is-ancestor",
                          str(candidate_plan.get("base_commit", "")), "HEAD"],
    }
    try:
        head = subprocess.run(commands["head"], cwd=ROOT, check=True, capture_output=True,
                              text=True, timeout=10).stdout.strip()
        status = subprocess.run(commands["status"], cwd=ROOT, check=True, capture_output=True,
                                text=True, timeout=10).stdout
        ancestor = subprocess.run(commands["base_ancestor"], cwd=ROOT, capture_output=True,
                                  text=True, timeout=10)
    except Exception as exc:
        raise RuntimeError("could not validate Git provenance: %s" % exc) from exc
    if status.strip():
        raise RuntimeError("real benchmark requires a clean committed source tree")
    if ancestor.returncode != 0:
        raise RuntimeError("candidate plan base_commit is not an ancestor of current HEAD")
    source_paths = (
        Path(__file__), ROOT / "source_robustness.py",
        ROOT / "scripts" / "optimize_source_constrained.py",
        ROOT / "scripts" / "diagnose_source_feasibility.py",
        ROOT / "scripts" / "run_protected_pvband_experiment.py",
        Path(constrained.light_source_module.__file__),
        Path(constrained.source_training_module.__file__),
    )
    source_hashes = {
        str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path)
        for path in source_paths
    }
    return {
        "head": head,
        "clean_tree": True,
        "candidate_plan_base_commit": candidate_plan.get("base_commit"),
        "source_sha256": source_hashes,
        "runtime": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
        },
    }


def _validate_prior_run(path, candidate_plan_sha256, shared_identity,
                        candidate_index, rho):
    if path is None:
        if candidate_index > 0:
            raise ValueError("--prior-run is required to advance the candidate plan")
        return None
    prior_dir = Path(path).resolve()
    if not prior_dir.is_absolute() or not prior_dir.is_dir():
        raise ValueError("--prior-run must name an existing run directory")
    try:
        protocol = json.loads((prior_dir / "protocol.json").read_text(encoding="utf-8"))
        state = json.loads((prior_dir / "run_state.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("prior run lacks readable protocol/canonical state") from exc
    _verify_run_state_checksum(state)
    protocol_sha256 = sha256_file(prior_dir / "protocol.json")
    prior_identity = protocol.get("identity", {})
    if state.get("results", {}).get("protocol_sha256") != protocol_sha256:
        raise ValueError("prior canonical results do not match protocol.json")
    if protocol.get("identity_sha256") != _canonical_hash(prior_identity):
        raise ValueError("prior protocol identity hash is invalid")
    if (state.get("identity") != prior_identity
            or state.get("identity_sha256") != _canonical_hash(prior_identity)):
        raise ValueError("prior protocol and canonical state identities differ")
    if protocol.get("candidate_plan", {}).get("sha256") != candidate_plan_sha256:
        raise ValueError("prior run used a different candidate plan")
    if prior_identity.get("shared") != shared_identity:
        raise ValueError("prior run source, data, or runtime identity differs")
    prior_index = protocol.get("candidate", {}).get("index")
    prior_rho = protocol.get("candidate", {}).get("rho")
    prior_results = state.get("results", {})
    prior_status = prior_results.get("status")
    if prior_index == candidate_index - 1:
        if prior_status != "complete" or state.get("phase") != "complete":
            raise ValueError("advancing rho requires a completed previous candidate")
        prior_gate = prior_results.get("calibration", {}).get("gate", {})
        prior_seeds = prior_results.get("seeds", [])
        if prior_gate.get("passed") is not False:
            raise ValueError("advancing rho requires the previous frozen gate to fail")
        if len(prior_seeds) != len(SEEDS) or not all(
            row.get("complete") and row.get("fit_qualified")
            for row in prior_seeds
        ):
            raise ValueError("advancing rho requires all five prior seeds complete and fit-qualified")
    elif prior_index == candidate_index and float(prior_rho) == float(rho):
        if prior_status in {"timeout_during_training", "timeout_during_calibration", "interrupted",
                            "training", "preregistered", "calibration_gate_running"}:
            raise ValueError("resume the active or timed-out prior run instead of duplicating it")
        if prior_status not in {"failed", "training_failed"} or state.get("phase") != "fit_training":
            raise ValueError("same-rho retry requires a terminal pre-calibration training failure")
        calibration = prior_results.get("calibration", {})
        if calibration.get("per_seed_partial") or calibration.get("per_seed"):
            raise ValueError("same-rho retry is forbidden after calibration scoring began")
        if prior_status == "training_failed":
            last = prior_results.get("seeds", [])[-1:] or [{}]
            last = last[0]
            retryable_seed_statuses = {
                "solver_failure", "numerical_error", "no_progress",
                "verification_failed", "initial_mix_failed_verification",
            }
            if (last.get("status") not in retryable_seed_statuses
                    and last.get("fit_qualified") is not False):
                raise ValueError("training_failed prior run has no terminal solver/numerical failure")
        else:
            failure = prior_results.get("failure", {})
            if not failure.get("retryable_training_failure"):
                raise ValueError("failed prior run is not a classified retryable training failure")
    else:
        raise ValueError("prior run is neither the completed previous rho nor a failed same-rho attempt")
    return {
        "path": str(prior_dir),
        "candidate_index": int(prior_index),
        "rho": float(prior_rho),
        "attempt_number": int(protocol.get("candidate", {}).get("attempt_number", 1)),
        "status": prior_status,
        "run_state_sha256": sha256_file(prior_dir / "run_state.json"),
        "protocol_sha256": protocol_sha256,
    }


def _make_identity(diag, dataset_path, diagnostic_path, provenance, device, gpu,
                   candidate_plan_path, candidate_plan_sha256, candidate_index,
                   rho, iterations, checkpoint_interval, solver_time_limit,
                   basis_parity, prior_run):
    shared = {
        "inputs": {
            "dataset_sha256": diag["input"]["dataset_sha256"],
            "dataset_file_sha256": sha256_file(dataset_path),
            "diagnostic_file_sha256": sha256_file(diagnostic_path),
            "diagnostic_input": diag["input"],
            "basis_parity": basis_parity,
        },
        "source_provenance": provenance,
        "device": {"device": str(device), "gpu_name": gpu},
        "candidate_plan": {
            "path": str(Path(candidate_plan_path).resolve()),
            "sha256": candidate_plan_sha256,
        },
        "fixed_protocol": {
            "objective_id": OBJECTIVE_ID,
            "seeds": list(SEEDS),
            "doses": [LOW_DOSE, NOMINAL_DOSE, HIGH_DOSE],
            "calibration_gate": {
                "band_pixels": constrained.GATE["band_pixels"],
                "nominal_l2_pixels": constrained.GATE["L2_pixels"],
                "worst_dose_l2_pixels": constrained.GATE["worst_L2_pixels"],
                "each_seed_band_below": constrained.LP_CAL["band_pixels"],
                "no_blank": True,
            },
        },
    }
    previous_attempt = (prior_run.get("attempt_number", 0)
                        if prior_run and prior_run["candidate_index"] == candidate_index else 0)
    attempt_number = int(previous_attempt + 1)
    candidate = {
        "index": int(candidate_index), "rho": float(rho),
        "attempt_number": attempt_number,
        "iterations_per_seed": int(iterations),
        "checkpoint_interval": int(checkpoint_interval),
        "solver_time_limit_seconds": float(solver_time_limit),
        "armijo_c1": ARMIJO_C1,
        "minimum_line_search_gamma": MIN_LINE_SEARCH_GAMMA,
        "fw_gap_absolute_tolerance": FW_GAP_ABS_TOLERANCE,
        "fw_gap_relative_tolerance": FW_GAP_REL_TOLERANCE,
    }
    return {
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "shared": shared,
        "candidate": candidate,
        "prior_run": prior_run,
        "attempt_number": attempt_number,
    }


def _resolve_options(args, candidate_plan, saved_protocol=None):
    rhos = _validate_target_plan(candidate_plan)
    if saved_protocol is not None:
        saved_candidate = saved_protocol.get("candidate", {})
        saved_solver = saved_protocol.get("protocol", {})
        expected = {
            "candidate_index": int(saved_candidate.get("index", -1)),
            "rho": float(saved_candidate.get("rho", math.nan)),
            "iterations": int(saved_solver.get("max_iterations_per_seed", -1)),
            "checkpoint_interval": int(saved_solver.get("checkpoint_interval", -1)),
            "solver_time_limit": float(saved_solver.get("solver_time_limit_seconds", math.nan)),
        }
        for name in ("candidate_index", "rho", "iterations", "checkpoint_interval", "solver_time_limit"):
            supplied = getattr(args, name)
            if supplied is not None and supplied != expected[name]:
                raise ValueError("resume %s differs from the frozen protocol" % name.replace("_", "-"))
        args.candidate_index = expected["candidate_index"]
        args.rho = validate_rho(expected["rho"])
        args.iterations = expected["iterations"]
        args.checkpoint_interval = expected["checkpoint_interval"]
        args.solver_time_limit = expected["solver_time_limit"]
        if (args.checkpoint_interval != DEFAULT_CHECKPOINT_INTERVAL
                or args.solver_time_limit != 60.0):
            raise ValueError("saved checkpoint interval or solver time limit differs from frozen defaults")
        return _candidate_plan_selection(candidate_plan, args.candidate_index,
                                         args.rho, args.iterations)
    args.candidate_index = 0 if args.candidate_index is None else args.candidate_index
    if args.candidate_index < 0 or args.candidate_index >= len(rhos):
        raise ValueError("candidate index is outside the registered rho sequence")
    args.rho = rhos[args.candidate_index] if args.rho is None else validate_rho(args.rho)
    args.iterations = (int(candidate_plan["initial_iterations_per_seed"])
                       if args.iterations is None else args.iterations)
    args.checkpoint_interval = (DEFAULT_CHECKPOINT_INTERVAL
                                if args.checkpoint_interval is None else args.checkpoint_interval)
    args.solver_time_limit = 60.0 if args.solver_time_limit is None else args.solver_time_limit
    selection = _candidate_plan_selection(
        candidate_plan, args.candidate_index, args.rho, args.iterations
    )
    if (args.checkpoint_interval != DEFAULT_CHECKPOINT_INTERVAL
            or args.solver_time_limit != 60.0):
        raise ValueError("checkpoint interval and solver time limit are frozen at 25 and 60 seconds")
    return selection


def _build_protocol(args, diag, dataset_path, diagnostic_path, m_lp, floor,
                    basis_minima, device, gpu, candidate_plan, candidate_plan_path,
                    identity, prior_run):
    return {
        "schema_version": 2,
        "status": "preregistered",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective_id": OBJECTIVE_ID,
        "identity": identity,
        "identity_sha256": _canonical_hash(identity),
        "candidate_plan": {
            "path": str(Path(candidate_plan_path).resolve()),
            "sha256": sha256_file(candidate_plan_path),
            "schema_version": candidate_plan["schema_version"],
        },
        "candidate": {
            "index": int(args.candidate_index), "rho": float(args.rho),
            "attempt_number": int(identity["attempt_number"]),
            "prior_run": prior_run,
        },
        "git_head": identity["shared"]["source_provenance"]["head"],
        "source_sha256": identity["shared"]["source_provenance"]["source_sha256"],
        "dataset_sha256": diag["input"]["dataset_sha256"],
        "dataset_file_sha256": sha256_file(dataset_path),
        "diagnostic_file_sha256": sha256_file(diagnostic_path),
        "fit_ids": [row["layout_id"] for row in diag["input"]["fit_masks"]],
        "calibration_ids": [row["layout_id"] for row in diag["input"]["calibration_masks"]],
        "final3_status": "closed; not indexed or evaluated",
        "heldout_status": "not indexed or evaluated",
        "device": str(device), "gpu_name": gpu,
        "runtime": identity["shared"]["source_provenance"]["runtime"],
        "protocol": {
            "scope": "source-only; 49 active source weights; fixed fit masks and targets",
            "process_doses": [LOW_DOSE, NOMINAL_DOSE, HIGH_DOSE],
            "objective": (
                "mean over fit pixels/layouts of "
                "relu(rho*mLP - min_d y_i*(d*I_i(w)-T))^2 / mLP^2, "
                "d in {0.98,1.02}; worst dose is 0.98 for target-positive pixels "
                "and 1.02 for target-negative pixels on the nonnegative-intensity region"
            ),
            "threshold": THRESHOLD, "rho": float(args.rho),
            "candidate_index": int(args.candidate_index),
            "lp_margin": float(m_lp), "required_nominal_margin_floor": float(floor),
            "hinge_normalization": "mLP^2; positive constant, preserves minimizers",
            "basis_nonnegative_roundoff_tolerance": BASIS_NONNEGATIVE_TOL,
            "basis_minima": basis_minima,
            "optimizer": "float64 Frank-Wolfe with existing HiGHS float64 LMO and Armijo halving",
            "fw_gap_stopping": {
                "absolute_tolerance": FW_GAP_ABS_TOLERANCE,
                "relative_tolerance": FW_GAP_REL_TOLERANCE,
                "formula": "gap <= abs_tol + rel_tol * max(1, abs(objective_value))",
                "interpretation": "floating-point numerical stopping rule, not a rigorous global certificate or quality guarantee",
            },
            "fit_selection": (
                "minimum fit hard PV-band, then fit worst-dose L2, beta=800 smooth PV score, "
                "then earliest checkpoint; require nominal fit L2=0 and polytope feasibility"
            ),
            "seeds": list(SEEDS), "max_iterations_per_seed": int(args.iterations),
            "checkpoint_interval": int(args.checkpoint_interval),
            "line_search": {"armijo_c1": ARMIJO_C1,
                            "minimum_gamma": MIN_LINE_SEARCH_GAMMA},
            "solver_time_limit_seconds": float(args.solver_time_limit),
        },
        "constraints": {
            "nominal_lp_region": (
                "w>=0; sum(w)=1; y_i*(I_i(w)-T)>=rho*mLP at every pixel in the four fit layouts"
            ),
            "float32_nominal_fit_l2": 0,
            "rho_is_configurable_within_candidate_plan": True,
        },
        "calibration_gate": {
            "frozen_before_scoring": True,
            "reference_lp_anchor_means": dict(constrained.LP_CAL),
            "mean_band_pixels_max": constrained.GATE["band_pixels"],
            "mean_nominal_l2_pixels_max": constrained.GATE["L2_pixels"],
            "mean_worst_dose_l2_pixels_max": constrained.GATE["worst_L2_pixels"],
            "each_seed_band_pixels_strictly_less_than": constrained.LP_CAL["band_pixels"],
            "no_blank_positive_target_any_dose": True,
            "all_five_seeds_must_finish_and_fit_qualify": True,
            "calibration_gradients": False,
            "role": "development-only; reused synthetic calibration, not independent generalization",
        },
        "invocation_timeout_seconds": float(args.timeout_seconds),
    }


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run_identity(args, diag, dataset_path, diagnostic_path, provenance,
                  device, gpu, plan_path, plan_sha256, basis_parity, prior_run):
    return _make_identity(
        diag, dataset_path, diagnostic_path, provenance, device, gpu,
        plan_path, plan_sha256, args.candidate_index, args.rho,
        args.iterations, args.checkpoint_interval, args.solver_time_limit,
        basis_parity, prior_run,
    )


def _validate_saved_candidate(candidate, seed, order, anchor, poly,
                              fit_rows, basis32):
    weights = np.asarray(candidate.get("weights"), dtype=np.float64)
    if weights.shape != np.asarray(anchor).shape or not np.isfinite(weights).all():
        raise ValueError("saved checkpoint has invalid source weights")
    if int(candidate.get("checkpoint_order", -1)) != order:
        raise ValueError("saved checkpoint order is not contiguous")
    if not poly.verify(weights).get("passed"):
        raise ValueError("saved checkpoint violates the registered nominal polytope")
    if not constrained.nominal_fit_check(fit_rows, basis32, weights).get("passed"):
        raise ValueError("saved checkpoint fails the nominal float32 fit check")
    if not candidate.get("fit_qualified"):
        raise ValueError("saved checkpoint is not fit-qualified")
    if not candidate.get("label") or not candidate.get("weights_file"):
        raise ValueError("saved checkpoint metadata is incomplete")
    return weights


def _validate_completed_seeds(records, anchor, poly, fit_rows, basis32):
    if not isinstance(records, list) or len(records) > len(SEEDS):
        raise ValueError("completed-seed list exceeds registered seeds")
    if [row.get("seed") for row in records] != list(SEEDS[:len(records)]):
        raise ValueError("completed seeds are not a registered prefix")
    for record in records:
        if record.get("status") not in {"complete", "complete_stationary"}:
            raise ValueError("saved seed record is not terminal-successful")
        if record.get("complete") is not True or record.get("fit_qualified") is not True:
            raise ValueError("saved seed record is not fit-qualified")
        weights_rows = record.get("candidate_weights")
        summaries = record.get("checkpoints")
        if (not isinstance(weights_rows, list) or not isinstance(summaries, list)
                or len(weights_rows) != len(summaries) or not summaries):
            raise ValueError("saved seed is missing candidate checkpoint state")
        candidates = []
        for order, (weights, summary) in enumerate(zip(weights_rows, summaries)):
            candidate = dict(summary)
            candidate["weights"] = weights
            _validate_saved_candidate(
                candidate, int(record["seed"]), order, anchor, poly,
                fit_rows, basis32,
            )
            candidates.append(candidate)
        selected = _select_fit_checkpoint(candidates)
        if selected is None or selected["label"] != record.get("selected_training_candidate"):
            raise ValueError("saved seed selection does not match its fit checkpoints")
        if not np.array_equal(
            np.asarray(record.get("selected_weights"), dtype=np.float64),
            np.asarray(selected["weights"], dtype=np.float64),
        ):
            raise ValueError("saved selected weights differ from the fit-only selector")


def _all_seeds_fit_qualified(results):
    records = results.get("seeds", [])
    return (
        len(records) == len(SEEDS)
        and [row.get("seed") for row in records] == list(SEEDS)
        and all(row.get("complete") is True and row.get("fit_qualified") is True
                for row in records)
    )


def _validate_fw_state(fw_state, seed, anchor, poly, fit_rows, basis32, iterations):
    if not isinstance(fw_state, dict) or fw_state.get("schema_version") != 1:
        raise ValueError("invalid Frank-Wolfe resume cursor schema")
    if int(fw_state.get("seed", -1)) != int(seed):
        raise ValueError("Frank-Wolfe cursor seed mismatch")
    if fw_state.get("stage") not in {"seed_start", "iterations"}:
        raise ValueError("Frank-Wolfe cursor has an invalid stage")
    current = np.asarray(fw_state.get("current_weights"), dtype=np.float64)
    if current.shape != np.asarray(anchor).shape or not np.isfinite(current).all():
        raise ValueError("Frank-Wolfe cursor has invalid current weights")
    if not poly.verify(current).get("passed"):
        raise ValueError("Frank-Wolfe cursor current point violates the registered polytope")
    if not constrained.nominal_fit_check(fit_rows, basis32, current).get("passed"):
        raise ValueError("Frank-Wolfe cursor current point fails the nominal fit check")
    next_step = fw_state.get("next_step")
    if not isinstance(next_step, int) or not 1 <= next_step <= iterations + 1:
        raise ValueError("Frank-Wolfe cursor next step is outside the frozen budget")
    record = fw_state.get("record")
    candidates = fw_state.get("candidates")
    if not isinstance(record, dict) or not isinstance(candidates, list) or not candidates:
        raise ValueError("Frank-Wolfe cursor lacks its record or anchor candidate")
    if record.get("seed") != int(seed) or record.get("status") not in {"running", "verification_failed"}:
        raise ValueError("Frank-Wolfe cursor record does not match the active seed")
    if fw_state.get("stage") == "seed_start" and (
        next_step != 1 or len(candidates) != 1 or candidates[0].get("label") != "LP anchor"
    ):
        raise ValueError("seed-start cursor must contain only the LP anchor")
    for order, candidate in enumerate(candidates):
        _validate_saved_candidate(
            candidate, int(seed), order, anchor, poly, fit_rows, basis32,
        )
    if int(fw_state.get("next_checkpoint_order", -1)) != len(candidates):
        raise ValueError("Frank-Wolfe cursor checkpoint order is inconsistent")
    completed = int(record.get("iterations_completed", -1))
    if completed < 0 or completed >= next_step or completed > iterations:
        raise ValueError("Frank-Wolfe cursor completed-step count is inconsistent")


def _repair_candidate_artifacts(out_dir, state, poly, fit_rows, basis32,
                                rho, lp_margin, protocol_sha256):
    descriptors = []
    for record in state["results"].get("seeds", []):
        weights_rows, summaries = record.get("candidate_weights", []), record.get("checkpoints", [])
        if len(weights_rows) != len(summaries):
            raise ValueError("saved completed-seed weights and checkpoint summaries differ")
        for weights, summary in zip(
            weights_rows, summaries
        ):
            candidate = dict(summary)
            candidate["weights"] = np.asarray(weights, dtype=np.float64)
            descriptors.append((int(record["seed"]), candidate, summary))
    active = state.get("fw_state")
    if active is not None:
        for candidate in active["candidates"]:
            descriptors.append((int(active["seed"]), candidate, candidate))
    for seed, candidate, _ in descriptors:
        weights = np.asarray(candidate["weights"], dtype=np.float64)
        if (not poly.verify(weights).get("passed")
                or not constrained.nominal_fit_check(fit_rows, basis32, weights).get("passed")):
            raise ValueError("resume checkpoint failed independent fit/polytope validation")
        expected_path = _expected_checkpoint_path(
            out_dir, seed, int(candidate["checkpoint_order"])
        )
        if str(Path(candidate["weights_file"]).resolve()) != str(expected_path.resolve()):
            raise ValueError("checkpoint path escapes or disagrees with the run directory")
    for seed, candidate, summary in descriptors:
        repaired = _verify_or_repair_checkpoint(
            out_dir, candidate, seed, rho, lp_margin, protocol_sha256,
        )
        summary["weights_file"] = repaired["weights_file"]
        summary["weights_sha256"] = repaired["weights_sha256"]


def _validate_checkpoint_artifacts_readonly(out_dir, state, rho, lp_margin,
                                            protocol_sha256):
    descriptors = []
    for record in state["results"].get("seeds", []):
        weights_rows, summaries = record.get("candidate_weights", []), record.get("checkpoints", [])
        if len(weights_rows) != len(summaries):
            raise ValueError("completed run has incomplete checkpoint metadata")
        for weights, summary in zip(weights_rows, summaries):
            candidate = dict(summary)
            candidate["weights"] = np.asarray(weights, dtype=np.float64)
            descriptors.append((int(record["seed"]), candidate))
    for seed, candidate in descriptors:
        expected = _expected_checkpoint_path(
            out_dir, seed, int(candidate["checkpoint_order"])
        )
        if str(Path(candidate["weights_file"]).resolve()) != str(expected.resolve()):
            raise ValueError("checkpoint path escapes or disagrees with the completed run")
        if not expected.is_file() or sha256_file(expected) != candidate.get("weights_sha256"):
            raise ValueError("completed checkpoint is missing or its hash differs")
        saved = torch.load(expected, map_location="cpu", weights_only=True)
        expected_metadata = {
            "objective_id": OBJECTIVE_ID, "seed": seed,
            "checkpoint_order": int(candidate["checkpoint_order"]),
            "label": candidate["label"], "rho": float(rho),
            "lp_margin": float(lp_margin),
            "protocol_sha256": protocol_sha256, "fit_only": True,
        }
        saved_weights = saved["weights_supported_float64"].detach().cpu().numpy()
        if (not np.array_equal(saved_weights, candidate["weights"])
                or saved.get("metadata") != expected_metadata):
            raise ValueError("completed checkpoint content or metadata differs")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--diagnostic-file", required=True)
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--prior-run")
    parser.add_argument("--output-root")
    parser.add_argument("--resume-run")
    parser.add_argument("--rho", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--solver-time-limit", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-gpu", default="NVIDIA GeForce RTX 5090")
    args = parser.parse_args(argv)
    if bool(args.resume_run) == bool(args.output_root):
        parser.error("specify exactly one of --output-root or --resume-run")
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.checkpoint_interval is not None and args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be positive")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    if args.solver_time_limit is not None and (
        not math.isfinite(args.solver_time_limit) or args.solver_time_limit <= 0
    ):
        parser.error("--solver-time-limit must be finite and positive")
    if args.output_root and not os.path.isabs(args.output_root):
        parser.error("--output-root must be absolute")
    if args.resume_run and not os.path.isabs(args.resume_run):
        parser.error("--resume-run must be absolute")
    return args


def run(args):
    args._run_lock = None
    args._run_state = None
    args._last_canonical_state = None
    args._active_output = None
    out_dir = None
    try:
        plan_path = Path(args.candidate_plan).resolve()
        candidate_plan = _read_json(plan_path)
        _validate_target_plan(candidate_plan)
        plan_sha256 = sha256_file(plan_path)
        dataset_path = Path(args.dataset_file).resolve()
        diagnostic_path = Path(args.diagnostic_file).resolve()
        if Path(candidate_plan.get("dataset", "")).resolve() != dataset_path:
            raise ValueError("dataset path differs from the registered candidate plan")
        if Path(candidate_plan.get("diagnostic", "")).resolve() != diagnostic_path:
            raise ValueError("diagnostic path differs from the registered candidate plan")
        saved_protocol = None
        if args.resume_run:
            out_dir = Path(args.resume_run).resolve()
            if not out_dir.is_dir():
                raise ValueError("--resume-run must name an existing run directory")
            args._run_lock = constrained.RunLock(out_dir).acquire()
            saved_protocol = _read_json(out_dir / "protocol.json")
            if str(plan_path) != saved_protocol.get("candidate_plan", {}).get("path"):
                raise ValueError("resume candidate-plan path differs from frozen protocol")
            if plan_sha256 != saved_protocol.get("candidate_plan", {}).get("sha256"):
                raise ValueError("candidate plan changed since run registration")
        _resolve_options(args, candidate_plan, saved_protocol)
        if saved_protocol is not None:
            prior_info = saved_protocol.get("candidate", {}).get("prior_run")
            frozen_prior = prior_info.get("path") if prior_info else None
            if args.prior_run and str(Path(args.prior_run).resolve()) != frozen_prior:
                raise ValueError("resume prior-run path differs from frozen lineage")
            args.prior_run = frozen_prior
        if args.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("the registered quality run requires CUDA")
        device = torch.device(args.device)
        gpu = torch.cuda.get_device_name(device)
        if gpu != args.expected_gpu:
            raise RuntimeError("expected %s; found %s" % (args.expected_gpu, gpu))
        torch.cuda.synchronize(device)

        diag, fit, cal, fit_rows, cal_rows = constrained.load_inputs(
            dataset_path, diagnostic_path, device
        )
        if constrained.diagnostic.sha256_file(dataset_path) != diag["input"]["dataset_sha256"]:
            raise ValueError("dataset SHA256 differs from registered diagnostic")
        expected_basis = {x["layout_id"]: x for x in diag["input"]["bases"]}
        basis32, _basis64_cpu, basis_gpu, parity = constrained.prepare_bases(
            fit_rows, cal_rows, device, expected_basis
        )
        basis_minima = validate_nonnegative_bases(basis_gpu, BASIS_NONNEGATIVE_TOL)
        support = constrained.support_mask()
        nominal_lp = diag["scenarios"]["fit_only.nominal"]
        if nominal_lp.get("status") != "positive_margin_feasible":
            raise ValueError("fit-only nominal LP is not positive-margin feasible")
        anchor = constrained.compress(nominal_lp["source_weights_full_grid"], support)
        lp_margin = float(nominal_lp["lp_optimal_margin"])
        if not math.isfinite(lp_margin) or lp_margin <= 0:
            raise ValueError("registered nominal LP margin must be finite and positive")
        targets = [row["target"].numpy() for row in fit_rows]
        arrays64 = [basis_gpu[row["layout_id"]].detach().cpu().numpy() for row in fit_rows]
        matrix64, labels64 = constrained.fit_matrix(arrays64, targets)
        reported_margin = constrained.signed_margin(matrix64, labels64, anchor)
        if abs(reported_margin - lp_margin) > constrained.TOL:
            raise ValueError("LP anchor failed independent float64 margin verification")
        poly = constrained.build_polytope(arrays64, targets, lp_margin, rho=args.rho)
        anchor_check = poly.verify(anchor)
        if not anchor_check["passed"]:
            raise ValueError("LP anchor violates the registered nominal region")
        if not constrained.nominal_fit_check(fit_rows, basis32, anchor)["passed"]:
            raise ValueError("LP anchor fails float32 nominal hard-fit check")
        if len(fit_rows) != 4 or len(cal_rows) != 4:
            raise ValueError("registered experiment requires four fit and four calibration layouts")
        if fit.pixel_size_nm != 4.0 or tuple(fit.masks.shape[-2:]) != (128, 128):
            raise ValueError("registered experiment requires fixed 128x128, 4 nm masks")

        provenance = _git_provenance(candidate_plan)
        provisional_identity = _run_identity(
            args, diag, dataset_path, diagnostic_path, provenance,
            device, gpu, plan_path, plan_sha256, parity, None,
        )
        shared_identity = provisional_identity["shared"]
        prior_run = _validate_prior_run(
            args.prior_run, plan_sha256, shared_identity,
            args.candidate_index, args.rho,
        )
        identity = _run_identity(
            args, diag, dataset_path, diagnostic_path, provenance,
            device, gpu, plan_path, plan_sha256, parity, prior_run,
        )

        if saved_protocol is not None:
            protocol = saved_protocol
            if protocol.get("identity") != identity or protocol.get(
                "identity_sha256"
            ) != _canonical_hash(identity):
                raise ValueError("resume identity mismatch (source, runtime, data, plan, or config changed)")
            protocol_sha256 = sha256_file(out_dir / "protocol.json")
            state = _load_run_state(out_dir, identity)
            results = state["results"]
            if results.get("protocol_sha256") != protocol_sha256:
                raise ValueError("canonical state points to a different protocol file")
            if (results.get("final3_status") != "closed; not indexed or evaluated"
                    or results.get("heldout_status") != "not indexed or evaluated"):
                raise ValueError("resume state violates the closed final3 protocol")
            _validate_completed_seeds(results.get("seeds", []), anchor, poly,
                                      fit_rows, basis32)
            current_seed, fw_state = state.get("current_seed"), state.get("fw_state")
            if current_seed is None and fw_state is not None:
                raise ValueError("Frank-Wolfe cursor exists without an active seed")
            if current_seed is not None:
                next_seeds = list(SEEDS[len(results.get("seeds", [])):])
                if not next_seeds or current_seed != next_seeds[0]:
                    raise ValueError("active seed is not the next registered seed")
                if fw_state is None:
                    progress = state.get("progress", {})
                    if not (
                        progress.get("event") == "seed_started"
                        or (progress.get("event") == "keyboard_interrupt"
                            and progress.get("interrupted_from_event") == "seed_started")
                    ):
                        raise ValueError("active seed without a cursor is not a committed seed-start state")
                else:
                    _validate_fw_state(fw_state, current_seed, anchor, poly,
                                       fit_rows, basis32, args.iterations)
            if state["phase"] == "calibration_gate":
                if current_seed is not None or not _all_seeds_fit_qualified(results):
                    raise ValueError("calibration phase requires all five fit selections frozen")
            if state["phase"] == "complete" and (
                not _all_seeds_fit_qualified(results)
                or results.get("calibration", {}).get("gate") is None
            ):
                raise ValueError("complete run lacks five fit selections or a gate")
            if state["phase"] == "complete":
                _validate_checkpoint_artifacts_readonly(
                    out_dir, state, args.rho, lp_margin, protocol_sha256,
                )
                args._run_state = state
                args._last_canonical_state = copy.deepcopy(state)
                args._active_output = out_dir
                return out_dir
            _repair_candidate_artifacts(
                out_dir, state, poly, fit_rows, basis32, args.rho,
                lp_margin, protocol_sha256,
            )
            _recover_sidecars(out_dir, state)
            _write_canonical_state(out_dir, state)
        else:
            out_root = _require_output_outside_source_tree(args.output_root)
            out_root.mkdir(parents=True, exist_ok=True)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
            out_dir = out_root / run_id
            out_dir.mkdir(parents=False, exist_ok=False)
            args._run_lock = constrained.RunLock(out_dir).acquire()
            (out_dir / "weights").mkdir()
            floor = float(args.rho * lp_margin)
            protocol = _build_protocol(
                args, diag, dataset_path, diagnostic_path, lp_margin, floor,
                basis_minima, device, gpu, candidate_plan, plan_path,
                identity, prior_run,
            )
            constrained.atomic_json(out_dir / "protocol.json", protocol)
            protocol_sha256 = sha256_file(out_dir / "protocol.json")
            results = {
                "schema_version": 2, "status": "preregistered",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "result_directory": str(out_dir), "objective_id": OBJECTIVE_ID,
                "final3_status": "closed; not indexed or evaluated",
                "heldout_status": "not indexed or evaluated",
                "protocol": protocol, "protocol_sha256": protocol_sha256,
                "basis_hashes_verified": True, "basis_parity": parity,
                "anchor_polytope_verification": anchor_check,
                "rho": float(args.rho), "candidate_index": args.candidate_index,
                "lp_margin": lp_margin, "required_nominal_margin_floor": floor,
                "gradient_diagnostics_at_anchor": {"status": "pending"},
                "seeds": [],
                "calibration": {"status": "closed_until_all_seeds_fit_selected"},
                "selection": {"status": "closed_until_calibration_gate"},
                "runtime": {"cumulative_seconds": 0.0, "invocation_count": 0},
            }
            state = {
                "schema_version": RUN_STATE_SCHEMA_VERSION,
                "identity": identity, "identity_sha256": _canonical_hash(identity),
                "phase": "fit_training", "current_seed": None, "fw_state": None,
                "results": results,
                "progress": {
                    "status": "preregistered", "phase": "fit_training",
                    "event": "protocol_frozen_before_gradient_diagnostics",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
            }
            args._run_state = state
            args._active_output = out_dir
            _write_canonical_state(out_dir, state)

        args._run_state = state
        args._active_output = out_dir
        args._last_canonical_state = copy.deepcopy(state)
        if state["phase"] == "complete":
            return out_dir
        invocation_started = time.monotonic()
        runtime = state["results"].setdefault("runtime", {})
        base_seconds = float(runtime.get("cumulative_seconds", 0.0))
        base_invocations = int(runtime.get("invocation_count", 0))
        deadline = invocation_started + float(args.timeout_seconds)

        def commit(event, status=None, phase=None, **fields):
            if status is not None:
                state["results"]["status"] = status
            if phase is not None:
                state["phase"] = phase
            state["progress"].update(fields)
            state["progress"].update({
                "status": state["results"].get("status"), "phase": state["phase"],
                "event": event, "updated_utc": datetime.now(timezone.utc).isoformat(),
            })
            _mark_runtime(state, invocation_started, base_seconds,
                          base_invocations, args.timeout_seconds)
            _write_canonical_state(out_dir, state)
            args._last_canonical_state = copy.deepcopy(state)

        if state["phase"] == "fit_training":
            if state["results"].get("gradient_diagnostics_at_anchor", {}).get("status") == "pending":
                state["results"]["gradient_diagnostics_at_anchor"] = _metric_bundle(
                    fit_rows, basis32, basis_gpu, anchor, lp_margin, args.rho
                )
                commit("anchor_fit_diagnostics_complete", status="training")
            results = state["results"]
            while len(results.get("seeds", [])) < len(SEEDS):
                if time.monotonic() >= deadline and state.get("current_seed") is None:
                    commit("timeout_before_next_seed", status="timeout_during_training")
                    return out_dir
                if state.get("current_seed") is None:
                    seed = SEEDS[len(results["seeds"])]
                    state["current_seed"] = int(seed)
                    state["fw_state"] = None
                seed = int(state["current_seed"])

                def save_fw_cursor(snapshot, event, run_status):
                    state["current_seed"] = seed
                    state["fw_state"] = snapshot
                    commit(event, status=run_status, phase="fit_training",
                           seed=seed, iterations_completed=int(
                               snapshot.get("record", {}).get("iterations_completed", 0)
                           ), next_step=snapshot.get("next_step"))

                record, selected, cursor = _fw_seed(
                    seed, anchor, poly, fit_rows, basis32, basis_gpu, lp_margin,
                    args.rho, protocol_sha256, out_dir, deadline,
                    args.solver_time_limit, args.iterations,
                    args.checkpoint_interval, save_fw_cursor,
                    resume_state=state.get("fw_state"),
                )
                if record.get("status") == "timeout":
                    if cursor is not None:
                        state["fw_state"] = cursor
                    commit("timeout_during_seed", status="timeout_during_training",
                           seed=seed,
                           iterations_completed=int(record.get("iterations_completed", 0)))
                    return out_dir
                record["complete"] = record.get("status") in {"complete", "complete_stationary"}
                record["fit_qualified"] = bool(selected is not None and selected.get("fit_qualified"))
                if selected is not None:
                    record["selected_training_candidate"] = selected["label"]
                    record["selected_weights"] = selected["weights"].tolist()
                    record["selected_checkpoint_order"] = selected["checkpoint_order"]
                results["seeds"].append(record)
                state["current_seed"], state["fw_state"] = None, None
                if not record["complete"] or not record["fit_qualified"]:
                    results["calibration"] = {
                        "status": "not_evaluated_incomplete_or_unqualified_seed_set"
                    }
                    results["selection"] = {
                        "status": "calibration_closed_training_failure",
                        "qualified_arms": [], "selected_arm": None,
                    }
                    commit("fit_seed_failed_calibration_closed", status="training_failed",
                           phase="fit_training", seed=seed, seed_status=record["status"])
                    return out_dir
                commit("fit_seed_complete", status="training", seed=seed,
                       selected_candidate=record["selected_training_candidate"])
            if not _all_seeds_fit_qualified(results):
                results["calibration"] = {
                    "status": "not_evaluated_incomplete_or_unqualified_seed_set"
                }
                commit("calibration_closed_incomplete_fit", status="training_failed")
                return out_dir
            if time.monotonic() >= deadline:
                commit("timeout_before_calibration", status="timeout_during_training")
                return out_dir
            results["selection"] = {
                "status": "fit_selections_frozen", "candidate_index": args.candidate_index,
                "rho": args.rho,
            }
            commit("all_five_fit_selections_frozen", status="calibration_gate_running",
                   phase="calibration_gate")

        if state["phase"] == "calibration_gate":
            results = state["results"]
            if not _all_seeds_fit_qualified(results):
                results["calibration"] = {
                    "status": "not_evaluated_incomplete_or_unqualified_seed_set"
                }
                commit("calibration_closed_incomplete_fit", status="training_failed")
                return out_dir
            cal_state = results.setdefault("calibration", {})
            anchor_calibration = cal_state.get("lp_anchor_control")
            if anchor_calibration is None:
                anchor_calibration = constrained.metrics(cal_rows, basis32, anchor)
                constrained._control_check(anchor_calibration, constrained.LP_CAL, "LP anchor")
                cal_state.update({"status": "scoring_after_all_fit_selections",
                                  "lp_anchor_control": anchor_calibration,
                                  "per_seed_partial": []})
                commit("calibration_anchor_control_saved", status="calibration_gate_running")
            if time.monotonic() >= deadline:
                commit("timeout_during_calibration", status="timeout_during_calibration")
                return out_dir
            scored = {int(row["seed"]) for row in cal_state.get("per_seed_partial", [])}
            for record in results["seeds"]:
                seed = int(record["seed"])
                if seed in scored:
                    continue
                if time.monotonic() >= deadline:
                    commit("timeout_during_calibration", status="timeout_during_calibration",
                           calibration_seeds_scored=len(scored))
                    return out_dir
                calibration = constrained.metrics(
                    cal_rows, basis32,
                    np.asarray(record["selected_weights"], dtype=np.float64),
                )
                cal_state.setdefault("per_seed_partial", []).append({
                    "seed": seed, "calibration": calibration,
                })
                scored.add(seed)
                commit("candidate_calibration_scored", status="calibration_gate_running",
                       calibration_seed=seed, calibration_seeds_scored=len(scored))
            if time.monotonic() >= deadline:
                commit("timeout_before_gate_evaluation", status="timeout_during_calibration")
                return out_dir
            calibration_records = sorted(
                cal_state.get("per_seed_partial", []), key=lambda row: int(row["seed"])
            )
            if [row["seed"] for row in calibration_records] != list(SEEDS):
                raise ValueError("calibration records do not cover all five fit-selected seeds")
            aggregate = _calibration_aggregate(calibration_records)
            no_blank = all(_no_blank(row["calibration"]) for row in calibration_records)
            gate = constrained._gate(
                aggregate, anchor_calibration, no_blank, complete_seeds=True,
            )
            results["calibration"] = {
                "status": "measured_development_only",
                "lp_anchor_control": anchor_calibration,
                "per_seed": calibration_records, "aggregate": aggregate,
                "no_positive_target_blank_any_dose": no_blank, "gate": gate,
                "calibration_role": "development only; not independent generalization evidence",
                "calibration_gradients": False,
            }
            results["selection"] = {
                "status": "development_gate_complete",
                "qualified_arms": ["robust_corner_hinge"] if gate["passed"] else [],
                "selected_arm": "robust_corner_hinge" if gate["passed"] else None,
                "candidate_index": args.candidate_index, "rho": args.rho,
                "calibration_role": "development only; no generalization claim",
            }
            results["elapsed_seconds"] = max(0.0, time.monotonic() - invocation_started)
            commit("development_gate_complete", status="complete", phase="complete",
                   gate_passed=bool(gate["passed"]))
        return out_dir
    except KeyboardInterrupt:
        state = copy.deepcopy(getattr(args, "_last_canonical_state", None))
        if out_dir is not None and state is not None and state.get("phase") != "complete":
            try:
                state["results"]["status"] = "interrupted"
                previous_event = state["progress"].get("event")
                state["progress"].update({"status": "interrupted", "event": "keyboard_interrupt",
                                          "interrupted_from_event": previous_event,
                                          "updated_utc": datetime.now(timezone.utc).isoformat()})
                _write_canonical_state(out_dir, state)
                args._run_state = state
                args._last_canonical_state = copy.deepcopy(state)
            except BaseException:
                pass
        raise
    except constrained.SidecarWriteError:
        # The canonical snapshot was committed first; resume repairs the sidecars.
        raise
    except Exception as exc:
        state = copy.deepcopy(getattr(args, "_last_canonical_state", None))
        if out_dir is not None and state is not None and state.get("phase") != "complete":
            try:
                state["results"]["status"] = "failed"
                state["results"]["failure"] = {
                    "type": type(exc).__name__, "message": str(exc),
                    "retryable_training_failure": bool(
                        state.get("phase") == "fit_training"
                        and isinstance(exc, (FloatingPointError, np.linalg.LinAlgError, RuntimeError))
                    ),
                }
                if state.get("phase") == "fit_training":
                    state["results"]["calibration"] = {"status": "not_evaluated_failure"}
                    state["results"]["selection"] = {
                        "status": "calibration_closed_failure",
                        "qualified_arms": [], "selected_arm": None,
                    }
                state["progress"].update({"status": "failed", "event": "exception",
                                          "error_type": type(exc).__name__,
                                          "message": str(exc),
                                          "updated_utc": datetime.now(timezone.utc).isoformat()})
                _write_canonical_state(out_dir, state)
                args._run_state = state
                args._last_canonical_state = copy.deepcopy(state)
            except BaseException:
                pass
        raise
    finally:
        if args._run_lock is not None:
            args._run_lock.release()


def main(argv=None):
    args = None
    try:
        args = _parse_args(argv)
        out_dir = run(args)
        canonical_path = out_dir / "run_state.json"
        if canonical_path.is_file():
            results = _read_json(canonical_path)["results"]
        else:
            results = _read_json(out_dir / "results.json")
        print(json.dumps({
            "result_directory": str(out_dir),
            "status": results["status"],
            "selection": results.get("selection"),
        }, ensure_ascii=False))
        return 0 if results["status"] == "complete" else 1
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "message": "resume with the same run identity"}),
              file=sys.stderr)
        return 130
    except Exception as exc:
        # Resume validation failures are read-only. run() only writes a state after
        # a new canonical snapshot exists or a valid resume has been accepted.
        out_dir = getattr(args, "_active_output", None) if args is not None else None
        print(json.dumps({
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "result_directory": str(out_dir) if out_dir is not None else None,
        }, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
