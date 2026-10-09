"""Prospective source-only FIT coverage experiment.

The runner is intentionally separate from the legacy optimization scripts.
It requires an externally frozen schema-5 or schema-6 plan and does not touch
dataset or optical inputs until the explicit ``--run`` mode is selected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
from scipy.optimize import linprog
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import source_coverage as coverage
from scripts import optimize_source_constrained as constrained
from scripts import diagnose_source_pareto as pareto
from light_source import DifferentiableAbbeLitho, PixelatedLightSource, resist_image
from source_robustness import (
    critical_corner_softcount_value,
    critical_corner_softcount_value_gradient,
)
from source_training import SourceDataset


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    return coverage.sha256_file(path)


def _absolute_file(path_value: str, name: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(name + " must be an existing absolute file")
    return path.resolve()


def _verify_git_ancestor(root: Path, base_commit: str) -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", base_commit, "HEAD"],
        cwd=root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def _source_paths() -> tuple[Path, ...]:
    return (
        ROOT / "source_coverage.py", ROOT / "source_pareto.py",
        ROOT / "scripts" / "optimize_source_coverage.py",
        ROOT / "tests" / "test_source_coverage.py",
        ROOT / "docs" / "SOURCE_COVERAGE.md",
        ROOT / "scripts" / "run_protected_pvband_experiment.py",
        ROOT / "scripts" / "diagnose_source_pareto.py",
        ROOT / "scripts" / "optimize_source_constrained.py",
        ROOT / "scripts" / "optimize_source_robust_corners.py",
        ROOT / "scripts" / "diagnose_source_feasibility.py",
        ROOT / "light_source.py", ROOT / "source_robustness.py",
        ROOT / "source_training.py",
    )


def _read_hashed_json(path: Path) -> tuple[dict, bytes, str]:
    raw, digest = coverage.read_hashed_bytes(path)
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("registered JSON must contain an object: " + str(path))
    return parsed, raw, digest


def _validate_identities(plan_path: Path, plan_sha: str, previous_sha: str,
                          plan: dict) -> dict:
    if previous_sha.lower() != coverage.PINNED_PREVIOUS_REPORT_SHA256:
        raise ValueError("--expected-previous-sha256 differs from the required predecessor pin")
    frozen_plan, _plan_bytes, actual_plan_sha = _read_hashed_json(plan_path)
    coverage.validate_plan_payload(frozen_plan)
    if actual_plan_sha != plan_sha:
        raise ValueError("frozen plan changed during the coverage attempt")
    if frozen_plan != plan:
        raise ValueError("parsed frozen plan content changed during identity check")

    expected = plan["input_hashes"]
    path_map = {
        "dataset_sha256": _absolute_file(plan["dataset_file"], "dataset_file"),
        "diagnostic_sha256": _absolute_file(plan["diagnostic_file"], "diagnostic_file"),
        "source_manifest_sha256": _absolute_file(plan["prerequisite_manifest"], "prerequisite_manifest"),
    }
    raws, actual_input_hashes = {}, {}
    for field, path in path_map.items():
        raw, digest = coverage.read_hashed_bytes(path)
        raws[field] = raw
        actual_input_hashes[field] = digest
        if digest != expected[field]:
            raise ValueError("coverage input hash mismatch: " + field)
    diagnostic = json.loads(raws["diagnostic_sha256"].decode("utf-8"))
    manifest = json.loads(raws["source_manifest_sha256"].decode("utf-8"))
    if not isinstance(diagnostic, dict) or not isinstance(manifest, dict):
        raise ValueError("diagnostic and source prerequisite manifest must be JSON objects")
    _validate_diagnostic_contract(diagnostic)

    previous_path = _absolute_file(plan["previous_report"]["path"], "previous_report.path")
    previous_plan_path = _absolute_file(plan["previous_plan"]["path"], "previous_plan.path")
    previous_report, previous_report_bytes, actual_previous_report_sha = _read_hashed_json(previous_path)
    previous_plan, _previous_plan_bytes, actual_previous_plan_sha = _read_hashed_json(previous_plan_path)
    if actual_previous_report_sha != coverage.PINNED_PREVIOUS_REPORT_SHA256:
        raise ValueError("previous Pareto JSON no longer matches the required SHA256")
    if actual_previous_plan_sha != coverage.PINNED_PREVIOUS_PLAN_SHA256:
        raise ValueError("previous Pareto plan no longer matches the required SHA256")
    previous = coverage.validate_previous_report(previous_report)
    superseded_identity = None
    if plan.get("schema_version") == 6:
        superseded = plan["superseded_coverage_attempt"]
        superseded_plan_path = _absolute_file(
            superseded["plan_path"], "superseded_coverage_attempt.plan_path",
        )
        superseded_report_path = _absolute_file(
            superseded["report_path"], "superseded_coverage_attempt.report_path",
        )
        superseded_plan, superseded_plan_sha = coverage.validate_plan_file(
            superseded_plan_path, coverage.PINNED_SUPERSEDED_COVERAGE_PLAN_SHA256,
        )
        if superseded_plan.get("schema_version") != 5:
            raise ValueError("superseded coverage plan is not the consumed schema-5 plan")
        superseded_report, _superseded_report_bytes, superseded_report_sha = _read_hashed_json(
            superseded_report_path
        )
        if superseded_report_sha != coverage.PINNED_SUPERSEDED_COVERAGE_REPORT_SHA256:
            raise ValueError("superseded coverage report no longer matches its pinned SHA256")
        coverage.validate_superseded_coverage_report(
            superseded_report, coverage.PINNED_SUPERSEDED_COVERAGE_PLAN_SHA256,
        )
        superseded_identity = {
            "plan_sha256": coverage.PINNED_SUPERSEDED_COVERAGE_PLAN_SHA256,
            "report_sha256": coverage.PINNED_SUPERSEDED_COVERAGE_REPORT_SHA256,
        }
    source_identity = coverage.verify_clean_source_commit(ROOT, _source_paths())
    _verify_git_ancestor(ROOT, plan["base_commit"])
    input_identity = {
        **actual_input_hashes,
        "previous_report_sha256": actual_previous_report_sha,
        "previous_plan_sha256": actual_previous_plan_sha,
    }
    if superseded_identity is not None:
        input_identity["superseded_coverage_attempt"] = superseded_identity
    return {
        "dataset_path": str(path_map["dataset_sha256"]),
        "dataset_bytes": raws["dataset_sha256"],
        "diagnostic_path": str(path_map["diagnostic_sha256"]),
        "diagnostic": diagnostic,
        "manifest_path": str(path_map["source_manifest_sha256"]),
        "manifest": manifest,
        "previous_report_path": str(previous_path),
        "previous_plan_path": str(previous_plan_path),
        "input_identity": input_identity,
        "previous_weights": previous, "previous_report": previous_report,
        "previous_report_bytes": previous_report_bytes,
        "source_identity": source_identity, "lineage_artifacts": None,
    }


def _validate_diagnostic_contract(diagnostic: dict) -> None:
    inp = diagnostic.get("input")
    if not isinstance(inp, dict):
        raise ValueError("diagnostic is missing the registered input identity")
    if inp.get("dataset_sha256") != "1fb6555fbf1dc4b4748f05f37d557977df5bbfd3b5b8abf5853c68a04716d1d0":
        raise ValueError("diagnostic dataset identity differs from frozen source data")
    for field, ids in (("fit_masks", coverage.ORIGINAL_FIT_LAYOUT_IDS),
                       ("fit_targets", coverage.ORIGINAL_FIT_LAYOUT_IDS),
                       ("calibration_masks", coverage.CALIBRATION_LAYOUT_IDS),
                       ("calibration_targets", coverage.CALIBRATION_LAYOUT_IDS)):
        rows = inp.get(field)
        if (not isinstance(rows, list) or [row.get("layout_id") for row in rows
                if isinstance(row, dict)] != list(ids)):
            raise ValueError("diagnostic %s IDs/order differ from frozen lineage" % field)
        if any(set(row) != {"layout_id", "sha256"}
               or not isinstance(row["sha256"], str) or len(row["sha256"]) != 64
               for row in rows):
            raise ValueError("diagnostic %s hash descriptors are malformed" % field)
    bases = inp.get("bases")
    expected_ids = list(coverage.ORIGINAL_FIT_LAYOUT_IDS + coverage.CALIBRATION_LAYOUT_IDS)
    expected_splits = ["fit"] * 4 + ["calibration"] * 4
    if not isinstance(bases, list) or [row.get("layout_id") for row in bases
                                      if isinstance(row, dict)] != expected_ids:
        raise ValueError("diagnostic basis descriptors differ from frozen FIT/calibration order")
    for row, expected_split in zip(bases, expected_splits):
        if (set(row) != {"layout_id", "split", "sha256", "shape", "dtype"}
                or row.get("split") != expected_split
                or row.get("shape") != [1, 49, 128, 128]
                or row.get("dtype") != "torch.float32"
                or not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64):
            raise ValueError("diagnostic float32 basis descriptor is malformed")


def _check_lineage_artifacts(identity: dict) -> None:
    for item in identity.get("lineage_artifacts") or ():
        raw, digest = coverage.read_hashed_bytes(item["path"])
        if digest != item["sha256"]:
            raise ValueError("prerequisite lineage artifact changed: " + item["path"])
        if Path(item["path"]).suffix == ".json":
            json.loads(raw.decode("utf-8"))

def _assert_identity_unchanged(plan_path: Path, plan_sha: str,
                               previous_sha: str, plan: dict, baseline: dict) -> dict:
    current = _validate_identities(plan_path, plan_sha, previous_sha, plan)
    if (current["input_identity"] != baseline["input_identity"]
            or current["source_identity"] != baseline["source_identity"]):
        raise ValueError("pinned input or committed source identity changed during coverage run")
    current["lineage_artifacts"] = baseline.get("lineage_artifacts")
    _check_lineage_artifacts(current)
    return current


def _check_output_root(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError("output root must be absolute")
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        return path
    raise ValueError("output root must be outside the source repository")


def _preflight(args) -> tuple[dict, str, dict]:
    plan_path = _absolute_file(args.plan_file, "plan_file")
    plan, plan_sha = coverage.validate_plan_file(
        plan_path, args.expected_plan_sha256
    )
    identities = _validate_identities(
        plan_path, plan_sha, args.expected_previous_sha256, plan
    )
    if Path(plan["prerequisite_manifest"]).resolve().parent != plan_path.parent:
        raise ValueError("attempt marker and frozen plan must share the manifest parent directory")
    return plan, plan_sha, identities


def _load_fit_only(dataset_bytes: bytes) -> SourceDataset:
    """Deserialize the already-hashed archive bytes, then access only `fit`."""
    payload = torch.load(io.BytesIO(dataset_bytes), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "fit" not in payload:
        raise ValueError("source dataset archive lacks its original FIT entry")
    return SourceDataset(**payload["fit"])


def _as_numpy_rows(fit: SourceDataset) -> list[dict]:
    rows = []
    for index, layout_id in enumerate(fit.layout_ids):
        rows.append({
            "layout_id": layout_id,
            "mask": fit.masks[index, 0].detach().cpu().numpy().astype(np.float32, copy=True),
            "target": fit.targets[index, 0].detach().cpu().numpy().astype(np.float32, copy=True),
            "split": "fit",
            "family": "original_fit",
        })
    return rows


def _generate_new_targets(layouts, device, physical: dict) -> list[dict]:
    """Generate only the three new FIT targets with the pinned original teacher."""
    from scripts import run_protected_pvband_experiment as original_protocol

    if (original_protocol.RASTER != physical["raster"]
            or original_protocol.PIXEL != physical["pixel_nm"]
            or original_protocol.THRESHOLD != physical["threshold"]
            or original_protocol.STEEPNESS != physical["steepness"]
            or original_protocol.BINARY != 0.5):
        raise ValueError("frozen teacher/resist constants differ from the coverage plan")
    teacher_source = original_protocol.teacher_source(device)
    teacher = DifferentiableAbbeLitho(
        teacher_source,
        numerical_aperture=physical["NA"],
        wavelength_nm=physical["wavelength_nm"],
        pixel_size_nm=original_protocol.PIXEL,
        source_chunk_size=8,
        cache_max_bytes=0,
    ).to(device)
    rows = []
    with torch.no_grad():
        for layout in layouts:
            mask = torch.as_tensor(layout.mask, dtype=torch.float32, device=device)
            aerial = teacher(mask[None, None])
            printed = resist_image(
                aerial, threshold=physical["threshold"],
                steepness=physical["steepness"], dose=1.0,
            )
            target = (printed >= original_protocol.BINARY).to(torch.float32)[0, 0]
            rows.append({
                "layout_id": layout.layout_id,
                "family": layout.family,
                "mask": np.ascontiguousarray(layout.mask, dtype=np.float32),
                "target": target.detach().cpu().numpy().astype(np.float32, copy=True),
                "split": "fit",
            })
    del teacher
    return rows


def _novelty_check(original_rows: list[dict], new_rows: list[dict], generation: dict) -> list[dict]:
    old_mask_hashes = {coverage.sha256_array(row["mask"]) for row in original_rows}
    old_target_hashes = {
        coverage.sha256_array(coverage.target_hashable(row["target"]))
        for row in original_rows
    }
    seen_masks, seen_targets, audit = set(old_mask_hashes), set(old_target_hashes), []
    for row in new_rows:
        mask_hash = coverage.sha256_array(row["mask"])
        target = coverage.target_hashable(row["target"])
        target_hash = coverage.sha256_array(target)
        if mask_hash in seen_masks or target_hash in seen_targets:
            raise ValueError("new FIT mask/target duplicates permitted original FIT or prior new FIT")
        positive_fraction = float(target.mean())
        if not generation["positive_fraction_min"] <= positive_fraction <= generation["positive_fraction_max"]:
            raise ValueError("new FIT target fails the fixed positive-fraction screen")
        seen_masks.add(mask_hash)
        seen_targets.add(target_hash)
        audit.append({
            "layout_id": row["layout_id"], "family": row["family"],
            "mask_sha256": mask_hash, "target_sha256": target_hash,
            "target_positive_fraction": positive_fraction,
        })
    return audit


def _prepare_basis(rows: list[dict], device, physical: dict, expected_bases=None):
    uniform_source = PixelatedLightSource(
        physical["source_grid"], sigma_inner=physical["sigma_inner"],
        sigma_outer=physical["sigma_outer"],
    )
    simulator = DifferentiableAbbeLitho(
        uniform_source, numerical_aperture=physical["NA"],
        wavelength_nm=physical["wavelength_nm"], pixel_size_nm=physical["pixel_nm"],
        source_chunk_size=8, cache_max_bytes=0,
    ).to(device)
    basis32, basis64, basis_torch, targets, parity = {}, {}, {}, {}, []
    bytes_per_basis = 49 * physical["raster"] * physical["raster"] * 4
    if bytes_per_basis > 512 * 1024**2:
        raise MemoryError("one float32 basis exceeds the frozen optical memory budget")
    for row in rows:
        mask = torch.as_tensor(row["mask"], dtype=torch.float32, device=device)
        basis = simulator.prepare_basis(mask[None, None], defocus_nm=0.0,
                                        max_bytes=512 * 1024**2)
        intensity_all = basis.intensities.detach().to(dtype=torch.float32, device="cpu").contiguous()
        intensity = intensity_all[0]
        if intensity.shape[0] != 49:
            raise ValueError("prepared basis does not have the registered 49 source samples")
        if expected_bases is not None and row["layout_id"] in expected_bases:
            descriptor = expected_bases[row["layout_id"]]
            if (constrained.sha256_tensor(intensity_all) != descriptor["sha256"]
                    or list(intensity_all.shape) != descriptor["shape"]
                    or str(intensity_all.dtype) != descriptor["dtype"]):
                raise ValueError("float32 basis hash/shape/dtype differs for %s" % row["layout_id"])
        basis32[row["layout_id"]] = intensity.detach().cpu().numpy().copy()
        array64 = intensity.detach().to(dtype=torch.float64, device="cpu").numpy().copy()
        if not np.isfinite(array64).all() or float(array64.min(initial=0.0)) < 0.0:
            raise ValueError("FIT basis is non-finite or negative")
        basis64[row["layout_id"]] = array64
        basis_torch[row["layout_id"]] = intensity.detach().to(dtype=torch.float64)
        targets[row["layout_id"]] = torch.as_tensor(
            row["target"], dtype=torch.bool, device=device
        )
        # Check direct simulator/basis parity for every pinned original FIT or
        # calibration layout, matching the legacy diagnostic tolerance.
        mask_t = mask
        with torch.no_grad():
            direct = simulator(mask_t)
            evaluated = simulator.evaluate_basis(basis)
            torch.testing.assert_close(direct, evaluated, rtol=1e-5, atol=1e-6)
            parity.append({"layout_id": row["layout_id"],
                           "max_abs_error": float((direct - evaluated).abs().max().item())})
        del basis
    return simulator, basis32, basis64, basis_torch, targets, parity


def _evaluate_rows(rows, basis32, weights, physical):
    records = []
    for row in rows:
        metric = coverage.hard_metrics(
            basis32[row["layout_id"]], row["target"], weights,
            threshold=physical["threshold"], steepness=physical["steepness"],
        )
        records.append({"layout_id": row["layout_id"], **metric})
    return {"mean": coverage.aggregate_metrics(records), "per_layout": records,
            "no_blank_positive_target_any_dose": all(
                row["no_blank_positive_target_any_dose"] for row in records
            )}


def _linprog_lmo(domain, gradient, deadline: float) -> dict:
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        return {"status": "deadline", "weights": None}
    limit = min(60.0, remaining)
    n = domain.source_count
    result = linprog(
        np.asarray(gradient, dtype=np.float64),
        A_ub=domain.a_ub, b_ub=domain.b_ub,
        A_eq=np.ones((1, n), dtype=np.float64), b_eq=np.ones(1, dtype=np.float64),
        bounds=tuple((0.0, None) for _ in range(n)), method="highs",
        options={"time_limit": limit, "primal_feasibility_tolerance": 1e-9,
                 "dual_feasibility_tolerance": 1e-9, "presolve": True},
    )
    record = {
        "status_code": int(result.status), "message": str(result.message),
        "success": bool(result.success), "time_limit_seconds": float(limit),
        "iterations": int(getattr(result, "nit", 0) or 0),
    }
    if not result.success or result.x is None:
        status = "deadline" if int(result.status) == 1 else "solver_failure"
        record["terminal_class"] = "time_limit" if status == "deadline" else "solver_failure"
        return {"status": status, "weights": None, "record": record}
    weights = np.asarray(result.x, dtype=np.float64)
    residual = domain.verify(weights)
    if not residual["passed"]:
        return {"status": "residual_failure", "weights": None,
                "rejected_x": weights.tolist(), "record": record, "residual": residual}
    return {"status": "optimal_verified", "weights": weights,
            "objective_value": float(np.asarray(gradient) @ weights),
            "record": record, "residual": residual}


def _armijo(current, vertex, loss, gap, bases, targets, beta, deadline):
    gamma = 1.0
    evaluations = 0
    while gamma >= 2.0 ** -24:
        if time.monotonic() >= deadline:
            return {"accepted": False, "timed_out": True, "evaluations": evaluations}
        trial = current + gamma * (vertex - current)
        value = critical_corner_softcount_value(trial, bases, targets, beta)
        evaluations += 1
        if math.isfinite(value) and value <= loss - 1e-4 * gamma * gap:
            return {"accepted": True, "gamma": gamma, "value": value,
                    "evaluations": evaluations, "weights": trial}
        gamma *= 0.5
    return {"accepted": False, "timed_out": False, "evaluations": evaluations}


def _initial_weights(reference, seed, domain):
    candidate = coverage.simplex_jitter(reference, seed, 1e-4)
    check = domain.verify(candidate)
    if check["passed"]:
        return candidate, {"used_jitter": True, "fallback": False, "check": check}
    ref_check = domain.verify(reference)
    if not ref_check["passed"]:
        raise ValueError("reference fallback is not feasible in the frozen guard domain")
    return reference.copy(), {"used_jitter": False, "fallback": True,
                              "rejected_jitter_check": check, "check": ref_check}


def _initial_feasible_weights(reference, seed, domain, candidate_auditor,
                              previous_starts, deadline, protocol=None,
                              progress_callback=None):
    """Find the first distinct audited convex mixture from seeded feasible LMO vertices."""
    protocol = protocol or coverage.SCHEMA6_INITIALIZATION_PROTOCOL
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    previous = [np.asarray(value, dtype=np.float64).reshape(-1)
                for value in (previous_starts or [])]
    previous_hashes = {
        coverage.sha256_array(np.asarray(value, dtype="<f8")) for value in previous
    }
    record = {
        "algorithm": "schema6_seeded_feasible_lmo_mixtures",
        "seed": int(seed), "status": "searching", "fallback": False,
        "used_jitter": False, "reference_weights_sha256": coverage.sha256_array(
            np.asarray(reference, dtype="<f8")
        ),
        "prior_start_count": len(previous), "prior_start_hashes": sorted(previous_hashes),
        "minimum_l1_distance": float(protocol["minimum_l1_distance"]),
        "directions": [], "directions_attempted": 0, "lmo_solves": 0,
        "started_monotonic": float(time.monotonic()),
        "initialization_budget_seconds": float(protocol["maximum_initialization_seconds_per_seed"]),
    }
    min_distance = float(protocol["minimum_l1_distance"])
    max_directions = int(protocol["maximum_lmo_directions_per_seed"])
    alphas = tuple(float(value) for value in protocol["mixing_alphas"])
    reference_hash = record["reference_weights_sha256"]

    def notify(event: str, details: dict | None = None) -> None:
        record["elapsed_seconds"] = float(max(0.0, time.monotonic() - record["started_monotonic"]))
        record["last_event"] = event
        if details is not None:
            record["last_event_details"] = details
        if progress_callback is not None:
            progress_callback(record)

    for direction_index in range(1, max_directions + 1):
        if time.monotonic() >= deadline:
            record["status"] = "timeout"
            record["failure_reason"] = "initialization_deadline_before_lmo"
            notify("deadline_before_direction", {"direction_index": direction_index})
            break
        direction = np.asarray(rng.normal(size=reference.shape), dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if not math.isfinite(direction_norm) or direction_norm <= 0.0:
            record["status"] = "direction_generation_failure"
            record["failure_reason"] = "random_direction_has_invalid_l2_norm"
            notify("invalid_random_direction", {"direction_index": direction_index})
            break
        direction /= direction_norm
        direction_row = {
            "direction_index": direction_index,
            "elapsed_before_direction_seconds": float(
                max(0.0, time.monotonic() - record["started_monotonic"])
            ),
            "objective": direction.tolist(),
            "objective_sha256": coverage.sha256_array(np.asarray(direction, dtype="<f8")),
            "objective_l2_norm": float(np.linalg.norm(direction)),
            "lmo_status": "pending", "candidates": [],
        }
        record["directions_attempted"] = direction_index
        record["lmo_solves"] += 1
        lmo_started = time.monotonic()
        try:
            lmo = _linprog_lmo(domain, direction, deadline)
        except Exception as exc:
            direction_row["lmo_elapsed_seconds"] = float(
                max(0.0, time.monotonic() - lmo_started)
            )
            direction_row.update({
                "lmo_status": "exception",
                "lmo_failure": {"type": type(exc).__name__, "message": str(exc)},
            })
            record["directions"].append(direction_row)
            record["status"] = "lmo_failure"
            record["failure_reason"] = "lmo_exception"
            notify("lmo_exception", {"direction_index": direction_index})
            break
        direction_row["lmo_elapsed_seconds"] = float(
            max(0.0, time.monotonic() - lmo_started)
        )
        direction_row["lmo_status"] = lmo.get("status")
        direction_row["lmo_record"] = lmo.get("record")
        direction_row["lmo_residual"] = lmo.get("residual")
        direction_row["lmo_objective_value"] = lmo.get("objective_value")
        if lmo.get("weights") is not None:
            vertex = np.asarray(lmo["weights"], dtype=np.float64).reshape(-1)
            direction_row["lmo_vertex"] = vertex.tolist()
            direction_row["lmo_vertex_sha256"] = coverage.sha256_array(
                np.asarray(vertex, dtype="<f8")
            )
        else:
            vertex = None
        if lmo.get("status") != "optimal_verified" or vertex is None:
            record["directions"].append(direction_row)
            record["status"] = "timeout" if lmo.get("status") == "deadline" else "lmo_failure"
            record["failure_reason"] = "lmo_" + str(lmo.get("status"))
            notify("lmo_failed", {"direction_index": direction_index})
            break

        for alpha in alphas:
            if time.monotonic() >= deadline:
                direction_row["candidates"].append({
                    "alpha": alpha, "status": "timeout_before_candidate_audit",
                })
                record["directions"].append(direction_row)
                record["status"] = "timeout"
                record["failure_reason"] = "initialization_deadline_during_candidate_search"
                notify("deadline_during_candidate_search", {
                    "direction_index": direction_index, "alpha": alpha,
                })
                return None, record
            candidate = reference + alpha * (vertex - reference)
            candidate = np.asarray(candidate, dtype=np.float64)
            candidate_hash = coverage.sha256_array(np.asarray(candidate, dtype="<f8"))
            distances = [{"role": "seed17_reference", "l1": float(np.abs(candidate - reference).sum())}]
            distances.extend({"role": "previous_start_%d" % index,
                              "l1": float(np.abs(candidate - value).sum())}
                             for index, value in enumerate(previous))
            distinct = candidate_hash != reference_hash and candidate_hash not in previous_hashes
            distinct = distinct and all(item["l1"] > min_distance for item in distances)
            audit_started = time.monotonic()
            try:
                audits = candidate_auditor(candidate)
            except Exception as exc:
                audits = {"passed": False, "audit_exception": {
                    "type": type(exc).__name__, "message": str(exc),
                }}
            if audits.get("deadline_exhausted"):
                direction_row["candidates"].append({
                    "alpha": alpha, "weights": candidate.tolist(),
                    "weights_sha256": candidate_hash, "l1_distances": distances,
                    "audit_elapsed_seconds": float(max(0.0, time.monotonic() - audit_started)),
                    "elapsed_seconds": float(max(0.0, time.monotonic() - record["started_monotonic"])),
                    "distinct": bool(distinct), "audits": audits,
                    "status": "timeout_during_candidate_audit",
                })
                record["directions"].append(direction_row)
                record["status"] = "timeout"
                record["failure_reason"] = "initialization_deadline_during_candidate_audit"
                notify("deadline_during_candidate_audit", {
                    "direction_index": direction_index, "alpha": alpha,
                })
                record.pop("started_monotonic", None)
                return None, record
            if time.monotonic() >= deadline:
                direction_row["candidates"].append({
                    "alpha": alpha, "weights": candidate.tolist(),
                    "weights_sha256": candidate_hash, "l1_distances": distances,
                    "audit_elapsed_seconds": float(max(0.0, time.monotonic() - audit_started)),
                    "elapsed_seconds": float(max(0.0, time.monotonic() - record["started_monotonic"])),
                    "distinct": bool(distinct), "audits": audits,
                    "status": "timeout_after_candidate_audit",
                })
                record["directions"].append(direction_row)
                record["status"] = "timeout"
                record["failure_reason"] = "initialization_deadline_after_candidate_audit"
                notify("deadline_after_candidate_audit", {
                    "direction_index": direction_index, "alpha": alpha,
                })
                record.pop("started_monotonic", None)
                return None, record
            feasible = bool(audits.get("passed"))
            accepted = bool(feasible and distinct)
            candidate_row = {
                "alpha": alpha, "weights": candidate.tolist(),
                "weights_sha256": candidate_hash, "l1_distances": distances,
                "audit_elapsed_seconds": float(max(0.0, time.monotonic() - audit_started)),
                "elapsed_seconds": float(max(0.0, time.monotonic() - record["started_monotonic"])),
                "distinct": bool(distinct), "audits": audits,
                "status": "accepted" if accepted else (
                    "rejected_infeasible" if not feasible else "rejected_not_distinct"
                ),
            }
            direction_row["candidates"].append(candidate_row)
            if accepted:
                direction_row["selected_alpha"] = alpha
                record["directions"].append(direction_row)
                record.update({
                    "status": "feasible_start_found",
                    "selected_direction_index": direction_index,
                    "selected_alpha": alpha,
                    "weights": candidate.tolist(),
                    "weights_sha256": candidate_hash,
                    "accepted_audits": audits,
                })
                notify("feasible_start_found", {
                    "direction_index": direction_index,
                    "alpha": alpha, "weights_sha256": candidate_hash,
                })
                if time.monotonic() >= deadline:
                    record["status"] = "timeout"
                    record["failure_reason"] = "initialization_deadline_during_start_recording"
                    notify("deadline_during_start_recording", {
                        "direction_index": direction_index, "alpha": alpha,
                    })
                    record.pop("started_monotonic", None)
                    return None, record
                record.pop("started_monotonic", None)
                return candidate, record
            notify("candidate_rejected", {
                "direction_index": direction_index, "alpha": alpha,
                "status": candidate_row["status"],
            })
        else:
            record["directions"].append(direction_row)
            continue
        break

    if record["status"] == "searching":
        record["status"] = "no_distinct_feasible_start"
        record["failure_reason"] = "registered_lmo_direction_budget_exhausted"
    notify("initialization_failed", {"status": record["status"]})
    record.pop("started_monotonic", None)
    return None, record


def _per_layout_objective_diagnostics(weights, rows, bases_torch, targets_torch,
                                      beta, snapshot_id, deadline, progress_callback=None):
    """Record per-layout losses/gradients and assert parity with the equal-weight objective."""
    layout_ids = [row["layout_id"] for row in rows]
    record = {
        "snapshot": snapshot_id, "beta": float(beta), "status": "in_progress",
        "layout_ids": layout_ids, "per_layout": [],
        "parity_tolerance": 1e-10,
    }

    def notify() -> None:
        if progress_callback is not None:
            progress_callback(record)

    if time.monotonic() >= deadline:
        record["status"] = "timeout"
        record["failure_reason"] = "deadline_before_aggregate_diagnostic"
        notify()
        return record
    all_bases = {key: bases_torch[key] for key in layout_ids}
    all_targets = {key: targets_torch[key] for key in layout_ids}
    aggregate_loss, aggregate_gradient = critical_corner_softcount_value_gradient(
        weights, all_bases, all_targets, beta,
    )
    aggregate_gradient = np.asarray(aggregate_gradient, dtype=np.float64)
    record["aggregate_loss"] = float(aggregate_loss)
    record["aggregate_gradient"] = aggregate_gradient.tolist()
    notify()
    if time.monotonic() >= deadline:
        record["status"] = "timeout"
        record["failure_reason"] = "deadline_after_aggregate_diagnostic"
        notify()
        return record
    for layout_id in layout_ids:
        if time.monotonic() >= deadline:
            record["status"] = "timeout"
            record["failure_reason"] = "deadline_during_per_layout_diagnostics"
            notify()
            return record
        loss, gradient = critical_corner_softcount_value_gradient(
            weights, {layout_id: bases_torch[layout_id]},
            {layout_id: targets_torch[layout_id]}, beta,
        )
        gradient = np.asarray(gradient, dtype=np.float64)
        record["per_layout"].append({
            "layout_id": layout_id, "softcount_loss": float(loss),
            "gradient": gradient.tolist(),
            "gradient_l2_norm": float(np.linalg.norm(gradient)),
        })
        notify()
    mean_loss = float(np.mean([row["softcount_loss"] for row in record["per_layout"]]))
    mean_gradient = np.mean(
        np.stack([np.asarray(row["gradient"], dtype=np.float64)
                  for row in record["per_layout"]]), axis=0,
    )
    loss_error = abs(mean_loss - float(aggregate_loss))
    gradient_error = float(np.max(np.abs(mean_gradient - aggregate_gradient), initial=0.0))
    record["mean_per_layout_loss"] = mean_loss
    record["mean_gradient"] = mean_gradient.tolist()
    record["loss_parity_abs_error"] = float(loss_error)
    record["gradient_parity_max_abs_error"] = gradient_error
    record["status"] = "complete" if max(loss_error, gradient_error) <= 1e-10 else "parity_failure"
    if record["status"] == "parity_failure":
        record["failure_reason"] = "per_layout_mean_does_not_match_aggregate_objective"
    if time.monotonic() >= deadline:
        record["status"] = "timeout"
        record["failure_reason"] = "deadline_after_per_layout_diagnostics"
    notify()
    if time.monotonic() >= deadline and record["status"] == "complete":
        record["status"] = "timeout"
        record["failure_reason"] = "deadline_during_diagnostic_progress_write"
        notify()
    return record


def _checkpoint_metrics(rows, old_rows, new_rows, basis32, weights, bases_torch,
                        targets_torch, beta, anchor, reference, physical, order):
    all_eval = _evaluate_rows(rows, basis32, weights, physical)
    old_eval = _evaluate_rows(old_rows, basis32, weights, physical)
    new_eval = _evaluate_rows(new_rows, basis32, weights, physical)
    old_gate = (
        old_eval["mean"]["band_pixels"] <= 234.5
        and old_eval["mean"]["L2_pixels"] <= 0.0
        and old_eval["mean"]["L2_worst_dose_pixels"] <= 150.25
        and old_eval["no_blank_positive_target_any_dose"]
    )
    training_objective = critical_corner_softcount_value(
        weights, bases_torch, targets_torch, beta
    )
    selection_objective = critical_corner_softcount_value(
        weights, bases_torch, targets_torch, coverage.BETAS[-1]
    )
    guard_audit = coverage.audit_float32_critical_guards(
        basis32, {row["layout_id"]: row["target"] for row in rows},
        anchor, reference, weights,
        [row["layout_id"] for row in old_rows],
        [row["layout_id"] for row in new_rows],
    )
    return {
        "checkpoint_order": int(order), "beta": float(beta),
        "training_soft_objective": float(training_objective),
        "selection_soft_objective": float(selection_objective),
        "selection_soft_objective_beta": float(coverage.BETAS[-1]),
        "all_fit": all_eval, "original_fit_mean": old_eval["mean"],
        "new_fit_mean": new_eval["mean"],
        "no_blank_positive_target_any_dose": (
            old_eval["no_blank_positive_target_any_dose"]
            and new_eval["no_blank_positive_target_any_dose"]
        ),
        "float32_critical_guard_audit": guard_audit,
        "original_fit_gate_passed": bool(old_gate),
        "new_fit_gate_passed": False,
        "original_nominal_polytope": None,
        "guard_domain": None,
        "l1_to_lp_anchor": float(np.abs(weights - anchor).sum()),
        "l1_to_reference": float(np.abs(weights - reference).sum()),
        "weights": np.asarray(weights, dtype=np.float64).tolist(),
    }


def _fit_checkpoint_qualified(row, new_reference_mean, original_poly, domain, weights):
    new = row["new_fit_mean"]
    ref = new_reference_mean
    row["new_fit_gate_passed"] = bool(
        new["band_pixels"] < ref["band_pixels"]
        and new["L2_pixels"] <= ref["L2_pixels"]
        and new["L2_worst_dose_pixels"] <= ref["L2_worst_dose_pixels"]
    )
    row["original_nominal_polytope"] = original_poly
    row["guard_domain"] = domain.verify(weights)
    row["qualified"] = bool(
        row["original_fit_gate_passed"]
        and row["new_fit_gate_passed"]
        and row["no_blank_positive_target_any_dose"]
        and row["original_nominal_polytope"]["passed"]
        and row["guard_domain"]["passed"]
        and row["float32_critical_guard_audit"]["passed"]
    )
    return row


def _incumbent_snapshot(seed, weights, steps_completed, phase, local_step, beta,
                        seed_started, event, details=None):
    snapshot = {
        "seed": int(seed), "steps_completed": int(steps_completed),
        "phase": phase, "local_step": local_step, "beta": beta,
        "elapsed_seconds": float(max(0.0, time.monotonic() - seed_started)),
        "event": event,
        "weights": np.asarray(weights, dtype=np.float64).tolist(),
        "weights_sha256": coverage.sha256_array(np.asarray(weights, dtype="<f8")),
    }
    if details is not None:
        snapshot["details"] = details
    return snapshot


def _run_seed(seed, ref, anchor, domain, all_rows, old_rows, new_rows,
              basis32, basis64, basis_torch, targets_torch, physical, new_reference_mean,
              seed_started, seed_deadline, checkpoint_callback, iteration_callback,
              initialization_protocol=None, diagnostics_protocol=None,
              prior_starts=None):
    history, checkpoints, order = [], [], 0
    diagnostics = []
    iteration_sink = iteration_callback

    def iteration_callback(snapshot, current_history, current_checkpoints):
        if initialization_protocol is not None:
            details = snapshot.get("details")
            if not (isinstance(details, dict) and "initialization" in details):
                snapshot["details"] = {
                    "optimizer_event_details": details,
                    "initialization": init_record,
                    "per_layout_diagnostics": diagnostics,
                }
        iteration_sink(snapshot, current_history, current_checkpoints)

    if initialization_protocol is None:
        # Preserve the already-consumed schema-5 jitter/fallback behavior verbatim.
        current, init_record = _initial_weights(ref, seed, domain)
        last_incumbent = _incumbent_snapshot(
            seed, current, 0, None, None, None, seed_started, "initialized", init_record,
        )
        iteration_callback(last_incumbent, history, checkpoints)
        initial_poly = coverage.verify_original_nominal_polytope(
                [basis64[row["layout_id"]] for row in old_rows],
                [row["target"] for row in old_rows], current)
        if not initial_poly["passed"]:
            current = ref.copy()
            init_record = {**init_record, "used_jitter": False, "fallback": True,
                           "jitter_full_nominal_check": initial_poly,
                           "check": domain.verify(ref)}
            if not coverage.verify_original_nominal_polytope(
                    [basis64[row["layout_id"]] for row in old_rows],
                    [row["target"] for row in old_rows], ref)["passed"]:
                raise ValueError("seed17 reference fails the full original nominal polytope")
        last_incumbent = _incumbent_snapshot(
            seed, current, 0, None, None, None, seed_started,
            "warm_start_feasible", init_record,
        )
        iteration_callback(last_incumbent, history, checkpoints)
    else:
        current = np.asarray(ref, dtype=np.float64).copy()
        init_state = {
            "algorithm": "schema6_seeded_feasible_lmo_mixtures", "seed": int(seed),
            "status": "searching", "fallback": False, "used_jitter": False,
            "per_layout_diagnostics": diagnostics,
        }
        init_record = init_state
        last_incumbent = _incumbent_snapshot(
            seed, current, 0, None, None, None, seed_started,
            "feasible_start_search_started", {"initialization": init_state},
        )
        iteration_callback(last_incumbent, history, checkpoints)

        def publish_schema6(event, active_diagnostic=None):
            nonlocal last_incumbent
            details = {"initialization": init_record,
                       "per_layout_diagnostics": diagnostics}
            if active_diagnostic is not None:
                details["active_per_layout_diagnostic"] = active_diagnostic
            last_incumbent = _incumbent_snapshot(
                seed, current, len(history), None, None, None, seed_started,
                event, details,
            )
            iteration_callback(last_incumbent, history, checkpoints)

        def init_progress(record):
            init_state.clear()
            init_state.update(record)
            init_state["per_layout_diagnostics"] = diagnostics
            publish_schema6("feasible_start_search_progress")

        init_deadline = min(
            seed_deadline,
            time.monotonic() + float(initialization_protocol[
                "maximum_initialization_seconds_per_seed"]),
        )

        def candidate_auditor(candidate):
            checks = {}
            if time.monotonic() >= init_deadline:
                return {"passed": False, "deadline_exhausted": True,
                        "failure_reason": "deadline_before_candidate_audit"}
            checks["guard_domain"] = domain.verify(candidate)
            if time.monotonic() >= init_deadline:
                return {"passed": False, "deadline_exhausted": True,
                        "checks": checks, "failure_reason": "deadline_after_guard_domain"}
            checks["original_nominal_polytope"] = coverage.verify_original_nominal_polytope(
                [basis64[row["layout_id"]] for row in old_rows],
                [row["target"] for row in old_rows], candidate,
            )
            if time.monotonic() >= init_deadline:
                return {"passed": False, "deadline_exhausted": True,
                        "checks": checks, "failure_reason": "deadline_after_full_nominal_audit"}
            checks["float32_critical_guard_audit"] = coverage.audit_float32_critical_guards(
                basis32, {row["layout_id"]: row["target"] for row in all_rows},
                anchor, ref, candidate,
                [row["layout_id"] for row in old_rows],
                [row["layout_id"] for row in new_rows],
            )
            checks["passed"] = bool(
                checks["guard_domain"].get("passed")
                and checks["original_nominal_polytope"].get("passed")
                and checks["float32_critical_guard_audit"].get("passed")
            )
            if time.monotonic() >= init_deadline:
                checks["passed"] = False
                checks["deadline_exhausted"] = True
                checks["failure_reason"] = "deadline_during_float32_audit"
            return checks

        current, init_record = _initial_feasible_weights(
            ref, seed, domain, candidate_auditor, prior_starts,
            init_deadline, protocol=initialization_protocol,
            progress_callback=init_progress,
        )
        init_state.clear()
        init_state.update(init_record)
        init_record = init_state
        init_record["per_layout_diagnostics"] = diagnostics
        if current is None:
            status = "timeout" if init_record.get("status") == "timeout" else "initialization_failed"
            publish_schema6("feasible_start_search_failed")
            return {"seed": seed, "status": status, "initialization": init_record,
                    "per_layout_diagnostics": diagnostics, "steps_completed": 0,
                    "history": history, "checkpoints": checkpoints,
                    "selected": None, "last_incumbent": last_incumbent}
        current = np.asarray(current, dtype=np.float64)
        if time.monotonic() >= seed_deadline:
            init_record["status"] = "timeout_after_feasible_start"
            publish_schema6("deadline_after_feasible_start")
            return {"seed": seed, "status": "timeout", "initialization": init_record,
                    "per_layout_diagnostics": diagnostics, "steps_completed": 0,
                    "history": history, "checkpoints": checkpoints,
                    "selected": None, "last_incumbent": last_incumbent}
        initial_poly = coverage.verify_original_nominal_polytope(
            [basis64[row["layout_id"]] for row in old_rows],
            [row["target"] for row in old_rows], current,
        )
        initial_domain = domain.verify(current)
        if not initial_poly["passed"] or not initial_domain["passed"]:
            init_record["status"] = "accepted_start_recheck_failed"
            init_record["post_search_recheck"] = {
                "original_nominal_polytope": initial_poly,
                "guard_domain": initial_domain,
            }
            publish_schema6("accepted_start_recheck_failed")
            return {"seed": seed, "status": "initialization_failed",
                    "initialization": init_record, "per_layout_diagnostics": diagnostics,
                    "steps_completed": 0, "history": history,
                    "checkpoints": checkpoints, "selected": None,
                    "last_incumbent": last_incumbent}
        publish_schema6("warm_start_feasible")

        def run_diagnostic(snapshot_id, beta):
            active = {"snapshot": snapshot_id, "status": "starting", "beta": beta}
            init_record["active_per_layout_diagnostic"] = active
            publish_schema6("per_layout_diagnostic_started", active)

            def diagnostic_progress(record):
                init_record["active_per_layout_diagnostic"] = record
                publish_schema6("per_layout_diagnostic_progress", record)

            try:
                result = _per_layout_objective_diagnostics(
                    current, all_rows,
                    basis_torch, targets_torch, beta, snapshot_id,
                    seed_deadline, progress_callback=diagnostic_progress,
                )
            except Exception as exc:
                result = {"snapshot": snapshot_id, "beta": float(beta),
                          "status": "exception", "failure_reason": str(exc),
                          "failure_type": type(exc).__name__}
            diagnostics.append(result)
            init_record.pop("active_per_layout_diagnostic", None)
            publish_schema6("per_layout_diagnostic_finished", result)
            return result

        first_diagnostic = run_diagnostic("initial_beta_200", coverage.BETAS[0])
        if first_diagnostic.get("status") != "complete":
            status = "timeout" if first_diagnostic.get("status") == "timeout" else "diagnostics_failure"
            return {"seed": seed, "status": status, "initialization": init_record,
                    "per_layout_diagnostics": diagnostics, "steps_completed": 0,
                    "history": history, "checkpoints": checkpoints,
                    "selected": None, "last_incumbent": last_incumbent}
    bases_subset = {key: basis_torch[key] for key in [row["layout_id"] for row in all_rows]}
    targets_subset = {key: targets_torch[key] for key in [row["layout_id"] for row in all_rows]}
    for phase, beta in enumerate(coverage.BETAS):
        if diagnostics_protocol is not None and phase > 0:
            snapshot_id = "transition_beta_%d" % int(beta)
            boundary_diagnostic = run_diagnostic(snapshot_id, beta)
            if boundary_diagnostic.get("status") != "complete":
                status = "timeout" if boundary_diagnostic.get("status") == "timeout" else "diagnostics_failure"
                return {"seed": seed, "status": status, "initialization": init_record,
                        "per_layout_diagnostics": diagnostics, "steps_completed": len(history),
                        "history": history, "checkpoints": checkpoints,
                        "selected": None, "last_incumbent": last_incumbent}
        for local_step in range(coverage.STEPS_PER_BETA):
            if time.monotonic() >= seed_deadline:
                last_incumbent = _incumbent_snapshot(
                    seed, current, len(history), phase, local_step, beta,
                    seed_started, "deadline_before_iteration",
                )
                iteration_callback(last_incumbent, history, checkpoints)
                return {"seed": seed, "status": "timeout", "initialization": init_record,
                        "steps_completed": len(history), "history": history,
                        "checkpoints": checkpoints, "selected": None,
                        "last_incumbent": last_incumbent}
            try:
                loss, gradient = critical_corner_softcount_value_gradient(
                    current, bases_subset, targets_subset, beta
                )
            except Exception as exc:
                last_incumbent = _incumbent_snapshot(
                    seed, current, len(history), phase, local_step, beta,
                    seed_started, "gradient_failure",
                    {"failure_type": type(exc).__name__, "message": str(exc)},
                )
                iteration_callback(last_incumbent, history, checkpoints)
                return {"seed": seed, "status": "gradient_failure",
                        "initialization": init_record, "steps_completed": len(history),
                        "history": history, "checkpoints": checkpoints,
                        "failure": {"type": type(exc).__name__, "message": str(exc)},
                        "selected": None, "last_incumbent": last_incumbent}
            if not math.isfinite(loss) or not np.isfinite(gradient).all():
                last_incumbent = _incumbent_snapshot(
                    seed, current, len(history), phase, local_step, beta,
                    seed_started, "numerical_failure",
                    {"loss_finite": bool(math.isfinite(loss)),
                     "gradient_finite": bool(np.isfinite(gradient).all())},
                )
                iteration_callback(last_incumbent, history, checkpoints)
                return {"seed": seed, "status": "numerical_failure",
                        "initialization": init_record, "steps_completed": len(history),
                        "history": history, "checkpoints": checkpoints,
                        "failure": {"type": "FloatingPointError",
                                    "message": "non-finite softcount objective or gradient"},
                        "selected": None, "last_incumbent": last_incumbent}
            try:
                lmo = _linprog_lmo(domain, gradient, seed_deadline)
            except Exception as exc:
                last_incumbent = _incumbent_snapshot(
                    seed, current, len(history), phase, local_step, beta,
                    seed_started, "lmo_exception",
                    {"failure_type": type(exc).__name__, "message": str(exc)},
                )
                iteration_callback(last_incumbent, history, checkpoints)
                return {"seed": seed, "status": "lmo_failure",
                        "initialization": init_record, "steps_completed": len(history),
                        "history": history, "checkpoints": checkpoints,
                        "failure": {"type": type(exc).__name__, "message": str(exc)},
                        "selected": None, "last_incumbent": last_incumbent}
            if lmo["status"] != "optimal_verified":
                failed_lmo = {k: v for k, v in lmo.items() if k != "weights"}
                last_incumbent = _incumbent_snapshot(
                    seed, current, len(history), phase, local_step, beta,
                    seed_started, "lmo_failure", failed_lmo,
                )
                iteration_callback(last_incumbent, history, checkpoints)
                return {"seed": seed, "status": lmo["status"],
                        "initialization": init_record, "steps_completed": len(history),
                        "history": history, "checkpoints": checkpoints,
                        "lmo_failure": failed_lmo,
                        "rejected_lmo_weights": lmo.get("rejected_x"),
                        "lmo_residual": lmo.get("residual"),
                        "selected": None, "last_incumbent": last_incumbent}
            vertex = lmo["weights"]
            gap = float(gradient @ (current - vertex))
            gap_class = coverage.classify_fw_gap(gap)
            if gap_class == "nonfinite_failure":
                last_incumbent = _incumbent_snapshot(
                    seed, current, len(history), phase, local_step, beta,
                    seed_started, "numerical_failure", {"gap_finite": bool(math.isfinite(gap))},
                )
                iteration_callback(last_incumbent, history, checkpoints)
                return {"seed": seed, "status": "numerical_failure",
                        "initialization": init_record, "steps_completed": len(history),
                        "history": history, "checkpoints": checkpoints,
                        "failure": {"type": "FloatingPointError", "message": "non-finite Frank-Wolfe gap"},
                        "selected": None, "last_incumbent": last_incumbent}
            if gap_class == "negative_gap_failure":
                last_incumbent = _incumbent_snapshot(
                    seed, current, len(history), phase, local_step, beta,
                    seed_started, "negative_gap_failure",
                    {"gap": gap, "lmo": lmo.get("record")},
                )
                iteration_callback(last_incumbent, history, checkpoints)
                return {"seed": seed, "status": "negative_gap_failure",
                        "initialization": init_record, "steps_completed": len(history),
                        "history": history, "checkpoints": checkpoints,
                        "failure": {"gap": gap, "lmo": lmo.get("record")},
                        "selected": None, "last_incumbent": last_incumbent}
            if gap_class == "stationary_tolerance":
                step_record = {"phase": phase, "beta": beta, "local_step": local_step,
                               "gap": gap, "status": "stationary_tolerance",
                               "lmo": lmo["record"]}
            else:
                try:
                    line = _armijo(current, vertex, loss, gap, bases_subset,
                                   targets_subset, beta, seed_deadline)
                except Exception as exc:
                    last_incumbent = _incumbent_snapshot(
                        seed, current, len(history), phase, local_step, beta,
                        seed_started, "line_search_failure",
                        {"failure_type": type(exc).__name__, "message": str(exc), "gap": gap},
                    )
                    iteration_callback(last_incumbent, history, checkpoints)
                    return {"seed": seed, "status": "line_search_failure",
                            "initialization": init_record, "steps_completed": len(history),
                            "history": history, "checkpoints": checkpoints,
                            "failure": {"type": type(exc).__name__, "message": str(exc), "gap": gap},
                            "selected": None, "last_incumbent": last_incumbent}
                if line.get("timed_out"):
                    last_incumbent = _incumbent_snapshot(
                        seed, current, len(history), phase, local_step, beta,
                        seed_started, "line_search_deadline",
                        {k: v for k, v in line.items() if k != "weights"},
                    )
                    iteration_callback(last_incumbent, history, checkpoints)
                    return {"seed": seed, "status": "timeout",
                            "initialization": init_record,
                            "steps_completed": len(history), "history": history,
                            "checkpoints": checkpoints, "selected": None,
                            "last_incumbent": last_incumbent}
                if not line.get("accepted"):
                    last_incumbent = _incumbent_snapshot(
                        seed, current, len(history), phase, local_step, beta,
                        seed_started, "line_search_no_progress",
                        {"gap": gap, "line_search_evaluations": line["evaluations"]},
                    )
                    iteration_callback(last_incumbent, history, checkpoints)
                    return {"seed": seed, "status": "no_progress",
                            "initialization": init_record,
                            "steps_completed": len(history), "history": history,
                            "checkpoints": checkpoints,
                            "failed_step": {"phase": phase, "beta": beta,
                                            "local_step": local_step, "gap": gap},
                            "selected": None, "last_incumbent": last_incumbent}
                current = np.asarray(line["weights"], dtype=np.float64)
                residual = domain.verify(current)
                if not residual["passed"]:
                    last_incumbent = _incumbent_snapshot(
                        seed, current, len(history), phase, local_step, beta,
                        seed_started, "accepted_iterate_domain_failure",
                        {"domain_residual": residual, "gap": gap},
                    )
                    iteration_callback(last_incumbent, history, checkpoints)
                    return {"seed": seed, "status": "constraint_failure",
                            "initialization": init_record, "steps_completed": len(history),
                            "history": history, "checkpoints": checkpoints,
                            "failure": {"domain_residual": residual, "gap": gap},
                            "selected": None, "last_incumbent": last_incumbent}
                step_record = {"phase": phase, "beta": beta, "local_step": local_step,
                               "gap": gap, "status": "accepted",
                               "line_search": {k: v for k, v in line.items() if k != "weights"},
                               "lmo": lmo["record"], "domain_residual": residual}
            step_record.update({
                "gradient": np.asarray(gradient, dtype=np.float64).tolist(),
                "lmo_vertex": np.asarray(vertex, dtype=np.float64).tolist(),
                "lmo_objective": float(lmo["objective_value"]),
                "weights_after": np.asarray(current, dtype=np.float64).tolist(),
                "weights_after_sha256": coverage.sha256_array(np.asarray(current, dtype="<f8")),
            })
            history.append(step_record)
            order += 1
            last_incumbent = _incumbent_snapshot(
                seed, current, len(history), phase, local_step, beta,
                seed_started, step_record["status"],
                {"gap": gap},
            )
            iteration_callback(last_incumbent, history, checkpoints)
            boundary = local_step == coverage.STEPS_PER_BETA - 1
            periodic = order % coverage.CHECKPOINT_INTERVAL == 0
            if periodic or boundary:
                poly = coverage.verify_original_nominal_polytope(
                    [basis64[row["layout_id"]] for row in old_rows],
                    [row["target"] for row in old_rows], current,
                )
                record = _checkpoint_metrics(
                    all_rows, old_rows, new_rows, basis32, current,
                    bases_subset, targets_subset, beta, anchor, ref, physical, order,
                )
                record = _fit_checkpoint_qualified(
                    record, new_reference_mean, poly, domain, current
                )
                record["phase_boundary"] = bool(boundary)
                record["weights_sha256"] = coverage.sha256_array(
                    np.asarray(current, dtype="<f8")
                )
                checkpoints.append(record)
                checkpoint_callback(seed, record, history, checkpoints, last_incumbent)
    if diagnostics_protocol is not None:
        final_diagnostic = run_diagnostic("final_beta_800", coverage.BETAS[-1])
        if final_diagnostic.get("status") != "complete":
            status = "timeout" if final_diagnostic.get("status") == "timeout" else "diagnostics_failure"
            return {"seed": seed, "status": status, "initialization": init_record,
                    "per_layout_diagnostics": diagnostics, "steps_completed": len(history),
                    "history": history, "checkpoints": checkpoints,
                    "selected": None, "last_incumbent": last_incumbent}
    qualified = [row for row in checkpoints if row.get("qualified")]
    selected = min(qualified, key=coverage.checkpoint_rank) if qualified else None
    result = {
        "seed": seed, "status": "complete" if len(history) == coverage.STEPS_PER_SEED else "incomplete",
        "initialization": init_record, "steps_completed": len(history),
        "history": history, "checkpoints": checkpoints,
        "last_incumbent": last_incumbent,
        "qualified_checkpoint_count": len(qualified), "selected": selected,
    }
    if diagnostics_protocol is not None:
        result["per_layout_diagnostics"] = diagnostics
    return result


def _calibration_rows_from_dataset(cal_dataset, diagnostic: dict) -> list[dict]:
    if (len(cal_dataset.layout_ids) != 4
            or tuple(cal_dataset.layout_ids) != coverage.CALIBRATION_LAYOUT_IDS
            or float(cal_dataset.pixel_size_nm) != 4.0
            or tuple(cal_dataset.targets.shape[-2:]) != (128, 128)
            or tuple(cal_dataset.masks.shape[-2:]) != (128, 128)):
        raise ValueError("reused calibration must match its four ordered 128x128 4 nm layouts")
    rows = [{
        "layout_id": layout_id,
        "mask": cal_dataset.masks[index, 0].detach().cpu().numpy().astype(np.float32, copy=True),
        "target": cal_dataset.targets[index, 0].detach().cpu().numpy().astype(np.float32, copy=True),
        "split": "calibration",
    } for index, layout_id in enumerate(cal_dataset.layout_ids)]
    coverage.validate_layout_rows(
        rows, coverage.CALIBRATION_LAYOUT_IDS,
        diagnostic["input"]["calibration_masks"],
        diagnostic["input"]["calibration_targets"], "calibration",
    )
    return rows


def _calibration_metrics(device, source_weights, physical: dict, diagnostic: dict):
    """Open the four-layout reused development generator only after FIT freeze."""
    from scripts import run_protected_pvband_experiment as original_protocol

    teacher, cal_dataset, cal_rows_raw = original_protocol.generate_calibration(device)
    del teacher
    rows = _calibration_rows_from_dataset(cal_dataset, diagnostic)
    expected_bases = {row["layout_id"]: row for row in diagnostic["input"]["bases"]}
    _sim, basis32, _basis64, _basis_t, _targets, parity = _prepare_basis(
        rows, device, physical, expected_bases
    )
    if len(parity) != 4:
        raise ValueError("direct simulator/basis parity must be checked for all four calibration layouts")
    per_seed = []
    for seed, weights in source_weights.items():
        metrics = _evaluate_rows(rows, basis32, weights, physical)
        per_seed.append({"seed": int(seed), **metrics})
    means = coverage.aggregate_metrics([row["mean"] for row in per_seed])
    gates = {
        "mean_band_pixels": means["band_pixels"] <= 239.4,
        "mean_nominal_l2_pixels": means["L2_pixels"] <= 56.175,
        "mean_worst_dose_l2_pixels": means["L2_worst_dose_pixels"] <= 157.2375,
        "every_seed_band_below_268": all(
            row["mean"]["band_pixels"] < 268.0 for row in per_seed
        ),
        "no_blank_positive_target_any_dose": all(
            row["no_blank_positive_target_any_dose"] for row in per_seed
        ),
        "all_five_complete_fit_qualified": len(per_seed) == 5,
    }
    return {"status": "scored_once_after_fit_freeze", "mean": means,
            "per_seed": per_seed, "gate": gates,
            "lineage": {"layout_ids": list(coverage.CALIBRATION_LAYOUT_IDS),
                        "basis_parity": parity},
            "passed": all(gates.values()),
            "selection_input": "none; calibration is gate-only and never a loss/tie input"}

def _seed17_reference_metrics(reference_metrics: dict, expected_rows: list[dict]) -> None:
    observed_rows = reference_metrics["per_layout"]
    if [row["layout_id"] for row in observed_rows] != list(coverage.ORIGINAL_FIT_LAYOUT_IDS):
        raise ValueError("float32 reference metric order differs from the frozen original FIT order")
    observed = {row["layout_id"]: row for row in observed_rows}
    for expected in expected_rows:
        row = observed[expected["layout_id"]]
        for key in ("band_pixels", "L2_pixels", "L2_worst_dose_pixels"):
            if int(row[key]) != expected[key]:
                raise ValueError("seed17 reference FIT metric mismatch for %s/%s" %
                                 (expected["layout_id"], key))
    mean = reference_metrics["mean"]
    if (mean["band_pixels"] > 234.5 or mean["L2_pixels"] != 0.0
            or mean["L2_worst_dose_pixels"] > 150.25):
        raise ValueError("pinned seed17 reference does not match immutable original FIT gates")


def _prepare_original_controls(plan: dict, plan_sha: str, args, identity: dict):
    """Run deterministic original-FIT checks before the one-use marker boundary."""
    identity = _assert_identity_unchanged(
        Path(args.plan_file).resolve(), plan_sha,
        args.expected_previous_sha256, plan, identity,
    )
    fit = _load_fit_only(identity["dataset_bytes"])
    if (float(fit.pixel_size_nm) != float(plan["physical"]["pixel_nm"])
            or tuple(fit.layout_ids) != coverage.ORIGINAL_FIT_LAYOUT_IDS
            or tuple(fit.masks.shape[-2:]) != (128, 128)
            or tuple(fit.targets.shape[-2:]) != (128, 128)):
        raise ValueError("original FIT IDs, order, pixel size, or raster differ from frozen contract")
    rows = _as_numpy_rows(fit)
    coverage.validate_layout_rows(
        rows, coverage.ORIGINAL_FIT_LAYOUT_IDS,
        identity["diagnostic"]["input"]["fit_masks"],
        identity["diagnostic"]["input"]["fit_targets"], "FIT",
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the registered source coverage run requires the CUDA optical backend")
    expected_bases = {row["layout_id"]: row for row in identity["diagnostic"]["input"]["bases"]}
    _sim, basis32, basis64, basis_torch, targets_torch, fresh_parity = _prepare_basis(
        rows, device, plan["physical"], expected_bases,
    )
    if len(fresh_parity) != 4:
        raise ValueError("fresh direct basis parity must be checked for all original FIT layouts")
    expected_inputs = {
        "dataset_sha256": plan["input_hashes"]["dataset_sha256"],
        "dataset_file_sha256": plan["input_hashes"]["dataset_sha256"],
        "diagnostic_file_sha256": plan["input_hashes"]["diagnostic_sha256"],
        "diagnostic_input": identity["diagnostic"]["input"],
        # This is the prior canonical identity, separate from fresh parity above.
        "basis_parity": identity["previous_weights"]["canonical_basis_parity"],
    }
    prerequisite = pareto.validate_prerequisite_manifest(
        identity["manifest_path"], coverage.PINNED_SOURCE_MANIFEST_SHA256, expected_inputs,
    )
    artifacts = pareto._merge_artifact_paths(prerequisite["artifact_paths"])
    if len(artifacts) != 14:
        raise ValueError("registered prerequisite lineage must contain exactly 14 verified artifacts")
    pinned_artifacts = identity["previous_report"].get("preflight", {}).get(
        "prerequisite_manifest", {}).get("artifact_paths")
    if pareto._merge_artifact_paths(pinned_artifacts or []) != artifacts:
        raise ValueError("validated prerequisite artifacts differ from pinned previous report")
    identity["lineage_artifacts"] = artifacts
    _check_lineage_artifacts(identity)
    identity = _assert_identity_unchanged(
        Path(args.plan_file).resolve(), plan_sha,
        args.expected_previous_sha256, plan, identity,
    )

    ids = [row["layout_id"] for row in rows]
    old_targets = [row["target"] for row in rows]
    bases64 = [basis64[key] for key in ids]
    anchor, reference = (identity["previous_weights"]["lp_anchor"],
                         identity["previous_weights"]["reference_weights"])
    original_domain, original_guard_counts = coverage.build_guarded_domain(
        bases64, old_targets, ids, [], [], [], anchor, reference,
    )
    anchor_poly = coverage.verify_original_nominal_polytope(bases64, old_targets, anchor)
    reference_poly = coverage.verify_original_nominal_polytope(bases64, old_targets, reference)
    if not anchor_poly["passed"] or not reference_poly["passed"]:
        raise ValueError("LP anchor or seed-17 reference fails the full original nominal polytope")
    if not original_domain.verify(reference)["passed"]:
        raise ValueError("seed-17 reference fails original-FIT warm-start guards")
    reference_fit = _evaluate_rows(rows, basis32, reference, plan["physical"])
    _seed17_reference_metrics(reference_fit, identity["previous_weights"]["fit_metrics_per_layout"])
    reference_audit = coverage.audit_float32_critical_guards(
        basis32, {row["layout_id"]: row["target"] for row in rows},
        anchor, reference, reference, ids, [],
    )
    if not reference_audit["passed"]:
        raise ValueError("seed-17 reference fails original FIT float32 critical-guard audit")
    return {
        "identity": identity, "fit": fit, "rows": rows, "device": device,
        "basis32": basis32, "basis64": basis64, "basis_torch": basis_torch,
        "targets_torch": targets_torch, "fresh_fit_parity": fresh_parity,
        "anchor": anchor, "reference": reference,
        "original_domain": original_domain,
        "original_guard_summary": original_guard_counts,
        "reference_fit": reference_fit, "reference_poly": reference_poly,
        "reference_audit": reference_audit,
    }


def _record_run_failure(report: dict, exc: Exception, calibration_opened: bool) -> None:
    report["status"] = "error"
    report["error"] = {"type": type(exc).__name__, "message": str(exc),
                       "traceback": traceback.format_exc()}
    report["calibration_status"] = (
        "opened_then_failed_no_retry" if calibration_opened else "closed"
    )
    report["coverage_attempt_consumed"] = bool(report.get("coverage_attempt_consumed"))


def _start_seed_clock(identity_recheck, monotonic=time.monotonic) -> tuple[float, float]:
    """Complete the boundary identity check before starting any solver budget."""
    checked_at = monotonic()
    identity_recheck()
    elapsed = max(0.0, monotonic() - checked_at)
    return monotonic(), elapsed


def run(args) -> Path:
    plan, plan_sha, identity = _preflight(args)
    output_root = _check_output_root(args.output_root)
    run_dir = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                             + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "coverage_report.json"
    schema6 = plan.get("schema_version") == 6
    report = {
        "schema_version": 2 if schema6 else 1, "objective_id": coverage.OBJECTIVE_ID,
        "status": "preflight_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan_sha, "previous_report_sha256": coverage.PINNED_PREVIOUS_REPORT_SHA256,
        "input_hashes": plan["input_hashes"],
        "previous_plan_sha256": coverage.PINNED_PREVIOUS_PLAN_SHA256,
        "source_manifest_sha256": coverage.PINNED_SOURCE_MANIFEST_SHA256,
        "attempt_marker": None, "coverage_attempt_consumed": False,
        "source_identity": identity["source_identity"],
        "calibration_status": "closed", "final3_status": "never indexed or evaluated",
        "seeds": [], "fit_frozen": False,
    }
    if schema6:
        report["superseded_coverage_attempt"] = plan["superseded_coverage_attempt"]
        report["initialization_protocol"] = plan["initialization_protocol"]
        report["diagnostics_protocol"] = plan["diagnostics_protocol"]
        report["fixed_fit_layout_hashes"] = plan["fixed_fit_layout_hashes"]
    _atomic_json(report_path, report)
    calibration_opened = False
    try:
        preseed_audit_started = time.monotonic()
        context = _prepare_original_controls(plan, plan_sha, args, identity)
        identity = context["identity"]
        report.update({
            "status": "original_fit_preseed_audit_passed",
            "preseed_original_audit_seconds": float(time.monotonic() - preseed_audit_started),
            "fresh_original_fit_basis_parity": context["fresh_fit_parity"],
            "historical_lineage_basis_parity": identity["previous_weights"]["canonical_basis_parity"],
            "lineage_artifact_count": len(identity["lineage_artifacts"]),
            "original_guard_summary": context["original_guard_summary"],
            "reference_original_fit": context["reference_fit"],
            "reference_original_polytope": context["reference_poly"],
            "reference_float32_guard_audit": context["reference_audit"],
        })
        _atomic_json(report_path, report)

        # Consume immediately before any new geometry, mask, target, basis, or score.
        attempt_marker = coverage.create_attempt_marker(plan["prerequisite_manifest"], plan_sha)
        report["attempt_marker"] = str(attempt_marker)
        report["coverage_attempt_consumed"] = True
        report["status"] = "coverage_attempt_consumed_before_new_fit_generation"
        _atomic_json(report_path, report)
        layouts = coverage.make_coverage_masks(plan["physical"]["raster"])
        new_rows = _generate_new_targets(layouts, context["device"], plan["physical"])
        novelty = _novelty_check(context["rows"], new_rows, plan["new_fit_generation"])
        if tuple(row["layout_id"] for row in new_rows) != coverage.LAYOUT_IDS:
            raise ValueError("new FIT layout IDs differ from the frozen plan")
        if schema6:
            actual_fixed_hashes = [
                {key: row[key] for key in ("layout_id", "mask_sha256", "target_sha256")}
                for row in novelty
            ]
            if actual_fixed_hashes != plan["fixed_fit_layout_hashes"]:
                raise ValueError("generated fixed FIT masks/targets differ from consumed-attempt hashes")
        _sim, new_basis32, new_basis64, new_basis_torch, new_targets_torch, new_parity = _prepare_basis(
            new_rows, context["device"], plan["physical"],
        )
        all_rows = context["rows"] + new_rows
        basis32, basis64 = dict(context["basis32"]), dict(context["basis64"])
        basis_torch, targets_torch = dict(context["basis_torch"]), dict(context["targets_torch"])
        basis32.update(new_basis32); basis64.update(new_basis64)
        basis_torch.update(new_basis_torch); targets_torch.update(new_targets_torch)
        identity = _assert_identity_unchanged(
            Path(args.plan_file).resolve(), plan_sha,
            args.expected_previous_sha256, plan, identity,
        )

        original_ids = [row["layout_id"] for row in context["rows"]]
        new_ids = [row["layout_id"] for row in new_rows]
        all_targets = [row["target"] for row in all_rows]
        previous = identity["previous_weights"]
        anchor, reference = context["anchor"], context["reference"]
        domain, guard_summary = coverage.build_guarded_domain(
            [basis64[key] for key in original_ids],
            [row["target"] for row in context["rows"]], original_ids,
            [basis64[key] for key in new_ids],
            [row["target"] for row in new_rows], new_ids, anchor, reference,
        )
        reference_poly = coverage.verify_original_nominal_polytope(
            [basis64[key] for key in original_ids],
            [row["target"] for row in context["rows"]], reference,
        )
        if not reference_poly["passed"] or not domain.verify(reference)["passed"]:
            raise ValueError("seed-17 reference fails original polytope or augmented warm-start audit")
        ref_old = context["reference_fit"]
        ref_new = _evaluate_rows(new_rows, basis32, reference, plan["physical"])
        reference_guard_audit = coverage.audit_float32_critical_guards(
            basis32, {row["layout_id"]: row["target"] for row in all_rows},
            anchor, reference, reference, original_ids, new_ids,
        )
        if not reference_guard_audit["passed"]:
            raise ValueError("seed-17 reference fails the augmented float32 critical-guard audit")
        report.update({
            "status": "fit_running", "new_fit_layouts": novelty,
            "new_fit_direct_basis_parity": new_parity,
            "guard_summary": guard_summary,
            "reference_new_fit": ref_new,
            "reference_augmented_polytope": reference_poly,
            "reference_guard_check": domain.verify(reference),
            "reference_augmented_float32_guard_audit": reference_guard_audit,
            "basis_scope": "original four FIT and three new FIT only; no non-FIT split indexed",
        })
        report["pre_solver_setup_seconds"] = float(time.monotonic() - preseed_audit_started)
        _atomic_json(report_path, report)
        # The registered 1,800-second solver budget starts after all preseed audits.
        solver_budget_started = time.monotonic()
        report["solver_budget_started_after_preseed_audit"] = True
        report["solver_budget_start_monotonic"] = solver_budget_started
        _atomic_json(report_path, report)

        selected_by_seed = {}
        accepted_starts = []
        for seed in coverage.SEEDS:
            total_spent = float(sum(row.get("wall_seconds", 0.0) for row in report["seeds"]))
            total_remaining = coverage.TOTAL_LIMIT_SECONDS - total_spent
            if total_remaining <= 0:
                report["status"] = "timeout"
                break
            seed_budget = min(coverage.PER_SEED_LIMIT_SECONDS, total_remaining)
            seed_started, identity_seconds = _start_seed_clock(
                lambda: _assert_identity_unchanged(
                    Path(args.plan_file).resolve(), plan_sha,
                    args.expected_previous_sha256, plan, identity,
                )
            )
            report.setdefault("pre_seed_identity_rechecks", []).append({
                "seed": int(seed), "wall_seconds": identity_seconds,
                "excluded_from_solver_budget": True,
            })
            seed_deadline = seed_started + seed_budget

            def checkpoint_callback(seed_value, checkpoint, history, checkpoints, current_incumbent):
                progress_path = run_dir / ("seed_%03d_progress.json" % int(seed_value))
                progress = {"seed": int(seed_value), "history": list(history),
                            "checkpoints": list(checkpoints),
                            "current_incumbent": current_incumbent,
                            "last_checkpoint": checkpoint}
                _atomic_json(progress_path, progress)
                report["current_checkpoints"] = progress["checkpoints"]
                report["current_incumbent"] = current_incumbent
                report["active_seed"] = int(seed_value)
                report["active_progress_path"] = str(progress_path)
                report["last_checkpoint"] = {
                    "seed": int(seed_value), "order": checkpoint["checkpoint_order"],
                    "beta": checkpoint["beta"], "weights": checkpoint["weights"],
                    "weights_sha256": checkpoint["weights_sha256"],
                    "qualified": checkpoint["qualified"],
                }
                report["solver_time_spent_seconds"] = float(
                    sum(row.get("wall_seconds", 0.0) for row in report["seeds"])
                    + max(0.0, time.monotonic() - seed_started)
                )
                _atomic_json(report_path, report)

            def iteration_callback(snapshot, history, checkpoints):
                progress_path = run_dir / ("seed_%03d_progress.json" % int(seed))
                _atomic_json(progress_path, {
                    "seed": int(seed), "history": list(history),
                    "checkpoints": list(checkpoints), "current_incumbent": snapshot,
                })
                report["active_seed"] = int(seed)
                report["active_step"] = int(snapshot["steps_completed"])
                report["current_incumbent"] = snapshot
                report["active_progress_path"] = str(progress_path)
                report["solver_time_spent_seconds"] = float(
                    sum(row.get("wall_seconds", 0.0) for row in report["seeds"])
                    + max(0.0, time.monotonic() - seed_started)
                )
                _atomic_json(report_path, report)

            progress_path = run_dir / ("seed_%03d_progress.json" % int(seed))
            report["current_incumbent"] = None
            report["current_checkpoints"] = []
            report["active_seed"] = int(seed)
            report["active_step"] = 0
            report["active_progress_path"] = str(progress_path)
            try:
                seed_options = {}
                if schema6:
                    seed_options = {
                        "initialization_protocol": plan["initialization_protocol"],
                        "diagnostics_protocol": plan["diagnostics_protocol"],
                        "prior_starts": accepted_starts,
                    }
                result = _run_seed(
                    seed, reference, anchor, domain, all_rows, context["rows"], new_rows,
                    basis32, basis64, basis_torch, targets_torch, plan["physical"],
                    ref_new["mean"], seed_started, seed_deadline,
                    checkpoint_callback, iteration_callback,
                    **seed_options,
                )
            except Exception as seed_exc:
                seed_traceback = traceback.format_exc()
                partial_progress = {}
                try:
                    partial_progress = json.loads(progress_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    partial_progress = {}
                if partial_progress.get("seed") != int(seed):
                    partial_progress = {}
                history = partial_progress.get("history", [])
                checkpoints = partial_progress.get("checkpoints", [])
                incumbent = partial_progress.get("current_incumbent")
                if not isinstance(incumbent, dict) or incumbent.get("seed") != int(seed):
                    incumbent = report.get("current_incumbent")
                if not isinstance(incumbent, dict) or incumbent.get("seed") != int(seed):
                    incumbent = None
                incumbent_details = (incumbent or {}).get("details", {})
                partial_initialization = incumbent_details.get("initialization")
                partial_diagnostics = incumbent_details.get("per_layout_diagnostics", [])
                seed_failure_record = {
                    "seed": int(seed), "status": "solver_exception",
                    "steps_completed": int((incumbent or {}).get("steps_completed", len(history))),
                    "history": history, "checkpoints": checkpoints,
                    "last_incumbent": incumbent, "selected": None,
                    "wall_seconds": float(max(0.0, time.monotonic() - seed_started)),
                    "pre_seed_identity_recheck_seconds": float(identity_seconds),
                    "failure": {"type": type(seed_exc).__name__,
                                "message": str(seed_exc), "traceback": seed_traceback},
                }
                if schema6:
                    seed_failure_record["initialization"] = partial_initialization
                    seed_failure_record["per_layout_diagnostics"] = partial_diagnostics
                report["seeds"].append(seed_failure_record)
                report["active_seed"] = None
                report["active_step"] = None
                report["active_progress_path"] = str(progress_path) if progress_path.exists() else None
                report["solver_time_spent_seconds"] = float(
                    sum(row.get("wall_seconds", 0.0) for row in report["seeds"])
                )
                report["status"] = "solver_exception"
                _atomic_json(report_path, report)
                raise
            initialization = result.get("initialization")
            if (schema6 and isinstance(initialization, dict)
                    and initialization.get("status") == "feasible_start_found"):
                accepted_starts.append(np.asarray(initialization["weights"], dtype=np.float64))
                result["accepted_start_index"] = len(accepted_starts)
                result["accepted_start_hashes_in_order"] = [
                    coverage.sha256_array(np.asarray(value, dtype="<f8"))
                    for value in accepted_starts
                ]
            elapsed = time.monotonic() - seed_started
            if elapsed > seed_budget:
                result["status"] = "timeout"
                result["selected"] = None
                result["deadline_exceeded_during_solver_and_snapshots"] = True
            result["wall_seconds"] = float(elapsed)
            result["pre_seed_identity_recheck_seconds"] = float(identity_seconds)
            report["solver_time_spent_seconds"] = float(total_spent + elapsed)
            report["seeds"].append(result)
            report["active_seed"] = None
            report["active_step"] = None
            report["active_progress_path"] = None
            report["current_incumbent"] = result.get("last_incumbent")
            _atomic_json(report_path, report)
            identity_started = time.monotonic()
            identity = _assert_identity_unchanged(
                Path(args.plan_file).resolve(), plan_sha,
                args.expected_previous_sha256, plan, identity,
            )
            report["seeds"][-1]["identity_recheck_seconds"] = float(time.monotonic() - identity_started)
            _atomic_json(report_path, report)
            if result["status"] != "complete":
                report["status"] = result["status"]
                break
            if result.get("selected") is not None:
                selected_by_seed[int(seed)] = np.asarray(result["selected"]["weights"], dtype=np.float64)
            report["status"] = "fit_running"
            _atomic_json(report_path, report)

        if (len(report["seeds"]) != len(coverage.SEEDS)
                or len(selected_by_seed) != len(coverage.SEEDS)
                or any(row.get("status") != "complete" for row in report["seeds"])):
            if len(report["seeds"]) == len(coverage.SEEDS):
                report["status"] = "no_fit_qualified_checkpoint"
            report["status"] = report.get("status", "fit_incomplete")
            report["calibration_status"] = "closed"
            _atomic_json(report_path, report)
            return report_path

        fit_freeze_check = coverage.fit_freeze_eligibility(report["seeds"])
        if fit_freeze_check["passed"]:
            frozen_audits = []
            target_map = {row["layout_id"]: row["target"] for row in all_rows}
            for row in report["seeds"]:
                audit = coverage.audit_float32_critical_guards(
                    basis32, target_map, anchor, reference,
                    row["selected"]["weights"], original_ids, new_ids,
                )
                frozen_audits.append({"seed": row["seed"], **audit})
            report["pre_calibration_float32_guard_audits"] = frozen_audits
            if any(not row["passed"] for row in frozen_audits):
                fit_freeze_check = {"passed": False,
                                    "reason": "pre_calibration_float32_guard_audit_failed"}
        if not fit_freeze_check["passed"]:
            report["status"] = "fit_freeze_ineligible"
            report["fit_freeze_check"] = fit_freeze_check
            report["calibration_status"] = "closed"
            _atomic_json(report_path, report)
            return report_path

        fit_selection = []
        for row in report["seeds"]:
            selected = row["selected"]
            fit_selection.append({
                "seed": row["seed"], "checkpoint_order": selected["checkpoint_order"],
                "weights": selected["weights"], "weights_sha256": selected["weights_sha256"],
                "fit_rank": list(coverage.checkpoint_rank(selected)),
            })
        report["fit_selection"] = fit_selection
        best_fit = min(fit_selection, key=lambda row: tuple(row["fit_rank"]))
        report["selected_fit_candidate"] = {
            "seed": best_fit["seed"], "checkpoint_order": best_fit["checkpoint_order"],
            "weights": best_fit["weights"], "weights_sha256": best_fit["weights_sha256"],
            "fit_rank": best_fit["fit_rank"], "selection_input": "FIT-only registered rank; calibration excluded",
        }
        report["fit_frozen"] = True
        report["fit_freeze_check"] = fit_freeze_check
        report["fit_frozen_sha256"] = coverage.sha256_bytes(
            json.dumps(fit_selection, sort_keys=True, separators=(",", ":")).encode()
        )
        report["status"] = "fit_frozen"
        _atomic_json(report_path, report)
        identity = _assert_identity_unchanged(
            Path(args.plan_file).resolve(), plan_sha,
            args.expected_previous_sha256, plan, identity,
        )

        report["calibration_status"] = "opened_once_in_progress"
        report["calibration_attempt_consumed"] = True
        _atomic_json(report_path, report)
        calibration_opened = True
        report["calibration"] = _calibration_metrics(
            context["device"], selected_by_seed, plan["physical"], identity["diagnostic"]
        )
        identity = _assert_identity_unchanged(
            Path(args.plan_file).resolve(), plan_sha,
            args.expected_previous_sha256, plan, identity,
        )
        report["calibration_status"] = "scored_once_after_fit_freeze"
        report["status"] = "complete" if report["calibration"]["passed"] else "complete_calibration_gate_failed"
        report["calibration_gate_passed"] = bool(report["calibration"]["passed"])
        report["runtime_wall_seconds"] = float(time.time() - args.started_at)
        _atomic_json(report_path, report)
        return report_path
    except Exception as exc:
        _record_run_failure(report, exc, calibration_opened)
        _atomic_json(report_path, report)
        raise

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-previous-sha256", required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "preflight":
            plan, plan_sha, identities = _preflight(args)
            print(json.dumps({
                "status": "preflight_passed_no_dataset_deserialized_no_optics_prepared",
                "plan_sha256": plan_sha,
                "previous_report_sha256": coverage.PINNED_PREVIOUS_REPORT_SHA256,
                "source_identity": identities["source_identity"],
            }, indent=2))
            return 0
        if not args.output_root:
            raise ValueError("--output-root is required for --mode run")
        args.started_at = time.time()
        path = run(args)
        print(json.dumps({"status": "finished", "report_path": str(path)}, indent=2))
        return 0
    except Exception as exc:
        print("source coverage failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
