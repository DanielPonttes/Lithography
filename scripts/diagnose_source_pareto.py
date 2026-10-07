"""Run the preregistered source-only Pareto protected-corner MILP.

This command verifies the completed hinge/softcount family and the frozen
inputs before solving. Calibration is opened only after all five FIT MILPs are
optimal, zero-gap, and pass the original float32 hard-print qualification.
"""
from __future__ import annotations

import argparse
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
import torch

from source_pareto import (
    EPSILON, HIGH_DOSE, LP_MARGIN, OBJECTIVE_ID, RHO, SEEDS,
    TOTAL_SOLVER_LIMIT_SECONDS, PER_SEED_LIMIT_SECONDS,
    PINNED_PLAN_SHA256, PINNED_SOURCE_MANIFEST_SHA256,
    build_sparse_milp, critical_signed_margins, five_seed_eligibility, hash_sparse_model,
    pinned_solver_backend, solve_highs,
)
from scripts import optimize_source_constrained as constrained
from scripts import optimize_source_robust_corners as robust
from scripts import run_protected_pvband_experiment as experiment

EXPECTED_GATES = {
    "mean_band_pixels_max": 239.4,
    "mean_nominal_l2_pixels_max": 56.175,
    "mean_worst_dose_l2_pixels_max": 157.2375,
    "each_seed_band_pixels_strictly_less_than": 268,
    "no_blank_positive_target_any_dose": True,
    "all_five_complete_fit_qualified": True,
}
EXPECTED_SCOPE = {
    "source_only": True,
    "fixed_masks": True,
    "number_of_source_weights": 49,
    "calibration_role": "reused development data, never optimizer or tie-break input",
    "final3_access": "never indexed or evaluated",
}
EXPECTED_PROTOCOL = {
    "epsilon": EPSILON,
    "rho": RHO,
    "seeds": list(SEEDS),
    "per_seed_time_limit_seconds": PER_SEED_LIMIT_SECONDS,
    "total_solver_time_limit_seconds": TOTAL_SOLVER_LIMIT_SECONDS,
    "final3_status": "closed; not indexed or evaluated",
    "gates": EXPECTED_GATES,
}
EXPECTED_TEXT = {
    "solver_eligibility": (
        "all five optimal zero-gap solutions and original-float32 fit qualification "
        "before calibration; time_limit incumbent is diagnostic incomplete, "
        "calibration closed; any numerical violation closes calibration"
    ),
    "objective_interpretation": "epsilon-buffered unrepaired anchor-error count, not exact hard-PV optimum",
    "no_new_corner_errors": "all originally correct critical FIT pixels remain correct in original float32 sigmoid-50 hard print",
    "candidate_selection": "FIT only; no calibration tie break",
    "retries": "no budget reset or silent retry; every attempt preserved; new attempt requires new prospective plan",
}
SEED_RHOS = (0.5, 0.05, 0.01)
SEED_IDS = (17, 29, 43, 71, 101)


def _validate_runtime_gate_constants():
    runtime = {
        "mean_band_pixels_max": constrained.GATE["band_pixels"],
        "mean_nominal_l2_pixels_max": constrained.GATE["L2_pixels"],
        "mean_worst_dose_l2_pixels_max": constrained.GATE["worst_L2_pixels"],
        "each_seed_band_pixels_strictly_less_than": constrained.LP_CAL["band_pixels"],
        "no_blank_positive_target_any_dose": True,
        "all_five_complete_fit_qualified": True,
    }
    if runtime != EXPECTED_GATES:
        raise RuntimeError("runtime FIT/development gates differ from the frozen Pareto plan")
    return runtime


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("could not read registered JSON: %s" % path) from exc


def _read_hashed_json(path):
    """Parse exactly the byte sequence whose SHA256 is recorded."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("could not parse registered JSON: %s" % path) from exc
    return parsed, digest, raw


def attempt_marker_path(plan_path, plan_sha256):
    """Return the one-use marker location tied to these frozen plan bytes."""
    return Path(plan_path).resolve().parent / ("source_pareto_attempt_%s.json" % plan_sha256)


def ensure_attempt_available(plan_path, plan_sha256):
    marker = attempt_marker_path(plan_path, plan_sha256)
    if marker.exists():
        raise RuntimeError("solver attempt already consumed for this candidate-plan SHA256")
    return marker


def claim_solver_attempt(plan_path, plan_sha256):
    """Atomically consume this plan's sole solver budget before solver entry."""
    marker = ensure_attempt_available(plan_path, plan_sha256)
    payload = (json.dumps({
        "schema_version": 1,
        "status": "attempt_consumed_before_first_solver_call",
        "candidate_plan_sha256": plan_sha256,
        "candidate_plan_path": str(Path(plan_path).resolve()),
        "total_solver_time_limit_seconds": TOTAL_SOLVER_LIMIT_SECONDS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(str(marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # An incomplete exclusive marker is deliberately retained fail-closed.
        raise
    return {"path": str(marker), "sha256": hashlib.sha256(payload).hexdigest()}


def _require_absolute_file(value, label):
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("%s must name an existing absolute file" % label)
    return path.resolve()


def validate_candidate_plan(plan, plan_path, dataset_path, diagnostic_path, *, plan_bytes=None):
    """Validate the exact frozen plan; CLI values cannot override its protocol."""
    if not isinstance(plan, dict) or plan.get("schema_version") != 4:
        raise ValueError("candidate plan must use schema version 4")
    if plan.get("status") != "prospective_candidate_plan":
        raise ValueError("candidate plan is not prospectively frozen")
    if plan.get("objective") != OBJECTIVE_ID:
        raise ValueError("candidate plan objective differs from this implementation")
    if plan.get("base_commit") != "a71d3ed465464b173231f5be2023250e489df809":
        raise ValueError("candidate plan base commit differs from the registered baseline")
    if Path(plan.get("dataset", "")).resolve() != dataset_path:
        raise ValueError("dataset path differs from the frozen candidate plan")
    if Path(plan.get("diagnostic", "")).resolve() != diagnostic_path:
        raise ValueError("diagnostic path differs from the frozen candidate plan")
    prereq = plan.get("prerequisite_manifest")
    if not isinstance(prereq, str) or not Path(prereq).is_absolute():
        raise ValueError("prerequisite manifest path must be absolute")
    if Path(prereq).resolve().parent != Path(plan_path).resolve().parent:
        raise ValueError("candidate plan and prerequisite manifest must share a directory")
    if plan.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("candidate plan changed a fixed solver, seed, margin, or gate")
    if plan.get("scope") != EXPECTED_SCOPE:
        raise ValueError("candidate plan changed the registered source-only scope")
    for key, expected in EXPECTED_TEXT.items():
        if plan.get(key) != expected:
            raise ValueError("candidate plan changed registered interpretation field %s" % key)
    raw = Path(plan_path).read_bytes() if plan_bytes is None else bytes(plan_bytes)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("candidate plan bytes are not valid UTF-8 JSON") from exc
    if parsed != plan:
        raise ValueError("parsed candidate plan differs from the exact hashed plan bytes")
    plan_sha = hashlib.sha256(raw).hexdigest()
    if plan_sha != PINNED_PLAN_SHA256:
        raise ValueError("candidate plan SHA256 differs from the frozen plan")
    return {"path": str(plan_path), "sha256": plan_sha,
            "base_commit": plan["base_commit"]}


def git_provenance(candidate_plan):
    """Require a clean committed source tree descended from the frozen base."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              check=True, capture_output=True, text=True,
                              timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                                cwd=ROOT, check=True, capture_output=True,
                                text=True, timeout=10).stdout
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor",
                                   candidate_plan["base_commit"], "HEAD"],
                                  cwd=ROOT, capture_output=True, text=True,
                                  timeout=10)
    except Exception as exc:
        raise RuntimeError("could not validate Git provenance: %s" % exc) from exc
    if status.strip():
        raise RuntimeError("production MILP requires a clean committed source tree")
    if ancestor.returncode != 0:
        raise RuntimeError("candidate plan base_commit is not an ancestor of current HEAD")
    source_paths = (
        Path(__file__), Path(__import__("source_pareto").__file__),
        Path(robust.__file__), Path(constrained.__file__),
        Path(constrained.diagnostic.__file__), Path(experiment.__file__),
        Path(constrained.light_source_module.__file__),
        Path(constrained.source_training_module.__file__),
    )
    hashes = {}
    for path in source_paths:
        try:
            key = str(path.resolve().relative_to(ROOT.resolve()))
        except ValueError as exc:
            raise RuntimeError("source provenance file is outside the checkout") from exc
        hashes[key] = sha256_file(path)
    return {"head": head, "clean_tree": True,
            "candidate_plan_base_commit": candidate_plan["base_commit"],
            "source_sha256": hashes,
            "runtime": {"python": sys.version.split()[0],
                        "numpy": np.__version__,
                        "scipy": __import__("scipy").__version__,
                        "torch": str(torch.__version__),
                        "torch_cuda": torch.version.cuda}}


def _merge_artifact_paths(artifact_paths):
    """Canonicalize artifact identities and reject conflicting duplicate paths."""
    merged = {}
    for item in artifact_paths:
        path = str(Path(item["path"]).resolve())
        digest = item.get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest.lower())):
            raise ValueError("prerequisite artifact has an invalid SHA256")
        digest = digest.lower()
        previous = merged.get(path)
        if previous is not None and previous != digest:
            raise ValueError("duplicate prerequisite artifact path has conflicting SHA256")
        merged[path] = digest
    return [{"path": path, "sha256": merged[path]} for path in sorted(merged)]


def _lineage_artifact_paths(lineage):
    """Hash and parse protocol/state bytes for the listed attempt and all ancestors."""
    artifacts = []
    for summary in lineage:
        run_dir = Path(summary["path"]).resolve()
        for filename, key in (("protocol.json", "protocol_sha256"),
                              ("run_state.json", "run_state_sha256")):
            path = run_dir / filename
            _parsed, digest, _raw = _read_hashed_json(path)
            if digest != summary.get(key):
                raise ValueError("source-family lineage artifact changed during validation")
            artifacts.append({"path": str(path), "sha256": digest})
    return _merge_artifact_paths(artifacts)


def _identity_snapshot(plan_path, plan_bytes, plan, dataset_path,
                       dataset_file_sha256_initial, diagnostic_path,
                       diagnostic_object, diagnostic_sha256, provenance, prerequisite):
    artifacts = _merge_artifact_paths(prerequisite["artifact_paths"])
    dataset_sha256 = sha256_file(dataset_path)
    diagnostic_dataset_sha256 = diagnostic_object.get("input", {}).get("dataset_sha256")
    if (dataset_sha256 != dataset_file_sha256_initial
            or dataset_sha256 != diagnostic_dataset_sha256):
        raise RuntimeError("dataset changed after its registered preflight SHA256")
    parsed_plan, parsed_plan_sha, parsed_plan_bytes = _read_hashed_json(plan_path)
    parsed_diagnostic, parsed_diagnostic_sha, _diagnostic_bytes = _read_hashed_json(
        diagnostic_path
    )
    if (parsed_plan_bytes != plan_bytes or parsed_plan_sha != hashlib.sha256(plan_bytes).hexdigest()
            or parsed_plan != plan or parsed_diagnostic_sha != diagnostic_sha256
            or parsed_diagnostic != diagnostic_object):
        raise RuntimeError("registered JSON changed while capturing the identity snapshot")
    json_hashes = {str(Path(plan_path).resolve()): parsed_plan_sha,
                   str(Path(diagnostic_path).resolve()): parsed_diagnostic_sha}
    json_hashes.update({item["path"]: item["sha256"] for item in artifacts})
    return {
        "candidate_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "dataset_file_sha256": dataset_file_sha256_initial,
        "diagnostic_file_sha256": sha256_file(diagnostic_path),
        "prerequisite_artifacts": {
            item["path"]: item["sha256"] for item in artifacts
        },
        "json_sha256": json_hashes,
        "git": provenance,
    }


def _recheck_identity(snapshot, plan_path, plan, dataset_path, diagnostic_path,
                      prerequisite, stage):
    """Re-read exact identities around each expensive/late stage."""
    try:
        artifact_paths = _merge_artifact_paths(prerequisite["artifact_paths"])
        parsed_hashes = {}
        parsed_json = {}
        for raw_path in snapshot["json_sha256"]:
            parsed, digest, _raw = _read_hashed_json(raw_path)
            parsed_hashes[raw_path] = digest
            parsed_json[raw_path] = parsed
        plan_key = str(Path(plan_path).resolve())
        if parsed_json[plan_key] != plan:
            raise ValueError("candidate plan parsed content changed")
        current = {
            "candidate_plan_sha256": parsed_hashes[plan_key],
            "dataset_file_sha256": sha256_file(dataset_path),
            "diagnostic_file_sha256": parsed_hashes[str(Path(diagnostic_path).resolve())],
            "prerequisite_artifacts": {
                item["path"]: parsed_hashes[
                    item["path"]
                ] for item in artifact_paths
            },
            "json_sha256": parsed_hashes,
            "git": git_provenance(plan),
        }
    except Exception as exc:
        raise RuntimeError("registered input or source identity drifted %s" % stage) from exc
    if current != snapshot:
        raise RuntimeError("registered input or source identity drifted %s" % stage)


def _validate_model_partition(rows, partition, model):
    """Require exact float32/float64 pixel maps and all registered counts."""
    expected_errors = tuple(
        (row["layout_id"], int(pixel))
        for row in rows
        for pixel in np.flatnonzero(partition["errors"][row["layout_id"]].reshape(-1))
    )
    expected_protected = {
        (row["layout_id"], int(pixel))
        for row in rows
        for pixel in np.flatnonzero(partition["protected"][row["layout_id"]].reshape(-1))
    }
    actual_protected = set(model.protected_pixel_indices)
    pruned_simplex = set(model.protected_pruned_simplex_pixel_indices)
    pruned_nominal = set(model.protected_pruned_nominal_pixel_indices)
    all_classified_protected = actual_protected | pruned_simplex | pruned_nominal
    all_expected = set(expected_protected)
    expected_nominal = expected_errors
    if (model.binary_count != 1331
            or model.binary_pixel_indices != expected_errors
            or len(set(model.binary_pixel_indices)) != model.binary_count
            or model.nominal_pixel_indices != expected_nominal
            or model.nominal_row_count != model.binary_count
            or model.nominal_original_row_count != 65536
            or model.nominal_pruned_protected_count != 64205
            or partition["anchor_error_pixels"] != len(expected_errors)
            or model.protected_count + model.protected_pruned_simplex
            + model.protected_pruned_nominal != 64205
            or len(expected_protected) != 64205
            or all_classified_protected != all_expected
            or len(actual_protected) != model.protected_count
            or len(pruned_simplex) != model.protected_pruned_simplex
            or len(pruned_nominal) != model.protected_pruned_nominal
            or (actual_protected & pruned_simplex)
            or (actual_protected & pruned_nominal)
            or (pruned_simplex & pruned_nominal)
            or partition["float64_float32_classification_conflicts"] != 0):
        raise ValueError("MILP float64 rows do not map exactly to the registered float32 FIT partition")
    return {"binary_error_pixels": len(expected_errors),
            "originally_correct_pixels": len(expected_protected),
            "nominal_rows_original": model.nominal_original_row_count,
            "nominal_rows_pruned_by_protected_corner_proof": model.nominal_pruned_protected_count,
            "nominal_rows_kept_for_anchor_errors": model.nominal_row_count,
            "protected_rows_kept": model.protected_count,
            "protected_rows_pruned": model.protected_pruned_simplex + model.protected_pruned_nominal,
            "error_pixel_mapping_matches_float32": True,
            "protected_pixel_mapping_certified": True,
            "complete_partition_mapping_matches_float32": True}


def _audit_full_incumbent(model, solver_row):
    """Recompute feasibility, integrality, and objective from the full x+z vector."""
    values = np.asarray(solver_row.get("values", ()), dtype=np.float64).reshape(-1)
    if values.size != model.objective.size or not np.isfinite(values).all():
        return {"passed": False, "reason": "incumbent vector missing, malformed, or nonfinite"}
    residual = __import__("source_pareto").check_model_residual(model, values, tolerance=1e-9)
    z = values[model.source_count:]
    integral = bool(np.all(np.abs(z - np.rint(z)) <= 1e-9))
    bounds = bool(np.all(z >= -1e-9) and np.all(z <= 1.0 + 1e-9))
    objective = float(np.dot(model.objective, values))
    reported = solver_row.get("objective")
    objective_match = (reported is not None and math.isfinite(float(reported))
                       and abs(objective - float(reported)) <= 1e-9)
    checks = {"sparse_model_residual": residual["passed"], "integrality": integral,
              "binary_bounds": bounds, "objective_matches_solver": objective_match,
              "objective_matches_binary_count": abs(objective - float(np.rint(z).sum())) <= 1e-9}
    little_endian = np.asarray(values, dtype="<f8")
    return {"passed": all(checks.values()), "checks": checks,
            "residual": residual, "objective_recomputed": objective,
            "binary_count_recomputed": int(np.rint(z).sum()),
            "maximum_integrality_violation": float(np.max(np.abs(z - np.rint(z)), initial=0.0)),
            "vector_sha256_float64_le": hashlib.sha256(little_endian.tobytes()).hexdigest(),
            "vector_length": int(values.size), "source_count": int(model.source_count),
            "binary_count": int(model.binary_count)}


def _expected_gate(family):
    _validate_runtime_gate_constants()
    if family == "hinge":
        return {
            "mean_band_pixels_max": constrained.GATE["band_pixels"],
            "mean_nominal_l2_pixels_max": constrained.GATE["L2_pixels"],
            "mean_worst_dose_l2_pixels_max": constrained.GATE["worst_L2_pixels"],
            "each_seed_band_pixels_strictly_less_than": constrained.LP_CAL["band_pixels"],
            "no_blank_positive_target_any_dose": True,
        }
    return {
        "mean_band_pixels_max": constrained.GATE["band_pixels"],
        "mean_nominal_l2_pixels_max": constrained.GATE["L2_pixels"],
        "mean_worst_dose_l2_pixels_max": constrained.GATE["worst_L2_pixels"],
        "each_seed_band_pixels_strictly_less_than": constrained.LP_CAL["band_pixels"],
        "no_blank_positive_target_any_dose": True,
    }


def _validate_recorded_gate(results):
    calibration = results.get("calibration", {})
    if (calibration.get("status") != "measured_development_only"
            or calibration.get("calibration_gradients") is not False
            or calibration.get("calibration_role")
            != "development only; not independent generalization evidence"):
        raise ValueError("source-family calibration was not a closed development-only gate")
    records = calibration.get("per_seed", [])
    if [row.get("seed") for row in records] != list(SEED_IDS):
        raise ValueError("source-family gate records must contain five ordered solver seeds")
    recomputed = constrained._aggregate_cal(records)
    if recomputed != calibration.get("aggregate"):
        raise ValueError("source-family calibration aggregate does not match per-seed metrics")
    no_blank = all(constrained._no_blank_print(row["calibration"]) for row in records)
    if calibration.get("no_positive_target_blank_any_dose") is not no_blank:
        raise ValueError("source-family no-blank summary does not match per-layout hard prints")
    expected_gate = constrained._gate(
        recomputed, calibration.get("lp_anchor_control", {}), no_blank,
        complete_seeds=True,
    )
    if expected_gate != calibration.get("gate"):
        raise ValueError("source-family frozen gate checks do not match the recorded metrics")
    if expected_gate.get("passed") is not False:
        raise ValueError("prerequisite gate unexpectedly passed")


def _validate_family_run(entry, family, index, rho, expected_inputs,
                         plan_sha256, listed_previous=None):
    if not isinstance(entry, dict):
        raise ValueError("source-family entry must be an object")
    if (entry.get("kind") != family or entry.get("candidate_index") != index
            or float(entry.get("rho", math.nan)) != rho
            or entry.get("status") != "complete" or entry.get("gate_passed") is not False):
        raise ValueError("source-family entries must be the completed ordered rho sequence")
    raw_dir = entry.get("run_directory")
    if not isinstance(raw_dir, str) or not Path(raw_dir).is_absolute():
        raise ValueError("source-family run_directory must be absolute")
    run_dir = Path(raw_dir).resolve()
    protocol_path = run_dir / "protocol.json"
    state_path = run_dir / "run_state.json"
    if not protocol_path.is_file() or not state_path.is_file():
        raise ValueError("source-family run is missing protocol.json or run_state.json")
    protocol, protocol_sha, protocol_bytes = _read_hashed_json(protocol_path)
    state, state_sha, state_bytes = _read_hashed_json(state_path)
    if (protocol_sha != entry.get("protocol_sha256")
            or state_sha != entry.get("run_state_sha256")):
        raise ValueError("source-family run protocol/state hash changed")
    robust._verify_run_state_checksum(state)
    identity = protocol.get("identity", {})
    shared = identity.get("shared", {})
    identity_sha = robust._canonical_hash(identity)
    if (entry.get("identity_sha256") != identity_sha
            or protocol.get("identity_sha256") != identity_sha
            or state.get("identity") != identity
            or state.get("identity_sha256") != identity_sha):
        raise ValueError("source-family identity hashes disagree")
    expected_objective = robust.OBJECTIVE_ID if family == "hinge" else robust.SOFTCOUNT_OBJECTIVE_ID
    candidate = protocol.get("candidate", {})
    frozen_candidate = identity.get("candidate", {})
    if (protocol.get("objective_id") != expected_objective
            or protocol.get("git_head") != entry.get("git_head")
            or candidate.get("index") != index
            or float(candidate.get("rho", math.nan)) != rho
            or frozen_candidate.get("index") != index
            or float(frozen_candidate.get("rho", math.nan)) != rho
            or protocol.get("candidate_plan", {}).get("sha256") != entry.get("candidate_plan_sha256")
            or protocol.get("dataset_sha256") != expected_inputs["dataset_sha256"]
            or protocol.get("dataset_file_sha256") != expected_inputs["dataset_file_sha256"]
            or protocol.get("diagnostic_file_sha256") != expected_inputs["diagnostic_file_sha256"]
            or shared.get("inputs") != expected_inputs
            or shared.get("source_provenance", {}).get("head") != entry.get("git_head")):
        raise ValueError("source-family objective, code, plan, or data identity drifted")
    if shared.get("fixed_protocol", {}).get("objective_id") != expected_objective:
        raise ValueError("source-family shared fixed_protocol objective differs from its family")
    results = state.get("results", {})
    if (results.get("status") != "complete" or state.get("phase") != "complete"
            or results.get("objective_id") != expected_objective
            or results.get("protocol_sha256") != entry.get("protocol_sha256")
            or results.get("final3_status") != "closed; not indexed or evaluated"
            or results.get("heldout_status") != "not indexed or evaluated"):
        raise ValueError("source-family run is not complete with held-out data closed")
    gate = results.get("calibration", {}).get("gate", {})
    if gate.get("passed") is not False:
        raise ValueError("Pareto protected experiment requires all six prerequisite gates to fail")
    _validate_recorded_gate(results)
    recorded_gate = protocol.get("calibration_gate", {})
    if any(recorded_gate.get(key) != value for key, value in _expected_gate(family).items()):
        raise ValueError("source-family run changed the immutable calibration gates")
    seeds = results.get("seeds", [])
    expected_seeds = [{"seed": seed, "complete": True, "fit_qualified": True}
                      for seed in SEED_IDS]
    if ([row.get("seed") for row in seeds] != list(SEED_IDS)
            or not all(row.get("complete") is True and row.get("fit_qualified") is True
                       for row in seeds)):
        raise ValueError("source-family canonical state does not contain five qualified FIT seeds")
    listed_seeds = entry.get("seeds")
    if listed_seeds != expected_seeds:
        raise ValueError("source-family manifest seed summary is incomplete or changed")

    next_rho = SEED_RHOS[index + 1] if index + 1 < len(SEED_RHOS) else rho
    lineage = []
    summary = robust._validate_prior_run(
        str(run_dir), plan_sha256, shared, index + 1, next_rho,
        _lineage=lineage,
    )
    if (summary is None or summary["protocol_sha256"] != entry["protocol_sha256"]
            or summary["run_state_sha256"] != entry["run_state_sha256"]
            or summary["candidate_index"] != index or summary["rho"] != rho):
        raise ValueError("source-family lineage did not validate the listed run")
    ancestors = lineage[1:]
    if listed_previous is None:
        if index == 0:
            if any(item["candidate_index"] != 0 for item in ancestors):
                raise ValueError("first family run does not reach its attempt-one origin")
    else:
        while ancestors and ancestors[0]["candidate_index"] == index:
            ancestors = ancestors[1:]
        ancestor = ancestors[0] if ancestors else None
        previous_path = str(Path(listed_previous["run_directory"]).resolve())
        if (ancestor is None or ancestor["candidate_index"] != index - 1
                or ancestor["path"] != previous_path
                or ancestor["protocol_sha256"] != listed_previous["protocol_sha256"]
                or ancestor["run_state_sha256"] != listed_previous["run_state_sha256"]):
            raise ValueError("source-family prior chain does not pass through the listed predecessor")
    lineage_artifacts = _lineage_artifact_paths(lineage)
    return {"run_directory": str(run_dir), "protocol_sha256": entry["protocol_sha256"],
            "run_state_sha256": entry["run_state_sha256"], "identity_sha256": identity_sha,
            "candidate_plan_sha256": entry["candidate_plan_sha256"], "git_head": entry["git_head"],
            "kind": family, "candidate_index": index, "rho": rho,
            "shared_prerequisite_manifest": shared.get("prerequisite_manifest"),
            "artifact_paths": lineage_artifacts}


def _validate_outer_hinge_entries(hinge_entries, legacy_manifest):
    """Bind outer rows to legacy rows and to their validated header identity."""
    header_head = legacy_manifest.get("code_commit")
    for new_entry, old_entry in zip(hinge_entries, legacy_manifest["runs"]):
        for new_key, old_key in (("candidate_index", "candidate_index"),
                                 ("rho", "rho"), ("run_directory", "run_directory"),
                                 ("protocol_sha256", "protocol_sha256"),
                                 ("run_state_sha256", "run_state_sha256"),
                                 ("identity_sha256", "identity_sha256")):
            if new_entry.get(new_key) != old_entry.get(old_key):
                raise ValueError("outer hinge run list differs from the validated legacy manifest")
        if (new_entry.get("git_head") != header_head
                or new_entry.get("status") != old_entry.get("status")
                or new_entry.get("gate_passed") != old_entry.get("gate_passed")
                or new_entry.get("seeds") != old_entry.get("seeds")):
            raise ValueError("outer hinge family metadata differs from the validated legacy manifest")


def _validate_softcount_predecessor(shared, hinge_manifest_path, hinge_manifest_sha256):
    expected = {"path": str(Path(hinge_manifest_path).resolve()),
                "sha256": hinge_manifest_sha256}
    if shared.get("prerequisite_manifest") != expected:
        raise ValueError("softcount shared prerequisite does not match the listed hinge manifest")
    return expected


def validate_prerequisite_manifest(path, manifest_sha256, expected_inputs):
    manifest_path = _require_absolute_file(path, "prerequisite manifest")
    manifest, actual_sha, manifest_bytes = _read_hashed_json(manifest_path)
    if manifest_sha256 and actual_sha != manifest_sha256:
        raise ValueError("prerequisite manifest SHA256 differs from its frozen plan")
    if (manifest.get("schema_version") != 1
            or manifest.get("status") != "completed_source_families_gate_failed"
            or manifest.get("dataset_sha256") != expected_inputs["dataset_sha256"]
            or manifest.get("diagnostic_file_sha256") != expected_inputs["diagnostic_file_sha256"]):
        raise ValueError("prerequisite manifest header/data identity is invalid")

    hinge_entries = manifest.get("hinge_runs")
    soft_entries = manifest.get("softcount_runs")
    if (not isinstance(hinge_entries, list) or len(hinge_entries) != 3
            or not isinstance(soft_entries, list) or len(soft_entries) != 3):
        raise ValueError("prerequisite manifest must list exactly three runs per source family")
    hinge_ref = manifest.get("hinge_manifest", {})
    hinge_path = _require_absolute_file(hinge_ref.get("path"), "legacy hinge manifest")
    legacy_manifest, legacy_sha, legacy_bytes = _read_hashed_json(hinge_path)
    if legacy_sha != hinge_ref.get("sha256"):
        raise ValueError("legacy hinge manifest hash differs from the pinned manifest")
    if (len(legacy_manifest.get("runs", [])) != 3
            or legacy_manifest.get("runs") is None):
        raise ValueError("legacy hinge manifest must contain three entries")
    hinge_plan_hash = hinge_entries[0].get("candidate_plan_sha256")
    hinge_head = hinge_entries[0].get("git_head")
    if (legacy_manifest.get("candidate_plan_sha256") != hinge_plan_hash
            or legacy_manifest.get("code_commit") != hinge_head):
        raise ValueError("outer hinge family plan/head differs from the legacy manifest header")
    legacy_validation = robust._validate_hinge_family_manifest(
        {
            "prerequisite_manifest": str(hinge_path),
            "prerequisite_family": {
                "candidate_plan_sha256": hinge_plan_hash,
                "code_commit": hinge_head,
            },
        }, expected_inputs,
    )
    if legacy_validation.get("sha256") != hinge_ref.get("sha256"):
        raise ValueError("legacy hinge validation returned a different artifact hash")

    families = {"hinge": [], "softcount": []}
    for family, entries in (("hinge", hinge_entries), ("softcount", soft_entries)):
        if [entry.get("candidate_index") for entry in entries] != [0, 1, 2]:
            raise ValueError("source-family manifest run order changed")
        family_heads = {entry.get("git_head") for entry in entries}
        family_plans = {entry.get("candidate_plan_sha256") for entry in entries}
        expected_head = (
            "300232577f93c2162de54b3cb8efa83ac5cd670d" if family == "hinge"
            else "a71d3ed465464b173231f5be2023250e489df809"
        )
        if family_heads != {expected_head} or len(family_plans) != 1 or None in family_plans:
            raise ValueError("source-family code head or candidate-plan identity changed")
        family_shared = None
        for index, (entry, rho) in enumerate(zip(entries, SEED_RHOS)):
            previous = entries[index - 1] if index else None
            checked = _validate_family_run(
                entry, family, index, rho, expected_inputs,
                entry.get("candidate_plan_sha256"), previous,
            )
            shared = checked.get("shared_prerequisite_manifest")
            if family == "softcount":
                _validate_softcount_predecessor(
                    {"prerequisite_manifest": shared}, hinge_path, hinge_ref["sha256"]
                )
            protocol, protocol_sha, _protocol_bytes = _read_hashed_json(
                Path(checked["run_directory"]) / "protocol.json")
            shared = protocol.get("identity", {}).get("shared", {})
            shared_hash = robust._canonical_hash(shared)
            if family_shared is None:
                family_shared = shared_hash
            elif shared_hash != family_shared:
                raise ValueError("source-family runs do not share one source/data identity")
            families[family].append({**checked, "shared_identity_sha256": shared_hash})
    # The legacy validator ties hinge run hashes and lineage to its own list;
    # require the prospective outer manifest to name those same three runs.
    _validate_outer_hinge_entries(hinge_entries, legacy_manifest)
    artifact_paths = [{"path": str(manifest_path), "sha256": actual_sha},
                      {"path": str(hinge_path), "sha256": legacy_sha}]
    for family_runs in families.values():
        for checked in family_runs:
            artifact_paths.extend(checked["artifact_paths"])
    artifact_paths = _merge_artifact_paths(artifact_paths)
    return {"path": str(manifest_path), "sha256": actual_sha,
            "hinge_manifest": legacy_validation, "families": families,
            "status": manifest["status"], "artifact_paths": artifact_paths}


def _critical_float32_classification(rows, basis32, weights):
    """Return per-layout critical hard errors using the original float32 path."""
    result = {}
    for row in rows:
        tensor = basis32[row["layout_id"]].intensities[0]
        w = torch.as_tensor(weights, dtype=tensor.dtype, device=tensor.device)
        aerial = torch.einsum("nhw,n->hw", tensor, w)
        target = torch.as_tensor(row["target"], dtype=torch.bool, device=tensor.device)
        corner = {}
        for name, dose in (("d0.98", 0.98), ("d1.02", 1.02)):
            printed = constrained.diagnostic.resist_image(
                aerial[None, None], dose=dose, threshold=experiment.THRESHOLD,
                steepness=experiment.STEEPNESS,
            )[0, 0] >= experiment.BINARY
            corner[name] = printed.detach().cpu().numpy()
        result[row["layout_id"]] = {
            "target": target.detach().cpu().numpy(),
            "critical": corner,
        }
    return result


def _anchor_error_partition(rows, basis_arrays64, basis32, anchor):
    float32 = _critical_float32_classification(rows, basis32, anchor)
    errors_by_layout, protected_by_layout = {}, {}
    conflicts = 0
    min_correct = {}
    total_errors = 0
    total_protected = 0
    for row, basis in zip(rows, basis_arrays64):
        layout_id = row["layout_id"]
        target = np.asarray(row["target"], dtype=bool)
        aerial = np.einsum("n,nhw->hw", np.asarray(anchor, dtype=np.float64), basis)
        low_f64 = 0.98 * aerial >= experiment.THRESHOLD
        high_f64 = 1.02 * aerial >= experiment.THRESHOLD
        low_f32 = float32[layout_id]["critical"]["d0.98"]
        high_f32 = float32[layout_id]["critical"]["d1.02"]
        conflicts += int(np.count_nonzero(low_f64 != low_f32))
        conflicts += int(np.count_nonzero(high_f64 != high_f32))
        critical_correct = np.where(target, low_f32 == target, high_f32 == target)
        signed = critical_signed_margins(basis, target, anchor).reshape(target.shape)
        if critical_correct.any():
            min_correct[layout_id] = float(signed[critical_correct].min())
        else:
            min_correct[layout_id] = None
        errors = ~critical_correct
        errors_by_layout[layout_id] = errors
        protected_by_layout[layout_id] = critical_correct
        total_errors += int(errors.sum())
        total_protected += int(critical_correct.sum())
    if conflicts:
        raise ValueError("float64 threshold and original float32 hard-print classifications conflict")
    if any(value is not None and value < EPSILON + 1e-9 for value in min_correct.values()):
        raise ValueError("LP anchor does not protect every originally correct FIT pixel with epsilon")
    return {"errors": errors_by_layout, "protected": protected_by_layout,
            "float64_float32_classification_conflicts": conflicts,
            "anchor_error_pixels": total_errors,
            "protected_pixels": total_protected,
            "minimum_correct_critical_margin_by_layout": min_correct}


def _fit_qualification(rows, basis32, basis_arrays64, targets, anchor, poly, model, solver_record):
    values = solver_record.get("values")
    if values is None:
        return {"passed": False, "reason": "no_incumbent"}
    source_count = model.source_count
    weights = np.asarray(values[:source_count], dtype=np.float64)
    z = np.asarray(values[source_count:], dtype=np.float64)
    residual = solver_record.get("residual", {})
    poly_check = poly.verify(weights, tol=1e-9)
    nominal_check = constrained.nominal_fit_check(rows, basis32, weights)
    anchor_partition = _anchor_error_partition(rows, basis_arrays64, basis32, anchor)
    candidate_hard = _critical_float32_classification(rows, basis32, weights)
    no_new_errors_by_layout = {}
    for row in rows:
        layout_id = row["layout_id"]
        target = np.asarray(row["target"], dtype=bool)
        protected = anchor_partition["protected"][layout_id]
        result = candidate_hard[layout_id]
        no_new = True
        for printed in result["critical"].values():
            no_new = no_new and bool(np.all(printed[protected] == target[protected]))
        no_new_errors_by_layout[layout_id] = no_new
    model_check = residual.get("passed") is True
    objective = float(np.dot(model.objective, values))
    integer_count = int(round(float(z.sum()))) if z.size else 0
    z_integral = bool(np.max(np.abs(z - np.rint(z)), initial=0.0) <= 1e-9)
    z_bounds = bool(np.all(z >= -1e-9) and np.all(z <= 1.0 + 1e-9))
    buffer_check = True
    margins = []
    binary_cursor = 0
    for layout_index, (basis, target) in enumerate(zip(basis_arrays64, targets)):
        signed = critical_signed_margins(basis, target, weights)
        old = anchor_partition["errors"][rows[layout_index]["layout_id"]].reshape(-1)
        protected = anchor_partition["protected"][rows[layout_index]["layout_id"]].reshape(-1)
        buffer_check = buffer_check and bool(np.all(signed[protected] >= EPSILON - 1e-9))
        for pixel in np.flatnonzero(old):
            z_value = z[binary_cursor]
            big_m = model.binary_big_m[binary_cursor]
            buffer_check = buffer_check and bool(
                signed[pixel] + (big_m * z_value) >= EPSILON - 1e-9
            )
            margins.append(float(signed[pixel]))
            binary_cursor += 1
    fit_metrics = constrained.metrics(rows, basis32, weights)
    no_blank_fit = constrained._no_blank_print(fit_metrics)
    checks = {
        "solver_optimal_zero_gap": bool(solver_record.get("optimal_zero_gap")),
        "sparse_model_residual": model_check,
        "integrality": z_integral,
        "binary_bounds": z_bounds,
        "objective_matches_binary_count": abs(objective - integer_count) <= 1e-9,
        "objective_matches_solver_report": (
            solver_record.get("objective") is not None
            and abs(objective - float(solver_record["objective"])) <= 1e-9
        ),
        "nominal_polytope": poly_check["passed"],
        "float32_nominal_fit_zero": nominal_check["passed"],
        "protected_buffer_rows": buffer_check,
        "original_float32_no_new_critical_corner_errors": all(no_new_errors_by_layout.values()),
        "no_blank_positive_target_at_any_fit_dose": no_blank_fit,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "weights": weights.tolist(),
        "buffered_count_objective": objective,
        "unrepaired_anchor_error_count": integer_count,
        "no_new_errors_by_layout": no_new_errors_by_layout,
        "polytope_check": poly_check,
        "nominal_fit_check": nominal_check,
        "fit_metrics": fit_metrics,
        "no_blank_positive_target_at_any_fit_dose": no_blank_fit,
        "error_margin_after_solution": margins,
    }


def _metric_summary(metrics):
    return metrics.get("mean", {})


def _generate_calibration_after_fit_freeze(device, diag, dataset_sha, fit_qualified):
    """Calibration is not reachable until five eligible FIT candidates freeze."""
    eligibility = five_seed_eligibility(fit_qualified)
    if not eligibility["passed"]:
        raise RuntimeError("calibration access attempted before all five FIT solutions qualified")
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("registered calibration generation requires CUDA")
    if dataset_sha != diag["input"]["dataset_sha256"]:
        raise ValueError("dataset identity changed before calibration access")
    teacher, calibration, rows = experiment.generate_calibration(device)
    del teacher
    if (len(calibration.masks) != 4 or calibration.pixel_size_nm != 4.0
            or tuple(calibration.targets.shape[-2:]) != (128, 128)):
        raise ValueError("regenerated calibration must match the registered four 128x128 layouts")
    cal_rows = constrained.layout_rows(calibration, "calibration")
    constrained.check_hashes(cal_rows, diag["input"]["calibration_masks"], "mask", "calibration mask")
    constrained.check_hashes(cal_rows, diag["input"]["calibration_targets"], "target", "calibration target")
    return cal_rows, rows


def _calibration_gate(fit_qualified, cal_rows, cal_basis, anchor_band=constrained.LP_CAL["band_pixels"]):
    records = []
    per_seed_no_blank = []
    for row in fit_qualified:
        metrics = constrained.metrics(cal_rows, cal_basis, np.asarray(row["weights"], dtype=np.float64))
        records.append({"seed": int(row["seed"]), "calibration": metrics})
        per_seed_no_blank.append(constrained._no_blank_print(metrics))
    aggregate = constrained._aggregate_cal(records)
    gate = constrained._gate(
        aggregate, {"mean": {"band_pixels": float(anchor_band)}},
        all(per_seed_no_blank), complete_seeds=True,
    )
    return {"aggregate": aggregate, "per_seed": records,
            "no_blank_each_seed": per_seed_no_blank, "gate": gate,
            "interpretation": "reused development data; not independent generalization evidence"}


def _record_attempt_failure(report, result_path, started, exc):
    calibration_opened = report.get("calibration_attempt_consumed") is True
    if isinstance(exc, KeyboardInterrupt):
        report["status"] = ("interrupted; opened_then_failed; no_retry" if calibration_opened
                             else "interrupted; calibration_closed")
        report["failure"] = {"type": "KeyboardInterrupt", "message": "attempt interrupted"}
    else:
        report["status"] = ("failed; opened_then_failed; no_retry" if calibration_opened
                             else "failed; calibration_closed")
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    report["calibration_status"] = "opened_then_failed; no_retry" if calibration_opened else "closed"
    report["runtime_wall_seconds"] = float(time.monotonic() - started)
    atomic_json(result_path, report)
    return result_path


def _run_registered_attempt(report, result_path, started, *, model, fit_rows,
                            anchor_fit_metrics,
                            identity_check, claim_attempt, solver, qualifier,
                            open_calibration, score_calibration):
    """Testable five-seed FIT freeze and one-shot calibration state machine."""
    fit_qualified = []
    solver_spent = 0.0
    for seed in SEEDS:
        remaining = TOTAL_SOLVER_LIMIT_SECONDS - solver_spent
        if remaining <= 0:
            report["status"] = "solver_budget_exhausted; calibration_closed"
            report["calibration_status"] = "closed"
            break
        time_limit = min(PER_SEED_LIMIT_SECONDS, remaining)
        report["status"] = "solving_fit_seed_%d" % seed
        report["active_seed"] = int(seed)
        atomic_json(result_path, report)
        identity_check("before solver seed %d" % seed)
        if report.get("attempt_marker", {}).get("consumed") is not True:
            claim_attempt()
        solver_row = solver(model, seed, time_limit, require_pinned=True)
        solver_spent += max(float(solver_row.get("external_wall_seconds") or 0.0),
                            float(solver_row.get("solver_wall_seconds") or 0.0))
        seed_record = {
            "seed": int(seed),
            "complete": solver_row.get("optimal_zero_gap") is True,
            "solver": {key: value for key, value in solver_row.items() if key != "values"},
            "fit_qualified": False,
            "fit": {"passed": False, "status": "qualification_pending"},
        }
        seed_record["buffered_count_objective"] = solver_row.get("objective")
        if solver_row.get("incumbent_present"):
            incumbent = np.asarray(solver_row["values"], dtype="<f8")
            seed_record["incumbent_source_weights"] = incumbent[:model.source_count].tolist()
            seed_record["incumbent_vector"] = {
                "values": incumbent.tolist(),
                "sha256_float64_le": hashlib.sha256(incumbent.tobytes()).hexdigest(),
                "source_count": int(model.source_count),
                "binary_count": int(model.binary_count),
                "audit": {"passed": False, "status": "pending"},
            }
        report["seeds"].append(seed_record)
        report["solver_time_spent_seconds"] = solver_spent
        report["seeds_not_started"] = [later for later in SEEDS if later > seed]
        report["status"] = "qualifying_fit_seed_%d" % seed
        report["runtime_wall_seconds"] = float(time.monotonic() - started)
        atomic_json(result_path, report)
        identity_check("after solver seed %d" % seed)
        if solver_row.get("post_run_error", {}).get("interrupted"):
            raise KeyboardInterrupt

        if solver_row.get("incumbent_present"):
            incumbent_audit = _audit_full_incumbent(model, solver_row)
            seed_record["incumbent_vector"]["audit"] = incumbent_audit
            seed_record["fit"]["full_incumbent_audit"] = incumbent_audit
            report["runtime_wall_seconds"] = float(time.monotonic() - started)
            atomic_json(result_path, report)
            qualification_input = dict(solver_row)
            if incumbent_audit.get("residual") is not None:
                qualification_input["residual"] = incumbent_audit["residual"]
            qualified = qualifier(qualification_input)
            qualified["passed"] = bool(qualified.get("passed")
                                       and incumbent_audit.get("passed"))
            qualified["full_incumbent_audit"] = incumbent_audit
            seed_record["incumbent_source_weights"] = qualified.get("weights")
        else:
            incumbent_audit = {"passed": False, "reason": "no_incumbent"}
            qualified = {"passed": False, "reason": "no_incumbent",
                         "full_incumbent_audit": incumbent_audit}
        seed_record["fit_qualified"] = bool(
            solver_row.get("optimal_zero_gap") is True and qualified.get("passed")
        )
        seed_record["fit"] = qualified
        report["runtime_wall_seconds"] = float(time.monotonic() - started)
        atomic_json(result_path, report)
        if not seed_record["fit_qualified"]:
            report["status"] = "incomplete_or_unqualified_fit; calibration_closed"
            report["calibration_status"] = "closed"
            break
        fit_qualified.append({
            "seed": int(seed), "solver": seed_record["solver"],
            "fit_qualified": True, "weights": qualified["weights"],
            "buffered_count_objective": qualified["buffered_count_objective"],
            "fit_metrics": qualified["fit_metrics"],
        })
    else:
        report["seeds_not_started"] = []

    eligibility = five_seed_eligibility([
        {"seed": row["seed"], "fit_qualified": row["fit_qualified"], "solver": row["solver"]}
        for row in report["seeds"]
    ])
    report["fit_eligibility"] = eligibility
    report["solver_time_spent_seconds"] = float(solver_spent)
    report["runtime_wall_seconds"] = float(time.monotonic() - started)
    if not eligibility["passed"]:
        report["status"] = "incomplete_or_unqualified_fit; calibration_closed"
        report["calibration_status"] = "closed"
        atomic_json(result_path, report)
        return result_path

    report["fit_anchor_summary"] = _metric_summary(anchor_fit_metrics)
    fit_improvement = any(
        row["fit_metrics"].get("mean", {}).get("band_pixels") is not None
        and row["fit_metrics"]["mean"]["band_pixels"] < anchor_fit_metrics["mean"]["band_pixels"]
        for row in fit_qualified
    )
    report["fit_candidate_summary"] = [
        {"seed": row["seed"], "buffered_count_objective": row["buffered_count_objective"],
         "fit_metrics": _metric_summary(row["fit_metrics"])} for row in fit_qualified
    ]
    fit_ranked = sorted(fit_qualified, key=lambda row: (
        int(round(row["buffered_count_objective"])),
        row["fit_metrics"]["mean"]["band_pixels"],
        row["fit_metrics"]["mean"]["L2_worst_dose_pixels"],
        row["fit_metrics"]["mean"]["L2_pixels"], row["seed"],
    ))
    report["fit_candidate_ranking"] = [
        {"rank": rank, "seed": row["seed"],
         "buffered_count_objective": row["buffered_count_objective"],
         "fit_metrics": _metric_summary(row["fit_metrics"])}
        for rank, row in enumerate(fit_ranked, start=1)
    ]
    report["fit_selected_seed"] = int(fit_ranked[0]["seed"])
    report["fit_improvement_over_anchor"] = bool(fit_improvement)
    report["fit_selection"] = {
        "rule": "buffered count objective, then hard FIT PV, worst-dose L2, nominal L2, then registered seed order; calibration never breaks ties",
        "seed_order": list(SEEDS), "all_five_seed_weights_frozen": True,
        "selected_seed": int(fit_ranked[0]["seed"]),
    }
    if not fit_improvement:
        report["status"] = "complete_fit_plateau; calibration_closed; no_global_impossibility_claim"
        report["calibration_status"] = "closed"
        report["plateau_interpretation"] = "zero FIT-PV improvement over the LP anchor; no global impossibility claim"
        atomic_json(result_path, report)
        return result_path

    identity_check("before calibration access")
    report["calibration_status"] = "opened_no_retry; in_progress_after_five_fit_freeze"
    report["calibration_attempt_consumed"] = True
    atomic_json(result_path, report)
    cal_context = open_calibration(fit_qualified)
    identity_check("after calibration generation")
    report["calibration"] = score_calibration(fit_qualified, cal_context)
    identity_check("after calibration scoring")
    report["calibration_status"] = "scored_once_after_fit_freeze; no_retry"
    report["status"] = "complete"
    report["runtime_wall_seconds"] = float(time.monotonic() - started)
    atomic_json(result_path, report)
    return result_path


def run(args):
    plan_path = _require_absolute_file(args.candidate_plan, "candidate plan")
    dataset_path = _require_absolute_file(args.dataset_file, "dataset")
    diagnostic_path = _require_absolute_file(args.diagnostic_file, "diagnostic")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        raise ValueError("output root must be absolute")
    output_root = output_root.resolve()
    try:
        output_root.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output root must be outside the source tree")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                             + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir()
    result_path = run_dir / "diagnostic.json"
    started = time.monotonic()
    plan_bytes = plan_path.read_bytes()
    plan_sha_initial = hashlib.sha256(plan_bytes).hexdigest()
    report = {
        "schema_version": 1,
        "objective_id": OBJECTIVE_ID,
        "status": "preflight",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "result_directory": str(run_dir),
        "calibration_status": "closed",
        "final3_status": "closed; not indexed or evaluated",
        "heldout_status": "not indexed or evaluated",
        "plan": {"path": str(plan_path), "sha256": plan_sha_initial},
        "inputs": {"dataset_file": str(dataset_path), "dataset_file_sha256": sha256_file(dataset_path),
                   "diagnostic_file": str(diagnostic_path), "diagnostic_file_sha256": sha256_file(diagnostic_path)},
        "protocol": {
            "rho": RHO, "lp_margin": LP_MARGIN, "nominal_margin_floor": RHO * LP_MARGIN,
            "epsilon": EPSILON, "critical_doses": [0.98, 1.02],
            "solver": "scipy.optimize._highspy._core._Highs sparse CSC MILP",
            "solver_primal_feasibility_tolerance": 1e-9,
            "solver_dual_feasibility_tolerance": 1e-9,
            "solver_mip_feasibility_tolerance": 1e-9,
            "solver_mip_rel_gap": 0.0, "solver_mip_abs_gap": 0.0,
            "solver_threads": 4, "time_limit_seconds_per_seed": 360.0,
            "total_solver_time_budget_seconds": 1800.0,
            "attempts_are_solver_restarts_not_independent_datasets": True,
            "objective_interpretation": EXPECTED_TEXT["objective_interpretation"],
        },
        "preflight": {},
        "anchor": {},
        "model": {},
        "seeds": [],
    }
    atomic_json(result_path, report)
    attempt_marker = None
    try:
        _validate_runtime_gate_constants()
        plan, parsed_plan_sha, _same_plan_bytes = _read_hashed_json(plan_path)
        if parsed_plan_sha != plan_sha_initial or _same_plan_bytes != plan_bytes:
            raise RuntimeError("candidate plan changed while initializing the attempt")
        plan_info = validate_candidate_plan(
            plan, plan_path, dataset_path, diagnostic_path, plan_bytes=plan_bytes
        )
        report["plan"].update(plan_info)
        provenance = git_provenance(plan)
        report["preflight"]["git"] = provenance
        attempt_marker = ensure_attempt_available(plan_path, plan_info["sha256"])
        report["attempt_marker"] = {"path": str(attempt_marker), "consumed": False}
        # A hard pin check happens before any solve and before any calibration access.
        _core, highs_version = pinned_solver_backend(require_pinned=True)
        report["preflight"]["runtime"] = {**provenance["runtime"], "highs": highs_version,
                                          "production_pin_passed": True}

        diag, diagnostic_sha, _diagnostic_bytes = _read_hashed_json(diagnostic_path)
        if diag.get("heldout_status") not in ("not_accessed", "not indexed or evaluated"):
            raise ValueError("registered feasibility diagnostic has unexpected held-out status")
        dataset_sha = str(diag.get("input", {}).get("dataset_sha256", ""))
        dataset_sha_file = sha256_file(dataset_path)
        if dataset_sha_file != dataset_sha:
            raise ValueError("dataset file SHA256 differs from the registered diagnostic")
        if diag.get("physical_parameters", {}).get("source_grid") != 9:
            raise ValueError("diagnostic source grid differs from the fixed 9x9 source")
        if (diag.get("physical_parameters", {}).get("sigma_inner") != 0.3
                or diag.get("physical_parameters", {}).get("sigma_outer") != 0.9):
            raise ValueError("diagnostic source support masks differ from the fixed annulus")
        manifest_path = Path(plan["prerequisite_manifest"]).resolve()
        manifest, manifest_sha_initial, _manifest_bytes = _read_hashed_json(manifest_path)
        if manifest_sha_initial != PINNED_SOURCE_MANIFEST_SHA256:
            raise ValueError("source-family manifest SHA256 differs from the frozen artifact")
        if manifest.get("dataset_sha256") != dataset_sha:
            raise ValueError("prerequisite manifest dataset SHA256 differs")
        if manifest.get("diagnostic_file_sha256") != diagnostic_sha:
            raise ValueError("prerequisite manifest diagnostic SHA256 differs")

        # constrained.load_fit indexes only payload['fit']; held-out/final3
        # payload keys are not looked up or passed to a metric/optics routine.
        fit = constrained.load_fit(dataset_path)
        if (len(fit.masks) != 4 or fit.pixel_size_nm != 4.0
                or tuple(fit.masks.shape[-2:]) != (128, 128)):
            raise ValueError("registered FIT payload must contain four fixed 128x128 4 nm layouts")
        fit_rows = constrained.layout_rows(fit, "fit")
        constrained.check_hashes(fit_rows, diag["input"]["fit_masks"], "mask", "fit mask")
        constrained.check_hashes(fit_rows, diag["input"]["fit_targets"], "target", "fit target")
        expected_basis = {item["layout_id"]: item for item in diag["input"]["bases"]}
        if len(expected_basis) != 8:
            raise ValueError("diagnostic must contain four FIT and four calibration basis hashes")
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("registered basis preparation requires CUDA")
        basis32, fit_cpu64, basis_gpu64, basis_parity = constrained.prepare_bases(
            fit_rows, [], device, expected_basis
        )
        if len(basis32) != 4 or len(fit_cpu64) != 4 or len(basis_gpu64) != 4:
            raise ValueError("basis preparation must use FIT layouts only")
        basis_minima = {}
        for layout_id, basis in basis32.items():
            values = basis.intensities[0]
            basis_minima[layout_id] = float(values.min().item())
            if basis_minima[layout_id] < 0:
                raise ValueError("original hash-checked FIT optical basis is negative")
        support = constrained.support_mask()
        if int(support.sum()) != 49:
            raise ValueError("fixed annular source support does not contain 49 active weights")

        expected_inputs = {
            "dataset_sha256": dataset_sha,
            "dataset_file_sha256": dataset_sha_file,
            "diagnostic_file_sha256": diagnostic_sha,
            "diagnostic_input": diag["input"],
            "basis_parity": basis_parity,
        }
        prerequisite = validate_prerequisite_manifest(
            manifest_path, PINNED_SOURCE_MANIFEST_SHA256, expected_inputs
        )
        if sha256_file(manifest_path) != manifest_sha_initial:
            raise ValueError("prerequisite manifest changed during preflight")

        nominal_lp = diag.get("scenarios", {}).get("fit_only.nominal", {})
        if nominal_lp.get("status") != "positive_margin_feasible":
            raise ValueError("registered fit-only nominal LP is not positive-margin feasible")
        if not math.isclose(float(nominal_lp.get("lp_optimal_margin", math.nan)),
                            LP_MARGIN, rel_tol=0.0, abs_tol=1e-18):
            raise ValueError("registered LP anchor margin differs from the fixed source-family value")
        anchor_full = nominal_lp.get("source_weights_full_grid")
        anchor = constrained.compress(anchor_full, support)
        bases = [fit_cpu64[row["layout_id"]] for row in fit_rows]
        targets = [row["target"].detach().cpu().numpy().astype(bool) for row in fit_rows]
        poly = constrained.build_polytope(bases, targets, LP_MARGIN, rho=RHO)
        poly_check = poly.verify(anchor)
        if not poly_check["passed"]:
            raise ValueError("registered LP anchor is outside the fixed nominal polytope")
        nominal_anchor = constrained.nominal_fit_check(fit_rows, basis32, anchor)
        if not nominal_anchor["passed"]:
            raise ValueError("registered LP anchor does not have zero nominal FIT errors")
        partition = _anchor_error_partition(fit_rows, bases, basis32, anchor)
        if (partition["anchor_error_pixels"] != 1331
                or partition["protected_pixels"] != 64205
                or partition["float64_float32_classification_conflicts"] != 0):
            raise ValueError("anchor critical-corner classification differs from the registered preflight")
        model = build_sparse_milp(
            bases, targets, anchor, poly, lp_margin=LP_MARGIN, rho=RHO,
            epsilon=EPSILON, solver_tolerance=1e-9,
            layout_ids=[row["layout_id"] for row in fit_rows],
        )
        partition_map = _validate_model_partition(fit_rows, partition, model)
        report["preflight"].update({
            "dataset_sha256": dataset_sha,
            "diagnostic_file_sha256": sha256_file(diagnostic_path),
            "prerequisite_manifest": prerequisite,
            "fit_layout_ids": [row["layout_id"] for row in fit_rows],
            "basis_parity": basis_parity,
            "fit_basis_minima": basis_minima,
            "calibration_basis_hashes_read_only": True,
            "fit_only_basis_preparation": True,
        })
        report["anchor"] = {
            "source_weights_supported_float64": anchor.tolist(),
            "nominal_polytope": poly_check,
            "nominal_fit_zero": nominal_anchor,
            "critical_error_pixels": partition["anchor_error_pixels"],
            "protected_pixels": partition["protected_pixels"],
            "classification_conflicts_float64_vs_float32": partition["float64_float32_classification_conflicts"],
            "minimum_correct_critical_margin_by_layout": partition["minimum_correct_critical_margin_by_layout"],
        }
        report["model"] = {
            "variables": {"source_weights": model.source_count, "error_binaries": model.binary_count,
                          "total": int(model.objective.size)},
            "rows": int(model.matrix.shape[0]),
            "nonzeros": int(model.matrix.nnz),
            "nominal_rows": model.nominal_row_count,
            "nominal_rows_original": model.nominal_original_row_count,
            "nominal_rows_pruned_by_protected_corner_proof": model.nominal_pruned_protected_count,
            "nominal_rows_kept_for_anchor_errors": model.nominal_row_count,
            "protected_rows_kept": model.protected_count,
            "protected_rows_pruned_by_simplex_bound": model.protected_pruned_simplex,
            "protected_rows_pruned_by_nominal_bound": model.protected_pruned_nominal,
            "protected_rows_pruned_total": model.protected_pruned_simplex + model.protected_pruned_nominal,
            "partition_mapping": partition_map,
            "binary_rows": model.binary_count,
            "pruning_policy": (
                "exactly omit nominal rows for all originally correct critical-corner FIT pixels; "
                "their kept critical constraints or certified simplex lower bounds imply the nominal "
                "margin floor; retain every nominal row for anchor-error pixels; no approximate "
                "quantization or row deduplication"
            ),
            "warm_start_residual": __import__("source_pareto").check_model_residual(model, model.anchor_start),
            "model_sha256": hash_sparse_model(model),
        }
        report["hashes"] = {
            "candidate_plan_sha256": plan_info["sha256"],
            "prerequisite_manifest_sha256": prerequisite["sha256"],
            "dataset_file_sha256": sha256_file(dataset_path),
            "diagnostic_file_sha256": sha256_file(diagnostic_path),
            "source_sha256": provenance["source_sha256"],
            "fit_basis_sha256": {
                item["layout_id"]: item["sha256"] for item in diag["input"]["bases"]
                if item.get("split") == "fit"
            },
        }
        identity_snapshot = _identity_snapshot(
            plan_path, plan_bytes, plan, dataset_path, dataset_sha_file,
            diagnostic_path, diag, diagnostic_sha, provenance, prerequisite
        )
        anchor_fit_metrics = constrained.metrics(fit_rows, basis32, anchor)

        def check_identity(stage):
            _recheck_identity(identity_snapshot, plan_path, plan, dataset_path,
                              diagnostic_path, prerequisite, stage)

        def consume_attempt():
            if not report["attempt_marker"]["consumed"]:
                report["attempt_marker"].update(
                    claim_solver_attempt(plan_path, plan_info["sha256"])
                )
                report["attempt_marker"]["consumed"] = True
                atomic_json(result_path, report)

        def qualify_fit(solver_row):
            return _fit_qualification(
                fit_rows, basis32, bases, targets, anchor, poly, model, solver_row
            )

        def open_calibration(frozen_fit_rows):
            cal_rows, _generated_rows = _generate_calibration_after_fit_freeze(
                device, diag, dataset_sha, [
                    {"seed": row["seed"], "fit_qualified": True, "solver": row["solver"]}
                    for row in frozen_fit_rows
                ]
            )
            cal_expected = {item["layout_id"]: item for item in diag["input"]["bases"]}
            cal_basis32, _unused_cpu, _unused_gpu, cal_parity = constrained.prepare_bases(
                [], cal_rows, device, cal_expected
            )
            report["calibration_basis_parity"] = cal_parity
            return cal_rows, cal_basis32

        def score_calibration(frozen_fit_rows, cal_context):
            cal_rows, cal_basis32 = cal_context
            return _calibration_gate(frozen_fit_rows, cal_rows, cal_basis32)

        return _run_registered_attempt(
            report, result_path, started, model=model, fit_rows=fit_rows,
            anchor_fit_metrics=anchor_fit_metrics, identity_check=check_identity,
            claim_attempt=consume_attempt, solver=solve_highs,
            qualifier=qualify_fit, open_calibration=open_calibration,
            score_calibration=score_calibration,
        )
        return result_path
    except KeyboardInterrupt:
        return _record_attempt_failure(report, result_path, started, KeyboardInterrupt())
    except Exception as exc:
        return _record_attempt_failure(report, result_path, started, exc)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", required=True, type=Path)
    parser.add_argument("--diagnostic-file", required=True, type=Path)
    parser.add_argument("--candidate-plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result_path = run(args)
    result = read_json(result_path)
    print("Diagnostic:", result_path, flush=True)
    print("Status:", result.get("status"), flush=True)
    print("Calibration:", result.get("calibration_status"), flush=True)
    return 0 if str(result.get("status", "")).startswith("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
