"""Preflight or directly verify a frozen source against fixed FIT layouts.

``direct-verify`` is a one-source parity/quality confirmation. It loads only
the pinned FIT split and never creates an event-search attempt marker. The
prospective event search itself consumes a distinct marker only after its
separate plan has been frozen and approved for execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from types import SimpleNamespace
import time
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import source_coverage as coverage
import source_event_search as event_search
from light_source import DifferentiableAbbeLitho, PixelatedLightSource, resist_image
from scripts import optimize_source_coverage as parent
from scripts import optimize_source_constrained as constrained


class FixedWeightsSource(PixelatedLightSource):
    """Expose the provided 49 float32 weights directly to the Abbe simulator."""

    def __init__(self, weights, *, grid_size=9, sigma_inner=0.3, sigma_outer=0.9):
        super().__init__(grid_size=grid_size, sigma_inner=sigma_inner, sigma_outer=sigma_outer)
        coordinates, _ = super().distribution()
        vector = event_search.validate_weights(weights)
        if vector.size != coordinates.shape[0]:
            raise ValueError("fixed source vector does not match active pupil support")
        self.register_buffer("_fixed_weights", torch.as_tensor(vector, dtype=torch.float32))

    def distribution(self):
        return self._coordinates[self._pupil_support], self._fixed_weights

    def set_weights(self, weights):
        vector = event_search.validate_weights(weights)
        if vector.size != self._fixed_weights.numel():
            raise ValueError("fixed source vector does not match the simulator source grid")
        with torch.no_grad():
            self._fixed_weights.copy_(torch.as_tensor(
                vector, dtype=self._fixed_weights.dtype, device=self._fixed_weights.device,
            ))


def _load_weights(path: Path) -> tuple[np.ndarray, dict, str, dict]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or "weights" not in payload:
        raise ValueError("weights JSON must contain an explicit `weights` array")
    weights = event_search.validate_weights(payload["weights"])
    computed64 = event_search.array_sha256(weights, "<f8")
    computed32 = event_search.array_sha256(weights, "<f4")
    for key, actual in (("weights_f64_sha256", computed64), ("weights_f32_sha256", computed32)):
        recorded = payload.get(key)
        if recorded is not None and recorded != actual:
            raise ValueError("weights JSON %s does not match the explicit source vector" % key)
    return weights, {"weights_f64_sha256": computed64, "weights_f32_sha256": computed32}, digest, payload


def _preflight(plan_path: Path, expected_sha: str, expected_previous_sha: str,
               device: str) -> tuple[dict, str, dict]:
    args = SimpleNamespace(plan_file=str(plan_path), expected_plan_sha256=expected_sha,
                           expected_previous_sha256=expected_previous_sha, device=device)
    plan, plan_sha, identity = parent._preflight(args)
    if plan.get("schema_version") != 8 or plan.get("objective_id") != coverage.OBJECTIVE_ID_V8:
        raise ValueError("direct verification requires the frozen schema-8 FIT coverage plan")
    if plan_sha != coverage.PINNED_PLAN_SHA256_V8:
        raise ValueError("direct verification plan differs from the frozen schema-8 plan pin")
    return plan, plan_sha, identity


def _fit_rows(identity: dict, plan: dict, device: torch.device) -> tuple[list[dict], dict]:
    fit = parent._load_fit_only(identity["dataset_bytes"])
    physical = plan["physical"]
    if (float(fit.pixel_size_nm) != float(physical["pixel_nm"])
            or tuple(fit.layout_ids) != coverage.ORIGINAL_FIT_LAYOUT_IDS
            or tuple(fit.masks.shape[-2:]) != (128, 128)
            or tuple(fit.targets.shape[-2:]) != (128, 128)):
        raise ValueError("original FIT IDs/order, pixel size, or raster differ from the frozen contract")
    rows = parent._as_numpy_rows(fit)
    coverage.validate_layout_rows(
        rows, coverage.ORIGINAL_FIT_LAYOUT_IDS,
        identity["diagnostic"]["input"]["fit_masks"],
        identity["diagnostic"]["input"]["fit_targets"], "FIT",
    )
    fixed = parent._generate_new_targets(
        coverage.make_coverage_masks(physical["raster"]), device, physical,
    )
    observed = [{
        "layout_id": row["layout_id"],
        "mask_sha256": coverage.sha256_array(row["mask"]),
        "target_sha256": coverage.sha256_array(coverage.target_hashable(row["target"])),
    } for row in fixed]
    if observed != coverage.PINNED_FIXED_FIT_LAYOUT_HASHES:
        raise ValueError("generated fixed FIT masks/targets differ from the pinned three-layout contract")
    if plan.get("fixed_fit_layout_hashes") != coverage.PINNED_FIXED_FIT_LAYOUT_HASHES:
        raise ValueError("plan fixed FIT layout identity differs from the source-only contract")
    return rows + fixed, {"original_fit_layout_count": len(rows),
                          "fixed_new_fit_layout_count": len(fixed),
                          "fixed_new_fit_layout_hashes": observed}


def _hard_metrics_from_aerial(aerial: torch.Tensor, target: np.ndarray,
                              threshold: float, steepness: float) -> dict:
    target_t = torch.as_tensor(np.asarray(target, dtype=bool), dtype=torch.bool,
                               device=aerial.device)
    doses = event_search.DOSES
    with torch.no_grad():
        binary = torch.stack([
            resist_image(aerial, dose=dose, threshold=threshold, steepness=steepness) >= 0.5
            for dose in doses
        ]).reshape(len(doses), -1)
    expected = target_t.reshape(1, -1)
    per_dose = (binary != expected).sum(dim=1)
    positive = int(target_t.sum().item())
    blank = [int(row.sum().item()) == 0 and positive > 0 for row in binary]
    return {
        "L2_pixels": int(per_dose[1].item()),
        "L2_worst_dose_pixels": int(per_dose.max().item()),
        "band_pixels": int((binary.any(dim=0) != binary.all(dim=0)).sum().item()),
        "per_dose_L2_pixels": [int(value) for value in per_dose.tolist()],
        "positive_target_pixels": positive,
        "no_blank_positive_target_any_dose": not any(blank),
    }


def _aggregate(rows: list[dict]) -> dict:
    return coverage.aggregate_metrics(rows)


def _route_summary(rows: list[dict]) -> dict:
    original_n = len(coverage.ORIGINAL_FIT_LAYOUT_IDS)
    return {
        "per_layout": rows,
        "original_fit_mean": _aggregate(rows[:original_n]),
        "new_fit_mean": _aggregate(rows[original_n:]),
        "no_blank_positive_target_any_dose": all(
            row["no_blank_positive_target_any_dose"] for row in rows
        ),
    }


def _verify_weights_on_fit(rows: list[dict], weights: np.ndarray, reference: np.ndarray,
                           identity: dict, plan: dict, device: torch.device) -> dict:
    physical = plan["physical"]
    source = FixedWeightsSource(
        weights, grid_size=physical["source_grid"], sigma_inner=physical["sigma_inner"],
        sigma_outer=physical["sigma_outer"],
    ).to(device)
    simulator = DifferentiableAbbeLitho(
        source, numerical_aperture=physical["NA"], wavelength_nm=physical["wavelength_nm"],
        pixel_size_nm=physical["pixel_nm"], source_chunk_size=8, cache_max_bytes=0,
    ).to(device)
    expected_bases = {row["layout_id"]: row
                      for row in identity["diagnostic"]["input"]["bases"]}
    expected_ids = set(coverage.ORIGINAL_FIT_LAYOUT_IDS)
    if not expected_ids.issubset(expected_bases):
        raise ValueError("pinned diagnostic lacks a basis identity for every original FIT layout")

    routes = ("direct_gpu", "weighted_basis_gpu", "canonical_cpu")
    per_vector = {name: {route: [] for route in routes}
                  for name in ("candidate", "reference")}
    parity = []
    basis_prepare_seconds = 0.0
    evaluation_started = time.monotonic()
    for row in rows:
        mask = torch.as_tensor(row["mask"], dtype=torch.float32, device=device)[None, None]
        basis_started = time.monotonic()
        basis = simulator.prepare_basis(mask.detach(), defocus_nm=0.0, max_bytes=512 * 1024**2)
        intensities = basis.intensities.detach().to(dtype=torch.float32).contiguous()
        basis_prepare_seconds += time.monotonic() - basis_started
        if intensities.shape[1:] != (event_search.SOURCE_COUNT, 128, 128):
            raise ValueError("canonical FIT basis must contain 49x128x128 float32 intensities")
        descriptor = expected_bases.get(row["layout_id"])
        if descriptor is not None:
            if (constrained.sha256_tensor(intensities.cpu()) != descriptor["sha256"]
                    or list(intensities.shape) != descriptor["shape"]
                    or str(intensities.dtype) != descriptor["dtype"]):
                raise ValueError("direct-verification basis differs from frozen FIT basis for %s"
                                 % row["layout_id"])

        for name, vector in (("candidate", weights), ("reference", reference)):
            source.set_weights(vector)
            with torch.no_grad():
                direct = simulator(mask)
                weighted = simulator.evaluate_basis(basis, defocus_nm=0.0)
            maximum = float((direct - weighted).abs().max().item())
            aerial_equal = bool(torch.allclose(direct, weighted, rtol=1e-5, atol=1e-6))
            direct_metrics = _hard_metrics_from_aerial(
                direct, row["target"], physical["threshold"], physical["steepness"],
            )
            basis_metrics = _hard_metrics_from_aerial(
                weighted, row["target"], physical["threshold"], physical["steepness"],
            )
            cpu_metrics = coverage.hard_metrics(
                intensities[0].detach().cpu().numpy(), row["target"], vector,
                threshold=physical["threshold"], steepness=physical["steepness"],
            )
            per_vector[name]["direct_gpu"].append({"layout_id": row["layout_id"], **direct_metrics})
            per_vector[name]["weighted_basis_gpu"].append({"layout_id": row["layout_id"], **basis_metrics})
            per_vector[name]["canonical_cpu"].append({"layout_id": row["layout_id"], **cpu_metrics})
            parity.append({"source": name, "layout_id": row["layout_id"],
                           "max_abs_aerial_error": maximum,
                           "aerial_allclose": aerial_equal,
                           "hard_metrics_equal": direct_metrics == basis_metrics,
                           "canonical_cpu_matches_direct": cpu_metrics == direct_metrics,
                           "canonical_cpu_matches_basis": cpu_metrics == basis_metrics,
                           "direct_hard_metrics": direct_metrics,
                           "weighted_basis_hard_metrics": basis_metrics,
                           "canonical_cpu_hard_metrics": cpu_metrics})
        del basis, intensities

    evaluation_seconds = time.monotonic() - evaluation_started
    summaries = {name: {route: _route_summary(rows_for_route)
                        for route, rows_for_route in routes_by_name.items()}
                 for name, routes_by_name in per_vector.items()}
    return {
        "candidate": summaries["candidate"]["direct_gpu"],
        "reference": summaries["reference"]["direct_gpu"],
        "all_routes": summaries,
        "direct_vs_weighted_basis": {
            "passed": all(row["aerial_allclose"] and row["hard_metrics_equal"]
                           and row["canonical_cpu_matches_direct"]
                           and row["canonical_cpu_matches_basis"] for row in parity),
            "layout_source_checks": parity,
            "comparison": "aerial tensors within rtol=1e-5, atol=1e-6; exact direct-GPU, basis-GPU and canonical-CPU hard counts",
        },
        "timings": {"basis_prepare_seconds": basis_prepare_seconds,
                    "candidate_and_reference_evaluation_seconds": evaluation_seconds},
    }


def _aggregate_hard_gates(candidate: dict, reference: dict) -> dict:
    """Apply every frozen old/new aggregate gate to canonical route summaries."""
    old, ref_old = candidate["original_fit_mean"], reference["original_fit_mean"]
    new, ref_new = candidate["new_fit_mean"], reference["new_fit_mean"]
    return {
        "all_fit_no_blank": (candidate["no_blank_positive_target_any_dose"] is True
                             and reference["no_blank_positive_target_any_dose"] is True),
        "original_band_nonincreasing": old["band_pixels"] <= ref_old["band_pixels"],
        "original_nominal_l2_nonincreasing": old["L2_pixels"] <= ref_old["L2_pixels"],
        "original_worst_l2_nonincreasing": (
            old["L2_worst_dose_pixels"] <= ref_old["L2_worst_dose_pixels"]),
        "original_frozen_band_limit": old["band_pixels"] <= 234.5,
        "original_frozen_nominal_l2_limit": old["L2_pixels"] <= 0.0,
        "original_frozen_worst_l2_limit": old["L2_worst_dose_pixels"] <= 150.25,
        "new_band_strictly_lower": new["band_pixels"] < ref_new["band_pixels"],
        "new_nominal_l2_nonincreasing": new["L2_pixels"] <= ref_new["L2_pixels"],
        "new_worst_l2_nonincreasing": (
            new["L2_worst_dose_pixels"] <= ref_new["L2_worst_dose_pixels"]),
    }


def _direct_gain_confirmed(parity_passed: bool, hard_gates: dict,
                           historical_parity: dict | None) -> bool:
    return (parity_passed and bool(hard_gates)
            and all(value is True for value in hard_gates.values())
            and (historical_parity is None or historical_parity.get("verified") is True))


def _write_new_json(path: Path, payload: dict) -> None:
    path = path.expanduser()
    if os.path.lexists(path):
        raise FileExistsError("output JSON already exists: %s" % path)
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output JSON must be outside the source repository")
    if os.path.lexists(path):
        raise FileExistsError("output JSON already exists: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-%d-%s" % (os.getpid(), uuid.uuid4().hex))
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    try:
        with temp.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, AttributeError):
            pass
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _validate_new_output_path(path: Path) -> Path:
    path = path.expanduser()
    if os.path.lexists(path):
        raise FileExistsError("output JSON already exists: %s" % path)
    path = path.resolve(strict=False)
    if os.path.lexists(path):
        raise FileExistsError("output JSON already exists: %s" % path)
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        return path
    raise ValueError("output JSON must be outside the source repository")


def preflight(plan_path: Path, expected_plan_sha256: str,
              expected_previous_sha256: str, device: str) -> dict:
    plan, plan_sha, identity = _preflight(
        plan_path, expected_plan_sha256, expected_previous_sha256, device,
    )
    return {
        "status": "preflight_passed_no_dataset_deserialized_no_optics_no_marker",
        "objective_id": event_search.PROTOCOL_ID,
        "coverage_plan_sha256": plan_sha,
        "source_manifest_sha256": plan["input_hashes"]["source_manifest_sha256"],
        "dataset_sha256": plan["input_hashes"]["dataset_sha256"],
        "diagnostic_sha256": plan["input_hashes"]["diagnostic_sha256"],
        "source_identity_files_checked": len(identity.get("source_identity", {}).get("files", {})),
        "dataset_scope": "not deserialized during preflight; direct-verify reads FIT only",
        "calibration_status": "closed by design",
        "final3_status": "never indexed or evaluated",
    }


def direct_verify(plan_path: Path, expected_plan_sha256: str,
                  expected_previous_sha256: str, weights_path: Path,
                  output_path: Path, device_name: str,
                  source_report_path: Path | None = None,
                  expected_source_report_sha256: str | None = None) -> Path:
    total_started = time.monotonic()
    output_path = _validate_new_output_path(output_path)
    pinned_bytes_before = {
        "weights_file_sha256": event_search.sha256_file(weights_path),
        "source_event_search.py": event_search.sha256_file(ROOT / "source_event_search.py"),
        "scripts/diagnose_source_events.py": event_search.sha256_file(Path(__file__)),
    }
    if source_report_path is not None:
        pinned_bytes_before["source_report_sha256"] = event_search.sha256_file(source_report_path)
    plan, plan_sha, identity = _preflight(
        plan_path, expected_plan_sha256, expected_previous_sha256, device_name,
    )
    weights, vector_hashes, weights_file_sha, weights_payload = _load_weights(weights_path)
    provenance = {"verified": False, "reason": "no source report supplied"}
    historical_metrics = None
    if (source_report_path is None) != (expected_source_report_sha256 is None):
        raise ValueError("source report path and expected SHA256 must be supplied together")
    if source_report_path is not None:
        report_raw = source_report_path.read_bytes()
        actual_report_sha = hashlib.sha256(report_raw).hexdigest()
        if actual_report_sha != str(expected_source_report_sha256).lower():
            raise ValueError("selected source report bytes differ from the required SHA256")
        source_report = json.loads(report_raw.decode("utf-8"))
        if (not isinstance(source_report, dict)
                or source_report.get("objective_id") != "fit_only_fixed_source_segment_hard_metric_sweep_v1"
                or source_report.get("status") != "complete"
                or source_report.get("calibration_status") != "closed by design"
                or source_report.get("final3_status") != "never indexed or evaluated"):
            raise ValueError("source report is not the complete pinned FIT-only segment result")
        selected = source_report.get("selected_qualified_fit_point")
        selected_metrics = selected.get("fit_metrics") if isinstance(selected, dict) else None
        if (weights_payload.get("report_sha256") != actual_report_sha
                or weights_payload.get("plan_sha256") != source_report.get("plan_sha256")
                or not isinstance(selected, dict)
                or not isinstance(selected_metrics, dict)
                or selected.get("weights_f64_sha256") != vector_hashes["weights_f64_sha256"]
                or not np.array_equal(np.asarray(selected.get("weights"), dtype=np.float64), weights)):
            raise ValueError("weights file does not identify the report's selected source vector")
        if selected_metrics.get("qualified") is not True:
            raise ValueError("source report selected point is not hard-qualified")
        historical_metrics = selected_metrics.get("all_fit", {}).get("per_layout")
        if not isinstance(historical_metrics, list):
            raise ValueError("source report lacks its selected point's per-layout FIT hard metrics")
        provenance = {
            "verified": True, "report_file": str(source_report_path.resolve()),
            "report_sha256": actual_report_sha, "plan_sha256": source_report["plan_sha256"],
            "selected_candidate": {key: selected.get(key) for key in (
                "slot_index", "seed", "segment_id", "alpha", "weights_f64_sha256",
            )},
            "historical_fit_metrics": selected_metrics["all_fit"],
        }
    reference = event_search.validate_weights(identity["previous_weights"]["reference_weights"])
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("canonical direct verification requires the registered CUDA optical backend")
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "torch_num_threads": int(torch.get_num_threads()),
    }
    environment["gpu_name"] = torch.cuda.get_device_name(device)
    environment["gpu_properties"] = {
        "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "capability": list(torch.cuda.get_device_capability(device)),
    }
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    environment["gpu_memory_at_verification_start"] = {
        "free_bytes": int(free_memory), "total_bytes": int(total_memory),
    }
    fit_started = time.monotonic()
    rows, layout_audit = _fit_rows(identity, plan, device)
    fit_load_seconds = time.monotonic() - fit_started
    metrics = _verify_weights_on_fit(rows, weights, reference, identity, plan, device)
    historical_parity = {"verified": False, "per_layout": [],
                         "interpretation": "no pinned source report supplied"}
    if historical_metrics is not None:
        historical_by_id = {row["layout_id"]: row for row in historical_metrics}
        mismatches = []
        observed = []
        canonical_rows = metrics["all_routes"]["candidate"]["canonical_cpu"]["per_layout"]
        for row in canonical_rows:
            old = historical_by_id.get(row["layout_id"])
            if old is None:
                mismatches.append({"layout_id": row["layout_id"], "reason": "missing_historical_layout"})
                continue
            fields = ("L2_pixels", "L2_worst_dose_pixels", "band_pixels",
                      "per_dose_L2_pixels", "positive_target_pixels",
                      "no_blank_positive_target_any_dose")
            differences = {field: {"historical": old.get(field), "direct": row.get(field)}
                           for field in fields if old.get(field) != row.get(field)}
            observed.append({"layout_id": row["layout_id"], "matched": not differences,
                             "differences": differences})
            if differences:
                mismatches.append(observed[-1])
        historical_parity = {
            "verified": (metrics["direct_vs_weighted_basis"]["passed"]
                         and not mismatches and len(observed) == len(canonical_rows)),
            "per_layout": observed, "mismatches": mismatches,
            "interpretation": ("canonical CPU hard counts compared with the pinned selected FIT report; verified only when all three evaluator routes agree"
                               if metrics["direct_vs_weighted_basis"]["passed"] else
                               "unverified: evaluator parity failed, so historical comparison is not evidence of source gain"),
        }
    candidate_mean = metrics["candidate"]["new_fit_mean"]
    reference_mean = metrics["reference"]["new_fit_mean"]
    delta = {key: float(candidate_mean[key] - reference_mean[key])
             for key in ("band_pixels", "L2_pixels", "L2_worst_dose_pixels")}
    parity_passed = bool(metrics["direct_vs_weighted_basis"]["passed"])
    hard_gate_results = _aggregate_hard_gates(
        metrics["candidate"], metrics["reference"],
    )
    hard_gates_passed = all(hard_gate_results.values())
    direct_gain_confirmed = _direct_gain_confirmed(
        parity_passed, hard_gate_results,
        historical_parity if historical_metrics is not None else None,
    )
    # Re-run the registered preflight after optics. It rehashes the frozen code,
    # plans, datasets and all pinned artifacts without opening calibration/final3.
    rechecked_plan, rechecked_sha, _rechecked_identity = _preflight(
        plan_path, expected_plan_sha256, expected_previous_sha256, device_name,
    )
    if rechecked_sha != plan_sha or rechecked_plan.get("input_hashes") != plan.get("input_hashes"):
        raise RuntimeError("frozen plan or inputs changed during direct verification")
    pinned_bytes_after = {
        "weights_file_sha256": event_search.sha256_file(weights_path),
        "source_event_search.py": event_search.sha256_file(ROOT / "source_event_search.py"),
        "scripts/diagnose_source_events.py": event_search.sha256_file(Path(__file__)),
    }
    if source_report_path is not None:
        pinned_bytes_after["source_report_sha256"] = event_search.sha256_file(source_report_path)
    if pinned_bytes_after != pinned_bytes_before:
        raise RuntimeError("pinned source code, weights, or selected report changed during verification")
    report = {
        "schema_version": 1, "objective_id": "source_only_direct_forward_confirmation_v1",
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "coverage_plan_sha256": plan_sha,
        "weights_file": str(weights_path.resolve()), "weights_file_sha256": weights_file_sha,
        "source_report_provenance": provenance,
        "candidate_weights": {**vector_hashes, "source_count": int(weights.size),
                               "unit_flux_f64": float(weights.sum()),
                               "weights": weights.tolist()},
        "reference_weights": {
            "weights_f64_sha256": event_search.array_sha256(reference, "<f8"),
            "weights_f32_sha256": event_search.array_sha256(reference, "<f4"),
            "source_count": int(reference.size), "unit_flux_f64": float(reference.sum()),
        },
        "source_code_sha256": {
            "source_event_search.py": event_search.sha256_file(ROOT / "source_event_search.py"),
            "scripts/diagnose_source_events.py": event_search.sha256_file(Path(__file__)),
        },
        "input_hashes": {key: value for key, value in plan["input_hashes"].items()},
        "physical": {key: plan["physical"][key] for key in (
            "source_grid", "sigma_inner", "sigma_outer", "NA", "wavelength_nm",
            "pixel_nm", "raster", "doses", "focus", "threshold", "steepness",
        )},
        "dataset_scope": "original FIT split plus exactly three pinned fixed new FIT layouts",
        "calibration_status": "closed by design", "final3_status": "never indexed or evaluated",
        "attempt_marker": "not created; this direct parity confirmation is not an event-search attempt",
        "layout_audit": layout_audit, "metrics": metrics,
        "historical_report_hard_metric_parity": historical_parity,
        "new_fit_candidate_minus_reference_mean": delta,
        "verification": {
            "evaluator_parity_passed": parity_passed,
            "aggregate_hard_quality_gates_passed": hard_gates_passed,
            "aggregate_hard_quality_gate_results": hard_gate_results,
            "direct_source_gain_confirmed": bool(direct_gain_confirmed),
            "historical_selected_report_parity_passed": (
                bool(historical_parity["verified"]) if historical_metrics is not None else None
            ),
            "interpretation": (
                "direct simulator, weighted basis and canonical CPU hard counts agree; candidate passes all frozen aggregate hard gates and any supplied historical parity check"
                if direct_gain_confirmed else
                "source gain is not confirmed because evaluator parity, an aggregate hard gate, or a supplied historical parity check failed"
            ),
        },
        "environment": environment,
        "timings": {
            "fit_load_and_layout_audit_seconds": fit_load_seconds,
            **metrics["timings"],
            "total_seconds_including_preflight_and_pin_recheck": time.monotonic() - total_started,
        },
        "pin_recheck": {"passed": True, "before": pinned_bytes_before,
                        "after": pinned_bytes_after, "coverage_plan_sha256_after": rechecked_sha},
        "interpretation": (
            "direct simulator reproduces weighted-basis aerial images and canonical CPU hard counts; FIT-only aggregate source gain confirmed"
            if direct_gain_confirmed else
            "FIT-only verification completed, but evaluator parity or the frozen aggregate hard gates do not support a source gain claim"
        ),
    }
    _write_new_json(output_path, report)
    return output_path.resolve()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "direct-verify"), required=True)
    p.add_argument("--coverage-plan-file", required=True)
    p.add_argument("--expected-coverage-plan-sha256", required=True)
    p.add_argument("--expected-previous-sha256", required=True)
    p.add_argument("--weights-file", help="JSON with an explicit 49-entry `weights` array")
    p.add_argument("--source-report-file", help="optional pinned FIT-only event/grid report for vector provenance")
    p.add_argument("--expected-source-report-sha256")
    p.add_argument("--output-file", help="new JSON result path outside the source repository")
    p.add_argument("--device", default="cuda")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.mode == "preflight":
            print(json.dumps(preflight(Path(args.coverage_plan_file),
                                       args.expected_coverage_plan_sha256,
                                       args.expected_previous_sha256, args.device), indent=2))
            return 0
        if not args.weights_file or not args.output_file:
            raise ValueError("--weights-file and --output-file are required for direct-verify")
        result = direct_verify(
            Path(args.coverage_plan_file), args.expected_coverage_plan_sha256,
            args.expected_previous_sha256, Path(args.weights_file), Path(args.output_file), args.device,
            Path(args.source_report_file) if args.source_report_file else None,
            args.expected_source_report_sha256,
        )
        report = json.loads(result.read_text(encoding="utf-8"))
        summary = {
            "status": "complete", "report_path": str(result),
            "direct_vs_weighted_basis_passed": report["metrics"]["direct_vs_weighted_basis"]["passed"],
            "historical_report_hard_metric_parity": report["historical_report_hard_metric_parity"]["verified"],
            "direct_source_gain_confirmed": report["verification"]["direct_source_gain_confirmed"],
            "candidate_new_fit_mean": report["metrics"]["candidate"]["new_fit_mean"],
            "reference_new_fit_mean": report["metrics"]["reference"]["new_fit_mean"],
            "candidate_minus_reference": report["new_fit_candidate_minus_reference_mean"],
        }
        print(json.dumps(summary, indent=2))
        success = summary["direct_source_gain_confirmed"]
        return 0 if success else 3
    except Exception as exc:
        print("source event direct verification failed: %s: %s" % (type(exc).__name__, exc),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
