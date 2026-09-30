"""Source-global strict full-pixel feasibility diagnostic; no training or final-test access."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import linprog
import torch

from light_source import DifferentiableAbbeLitho, PixelatedLightSource, resist_image
from source_training import SourceDataset
from scripts import run_protected_pvband_experiment as experiment

GRID = 9
SIGMA_INNER = 0.3
SIGMA_OUTER = 0.9
MARGIN_TOL = 1e-8
RESIDUAL_TOL = 2e-8
MAX_BASIS_BYTES = 512 * 1024**2


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def extract_fit_payload(payload):
    """Read only fit; never inspect/index another split key."""
    return payload["fit"]


def load_fit_dataset(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return SourceDataset(**extract_fit_payload(payload))


def solve_max_margin(layouts, doses=(1.0,), threshold=0.225, time_limit=60.0,
                     method="highs", presolve=True):
    """LP over source weights and a free margin, with every pixel constrained."""
    if not layouts:
        raise ValueError("at least one layout is required")
    doses = tuple(float(x) for x in doses)
    if not doses or any(not math.isfinite(x) or x <= 0 for x in doses):
        raise ValueError("doses must be finite and positive")
    if not math.isfinite(float(threshold)) or time_limit <= 0:
        raise ValueError("threshold and time limit must be valid")
    if method not in ("highs", "highs-ipm") or not isinstance(presolve, bool):
        raise ValueError("method must be highs/highs-ipm and presolve must be boolean")
    prepared, count, rows = [], None, 0
    for item in layouts:
        basis = np.asarray(item["basis"], dtype=np.float64)
        target = np.asarray(item["target"])
        if basis.ndim != 3 or basis.shape[-2:] != target.shape:
            raise ValueError("basis must be (source,H,W) and target (H,W)")
        if not np.isfinite(basis).all() or not np.isfinite(target).all():
            raise ValueError("basis and target must be finite")
        if not np.isin(target, (0, 1, False, True)).all():
            raise ValueError("target must be binary")
        if count is None:
            count = basis.shape[0]
        elif count != basis.shape[0]:
            raise ValueError("all layouts must use the same source support")
        flat = basis.reshape(count, -1).T
        fg = target.astype(bool, copy=False).reshape(-1)
        prepared.append((flat, fg, ~fg))
        rows += fg.size
    if not count:
        raise ValueError("basis must have source samples")

    # FG: -min(dose)*B*s + m <= -threshold.
    # BG:  max(dose)*B*s + m <=  threshold.
    aub = np.empty((rows, count + 1), dtype=np.float64)
    bub = np.empty(rows, dtype=np.float64)
    cursor = 0
    for matrix, fg, bg in prepared:
        for mask, sign, dose, bound in (
            (fg, -1.0, min(doses), -float(threshold)),
            (bg, 1.0, max(doses), float(threshold)),
        ):
            n = int(mask.sum())
            if not n:
                continue
            end = cursor + n
            aub[cursor:end, :count] = sign * dose * matrix[mask]
            aub[cursor:end, count] = 1.0
            bub[cursor:end] = bound
            cursor = end
    c = np.zeros(count + 1, dtype=np.float64)
    c[-1] = -1.0
    aeq = np.zeros((1, count + 1), dtype=np.float64)
    aeq[0, :count] = 1.0
    result = linprog(
        c, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * count + [(None, None)],
        method=method,
        options={
            "time_limit": float(time_limit),
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
            "ipm_optimality_tolerance": 1e-10,
            "presolve": presolve,
        },
    )
    solver = {
        "status_code": int(result.status), "status": str(result.message),
        "success": bool(result.success), "iterations": int(getattr(result, "nit", 0) or 0),
        "constraint_rows": rows, "variables": count + 1,
        "method": "scipy.optimize.linprog(method=%r), float64" % method,
        "presolve": presolve,
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
        "ipm_optimality_tolerance": 1e-10,
        "fg_dose_bound": min(doses), "bg_dose_bound": max(doses),
    }
    if not result.success or result.x is None:
        return {"solver": solver, "lp_margin": None, "weights": None}
    raw = np.asarray(result.x[:count], dtype=np.float64)
    raw_sum = float(raw.sum())
    if raw.min(initial=0.0) < -1e-8 or not math.isfinite(raw_sum) or raw_sum <= 0:
        solver.update(success=False, status="solver returned invalid source weights")
        return {"solver": solver, "lp_margin": float(result.x[-1]), "weights": None}
    weights = np.maximum(raw, 0.0)
    weights /= weights.sum()
    solver["source_weight_sum_before_cleanup"] = raw_sum
    solver["source_weight_min_before_cleanup"] = float(raw.min())
    return {"solver": solver, "lp_margin": float(result.x[-1]), "weights": weights}


def _weight_grid(weights, support):
    result = np.zeros(support.shape, dtype=np.float64)
    result[support] = weights
    return result


def _gauges(weights, coords, support):
    radius = np.sqrt((coords**2).sum(axis=1))
    return {
        "flux_sum": float(weights.sum()),
        "active_weight_min": float(weights.min()),
        "active_weight_max": float(weights.max()),
        "support_pixels": int(support.sum()),
        "outside_support_flux": 0.0,
        "weighted_mean_sigma_radius": float(weights @ radius),
        "weighted_rms_sigma_radius": float(np.sqrt(weights @ radius**2)),
        "maximum_supported_sigma_radius": float(radius.max()),
    }


def _prepare_float32_basis(simulator, mask, verify_parity=False, max_bytes=MAX_BASIS_BYTES):
    basis = simulator.prepare_basis(mask, defocus_nm=0.0, max_bytes=max_bytes)
    if verify_parity:
        with torch.no_grad():
            torch.testing.assert_close(
                simulator(mask), simulator.evaluate_basis(basis),
                rtol=1e-5, atol=1e-6,
            )
    # Basis.to() mutates the module, so do this only after any parity check.
    return basis.cpu()


def _layout_rows(dataset, split):
    """Create explicit HxW targets and retain masks as 1xHxW for basis prep."""
    return [
        {
            "layout_id": layout_id,
            "split": split,
            "mask": dataset.masks[index],
            "target": dataset.targets[index, 0],
        }
        for index, layout_id in enumerate(dataset.layout_ids)
    ]


def _metrics(layout, basis, weights, doses):
    intensities = basis.intensities[0]
    weights_t = torch.as_tensor(weights, dtype=intensities.dtype, device=intensities.device)
    aerial = torch.einsum("nhw,n->hw", intensities, weights_t)
    target = torch.as_tensor(layout["target"], dtype=torch.bool, device=intensities.device)
    npos = int(target.sum().item())
    corners, binaries, slacks = [], [], []
    for dose in doses:
        printed = resist_image(aerial[None, None], dose=dose,
                               threshold=experiment.THRESHOLD,
                               steepness=experiment.STEEPNESS)[0, 0]
        binary = printed >= experiment.BINARY
        direct_binary = dose * aerial >= experiment.THRESHOLD
        fp = int((binary & ~target).sum().item())
        fn = int((~binary & target).sum().item())
        fg = float((dose * aerial[target] - experiment.THRESHOLD).min().item()) if npos else None
        bg = float((experiment.THRESHOLD - dose * aerial[~target]).min().item()) if bool((~target).any()) else None
        slack = min(v for v in (fg, bg) if v is not None)
        corners.append({
            "corner": "nominal" if dose == 1.0 else "d%g" % dose,
            "dose": float(dose),
            "predicted_positive_pixels": int(binary.sum().item()),
            "target_positive_pixels": npos,
            "false_positive_pixels": fp,
            "false_negative_pixels": fn,
            "recall": (npos - fn) / npos if npos else None,
            "L2_pixels": fp + fn,
            "minimum_foreground_signed_slack": fg,
            "minimum_background_signed_slack": bg,
            "minimum_signed_constraint_slack": slack,
            "intensity_vs_resist_hardprint_disagreements": int((binary != direct_binary).sum().item()),
            "violation_beyond_lp_margin": None,
        })
        binaries.append(binary)
        slacks.append(slack)
    nominal = next((i for i, d in enumerate(doses) if d == 1.0), None)
    band = int((torch.stack(binaries).any(0) != torch.stack(binaries).all(0)).sum().item()) if len(doses) > 1 else 0
    return {
        "layout_id": layout["layout_id"], "split": layout["split"],
        "target_positive_pixels": npos,
        "L2_pixels": corners[nominal]["L2_pixels"] if nominal is not None else None,
        "L2_worst_dose_pixels": max(c["L2_pixels"] for c in corners),
        "band_pixels": band, "per_corner": corners,
        "minimum_direct_margin": float(min(slacks)),
    }


def _summary(records):
    out = {}
    for split in ("fit", "calibration"):
        group = [x for x in records if x["split"] == split]
        if group:
            out[split] = {
                key: float(sum(x[key] for x in group) / len(group))
                for key in ("L2_pixels", "L2_worst_dose_pixels", "band_pixels")
                if all(x[key] is not None for x in group)
            }
    return out


def _evaluate(layouts, bases, weights, doses, lp_margin=None):
    # Optical simulation and hard prints remain float32, matching the registered
    # experiment. Independently verify constraints in float64 from those bases.
    records = [_metrics(x, bases[x["layout_id"]], weights, doses) for x in layouts]
    direct_margins = []
    for layout, record in zip(layouts, records):
        basis64 = bases[layout["layout_id"]].intensities[0].detach().cpu().numpy().astype(np.float64)
        aerial64 = np.einsum("n,nhw->hw", np.asarray(weights, dtype=np.float64), basis64)
        target = torch.as_tensor(layout["target"]).detach().cpu().numpy().astype(bool)
        for corner in record["per_corner"]:
            dose = corner["dose"]
            fg = float((dose * aerial64[target] - experiment.THRESHOLD).min()) if target.any() else None
            bg = float((experiment.THRESHOLD - dose * aerial64[~target]).min()) if (~target).any() else None
            slack = min(x for x in (fg, bg) if x is not None)
            corner["minimum_foreground_signed_slack"] = fg
            corner["minimum_background_signed_slack"] = bg
            corner["minimum_signed_constraint_slack"] = slack
            corner["violation_beyond_lp_margin"] = None
            direct_margins.append(slack)
        record["minimum_direct_margin"] = min(c["minimum_signed_constraint_slack"] for c in record["per_corner"])
    direct_margin = min(direct_margins)
    if lp_margin is not None:
        for record in records:
            for corner in record["per_corner"]:
                corner["violation_beyond_lp_margin"] = max(
                    0.0, lp_margin - corner["minimum_signed_constraint_slack"]
                )
        violation = max(0.0, lp_margin - direct_margin)
        residual = {
            "minimum_signed_constraint_slack": float(direct_margin),
            "lp_margin_minus_direct_margin": float(lp_margin - direct_margin),
            "maximum_positive_constraint_violation": float(violation),
            "tolerance": RESIDUAL_TOL, "passed": violation <= RESIDUAL_TOL,
        }
    else:
        residual = None
    return records, direct_margin, residual


def _solve_verified_lp(lp_layouts, selected, bases, doses, initial_time_limit,
                       retry_time_limit):
    """Retry a successful LP only when independent direct residuals fail."""
    initial = solve_max_margin(
        lp_layouts, doses, experiment.THRESHOLD, initial_time_limit,
        method="highs", presolve=True,
    )
    if initial["weights"] is None:
        return {
            "result": initial, "records": None, "direct_margin": None,
            "residual": None, "numerical_retry": None, "solver_call_count": 1,
        }
    records, direct_margin, residual = _evaluate(
        selected, bases, initial["weights"], doses, initial["lp_margin"]
    )
    chosen = initial
    retry_record = None
    calls = 1
    if initial["solver"]["success"] and not residual["passed"]:
        initial_snapshot = {
            "lp_optimal_margin": initial["lp_margin"],
            "direct_verified_margin": direct_margin,
            "direct_residual": residual,
            "solver": initial["solver"],
        }
        limit = float(retry_time_limit())
        if limit <= 0:
            retry_record = {
                "method": "highs-ipm", "presolve": False,
                "status": "global_timeout_before_retry",
            }
        else:
            retry = solve_max_margin(
                lp_layouts, doses, experiment.THRESHOLD, limit,
                method="highs-ipm", presolve=False,
            )
            calls += 1
            retry_record = {
                "method": retry["solver"]["method"],
                "presolve": retry["solver"]["presolve"],
                "solver": retry["solver"],
                "lp_optimal_margin": retry["lp_margin"],
            }
            if retry["weights"] is not None:
                retry_records, retry_margin, retry_residual = _evaluate(
                    selected, bases, retry["weights"], doses, retry["lp_margin"]
                )
                retry_record.update({
                    "direct_verified_margin": retry_margin,
                    "direct_residual": retry_residual,
                })
                if retry["solver"]["success"] and retry_residual["passed"]:
                    chosen = retry
                    records, direct_margin, residual = (
                        retry_records, retry_margin, retry_residual
                    )
                    retry_record["selected"] = True
                else:
                    retry_record["selected"] = False
            else:
                retry_record["selected"] = False
        retry_record["reason"] = "initial_numerical_verification_failed"
        retry_record["initial"] = initial_snapshot
        retry_record.setdefault("selected", False)
    return {
        "result": chosen, "records": records, "direct_margin": direct_margin,
        "residual": residual, "numerical_retry": retry_record,
        "solver_call_count": calls,
    }


def _status(lp_margin, residual_passed=True):
    if not residual_passed:
        return "numerical_verification_failed"
    if lp_margin > MARGIN_TOL:
        return "positive_margin_feasible"
    if lp_margin < -MARGIN_TOL:
        return "infeasible_under_strict_full_pixel_constraints"
    return "indeterminate_boundary"


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def run(args):
    dataset_path = Path(args.dataset_file).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(str(dataset_path))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.solver_time_limit <= 0 or args.timeout_seconds <= 0:
        raise ValueError("time limits must be positive")
    device = torch.device(args.device)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir()
    path = run_dir / "diagnostic.json"
    started = time.monotonic()
    deadline = started + args.timeout_seconds
    report = {
        "schema_version": 1, "status": "loading_fit_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "result_directory": str(run_dir), "heldout_status": "not_accessed",
        "input": {"dataset_file": str(dataset_path), "dataset_sha256": sha256_file(dataset_path),
                  "fit_masks": [], "fit_targets": [], "calibration_masks": [],
                  "calibration_targets": [], "bases": []},
        "protocol": {
            "device": str(device), "only_previous_split": "fit",
            "calibration_role": "development only; pooled scenarios do not measure generalization",
            "threshold": experiment.THRESHOLD, "steepness": experiment.STEEPNESS,
            "binary_threshold": experiment.BINARY, "doses": list(experiment.DOSES),
            "solver": "scipy.optimize.linprog(method='highs'), float64",
            "solver_primal_feasibility_tolerance": 1e-9,
            "solver_dual_feasibility_tolerance": 1e-9,
            "solver_ipm_optimality_tolerance": 1e-10,
            "numerical_retry": "only after successful highs solve fails float64 direct residual tolerance: highs-ipm with presolve=False, unchanged constraints and tolerances",
            "solver_time_limit_seconds_per_lp": args.solver_time_limit,
            "global_timeout_seconds": args.timeout_seconds,
            "margin_tolerance": MARGIN_TOL,
            "negative_margin_interpretation": "infeasible only under strict full-pixel constraints; does not prove the experiment's permissive gate impossible",
            "training": "none; no Adam or source fitting",
        },
        "physical_parameters": None, "baselines": {}, "scenarios": {},
    }
    _atomic_json(path, report)

    payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    fit = SourceDataset(**extract_fit_payload(payload))
    if len(fit.masks) != 4 or fit.pixel_size_nm != experiment.PIXEL or tuple(fit.targets.shape[-2:]) != (experiment.RASTER, experiment.RASTER):
        raise ValueError("previous fit must contain four registered 128x128 layouts at the experiment pixel size")
    for i, layout_id in enumerate(fit.layout_ids):
        report["input"]["fit_masks"].append({"layout_id": layout_id, "sha256": sha256_tensor(fit.masks[i])})
        report["input"]["fit_targets"].append({"layout_id": layout_id, "sha256": sha256_tensor(fit.targets[i])})

    teacher, calibration, _rows = experiment.generate_calibration(device)
    if len(calibration.masks) != 4 or set(fit.layout_ids) & set(calibration.layout_ids):
        raise ValueError("expected four distinct regenerated calibration layouts")
    for i, layout_id in enumerate(calibration.layout_ids):
        report["input"]["calibration_masks"].append({"layout_id": layout_id, "sha256": sha256_tensor(calibration.masks[i])})
        report["input"]["calibration_targets"].append({"layout_id": layout_id, "sha256": sha256_tensor(calibration.targets[i])})

    source = PixelatedLightSource(GRID, sigma_inner=SIGMA_INNER, sigma_outer=SIGMA_OUTER).to(device)
    simulator = DifferentiableAbbeLitho(
        source, numerical_aperture=1.35, wavelength_nm=193.0,
        pixel_size_nm=experiment.PIXEL, source_chunk_size=8, cache_max_bytes=0,
    ).to(device)
    support = simulator.source._pupil_support.detach().cpu().numpy().astype(bool)
    coords = simulator.source._coordinates.detach().cpu().numpy()[support]
    report["physical_parameters"] = {
        **simulator.optical_config(), "raster": [experiment.RASTER, experiment.RASTER],
        "source_grid": GRID, "sigma_inner": SIGMA_INNER, "sigma_outer": SIGMA_OUTER,
        "active_source_samples": int(support.sum()), "threshold": experiment.THRESHOLD,
        "steepness": experiment.STEEPNESS, "binary_threshold": experiment.BINARY,
        "doses": list(experiment.DOSES),
    }
    layouts = _layout_rows(fit, "fit") + _layout_rows(calibration, "calibration")

    bases = {}
    for item in layouts:
        if time.monotonic() >= deadline:
            report["status"] = "timeout_during_basis_preparation"
            _atomic_json(path, report)
            return path
        mask = item["mask"][None].to(device=device, dtype=torch.float32)
        basis = _prepare_float32_basis(
            simulator, mask,
            verify_parity=item["layout_id"] in (fit.layout_ids[0], calibration.layout_ids[0]),
        )
        bases[item["layout_id"]] = basis
        report["input"]["bases"].append({
            "layout_id": item["layout_id"], "split": item["split"],
            "sha256": sha256_tensor(basis.intensities),
            "shape": list(basis.intensities.shape), "dtype": str(basis.intensities.dtype),
        })
        report["status"] = "bases_prepared:%s" % item["layout_id"]
        _atomic_json(path, report)

    # Baseline distributions are diagnostic controls only: A0 has no jitter.
    initial = PixelatedLightSource(GRID, sigma_inner=SIGMA_INNER, sigma_outer=SIGMA_OUTER)
    _, init_w = initial.distribution()
    _, teacher_w = teacher.source.distribution()
    baseline_weights = {
        "A0_initial_annulus_no_jitter": init_w.detach().cpu().double().numpy(),
        "known_teacher": teacher_w.detach().cpu().double().numpy(),
    }
    all_doses = list(experiment.DOSES)
    for name, weights in baseline_weights.items():
        entry = {"source_weights_full_grid": _weight_grid(weights, support).tolist(),
                 "source_gauges": _gauges(weights, coords, support)}
        for mode, doses in (("nominal", [1.0]), ("robust3doses", all_doses)):
            records, _, _ = _evaluate(layouts, bases, weights, doses)
            entry[mode] = {"metrics_by_split": _summary(records), "per_layout_corners": records}
        report["baselines"][name] = entry
        _atomic_json(path, report)

    def lp_data(selected):
        return [{"basis": bases[x["layout_id"]].intensities[0].numpy(),
                 "target": x["target"].numpy()} for x in selected]

    def solve_case(selected, doses):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"status": "global_timeout"}
        resolved = _solve_verified_lp(
            lp_data(selected), selected, bases, doses,
            min(args.solver_time_limit, remaining),
            lambda: min(args.solver_time_limit, max(0.0, deadline - time.monotonic())),
        )
        result = resolved["result"]
        if result["weights"] is None:
            failed = {"status": "solver_not_optimal", "solver": result["solver"],
                      "lp_optimal_margin": result["lp_margin"],
                      "solver_call_count": resolved["solver_call_count"]}
            if resolved["numerical_retry"] is not None:
                failed["numerical_retry"] = resolved["numerical_retry"]
            return failed
        records = resolved["records"]
        direct_margin = resolved["direct_margin"]
        residual = resolved["residual"]
        scenario = {
            "status": _status(result["lp_margin"], residual["passed"]), "lp_optimal_margin": result["lp_margin"],
            "direct_verified_margin": direct_margin, "margin_tolerance": MARGIN_TOL,
            "direct_residual": residual, "solver": result["solver"],
            "solver_call_count": resolved["solver_call_count"],
            "source_weights_full_grid": _weight_grid(result["weights"], support).tolist(),
            "source_gauges": _gauges(result["weights"], coords, support),
            "dose_set": list(doses), "metrics_by_split": _summary(records),
            "per_layout_corners": records,
        }
        if resolved["numerical_retry"] is not None:
            scenario["numerical_retry"] = resolved["numerical_retry"]
        if any(x["split"] == "calibration" for x in selected):
            scenario["calibration_interpretation"] = "development constraints only; not a generalization result"
        elif len(selected) == 4 and doses != [1.0]:
            cal_records, cal_margin, _ = _evaluate(layouts[4:], bases, result["weights"], all_doses)
            scenario["calibration_diagnostic_only"] = {
                "role": "evaluation only; calibration did not enter this LP and was not used to choose or tune weights",
                "dose_set": all_doses,
                "metrics_by_split": _summary(cal_records),
                "per_layout_corners": cal_records,
                "minimum_direct_margin": cal_margin,
            }
        if len(selected) > 1 and doses == [1.0]:
            eval_records, eval_margin, _ = _evaluate(layouts, bases, result["weights"], all_doses)
            scenario["evaluation_3doses"] = {
                "role": "diagnostic evaluation only; these dose corners do not change the nominal LP",
                "dose_set": all_doses,
                "metrics_by_split": _summary(eval_records),
                "per_layout_corners": eval_records,
                "minimum_direct_margin": eval_margin,
            }
        return scenario

    fit_layouts = layouts[:4]
    pooled = [
        ("fit_only.nominal", fit_layouts, [1.0]),
        ("fit_only.robust3doses", fit_layouts, all_doses),
        ("pooled8_development.nominal", layouts, [1.0]),
        ("pooled8_development.robust3doses", layouts, all_doses),
    ]
    for name, selected, doses in pooled:
        report["status"] = "solving:%s" % name
        _atomic_json(path, report)
        report["scenarios"][name] = solve_case(selected, doses)
        _atomic_json(path, report)
    report["scenarios"]["individual_layouts"] = {}
    for item in layouts:
        modes = {"split": item["split"]}
        for mode, doses in (("nominal", [1.0]), ("robust3doses", all_doses)):
            modes[mode] = solve_case([item], doses)
            report["scenarios"]["individual_layouts"][item["layout_id"]] = modes
            _atomic_json(path, report)
    cases = [report["scenarios"][name] for name, _, _ in pooled]
    for layout_id in (x["layout_id"] for x in layouts):
        cases.extend(report["scenarios"]["individual_layouts"][layout_id][mode]
                     for mode in ("nominal", "robust3doses"))
    if any(case.get("status") == "global_timeout" for case in cases):
        report["status"] = "timeout"
    elif len(cases) != 20 or any(
        case.get("status") not in ("positive_margin_feasible", "infeasible_under_strict_full_pixel_constraints", "indeterminate_boundary")
        or not case.get("direct_residual", {}).get("passed", False)
        for case in cases
    ):
        report["status"] = "partial_solver_or_verification_failure"
    else:
        report["status"] = "complete"
    report["completed_lp_cases"] = sum(case.get("status") != "global_timeout" for case in cases)
    report["elapsed_seconds"] = float(time.monotonic() - started)
    report["heldout_status"] = "not_accessed"
    _atomic_json(path, report)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--solver-time-limit", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)
    path = run(args)
    print("Diagnostic:", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
