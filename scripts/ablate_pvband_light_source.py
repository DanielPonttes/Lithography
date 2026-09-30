"""Short, preregistered synthetic dose-only PV-band ablation.

All quality fits are intentionally short and require the local RTX 5060 Ti.
The three final layouts are target-generated before training and are evaluated
only if a calibration arm passes the fixed selection rule.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from light_source import DifferentiableAbbeLitho, PixelatedLightSource, resist_image
from source_training import (
    ProcessCorner, SourceDataset, SourceFitConfig, _cpu_state, _prepare,
    evaluate_source, fit_source, validate_splits,
)

RESULTS_ROOT = Path("D:/Codex/Lithography/work/pvband_ablation")
SEEDS = (17, 29, 43)
PIXEL_SIZE_NM = 4.0
RASTER = 128
CORNERS = (
    ProcessCorner("d0.98", dose=0.98),
    ProcessCorner("nominal", dose=1.0),
    ProcessCorner("d1.02", dose=1.02),
)
LEGACY_SEEN = ("val_vls_pitch22_width7", "val_crossbar_endcaps", "val_chevron_contacts")
ARMS = {
    "A0": {"objective": "envelope_squared", "band_weight": 0.5},
    "A1": {"objective": "envelope_squared", "band_weight": 2.0},
    "A2": {"objective": "worst_dose_surrogate", "surrogate_steepness": 50.0,
           "surrogate_band_weight": 0.5},
    "A3": {"objective": "worst_dose"},
}


def sha256_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def rect(mask, y0, y1, x0, x1):
    h, w = mask.shape
    y0, y1 = min(h, max(0, y0)), min(h, max(0, y1))
    x0, x1 = min(w, max(0, x0)), min(w, max(0, x1))
    if y0 < y1 and x0 < x1:
        mask[y0:y1, x0:x1] = 1.0


def make_layouts(size=RASTER):
    """Exact existing fit/calibration geometries plus three fixed new tests."""
    rows = []

    def fresh(layout_id, family, geometry):
        mask = torch.zeros((size, size), dtype=torch.float32)
        rows.append({"layout_id": layout_id, "family": family,
                     "geometry": geometry, "mask": mask})
        return mask

    mask = fresh("train_vls_pitch24_width9", "vertical_line_space",
                 "vertical clear bars, 36 nm width and 96 nm pitch")
    for x in range(8, size, 24):
        rect(mask, 0, size, x, x + 9)

    mask = fresh("train_hls_pitch28_width10", "horizontal_line_space",
                 "horizontal clear bars, 40 nm width and 112 nm pitch")
    for y in range(10, size, 28):
        rect(mask, y, y + 10, 0, size)

    mask = fresh("train_l_contours", "line_contour",
                 "three separated L contours with 40 nm line width")
    for x, y in ((18, 18), (61, 36), (94, 74)):
        rect(mask, y, y + 10, x, x + 42)
        rect(mask, y, y + 42, x, x + 10)

    mask = fresh("train_t_junctions", "junction",
                 "T junctions with 36 nm trunk and 40 nm cap widths")
    for x, y in ((25, 18), (76, 62)):
        rect(mask, y, y + 52, x, x + 9)
        rect(mask, y, y + 9, x - 17, x + 26)

    mask = fresh("train_serpentine_line_ends", "line_end_and_jog",
                 "two jogged contours with capped line ends")
    for x0, y0 in ((13, 20), (71, 67)):
        rect(mask, y0, y0 + 9, x0, x0 + 37)
        rect(mask, y0, y0 + 31, x0 + 28, x0 + 37)
        rect(mask, y0 + 22, y0 + 31, x0 + 28, x0 + 58)

    mask = fresh("train_contact_array_pitch30", "contact_array",
                 "3 by 3 array of 112 nm square contacts at 176 nm pitch")
    for y in (6, 50, 94):
        for x in (6, 50, 94):
            rect(mask, y, y + 28, x, x + 28)

    mask = fresh("test_lines_diagonal_space", "lines",
                 "three shallow diagonal clear bands, each 56 nm wide, clipped to the raster; predeclared test")
    for y in range(size):
        for x0 in (18 + y // 4, 62 - y // 5, 105 + y // 7):
            rect(mask, y, y + 1, x0 - 7, x0 + 7)

    mask = fresh("test_contacts_staggered_array", "contacts",
                 "nine 84 nm square contacts in a staggered 3 by 3 array; predeclared test")
    for row, y in enumerate((14, 53, 92)):
        shift = (0, 12, -5)[row]
        for x in (8 + shift, 52 + shift, 96 + shift):
            rect(mask, y, y + 21, x, x + 21)

    mask = fresh("test_junction_asymmetric_cross", "junctions",
                 "asymmetric cross junction with offset side branches and varied capped arms; predeclared test")
    rect(mask, 18, 112, 60, 68)
    rect(mask, 58, 66, 17, 112)
    rect(mask, 30, 38, 34, 60)
    rect(mask, 88, 96, 68, 101)
    return rows[:4], rows[4:6], rows[6:]


def teacher_source(device, grid_size=9):
    source = PixelatedLightSource(grid_size=grid_size, sigma_inner=0.3, sigma_outer=0.9)
    coords, support = source._coordinates, source._pupil_support
    x, y = coords[..., 0], coords[..., 1]
    lobe_a = -((x - 0.45).square() + (y - 0.225).square()) / (2 * 0.24**2)
    lobe_b = -((x + 0.225).square() + (y + 0.45).square()) / (2 * 0.27**2)
    with torch.no_grad():
        logits = torch.logaddexp(lobe_a, lobe_b + math.log(0.55))
        source.logits.copy_(torch.where(support, logits, source.logits))
    return source.to(device)


def make_simulator(device, seed):
    source = PixelatedLightSource(grid_size=9, sigma_inner=0.3, sigma_outer=0.9)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(source.logits.shape, generator=generator, dtype=source.logits.dtype)
    with torch.no_grad():
        source.logits.add_(noise * 0.02 * source._pupil_support)
    return DifferentiableAbbeLitho(
        source, numerical_aperture=1.35, wavelength_nm=193.0,
        pixel_size_nm=PIXEL_SIZE_NM, source_chunk_size=8, cache_max_bytes=0,
    ).to(device)


def materialize_datasets(device, config):
    fit_rows, calibration_rows, test_rows = make_layouts()
    teacher = DifferentiableAbbeLitho(
        teacher_source(device), numerical_aperture=1.35, wavelength_nm=193.0,
        pixel_size_nm=PIXEL_SIZE_NM, source_chunk_size=8, cache_max_bytes=0,
    ).to(device)
    all_rows = fit_rows + calibration_rows + test_rows
    with torch.no_grad():
        for row in all_rows:
            aerial = teacher(row["mask"].to(device))
            soft = resist_image(aerial, dose=1.0, threshold=config.threshold,
                                steepness=config.steepness)
            row["target"] = (soft >= config.binary_threshold).float().cpu()[0]
    splits = {
        "fit": SourceDataset(
            torch.stack([r["mask"] for r in fit_rows]),
            torch.stack([r["target"] for r in fit_rows]),
            tuple(r["layout_id"] for r in fit_rows), PIXEL_SIZE_NM,
        ),
        "calibration": SourceDataset(
            torch.stack([r["mask"] for r in calibration_rows]),
            torch.stack([r["target"] for r in calibration_rows]),
            tuple(r["layout_id"] for r in calibration_rows), PIXEL_SIZE_NM,
        ),
        "final_test": SourceDataset(
            torch.stack([r["mask"] for r in test_rows]),
            torch.stack([r["target"] for r in test_rows]),
            tuple(r["layout_id"] for r in test_rows), PIXEL_SIZE_NM,
        ),
    }
    return teacher, splits, all_rows


def preflight(splits, rows):
    validate_splits(splits["fit"], splits["calibration"])
    ids = [layout_id for split in splits.values() for layout_id in split.layout_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("layout IDs overlap across fit, calibration, and final test")
    mask_hashes, target_hashes = [], []
    for split_name, dataset in splits.items():
        if dataset.pixel_size_nm != PIXEL_SIZE_NM or tuple(dataset.masks.shape[-2:]) != (RASTER, RASTER):
            raise ValueError("unexpected pixel size or raster shape in " + split_name)
        counts = [int(t.sum().item()) for t in dataset.targets]
        fractions = [float(t.mean().item()) for t in dataset.targets]
        if any(not 0.01 <= v <= 0.99 for v in fractions):
            raise ValueError("target failed fixed 1%-99% nontriviality screen in " + split_name)
        if len({sha256_tensor(t) for t in dataset.targets}) != len(dataset.targets):
            raise ValueError("duplicate targets inside " + split_name)
        if split_name != "fit" and len(dataset.targets) < 2:
            raise ValueError("held-out split must contain at least two targets")
        target_hashes.extend(sha256_tensor(t) for t in dataset.targets)
        mask_hashes.extend(sha256_tensor(m) for m in dataset.masks)
        dataset._preflight_counts = counts
    if len(set(mask_hashes)) != len(mask_hashes):
        raise ValueError("identical masks overlap across splits")
    if len(set(target_hashes)) != len(target_hashes):
        raise ValueError("identical targets overlap across splits")
    by_id = {row["layout_id"]: row for row in rows}
    result = {}
    for split_name, dataset in splits.items():
        result[split_name] = []
        for index, layout_id in enumerate(dataset.layout_ids):
            row = by_id[layout_id]
            result[split_name].append({
                "layout_id": layout_id, "family": row["family"],
                "geometry": row["geometry"],
                "mask_sha256": sha256_tensor(dataset.masks[index]),
                "target_sha256": sha256_tensor(dataset.targets[index]),
                "target_positive_pixels": dataset._preflight_counts[index],
                "target_positive_fraction": float(dataset.targets[index].mean().item()),
            })
    return result


def config_for(seed, steps, arm):
    values = dict(
        steps=steps, learning_rate=0.01, threshold=0.225, steepness=50.0,
        binary_threshold=0.5, max_basis_bytes=512 * 1024**2,
        max_total_basis_bytes=512 * 1024**2, max_device_basis_bytes=256 * 1024**2,
        seed=seed, verify_basis=True,
    )
    values.update(ARMS[arm])
    return SourceFitConfig(**values)


def aggregate(records, split, metric):
    values = [run["fit_report"]["after"][split]["mean"][metric]
              for run in records]
    return sum(values) / len(values)


def choose_arm(arm_runs):
    baseline = arm_runs["A0"]
    a0_band = aggregate(baseline, "validation", "band_pixels")
    a0_nominal = aggregate(baseline, "validation", "L2_pixels")
    a0_worst = aggregate(baseline, "validation", "L2_worst_dose_pixels")
    checks = {}
    for arm in ("A1", "A2", "A3"):
        runs = arm_runs[arm]
        band = aggregate(runs, "validation", "band_pixels")
        nominal = aggregate(runs, "validation", "L2_pixels")
        worst = aggregate(runs, "validation", "L2_worst_dose_pixels")
        seed_improvements = [
            run["fit_report"]["after"]["validation"]["mean"]["band_pixels"]
            < baseline[index]["fit_report"]["after"]["validation"]["mean"]["band_pixels"]
            for index, run in enumerate(runs)
        ]
        checks[arm] = {
            "pv_band_mean_reduction_at_least_10pct": band <= 0.90 * a0_band,
            "nominal_L2_mean_at_most_5pct_worse": nominal <= 1.05 * a0_nominal,
            "worst_dose_L2_mean_at_most_5pct_worse": worst <= 1.05 * a0_worst,
            "pv_band_improves_in_each_seed": all(seed_improvements),
            "per_seed_pv_band_improves": seed_improvements,
            "calibration_means": {"band_pixels": band, "L2_pixels": nominal,
                                  "L2_worst_dose_pixels": worst},
        }
    eligible = [arm for arm, check in checks.items()
                if check["pv_band_mean_reduction_at_least_10pct"]
                and check["nominal_L2_mean_at_most_5pct_worse"]
                and check["worst_dose_L2_mean_at_most_5pct_worse"]
                and check["pv_band_improves_in_each_seed"]]
    selected = min(eligible, key=lambda arm: (
        checks[arm]["calibration_means"]["band_pixels"],
        checks[arm]["calibration_means"]["L2_pixels"], arm,
    )) if eligible else None
    return {
        "baseline_A0_calibration_means": {"band_pixels": a0_band,
                                          "L2_pixels": a0_nominal,
                                          "L2_worst_dose_pixels": a0_worst},
        "candidate_checks": checks,
        "eligible_arms": eligible,
        "selected_arm": selected,
        "selection_rule": (
            "Among A1-A3, require mean calibration PV-band <= 0.90*A0, mean nominal and worst-dose L2 <= 1.05*A0, "
            "and lower PV-band than A0 for every seed; choose the eligible arm with lowest mean PV-band, then "
            "lowest nominal L2, then lexicographic arm ID. All comparisons use only the two calibration layouts."
        ),
    }


def selected_final_test_arm(selection):
    """Return a frozen, recognized candidate; otherwise keep final-test gate closed."""
    if not isinstance(selection, dict):
        return None
    arm = selection.get("selected_arm")
    return arm if arm in ARMS else None


def eval_final_test(sim, dataset, config):
    prepared, verification = _prepare(sim, dataset, (0.0,), config)
    try:
        result = evaluate_source(sim, dataset, prepared, CORNERS, config)
    finally:
        for bases in prepared:
            for basis in bases.values():
                basis.cpu()
    result["basis_verification"] = verification
    return result


def require_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this ablation requires the local RTX 5060 Ti")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    if name != "NVIDIA GeForce RTX 5060 Ti":
        raise RuntimeError("refusing to train on unexpected GPU: " + name)
    torch.cuda.synchronize(device)
    return device, name


def run(args):
    device, gpu_name = require_device()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / (stamp + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "ablation.json"
    report = {
        "schema_version": 1, "status": "preflight_running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "result_directory": str(run_dir),
        "system": {"python": sys.version, "platform": platform.platform(),
                   "torch": str(torch.__version__), "cuda": torch.version.cuda,
                   "gpu_name": gpu_name, "device": str(device)},
        "scope": {
            "backend": "experimental_scalar_abbe", "synthetic_targets_only": True,
            "uses_real_training_data": False, "socs_parity_claim": False,
            "EPE": None, "shots": None, "SOCS": "not evaluated",
            "limitations": [
                "sample is small and synthetic; the result does not validate fab performance or the paper hypothesis",
                "EPE, shot count, and SOCS are not evaluated",
            ],
        },
        "protocol": {
            "working_directory": str(Path.cwd()),
            "command": "%s scripts/ablate_pvband_light_source.py --steps %d --seeds 17 29 43 --output-root %s" %
                       (sys.executable, args.steps, str(args.output_root)),
            "raster_shape": [RASTER, RASTER], "pixel_size_nm": PIXEL_SIZE_NM,
            "threshold": 0.225, "resist_steepness": 50.0, "binary_threshold": 0.5,
            "corners": [{"name": c.name, "dose": c.dose, "defocus_nm": c.defocus_nm}
                        for c in CORNERS],
            "focus": "none", "teacher_target_dose": 1.0,
            "seeds": list(args.seeds), "steps": args.steps,
            "optimizer": {"name": "Adam", "learning_rate": 0.01},
            "initialization": "same 0.02 Gaussian logit jitter per seed and arm; exact initial-logit hashes checked",
            "fit_layout_ids": ["train_vls_pitch24_width9", "train_hls_pitch28_width10",
                               "train_l_contours", "train_t_junctions"],
            "calibration_layout_ids": ["train_serpentine_line_ends", "train_contact_array_pitch30"],
            "legacy_seen_excluded": list(LEGACY_SEEN),
            "arms": ARMS,
            "A2_definition": "worst-dose per-pixel MSE plus 0.5 * mean(max_c q_c - min_c q_c), q_c=sigmoid(50*(printed_c-0.5)); this is zero for identical corner prints and approaches the binary corner-envelope indicator",
            "A3_definition": "worst-dose per-pixel MSE only; ablation isolates the surrogate-band term",
            "selection_rule_frozen_before_fit": (
                "A1-A3 require calibration mean PV-band reduction >=10% vs A0, nominal and worst-dose L2 <=5% worse, "
                "and per-seed PV-band improvement; choose lowest eligible calibration PV-band, then nominal L2, then ID"
            ),
            "preflight_target_screen": "each target positive fraction 1%-99%; no repeated masks, targets, or layout IDs across any split",
            "posthoc_binary_metrics": (
                "For every layout and process corner, report predicted positive pixels and target positive pixels, "
                "plus false positives and false negatives; these diagnostics do not enter the objective or selection."
            ),
            "preflight_design_history": [{
                "layout_id": "test_lines_diagonal_space",
                "discarded_before_training": "two 32 nm diagonal bands yielded 46/16384 positive target pixels (0.28%)",
                "reason": "failed the predeclared 1%-99% nontriviality screen; no fit or final-test metrics had been viewed",
                "final_geometry": "three shallow diagonal clear bands, each 56 nm wide",
            }],
        },
        "layouts": {}, "initial_logit_hashes": {}, "arms_run": {},
        "selection": None, "final_test": None,
    }
    write_json(report_path, report)
    try:
        base_config = SourceFitConfig(steps=1, learning_rate=0.01, band_weight=0.5,
                                      threshold=0.225, steepness=50.0, binary_threshold=0.5,
                                      max_basis_bytes=512 * 1024**2,
                                      max_total_basis_bytes=512 * 1024**2,
                                      max_device_basis_bytes=256 * 1024**2)
        teacher, splits, rows = materialize_datasets(device, base_config)
        layouts = preflight(splits, rows)
        report["layouts"] = layouts
        report["teacher_source_weights"] = teacher.source.weight_map().detach().cpu().tolist()
        torch.save({name: split.payload() for name, split in splits.items()}, run_dir / "datasets.pt")
        del teacher
        report["status"] = "preflight_passed"
        write_json(report_path, report)

        states = {}
        for seed in args.seeds:
            seed_hashes = {}
            reference = None
            for arm in ARMS:
                sim = make_simulator(device, seed)
                state_hash = sha256_tensor(sim.source.logits)
                seed_hashes[arm] = state_hash
                if reference is None:
                    reference = state_hash
                elif state_hash != reference:
                    raise AssertionError("initial source logits differ across arms for seed %d" % seed)
                del sim
            report["initial_logit_hashes"][str(seed)] = seed_hashes
        report["status"] = "fitting"
        write_json(report_path, report)

        arm_runs = {arm: [] for arm in ARMS}
        for arm in ARMS:
            report["arms_run"][arm] = []
            for seed in args.seeds:
                sim = make_simulator(device, seed)
                cfg = config_for(seed, args.steps, arm)
                fit_report = fit_source(sim, splits["fit"], splits["calibration"],
                                        corners=CORNERS, config=cfg)
                run_record = {
                    "seed": seed,
                    "initial_logits_sha256": report["initial_logit_hashes"][str(seed)][arm],
                    "fit_report": fit_report,
                    "final_source_state": _cpu_state(sim.source),
                }
                arm_runs[arm].append(run_record)
                report["arms_run"][arm].append({
                    "seed": seed,
                    "initial_logits_sha256": run_record["initial_logits_sha256"],
                    "fit_report": fit_report,
                })
                write_json(report_path, report)
                del sim
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)

        report["selection"] = choose_arm(arm_runs)
        selected = selected_final_test_arm(report["selection"])
        report["status"] = "selected_for_final_test" if selected else "no_arm_met_calibration_rule"
        write_json(report_path, report)
        if selected is None:
            report["final_test"] = {
                "status": "not_run_no_arm_met_calibration_rule",
                "explanation": "The fixed calibration criteria rejected A1, A2, and A3; the new final test remains unopened.",
            }
            report["status"] = "done_no_qualifying_arm"
            write_json(report_path, report)
            return report

        selected_runs = []
        chosen_config_template = ARMS[selected]
        for record in arm_runs[selected]:
            sim = make_simulator(device, record["seed"])
            sim.source.load_state_dict(record["final_source_state"])
            cfg = config_for(record["seed"], args.steps, selected)
            test_metrics = eval_final_test(sim, splits["final_test"], cfg)
            selected_runs.append({"seed": record["seed"], "metrics": test_metrics})
            del sim
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
        report["final_test"] = {
            "status": "evaluated_after_calibration_freeze",
            "frozen_arm": selected,
            "frozen_arm_config": chosen_config_template,
            "runs": selected_runs,
        }
        report["status"] = "done"
        write_json(report_path, report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(report_path, report)
        (run_dir / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs=3, default=list(SEEDS))
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args(argv)
    if args.steps != 40:
        parser.error("the preregistered ablation fixes exactly 40 Adam steps")
    if tuple(args.seeds) != SEEDS:
        parser.error("the preregistered ablation fixes seeds 17, 29, 43 in that order")
    report = run(args)
    print("Status:", report["status"])
    print("Report:", (Path(report["result_directory"]) / "ablation.json").resolve())
    print("Selection:", (report.get("selection") or {}).get("selected_arm"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
