"""Paired end-to-end A/B benchmark for source-basis residency in fit_source.

Only the training-basis residency policy changes between arms. The fixed-mask
synthetic fixtures are reused from benchmark_light_source.py; results do not
establish SOCS parity, EPE, shots, or performance on real LithoBench data.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from benchmark_light_source import (
    CORNERS, DEFAULT_SEEDS, initial_validation, make_datasets, make_simulator, tensor_hash,
)
from source_training import SourceFitConfig, fit_source, validate_splits


HARD_COUNT_FIELDS = {
    "L2_pixels", "L2_worst_dose_pixels", "band_pixels", "flip_window_pixels",
    "predicted_positive_pixels", "target_positive_pixels",
    "false_positive_pixels", "false_negative_pixels",
}
PARITY_RTOL = 1e-6
PARITY_ATOL = 1e-7


def _sanitize_nonfinite(value):
    if isinstance(value, float) and not math.isfinite(value):
        label = "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
        return {"__nonfinite_float__": label}
    if isinstance(value, dict):
        return {key: _sanitize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_nonfinite(item) for item in value]
    return value


def write_json(path, payload, sanitize_nonfinite=False):
    temp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    serializable = _sanitize_nonfinite(payload) if sanitize_nonfinite else payload
    temp.write_text(json.dumps(serializable, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def record_failure(output_path, report, exc):
    report["status"] = "failed"
    report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    # Non-finite values remain visible as explicit markers in failure artifacts.
    write_json(output_path, report, sanitize_nonfinite=True)
    return report


def source_hashes():
    names = ("light_source.py", "source_training.py", "scripts/train_light_source.py",
             "scripts/benchmark_light_source.py", "scripts/benchmark_source_residency.py")
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in names
    }


def git_provenance():
    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    return {"head": git("rev-parse", "HEAD"),
            "dirty_paths": git("status", "--porcelain", "--untracked-files=all").splitlines()}


def initialize_provenance(output_path, report):
    report["provenance"] = {
        "git_status": "checking", "source_sha256": source_hashes(),
    }
    write_json(output_path, report)
    try:
        report["provenance"]["git"] = git_provenance()
        report["provenance"]["git_status"] = "available"
    except Exception as exc:
        report["provenance"]["git_status"] = "unavailable"
        report["provenance"]["git_error"] = {"type": type(exc).__name__, "message": str(exc)}
        record_failure(output_path, report,
                       RuntimeError("Git provenance is required; benchmark stopped: %s" % exc))
        raise RuntimeError("Git provenance is required; benchmark stopped") from exc
    write_json(output_path, report)
    return report["provenance"]


def fresh_output_dir(root):
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("--output-root must be an absolute path")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / ("source_residency_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                        + "_" + uuid.uuid4().hex[:8])
    candidate.mkdir(exist_ok=False)
    return candidate


def make_config(args, seed, arm, steps=None):
    return SourceFitConfig(
        steps=args.steps if steps is None else steps,
        learning_rate=0.01, band_weight=args.band_weight,
        threshold=0.225, steepness=50.0, binary_threshold=0.5,
        max_basis_bytes=args.max_basis_mib * 1024 ** 2,
        max_total_basis_bytes=args.max_total_basis_mib * 1024 ** 2,
        max_device_basis_bytes=args.max_device_basis_mib * 1024 ** 2,
        seed=int(seed), verify_basis=True,
        device_basis_residency=arm,
        max_resident_train_basis_bytes=args.resident_budget_mib * 1024 ** 2,
    )


def create_model(device, pixel_size_nm, seed):
    return make_simulator(device, pixel_size_nm, seed=int(seed), jitter=0.02,
                          grid_size=9, chunk_size=8, cache_bytes=0)


def _hard_field(path):
    return path.rsplit(".", 1)[-1] in HARD_COUNT_FIELDS


def compare_tree(left, right, path="root", stats=None):
    if stats is None:
        stats = {"ok": True, "max_abs_difference": 0.0,
                 "max_relative_difference": 0.0, "mismatch_count": 0,
                 "bitwise_equal": True, "examples": []}

    def mismatch(where, reason):
        stats["ok"] = False
        stats["mismatch_count"] += 1
        stats["bitwise_equal"] = False
        if len(stats["examples"]) < 20:
            stats["examples"].append({"path": where, "reason": reason})

    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            missing_from_right = sorted(str(key) for key in left.keys() - right.keys())
            missing_from_left = sorted(str(key) for key in right.keys() - left.keys())
            mismatch(path, "object keys differ; missing_from_right=%s missing_from_left=%s"
                     % (missing_from_right, missing_from_left))
            return stats
        for key in left:
            compare_tree(left[key], right[key], path + "." + str(key), stats)
        return stats
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            mismatch(path, "sequence lengths differ")
            return stats
        for index, (a, b) in enumerate(zip(left, right)):
            compare_tree(a, b, path + "[" + str(index) + "]", stats)
        return stats
    if isinstance(left, bool) or isinstance(right, bool):
        if type(left) is not type(right) or left != right:
            mismatch(path, "boolean value or type differs")
        return stats
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        left_number, right_number = float(left), float(right)
        if not math.isfinite(left_number) or not math.isfinite(right_number):
            mismatch(path, "non-finite numeric value (NaN or infinity)")
            return stats
        difference = abs(left_number - right_number)
        if not math.isfinite(difference):
            mismatch(path, "numeric difference overflowed to a non-finite value")
            return stats
        relative = difference / max(abs(left_number), abs(right_number), 1e-30)
        stats["max_abs_difference"] = max(stats["max_abs_difference"], difference)
        stats["max_relative_difference"] = max(stats["max_relative_difference"], relative)
        if difference != 0.0:
            stats["bitwise_equal"] = False
        if _hard_field(path):
            if left != right:
                mismatch(path, "hard pixel count differs")
        elif difference > PARITY_ATOL + PARITY_RTOL * abs(right_number):
            mismatch(path, "numeric difference exceeds declared tolerance")
        return stats
    if left != right:
        mismatch(path, "values differ")
    if stats["max_abs_difference"] != 0.0:
        stats["bitwise_equal"] = False
    return stats


def compare_fit_outputs(cpu_fit, device_fit):
    fields = (
        "before", "after", "history", "source_weights_before", "source_weights_after",
        "basis_verification", "train_masks_sha256", "validation_masks_sha256",
        "train_targets_sha256", "validation_targets_sha256",
    )
    cpu_view = {key: cpu_fit[key] for key in fields}
    device_view = {key: device_fit[key] for key in fields}
    return compare_tree(cpu_view, device_view)


def one_fit(device, train, validation, args, seed, arm, steps=None, capture_memory=False):
    model = create_model(device, train.pixel_size_nm, seed)
    initial_hash = tensor_hash(model.source.logits.detach())
    if model.source.logits.dtype != torch.float32:
        raise RuntimeError("benchmark requires float32 source weights")
    torch.cuda.synchronize(device)
    if capture_memory:
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
    else:
        memory_before = None
    start = time.perf_counter()
    result = fit_source(
        model, train, validation, corners=CORNERS,
        config=make_config(args, seed, arm, steps=steps), output_dir=None,
    )
    torch.cuda.synchronize(device)
    full_fit_seconds = time.perf_counter() - start
    memory = None
    if capture_memory:
        memory = {
            "before": memory_before,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "after_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "after_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
    compact = {
        "initial_source_logits_sha256": initial_hash,
        "preparation_seconds": float(result["preparation_seconds"]),
        "optimization_seconds": float(result["optimization_seconds"]),
        "full_fit_wall_seconds": float(full_fit_seconds),
        "requested_device_basis_residency": result["requested_device_basis_residency"],
        "effective_device_basis_residency": result["effective_device_basis_residency"],
        "estimated_train_device_basis_bytes": result["estimated_train_device_basis_bytes"],
        "resident_basis_estimate_scope": result["resident_basis_estimate_scope"],
        "basis_verification": result["basis_verification"],
        "train_masks_sha256": result["train_masks_sha256"],
        "validation_masks_sha256": result["validation_masks_sha256"],
        "train_targets_sha256": result["train_targets_sha256"],
        "validation_targets_sha256": result["validation_targets_sha256"],
        "before": result["before"], "after": result["after"],
        "history": result["history"],
        "source_weights_before": result["source_weights_before"],
        "source_weights_after": result["source_weights_after"],
        "memory": memory,
    }
    del model, result
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return compact


def summarize_ratios(runs):
    phases = ("preparation_seconds", "optimization_seconds", "full_fit_wall_seconds")
    summary = {}
    for phase in phases:
        paired = []
        by_seed = {}
        for row in runs:
            ratio = row["cpu"][phase] / row["device"][phase]
            paired.append(ratio)
            by_seed.setdefault(str(row["seed"]), []).append(ratio)
        summary[phase] = {
            "interpretation": "CPU seconds / device-resident seconds; >1 favors residency",
            "paired_median": float(statistics.median(paired)),
            "paired_min": float(min(paired)), "paired_max": float(max(paired)),
            "paired_count": len(paired),
            "by_seed": {
                seed: {"median": float(statistics.median(values)),
                       "min": float(min(values)), "max": float(max(values)),
                       "paired_count": len(values)}
                for seed, values in by_seed.items()
            },
        }
    return summary


def execution_order_counts(runs):
    return {
        "cpu_first": sum(row["execution_order"][0] == "cpu" for row in runs),
        "device_first": sum(row["execution_order"][0] == "device" for row in runs),
        "run_count": len(runs),
    }


def planned_execution_order_counts(seeds, repetitions):
    cpu_first = sum(
        (repetition + seed_index) % 2 == 0
        for seed_index, _ in enumerate(seeds)
        for repetition in range(repetitions)
    )
    return {"cpu_first": cpu_first, "device_first": len(seeds) * repetitions - cpu_first,
            "run_count": len(seeds) * repetitions}


def warmup(device, train, validation, args, seeds):
    rows = []
    warmup_steps = args.steps if args.warmup_steps is None else args.warmup_steps
    for seed in seeds:
        for arm in ("cpu", "device"):
            run = one_fit(device, train, validation, args, seed, arm,
                          steps=warmup_steps, capture_memory=False)
            if (run["requested_device_basis_residency"] != arm
                    or run["effective_device_basis_residency"] != arm):
                raise AssertionError("warm-up arm %s did not use its requested residency policy" % arm)
            rows.append({"seed": int(seed), "arm": arm,
                         "steps": warmup_steps,
                         "initial_source_logits_sha256": run["initial_source_logits_sha256"]})
    return rows


def run_benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the source-residency A/B benchmark")
    if args.steps < 1 or (args.warmup_steps is not None and args.warmup_steps < 1):
        raise ValueError("--steps and an explicit --warmup-steps must be positive")
    if args.repetitions < 5:
        raise ValueError("--repetitions must be at least 5")
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("exactly three distinct seeds are required")
    if args.resident_budget_mib < 1:
        raise ValueError("--resident-budget-mib must be positive")
    for name in ("max_basis_mib", "max_total_basis_mib", "max_device_basis_mib"):
        if getattr(args, name) < 1:
            raise ValueError("--" + name.replace("_", "-") + " must be positive")

    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(args.torch_threads)
    except RuntimeError:
        pass
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    out_dir = fresh_output_dir(args.output_root)
    output_path = out_dir / "benchmark.json"
    config = make_config(args, args.seeds[0], "cpu")
    planned_order_counts = planned_execution_order_counts(args.seeds, args.repetitions)
    report = {
        "schema_version": 1, "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "result_directory": str(out_dir), "output_json": str(output_path),
        "system": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "torch_cuda_runtime": torch.version.cuda,
            "cuda_device": str(device), "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "float32_only": True, "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        },
        "protocol": {
            "scope": "only training optical-basis residency differs between paired fits",
            "raster": args.raster, "pixel_size_nm": 4.0,
            "steps": args.steps,
            "warmup_steps_excluded": args.steps if args.warmup_steps is None else args.warmup_steps,
            "repetitions_per_seed": args.repetitions, "seeds": list(args.seeds),
            "arm_order": "alternates by repetition and seed index; no extra repetition is added to equalize counts",
            "planned_execution_order_counts": planned_order_counts,
            "default_five_repetition_order_counts": {"cpu_first": 8, "device_first": 7, "run_count": 15},
            "corners": [asdict(corner) for corner in CORNERS],
            "fit_config_common": asdict(config),
            "only_arm_specific_setting": "device_basis_residency (cpu versus device)",
            "resident_budget_mib": args.resident_budget_mib,
            "resident_budget_scope": "estimated intensity tensor payload only; not a cap on total CUDA peak memory",
            "warmup": "same full fit_source workload as measured by default, same fixed-mask data and initial source per seed, each arm separately; excluded from reported timings",
            "timings": {
                "preparation_seconds": "fit_source basis preparation including validation basis generation, CUDA synchronized at both boundaries",
                "optimization_seconds": "fit_source training loop, CUDA synchronized at both boundaries",
                "full_fit_wall_seconds": "external wall timer around full fit_source including before/after metrics; excludes output serialization",
            },
            "parity": {"rtol": PARITY_RTOL, "atol": PARITY_ATOL,
                       "hard_pixel_counts": "exact equality required"},
            "output": "one unique benchmark.json; fits use output_dir=None and write no checkpoints",
        },
        "fixture_preflight": None, "warmup": [], "runs": [],
        "paired_speedup_ratios": None,
        "execution_order_counts": {"cpu_first": 0, "device_first": 0, "run_count": 0},
        "limitations": [
            "synthetic scalar-Abbe fixtures only; no SOCS parity, EPE, shot-count, or real-layout claim",
            "residency may reduce basis transfer overhead; speedup is specific to this GPU, PyTorch, and fixture configuration",
            "hard-count parity is exact, but continuous values use the declared floating-point tolerance",
            "no MRC, dynamic weights, masks, objectives, initialization, precision, item counts, or update order are changed",
        ],
    }
    initialize_provenance(output_path, report)
    try:
        teacher, train, validation, raw = make_datasets(device, args.raster, 4.0, config)
        del teacher
        torch.cuda.synchronize(device)
        validate_splits(train, validation)
        pixels = args.raster * args.raster
        all_targets = list(train.targets) + list(validation.targets)
        counts = [int(target.sum().item()) for target in all_targets]
        distinct_train = len({tensor_hash(target) for target in train.targets})
        distinct_validation = len({tensor_hash(target) for target in validation.targets})
        target_ok = (
            min(counts) / pixels >= 0.01 and max(counts) / pixels <= 0.99
            and distinct_train >= 2 and distinct_validation >= 2
        )
        if not target_ok:
            raise ValueError(
                "synthetic fixture preflight failed the 1%-99% nontrivial target / distinct-target screen"
            )
        report["fixture_preflight"] = {
            "status": "passed", "layout_generator": "benchmark_light_source.make_layouts",
            "raster_shape": list(train.masks.shape[-2:]),
            "train_layout_ids": list(train.layout_ids),
            "validation_layout_ids": list(validation.layout_ids),
            "target_positive_pixels": {
                **{name: n for name, n in zip(train.layout_ids, counts[:len(train.masks)])},
                **{name: n for name, n in zip(validation.layout_ids, counts[len(train.masks):])},
            },
            "target_screen": "passed: every target 1%-99% positive and at least two distinct targets per split",
            "split_validation": "passed",
            "initial_source_screen": {"status": "running", "minimum_heldout_L2_ratio": 0.001,
                                      "runs": []},
            "tensor_sha256_by_layout": {
                "train_masks": {name: tensor_hash(value) for name, value in zip(train.layout_ids, train.masks)},
                "train_targets": {name: tensor_hash(value) for name, value in zip(train.layout_ids, train.targets)},
                "validation_masks": {name: tensor_hash(value) for name, value in zip(validation.layout_ids, validation.masks)},
                "validation_targets": {name: tensor_hash(value) for name, value in zip(validation.layout_ids, validation.targets)},
            },
        }
        write_json(output_path, report)
        for seed in args.seeds:
            model = create_model(device, train.pixel_size_nm, seed)
            initial = initial_validation(model, validation, CORNERS, config)
            ratio = initial["mean"]["L2_pixels"] / (len(validation.masks) * pixels)
            screen_row = {
                "seed": int(seed),
                "initial_source_logits_sha256": tensor_hash(model.source.logits.detach()),
                "heldout_L2_ratio": float(ratio),
            }
            report["fixture_preflight"]["initial_source_screen"]["runs"].append(screen_row)
            del model, initial
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            write_json(output_path, report)
            if ratio < 0.001:
                report["fixture_preflight"]["initial_source_screen"]["status"] = "failed"
                raise ValueError(
                    "raster %d failed the initial held-out L2 nontriviality screen for seed %d"
                    % (args.raster, seed)
                )
        report["fixture_preflight"]["initial_source_screen"]["status"] = "passed"
        write_json(output_path, report)
        report["warmup"] = warmup(device, train, validation, args, args.seeds)
        write_json(output_path, report)

        for seed_index, seed in enumerate(args.seeds):
            for repetition in range(args.repetitions):
                cpu_first = (repetition + seed_index) % 2 == 0
                order = ("cpu", "device") if cpu_first else ("device", "cpu")
                pair = {"seed": int(seed), "repetition": repetition,
                        "execution_order": list(order)}
                for arm in order:
                    pair[arm] = one_fit(device, train, validation, args, seed, arm,
                                        capture_memory=True)
                policy_errors = [
                    "%s requested=%s effective=%s" % (
                        arm, pair[arm]["requested_device_basis_residency"],
                        pair[arm]["effective_device_basis_residency"],
                    )
                    for arm in order
                    if pair[arm]["requested_device_basis_residency"] != arm
                    or pair[arm]["effective_device_basis_residency"] != arm
                ]
                if pair["cpu"]["initial_source_logits_sha256"] != pair["device"]["initial_source_logits_sha256"]:
                    raise AssertionError("paired arms did not start from identical source logits")
                parity = compare_fit_outputs(pair["cpu"], pair["device"])
                pair["parity"] = parity
                report["runs"].append(pair)
                report["execution_order_counts"] = execution_order_counts(report["runs"])
                report["paired_speedup_ratios"] = summarize_ratios(report["runs"])
                if policy_errors:
                    raise AssertionError("residency policy mismatch: " + "; ".join(policy_errors))
                if not parity["ok"]:
                    raise AssertionError(
                        "CPU/device fit parity failed for seed %s repetition %s: %s"
                        % (seed, repetition, parity["examples"])
                    )
                write_json(output_path, report)
        report["status"] = "done"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_path, report)
        print("Saved source-residency benchmark: " + str(output_path))
        return report
    except Exception as exc:
        record_failure(output_path, report, exc)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path,
                        help="absolute writable result root; a unique run directory is created")
    parser.add_argument("--raster", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="excluded warm-up steps per arm and seed (default: same as --steps)")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seeds", nargs=3, type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--band-weight", type=float, default=0.5)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--resident-budget-mib", type=int, default=512)
    parser.add_argument("--max-basis-mib", type=int, default=512)
    parser.add_argument("--max-total-basis-mib", type=int, default=512)
    parser.add_argument("--max-device-basis-mib", type=int, default=256)
    args = parser.parse_args(argv)
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    try:
        run_benchmark(args)
    except Exception as exc:
        print("source-residency benchmark failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
