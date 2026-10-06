"""Short synthetic GPU benchmark for the experimental scalar Abbe light source.

Outputs do not establish SOCS parity, EPE, shot count, or article results.
By default, results are written below D:/Codex/Lithography/work/light_source_benchmark;
--output-root selects another absolute result root.
"""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from light_source import DifferentiableAbbeLitho, PixelatedLightSource, resist_image
from source_training import (
    ProcessCorner, SourceDataset, SourceFitConfig, evaluate_source, fit_source, validate_splits,
)

RESULTS_ROOT = Path("D:/Codex/Lithography/work/light_source_benchmark")
DEFAULT_SEEDS = (17, 29, 43)
CORNERS = (
    ProcessCorner("d0.98", dose=0.98),
    ProcessCorner("nominal", dose=1.0),
    ProcessCorner("d1.02", dose=1.02),
)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is intentionally disabled.")
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    return device


def write_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def tensor_hash(tensor):
    raw = tensor.detach().to(device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def rows_as_bits(tensor):
    value = tensor.detach().to(device="cpu", dtype=torch.uint8)
    return ["".join("1" if bit else "0" for bit in row.tolist()) for row in value]


def rect(mask, y0, y1, x0, x1):
    h, w = mask.shape
    mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 1.0


def make_layouts(size=128):
    """Predeclared 4 nm lines, line-ends, junctions, jogs, and contacts."""
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

    mask = fresh("val_vls_pitch22_width7", "vertical_line_space",
                 "held-out vertical bars, 96 nm width and 176 nm pitch")
    for x in (4, 48, 92):
        rect(mask, 0, size, x, x + 24)

    mask = fresh("val_crossbar_endcaps", "junction_and_line_end",
                 "held-out crossed line ends and caps, 32 nm width")
    rect(mask, 16, 24, 19, 109)
    rect(mask, 50, 58, 39, 96)
    rect(mask, 23, 51, 19, 27)
    rect(mask, 23, 51, 101, 109)
    rect(mask, 58, 86, 39, 47)
    rect(mask, 58, 86, 88, 96)

    mask = fresh("val_chevron_contacts", "chevron_and_contacts",
                 "held-out chevrons and two isolated 104 nm contacts")
    for x0, y0 in ((22, 23), (71, 68)):
        for k in range(5):
            rect(mask, y0 + 8*k, y0 + 8*k + 16, x0 + 7*k, x0 + 7*k + 16)
            rect(mask, y0 + 8*k, y0 + 8*k + 16, x0 + 56 - 7*k, x0 + 72 - 7*k)
    for y, x in ((88, 8), (8, 88)):
        rect(mask, y, y + 26, x, x + 26)

    return ([row for row in rows if row["layout_id"].startswith("train_")],
            [row for row in rows if row["layout_id"].startswith("val_")])


def teacher_source(device, grid_size=9):
    source = PixelatedLightSource(grid_size=grid_size, sigma_inner=0.3, sigma_outer=0.9)
    coords, support = source._coordinates, source._pupil_support
    x, y = coords[..., 0], coords[..., 1]
    # Fixed asymmetric two-lobe map: controlled synthetic teacher, not fab data.
    lobe_a = -((x - 0.45).square() + (y - 0.225).square()) / (2 * 0.24**2)
    lobe_b = -((x + 0.225).square() + (y + 0.45).square()) / (2 * 0.27**2)
    with torch.no_grad():
        logits = torch.logaddexp(lobe_a, lobe_b + math.log(0.55))
        source.logits.copy_(torch.where(support, logits, source.logits))
    return source.to(device)


def make_simulator(device, pixel_size_nm, seed=None, jitter=0.02,
                   grid_size=9, chunk_size=8, cache_bytes=0):
    source = PixelatedLightSource(grid_size=grid_size, sigma_inner=0.3, sigma_outer=0.9)
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(source.logits.shape, generator=generator,
                            dtype=source.logits.dtype)
        with torch.no_grad():
            source.logits.add_(noise * float(jitter) * source._pupil_support)
    return DifferentiableAbbeLitho(
        source, numerical_aperture=1.35, wavelength_nm=193.0,
        pixel_size_nm=float(pixel_size_nm), source_chunk_size=chunk_size,
        cache_max_bytes=cache_bytes,
    ).to(device)


def make_datasets(device, size, pixel_size_nm, config):
    train_rows, val_rows = make_layouts(size)
    teacher = DifferentiableAbbeLitho(
        teacher_source(device), numerical_aperture=1.35, wavelength_nm=193.0,
        pixel_size_nm=pixel_size_nm, source_chunk_size=8, cache_max_bytes=0,
    ).to(device)
    with torch.no_grad():
        for row in train_rows + val_rows:
            aerial = teacher(row["mask"].to(device))
            soft = resist_image(aerial, dose=1.0, threshold=config.threshold,
                                steepness=config.steepness)
            target = (soft >= config.binary_threshold).float().cpu()[0]
            row["target"] = target
            row["target_sha256"] = tensor_hash(target)
            row["positive_pixels"] = int(target.sum().item())
            row["positive_fraction"] = float(target.mean().item())

    train = SourceDataset(
        torch.stack([r["mask"] for r in train_rows]),
        torch.stack([r["target"] for r in train_rows]),
        tuple(r["layout_id"] for r in train_rows), pixel_size_nm,
    )
    validation = SourceDataset(
        torch.stack([r["mask"] for r in val_rows]),
        torch.stack([r["target"] for r in val_rows]),
        tuple(r["layout_id"] for r in val_rows), pixel_size_nm,
    )
    split_error = None
    try:
        validate_splits(train, validation)
    except ValueError as exc:
        split_error = str(exc)
    train_mask_hashes = {tensor_hash(t): name for t, name in zip(train.masks, train.layout_ids)}
    train_target_hashes = {tensor_hash(t): name for t, name in zip(train.targets, train.layout_ids)}
    mask_overlap = {
        name: train_mask_hashes[tensor_hash(mask)]
        for name, mask in zip(validation.layout_ids, validation.masks)
        if tensor_hash(mask) in train_mask_hashes
    }
    target_overlap = {
        name: train_target_hashes[tensor_hash(target)]
        for name, target in zip(validation.layout_ids, validation.targets)
        if tensor_hash(target) in train_target_hashes
    }
    raw_layouts = []
    for split, rows in (("train", train_rows), ("validation", val_rows)):
        for row in rows:
            raw_layouts.append({
                "split": split, "layout_id": row["layout_id"],
                "family": row["family"], "geometry": row["geometry"],
                "mask_sha256": tensor_hash(row["mask"]),
                "target_sha256": row["target_sha256"],
                "target_positive_pixels": row["positive_pixels"],
                "target_positive_fraction": row["positive_fraction"],
                "mask_rows_bits": rows_as_bits(row["mask"]),
                "target_rows_bits": rows_as_bits(row["target"]),
            })
    raw = {
        "synthetic": True, "pixel_size_nm": pixel_size_nm,
        "raster_shape": [size, size],
        "split_integrity": {
            "validate_splits_error": split_error,
            "cross_split_identical_masks": mask_overlap,
            "cross_split_identical_targets": target_overlap,
            "usable_for_heldout_training": split_error is None,
        },
        "teacher_source_weights": teacher.source.weight_map().detach().cpu().tolist(),
        "target_generation": {
            "backend": "experimental_scalar_abbe",
            "teacher": "fixed asymmetric two-lobe synthetic source",
            "dose": 1.0, "threshold": config.threshold,
            "steepness": config.steepness,
            "binary_threshold": config.binary_threshold,
        },
        "layouts": raw_layouts,
    }
    return teacher, train, validation, raw


def percentile(values, q):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    position = max(0.0, min(float(len(ordered)-1), q * (len(ordered)-1)))
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0-fraction) + ordered[upper] * fraction


def sample_stats(values):
    vals = [float(v) for v in values]
    return {
        "samples": vals, "mean": statistics.mean(vals),
        "median_p50": percentile(vals, 0.50), "p95": percentile(vals, 0.95),
        "stdev_sample": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals), "max": max(vals),
    }


def timed_repetitions(step, device, repetitions, iterations, warmup=5):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            step()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) / iterations)
    return samples


def direct_step(sim, mask, probe):
    sim.source.logits.grad = None
    image = sim(mask)
    (image * probe).sum().backward()


def basis_step(sim, basis, probe):
    sim.source.logits.grad = None
    image = sim.evaluate_basis(basis)
    (image * probe).sum().backward()


def cpu_basis_step(sim, cpu_basis, device, probe):
    sim.source.logits.grad = None
    # Module.to mutates a module. A shallow module copy with its own buffer
    # mapping models a CPU-held production basis copied to CUDA each step.
    gpu_basis = copy.copy(cpu_basis)
    gpu_basis._buffers = cpu_basis._buffers.copy()
    gpu_basis.to(device)
    image = sim.evaluate_basis(gpu_basis)
    (image * probe).sum().backward()
    del gpu_basis, image


def memory_envelope(step, device, repeats):
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        step()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return {"allocated_before_bytes": int(baseline),
            "peak_allocated_bytes": int(peak),
            "peak_increment_bytes": int(max(0, peak - baseline))}


def gradient_parity(sim, mask, probe, basis):
    direct = sim(mask)
    direct_grad, = torch.autograd.grad((direct * probe).sum(), sim.source.logits)
    compiled = sim.evaluate_basis(basis)
    basis_grad, = torch.autograd.grad((compiled * probe).sum(), sim.source.logits)
    diff = direct_grad.detach() - basis_grad.detach()
    cosine = torch.nn.functional.cosine_similarity(
        direct_grad.detach().reshape(1, -1), basis_grad.detach().reshape(1, -1)
    ).item()
    return {
        "image_max_abs_error": float((direct.detach() - compiled.detach()).abs().max().item()),
        "image_allclose_rtol_1e-5_atol_1e-6": bool(torch.allclose(
            direct.detach(), compiled.detach(), rtol=1e-5, atol=1e-6)),
        "source_gradient_max_abs_error": float(diff.abs().max().item()),
        "source_gradient_relative_l2_error": float(
            diff.norm().item() / max(direct_grad.detach().norm().item(), 1e-30)),
        "source_gradient_cosine_similarity": float(cosine),
        "direct_gradient_l2_norm": float(direct_grad.detach().norm().item()),
        "basis_gradient_l2_norm": float(basis_grad.detach().norm().item()),
    }


def break_even(direct_step_s, basis_setup, basis_step_s):
    """First step count where the warmed basis route is strictly faster."""
    slope = float(direct_step_s) - float(basis_step_s)
    if slope <= 0:
        return None
    n = max(1, math.floor(float(basis_setup) / slope) + 1)
    while basis_setup + n*basis_step_s >= n*direct_step_s:
        n += 1
    return n


def benchmark_resolution(device, size, repetitions, iterations, mask_tensor):
    sim = make_simulator(device, 4.0, cache_bytes=0)
    mask = mask_tensor.to(device).detach()
    probe = torch.linspace(0.25, 1.25, size*size, dtype=torch.float32, device=device)
    probe = probe.reshape(1, 1, size, size)
    source_count = int(sim.source.distribution()[1].numel())
    required_cache_bytes = source_count * size * size * 8

    # First call records CUDA/library startup separately. The following
    # synchronized warmups keep one-time initialization out of steady samples.
    torch.cuda.synchronize(device)
    cold_prepare_start = time.perf_counter()
    cold_basis = sim.prepare_basis(mask)
    torch.cuda.synchronize(device)
    cold_prepare_seconds = time.perf_counter() - cold_prepare_start
    del cold_basis
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    for _ in range(5):
        warm_basis = sim.prepare_basis(mask)
        torch.cuda.synchronize(device)
        del warm_basis
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    prep_samples = []
    basis_gpu = None
    torch.cuda.synchronize(device)
    prep_baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repetitions):
        if basis_gpu is not None:
            del basis_gpu
            basis_gpu = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        basis_gpu = sim.prepare_basis(mask)
        torch.cuda.synchronize(device)
        prep_samples.append(time.perf_counter() - start)
    prep_peak = int(torch.cuda.max_memory_allocated(device))
    gpu_basis_bytes = int(basis_gpu.intensities.numel() * basis_gpu.intensities.element_size())

    # Direct path with transfer cache disabled: transfer construction repeats.
    sim.cache_max_bytes = 0
    sim.clear_cache()
    uncached_fn = lambda: direct_step(sim, mask, probe)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    uncached_fn()
    torch.cuda.synchronize(device)
    uncached_first = time.perf_counter() - start
    uncached_memory = memory_envelope(uncached_fn, device, max(1, min(repetitions, 3)))
    uncached_samples = timed_repetitions(uncached_fn, device, repetitions, iterations)

    # Direct path with a cache large enough for all active source transfers.
    sim.cache_max_bytes = int(required_cache_bytes * 2)
    sim.clear_cache()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    direct_step(sim, mask, probe)
    torch.cuda.synchronize(device)
    cached_first = time.perf_counter() - start
    cached_memory = memory_envelope(uncached_fn, device, max(1, min(repetitions, 3)))
    cached_samples = timed_repetitions(uncached_fn, device, repetitions, iterations)
    cache_bytes = int(sim._cache_bytes)
    sim.clear_cache()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    resident_samples = timed_repetitions(
        lambda: basis_step(sim, basis_gpu, probe), device, repetitions, iterations)
    resident_memory = memory_envelope(
        lambda: basis_step(sim, basis_gpu, probe), device, max(1, min(repetitions, 3)))
    parity = gradient_parity(sim, mask, probe, basis_gpu)

    # Warm D2H and H2D copy routes separately, then measure one D2H handoff
    # as the one-time production setup cost.
    torch.cuda.synchronize(device)
    copy_warm_start = time.perf_counter()
    warm_cpu_basis = basis_gpu.cpu()
    torch.cuda.synchronize(device)
    warm_gpu_basis = warm_cpu_basis.to(device)
    torch.cuda.synchronize(device)
    copy_warmup_seconds = time.perf_counter() - copy_warm_start
    del warm_cpu_basis, warm_gpu_basis
    gc.collect()
    torch.cuda.synchronize(device)
    handoff_start = time.perf_counter()
    basis_cpu = basis_gpu.cpu()
    torch.cuda.synchronize(device)
    handoff_seconds = time.perf_counter() - handoff_start
    cpu_basis_bytes = int(basis_cpu.intensities.numel() * basis_cpu.intensities.element_size())
    del basis_gpu
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    transfer_samples = timed_repetitions(
        lambda: cpu_basis_step(sim, basis_cpu, device, probe), device, repetitions, iterations)
    transfer_memory = memory_envelope(
        lambda: cpu_basis_step(sim, basis_cpu, device, probe), device,
        max(1, min(repetitions, 3)))

    prep = sample_stats(prep_samples)
    direct_uncached = sample_stats(uncached_samples)
    direct_cached = sample_stats(cached_samples)
    resident = sample_stats(resident_samples)
    transfer = sample_stats(transfer_samples)
    prep_mean = prep["mean"]
    resident_setup = prep_mean
    production_setup = prep_mean + handoff_seconds
    return_value = {
        "raster_shape": [size, size], "pixel_size_nm": 4.0,
        "source_grid_size": sim.source.grid_size, "active_source_points": source_count,
        "source_chunk_size": sim.source_chunk_size,
        "repetitions": repetitions, "iterations_per_repetition": iterations,
        "warmup_steps_per_arm": 5,
        "prepare_warmup_calls": 5,
        "basis_transfer_warmup": "one synchronized CPU roundtrip (GPU->CPU->GPU) before measuring production handoff and step transfer",
        "timing_unit": "seconds per source-gradient step (forward + common probe loss + backward)",
        "synchronization": "torch.cuda.synchronize before and after each timed repetition and preparation",
        "direct_cache_disabled": {
            "cache_max_bytes": 0, "first_call_seconds": uncached_first,
            "step_seconds": direct_uncached, "memory": uncached_memory,
        },
        "direct_cache_enabled": {
            "cache_max_bytes": int(sim.cache_max_bytes),
            "cache_required_bytes_estimate": required_cache_bytes,
            "cache_bytes_after_build": cache_bytes,
            "first_step_including_cache_build_seconds": cached_first,
            "cold_first_step_used_for_break_even": False,
            "steady_step_seconds": direct_cached, "memory": cached_memory,
        },
        "basis_prepare": {
            "cold_first_prepare_seconds_including_initialization": cold_prepare_seconds,
            "prepare_seconds": prep, "peak_allocated_bytes": prep_peak,
            "allocated_before_bytes": prep_baseline,
            "peak_increment_bytes": int(max(0, prep_peak-prep_baseline)),
            "gpu_persistent_intensity_bytes": gpu_basis_bytes,
            "cpu_roundtrip_warmup_seconds": copy_warmup_seconds,
            "cpu_handoff_seconds": handoff_seconds,
            "cpu_persistent_intensity_bytes": cpu_basis_bytes,
        },
        "basis_gpu_resident": {"step_seconds": resident, "memory": resident_memory},
        "basis_cpu_to_gpu_each_step": {
            "step_seconds_including_cpu_gpu_roundtrip_contraction_and_backward": transfer,
            "memory": transfer_memory,
        },
        "break_even_steps": {
            "uncached_direct_vs_basis_gpu_resident": break_even(
                direct_uncached["mean"], resident_setup, resident["mean"]),
            "cached_direct_vs_basis_gpu_resident": break_even(
                direct_cached["mean"], resident_setup, resident["mean"]),
            "uncached_direct_vs_production_cpu_gpu": break_even(
                direct_uncached["mean"], production_setup, transfer["mean"]),
            "cached_direct_vs_production_cpu_gpu": break_even(
                direct_cached["mean"], production_setup, transfer["mean"]),
            "calculation": "strictly first integer n where n*warmed_direct_step > warmed_basis_prepare+one_D2H_handoff+n*warmed_basis_step; cold direct/cache-build time is retained as raw data and gets no break-even credit",
        },
        "parity": parity,
    }
    del basis_cpu, sim, mask, probe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return return_value


def initial_validation(sim, validation, corners, config):
    prepared = []
    for mask in validation.masks:
        basis = sim.prepare_basis(mask[None].to(sim.source.logits.device))
        prepared.append({0.0: basis.cpu()})
    return evaluate_source(sim, validation, prepared, corners, config)


def quality_arm(device, seeds, steps, raster, band_weight, output_path, report):
    config = SourceFitConfig(
        steps=steps, learning_rate=0.01, band_weight=band_weight,
        threshold=0.225, steepness=50.0, binary_threshold=0.5,
        max_basis_bytes=512*1024**2, max_total_basis_bytes=512*1024**2,
        max_device_basis_bytes=256*1024**2, verify_basis=True,
    )
    pixel_size_nm = 4.0
    teacher, train, validation, raw = make_datasets(device, raster, pixel_size_nm, config)
    counts = [int(t.sum().item()) for t in train.targets]
    counts += [int(t.sum().item()) for t in validation.targets]
    pixels = raster*raster
    min_frac, max_frac = min(counts)/pixels, max(counts)/pixels
    target_ok = (
        min_frac >= 0.01 and max_frac <= 0.99
        and len({tensor_hash(x) for x in train.targets}) >= 2
        and len({tensor_hash(x) for x in validation.targets}) >= 2
    )
    raw["target_quality_precheck"] = {
        "positive_pixels_by_layout": {
            **{name: n for name, n in zip(train.layout_ids, counts[:len(train.layout_ids)])},
            **{name: n for name, n in zip(validation.layout_ids, counts[len(train.layout_ids):])},
        },
        "nonconstant_and_nontrivial_fraction_range_1pct_to_99pct": target_ok,
        "criterion": "each target has 1%-99% positive pixels, and each split has at least two distinct targets",
    }
    report["quality"] = {
        "status": "running", "raster_shape": [raster, raster],
        "pixel_size_nm": pixel_size_nm, "train_layout_count": len(train.masks),
        "heldout_layout_count": len(validation.masks), "seeds": list(seeds),
        "fit_config": {
            **config.__dict__, "seed_initialization_jitter_std": 0.02,
            "same_layouts_targets_corners_and_optimizer_settings_each_seed": True,
        },
        "corners": [{"name": c.name, "dose": c.dose, "defocus_nm": c.defocus_nm}
                    for c in CORNERS],
        "dose_corner_interpretation": "same 0.98/1.00/1.02 intensity corners before/after and for all seeds; teacher target generated at nominal dose 1.0",
        "metric_definition": {
            "L2_pixels": "nominal binary student print XOR binary teacher target",
            "L2_area_nm2": "L2_pixels * 4 nm * 4 nm",
            "band_pixels": "student binary print envelope across three dose corners; not error against teacher envelope",
            "band_area_nm2": "band_pixels * 4 nm * 4 nm",
            "objective": "continuous teacher-target fidelity plus band_weight times squared student dose-envelope width",
        },
        "target_counts_precheck": raw["target_quality_precheck"],
        "split_integrity": raw["split_integrity"],
        "runs": [], "raw_data": raw,
    }
    write_json(output_path, report)

    runs, initial_l2_ratios = [], []
    for seed in seeds:
        sim = make_simulator(device, pixel_size_nm, seed=seed, jitter=0.02,
                             grid_size=9, chunk_size=8, cache_bytes=0)
        initial = initial_validation(sim, validation, CORNERS, config)
        init_ratio = initial["mean"]["L2_pixels"] / (len(validation.masks)*pixels)
        initial_l2_ratios.append(init_ratio)
        record = {
            "seed": int(seed),
            "initial_source_logits_sha256": tensor_hash(sim.source.logits.detach()),
            "initial_heldout_metrics_before_optimizer": initial,
            "initial_heldout_L2_ratio": init_ratio,
            "status": "preflight_complete",
        }
        runs.append(record)
        report["quality"]["runs"] = runs
        write_json(output_path, report)
        del sim, initial
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    initial_nontrivial = all(v >= 0.001 for v in initial_l2_ratios)
    split_ok = raw["split_integrity"]["usable_for_heldout_training"]
    run_fit = bool(target_ok and initial_nontrivial and split_ok)
    if run_fit:
        for record in runs:
            seed = int(record["seed"])
            sim = make_simulator(device, pixel_size_nm, seed=seed, jitter=0.02,
                                 grid_size=9, chunk_size=8, cache_bytes=0)
            fit = fit_source(
                sim, train, validation, corners=CORNERS,
                config=SourceFitConfig(
                    steps=steps, learning_rate=0.01, band_weight=band_weight,
                    threshold=0.225, steepness=50.0, binary_threshold=0.5,
                    max_basis_bytes=512*1024**2, max_total_basis_bytes=512*1024**2,
                    max_device_basis_bytes=256*1024**2, seed=seed, verify_basis=True,
                ),
            )
            matches = (
                record["initial_heldout_metrics_before_optimizer"]
                == fit["before"]["validation"]
            )
            record["fit_report"] = fit
            record["preflight_matches_fit_before_validation"] = matches
            record["status"] = "done" if matches else "preflight_mismatch"
            write_json(output_path, report)
            del sim, fit
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
    else:
        for record in runs:
            if not split_ok:
                record["status"] = "not_run_invalid_split"
                record["reason"] = raw["split_integrity"]["validate_splits_error"]
            else:
                record["status"] = "not_run_nontriviality_screen"
                record["reason"] = (
                    "targets failed the 1%-99% target-variation screen"
                    if not target_ok
                    else "at least one seed had initial heldout L2 below 0.1% of heldout pixels"
                )
            record["fit_report"] = None
        write_json(output_path, report)

    preflight_matches = all(
        record.get("preflight_matches_fit_before_validation", True) for record in runs
    )
    informative = bool(run_fit and preflight_matches)
    report["quality"]["informative"] = informative
    report["quality"]["informative_criteria"] = {
        "targets_nontrivial": target_ok,
        "train_validation_splits_independent": split_ok,
        "each_seed_initial_heldout_L2_ratio_at_least_0_1_percent": initial_nontrivial,
        "initial_heldout_L2_ratios": initial_l2_ratios,
        "preflight_baseline_matches_fit_report": preflight_matches,
        "warning": None if informative else "predeclared synthetic nontriviality or split-independence screen failed; no source fitting was run unless targets were nontrivial and splits independent, and no quality-improvement claim is supported",
    }
    report["quality"]["teacher_source_weights"] = teacher.source.weight_map().detach().cpu().tolist()
    if informative:
        report["quality"]["status"] = "done"
    elif not split_ok:
        report["quality"]["status"] = "not_run_invalid_split"
    elif not target_ok or not initial_nontrivial:
        report["quality"]["status"] = "not_run_nontriviality_screen"
    else:
        report["quality"]["status"] = "preflight_mismatch"
    return report


def system_info(device):
    p = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()), "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(p.total_memory),
        "gpu_compute_capability": [int(p.major), int(p.minor)], "device": str(device),
    }


def new_output_dir(requested, requested_root=None):
    root = RESULTS_ROOT if requested_root is None else Path(requested_root)
    if not root.is_absolute():
        if requested_root is None:
            raise ValueError(
                f"default result root is not absolute on this platform: {root}; "
                "pass --output-root with an absolute path"
            )
        raise ValueError(f"--output-root must be absolute, got {root}")
    root = root.resolve()
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = root / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    else:
        candidate = Path(requested).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"results must be below {root}") from exc
    root.mkdir(parents=True, exist_ok=True)
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="absolute root for results (default: D:/Codex/Lithography/work/light_source_benchmark)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="new directory below --output-root (default: unique run directory)")
    parser.add_argument("--microbenchmark-sizes", nargs="+", type=int, default=(256, 512))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--quality-raster", type=int, default=128)
    parser.add_argument("--quality-steps", type=int, default=40)
    parser.add_argument("--quality-seeds", nargs=3, type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--band-weight", type=float, default=0.5)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-microbenchmark", action="store_true")
    args = parser.parse_args(argv)
    if args.repetitions < 5 or args.iterations < 20:
        parser.error("--repetitions >=5 and --iterations >=20 are required")
    if args.quality_steps < 1 or args.quality_raster < 32:
        parser.error("quality steps must be positive and raster must be >=32")
    if len(set(args.quality_seeds)) != 3:
        parser.error("exactly three distinct fixed quality seeds are required")
    if any(size < 32 for size in args.microbenchmark_sizes):
        parser.error("microbenchmark sizes must be >=32")
    device = require_cuda()
    out_dir = new_output_dir(args.output_dir, args.output_root)
    output_path = out_dir / "benchmark.json"
    report = {
        "schema_version": 1, "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "result_directory": str(out_dir), "storage_drive": out_dir.drive,
        "system": system_info(device),
        "scope": {
            "backend": "experimental_scalar_abbe",
            "synthetic_targets_only": True, "uses_real_training_data": False,
            "socs_parity_claim": False, "EPE": None, "shots": None,
            "limitations": [
                "synthetic scalar-Abbe targets do not validate the paper hypothesis",
                "no SOCS parity, EPE, shot-count, or real LithoBench claim is made",
                "microbenchmark is specific to this GPU/software stack",
                "quality arm fits a known synthetic teacher on disjoint synthetic layouts",
            ],
        },
        "protocol": {
            "microbenchmark_sizes": list(args.microbenchmark_sizes),
            "repetitions": args.repetitions, "iterations_per_repetition": args.iterations,
            "quality_raster": args.quality_raster, "quality_steps": args.quality_steps,
            "quality_seeds": list(args.quality_seeds), "quality_band_weight": args.band_weight,
            "output_json": str(output_path),
            "pilot_history": [
                {
                    "run_directory": "run_20260929T001316Z_03353d31",
                    "status": "pilot_only_no_quality_fit",
                    "outcome": "first quality construction stopped at split validation because binary targets overlapped; no fit was run",
                },
                {
                    "run_directory": "run_20260929T001526Z_55282ac2",
                    "status": "pilot_only_no_quality_fit",
                    "outcome": "three synthetic target layouts were empty and two heldout targets duplicated the empty training contact target; no fit was run",
                },
                {
                    "run_directory": "run_20260929T001723Z_c5bbf84b",
                    "status": "pilot_only_no_quality_fit",
                    "outcome": "same empty-target overlap persisted; first prepare included CUDA initialization outlier, so its timing was superseded by warmed preparation samples",
                },
            ],
            "single_geometry_correction_after_pilots": {
                "scope": "heldout line width/pitch plus contact/chevron geometry; same layout IDs/families, split, optics, seeds, dose corners, and fit hyperparameters",
                "training_contact_array": "28-pixel (112 nm) square contacts on 44-pixel (176 nm) pitch",
                "heldout_contact_features": "two 26-pixel (104 nm) square contacts, separated by 80 pixels in both raster axes",
                "heldout_line_and_chevron_features": "heldout line width 24 pixels (96 nm) at 44-pixel (176 nm) pitch; chevron stroke 16 pixels (64 nm)",
                "purpose": "make the narrowest target features resolvable by the declared scalar Abbe model; apply identically before any quality fit",
            },
        },
        "microbenchmark": {
            "status": "skipped" if args.skip_microbenchmark else "running", "sizes": [],
        },
    }
    write_json(output_path, report)
    try:
        if not args.skip_microbenchmark:
            train_rows, _ = make_layouts(max(args.microbenchmark_sizes))
            reference_mask = train_rows[0]["mask"]
            for size in args.microbenchmark_sizes:
                if size == reference_mask.shape[0]:
                    mask = reference_mask
                else:
                    mask = torch.nn.functional.interpolate(
                        reference_mask[None, None], size=(size, size), mode="nearest"
                    )[0, 0]
                item = benchmark_resolution(device, size, args.repetitions, args.iterations, mask)
                report["microbenchmark"]["sizes"].append(item)
                write_json(output_path, report)
        if args.skip_quality:
            report["quality"] = {"status": "skipped"}
        else:
            quality_arm(device, tuple(args.quality_seeds), args.quality_steps,
                        args.quality_raster, args.band_weight, output_path, report)
        report["microbenchmark"]["status"] = "skipped" if args.skip_microbenchmark else "done"
        report["status"] = "done"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_path, report)
        print(f"Saved benchmark JSON: {output_path}")
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
