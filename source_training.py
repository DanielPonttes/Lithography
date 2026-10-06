"""Global illumination fitting with fixed masks and explicit held-out layouts.

Only source logits are optimized. Metrics describe the configured Abbe model;
they do not establish equivalence to the existing LithoBench SOCS simulator.
"""

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Optional

import torch
import torch.nn.functional as F

from light_source import resist_image


def _image_batch(value, name):
    value = torch.as_tensor(value).detach().cpu().float().clone()
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[1] != 1 or min(value.shape) < 1:
        raise ValueError(name + " must have shape (N,1,H,W) or (N,H,W)")
    if not torch.isfinite(value).all() or not ((value == 0) | (value == 1)).all():
        raise ValueError(name + " must contain finite binary pixels (0 or 1)")
    return value


@dataclass
class SourceDataset:
    masks: torch.Tensor
    targets: torch.Tensor
    layout_ids: tuple
    pixel_size_nm: float
    group_ids: tuple = ()

    def __post_init__(self):
        self.masks = _image_batch(self.masks, "masks")
        self.targets = _image_batch(self.targets, "targets")
        if self.masks.shape != self.targets.shape:
            raise ValueError("mask and target batches must have identical shapes")
        self.layout_ids = tuple(self.layout_ids)
        if len(self.layout_ids) != len(self.masks):
            raise ValueError("one layout_id is required per sample")
        if any(not isinstance(x, str) or not x for x in self.layout_ids):
            raise ValueError("layout_ids must be non-empty strings")
        if len(set(self.layout_ids)) != len(self.layout_ids):
            raise ValueError("layout_ids must be unique within a split")
        self.group_ids = tuple(self.group_ids) if self.group_ids else self.layout_ids
        if len(self.group_ids) != len(self.masks) or any(
            not isinstance(x, str) or not x for x in self.group_ids
        ):
            raise ValueError("one non-empty group_id is required per sample")
        self.pixel_size_nm = float(self.pixel_size_nm)
        if not math.isfinite(self.pixel_size_nm) or self.pixel_size_nm <= 0:
            raise ValueError("pixel_size_nm must be finite and positive")

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("dataset must be a dictionary of tensors and metadata")
        return cls(**payload)

    def payload(self):
        return {
            "masks": self.masks, "targets": self.targets,
            "layout_ids": list(self.layout_ids), "group_ids": list(self.group_ids),
            "pixel_size_nm": self.pixel_size_nm,
        }


def _target_hash(target):
    return hashlib.sha256(target.contiguous().numpy().tobytes()).hexdigest()


def validate_splits(train, validation):
    if train.pixel_size_nm != validation.pixel_size_nm:
        raise ValueError("train and validation pixel sizes must match")
    if train.masks.shape[-2:] != validation.masks.shape[-2:]:
        raise ValueError("train and validation raster sizes must match")
    if set(train.layout_ids) & set(validation.layout_ids):
        raise ValueError("layout_ids overlap between training and validation")
    if set(train.group_ids) & set(validation.group_ids):
        raise ValueError("group_ids overlap between training and validation")
    train_hashes = {_target_hash(t) for t in train.targets}
    if any(_target_hash(t) in train_hashes for t in validation.targets):
        raise ValueError("an identical target appears in both splits")
    mask_hashes = {_target_hash(m) for m in train.masks}
    if any(_target_hash(m) in mask_hashes for m in validation.masks):
        raise ValueError("an identical mask appears in both splits")


@dataclass(frozen=True)
class ProcessCorner:
    name: str
    dose: float = 1.0
    defocus_nm: float = 0.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("corner name must be a non-empty string")
        if not math.isfinite(self.dose) or self.dose <= 0:
            raise ValueError("corner dose must be finite and positive")
        if not math.isfinite(self.defocus_nm):
            raise ValueError("defocus_nm must be finite")
        if self.name == "nominal" and (self.dose != 1 or self.defocus_nm != 0):
            raise ValueError("nominal corner must have dose=1 and defocus_nm=0")


def process_grid(doses=(0.98, 1.0, 1.02), defocus_nm=(0.0,)):
    corners = []
    for z in dict.fromkeys(float(x) for x in defocus_nm):
        for d in dict.fromkeys(float(x) for x in doses):
            name = "nominal" if z == 0 and d == 1 else "z%g_d%g" % (z, d)
            corners.append(ProcessCorner(name, d, z))
    if not any(c.name == "nominal" for c in corners):
        raise ValueError("process grid must include zero focus and unit dose")
    return tuple(corners)


@dataclass(frozen=True)
class SourceFitConfig:
    steps: int = 50
    learning_rate: float = 0.01
    band_weight: float = 0.5
    threshold: float = 0.225
    steepness: float = 50.0
    binary_threshold: float = 0.5
    max_basis_bytes: int = 512 * 1024 ** 2
    max_total_basis_bytes: int = 512 * 1024 ** 2
    max_device_basis_bytes: int = 256 * 1024 ** 2
    seed: int = 42
    verify_basis: bool = True
    # New objective controls stay last to preserve the original positional API.
    objective: str = "envelope_squared"
    surrogate_steepness: float = 50.0
    surrogate_band_weight: float = 0.5
    # Opt-in residency applies to training bases only. Validation stays on CPU.
    device_basis_residency: str = "cpu"
    max_resident_train_basis_bytes: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.band_weight) or self.band_weight < 0:
            raise ValueError("band_weight must be finite and non-negative")
        if self.objective not in ("envelope_squared", "worst_dose_surrogate", "worst_dose"):
            raise ValueError("objective must be envelope_squared, worst_dose_surrogate, or worst_dose")
        if not math.isfinite(self.surrogate_steepness) or self.surrogate_steepness <= 0:
            raise ValueError("surrogate_steepness must be finite and positive")
        if not math.isfinite(self.surrogate_band_weight) or self.surrogate_band_weight < 0:
            raise ValueError("surrogate_band_weight must be finite and non-negative")
        if not math.isfinite(self.threshold) or not math.isfinite(self.steepness) or self.steepness <= 0:
            raise ValueError("resist threshold/steepness must be finite; steepness > 0")
        if not math.isfinite(self.binary_threshold) or not 0 < self.binary_threshold < 1:
            raise ValueError("binary_threshold must be in (0,1)")
        budgets = (self.max_basis_bytes, self.max_total_basis_bytes, self.max_device_basis_bytes)
        if any(isinstance(b, bool) or not isinstance(b, int) or b < 1 for b in budgets):
            raise ValueError("basis memory budgets must be positive integers")
        if self.device_basis_residency not in ("cpu", "device", "auto"):
            raise ValueError("device_basis_residency must be cpu, device, or auto")
        resident_budget = self.max_resident_train_basis_bytes
        if resident_budget is not None and (
            isinstance(resident_budget, bool) or not isinstance(resident_budget, int)
            or resident_budget < 1
        ):
            raise ValueError("max_resident_train_basis_bytes must be a positive integer when set")
        if self.device_basis_residency != "cpu" and resident_budget is None:
            raise ValueError("device/auto residency requires an explicit aggregate training-basis budget")


def _prepare(simulator, dataset, focuses, config, keep_on_device=False):
    result = []
    verified = []
    device = simulator.source.logits.device
    for index, mask in enumerate(dataset.masks):
        bases = {}
        for focus in focuses:
            basis = simulator.prepare_basis(
                mask[None].to(device), defocus_nm=focus,
                max_bytes=config.max_basis_bytes,
            )
            if config.verify_basis and index == 0:
                with torch.no_grad():
                    direct = simulator(mask[None].to(device), defocus_nm=focus)
                    compiled = simulator.evaluate_basis(basis)
                    torch.testing.assert_close(direct, compiled, rtol=1e-5, atol=1e-6)
                    verified.append({"layout_id": dataset.layout_ids[index],
                                     "defocus_nm": focus,
                                     "max_abs_error": float((direct - compiled).abs().max().item())})
            bases[focus] = basis if keep_on_device else basis.cpu()
        result.append(bases)
    return result, verified


def _cpu_state(module):
    return {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
            for k, v in module.state_dict().items()}


def _printed(simulator, bases, corners, config):
    device = simulator.source.logits.device
    aerials = {
        z: simulator.evaluate_basis(
            b if b.intensities.device == device else b.to(device), defocus_nm=z
        )
        for z, b in bases.items()
    }
    return torch.stack([
        resist_image(aerials[c.defocus_nm], dose=c.dose,
                     threshold=config.threshold, steepness=config.steepness)
        for c in corners
    ])


def _release(bases, keep_on_device=False):
    if not keep_on_device:
        for basis in bases.values():
            basis.cpu()


def _objective_components(printed, target, config):
    residual_sq = (printed - target[None]).square()
    fidelity = residual_sq.mean()
    worst_dose_fidelity = residual_sq.max(dim=0).values.mean()
    envelope = printed.max(dim=0).values - printed.min(dim=0).values
    q = torch.sigmoid(config.surrogate_steepness * (printed - config.binary_threshold))
    surrogate_band = (q.max(dim=0).values - q.min(dim=0).values).mean()
    return {
        "continuous_fidelity_mse": fidelity,
        "worst_dose_fidelity_mse": worst_dose_fidelity,
        "envelope_continuous_mean": envelope.mean(),
        "envelope_continuous_mse": envelope.square().mean(),
        "surrogate_pv_band": surrogate_band,
    }


def _objective(printed, target, config):
    terms = _objective_components(printed, target, config)
    if config.objective == "worst_dose_surrogate":
        return terms["worst_dose_fidelity_mse"] + (
            config.surrogate_band_weight * terms["surrogate_pv_band"]
        )
    if config.objective == "worst_dose":
        return terms["worst_dose_fidelity_mse"]
    return terms["continuous_fidelity_mse"] + config.band_weight * terms["envelope_continuous_mse"]


@torch.no_grad()
def evaluate_source(simulator, dataset, prepared, corners, config, keep_on_device=False):
    nominal = next(i for i, c in enumerate(corners) if c.name == "nominal")
    records = []
    for index, bases in enumerate(prepared):
        try:
            printed = _printed(simulator, bases, corners, config)
            target = dataset.targets[index:index + 1].to(printed.device)
            binary = printed >= config.binary_threshold
            l2 = float((binary[nominal] != target.bool()).sum().item())
            per_corner_l2 = (binary != target.bool()[None]).reshape(len(corners), -1).sum(dim=1)
            band = float((binary.any(dim=0) != binary.all(dim=0)).sum().item())
            target_binary = target.bool()
            per_corner_binary_metrics = []
            for corner_index, corner in enumerate(corners):
                corner_binary = binary[corner_index]
                per_corner_binary_metrics.append({
                    "corner": asdict(corner),
                    "predicted_positive_pixels": int(corner_binary.sum().item()),
                    "target_positive_pixels": int(target_binary.sum().item()),
                    "false_positive_pixels": int((corner_binary & ~target_binary).sum().item()),
                    "false_negative_pixels": int((~corner_binary & target_binary).sum().item()),
                })
            terms = _objective_components(printed, target, config)
            zero_focus = [(i, c) for i, c in enumerate(corners) if c.defocus_nm == 0]
            low = [(i, c) for i, c in zero_focus if c.dose == min(x.dose for _, x in zero_focus)]
            high = [(i, c) for i, c in zero_focus if c.dose == max(x.dose for _, x in zero_focus)]
            flip_window = None
            if low and high and low[0][1].dose < high[0][1].dose:
                flip_window = float((binary[low[0][0]] != binary[high[0][0]]).sum().item())
            records.append({
                "layout_id": dataset.layout_ids[index], "L2_pixels": l2,
                "L2_worst_dose_pixels": float(per_corner_l2.max().item()),
                "band_pixels": band,
                "flip_window_pixels": flip_window,
                "per_corner_binary_metrics": per_corner_binary_metrics,
                **{key: float(value.item()) for key, value in terms.items()},
                "L2_area_nm2": l2 * dataset.pixel_size_nm ** 2,
                "band_area_nm2": band * dataset.pixel_size_nm ** 2,
                "objective": float(_objective(printed, target, config).item()),
            })
        finally:
            _release(bases, keep_on_device=keep_on_device)
    mean_keys = (
        "L2_pixels", "L2_worst_dose_pixels", "band_pixels", "flip_window_pixels",
        "continuous_fidelity_mse", "worst_dose_fidelity_mse",
        "envelope_continuous_mean", "envelope_continuous_mse", "surrogate_pv_band",
        "L2_area_nm2", "band_area_nm2", "objective",
    )
    means = {
        key: (None if all(r[key] is None for r in records) else
              sum(r[key] for r in records if r[key] is not None)
              / sum(r[key] is not None for r in records))
        for key in mean_keys
    }
    return {"mean": means, "per_layout": records}


def fit_source(simulator, train, validation, corners=None, config=None, output_dir=None):
    """Fit one shared source; no masks or validation data receive updates.

    By default bases are held on CPU and transferred one layout at a time.
    Opt-in device residency retains training bases within an aggregate budget;
    validation bases always stay on CPU. A basis is reused for all doses at the
    same defocus. An optimizer step accumulates the mean loss over every layout.
    """
    config = config or SourceFitConfig()
    corners = tuple(process_grid() if corners is None else corners)
    if len({c.name for c in corners}) != len(corners):
        raise ValueError("corner names must be unique")
    if sum(c.name == "nominal" for c in corners) != 1:
        raise ValueError("exactly one nominal corner is required")
    validate_splits(train, validation)
    if simulator.pixel_size_nm != train.pixel_size_nm:
        raise ValueError("simulator pixel size does not match the dataset")
    focuses = tuple(dict.fromkeys(c.defocus_nm for c in corners))
    if any(z != 0 for z in focuses) and getattr(simulator, "refractive_index", None) is None:
        raise ValueError("defocus requires an explicit refractive_index")

    _, weights = simulator.source.distribution()
    h, w = train.masks.shape[-2:]
    # The simulator promotes float32 masks when source parameters are double.
    real_bytes = 8 if simulator.source.logits.dtype == torch.float64 else 4
    per_basis = weights.numel() * h * w * real_bytes
    total = per_basis * len(focuses) * (len(train.masks) + len(validation.masks))
    per_layout = per_basis * len(focuses)
    if per_layout > config.max_device_basis_bytes:
        raise MemoryError("all focus bases for one layout require %d device bytes; "
                          "reduce the raster/focus grid or raise the explicit device budget" % per_layout)
    if per_basis > config.max_basis_bytes or total > config.max_total_basis_bytes:
        raise MemoryError("optical bases require %d bytes per mask/focus and %d total; "
                          "reduce the dataset/raster or raise explicit memory budgets" % (per_basis, total))

    estimated_train_device_bytes = per_basis * len(focuses) * len(train.masks)
    residency_budget = config.max_resident_train_basis_bytes
    residency_fallback_reason = None
    device = simulator.source.logits.device
    if config.device_basis_residency == "device" and device.type == "cpu":
        raise ValueError("device basis residency requires a CUDA simulator; got CPU")
    if config.device_basis_residency == "device":
        if estimated_train_device_bytes > residency_budget:
            raise MemoryError(
                "training bases require %d aggregate device bytes, exceeding explicit "
                "residency budget %d" % (estimated_train_device_bytes, residency_budget)
            )
        effective_residency = "device"
    elif config.device_basis_residency == "auto":
        if device.type == "cpu":
            effective_residency = "cpu"
            residency_fallback_reason = "simulator is on CPU; automatic device residency requires CUDA"
        elif estimated_train_device_bytes <= residency_budget:
            effective_residency = "device"
        else:
            effective_residency = "cpu"
            residency_fallback_reason = (
                "estimated training basis bytes %d exceed explicit residency budget %d"
                % (estimated_train_device_bytes, residency_budget)
            )
    else:
        effective_residency = "cpu"

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    train_bases, verified_train = _prepare(
        simulator, train, focuses, config, keep_on_device=effective_residency == "device"
    )
    validation_bases, verified_validation = _prepare(simulator, validation, focuses, config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - start
    initial_state = _cpu_state(simulator.source)
    initial_weights = simulator.source.weight_map().detach().cpu().tolist()
    before = {
        "train": evaluate_source(
            simulator, train, train_bases, corners, config,
            keep_on_device=effective_residency == "device",
        ),
        "validation": evaluate_source(simulator, validation, validation_bases, corners, config),
    }

    optimizer = torch.optim.Adam([simulator.source.logits], lr=config.learning_rate)
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for index, bases in enumerate(train_bases):
            try:
                printed = _printed(simulator, bases, corners, config)
                target = train.targets[index:index + 1].to(printed.device)
                loss = _objective(printed, target, config) / len(train_bases)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite source training loss")
                loss.backward()
                train_loss += float(loss.detach().item())
            finally:
                _release(bases, keep_on_device=effective_residency == "device")
        if simulator.source.logits.grad is None or not torch.isfinite(simulator.source.logits.grad).all():
            raise FloatingPointError("source gradient is missing or non-finite")
        optimizer.step()
        history.append({"step": step, "train_objective_before_update": train_loss})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    optimization_seconds = time.perf_counter() - start
    after = {
        "train": evaluate_source(
            simulator, train, train_bases, corners, config,
            keep_on_device=effective_residency == "device",
        ),
        "validation": evaluate_source(simulator, validation, validation_bases, corners, config),
    }
    has_focus = any(c.defocus_nm != 0 for c in corners)
    has_dose = len({c.dose for c in corners}) > 1
    band_type = "focus_and_dose" if has_focus and has_dose else (
        "focus_only" if has_focus else ("dose_only" if has_dose else "none"))
    report = {
        "schema_version": 1, "backend": "experimental_scalar_abbe",
        "comparable_to_original_socs": False, "band_type": band_type,
        "dose_convention": "intensity", "torch_version": str(torch.__version__),
        "optical_config": simulator.optical_config(), "fit_config": asdict(config),
        "source_config": {"grid_size": simulator.source.grid_size,
                          "sigma_inner": simulator.source.sigma_inner,
                          "sigma_outer": simulator.source.sigma_outer},
        "corners": [asdict(c) for c in corners],
        "train_ids": list(train.layout_ids), "validation_ids": list(validation.layout_ids),
        "train_group_ids": list(train.group_ids), "validation_group_ids": list(validation.group_ids),
        "raster_shape": [h, w], "estimated_basis_bytes": total,
        "estimated_device_basis_bytes_per_layout": per_layout,
        "estimated_train_device_basis_bytes": estimated_train_device_bytes,
        "resident_basis_estimate_scope": (
            "intensity tensor payload only; excludes basis metadata/buffers, internal clones, "
            "validation construction temporaries, and other CUDA allocations"
        ),
        "requested_device_basis_residency": config.device_basis_residency,
        "effective_device_basis_residency": effective_residency,
        "max_resident_train_basis_bytes": residency_budget,
        "device_basis_residency_fallback_reason": residency_fallback_reason,
        "basis_verification": verified_train + verified_validation,
        "train_masks_sha256": _target_hash(train.masks),
        "validation_masks_sha256": _target_hash(validation.masks),
        "train_targets_sha256": _target_hash(train.targets),
        "validation_targets_sha256": _target_hash(validation.targets),
        "before": before, "after": after, "history": history,
        "source_weights_before": initial_weights,
        "source_weights_after": simulator.source.weight_map().detach().cpu().tolist(),
        "preparation_seconds": preparation_seconds, "optimization_seconds": optimization_seconds,
        "EPE": None, "shots": None,
        "metric_notes": ["L2 is nominal binary pixel mismatch at the declared raster.",
                         "Band is XOR of binary envelopes over the configured corners.",
                         "EPE and shots require a separate canonical evaluation; source-only fitting leaves masks fixed.",
                         "Validation is evaluated before/after and never used for gradient updates."],
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "source_state": _cpu_state(simulator.source),
            "initial_source_state": initial_state, "optical_config": report["optical_config"],
            "source_config": report["source_config"], "corners": report["corners"],
            "fit_config": report["fit_config"], "train_ids": report["train_ids"],
            "validation_ids": report["validation_ids"],
            "train_group_ids": report["train_group_ids"],
            "validation_group_ids": report["validation_group_ids"],
            "train_masks_sha256": report["train_masks_sha256"],
            "validation_masks_sha256": report["validation_masks_sha256"],
            "train_targets_sha256": report["train_targets_sha256"],
            "validation_targets_sha256": report["validation_targets_sha256"],
        }
        torch.save(checkpoint, out / "source.pt")
        (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
