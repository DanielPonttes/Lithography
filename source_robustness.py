"""Target-aware, source-only process-corner robustness objectives and diagnostics."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch

THRESHOLD = 0.225
LOW_DOSE = 0.98
NOMINAL_DOSE = 1.0
HIGH_DOSE = 1.02
RESIST_STEEPNESS = 50.0
SMOOTH_PV_BETA = 800.0
BASIS_NONNEGATIVE_TOL = 0.0
SOFTCOUNT_OBJECTIVE_ID = "target_aware_critical_corner_softcount_v1"
SOFTCOUNT_BETA_SCHEDULE = (200.0, 400.0, 800.0)


def validate_rho(rho: float) -> float:
    value = float(rho)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("rho must be finite and in (0, 1]")
    return value


def validate_margin_scale(lp_margin: float, rho: float) -> tuple[float, float]:
    rho = validate_rho(rho)
    lp_margin = float(lp_margin)
    if not math.isfinite(lp_margin) or lp_margin <= 0:
        raise ValueError("LP margin must be finite and positive")
    return lp_margin, rho * lp_margin


def validate_nonnegative_bases(
    bases: Mapping[str, torch.Tensor], tolerance: float = BASIS_NONNEGATIVE_TOL
) -> dict[str, float]:
    """Require the optical basis to be nonnegative up to declared roundoff."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("basis tolerance must be finite and non-negative")
    minima = {}
    for name, basis in bases.items():
        if not isinstance(basis, torch.Tensor) or basis.ndim != 3:
            raise ValueError("basis for %s must have shape (source, height, width)" % name)
        if not torch.isfinite(basis).all():
            raise ValueError("basis for %s contains non-finite values" % name)
        minimum = float(basis.detach().min().item())
        if minimum < -tolerance:
            raise ValueError(
                "basis for %s is negative beyond roundoff tolerance: %.17g < -%.17g"
                % (name, minimum, tolerance)
            )
        minima[name] = minimum
    if not minima:
        raise ValueError("at least one fit basis is required")
    return minima


def _validate_aerial_target(
    aerial: torch.Tensor, target: torch.Tensor, name: str = "layout"
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(aerial, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("aerial and target must be tensors")
    if aerial.ndim != 2 or target.ndim != 2 or aerial.shape != target.shape:
        raise ValueError("%s aerial/target must be matching height-by-width tensors" % name)
    if not torch.isfinite(aerial).all():
        raise ValueError("%s aerial contains non-finite values" % name)
    if target.dtype != torch.bool:
        if not (target.is_floating_point() or target.dtype in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        )):
            raise ValueError("%s target must be binary" % name)
        if target.is_floating_point() and not torch.isfinite(target).all():
            raise ValueError("%s target contains non-finite values" % name)
        if not torch.all((target == 0) | (target == 1)):
            raise ValueError("%s target must contain only 0 and 1" % name)
    target_bool = target.to(device=aerial.device, dtype=torch.bool)
    return aerial, target_bool


def worst_corner_squared_hinge_from_aerial(
    aerial: torch.Tensor,
    target: torch.Tensor,
    lp_margin: float,
    rho: float,
    threshold: float = THRESHOLD,
) -> torch.Tensor:
    """Smooth convex extension of the worst-dose squared hinge on I >= 0.

    For clear target pixels the low-dose corner is worst; for dark target pixels
    the high-dose corner is worst. This is exactly the min over dose corners
    {0.98, 1.02} whenever aerial intensity is nonnegative.
    """
    lp_margin, margin_floor = validate_margin_scale(lp_margin, rho)
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    aerial, target_bool = _validate_aerial_target(aerial, target)
    signed_margin = torch.where(
        target_bool,
        LOW_DOSE * aerial - float(threshold),
        float(threshold) - HIGH_DOSE * aerial,
    )
    violation = torch.relu(margin_floor - signed_margin)
    return violation.square().mean() / (lp_margin * lp_margin)


def explicit_worst_corner_squared_hinge_from_aerial(
    aerial: torch.Tensor,
    target: torch.Tensor,
    lp_margin: float,
    rho: float,
    threshold: float = THRESHOLD,
) -> torch.Tensor:
    """Reference min-corner expression, useful for parity checks."""
    lp_margin, margin_floor = validate_margin_scale(lp_margin, rho)
    aerial, target_bool = _validate_aerial_target(aerial, target)
    sign = torch.where(target_bool, 1.0, -1.0).to(dtype=aerial.dtype)
    margins = torch.stack(
        (
            sign * (LOW_DOSE * aerial - float(threshold)),
            sign * (HIGH_DOSE * aerial - float(threshold)),
        ),
        dim=0,
    )
    violation = torch.relu(margin_floor - margins.min(dim=0).values)
    return violation.square().mean() / (lp_margin * lp_margin)


def _aerials(
    weights: torch.Tensor, bases: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        name: torch.einsum("n,nhw->hw", weights, basis)
        for name, basis in bases.items()
    }


def robust_corner_squared_hinge(
    weights: torch.Tensor,
    bases: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    lp_margin: float,
    rho: float,
    threshold: float = THRESHOLD,
) -> torch.Tensor:
    """Mean normalized worst-dose squared hinge across fit pixels/layouts."""
    if set(bases) != set(targets) or not bases:
        raise ValueError("fit basis and target layout IDs must match and be non-empty")
    lp_margin, _ = validate_margin_scale(lp_margin, rho)
    losses = []
    for name, basis in bases.items():
        if basis.ndim != 3 or basis.shape[0] != weights.numel():
            raise ValueError("basis source dimension mismatch for %s" % name)
        aerial = torch.einsum("n,nhw->hw", weights, basis)
        target = torch.as_tensor(targets[name], device=aerial.device)
        losses.append(
            worst_corner_squared_hinge_from_aerial(
                aerial, target, lp_margin, rho, threshold
            ).reshape(())
        )
    return torch.stack(losses).mean()


def robust_corner_value_gradient(
    weights: np.ndarray,
    bases: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    lp_margin: float,
    rho: float,
    threshold: float = THRESHOLD,
) -> tuple[float, np.ndarray]:
    """Evaluate loss and its 49-dimensional float64 source-weight gradient."""
    device = next(iter(bases.values())).device
    variable = torch.tensor(
        np.asarray(weights, dtype=np.float64), dtype=torch.float64,
        device=device, requires_grad=True,
    )
    loss = robust_corner_squared_hinge(
        variable, bases, targets, lp_margin, rho, threshold
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite robust-corner hinge loss")
    loss.backward()
    if variable.grad is None or not torch.isfinite(variable.grad).all():
        raise FloatingPointError("missing or non-finite robust-corner gradient")
    return float(loss.detach().item()), variable.grad.detach().cpu().numpy().copy()


def robust_corner_value(
    weights: np.ndarray,
    bases: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    lp_margin: float,
    rho: float,
    threshold: float = THRESHOLD,
) -> float:
    device = next(iter(bases.values())).device
    variable = torch.as_tensor(
        np.asarray(weights, dtype=np.float64), dtype=torch.float64, device=device
    )
    with torch.no_grad():
        loss = robust_corner_squared_hinge(
            variable, bases, targets, lp_margin, rho, threshold
        )
    value = float(loss.item())
    if not math.isfinite(value):
        raise FloatingPointError("non-finite robust-corner hinge value")
    return value


def fit_objective_gradient_diagnostics(
    weights: np.ndarray,
    bases: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    lp_margin: float,
    rho: float,
    threshold: float = THRESHOLD,
) -> dict:
    """Compare source-logit-independent weight gradients on fixed fit layouts."""
    if set(bases) != set(targets) or not bases:
        raise ValueError("fit basis and target layout IDs must match and be non-empty")
    device = next(iter(bases.values())).device
    variable = torch.tensor(
        np.asarray(weights, dtype=np.float64), dtype=torch.float64,
        device=device, requires_grad=True,
    )
    terms = {
        "nominal_fidelity_mse": [],
        "dose_0.98_fidelity_mse": [],
        "dose_1.02_fidelity_mse": [],
        "smooth_pv_beta800": [],
    }
    for name, basis in bases.items():
        if basis.ndim != 3 or basis.shape[0] != variable.numel():
            raise ValueError("basis source dimension mismatch for %s" % name)
        aerial = torch.einsum("n,nhw->hw", variable, basis)
        _, target_bool = _validate_aerial_target(
            aerial, torch.as_tensor(targets[name], device=aerial.device), name
        )
        target = target_bool.to(dtype=aerial.dtype)
        for label, dose in (
            ("nominal_fidelity_mse", NOMINAL_DOSE),
            ("dose_0.98_fidelity_mse", LOW_DOSE),
            ("dose_1.02_fidelity_mse", HIGH_DOSE),
        ):
            printed = torch.sigmoid(
                RESIST_STEEPNESS * (dose * aerial - threshold)
            )
            terms[label].append((printed - target).square().mean())
        terms["smooth_pv_beta800"].append(
            (
                torch.sigmoid(SMOOTH_PV_BETA * (HIGH_DOSE * aerial - threshold))
                - torch.sigmoid(SMOOTH_PV_BETA * (LOW_DOSE * aerial - threshold))
            ).mean()
        )
    scalar_terms = {name: torch.stack(values).mean() for name, values in terms.items()}
    scalar_terms["robust_corner_squared_hinge"] = robust_corner_squared_hinge(
        variable, bases, targets, lp_margin, rho, threshold
    )
    names = tuple(scalar_terms)
    gradients = {}
    values = {}
    for index, name in enumerate(names):
        gradient, = torch.autograd.grad(
            scalar_terms[name], variable, retain_graph=index + 1 < len(names)
        )
        if not torch.isfinite(gradient).all() or not torch.isfinite(scalar_terms[name]):
            raise FloatingPointError("non-finite gradient diagnostic for %s" % name)
        gradients[name] = gradient.detach().cpu().numpy().copy()
        values[name] = float(scalar_terms[name].detach().item())
    norms = {name: float(np.linalg.norm(gradient)) for name, gradient in gradients.items()}
    tangent_gradients = {
        name: gradient - float(np.mean(gradient))
        for name, gradient in gradients.items()
    }
    tangent_norms = {
        name: float(np.linalg.norm(gradient))
        for name, gradient in tangent_gradients.items()
    }
    pairwise_cosines = {}
    tangent_pairwise_cosines = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            denominator = norms[left] * norms[right]
            cosine = (
                float(np.dot(gradients[left], gradients[right]) / denominator)
                if denominator > 0 else None
            )
            pairwise_cosines[left + "__" + right] = cosine
            tangent_denominator = tangent_norms[left] * tangent_norms[right]
            tangent_cosine = (
                float(np.dot(tangent_gradients[left], tangent_gradients[right])
                      / tangent_denominator)
                if tangent_denominator > 0 else None
            )
            tangent_pairwise_cosines[left + "__" + right] = tangent_cosine
    return {
        "objective_values": values,
        "gradient_l2_norms": norms,
        "pairwise_gradient_cosines": pairwise_cosines,
        "simplex_tangent_gradient_l2_norms": tangent_norms,
        "simplex_tangent_pairwise_gradient_cosines": tangent_pairwise_cosines,
        "gradient_space": (
            "49 source weights; fixed fit layouts only. Raw gradients include "
            "the simplex-normal component; tangent projection subtracts the "
            "mean component. Neither cosine accounts for active LP constraints."
        ),
    }


def signed_margin_quantiles(
    weights: np.ndarray,
    basis: torch.Tensor,
    target: torch.Tensor,
    doses: tuple[float, ...] = (LOW_DOSE, NOMINAL_DOSE, HIGH_DOSE),
    threshold: float = THRESHOLD,
) -> dict[str, dict]:
    """Return per-pixel target-signed intensity margin summaries by dose."""
    if basis.ndim != 3:
        raise ValueError("basis must have shape (source, height, width)")
    device = basis.device
    w = torch.as_tensor(weights, dtype=torch.float64, device=device)
    aerial = torch.einsum(
        "n,nhw->hw", w, basis.to(dtype=torch.float64)
    )
    _, target_bool = _validate_aerial_target(
        aerial, torch.as_tensor(target, device=device), "layout"
    )
    out = {}
    for dose in doses:
        if not math.isfinite(float(dose)) or dose <= 0:
            raise ValueError("dose must be finite and positive")
        signed = torch.where(
            target_bool,
            float(dose) * aerial - threshold,
            threshold - float(dose) * aerial,
        ).reshape(-1)
        quantiles = torch.quantile(
            signed, torch.tensor(
                [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0],
                dtype=signed.dtype, device=signed.device,
            ),
        )
        out["nominal" if dose == NOMINAL_DOSE else "d%g" % dose] = {
            "dose": float(dose),
            "pixel_count": int(signed.numel()),
            "negative_margin_pixels": int((signed < 0).sum().item()),
            "minimum_signed_margin": float(quantiles[0].item()),
            "signed_margin_quantiles": {
                key: float(value.item())
                for key, value in zip(
                    ("p00", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "p100"),
                    quantiles,
                )
            },
        }
    return out


def critical_corner_softcount_from_aerial(
    aerial: torch.Tensor,
    target: torch.Tensor,
    beta: float,
    threshold: float = THRESHOLD,
) -> torch.Tensor:
    """Mean target-aware critical-corner soft error probability for one layout.

    The low-dose corner is limiting for target-positive pixels and the
    high-dose corner is limiting for target-negative pixels. This is a smooth,
    nonconvex surrogate for the corresponding hard error count; it has no LP
    margin offset or margin normalization.
    """
    beta = float(beta)
    threshold = float(threshold)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("softcount beta must be finite and positive")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    aerial, target_bool = _validate_aerial_target(aerial, target)
    critical_margin = torch.where(
        target_bool,
        LOW_DOSE * aerial - threshold,
        threshold - HIGH_DOSE * aerial,
    )
    return torch.sigmoid(-beta * critical_margin).mean()


def target_aware_critical_corner_softcount(
    weights: torch.Tensor,
    bases: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    beta: float,
    threshold: float = THRESHOLD,
) -> torch.Tensor:
    """Equal-layout mean of critical-corner soft pixel error probabilities."""
    if set(bases) != set(targets) or not bases:
        raise ValueError("fit basis and target layout IDs must match and be non-empty")
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("softcount beta must be finite and positive")
    losses = []
    for name, basis in bases.items():
        if basis.ndim != 3 or basis.shape[0] != weights.numel():
            raise ValueError("basis source dimension mismatch for %s" % name)
        aerial = torch.einsum("n,nhw->hw", weights, basis)
        target = torch.as_tensor(targets[name], device=aerial.device)
        losses.append(critical_corner_softcount_from_aerial(
            aerial, target, beta, threshold
        ).reshape(()))
    return torch.stack(losses).mean()


def critical_corner_softcount_value_gradient(
    weights: np.ndarray,
    bases: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    beta: float,
    threshold: float = THRESHOLD,
) -> tuple[float, np.ndarray]:
    """Evaluate the softcount and 49-dimensional float64 source gradient."""
    if not bases:
        raise ValueError("at least one fit basis is required")
    device = next(iter(bases.values())).device
    variable = torch.tensor(
        np.asarray(weights, dtype=np.float64), dtype=torch.float64,
        device=device, requires_grad=True,
    )
    loss = target_aware_critical_corner_softcount(
        variable, bases, targets, beta, threshold
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite critical-corner softcount loss")
    loss.backward()
    if variable.grad is None or not torch.isfinite(variable.grad).all():
        raise FloatingPointError("missing or non-finite critical-corner softcount gradient")
    return float(loss.detach().item()), variable.grad.detach().cpu().numpy().copy()


def critical_corner_softcount_value(
    weights: np.ndarray,
    bases: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    beta: float,
    threshold: float = THRESHOLD,
) -> float:
    if not bases:
        raise ValueError("at least one fit basis is required")
    device = next(iter(bases.values())).device
    variable = torch.as_tensor(
        np.asarray(weights, dtype=np.float64), dtype=torch.float64, device=device
    )
    with torch.no_grad():
        loss = target_aware_critical_corner_softcount(
            variable, bases, targets, beta, threshold
        )
    value = float(loss.item())
    if not math.isfinite(value):
        raise FloatingPointError("non-finite critical-corner softcount value")
    return value


def smooth_pv_from_aerial(
    aerial: torch.Tensor,
    beta: float,
    threshold: float = THRESHOLD,
    low_dose: float = LOW_DOSE,
    high_dose: float = HIGH_DOSE,
) -> torch.Tensor:
    """Mean smooth dose-transition indicator for one FIT layout.

    This target-independent surrogate is the mean difference between the
    high-dose and low-dose soft prints. It is not the thresholded hard PV-band
    count. The beta value is supplied by the caller and follows the registered
    continuation schedule.
    """
    beta = float(beta)
    threshold = float(threshold)
    low_dose = float(low_dose)
    high_dose = float(high_dose)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("smooth PV beta must be finite and positive")
    if not math.isfinite(threshold):
        raise ValueError("smooth PV threshold must be finite")
    if (not math.isfinite(low_dose) or not math.isfinite(high_dose)
            or low_dose <= 0 or high_dose <= low_dose):
        raise ValueError("smooth PV doses must be finite, positive, and increasing")
    if not isinstance(aerial, torch.Tensor) or aerial.ndim != 2:
        raise ValueError("smooth PV aerial must be a height-by-width tensor")
    if not torch.isfinite(aerial).all():
        raise ValueError("smooth PV aerial contains non-finite values")
    low_print = torch.sigmoid(beta * (low_dose * aerial - threshold))
    high_print = torch.sigmoid(beta * (high_dose * aerial - threshold))
    return (high_print - low_print).mean()


def smooth_pv_value_gradient(
    weights: np.ndarray,
    bases: Mapping[str, torch.Tensor],
    beta: float,
    threshold: float = THRESHOLD,
    low_dose: float = LOW_DOSE,
    high_dose: float = HIGH_DOSE,
) -> tuple[float, np.ndarray]:
    """Evaluate an equal-layout smooth-PV mean and its source gradient."""
    if not bases:
        raise ValueError("at least one FIT basis is required")
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.ndim != 1 or not np.isfinite(weight_array).all():
        raise ValueError("smooth PV source weights must be a finite vector")
    device = next(iter(bases.values())).device
    variable = torch.tensor(weight_array, dtype=torch.float64, device=device,
                            requires_grad=True)
    per_layout = []
    for name, basis in bases.items():
        if not isinstance(basis, torch.Tensor) or basis.ndim != 3:
            raise ValueError("basis for %s must have shape (source, height, width)" % name)
        if basis.shape[0] != variable.numel():
            raise ValueError("basis source dimension mismatch for %s" % name)
        if not torch.isfinite(basis).all():
            raise ValueError("basis for %s contains non-finite values" % name)
        aerial = torch.einsum("n,nhw->hw", variable, basis)
        per_layout.append(smooth_pv_from_aerial(
            aerial, beta, threshold, low_dose, high_dose,
        ).reshape(()))
    loss = torch.stack(per_layout).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite smooth PV loss")
    loss.backward()
    if variable.grad is None or not torch.isfinite(variable.grad).all():
        raise FloatingPointError("missing or non-finite smooth PV source gradient")
    return float(loss.detach().item()), variable.grad.detach().cpu().numpy().copy()


def smooth_pv_value(
    weights: np.ndarray,
    bases: Mapping[str, torch.Tensor],
    beta: float,
    threshold: float = THRESHOLD,
    low_dose: float = LOW_DOSE,
    high_dose: float = HIGH_DOSE,
) -> float:
    """Evaluate an equal-layout smooth-PV mean without constructing gradients."""
    if not bases:
        raise ValueError("at least one FIT basis is required")
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.ndim != 1 or not np.isfinite(weight_array).all():
        raise ValueError("smooth PV source weights must be a finite vector")
    device = next(iter(bases.values())).device
    variable = torch.as_tensor(weight_array, dtype=torch.float64, device=device)
    values = []
    with torch.no_grad():
        for name, basis in bases.items():
            if not isinstance(basis, torch.Tensor) or basis.ndim != 3:
                raise ValueError("basis for %s must have shape (source, height, width)" % name)
            if basis.shape[0] != variable.numel():
                raise ValueError("basis source dimension mismatch for %s" % name)
            if not torch.isfinite(basis).all():
                raise ValueError("basis for %s contains non-finite values" % name)
            aerial = torch.einsum("n,nhw->hw", variable, basis)
            values.append(smooth_pv_from_aerial(
                aerial, beta, threshold, low_dose, high_dose,
            ).reshape(()))
        loss = torch.stack(values).mean()
    result = float(loss.item())
    if not math.isfinite(result):
        raise FloatingPointError("non-finite smooth PV value")
    return result
