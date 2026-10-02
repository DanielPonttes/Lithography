"""Constrained source-only LP and Frank-Wolfe experiment; final3 stays closed."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib, json, math, os, sys, time, uuid
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.optimize import linprog
import torch
import light_source as light_source_module
import source_training as source_training_module
from light_source import DifferentiableAbbeLitho, PixelatedLightSource
from scripts import diagnose_source_feasibility as diagnostic
from scripts import run_protected_pvband_experiment as experiment

GRID, INNER, OUTER = 9, 0.3, 0.9
THRESHOLD, DOSES = 0.225, (0.98, 1.0, 1.02)
RHO, TOL = 0.5, 2e-8
SEEDS, BETAS = (17, 29, 43, 71, 101), (200.0, 400.0, 800.0)
LP_CAL = {"band_pixels": 268.0, "L2_pixels": 53.5, "L2_worst_dose_pixels": 149.75}
FIXED_CAL = {"band_pixels": 266.0, "L2_pixels": 461.75, "L2_worst_dose_pixels": 534.5}
GATE = {"band_pixels": 239.4, "L2_pixels": 56.175, "worst_L2_pixels": 157.2375}
REFERENCES = [
    {"citation": "Jia and Lam (2011), SMO/source contrast; source-only adaptation",
     "url": "https://hub.hku.hk/bitstream/10722/155667/1/content.pdf"},
    {"citation": "US7057709B2, source LP and process-window constraints",
     "url": "https://patents.google.com/patent/US7057709B2/en"},
    {"citation": "Lacoste-Julien (2016), nonconvex Frank-Wolfe stationarity",
     "url": "https://arxiv.org/abs/1607.00345"},
]


def sha256_tensor(value):
    array = value.detach().cpu().contiguous().numpy() if isinstance(value, torch.Tensor) else np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class SidecarWriteError(OSError):
    """A display sidecar failed after the canonical snapshot was committed."""
    def __init__(self, path, cause):
        self.path = Path(path)
        super().__init__("could not write run sidecar %s: %s" % (self.path, cause))


def _atomic_sidecar(path, payload):
    try:
        atomic_json(path, payload)
    except Exception as exc:
        raise SidecarWriteError(path, exc) from exc


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def only_fit(payload):
    """Index only fit. torch.load deserializes the aggregate archive, but no other split is indexed or evaluated."""
    return payload["fit"]


def load_fit(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return diagnostic.SourceDataset(**only_fit(payload))


def layout_rows(dataset, split):
    return [{"layout_id": name, "split": split, "mask": dataset.masks[i],
             "target": dataset.targets[i, 0]}
            for i, name in enumerate(dataset.layout_ids)]


def check_hashes(rows, expected, key, label):
    by_id = {x["layout_id"]: x["sha256"] for x in expected}
    if set(by_id) != {x["layout_id"] for x in rows}:
        raise ValueError("%s layout IDs differ from diagnostic" % label)
    for row in rows:
        if sha256_tensor(row[key]) != by_id[row["layout_id"]]:
            raise ValueError("%s hash mismatch for %s" % (label, row["layout_id"]))


def support_mask():
    source = PixelatedLightSource(GRID, sigma_inner=INNER, sigma_outer=OUTER)
    return source._pupil_support.cpu().numpy().astype(bool)


def compress(grid, support):
    grid = np.asarray(grid, dtype=np.float64)
    if grid.shape != support.shape or not np.isfinite(grid).all() or np.any(grid[~support] != 0):
        raise ValueError("invalid source grid/support")
    weights = grid[support].copy()
    if weights.min(initial=0.0) < 0.0 or abs(float(weights.sum()) - 1.0) > TOL:
        raise ValueError("diagnostic source weights violate the simplex")
    return weights


def compress_fixed_control(grid, support):
    """Normalize only the stored float32 fixed-source control for roundoff."""
    grid = np.asarray(grid, dtype=np.float64)
    if grid.shape != support.shape or not np.isfinite(grid).all() or np.any(grid[~support] != 0):
        raise ValueError("invalid fixed-control source grid/support")
    weights = grid[support].copy()
    if weights.min(initial=0.0) < 0.0:
        raise ValueError("fixed-control source contains a negative weight")
    raw_sum = float(weights.sum())
    tolerance = 8.0 * float(np.finfo(np.float32).eps)
    if raw_sum <= 0.0 or abs(raw_sum - 1.0) > tolerance:
        raise ValueError("fixed-control flux error exceeds float32 roundoff tolerance")
    weights /= raw_sum
    verified = compress(expand(weights, support), support)
    metadata = {
        "raw_float32_grid_flux_sum": raw_sum,
        "normalized_flux_sum": float(verified.sum()),
        "flux_correction_absolute": abs(1.0 - raw_sum),
        "allowed_float32_roundoff": tolerance,
        "normalization_applied": raw_sum != 1.0,
    }
    return verified, metadata


def expand(weights, support):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != int(support.sum()):
        raise ValueError("weights do not match source support")
    grid = np.zeros(support.shape, dtype=np.float64)
    grid[support] = weights
    return grid


def fit_matrix(bases, targets):
    if not bases or len(bases) != len(targets):
        raise ValueError("fit bases/targets must be paired")
    matrices, labels, count = [], [], None
    for basis, target in zip(bases, targets):
        basis, target = np.asarray(basis, dtype=np.float64), np.asarray(target)
        if basis.ndim != 3 or target.ndim != 2 or basis.shape[1:] != target.shape:
            raise ValueError("basis must be (N,H,W), target (H,W)")
        if not np.isfinite(basis).all() or not np.isin(target, (0, 1, False, True)).all():
            raise ValueError("basis must be finite and target binary")
        count = basis.shape[0] if count is None else count
        if basis.shape[0] != count:
            raise ValueError("source support differs across masks")
        matrices.append(basis.reshape(count, -1).T)
        labels.append(target.astype(bool).reshape(-1))
    return np.concatenate(matrices), np.concatenate(labels)


def signed_margin(matrix, labels, weights):
    sign = np.where(np.asarray(labels, dtype=bool), 1.0, -1.0)
    return float(np.min(sign * (np.asarray(matrix) @ np.asarray(weights) - THRESHOLD)))


@dataclass
class Polytope:
    matrix: np.ndarray
    labels: np.ndarray
    margin_floor: float
    aub: np.ndarray
    bub: np.ndarray
    aeq: np.ndarray
    beq: np.ndarray
    bounds: tuple

    def verify(self, weights, tol=TOL):
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.size != self.matrix.shape[1] or not np.isfinite(weights).all():
            return {"passed": False, "reason": "invalid_shape_or_nonfinite"}
        margin = signed_margin(self.matrix, self.labels, weights)
        violation = float(np.max(self.aub @ weights - self.bub, initial=-np.inf))
        flux, minimum = float(weights.sum()), float(weights.min(initial=0.0))
        ok = minimum >= -tol and abs(flux - 1) <= tol and violation <= tol and margin >= self.margin_floor - tol
        return {"passed": bool(ok), "flux_sum": flux, "minimum_weight": minimum,
                "minimum_signed_margin": margin, "required_margin_floor": self.margin_floor,
                "max_positive_violation": max(0.0, violation), "tolerance": tol}


def build_polytope(bases, targets, lp_margin, rho=RHO):
    matrix, labels = fit_matrix(bases, targets)
    if not math.isfinite(float(lp_margin)) or lp_margin <= 0 or not 0 < rho <= 1:
        raise ValueError("positive LP margin and rho in (0,1] required")
    sign = np.where(labels, 1.0, -1.0)
    floor = float(rho * lp_margin)
    n = matrix.shape[1]
    return Polytope(matrix, labels, floor, -sign[:, None] * matrix,
                    -sign * THRESHOLD - floor, np.ones((1, n)),
                    np.ones(1), tuple((0.0, None) for _ in range(n)))

def _candidate(raw, poly):
    raw = np.asarray(raw, dtype=np.float64)
    n = poly.matrix.shape[1]
    if raw.size < n or not np.isfinite(raw).all() or raw[:n].min(initial=0) < -TOL:
        return None
    weights = np.maximum(raw[:n], 0.0)
    total = float(weights.sum())
    if total <= 0:
        return None
    weights /= total
    return weights if poly.verify(weights)["passed"] else None


def solve_lp(poly, objective, deadline, time_limit=60.0,
             extra_aub=None, extra_bub=None, extra_bounds=None, check=None):
    """HiGHS float64; retry only success with failed direct residual."""
    n = poly.matrix.shape[1]
    bounds = poly.bounds if extra_bounds is None else extra_bounds
    aub, bub = poly.aub, poly.bub
    aeq = poly.aeq
    if len(bounds) > n:
        aub = np.pad(aub, ((0, 0), (0, len(bounds) - n)))
        aeq = np.pad(aeq, ((0, 0), (0, len(bounds) - n)))
    if extra_aub is not None:
        extra_aub = np.asarray(extra_aub, dtype=np.float64)
        if extra_aub.ndim != 2 or extra_aub.shape[1] != len(bounds):
            raise ValueError("extra LP constraints must match variable dimensions")
        aub = np.concatenate((aub, extra_aub))
        bub = np.concatenate((bub, np.asarray(extra_bub, dtype=np.float64)))
    if len(objective) != len(bounds):
        raise ValueError("LP objective/bounds dimension mismatch")
    attempts = []
    for method, presolve in (("highs", True), ("highs-ipm", False)):
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            return {"status": "deadline", "weights": None, "attempts": attempts}
        limit = min(float(time_limit), remaining)
        result = linprog(
            objective, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=poly.beq,
            bounds=bounds, method=method,
            options={"time_limit": limit, "primal_feasibility_tolerance": 1e-9,
                     "dual_feasibility_tolerance": 1e-9,
                     "ipm_optimality_tolerance": 1e-10, "presolve": presolve},
        )
        attempt = {"method": method, "presolve": presolve,
                   "success": bool(result.success), "status_code": int(result.status),
                   "message": str(result.message), "iterations": int(getattr(result, "nit", 0) or 0),
                   "time_limit_seconds": limit}
        attempts.append(attempt)
        if not result.success or result.x is None:
            return {"status": "solver_failure", "weights": None, "attempts": attempts}
        x = np.asarray(result.x, dtype=np.float64)
        weights = _candidate(x, poly)
        extra_ok = True if check is None else bool(check(x, weights))
        attempt["direct_polytope_verification"] = (
            poly.verify(weights) if weights is not None else {"passed": False}
        )
        attempt["additional_verification_passed"] = extra_ok
        if weights is not None and extra_ok:
            return {"status": "optimal_verified", "weights": weights,
                    "objective_value": float(np.asarray(objective) @ x),
                    "attempts": attempts}
    return {"status": "numerical_residual_failure", "weights": None, "attempts": attempts}


def solve_lmo(poly, objective, deadline, time_limit=60.0):
    return solve_lp(poly, objective, deadline, time_limit)


def boundary_pairs(targets):
    fg, bg, offset = [], [], 0
    for target in targets:
        target = np.asarray(target, dtype=bool)
        h, w = target.shape
        for y in range(h):
            for x in range(w - 1):
                left, right = target[y, x], target[y, x + 1]
                if left != right:
                    fg.append(offset + y * w + (x if left else x + 1))
                    bg.append(offset + y * w + (x + 1 if left else x))
        for y in range(h - 1):
            for x in range(w):
                top, bottom = target[y, x], target[y + 1, x]
                if top != bottom:
                    fg.append(offset + y * w + (x if top else x + w))
                    bg.append(offset + y * w + (x + w if top else x))
        offset += h * w
    if not fg:
        raise ValueError("fit targets contain no adjacent boundary pairs")
    return np.asarray(fg, dtype=np.int64), np.asarray(bg, dtype=np.int64)


def edge_contrast_matrix(matrix, targets):
    fg, bg = boundary_pairs(targets)
    return matrix[fg] - matrix[bg]


def solve_edge_lp(poly, contrasts, deadline, time_limit=60.0):
    n = poly.matrix.shape[1]
    extra = np.zeros((len(contrasts), n + 1), dtype=np.float64)
    extra[:, :n], extra[:, n] = -contrasts, 1.0
    return solve_lp(
        poly, np.r_[np.zeros(n), -1.0], deadline, time_limit,
        extra_aub=extra, extra_bub=np.zeros(len(contrasts)),
        extra_bounds=poly.bounds + ((None, None),),
        check=lambda x, w: w is not None and
             float(x[-1]) <= float(np.min(contrasts @ w)) + TOL,
    )


def aerials(weights, bases_gpu):
    device = next(iter(bases_gpu.values())).device
    w = torch.as_tensor(weights, dtype=torch.float64, device=device)
    return {name: torch.einsum("n,nhw->hw", w, basis)
            for name, basis in bases_gpu.items()}


def smooth_value(weights, bases_gpu, beta):
    ims = aerials(weights, bases_gpu)
    vals = [torch.sigmoid(beta * (1.02 * x - THRESHOLD))
            - torch.sigmoid(beta * (0.98 * x - THRESHOLD))
            for x in ims.values()]
    return float(torch.cat([x.reshape(-1) for x in vals]).mean().item())


def smooth_gradient(weights, bases_gpu, beta):
    device = next(iter(bases_gpu.values())).device
    w = torch.tensor(weights, dtype=torch.float64, device=device, requires_grad=True)
    ims = aerials(w, bases_gpu)
    vals = [torch.sigmoid(beta * (1.02 * x - THRESHOLD))
            - torch.sigmoid(beta * (0.98 * x - THRESHOLD))
            for x in ims.values()]
    loss = torch.cat([x.reshape(-1) for x in vals]).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite smooth PV-band")
    loss.backward()
    if w.grad is None or not torch.isfinite(w.grad).all():
        raise FloatingPointError("missing or nonfinite FW gradient")
    return float(loss.detach().item()), w.grad.detach().cpu().numpy().copy()


def line_search(weights, direction, value, gap, bases_gpu, beta):
    """Armijo search using one precomputed I(w) and I(direction) contraction."""
    if gap <= 0:
        return {"accepted": False, "gamma": 0.0, "value": value, "evaluations": 0}
    current, delta = aerials(weights, bases_gpu), aerials(direction, bases_gpu)
    gamma, evaluations = 1.0, 0
    while gamma >= 2.0 ** -24:
        evaluations += 1
        vals = []
        for name, image in current.items():
            trial = image + gamma * delta[name]
            vals.append(torch.sigmoid(beta * (1.02 * trial - THRESHOLD))
                        - torch.sigmoid(beta * (0.98 * trial - THRESHOLD)))
        trial_value = float(torch.cat([x.reshape(-1) for x in vals]).mean().item())
        if math.isfinite(trial_value) and trial_value <= value - 1e-4 * gamma * gap:
            return {"accepted": True, "gamma": gamma, "value": trial_value,
                    "evaluations": evaluations}
        gamma *= 0.5
    return {"accepted": False, "gamma": 0.0, "value": value,
            "evaluations": evaluations}


def metrics(rows, basis32, weights):
    records, margin, _ = diagnostic._evaluate(rows, basis32, weights, list(DOSES))
    return {"mean": diagnostic._summary(records).get(rows[0]["split"], {}),
            "per_layout": records, "minimum_margin_all_doses": float(margin)}


def nominal_fit_check(fit_rows, basis32, weights):
    """Check zero nominal hard-print errors with the diagnostic float32 path."""
    records = [diagnostic._metrics(row, basis32[row["layout_id"]], weights, (1.0,))
               for row in fit_rows]
    per_layout = {row["layout_id"]: int(metric["L2_pixels"])
                  for row, metric in zip(fit_rows, records)}
    return {"passed": all(value == 0 for value in per_layout.values()),
            "per_layout_L2_pixels": per_layout,
            "total_L2_pixels": sum(per_layout.values()),
            "precision": "original hash-checked float32 basis and sigmoid-50 hard print"}


def save_weights(out_dir, name, weights, support, metadata):
    path = Path(out_dir) / "weights" / (name + ".pt")
    atomic_torch(path, {"weights_supported_float64": torch.tensor(weights, dtype=torch.float64),
                        "weights_full_grid_float64": torch.tensor(expand(weights, support)),
                        "metadata": metadata})
    return str(path)


def prepare_bases(fit_rows, cal_rows, device, expected):
    """Keep the hash-checked float32 bases for metrics; float64 GPU basis is fit-only."""
    sim32 = DifferentiableAbbeLitho(
        PixelatedLightSource(GRID, INNER, OUTER), numerical_aperture=1.35,
        wavelength_nm=193.0, pixel_size_nm=4.0, source_chunk_size=8,
        cache_max_bytes=0,
    ).to(device)
    hard, fit_gpu, fit_cpu, parity = {}, {}, {}, []
    for i, row in enumerate(fit_rows + cal_rows):
        name = row["layout_id"]
        mask = row["mask"][None].to(device=device, dtype=torch.float32)
        b32 = sim32.prepare_basis(mask)
        cpu32 = b32.intensities.detach().cpu().contiguous().clone()
        reference = expected.get(name)
        if reference is None or sha256_tensor(cpu32) != reference["sha256"]:
            raise ValueError("float32 basis hash mismatch for %s" % name)
        if list(cpu32.shape) != reference["shape"] or str(cpu32.dtype) != reference["dtype"]:
            raise ValueError("basis shape/dtype mismatch for %s" % name)
        hard[name] = SimpleNamespace(intensities=cpu32)
        if i == 0:
            with torch.no_grad():
                torch.testing.assert_close(sim32(mask), sim32.evaluate_basis(b32),
                                           rtol=1e-5, atol=1e-6)
                parity.append({"layout_id": name,
                               "max_abs_error": float((sim32(mask) - sim32.evaluate_basis(b32)).abs().max())})
    for row in fit_rows:
        name = row["layout_id"]
        # Promote the exact hash-checked float32 optical basis used by the
        # diagnostic. Do not recompute optics in another precision.
        fit_gpu[name] = hard[name].intensities[0].to(
            device=device, dtype=torch.float64
        ).contiguous()
        fit_cpu[name] = hard[name].intensities[0].numpy().astype(np.float64, copy=True)
    return hard, fit_cpu, fit_gpu, parity


def load_inputs(dataset_path, diagnostic_path, device):
    diag = json.loads(Path(diagnostic_path).read_text(encoding="utf-8"))
    if diagnostic.sha256_file(dataset_path) != diag["input"]["dataset_sha256"]:
        raise ValueError("dataset SHA256 differs from feasibility diagnostic")
    fit = load_fit(dataset_path)  # extracts only payload['fit']
    if len(fit.masks) != 4 or fit.pixel_size_nm != 4.0:
        raise ValueError("expected four registered 4 nm fit layouts")
    _teacher, cal, _rows = experiment.generate_calibration(device)
    if len(cal.masks) != 4 or tuple(cal.masks.shape[-2:]) != (128, 128):
        raise ValueError("expected four regenerated 128x128 calibration layouts")
    fit_rows, cal_rows = layout_rows(fit, "fit"), layout_rows(cal, "calibration")
    inp = diag["input"]
    check_hashes(fit_rows, inp["fit_masks"], "mask", "fit mask")
    check_hashes(fit_rows, inp["fit_targets"], "target", "fit target")
    check_hashes(cal_rows, inp["calibration_masks"], "mask", "calibration mask")
    check_hashes(cal_rows, inp["calibration_targets"], "target", "calibration target")
    return diag, fit, cal, fit_rows, cal_rows


def candidate_row(weights, order, poly, fit_rows, basis32, basis_gpu):
    verify = poly.verify(weights)
    fit_metrics = metrics(fit_rows, basis32, weights)
    return {
        "weights": np.asarray(weights, dtype=np.float64).copy(),
        "checkpoint_order": int(order), "polytope": verify,
        "fit_metrics": fit_metrics,
        "smooth_beta800": smooth_value(weights, basis_gpu, 800.0),
        "fit_qualified": bool(verify["passed"] and fit_metrics["mean"]["L2_pixels"] == 0.0
                              and all(x["L2_pixels"] == 0 for x in fit_metrics["per_layout"])),
    }


def choose_fit_checkpoint(candidates):
    eligible = [x for x in candidates if x["fit_qualified"]]
    if not eligible:
        return None
    return min(eligible, key=lambda x: (
        x["fit_metrics"]["mean"]["band_pixels"],
        x["fit_metrics"]["mean"]["L2_worst_dose_pixels"],
        x["smooth_beta800"], x["checkpoint_order"],
    ))


def report_fw_checkpoint(flush, seed, detail):
    """Relay a seed checkpoint without forwarding its duplicate event key."""
    fields = {key: value for key, value in detail.items() if key != "event"}
    flush("frank_wolfe_checkpoint", current_arm="frank_wolfe",
          current_seed=seed, **fields)


def _candidate_descriptors(candidates):
    return [{"weights": np.asarray(x["weights"], dtype=np.float64).tolist(),
             "checkpoint_order": int(x["checkpoint_order"]), "label": x["label"]}
            for x in candidates]


def _finish_fw_record(record, candidates):
    if record["status"] == "running":
        record["status"] = "complete"
    best = choose_fit_checkpoint(candidates)
    if best is not None:
        record["selected_training_candidate"] = best["label"]
        record["selected_training_metrics"] = best["fit_metrics"]["mean"]
        record["selected_training_weights_full_grid"] = None
        record["selected_weights"] = best["weights"].tolist()
    else:
        record["selected_training_candidate"] = None
    record["selection_rule"] = "fit hard PV band, then fit worst-dose L2, common beta=800, earliest; nominal fit L2=0"
    return record, best


def _fw_snapshot(seed, current, phase_index, next_step, order, record,
                 candidates, checkpoint_kind=None, stage="iterations"):
    return {
        "schema_version": 1, "seed": int(seed), "stage": stage,
        "phase_index": int(phase_index), "next_step": int(next_step),
        "current_weights": np.asarray(current, dtype=np.float64).tolist(),
        "next_checkpoint_order": int(order), "record": copy.deepcopy(record),
        "candidates": _candidate_descriptors(candidates),
        "checkpoint_kind": checkpoint_kind,
    }


def fw_seed(seed, anchor, poly, fit_rows, basis32, basis_gpu,
            deadline, solver_limit, iterations, interval, progress,
            resume_state=None, state_callback=None):
    """Run one seed, optionally restoring its exact next FW step.

    A timeout snapshot keeps the latest accepted point in `current_weights`.
    That point does not become a selection candidate unless it is a registered
    periodic checkpoint or completed block end.
    """
    if resume_state is None or resume_state.get("stage") == "seed_start":
        rng = np.random.default_rng(seed)
        random_vertex = solve_lmo(poly, rng.normal(size=len(anchor)), deadline, solver_limit)
        record = {"seed": seed, "status": "running", "iterations_completed": 0,
                  "random_vertex_solver": {k: v for k, v in random_vertex.items() if k != "weights"},
                  "checkpoint_history": [], "blocks": []}
        anchor_hard = nominal_fit_check(fit_rows, basis32, anchor)
        candidates = [candidate_row(anchor, 0, poly, fit_rows, basis32, basis_gpu)]
        candidates[0]["label"] = "LP anchor"
        candidates[0]["float32_nominal_fit"] = anchor_hard
        if not anchor_hard["passed"]:
            record["status"] = "anchor_nominal_fit_failed"
            record["initial_float32_nominal_fit"] = anchor_hard
            return record, None
        if random_vertex["weights"] is None:
            if time.monotonic() >= deadline or random_vertex["status"] == "deadline":
                record["status"] = "timeout"
                if state_callback:
                    state_callback({"schema_version": 1, "seed": seed,
                                    "stage": "seed_start"})
                return record, choose_fit_checkpoint(candidates)
            record["status"] = random_vertex["status"]
            record["selected_training_candidate"] = "LP anchor"
            record["selected_training_weights_full_grid"] = None
            return record, choose_fit_checkpoint(candidates)
        current = 0.95 * anchor + 0.05 * random_vertex["weights"]
        initial_poly_check = poly.verify(current)
        initial_hard_check = nominal_fit_check(fit_rows, basis32, current)
        record["initial_float32_nominal_fit"] = initial_hard_check
        if not initial_poly_check["passed"] or not initial_hard_check["passed"]:
            record["status"] = "initial_mix_failed_verification"
            record["initial_polytope_check"] = initial_poly_check
            return record, choose_fit_checkpoint(candidates)
        candidates.append(candidate_row(current, 1, poly, fit_rows, basis32, basis_gpu))
        candidates[-1]["label"] = "initial 5% feasible-vertex mix"
        phase_index, next_step, order = 0, 1, 2
    else:
        if resume_state.get("schema_version") != 1 or resume_state.get("seed") != seed:
            raise ValueError("Frank-Wolfe resume snapshot schema or seed mismatch")
        if resume_state.get("stage") in ("complete_seed", "terminal_seed"):
            record = copy.deepcopy(resume_state["record"])
            candidates = [candidate_row(
                np.asarray(x["weights"], dtype=np.float64), x["checkpoint_order"],
                poly, fit_rows, basis32, basis_gpu
            ) for x in resume_state["candidates"]]
            for candidate, descriptor in zip(candidates, resume_state["candidates"]):
                candidate["label"] = descriptor["label"]
            return _finish_fw_record(record, candidates)
        record = copy.deepcopy(resume_state["record"])
        candidates = [candidate_row(
            np.asarray(x["weights"], dtype=np.float64), x["checkpoint_order"],
            poly, fit_rows, basis32, basis_gpu
        ) for x in resume_state["candidates"]]
        for candidate, descriptor in zip(candidates, resume_state["candidates"]):
            candidate["label"] = descriptor["label"]
        current = np.asarray(resume_state["current_weights"], dtype=np.float64)
        phase_index = int(resume_state["phase_index"])
        next_step = int(resume_state["next_step"])
        order = int(resume_state["next_checkpoint_order"])
        record["status"] = "running"
        if record["blocks"] and record["blocks"][-1].get("status") == "running":
            # The open block belongs to this phase and resumes at next_step.
            pass

    def persist(phase, step, kind=None, stage="iterations"):
        if state_callback:
            state_callback(_fw_snapshot(seed, current, phase, step, order, record,
                                        candidates, kind, stage))

    if resume_state is None or resume_state.get("stage") == "seed_start":
        persist(phase_index, next_step, "seed_initialized")

    if resume_state is not None and resume_state.get("stage") == "complete_seed":
        return _finish_fw_record(record, candidates)

    for beta_index in range(phase_index, len(BETAS)):
        beta = BETAS[beta_index]
        active = bool(record["blocks"] and record["blocks"][-1].get("status") == "running"
                      and float(record["blocks"][-1]["beta"]) == beta)
        if active:
            block = record["blocks"][-1]
            start_at = next_step
        else:
            block = {"beta": beta, "start_value": smooth_value(current, basis_gpu, beta),
                     "steps": [], "status": "running"}
            record["blocks"].append(block)
            start_at = 1
        stop = False
        for step in range(start_at, iterations + 1):
            if time.monotonic() >= deadline:
                record["status"] = "running"
                persist(beta_index, step, "timeout_current_trajectory")
                timed_out = copy.deepcopy(record)
                timed_out["status"] = "timeout"
                return timed_out, choose_fit_checkpoint(candidates)
            value, grad = smooth_gradient(current, basis_gpu, beta)
            lmo = solve_lmo(poly, grad, deadline, solver_limit)
            if lmo["weights"] is None and (time.monotonic() >= deadline or lmo["status"] == "deadline"):
                # Retry this same step on resume; do not add a failed trajectory point.
                persist(beta_index, step, "timeout_lmo_retry")
                timed_out = copy.deepcopy(record)
                timed_out["status"] = "timeout"
                return timed_out, choose_fit_checkpoint(candidates)
            item = {"step": step, "loss": value, "lmo_status": lmo["status"],
                    "lmo_attempts": lmo["attempts"]}
            if lmo["weights"] is None:
                block["status"], record["status"], stop = lmo["status"], lmo["status"], True
                block["steps"].append(item)
                break
            gap = float(grad @ (current - lmo["weights"]))
            item["fw_stationarity_gap"] = gap
            item["gap_meaning"] = "nonconvex stationarity diagnostic; not a global PV certificate"
            if not math.isfinite(gap) or gap < -1e-8:
                block["status"], record["status"], stop = "numerical_error", "numerical_error", True
                block["steps"].append(item)
                break
            if gap <= 1e-10:
                item["status"], block["status"] = "stationary_tolerance", "stationary"
                block["steps"].append(item)
                break
            ls = line_search(current, lmo["weights"] - current, value, gap, basis_gpu, beta)
            item["line_search"] = ls
            if not ls["accepted"]:
                item["status"], block["status"] = "no_decrease", "no_progress"
                block["steps"].append(item)
                break
            previous = current
            proposal = previous + ls["gamma"] * (lmo["weights"] - previous)
            check = poly.verify(proposal)
            hard_check = nominal_fit_check(fit_rows, basis32, proposal)
            item["constraint_check"] = check
            item["float32_nominal_fit"] = hard_check
            if not check["passed"] or not hard_check["passed"]:
                item["status"] = "verification_failed"
                block["status"], record["status"], stop = "verification_failed", "numerical_error", True
                block["steps"].append(item)
                break
            current = proposal
            item["status"] = "accepted"
            block["steps"].append(item)
            record["iterations_completed"] += 1
            if step % interval == 0 or step == iterations:
                row = candidate_row(current, order, poly, fit_rows, basis32, basis_gpu)
                row["label"] = "beta%d step%d" % (int(beta), step)
                candidates.append(row)
                record["checkpoint_history"].append({
                    "label": row["label"], "fit_mean": row["fit_metrics"]["mean"],
                    "smooth_beta800": row["smooth_beta800"], "polytope": row["polytope"],
                })
                progress({"event": "checkpoint", "seed": seed, "beta": beta, "step": step,
                          "fit_band": row["fit_metrics"]["mean"]["band_pixels"]})
                order += 1
                persist(beta_index, step + 1, "periodic")
        if block["status"] == "running":
            block["status"] = "complete"
        block["end_value"] = smooth_value(current, basis_gpu, beta)
        # Timeout returns inside the loop with the trajectory snapshot only.
        # Other exits keep the pre-existing block-end candidate semantics.
        end_row = candidate_row(current, order, poly, fit_rows, basis32, basis_gpu)
        end_row["label"] = "beta%d block-end step%d" % (int(beta), len(block["steps"]))
        candidates.append(end_row)
        record["checkpoint_history"].append({
            "label": end_row["label"], "fit_mean": end_row["fit_metrics"]["mean"],
            "smooth_beta800": end_row["smooth_beta800"], "polytope": end_row["polytope"],
        })
        progress({"event": "checkpoint", "seed": seed, "beta": beta,
                  "step": len(block["steps"]), "checkpoint": "block_end",
                  "fit_band": end_row["fit_metrics"]["mean"]["band_pixels"]})
        order += 1
        stage = ("terminal_seed" if stop else
                 "complete_seed" if beta_index + 1 == len(BETAS) else "iterations")
        persist(beta_index + 1, 1, "block_end", stage)
        if stop:
            break
        phase_index, next_step = beta_index + 1, 1

    if record["status"] == "running" and phase_index >= len(BETAS):
        record, best = _finish_fw_record(record, candidates)
        # Persist the terminal seed before the caller updates the run results.
        if state_callback:
            state_callback(_fw_snapshot(seed, current, len(BETAS), 1, order,
                                        record, candidates, "seed_complete", "complete_seed"))
        return record, best
    if record["status"] == "running":
        record["status"] = "numerical_error" if record["blocks"] and record["blocks"][-1].get("status") == "verification_failed" else "complete"
    return _finish_fw_record(record, candidates)



def _control_check(metrics_obj, expected, label):
    for key, value in expected.items():
        actual = float(metrics_obj["mean"][key])
        if not math.isclose(actual, value, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("%s %s=%s expected=%s" % (label, key, actual, value))


def _no_blank_print(metrics_obj):
    return all(
        row["target_positive_pixels"] == 0
        or all(c["predicted_positive_pixels"] > 0 for c in row["per_corner"])
        for row in metrics_obj["per_layout"]
    )


def _aggregate_cal(records):
    keys = ("band_pixels", "L2_pixels", "L2_worst_dose_pixels")
    per_seed = [{"seed": x["seed"], **{k: x["calibration"]["mean"][k] for k in keys}}
                for x in records]
    return {
        "mean": {k: sum(x[k] for x in per_seed) / len(per_seed) for k in keys} if per_seed else {},
        "per_seed": per_seed,
        "per_seed_band_pixels": [x["band_pixels"] for x in per_seed],
    }


def _gate(cal, anchor, no_blank, complete_seeds=True):
    m, bands = cal["mean"], cal.get("per_seed_band_pixels", [cal["mean"]["band_pixels"]])
    checks = {
        "mean_band_le_239_4": m["band_pixels"] <= GATE["band_pixels"],
        "mean_nominal_L2_le_56_175": m["L2_pixels"] <= GATE["L2_pixels"],
        "mean_worst_dose_L2_le_157_2375": m["L2_worst_dose_pixels"] <= GATE["worst_L2_pixels"],
        "every_seed_band_below_anchor": bool(bands) and all(x < anchor["mean"]["band_pixels"] for x in bands),
        "no_positive_target_blank_at_any_dose": bool(no_blank),
        "all_registered_seeds_complete": bool(complete_seeds),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "calibration_role": "development-only gate, not independent generalization evidence"}


def _save_protocol(path, args, diag, gpu, margins, floor, fixed_normalization):
    protocol = {
        "schema_version": 1, "status": "preregistered",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "heldout_status": "not_indexed_or_evaluated", "final3_status": "closed",
        "dataset_sha256": diag["input"]["dataset_sha256"],
        "fit_ids": [x["layout_id"] for x in diag["input"]["fit_masks"]],
        "calibration_ids": [x["layout_id"] for x in diag["input"]["calibration_masks"]],
        "gpu": gpu, "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "fixed_control_flux_normalization": fixed_normalization,
        "optics": {"NA": 1.35, "wavelength_nm": 193.0, "pixel_size_nm": 4.0,
                   "raster": [128, 128], "grid": GRID, "sigma_inner": INNER,
                   "sigma_outer": OUTER, "threshold": THRESHOLD, "doses": list(DOSES)},
        "fit_polytope": {
            "definition": "w>=0, sum(w)=1, signed(target)*(basis64@w-.225)>=rho*mLP at every nominal pixel of all four fit layouts",
            "margin_checks": margins, "rho": RHO, "required_margin_floor": floor,
            "residual_tolerance": TOL,
        },
        "solver": {"HiGHS_float64": True, "per_LP_time_limit_seconds": args.solver_time_limit,
                   "retry": "only successful LP with failed independent residual; highs-ipm/presolve=False, same constraints/tolerances"},
        "arms": {
            "edge_contrast_LP": "maximize minimum adjacent horizontal/vertical bright-minus-dark aerial intensity on fit target boundaries; finite-difference proxy, not NILS",
            "Frank_Wolfe": {
                "objective": "mean sigmoid(beta*(1.02*I-.225))-sigmoid(beta*(.98*I-.225)) on fit pixels",
                "betas": list(BETAS), "seeds": list(SEEDS),
                "initialization": "0.95*LP anchor + 0.05 seeded feasible random-objective LP vertex",
                "max_iterations_per_beta": args.iterations_per_block,
                "selection": "training hard PV band, training worst-dose L2, common beta=800 score, earliest; nominal fit L2=0 required",
                "line_search": "monotone Armijo, precompute I(w) and I(direction) once per iteration",
                "FW_gap": "nonconvex stationarity diagnostic, not a global-optimum or hard-PV certificate",
            },
        },
        "calibration_gate": {
            "frozen_before_candidate_calibration_scoring": True,
            "LP_anchor": LP_CAL, "fixed_annulus": FIXED_CAL,
            "mean_band_max": GATE["band_pixels"], "mean_nominal_L2_max": GATE["L2_pixels"],
            "mean_worst_L2_max": GATE["worst_L2_pixels"],
            "each_seed_band_below_LP_anchor": True, "no_blank_positive_target_any_dose": True,
            "calibration_gradients": False,
        },
        "references": REFERENCES,
        "limits": ["never index or evaluate pooled8 or final3", "no mask/MRC optimization",
                   "no dynamic hotspot weights", "scalar-Abbe source-only experiment is not SOCS parity",
                   "smooth PV and edge contrast are not hard-PV guarantees"],
        "timeout_seconds": args.timeout_seconds,
    }
    atomic_json(path, protocol)
    return protocol


def persist_run_error(out_dir, exc):
    """Persist a post-creation failure while retaining any partial results."""
    out_dir = Path(out_dir)
    now = datetime.now(timezone.utc).isoformat()
    detail = {"type": type(exc).__name__, "message": str(exc)}
    for filename in ("results.json", "progress.json"):
        path = out_dir / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {}
        payload.update({"status": "error", "error": detail, "updated_utc": now})
        atomic_json(path, payload)


def _canonical_hash(payload):
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                     allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunLock:
    """Exclusive run-directory ownership; stale locks are never stolen."""
    def __init__(self, run_dir):
        self.path = Path(run_dir) / ".run.lock"
        self.token = uuid.uuid4().hex
        self.owned = False

    def acquire(self):
        payload = {"token": self.token, "pid": os.getpid(),
                   "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME"),
                   "started_utc": datetime.now(timezone.utc).isoformat()}
        try:
            descriptor = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            try:
                owner = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                owner = {"status": "unreadable"}
            raise RuntimeError("run directory is locked; inspect %s before manual recovery (owner=%s)" %
                               (self.path, json.dumps(owner, sort_keys=True))) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
        self.owned = True
        return self

    def release(self):
        if not self.owned:
            return
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
            if owner.get("token") == self.token:
                self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self.owned = False


def _resume_identity(args, diag, gpu, parity):
    inp = diag["input"]
    return {
        "schema_version": 1,
        "dataset_sha256": inp["dataset_sha256"],
        "dataset_file_sha256": _file_sha256(args.dataset_file),
        "diagnostic_file_sha256": _file_sha256(args.diagnostic_file),
        "fit_masks": inp["fit_masks"], "fit_targets": inp["fit_targets"],
        "calibration_masks": inp["calibration_masks"],
        "calibration_targets": inp["calibration_targets"],
        "basis_descriptors": inp["bases"],
        "basis_parity": parity,
        "implementation_sources": {
            "runner": _file_sha256(__file__),
            "light_source": _file_sha256(light_source_module.__file__),
            "diagnostic": _file_sha256(diagnostic.__file__),
            "protected_experiment": _file_sha256(experiment.__file__),
            "source_training": _file_sha256(source_training_module.__file__),
        },
        "runtime": {"gpu": gpu, "python": sys.version.split()[0],
                    "numpy": np.__version__, "scipy": scipy.__version__,
                    "torch": str(torch.__version__), "cuda": torch.version.cuda},
        "solver": {"device": args.device, "expected_gpu": args.expected_gpu,
                   "per_lp_time_limit_seconds": float(args.solver_time_limit),
                   "iterations_per_block": int(args.iterations_per_block),
                   "checkpoint_interval": int(args.checkpoint_interval),
                   "seeds": list(SEEDS), "betas": list(BETAS),
                   "grid": GRID, "sigma_inner": INNER, "sigma_outer": OUTER,
                   "threshold": THRESHOLD, "doses": list(DOSES),
                   "rho": RHO, "tolerance": TOL},
    }


def _new_run_state(identity, results, phase="controls"):
    return _seal_run_state({"schema_version": 1, "identity": identity,
                            "identity_sha256": _canonical_hash(identity), "phase": phase,
                            "current_seed": None, "fw_state": None,
                            "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "progress": {}, "results": results})


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


def _load_run_state(out, identity):
    path = Path(out) / "run_state.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("run has no resumable run_state.json; refusing to infer trajectory") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("canonical run_state.json is corrupt; refusing stale sidecars") from exc
    if state.get("schema_version") != 1:
        raise ValueError("unsupported resume-state schema")
    _verify_run_state_checksum(state)
    if state.get("identity") != identity or state.get("identity_sha256") != _canonical_hash(identity):
        raise ValueError("resume identity mismatch (dataset, diagnostic, runtime, solver, or runner changed)")
    if state.get("phase") not in ("controls", "edge_contrast_lp", "frank_wolfe",
                                  "calibration_gate", "complete"):
        raise ValueError("invalid canonical run phase")
    if not isinstance(state.get("results"), dict):
        raise ValueError("canonical run state has no results object")
    status = state["results"].get("status")
    if status == "complete" and state["phase"] == "complete":
        return state
    if status == "complete" or state["phase"] == "complete":
        raise ValueError("canonical complete status and phase disagree")
    active_statuses = {
        "preflight_complete": "controls", "controls_running": "controls",
        "edge_contrast_lp_running": "edge_contrast_lp",
        "frank_wolfe_running": "frank_wolfe",
        "calibration_gate_running": "calibration_gate",
        "controls_verified": "edge_contrast_lp",
        "timeout_after_controls": "edge_contrast_lp",
        "timeout_during_edge_lp": "edge_contrast_lp",
        "timeout_after_edge_lp": "frank_wolfe",
        "timeout_during_frank_wolfe": "frank_wolfe",
        "timeout_before_calibration_gate": "calibration_gate",
        "timeout_during_calibration_gate": "calibration_gate",
        "calibration_incomplete": "calibration_gate",
    }
    resumable_phases = {"controls", "edge_contrast_lp", "frank_wolfe", "calibration_gate"}
    if status == "interrupted" and state["phase"] in resumable_phases:
        pass
    elif status in active_statuses and state["phase"] == active_statuses[status]:
        pass
    else:
        raise ValueError("run status %r is not safely resumable" % status)
    return state


def _validate_fw_resume_state(fw_state, seed, anchor, poly, fit_rows, basis32,
                              support, iterations):
    if fw_state is None:
        return
    if not isinstance(fw_state, dict) or fw_state.get("schema_version") != 1:
        raise ValueError("invalid Frank-Wolfe checkpoint schema")
    if fw_state.get("seed") != seed or seed not in SEEDS:
        raise ValueError("Frank-Wolfe checkpoint seed mismatch")
    stage = fw_state.get("stage")
    if stage == "seed_start":
        return
    if stage not in ("iterations", "complete_seed", "terminal_seed"):
        raise ValueError("invalid Frank-Wolfe checkpoint stage")
    record = fw_state.get("record")
    descriptors = fw_state.get("candidates")
    if not isinstance(record, dict) or record.get("seed") != seed or not isinstance(descriptors, list):
        raise ValueError("Frank-Wolfe checkpoint record is malformed")
    terminal_failures = {"solver_failure", "numerical_residual_failure", "numerical_error",
                         "deadline", "infeasible", "unbounded", "iteration_limit",
                         "verification_failed"}
    if stage == "complete_seed" and record.get("status") not in ("running", "complete"):
        raise ValueError("terminal seed snapshot has an invalid status")
    if stage == "terminal_seed" and record.get("status") not in terminal_failures:
        raise ValueError("failed seed snapshot has a nonterminal status")
    if stage == "iterations":
        phase = fw_state.get("phase_index")
        step = fw_state.get("next_step")
        if not isinstance(phase, int) or phase < 0 or phase >= len(BETAS):
            raise ValueError("Frank-Wolfe next beta phase is invalid")
        if not isinstance(step, int) or step < 1 or step > iterations + 1:
            raise ValueError("Frank-Wolfe next step is invalid")
    orders, labels = [], []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("label"), str):
            raise ValueError("Frank-Wolfe candidate descriptor is malformed")
        weights = compress(expand(descriptor.get("weights"), support), support)
        if not poly.verify(weights)["passed"] or not nominal_fit_check(fit_rows, basis32, weights)["passed"]:
            raise ValueError("saved Frank-Wolfe candidate failed feasibility or nominal-fit validation")
        orders.append(descriptor.get("checkpoint_order"))
        labels.append(descriptor["label"])
    if orders != list(range(len(descriptors))) or len(set(labels)) != len(labels):
        raise ValueError("Frank-Wolfe candidate order or labels are corrupt")
    if fw_state.get("next_checkpoint_order") != len(descriptors):
        raise ValueError("Frank-Wolfe next candidate order is corrupt")
    if len(descriptors) < 2 or len(record.get("checkpoint_history", [])) != len(descriptors) - 2:
        raise ValueError("Frank-Wolfe eligible-candidate history is inconsistent")
    current = compress(expand(fw_state.get("current_weights"), support), support)
    if not poly.verify(current)["passed"] or not nominal_fit_check(fit_rows, basis32, current)["passed"]:
        raise ValueError("saved Frank-Wolfe trajectory failed feasibility or nominal-fit validation")
    blocks = record.get("blocks", [])
    accepted = 0
    for block in blocks:
        step_ids = [x.get("step") for x in block.get("steps", [])]
        if step_ids != list(range(1, len(step_ids) + 1)):
            raise ValueError("Frank-Wolfe block step history is corrupt")
        accepted += sum(x.get("status") == "accepted" for x in block.get("steps", []))
    if accepted != record.get("iterations_completed"):
        raise ValueError("Frank-Wolfe accepted-step counter is inconsistent")
    if stage == "iterations":
        active = bool(blocks and blocks[-1].get("status") == "running")
        if active:
            if len(blocks) != fw_state["phase_index"] + 1:
                raise ValueError("Frank-Wolfe active block does not match next beta phase")
            if float(blocks[-1].get("beta")) != BETAS[fw_state["phase_index"]]:
                raise ValueError("Frank-Wolfe active block beta is corrupt")
            if len(blocks[-1].get("steps", [])) + 1 != fw_state["next_step"]:
                raise ValueError("Frank-Wolfe next step does not follow saved block history")
            if any(x.get("status") != "accepted" for x in blocks[-1].get("steps", [])):
                raise ValueError("active Frank-Wolfe block contains an unaccepted step")
        else:
            normal_blocks = {"complete", "stationary", "no_progress"}
            if (len(blocks) != fw_state["phase_index"] or fw_state["next_step"] != 1
                    or (blocks and blocks[-1].get("status") not in normal_blocks)):
                raise ValueError("Frank-Wolfe next phase does not follow completed blocks")
    elif stage == "complete_seed":
        if (fw_state.get("phase_index") != len(BETAS) or len(blocks) != len(BETAS)
                or fw_state.get("next_step") != 1 or not blocks
                or blocks[-1].get("status") not in {"complete", "stationary", "no_progress"}
                or record.get("status") not in ("running", "complete")):
            raise ValueError("completed seed cursor or block history is inconsistent")
    elif stage == "terminal_seed":
        if (fw_state.get("phase_index") != len(blocks) or not 1 <= len(blocks) <= len(BETAS)
                or fw_state.get("next_step") != 1
                or blocks[-1].get("status") not in terminal_failures):
            raise ValueError("failed seed cursor or terminal block is inconsistent")


def _validate_completed_seed_records(records, anchor, poly, fit_rows, basis32, support):
    seen = set()
    for record in records:
        seed = record.get("seed")
        if seed not in SEEDS or seed in seen:
            raise ValueError("completed Frank-Wolfe seed list is corrupt")
        seen.add(seed)
        status = record.get("status")
        if status not in {"complete", "solver_failure", "numerical_residual_failure",
                          "numerical_error", "deadline", "infeasible", "unbounded",
                          "iteration_limit", "anchor_nominal_fit_failed",
                          "initial_mix_failed_verification"}:
            raise ValueError("Frank-Wolfe results contain a nonterminal seed record")
        if status == "complete":
            if "selected_weights" not in record:
                raise ValueError("completed seed has no selected training weights")
            weights = compress(expand(record["selected_weights"], support), support)
            if not poly.verify(weights)["passed"] or not nominal_fit_check(fit_rows, basis32, weights)["passed"]:
                raise ValueError("completed seed selection failed feasibility or nominal-fit validation")
            full = np.asarray(record.get("selected_training_weights_full_grid"), dtype=np.float64)
            full_weights = compress(full, support)
            if not np.allclose(weights, full_weights, rtol=0.0, atol=TOL):
                raise ValueError("completed seed weight artifacts disagree")
        elif "selected_weights" in record:
            weights = compress(expand(record["selected_weights"], support), support)
            if not poly.verify(weights)["passed"] or not nominal_fit_check(fit_rows, basis32, weights)["passed"]:
                raise ValueError("failed seed's optional best candidate is infeasible")
    if [x["seed"] for x in records] != sorted(seen, key=SEEDS.index):
        raise ValueError("Frank-Wolfe seed records are not in registered order")
    if seen != {x["seed"] for x in records}:
        raise ValueError("completed seed set is inconsistent")


def _validate_saved_controls(results):
    controls = results.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("saved run phase requires verified controls")
    for key in ("fit_only_lp_anchor", "fixed_annulus"):
        row = controls.get(key)
        if not isinstance(row, dict) or not isinstance(row.get("fit"), dict):
            raise ValueError("saved control metrics are malformed")
    _control_check(controls["fit_only_lp_anchor"].get("calibration"), LP_CAL, "LP anchor")
    _control_check(controls["fixed_annulus"].get("calibration"), FIXED_CAL, "fixed annulus")


def _validate_saved_edge(results, phase, poly, fit_rows, basis32, support):
    if phase not in ("frank_wolfe", "calibration_gate", "complete"):
        return
    edge = results.get("edge_contrast_lp")
    if not isinstance(edge, dict):
        raise ValueError("Frank-Wolfe phase has no settled edge LP result")
    full = edge.get("weights_full_grid")
    if full is not None:
        weights = compress(full, support)
        if not poly.verify(weights)["passed"] or not nominal_fit_check(fit_rows, basis32, weights)["passed"]:
            raise ValueError("saved edge candidate failed feasibility or nominal-fit validation")
        fit_metrics = edge.get("fit")
        if not isinstance(fit_metrics, dict) or not isinstance(fit_metrics.get("mean"), dict):
            raise ValueError("saved edge candidate has malformed fit metrics")
        for key in ("band_pixels", "L2_pixels", "L2_worst_dose_pixels"):
            if not math.isfinite(float(fit_metrics["mean"][key])):
                raise ValueError("saved edge candidate fit metrics are non-finite")
        expected_qualified = (float(fit_metrics["mean"]["L2_pixels"]) == 0.0
                              and all(float(x["L2_pixels"]) == 0.0
                                      for x in fit_metrics.get("per_layout", [])))
        if bool(edge.get("fit_qualified")) != expected_qualified:
            raise ValueError("saved edge candidate qualification disagrees with fit metrics")
    elif edge.get("fit_qualified"):
        raise ValueError("qualified edge candidate has no saved weights")

    cal = edge.get("calibration")
    if isinstance(cal, dict) and "mean" in cal:
        for key in ("band_pixels", "L2_pixels", "L2_worst_dose_pixels"):
            if not math.isfinite(float(cal["mean"][key])):
                raise ValueError("saved edge calibration metrics are non-finite")
        if edge.get("no_positive_target_blank_any_dose") != _no_blank_print(cal):
            raise ValueError("saved edge calibration blank check is inconsistent")
        if edge.get("gate") != _gate(cal, results["controls"]["fit_only_lp_anchor"]["calibration"],
                                      edge["no_positive_target_blank_any_dose"]):
            raise ValueError("saved edge calibration gate is inconsistent")


def _pending_fw_seeds(records):
    settled = {x["seed"] for x in records
               if x.get("status") not in ("running", "timeout", "initializing")}
    return [seed for seed in SEEDS if seed not in settled]


def _all_seeds_settled(records):
    return not _pending_fw_seeds(records) and len(records) == len(SEEDS)


def _all_seeds_succeeded(records):
    return _all_seeds_settled(records) and all(x.get("status") == "complete" for x in records)


def _metric_mean(metrics_obj):
    return {key: float(metrics_obj["mean"][key])
            for key in ("band_pixels", "L2_pixels", "L2_worst_dose_pixels")}


def _build_comparison(results, all_seeds_settled, calibration_complete=None):
    arms = []
    for key, label in (("fit_only_lp_anchor", "Fit-only LP anchor"),
                       ("fixed_annulus", "Fixed annulus")):
        control = (results.get("controls") or {}).get(key)
        arms.append({"arm": key, "label": label,
                     "fit": {"status": "measured", "mean": _metric_mean(control["fit"])},
                     "calibration": {"status": "measured", "mean": _metric_mean(control["calibration"])}})
    edge = results.get("edge_contrast_lp") or {}
    edge_cal = edge.get("calibration")
    if isinstance(edge_cal, dict) and "mean" in edge_cal:
        edge_cal_summary = {"status": "measured", "mean": _metric_mean(edge_cal)}
    else:
        edge_cal_summary = {"status": edge_cal.get("status", "not_evaluated") if isinstance(edge_cal, dict)
                            else "not_evaluated"}
    arms.append({"arm": "edge_contrast_lp", "label": "Edge contrast LP",
                 "fit": ({"status": "measured", "mean": _metric_mean(edge["fit"]) }
                         if edge.get("fit") and "mean" in edge["fit"] else
                         {"status": edge.get("status", "not_solved")}),
                 "calibration": edge_cal_summary})
    seeds = [x for x in results.get("frank_wolfe", [])
             if x.get("status") == "complete" and x.get("selected_training_metrics")]
    all_successful = _all_seeds_succeeded(results.get("frank_wolfe", []))
    if seeds:
        keys = ("band_pixels", "L2_pixels", "L2_worst_dose_pixels")
        per_seed_fit = [{"seed": x["seed"], **_metric_mean(x["selected_training_metrics"])}
                        for x in seeds]
        fit_summary = {key: sum(x[key] for x in per_seed_fit) / len(per_seed_fit) for key in keys}
        fit_detail = {"status": "measured" if all_successful else "measured_partial_successes",
                      "mean": fit_summary, "per_seed": per_seed_fit}
    else:
        fit_detail = {"status": "not_measured"}
    fw = results.get("frank_wolfe_arm") or {}
    fw_cal = fw.get("calibration")
    if all_successful and fw_cal and "mean" in fw_cal:
        fw_cal_detail = {"status": "measured", "mean": _metric_mean(fw_cal),
                         "per_seed": fw_cal.get("per_seed", [])}
    else:
        fw_cal_detail = {"status": "not_evaluated_incomplete_seed_set" if not all_seeds_settled
                         else "not_evaluated_fw_incomplete_seed_set" if not all_successful
                         else fw.get("status", "not_evaluated")}
    arms.append({"arm": "frank_wolfe", "label": "Frank-Wolfe",
                 "fit": fit_detail, "calibration": fw_cal_detail})
    status = ("calibration_closed_incomplete_seed_set" if not all_seeds_settled else
              "calibration_incomplete" if calibration_complete is False else
              "ready_fw_incomplete" if not all_successful else "ready")
    return {"status": status,
            "metric_keys": ["band_pixels", "L2_pixels", "L2_worst_dose_pixels"],
            "calibration_role": "development-only; no generalization claim",
            "arms": arms}


def _mark_candidate_calibration_closed(results):
    edge = results.get("edge_contrast_lp")
    if edge:
        edge["calibration"] = {"status": "not_evaluated_incomplete_seed_set"}
    for record in results.get("frank_wolfe", []):
        if not (isinstance(record.get("calibration"), dict)
                and "mean" in record["calibration"]):
            record["calibration"] = {"status": "not_evaluated_incomplete_seed_set"}
    results["selection"] = {"status": "calibration_closed_incomplete_seed_set",
                            "qualified_arms": [], "selected_arm": None,
                            "final3": "closed; not indexed or evaluated"}


def _run_resumable(args):
    if args.resume_run:
        out = Path(args.resume_run)
        if not out.is_absolute() or not out.is_dir():
            raise ValueError("--resume-run must name an existing absolute run directory")
        args._run_lock = RunLock(out).acquire()
    elif not Path(args.output_root).is_absolute():
        raise ValueError("--output-root must be absolute")
    if args.timeout_seconds <= 0 or args.solver_time_limit <= 0:
        raise ValueError("timeouts must be positive")
    if args.iterations_per_block < 1 or args.checkpoint_interval < 1:
        raise ValueError("iteration counts must be positive")
    deadline = time.monotonic() + args.timeout_seconds
    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("run requires CUDA")
    gpu = torch.cuda.get_device_name(torch.device("cuda"))
    if gpu != args.expected_gpu:
        raise RuntimeError("expected %s; found %s" % (args.expected_gpu, gpu))
    torch.cuda.synchronize(torch.device("cuda"))

    diag = json.loads(Path(args.diagnostic_file).read_text(encoding="utf-8"))
    if diagnostic.sha256_file(args.dataset_file) != diag["input"]["dataset_sha256"]:
        raise ValueError("dataset file SHA256 differs from feasibility diagnostic")
    fit = load_fit(args.dataset_file)
    if len(fit.masks) != 4 or fit.pixel_size_nm != 4.0:
        raise ValueError("expected four registered 4 nm fit layouts")
    _teacher, cal, _cal_rows = experiment.generate_calibration("cuda")
    if len(cal.masks) != 4 or tuple(cal.masks.shape[-2:]) != (128, 128):
        raise ValueError("expected four regenerated 128x128 calibration layouts")
    fit_rows, cal_rows = layout_rows(fit, "fit"), layout_rows(cal, "calibration")
    inp = diag["input"]
    check_hashes(fit_rows, inp["fit_masks"], "mask", "fit mask")
    check_hashes(fit_rows, inp["fit_targets"], "target", "fit target")
    check_hashes(cal_rows, inp["calibration_masks"], "mask", "calibration mask")
    check_hashes(cal_rows, inp["calibration_targets"], "target", "calibration target")
    expected_basis = {x["layout_id"]: x for x in inp["bases"]}
    basis32, basis64_cpu, basis64_gpu, parity = prepare_bases(
        fit_rows, cal_rows, "cuda", expected_basis
    )
    support = support_mask()
    lp = diag["scenarios"]["fit_only.nominal"]
    if lp.get("status") != "positive_margin_feasible":
        raise ValueError("fit-only nominal LP is not positive-margin feasible")
    anchor = compress(lp["source_weights_full_grid"], support)
    fixed, fixed_normalization = compress_fixed_control(
        diag["baselines"]["A0_initial_annulus_no_jitter"]["source_weights_full_grid"], support)
    targets = [x["target"].numpy() for x in fit_rows]
    arrays64 = [basis64_cpu[x["layout_id"]] for x in fit_rows]
    arrays32 = [basis32[x["layout_id"]].intensities[0].numpy() for x in fit_rows]
    matrix64, labels64 = fit_matrix(arrays64, targets)
    matrix32, labels32 = fit_matrix(arrays32, targets)
    reported = float(lp["lp_optimal_margin"])
    margin64, margin32 = signed_margin(matrix64, labels64, anchor), signed_margin(matrix32, labels32, anchor)
    if (reported <= 0 or margin32 <= 0 or margin64 <= 0
            or abs(margin32 - reported) > TOL or abs(margin64 - reported) > TOL):
        raise ValueError("LP anchor failed independent fit margin verification")
    poly = build_polytope(arrays64, targets, reported, RHO)
    anchor_check = poly.verify(anchor)
    if not anchor_check["passed"]:
        raise ValueError("LP anchor violates the constrained polytope")
    contrasts = edge_contrast_matrix(matrix64, targets)
    identity = _resume_identity(args, diag, gpu, parity)

    if args.resume_run:
        state = _load_run_state(out, identity)
        results = state["results"]
        if results.get("final3_status") != "closed" or results.get("heldout_status") != "not_indexed_or_evaluated":
            raise ValueError("resume state violates the closed final3 protocol")
        _validate_completed_seed_records(results.get("frank_wolfe", []), anchor, poly,
                                         fit_rows, basis32, support)
        if state["phase"] != "controls":
            _validate_saved_controls(results)
        _validate_saved_edge(results, state["phase"], poly, fit_rows, basis32, support)
        active_seed, fw_state = state.get("current_seed"), state.get("fw_state")
        if state["phase"] == "frank_wolfe" and ((fw_state is None) != (active_seed is None)):
            raise ValueError("Frank-Wolfe current seed and trajectory state disagree")
        if state["phase"] != "frank_wolfe" and (fw_state is not None or active_seed is not None):
            raise ValueError("non-training phase has an active Frank-Wolfe trajectory")
        if active_seed is not None:
            pending_seeds = _pending_fw_seeds(results.get("frank_wolfe", []))
            if not pending_seeds or active_seed != pending_seeds[0]:
                raise ValueError("current seed is not the next incomplete seed")
            _validate_fw_resume_state(fw_state, active_seed, anchor, poly, fit_rows,
                                      basis32, support, args.iterations_per_block)
        if state["phase"] == "calibration_gate" and not _all_seeds_settled(results.get("frank_wolfe", [])):
            raise ValueError("calibration phase requires all registered training seeds settled")
        if state["phase"] == "complete":
            if (not _all_seeds_settled(results.get("frank_wolfe", []))
                    or (results.get("selection") or {}).get("status") != "development_gate_complete"):
                raise ValueError("canonical complete run has unsettled seeds or incomplete selection")
        args._active_output = out
        args._state_accepted = True
        progress = copy.deepcopy(state.get("progress") or {})
        if state["phase"] == "complete":
            _atomic_sidecar(out / "results.json", results)
            _atomic_sidecar(out / "progress.json", progress)
            args._resume_noop = True
            return out
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        out = Path(args.output_root) / run_id
        out.mkdir(parents=True, exist_ok=False)
        args._run_lock = RunLock(out).acquire()
        protocol = _save_protocol(out / "protocol.json", args, diag, gpu,
                                  {"diagnostic": reported, "float32_recomputed": margin32,
                                   "float64_recomputed": margin64, "used": reported},
                                  poly.margin_floor, fixed_normalization)
        results = {"schema_version": 1, "status": "initializing",
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "result_directory": str(out), "heldout_status": "not_indexed_or_evaluated",
                   "final3_status": "closed", "protocol": protocol,
                   "fixed_control_flux_normalization": fixed_normalization,
                   "basis_hashes_verified": True, "basis_parity": parity,
                   "anchor_polytope_verification": anchor_check,
                   "edge_pair_count": int(len(contrasts)), "controls": None,
                   "edge_contrast_lp": None, "frank_wolfe": [],
                   "selection": {"status": "closed_until_training_frozen"}}
        state = _new_run_state(identity, results)
        progress = {"status": "initializing", "phase": "controls",
                    "updated_utc": datetime.now(timezone.utc).isoformat()}
        args._state_accepted = True
        args._active_output = out

    args._run_state = state
    args._run_results = results
    runtime = results.setdefault("runtime", {})
    args._runtime_base_seconds = float(runtime.get("cumulative_seconds", 0.0))
    args._runtime_base_invocations = int(runtime.get("invocation_count", 0))
    missing = object()

    def flush(event, status=None, phase=None, current_seed=missing,
              fw_state=missing, **fields):
        if status is not None:
            results["status"] = status
        if phase is not None:
            state["phase"] = phase
        if current_seed is not missing:
            state["current_seed"] = current_seed
        if fw_state is not missing:
            state["fw_state"] = fw_state
        progress.update(fields)
        progress.update({"status": results.get("status"), "phase": state["phase"],
                         "event": event,
                         "current_seed": state.get("current_seed"),
                         "updated_utc": datetime.now(timezone.utc).isoformat()})
        elapsed = max(0.0, time.monotonic() - args._invocation_started)
        runtime = results.setdefault("runtime", {})
        runtime["cumulative_seconds"] = args._runtime_base_seconds + elapsed
        runtime["last_invocation_seconds"] = elapsed
        runtime["last_timeout_budget_seconds"] = float(args.timeout_seconds)
        runtime["invocation_count"] = args._runtime_base_invocations + 1
        state["progress"] = progress
        state["updated_utc"] = progress["updated_utc"]
        # The canonical snapshot is always replaced before the display sidecars.
        _seal_run_state(state)
        atomic_json(out / "run_state.json", state)
        args._last_canonical_state = copy.deepcopy(state)
        _atomic_sidecar(out / "progress.json", progress)
        _atomic_sidecar(out / "results.json", results)

    args._commit_state = flush
    if args.resume_run:
        state["progress"] = progress
        running = {"controls": "controls_running", "edge_contrast_lp": "edge_contrast_lp_running",
                   "frank_wolfe": "frank_wolfe_running", "calibration_gate": "calibration_gate_running"}.get(state["phase"])
        if state["phase"] != "complete":
            flush("resume_started", status=running or "resuming")
    else:
        flush("preflight_complete", status="preflight_complete")

    if state["phase"] == "controls":
        flush("controls_started", status="controls_running", phase="controls")
        controls = {
            "fit_only_lp_anchor": {"fit": metrics(fit_rows, basis32, anchor),
                                    "calibration": metrics(cal_rows, basis32, anchor)},
            "fixed_annulus": {"fit": metrics(fit_rows, basis32, fixed),
                              "calibration": metrics(cal_rows, basis32, fixed),
                              "source_flux_normalization": fixed_normalization},
        }
        _control_check(controls["fit_only_lp_anchor"]["calibration"], LP_CAL, "LP anchor")
        _control_check(controls["fixed_annulus"]["calibration"], FIXED_CAL, "fixed annulus")
        results["controls"] = controls
        save_weights(out, "lp_anchor", anchor, support, {"kind": "LP fit-only control"})
        save_weights(out, "fixed_annulus", fixed, support,
                     {"kind": "fixed-source control", "flux_normalization": fixed_normalization})
        flush("controls_verified", status="edge_contrast_lp_running", phase="edge_contrast_lp")
        if time.monotonic() >= deadline:
            results["comparison"] = _build_comparison(results, False)
            flush("timeout_after_controls", status="timeout_after_controls")
            return out

    if state["phase"] == "edge_contrast_lp":
        flush("edge_contrast_lp_started", status="edge_contrast_lp_running",
              phase="edge_contrast_lp", current_arm="edge_contrast_lp")
        edge = solve_edge_lp(poly, contrasts, deadline, args.solver_time_limit)
        edge_candidate = None
        if edge["weights"] is not None:
            edge_candidate = candidate_row(edge["weights"], 0, poly, fit_rows, basis32, basis64_gpu)
            edge_candidate["minimum_adjacent_edge_contrast"] = float(np.min(contrasts @ edge["weights"]))
            edge_candidate["contrast_note"] = "finite-difference intensity proxy, not NILS or a hard-PV guarantee"
            results["edge_contrast_lp"] = {
                "status": edge["status"], "solver": {k: v for k, v in edge.items() if k != "weights"},
                "fit": edge_candidate["fit_metrics"], "polytope": edge_candidate["polytope"],
                "minimum_adjacent_edge_contrast": edge_candidate["minimum_adjacent_edge_contrast"],
                "contrast_note": edge_candidate["contrast_note"],
                "weights_full_grid": expand(edge["weights"], support).tolist(),
                "fit_qualified": edge_candidate["fit_qualified"],
            }
            save_weights(out, "edge_contrast_lp", edge["weights"], support,
                         {"kind": "constrained minimum edge-contrast LP"})
        else:
            results["edge_contrast_lp"] = {
                "status": edge["status"], "solver": {k: v for k, v in edge.items() if k != "weights"}}
        if edge["weights"] is None and (time.monotonic() >= deadline or edge["status"] == "deadline"):
            results["comparison"] = _build_comparison(results, False)
            flush("timeout_during_edge_lp", status="timeout_during_edge_lp")
            return out
        flush("edge_contrast_lp_complete", status="frank_wolfe_running",
              phase="frank_wolfe", current_arm="frank_wolfe")
        if time.monotonic() >= deadline:
            results["comparison"] = _build_comparison(results, False)
            flush("timeout_after_edge_lp", status="timeout_after_edge_lp")
            return out

    training_complete = _all_seeds_succeeded(results.get("frank_wolfe", []))
    if state["phase"] == "frank_wolfe":
        records = results.setdefault("frank_wolfe", [])
        _validate_completed_seed_records(records, anchor, poly, fit_rows, basis32, support)
        active_seed, active_state = state.get("current_seed"), state.get("fw_state")
        settled_seeds = {x["seed"] for x in records
                         if x.get("status") not in ("running", "timeout", "initializing")}
        for seed in SEEDS:
            if seed in settled_seeds:
                continue
            prior = [x for x in records if x.get("seed") == seed]
            if prior:
                raise ValueError("cannot resume a nonterminal seed already present in results")
            if active_seed is not None and active_seed != seed:
                raise ValueError("active seed does not match the registered seed order")
            if active_seed is None:
                if time.monotonic() >= deadline:
                    active_seed = seed
                    active_state = {"schema_version": 1, "seed": seed, "stage": "seed_start"}
                    flush("timeout_before_seed", status="timeout_during_frank_wolfe",
                          phase="frank_wolfe", current_arm="frank_wolfe",
                          current_seed=seed, fw_state=active_state)
                    _mark_candidate_calibration_closed(results)
                    results["comparison"] = _build_comparison(results, False)
                    flush("timeout_before_seed", status="timeout_during_frank_wolfe")
                    return out
                active_seed = seed
                active_state = {"schema_version": 1, "seed": seed, "stage": "seed_start"}
                flush("frank_wolfe_seed_started", status="frank_wolfe_running",
                      phase="frank_wolfe", current_arm="frank_wolfe",
                      current_seed=seed, fw_state=active_state)

            def on_fw_state(snapshot):
                phase_index = snapshot.get("phase_index", 0)
                detail = {"beta": BETAS[phase_index] if phase_index < len(BETAS) else None,
                          "step": snapshot.get("next_step"),
                          "checkpoint": snapshot.get("checkpoint_kind"),
                          "trajectory_stage": snapshot.get("stage")}
                flush("frank_wolfe_checkpoint", status="frank_wolfe_running",
                      phase="frank_wolfe", current_arm="frank_wolfe",
                      current_seed=seed, fw_state=snapshot, **detail)

            record, best = fw_seed(
                seed, anchor, poly, fit_rows, basis32, basis64_gpu,
                deadline, args.solver_time_limit, args.iterations_per_block,
                args.checkpoint_interval,
                lambda detail: report_fw_checkpoint(flush, seed, detail),
                resume_state=active_state, state_callback=on_fw_state,
            )
            if record["status"] == "timeout":
                _mark_candidate_calibration_closed(results)
                results["comparison"] = _build_comparison(results, False)
                flush("timeout_during_frank_wolfe", status="timeout_during_frank_wolfe",
                      phase="frank_wolfe", current_arm="frank_wolfe",
                      current_seed=seed, fw_state=state.get("fw_state"))
                return out
            if best is not None:
                save_weights(out, "frank_wolfe_seed_%d" % seed, best["weights"], support,
                             {"seed": seed, "selected_checkpoint": best.get("label")})
                record["selected_training_candidate"] = best.get("label")
                record["selected_training_weights_full_grid"] = expand(best["weights"], support).tolist()
                record["selected_training_metrics"] = best["fit_metrics"]
            records.append(record)
            records.sort(key=lambda row: SEEDS.index(row["seed"]))
            active_seed, active_state = None, None
            settled_seeds.add(seed)
            flush("frank_wolfe_seed_complete", status="frank_wolfe_running",
                  phase="frank_wolfe", current_arm="frank_wolfe",
                  current_seed=None, fw_state=None, seed_status=record["status"])

        training_settled = _all_seeds_settled(records)
        training_complete = _all_seeds_succeeded(records)
        if not training_settled:
            for record in records:
                if "calibration" not in record:
                    record["calibration"] = {"status": "not_evaluated_incomplete_seed_set"}
            if results.get("edge_contrast_lp") is not None:
                results["edge_contrast_lp"]["calibration"] = {"status": "not_evaluated_incomplete_seed_set"}
            results["frank_wolfe_arm"] = {
                "status": "incomplete_seed_set",
                "completed_qualified_seeds": sum(x.get("status") == "complete" for x in records),
                "required_seed_count": len(SEEDS),
            }
            results["selection"] = {"status": "calibration_closed_incomplete_seed_set",
                                    "qualified_arms": [], "selected_arm": None,
                                    "final3": "closed; not indexed or evaluated"}
            results["comparison"] = _build_comparison(results, False)
            flush("incomplete_seed_set", status="timeout_during_frank_wolfe",
                  phase="frank_wolfe", current_seed=None, fw_state=None)
            return out

        if not training_complete:
            for record in records:
                if record.get("status") != "complete":
                    record["calibration"] = {"status": "not_evaluated_fw_incomplete_seed_set"}
            results["frank_wolfe_arm"] = {
                "status": "incomplete_seed_set",
                "completed_qualified_seeds": sum(x.get("status") == "complete" for x in records),
                "required_seed_count": len(SEEDS),
            }

        flush("training_seeds_frozen", status="calibration_gate_running",
              phase="calibration_gate", current_arm="calibration_gate",
              current_seed=None, fw_state=None)
        if time.monotonic() >= deadline:
            results["comparison"] = _build_comparison(results, True, False)
            flush("timeout_before_calibration_gate", status="timeout_before_calibration_gate")
            return out

    if state["phase"] == "calibration_gate":
        anchor_cal = results["controls"]["fit_only_lp_anchor"]["calibration"]
        edge = results.get("edge_contrast_lp") or {}
        if edge.get("fit_qualified"):
            edge_cal = edge.get("calibration")
            if not (isinstance(edge_cal, dict) and "mean" in edge_cal):
                if time.monotonic() >= deadline:
                    edge["calibration"] = {"status": "not_evaluated_timeout"}
                    results["comparison"] = _build_comparison(results, True, False)
                    flush("timeout_during_calibration_gate", status="timeout_during_calibration_gate",
                          current_arm="calibration_gate")
                    return out
                edge_weights = compress(edge["weights_full_grid"], support)
                edge_cal = metrics(cal_rows, basis32, edge_weights)
                edge["calibration"] = edge_cal
                no_blank = _no_blank_print(edge_cal)
                edge["no_positive_target_blank_any_dose"] = no_blank
                edge["gate"] = _gate(edge_cal, anchor_cal, no_blank)
                flush("edge_candidate_calibration_complete", status="calibration_gate_running",
                      phase="calibration_gate", current_arm="calibration_gate")
        elif edge:
            edge["calibration"] = {"status": "not_evaluated_unqualified"}

        records = results.get("frank_wolfe", [])
        if training_complete:
            for record in records:
                cal_record = record.get("calibration")
                if isinstance(cal_record, dict) and "mean" in cal_record:
                    continue
                if time.monotonic() >= deadline:
                    record["calibration"] = {"status": "not_evaluated_timeout"}
                    results["comparison"] = _build_comparison(results, True, False)
                    flush("timeout_during_calibration_gate", status="timeout_during_calibration_gate",
                          current_arm="calibration_gate")
                    return out
                weights = np.asarray(record["selected_weights"], dtype=np.float64)
                cal_metrics = metrics(cal_rows, basis32, weights)
                record["calibration"] = cal_metrics
                record["no_positive_target_blank_any_dose"] = _no_blank_print(cal_metrics)
                results["comparison"] = _build_comparison(results, True, False)
                flush("frank_wolfe_seed_calibration_complete", status="calibration_gate_running",
                      phase="calibration_gate", current_arm="calibration_gate",
                      calibration_seed=record["seed"])

        calibrated = [x for x in records
                      if x.get("status") == "complete"
                      and isinstance(x.get("calibration"), dict)
                      and "mean" in x["calibration"]]
        if training_complete and len(calibrated) != len(SEEDS):
            results["comparison"] = _build_comparison(results, True, False)
            flush("calibration_incomplete", status="calibration_incomplete")
            return out
        if training_complete:
            fw_cal = _aggregate_cal(calibrated)
            no_blank = all(x["no_positive_target_blank_any_dose"] for x in calibrated)
            results["frank_wolfe_arm"] = {
                "calibration": fw_cal,
                "gate": _gate(fw_cal, anchor_cal, no_blank, complete_seeds=True),
            }
        edge_pass = bool(edge.get("gate", {}).get("passed"))
        fw_pass = bool((results.get("frank_wolfe_arm") or {}).get("gate", {}).get("passed"))
        qualified = [name for name, passed in (("edge_contrast_lp", edge_pass),
                                                ("frank_wolfe", fw_pass)) if passed]
        results["selection"] = {
            "status": "development_gate_complete", "qualified_arms": qualified,
            "selected_arm": qualified[0] if len(qualified) == 1 else (
                min(qualified, key=lambda name: (
                    edge["calibration"]["mean"]["band_pixels"] if name == "edge_contrast_lp"
                    else results["frank_wolfe_arm"]["calibration"]["mean"]["band_pixels"]
                )) if qualified else None
            ),
            "calibration_role": "development-only; no generalization claim",
            "final3": "closed; not indexed or evaluated",
        }
        results["comparison"] = _build_comparison(results, True, True)
        results["status"] = "complete"
        flush("run_complete", status="complete", phase="complete",
              current_arm=None, current_seed=None, fw_state=None)
        return out
    raise ValueError("unsupported resume phase %r" % state.get("phase"))


def _persist_emergency_state(args, status, exc=None):
    """Use the newest valid canonical generation, falling back to known memory."""
    out = Path(args._active_output)
    try:
        state = json.loads((out / "run_state.json").read_text(encoding="utf-8"))
        _verify_run_state_checksum(state)
        if state.get("schema_version") != 1 or not isinstance(state.get("results"), dict):
            raise ValueError("invalid canonical run state")
    except BaseException:
        state = copy.deepcopy(args._last_canonical_state)
    if state is None:
        return

    # A committed complete generation is terminal. Never downgrade it because
    # a later sidecar write or cleanup step failed.
    complete = state.get("phase") == "complete" and state.get("results", {}).get("status") == "complete"
    if complete:
        args._last_canonical_state = copy.deepcopy(state)
        progress = state.get("progress") or {}
        _atomic_sidecar(out / "progress.json", progress)
        _atomic_sidecar(out / "results.json", state["results"])
        return

    # status=None denotes a sidecar-only failure: retain the canonical bytes
    # and repair display files from that generation if possible.
    if status is None:
        args._last_canonical_state = copy.deepcopy(state)
        _atomic_sidecar(out / "progress.json", state.get("progress") or {})
        _atomic_sidecar(out / "results.json", state["results"])
        return

    state = copy.deepcopy(state)
    results = state["results"]
    now = datetime.now(timezone.utc).isoformat()
    results["status"] = status
    if exc is not None:
        results["error"] = {"type": type(exc).__name__, "message": str(exc)}
    progress = state.setdefault("progress", {})
    progress.update({"status": status, "event": status, "updated_utc": now})
    state["updated_utc"] = now
    _seal_run_state(state)
    atomic_json(out / "run_state.json", state)
    args._last_canonical_state = copy.deepcopy(state)
    _atomic_sidecar(out / "progress.json", progress)
    _atomic_sidecar(out / "results.json", results)


def run(args):
    args._run_lock = None
    args._active_output = None
    args._state_accepted = False
    args._resume_noop = False
    args._emergency_committed = False
    args._last_canonical_state = None
    args._invocation_started = time.monotonic()
    try:
        return _run_resumable(args)
    except KeyboardInterrupt:
        if args._state_accepted and args._run_lock is not None and args._run_lock.owned:
            try:
                _persist_emergency_state(args, "interrupted")
            except BaseException:
                pass
            args._emergency_committed = True
        raise
    except BaseException as exc:
        if args._state_accepted and args._run_lock is not None and args._run_lock.owned:
            try:
                if isinstance(exc, SidecarWriteError):
                    _persist_emergency_state(args, None)
                else:
                    _persist_emergency_state(args, "error", exc)
            except BaseException:
                pass
            args._emergency_committed = True
        raise
    finally:
        pending_interrupt = None
        if (args._state_accepted and not args._resume_noop and not args._emergency_committed
                and hasattr(args, "_commit_state")):
            try:
                args._commit_state("invocation_end", status=args._run_results.get("status"))
            except KeyboardInterrupt as exc:
                try:
                    _persist_emergency_state(args, "interrupted")
                except BaseException:
                    pass
                args._emergency_committed = True
                pending_interrupt = exc
            except BaseException as exc:
                try:
                    _persist_emergency_state(args, None if isinstance(exc, SidecarWriteError) else "error",
                                             None if isinstance(exc, SidecarWriteError) else exc)
                except BaseException:
                    pass
                args._emergency_committed = True
                print(json.dumps({"status": "sidecar_write_error" if isinstance(exc, SidecarWriteError)
                                  else "final_snapshot_error", "message": str(exc)},
                                 ensure_ascii=False), file=sys.stderr)
        if args._run_lock is not None:
            args._run_lock.release()
        if pending_interrupt is not None:
            raise pending_interrupt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--diagnostic-file", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--resume-run", help="resume an existing run directory with matching identity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-gpu", default="NVIDIA GeForce RTX 5090")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--solver-time-limit", type=float, default=60.0)
    parser.add_argument("--iterations-per-block", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    args = parser.parse_args(argv)
    if not args.resume_run and not args.output_root:
        parser.error("--output-root is required for a new run")
    return args


def main(argv=None):
    args = None
    try:
        args = parse_args(argv)
        out = run(args)
        results = json.loads((out / "results.json").read_text(encoding="utf-8"))
        print(json.dumps({"result_directory": str(out), "status": results["status"]}))
        return 0 if results["status"] == "complete" else 1
    except Exception as exc:
        out = getattr(args, "_active_output", None) if args is not None else None
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "message": str(exc), "result_directory": str(out) if out else None},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
