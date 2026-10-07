"""Sparse protected-error MILP for the registered source-only experiment.

The binary objective counts anchor errors that cannot be repaired with a fixed
epsilon buffer. It is a conservative buffered count, not an exact hard-PV
objective. This module contains no data loading or calibration access.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import sys
import time
from typing import Sequence

import numpy as np
from scipy import sparse

OBJECTIVE_ID = "target_aware_pareto_protected_buffered_count_v1"
LP_MARGIN = 0.0002215098124816199
RHO = 0.01
EPSILON = 2e-6
THRESHOLD = 0.225
LOW_DOSE = 0.98
HIGH_DOSE = 1.02
SOLVER_TOLERANCE = 1e-9
PRIMAL_TOLERANCE = 1e-9
DUAL_TOLERANCE = 1e-9
MIP_TOLERANCE = 1e-9
SEEDS = (17, 29, 43, 71, 101)
PER_SEED_LIMIT_SECONDS = 360.0
TOTAL_SOLVER_LIMIT_SECONDS = 1800.0
PINNED_SCIPY = "1.17.1"
PINNED_HIGHS = "1.12.0"
PINNED_PLAN_SHA256 = "8a3b00d1b9ab073454598994e92cdca3564a34fd3c412414d2e85caa9221a6a6"
PINNED_SOURCE_MANIFEST_SHA256 = "50e0b813e0f2db7e42deb2b901417e130f5c0a935f7a469c4f8e37bb3718d343"


@dataclass(frozen=True)
class SparseMilp:
    matrix: sparse.csc_matrix
    row_lower: np.ndarray
    row_upper: np.ndarray
    objective: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    integrality: np.ndarray
    source_count: int
    binary_count: int
    binary_margins: np.ndarray
    binary_big_m: np.ndarray
    binary_layout_ids: tuple[str, ...]
    binary_pixel_indices: tuple[tuple[str, int], ...]
    nominal_pixel_indices: tuple[tuple[str, int], ...]
    nominal_original_row_count: int
    nominal_pruned_protected_count: int
    protected_count: int
    protected_pixel_indices: tuple[tuple[str, int], ...]
    protected_pruned_simplex_pixel_indices: tuple[tuple[str, int], ...]
    protected_pruned_nominal_pixel_indices: tuple[tuple[str, int], ...]
    protected_pruned_simplex: int
    protected_pruned_nominal: int
    nominal_row_count: int
    anchor_start: np.ndarray
    epsilon: float
    rho: float
    lp_margin: float
    nominal_floor: float


def hash_sparse_model(model: SparseMilp) -> str:
    """Hash canonical sparse coefficients, bounds, costs, and variable types."""
    digest = hashlib.sha256()
    for value in (model.matrix.shape, model.matrix.indptr, model.matrix.indices,
                  model.matrix.data, model.row_lower, model.row_upper,
                  model.objective, model.variable_lower, model.variable_upper,
                  model.integrality):
        if isinstance(value, tuple):
            digest.update(np.asarray(value, dtype=np.int64).tobytes())
        else:
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes())
    digest.update(repr(model.binary_pixel_indices).encode("utf-8"))
    digest.update(repr(model.nominal_pixel_indices).encode("utf-8"))
    digest.update(repr(model.protected_pixel_indices).encode("utf-8"))
    digest.update(repr(model.protected_pruned_simplex_pixel_indices).encode("utf-8"))
    digest.update(repr(model.protected_pruned_nominal_pixel_indices).encode("utf-8"))
    return digest.hexdigest()


def _as_layout_arrays(bases: Sequence[np.ndarray], targets: Sequence[np.ndarray]):
    if not bases or len(bases) != len(targets):
        raise ValueError("matching non-empty FIT basis and target lists are required")
    source_count = None
    out = []
    for index, (basis, target) in enumerate(zip(bases, targets)):
        b = np.asarray(basis, dtype=np.float64)
        t = np.asarray(target, dtype=bool)
        if b.ndim != 3 or t.ndim != 2 or b.shape[1:] != t.shape:
            raise ValueError("FIT basis/target shape mismatch at layout %d" % index)
        if source_count is None:
            source_count = b.shape[0]
        if b.shape[0] != source_count or not np.isfinite(b).all():
            raise ValueError("FIT bases must share a finite source dimension")
        if float(b.min(initial=0.0)) < 0.0:
            raise ValueError("FIT bases must be nonnegative")
        out.append((np.ascontiguousarray(b), np.ascontiguousarray(t)))
    return source_count, out


def critical_signed_margins(basis: np.ndarray, target: np.ndarray,
                            weights: np.ndarray) -> np.ndarray:
    """Float64 target-signed critical-corner margins in row-major pixel order."""
    b = np.asarray(basis, dtype=np.float64)
    t = np.asarray(target, dtype=bool)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if b.ndim != 3 or b.shape[0] != w.size or b.shape[1:] != t.shape:
        raise ValueError("basis, target, and source weights have incompatible shapes")
    aerial = np.einsum("n,nhw->hw", w, b, optimize=True)
    return np.where(t, LOW_DOSE * aerial - THRESHOLD,
                    THRESHOLD - HIGH_DOSE * aerial).reshape(-1)


def build_sparse_milp(
    bases: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    anchor: np.ndarray,
    polytope,
    *,
    lp_margin: float = LP_MARGIN,
    rho: float = RHO,
    epsilon: float = EPSILON,
    solver_tolerance: float = SOLVER_TOLERANCE,
    layout_ids: Sequence[str] | None = None,
) -> SparseMilp:
    """Build nominal-polytope + protected critical pixels + one binary/error.

    The passed polytope must be the existing constrained.build_polytope result
    for these FIT bases and targets. No rows are approximately quantized or
    deduplicated. Protected-row pruning uses only explicit lower-bound proofs.
    """
    n, layouts = _as_layout_arrays(bases, targets)
    if (not math.isfinite(lp_margin) or lp_margin <= 0
            or not math.isfinite(rho) or not 0 < rho <= 1
            or not math.isfinite(epsilon) or epsilon <= 0
            or not math.isfinite(solver_tolerance) or solver_tolerance < 0):
        raise ValueError("invalid fixed MILP margin parameters")
    w0 = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if w0.size != n or not np.isfinite(w0).all():
        raise ValueError("LP anchor has an invalid shape or non-finite weights")
    if not math.isclose(float(rho), RHO, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("registered rho is fixed at 0.01")
    if not math.isclose(float(epsilon), EPSILON, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("registered epsilon is fixed at 2e-6")
    if not math.isclose(float(lp_margin), LP_MARGIN, rel_tol=0.0, abs_tol=1e-18):
        raise ValueError("registered nominal LP margin changed")
    ids = tuple(layout_ids or ("fit_%d" % i for i in range(len(layouts))))
    if len(ids) != len(layouts) or len(set(ids)) != len(ids):
        raise ValueError("unique FIT layout IDs are required")

    q = float(rho * lp_margin)
    nominal_floor = q
    # Verify the complete nominal polytope's ordered rows, labels and limits
    # against the exact FIT bases before removing any redundant rows.
    expected_matrix = np.concatenate(
        [basis.reshape(n, -1).T for basis, _target in layouts], axis=0
    )
    expected_labels = np.concatenate([target.reshape(-1) for _basis, target in layouts])
    expected_sign = np.where(expected_labels, 1.0, -1.0)
    expected_aub = -expected_sign[:, None] * expected_matrix
    expected_bub = -expected_sign * THRESHOLD - q
    try:
        poly_matrix = np.asarray(polytope.matrix, dtype=np.float64)
        raw_labels = np.asarray(polytope.labels)
        poly_labels = raw_labels.astype(bool)
        poly_aub = np.asarray(polytope.aub, dtype=np.float64)
        poly_bub = np.asarray(polytope.bub, dtype=np.float64).reshape(-1)
        poly_floor = float(polytope.margin_floor)
        poly_aeq = np.asarray(polytope.aeq, dtype=np.float64)
        poly_beq = np.asarray(polytope.beq, dtype=np.float64).reshape(-1)
        poly_bounds = tuple(tuple(bound) for bound in polytope.bounds)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("nominal polytope is missing its full FIT matrix and labels") from exc
    if (not np.isin(raw_labels, (0, 1, False, True)).all()
            or poly_floor != q
            or not np.array_equal(poly_matrix, expected_matrix)
            or not np.array_equal(poly_labels, expected_labels)
            or not np.array_equal(poly_aub, expected_aub)
            or not np.array_equal(poly_bub, expected_bub)
            or not np.array_equal(poly_aeq, np.ones((1, n), dtype=np.float64))
            or not np.array_equal(poly_beq, np.ones(1, dtype=np.float64))
            or poly_bounds != tuple((0.0, None) for _ in range(n))):
        raise ValueError("nominal polytope matrix, labels, order, or RHS differ from FIT bases/targets")
    nominal_original_row_count = int(expected_matrix.shape[0])

    protected_blocks = []
    error_blocks = []
    binary_margins = []
    binary_big_m = []
    binary_layout_ids = []
    binary_pixel_indices = []
    protected_pixel_indices = []
    protected_pruned_simplex_pixel_indices = []
    protected_pruned_nominal_pixel_indices = []
    nominal_pixel_indices = []
    nominal_row_indices = []
    protected_rhs_blocks = []
    error_rhs_blocks = []
    protected_count = 0
    pruned_simplex = 0
    pruned_nominal = 0
    nominal_offset = 0

    # These are valid lower bounds from the registered nominal signed-margin
    # floor. They are independent of the pixel and source distribution.
    nominal_pos_bound = LOW_DOSE * q - (1.0 - LOW_DOSE) * THRESHOLD
    nominal_neg_bound = HIGH_DOSE * q - (HIGH_DOSE - 1.0) * THRESHOLD

    for layout_index, (basis, target) in enumerate(layouts):
        flat = basis.reshape(n, -1).T
        target_flat = target.reshape(-1)
        sign = np.where(target_flat, 1.0, -1.0)
        doses = np.where(target_flat, LOW_DOSE, HIGH_DOSE)
        margins = critical_signed_margins(basis, target, w0)
        wrong = margins < 0.0
        correct = ~wrong

        simplex_lb = np.where(
            target_flat,
            LOW_DOSE * np.min(flat, axis=1) - THRESHOLD,
            THRESHOLD - HIGH_DOSE * np.max(flat, axis=1),
        )
        nominal_lb = np.where(target_flat, nominal_pos_bound, nominal_neg_bound)
        certified_lb = np.maximum(simplex_lb, nominal_lb)

        wrong_indices = np.flatnonzero(wrong)
        nominal_row_indices.extend((nominal_offset + wrong_indices).tolist())
        nominal_pixel_indices.extend((ids[layout_index], int(pixel))
                                     for pixel in wrong_indices)
        nominal_offset += int(target_flat.size)
        for pixel in wrong_indices:
            lower_bound = float(certified_lb[pixel])
            big_m = max(0.0, float(epsilon) - lower_bound)
            binary_margins.append(float(margins[pixel]))
            binary_big_m.append(big_m)
            binary_layout_ids.append(ids[layout_index])
            binary_pixel_indices.append((ids[layout_index], int(pixel)))

        # Protected critical rows are pruned only by their simplex vertex
        # lower bound. The nominal floor remains available for tight big-M
        # values on anchor errors, but is not used to prove a protected row.
        simplex_safe = correct & (simplex_lb >= epsilon + solver_tolerance)
        nominal_safe = np.zeros_like(correct)
        keep = correct & ~simplex_safe
        keep_indices = np.flatnonzero(keep)
        pruned_simplex += int(simplex_safe.sum())
        pruned_nominal += int(nominal_safe.sum())
        protected_count += int(keep_indices.size)
        protected_pixel_indices.extend((ids[layout_index], int(pixel))
                                       for pixel in keep_indices)
        protected_pruned_simplex_pixel_indices.extend(
            (ids[layout_index], int(pixel)) for pixel in np.flatnonzero(simplex_safe)
        )
        protected_pruned_nominal_pixel_indices.extend(
            (ids[layout_index], int(pixel)) for pixel in np.flatnonzero(nominal_safe)
        )
        if keep_indices.size:
            coeff = (-sign[keep_indices] * doses[keep_indices])[:, None] * flat[keep_indices]
            protected_blocks.append(sparse.csr_matrix(coeff))
            protected_rhs_blocks.append(-sign[keep_indices] * THRESHOLD - epsilon)

        if wrong_indices.size:
            coeff = (-sign[wrong_indices] * doses[wrong_indices])[:, None] * flat[wrong_indices]
            error_blocks.append(sparse.csr_matrix(coeff))
            error_rhs_blocks.append(-sign[wrong_indices] * THRESHOLD - epsilon)

    errors = len(binary_margins)
    nominal_row_indices_array = np.asarray(nominal_row_indices, dtype=np.int64)
    if (nominal_row_indices_array.size != errors
            or tuple(nominal_pixel_indices) != tuple(binary_pixel_indices)
            or nominal_original_row_count - errors != len(protected_pixel_indices)
            + len(protected_pruned_simplex_pixel_indices)
            + len(protected_pruned_nominal_pixel_indices)):
        raise AssertionError("nominal row pruning does not match the complete FIT partition")
    # Every originally correct critical-corner pixel either keeps its stronger
    # critical row or is certified redundant by the simplex lower bound. Its
    # nominal row is therefore implied; retain only the anchor-error rows.
    nominal_matrix = sparse.csr_matrix(expected_aub[nominal_row_indices_array, :])
    nominal_matrix.eliminate_zeros()
    nominal_rhs = expected_bub[nominal_row_indices_array]
    # Weight rows occupy the first n columns. Binary slack coefficients are
    # appended by an offset diagonal block, never a dense identity matrix.
    zero_binary_block = lambda rows: sparse.csr_matrix((rows, errors), dtype=np.float64)
    base_rows = [sparse.hstack((nominal_matrix, zero_binary_block(nominal_matrix.shape[0])),
                               format="csr")]
    if protected_blocks:
        protected_matrix = sparse.vstack(protected_blocks, format="csr")
        base_rows.append(sparse.hstack((protected_matrix, zero_binary_block(protected_matrix.shape[0])),
                                       format="csr"))
    if error_blocks:
        weight_error_rows = sparse.vstack(error_blocks, format="csr")
        binary_diagonal = -sparse.diags(np.asarray(binary_big_m, dtype=np.float64), format="csr")
        base_rows.append(sparse.hstack((weight_error_rows, binary_diagonal), format="csr"))
    weight_only = sparse.vstack(base_rows, format="csr")
    if errors and not error_blocks:
        raise AssertionError("binary variables have no matching critical-error rows")
    protected_rhs = (np.concatenate(protected_rhs_blocks)
                     if protected_rhs_blocks else np.empty(0))
    error_rhs = np.concatenate(error_rhs_blocks) if error_rhs_blocks else np.empty(0)
    row_upper = np.concatenate((nominal_rhs, protected_rhs, error_rhs, np.array([1.0])))
    row_lower = np.full(row_upper.size, -np.inf, dtype=np.float64)
    row_lower[-1] = 1.0
    # Last row is the simplex equality. Add it to the matrix after all pixel rows.
    simplex = sparse.csr_matrix((np.ones(n), (np.zeros(n, dtype=np.int32), np.arange(n))),
                                shape=(1, n + errors))
    matrix = sparse.vstack((weight_only, simplex), format="csc")
    # The nominal polytope is all first, followed by protected rows then error rows.
    if row_upper.size != matrix.shape[0]:
        raise AssertionError("MILP row bounds do not match sparse matrix")

    objective = np.concatenate((np.zeros(n, dtype=np.float64), np.ones(errors, dtype=np.float64)))
    variable_lower = np.zeros(n + errors, dtype=np.float64)
    variable_upper = np.concatenate((np.ones(n, dtype=np.float64), np.ones(errors, dtype=np.float64)))
    integrality = np.concatenate((np.zeros(n, dtype=np.uint8), np.ones(errors, dtype=np.uint8)))
    anchor_start = np.concatenate((w0, np.ones(errors, dtype=np.float64)))
    model = SparseMilp(
        matrix=matrix,
        row_lower=row_lower,
        row_upper=row_upper,
        objective=objective,
        variable_lower=variable_lower,
        variable_upper=variable_upper,
        integrality=integrality,
        source_count=n,
        binary_count=errors,
        binary_margins=np.asarray(binary_margins, dtype=np.float64),
        binary_big_m=np.asarray(binary_big_m, dtype=np.float64),
        binary_layout_ids=tuple(binary_layout_ids),
        binary_pixel_indices=tuple(binary_pixel_indices),
        nominal_pixel_indices=tuple(nominal_pixel_indices),
        nominal_original_row_count=nominal_original_row_count,
        nominal_pruned_protected_count=nominal_original_row_count - int(nominal_matrix.shape[0]),
        protected_count=protected_count,
        protected_pixel_indices=tuple(protected_pixel_indices),
        protected_pruned_simplex_pixel_indices=tuple(protected_pruned_simplex_pixel_indices),
        protected_pruned_nominal_pixel_indices=tuple(protected_pruned_nominal_pixel_indices),
        protected_pruned_simplex=pruned_simplex,
        protected_pruned_nominal=pruned_nominal,
        nominal_row_count=int(nominal_matrix.shape[0]),
        anchor_start=anchor_start,
        epsilon=float(epsilon),
        rho=float(rho),
        lp_margin=float(lp_margin),
        nominal_floor=q,
    )
    start_check = check_model_residual(model, anchor_start, tolerance=solver_tolerance)
    if not start_check["passed"]:
        raise ValueError("LP anchor with error binaries set to one is not a feasible MILP warm start")
    return model


def check_model_residual(model: SparseMilp, values: np.ndarray,
                         tolerance: float = SOLVER_TOLERANCE) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size != model.objective.size or not np.isfinite(x).all():
        return {"passed": False, "reason": "invalid_solution_shape_or_values"}
    activity = np.asarray(model.matrix @ x, dtype=np.float64).reshape(-1)
    lower_v = np.maximum(model.row_lower - activity, 0.0)
    upper_v = np.maximum(activity - model.row_upper, 0.0)
    row_violation = np.maximum(lower_v, upper_v)
    bound_v = np.maximum(model.variable_lower - x, 0.0)
    bound_v = np.maximum(bound_v, np.maximum(x - model.variable_upper, 0.0))
    int_mask = model.integrality.astype(bool)
    integrality = float(np.max(np.abs(x[int_mask] - np.rint(x[int_mask])), initial=0.0))
    max_row = float(np.max(row_violation, initial=0.0))
    max_bound = float(np.max(bound_v, initial=0.0))
    return {
        "passed": bool(max(max_row, max_bound, integrality) <= tolerance),
        "maximum_row_violation": max_row,
        "maximum_variable_bound_violation": max_bound,
        "maximum_integrality_violation": integrality,
        "tolerance": float(tolerance),
    }


def validate_solver_record(record: dict, *, require_optimal: bool = True) -> dict:
    """Fail-closed eligibility check independent of the particular solver API."""
    if not isinstance(record, dict):
        return {"passed": False, "reason": "solver_record_not_object"}
    checks = {
        "model_optimal": record.get("model_status") == "kOptimal",
        "solver_run_ok": record.get("run_status") == "kOk",
        "model_accepted": record.get("pass_model_status") == "kOk",
        "warm_start_accepted": record.get("warm_start_status") == "kOk",
        "objective_finite": _finite_number(record.get("objective")),
        "dual_bound_finite": _finite_number(record.get("dual_bound")),
        "integer_objective": (
            _finite_number(record.get("objective"))
            and abs(float(record.get("objective")) - round(float(record.get("objective"))))
            <= MIP_TOLERANCE
        ),
        "zero_mip_gap": _finite_number(record.get("mip_gap")) and record.get("mip_gap") == 0.0,
        "zero_objective_gap": (
            _finite_number(record.get("objective"))
            and _finite_number(record.get("dual_bound"))
            and record.get("objective") == record.get("dual_bound")
        ),
        "options_passed": record.get("options_passed") is True,
        "residual_passed": record.get("residual", {}).get("passed") is True,
        "integrality_passed": record.get("residual", {}).get("maximum_integrality_violation", math.inf)
        <= MIP_TOLERANCE,
    }
    passed = all(checks.values()) if require_optimal else (
        checks["objective_finite"] and checks["residual_passed"]
    )
    return {"passed": bool(passed), "checks": checks}


def five_seed_eligibility(seed_rows: Sequence[dict]) -> dict:
    ordered = list(seed_rows)
    checks = {
        "five_registered_seeds": len(ordered) == len(SEEDS),
        "seed_order": [x.get("seed") for x in ordered] == list(SEEDS),
        "all_optimal_zero_gap": len(ordered) == len(SEEDS) and all(
            x.get("fit_qualified") is True
            and validate_solver_record(x.get("solver", {}))["passed"]
            for x in ordered
        ),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "calibration_status": "eligible only after all checks pass" if all(checks.values()) else "closed"}


def _finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def pinned_solver_backend(*, require_pinned: bool = True):
    """Return the SciPy-bundled private HiGHS API after hard pin checks."""
    import scipy
    try:
        from scipy.optimize import _highspy
        core = _highspy._core
    except Exception as exc:
        raise RuntimeError("pinned SciPy private HiGHS backend is unavailable") from exc
    highs = core._Highs()
    highs_version = str(highs.version())
    if require_pinned:
        if tuple(sys.version_info[:2]) != (3, 12):
            raise RuntimeError("production MILP requires Python 3.12")
        if scipy.__version__ != PINNED_SCIPY or highs_version != PINNED_HIGHS:
            raise RuntimeError(
                "production MILP requires scipy %s with bundled HiGHS %s; found scipy %s / HiGHS %s"
                % (PINNED_SCIPY, PINNED_HIGHS, scipy.__version__, highs_version)
            )
    required = ("HighsLp", "HighsVarType", "MatrixFormat", "kHighsInf")
    missing = [name for name in required if not hasattr(core, name)]
    for name in ("passModel", "setSolution", "run", "getSolution", "getInfo",
                 "getModelStatus", "getRunTime"):
        if not hasattr(highs, name):
            missing.append("_Highs." + name)
    info = highs.getInfo()
    for name in ("objective_function_value", "mip_dual_bound", "mip_gap", "mip_node_count"):
        if not hasattr(info, name):
            missing.append("HighsInfo." + name)
    if missing:
        raise RuntimeError("SciPy private HiGHS API is missing: " + ", ".join(missing))
    return core, highs_version


def solve_highs(model: SparseMilp, seed: int, time_limit: float, *,
                require_pinned: bool = True) -> dict:
    """Run exactly one HiGHS attempt with checked options and LP warm start."""
    if seed not in SEEDS:
        raise ValueError("solver seed is not registered")
    if not math.isfinite(float(time_limit)) or not 0 < time_limit <= PER_SEED_LIMIT_SECONDS:
        raise ValueError("per-seed time limit must be in (0, 360]")
    core, highs_version = pinned_solver_backend(require_pinned=require_pinned)
    highs = core._Highs()
    expected_options = {
        "output_flag": False,
        "primal_feasibility_tolerance": PRIMAL_TOLERANCE,
        "dual_feasibility_tolerance": DUAL_TOLERANCE,
        "mip_feasibility_tolerance": MIP_TOLERANCE,
        "mip_rel_gap": 0.0,
        "mip_abs_gap": 0.0,
        "threads": 4,
        "random_seed": int(seed),
        "time_limit": float(time_limit),
    }
    option_results = {}
    for name, value in expected_options.items():
        status = highs.setOptionValue(name, value)
        option_results[name] = getattr(status, "name", str(status))
        if option_results[name] != "kOk":
            raise RuntimeError("HiGHS refused pinned option %s: %s" % (name, option_results[name]))
    option_readback = {}
    for name, expected in expected_options.items():
        status, actual = highs.getOptionValue(name)
        status_name = getattr(status, "name", str(status))
        option_readback[name] = {"status": status_name, "value": actual}
        if status_name != "kOk" or actual != expected:
            raise RuntimeError("HiGHS option readback differs for %s" % name)

    lp = core.HighsLp()
    variable_count = model.objective.size
    row_count = model.row_upper.size
    matrix = model.matrix
    if matrix.shape != (row_count, variable_count):
        raise ValueError("sparse MILP dimensions are inconsistent")
    lp.num_col_ = int(variable_count)
    lp.num_row_ = int(row_count)
    lp.col_cost_ = np.asarray(model.objective, dtype=np.float64)
    lp.col_lower_ = np.asarray(model.variable_lower, dtype=np.float64)
    lp.col_upper_ = np.asarray(model.variable_upper, dtype=np.float64)
    lp.row_lower_ = np.where(np.isneginf(model.row_lower), -float(core.kHighsInf),
                             np.asarray(model.row_lower, dtype=np.float64))
    lp.row_upper_ = np.where(np.isposinf(model.row_upper), float(core.kHighsInf),
                             np.asarray(model.row_upper, dtype=np.float64))
    lp.integrality_ = [
        core.HighsVarType.kInteger if value else core.HighsVarType.kContinuous
        for value in model.integrality
    ]
    high_matrix = lp.a_matrix_
    high_matrix.format_ = core.MatrixFormat.kColwise
    high_matrix.start_ = matrix.indptr.astype(np.int32, copy=True)
    high_matrix.index_ = matrix.indices.astype(np.int32, copy=True)
    high_matrix.value_ = matrix.data.astype(np.float64, copy=True)
    pass_status = highs.passModel(lp)
    if getattr(pass_status, "name", str(pass_status)) != "kOk":
        raise RuntimeError("HiGHS rejected the sparse model: %s" % pass_status)
    start_indices = np.arange(variable_count, dtype=np.int32)
    warm_status = highs.setSolution(variable_count, start_indices,
                                    np.asarray(model.anchor_start, dtype=np.float64))
    if getattr(warm_status, "name", str(warm_status)) != "kOk":
        raise RuntimeError("HiGHS rejected the feasible LP-anchor warm start")

    started = time.monotonic()
    run_exception = None
    run_status = None
    try:
        run_status = highs.run()
    except BaseException as exc:
        run_exception = exc
    wall_seconds = time.monotonic() - started
    try:
        model_status = highs.getModelStatus()
        model_status_name = getattr(model_status, "name", str(model_status))
    except Exception as exc:
        model_status_name = "unavailable"
        if run_exception is None:
            run_exception = exc
    try:
        info = highs.getInfo()
        info_exception = None
    except Exception as exc:
        info = None
        info_exception = exc
    if info_exception is not None and run_exception is None:
        run_exception = info_exception
    try:
        solution = highs.getSolution()
    except Exception as exc:
        solution = None
        if run_exception is None:
            run_exception = exc
    values = np.asarray(getattr(solution, "col_value", ()), dtype=np.float64)
    has_values = values.size == variable_count and np.isfinite(values).all()
    residual = check_model_residual(model, values) if has_values else {
        "passed": False, "reason": "solver_returned_no_finite_incumbent"
    }
    objective_raw = getattr(info, "objective_function_value", None)
    dual_raw = getattr(info, "mip_dual_bound", None)
    gap_raw = getattr(info, "mip_gap", None)
    nodes_raw = getattr(info, "mip_node_count", None)
    objective = float(objective_raw) if _finite_number(objective_raw) else None
    dual_bound = float(dual_raw) if _finite_number(dual_raw) else None
    mip_gap = float(gap_raw) if _finite_number(gap_raw) else None
    try:
        solver_wall = float(highs.getRunTime())
    except Exception:
        solver_wall = float(wall_seconds)
    record = {
        "seed": int(seed),
        "model_status": model_status_name,
        "run_status": (getattr(run_status, "name", str(run_status))
                       if run_status is not None else "kError"),
        "objective": objective,
        "dual_bound": dual_bound,
        "mip_gap": mip_gap,
        "node_count": int(nodes_raw) if _finite_number(nodes_raw) else None,
        "solver_wall_seconds": solver_wall,
        "external_wall_seconds": float(wall_seconds),
        "time_limit_seconds": float(time_limit),
        "highs_version": highs_version,
        "scipy_version": __import__("scipy").__version__,
        "option_set_status": option_results,
        "option_readback": option_readback,
        "options_passed": True,
        "pass_model_status": getattr(pass_status, "name", str(pass_status)),
        "warm_start_status": getattr(warm_status, "name", str(warm_status)),
        "residual": residual,
        "incumbent_present": bool(has_values),
        "values": values.tolist() if has_values else None,
    }
    if run_exception is not None:
        record["post_run_error"] = {
            "type": type(run_exception).__name__, "message": str(run_exception),
            "interrupted": isinstance(run_exception, KeyboardInterrupt),
        }
        record["run_status"] = "kError"
    record["optimal_zero_gap"] = validate_solver_record(record)["passed"]
    return record
