"""Deterministic affine threshold-event proposals for source-only searches.

Event locations are computed in float64 from the affine weighted-basis model.
They are proposals only: candidate qualification and ranking must use the
canonical float32 simulator output and explicit hard-metric gates.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "source_only_affine_dose_threshold_events_v1"
SOURCE_COUNT = 49
DOSES = (0.98, 1.0, 1.02)
THRESHOLD = 0.225
STEEPNESS = 50.0
DEFAULT_ALPHA_TOLERANCE = 1e-12


def _strict_integer(value, *, name: str, minimum: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or int(value) < minimum):
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(value)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def array_sha256(values: np.ndarray, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    return sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def validate_weights(weights, *, source_count: int = SOURCE_COUNT,
                     simplex_tolerance: float = 1e-9) -> np.ndarray:
    """Return a validated float64 nonnegative unit-flux source vector."""
    source_count = _strict_integer(source_count, name="source_count", minimum=1)
    if not math.isfinite(float(simplex_tolerance)) or simplex_tolerance < 0:
        raise ValueError("simplex_tolerance must be finite and nonnegative")
    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    if vector.size != source_count:
        raise ValueError("source weights must have exactly %d entries" % source_count)
    if not np.isfinite(vector).all():
        raise ValueError("source weights must be finite")
    if np.any(vector < 0.0):
        raise ValueError("source weights must be nonnegative")
    if abs(float(vector.sum()) - 1.0) > simplex_tolerance:
        raise ValueError("source weights must have unit flux")
    return vector.copy()


def _validate_intensity_pair(start, end) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(start, dtype=np.float64)
    right = np.asarray(end, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("endpoint intensity arrays must have the same non-empty shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("endpoint intensity arrays must be finite")
    return left.reshape(-1), right.reshape(-1)


def threshold_event_alphas(start_intensity, end_intensity, *,
                           doses: Sequence[float] = DOSES,
                           threshold: float = THRESHOLD,
                           alpha_tolerance: float = DEFAULT_ALPHA_TOLERANCE) -> dict:
    """Return segment knots and interval midpoints at dose-threshold events.

    For every pixel and dose, the ideal weighted intensity is affine in alpha.
    The crossing solves ``dose * I(alpha) == threshold``. Constant pixels that
    lie exactly on the threshold add no isolated event; their hard state is
    constant over the whole segment and endpoints already represent it.
    Near-duplicate roots are merged in sorted order using the supplied
    tolerance. The returned float64 roots are never treated as hard decisions.
    """
    left, right = _validate_intensity_pair(start_intensity, end_intensity)
    try:
        dose_values = tuple(float(dose) for dose in doses)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("doses must be finite positive numbers") from exc
    if not dose_values or any(not math.isfinite(dose) or dose <= 0.0 for dose in dose_values):
        raise ValueError("doses must be finite positive numbers")
    threshold = float(threshold)
    alpha_tolerance = float(alpha_tolerance)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and nonnegative")
    if not math.isfinite(alpha_tolerance) or not 0.0 <= alpha_tolerance < 0.5:
        raise ValueError("alpha_tolerance must be finite and lie in [0, 0.5)")

    roots: list[np.ndarray] = []
    constant_on_threshold = 0
    delta = right - left
    changing = delta != 0.0
    for dose in dose_values:
        level = threshold / dose
        constant_on_threshold += int(np.count_nonzero((~changing) & (left == level)))
        if not np.any(changing):
            continue
        alpha = np.empty_like(delta)
        alpha[~changing] = np.nan
        alpha[changing] = (level - left[changing]) / delta[changing]
        valid = np.isfinite(alpha) & (alpha >= -alpha_tolerance) & (alpha <= 1.0 + alpha_tolerance)
        if np.any(valid):
            roots.append(np.clip(alpha[valid], 0.0, 1.0))

    ordered = np.sort(np.concatenate(([0.0, 1.0], *roots)).astype(np.float64, copy=False))
    knots: list[float] = []
    for value in ordered:
        value = float(value)
        if not knots or value - knots[-1] > alpha_tolerance:
            knots.append(value)
        elif value == 1.0:
            # Preserve the closed endpoint if a near-one root merged with it.
            knots[-1] = 1.0

    # Endpoints are guaranteed even when a root is within tolerance of one.
    knots[0] = 0.0
    knots[-1] = 1.0
    midpoints = [(left_alpha + right_alpha) / 2.0
                 for left_alpha, right_alpha in zip(knots, knots[1:])
                 if right_alpha > left_alpha]
    proposal_alphas = sorted(set(knots + midpoints))
    return {
        "knots": knots,
        "midpoints": midpoints,
        "proposal_alphas": proposal_alphas,
        "isolated_crossing_count": int(sum(root.size for root in roots)),
        "constant_pixels_on_threshold_count": constant_on_threshold,
        "dose_thresholds": [{"dose": dose, "aerial_intensity": threshold / dose}
                            for dose in dose_values],
        "interpretation": "float64 ideal-affine proposals; actual float32 sigmoid hard outputs decide qualification",
    }


def _merge_alpha_proposals(proposals: Sequence[float], tolerance: float) -> list[float]:
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or not 0.0 <= tolerance < 0.5:
        raise ValueError("alpha tolerance must be finite and lie in [0, 0.5)")
    values = np.asarray([0.0, 1.0, *proposals], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("alpha proposals must be finite")
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError("alpha proposals must lie on the closed unit segment")
    values = np.sort(np.clip(values, 0.0, 1.0))
    result: list[float] = []
    for value in values:
        value = float(value)
        if not result or value - result[-1] > tolerance:
            result.append(value)
        elif value == 1.0:
            result[-1] = 1.0
    result[0], result[-1] = 0.0, 1.0
    return result


def interpolate_weights(start, end, alpha: float) -> np.ndarray:
    """Interpolate source weights in float64 without projection or renormalizing."""
    left = validate_weights(start)
    right = validate_weights(end)
    if left.shape != right.shape:
        raise ValueError("source segment endpoints must have equal shape")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    if alpha == 0.0:
        return left
    if alpha == 1.0:
        return right
    return (1.0 - alpha) * left + alpha * right


def segment_event_candidates(reference_weights, endpoint_weights,
                             reference_intensities: Mapping[str, np.ndarray],
                             endpoint_intensities: Mapping[str, np.ndarray], *,
                             segment_id: str = "segment",
                             doses: Sequence[float] = DOSES,
                             threshold: float = THRESHOLD,
                             alpha_tolerance: float = DEFAULT_ALPHA_TOLERANCE) -> tuple[list[dict], dict]:
    """Build deterministically deduplicated candidates for one source segment."""
    reference = validate_weights(reference_weights)
    endpoint = validate_weights(endpoint_weights)
    if reference.shape != endpoint.shape:
        raise ValueError("segment endpoints must have equal source counts")
    if not isinstance(segment_id, str) or not segment_id:
        raise ValueError("segment_id must be a non-empty string")
    if set(reference_intensities) != set(endpoint_intensities) or not reference_intensities:
        raise ValueError("reference and endpoint intensities must cover identical layouts")

    events: list[float] = []
    layout_audits = []
    for layout_id in sorted(reference_intensities):
        summary = threshold_event_alphas(
            reference_intensities[layout_id], endpoint_intensities[layout_id],
            doses=doses, threshold=threshold, alpha_tolerance=alpha_tolerance,
        )
        events.extend(summary["knots"])
        layout_audits.append({"layout_id": layout_id, **summary})
    knots = _merge_alpha_proposals(events, alpha_tolerance)
    alphas = sorted(set(knots + [0.5 * (a + b) for a, b in zip(knots, knots[1:]) if b > a]))

    proposals = []
    for alpha in alphas:
        role = "reference_endpoint" if alpha == 0.0 else "segment_endpoint" if alpha == 1.0 else (
            "threshold_event" if alpha in knots else "interval_midpoint")
        weights = interpolate_weights(reference, endpoint, alpha)
        proposals.append({"segment_id": segment_id, "alpha": alpha, "proposal_role": role,
                          "weights": weights, "weights_f64_sha256": array_sha256(weights, "<f8"),
                          "weights_f32_sha256": array_sha256(weights, "<f4")})

    unique: list[dict] = []
    by_hash: dict[str, dict] = {}
    duplicate_count = 0
    for row in proposals:
        prior = by_hash.get(row["weights_f64_sha256"])
        if prior is None:
            row["proposal_roles"] = [row.pop("proposal_role")]
            row["alphas"] = [row["alpha"]]
            by_hash[row["weights_f64_sha256"]] = row
            unique.append(row)
        else:
            duplicate_count += 1
            prior["proposal_roles"].append(row["proposal_role"])
            prior["alphas"].append(row["alpha"])
    for index, row in enumerate(unique):
        row["candidate_order"] = index
        row["candidate_id"] = "%s:%06d" % (segment_id, index)
        row["roles"] = list(dict.fromkeys(row.get("proposal_roles", ("event",))))
    audit = {"segment_id": segment_id, "layout_events": layout_audits,
             "unique_knot_count": len(knots), "proposal_count_before_weight_dedup": len(proposals),
             "candidate_count": len(unique), "deduplicated_proposal_count": duplicate_count,
             "candidate_generation": "sorted layout IDs, sorted threshold knots, then interval midpoints"}
    return unique, audit


def incumbent_candidates(*, reference_weights, initial_incumbent_weights,
                         best_known_incumbent_weights, segment_candidates: Sequence[dict]) -> list[dict]:
    """Put reference and both frozen incumbents ahead of event proposals.

    Identical vectors are represented once but retain every role in
    ``roles``. This guarantees a reference/start/best-known incumbent cannot
    disappear through event deduplication.
    """
    rows = []
    for role, vector in (("reference", reference_weights),
                         ("initial_incumbent", initial_incumbent_weights),
                         ("best_known_incumbent", best_known_incumbent_weights)):
        weights = validate_weights(vector)
        rows.append({"candidate_id": role, "candidate_order": len(rows), "roles": [role],
                     "proposal_role": role, "weights": weights,
                     "weights_f64_sha256": array_sha256(weights, "<f8"),
                     "weights_f32_sha256": array_sha256(weights, "<f4"), "alphas": []})
    rows.extend(dict(row) for row in segment_candidates)

    unique: list[dict] = []
    positions: dict[str, dict] = {}
    for row in rows:
        weights = validate_weights(row["weights"])
        digest = array_sha256(weights, "<f8")
        if digest in positions:
            existing = positions[digest]
            role_values = row.get("roles", [row.get("proposal_role", "event")])
            existing["roles"].extend(role for role in role_values if role not in existing["roles"])
            if row.get("alphas"):
                existing.setdefault("alphas", []).extend(row["alphas"])
            existing.setdefault("aliases", []).append(row.get("candidate_id"))
            continue
        normalized = dict(row)
        normalized["weights"] = weights
        normalized["weights_f64_sha256"] = digest
        normalized["weights_f32_sha256"] = array_sha256(weights, "<f4")
        normalized.setdefault("roles", [normalized.get("proposal_role", "event")])
        normalized["candidate_order"] = len(unique)
        unique.append(normalized)
        positions[digest] = normalized
    for index, row in enumerate(unique):
        row["candidate_order"] = index
    return unique


def create_event_attempt_marker(manifest_path: str | Path, plan_sha256: str) -> Path:
    """Consume a prospective event-search plan with a distinct marker name."""
    digest = str(plan_sha256).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("plan_sha256 must be a lowercase 64-character SHA256")
    parent = Path(manifest_path).resolve().parent
    marker = parent / ("source_event_attempt_" + digest + ".consumed")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(marker, flags, 0o600)
    try:
        payload = ("{\"plan_sha256\":\"%s\",\"status\":\"consumed\",\"objective\":\"%s\"}\n"
                   % (digest, PROTOCOL_ID)).encode("utf-8")
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return marker


def canonical_identity(*, code_paths: Sequence[str | Path], input_paths: Sequence[str | Path],
                       named_vectors: Mapping[str, Sequence[float]]) -> dict:
    """Build stable code/input/source-vector pins for a prospective plan."""
    def path_key(path):
        return Path(path).as_posix()

    normalized_vectors = {str(name): validate_weights(weights)
                          for name, weights in named_vectors.items()}
    return {
        "code_sha256": {path_key(path): sha256_file(path)
                        for path in sorted((Path(p) for p in code_paths), key=path_key)},
        "input_sha256": {path_key(path): sha256_file(path)
                         for path in sorted((Path(p) for p in input_paths), key=path_key)},
        "source_vectors": {
            name: {"source_count": int(weights.size),
                   "weights_f64_sha256": array_sha256(weights, "<f8"),
                   "weights_f32_sha256": array_sha256(weights, "<f4")}
            for name, weights in sorted(normalized_vectors.items())
        },
        "identity_rule": "SHA256 code/input bytes and exact little-endian float64/float32 source vectors",
    }


def validate_identity(identity: dict, *, base_dir: str | Path = ".",
                      named_vectors: Mapping[str, Sequence[float]] | None = None) -> None:
    """Raise if any pinned code or input file no longer has its frozen hash."""
    if not isinstance(identity, dict) or not isinstance(identity.get("code_sha256"), dict) \
            or not isinstance(identity.get("input_sha256"), dict):
        raise ValueError("identity must contain code_sha256 and input_sha256 maps")
    root = Path(base_dir)
    for category in ("code_sha256", "input_sha256"):
        for relative, expected in identity[category].items():
            actual = sha256_file(root / relative)
            if actual != expected:
                raise ValueError("pinned %s hash mismatch: %s" % (category, relative))
    vector_pins = identity.get("source_vectors", {})
    if not isinstance(vector_pins, dict):
        raise ValueError("identity source_vectors must be a map")
    if named_vectors is not None:
        if set(named_vectors) != set(vector_pins):
            raise ValueError("named source vectors do not match the identity pin names")
        for name, weights in named_vectors.items():
            vector = validate_weights(weights)
            pin = vector_pins[name]
            if (not isinstance(pin, dict) or pin.get("source_count") != int(vector.size)
                    or pin.get("weights_f64_sha256") != array_sha256(vector, "<f8")
                    or pin.get("weights_f32_sha256") != array_sha256(vector, "<f4")):
                raise ValueError("pinned source vector mismatch: %s" % name)


def rank_qualified_candidates(records: Sequence[dict]) -> list[dict]:
    """Return only actually attempted, hard-qualified records in stable order."""
    qualified = []
    for row in records:
        if (row.get("status") != "attempted" or row.get("hard_qualified") is not True
                or not isinstance(row.get("actual_hard_metrics"), dict)):
            continue
        metrics = row["actual_hard_metrics"]
        required = ("new_fit_mean", "original_fit_mean")
        if any(not isinstance(metrics.get(key), dict) for key in required):
            raise ValueError("qualified candidate is missing aggregate actual hard metrics")
        new, original = metrics["new_fit_mean"], metrics["original_fit_mean"]
        rank = (float(new["band_pixels"]), float(new["L2_worst_dose_pixels"]),
                float(new["L2_pixels"]), float(original["band_pixels"]),
                float(original["L2_worst_dose_pixels"]), float(original["L2_pixels"]),
                int(row.get("candidate_order", 0)))
        qualified.append((rank, row))
    return [row for _rank, row in sorted(qualified, key=lambda pair: pair[0])]


def fair_quality_runtime_protocol(*, workload_sha256: str,
                                  candidate_evaluation_budget: int,
                                  wall_clock_budget_seconds: float,
                                  torch_threads: int = 1) -> dict:
    """Describe a paired comparison that separates quality from runtime.

    The caller must pin the exact FIT workload (masks, targets and basis),
    thread count, device and incumbent vectors in the prospective plan. This
    function supplies the shared comparison rules rather than claiming any
    measured speedup.
    """
    if len(str(workload_sha256)) != 64 or any(
            char not in "0123456789abcdef" for char in str(workload_sha256).lower()):
        raise ValueError("workload_sha256 must be a 64-character hexadecimal digest")
    candidate_evaluation_budget = _strict_integer(
        candidate_evaluation_budget, name="candidate_evaluation_budget", minimum=3,
    )
    if not math.isfinite(float(wall_clock_budget_seconds)) or wall_clock_budget_seconds <= 0.0:
        raise ValueError("wall_clock_budget_seconds must be finite and positive")
    torch_threads = _strict_integer(torch_threads, name="torch_threads", minimum=1)
    return {
        "workload_sha256": str(workload_sha256).lower(),
        "candidate_evaluation_budget_per_method": candidate_evaluation_budget,
        "wall_clock_budget_seconds_per_method": float(wall_clock_budget_seconds),
        "torch_threads": torch_threads,
        "quality_comparison": {
            "same_frozen_fit_masks_targets_basis_and_source_grid": True,
            "same_reference_initial_incumbent_and_best_known_incumbent": True,
            "same_float32_hard_metric_evaluator_and_qualification_gates": True,
            "same_candidate_evaluation_count_limit": candidate_evaluation_budget,
            "report_best_qualified_metrics_and_full_per_layout_counts": True,
            "unattempted_candidates": "reported as not attempted; never evidence of no qualifying point",
            "selection_inputs": "FIT hard metrics only; calibration closed; final3 never indexed or evaluated",
        },
        "runtime_comparison": {
            "same_wall_clock_budget_per_method": float(wall_clock_budget_seconds),
            "same_device_process_thread_count_and_memory_limits": True,
            "cold_setup_and_basis_preparation": "measured separately and included in total wall time",
            "candidate_generation_time": "included in total wall time and separately reported",
            "cross_method_score_cache": "disabled",
            "within_method_duplicate_policy": "deduplicate exact float64 source vectors before scoring; report float32 aliases",
            "run_order": "paired repetitions in both event-grid and grid-event order; report each and the median",
            "speedup_claim": "only from complete paired timing records with identical workload SHA256",
        },
        "negative_result_scope": "fixed candidate proposal set and explicit budget only; not global or exhaustive float32 infeasibility",
    }


def score_candidate_set(candidates: Sequence[dict], basis32_by_layout: Mapping[str, np.ndarray],
                        targets_by_layout: Mapping[str, np.ndarray], *,
                        original_layout_ids: Sequence[str], new_layout_ids: Sequence[str],
                        max_candidates: int | None = None,
                        wall_clock_budget_seconds: float | None = None,
                        additional_source_auditor: Callable[[np.ndarray], dict] | None = None) -> dict:
    """Audit each attempted candidate with canonical float32 hard metrics.

    The first distinct reference/start/best-known rows are protected from a
    candidate-count limit. Candidates beyond the explicit count or time
    budget remain in the report as ``not_attempted``. No float64 event estimate
    participates in hard qualification or rank.
    """
    import time

    import source_coverage as coverage

    originals = tuple(str(value) for value in original_layout_ids)
    new = tuple(str(value) for value in new_layout_ids)
    all_ids = originals + new
    if not originals or not new or len(set(all_ids)) != len(all_ids):
        raise ValueError("original and new FIT layout IDs must be non-empty and disjoint")
    if set(all_ids) != set(basis32_by_layout) or set(all_ids) != set(targets_by_layout):
        raise ValueError("basis and target maps must exactly cover the pinned FIT layout IDs")
    if max_candidates is not None:
        max_candidates = _strict_integer(max_candidates, name="max_candidates", minimum=1)
    if wall_clock_budget_seconds is not None and (
            not math.isfinite(float(wall_clock_budget_seconds)) or wall_clock_budget_seconds <= 0.0):
        raise ValueError("wall_clock_budget_seconds must be finite and positive")

    anchor_hashes = set()
    for row in candidates:
        roles = set(row.get("roles", ()))
        if roles.intersection({"reference", "initial_incumbent", "best_known_incumbent"}):
            anchor_hashes.add(row.get("weights_f64_sha256") or array_sha256(row["weights"], "<f8"))
    if not anchor_hashes:
        raise ValueError("candidate set must preserve reference and incumbent source vectors")
    if max_candidates is not None and max_candidates < len(anchor_hashes):
        raise ValueError("max_candidates is too small to audit all distinct protected incumbents")

    protected_roles = {"reference", "initial_incumbent", "best_known_incumbent"}
    ordered = sorted(candidates, key=lambda row: (
        0 if set(row.get("roles", ())).intersection(protected_roles) else 1,
        int(row.get("candidate_order", 0)),
    ))
    started = time.monotonic()
    results = []
    reference_metrics = None
    attempted = 0
    protected_elapsed = 0.0
    failure = None
    for row_index, row in enumerate(ordered):
        reason = None
        is_protected = bool(set(row.get("roles", ())).intersection(protected_roles))
        if max_candidates is not None and attempted >= max_candidates:
            reason = "candidate_evaluation_budget"
        elif (wall_clock_budget_seconds is not None
              and time.monotonic() - started >= float(wall_clock_budget_seconds)
              and not is_protected):
            reason = "wall_clock_budget"
        result = {
            "candidate_id": row.get("candidate_id", "candidate-%06d" % int(row.get("candidate_order", 0))),
            "candidate_order": int(row.get("candidate_order", len(results))),
            "roles": list(row.get("roles", ())),
            "weights": np.asarray(row["weights"], dtype=np.float64).tolist(),
            "weights_f64_sha256": array_sha256(row["weights"], "<f8"),
            "weights_f32_sha256": array_sha256(row["weights"], "<f4"),
        }
        if reason:
            result.update(status="not_attempted", reason=reason, actual_hard_metrics=None,
                          hard_qualified=False)
            results.append(result)
            continue

        attempted += 1
        candidate_started = time.monotonic()
        try:
            weights = validate_weights(row["weights"])
            per_layout = []
            for layout_id in all_ids:
                metric = coverage.hard_metrics(
                    np.asarray(basis32_by_layout[layout_id], dtype=np.float32),
                    np.asarray(targets_by_layout[layout_id], dtype=bool), weights,
                    threshold=THRESHOLD, steepness=STEEPNESS,
                )
                per_layout.append({"layout_id": layout_id, **metric})
            original_rows = per_layout[:len(originals)]
            new_rows = per_layout[len(originals):]
            actual = {
                "per_layout": per_layout,
                "original_fit_mean": coverage.aggregate_metrics(original_rows),
                "new_fit_mean": coverage.aggregate_metrics(new_rows),
                "no_blank_positive_target_any_dose": all(
                    item["no_blank_positive_target_any_dose"] for item in per_layout
                ),
            }
            source_audit = (additional_source_auditor(weights.copy())
                            if additional_source_auditor is not None
                            else {"passed": True, "scope": "nonnegative unit-flux simplex only"})
            if not isinstance(source_audit, dict) or not isinstance(source_audit.get("passed"), bool):
                raise ValueError("additional_source_auditor must return a dict with boolean `passed`")
            if "reference" in set(row.get("roles", ())):
                reference_metrics = actual
            result.update(status="attempted", actual_hard_metrics=actual,
                          additional_source_audit=source_audit,
                          float64_event_data_used_for_qualification=False)
            results.append(result)
        except Exception as exc:
            failure = {"candidate_id": result["candidate_id"],
                       "error_type": type(exc).__name__, "error": str(exc)}
            result.update(status="error", reason="candidate_audit_failed",
                          actual_hard_metrics=None, hard_qualified=False,
                          error_type=type(exc).__name__, error=str(exc))
            results.append(result)
            for rest in ordered[row_index + 1:]:
                results.append({
                    "candidate_id": rest.get("candidate_id", "candidate-%06d" % int(rest.get("candidate_order", 0))),
                    "candidate_order": int(rest.get("candidate_order", len(results))),
                    "roles": list(rest.get("roles", ())),
                    "weights": np.asarray(rest["weights"], dtype=np.float64).tolist(),
                    "weights_f64_sha256": array_sha256(rest["weights"], "<f8"),
                    "weights_f32_sha256": array_sha256(rest["weights"], "<f4"),
                    "status": "not_attempted", "reason": "prior_candidate_audit_error",
                    "actual_hard_metrics": None, "hard_qualified": False,
                })
            break
        finally:
            if is_protected:
                protected_elapsed += time.monotonic() - candidate_started

    if reference_metrics is None:
        for row in results:
            row["hard_qualified"] = False
            if row["status"] == "attempted":
                row["hard_qualification_gates"] = {"reference_hard_metrics_available": False}
    else:
        ref_original = reference_metrics["original_fit_mean"]
        ref_new = reference_metrics["new_fit_mean"]
        for row in results:
            actual = row.get("actual_hard_metrics")
            if row["status"] != "attempted" or actual is None:
                continue
            old, new_metrics = actual["original_fit_mean"], actual["new_fit_mean"]
            gates = {
                "all_fit_layouts_no_blank_positive_target": actual["no_blank_positive_target_any_dose"],
                "original_nominal_l2_nonincreasing": old["L2_pixels"] <= ref_original["L2_pixels"],
                "original_worst_dose_l2_nonincreasing": old["L2_worst_dose_pixels"] <= ref_original["L2_worst_dose_pixels"],
                "original_band_nonincreasing": old["band_pixels"] <= ref_original["band_pixels"],
                "new_nominal_l2_nonincreasing": new_metrics["L2_pixels"] <= ref_new["L2_pixels"],
                "new_worst_dose_l2_nonincreasing": new_metrics["L2_worst_dose_pixels"] <= ref_new["L2_worst_dose_pixels"],
                "new_band_strictly_lower": new_metrics["band_pixels"] < ref_new["band_pixels"],
                "additional_source_audit_passed": row.get("additional_source_audit", {}).get("passed") is True,
            }
            row["hard_qualification_gates"] = gates
            row["hard_qualified"] = bool(all(gates.values()))
    ranked = rank_qualified_candidates(results)
    elapsed = time.monotonic() - started
    complete = all(row["status"] == "attempted" for row in results)
    return {
        "status": "error" if failure else "complete" if complete else "incomplete_budget",
        "error": failure,
        "candidate_count": len(ordered), "attempted_candidates": attempted,
        "qualified_candidates": sum(row.get("hard_qualified") is True for row in results),
        "elapsed_seconds": elapsed,
        "protected_incumbent_elapsed_seconds": protected_elapsed,
        "budgets": {"max_candidates": max_candidates,
                    "wall_clock_budget_seconds": wall_clock_budget_seconds,
                    "wall_clock_includes_protected_incumbent_scoring": True,
                    "protected_incumbents_are_scored_even_if_they_exceed_wall_budget": True,
                    "incomplete_interpretation": "not evidence of no qualifying point"},
        "qualification_policy": "actual float32 hard counts: no blank; original means nonincreasing; new nominal/worst nonincreasing and new band strictly lower than reference",
        "selected_qualified_candidate_id": ranked[0]["candidate_id"] if ranked else None,
        "ranked_qualified_candidate_ids": [row["candidate_id"] for row in ranked],
        "candidate_records": results,
    }
