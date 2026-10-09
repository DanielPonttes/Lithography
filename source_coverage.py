"""Prospective source-only FIT coverage helpers.

This module defines only three new FIT masks and the small, source-weight
feasible guard set used by the coverage experiment. It never loads a dataset,
prepares an optical basis, or creates development/held-out layouts at import.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Sequence

import numpy as np
from scipy import sparse

OBJECTIVE_ID = "target_aware_fit_coverage_softcount_v1"
PINNED_PLAN_SHA256 = "7b4cbc414bc2b2a6ee72fac3a633e27adb2bcb2b299b1adb79dbfa9cbc3fc6d2"
PINNED_PLAN_SHA256_V6 = "c6dffeca8919dc8b0299ca6d35b05131b7bc67b8b61d45aba86377c7199915bc"
PINNED_SUPERSEDED_COVERAGE_REPORT_SHA256 = "263c1bb7e0de8bfcfff133e58a0b1c417f78cb565b9b1c9d4c3fa995cb766320"
PINNED_SUPERSEDED_COVERAGE_PLAN_SHA256 = PINNED_PLAN_SHA256
PINNED_FIXED_FIT_LAYOUT_HASHES = [
    {
        "layout_id": "fit_coverage_finite_ribbons_v1",
        "mask_sha256": "359df6231a4f6c430944e5c3e246eeb83b0d738ed57056ee2cf2eb01ae445781",
        "target_sha256": "8240edd4f8946fb6415c956b2c738457f8d8a42ac3b85eef816b1827839f42bb",
    },
    {
        "layout_id": "fit_coverage_asymmetric_line_ends_v1",
        "mask_sha256": "d730598006c5dd397b51491244188d68a93d3ed906463bb30294d68ae830648a",
        "target_sha256": "f3fc4cb7457ce531282a90c9c3156d6df9436ecf1910d37acdccbce13bffa83c",
    },
    {
        "layout_id": "fit_coverage_irregular_contacts_v1",
        "mask_sha256": "593d63a7aaad61597634dba821cf52fd70316b5faeb48bcc762a8f9cba5357f8",
        "target_sha256": "ab63be0ee7a87e91319d15cbd2306c3acc04813b718e755530ebbe4e4e2161f5",
    },
]
SCHEMA6_BASE_COMMIT = "a32222f24f6514d723236927971256ed6c3e1d98"
SCHEMA6_INITIALIZATION_PROTOCOL = {
    "algorithm": "seeded random linear minimization vertices mixed with the seed17 reference",
    "random_generator": "numpy.default_rng(seed); standard_normal(49); normalize each direction by its L2 norm",
    "lmo_direction_objective": "minimize the normalized direction over the guarded source domain",
    "maximum_lmo_directions_per_seed": 8,
    "mixing_alphas": [0.5, 0.25, 0.125],
    "candidate_order": "direction order, then alpha order; first candidate passing every audit and distinctness check",
    "minimum_l1_distance": 1e-4,
    "distinctness": "candidate hash must be new and L1 distance must exceed the minimum from the seed17 reference and every previously accepted start",
    "candidate_audits": [
        "full augmented guarded source domain",
        "full original nominal q polytope",
        "float32 critical guard audit",
    ],
    "start_selection": "feasibility and registered distinctness only; no image metric or loss ranking",
    "failure_policy": "no reference fallback; fail closed when the bounded search finds no distinct audited start",
    "maximum_initialization_seconds_per_seed": 60.0,
    "lmo_time_limit": "remaining initialization deadline, capped at 60 seconds per LMO",
    "budget_accounting": "initialization and diagnostics count inside the existing 360-second seed and 1800-second total budgets",
}
SCHEMA6_DIAGNOSTICS_PROTOCOL = {
    "snapshots": [
        "initial_beta_200",
        "transition_beta_400",
        "transition_beta_800",
        "final_beta_800",
    ],
    "per_layout_fields": ["softcount_loss", "gradient", "gradient_l2_norm"],
    "mean_parity_absolute_tolerance": 1e-10,
    "parity_check": "arithmetic mean of per-layout losses and gradients equals the existing seven-layout aggregate",
    "budget_accounting": "all aggregate and per-layout evaluations count inside the seed and total solver deadlines",
    "optimizer_or_ranking_influence": False,
}
PINNED_PREVIOUS_REPORT_SHA256 = "984dd65885ffc66b4eff3a70b4b8a81707841a777e8dd33ede9fe356daa5eb44"
PINNED_PREVIOUS_PLAN_SHA256 = "8a3b00d1b9ab073454598994e92cdca3564a34fd3c412414d2e85caa9221a6a6"
PINNED_SOURCE_MANIFEST_SHA256 = "50e0b813e0f2db7e42deb2b901417e130f5c0a935f7a469c4f8e37bb3718d343"
PINNED_BASIS_PARITY = [{"layout_id": "train_vls_pitch24_width9", "max_abs_error": 2.9802322387695312e-08}]
PINNED_BASIS_PARITY_SHA256 = "2c8b81b8eb9edca0037a159d69697326d6247492e06838963824a11e316df45c"

LP_MARGIN = 0.0002215098124816199
RHO = 0.01
NOMINAL_FLOOR = LP_MARGIN * RHO
EPSILON = 2e-6
SOLVER_TOLERANCE = 1e-9
THRESHOLD = 0.225
LOW_DOSE = 0.98
HIGH_DOSE = 1.02
SEEDS = (17, 29, 43, 71, 101)
BETAS = (200.0, 400.0, 800.0)
STEPS_PER_BETA = 68
STEPS_PER_SEED = len(BETAS) * STEPS_PER_BETA
CHECKPOINT_INTERVAL = 25
PER_SEED_LIMIT_SECONDS = 360.0
TOTAL_LIMIT_SECONDS = 1800.0
NEW_GUARDS_PER_CLASS = 128

LAYOUT_IDS = (
    "fit_coverage_finite_ribbons_v1",
    "fit_coverage_asymmetric_line_ends_v1",
    "fit_coverage_irregular_contacts_v1",
)
CALIBRATION_LAYOUT_IDS = (
    "cal_lines_phase_shifted_ribbons", "cal_contacts_hexagonal_array",
    "cal_junctions_asymmetric_y_tree", "cal_junctions_offset_double_t",
)
ORIGINAL_FIT_LAYOUT_IDS = (
    "train_vls_pitch24_width9", "train_hls_pitch28_width10",
    "train_l_contours", "train_t_junctions",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hashed_bytes(path: str | Path) -> tuple[bytes, str]:
    raw = Path(path).read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _validate_schema5_plan_payload(plan: dict) -> None:
    """Fail closed on any change to the externally frozen schema-5 protocol."""
    if not isinstance(plan, dict):
        raise ValueError("coverage plan must be an object")
    expected_top_level = {
        "schema_version", "status", "created_utc", "objective_id", "base_commit",
        "dataset_file", "diagnostic_file", "prerequisite_manifest", "previous_report",
        "previous_plan", "input_hashes", "scope", "physical", "protocol",
        "new_fit_generation", "completion",
    }
    if set(plan) != expected_top_level:
        raise ValueError("coverage plan fields differ from the frozen schema-5 contract")
    if (plan.get("schema_version") != 5
            or plan.get("status") != "prospective_candidate_plan"
            or plan.get("objective_id") != OBJECTIVE_ID):
        raise ValueError("coverage plan schema/status/objective differs from the frozen protocol")
    if plan.get("base_commit") != "7483c1e7c8003f3e2f61caa6324a30d0662a8121":
        raise ValueError("coverage plan base commit differs from the registered predecessor")
    previous = plan.get("previous_report")
    if (not isinstance(previous, dict) or set(previous) != {"path", "sha256", "selected_seed", "selection_role"}
            or previous.get("sha256") != PINNED_PREVIOUS_REPORT_SHA256
            or previous.get("selected_seed") != 17
            or previous.get("selection_role") != "previously frozen FIT-only ranking; no calibration selection"):
        raise ValueError("coverage plan previous report lineage differs from the frozen FIT-only parent")
    old_plan = plan.get("previous_plan")
    if (not isinstance(old_plan, dict) or set(old_plan) != {"path", "sha256"}
            or old_plan.get("sha256") != PINNED_PREVIOUS_PLAN_SHA256):
        raise ValueError("coverage plan previous-plan lineage differs from the registered plan")
    input_hashes = plan.get("input_hashes")
    expected_input_hashes = {
        "dataset_sha256": "1fb6555fbf1dc4b4748f05f37d557977df5bbfd3b5b8abf5853c68a04716d1d0",
        "diagnostic_sha256": "cd236f09d6c0ad32398d3ec638f1d2433a2a8cc6b5060016c181c15358d8901b",
        "source_manifest_sha256": PINNED_SOURCE_MANIFEST_SHA256,
    }
    if input_hashes != expected_input_hashes:
        raise ValueError("coverage plan input hashes differ from the frozen lineage")
    scope = plan.get("scope")
    if (not isinstance(scope, dict)
            or set(scope) != {"source_only", "fixed_masks", "number_of_source_weights", "calibration_role", "final3_access"}
            or scope.get("source_only") is not True
            or scope.get("fixed_masks") is not True
            or scope.get("number_of_source_weights") != 49
            or scope.get("final3_access") != "never indexed or evaluated"
            or scope.get("calibration_role") != "reused development data, never optimizer or tie-break input"):
        raise ValueError("coverage plan scope differs from source-only closed-final3 protocol")
    physical = plan.get("physical")
    if not isinstance(physical, dict):
        raise ValueError("coverage plan physical configuration is required")
    expected_physical = {
        "source_grid": 9, "sigma_inner": 0.3, "sigma_outer": 0.9,
        "NA": 1.35, "wavelength_nm": 193, "pixel_nm": 4, "raster": 128,
        "doses": [0.98, 1, 1.02], "focus": 0, "threshold": THRESHOLD,
        "steepness": 50,
    }
    if physical != expected_physical:
        raise ValueError("coverage plan optical/resist configuration differs from the registered values")
    protocol = plan.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("coverage plan protocol is required")
    exact_protocol = {
        "rho": RHO, "lp_margin": LP_MARGIN, "epsilon": EPSILON,
        "solver_tolerance": SOLVER_TOLERANCE,
        "seeds": list(SEEDS), "betas": list(BETAS),
        "steps_per_beta": STEPS_PER_BETA, "total_steps_per_seed": STEPS_PER_SEED,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "checkpoint_at_beta_transition_and_final": True,
        "initial_simplex_jitter": 1e-4,
        "fallback_if_guard_infeasible": "exact previous FIT-only seed17 reference; record fallback",
        "per_seed_time_limit_seconds": PER_SEED_LIMIT_SECONDS,
        "total_time_limit_seconds": TOTAL_LIMIT_SECONDS,
        "original_guards": "full original nominal q plus original LP-anchor critical protections epsilon; preserve original float32 hard qualification",
        "reference_guards": "all originally FIT-only seed17 reference-correct critical pixels, floor min(epsilon,0.5*positive float64 reference margin); original protections unchanged",
        "new_guards": "deterministic row-major sampling, at most128 target-positive and128 target-negative indices per new layout; include only positive reference critical margin; floor min(epsilon,0.5*margin)",
        "objective": "critical softcount mean with static equal weight per each of7FITlayouts; no dynamic/hotspot weights; existing target-aware critical dose softcount",
        "fit_selection": "qualified checkpoints only; new3 mean hard PV, new3 mean worst-dose L2, new3 mean nominal L2, soft objective, L1 to original LP anchor, registered checkpoint order; never calibration",
        "new_fit_gate": "strict reduction of new3 mean hard PV versus fixed seed17 reference, mean nominal and worst-dose L2 no greater than same reference, no positive-target blank at any dose",
        "calibration_boundary": "once only after all5 complete qualified FIT-only source vectors freeze and identities rechecked; own FW eligibility, never fabricate MILP optimal certificates; failures after opening remain opened_then_failed no_retry",
        "retries": "one exclusive consumed marker beside plan with same parent as original prerequisite manifest; no silent retry or budget reset; new attempt requires new prospective plan",
        "interpretation": "nonconvex surrogate experiment; computational restarts, not independent generalization; no global optimum claim; final3 closed",
    }
    if set(protocol) != set(exact_protocol) | {"original_fit_gate", "calibration_gate"}:
        raise ValueError("coverage plan protocol fields differ from the frozen contract")
    for key, expected in exact_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError("coverage plan protocol field differs from frozen value: " + key)
    validate_budget_values(
        protocol["per_seed_time_limit_seconds"],
        protocol["total_time_limit_seconds"],
        protocol["total_steps_per_seed"],
    )
    if (not isinstance(protocol.get("original_fit_gate"), dict)
            or set(protocol["original_fit_gate"]) != {
                "mean_band_pixels_max", "mean_nominal_l2_pixels_max", "mean_worst_dose_l2_pixels_max"
            }
            or protocol.get("original_fit_gate") != {
        "mean_band_pixels_max": 234.5,
        "mean_nominal_l2_pixels_max": 0,
        "mean_worst_dose_l2_pixels_max": 150.25,
    }):
        raise ValueError("coverage plan original FIT gate changed")
    if (not isinstance(protocol.get("calibration_gate"), dict)
            or set(protocol["calibration_gate"]) != {
                "mean_band_pixels_max", "mean_nominal_l2_pixels_max",
                "mean_worst_dose_l2_pixels_max", "each_seed_band_pixels_strictly_less_than",
                "no_blank_positive_target_any_dose", "all_five_complete_fit_qualified",
            }
            or protocol.get("calibration_gate") != {
        "mean_band_pixels_max": 239.4,
        "mean_nominal_l2_pixels_max": 56.175,
        "mean_worst_dose_l2_pixels_max": 157.2375,
        "each_seed_band_pixels_strictly_less_than": 268,
        "no_blank_positive_target_any_dose": True,
        "all_five_complete_fit_qualified": True,
    }):
        raise ValueError("coverage plan calibration gate changed")
    generation = plan.get("new_fit_generation")
    expected_ids = list(LAYOUT_IDS)
    if (not isinstance(generation, dict)
            or set(generation) != {
                "teacher", "policy", "layout_ids", "novelty",
                "positive_fraction_min", "positive_fraction_max",
            }
            or generation.get("layout_ids") != expected_ids
            or generation.get("teacher") != "same frozen original synthetic teacher and physics, targets fixed independently of candidate and LP anchor before optimization"
            or generation.get("policy") != "three independent functions; no legacy make_layouts or heldout generation/access; generator source pinned by clean reviewed committed code before execution; no adaptive geometry correction based on scores"
            or generation.get("novelty") != "duplicate mask/target hashes checked only against permitted original FIT metadata and new3; no calibration/final3 data access"
            or generation.get("positive_fraction_min") != 0.01
            or generation.get("positive_fraction_max") != 0.99):
        raise ValueError("coverage plan new-FIT generation policy differs from frozen values")
    if plan.get("completion") != "all5 complete204steps or registered convergence accounting; deadline preserves incumbent and closes calibration; same immutable control gates":
        raise ValueError("coverage plan completion policy changed")


def _validate_schema6_plan_payload(plan: dict) -> None:
    if not isinstance(plan, dict):
        raise ValueError("coverage plan must be an object")
    added_fields = {
        "superseded_coverage_attempt", "fixed_fit_layout_hashes",
        "initialization_protocol", "diagnostics_protocol",
    }
    expected_fields = {
        "schema_version", "status", "created_utc", "objective_id", "base_commit",
        "dataset_file", "diagnostic_file", "prerequisite_manifest", "previous_report",
        "previous_plan", "input_hashes", "scope", "physical", "protocol",
        "new_fit_generation", "completion",
    } | added_fields
    if set(plan) != expected_fields:
        raise ValueError("coverage plan fields differ from the frozen schema-6 contract")
    if (plan.get("schema_version") != 6
            or plan.get("status") != "prospective_candidate_plan"
            or plan.get("objective_id") != OBJECTIVE_ID
            or plan.get("base_commit") != SCHEMA6_BASE_COMMIT):
        raise ValueError("coverage plan schema/status/objective/base commit differs from schema 6")
    superseded = plan.get("superseded_coverage_attempt")
    expected_superseded = {
        "status": "no_fit_qualified_checkpoint",
        "plan_path": "/home/daniel/experiments/robust-source-quality-20261005-766872/coverage_candidate_plan_7b4cbc4.json",
        "plan_sha256": PINNED_SUPERSEDED_COVERAGE_PLAN_SHA256,
        "report_path": "/home/daniel/experiments/robust-source-quality-20261005-766872/coverage-runs/20261007T035331Z_64bed05d/coverage_report.json",
        "report_sha256": PINNED_SUPERSEDED_COVERAGE_REPORT_SHA256,
    }
    if superseded != expected_superseded:
        raise ValueError("schema-6 plan does not preserve the consumed schema-5 coverage attempt")
    if plan.get("fixed_fit_layout_hashes") != PINNED_FIXED_FIT_LAYOUT_HASHES:
        raise ValueError("schema-6 fixed FIT mask/target hashes differ from the consumed attempt")
    if plan.get("initialization_protocol") != SCHEMA6_INITIALIZATION_PROTOCOL:
        raise ValueError("schema-6 feasible-start protocol differs from its frozen values")
    if plan.get("diagnostics_protocol") != SCHEMA6_DIAGNOSTICS_PROTOCOL:
        raise ValueError("schema-6 per-layout diagnostic protocol differs from its frozen values")

    # Reuse the complete schema-5 checks for all inherited physics, objective,
    # lineage, gates, fixed layouts, and completion policy. Only the starting
    # rule and the code base commit are intentionally new in schema 6.
    inherited = dict(plan)
    for key in added_fields:
        inherited.pop(key)
    inherited["schema_version"] = 5
    inherited["base_commit"] = "7483c1e7c8003f3e2f61caa6324a30d0662a8121"
    inherited_protocol = dict(plan.get("protocol", {}))
    inherited_protocol["initial_simplex_jitter"] = 1e-4
    inherited_protocol["fallback_if_guard_infeasible"] = (
        "exact previous FIT-only seed17 reference; record fallback"
    )
    inherited["protocol"] = inherited_protocol
    _validate_schema5_plan_payload(inherited)

    protocol = plan.get("protocol")
    if (protocol.get("initial_simplex_jitter") != 0.0
            or protocol.get("fallback_if_guard_infeasible")
            != "no fallback; fail closed after registered feasible-start search"):
        raise ValueError("schema-6 plan must disable jitter and reference fallback")


def validate_superseded_coverage_report(report: dict, expected_plan_sha256: str) -> None:
    """Validate only the FIT/attempt boundary of the consumed schema-5 report.

    Deliberately does not inspect a calibration result or any calibration metrics.
    """
    if (not isinstance(report, dict)
            or report.get("schema_version") != 1
            or report.get("objective_id") != OBJECTIVE_ID
            or report.get("status") != "no_fit_qualified_checkpoint"
            or report.get("plan_sha256") != expected_plan_sha256
            or report.get("coverage_attempt_consumed") is not True
            or report.get("calibration_status") != "closed"
            or report.get("final3_status") != "never indexed or evaluated"):
        raise ValueError("superseded schema-5 coverage report is not the closed failed FIT attempt")
    observed_layouts = report.get("new_fit_layouts")
    if (not isinstance(observed_layouts, list)
            or any(not isinstance(row, dict) for row in observed_layouts)):
        raise ValueError("superseded coverage report lacks well-formed fixed FIT layout identities")
    observed_hashes = [
        {key: row.get(key) for key in ("layout_id", "mask_sha256", "target_sha256")}
        for row in observed_layouts
    ]
    if observed_hashes != PINNED_FIXED_FIT_LAYOUT_HASHES:
        raise ValueError("superseded coverage report FIT mask/target hashes differ from schema 6")
    seeds = report.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != len(SEEDS):
        raise ValueError("superseded coverage report does not contain all five seeds")
    fallback_hashes = []
    for seed, row in zip(SEEDS, seeds):
        if not isinstance(row, dict):
            raise ValueError("superseded coverage report seed history is malformed")
        initialization = row.get("initialization")
        last_incumbent = row.get("last_incumbent")
        rejected_jitter = (initialization.get("rejected_jitter_check")
                           if isinstance(initialization, dict) else None)
        reference_check = initialization.get("check") if isinstance(initialization, dict) else None
        final_weights_sha = (last_incumbent.get("weights_sha256")
                             if isinstance(last_incumbent, dict) else None)
        if (row.get("seed") != seed
                or row.get("status") != "complete"
                or row.get("steps_completed") != STEPS_PER_SEED
                or not isinstance(initialization, dict)
                or initialization.get("fallback") is not True
                or initialization.get("used_jitter") is not False
                or not isinstance(rejected_jitter, dict)
                or rejected_jitter.get("passed") is not False
                or not isinstance(reference_check, dict)
                or reference_check.get("passed") is not True
                or row.get("qualified_checkpoint_count") != 0
                or row.get("selected") is not None
                or not isinstance(final_weights_sha, str) or len(final_weights_sha) != 64):
            raise ValueError("superseded coverage report seed history differs from the consumed attempt")
        fallback_hashes.append(final_weights_sha)
    if len(set(fallback_hashes)) != 1:
        raise ValueError("superseded coverage report seeds did not share the reference fallback")


def validate_plan_payload(plan: dict) -> None:
    """Dispatch the immutable v5 protocol or the separately pinned v6 protocol."""
    if not isinstance(plan, dict):
        raise ValueError("coverage plan must be an object")
    if plan.get("schema_version") == 5:
        _validate_schema5_plan_payload(plan)
    elif plan.get("schema_version") == 6:
        _validate_schema6_plan_payload(plan)
    else:
        raise ValueError("coverage plan schema version is not registered")


def validate_plan_file(path: str | Path, expected_sha256: str) -> tuple[dict, str]:
    raw, actual_sha = read_hashed_bytes(path)
    expected = str(expected_sha256).lower()
    # Keep every non-v6 request on the original schema-5 pin/error path.
    if expected != PINNED_PLAN_SHA256_V6:
        if expected != PINNED_PLAN_SHA256 or actual_sha != PINNED_PLAN_SHA256:
            raise ValueError("plan SHA256 does not match the required frozen plan pin")
        plan = json.loads(raw.decode("utf-8"))
        _validate_schema5_plan_payload(plan)
        return plan, actual_sha
    if actual_sha != PINNED_PLAN_SHA256_V6:
        raise ValueError("plan SHA256 does not match the required frozen schema-6 plan pin")
    plan = json.loads(raw.decode("utf-8"))
    validate_plan_payload(plan)
    return plan, actual_sha


def validate_previous_report(report: dict) -> dict:
    """Read only FIT lineage/weights from the pinned Pareto JSON."""
    if not isinstance(report, dict) or report.get("objective_id") != "target_aware_pareto_protected_buffered_count_v1":
        raise ValueError("previous report has the wrong objective identity")
    if report.get("status") != "complete" or report.get("fit_improvement_over_anchor") is not True:
        raise ValueError("previous FIT-only source run did not complete with improvement")
    selection = report.get("fit_selection")
    if not isinstance(selection, dict) or selection.get("selected_seed") != 17:
        raise ValueError("previous FIT-only selection is not the frozen seed 17")
    if selection.get("all_five_seed_weights_frozen") is not True:
        raise ValueError("previous report did not freeze all five FIT-only weights")
    seeds = report.get("seeds")
    if not isinstance(seeds, list) or [row.get("seed") for row in seeds] != list(SEEDS):
        raise ValueError("previous report seed list/order differs from the frozen five seeds")
    for row in seeds:
        fit = row.get("fit") if isinstance(row, dict) else None
        solver = row.get("solver") if isinstance(row, dict) else None
        if (not isinstance(fit, dict) or fit.get("passed") is not True
                or not isinstance(solver, dict) or solver.get("optimal_zero_gap") is not True):
            raise ValueError("previous seed lacks a qualified optimal zero-gap FIT record")
    preflight = report.get("preflight")
    fit_layout_ids = preflight.get("fit_layout_ids") if isinstance(preflight, dict) else None
    canonical_parity = preflight.get("basis_parity") if isinstance(preflight, dict) else None
    validate_canonical_basis_parity(canonical_parity)
    if fit_layout_ids != list(ORIGINAL_FIT_LAYOUT_IDS):
        raise ValueError("previous report FIT layout order differs from the frozen original FIT contract")
    selected = next(row for row in seeds if row["seed"] == 17)
    fit_metrics = selected["fit"].get("fit_metrics")
    per_layout = fit_metrics.get("per_layout") if isinstance(fit_metrics, dict) else None
    if (not isinstance(per_layout, list) or len(per_layout) != len(ORIGINAL_FIT_LAYOUT_IDS)
            or [row.get("layout_id") for row in per_layout if isinstance(row, dict)]
            != list(ORIGINAL_FIT_LAYOUT_IDS)):
        raise ValueError("previous seed17 per-layout FIT metrics are missing or reordered")
    required_metrics = ("band_pixels", "L2_pixels", "L2_worst_dose_pixels")
    expected_by_id = {
        "train_vls_pitch24_width9": (512, 0, 256),
        "train_hls_pitch28_width10": (256, 0, 256),
        "train_l_contours": (111, 0, 59),
        "train_t_junctions": (59, 0, 30),
    }
    for metric_row in per_layout:
        for key in required_metrics:
            value = metric_row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("previous seed17 per-layout FIT metric is not an integer count")
        expected_values = expected_by_id[metric_row["layout_id"]]
        observed = (metric_row["band_pixels"], metric_row["L2_pixels"],
                    metric_row["L2_worst_dose_pixels"])
        if observed != expected_values:
            raise ValueError("previous seed17 FIT per-layout metrics differ from pinned lineage")
    weights = np.asarray(selected["fit"].get("weights"), dtype=np.float64)
    anchor = np.asarray(report.get("anchor", {}).get("source_weights_supported_float64"), dtype=np.float64)
    if (weights.shape != (49,) or anchor.shape != (49,)
            or not np.isfinite(weights).all() or not np.isfinite(anchor).all()
            or np.any(weights < 0.0) or np.any(anchor < 0.0)
            or abs(float(weights.sum()) - 1.0) > 1e-9
            or abs(float(anchor.sum()) - 1.0) > 1e-9):
        raise ValueError("previous seed17/reference or LP anchor has invalid 49-weight simplex values")
    return {
        "reference_weights": weights, "lp_anchor": anchor, "selected_seed": 17,
        "fit_layout_ids": tuple(fit_layout_ids),
        "fit_metrics_per_layout": per_layout,
        "canonical_basis_parity": canonical_parity,
    }


def validate_canonical_basis_parity(value) -> None:
    if (not isinstance(value, list) or len(value) != 1
            or not isinstance(value[0], dict)
            or set(value[0]) != {"layout_id", "max_abs_error"}
            or value[0].get("layout_id") != ORIGINAL_FIT_LAYOUT_IDS[0]
            or isinstance(value[0].get("max_abs_error"), bool)
            or not isinstance(value[0].get("max_abs_error"), (int, float))
            or not math.isfinite(float(value[0]["max_abs_error"]))
            or float(value[0]["max_abs_error"]) > 1e-6
            or value[0]["max_abs_error"] != PINNED_BASIS_PARITY[0]["max_abs_error"]
            or sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
            != PINNED_BASIS_PARITY_SHA256):
        raise ValueError("previous-report canonical FIT basis parity differs from its pinned identity")


def validate_budget_values(per_seed_seconds, total_seconds, steps_per_seed) -> None:
    if (per_seed_seconds != PER_SEED_LIMIT_SECONDS
            or total_seconds != TOTAL_LIMIT_SECONDS
            or steps_per_seed != STEPS_PER_SEED):
        raise ValueError("coverage solver budgets/step count differ from the frozen plan")


def fit_freeze_eligibility(seed_records: Sequence[dict]) -> dict:
    """Fail closed before calibration unless all five FIT-only vectors are frozen."""
    if not isinstance(seed_records, (list, tuple)) or len(seed_records) != len(SEEDS):
        return {"passed": False, "reason": "five_seed_records_required"}
    for expected_seed, row in zip(SEEDS, seed_records):
        if (not isinstance(row, dict) or row.get("seed") != expected_seed
                or row.get("status") != "complete"
                or row.get("steps_completed") != STEPS_PER_SEED
                or not isinstance(row.get("selected"), dict)
                or row["selected"].get("qualified") is not True):
            return {"passed": False, "reason": "incomplete_or_unqualified_fit_seed",
                    "seed": expected_seed}
        weights = np.asarray(row["selected"].get("weights"), dtype=np.float64)
        if (weights.shape != (49,) or not np.isfinite(weights).all()
                or np.any(weights < 0.0) or abs(float(weights.sum()) - 1.0) > SOLVER_TOLERANCE):
            return {"passed": False, "reason": "invalid_frozen_source_vector",
                    "seed": expected_seed}
    return {"passed": True, "seed_order": list(SEEDS), "all_five_fit_vectors_frozen": True}


def verify_clean_source_commit(root: str | Path, source_paths: Sequence[str | Path]) -> dict:
    """Capture a clean committed source identity for before/after rechecks."""
    root = Path(root).resolve()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                            cwd=root, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise ValueError("coverage run requires a clean committed source tree")
    hashes = {}
    for path in source_paths:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("source identity path is outside the repository") from exc
        hashes[str(resolved.relative_to(root))] = sha256_file(resolved)
    payload = json.dumps({"head": head, "files": hashes}, sort_keys=True, separators=(",", ":")).encode()
    return {"head": head, "files": hashes, "sha256": sha256_bytes(payload)}


def create_attempt_marker(manifest_path: str | Path, plan_sha256: str) -> Path:
    """Consume one prospective plan once, atomically and without retry."""
    parent = Path(manifest_path).resolve().parent
    marker = parent / ("coverage_attempt_" + plan_sha256 + ".consumed")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(marker, flags, 0o600)
    try:
        payload = json.dumps({"plan_sha256": plan_sha256, "status": "consumed"},
                             sort_keys=True).encode("utf-8")
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return marker


@dataclass(frozen=True)
class CoverageLayout:
    layout_id: str
    family: str
    mask: np.ndarray


@dataclass(frozen=True)
class GuardedDomain:
    """Sparse 49-D source simplex with original and prospective FIT guards."""

    a_ub: sparse.csr_matrix
    b_ub: np.ndarray
    guard_counts: dict
    source_count: int
    tolerance: float = SOLVER_TOLERANCE

    def verify(self, weights: np.ndarray) -> dict:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != self.source_count or not np.isfinite(w).all():
            return {"passed": False, "reason": "invalid_shape_or_nonfinite"}
        row_violation = np.asarray(self.a_ub @ w - self.b_ub).reshape(-1)
        max_row = float(np.maximum(row_violation, 0.0).max(initial=0.0))
        max_bound = float(np.maximum(-w, 0.0).max(initial=0.0))
        simplex_error = abs(float(w.sum()) - 1.0)
        maximum = max(max_row, max_bound, simplex_error)
        return {
            "passed": bool(maximum <= self.tolerance),
            "maximum_row_violation": max_row,
            "maximum_bound_violation": max_bound,
            "simplex_sum_error": simplex_error,
            "tolerance": self.tolerance,
        }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _rect(mask: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> None:
    h, w = mask.shape
    y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
    if y0 < y1 and x0 < x1:
        mask[y0:y1, x0:x1] = 1.0


def make_finite_ribbons_mask(size: int = 128) -> np.ndarray:
    if size != 128:
        raise ValueError("coverage raster is fixed at 128 by 128 pixels")
    ribbons = np.zeros((size, size), dtype=np.float32)
    # Finite ribbons intentionally vary width, span, orientation, and placement.
    for y0, y1, x0, x1 in (
        (9, 21, 8, 112), (29, 43, 21, 119), (58, 71, 4, 94),
        (81, 97, 28, 124), (106, 118, 14, 87),
    ):
        _rect(ribbons, y0, y1, x0, x1)
    for x0, x1, y0, y1 in ((16, 28, 21, 54), (93, 107, 52, 78)):
        _rect(ribbons, y0, y1, x0, x1)
    return ribbons


def make_asymmetric_line_ends_mask(size: int = 128) -> np.ndarray:
    if size != 128:
        raise ValueError("coverage raster is fixed at 128 by 128 pixels")
    ends = np.zeros((size, size), dtype=np.float32)
    # Asymmetric finite line ends with one jog; these coordinates are new.
    for x0, width, y0, y1 in ((12, 12, 10, 48), (40, 14, 24, 75),
                              (76, 13, 8, 56), (103, 12, 48, 113)):
        _rect(ends, y0, y1, x0, x0 + width)
    _rect(ends, 61, 75, 40, 67)
    _rect(ends, 61, 88, 53, 67)
    _rect(ends, 91, 105, 80, 112)
    return ends


def make_irregular_contacts_mask(size: int = 128) -> np.ndarray:
    if size != 128:
        raise ValueError("coverage raster is fixed at 128 by 128 pixels")
    contacts = np.zeros((size, size), dtype=np.float32)
    # Two independently placed rows; sizes are 20-24 px and all components separate.
    contacts_spec = (
        (18, 20, 20), (47, 21, 22), (80, 18, 20), (111, 22, 22),
        (16, 89, 22), (47, 87, 20), (79, 92, 24), (112, 88, 20),
    )
    for cx, cy, width in contacts_spec:
        y0, x0 = cy - width // 2, cx - width // 2
        _rect(contacts, y0, y0 + width, x0, x0 + width)
    return contacts


def make_coverage_masks(size: int = 128) -> tuple[CoverageLayout, ...]:
    """Call only the three new, isolated FIT geometry functions."""
    if size != 128:
        raise ValueError("coverage raster is fixed at 128 by 128 pixels")
    layouts = (
        CoverageLayout(LAYOUT_IDS[0], "finite_ribbons", make_finite_ribbons_mask(size)),
        CoverageLayout(LAYOUT_IDS[1], "asymmetric_line_ends", make_asymmetric_line_ends_mask(size)),
        CoverageLayout(LAYOUT_IDS[2], "irregular_contact_array", make_irregular_contacts_mask(size)),
    )
    if any(not np.any(row.mask) or not np.isfinite(row.mask).all() for row in layouts):
        raise ValueError("coverage masks must be non-empty and finite")
    if len({sha256_array(row.mask) for row in layouts}) != len(layouts):
        raise ValueError("coverage mask generator produced duplicates")
    return layouts


def validate_layout_rows(rows: Sequence[dict], expected_ids: Sequence[str],
                         mask_descriptors: Sequence[dict],
                         target_descriptors: Sequence[dict], label: str) -> dict:
    """Check ordered raster identities using hashes of the exact row bytes."""
    ids = [row.get("layout_id") for row in rows]
    if ids != list(expected_ids):
        raise ValueError(label + " layout IDs/order differ from frozen lineage")
    expected_masks = {row["layout_id"]: row["sha256"] for row in mask_descriptors}
    expected_targets = {row["layout_id"]: row["sha256"] for row in target_descriptors}
    if set(expected_masks) != set(ids) or set(expected_targets) != set(ids):
        raise ValueError(label + " diagnostic hash IDs differ from frozen layout order")
    observed = []
    for row in rows:
        mask_hash = sha256_array(np.asarray(row["mask"], dtype=np.float32))
        target_hash = sha256_array(target_hashable(row["target"]))
        if mask_hash != expected_masks[row["layout_id"]]:
            raise ValueError(label + " mask hash mismatch for " + row["layout_id"])
        if target_hash != expected_targets[row["layout_id"]]:
            raise ValueError(label + " target hash mismatch for " + row["layout_id"])
        observed.append({"layout_id": row["layout_id"],
                         "mask_sha256": mask_hash, "target_sha256": target_hash})
    return {"layout_ids": ids, "rows": observed}


def classify_fw_gap(gap: float, tolerance: float = SOLVER_TOLERANCE) -> str:
    value = float(gap)
    if not math.isfinite(value):
        return "nonfinite_failure"
    if value < -tolerance:
        return "negative_gap_failure"
    if value <= tolerance:
        return "stationary_tolerance"
    return "continue"


def target_hashable(value) -> np.ndarray:
    """Return a canonical binary float32 raster matching the legacy tensor hash."""
    array = np.asarray(value)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("target must be a finite two-dimensional raster")
    if not np.isin(array, (0, 1, False, True)).all():
        raise ValueError("target must contain only binary values")
    return np.ascontiguousarray(array, dtype=np.float32)


def critical_signed_margins(
    basis: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    b = np.asarray(basis, dtype=np.float64)
    t = np.asarray(target, dtype=bool)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if b.ndim != 3 or t.ndim != 2 or b.shape[0] != w.size or b.shape[1:] != t.shape:
        raise ValueError("basis, target, and weights have incompatible shapes")
    if not np.isfinite(b).all() or not np.isfinite(w).all() or np.any(b < 0.0):
        raise ValueError("basis and weights must be finite; basis must be nonnegative")
    aerial = np.einsum("n,nhw->hw", w, b, optimize=True)
    return np.where(t, LOW_DOSE * aerial - THRESHOLD,
                    THRESHOLD - HIGH_DOSE * aerial)


def deterministic_guard_indices(target: np.ndarray, per_class: int = NEW_GUARDS_PER_CLASS) -> np.ndarray:
    """Row-major fixed-stride sample, balanced by target class and weight-independent."""
    t = np.asarray(target, dtype=bool)
    if t.ndim != 2 or per_class < 1:
        raise ValueError("target must be a 2-D raster and per_class positive")
    selected = []
    flat = t.reshape(-1)
    for label in (True, False):
        candidates = np.flatnonzero(flat == label)
        if candidates.size:
            positions = np.linspace(0, candidates.size - 1,
                                    min(per_class, candidates.size), dtype=np.int64)
            selected.extend(candidates[positions].tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _simplex_lower_bound(flat: np.ndarray, positive: np.ndarray) -> np.ndarray:
    return np.where(positive,
                    LOW_DOSE * np.min(flat, axis=1) - THRESHOLD,
                    THRESHOLD - HIGH_DOSE * np.max(flat, axis=1))


def _critical_rows(
    basis: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    floors: np.ndarray,
    layout_id: str,
    tolerance: float,
    include_details: bool = False,
) -> tuple[sparse.csr_matrix, np.ndarray, dict]:
    n = basis.shape[0]
    flat = np.asarray(basis, dtype=np.float64).reshape(n, -1).T
    t = np.asarray(target, dtype=bool).reshape(-1)
    ids = np.asarray(indices, dtype=np.int64).reshape(-1)
    f = np.asarray(floors, dtype=np.float64).reshape(-1)
    if ids.size != f.size or np.any(ids < 0) or np.any(ids >= t.size):
        raise ValueError("guard indices and floors are inconsistent")
    if not np.isfinite(f).all() or np.any(f <= 0.0):
        raise ValueError("critical guard floors must be finite and positive")
    if ids.size == 0:
        return sparse.csr_matrix((0, n)), np.empty(0), {
            "layout_id": layout_id, "candidate_rows": 0, "kept_rows": 0,
            "pruned_simplex_rows": 0,
        }
    positive = t[ids]
    sign = np.where(positive, 1.0, -1.0)
    dose = np.where(positive, LOW_DOSE, HIGH_DOSE)
    simplex_lb = _simplex_lower_bound(flat[ids], positive)
    safe = simplex_lb >= f + tolerance
    keep = ~safe
    rows = (-sign[keep] * dose[keep])[:, None] * flat[ids[keep]]
    rhs = -f[keep] - sign[keep] * THRESHOLD
    info = {
        "layout_id": layout_id,
        "candidate_rows": int(ids.size),
        "kept_rows": int(keep.sum()),
        "pruned_simplex_rows": int(safe.sum()),
    }
    if include_details:
        info["candidate_indices_sha256"] = sha256_array(np.asarray(ids, dtype="<i8"))
        info["kept_indices_sha256"] = sha256_array(np.asarray(ids[keep], dtype="<i8"))
        info["kept_floors_sha256"] = sha256_array(np.asarray(f[keep], dtype="<f8"))
    return sparse.csr_matrix(rows), rhs, info


def _nominal_arrays(bases: Sequence[np.ndarray], targets: Sequence[np.ndarray]):
    if not bases or len(bases) != len(targets):
        raise ValueError("matching original FIT bases and targets are required")
    n = np.asarray(bases[0]).shape[0]
    mats, labels = [], []
    for basis, target in zip(bases, targets):
        b = np.asarray(basis, dtype=np.float64)
        t = np.asarray(target, dtype=bool)
        if b.ndim != 3 or b.shape[0] != n or b.shape[1:] != t.shape:
            raise ValueError("original FIT basis/target shape mismatch")
        if not np.isfinite(b).all() or np.any(b < 0.0):
            raise ValueError("original FIT basis must be finite and nonnegative")
        mats.append(b.reshape(n, -1).T)
        labels.append(t.reshape(-1))
    return np.concatenate(mats), np.concatenate(labels)


def verify_original_nominal_polytope(
    bases: Sequence[np.ndarray], targets: Sequence[np.ndarray], weights: np.ndarray,
    tolerance: float = 1e-9,
) -> dict:
    """Check every original FIT nominal-q row, without protected-row pruning."""
    matrix, labels = _nominal_arrays(bases, targets)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != matrix.shape[1] or not np.isfinite(w).all():
        return {"passed": False, "reason": "invalid_weight_shape_or_nonfinite"}
    sign = np.where(labels, 1.0, -1.0)
    margin = sign * (matrix @ w - THRESHOLD)
    violation = float(np.maximum(NOMINAL_FLOOR - margin, 0.0).max(initial=0.0))
    sum_error = abs(float(w.sum()) - 1.0)
    min_weight_violation = float(np.maximum(-w, 0.0).max(initial=0.0))
    maximum = max(violation, sum_error, min_weight_violation)
    return {
        "passed": bool(maximum <= tolerance),
        "minimum_signed_margin": float(margin.min(initial=np.inf)),
        "required_margin_floor": NOMINAL_FLOOR,
        "max_positive_violation": violation,
        "flux_sum": float(w.sum()),
        "minimum_weight": float(w.min(initial=0.0)),
        "row_count": int(matrix.shape[0]),
        "tolerance": tolerance,
    }


def build_guarded_domain(
    original_bases: Sequence[np.ndarray],
    original_targets: Sequence[np.ndarray],
    original_layout_ids: Sequence[str],
    new_bases: Sequence[np.ndarray],
    new_targets: Sequence[np.ndarray],
    new_layout_ids: Sequence[str],
    lp_anchor: np.ndarray,
    reference_weights: np.ndarray,
    tolerance: float = SOLVER_TOLERANCE,
) -> tuple[GuardedDomain, dict]:
    """Build a pruned LMO domain and verify the full original q polytope.

    Full original q is verified before pruning. A q row can leave the LMO only
    when an exact simplex bound proves it or the same-pixel anchor critical
    protection with epsilon 2e-6 implies a strictly stronger nominal margin.
    Reference guards are independently added with positive capped floors.
    """
    original_layout_ids = tuple(original_layout_ids)
    new_layout_ids = tuple(new_layout_ids)
    if len(original_layout_ids) != len(original_bases) or len(new_layout_ids) != len(new_bases):
        raise ValueError("layout IDs must match basis lists")
    if len(set(original_layout_ids + tuple(new_layout_ids))) != len(original_layout_ids) + len(new_layout_ids):
        raise ValueError("layout IDs must be unique")
    n = np.asarray(original_bases[0]).shape[0]
    anchor = np.asarray(lp_anchor, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_weights, dtype=np.float64).reshape(-1)
    if (anchor.size != n or reference.size != n or not np.isfinite(anchor).all()
            or not np.isfinite(reference).all()):
        raise ValueError("anchor/reference weights do not match the source basis")
    if (anchor.min(initial=0.0) < -tolerance or reference.min(initial=0.0) < -tolerance
            or abs(float(anchor.sum()) - 1.0) > tolerance
            or abs(float(reference.sum()) - 1.0) > tolerance):
        raise ValueError("anchor and reference must lie on the source simplex")

    full_matrix, full_labels = _nominal_arrays(original_bases, original_targets)
    sign = np.where(full_labels, 1.0, -1.0)
    nominal_a = -sign[:, None] * full_matrix
    nominal_b = -sign * THRESHOLD - NOMINAL_FLOOR
    anchor_nominal = verify_original_nominal_polytope(
        original_bases, original_targets, anchor, tolerance
    )
    reference_nominal = verify_original_nominal_polytope(
        original_bases, original_targets, reference, tolerance
    )
    if not anchor_nominal["passed"] or not reference_nominal["passed"]:
        raise ValueError("LP anchor or reference violates the full original nominal polytope")

    # Prove nominal rows redundant either over the full source simplex or via
    # the stronger same-pixel critical protection at the frozen LP anchor.
    nominal_simplex_lb = np.where(
        full_labels,
        np.min(full_matrix, axis=1) - THRESHOLD,
        THRESHOLD - np.max(full_matrix, axis=1),
    )
    nominal_simplex_safe = nominal_simplex_lb >= NOMINAL_FLOOR + tolerance
    anchor_critical_correct = np.zeros(full_labels.size, dtype=bool)
    row_offset = 0
    for basis, target in zip(original_bases, original_targets):
        margins = critical_signed_margins(basis, target, anchor).reshape(-1)
        anchor_critical_correct[row_offset:row_offset + margins.size] = margins >= EPSILON - tolerance
        row_offset += margins.size
    if row_offset != full_labels.size:
        raise ValueError("original FIT q rows and critical guard rows lost layout alignment")
    anchor_protection_safe = anchor_critical_correct
    nominal_pruned = nominal_simplex_safe | anchor_protection_safe
    nominal_kept = ~nominal_pruned
    nominal_rows = sparse.csr_matrix(nominal_a[nominal_kept])
    nominal_rhs = nominal_b[nominal_kept]

    blocks = [nominal_rows]
    rhs_blocks = [nominal_rhs]
    counts = {
        "original_nominal_full_rows_verified": int(full_matrix.shape[0]),
        "original_nominal_rows_kept_after_exact_pruning": int(nominal_kept.sum()),
        "original_nominal_rows_pruned_by_simplex_proof": int(nominal_simplex_safe.sum()),
        "original_nominal_rows_pruned_by_anchor_critical_dominance": int((anchor_protection_safe & ~nominal_simplex_safe).sum()),
        "original_nominal_kept_mask_sha256": sha256_array(np.asarray(nominal_kept, dtype=np.uint8)),
        "original_nominal_pruning_certificates": {
            "simplex_vertex_bounds": int(nominal_simplex_safe.sum()),
            "same_pixel_anchor_critical_epsilon_dominance": int((anchor_protection_safe & ~nominal_simplex_safe).sum()),
            "simplex_pruned_mask_sha256": sha256_array(np.asarray(nominal_simplex_safe, dtype=np.uint8)),
            "anchor_dominance_mask_sha256": sha256_array(np.asarray(anchor_protection_safe & ~nominal_simplex_safe, dtype=np.uint8)),
            "critical_floor": EPSILON, "nominal_floor": NOMINAL_FLOOR,
            "proof": "low-dose positive or high-dose negative anchor protection implies same-pixel nominal margin strictly above nominal floor",
        },
        "original_anchor_protections": [],
        "reference_guards_original": [],
        "reference_guards_new": [],
    }

    for layout_id, basis, target in zip(original_layout_ids, original_bases, original_targets):
        b = np.asarray(basis, dtype=np.float64)
        t = np.asarray(target, dtype=bool)
        pixel_count = t.size
        margins = critical_signed_margins(b, t, anchor).reshape(-1)
        if np.any((margins >= 0.0) & (margins < EPSILON - tolerance)):
            raise ValueError("original LP anchor correct pixel lacks the registered epsilon buffer")
        indices = np.flatnonzero(margins >= 0.0)
        floors = np.full(indices.size, EPSILON, dtype=np.float64)
        block, rhs, info = _critical_rows(b, t, indices, floors, layout_id, tolerance,
                                           include_details=True)
        blocks.append(block)
        rhs_blocks.append(rhs)
        counts["original_anchor_protections"].append(info)

        ref_margin = critical_signed_margins(b, t, reference).reshape(-1)
        ref_indices_all = np.flatnonzero(ref_margin > 0.0)
        anchor_protected = margins >= EPSILON - tolerance
        dominated = anchor_protected[ref_indices_all]
        ref_indices = ref_indices_all[~dominated]
        ref_floors = np.minimum(EPSILON, 0.5 * ref_margin[ref_indices])
        block, rhs, info = _critical_rows(b, t, ref_indices, ref_floors, layout_id, tolerance,
                                           include_details=True)
        info["candidate_rows_before_anchor_dominance"] = int(ref_indices_all.size)
        info["pruned_by_same_pixel_anchor_protection"] = int(dominated.sum())
        info["dominance_certificate"] = "same signed low/high-dose row and reference floor <= LP-anchor epsilon"
        blocks.append(block)
        rhs_blocks.append(rhs)
        counts["reference_guards_original"].append(info)
        if b.shape[0] != n:
            raise ValueError("original FIT source dimension differs across layouts")

    for layout_id, basis, target in zip(new_layout_ids, new_bases, new_targets):
        b = np.asarray(basis, dtype=np.float64)
        t = np.asarray(target, dtype=bool)
        if b.ndim != 3 or b.shape[0] != n or b.shape[1:] != t.shape:
            raise ValueError("new FIT basis/target/source dimension mismatch")
        ref_margin = critical_signed_margins(b, t, reference).reshape(-1)
        sampled = deterministic_guard_indices(t)
        indices = sampled[ref_margin[sampled] > 0.0]
        floors = np.minimum(EPSILON, 0.5 * ref_margin[indices])
        block, rhs, info = _critical_rows(b, t, indices, floors, layout_id, tolerance,
                                           include_details=True)
        info["sampled_indices_sha256"] = sha256_array(np.asarray(sampled, dtype="<i8"))
        blocks.append(block)
        rhs_blocks.append(rhs)
        counts["reference_guards_new"].append(info)

    a_ub = sparse.vstack(blocks, format="csr")
    b_ub = np.concatenate(rhs_blocks) if rhs_blocks else np.empty(0, dtype=np.float64)
    domain = GuardedDomain(a_ub, b_ub, counts, n, tolerance)
    anchor_check = domain.verify(anchor)
    reference_check = domain.verify(reference)
    if not reference_check["passed"]:
        raise ValueError("seed17 reference is not feasible in the frozen augmented guard domain")
    counts["anchor_domain_check"] = anchor_check
    counts["anchor_augmented_feasibility_is_informational"] = True
    counts["reference_domain_check"] = reference_check
    counts["constraint_rows_kept"] = int(a_ub.shape[0])
    counts["constraint_nonzeros"] = int(a_ub.nnz)
    return domain, counts


def audit_float32_critical_guards(
    basis32_by_id: dict[str, np.ndarray],
    targets_by_id: dict[str, np.ndarray],
    lp_anchor: np.ndarray,
    reference_weights: np.ndarray,
    candidate_weights: np.ndarray,
    original_layout_ids: Sequence[str],
    new_layout_ids: Sequence[str],
) -> dict:
    """Pixelwise audit that frozen anchor/reference critical-correct pixels remain printed.

    It follows the float32 torch hard-print path at low dose for positive targets
    and high dose for negative targets. The selected new-layout guards are the
    fixed row-major 128-per-class sample, filtered by reference correctness.
    """
    import torch
    from light_source import resist_image

    candidate = torch.as_tensor(np.asarray(candidate_weights), dtype=torch.float32)
    failures, groups = [], []
    for role, weights in (("lp_anchor", lp_anchor), ("seed17_reference", reference_weights)):
        w64 = np.asarray(weights, dtype=np.float64).reshape(-1)
        for layout_id in original_layout_ids:
            basis = np.asarray(basis32_by_id[layout_id], dtype=np.float32)
            target = np.asarray(targets_by_id[layout_id], dtype=bool)
            margins = critical_signed_margins(basis.astype(np.float64), target, w64).reshape(-1)
            indices = np.flatnonzero(margins >= (EPSILON - SOLVER_TOLERANCE) if role == "lp_anchor" else margins > 0.0)
            groups.append((role, layout_id, indices))
    ref = np.asarray(reference_weights, dtype=np.float64).reshape(-1)
    for layout_id in new_layout_ids:
        basis = np.asarray(basis32_by_id[layout_id], dtype=np.float32)
        target = np.asarray(targets_by_id[layout_id], dtype=bool)
        margins = critical_signed_margins(basis.astype(np.float64), target, ref).reshape(-1)
        sampled = deterministic_guard_indices(target)
        indices = sampled[margins[sampled] > 0.0]
        groups.append(("seed17_reference_sampled_new", layout_id, indices))

    results = []
    for role, layout_id, indices in groups:
        basis_t = torch.as_tensor(basis32_by_id[layout_id], dtype=torch.float32)
        target = torch.as_tensor(targets_by_id[layout_id], dtype=torch.bool)
        if indices.size:
            aerial = torch.einsum("nhw,n->hw", basis_t, candidate)
            low = resist_image(aerial, dose=LOW_DOSE, threshold=THRESHOLD, steepness=50.0) >= 0.5
            high = resist_image(aerial, dose=HIGH_DOSE, threshold=THRESHOLD, steepness=50.0) >= 0.5
            idx = torch.as_tensor(indices, dtype=torch.long)
            positive = target.reshape(-1)[idx]
            actual = torch.where(positive, low.reshape(-1)[idx], high.reshape(-1)[idx])
            correct = actual == positive
            wrong = indices[~correct.cpu().numpy()]
        else:
            wrong = np.empty(0, dtype=np.int64)
        group = {"role": role, "layout_id": layout_id, "pixel_count": int(indices.size),
                 "correct_count": int(indices.size - wrong.size),
                 "failed_count": int(wrong.size),
                 "indices_sha256": sha256_array(np.asarray(indices, dtype="<i8"))}
        if wrong.size:
            group["first_failed_pixel_indices"] = wrong[:32].tolist()
            failures.append({"role": role, "layout_id": layout_id,
                             "failed_pixel_indices": wrong[:32].tolist(),
                             "failed_count": int(wrong.size)})
        # Counts and index digest retain the audit without serializing every raster coordinate.
        results.append(group)
    return {"passed": not failures, "groups": results, "failures": failures,
            "pixel_count": int(sum(row["pixel_count"] for row in results)),
            "correct_count": int(sum(row["correct_count"] for row in results)),
            "failed_count": int(sum(row["failed_count"] for row in results))}


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the unit simplex."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0 or not np.isfinite(v).all():
        raise ValueError("simplex projection requires a non-empty finite vector")
    ordered = np.sort(v)[::-1]
    cssv = np.cumsum(ordered) - 1.0
    candidates = ordered - cssv / np.arange(1, v.size + 1) > 0
    rho = int(np.flatnonzero(candidates)[-1])
    theta = cssv[rho] / float(rho + 1)
    result = np.maximum(v - theta, 0.0)
    result /= result.sum()
    return result


def simplex_jitter(reference: np.ndarray, seed: int, scale: float = 1e-4) -> np.ndarray:
    if not math.isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("simplex jitter scale must be finite and positive")
    rng = np.random.default_rng(int(seed))
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if not np.isfinite(reference).all():
        raise ValueError("reference weights must be finite")
    return project_simplex(reference + rng.normal(0.0, float(scale), size=reference.shape))


def hard_metrics(
    basis32: np.ndarray, target: np.ndarray, weights: np.ndarray,
    threshold: float = THRESHOLD, steepness: float = 50.0,
) -> dict:
    """Float32 hard print metrics for one layout over the fixed dose grid."""
    import torch
    from light_source import resist_image

    basis = torch.as_tensor(np.asarray(basis32), dtype=torch.float32)
    target_t = torch.as_tensor(np.asarray(target, dtype=bool), dtype=torch.bool)
    w = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
    if basis.ndim != 3 or basis.shape[0] != w.numel() or basis.shape[1:] != target_t.shape:
        raise ValueError("float32 basis, target, and weights have incompatible shapes")
    aerial = torch.einsum("nhw,n->hw", basis, w)
    binary = torch.stack([
        resist_image(aerial, dose=dose, threshold=threshold, steepness=steepness) >= 0.5
        for dose in (LOW_DOSE, 1.0, HIGH_DOSE)
    ])
    per_corner = (binary != target_t[None]).reshape(3, -1).sum(dim=1)
    positive = int(target_t.sum().item())
    blank = [int(row.sum().item()) == 0 and positive > 0 for row in binary]
    return {
        "L2_pixels": int(per_corner[1].item()),
        "L2_worst_dose_pixels": int(per_corner.max().item()),
        "band_pixels": int((binary.any(dim=0) != binary.all(dim=0)).sum().item()),
        "per_dose_L2_pixels": [int(x) for x in per_corner.tolist()],
        "positive_target_pixels": positive,
        "no_blank_positive_target_any_dose": not any(blank),
    }


def aggregate_metrics(per_layout: Sequence[dict]) -> dict:
    if not per_layout:
        raise ValueError("at least one layout metric is required")
    keys = ("L2_pixels", "L2_worst_dose_pixels", "band_pixels")
    return {key: float(np.mean([row[key] for row in per_layout])) for key in keys}


def checkpoint_rank(row: dict) -> tuple:
    """Frozen FIT-only ranking; L1 is the final metric tie-break."""
    metrics = row["new_fit_mean"]
    return (
        metrics["band_pixels"], metrics["L2_worst_dose_pixels"],
        metrics["L2_pixels"], row["selection_soft_objective"], row["l1_to_lp_anchor"],
        row["checkpoint_order"],
    )
