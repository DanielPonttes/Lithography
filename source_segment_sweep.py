"""Frozen, FIT-only hard-metric sweep over registered source-weight segments.

This diagnostic is intentionally separate from the coverage optimizer. It
scores a fixed grid of convex mixtures of already frozen vectors and never
selects or reads calibration/final layouts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Iterable, Sequence

import numpy as np

import source_coverage as coverage

OBJECTIVE_ID = "fit_only_fixed_source_segment_hard_metric_sweep_v1"
SCHEMA_VERSION = 1
PINNED_PLAN_SHA256 = "5c38f94ff79b7f4e2738664b211d4b0c1c0692f95ecb1ab56548736e50b1170d"
BASE_COMMIT = "4c36a6745bc38ca42c17b7e53aff49d48b54bbed"
PARENT_SCHEMA8_PLAN_SHA256 = coverage.PINNED_PLAN_SHA256_V8
PARENT_SCHEMA8_REPORT_SHA256 = "b3a2ba2d20a9a19584d35d15ebc5e942b9a78bbed85fb4f4fbe58df7a8d543c0"
SCHEMA6_REPORT_SHA256 = "1b830b964b9e8ccb84f4e8306e7419a2be221e687222ded57a4171894747289c"
SCHEMA7_REPORT_SHA256 = "ec993a4a2b5fb566816c05cad863c84ac0b525d0fc36d6dc3da3fe2d6798ece5"
SCHEMA8_REPORT_SHA256 = PARENT_SCHEMA8_REPORT_SHA256
REPORT_FILES = {
    "schema6": "schema6_a7f5104_report.json",
    "schema7": "schema7_a5c9d3b_report.json",
    "schema8": "schema8_db12cd8_report.json",
}
REPORT_REMOTE_PATHS = {
    "schema6": "/home/daniel/experiments/robust-source-quality-20261005-766872/feasible-start-runs/20261009T201624Z_380d04cf/coverage_report.json",
    "schema7": "/home/daniel/experiments/robust-source-quality-20261005-766872/new3-static-runs/20261009T205445Z_7bce872b/coverage_report.json",
    "schema8": "/home/daniel/experiments/robust-source-quality-20261005-766872/direct-pv-runs/20261009T213242Z_2af05efa/coverage_report.json",
}
REPORT_PLAN_PATHS = {
    "schema6": "/home/daniel/experiments/robust-source-quality-20261005-766872/feasible_starts_candidate_plan.json",
    "schema7": "/home/daniel/experiments/robust-source-quality-20261005-766872/new3_static_objective_candidate_plan.json",
    "schema8": "/home/daniel/experiments/robust-source-quality-20261005-766872/direct_pv_objective_candidate_plan.json",
}
REPORT_PLAN_SHAS = {
    "schema6": coverage.PINNED_PLAN_SHA256_V6,
    "schema7": coverage.PINNED_PLAN_SHA256_V7,
    "schema8": PARENT_SCHEMA8_PLAN_SHA256,
}
REPORT_SHAS = {
    "schema6": SCHEMA6_REPORT_SHA256,
    "schema7": SCHEMA7_REPORT_SHA256,
    "schema8": SCHEMA8_REPORT_SHA256,
}
SEEDS = (17, 29, 43, 71, 101)
SEGMENT_KINDS = ("schema8_initialization", "schema6_endpoint", "schema7_endpoint", "schema8_endpoint")
ALPHA_DENOMINATOR = 256
SLOTS_PER_SEGMENT = ALPHA_DENOMINATOR + 1
SEGMENTS_PER_SEED = len(SEGMENT_KINDS)
SLOTS_PER_SEED = SEGMENTS_PER_SEED * SLOTS_PER_SEGMENT
TOTAL_SLOTS = len(SEEDS) * SLOTS_PER_SEED
SETUP_LIMIT_SECONDS = 300.0
PER_SEED_LIMIT_SECONDS = 600.0
TOTAL_LIMIT_SECONDS = 3600.0
CHECKPOINT_INTERVAL = 128
SCREEN_CPU_THREADS = 1
F64_DISTANCE_TOLERANCE = 1e-9

PHYSICAL = {
    "source_grid": 9, "sigma_inner": 0.3, "sigma_outer": 0.9,
    "NA": 1.35, "wavelength_nm": 193, "pixel_nm": 4, "raster": 128,
    "doses": [0.98, 1.0, 1.02], "focus": 0, "threshold": 0.225,
    "steepness": 50,
}
LAYOUT_HASHES = coverage.PINNED_FIXED_FIT_LAYOUT_HASHES
PARENT_ROOT = "/home/daniel/experiments/robust-source-quality-20261005-766872"
PARENT_SCHEMA8_PLAN_PATH = PARENT_ROOT + "/direct_pv_objective_candidate_plan.json"
PARENT_SCHEMA8_REPORT_PATH = PARENT_ROOT + "/direct-pv-runs/20261009T213242Z_2af05efa/coverage_report.json"
PARENT_SCHEMA8_SUPERSEDES = {
    "schema6": {"status": "no_fit_qualified_checkpoint", "plan_path": REPORT_PLAN_PATHS["schema6"],
                "plan_sha256": REPORT_PLAN_SHAS["schema6"], "report_path": REPORT_REMOTE_PATHS["schema6"],
                "report_sha256": REPORT_SHAS["schema6"]},
    "schema7": {"status": "no_fit_qualified_checkpoint", "plan_path": REPORT_PLAN_PATHS["schema7"],
                "plan_sha256": REPORT_PLAN_SHAS["schema7"], "report_path": REPORT_REMOTE_PATHS["schema7"],
                "report_sha256": REPORT_SHAS["schema7"]},
}
SOURCE_FILES = (
    "source_segment_sweep.py", "scripts/diagnose_source_segments.py",
    "tests/test_source_segment_sweep.py", "docs/SOURCE_SEGMENT_SWEEP.md",
    "source_coverage.py", "source_pareto.py", "source_robustness.py",
    "source_training.py", "light_source.py",
    "scripts/optimize_source_coverage.py", "scripts/optimize_source_constrained.py",
    "scripts/optimize_source_robust_corners.py", "scripts/diagnose_source_pareto.py",
    "scripts/diagnose_source_feasibility.py", "scripts/run_protected_pvband_experiment.py",
    "tests/test_source_coverage.py", "docs/SOURCE_COVERAGE.md",
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_sha256(value: dict) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8"))


def sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return sha256_bytes(arr.tobytes(order="C"))


def weight_hash64(weights: Sequence[float]) -> str:
    return sha256_array(np.asarray(weights, dtype="<f8").reshape(-1))


def weight_hash32(weights: Sequence[float]) -> str:
    return sha256_array(np.asarray(weights, dtype="<f4").reshape(-1))


def module_hash_with_plan_pin(path: Path) -> str:
    raw = path.read_bytes()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    pattern = r'(?m)^PINNED_PLAN_SHA256 = "[0-9a-f]{64}"$'
    masked, count = re.subn(pattern,
                             'PINNED_PLAN_SHA256 = "' + "0" * 64 + '"', text)
    if count != 1:
        raise ValueError("source segment plan hash pin is not uniquely normalizable")
    return sha256_bytes(masked.encode("utf-8"))


def source_file_hashes(repo_root: Path) -> dict[str, str]:
    rows = {}
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise ValueError("missing source identity file: " + relative)
        rows[relative] = (module_hash_with_plan_pin(path) if relative == "source_segment_sweep.py"
                          else sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n")))
    return rows


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen lineage JSON must be an object: " + str(path))
    return value


def _endpoint_vectors(report: dict, key: str, expected_schema: int,
                      expected_objective: str, expected_plan_sha: str) -> dict[int, dict]:
    if (report.get("schema_version") != expected_schema
            or report.get("objective_id") != expected_objective
            or report.get("status") != "no_fit_qualified_checkpoint"
            or report.get("plan_sha256") != expected_plan_sha
            or report.get("coverage_attempt_consumed") is not True
            or report.get("calibration_status") != "closed"
            or report.get("final3_status") != "never indexed or evaluated"):
        raise ValueError(key + " report is not the pinned complete failed FIT-only attempt")
    seeds = report.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != len(SEEDS):
        raise ValueError(key + " report must contain exactly five seeds")
    result = {}
    for seed, row in zip(SEEDS, seeds):
        if (not isinstance(row, dict) or row.get("seed") != seed
                or row.get("status") != "complete"
                or row.get("steps_completed") != 204
                or row.get("qualified_checkpoint_count") != 0
                or row.get("selected") is not None):
            raise ValueError(key + " report seed state is not a complete unqualified FIT attempt")
        init = row.get("initialization")
        checks = row.get("checkpoints")
        if (not isinstance(init, dict) or init.get("status") != "feasible_start_found"
                or init.get("fallback") is not False or init.get("used_jitter") is not False
                or not isinstance(checks, list) or len(checks) != 11
                or any(not isinstance(check, dict) or check.get("qualified") is not False
                       for check in checks)):
            raise ValueError(key + " report lacks the registered feasible start or eleven failed checkpoints")
        endpoint = checks[-1]
        if endpoint.get("checkpoint_order") != 204 or endpoint.get("weights") is None:
            raise ValueError(key + " terminal endpoint is not checkpoint 204")
        init_w = np.asarray(init.get("weights"), dtype=np.float64).reshape(-1)
        end_w = np.asarray(endpoint["weights"], dtype=np.float64).reshape(-1)
        if init_w.size != 49 or end_w.size != 49 or not np.isfinite(init_w).all() or not np.isfinite(end_w).all():
            raise ValueError(key + " endpoint weight vector must be finite length-49 float64")
        if init.get("weights_sha256") != weight_hash64(init_w):
            raise ValueError(key + " initialization vector hash mismatch")
        if endpoint.get("weights_sha256") != weight_hash64(end_w):
            raise ValueError(key + " terminal vector hash mismatch")
        result[seed] = {
            "initialization": init_w.tolist(), "initialization_sha256": weight_hash64(init_w),
            "endpoint": end_w.tolist(), "endpoint_sha256": weight_hash64(end_w),
            "float32_endpoint_sha256": weight_hash32(end_w),
        }
    return result


def _load_and_pin_report(path: Path, expected_sha: str, key: str,
                         expected_schema: int, expected_objective: str,
                         expected_plan_sha: str) -> dict[int, dict]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise ValueError(key + " report bytes differ from the frozen SHA256")
    report = json.loads(raw.decode("utf-8"))
    if not isinstance(report, dict):
        raise ValueError(key + " report must contain an object")
    return _endpoint_vectors(report, key, expected_schema, expected_objective, expected_plan_sha)


def collect_frozen_endpoints(local_root: Path) -> dict[int, dict]:
    reports = {
        "schema6": _load_and_pin_report(
            local_root / REPORT_FILES["schema6"], REPORT_SHAS["schema6"], "schema6", 2,
            coverage.OBJECTIVE_ID, REPORT_PLAN_SHAS["schema6"]),
        "schema7": _load_and_pin_report(
            local_root / REPORT_FILES["schema7"], REPORT_SHAS["schema7"], "schema7", 3,
            coverage.OBJECTIVE_ID_V7, REPORT_PLAN_SHAS["schema7"]),
        "schema8": _load_and_pin_report(
            local_root / REPORT_FILES["schema8"], REPORT_SHAS["schema8"], "schema8", 4,
            coverage.OBJECTIVE_ID_V8, REPORT_PLAN_SHAS["schema8"]),
    }
    combined: dict[int, dict] = {}
    for seed in SEEDS:
        rows = {name: report[seed] for name, report in reports.items()}
        starts = {rows[name]["initialization_sha256"] for name in rows}
        if len(starts) != 1:
            raise ValueError("schema6/7/8 feasible starts differ for seed %d" % seed)
        combined[seed] = {
            "initialization": rows["schema8"]["initialization"],
            "initialization_sha256": rows["schema8"]["initialization_sha256"],
            "schema6_endpoint": rows["schema6"]["endpoint"],
            "schema6_endpoint_sha256": rows["schema6"]["endpoint_sha256"],
            "schema6_endpoint_float32_sha256": rows["schema6"]["float32_endpoint_sha256"],
            "schema7_endpoint": rows["schema7"]["endpoint"],
            "schema7_endpoint_sha256": rows["schema7"]["endpoint_sha256"],
            "schema7_endpoint_float32_sha256": rows["schema7"]["float32_endpoint_sha256"],
            "schema8_endpoint": rows["schema8"]["endpoint"],
            "schema8_endpoint_sha256": rows["schema8"]["endpoint_sha256"],
            "schema8_endpoint_float32_sha256": rows["schema8"]["float32_endpoint_sha256"],
        }
    return combined


def _fixed_protocol() -> dict:
    return {
        "seed_order": list(SEEDS),
        "segment_order": list(SEGMENT_KINDS),
        "alpha_grid": {"numerator": "k", "denominator": ALPHA_DENOMINATOR,
                       "k_inclusive": [0, ALPHA_DENOMINATOR]},
        "slot_count": TOTAL_SLOTS,
        "weight_rule": "float64 (1-alpha)*reference + alpha*endpoint; no projection, clipping, or renormalization",
        "reference": "the previously frozen FIT-only seed-17 Pareto reference weights",
        "endpoints": "each seed's schema-8 feasible start and terminal step-204 schema-6/7/8 vectors, pinned by report SHA and vector SHA",
        "candidate_order": "seed order, then the four listed segment types, then increasing k",
        "duplicate_policy": "retain and audit every slot; exact-float64 and float32 metric caches may reuse score results but never remove a slot",
        "hard_metrics": "existing full float32 raster hard_metrics at doses 0.98, 1.00, 1.02 for all four original and three fixed new FIT layouts",
        "cpu_thread_protocol": {
            "screen_threads": SCREEN_CPU_THREADS,
            "reference": "exact per-layout hard metric parity against original host thread count before grid scoring",
            "qualified": "every screen-qualified slot is fully re-evaluated without cache at original host thread count; only golden-qualified points enter ranks",
            "negative_interpretation": "a negative result concerns this fixed one-thread screening grid; it is not an exhaustive original-thread search or a continuous/global infeasibility proof",
            "restore": "original process thread setting restored on every exit",
        },
        "feasibility": ["full float64 augmented guarded source domain", "full original nominal q polytope",
                        "float32 critical guard audit", "full hard FIT qualification gates"],
        "selection_rank": "common existing all-seven critical-corner beta-800 checkpoint_rank; report a qualifying FIT point separately from five-seed qualification",
        "calibration": "closed by design for the entire diagnostic, including if a FIT candidate passes",
        "final3": "never indexed or evaluated",
        "adaptive_refinement": False,
    }


def make_candidate_plan(repo_root: Path, local_lineage_root: Path,
                        parent_manifest_path: str) -> dict:
    endpoints = collect_frozen_endpoints(local_lineage_root)
    source_hashes = source_file_hashes(repo_root)
    endpoint_sha = {
        str(seed): {field: endpoints[seed][field] for field in (
            "initialization_sha256", "schema6_endpoint_sha256", "schema7_endpoint_sha256",
            "schema8_endpoint_sha256", "schema6_endpoint_float32_sha256",
            "schema7_endpoint_float32_sha256", "schema8_endpoint_float32_sha256")}
        for seed in SEEDS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "prospective_candidate_plan",
        "objective_id": OBJECTIVE_ID,
        "base_commit": BASE_COMMIT,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lineage": {
            "schema8_parent_plan_path": PARENT_SCHEMA8_PLAN_PATH,
            "schema8_parent_plan_sha256": PARENT_SCHEMA8_PLAN_SHA256,
            "schema8_parent_report_path": PARENT_SCHEMA8_REPORT_PATH,
            "schema8_parent_report_sha256": PARENT_SCHEMA8_REPORT_SHA256,
            "schema6_report_path": REPORT_REMOTE_PATHS["schema6"],
            "schema6_report_sha256": REPORT_SHAS["schema6"],
            "schema7_report_path": REPORT_REMOTE_PATHS["schema7"],
            "schema7_report_sha256": REPORT_SHAS["schema7"],
            "schema6_plan_path": REPORT_PLAN_PATHS["schema6"],
            "schema6_plan_sha256": REPORT_PLAN_SHAS["schema6"],
            "schema7_plan_path": REPORT_PLAN_PATHS["schema7"],
            "schema7_plan_sha256": REPORT_PLAN_SHAS["schema7"],
            "previous_report_sha256": coverage.PINNED_PREVIOUS_REPORT_SHA256,
            "previous_plan_sha256": coverage.PINNED_PREVIOUS_PLAN_SHA256,
            "source_manifest_sha256": coverage.PINNED_SOURCE_MANIFEST_SHA256,
            "source_manifest_path": parent_manifest_path,
            "dataset_file": "/home/daniel/experiments/constrained-source-20260929-cb2e6c/previous-datasets.pt",
            "diagnostic_file": "/home/daniel/experiments/constrained-source-20260929-cb2e6c/diagnostic.json",
        },
        "input_hashes": {
            "dataset_sha256": "1fb6555fbf1dc4b4748f05f37d557977df5bbfd3b5b8abf5853c68a04716d1d0",
            "diagnostic_sha256": "cd236f09d6c0ad32398d3ec638f1d2433a2a8cc6b5060016c181c15358d8901b",
            "source_manifest_sha256": coverage.PINNED_SOURCE_MANIFEST_SHA256,
        },
        "fixed_fit_layout_hashes": LAYOUT_HASHES,
        "physical": PHYSICAL,
        "endpoints": {str(seed): endpoints[seed] for seed in SEEDS},
        "endpoint_vector_hashes": endpoint_sha,
        "protocol": _fixed_protocol(),
        "budgets": {
            "setup_seconds_max": SETUP_LIMIT_SECONDS,
            "per_seed_seconds_max": PER_SEED_LIMIT_SECONDS,
            "total_seconds_max": TOTAL_LIMIT_SECONDS,
            "progress_checkpoint_every_slots": CHECKPOINT_INTERVAL,
            "incomplete_timeout_interpretation": "incomplete work is not evidence of no qualifying point",
        },
        "scope": {
            "source_only": True, "fixed_masks": True, "number_of_source_weights": 49,
            "calibration_status": "closed by design", "final3_status": "never indexed or evaluated",
        },
        "source_identity": {
            "base_commit_ancestor": BASE_COMMIT,
            "text_hash_rule": "UTF-8 bytes with CRLF normalized to LF; only the segment module plan pin is zeroed",
            "files": source_hashes,
        },
        "completion": "all 5140 ordered slots across all five seeds, or explicit timeout/error with partial slot telemetry; no early stop on passing or failing candidates",
    }


def validate_plan(plan: dict, expected_sha256: str, actual_sha256: str) -> None:
    required = {
        "schema_version", "status", "objective_id", "base_commit", "created_utc",
        "lineage", "input_hashes", "fixed_fit_layout_hashes", "physical", "endpoints",
        "endpoint_vector_hashes", "protocol", "budgets", "scope", "source_identity", "completion",
    }
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError("segment sweep plan fields differ from the registered contract")
    if (plan["schema_version"] != SCHEMA_VERSION or plan["status"] != "prospective_candidate_plan"
            or plan["objective_id"] != OBJECTIVE_ID or plan["base_commit"] != BASE_COMMIT):
        raise ValueError("segment sweep plan schema/objective/base commit mismatch")
    if actual_sha256 != expected_sha256.lower() or actual_sha256 != PINNED_PLAN_SHA256:
        raise ValueError("segment sweep plan SHA256 does not match the frozen pin")
    if plan["lineage"].get("schema8_parent_plan_sha256") != PARENT_SCHEMA8_PLAN_SHA256:
        raise ValueError("segment sweep schema-8 plan lineage mismatch")
    expected_lineage_fields = {
        "schema8_parent_plan_path", "schema8_parent_plan_sha256", "schema8_parent_report_path",
        "schema8_parent_report_sha256", "schema6_report_path", "schema6_report_sha256",
        "schema7_report_path", "schema7_report_sha256", "schema6_plan_path", "schema6_plan_sha256",
        "schema7_plan_path", "schema7_plan_sha256", "previous_report_sha256", "previous_plan_sha256",
        "source_manifest_sha256", "source_manifest_path", "dataset_file", "diagnostic_file",
    }
    if set(plan["lineage"]) != expected_lineage_fields:
        raise ValueError("segment sweep lineage fields differ from the registered contract")
    expected_lineage = {
        "schema8_parent_report_sha256": PARENT_SCHEMA8_REPORT_SHA256,
        "schema6_report_sha256": SCHEMA6_REPORT_SHA256,
        "schema7_report_sha256": SCHEMA7_REPORT_SHA256,
        "schema6_plan_sha256": REPORT_PLAN_SHAS["schema6"],
        "schema7_plan_sha256": REPORT_PLAN_SHAS["schema7"],
        "previous_report_sha256": coverage.PINNED_PREVIOUS_REPORT_SHA256,
        "previous_plan_sha256": coverage.PINNED_PREVIOUS_PLAN_SHA256,
        "source_manifest_sha256": coverage.PINNED_SOURCE_MANIFEST_SHA256,
    }
    for field, expected in expected_lineage.items():
        if plan["lineage"].get(field) != expected:
            raise ValueError("segment sweep lineage pin mismatch: " + field)
    if plan["input_hashes"] != {
            "dataset_sha256": "1fb6555fbf1dc4b4748f05f37d557977df5bbfd3b5b8abf5853c68a04716d1d0",
            "diagnostic_sha256": "cd236f09d6c0ad32398d3ec638f1d2433a2a8cc6b5060016c181c15358d8901b",
            "source_manifest_sha256": coverage.PINNED_SOURCE_MANIFEST_SHA256}:
        raise ValueError("segment sweep input hashes differ from frozen lineage")
    if plan["fixed_fit_layout_hashes"] != LAYOUT_HASHES or plan["physical"] != PHYSICAL:
        raise ValueError("segment sweep fixed layouts or physical protocol changed")
    if plan["protocol"] != _fixed_protocol():
        raise ValueError("segment sweep grid, order, ranking, or split boundary changed")
    if plan["budgets"] != {
            "setup_seconds_max": SETUP_LIMIT_SECONDS, "per_seed_seconds_max": PER_SEED_LIMIT_SECONDS,
            "total_seconds_max": TOTAL_LIMIT_SECONDS, "progress_checkpoint_every_slots": CHECKPOINT_INTERVAL,
            "incomplete_timeout_interpretation": "incomplete work is not evidence of no qualifying point"}:
        raise ValueError("segment sweep budget differs from registered values")
    if plan["scope"] != {"source_only": True, "fixed_masks": True, "number_of_source_weights": 49,
                         "calibration_status": "closed by design", "final3_status": "never indexed or evaluated"}:
        raise ValueError("segment sweep scope differs from closed split protocol")
    if plan["source_identity"].get("base_commit_ancestor") != BASE_COMMIT:
        raise ValueError("segment sweep source ancestor pin mismatch")
    if (set(plan["source_identity"]) != {"base_commit_ancestor", "files", "text_hash_rule"}
            or plan["source_identity"]["text_hash_rule"] != "UTF-8 bytes with CRLF normalized to LF; only the segment module plan pin is zeroed"):
        raise ValueError("segment sweep source identity fields differ")
    if set(plan["source_identity"].get("files", {})) != set(SOURCE_FILES):
        raise ValueError("segment sweep source identity file set differs from frozen bundle")
    if not isinstance(plan["endpoints"], dict) or not isinstance(plan["endpoint_vector_hashes"], dict):
        raise ValueError("segment sweep endpoint table is malformed")
    if set(plan["endpoints"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("segment sweep endpoint seeds/order differ")
    for seed in SEEDS:
        row = plan["endpoints"][str(seed)]
        hashes = plan["endpoint_vector_hashes"][str(seed)]
        expected_endpoint_fields = {
            "initialization", "initialization_sha256", "schema6_endpoint", "schema6_endpoint_sha256",
            "schema6_endpoint_float32_sha256", "schema7_endpoint", "schema7_endpoint_sha256",
            "schema7_endpoint_float32_sha256", "schema8_endpoint", "schema8_endpoint_sha256",
            "schema8_endpoint_float32_sha256",
        }
        if set(row) != expected_endpoint_fields or set(hashes) != {
                "initialization_sha256", "schema6_endpoint_sha256", "schema7_endpoint_sha256",
                "schema8_endpoint_sha256", "schema6_endpoint_float32_sha256",
                "schema7_endpoint_float32_sha256", "schema8_endpoint_float32_sha256"}:
            raise ValueError("segment sweep endpoint record fields differ from the registered contract")
        for name in ("initialization", "schema6_endpoint", "schema7_endpoint", "schema8_endpoint"):
            vector = np.asarray(row.get(name), dtype=np.float64).reshape(-1)
            hash_field = name + "_sha256"
            if vector.size != 49 or not np.isfinite(vector).all() or row.get(hash_field) != weight_hash64(vector):
                raise ValueError("segment sweep endpoint vector/hash mismatch: %s/%s" % (seed, name))
            if hashes.get(hash_field) != row[hash_field]:
                raise ValueError("segment sweep endpoint hash table mismatch: %s/%s" % (seed, name))
        for field in ("schema6_endpoint", "schema7_endpoint", "schema8_endpoint"):
            f32_field = field + "_float32_sha256"
            if row.get(f32_field) != weight_hash32(row[field]) or hashes.get(f32_field) != row[f32_field]:
                raise ValueError("segment sweep float32 endpoint hash mismatch: %s/%s" % (seed, field))
        if row.get("initialization_sha256") != weight_hash64(row["initialization"]):
            raise ValueError("segment sweep initialization hash mismatch: %s" % seed)


def iter_slots(plan: dict, reference_weights: Sequence[float]) -> Iterable[dict]:
    reference = np.asarray(reference_weights, dtype=np.float64).reshape(-1)
    if reference.size != 49 or not np.isfinite(reference).all():
        raise ValueError("reference weights must be finite length 49")
    for seed in SEEDS:
        endpoints = plan["endpoints"][str(seed)]
        for segment_index, (segment_id, field) in enumerate(zip(SEGMENT_KINDS,
                ("initialization", "schema6_endpoint", "schema7_endpoint", "schema8_endpoint"))):
            endpoint = np.asarray(endpoints[field], dtype=np.float64).reshape(-1)
            for k in range(ALPHA_DENOMINATOR + 1):
                alpha = k / float(ALPHA_DENOMINATOR)
                weights = (1.0 - alpha) * reference + alpha * endpoint
                slot_index = (seed_index(seed) * SLOTS_PER_SEED
                              + segment_index * SLOTS_PER_SEGMENT + k)
                yield {
                    "slot_index": slot_index, "seed": seed, "segment_id": segment_id,
                    "segment_order": segment_index, "alpha_numerator": k,
                    "alpha_denominator": ALPHA_DENOMINATOR, "alpha": alpha,
                    "weights": weights,
                    "weights_f64_sha256": weight_hash64(weights),
                    "weights_f32_sha256": weight_hash32(weights),
                }


def seed_index(seed: int) -> int:
    try:
        return SEEDS.index(int(seed))
    except ValueError as exc:
        raise ValueError("seed is not registered") from exc


def audit_float64_candidate(weights: Sequence[float], reference: Sequence[float],
                            anchor: Sequence[float], original_bases: Sequence[np.ndarray],
                            original_targets: Sequence[np.ndarray], domain) -> dict:
    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if vector.size != 49 or not np.isfinite(vector).all():
        return {"passed": False, "reason": "invalid_vector"}
    simplex = {"passed": bool(vector.min(initial=0.0) >= -1e-9
                               and abs(float(vector.sum()) - 1.0) <= 1e-9),
               "flux_sum": float(vector.sum()), "minimum_weight": float(vector.min(initial=0.0))}
    nominal = coverage.verify_original_nominal_polytope(original_bases, original_targets, vector)
    guarded = domain.verify(vector)
    return {"passed": bool(simplex["passed"] and nominal["passed"] and guarded["passed"]),
            "simplex": simplex, "original_nominal_polytope": nominal, "guard_domain": guarded}


def source_identity(repo_root: Path) -> dict:
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                            check=True, capture_output=True, text=True).stdout
    if status.strip():
        raise ValueError("segment sweep requires a clean committed source tree")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                          check=True, capture_output=True, text=True).stdout.strip()
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=repo_root,
                              capture_output=True)
    if ancestor.returncode != 0:
        raise ValueError("segment sweep code is not descended from its frozen base commit")
    actual = source_file_hashes(repo_root)
    return {"head": head, "files": actual}


def validate_source_identity(repo_root: Path, expected_files: dict) -> dict:
    actual = source_identity(repo_root)
    if actual["files"] != expected_files:
        raise ValueError("segment sweep source hashes differ from the frozen plan")
    return actual


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
