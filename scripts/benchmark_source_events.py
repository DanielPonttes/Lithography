"""Freeze, preflight, or run the prospective paired source-event/grid benchmark.

``freeze`` reads metadata and hashes only. ``preflight`` rechecks those pins
without deserializing FIT data or preparing optics. ``run`` consumes a distinct
one-use event marker before the original FIT loader, optics, or scoring.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import source_coverage as coverage
import source_event_search as event_search
from scripts import diagnose_source_pareto as pareto
from scripts import optimize_source_coverage as parent
from source_robustness import critical_corner_softcount_value


SCHEMA_VERSION = 1
OBJECTIVE_ID = "prospective_source_event_vs_grid_quality_time_v1"
REPEATS = 3
SEED_ORDER = (17, 29, 43, 71, 101)
SEGMENT_NAMES = ("schema8_initialization", "schema6_endpoint",
                 "schema7_endpoint", "schema8_endpoint")
GRID_DENOMINATOR = 256
GRID_PROPOSALS_PER_SEED = len(SEGMENT_NAMES) * (GRID_DENOMINATOR + 1)
PROTECTED_INCUMBENTS = 3
ATTEMPT_CAP_PER_ARM_SEED = GRID_PROPOSALS_PER_SEED + PROTECTED_INCUMBENTS
WALL_BUDGET_SECONDS = 600.0
PROGRESS_EVERY = 10
MATCHED_ATTEMPTS = (3, 259, 515, 771, 1027, 1031)
MATCHED_WALL_SECONDS = (60.0, 300.0, 600.0)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _exclusive_json(path: Path, payload: dict) -> None:
    path = path.expanduser()
    if os.path.lexists(path):
        raise FileExistsError("refusing to replace existing frozen plan/report: %s" % path)
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-%d-%s" % (os.getpid(), uuid.uuid4().hex))
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, AttributeError):
            pass
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_progress(path: Path, payload: dict) -> float:
    started = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return time.monotonic() - started


def _absolute_file(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError("%s must be an existing absolute file" % name)
    return path.resolve()


def _assert_outside_repo(path: Path, label: str) -> Path:
    path = path.expanduser().resolve(strict=False)
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        return path
    raise ValueError("%s must be outside the source repository" % label)


def _parent_preflight(coverage_plan: Path, coverage_sha: str,
                      previous_sha: str) -> tuple[dict, str, dict, list[dict]]:
    args = argparse.Namespace(plan_file=str(coverage_plan),
                              expected_plan_sha256=coverage_sha,
                              expected_previous_sha256=previous_sha,
                              device="cuda")
    plan, plan_sha, identity = parent._preflight(args)
    if plan.get("schema_version") != 8 or plan_sha != coverage.PINNED_PLAN_SHA256_V8:
        raise ValueError("source event benchmark requires the registered schema-8 parent plan")
    expected_inputs = {
        "dataset_sha256": plan["input_hashes"]["dataset_sha256"],
        "dataset_file_sha256": plan["input_hashes"]["dataset_sha256"],
        "diagnostic_file_sha256": plan["input_hashes"]["diagnostic_sha256"],
        "diagnostic_input": identity["diagnostic"]["input"],
        "basis_parity": identity["previous_weights"]["canonical_basis_parity"],
    }
    prerequisite = pareto.validate_prerequisite_manifest(
        identity["manifest_path"], coverage.PINNED_SOURCE_MANIFEST_SHA256,
        expected_inputs,
    )
    artifacts = pareto._merge_artifact_paths(prerequisite["artifact_paths"])
    if len(artifacts) != 14:
        raise ValueError("parent preflight did not verify exactly 14 lineage artifacts")
    pinned_artifacts = identity["previous_report"].get("preflight", {}).get(
        "prerequisite_manifest", {}).get("artifact_paths")
    if pareto._merge_artifact_paths(pinned_artifacts or []) != artifacts:
        raise ValueError("parent manifest artifact list differs from the pinned predecessor")
    return plan, plan_sha, identity, artifacts


def _vector_record(weights) -> dict:
    vector = event_search.validate_weights(weights)
    return {"weights": vector.tolist(), "source_count": int(vector.size),
            "weights_f64_sha256": event_search.array_sha256(vector, "<f8"),
            "weights_f32_sha256": event_search.array_sha256(vector, "<f4")}


def _validate_segment_inputs(segment_plan: dict, segment_plan_sha: str,
                             segment_report: dict, segment_report_sha: str,
                             selected_file: dict, selected_file_sha: str,
                             coverage_plan_sha: str) -> tuple[dict, dict]:
    if (segment_plan.get("objective_id") != "fit_only_fixed_source_segment_hard_metric_sweep_v1"
            or segment_plan.get("status") != "prospective_candidate_plan"):
        raise ValueError("source segment plan is not the registered frozen FIT-only plan")
    if (segment_plan.get("lineage", {}).get("schema8_parent_plan_sha256") != coverage_plan_sha
            or segment_plan.get("fixed_fit_layout_hashes") != coverage.PINNED_FIXED_FIT_LAYOUT_HASHES
            or segment_plan.get("protocol", {}).get("seed_order") != list(SEED_ORDER)
            or segment_plan.get("protocol", {}).get("alpha_grid") != {
                "numerator": "k", "denominator": GRID_DENOMINATOR,
                "k_inclusive": [0, GRID_DENOMINATOR],
            }):
        raise ValueError("segment plan lineage, new FIT identity, seed order, or grid differs")
    if segment_report.get("objective_id") != "fit_only_fixed_source_segment_hard_metric_sweep_v1":
        raise ValueError("segment report objective differs from fixed-source FIT-only sweep")
    if (segment_report.get("status") != "complete"
            or segment_report.get("plan_sha256") != segment_plan_sha
            or segment_report.get("calibration_status") != "closed by design"
            or segment_report.get("final3_status") != "never indexed or evaluated"):
        raise ValueError("segment report is incomplete or opened a forbidden split")
    selected = segment_report.get("selected_qualified_fit_point")
    if (not isinstance(selected, dict) or selected.get("slot_index") != 4755
            or selected.get("seed") != 101
            or not isinstance(selected.get("fit_metrics"), dict)
            or selected["fit_metrics"].get("qualified") is not True):
        raise ValueError("pinned best-known incumbent is not qualified slot 4755 / seed 101")
    weights = event_search.validate_weights(selected_file.get("weights"))
    if (selected_file.get("report_sha256") != segment_report_sha
            or selected_file.get("plan_sha256") != segment_plan_sha
            or selected_file.get("weights_f64_sha256") != event_search.array_sha256(weights, "<f8")
            or selected_file.get("weights_f32_sha256") != event_search.array_sha256(weights, "<f4")
            or selected.get("weights_f64_sha256") != event_search.array_sha256(weights, "<f8")
            or not np.array_equal(np.asarray(selected.get("weights"), dtype=np.float64), weights)):
        raise ValueError("selected source JSON does not match the report's best-known vector")
    per_seed = segment_plan.get("endpoints")
    if not isinstance(per_seed, dict) or set(per_seed) != {str(seed) for seed in SEED_ORDER}:
        raise ValueError("segment plan must pin all five seed endpoint vectors")
    normalized = {}
    for seed in SEED_ORDER:
        row = per_seed[str(seed)]
        vectors = {}
        for name, field, sha_field in (
                ("schema8_initialization", "initialization", "initialization_sha256"),
                ("schema6_endpoint", "schema6_endpoint", "schema6_endpoint_sha256"),
                ("schema7_endpoint", "schema7_endpoint", "schema7_endpoint_sha256"),
                ("schema8_endpoint", "schema8_endpoint", "schema8_endpoint_sha256")):
            vector = event_search.validate_weights(row[field])
            digest = event_search.array_sha256(vector, "<f8")
            if digest != row.get(sha_field):
                raise ValueError("seed %s %s source vector hash mismatch" % (seed, name))
            vectors[name] = _vector_record(vector)
        normalized[str(seed)] = vectors
    return normalized, _vector_record(weights)


def _code_paths() -> list[Path]:
    parent_paths = list(parent._source_paths())
    extension_paths = [
        ROOT / "source_event_search.py", ROOT / "scripts" / "diagnose_source_events.py",
        ROOT / "scripts" / "benchmark_source_events.py",
        ROOT / "tests" / "test_source_event_search.py",
        ROOT / "tests" / "test_source_event_benchmark.py",
        ROOT / "docs" / "SOURCE_EVENT_SEARCH.md",
    ]
    paths = list(dict.fromkeys(path.resolve() for path in parent_paths + extension_paths))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("prospective code pin files are missing: " + ", ".join(missing))
    return paths


def freeze(*, coverage_plan_file: Path, expected_coverage_plan_sha256: str,
           expected_previous_sha256: str, segment_plan_file: Path,
           expected_segment_plan_sha256: str, segment_report_file: Path,
           expected_segment_report_sha256: str, selected_weights_file: Path,
           output_plan_file: Path) -> tuple[Path, str]:
    output_plan_file = output_plan_file.expanduser()
    if os.path.lexists(output_plan_file):
        raise FileExistsError("frozen event plan already exists: %s" % output_plan_file)
    output_plan_file = _assert_outside_repo(output_plan_file, "event plan")
    plan, plan_sha, identity, artifacts = _parent_preflight(
        _absolute_file(coverage_plan_file, "coverage plan"),
        expected_coverage_plan_sha256, expected_previous_sha256,
    )
    segment_plan_file = _absolute_file(segment_plan_file, "segment plan")
    segment_report_file = _absolute_file(segment_report_file, "segment report")
    selected_weights_file = _absolute_file(selected_weights_file, "selected weights")
    segment_plan_raw = segment_plan_file.read_bytes()
    segment_report_raw = segment_report_file.read_bytes()
    selected_raw = selected_weights_file.read_bytes()
    segment_plan_sha = hashlib.sha256(segment_plan_raw).hexdigest()
    segment_report_sha = hashlib.sha256(segment_report_raw).hexdigest()
    selected_sha = hashlib.sha256(selected_raw).hexdigest()
    if segment_plan_sha != expected_segment_plan_sha256.lower():
        raise ValueError("segment plan does not match its expected SHA256")
    if segment_report_sha != expected_segment_report_sha256.lower():
        raise ValueError("segment report does not match its expected SHA256")
    segment_plan = json.loads(segment_plan_raw.decode("utf-8"))
    segment_report = json.loads(segment_report_raw.decode("utf-8"))
    selected_payload = json.loads(selected_raw.decode("utf-8"))
    vectors, best_known = _validate_segment_inputs(
        segment_plan, segment_plan_sha, segment_report, segment_report_sha,
        selected_payload, selected_sha, plan_sha,
    )
    reference = _vector_record(identity["previous_weights"]["reference_weights"])
    anchor = _vector_record(identity["previous_weights"]["lp_anchor"])
    if output_plan_file.resolve(strict=False).parent != Path(plan["prerequisite_manifest"]).resolve().parent:
        raise ValueError("event plan must be beside the prerequisite manifest so its attempt marker is colocated")
    input_paths = [coverage_plan_file.resolve(), segment_plan_file,
                   segment_report_file, selected_weights_file,
                   Path(plan["dataset_file"]).resolve(),
                   Path(plan["diagnostic_file"]).resolve(),
                   Path(plan["prerequisite_manifest"]).resolve()]
    input_paths.extend(Path(item["path"]).resolve() for item in artifacts)
    input_pins = {str(path): _sha(path) for path in sorted(set(input_paths), key=str)}
    code_pins = {str(path.relative_to(ROOT.resolve())): _sha(path)
                 for path in _code_paths()}
    source_identity = {
        "parent_source_identity": identity["source_identity"],
        "parent_lineage_artifact_count": len(artifacts),
        "parent_lineage_artifacts": artifacts,
    }
    payload = {
        "schema_version": SCHEMA_VERSION, "objective_id": OBJECTIVE_ID,
        "status": "frozen_prospective_quality_time_plan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "coverage_plan_file": str(coverage_plan_file.resolve()),
        "coverage_plan_sha256": plan_sha,
        "expected_previous_sha256": expected_previous_sha256.lower(),
        "segment_plan_file": str(segment_plan_file),
        "segment_plan_sha256": segment_plan_sha,
        "segment_report_file": str(segment_report_file),
        "segment_report_sha256": segment_report_sha,
        "selected_weights_file": str(selected_weights_file),
        "selected_weights_file_sha256": selected_sha,
        "source_vectors": {
            "reference": reference, "lp_anchor": anchor,
            "best_known_slot_4755_seed_101": best_known,
            "seed_endpoints": vectors,
        },
        "best_known_historical_fit_metrics": segment_report["selected_qualified_fit_point"]["fit_metrics"]["all_fit"]["per_layout"],
        "reference_new_fit_metrics": segment_report["reference_new_fit"],
        "fixed_fit_layout_hashes": coverage.PINNED_FIXED_FIT_LAYOUT_HASHES,
        "physical": plan["physical"],
        "input_sha256": input_pins, "code_sha256": code_pins,
        "source_identity": source_identity,
        "lineage_artifact_count": len(artifacts),
        "protocol": {
            "seed_order": list(SEED_ORDER), "segments_per_seed": list(SEGMENT_NAMES),
            "segment_start": "frozen seed-17 reference source vector",
            "event_candidates": "per-layout/dose affine crossings, segment endpoints and interval midpoints; exact float64 source duplicates deduplicated; when more than 257 proposals exist, a deterministic stratified subset of 257 is evaluated per segment. This is not an exhaustive threshold partition and may miss narrow feasible intervals.",
            "grid_candidates": "every k/256 point, k=0..256, on each of four segments per seed; duplicate slots retained and scored; cross-method caches disabled",
            "protected_incumbent_roles": ["reference", "that seed's pinned initialization", "best-known FIT-qualified slot 4755 seed 101"],
            "candidate_cap_per_arm_seed_including_protected": ATTEMPT_CAP_PER_ARM_SEED,
            "wall_clock_budget_seconds_per_arm_seed": WALL_BUDGET_SECONDS,
            "wall_budget_semantics": "10-minute arm budget includes proposal setup, all distinct protected incumbents, canonical scoring, original-host-thread golden audits, source guards, and progress writes; checked between candidates, so an in-flight canonical+golden audit completes and may overrun the cap; protected incumbents are always evaluated first",
            "protected_incumbents": "all distinct protected vectors scored first even if they overrun the wall budget; attempts count toward candidate cap",
            "hard_evaluator": "canonical source_coverage.hard_metrics float32 at doses .98/1/1.02 on all seven FIT layouts; primary score at one CPU thread, then exact per-layout hard-count audit at the original host torch thread count captured at run start for reference/best-known and every otherwise-qualified candidate; no score cache",
            "thread_count_stability": "reference and best-known vectors must match exactly between one thread and the recorded original host thread count or the run aborts; every otherwise-qualified proposal is audited the same way and rejected on any mismatch; restore the one-thread arm setting after each audit",
            "every_candidate_domain_audit": ["augmented guarded source domain", "full original nominal polytope", "float32 critical guards"],
            "qualification": "same hard gates as schema-8 FIT selection plus every source/domain guard",
            "rank": "source_coverage.checkpoint_rank using all-seven critical-corner beta-800 soft objective, LP-anchor L1, and stable candidate order",
            "paired_repeats": REPEATS,
            "paired_order": ["event_then_grid", "grid_then_event", "event_then_grid"],
            "fixed_seeds_are_paired_repeats_not_independent_groups": True,
            "shared_setup": "original and exactly three fixed new FIT bases are prepared once outside per-arm timers and reported separately; arm timers include proposal setup, anchor scoring, candidate scoring/audits and periodic report writes",
            "cache_policy": "no within-arm or cross-arm score cache; both methods use one CPU thread for primary canonical hard metrics and the same original-host-thread golden audit on protected baselines and otherwise-qualified candidates; prior resident-only 1.71x timing is not an algorithm speedup claim",
            "primary_quality_time_outcome": "time to improve the preserved best-known qualified checkpoint rank; this rank includes the soft objective, LP-anchor distance and stable-order tie-breaks, so a rank improvement alone is not a hard-PV or additional-pixel-coverage improvement",
            "separate_hard_pv_outcome": "time to a strictly better tuple of new-FIT hard band, worst-dose L2, and nominal L2 pixel counts than the preserved best-known checkpoint; report separately from checkpoint-rank improvement",
            "secondary_outcomes": ["quality and hard-PV counts at matched candidate attempt counts", "quality and hard-PV counts at 60/300/600-second wall checkpoints", "complete arm runtime"],
            "incomplete_arm": "remaining candidates marked not_attempted; timeout/interruption is not evidence of no qualified point",
            "calibration_status": "closed by design", "final3_status": "never indexed or evaluated",
        },
    }
    _exclusive_json(output_plan_file, payload)
    return output_plan_file.resolve(), _sha(output_plan_file)


def _load_json_with_sha(path: Path, expected: str, name: str) -> dict:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError("%s SHA256 differs from frozen event plan" % name)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("%s must contain a JSON object" % name)
    return value


def _validate_event_plan(plan: dict) -> None:
    protocol = plan.get("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("frozen event plan protocol must be an object")
    if (plan.get("schema_version") != SCHEMA_VERSION
            or plan.get("objective_id") != OBJECTIVE_ID
            or plan.get("status") != "frozen_prospective_quality_time_plan"
            or plan.get("fixed_fit_layout_hashes") != coverage.PINNED_FIXED_FIT_LAYOUT_HASHES
            or plan.get("lineage_artifact_count") != 14):
        raise ValueError("frozen event plan schema, objective, fixed FITs, or lineage differs")
    if (protocol.get("candidate_cap_per_arm_seed_including_protected")
            != ATTEMPT_CAP_PER_ARM_SEED
            or protocol.get("wall_clock_budget_seconds_per_arm_seed")
            != WALL_BUDGET_SECONDS
            or not isinstance(protocol.get("event_candidates"), str)
            or "not an exhaustive threshold partition" not in protocol.get("event_candidates", "")
            or not isinstance(protocol.get("thread_count_stability"), str)
            or "original host thread count" not in protocol.get("thread_count_stability", "")):
        raise ValueError("event plan budget differs from this runner")
    if not isinstance(plan.get("code_sha256"), dict) or not isinstance(plan.get("input_sha256"), dict):
        raise ValueError("frozen event plan lacks code/input pins")
    vectors = plan.get("source_vectors")
    required_vectors = {"reference", "lp_anchor", "best_known_slot_4755_seed_101"}
    if not isinstance(vectors, dict) or not required_vectors.issubset(vectors):
        raise ValueError("frozen event plan lacks the reference, LP anchor, or best-known source")
    for name in required_vectors:
        record = vectors[name]
        weights = event_search.validate_weights(record.get("weights"))
        if (record.get("source_count") != event_search.SOURCE_COUNT
                or record.get("weights_f64_sha256") != event_search.array_sha256(weights, "<f8")
                or record.get("weights_f32_sha256") != event_search.array_sha256(weights, "<f4")):
            raise ValueError("source-vector identity mismatch: %s" % name)
    endpoint_map = vectors.get("seed_endpoints")
    if not isinstance(endpoint_map, dict) or set(endpoint_map) != {str(seed) for seed in SEED_ORDER}:
        raise ValueError("event plan must freeze endpoints for all five seeds")
    for seed, named in endpoint_map.items():
        if not isinstance(named, dict) or set(named) != set(SEGMENT_NAMES):
            raise ValueError("seed %s must contain exactly four pinned segment endpoints" % seed)
        for segment_name, record in named.items():
            weights = event_search.validate_weights(record.get("weights"))
            if (record.get("source_count") != event_search.SOURCE_COUNT
                    or record.get("weights_f64_sha256") != event_search.array_sha256(weights, "<f8")
                    or record.get("weights_f32_sha256") != event_search.array_sha256(weights, "<f4")):
                raise ValueError("source endpoint identity mismatch: %s/%s" % (seed, segment_name))


def _check_pins(plan: dict, *, run_parent_preflight: bool) -> tuple[dict, dict]:
    _validate_event_plan(plan)
    for relative, expected in plan["code_sha256"].items():
        actual = _sha(ROOT / relative)
        if actual != expected:
            raise ValueError("pinned code hash mismatch: %s" % relative)
    for filename, expected in plan["input_sha256"].items():
        actual = _sha(filename)
        if actual != expected:
            raise ValueError("pinned input hash mismatch: %s" % filename)
    parent_plan, parent_sha, identity, artifacts = _parent_preflight(
        Path(plan["coverage_plan_file"]), plan["coverage_plan_sha256"],
        plan["expected_previous_sha256"],
    )
    if parent_sha != plan["coverage_plan_sha256"]:
        raise ValueError("parent coverage plan differs from frozen event plan")
    if (plan.get("physical") != parent_plan.get("physical")
            or plan.get("fixed_fit_layout_hashes") != parent_plan.get("fixed_fit_layout_hashes")):
        raise ValueError("frozen event physical/layout contract differs from parent coverage plan")
    actual_artifact_pins = sorted(artifacts, key=lambda row: row["path"])
    if actual_artifact_pins != sorted(
            plan["source_identity"]["parent_lineage_artifacts"], key=lambda row: row["path"]):
        raise ValueError("parent's verified 14 lineage artifact pins changed")
    if run_parent_preflight:
        return parent_plan, identity
    return parent_plan, identity


def preflight(event_plan_file: Path, expected_event_plan_sha256: str,
              expected_previous_sha256: str) -> dict:
    plan = _load_json_with_sha(event_plan_file, expected_event_plan_sha256, "event plan")
    _validate_event_plan(plan)
    if plan.get("expected_previous_sha256") != expected_previous_sha256.lower():
        raise ValueError("--expected-previous-sha256 differs from frozen event plan")
    parent_plan, identity = _check_pins(plan, run_parent_preflight=True)
    return {
        "status": "preflight_passed_no_fit_deserialization_no_optics_no_scoring_no_marker",
        "event_plan_sha256": expected_event_plan_sha256,
        "coverage_plan_sha256": plan["coverage_plan_sha256"],
        "code_pin_count": len(plan["code_sha256"]),
        "input_pin_count": len(plan["input_sha256"]),
        "lineage_artifact_count": plan["lineage_artifact_count"],
        "source_manifest_sha256": parent_plan["input_hashes"]["source_manifest_sha256"],
        "dataset_scope": "metadata hashes only; no torch.load or split indexing",
        "calibration_status": "closed by design", "final3_status": "never indexed or evaluated",
        "marker_created": False,
    }


def _metrics(rows: list[dict], basis32: dict[str, np.ndarray], weights: np.ndarray,
             physical: dict) -> dict:
    per_layout = []
    for row in rows:
        hard = coverage.hard_metrics(
            basis32[row["layout_id"]], row["target"], weights,
            threshold=physical["threshold"], steepness=physical["steepness"],
        )
        per_layout.append({"layout_id": row["layout_id"], **hard})
    old_count = len(coverage.ORIGINAL_FIT_LAYOUT_IDS)
    original = per_layout[:old_count]
    new = per_layout[old_count:]
    return {
        "per_layout": per_layout,
        "original_fit_mean": coverage.aggregate_metrics(original),
        "new_fit_mean": coverage.aggregate_metrics(new),
        "no_blank_positive_target_any_dose": all(
            row["no_blank_positive_target_any_dose"] for row in per_layout
        ),
    }


def _canonical_metric_signature(rows: list[dict]) -> list[dict]:
    keys = ("layout_id", "L2_pixels", "L2_worst_dose_pixels", "band_pixels",
            "per_dose_L2_pixels", "positive_target_pixels",
            "no_blank_positive_target_any_dose")
    if not isinstance(rows, list) or any(not isinstance(row, dict)
                                         or any(key not in row for key in keys)
                                         for row in rows):
        raise ValueError("pinned/reference hard metrics lack canonical per-layout fields")
    return [{key: row[key] for key in keys} for row in rows]


def _golden_thread_metric_audit(rows: list[dict], basis32: dict[str, np.ndarray],
                                weights: np.ndarray, physical: dict,
                                single_thread_metrics: dict,
                                golden_thread_count: int) -> dict:
    if (type(golden_thread_count) is not int or golden_thread_count < 1):
        raise ValueError("golden audit thread count must be a positive integer")
    restore_threads = torch.get_num_threads()
    started = time.monotonic()
    try:
        torch.set_num_threads(golden_thread_count)
        golden_metrics = _metrics(rows, basis32, weights, physical)
    finally:
        torch.set_num_threads(restore_threads)
    mismatches = []
    primary_rows = {row["layout_id"]: row for row in single_thread_metrics["per_layout"]}
    golden_rows = {row["layout_id"]: row for row in golden_metrics["per_layout"]}
    for layout_id in sorted(set(primary_rows) | set(golden_rows)):
        primary, golden = primary_rows.get(layout_id), golden_rows.get(layout_id)
        if primary != golden:
            mismatches.append({"layout_id": layout_id,
                               "single_thread_metrics": primary,
                               "golden_thread_metrics": golden})
    aggregate_mismatches = {
        key: {"single_thread": single_thread_metrics.get(key),
              "golden_thread_metrics": golden_metrics.get(key)}
        for key in ("original_fit_mean", "new_fit_mean", "no_blank_positive_target_any_dose")
        if single_thread_metrics.get(key) != golden_metrics.get(key)
    }
    return {
        "passed": not mismatches and not aggregate_mismatches,
        "primary_thread_count": restore_threads,
        "golden_thread_count": golden_thread_count,
        "per_layout_hard_counts_exact": not mismatches,
        "aggregate_hard_counts_exact": not aggregate_mismatches,
        "per_layout_mismatches": mismatches,
        "aggregate_mismatches": aggregate_mismatches,
        "elapsed_seconds": time.monotonic() - started,
        "comparison": "exact per-layout, per-dose, no-blank and aggregate canonical hard metrics",
    }


def _hard_gates(actual: dict, reference: dict) -> dict:
    old, ref_old = actual["original_fit_mean"], reference["original_fit_mean"]
    new, ref_new = actual["new_fit_mean"], reference["new_fit_mean"]
    return {
        "all_fit_no_blank": actual["no_blank_positive_target_any_dose"],
        "original_frozen_band_limit": old["band_pixels"] <= 234.5,
        "original_frozen_nominal_l2_limit": old["L2_pixels"] <= 0.0,
        "original_frozen_worst_l2_limit": old["L2_worst_dose_pixels"] <= 150.25,
        "original_band_nonincreasing": old["band_pixels"] <= ref_old["band_pixels"],
        "original_l2_nonincreasing": old["L2_pixels"] <= ref_old["L2_pixels"],
        "original_worst_l2_nonincreasing": old["L2_worst_dose_pixels"] <= ref_old["L2_worst_dose_pixels"],
        "new_band_strictly_lower": new["band_pixels"] < ref_new["band_pixels"],
        "new_l2_nonincreasing": new["L2_pixels"] <= ref_new["L2_pixels"],
        "new_worst_l2_nonincreasing": new["L2_worst_dose_pixels"] <= ref_new["L2_worst_dose_pixels"],
    }


def _validate_direct_basis_parity(parity: list[dict], expected_layout_ids: tuple[str, ...],
                                  label: str) -> None:
    if not isinstance(parity, list) or [row.get("layout_id") for row in parity
                                       if isinstance(row, dict)] != list(expected_layout_ids):
        raise ValueError("%s direct simulator/basis parity is missing or has a layout/order mismatch" % label)
    for row in parity:
        error = row.get("max_abs_error")
        if not isinstance(error, (int, float)) or not math.isfinite(float(error)) or error < 0:
            raise ValueError("%s direct simulator/basis parity has an invalid max_abs_error" % label)


def _rank_record(weights: np.ndarray, actual: dict, basis_torch: dict,
                 targets_torch: dict, anchor: np.ndarray, order: int) -> dict:
    return {
        "new_fit_mean": actual["new_fit_mean"],
        "selection_soft_objective": float(critical_corner_softcount_value(
            weights, basis_torch, targets_torch, coverage.BETAS[-1],
        )),
        "l1_to_lp_anchor": float(np.abs(weights - anchor).sum()),
        "checkpoint_order": int(order),
    }


def _hard_pv_tuple(actual: dict) -> tuple[float, float, float]:
    metrics = actual["new_fit_mean"]
    return (float(metrics["band_pixels"]), float(metrics["L2_worst_dose_pixels"]),
            float(metrics["L2_pixels"]))


def _hard_pv_improves(actual: dict, baseline: dict) -> bool:
    candidate = _hard_pv_tuple(actual)
    incumbent = _hard_pv_tuple(baseline)
    return all(value <= old for value, old in zip(candidate, incumbent)) and any(
        value < old for value, old in zip(candidate, incumbent)
    )


def _quality_snapshot(best_qualified: dict | None, best_rank: tuple | None,
                      preserved_best_metrics: dict) -> dict:
    if best_qualified is None:
        return {"best_qualified_candidate_id": None, "best_rank": None,
                "best_new_fit_mean": None, "hard_pv_counts": None,
                "hard_pv_improved_preserved_best": False}
    metrics = best_qualified["actual_hard_metrics"]
    return {
        "best_qualified_candidate_id": best_qualified["candidate_id"],
        "best_rank": list(best_rank) if best_rank is not None else None,
        "best_new_fit_mean": metrics["new_fit_mean"],
        "hard_pv_counts": {key: metrics["new_fit_mean"][key]
                            for key in ("band_pixels", "L2_worst_dose_pixels", "L2_pixels")},
        "hard_pv_improved_preserved_best": _hard_pv_improves(
            metrics, preserved_best_metrics),
    }


def _optional_difference(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _paired_time_summary(pairs: list[dict], event_key: str, grid_key: str) -> dict:
    differences = []
    paired_values = []
    for pair in pairs:
        event_seconds = pair["arms"]["event"].get(event_key)
        grid_seconds = pair["arms"]["grid"].get(grid_key)
        if event_seconds is None or grid_seconds is None:
            continue
        differences.append(float(event_seconds) - float(grid_seconds))
        paired_values.append({"seed": pair["seed"], "paired_repeat": pair["paired_repeat"],
                              "event_seconds": float(event_seconds),
                              "grid_seconds": float(grid_seconds),
                              "event_minus_grid_seconds": float(event_seconds) - float(grid_seconds)})
    return {
        "paired_observations_with_both_times": paired_values,
        "paired_observation_count": len(differences),
        "median_event_minus_grid_seconds": float(np.median(differences)) if differences else None,
        "descriptive_only_fixed_seeds_and_repeats_not_independent_groups": True,
    }


def _segment_events(seed: int, plan: dict, basis64: dict) -> tuple[list[dict], list[dict]]:
    reference = np.asarray(plan["source_vectors"]["reference"]["weights"], dtype=np.float64)
    seed_endpoints = plan["source_vectors"]["seed_endpoints"][str(seed)]
    layout_ids = list(coverage.ORIGINAL_FIT_LAYOUT_IDS + coverage.LAYOUT_IDS)
    candidates, audit = [], []
    for segment_name in SEGMENT_NAMES:
        endpoint = np.asarray(seed_endpoints[segment_name]["weights"], dtype=np.float64)
        start_intensities = {key: np.einsum("nhw,n->hw", basis64[key], reference,
                                             optimize=True) for key in layout_ids}
        end_intensities = {key: np.einsum("nhw,n->hw", basis64[key], endpoint,
                                           optimize=True) for key in layout_ids}
        segment_rows, segment_audit = event_search.segment_event_candidates(
            reference, endpoint, start_intensities, end_intensities,
            segment_id="seed-%s:%s" % (seed, segment_name),
            doses=event_search.DOSES, threshold=event_search.THRESHOLD,
        )
        full_count = len(segment_rows)
        quota = GRID_DENOMINATOR + 1
        if full_count > quota:
            selected_indices = np.linspace(0, full_count - 1, quota, dtype=np.int64)
            segment_rows = [segment_rows[index] for index in selected_indices]
        segment_audit.update({"full_event_proposal_count": full_count,
                              "selected_event_proposal_count": len(segment_rows),
                              "deterministic_stratification_cap": quota,
                              "truncated": full_count > quota})
        # A layout can contribute thousands of affine roots. Keep the report
        # bounded while preserving count and deterministic digest evidence for
        # the complete pre-stratification knot/proposal arrays.
        compact_layout_audits = []
        for layout_audit in segment_audit.pop("layout_events", []):
            knots = layout_audit.get("knots", [])
            proposals = layout_audit.get("proposal_alphas", [])
            digest_payload = {"knots": knots, "proposal_alphas": proposals}
            compact_layout_audits.append({
                "layout_id": layout_audit["layout_id"],
                "isolated_crossing_count": layout_audit.get("isolated_crossing_count"),
                "constant_pixels_on_threshold_count": layout_audit.get(
                    "constant_pixels_on_threshold_count"),
                "unique_knot_count": len(knots),
                "proposal_alpha_count": len(proposals),
                "knot_and_proposal_alphas_sha256": hashlib.sha256(
                    _canonical_json(digest_payload)).hexdigest(),
            })
        segment_audit["layout_event_summaries"] = compact_layout_audits
        candidates.extend(segment_rows)
        audit.append(segment_audit)
    return candidates, audit


def _grid_candidates(seed: int, plan: dict) -> list[dict]:
    reference = np.asarray(plan["source_vectors"]["reference"]["weights"], dtype=np.float64)
    seed_endpoints = plan["source_vectors"]["seed_endpoints"][str(seed)]
    rows = []
    order = 0
    for segment_name in SEGMENT_NAMES:
        endpoint = np.asarray(seed_endpoints[segment_name]["weights"], dtype=np.float64)
        for k in range(GRID_DENOMINATOR + 1):
            alpha = k / GRID_DENOMINATOR
            weights = event_search.interpolate_weights(reference, endpoint, alpha)
            rows.append({"candidate_id": "seed-%s:%s:grid-%03d" % (seed, segment_name, k),
                         "candidate_order": order, "roles": ["grid_slot"],
                         "weights": weights, "alpha": alpha,
                         "segment_id": "seed-%s:%s" % (seed, segment_name)})
            order += 1
    return rows


def _arm(method: str, seed: int, repeat: int, plan: dict, rows: list[dict],
         basis32: dict, basis64: dict, basis_torch: dict, targets_torch: dict,
         anchor: np.ndarray, reference_weights: np.ndarray, best_known: np.ndarray,
         domain, targets_by_id: dict, progress_callback,
         golden_thread_count: int, retain_candidate_records: bool = True) -> dict:
    started = time.monotonic()
    protected = event_search.incumbent_candidates(
        reference_weights=reference_weights,
        initial_incumbent_weights=plan["source_vectors"]["seed_endpoints"][str(seed)]["schema8_initialization"]["weights"],
        best_known_incumbent_weights=best_known, segment_candidates=[],
    )
    selected_rank = None
    selected_record = None
    best_known_rank = None
    best_qualified = None
    attempted = 0
    records = []
    anchor_elapsed = 0.0
    scoring_elapsed = 0.0
    golden_audit_elapsed = 0.0
    proposal_seconds = 0.0
    generation_audit = []
    reference_metrics = None
    time_to_first_qualified = None
    time_to_rank_improve = None
    rank_improvement_candidate = None
    rank_improvement_hard_pv_status = None
    time_to_hard_pv_improve = None
    hard_pv_improvement_candidate = None
    matched_attempts = {}
    matched_wall = {}
    report_io_seconds = 0.0
    stop_reason = None

    # Score protected reference, seed initialization, and preserved best first.
    for row in protected:
        candidate_started = time.monotonic()
        weights = event_search.validate_weights(row["weights"])
        actual = _metrics(rows, basis32, weights, plan["physical"])
        if "reference" in row["roles"]:
            reference_metrics = actual
            expected_reference_new = plan["reference_new_fit_metrics"].get("per_layout")
            if _canonical_metric_signature(
                    actual["per_layout"][len(coverage.ORIGINAL_FIT_LAYOUT_IDS):]) != \
                    _canonical_metric_signature(expected_reference_new):
                raise ValueError("canonical reference new-FIT counts differ from the pinned segment report")
        if "best_known_incumbent" in row["roles"]:
            expected = plan["best_known_historical_fit_metrics"]
            if (_canonical_metric_signature(actual["per_layout"])
                    != _canonical_metric_signature(expected)):
                raise ValueError("canonical best-known hard metrics differ from the pinned segment report")
        source_audit = {
            "guarded_domain": domain.verify(weights),
            "original_nominal_polytope": coverage.verify_original_nominal_polytope(
                [basis64[key] for key in coverage.ORIGINAL_FIT_LAYOUT_IDS],
                [targets_by_id[key] for key in coverage.ORIGINAL_FIT_LAYOUT_IDS], weights,
            ),
            "float32_critical_guards": coverage.audit_float32_critical_guards(
                basis32, targets_by_id, anchor, reference_weights, weights,
                coverage.ORIGINAL_FIT_LAYOUT_IDS, coverage.LAYOUT_IDS,
            ),
        }
        gates = _hard_gates(actual, reference_metrics) if reference_metrics is not None else {}
        gates.update({key + "_passed": value.get("passed") is True
                      for key, value in source_audit.items()})
        qualified = bool(gates and all(gates.values()))
        thread_audit = None
        must_audit_protected = bool(set(row["roles"]).intersection(
            {"reference", "best_known_incumbent"}))
        if must_audit_protected or qualified:
            thread_audit = _golden_thread_metric_audit(
                rows, basis32, weights, plan["physical"], actual, golden_thread_count,
            )
            golden_audit_elapsed += thread_audit["elapsed_seconds"]
            if must_audit_protected and not thread_audit["passed"]:
                progress_callback({"type": "protected_thread_parity_failure",
                                   "candidate_id": row["candidate_id"],
                                   "roles": row["roles"],
                                   "weights_f64_sha256": event_search.array_sha256(weights, "<f8"),
                                   "thread_count_audit": thread_audit})
                raise ValueError("protected %s hard counts differ between one thread and the recorded original host thread count" % row["candidate_id"])
            gates["cpu_thread_golden_metric_parity_passed"] = thread_audit["passed"]
            qualified = qualified and thread_audit["passed"]
        rank = None
        if qualified:
            rank = _rank_record(weights, actual, basis_torch, targets_torch, anchor, attempted)
            rank_tuple = coverage.checkpoint_rank(rank)
            if "best_known_incumbent" in row["roles"]:
                best_known_rank = rank_tuple
            if selected_rank is None or rank_tuple < selected_rank:
                selected_rank, selected_record = rank_tuple, row["candidate_id"]
                best_qualified = {"candidate_id": row["candidate_id"], "rank": list(rank_tuple),
                                  "actual_hard_metrics": actual}
            if time_to_first_qualified is None:
                time_to_first_qualified = time.monotonic() - started
        elapsed_candidate = time.monotonic() - candidate_started
        anchor_elapsed += elapsed_candidate
        scoring_elapsed += elapsed_candidate
        attempted += 1
        records.append({"method": method, "seed": seed, "paired_repeat": repeat,
                        "candidate_id": row["candidate_id"], "candidate_order": attempted - 1,
                        "roles": row["roles"], "weights": weights.tolist(),
                        "weights_f64_sha256": event_search.array_sha256(weights, "<f8"),
                        "weights_f32_sha256": event_search.array_sha256(weights, "<f4"),
                        "status": "attempted", "actual_hard_metrics": actual,
                        "hard_qualification_gates": gates, "hard_qualified": qualified,
                        "source_domain_audit": source_audit, "rank_record": rank,
                        "elapsed_seconds": elapsed_candidate,
                        "elapsed_since_arm_start_seconds": time.monotonic() - started,
                        "thread_count_audit": thread_audit})
        report_io_seconds += progress_callback({"type": "candidate_audit",
                                                "candidate_audit_record": records[-1]})
    if reference_metrics is None or best_known_rank is None:
        raise ValueError("protected reference or best-known incumbent failed to qualify")

    best_known_reference_row = next(row for row in records
                                    if "best_known_incumbent" in row["roles"])
    best_known_reference_rank = best_known_rank
    best_known_reference_metrics = best_known_reference_row["actual_hard_metrics"]
    for incumbent_row in records:
        if (incumbent_row["hard_qualified"]
                and "best_known_incumbent" not in incumbent_row["roles"]
                and _hard_pv_improves(incumbent_row["actual_hard_metrics"],
                                      best_known_reference_metrics)
                and time_to_hard_pv_improve is None):
            time_to_hard_pv_improve = incumbent_row["elapsed_since_arm_start_seconds"]
            hard_pv_improvement_candidate = incumbent_row["candidate_id"]
        if (incumbent_row["hard_qualified"]
                and tuple(coverage.checkpoint_rank(incumbent_row["rank_record"])) < best_known_rank
                and time_to_rank_improve is None):
            time_to_rank_improve = incumbent_row["elapsed_since_arm_start_seconds"]
            rank_improvement_candidate = incumbent_row["candidate_id"]
            rank_improvement_hard_pv_status = _hard_pv_improves(
                incumbent_row["actual_hard_metrics"], best_known_reference_metrics)

    if attempted in MATCHED_ATTEMPTS:
        matched_attempts[str(attempted)] = {
            **_quality_snapshot(best_qualified, selected_rank,
                                best_known_reference_row["actual_hard_metrics"]),
        }
    for wall_mark in MATCHED_WALL_SECONDS:
        if time.monotonic() - started >= wall_mark:
            matched_wall[str(int(wall_mark))] = {
                "observed_at_seconds": time.monotonic() - started,
                "attempted_candidates": attempted,
                **_quality_snapshot(best_qualified, selected_rank,
                                    best_known_reference_row["actual_hard_metrics"]),
            }
    method_setup_started = time.monotonic()
    proposal_setup_status = "complete"
    if time.monotonic() - started >= WALL_BUDGET_SECONDS:
        proposal_rows = []
        proposal_setup_status = "skipped_protected_incumbents_exceeded_wall_budget"
        stop_reason = "wall_clock_budget"
    elif method == "event":
        proposal_rows, generation_audit = _segment_events(seed, plan, basis64)
        merged_rows = event_search.incumbent_candidates(
            reference_weights=reference_weights,
            initial_incumbent_weights=plan["source_vectors"]["seed_endpoints"][str(seed)]["schema8_initialization"]["weights"],
            best_known_incumbent_weights=best_known,
            segment_candidates=proposal_rows,
        )
        anchor_rows_by_hash = {row["weights_f64_sha256"]: row for row in records}
        proposal_rows = []
        for row in merged_rows:
            roles = set(row.get("roles", ()))
            if roles.intersection({"reference", "initial_incumbent", "best_known_incumbent"}):
                anchor_row = anchor_rows_by_hash.get(row["weights_f64_sha256"])
                if anchor_row is not None:
                    anchor_row["roles"] = list(row.get("roles", anchor_row["roles"]))
                    anchor_row["aliases"] = list(row.get("aliases", ()))
                continue
            proposal_rows.append(row)
    elif method == "grid":
        proposal_rows = _grid_candidates(seed, plan)
    else:
        raise ValueError("method must be event or grid")
    proposal_seconds = time.monotonic() - method_setup_started
    for wall_mark in MATCHED_WALL_SECONDS:
        key = str(int(wall_mark))
        elapsed_setup = time.monotonic() - started
        if key not in matched_wall and elapsed_setup >= wall_mark:
            matched_wall[key] = {"observed_at_seconds": elapsed_setup,
                                 "attempted_candidates": attempted,
                                 **_quality_snapshot(best_qualified, selected_rank,
                                                     best_known_reference_metrics)}

    attempt_budget = ATTEMPT_CAP_PER_ARM_SEED
    if time.monotonic() - started >= WALL_BUDGET_SECONDS:
        stop_reason = "wall_clock_budget"
    for candidate in proposal_rows:
        if stop_reason is not None:
            break
        if attempted >= attempt_budget:
            stop_reason = "candidate_evaluation_budget"
            break
        if time.monotonic() - started >= WALL_BUDGET_SECONDS:
            stop_reason = "wall_clock_budget"
            break
        candidate_started = time.monotonic()
        weights = event_search.validate_weights(candidate["weights"])
        actual = _metrics(rows, basis32, weights, plan["physical"])
        source_audit = {
            "guarded_domain": domain.verify(weights),
            "original_nominal_polytope": coverage.verify_original_nominal_polytope(
                [basis64[key] for key in coverage.ORIGINAL_FIT_LAYOUT_IDS],
                [targets_by_id[key] for key in coverage.ORIGINAL_FIT_LAYOUT_IDS], weights,
            ),
            "float32_critical_guards": coverage.audit_float32_critical_guards(
                basis32, targets_by_id, anchor, reference_weights, weights,
                coverage.ORIGINAL_FIT_LAYOUT_IDS, coverage.LAYOUT_IDS,
            ),
        }
        gates = _hard_gates(actual, reference_metrics)
        gates.update({key + "_passed": value.get("passed") is True
                      for key, value in source_audit.items()})
        qualified = bool(all(gates.values()))
        thread_audit = None
        if qualified:
            thread_audit = _golden_thread_metric_audit(
                rows, basis32, weights, plan["physical"], actual, golden_thread_count,
            )
            golden_audit_elapsed += thread_audit["elapsed_seconds"]
            gates["cpu_thread_golden_metric_parity_passed"] = thread_audit["passed"]
            qualified = thread_audit["passed"]
        rank = None
        if qualified:
            rank = _rank_record(weights, actual, basis_torch, targets_torch, anchor, attempted)
            rank_tuple = coverage.checkpoint_rank(rank)
            if selected_rank is None or rank_tuple < selected_rank:
                selected_rank, selected_record = rank_tuple, candidate["candidate_id"]
                best_qualified = {"candidate_id": candidate["candidate_id"],
                                  "rank": list(rank_tuple), "actual_hard_metrics": actual}
            if time_to_first_qualified is None:
                time_to_first_qualified = time.monotonic() - started
            if rank_tuple < best_known_reference_rank and time_to_rank_improve is None:
                time_to_rank_improve = time.monotonic() - started
                rank_improvement_candidate = candidate["candidate_id"]
                rank_improvement_hard_pv_status = _hard_pv_improves(
                    actual, best_known_reference_metrics)
            if (_hard_pv_improves(actual, best_known_reference_metrics)
                    and time_to_hard_pv_improve is None):
                time_to_hard_pv_improve = time.monotonic() - started
                hard_pv_improvement_candidate = candidate["candidate_id"]
        elapsed_candidate = time.monotonic() - candidate_started
        scoring_elapsed += elapsed_candidate
        attempted += 1
        records.append({"method": method, "seed": seed, "paired_repeat": repeat,
                        "candidate_id": candidate["candidate_id"],
                        "candidate_order": attempted - 1,
                        "roles": candidate.get("roles", [method + "_proposal"]),
                        "segment_id": candidate.get("segment_id"),
                        "alpha": candidate.get("alpha"), "alphas": candidate.get("alphas"),
                        "weights": weights.tolist(),
                        "weights_f64_sha256": event_search.array_sha256(weights, "<f8"),
                        "weights_f32_sha256": event_search.array_sha256(weights, "<f4"),
                        "status": "attempted", "actual_hard_metrics": actual,
                        "hard_qualification_gates": gates, "hard_qualified": qualified,
                        "source_domain_audit": source_audit, "rank_record": rank,
                        "elapsed_seconds": elapsed_candidate,
                        "elapsed_since_arm_start_seconds": time.monotonic() - started,
                        "thread_count_audit": thread_audit})
        report_io_seconds += progress_callback({"type": "candidate_audit",
                                                "candidate_audit_record": records[-1]})
        if attempted in MATCHED_ATTEMPTS:
            matched_attempts[str(attempted)] = {
                **_quality_snapshot(best_qualified, selected_rank,
                                    best_known_reference_metrics),
            }
        elapsed = time.monotonic() - started
        for wall_mark in MATCHED_WALL_SECONDS:
            key = str(int(wall_mark))
            if key not in matched_wall and elapsed >= wall_mark:
                matched_wall[key] = {"observed_at_seconds": elapsed,
                                     "attempted_candidates": attempted,
                                     **_quality_snapshot(best_qualified, selected_rank,
                                                         best_known_reference_metrics)}
        if attempted % PROGRESS_EVERY == 0:
            report_io_seconds += progress_callback({
                "type": "arm_progress", "method": method, "seed": seed,
                "repeat": repeat, "attempted": attempted,
                "elapsed_seconds": elapsed, "best_qualified_candidate_id": selected_record,
            })
    elapsed_total = time.monotonic() - started
    for count in MATCHED_ATTEMPTS:
        matched_attempts.setdefault(str(count), {
            "status": "not_reached_candidate_budget_or_proposal_exhaustion",
            "attempted_candidates": attempted,
            **_quality_snapshot(best_qualified, selected_rank,
                                best_known_reference_metrics),
        })
    for wall_mark in MATCHED_WALL_SECONDS:
        matched_wall.setdefault(str(int(wall_mark)), {
            "status": "not_reached_before_arm_completion_or_budget",
            "attempted_candidates": attempted,
            "elapsed_seconds": elapsed_total,
            **_quality_snapshot(best_qualified, selected_rank,
                                best_known_reference_metrics),
        })
    result = {
        "method": method, "seed": seed, "paired_repeat": repeat,
        "status": "complete" if stop_reason is None else "incomplete_budget",
        "stop_reason": stop_reason, "candidate_cap": attempt_budget,
        "attempted_candidates": attempted,
        "proposal_count": len(proposal_rows),
        "proposal_setup_status": proposal_setup_status,
        "not_attempted_candidates": max(0, len(proposal_rows) - max(0, attempted - len(protected))),
        "not_attempted_interpretation": "not evidence of no qualified point",
        "reference_hard_metrics": reference_metrics,
        "preserved_best_known_metrics": best_known_reference_metrics,
        "preserved_best_known_rank": list(best_known_rank),
        "best_qualified_candidate_id": selected_record,
        "best_qualified_rank": list(selected_rank) if selected_rank is not None else None,
        "best_qualified_record": best_qualified,
        "time_to_first_qualified_seconds": time_to_first_qualified,
        "time_to_hard_pv_improve_preserved_best_seconds": time_to_hard_pv_improve,
        "hard_pv_improvement_candidate_id": hard_pv_improvement_candidate,
        "time_to_checkpoint_rank_improve_preserved_best_seconds": time_to_rank_improve,
        "checkpoint_rank_improvement_candidate_id": rank_improvement_candidate,
        "checkpoint_rank_improvement_hard_pv_improved_preserved_best": rank_improvement_hard_pv_status,
        "rank_improvement_may_be_soft_tie_break_only": (
            rank_improvement_hard_pv_status is False
            if rank_improvement_hard_pv_status is not None else None),
        "matched_candidate_attempt_checkpoints": matched_attempts,
        "matched_wall_clock_checkpoints": matched_wall,
        "timings": {"arm_wall_seconds_including_periodic_report_io": elapsed_total,
                    "proposal_generation_seconds": proposal_seconds,
                    "protected_incumbent_scoring_seconds": anchor_elapsed,
                    "candidate_scoring_and_audit_seconds": scoring_elapsed,
                    "golden_thread_audit_seconds": golden_audit_elapsed,
                    "periodic_report_io_seconds": report_io_seconds},
        "event_generation_audit": generation_audit if method == "event" else [],
    }
    # Full per-candidate payloads live in the append-only JSONL journal. Returning
    # them into paired_runs would make every periodic progress rewrite grow with
    # the complete search history and dominate a long run with duplicate I/O.
    if retain_candidate_records:
        result["candidate_records"] = records
    else:
        result["candidate_audit_journal_records"] = len(records)
    return result


def run(event_plan_file: Path, expected_event_plan_sha256: str,
        expected_previous_sha256: str, output_root: Path) -> Path:
    run_started = time.monotonic()
    output_root = _assert_outside_repo(output_root, "output root")
    plan = _load_json_with_sha(event_plan_file, expected_event_plan_sha256, "event plan")
    if plan["expected_previous_sha256"] != expected_previous_sha256.lower():
        raise ValueError("--expected-previous-sha256 differs from the frozen event plan")
    parent_plan, identity = _check_pins(plan, run_parent_preflight=True)
    golden_thread_count = torch.get_num_threads()
    if golden_thread_count < 2:
        raise ValueError("source event benchmark requires at least two original host torch threads for the independent golden hard-count audit")
    manifest_path = Path(parent_plan["prerequisite_manifest"])
    run_dir = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                             + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.json"
    candidate_journal_path = run_dir / "candidate_audit.jsonl"
    report_path = run_dir / "benchmark_report.json"
    report = {
        "schema_version": 1, "objective_id": OBJECTIVE_ID,
        "status": "preflight_passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "event_plan_sha256": expected_event_plan_sha256,
        "coverage_plan_sha256": plan["coverage_plan_sha256"],
        "segment_plan_sha256": plan["segment_plan_sha256"],
        "segment_report_sha256": plan["segment_report_sha256"],
        "selected_weights_file_sha256": plan["selected_weights_file_sha256"],
        "pinned_code_sha256": plan["code_sha256"],
        "pinned_input_sha256": plan["input_sha256"],
        "source_identity": plan["source_identity"],
        "lineage_artifact_count": plan["lineage_artifact_count"],
        "frozen_search_protocol": plan["protocol"],
        "event_candidate_sampling_is_exhaustive": False,
        "narrow_feasible_intervals_may_be_missed_by_stratified_event_subset": True,
        "quality_interpretation": {
            "checkpoint_rank_can_improve_on_soft_objective_or_lp_distance_only": True,
            "hard_pv_improvement_is_reported_separately_and_requires_pareto_nonincrease_plus_strict_improvement": True,
        },
        "calibration_status": "closed by design", "final3_status": "never indexed or evaluated",
        "fit_deserialized": False, "optics_prepared": False, "attempt_marker": None,
        "shared_setup_timings": {}, "paired_runs": [], "errors": [],
        "candidate_audit_journal": str(candidate_journal_path.resolve()),
    }
    marker = event_search.create_event_attempt_marker(manifest_path, expected_event_plan_sha256)
    report["attempt_marker"] = str(marker)
    report["event_attempt_consumed_before_fit_load_or_optics"] = True
    report["status"] = "event_attempt_consumed_before_fit_load_or_optics"
    report["calibration_status"] = "closed"
    report["progress_last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    io_sec = _atomic_progress(progress_path, report)

    journal = candidate_journal_path.open("xb", buffering=0)
    journal_count = 0

    def save_progress(event: dict) -> float:
        nonlocal journal_count
        io_started = time.monotonic()
        if event.get("candidate_audit_record") is not None:
            encoded = _canonical_json(event["candidate_audit_record"])
            journal.write(encoded)
            journal_count += 1
            if journal_count % PROGRESS_EVERY == 0:
                os.fsync(journal.fileno())
        if event.get("type") != "candidate_audit":
            report["latest_progress_event"] = event
            report["progress_last_updated_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_progress(progress_path, report)
        return time.monotonic() - io_started

    try:
        # The exclusive event marker above intentionally precedes this FIT-only loader.
        context_started = time.monotonic()
        context = parent._prepare_original_controls(
            parent_plan, plan["coverage_plan_sha256"],
            argparse.Namespace(plan_file=plan["coverage_plan_file"],
                               expected_previous_sha256=expected_previous_sha256,
                               device="cuda"), identity,
        )
        if set(context["fit"].layout_ids) != set(coverage.ORIGINAL_FIT_LAYOUT_IDS):
            raise ValueError("parent FIT loader did not return exactly the four pinned original FIT layouts")
        layout_specs = coverage.make_coverage_masks(parent_plan["physical"]["raster"])
        new_rows = parent._generate_new_targets(layout_specs, context["device"], parent_plan["physical"])
        novelty = parent._novelty_check(context["rows"], new_rows, parent_plan["new_fit_generation"])
        actual_fixed = [{key: row[key] for key in ("layout_id", "mask_sha256", "target_sha256")}
                        for row in novelty]
        if (actual_fixed != coverage.PINNED_FIXED_FIT_LAYOUT_HASHES
                or actual_fixed != parent_plan["fixed_fit_layout_hashes"]):
            raise ValueError("generated new FIT layouts differ from the exact three pinned identities")
        _sim, new32, new64, new_torch, new_targets, new_parity = parent._prepare_basis(
            new_rows, context["device"], parent_plan["physical"],
        )
        _validate_direct_basis_parity(
            context["fresh_fit_parity"], coverage.ORIGINAL_FIT_LAYOUT_IDS, "original FIT")
        _validate_direct_basis_parity(new_parity, coverage.LAYOUT_IDS, "new FIT")
        rows = context["rows"] + new_rows
        basis32 = dict(context["basis32"]); basis32.update(new32)
        basis64 = dict(context["basis64"]); basis64.update(new64)
        basis_torch = dict(context["basis_torch"]); basis_torch.update(new_torch)
        targets_torch = dict(context["targets_torch"]); targets_torch.update(new_targets)
        targets_by_id = {row["layout_id"]: row["target"] for row in rows}
        original_ids = list(coverage.ORIGINAL_FIT_LAYOUT_IDS)
        new_ids = list(coverage.LAYOUT_IDS)
        anchor = np.asarray(plan["source_vectors"]["lp_anchor"]["weights"], dtype=np.float64)
        reference_weights = np.asarray(plan["source_vectors"]["reference"]["weights"], dtype=np.float64)
        best_known = np.asarray(plan["source_vectors"]["best_known_slot_4755_seed_101"]["weights"], dtype=np.float64)
        all_bases64 = [basis64[key] for key in original_ids]
        domain, guard_summary = coverage.build_guarded_domain(
            [basis64[key] for key in original_ids], [targets_by_id[key] for key in original_ids], original_ids,
            [basis64[key] for key in new_ids], [targets_by_id[key] for key in new_ids], new_ids,
            anchor, reference_weights,
        )
        reference_audit = coverage.audit_float32_critical_guards(
            basis32, targets_by_id, anchor, reference_weights, reference_weights,
            original_ids, new_ids,
        )
        if not reference_audit["passed"] or not domain.verify(reference_weights)["passed"]:
            raise ValueError("seed-17 reference fails full augmented guards")
        for record_name, record in (("reference", plan["source_vectors"]["reference"]),
                                    ("LP anchor", plan["source_vectors"]["lp_anchor"]),
                                    ("best-known", plan["source_vectors"]["best_known_slot_4755_seed_101"])):
            vector = np.asarray(record["weights"], dtype=np.float64)
            if (event_search.array_sha256(vector, "<f8") != record["weights_f64_sha256"]
                    or event_search.array_sha256(vector, "<f4") != record["weights_f32_sha256"]):
                raise ValueError("frozen %s source-vector identity changed" % record_name)
        report["fit_deserialized"] = True
        report["optics_prepared"] = True
        shared_setup_elapsed = time.monotonic() - context_started
        report["shared_setup_timings"] = {
            "parent_original_controls_new_fit_generation_basis_and_guard_setup_seconds": shared_setup_elapsed,
            "new_fit_layouts": novelty, "new_fit_direct_basis_parity": new_parity,
            "original_fit_direct_basis_parity": context["fresh_fit_parity"],
            "augmented_guard_summary": guard_summary,
            "reference_augmented_critical_guard_audit": reference_audit,
            "shared_setup_excluded_from_per_arm_wall_budgets": True,
        }
        report["environment"] = {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(context["device"]),
            "gpu_capability": list(torch.cuda.get_device_capability(context["device"])),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(context["device"]).total_memory),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "original_torch_num_threads": golden_thread_count,
            "per_arm_torch_num_threads": 1,
            "golden_audit_torch_num_threads": golden_thread_count,
        }
        original_threads = golden_thread_count
        torch.set_num_threads(1)
        try:
            report["status"] = "paired_search_running"
            io_sec += save_progress({"type": "shared_setup_complete", "method_cpu_threads": 1})
            for repeat in range(REPEATS):
                order = ("event", "grid") if repeat % 2 == 0 else ("grid", "event")
                for seed in SEED_ORDER:
                    pair = {"paired_repeat": repeat, "seed": seed, "order": list(order), "arms": {}}
                    report["current_pair"] = pair
                    for method in order:
                        arm = _arm(
                            method, seed, repeat, plan, rows, basis32, basis64,
                            basis_torch, targets_torch, anchor, reference_weights,
                            best_known, domain, targets_by_id, save_progress,
                            golden_thread_count, retain_candidate_records=False,
                        )
                        pair["arms"][method] = arm
                        report["status"] = "paired_search_running"
                        io_sec += save_progress({"type": "arm_complete", "method": method,
                                                 "seed": seed, "repeat": repeat,
                                                 "status": arm["status"],
                                                 "attempted": arm["attempted_candidates"]})
                    pair["primary_paired_result"] = {
                        "checkpoint_rank_improvement": {
                            "event_seconds": pair["arms"]["event"]["time_to_checkpoint_rank_improve_preserved_best_seconds"],
                            "grid_seconds": pair["arms"]["grid"]["time_to_checkpoint_rank_improve_preserved_best_seconds"],
                            "event_minus_grid_seconds": _optional_difference(
                                pair["arms"]["event"]["time_to_checkpoint_rank_improve_preserved_best_seconds"],
                                pair["arms"]["grid"]["time_to_checkpoint_rank_improve_preserved_best_seconds"]),
                            "interpretation": "checkpoint rank can improve through a soft objective or LP-anchor tie-break without any hard-PV improvement",
                        },
                        "hard_pv_improvement": {
                            "event_seconds": pair["arms"]["event"]["time_to_hard_pv_improve_preserved_best_seconds"],
                            "grid_seconds": pair["arms"]["grid"]["time_to_hard_pv_improve_preserved_best_seconds"],
                            "event_minus_grid_seconds": _optional_difference(
                                pair["arms"]["event"]["time_to_hard_pv_improve_preserved_best_seconds"],
                                pair["arms"]["grid"]["time_to_hard_pv_improve_preserved_best_seconds"]),
                            "candidate_ids": {"event": pair["arms"]["event"]["hard_pv_improvement_candidate_id"],
                                              "grid": pair["arms"]["grid"]["hard_pv_improvement_candidate_id"]},
                        },
                    }
                    report["paired_runs"].append(pair)
                    report.pop("current_pair", None)
                    report["progress_last_updated_utc"] = datetime.now(timezone.utc).isoformat()
                    io_sec += _atomic_progress(progress_path, report)
        finally:
            torch.set_num_threads(original_threads)
        report["paired_time_summaries"] = {
            "checkpoint_rank_improvement": _paired_time_summary(
                report["paired_runs"],
                "time_to_checkpoint_rank_improve_preserved_best_seconds",
                "time_to_checkpoint_rank_improve_preserved_best_seconds"),
            "hard_pv_improvement": _paired_time_summary(
                report["paired_runs"],
                "time_to_hard_pv_improve_preserved_best_seconds",
                "time_to_hard_pv_improve_preserved_best_seconds"),
        }
        report["status"] = ("complete" if all(
            arm["status"] == "complete" for pair in report["paired_runs"]
            for arm in pair.get("arms", {}).values()
        ) and len(report["paired_runs"]) == REPEATS * len(SEED_ORDER)
            else "incomplete_budget")
        # Final parent preflight and every explicit code/input hash are rechecked.
        _check_pins(plan, run_parent_preflight=True)
        report["end_pin_recheck_passed"] = True
        report["calibration_status"] = "closed by design"
        report["final3_status"] = "never indexed or evaluated"
        report["result_scope"] = "paired quality-time comparison on five fixed seeds; not independent groups and not a speedup claim"
        report["timings"] = {"shared_setup_seconds": report["shared_setup_timings"].get(
            "parent_original_controls_new_fit_generation_basis_and_guard_setup_seconds"),
            "progress_report_io_seconds": io_sec,
            "run_total_seconds": None}
        report["timings"]["run_total_seconds"] = time.monotonic() - run_started
    except KeyboardInterrupt:
        report["status"] = "interrupted_incomplete"
        report["incomplete_interpretation"] = "interruption is not evidence of no qualified point"
        report["calibration_status"] = "closed by design"
        report["final3_status"] = "never indexed or evaluated"
        try:
            _check_pins(plan, run_parent_preflight=True)
            report["end_pin_recheck_passed"] = True
        except Exception as pin_exc:
            report["end_pin_recheck_error"] = "%s: %s" % (type(pin_exc).__name__, pin_exc)
        journal.flush()
        os.fsync(journal.fileno())
        journal.close()
        _atomic_progress(progress_path, report)
        raise
    except Exception as exc:
        report["status"] = "error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
        report["calibration_status"] = "closed by design"
        report["final3_status"] = "never indexed or evaluated"
        try:
            _check_pins(plan, run_parent_preflight=True)
            report["end_pin_recheck_passed"] = True
        except Exception as pin_exc:
            report["end_pin_recheck_error"] = "%s: %s" % (type(pin_exc).__name__, pin_exc)
        _atomic_progress(progress_path, report)
    if not journal.closed:
        journal.flush()
        os.fsync(journal.fileno())
        journal.close()
    report["candidate_audit_journal_records"] = journal_count
    report["progress_file"] = str(progress_path.resolve())
    report["report_file"] = str(report_path.resolve())
    _exclusive_json(report_path, report)
    return report_path.resolve()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("freeze", "preflight", "run"))
    p.add_argument("--event-plan-file")
    p.add_argument("--expected-event-plan-sha256")
    p.add_argument("--expected-previous-sha256")
    p.add_argument("--coverage-plan-file")
    p.add_argument("--expected-coverage-plan-sha256")
    p.add_argument("--segment-plan-file")
    p.add_argument("--expected-segment-plan-sha256")
    p.add_argument("--segment-report-file")
    p.add_argument("--expected-segment-report-sha256")
    p.add_argument("--selected-weights-file")
    p.add_argument("--output-plan-file")
    p.add_argument("--output-root")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.mode == "freeze":
            required = (args.coverage_plan_file, args.expected_coverage_plan_sha256,
                        args.expected_previous_sha256, args.segment_plan_file,
                        args.expected_segment_plan_sha256, args.segment_report_file,
                        args.expected_segment_report_sha256, args.selected_weights_file,
                        args.output_plan_file)
            if any(value is None for value in required):
                raise ValueError("freeze requires parent/segment pins, selected weights and --output-plan-file")
            path, digest = freeze(
                coverage_plan_file=Path(args.coverage_plan_file),
                expected_coverage_plan_sha256=args.expected_coverage_plan_sha256,
                expected_previous_sha256=args.expected_previous_sha256,
                segment_plan_file=Path(args.segment_plan_file),
                expected_segment_plan_sha256=args.expected_segment_plan_sha256,
                segment_report_file=Path(args.segment_report_file),
                expected_segment_report_sha256=args.expected_segment_report_sha256,
                selected_weights_file=Path(args.selected_weights_file),
                output_plan_file=Path(args.output_plan_file),
            )
            print(json.dumps({"status": "frozen_metadata_only", "event_plan_file": str(path),
                              "event_plan_sha256": digest, "FIT_deserialized": False,
                              "optics_prepared": False, "marker_created": False}, indent=2))
            return 0
        if not args.event_plan_file or not args.expected_event_plan_sha256 or not args.expected_previous_sha256:
            raise ValueError("preflight/run require event plan SHA256 and expected previous report SHA256")
        if args.mode == "preflight":
            print(json.dumps(preflight(Path(args.event_plan_file), args.expected_event_plan_sha256,
                                       args.expected_previous_sha256), indent=2))
            return 0
        if not args.output_root:
            raise ValueError("run requires --output-root")
        result_path = run(Path(args.event_plan_file), args.expected_event_plan_sha256,
                          args.expected_previous_sha256, Path(args.output_root))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print(json.dumps({"status": result["status"], "report_file": str(result_path),
                          "event_plan_sha256": result["event_plan_sha256"],
                          "paired_arm_count": sum(len(pair.get("arms", {}))
                                                   for pair in result.get("paired_runs", [])),
                          "calibration_status": result["calibration_status"],
                          "final3_status": result["final3_status"]}, indent=2))
        return 0 if result["status"] == "complete" else 3 if result["status"] == "incomplete_budget" else 2
    except KeyboardInterrupt:
        print("source event benchmark interrupted; progress report marks unattempted work", file=sys.stderr)
        return 130
    except Exception as exc:
        print("source event benchmark failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
