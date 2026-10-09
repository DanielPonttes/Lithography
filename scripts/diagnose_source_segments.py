"""Run the prospective fixed-grid FIT-only source-segment diagnostic."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import uuid
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import source_coverage as coverage
import source_segment_sweep as sweep
from scripts import optimize_source_coverage as parent


def _load_plan(path: Path, expected_sha: str) -> tuple[dict, str]:
    raw = path.read_bytes()
    actual = sweep.sha256_bytes(raw)
    plan = json.loads(raw.decode("utf-8"))
    sweep.validate_plan(plan, expected_sha, actual)
    return plan, actual


def _validate_frozen_reports(plan: dict) -> tuple[dict[int, dict], dict]:
    endpoints_by_schema = {}
    reports = {}
    for key, schema, objective, plan_sha in (
        ("schema6", 2, coverage.OBJECTIVE_ID, sweep.REPORT_PLAN_SHAS["schema6"]),
        ("schema7", 3, coverage.OBJECTIVE_ID_V7, sweep.REPORT_PLAN_SHAS["schema7"]),
        ("schema8", 4, coverage.OBJECTIVE_ID_V8, sweep.REPORT_PLAN_SHAS["schema8"]),
    ):
        lineage = plan["lineage"]
        path_field, sha_field = key + "_report_path", key + "_report_sha256"
        path_value = (lineage["schema8_parent_report_path"] if key == "schema8"
                      else lineage[path_field])
        expected_sha = (lineage["schema8_parent_report_sha256"] if key == "schema8"
                        else lineage[sha_field])
        path = Path(path_value)
        raw = path.read_bytes()
        if sweep.sha256_bytes(raw) != expected_sha:
            raise ValueError(key + " report changed from its frozen SHA256")
        report = json.loads(raw.decode("utf-8"))
        if not isinstance(report, dict):
            raise ValueError(key + " report is not a JSON object")
        endpoints_by_schema[key] = sweep._endpoint_vectors(
            report, key, schema, objective, plan_sha,
        )
        reports[key] = report
    for seed in sweep.SEEDS:
        actual_start_hashes = {
            endpoints_by_schema[key][seed]["initialization_sha256"]
            for key in ("schema6", "schema7", "schema8")
        }
        if len(actual_start_hashes) != 1:
            raise ValueError("pinned schema 6/7/8 feasible starts no longer agree")
        frozen = plan["endpoints"][str(seed)]
        for schema, field in (("schema8", "initialization"), ("schema6", "schema6_endpoint"),
                              ("schema7", "schema7_endpoint"), ("schema8", "schema8_endpoint")):
            source_field = "initialization" if field == "initialization" else "endpoint"
            actual = endpoints_by_schema[schema][seed][source_field]
            expected = np.asarray(frozen[field], dtype=np.float64)
            if not np.array_equal(actual, expected):
                raise ValueError("frozen segment endpoint differs from report: %d/%s" % (seed, field))
    return plan["endpoints"], reports


def _parent_identity(plan: dict) -> tuple[dict, dict, str]:
    lineage = plan["lineage"]
    parent_path = Path(lineage["schema8_parent_plan_path"]).resolve()
    raw = parent_path.read_bytes()
    actual = sweep.sha256_bytes(raw)
    if actual != lineage["schema8_parent_plan_sha256"]:
        raise ValueError("parent schema-8 plan hash mismatch")
    parent_plan = json.loads(raw.decode("utf-8"))
    coverage.validate_plan_payload(parent_plan)
    if parent_plan.get("schema_version") != 8:
        raise ValueError("segment sweep parent is not the frozen schema-8 source plan")
    if (parent_plan.get("dataset_file") is None
            or parent_plan.get("prerequisite_manifest") != lineage["source_manifest_path"]
            or parent_plan.get("input_hashes", {}).get("source_manifest_sha256")
            != plan["input_hashes"]["source_manifest_sha256"]):
        raise ValueError("segment sweep parent plan input lineage mismatch")
    identity = parent._validate_identities(
        parent_path, actual, coverage.PINNED_PREVIOUS_REPORT_SHA256, parent_plan,
    )
    artifacts = parent.pareto._merge_artifact_paths(
        identity["previous_report"].get("preflight", {}).get(
            "prerequisite_manifest", {}).get("artifact_paths") or []
    )
    if len(artifacts) != 14:
        raise ValueError("segment lineage must contain exactly 14 frozen artifacts")
    identity["lineage_artifacts"] = artifacts
    parent._check_lineage_artifacts(identity)
    return parent_plan, identity, str(parent_path)


def _preflight(plan_path: Path, expected_sha: str) -> tuple[dict, str, dict, dict, str, dict]:
    plan, plan_sha = _load_plan(plan_path, expected_sha)
    source = sweep.validate_source_identity(ROOT, plan["source_identity"]["files"])
    endpoints, frozen_reports = _validate_frozen_reports(plan)
    parent_plan, parent_identity, parent_plan_path = _parent_identity(plan)
    return plan, plan_sha, parent_plan, parent_identity, parent_plan_path, {
        "source_identity": source, "endpoints": endpoints, "frozen_reports": frozen_reports,
    }


def _verify_reference_fit(reference_fit: dict, expected: dict) -> None:
    actual_rows = reference_fit["per_layout"]
    expected_rows = expected.get("per_layout") if isinstance(expected, dict) else None
    if not isinstance(expected_rows, list) or len(expected_rows) != len(actual_rows):
        raise ValueError("schema-8 report lacks its frozen reference FIT hard metrics")
    for actual, frozen in zip(actual_rows, expected_rows):
        for key in ("layout_id", "L2_pixels", "L2_worst_dose_pixels", "band_pixels",
                    "per_dose_L2_pixels", "no_blank_positive_target_any_dose"):
            if actual.get(key) != frozen.get(key):
                raise ValueError("recomputed reference hard metric differs: %s/%s" % (actual.get("layout_id"), key))


def _candidate_score(slot: dict, context: dict, ref_new_mean: dict,
                     anchor: np.ndarray, reference: np.ndarray, original_poly, domain,
                     physical_cache: dict | None = None) -> dict:
    weights = slot["weights"]
    audit = sweep.audit_float64_candidate(
        weights, reference, anchor,
        [context["basis64"][row["layout_id"]] for row in context["original_rows"]],
        [row["target"] for row in context["original_rows"]], domain,
    )
    # Only float32 physical outputs are reusable. Float64 feasibility, source
    # distances, the beta-800 objective and rank always belong to this slot.
    key = (torch.get_num_threads(), slot["weights_f32_sha256"])
    physical_fields = ("all_fit", "original_fit_mean", "new_fit_mean",
                       "no_blank_positive_target_any_dose", "float32_critical_guard_audit",
                       "original_fit_gate_passed")
    cached = physical_cache.get(key) if physical_cache is not None else None
    if cached is None:
        metric = parent._checkpoint_metrics(
            context["all_rows"], context["original_rows"], context["new_rows"],
            context["basis32"], weights, context["basis_torch"], context["targets_torch"],
            800.0, anchor, reference, context["physical"], slot["slot_index"],
            training_objective_kind=None,
        )
        if physical_cache is not None:
            physical_cache[key] = {name: copy.deepcopy(metric[name]) for name in physical_fields}
    else:
        metric = copy.deepcopy(cached)
        objective = parent.critical_corner_softcount_value(
            weights, context["basis_torch"], context["targets_torch"], 800.0,
        )
        metric.update(checkpoint_order=int(slot["slot_index"]), beta=800.0,
                      selection_soft_objective=float(objective), selection_soft_objective_beta=800.0,
                      training_soft_objective=float(objective), weights=weights.tolist(),
                      l1_to_lp_anchor=float(np.abs(weights - anchor).sum()),
                      l1_to_reference=float(np.abs(weights - reference).sum()))
    metric = parent._fit_checkpoint_qualified(
        metric, ref_new_mean, audit["original_nominal_polytope"], domain, weights,
    )
    metric["float64_candidate_audit"] = audit
    metric["original_nominal_polytope"] = audit["original_nominal_polytope"]
    metric["guard_domain"] = audit["guard_domain"]
    metric["qualified"] = bool(metric["qualified"] and audit["passed"])
    metric["physical_cache_hit"] = cached is not None
    return metric


def _golden_score(slot, context, ref_mean, anchor, reference, original_poly, domain, host_threads):
    screening_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(host_threads)
        return _candidate_score(slot, context, ref_mean, anchor, reference, original_poly, domain)
    finally:
        torch.set_num_threads(screening_threads)


def _slot_record(slot: dict) -> dict:
    return {
        "slot_index": slot["slot_index"], "seed": slot["seed"],
        "segment_id": slot["segment_id"], "segment_order": slot["segment_order"],
        "alpha_numerator": slot["alpha_numerator"],
        "alpha_denominator": slot["alpha_denominator"], "alpha": slot["alpha"],
        "weights": slot["weights"].tolist(),
        "weights_f64_sha256": slot["weights_f64_sha256"],
        "weights_f32_sha256": slot["weights_f32_sha256"],
    }


def _record_unattempted(slot: dict, reason: str) -> dict:
    row = _slot_record(slot)
    row.update({"status": "not_attempted", "reason": reason,
                "float64_audit": None, "hard_metrics": None, "qualified": False})
    return row


def _recheck(plan_path: Path, plan_sha: str, plan: dict, parent_plan_path: Path,
             parent_plan_sha: str) -> None:
    raw = plan_path.read_bytes()
    if sweep.sha256_bytes(raw) != plan_sha:
        raise ValueError("segment sweep plan changed during run")
    sweep.validate_source_identity(ROOT, plan["source_identity"]["files"])
    if coverage.sha256_file(parent_plan_path) != parent_plan_sha:
        raise ValueError("schema-8 parent plan changed during run")
    for key, path_value in (("dataset_sha256", plan["lineage"].get("dataset_file")),
                            ("diagnostic_sha256", plan["lineage"].get("diagnostic_file")),
                            ("source_manifest_sha256", plan["lineage"]["source_manifest_path"])):
        path = Path(path_value) if path_value else None
        if path is None or coverage.sha256_file(path) != plan["input_hashes"][key]:
            raise ValueError("pinned input changed during run: " + key)
    # Includes all 14 manifest artifacts and schema-6/7/8 plans, not just
    # the manifest JSON or the parent plan. Metadata only, no FIT loading.
    _parent_identity(plan)
    _validate_frozen_reports(plan)


def _seal_incomplete(report: dict, all_slots: list[dict], reason: str) -> None:
    for row in report["seeds"]:
        if row.get("status") == "running":
            row.update(status="error" if "error" in reason else "timeout", reason=reason)
    present = {row["slot_index"] for row in report["slot_records"]}
    report["slot_records"].extend(
        _record_unattempted(slot, reason) for slot in all_slots if slot["slot_index"] not in present
    )
    represented = {row["seed"] for row in report["seeds"]}
    for seed in sweep.SEEDS:
        if seed not in represented:
            report["seeds"].append({"seed": seed, "status": "not_attempted", "reason": reason,
                                    "slot_count": sweep.SLOTS_PER_SEED,
                                    "qualified_candidate_count": 0, "best_qualified_slot_index": None})
    report["five_seed_fit_qualification"] = False
    report["completion"] = "incomplete; not evidence of no passing point"


def run(plan_path: Path, expected_sha: str, output_root: Path, device: str) -> Path:
    setup_started = time.monotonic()
    plan, plan_sha, parent_plan, identity, parent_plan_path_str, verified = _preflight(plan_path, expected_sha)
    parent_plan_path = Path(parent_plan_path_str)
    parent_plan_sha = plan["lineage"]["schema8_parent_plan_sha256"]
    output_root = parent._check_output_root(str(output_root))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    run_dir = output_root / ("segment-sweep-" + run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "segment_sweep_report.json"
    report = {
        "schema_version": 1, "objective_id": sweep.OBJECTIVE_ID,
        "status": "preparing", "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan_sha, "base_commit": sweep.BASE_COMMIT,
        "calibration_status": "closed by design", "calibration_opened": False,
        "final3_status": "never indexed or evaluated",
        "source_identity": verified["source_identity"],
        "lineage": plan["lineage"], "fixed_fit_layout_hashes": plan["fixed_fit_layout_hashes"],
        "protocol": plan["protocol"], "budget": plan["budgets"],
        "planned_slots": sweep.TOTAL_SLOTS, "attempted_slots": 0,
        "slot_records": [], "seeds": [], "selected_qualified_fit_point": None,
        "five_seed_fit_qualification": False,
    }
    marker = None
    all_slots = []
    host_threads = torch.get_num_threads()
    try:
        _recheck(plan_path, plan_sha, plan, parent_plan_path, parent_plan_sha)
        # A fresh, exclusive plan marker is consumed before FIT deserialization,
        # geometry generation, basis preparation, or any candidate scoring.
        marker = coverage.create_attempt_marker(plan["lineage"]["source_manifest_path"], plan_sha)
        report["attempt_marker"] = str(marker)
        report["coverage_attempt_consumed"] = True
        report["status"] = "consumed_before_fit_loading"
        sweep.atomic_json(report_path, report)

        args = SimpleNamespace(plan_file=str(parent_plan_path),
                               expected_previous_sha256=coverage.PINNED_PREVIOUS_REPORT_SHA256,
                               device=device)
        original = parent._prepare_original_controls(parent_plan, parent_plan_sha, args, identity)
        original_rows = original["rows"]
        reference = np.asarray(original["reference"], dtype=np.float64)
        anchor = np.asarray(original["anchor"], dtype=np.float64)
        all_slots = list(sweep.iter_slots(plan, reference))
        physical = plan["physical"]

        layouts = coverage.make_coverage_masks(physical["raster"])
        new_rows = parent._generate_new_targets(layouts, original["device"], physical)
        novelty = parent._novelty_check(original_rows, new_rows, parent_plan["new_fit_generation"])
        observed_hashes = [{key: row[key] for key in ("layout_id", "mask_sha256", "target_sha256")}
                           for row in novelty]
        if observed_hashes != plan["fixed_fit_layout_hashes"]:
            raise ValueError("generated FIT masks/targets differ from frozen plan hashes")
        _sim, new32, new64, new_torch, new_targets_torch, new_parity = parent._prepare_basis(
            new_rows, original["device"], physical,
        )
        all_rows = original_rows + new_rows
        basis32, basis64 = dict(original["basis32"]), dict(original["basis64"])
        basis_torch, targets_torch = dict(original["basis_torch"]), dict(original["targets_torch"])
        basis32.update(new32); basis64.update(new64)
        basis_torch.update(new_torch); targets_torch.update(new_targets_torch)
        old_ids = [row["layout_id"] for row in original_rows]
        new_ids = [row["layout_id"] for row in new_rows]
        old_bases64 = [basis64[key] for key in old_ids]
        new_bases64 = [basis64[key] for key in new_ids]
        old_targets = [row["target"] for row in original_rows]
        new_targets = [row["target"] for row in new_rows]
        domain, guard_counts = coverage.build_guarded_domain(
            old_bases64, old_targets, old_ids, new_bases64, new_targets, new_ids,
            anchor, reference,
        )
        original_poly = coverage.verify_original_nominal_polytope(old_bases64, old_targets, reference)
        if not original_poly["passed"] or not domain.verify(reference)["passed"]:
            raise ValueError("frozen reference fails the augmented FIT source domain")
        golden_reference = parent._evaluate_rows(all_rows, basis32, reference, physical)
        torch.set_num_threads(sweep.SCREEN_CPU_THREADS)
        screen_reference = parent._evaluate_rows(all_rows, basis32, reference, physical)
        if screen_reference != golden_reference:
            raise ValueError("screen/reference hard metrics differ from original CPU threads")
        ref_new = parent._evaluate_rows(new_rows, basis32, reference, physical)
        _verify_reference_fit(ref_new, verified["frozen_reports"]["schema8"]["reference_new_fit"])
        reference_guard_audit = coverage.audit_float32_critical_guards(
            basis32, {row["layout_id"]: row["target"] for row in all_rows},
            anchor, reference, reference, old_ids, new_ids,
        )
        if not reference_guard_audit["passed"]:
            raise ValueError("reference fails the actual float32 critical-guard audit")
        setup_seconds = time.monotonic() - setup_started
        report.update({
            "status": "scoring", "setup_seconds": setup_seconds,
            "setup_limit_passed": setup_seconds <= sweep.SETUP_LIMIT_SECONDS,
            "generated_fit_layouts": novelty, "new_fit_basis_parity": new_parity,
            "guard_summary": guard_counts, "reference_new_fit": ref_new,
            "reference_float32_guard_audit": reference_guard_audit,
            "basis_scope": "original four FIT plus three fixed new FIT only",
            "cpu_threads": {"screen": sweep.SCREEN_CPU_THREADS, "golden": host_threads,
                            "reference_full_raster_parity": True},
        })
        context = {"original_rows": original_rows, "new_rows": new_rows,
                   "all_rows": all_rows, "basis32": basis32, "basis64": basis64,
                   "basis_torch": basis_torch, "targets_torch": targets_torch,
                   "physical": physical}
        sweep.atomic_json(report_path, report)
        if setup_seconds > sweep.SETUP_LIMIT_SECONDS:
            report.update(status="timeout", timeout_stage="setup", completion="incomplete; not evidence of no passing point")
            _seal_incomplete(report, all_slots, "setup_deadline")
            sweep.atomic_json(report_path, report)
            return report_path

        total_started = setup_started
        best = None
        best_rank = None
        best_seed_rank: dict[int, tuple] = {}
        seen64: dict[str, int] = {}
        seen32: dict[str, int] = {}
        physical_cache = {}
        total_deadline_hit = False
        for seed in sweep.SEEDS:
            seed_started = time.monotonic()
            seed_record = {"seed": seed, "status": "running", "slot_start": len(report["slot_records"]),
                           "slot_count": sweep.SLOTS_PER_SEED, "qualified_candidate_count": 0,
                           "best_qualified_slot_index": None}
            report["seeds"].append(seed_record)
            seed_slots = [slot for slot in all_slots if slot["seed"] == seed]
            for slot in seed_slots:
                total_elapsed = time.monotonic() - total_started
                seed_elapsed = time.monotonic() - seed_started
                if total_elapsed >= sweep.TOTAL_LIMIT_SECONDS:
                    report["slot_records"].extend(
                        _record_unattempted(rest, "total_deadline")
                        for rest in all_slots[slot["slot_index"]:]
                    )
                    seed_record["status"] = "timeout"
                    seed_record["timeout_reason"] = "total_deadline"
                    total_deadline_hit = True
                    break
                if seed_elapsed >= sweep.PER_SEED_LIMIT_SECONDS:
                    report["slot_records"].extend(
                        _record_unattempted(rest, "per_seed_deadline") for rest in seed_slots
                        if rest["slot_index"] >= slot["slot_index"]
                    )
                    seed_record["status"] = "timeout"
                    seed_record["timeout_reason"] = "per_seed_deadline"
                    break
                record = _slot_record(slot)
                record["duplicate_of_slot_index_f64"] = seen64.get(slot["weights_f64_sha256"])
                record["duplicate_of_slot_index_f32"] = seen32.get(slot["weights_f32_sha256"])
                seen64.setdefault(slot["weights_f64_sha256"], slot["slot_index"])
                seen32.setdefault(slot["weights_f32_sha256"], slot["slot_index"])
                record["status"] = "attempted"
                try:
                    scored = _candidate_score(
                        slot, context, ref_new["mean"], anchor, reference, original_poly, domain,
                        physical_cache,
                    )
                    record["screen_hard_metrics"] = scored
                    record["golden_audit"] = "not_required_screen_unqualified"
                    if scored["qualified"]:
                        scored = _golden_score(slot, context, ref_new["mean"], anchor, reference,
                                               original_poly, domain, host_threads)
                        record["golden_audit"] = "passed" if scored["qualified"] else "rejected"
                        record["golden_hard_metrics"] = scored
                    record["hard_metrics"] = scored
                    record["float64_audit"] = scored["float64_candidate_audit"]
                    record["qualified"] = bool(scored["qualified"])
                    if record["qualified"]:
                        seed_record["qualified_candidate_count"] += 1
                        rank = coverage.checkpoint_rank(scored)
                        record["fit_rank"] = list(rank)
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best = {"slot_index": slot["slot_index"], "seed": seed,
                                    "segment_id": slot["segment_id"], "alpha": slot["alpha"],
                                    "weights": slot["weights"].tolist(),
                                    "weights_f64_sha256": slot["weights_f64_sha256"],
                                    "fit_metrics": scored}
                            report["selected_qualified_fit_point"] = best
                        rank_tuple = tuple(record["fit_rank"])
                        if seed not in best_seed_rank or rank_tuple < best_seed_rank[seed]:
                            best_seed_rank[seed] = rank_tuple
                            seed_record["best_qualified_slot_index"] = slot["slot_index"]
                    report["attempted_slots"] += 1
                except Exception as exc:
                    record.update(status="error", error_type=type(exc).__name__, error=str(exc), qualified=False)
                    report["attempted_slots"] += 1
                    report["slot_records"].append(record)
                    report["status"] = "error"
                    report["error"] = {"type": type(exc).__name__, "message": str(exc),
                                       "traceback": traceback.format_exc()}
                    remaining = [rest for rest in all_slots[slot["slot_index"] + 1:]]
                    report["slot_records"].extend(_record_unattempted(rest, "scoring_error") for rest in remaining)
                    seed_record["status"] = "error"
                    _seal_incomplete(report, all_slots, "scoring_error")
                    _recheck(plan_path, plan_sha, plan, parent_plan_path, parent_plan_sha)
                    sweep.atomic_json(report_path, report)
                    return report_path
                report["slot_records"].append(record)
                completed_in_seed = sum(1 for x in report["slot_records"]
                                        if x.get("seed") == seed and x.get("status") == "attempted")
                if completed_in_seed % sweep.CHECKPOINT_INTERVAL == 0:
                    report["last_progress_utc"] = datetime.now(timezone.utc).isoformat()
                    report["elapsed_seconds"] = time.monotonic() - total_started
                    sweep.atomic_json(report_path, report)
            if seed_record["status"] == "running":
                seed_record["status"] = ("timeout" if time.monotonic() - seed_started
                                         >= sweep.PER_SEED_LIMIT_SECONDS else "complete")
            seed_record["wall_seconds"] = time.monotonic() - seed_started
            if report["status"] == "error":
                break
            if total_deadline_hit:
                break
            _recheck(plan_path, plan_sha, plan, parent_plan_path, parent_plan_sha)
            sweep.atomic_json(report_path, report)

        report["selected_qualified_fit_point"] = best
        if total_deadline_hit:
            _seal_incomplete(report, all_slots, "total_deadline")
        report["five_seed_fit_qualification"] = bool(
            len(report["seeds"]) == len(sweep.SEEDS)
            and all(seed["status"] == "complete" and seed["qualified_candidate_count"] > 0
                    for seed in report["seeds"])
        )
        report["elapsed_seconds"] = time.monotonic() - total_started
        if report["status"] != "error":
            complete = (report["attempted_slots"] == sweep.TOTAL_SLOTS
                        and len(report["slot_records"]) == sweep.TOTAL_SLOTS
                        and report["elapsed_seconds"] < sweep.TOTAL_LIMIT_SECONDS
                        and all(row["status"] == "complete" for row in report["seeds"]))
            report["status"] = "complete" if complete else "timeout"
            report["completion"] = ("all 5140 slots attempted" if complete else
                                    "incomplete; not evidence of no passing point")
        _recheck(plan_path, plan_sha, plan, parent_plan_path, parent_plan_sha)
        sweep.atomic_json(report_path, report)
        return report_path
    except Exception as exc:
        report["status"] = "error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
        report["calibration_status"] = "closed by design"
        report["calibration_opened"] = False
        report["final3_status"] = "never indexed or evaluated"
        if marker is not None:
            report["coverage_attempt_consumed"] = True
        _seal_incomplete(report, all_slots, "run_error")
        try:
            _recheck(plan_path, plan_sha, plan, parent_plan_path, parent_plan_sha)
        except Exception as identity_error:
            report["failure_identity_check"] = {"type": type(identity_error).__name__,
                                                 "message": str(identity_error)}
        sweep.atomic_json(report_path, report)
        return report_path
    finally:
        torch.set_num_threads(host_threads)


def _freeze_plan(args) -> int:
    repo_root = ROOT
    lineage_root = Path(args.lineage_root).resolve()
    target = Path(args.plan_file).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_manifest = sweep.PARENT_ROOT + "/source_family_manifest.json"
    plan = sweep.make_candidate_plan(repo_root, lineage_root, parent_manifest)
    encoded = (json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    plan_sha = sweep.sha256_bytes(encoded)
    source_file = ROOT / "source_segment_sweep.py"
    content = source_file.read_text(encoding="utf-8")
    updated, count = re.subn(r'(?m)^PINNED_PLAN_SHA256 = "[0-9a-f]{64}"$',
                             'PINNED_PLAN_SHA256 = "' + plan_sha + '"', content)
    if count != 1:
        raise ValueError("could not set unique prospective plan SHA pin")
    source_file.write_text(updated, encoding="utf-8", newline="\n")
    # Normalized hash is stable across setting the literal SHA pin.
    if sweep.source_file_hashes(ROOT)["source_segment_sweep.py"] != plan["source_identity"]["files"]["source_segment_sweep.py"]:
        raise ValueError("setting the plan SHA changed the normalized source identity")
    sweep.atomic_json(target, plan)
    print(json.dumps({"status": "prospective_plan_written", "plan_path": str(target),
                      "plan_sha256": plan_sha, "slots": sweep.TOTAL_SLOTS}, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("freeze-plan", "preflight", "run"), required=True)
    p.add_argument("--plan-file", required=True)
    p.add_argument("--expected-plan-sha256")
    p.add_argument("--lineage-root", default=r"D:\Codex\Lithography\work\server_benchmark\robust-source-quality-20261005-766872")
    p.add_argument("--output-root")
    p.add_argument("--device", default="cuda")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.mode == "freeze-plan":
            return _freeze_plan(args)
        if not args.expected_plan_sha256:
            raise ValueError("--expected-plan-sha256 is required")
        if args.mode == "preflight":
            plan, plan_sha, _parent_plan, identity, _parent_path, verified = _preflight(
                Path(args.plan_file), args.expected_plan_sha256,
            )
            print(json.dumps({"status": "preflight_passed_no_dataset_deserialized_no_optics_no_marker",
                              "plan_sha256": plan_sha, "source_identity": verified["source_identity"],
                              "source_manifest_sha256": plan["input_hashes"]["source_manifest_sha256"],
                              "seed_count": len(sweep.SEEDS), "slot_count": sweep.TOTAL_SLOTS,
                              "calibration_status": "closed by design",
                              "final3_status": "never indexed or evaluated"}, indent=2))
            return 0
        if not args.output_root:
            raise ValueError("--output-root is required for run mode")
        report = run(Path(args.plan_file), args.expected_plan_sha256,
                     Path(args.output_root), args.device)
        status = json.loads(report.read_text(encoding="utf-8"))["status"]
        print(json.dumps({"status": status, "report_path": str(report)}, indent=2))
        return 0 if status == "complete" else 3
    except Exception as exc:
        print("source segment sweep failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
