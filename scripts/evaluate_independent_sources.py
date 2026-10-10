#!/usr/bin/env python3
"""Evaluate three frozen illumination sources on supplied StdMetal/StdContact GLP layouts.

The evaluator intentionally has two phases. ``freeze`` parses and rasterizes
the complete external GLP set without evaluating any source. ``run`` consumes
one unique marker before importing the optical model and writes durable
per-layout progress. No optimization or layout selection by measured outcome
is performed here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
LIGHT_SOURCE_PATH = REPO_ROOT / "light_source.py"
# ``python scripts/evaluate_independent_sources.py ...`` puts only ``scripts/``
# at sys.path[0]. Add the checkout root so the CLI can import light_source.py
# from any working directory, not just when the caller sets PYTHONPATH.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FAMILY_NAMES = ("StdMetal271", "StdContact165")
EXPECTED_FILE_COUNTS = {"StdMetal271": 271, "StdContact165": 165}
EXPECTED_UPSTREAM_HEAD = "9c74e82218e377eaf6d02d113fc1ce6e36c92aa6"
EXPECTED_EVENT_REPORT_SHA256 = "04b6e03caaac23cd21ef07f287fd02d493650d098051c0b8f665e528b64b14df"
EXPECTED_EVENT_ARTIFACT_SHA256 = "60d54f2887dff4c5701ea6a2e1c289cc69ae662cbd7c7aa40d8a779cf52d95c2"
EXPECTED_EVENT_PLAN_SHA256 = "854b81a18ebd36589a5b50f6316059bfd45fa502c767e6bbb6f148d76f71a7cf"
EXPECTED_CACHED_COVERAGE_PLAN_SHA256 = "de435a135c44fad5346cbc2cc21985cd54042d4655b037d87df3bb4f132a1127"
EXPECTED_EVENT_CANDIDATE_ID = "seed-29:schema7_endpoint:000014"
EXPECTED_VECTOR_HASHES = {
    "reference": {
        "weights_f64_sha256": "e3e8f20d3574a48c782e9932c07b001eb45f0b3778ded612af086781014f69db",
        "weights_f32_sha256": "98ea7a216ca3ad6149abe7a8a5482da69990a441b855648ac67bb88268721d2e",
    },
    "best_known": {
        "weights_f64_sha256": "388ff3bd997b4cbe7543a9d0fce15e2c46c2854ee9659a623b30539235069e86",
        "weights_f32_sha256": "aa24fb03e849007cb3530f0de31968d88ad9af8089fe1f1da78841bb77df7f46",
    },
    "event_candidate": {
        "weights_f64_sha256": "36c66f0b29bdeced656a4734d906bb5cd667c26daa6808e89dd1131e5159f3c6",
        "weights_f32_sha256": "901b22f41ef587357dab92e5ef80f557a9d0bdd22fc86560944d5b6dd7f58b41",
    },
}
PHYSICAL = {
    "canvas_size_px": 1024,
    "sensitivity_canvas_size_px": 512,
    "pixel_size_nm": 4.0,
    "numerical_aperture": 1.35,
    "wavelength_nm": 193.0,
    "source_grid": 9,
    "sigma_inner": 0.3,
    "sigma_outer": 0.9,
    "focus_nm": 0.0,
    "doses": [0.98, 1.0, 1.02],
    "threshold": 0.225,
    "steepness": 50.0,
    "source_chunk_size": 4,
    "cache_max_bytes": 0,
    "basis_max_bytes": 1024**3,
    "aerial_rtol": 1e-5,
    "aerial_atol": 1e-6,
}
COMPUTE_POLICY = {
    "torch_num_threads": 1,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "basis_contraction": "float32 CPU weighted sum from one GPU-prepared basis per layout",
}
BOOTSTRAP_SEED = 20261010
BOOTSTRAP_DRAWS = 10000
MAX_COORD_X_NM = 1277
MAX_COORD_Y_NM = 1270
PROTOCOL_SCHEMA = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, raw


def _weight_array(values: object, name: str) -> np.ndarray:
    try:
        weights = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} weights must be a finite numeric vector") from exc
    if weights.size != 49 or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"{name} must contain 49 finite nonnegative weights")
    if abs(float(weights.sum(dtype=np.float64)) - 1.0) > 1e-8:
        raise ValueError(f"{name} weights do not sum to one within 1e-8")
    return weights


def vector_hashes(weights: object) -> dict:
    vector = _weight_array(weights, "source")
    return {
        "weights_f64_sha256": _sha256_bytes(np.asarray(vector, dtype="<f8").tobytes()),
        "weights_f32_sha256": _sha256_bytes(np.asarray(vector, dtype="<f4").tobytes()),
    }


def _validate_pinned_inputs(cached_plan_path: Path, event_path: Path,
                            event_report_path: Path) -> dict:
    plan, plan_raw = _read_json(cached_plan_path)
    event, event_raw = _read_json(event_path)
    report, report_raw = _read_json(event_report_path)
    if plan.get("coverage_plan_sha256") != EXPECTED_CACHED_COVERAGE_PLAN_SHA256:
        raise ValueError("cached candidate plan coverage-plan pin differs from the frozen value")
    physical = plan.get("physical", {})
    expected_plan_physical = {
        "source_grid": 9, "sigma_inner": 0.3, "sigma_outer": 0.9,
        "NA": 1.35, "wavelength_nm": 193, "pixel_nm": 4,
        "focus": 0, "doses": [0.98, 1, 1.02],
        "threshold": 0.225, "steepness": 50,
    }
    for key, expected in expected_plan_physical.items():
        if physical.get(key) != expected:
            raise ValueError(f"cached plan physical.{key} differs from the pinned source protocol")
    if report.get("status") != "complete" or report.get("objective_id") != "prospective_source_event_vs_grid_quality_time_v1":
        raise ValueError("uncached event report is not the expected completed benchmark report")
    report_sha = _sha256_bytes(report_raw)
    if report_sha != EXPECTED_EVENT_REPORT_SHA256 or event.get("event_report_sha256") != report_sha:
        raise ValueError("event candidate is not bound to the pinned uncached report bytes")
    event_file_sha = _sha256_bytes(event_raw)
    if event_file_sha != EXPECTED_EVENT_ARTIFACT_SHA256:
        raise ValueError("event candidate artifact differs from the pinned published JSON")
    if (report.get("event_plan_sha256") != EXPECTED_EVENT_PLAN_SHA256
            or event.get("event_plan_sha256") != report.get("event_plan_sha256")):
        raise ValueError("event candidate plan SHA does not match the pinned report")
    if event.get("candidate_id") != EXPECTED_EVENT_CANDIDATE_ID:
        raise ValueError("event candidate ID differs from the frozen selection")
    candidate = _weight_array(event.get("weights"), "event candidate")
    candidate_hashes = vector_hashes(candidate)
    if event.get("weights_f32_sha256") != candidate_hashes["weights_f32_sha256"]:
        raise ValueError("event candidate f32 vector hash is invalid")
    if event.get("weights_f64_sha256") != candidate_hashes["weights_f64_sha256"]:
        raise ValueError("event candidate f64 vector hash is invalid")
    if candidate_hashes != EXPECTED_VECTOR_HASHES["event_candidate"]:
        raise ValueError("event candidate vector does not match the pinned source vector")

    source_vectors = plan.get("source_vectors")
    if not isinstance(source_vectors, dict):
        raise ValueError("cached candidate plan lacks source_vectors")
    required = {
        "reference": "reference",
        "best_known": "best_known_slot_4755_seed_101",
    }
    result = {}
    for label, plan_key in required.items():
        record = source_vectors.get(plan_key)
        if not isinstance(record, dict):
            raise ValueError(f"cached candidate plan lacks source_vectors.{plan_key}")
        weights = _weight_array(record.get("weights"), label)
        actual = vector_hashes(weights)
        for field, value in actual.items():
            if record.get(field) != value or EXPECTED_VECTOR_HASHES[label][field] != value:
                raise ValueError(f"cached {label} {field} differs from the frozen pin")
        result[label] = {"weights": weights.tolist(), **actual}
    result["event_candidate"] = {
        "weights": candidate.tolist(), **candidate_hashes,
    }
    return {
        "vectors": result,
        "cached_plan_file": str(cached_plan_path.resolve()),
        "cached_plan_file_sha256": _sha256_bytes(plan_raw),
        "cached_coverage_plan_sha256": plan["coverage_plan_sha256"],
        "cached_plan_source_vectors": {
            "reference": "reference",
            "best_known": "best_known_slot_4755_seed_101",
        },
        "event_candidate_file": str(event_path.resolve()),
        "event_candidate_file_sha256": event_file_sha,
        "event_candidate_id": event["candidate_id"],
        "event_plan_sha256": report["event_plan_sha256"],
        "event_report_file": str(event_report_path.resolve()),
        "event_report_file_sha256": report_sha,
        "event_report_status": report["status"],
        "event_report_selected_weights_file_sha256": report.get("selected_weights_file_sha256"),
    }


def _polygon_area2(vertices: list[tuple[int, int]]) -> int:
    return sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(vertices, vertices[1:] + vertices[:1])
    )


def _segments_intersect(a, b, c, d) -> bool:
    """Return true for a non-adjacent orthogonal intersection or overlap."""
    ax, ay = a; bx, by = b; cx, cy = c; dx, dy = d
    a_vert = ax == bx
    c_vert = cx == dx
    if a_vert and c_vert:
        return ax == cx and max(min(ay, by), min(cy, dy)) <= min(max(ay, by), max(cy, dy))
    if not a_vert and not c_vert:
        return ay == cy and max(min(ax, bx), min(cx, dx)) <= min(max(ax, bx), max(cx, dx))
    if not a_vert:
        a, b, c, d = c, d, a, b
        ax, ay = a; bx, by = b; cx, cy = c; dx, dy = d
    return min(ay, by) <= cy <= max(ay, by) and min(cx, dx) <= ax <= max(cx, dx)


def _validate_simple_polygon(vertices: list[tuple[int, int]], source: str) -> None:
    edges = list(zip(vertices, vertices[1:] + vertices[:1]))
    count = len(edges)
    for i, (a, b) in enumerate(edges):
        if a == b:
            raise ValueError(f"{source}: polygon has a zero-length edge")
        if i == 0:
            continue
        for j in range(i):
            if j == i - 1 or (j == 0 and i == count - 1):
                continue
            c, d = edges[j]
            if _segments_intersect(a, b, c, d):
                raise ValueError(f"{source}: polygon is self-intersecting or has overlapping edges")


def parse_glp(path: Path) -> dict:
    """Parse the pinned StdMetal/StdContact header and Manhattan PGONs in nm."""
    raw = path.read_bytes()
    try:
        lines = raw.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: GLP is not UTF-8 text") from exc
    begin_seen = False
    equiv_seen = False
    cname = None
    level = None
    cell_seen = False
    end_seen = False
    polygons: list[list[tuple[int, int]]] = []
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if end_seen:
            raise ValueError(f"{path}:{line_number}: records after ENDMSG are unsupported")
        if line.startswith("BEGIN"):
            if begin_seen or equiv_seen or polygons or not re.fullmatch(
                r"BEGIN(?:\s+/\*.*\*/)?", line
            ):
                raise ValueError(f"{path}:{line_number}: malformed or repeated BEGIN")
            begin_seen = True
            continue
        fields = line.split()
        token = fields[0].upper()
        if token == "EQUIV":
            if not begin_seen or equiv_seen or fields[:4] != ["EQUIV", "1", "1000", "MICRON"]:
                raise ValueError(f"{path}:{line_number}: expected one EQUIV 1 1000 MICRON header")
            if fields[4:] != ["+X,+Y"]:
                raise ValueError(
                    f"{path}:{line_number}: unsupported EQUIV coordinate orientation; expected +X,+Y"
                )
            equiv_seen = True
        elif token == "CNAME":
            if not equiv_seen or cname is not None or len(fields) != 2 or cell_seen or polygons:
                raise ValueError(f"{path}:{line_number}: malformed, repeated, or misplaced CNAME")
            cname = fields[1]
        elif token == "LEVEL":
            if cname is None or level is not None or len(fields) != 2 or cell_seen or polygons:
                raise ValueError(f"{path}:{line_number}: malformed, repeated, or misplaced LEVEL")
            level = fields[1]
        elif token == "CELL":
            if level is None or cell_seen or fields != ["CELL", cname, "PRIME"]:
                raise ValueError(f"{path}:{line_number}: expected CELL <CNAME> PRIME after LEVEL")
            cell_seen = True
        elif token == "PGON":
            if not cell_seen or len(fields) < 11 or fields[1] != "N" or fields[2] != level:
                raise ValueError(f"{path}:{line_number}: malformed PGON or missing supported GLP header")
            coordinate_fields = fields[3:]
            if len(coordinate_fields) % 2:
                raise ValueError(f"{path}:{line_number}: PGON needs x/y coordinate pairs")
            try:
                flat = [int(value, 10) for value in coordinate_fields]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: GLP coordinates must be integer nanometers") from exc
            vertices = list(zip(flat[::2], flat[1::2]))
            if len(vertices) >= 2 and vertices[0] == vertices[-1]:
                vertices.pop()
            if len(set(vertices)) < 4 or _polygon_area2(vertices) == 0:
                raise ValueError(f"{path}:{line_number}: PGON has empty or collapsed area")
            for first, second in zip(vertices, vertices[1:] + vertices[:1]):
                if first[0] != second[0] and first[1] != second[1]:
                    raise ValueError(f"{path}:{line_number}: only Manhattan PGON edges are accepted")
            _validate_simple_polygon(vertices, f"{path}:{line_number}")
            polygons.append(vertices)
        elif token == "ENDMSG":
            if fields != ["ENDMSG"] or end_seen or not cell_seen or not polygons:
                raise ValueError(f"{path}:{line_number}: malformed or repeated ENDMSG")
            end_seen = True
        else:
            raise ValueError(f"{path}:{line_number}: unsupported GLP record {fields[0]!r}")
    if not begin_seen or not equiv_seen or cname is None or level is None or not cell_seen or not end_seen or not polygons:
        raise ValueError(f"{path}: missing supported BEGIN/EQUIV/CNAME/LEVEL/CELL header, PGON geometry, or ENDMSG")
    all_points = [point for polygon in polygons for point in polygon]
    min_x = min(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_x = max(point[0] for point in all_points)
    max_y = max(point[1] for point in all_points)
    if min_x < 0 or min_y < 0 or max_x > MAX_COORD_X_NM or max_y > MAX_COORD_Y_NM:
        raise ValueError(
            f"{path}: coordinates exceed the pinned {MAX_COORD_X_NM} x {MAX_COORD_Y_NM} nm input bounds"
        )
    return {
        "raw_sha256": _sha256_bytes(raw),
        "polygons": polygons,
        "bbox_nm": [min_x, min_y, max_x, max_y],
        "polygon_count": len(polygons),
    }


def _center_translation(bbox_nm: list[int], canvas_size_px: int,
                        pixel_size_nm: float = PHYSICAL["pixel_size_nm"]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = bbox_nm
    center = canvas_size_px * pixel_size_nm / 2.0
    dx = center - (min_x + max_x) / 2.0
    dy = center - (min_y + max_y) / 2.0
    # The positive half-tie rule is pinned to make translation independent of
    # the host language's banker's rounding behavior.
    dx = math.floor(dx / pixel_size_nm + 0.5) * pixel_size_nm
    dy = math.floor(dy / pixel_size_nm + 0.5) * pixel_size_nm
    return float(dx), float(dy)


def rasterize_polygons(polygons: list[list[tuple[int, int]]], bbox_nm: list[int],
                       canvas_size_px: int = 1024,
                       pixel_size_nm: float = PHYSICAL["pixel_size_nm"]) -> tuple[np.ndarray, dict]:
    if canvas_size_px <= 0 or pixel_size_nm <= 0 or not polygons:
        raise ValueError("canvas and geometry must be nonempty and positive")
    min_x, min_y, max_x, max_y = bbox_nm
    dx, dy = _center_translation(bbox_nm, canvas_size_px, pixel_size_nm)
    extent = canvas_size_px * pixel_size_nm
    if min_x + dx < 0 or min_y + dy < 0 or max_x + dx > extent or max_y + dy > extent:
        raise ValueError("translated geometry does not fit the fixed canvas; clipping is forbidden")
    mask = np.zeros((canvas_size_px, canvas_size_px), dtype=np.uint8)
    pixel_centers_x = (np.arange(canvas_size_px, dtype=np.float64) + 0.5) * pixel_size_nm
    for row in range(canvas_size_px):
        y_center_nm = (canvas_size_px - row - 0.5) * pixel_size_nm
        source_y = y_center_nm - dy
        for polygon in polygons:
            intersections = []
            for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
                if x1 != x2:
                    continue
                if min(y1, y2) <= source_y < max(y1, y2):
                    intersections.append(float(x1) + dx)
            intersections.sort()
            if len(intersections) % 2:
                raise ValueError("scanline has an odd number of polygon crossings")
            for left, right in zip(intersections[::2], intersections[1::2]):
                first = int(np.searchsorted(pixel_centers_x, left, side="left"))
                last = int(np.searchsorted(pixel_centers_x, right, side="left"))
                if last > first:
                    mask[row, first:last] = 1
    if not bool(mask.any()):
        raise ValueError("fixed rasterization produced an empty or subpixel-collapsed target")
    rows, columns = np.nonzero(mask)
    actual_bbox = [int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)]
    raw_mask_hash = _sha256_bytes(mask.tobytes(order="C"))
    packed = np.packbits(mask.reshape(-1), bitorder="big").tobytes()
    return mask, {
        "canvas_size_px": int(canvas_size_px),
        "pixel_size_nm": float(pixel_size_nm),
        "translation_nm": [dx, dy],
        "raster_bbox_px_halfopen": actual_bbox,
        "positive_pixel_count": int(mask.sum()),
        "raster_sha256": raw_mask_hash,
        "packed_mask_sha256": _sha256_bytes(packed),
        "packed_mask_bytes": len(packed),
        "raster_rule": "scanline pixel centers; ymin<=y<ymax, left<=x<right; polygon union",
    }


class _DisjointSet:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            low, high = sorted((a, b))
            self.parent[high] = low

    def components(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for item in sorted(self.parent):
            result[self.find(item)].append(item)
        return dict(result)


def cell_group_for_path(path: Path) -> str:
    stem = path.stem.split("__", 1)[0]
    return re.sub(r"_X\d+$", "", stem, flags=re.IGNORECASE)


def _list_glps(family_dirs: dict[str, Path]) -> list[dict]:
    selected = []
    for family in FAMILY_NAMES:
        root = family_dirs[family].resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"missing GLP family directory: {root}")
        files = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".glp"),
                       key=lambda p: p.relative_to(root).as_posix().casefold())
        if len(files) != EXPECTED_FILE_COUNTS[family]:
            raise ValueError(f"{family} contains {len(files)} GLP files; expected {EXPECTED_FILE_COUNTS[family]}")
        if not files:
            raise ValueError(f"no GLP files found under {root}")
        for path in files:
            selected.append({
                "family": family,
                "path": str(path.resolve()),
                "relative_path": path.relative_to(root).as_posix(),
                "cell_group": cell_group_for_path(path),
            })
    return sorted(selected, key=lambda row: (row["family"], row["relative_path"].casefold(), row["relative_path"]))


def _assert_output_outside_repo(output_dir: Path, source_roots: list[Path]) -> None:
    target = output_dir.resolve()
    repo = REPO_ROOT.resolve()
    try:
        target.relative_to(repo)
    except ValueError:
        pass
    else:
        raise ValueError("external run artifacts must be outside the evaluator Git repository")
    resolved_roots = [root.resolve() for root in source_roots]
    forbidden_roots = list(resolved_roots)
    checkout_roots = []
    for root in resolved_roots:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"], cwd=root, check=True,
                capture_output=True, text=True, timeout=10,
            )
            checkout_roots.append(Path(result.stdout.strip()).resolve())
        except (OSError, subprocess.SubprocessError):
            checkout_roots.append(None)
    if checkout_roots and all(root is not None for root in checkout_roots):
        unique_checkouts = {root for root in checkout_roots}
        if len(unique_checkouts) == 1:
            forbidden_roots.append(next(iter(unique_checkouts)))
    # The supplied upstream layout uses sibling benchmark/StdMetal and
    # benchmark/StdContact directories. If git metadata is unavailable, the
    # shared benchmark parent is still a bounded common data root.
    if (not (checkout_roots and all(root is not None for root in checkout_roots)
             and len({root for root in checkout_roots}) == 1)
            and len(resolved_roots) == 2
            and resolved_roots[0].parent == resolved_roots[1].parent
            and resolved_roots[0].parent.name.casefold() == "benchmark"
            and {root.name.casefold() for root in resolved_roots} == {"stdmetal", "stdcontact"}):
        benchmark_root = resolved_roots[0].parent
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"], cwd=benchmark_root, check=True,
                capture_output=True, text=True, timeout=10,
            )
            forbidden_roots.append(Path(result.stdout.strip()).resolve())
        except (OSError, subprocess.SubprocessError):
            forbidden_roots.append(benchmark_root)
    for root in forbidden_roots:
        try:
            target.relative_to(root)
        except ValueError:
            continue
        raise ValueError(f"external run artifacts must be outside source data: {root}")


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def freeze(args: argparse.Namespace) -> dict:
    family_dirs = {
        "StdMetal271": Path(args.metal_dir),
        "StdContact165": Path(args.contact_dir),
    }
    out_dir = Path(args.output_dir).resolve()
    _assert_output_outside_repo(out_dir, list(family_dirs.values()))
    if out_dir.exists():
        raise FileExistsError(f"freeze output already exists; refusing to overwrite: {out_dir}")
    if args.upstream_head != EXPECTED_UPSTREAM_HEAD:
        raise ValueError("upstream GLP repository HEAD differs from the pinned dataset lineage")
    pinned_inputs = _validate_pinned_inputs(
        Path(args.cached_plan), Path(args.event_candidate), Path(args.event_report)
    )
    files = _list_glps(family_dirs)
    all_layouts = []
    for item in files:
        parsed = parse_glp(Path(item["path"]))
        mask, raster_info = rasterize_polygons(parsed["polygons"], parsed["bbox_nm"])
        all_layouts.append({
            **item,
            "raw_glp_sha256": parsed["raw_sha256"],
            "bbox_nm": parsed["bbox_nm"],
            "polygon_count": parsed["polygon_count"],
            "raster": raster_info,
            "mask": mask,
        })

    # Exact raster duplicates are collapsed within each family by the
    # lexicographically first source path. Every alias remains pinned. A
    # duplicate geometry joining different cell stems joins their bootstrap
    # groups conservatively.
    dsu = _DisjointSet()
    by_family_hash: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_hash_all_families: dict[str, list[dict]] = defaultdict(list)
    for row in all_layouts:
        dsu.find(row["cell_group"])
        raster_hash = row["raster"]["raster_sha256"]
        by_family_hash[(row["family"], raster_hash)].append(row)
        by_hash_all_families[raster_hash].append(row)
    for aliases in by_family_hash.values():
        groups = sorted({row["cell_group"] for row in aliases})
        for group in groups[1:]:
            dsu.union(groups[0], group)
    for aliases in by_hash_all_families.values():
        groups = sorted({row["cell_group"] for row in aliases})
        for group in groups[1:]:
            dsu.union(groups[0], group)
    components = dsu.components()
    group_lookup = {group: "|".join(members) for members in components.values() for group in members}

    cache_plan, _ = _read_json(Path(args.cached_plan))
    excluded_fit = []
    for fit_row in cache_plan.get("fixed_fit_layout_hashes", []):
        excluded_fit.append({
            "layout_id": fit_row.get("layout_id"),
            "mask_sha256": fit_row.get("mask_sha256"),
            "target_sha256": fit_row.get("target_sha256"),
        })
    old_fit_target_hashes = {row.get("target_sha256") for row in excluded_fit}
    old_fit_mask_hashes = {row.get("mask_sha256") for row in excluded_fit}
    for row in all_layouts:
        if row["raster"]["raster_sha256"] in old_fit_target_hashes | old_fit_mask_hashes:
            raise ValueError(f"GLP raster unexpectedly matches an excluded FIT hash: {row['relative_path']}")

    # Build one retained record for each family/raster pair. Separate family
    # records are retained when geometry occurs in both families, but their
    # group components have already been joined above.
    retained = []
    for (family, raster_hash), aliases in sorted(by_family_hash.items()):
        aliases.sort(key=lambda row: (row["relative_path"].casefold(), row["relative_path"]))
        representative = aliases[0]
        layout_id = family + ":" + representative["relative_path"]
        packed = np.packbits(representative["mask"].reshape(-1), bitorder="big").tobytes()
        retained.append({
            "layout_id": layout_id,
            "family": family,
            "cell_group": group_lookup[representative["cell_group"]],
            "source_cell_group": representative["cell_group"],
            "source_glp": representative["path"],
            "relative_path": representative["relative_path"],
            "raw_glp_sha256": representative["raw_glp_sha256"],
            "bbox_nm": representative["bbox_nm"],
            "polygon_count": representative["polygon_count"],
            "raster": representative["raster"],
            "aliases": [
                {"relative_path": alias["relative_path"], "raw_glp_sha256": alias["raw_glp_sha256"],
                 "source_cell_group": alias["cell_group"]}
                for alias in aliases
            ],
            "alias_count": len(aliases),
            "packed_mask_bytes_internal": packed,
        })
    retained.sort(key=lambda row: (row["family"], row["relative_path"].casefold(), row["relative_path"]))

    # The masks are written as a compact binary snapshot in the external run
    # directory. This makes the run independent of edits after preflight.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=False, exist_ok=False)
    targets_dir = out_dir / "target_masks"
    targets_dir.mkdir()
    for row in retained:
        packed = row.pop("packed_mask_bytes_internal")
        filename = _sha256_bytes(row["layout_id"].encode("utf-8")) + ".mask"
        target_path = targets_dir / filename
        with target_path.open("xb") as stream:
            stream.write(packed)
            stream.flush()
            os.fsync(stream.fileno())
        row["target_snapshot"] = str(target_path.resolve())
        row["target_snapshot_sha256"] = _sha256_bytes(packed)

    files_by_family = {family: sum(row["family"] == family for row in all_layouts) for family in FAMILY_NAMES}
    cell_groups_by_family = {
        family: sorted({row["cell_group"] for row in all_layouts if row["family"] == family})
        for family in FAMILY_NAMES
    }
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_id": "independent-fixed-source-glp-" + _utc_now().replace(":", "").replace("-", ""),
        "created_utc": _utc_now(),
        "status": "frozen_preflight_only",
        "source_repository": {
            "upstream_repository": "/home/murilo/Documentos/Lithography/lithobench",
            "upstream_head": args.upstream_head,
            "local_evaluator_git_head": _git_head(),
            "evaluator_script_sha256": _sha256_file(SCRIPT_PATH),
            "light_source_sha256": _sha256_file(LIGHT_SOURCE_PATH),
        },
        "input_pins": pinned_inputs,
        "dataset": {
            "families": FAMILY_NAMES,
            "family_directories": {name: str(path.resolve()) for name, path in family_dirs.items()},
            "raw_file_counts": files_by_family,
            "raw_file_count_total": len(all_layouts),
            "deduplicated_layout_count": len(retained),
            "duplicate_alias_count": len(all_layouts) - len(retained),
            "cell_group_names_by_family": cell_groups_by_family,
            "cell_group_count_by_family": {key: len(value) for key, value in cell_groups_by_family.items()},
            "pooled_cell_group_components": list(components.values()),
            "pooled_cell_group_count": len(components),
            "source_files": [
                {key: row[key] for key in ("family", "path", "relative_path", "cell_group", "raw_glp_sha256")}
                for row in all_layouts
            ],
            "layouts": retained,
            "expected_coordinate_bounds_nm": {"max_x": MAX_COORD_X_NM, "max_y": MAX_COORD_Y_NM},
            "raster_policy": {
                "primary": "fixed 1024x1024 at 4 nm; center bounding box using a translation rounded to 4 nm; no scale, crop, or clipping",
                "pixel_rule": "scanline at pixel centers; vertical-edge crossings ymin<=y<ymax; fill intervals left<=x<right; PGON union",
                "geometry_units": "EQUIV 1 1000 MICRON; integer GLP coordinates interpreted as nanometers",
            "dedupe": "exact supplied-tile target raster hashes deduplicated within each family; lexicographically first path retained; aliases preserved",
                "group_union": "cell groups joined when exact target rasters tie, including ties across families",
                "cell_group_rule": "filename stem before first __, then remove trailing _X followed by digits",
            },
        },
        "physical": PHYSICAL,
        "compute_policy": COMPUTE_POLICY,
        "source_vectors": pinned_inputs["vectors"],
        "excluded_fit_metadata": {
            "fixed_fit_layout_hashes_from_cached_plan": excluded_fit,
            "evaluation_uses_fit_loader": False,
            "fit_hash_overlap_check": "all independent 1024x1024 target hashes differ from cached FIT mask/target hashes",
            "prior_human_exposure_assessed": False,
        },
        "statistics": {
            "primary_metric": "equal mean over cell-group means of each layout's PVband pixels divided by target-positive pixels",
            "primary_metrics": ["nominal mismatch pixels / target-positive pixels", "worst-dose mismatch pixels / target-positive pixels"],
            "secondary_metrics": ["nominal absolute relative area error", "worst-dose absolute relative area error"],
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "pinned cell-group component", "ci": "percentile 95%", "interpretation": "descriptive group-level interval; not validation of independent sampling"},
            "claim_gate": "candidate-v-reference PVband 95% group-bootstrap CI upper bound < 0; mean nominal and worst-dose spatial mismatch-fraction deltas <= 0; no blank candidate layout; full raster, method, and hard-mask parity validation passed",
        },
        "runtime_policy": {"time_budget_seconds": int(args.time_budget_seconds), "fatal_input_or_runtime_failure": "durable partial status; no scientific completion", "numerical_parity_mismatch": "retain every scored layout, continue; validation and claim gates fail"},
    }
    protocol_path = out_dir / "protocol.json"
    protocol_bytes = _json_bytes(protocol)
    with protocol_path.open("xb") as stream:
        stream.write(protocol_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    return {"status": protocol["status"], "protocol_file": str(protocol_path),
            "protocol_file_sha256": _sha256_bytes(protocol_bytes),
            "raw_file_count": len(all_layouts), "deduplicated_layout_count": len(retained),
            "pooled_cell_group_count": len(components), "output_dir": str(out_dir)}


def unpack_target(path: Path, canvas_size_px: int) -> np.ndarray:
    packed = path.read_bytes()
    expected = canvas_size_px * canvas_size_px
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")
    if bits.size < expected or np.any(bits[expected:]):
        raise ValueError(f"packed target mask has invalid length or nonzero padding bits: {path}")
    return bits[:expected].reshape(canvas_size_px, canvas_size_px).astype(np.uint8, copy=False)


def hard_metrics_from_aerial(aerial, target: np.ndarray, *, doses: list[float],
                             threshold: float, steepness: float) -> tuple[dict, list[np.ndarray]]:
    """Use the pinned sigmoid >= 0.5 decision for all dose corners."""
    import torch
    from light_source import resist_image

    if aerial.ndim != 2:
        aerial = aerial.reshape(aerial.shape[-2], aerial.shape[-1])
    target_bool = np.asarray(target, dtype=bool)
    predictions = []
    for dose in doses:
        printed = resist_image(aerial, dose=dose, threshold=threshold, steepness=steepness) >= 0.5
        predictions.append(printed.detach().to(device="cpu").numpy().astype(bool, copy=False))
    stack = np.stack(predictions, axis=0)
    target_count = int(target_bool.sum())
    if target_count <= 0:
        raise ValueError("independent target has no positive pixels")
    corner_errors = [int(np.count_nonzero(prediction != target_bool)) for prediction in stack]
    printed_counts = [int(prediction.sum()) for prediction in stack]
    relative_area_errors = [abs(count - target_count) / target_count for count in printed_counts]
    spatial_error_fractions = [error / target_count for error in corner_errors]
    pv_band = int(np.count_nonzero(np.any(stack, axis=0) != np.all(stack, axis=0)))
    result = {
        "target_positive_pixels": target_count,
        "printed_positive_pixels_by_dose": printed_counts,
        "per_dose_error_pixels": corner_errors,
        "nominal_error_pixels": corner_errors[1],
        "worst_dose_error_pixels": max(corner_errors),
        "nominal_error_fraction_of_target": spatial_error_fractions[1],
        "worst_dose_error_fraction_of_target": max(spatial_error_fractions),
        "pvband_pixels": pv_band,
        "pvband_fraction_of_target": pv_band / target_count,
        "nominal_absolute_relative_area_error": relative_area_errors[1],
        "worst_dose_absolute_relative_area_error": max(relative_area_errors),
        "no_blank_positive_target_any_dose": all(count > 0 for count in printed_counts),
    }
    return result, predictions


def _source_model(weights: list[float], device: str):
    import torch
    from light_source import DifferentiableAbbeLitho, PixelatedLightSource

    class FixedWeightsSource(PixelatedLightSource):
        """Fixed, non-renormalized weight vector on the pinned 9x9 source."""

        def __init__(self, vector):
            super().__init__(grid_size=PHYSICAL["source_grid"],
                             sigma_inner=PHYSICAL["sigma_inner"],
                             sigma_outer=PHYSICAL["sigma_outer"])
            active = self._pupil_support
            if int(active.sum().item()) != len(vector):
                raise ValueError("source vector length does not match active pupil support")
            value = torch.as_tensor(vector, dtype=torch.float32)
            if not bool(torch.isfinite(value).all().item()) or bool((value < 0).any().item()):
                raise ValueError("fixed source weights must be finite and nonnegative")
            if abs(float(value.double().sum().item()) - 1.0) > 1e-6:
                raise ValueError("fixed source weights must sum to one")
            self.register_buffer("_fixed_weights", value)

        def distribution(self):
            return self._coordinates[self._pupil_support], self._fixed_weights

    source = FixedWeightsSource(weights).to(device)
    model = DifferentiableAbbeLitho(
        source,
        numerical_aperture=PHYSICAL["numerical_aperture"],
        wavelength_nm=PHYSICAL["wavelength_nm"],
        pixel_size_nm=PHYSICAL["pixel_size_nm"],
        source_chunk_size=PHYSICAL["source_chunk_size"],
        cache_max_bytes=PHYSICAL["cache_max_bytes"],
    ).to(device)
    model.eval()
    return model


def _apply_compute_policy(torch) -> None:
    torch.set_num_threads(COMPUTE_POLICY["torch_num_threads"])
    torch.backends.cuda.matmul.allow_tf32 = COMPUTE_POLICY["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = COMPUTE_POLICY["cudnn_allow_tf32"]


def _compute_environment(torch, device: str) -> dict:
    cuda_device_name = None
    if device.startswith("cuda") and torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name(torch.device(device))
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_device_name": cuda_device_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def evaluate_layout_methods(mask: np.ndarray, target: np.ndarray,
                            source_vectors: dict[str, dict], device: str) -> dict[str, dict]:
    """Score all fixed vectors with one reusable GPU basis for this layout."""
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    _apply_compute_policy(torch)
    input_gpu = torch.as_tensor(mask, dtype=torch.float32, device=device)
    models = {
        source: _source_model(record["weights"], device)
        for source, record in source_vectors.items()
    }
    source_order = tuple(source_vectors)
    if not source_order:
        raise ValueError("at least one fixed source vector is required")
    with torch.no_grad():
        basis = models[source_order[0]].prepare_basis(
            input_gpu.detach(), defocus_nm=PHYSICAL["focus_nm"], max_bytes=PHYSICAL["basis_max_bytes"]
        )
        # Transfer the source-independent float32 intensities once. CPU1 is
        # the canonical path; its only per-source work is the fixed-weight sum.
        basis_cpu = basis.intensities.detach().to(device="cpu")[0]
        method_aerials = {}
        if device.startswith("cuda"):
            direct_key, weighted_key = "direct_gpu", "weighted_gpu"
        else:
            direct_key, weighted_key = "direct_cpu", "weighted_cpu"
        for source, record in source_vectors.items():
            direct = models[source](input_gpu, defocus_nm=PHYSICAL["focus_nm"]).reshape(mask.shape)
            weighted = models[source].evaluate_basis(basis, defocus_nm=PHYSICAL["focus_nm"]).reshape(mask.shape)
            cpu_weights = torch.as_tensor(record["weights"], dtype=torch.float32, device="cpu")
            canonical = torch.einsum("nhw,n->hw", basis_cpu, cpu_weights).reshape(mask.shape)
            method_aerials[source] = {
                direct_key: direct.detach().to(device="cpu").clone(),
                weighted_key: weighted.detach().to(device="cpu").clone(),
                "canonical_cpu_1thread_basis": canonical.detach().clone(),
            }
    del basis, basis_cpu, models, input_gpu
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    threshold = PHYSICAL["threshold"]
    steepness = PHYSICAL["steepness"]
    doses = PHYSICAL["doses"]
    results = {}
    for source, aerials in method_aerials.items():
        metrics = {}
        masks_by_method = {}
        for method, aerial in aerials.items():
            metric, printed = hard_metrics_from_aerial(
                aerial, target, doses=doses, threshold=threshold, steepness=steepness
            )
            metrics[method] = metric
            masks_by_method[method] = printed
        methods = tuple(aerials)
        parity = {}
        for left_index, left in enumerate(methods):
            for right in methods[left_index + 1:]:
                left_aerial = aerials[left].numpy()
                right_aerial = aerials[right].numpy()
                difference = np.abs(left_aerial.astype(np.float64) - right_aerial.astype(np.float64))
                close = np.isclose(left_aerial, right_aerial,
                                   rtol=PHYSICAL["aerial_rtol"], atol=PHYSICAL["aerial_atol"])
                per_dose_mismatch = [
                    int(np.count_nonzero(a != b))
                    for a, b in zip(masks_by_method[left], masks_by_method[right])
                ]
                parity[f"{left}_vs_{right}"] = {
                    "aerial_allclose": bool(close.all()),
                    "aerial_rtol": PHYSICAL["aerial_rtol"],
                    "aerial_atol": PHYSICAL["aerial_atol"],
                    "aerial_max_abs_error": float(difference.max(initial=0.0)),
                    "hard_mask_mismatch_pixels_by_dose": per_dose_mismatch,
                    "hard_masks_exact": not any(per_dose_mismatch),
                }
        results[source] = {
            "metrics": metrics,
            "parity": parity,
            "parity_passed": all(
                item["aerial_allclose"] and item["hard_masks_exact"]
                for item in parity.values()
            ),
        }
    return results


def evaluate_methods(mask: np.ndarray, target: np.ndarray, weights: list[float],
                     device: str) -> dict:
    """Single-vector compatibility wrapper used by focused synthetic checks."""
    return evaluate_layout_methods(
        mask, target, {"single": {"weights": weights}}, device
    )["single"]


def _append_jsonl(path: Path, value: dict) -> None:
    # Progress readers consume one JSON object per physical line. Keep nested
    # values compact so a runtime error cannot make the durable journal unreadable.
    raw = (json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _bootstrap_ci(values: list[float], rng: np.random.Generator) -> list[float] | None:
    if not values:
        return None
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 1:
        return [float(vector[0]), float(vector[0])]
    indices = rng.integers(0, vector.size, size=(BOOTSTRAP_DRAWS, vector.size))
    means = vector[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _group_values(rows: list[dict], metric_key: str,
                  family: str | None = None) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        families = row.get("families", [row.get("family")])
        if family is not None and family not in families:
            continue
        metric = row["metrics"]["canonical_cpu_1thread_basis"]
        target_count = metric["target_positive_pixels"]
        if metric_key == "pvband_rate":
            value = metric["pvband_pixels"] / target_count
        else:
            value = float(metric[metric_key])
        grouped[row["cell_group"]].append(value)
    return {group: float(np.mean(values)) for group, values in sorted(grouped.items())}


def _scope_summary(records: dict[str, dict[str, dict]], layout_by_id: dict[str, dict],
                   family: str | None) -> dict:
    result = {}
    for source in ("event_candidate", "reference", "best_known"):
        rows = []
        for layout_id, methods in records.get(source, {}).items():
            if "canonical_cpu_1thread_basis" not in methods.get("metrics", {}):
                continue
            info = layout_by_id[layout_id]
            rows.append({**methods, "layout_id": layout_id,
                         "family": info["family"], "families": [info["family"]],
                         "cell_group": info["cell_group"]})
        result[source] = {
            "layout_count": sum(1 for row in rows if family is None or family in row["families"]),
            "layout_raw_counts": {},
            "group_mean_metrics": {},
        }
        for metric_key in ("pvband_rate", "nominal_error_fraction_of_target",
                           "worst_dose_error_fraction_of_target",
                           "nominal_absolute_relative_area_error",
                           "worst_dose_absolute_relative_area_error"):
            group_values = _group_values(rows, metric_key, family)
            result[source]["group_mean_metrics"][metric_key] = (
                float(np.mean(list(group_values.values()))) if group_values else None
            )
            result[source].setdefault("groups", {})[metric_key] = group_values
        selected = [row for row in rows if family is None or family in row["families"]]
        result[source]["layout_raw_counts"] = {
            "layout_count": len(selected),
            "target_positive_pixels": int(sum(row["metrics"]["canonical_cpu_1thread_basis"]["target_positive_pixels"] for row in selected)),
            "pvband_pixels": int(sum(row["metrics"]["canonical_cpu_1thread_basis"]["pvband_pixels"] for row in selected)),
            "nominal_error_pixels": int(sum(row["metrics"]["canonical_cpu_1thread_basis"]["nominal_error_pixels"] for row in selected)),
            "worst_dose_error_pixels": int(sum(row["metrics"]["canonical_cpu_1thread_basis"]["worst_dose_error_pixels"] for row in selected)),
            "candidate_blank_layout_count": sum(not row["metrics"]["canonical_cpu_1thread_basis"]["no_blank_positive_target_any_dose"] for row in selected),
            "parity_failed_layout_count": sum(not row["parity_passed"] for row in selected),
        }

    for baseline in ("reference", "best_known"):
        delta = {}
        rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if family is None else FAMILY_NAMES.index(family) + 1)
                                   + (100 if baseline == "best_known" else 0))
        for metric_key in ("pvband_rate", "nominal_error_fraction_of_target",
                           "worst_dose_error_fraction_of_target",
                           "nominal_absolute_relative_area_error",
                           "worst_dose_absolute_relative_area_error"):
            candidate_groups = result["event_candidate"].get("groups", {}).get(metric_key, {})
            baseline_groups = result[baseline].get("groups", {}).get(metric_key, {})
            shared = sorted(set(candidate_groups) & set(baseline_groups))
            diffs = [candidate_groups[group] - baseline_groups[group] for group in shared]
            delta[metric_key] = {
                "group_deltas": {group: candidate_groups[group] - baseline_groups[group] for group in shared},
                "equal_group_mean_delta": float(np.mean(diffs)) if diffs else None,
                "bootstrap_95_percentile_ci": _bootstrap_ci(diffs, rng),
                "wins_ties_losses": {
                    "wins": sum(value < -1e-12 for value in diffs),
                    "ties": sum(abs(value) <= 1e-12 for value in diffs),
                    "losses": sum(value > 1e-12 for value in diffs),
                },
                "paired_group_count": len(shared),
            }
        result[f"event_candidate_minus_{baseline}"] = delta
    return result


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid progress JSONL at line {line_number}") from exc
    return result


def _summarize(protocol: dict, records: list[dict], scored: list[dict],
               status: str, fatal_error: str | None, records_path: Path,
               protocol_sha256: str) -> dict:
    layouts = protocol["dataset"]["layouts"]
    layout_by_id = {row["layout_id"]: row for row in layouts}
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    # These are the exact source identifiers emitted by run().
    source_ids = {"event_candidate": "event_candidate", "reference": "reference", "best_known": "best_known"}
    for row in records:
        source = row.get("source")
        layout_id = row.get("layout_id")
        if source in source_ids and layout_id in layout_by_id and row.get("status") == "scored":
            by_source[source_ids[source]][layout_id] = row
    scopes = {"pooled": _scope_summary(by_source, layout_by_id, None)}
    for family in FAMILY_NAMES:
        scopes[family] = _scope_summary(by_source, layout_by_id, family)
    candidate_metrics = by_source.get("event_candidate", {})
    all_layout_ids = {layout["layout_id"] for layout in layouts}
    expected_rows = len(all_layout_ids) * 3
    completed_rows = sum(len(by_source.get(source, {})) for source in source_ids.values())
    full_scoring = completed_rows == expected_rows
    all_parity = full_scoring and all(
        row["parity_passed"]
        for source in source_ids.values() for row in by_source.get(source, {}).values()
    )
    no_blank = full_scoring and all(
        row["metrics"]["canonical_cpu_1thread_basis"]["no_blank_positive_target_any_dose"]
        for row in candidate_metrics.values()
    )
    pooled = scopes["pooled"]
    pv_ci = pooled.get("event_candidate_minus_reference", {}).get("pvband_rate", {}).get("bootstrap_95_percentile_ci")
    pv_upper_below_zero = bool(pv_ci and pv_ci[1] < 0.0)
    nominal_delta = pooled.get("event_candidate_minus_reference", {}).get("nominal_error_fraction_of_target", {}).get("equal_group_mean_delta")
    worst_delta = pooled.get("event_candidate_minus_reference", {}).get("worst_dose_error_fraction_of_target", {}).get("equal_group_mean_delta")
    nonregression = nominal_delta is not None and nominal_delta <= 0.0 and worst_delta is not None and worst_delta <= 0.0
    error_rows = sum(row.get("status") != "scored" for row in records)
    validations_passed = full_scoring and all_parity and error_rows == 0 and status == "complete"
    return {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "protocol_file_sha256": protocol_sha256,
        "status": status,
        "fatal_error": fatal_error,
        "scoring": {
            "completed_source_layout_records": completed_rows,
            "expected_source_layout_records": expected_rows,
            "full_source_layout_coverage": full_scoring,
            "error_record_count": error_rows,
            "scored_layout_count_by_source": {source: len(by_source.get(source, {})) for source in source_ids.values()},
        },
        "validation": {
            "passed": validations_passed,
            "exact_hard_mask_and_aerial_parity_passed": all_parity,
            "candidate_no_blank_positive_target_all_doses": no_blank,
            "numerical_parity_mismatch_policy": "all completed layout results retained; no mismatched layout excluded",
        },
        "primary_claim_gate": {
            "pvband_95_group_bootstrap_upper_ci_below_zero": pv_upper_below_zero,
            "mean_nominal_spatial_error_nonregression": nominal_delta is not None and nominal_delta <= 0.0,
            "mean_worst_dose_spatial_error_nonregression": worst_delta is not None and worst_delta <= 0.0,
            "candidate_has_no_blank_layout": no_blank,
            "complete_parity_validation": validations_passed,
            "eligible": bool(pv_upper_below_zero and nonregression and no_blank and validations_passed and status == "complete"),
        },
        "scope_summaries": scopes,
        "records_file_sha256": _sha256_file(records_path) if records_path.exists() else None,
        "sensitivity": scored,
    }


def _validate_protocol(protocol_path: Path, expected_sha256: str) -> tuple[dict, str]:
    protocol, raw = _read_json(protocol_path)
    actual_sha256 = _sha256_bytes(raw)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"protocol SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("status") != "frozen_preflight_only":
        raise ValueError("protocol is not a frozen preflight-only version-1 protocol")
    if protocol.get("physical") != PHYSICAL:
        raise ValueError("frozen physical protocol differs from the evaluator constants")
    if protocol.get("compute_policy") != COMPUTE_POLICY:
        raise ValueError("frozen compute policy differs from the evaluator constants")
    source_repository = protocol.get("source_repository", {})
    if source_repository.get("evaluator_script_sha256") != _sha256_file(SCRIPT_PATH):
        raise ValueError("evaluator script differs from the stable freeze pin")
    if source_repository.get("light_source_sha256") != _sha256_file(LIGHT_SOURCE_PATH):
        raise ValueError("light_source.py differs from the stable freeze pin")
    if source_repository.get("upstream_head") != EXPECTED_UPSTREAM_HEAD:
        raise ValueError("upstream dataset lineage pin changed")
    pins = protocol.get("input_pins", {})
    checked_pins = _validate_pinned_inputs(
        Path(pins["cached_plan_file"]), Path(pins["event_candidate_file"]),
        Path(pins["event_report_file"]),
    )
    if checked_pins != pins:
        raise ValueError("cached plan, selected candidate, or uncached report changed after freeze")
    dataset = protocol.get("dataset", {})
    if dataset.get("raw_file_counts") != EXPECTED_FILE_COUNTS or dataset.get("raw_file_count_total") != 436:
        raise ValueError("frozen GLP source file counts do not match the required 271+165 set")
    if dataset.get("deduplicated_layout_count") != len(dataset.get("layouts", [])):
        raise ValueError("frozen deduplicated layout count does not match its manifest")
    for source in dataset.get("source_files", []):
        if _sha256_file(Path(source["path"])) != source["raw_glp_sha256"]:
            raise ValueError(f"source GLP hash changed after freeze: {source['path']}")
    for layout in dataset.get("layouts", []):
        snapshot = Path(layout["target_snapshot"])
        if _sha256_file(snapshot) != layout["target_snapshot_sha256"]:
            raise ValueError(f"frozen target mask changed after freeze: {snapshot}")
        target = unpack_target(snapshot, PHYSICAL["canvas_size_px"])
        if _sha256_bytes(target.tobytes(order="C")) != layout["raster"]["raster_sha256"]:
            raise ValueError(f"frozen target raster hash mismatch: {layout['layout_id']}")
    return protocol, actual_sha256


def preflight(args: argparse.Namespace) -> dict:
    protocol, digest = _validate_protocol(Path(args.protocol), args.expected_protocol_sha256)
    return {
        "status": "preflight_passed",
        "protocol_file": str(Path(args.protocol).resolve()),
        "protocol_file_sha256": digest,
        "metadata_only": True,
        "optics_prepared": False,
        "source_layout_counts": protocol["dataset"]["raw_file_counts"],
        "raw_glp_count": protocol["dataset"]["raw_file_count_total"],
        "deduplicated_layout_count": protocol["dataset"]["deduplicated_layout_count"],
        "pooled_cell_group_count": protocol["dataset"]["pooled_cell_group_count"],
        "event_candidate_id": protocol["input_pins"]["event_candidate_id"],
        "event_candidate_sha256": protocol["input_pins"]["event_candidate_file_sha256"],
        "event_report_sha256": protocol["input_pins"]["event_report_file_sha256"],
        "source_vector_hashes": {key: {field: value for field, value in vector.items() if field.endswith("sha256")}
                                 for key, vector in protocol["source_vectors"].items()},
        "compute_policy": protocol["compute_policy"],
    }


def _claim_error(original: dict, source: str, layout_id: str) -> dict:
    return {"status": "error", "source": source, "layout_id": layout_id,
            "error": original.get("error", "unknown error"), "error_kind": original.get("error_kind", "runtime")}


def _raster_for_sensitivity(layout: dict, canvas_size_px: int) -> tuple[np.ndarray, dict]:
    parsed = parse_glp(Path(layout["source_glp"]))
    mask, info = rasterize_polygons(parsed["polygons"], parsed["bbox_nm"], canvas_size_px=canvas_size_px)
    if parsed["raw_sha256"] != layout["raw_glp_sha256"]:
        raise ValueError(f"source GLP changed after freeze: {layout['source_glp']}")
    return mask, info


def run(args: argparse.Namespace) -> dict:
    protocol_path = Path(args.protocol).resolve()
    protocol, protocol_sha256 = _validate_protocol(protocol_path, args.expected_protocol_sha256)
    frozen_budget = int(protocol["runtime_policy"]["time_budget_seconds"])
    if args.time_budget_seconds <= 0 or args.time_budget_seconds > frozen_budget:
        raise ValueError("run time budget must be positive and cannot exceed the frozen protocol budget")

    run_dir = Path(args.run_dir).resolve()
    family_roots = [Path(value) for value in protocol["dataset"]["family_directories"].values()]
    _assert_output_outside_repo(run_dir, family_roots)
    if run_dir.exists():
        existing = [name for name in ("consumed.marker", "progress.json", "layout_results.jsonl",
                                      "padding_sensitivity.jsonl", "result.json") if (run_dir / name).exists()]
        if existing:
            raise FileExistsError(f"run output already contains {existing}; consumed run directories cannot be reused")
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "consumed.marker"
    # Only after protocol and file-hash preflight succeeds do we consume the
    # unique marker. It still precedes every optical import or scoring call.
    fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    marker_content = {
        "consumed_utc": _utc_now(),
        "run_nonce": uuid.uuid4().hex,
        "protocol_file": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "launcher_git_head": _git_head(),
    }
    with os.fdopen(fd, "wb") as stream:
        stream.write(_json_bytes(marker_content))
        stream.flush()
        os.fsync(stream.fileno())
    progress_path = run_dir / "progress.json"
    records_path = run_dir / "layout_results.jsonl"
    final_path = run_dir / "result.json"
    progress = {"status": "partial", "phase": "consumed", "started_utc": _utc_now(),
                "completed_records": 0, "expected_records": None, "last_layout_id": None, "last_source": None,
                "error": None}
    _atomic_json(progress_path, progress)
    fatal_error = None
    sensitivity_records = []
    start_time = time.monotonic()
    status = "partial"
    try:
        layouts = protocol["dataset"]["layouts"]
        expected_records = len(layouts) * 3
        progress.update({"phase": "scoring", "expected_records": expected_records})
        _atomic_json(progress_path, progress)

        # Recheck every raw GLP and frozen packed target before scoring. These
        # checks happen after the consumed marker but before optical setup.
        pins = protocol["input_pins"]
        if _validate_pinned_inputs(Path(pins["cached_plan_file"]),
                                   Path(pins["event_candidate_file"]),
                                   Path(pins["event_report_file"])) != pins:
            raise ValueError("source, candidate, or report pins changed after protocol preflight")
        for source_file in protocol["dataset"]["source_files"]:
            if _sha256_file(Path(source_file["path"])) != source_file["raw_glp_sha256"]:
                raise ValueError(f"source GLP hash changed after preflight: {source_file['path']}")
        for layout in layouts:
            if _sha256_file(Path(layout["source_glp"])) != layout["raw_glp_sha256"]:
                raise ValueError(f"source GLP hash changed after freeze: {layout['source_glp']}")
            snapshot = Path(layout["target_snapshot"])
            if _sha256_file(snapshot) != layout["target_snapshot_sha256"]:
                raise ValueError(f"frozen target mask changed after freeze: {snapshot}")
            unpacked = unpack_target(snapshot, PHYSICAL["canvas_size_px"])
            if _sha256_bytes(unpacked.tobytes(order="C")) != layout["raster"]["raster_sha256"]:
                raise ValueError(f"frozen target raster hash mismatch: {layout['layout_id']}")

        vectors = protocol["source_vectors"]
        source_order = ("event_candidate", "reference", "best_known")
        completed = 0
        for layout in layouts:
            target = unpack_target(Path(layout["target_snapshot"]), PHYSICAL["canvas_size_px"])
            if layout["raster"]["positive_pixel_count"] != int(target.sum()):
                raise ValueError(f"target positive-pixel count changed after freeze: {layout['layout_id']}")
            mask = target.astype(np.float32, copy=False)
            if time.monotonic() - start_time >= args.time_budget_seconds:
                raise TimeoutError("frozen evaluation wall-clock budget expired")
            progress.update({"last_layout_id": layout["layout_id"], "last_source": "all_sources_basis_reuse",
                             "completed_records": completed})
            _atomic_json(progress_path, progress)
            try:
                layout_scores = evaluate_layout_methods(mask, target, vectors, args.device)
            except RuntimeError as exc:
                error = str(exc)
                for source in source_order:
                    _append_jsonl(records_path, _claim_error({"error": error, "error_kind": "runtime"}, source, layout["layout_id"]))
                fatal_error = f"runtime failure at {layout['layout_id']}: {error}"
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                for source in source_order:
                    _append_jsonl(records_path, _claim_error({"error": error, "error_kind": "input_or_geometry"}, source, layout["layout_id"]))
                fatal_error = f"fatal failure at {layout['layout_id']}: {error}"
                raise
            for source in source_order:
                record = {
                    "status": "scored",
                    "source": source,
                    "weights_f64_sha256": vectors[source]["weights_f64_sha256"],
                    "weights_f32_sha256": vectors[source]["weights_f32_sha256"],
                    "layout_id": layout["layout_id"],
                    "family": layout["family"],
                    "cell_group": layout["cell_group"],
                    "target_raster_sha256": layout["raster"]["raster_sha256"],
                    "target_positive_pixels": layout["raster"]["positive_pixel_count"],
                    **layout_scores[source],
                    "completed_utc": _utc_now(),
                }
                _append_jsonl(records_path, record)
                completed += 1
                progress.update({"completed_records": completed, "last_source": source, "last_error": None})
                _atomic_json(progress_path, progress)
            if time.monotonic() - start_time >= args.time_budget_seconds:
                raise TimeoutError("frozen evaluation wall-clock budget expired after persisting the completed layout")

        # Prespecified, non-primary padding check on the first lexicographic
        # original GLP from each family. The 1024-pixel values reuse primary
        # scored rows; the 512-pixel runs use the same source vectors.
        progress.update({"phase": "padding_sensitivity", "completed_records": completed})
        _atomic_json(progress_path, progress)
        source_files = protocol["dataset"]["source_files"]
        for family in FAMILY_NAMES:
            first_source = next(row for row in source_files if row["family"] == family)
            representative = next(
                layout for layout in layouts
                if layout["family"] == family
                and any(alias["relative_path"] == first_source["relative_path"] for alias in layout["aliases"])
            )
            mask512, raster512 = _raster_for_sensitivity(representative, PHYSICAL["sensitivity_canvas_size_px"])
            if time.monotonic() - start_time >= args.time_budget_seconds:
                raise TimeoutError("frozen evaluation wall-clock budget expired during padding sensitivity")
            sensitivity_by_source = evaluate_layout_methods(
                mask512.astype(np.float32, copy=False), mask512, vectors, args.device
            )
            for source in source_order:
                scored512 = sensitivity_by_source[source]
                # Primary layout rows are keyed by both source and layout.
                primary_row = next(row for row in _read_records(records_path)
                                   if row.get("status") == "scored" and row.get("source") == source
                                   and row.get("layout_id") == representative["layout_id"])
                sensitivity_records.append({
                    "family": family,
                    "layout_id": representative["layout_id"],
                    "source": source,
                    "canvas_size_px": PHYSICAL["sensitivity_canvas_size_px"],
                    "raster": raster512,
                    "metrics_512": scored512["metrics"]["canonical_cpu_1thread_basis"],
                    "metrics_1024_primary": primary_row["metrics"]["canonical_cpu_1thread_basis"],
                    "pvband_fraction_delta_512_minus_1024": (
                        scored512["metrics"]["canonical_cpu_1thread_basis"]["pvband_fraction_of_target"]
                        - primary_row["metrics"]["canonical_cpu_1thread_basis"]["pvband_fraction_of_target"]
                    ),
                    "parity_512": scored512["parity"],
                    "parity_passed_512": scored512["parity_passed"],
                })
                _append_jsonl(run_dir / "padding_sensitivity.jsonl", sensitivity_records[-1])

        # Recheck all frozen inputs after scoring so a change during the run
        # cannot produce a complete result from a stale protocol snapshot.
        _validate_protocol(protocol_path, protocol_sha256)
        status = "complete"
    except TimeoutError as exc:
        status = "partial"
        fatal_error = fatal_error or str(exc)
    except Exception as exc:
        status = "partial"
        fatal_error = fatal_error or f"{type(exc).__name__}: {exc}"
        progress["traceback"] = traceback.format_exc(limit=8)

    progress.update({"status": status, "phase": "finished", "completed_utc": _utc_now(),
                     "completed_records": len([row for row in _read_records(records_path) if row.get("status") == "scored"]),
                     "fatal_error": fatal_error,
                     "elapsed_seconds": time.monotonic() - start_time})
    _atomic_json(progress_path, progress)
    result = _summarize(protocol, _read_records(records_path), sensitivity_records,
                        status, fatal_error, records_path, protocol_sha256)
    result["run"] = {
        "started_utc": marker_content["consumed_utc"],
        "completed_utc": _utc_now(),
        "device": args.device,
        "wall_seconds": progress["elapsed_seconds"],
        "progress_file": str(progress_path.resolve()),
        "records_file": str(records_path.resolve()),
        "consumed_marker": str(marker.resolve()),
        "consumed_marker_sha256": _sha256_file(marker),
        "compute_policy": COMPUTE_POLICY,
        "compute_environment": (
            _compute_environment(sys.modules["torch"], args.device)
            if "torch" in sys.modules else None
        ),
    }
    result["sensitivity_file_sha256"] = _sha256_file(run_dir / "padding_sensitivity.jsonl") if (run_dir / "padding_sensitivity.jsonl").exists() else None
    _atomic_json(final_path, result)
    progress["result_file"] = str(final_path.resolve())
    progress["result_file_sha256"] = _sha256_file(final_path)
    _atomic_json(progress_path, progress)
    return {"status": status, "result_file": str(final_path),
            "completed_source_layout_records": result["scoring"]["completed_source_layout_records"],
            "expected_source_layout_records": result["scoring"]["expected_source_layout_records"],
            "validation_passed": result["validation"]["passed"],
            "primary_claim_eligible": result["primary_claim_gate"]["eligible"],
            "fatal_error": fatal_error}


def _add_common_freeze_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cached-plan", required=True, help="Frozen cached candidate plan JSON outside Git")
    parser.add_argument("--event-candidate", required=True, help="Pinned selected uncached event candidate JSON")
    parser.add_argument("--event-report", required=True, help="Pinned uncached benchmark report JSON")
    parser.add_argument("--upstream-head", default=EXPECTED_UPSTREAM_HEAD, help="Pinned upstream LithoBench commit")
    parser.add_argument("--time-budget-seconds", type=int, default=21600)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze", help="Freeze independent GLP inputs and raster targets")
    freeze_parser.add_argument("--metal-dir", required=True, help="Directory containing all StdMetal GLPs (271 files)")
    freeze_parser.add_argument("--contact-dir", required=True, help="Directory containing all StdContact GLPs (165 files)")
    freeze_parser.add_argument("--output-dir", required=True, help="New, unique external run directory")
    _add_common_freeze_args(freeze_parser)
    preflight_parser = subparsers.add_parser("preflight", help="Verify immutable protocol metadata without optics")
    preflight_parser.add_argument("--protocol", required=True, help="Frozen protocol.json")
    preflight_parser.add_argument("--expected-protocol-sha256", required=True)
    run_parser = subparsers.add_parser("run", help="Consume a unique marker and score the frozen sources")
    run_parser.add_argument("--protocol", required=True, help="Frozen protocol.json")
    run_parser.add_argument("--run-dir", required=True, help="Unique external run directory; never reuse")
    run_parser.add_argument("--expected-protocol-sha256", required=True)
    run_parser.add_argument("--device", default="cuda", help="Torch device, normally cuda")
    run_parser.add_argument("--time-budget-seconds", type=int, default=21600)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            output = freeze(args)
        elif args.command == "preflight":
            output = preflight(args)
        else:
            output = run(args)
    except Exception as exc:
        parser.exit(2, f"ERROR: {type(exc).__name__}: {exc}\n")
    print(json.dumps(output, sort_keys=True))
    return 0 if output.get("status") in {"frozen_preflight_only", "preflight_passed", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
