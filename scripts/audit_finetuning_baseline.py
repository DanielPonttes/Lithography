#!/usr/bin/env python3
"""Read-only, stdlib-only inventory for a NeuralILT fine-tuning audit.

This tool hashes evidence and reads PNG headers. It never imports
PyTorch, loads a checkpoint, runs a model, or starts a training/benchmark job.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import zlib


CHECKPOINT = "work/MetalSet_NeuralILT/net.pth"
GLPS = [f"benchmark/ICCAD2013/M1_test{i}.glp" for i in range(1, 11)]
CONFIGS = [
    "config/lithosimple.txt",
    "config/curvilt512.txt",
    "config/curvilt1024.txt",
    "config/simpleilt.txt",
]
SOURCE_FILES = [
    "lithobench/ilt/neuralilt.py",
    "lithobench/model.py",
    "lithobench/evaluate.py",
    "pylitho/exact.py",
    "pyilt/evaluation.py",
    "pycommon/settings.py",
    "lithobench/dataset.py",
    "lithobench/train.py",
    "lithobench/test.py",
]
REQUIRED_SOURCE_FILES = {"lithobench/ilt/neuralilt.py", "lithobench/train.py", "lithobench/dataset.py"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(root: Path, relative: str) -> dict:
    path = root / relative
    record = {"path": relative, "status": "missing"}
    try:
        if not path.is_file():
            return record
        stat = path.stat()
        record.update(
            status="present",
            size_bytes=stat.st_size,
            sha256=sha256_file(path),
        )
    except OSError as exc:
        record.update(status="unreadable", error=type(exc).__name__)
    return record


def git_record(root: Path) -> dict:
    result = {"available": False, "commit": None, "dirty": None, "status_short": None}
    safe_root = str(root)

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-c", f"safe.directory={safe_root}", "-C", safe_root, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=15,
        )

    try:
        commit = run("rev-parse", "--verify", "HEAD")
        status = run("status", "--porcelain=v1", "--branch", "--untracked-files=normal")
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = type(exc).__name__
        return result

    if commit.returncode != 0 or status.returncode != 0:
        result["error"] = "git command failed"
        return result
    status_lines = status.stdout.splitlines()
    result.update(
        available=True,
        commit=commit.stdout.strip(),
        dirty=any(line and not line.startswith("##") for line in status_lines),
        status_short=status_lines,
    )
    return result


def png_record(path: Path) -> dict:
    """Read PNG IHDR only; pixel values and binary-mask status remain unknown."""
    result = {"path": path.name, "status": "unreadable", "pixel_binary_verified": None}
    try:
        with path.open("rb") as stream:
            if stream.read(8) != PNG_SIGNATURE:
                return {**result, "status": "invalid_png_signature"}
            chunk_head = stream.read(8)
            if len(chunk_head) != 8:
                return {**result, "status": "truncated_png"}
            length, kind = struct.unpack(">I4s", chunk_head)
            if kind != b"IHDR" or length != 13:
                return {**result, "status": "invalid_png_header"}
            data = stream.read(13)
            crc_bytes = stream.read(4)
            if len(data) != 13 or len(crc_bytes) != 4:
                return {**result, "status": "truncated_png"}
            if (zlib.crc32(kind + data) & 0xFFFFFFFF) != struct.unpack(">I", crc_bytes)[0]:
                return {**result, "status": "png_crc_mismatch"}
        header = struct.unpack(">IIBBBBB", data)
        width, height, bit_depth, color_type, compression, filtering, interlace = header
        result.update(
            status="header_read",
            width=width,
            height=height,
            bit_depth=bit_depth,
            color_type=color_type,
            interlace=interlace,
            pixel_binary_verified=None,
            pixel_binary_note="not checked; IHDR does not establish pixel values or mask provenance",
        )
    except (OSError, ValueError, struct.error, zlib.error) as exc:
        result.update(status="unreadable", error=type(exc).__name__)
    return result


def png_mask_inventory(root: Path) -> dict:
    directory = root / "saved/MetalSet_NeuralILT"
    found = []
    if directory.is_dir():
        for path in sorted(directory.rglob("*.png")):
            try:
                relative = path.relative_to(root).as_posix()
                record = file_record(root, relative)
                png = png_record(path)
                record["png_header_status"] = png.get("status")
                record["png_ihdr"] = {key: value for key, value in png.items() if key not in {"path", "status"}}
                label = re.search(r"mask([01])", path.as_posix(), flags=re.IGNORECASE)
                record["filename_label"] = f"mask{label.group(1)}" if label else None
                found.append(record)
            except OSError:
                continue
    counts = {f"mask{i}": sum(item["filename_label"] == f"mask{i}" for item in found) for i in (0, 1)}
    return {
        "directory": "saved/MetalSet_NeuralILT",
        "expected_png_count": 20,
        "found_png_count": len(found),
        "filename_label_counts": counts,
        "files": found,
        "provenance_verified": False,
    }


def matching_files(root: Path, predicate, walk_root: Path | None = None) -> list[dict]:
    records = []
    walk_root = walk_root or root
    if not walk_root.is_dir():
        return records
    for base, dirs, files in os.walk(walk_root):
        dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS)
        for name in sorted(files):
            if predicate(name):
                path = Path(base) / name
                relative = path.relative_to(root).as_posix()
                records.append(file_record(root, relative))
    return records


def source_markers(root: Path, records: list[dict]) -> list[dict]:
    markers = []
    patterns = {
        "finetuneFast": re.compile(r"finetuneFast", re.IGNORECASE),
        "train/evaluate entry points": re.compile(r"\bdef\s+(train|evaluate)\b"),
        "MSE or PV terms": re.compile(r"mseNom|mseMaxMin|PVBL2|pvband|MSELoss", re.IGNORECASE),
        "kernel/scaling loaders": re.compile(r"kernel(s)?\.pt|scale(s)?\.pt", re.IGNORECASE),
    }
    for record in records:
        relative = record.get("path")
        if record.get("status") != "present" or not relative or not relative.endswith(".py"):
            continue
        try:
            lines = (root / relative).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        matches = {}
        for label, pattern in patterns.items():
            numbers = [i for i, line in enumerate(lines, 1) if pattern.search(line)]
            if numbers:
                matches[label] = numbers[:50]
        if matches:
            markers.append({"path": relative, "line_markers": matches})
    return markers


def config_observations(root: Path, records: list[dict]) -> list[dict]:
    pattern = re.compile(r"\b(alpha|threshold|wavelength|dose|focus|na)\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
    observations = []
    for record in records:
        if record.get("status") != "present":
            continue
        relative = record["path"]
        try:
            lines = (root / relative).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            for match in pattern.finditer(line):
                observations.append({
                    "path": relative,
                    "line": number,
                    "key": match.group(1).lower(),
                    "numeric_value": float(match.group(2)),
                })
    return observations


def build_inventory(root: Path) -> dict:
    root = root.resolve()
    git = git_record(root)
    checkpoint = file_record(root, CHECKPOINT)
    glps = [file_record(root, path) for path in GLPS]
    configs = [file_record(root, path) for path in CONFIGS]
    source = [file_record(root, path) for path in SOURCE_FILES]
    source += matching_files(
        root,
        lambda name: name.lower() in {"neuralilt.py", "model.py", "evaluate.py", "evaluation.py", "exact.py", "settings.py"},
        walk_root=root / "lithobench",
    )
    unique = {record["path"]: record for record in source}
    source = [unique[key] for key in sorted(unique)]
    kernels = []
    for directory in (
        "kernel", "kernels", "kernel/scales", "kernels/scales", "scales",
        "pylitho/kernel", "pylitho/kernels", "pylitho/kernel/scales",
        "pylitho/kernels/scales", "pylitho/scales",
    ):
        kernels.extend(
            matching_files(
                root,
                lambda name: Path(name).suffix.lower() == ".pt",
                walk_root=root / directory,
            )
        )
    kernels = sorted({item["path"]: item for item in kernels}.values(), key=lambda item: item["path"])
    masks = png_mask_inventory(root)
    checkpoints = matching_files(
        root,
        lambda name: Path(name).suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors"},
        walk_root=root / "work/MetalSet_NeuralILT",
    )
    pv_candidates = matching_files(
        root,
        lambda name: Path(name).suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors"}
        and re.search(r"finetun|pv.?aware", name, flags=re.IGNORECASE) is not None,
        walk_root=root / "work",
    )
    markers = source_markers(root, source)

    missing = []
    for record in [checkpoint, *glps, *configs]:
        if record.get("status") == "missing":
            missing.append(record["path"])
    for record in source:
        if record["path"] in REQUIRED_SOURCE_FILES and record.get("status") == "missing":
            missing.append(record["path"])
    if masks["found_png_count"] != masks["expected_png_count"]:
        missing.append(f"expected 20 saved masks, found {masks['found_png_count']}")
    if not kernels:
        missing.append("kernel/kernels/scales .pt assets not found")
    blockers = [
        "No PV-aware fine-tuned checkpoint with verified provenance is established by this inventory.",
        "Hash inventory is a snapshot only; re-hash all inputs immediately before any scoring.",
    ]
    if checkpoint["status"] != "present":
        blockers.append(f"Expected checkpoint missing or unreadable: {CHECKPOINT}.")
    if not pv_candidates:
        blockers.append("No filename-marked PV-aware checkpoint candidate was found under work/.")
    if any(item["status"] != "present" for item in glps):
        blockers.append("The complete ten-layout M1_test1..10 holdout set was not verified.")
    if any(item["status"] != "present" for item in configs):
        blockers.append("One or more optical/ILT configuration files are missing or unreadable.")
    if any(item["path"] in REQUIRED_SOURCE_FILES and item.get("status") != "present" for item in source):
        blockers.append("Required upstream training/dataset/model source is missing or unreadable.")
    if masks["found_png_count"] != masks["expected_png_count"]:
        blockers.append("The expected twenty saved diagnostic mask PNGs were not all found.")
    if not kernels:
        blockers.append("No kernel/scaling .pt assets were found in the searched directories.")
    if git.get("dirty"):
        blockers.append("Upstream working tree is dirty; review and preserve the reported diff before replay.")

    return {
        "schema_version": 1,
        "status": "audit_only",
        "paper_baseline_reproduced": False,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "upstream_root": str(root),
        "git": git,
        "artifacts": {
            "init_checkpoint_expected": checkpoint,
            "checkpoint_files_in_model_work_dir": checkpoints,
            "filename_marked_pv_aware_checkpoint_candidates": pv_candidates,
            "glp_holdout_expected": glps,
            "saved_masks": masks,
            "configs_expected": configs,
            "config_numeric_observations": config_observations(root, configs),
            "source_files": source,
            "source_markers": markers,
            "kernel_and_scale_pt_files": kernels,
        },
        "missing": sorted(set(missing)),
        "blockers": blockers,
        "interpretation": {
            "checkpoint_provenance_verified": False,
            "saved_pngs_prove_model_or_training_provenance": False,
            "png_binary_pixels_verified": False,
            "model_or_training_executed": False,
            "credentials_or_environment_variables_collected": False,
        },
        "reviewed_comparison_reference": {
            "paper_alpha": 85,
            "inspected_upstream_lithosimple_alpha": 50,
            "paper_objective": "L2 + 0.1 * PV",
            "inspected_upstream_neuralilt_train_objective": "MSE nominal plus MSE min/max PV term, unit weight",
            "alpha_difference_proves_binary_mask_difference": False,
            "note": "Review reference for the inspected upstream snapshot; compare observed config values and source hashes before relying on it.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path, help="clone raiz do LithoBench a inventariar")
    parser.add_argument("--output", required=True, type=Path, help="novo arquivo JSON; nunca sobrescreve arquivo existente")
    args = parser.parse_args(argv)
    root = args.upstream_root.expanduser()
    output = args.output.expanduser()
    if not root.is_dir():
        parser.error(f"upstream root não é um diretório: {root}")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        inventory = build_inventory(root)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(inventory, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
    except FileExistsError:
        print(f"refusing to overwrite existing output: {output}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"audit could not be written ({type(exc).__name__}): {output}", file=sys.stderr)
        return 2
    print(f"Wrote audit-only inventory: {output} ({len(inventory['missing'])} missing, {len(inventory['blockers'])} blockers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
