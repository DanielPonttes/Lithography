#!/usr/bin/env python3
"""Preflight or explicitly train a reconstructed, non-original PV-aware NeuralILT.

Default operation uses only the Python standard library and writes a hashed
manifest. Training imports PyTorch/upstream code only after explicit opt-in.
This tool never claims to reproduce the unavailable paper checkpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time


PINNED_COMMIT = "9c74e82218e377eaf6d02d113fc1ce6e36c92aa6"
DATA_ROOT = Path("work/MetalSet")
CHECKPOINT_CANDIDATE = Path("work/MetalSet_NeuralILT/net.pth")
TEST_GLPS = [Path(f"benchmark/ICCAD2013/M1_test{i}.glp") for i in range(1, 11)]
CONFIG_PATHS = [
    Path("config/lithosimple.txt"),
    Path("config/curvilt512.txt"),
    Path("config/curvilt1024.txt"),
    Path("config/simpleilt.txt"),
]
SOURCE_PATHS = [
    Path("lithobench/ilt/neuralilt.py"),
    Path("lithobench/dataset.py"),
    Path("lithobench/model.py"),
    Path("lithobench/evaluate.py"),
    Path("lithobench/train.py"),
    Path("pylitho/exact.py"),
    Path("pyilt/evaluation.py"),
    Path("pycommon/settings.py"),
    Path("pycommon/utils.py"),
    Path("config/lithosimple.txt"),
]
KERNEL_PATHS = [
    Path("kernel/kernels/focus.pt"),
    Path("kernel/kernels/defocus.pt"),
    Path("kernel/kernels/ct_focus.pt"),
    Path("kernel/kernels/ct_defocus.pt"),
    Path("kernel/kernels/combo_focus.pt"),
    Path("kernel/kernels/combo_defocus.pt"),
    Path("kernel/kernels/combo_ct_focus.pt"),
    Path("kernel/kernels/combo_ct_defocus.pt"),
    Path("kernel/scales/focus.pt"),
    Path("kernel/scales/defocus.pt"),
    Path("kernel/scales/combo.pt"),
]
RECIPE = {
    "label": "reconstruction_not_replication",
    "input_size": [512, 512],
    "pretrain_epochs": 50,
    "finetune_epochs": 20,
    "batch_size": 4,
    "learning_rate": 0.001,
    "pretrain_loss": "mean squared error between UNet mask and pixelILT label",
    "finetune_loss": "MSE(printed_nominal, target) + 0.1 * mean(abs(printed_max - printed_min))",
    "mask_data_crop_and_flip": "upstream DataILT train path; 512 input, random crop/flips",
    "optimizer": "Adam; new optimizer for each stage; constant learning rate",
    "seed": 17,
    "amp": False,
    "tf32": False,
    "resume_supported": False,
    "bitwise_reproducible": False,
    "physical_raster_calibration_established": False,
    "quality_claim_eligible": False,
    "test_glps_used_for_training_or_validation": False,
    "optical_note": "LithoSim uses the pinned precomputed kernels; alpha=85 is an explicit config-copy override. Kernel calibration to the article's physical focus window is unverified.",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(root: Path, relative: Path) -> dict:
    path = root / relative
    result = {"path": relative.as_posix(), "status": "missing"}
    try:
        if not path.is_file():
            return result
        st = path.stat()
        result.update(status="present", size_bytes=st.st_size, sha256=sha256_file(path))
    except OSError as exc:
        result.update(status="unreadable", error=type(exc).__name__)
    return result


def git_state(root: Path) -> dict:
    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
            check=False, capture_output=True, text=True, timeout=20,
        )

    try:
        commit = run("rev-parse", "--verify", "HEAD")
        status = run("status", "--porcelain=v1", "--branch", "--untracked-files=normal")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__, "commit": None, "dirty": None}
    if commit.returncode or status.returncode:
        return {"available": False, "error": "git command failed", "commit": None, "dirty": None}
    lines = status.stdout.splitlines()
    return {
        "available": True,
        "commit": commit.stdout.strip(),
        "dirty": any(line and not line.startswith("##") for line in lines),
        "status_short": lines,
    }


def png_dimensions(path: Path) -> tuple[int, int] | None:
    """Inspect only PNG signature/IHDR; no pixel decoding or image library."""
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            return None
        head = stream.read(8)
        if len(head) != 8:
            return None
        length = int.from_bytes(head[:4], "big")
        if head[4:] != b"IHDR" or length != 13:
            return None
        data = stream.read(13)
        if len(data) != 13:
            return None
        return (int.from_bytes(data[:4], "big"), int.from_bytes(data[4:8], "big"))


def files_by_stem(directory: Path, suffix: str) -> dict[str, Path]:
    path = directory
    if not path.is_dir():
        return {}
    return {p.stem: p for p in path.glob(f"*{suffix}") if p.is_file()}


def config_dict(root: Path, relative: Path = Path("config/lithosimple.txt")) -> dict[str, str]:
    values: dict[str, str] = {}
    with (root / relative).open("r", encoding="utf-8") as stream:
        for line in stream:
            parts = line.strip().split()
            if len(parts) >= 2:
                values[parts[0]] = parts[1]
    return values


def build_manifest(root_arg: Path) -> dict:
    root = root_arg.expanduser().resolve()
    git = git_state(root)
    glp_files = files_by_stem(root / DATA_ROOT / "glp", ".glp")
    target_files = files_by_stem(root / DATA_ROOT / "target", ".png")
    pixel_files = files_by_stem(root / DATA_ROOT / "pixelILT", ".png")
    pair_ids = sorted(set(glp_files) & set(target_files) & set(pixel_files))
    test_records = [file_record(root, path) for path in TEST_GLPS]
    test_hashes = {record["sha256"] for record in test_records if record.get("sha256")}
    test_ids = {path.stem for path in TEST_GLPS}

    pairs = []
    eligible = []
    overlap_count = 0
    for pair_id in pair_ids:
        paths = {
            "glp": glp_files[pair_id],
            "target": target_files[pair_id],
            "pixelILT": pixel_files[pair_id],
        }
        records = {kind: file_record(root, path.relative_to(root)) for kind, path in paths.items()}
        same_hash = records["glp"].get("sha256") in test_hashes
        same_id = pair_id in test_ids
        overlap = same_hash or same_id
        if overlap:
            overlap_count += 1
        else:
            eligible.append(pair_id)
        dimensions = {}
        for kind, path in paths.items():
            try:
                dims = png_dimensions(path) if path.suffix.lower() == ".png" else None
            except OSError:
                dims = None
            dimensions[kind] = list(dims) if dims else None
        pairs.append({
            "id": pair_id,
            "files": records,
            "png_dimensions": dimensions,
            "split": "excluded_test_glp_overlap" if overlap else None,
            "test_glp_overlap": {"basename": same_id, "sha256": same_hash},
        })

    train_count = round(len(eligible) * 0.9)
    train_ids = set(eligible[:train_count])
    val_ids = set(eligible[train_count:])
    for pair in pairs:
        if pair["split"] is None:
            pair["split"] = "train" if pair["id"] in train_ids else "validation"

    config_inventory = []
    original_config = {}
    config_error = None
    for relative in CONFIG_PATHS:
        record = file_record(root, relative)
        try:
            values = config_dict(root, relative) if record["status"] == "present" else {}
            parse_error = None if values else "empty_or_unparsed"
        except (OSError, UnicodeError) as exc:
            values = {}
            parse_error = type(exc).__name__
        config_inventory.append({"file": record, "values": values, "parse_error": parse_error})
        if relative == Path("config/lithosimple.txt"):
            original_config = values
            config_error = parse_error
    effective_config = dict(original_config)
    effective_config["KernelDir"] = str((root / "kernel").resolve())
    effective_config["PrintSteepness"] = "85.0"

    source_records = [file_record(root, path) for path in SOURCE_PATHS]
    kernel_records = [file_record(root, path) for path in KERNEL_PATHS]
    test_holdout_hashes = sorted(test_hashes)
    manifest = {
        "schema_version": 1,
        "status": "reconstruction_manifest",
        "label": "reconstruction_not_replication",
        "paper_baseline_reproduced": False,
        "upstream_root": str(root),
        "upstream_git": git,
        "required_upstream_commit": PINNED_COMMIT,
        "commit_matches_pin": git.get("commit") == PINNED_COMMIT,
        "source_hashes": source_records,
        "config_inventory": config_inventory,
        "kernel_hashes": kernel_records,
        "runner": {
            "path": "scripts/reconstruct_pvaware_baseline.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "old_checkpoint_candidate_not_used": file_record(root, CHECKPOINT_CANDIDATE),
        "dataset": {
            "root": DATA_ROOT.as_posix(),
            "pairing": "intersection of glp/target/pixelILT basenames, lexically sorted as upstream filesMaskOpt",
            "paired_layout_count": len(pairs),
            "pairs": pairs,
            "training_pair_count": len(train_ids),
            "validation_pair_count": len(val_ids),
            "excluded_test_glp_overlap_pair_count": overlap_count,
            "split_rule": "first round(0.9*N) sorted non-test-overlap pairs train; remaining pairs validation",
            "test_glps": test_records,
            "all_ten_test_glps_present_and_hashed": len(test_records) == 10 and all(r.get("status") == "present" for r in test_records),
            "test_glp_sha256_set": test_holdout_hashes,
            "test_overlap_rule": "exclude when basename or full-file SHA-256 matches any of the ten test GLPs",
            "test_overlap_check_complete": len(test_records) == 10 and all(r.get("status") == "present" for r in test_records),
            "renamed_or_transformed_layout_aliases_detected": False,
        },
        "source_config": {
            "path": "config/lithosimple.txt",
            "parse_error": config_error,
            "original_values": original_config,
            "training_copy_overrides": {"KernelDir": effective_config.get("KernelDir"), "PrintSteepness": "85.0"},
            "effective_values": effective_config,
            "source_file_is_modified": False,
        },
        "recipe": RECIPE,
        "limitations": [
            "This is a seeded reconstruction using the audited upstream architecture and declared loss; it is not the paper's original PV-aware checkpoint or a claim of exact reproduction.",
            "The 512-pixel raster's physical calibration is unestablished; do not make optical-quality claims from this run.",
            "The upstream precomputed focus/defocus kernels do not establish the article's +/-25 nm focus calibration.",
            "Filename and exact-byte GLP checks do not detect renamed or transformed aliases of a held-out layout.",
            "No checkpoint resume implementation is offered; an interrupted run must be restarted from a new unique output folder.",
            "The ten paper test GLPs are for a separate evaluation phase and are never included in train/validation pairs.",
        ],
    }
    return manifest


def canonical_sha(manifest: dict) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(encoded)


def write_exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def validate_manifest(manifest: dict) -> list[str]:
    errors = []
    if not manifest["commit_matches_pin"]:
        errors.append(f"upstream HEAD must be {PINNED_COMMIT}")
    if not manifest["dataset"]["all_ten_test_glps_present_and_hashed"]:
        errors.append("the ten paper holdout GLPs are not all present/readable")
    if not manifest["dataset"]["pairs"]:
        errors.append("no complete work/MetalSet GLP/target/pixelILT pairs found")
    if manifest["dataset"]["training_pair_count"] < 1 or manifest["dataset"]["validation_pair_count"] < 1:
        errors.append("the train/validation split is empty")
    for record in manifest["source_hashes"] + manifest["kernel_hashes"]:
        if record["status"] != "present":
            errors.append(f"required upstream input missing/unreadable: {record['path']}")
    for pair in manifest["dataset"]["pairs"]:
        if pair["split"] not in {"train", "validation"}:
            continue
        for kind, record in pair["files"].items():
            if record["status"] != "present":
                errors.append(f"training/validation file missing/unreadable: {record['path']}")
        for kind in ("target", "pixelILT"):
            dimensions = pair["png_dimensions"].get(kind)
            if not dimensions or len(dimensions) != 2 or any(not isinstance(value, int) or value <= 0 for value in dimensions):
                errors.append(f"invalid PNG signature/IHDR dimensions for {pair['id']} {kind}")
    if manifest["source_config"]["parse_error"] or not manifest["source_config"]["original_values"]:
        errors.append("config/lithosimple.txt could not be parsed")
    return errors


def _restore_training_imports(root: Path):
    os.chdir(root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    return np, torch, F, DataLoader


def _import_upstream_training_components():
    from lithobench.ilt.neuralilt import UNet
    from lithobench.dataset import DataILT
    from pylitho.exact import LithoSim
    return UNet, DataILT, LithoSim


def set_seeds(np, torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def atomic_torch_save(torch, value, path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    if path.exists() or temp.exists():
        raise FileExistsError(path)
    torch.save(value, temp)
    os.replace(temp, path)


def atomic_replace_torch_save(torch, value, path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        raise FileExistsError(temp)
    torch.save(value, temp)
    os.replace(temp, path)


def journal_write(stream, record: dict) -> None:
    stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run_training(root: Path, output_dir: Path, manifest: dict, manifest_digest: str) -> dict:
    np, torch, F, DataLoader = _restore_training_imports(root)
    if not torch.cuda.is_available():
        raise RuntimeError("--train requer CUDA disponível; o preflight não usa GPU")
    torch.cuda.set_device(0)
    UNet, DataILT, LithoSim = _import_upstream_training_components()
    set_seeds(np, torch, RECIPE["seed"])
    device = torch.device("cuda")
    recipe = dict(RECIPE)
    recipe["pytorch_version"] = torch.__version__
    recipe["cuda_version"] = torch.version.cuda
    recipe["device_name"] = torch.cuda.get_device_name(0)

    pairs = manifest["dataset"]["pairs"]
    train_pairs = [p for p in pairs if p["split"] == "train"]
    val_pairs = [p for p in pairs if p["split"] == "validation"]
    train_data = DataILT(
        [str(root / p["files"]["target"]["path"]) for p in train_pairs],
        [str(root / p["files"]["pixelILT"]["path"]) for p in train_pairs],
        crop=True, size=(512, 512), cache=False,
    )
    val_data = DataILT(
        [str(root / p["files"]["target"]["path"]) for p in val_pairs],
        [str(root / p["files"]["pixelILT"]["path"]) for p in val_pairs],
        crop=False, size=(512, 512), cache=False,
    )
    loader_gen = torch.Generator()
    loader_gen.manual_seed(RECIPE["seed"])
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0, generator=loader_gen)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=0)

    base_config = dict(manifest["source_config"]["original_values"])
    base_config["KernelDir"] = str(root / "kernel")
    base_config["PrintSteepness"] = "85.0"
    simulator = LithoSim(base_config)
    net = torch.nn.DataParallel(UNet(), device_ids=[0], output_device=0).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    journal_path = output_dir / "epoch_journal.jsonl"
    timings = {}

    def epoch_pass(stage: str, epoch: int, train: bool) -> dict:
        net.train(train)
        sums = {"objective": 0.0, "l2_mse": 0.0, "pv_mean_abs": 0.0}
        batches = 0
        samples = 0
        context = torch.enable_grad if train else torch.no_grad
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with context():
            for target, label in (train_loader if train else val_loader):
                target = target.to(device)
                label = label.to(device)
                prediction = net(target)
                if stage == "pretrain":
                    mse = F.mse_loss(prediction, label)
                    objective, l2_mse, pv_term = mse, mse, torch.zeros((), device=device)
                else:
                    printed_nom, printed_max, printed_min = simulator(prediction.squeeze(1))
                    l2_mse = F.mse_loss(printed_nom.unsqueeze(1), target)
                    pv_term = torch.mean(torch.abs(printed_max - printed_min))
                    objective = l2_mse + 0.1 * pv_term
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    objective.backward()
                    optimizer.step()
                batch_size = int(target.shape[0])
                sums["objective"] += float(objective.detach().item()) * batch_size
                sums["l2_mse"] += float(l2_mse.detach().item()) * batch_size
                sums["pv_mean_abs"] += float(pv_term.detach().item()) * batch_size
                batches += 1
                samples += batch_size
        if not batches:
            raise RuntimeError(f"empty {stage} {'train' if train else 'validation'} loader")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = {key: value / samples for key, value in sums.items()}
        result["num_samples"] = samples
        result["num_batches"] = batches
        result["seconds"] = time.perf_counter() - start
        return result

    with journal_path.open("x", encoding="utf-8", newline="\n") as journal:
        journal_write(journal, {
            "event": "run_start",
            "status": "reconstruction_not_replication",
            "manifest_sha256": manifest_digest,
            "recipe": recipe,
            "cuda_visible_device_count": torch.cuda.device_count(),
        })
        for stage, epochs in (("pretrain", 50), ("finetune", 20)):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stage_start = time.perf_counter()
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
            for epoch in range(1, epochs + 1):
                training = epoch_pass(stage, epoch, True)
                validation = epoch_pass(stage, epoch, False)
                journal_write(journal, {
                    "event": "epoch",
                    "stage": stage,
                    "epoch": epoch,
                    "train": training,
                    "validation": validation,
                    "elapsed_stage_seconds": time.perf_counter() - stage_start,
                })
                atomic_replace_torch_save(torch, {
                    "schema_version": 1,
                    "label": "reconstruction_not_replication",
                    "resume_supported_by_runner": False,
                    "stage": stage,
                    "epoch": epoch,
                    "model_state": net.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                    "train_loader_generator_state": train_loader.generator.get_state(),
                }, output_dir / "last_epoch_recovery_state.pt")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings[f"{stage}_seconds"] = time.perf_counter() - stage_start
            weights_path = output_dir / ("init.pth" if stage == "pretrain" else "fine.pth")
            atomic_torch_save(torch, net.module.state_dict(), weights_path)
            journal_write(journal, {
                "event": "stage_checkpoint",
                "stage": stage,
                "checkpoint": weights_path.name,
                "sha256": sha256_file(weights_path),
            })
    return {
        "status": "completed_reconstruction_train",
        "label": "reconstruction_not_replication",
        "paper_baseline_reproduced": False,
        "manifest_sha256": manifest_digest,
        "training_seconds": timings,
        "init_checkpoint_sha256": sha256_file(output_dir / "init.pth"),
        "fine_checkpoint_sha256": sha256_file(output_dir / "fine.pth"),
        "inference_metrics": "not_run",
        "mask_refinement_metrics": "not_run",
        "source_fit_metrics": "not_run",
        "quality_claim_eligible": False,
        "resume_supported": False,
        "bitwise_reproducible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="pasta nova e exclusiva; deve ficar fora do upstream")
    parser.add_argument("--train", action="store_true", help="explicitamente iniciar os 50+20 epochs GPU")
    parser.add_argument("--manifest-sha256", help="digest exibido pelo preflight; obrigatório com --train")
    parser.add_argument("--accept-reconstruction-limitations", action="store_true", help="confirma rótulo aproximado, calibração 512 desconhecida e ausência de equivalência original")
    args = parser.parse_args(argv)

    root = args.upstream_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"upstream root não é diretório: {root}")
    if out == root or root in out.parents:
        parser.error("output-dir deve ficar fora do clone upstream")
    if args.train and (not args.manifest_sha256 or not args.accept_reconstruction_limitations):
        parser.error("--train exige --manifest-sha256 e --accept-reconstruction-limitations")
    if args.manifest_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.manifest_sha256):
        parser.error("--manifest-sha256 deve conter 64 dígitos hexadecimais")

    try:
        manifest = build_manifest(root)
        errors = validate_manifest(manifest)
        manifest["preflight_errors"] = errors
        manifest["preflight_status"] = "ready_for_opt_in_train" if not errors else "blocked"
        digest = canonical_sha(manifest)
        if args.train:
            if errors:
                print("Preflight blocked training:\n- " + "\n- ".join(errors), file=sys.stderr)
                return 2
            if digest.lower() != args.manifest_sha256.lower():
                print(f"manifest changed; computed sha256={digest}", file=sys.stderr)
                return 2
        out.mkdir(parents=True, exist_ok=False)
        manifest_output = dict(manifest)
        manifest_output["manifest_sha256"] = digest
        manifest_output["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        write_exclusive_json(out / "manifest.json", manifest_output)
        print(f"manifest_sha256={digest}")
        print(f"manifest_path={out / 'manifest.json'}")
        if not args.train:
            print("mode=preflight_only; no torch/GPU/model code imported")
            print(f"paired={manifest['dataset']['paired_layout_count']} train={manifest['dataset']['training_pair_count']} validation={manifest['dataset']['validation_pair_count']} excluded_test_overlap={manifest['dataset']['excluded_test_glp_overlap_pair_count']}")
            print(f"preflight_errors={len(errors)}")
            for error in errors:
                print(f"preflight_blocker={error}")
            return 0
        result = run_training(root, out, manifest, digest)
        write_exclusive_json(out / "training_summary.json", result)
        print(f"training_status=complete label={result['label']} quality_claim_eligible=false")
        return 0
    except FileExistsError:
        print(f"refusing to overwrite existing output directory: {out}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"runner failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
