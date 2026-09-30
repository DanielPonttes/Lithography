import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from light_source import DifferentiableAbbeLitho, PixelatedLightSource
from source_training import (
    ProcessCorner, SourceDataset, SourceFitConfig, fit_source, process_grid,
    validate_splits,
)


def fixtures(alternate_validation=False):
    masks = torch.zeros(2, 1, 32, 32)
    masks[0, 0, 6:22, 9:17] = 1
    masks[1, 0, 12:23, 4:27] = 1
    targets = masks.clone()
    targets[0, 0, 6:22, 17] = 1
    targets[1, 0, 23, 4:27] = 1
    validation = torch.zeros(1, 1, 32, 32)
    validation[0, 0, 5:25, 20:26] = 1
    val_target = validation.clone()
    if alternate_validation:
        val_target.zero_()
        val_target[0, 0, 10:20, 8:14] = 1
    return (
        SourceDataset(masks, targets, ("train_a", "train_b"), 8.0),
        SourceDataset(validation, val_target, ("val_a",), 8.0),
    )


def simulator():
    return DifferentiableAbbeLitho(
        PixelatedLightSource(grid_size=5), pixel_size_nm=8.0,
        source_chunk_size=3,
    )


class SourceTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fit_saves_safe_checkpoint_and_reuses_bases(self):
        train, val = fixtures()
        sim = simulator()
        original_masks = train.masks.clone()
        original_logits = sim.source.logits.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("torch.fft.fft2", wraps=torch.fft.fft2) as fft:
                report = fit_source(sim, train, val, config=SourceFitConfig(steps=4),
                                    output_dir=directory)
            self.assertEqual(fft.call_count, len(train.masks) + len(val.masks) + 2)
            self.assertEqual(len(report["basis_verification"]), 2)
            self.assertLess(max(v["max_abs_error"] for v in report["basis_verification"]), 1e-5)
            self.assertTrue(torch.equal(original_masks, train.masks))
            self.assertFalse(torch.equal(original_logits, sim.source.logits))
            self.assertAlmostEqual(float(sim.source.weight_map().sum().detach()), 1.0, places=6)
            self.assertEqual(report["band_type"], "dose_only")
            self.assertFalse(report["comparable_to_original_socs"])
            saved = torch.load(Path(directory) / "source.pt", weights_only=True)
            self.assertTrue(torch.equal(saved["source_state"]["logits"], sim.source.logits.cpu()))
            logged = json.loads((Path(directory) / "metrics.json").read_text())
            self.assertEqual(logged["validation_ids"], ["val_a"])
            self.assertEqual(saved["train_group_ids"], list(train.group_ids))
            self.assertEqual(saved["validation_targets_sha256"], logged["validation_targets_sha256"])
            self.assertIsNone(logged["EPE"])
            self.assertIsNone(logged["shots"])

    def test_validation_changes_cannot_change_source_updates(self):
        train, val = fixtures()
        train2, different_val = fixtures(alternate_validation=True)
        first, second = simulator(), simulator()
        fit_source(first, train, val, config=SourceFitConfig(steps=2))
        fit_source(second, train2, different_val, config=SourceFitConfig(steps=2))
        torch.testing.assert_close(first.source.logits, second.source.logits, rtol=0, atol=0)

    def test_overlapping_identifiers_groups_and_targets_are_rejected(self):
        train, val = fixtures()
        val.layout_ids = ("train_a",)
        with self.assertRaisesRegex(ValueError, "layout_ids overlap"):
            validate_splits(train, val)
        val.layout_ids = ("val_a",)
        val.group_ids = ("train_b",)
        with self.assertRaisesRegex(ValueError, "group_ids overlap"):
            validate_splits(train, val)
        val.group_ids = ("val_a",)
        val.targets = train.targets[0:1].clone()
        with self.assertRaisesRegex(ValueError, "identical target"):
            validate_splits(train, val)
        train, val = fixtures()
        val.masks = train.masks[0:1].clone()
        with self.assertRaisesRegex(ValueError, "identical mask"):
            validate_splits(train, val)

    def test_budget_and_physical_configuration_fail_before_precomputation(self):
        train, val = fixtures()
        sim = simulator()
        with mock.patch.object(sim, "prepare_basis", side_effect=AssertionError("FFT ran")):
            with self.assertRaises(MemoryError):
                fit_source(sim, train, val, config=SourceFitConfig(steps=1, max_total_basis_bytes=1))
            with self.assertRaisesRegex(MemoryError, "device bytes"):
                fit_source(sim, train, val, config=SourceFitConfig(steps=1, max_device_basis_bytes=1))
        with self.assertRaisesRegex(ValueError, "refractive_index"):
            fit_source(sim, train, val, corners=process_grid(defocus_nm=(0, 25)),
                       config=SourceFitConfig(steps=1))
        val.pixel_size_nm = 4.0
        with self.assertRaisesRegex(ValueError, "pixel sizes"):
            validate_splits(train, val)

    def test_dataset_and_corner_validation(self):
        with self.assertRaises(ValueError):
            SourceDataset(torch.ones(1, 1, 4, 4) * 0.5, torch.zeros(1, 1, 4, 4), ("a",), 1)
        with self.assertRaises(ValueError):
            ProcessCorner("nominal", 0.98, 0)
        with self.assertRaises(ValueError):
            process_grid(doses=(0.98,))
        with self.assertRaises(ValueError):
            SourceFitConfig(learning_rate=float("nan"))

    def test_focus_dose_fit_and_double_memory_accounting(self):
        train, val = fixtures()
        sim = DifferentiableAbbeLitho(PixelatedLightSource(5), numerical_aperture=0.9,
                                     refractive_index=1.2, pixel_size_nm=8).double()
        corners = process_grid(defocus_nm=(0, -25, 25))
        report = fit_source(sim, train, val, corners=corners, config=SourceFitConfig(steps=2))
        self.assertEqual(report["band_type"], "focus_and_dose")
        self.assertEqual(len(report["basis_verification"]), 6)
        n = sim.source.distribution()[1].numel()
        self.assertEqual(report["estimated_basis_bytes"], n * 32 * 32 * 8 * 3 * 3)
        with mock.patch.object(sim, "prepare_basis", side_effect=AssertionError("allocation")):
            with self.assertRaises(MemoryError):
                fit_source(sim, train, val, config=SourceFitConfig(
                    steps=1, max_total_basis_bytes=n * 32 * 32 * 4 * 3))
        with self.assertRaisesRegex(ValueError, "nominal"):
            fit_source(sim, train, val, corners=(), config=SourceFitConfig(steps=1))


if __name__ == "__main__":
    unittest.main()
