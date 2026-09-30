import unittest

import torch

from scripts.run_protected_pvband_experiment import (
    SEEDS, balanced_mse, choose_arm, contour_mask, final_test_arms, local_band,
    objective_parts, simulator,
)


def calibration_metrics(band, nominal, worst, zero_print=False):
    positive = 16
    return {
        "mean": {"band_pixels": float(band), "L2_pixels": float(nominal),
                 "L2_worst_dose_pixels": float(worst)},
        "per_layout": [{
            "layout_id": "cal_a", "target_positive_pixels": positive,
            "per_corner_binary_metrics": [
                {"predicted_positive_pixels": 10},
                {"predicted_positive_pixels": 0 if zero_print else positive},
                {"predicted_positive_pixels": positive},
            ],
        }],
    }


class ProtectedPvBandTests(unittest.TestCase):
    def test_balanced_worst_mse_avoids_background_dilution_and_rejects_blank(self):
        target = torch.zeros((1, 1, 10, 10))
        target[0, 0, 2, 2] = 1
        printed = torch.zeros((3, 1, 1, 10, 10))
        balanced = balanced_mse(printed, target)
        unbalanced = (printed - target.unsqueeze(0)).square().mean()
        self.assertAlmostEqual(float(balanced), 0.5, places=7)
        self.assertAlmostEqual(float(unbalanced), 0.01, places=7)
        with self.assertRaisesRegex(ValueError, "both classes"):
            balanced_mse(printed, torch.zeros_like(target))

    def test_fixed_contour_roi_local_band_and_gradients(self):
        target = torch.zeros((1, 1, 8, 8))
        target[..., 2:6, 2:6] = 1
        roi, perimeter = contour_mask(target, radius=2)
        self.assertGreater(perimeter, 0)
        self.assertGreater(int(roi.sum()), perimeter)
        identical = torch.full((3, 1, 1, 8, 8), 0.49, requires_grad=True)
        band = local_band(identical, target, roi, perimeter)
        self.assertEqual(float(band.detach()), 0.0)
        values = torch.linspace(0.35, 0.65, 3 * 64).reshape(3, 1, 1, 8, 8).requires_grad_()
        terms = objective_parts(values, target, "A5", roi, perimeter)
        terms["objective"].backward()
        self.assertTrue(torch.isfinite(terms["objective"]))
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_gate_compares_paired_seeds_and_opens_baseline_plus_winner(self):
        base = [{"calibration": calibration_metrics(100, 100, 200)} for _ in SEEDS]
        a4 = [{"calibration": calibration_metrics(88, 104, 204)} for _ in SEEDS]
        a5 = [{"calibration": calibration_metrics(80, 105, 210)} for _ in SEEDS]
        result = choose_arm({"A0": base, "A4": a4, "A5": a5})
        self.assertEqual(result["selected_arm"], "A5")
        self.assertEqual(final_test_arms(result), ["A0", "A5"])
        self.assertTrue(result["candidate_checks"]["A4"]["qualified"])
        self.assertTrue(result["candidate_checks"]["A5"]["qualified"])

    def test_empty_gate_and_zero_nominal_positive_layout_are_rejected(self):
        self.assertEqual(final_test_arms(None), [])
        self.assertEqual(final_test_arms({"selected_arm": "A4", "eligible_arms": []}), [])
        base = [{"calibration": calibration_metrics(100, 100, 200)} for _ in SEEDS]
        bad = [{"calibration": calibration_metrics(80, 100, 200, zero_print=True)} for _ in SEEDS]
        good = [{"calibration": calibration_metrics(90, 100, 200)} for _ in SEEDS]
        result = choose_arm({"A0": base, "A4": bad, "A5": good})
        self.assertFalse(result["candidate_checks"]["A4"]["qualified"])
        self.assertFalse(result["candidate_checks"]["A4"]["no_positive_layout_with_zero_nominal_print"])
        self.assertEqual(result["selected_arm"], "A5")

    def test_seed_initialization_is_paired_and_source_flux_is_unit_normalized(self):
        first, same, other = (simulator(torch.device("cpu"), seed) for seed in (29, 29, 43))
        self.assertTrue(torch.equal(first.source.logits, same.source.logits))
        self.assertFalse(torch.equal(first.source.logits, other.source.logits))
        for sim in (first, same, other):
            self.assertAlmostEqual(float(sim.source.weight_map().sum()), 1.0, places=6)

    def test_incomplete_arms_keep_final_gate_closed(self):
        result = choose_arm({"A0": [], "A4": [], "A5": []})
        self.assertIsNone(result["selected_arm"])
        self.assertEqual(result["final_gate_arms"], [])


if __name__ == "__main__":
    unittest.main()
