import unittest
from unittest import mock

import torch

from scripts.ablate_pvband_light_source import (
    ARMS, RASTER, choose_arm, make_simulator, preflight, selected_final_test_arm,
    sha256_tensor,
)
from source_training import (
    ProcessCorner, SourceDataset, SourceFitConfig, _objective, _objective_components,
    evaluate_source,
)


class PvBandAblationTests(unittest.TestCase):
    def test_new_objective_options_preserve_legacy_positional_config(self):
        config = SourceFitConfig(3, 0.02, 0.75, 0.21, 45.0, 0.5)
        self.assertEqual(config.steps, 3)
        self.assertEqual(config.band_weight, 0.75)
        self.assertEqual(config.threshold, 0.21)
        self.assertEqual(config.objective, "envelope_squared")

    def test_surrogate_is_zero_for_identical_uncertain_corners(self):
        printed = torch.full((3, 1, 1, 2, 2), 0.49)
        target = torch.zeros((1, 1, 2, 2))
        config = SourceFitConfig(objective="worst_dose_surrogate")
        value = _objective_components(printed, target, config)["surrogate_pv_band"]
        self.assertEqual(float(value), 0.0)

    def test_surrogate_approaches_binary_corner_envelope(self):
        # First pixel is below threshold at every corner; second flips.
        printed = torch.tensor([[[[[0.05, 0.40]]]],
                                [[[[0.20, 0.80]]]],
                                [[[[0.45, 0.95]]]]])
        target = torch.zeros((1, 1, 1, 2))
        config = SourceFitConfig(objective="worst_dose_surrogate", surrogate_steepness=1000.0)
        value = _objective_components(printed, target, config)["surrogate_pv_band"]
        self.assertAlmostEqual(float(value), 0.5, places=6)

    def test_surrogate_objective_has_finite_gradient(self):
        values = torch.tensor([0.47, 0.51, 0.54], requires_grad=True)
        printed = values.reshape(3, 1, 1, 1, 1)
        target = torch.zeros((1, 1, 1, 1))
        config = SourceFitConfig(objective="worst_dose_surrogate", surrogate_steepness=50.0,
                                 surrogate_band_weight=0.5)
        loss = _objective(printed, target, config)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_split_hash_integrity_preflight(self):
        def make_split(prefix, count, offset):
            masks = torch.zeros((count, 1, RASTER, RASTER))
            targets = torch.zeros_like(masks)
            for i in range(count):
                masks[i, 0, (offset + i * 3) % RASTER, :i + 1] = 1
                start = offset + i
                targets[i, 0, start:start + 20, start:start + 20] = 1
            ids = tuple(prefix + str(i) for i in range(count))
            return SourceDataset(masks, targets, ids, 4.0), ids

        fit, fit_ids = make_split("fit_", 4, 1)
        calibration, calibration_ids = make_split("cal_", 2, 20)
        final_test, test_ids = make_split("test_", 3, 50)
        splits = {"fit": fit, "calibration": calibration, "final_test": final_test}
        rows = []
        for split_name, ids in (("fit", fit_ids), ("calibration", calibration_ids),
                                ("final_test", test_ids)):
            dataset = splits[split_name]
            for i, layout_id in enumerate(ids):
                rows.append({"layout_id": layout_id, "family": split_name,
                             "geometry": "unit test", "mask": dataset.masks[i]})
        layouts = preflight(splits, rows)
        self.assertEqual([len(layouts[name]) for name in splits], [4, 2, 3])
        self.assertEqual(len({item["target_sha256"] for values in layouts.values() for item in values}), 9)

    def test_initial_logits_match_for_same_seed_across_arms(self):
        self.assertEqual(tuple(ARMS), ("A0", "A1", "A2", "A3"))
        first = make_simulator(torch.device("cpu"), 29)
        second = make_simulator(torch.device("cpu"), 29)
        try:
            self.assertEqual(sha256_tensor(first.source.logits), sha256_tensor(second.source.logits))
            torch.testing.assert_close(first.source.logits, second.source.logits, rtol=0, atol=0)
        finally:
            del first, second

    def test_choose_arm_inclusive_calibration_limits_and_tiebreak(self):
        def records(band, nominal, worst):
            return [{"seed": seed, "fit_report": {"after": {"validation": {"mean": {
                "band_pixels": band[i], "L2_pixels": nominal[i],
                "L2_worst_dose_pixels": worst[i],
            }}}}} for i, seed in enumerate((17, 29, 43))]

        baseline = records([100.0] * 3, [100.0] * 3, [200.0] * 3)
        boundary = records([90.0] * 3, [105.0] * 3, [210.0] * 3)
        runs = {"A0": baseline, "A1": boundary, "A2": boundary,
                "A3": records([90.0, 100.0, 80.0], [100.0] * 3, [200.0] * 3)}
        result = choose_arm(runs)
        self.assertTrue(result["candidate_checks"]["A1"]["pv_band_mean_reduction_at_least_10pct"])
        self.assertTrue(result["candidate_checks"]["A1"]["nominal_L2_mean_at_most_5pct_worse"])
        self.assertTrue(result["candidate_checks"]["A1"]["worst_dose_L2_mean_at_most_5pct_worse"])
        self.assertTrue(result["candidate_checks"]["A1"]["pv_band_improves_in_each_seed"])
        self.assertFalse(result["candidate_checks"]["A3"]["pv_band_improves_in_each_seed"])
        self.assertEqual(result["selected_arm"], "A1")

    def test_final_test_gate_stays_closed_without_qualified_arm(self):
        self.assertIsNone(selected_final_test_arm(None))
        self.assertIsNone(selected_final_test_arm({"selected_arm": None, "eligible_arms": []}))
        self.assertIsNone(selected_final_test_arm({"selected_arm": "unknown"}))
        self.assertEqual(selected_final_test_arm({"selected_arm": "A2"}), "A2")

    def test_per_corner_binary_diagnostics_match_hand_computed_pixels(self):
        mask = torch.zeros((1, 1, 2, 2))
        target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        dataset = SourceDataset(mask, target, ("handmade",), 4.0)
        printed = torch.tensor([
            [[[[0.8, 0.2], [0.7, 0.1]]]],
            [[[[0.8, 0.2], [0.1, 0.7]]]],
            [[[[0.2, 0.8], [0.1, 0.1]]]],
        ])
        corners = (ProcessCorner("d0.98", 0.98), ProcessCorner("nominal", 1.0),
                   ProcessCorner("d1.02", 1.02))
        config = SourceFitConfig(steps=1)
        with mock.patch("source_training._printed", return_value=printed), \
             mock.patch("source_training._release"):
            result = evaluate_source(object(), dataset, [{}], corners, config)
        layout = result["per_layout"][0]
        self.assertEqual(layout["per_corner_binary_metrics"][0]["target_positive_pixels"], 1)
        self.assertEqual(layout["L2_pixels"], 1.0)
        self.assertEqual(layout["L2_worst_dose_pixels"], 2.0)
        self.assertEqual(layout["band_pixels"], 4.0)
        self.assertEqual(layout["flip_window_pixels"], 3.0)
        self.assertEqual([m["predicted_positive_pixels"] for m in layout["per_corner_binary_metrics"]],
                         [2, 2, 1])
        self.assertEqual([m["false_positive_pixels"] for m in layout["per_corner_binary_metrics"]],
                         [1, 1, 1])
        self.assertEqual([m["false_negative_pixels"] for m in layout["per_corner_binary_metrics"]],
                         [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
