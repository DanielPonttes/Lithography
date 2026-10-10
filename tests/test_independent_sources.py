"""Focused synthetic checks for fixed-source independent GLP evaluation."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from scripts import evaluate_independent_sources as evaluator


def glp_text(vertices, layer="M1"):
    coordinates = " ".join(str(value) for point in vertices for value in point)
    return (
        "BEGIN     /* The metadata are invalid */\n"
        "EQUIV  1  1000  MICRON  +X,+Y\n"
        "CNAME Temp_Top\nLEVEL M1\nCELL Temp_Top PRIME\n"
        f"PGON N {layer} {coordinates}\nENDMSG\n"
    )


class IndependentSourceGeometryTests(unittest.TestCase):
    def test_pinned_upstream_head_is_the_full_40_character_commit(self):
        self.assertEqual(
            evaluator.EXPECTED_UPSTREAM_HEAD,
            "9c74e82218e377eaf6d02d113fc1ce6e36c92aa6",
        )
        self.assertRegex(evaluator.EXPECTED_UPSTREAM_HEAD, r"^[0-9a-f]{40}$")

    def test_output_path_must_be_outside_common_benchmark_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            benchmark = Path(temporary) / "lithobench" / "benchmark"
            metal = benchmark / "StdMetal"
            contact = benchmark / "StdContact"
            metal.mkdir(parents=True)
            contact.mkdir()
            with self.assertRaisesRegex(ValueError, "outside source data"):
                evaluator._assert_output_outside_repo(benchmark / "results", [metal, contact])
            evaluator._assert_output_outside_repo(Path(temporary) / "external-results", [metal, contact])

    def test_glp_units_manhattan_area_and_centered_half_open_raster(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tile.glp"
            path.write_text(glp_text([(0, 0), (8, 0), (8, 8), (0, 8)]), encoding="utf-8")
            parsed = evaluator.parse_glp(path)
        self.assertEqual(parsed["bbox_nm"], [0, 0, 8, 8])
        mask, info = evaluator.rasterize_polygons(parsed["polygons"], parsed["bbox_nm"], canvas_size_px=8)
        self.assertEqual(int(mask.sum()), 4)
        self.assertEqual(info["translation_nm"], [12.0, 12.0])
        self.assertEqual(info["raster_bbox_px_halfopen"], [3, 3, 5, 5])
        self.assertEqual(info["raster_sha256"], evaluator._sha256_bytes(mask.tobytes(order="C")))

    def test_actual_stdmetal_header_and_unsupported_orientation(self):
        actual_header = (
            "BEGIN     /* The metadata are invalid */\n"
            "EQUIV  1  1000  MICRON  +X,+Y\n"
            "CNAME Temp_Top\nLEVEL M1\n\nCELL Temp_Top PRIME\n"
            "   PGON N M1  0 0 0 65 65 65 65 0\nENDMSG\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "actual.glp"
            path.write_text(actual_header, encoding="utf-8")
            parsed = evaluator.parse_glp(path)
            self.assertEqual(parsed["bbox_nm"], [0, 0, 65, 65])
            self.assertEqual(parsed["polygon_count"], 1)
            path.write_text(actual_header.replace("+X,+Y", "-X,+Y"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coordinate orientation"):
                evaluator.parse_glp(path)

    def test_pinned_maximum_absolute_glp_coordinates(self):
        vertices = [(1273, 1266), (1277, 1266), (1277, 1270), (1273, 1270)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "boundary.glp"
            path.write_text(glp_text(vertices), encoding="utf-8")
            parsed = evaluator.parse_glp(path)
            self.assertEqual(parsed["bbox_nm"], [1273, 1266, 1277, 1270])
            path.write_text(glp_text([(1274, 1266), (1278, 1266),
                                      (1278, 1270), (1274, 1270)]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned 1277 x 1270 nm input bounds"):
                evaluator.parse_glp(path)

    def test_jsonl_append_round_trips_error_and_all_canonical_source_records(self):
        records = [
            {"status": "error", "source": "event_candidate", "layout_id": "synthetic:0",
             "error": "RuntimeError: first line" + chr(10) + "second line",
             "error_kind": "runtime"},
            {"status": "scored", "source": "event_candidate", "layout_id": "synthetic:0",
             "metrics": {"canonical_cpu_1thread_basis": {"values": [1, 2, 3]}}},
            {"status": "scored", "source": "reference", "layout_id": "synthetic:0",
             "metrics": {"canonical_cpu_1thread_basis": {"values": [4, 5, 6]}}},
            {"status": "scored", "source": "best_known", "layout_id": "synthetic:0",
             "metrics": {"canonical_cpu_1thread_basis": {"values": [7, 8, 9]}}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            for record in records:
                evaluator._append_jsonl(path, record)
            self.assertEqual(len(path.read_bytes().splitlines()), len(records))
            self.assertEqual(evaluator._read_records(path), records)
        self.assertEqual(
            {row["source"] for row in records if row["status"] == "scored"},
            {"event_candidate", "reference", "best_known"},
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is supplied on the scoring host")
    def test_standalone_cli_synthetic_64px_run_uses_real_light_source_api(self):
        driver = textwrap.dedent(r"""
            import hashlib, json, sys
            from pathlib import Path
            from runpy import run_path
            import numpy as np

            script_path = Path(sys.argv[1]).resolve()
            root = Path(sys.argv[2]).resolve()
            repository_root = script_path.parent.parent
            sys.path[:] = [entry for entry in sys.path
                           if Path(entry or ".").resolve() != repository_root]
            assert repository_root not in [Path(entry or ".").resolve() for entry in sys.path]
            evaluator = run_path(str(script_path), run_name="standalone_cli_smoke")
            evaluator = evaluator["main"].__globals__
            physical = evaluator["PHYSICAL"]
            physical["canvas_size_px"] = 64
            physical["sensitivity_canvas_size_px"] = 32
            source_families = (("StdMetal271", "StdMetal"), ("StdContact165", "StdContact"))
            glp_text = (
                "BEGIN     /* The metadata are invalid */\n"
                "EQUIV  1  1000  MICRON  +X,+Y\n"
                "CNAME Temp_Top\nLEVEL M1\nCELL Temp_Top PRIME\n"
                "PGON N M1 0 0 96 0 96 96 0 96\nENDMSG\n"
            )
            layouts, source_files, family_directories = [], [], {}
            for protocol_family, directory_name in source_families:
                family_dir = root / "dataset" / directory_name
                family_dir.mkdir(parents=True)
                family_directories[protocol_family] = str(family_dir)
                glp_path = family_dir / "synthetic.glp"
                glp_path.write_text(glp_text, encoding="utf-8")
                parsed = evaluator["parse_glp"](glp_path)
                mask, raster = evaluator["rasterize_polygons"](
                    parsed["polygons"], parsed["bbox_nm"], canvas_size_px=64
                )
                snapshot = root / "targets" / (directory_name + ".mask")
                snapshot.parent.mkdir(exist_ok=True)
                packed = np.packbits(mask.reshape(-1), bitorder="big").tobytes()
                snapshot.write_bytes(packed)
                source_files.append({
                    "family": protocol_family, "path": str(glp_path),
                    "relative_path": "synthetic.glp", "cell_group": "synthetic",
                    "raw_glp_sha256": parsed["raw_sha256"],
                })
                layouts.append({
                    "layout_id": protocol_family + ":synthetic.glp",
                    "family": protocol_family, "cell_group": "synthetic",
                    "source_glp": str(glp_path), "raw_glp_sha256": parsed["raw_sha256"],
                    "target_snapshot": str(snapshot),
                    "target_snapshot_sha256": hashlib.sha256(packed).hexdigest(),
                    "polygon_count": parsed["polygon_count"],
                    "raster": {**raster, "positive_pixel_count": int(mask.sum())},
                    "aliases": [{"relative_path": "synthetic.glp"}],
                })

            weights = {}
            for source, index in (("event_candidate", 24), ("reference", 16), ("best_known", 32)):
                vector = np.zeros(49, dtype=np.float64)
                vector[index] = 1.0
                weights[source] = {"weights": vector.tolist(), **evaluator["vector_hashes"](vector)}
            pins = {
                "cached_plan_file": "synthetic-cached-plan.json",
                "event_candidate_file": "synthetic-event-candidate.json",
                "event_report_file": "synthetic-event-report.json",
            }
            protocol = {
                "input_pins": pins,
                "runtime_policy": {"time_budget_seconds": 300},
                "source_vectors": weights,
                "dataset": {
                    "family_directories": family_directories,
                    "source_files": source_files,
                    "layouts": layouts,
                },
            }
            evaluator["_validate_protocol"] = lambda path, digest: (protocol, digest)
            evaluator["_validate_pinned_inputs"] = lambda *paths: pins
            protocol_path = root / "synthetic-protocol.json"
            protocol_path.write_text("{}" + chr(10), encoding="utf-8")
            run_dir = root / "synthetic-run"
            code = evaluator["main"]([
                "run", "--protocol", str(protocol_path), "--run-dir", str(run_dir),
                "--expected-protocol-sha256", "a" * 64, "--device", "cpu",
                "--time-budget-seconds", "300",
            ])
            if code != 0:
                raise SystemExit(code)
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            rows = evaluator["_read_records"](run_dir / "layout_results.jsonl")
            assert result["status"] == "complete", result
            assert result["scoring"]["completed_source_layout_records"] == 6, result["scoring"]
            assert len(rows) == 6, len(rows)
            assert {row["source"] for row in rows} == {"event_candidate", "reference", "best_known"}
            assert "light_source" in sys.modules
            print("synthetic_standalone_cli_smoke=passed")
        """)
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, "-B", "-c", driver,
                 str(evaluator.SCRIPT_PATH), temporary],
                cwd=temporary, capture_output=True, text=True, timeout=180,
            )
        self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
        self.assertIn("synthetic_standalone_cli_smoke=passed", completed.stdout)

    def test_union_raster_and_area_are_deterministic(self):
        rectangles = [
            [(0, 0), (8, 0), (8, 8), (0, 8)],
            [(8, 0), (16, 0), (16, 8), (8, 8)],
        ]
        first, first_info = evaluator.rasterize_polygons(rectangles, [0, 0, 16, 8], canvas_size_px=8)
        second, second_info = evaluator.rasterize_polygons(rectangles, [0, 0, 16, 8], canvas_size_px=8)
        self.assertEqual(int(first.sum()), 8)
        self.assertEqual(first_info["raster_sha256"], second_info["raster_sha256"])
        np.testing.assert_array_equal(first, second)

    def test_rejects_units_diagonal_edges_and_collapsed_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.glp"
            path.write_text("EQUIV 1 1 MICRON\nENDMSG\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "EQUIV"):
                evaluator.parse_glp(path)
            path.write_text(glp_text([(0, 0), (8, 0), (8, 8), (0, 8)]).replace("8 0", "8 1", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Manhattan"):
                evaluator.parse_glp(path)
        with self.assertRaisesRegex(ValueError, "empty or subpixel-collapsed"):
            evaluator.rasterize_polygons(
                [[(0, 0), (2, 0), (2, 2), (0, 2)]], [0, 0, 2, 2], canvas_size_px=8
            )
        with self.assertRaisesRegex(ValueError, "fit the fixed canvas"):
            evaluator.rasterize_polygons(
                [[(0, 0), (40, 0), (40, 8), (0, 8)]], [0, 0, 40, 8], canvas_size_px=8
            )

    def test_rejects_self_intersecting_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "crossing.glp"
            path.write_text(
                glp_text([(0, 0), (12, 0), (12, 12), (4, 12), (4, 4), (8, 4), (8, 8), (0, 8)]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "self-intersecting"):
                evaluator.parse_glp(path)

    def test_duplicate_geometry_unions_cell_groups_conservatively(self):
        groups = evaluator._DisjointSet()
        groups.union("metal_A", "metal_B")
        groups.union("metal_B", "contact_C")
        components = groups.components()
        self.assertEqual(components, {"contact_C": ["contact_C", "metal_A", "metal_B"]})

    def test_group_bootstrap_and_primary_claim_gate_are_deterministic(self):
        ci_a = evaluator._bootstrap_ci([-0.1, -0.2, -0.15], np.random.default_rng(20261010))
        ci_b = evaluator._bootstrap_ci([-0.1, -0.2, -0.15], np.random.default_rng(20261010))
        self.assertEqual(ci_a, ci_b)
        self.assertLess(ci_a[1], 0)

        layouts = [
            {"layout_id": "metal:a", "family": "StdMetal271", "cell_group": "A"},
            {"layout_id": "metal:b", "family": "StdMetal271", "cell_group": "B"},
        ]
        protocol = {"dataset": {"layouts": layouts}}
        records = []
        for source in ("event_candidate", "reference", "best_known"):
            for layout in layouts:
                is_candidate = source == "event_candidate"
                metrics = {
                    "target_positive_pixels": 100,
                    "pvband_pixels": 10 if is_candidate else 20,
                    "pvband_fraction_of_target": 0.1 if is_candidate else 0.2,
                    "nominal_error_fraction_of_target": 0.04 if is_candidate else 0.05,
                    "worst_dose_error_fraction_of_target": 0.08 if is_candidate else 0.09,
                    "nominal_absolute_relative_area_error": 0.1 if is_candidate else 0.2,
                    "worst_dose_absolute_relative_area_error": 0.2 if is_candidate else 0.3,
                    "no_blank_positive_target_any_dose": True,
                    "nominal_error_pixels": 4,
                    "worst_dose_error_pixels": 8,
                }
                records.append({
                    "status": "scored", "source": source, "layout_id": layout["layout_id"],
                    "metrics": {"canonical_cpu_1thread_basis": metrics}, "parity_passed": True,
                })
        with tempfile.TemporaryDirectory() as temporary:
            record_file = Path(temporary) / "records.jsonl"
            record_file.write_text("", encoding="utf-8")
            summary = evaluator._summarize(protocol, records, [], "complete", None,
                                           record_file, "a" * 64)
        self.assertTrue(summary["validation"]["passed"])
        self.assertEqual(summary["scoring"]["completed_source_layout_records"], 6)
        self.assertEqual(summary["scoring"]["expected_source_layout_records"], 6)
        self.assertEqual(summary["scoring"]["scored_layout_count_by_source"]["event_candidate"], 2)
        self.assertEqual(summary["scope_summaries"]["pooled"]["event_candidate"]["layout_count"], 2)
        self.assertTrue(summary["primary_claim_gate"]["eligible"])
        self.assertLess(summary["scope_summaries"]["pooled"][
            "event_candidate_minus_reference"]["pvband_rate"]["bootstrap_95_percentile_ci"][1], 0)
        self.assertTrue(summary["primary_claim_gate"]["mean_nominal_spatial_error_nonregression"])

        records[-1]["parity_passed"] = False
        with tempfile.TemporaryDirectory() as temporary:
            record_file = Path(temporary) / "records.jsonl"
            record_file.write_text("", encoding="utf-8")
            failed = evaluator._summarize(protocol, records, [], "complete", None,
                                          record_file, "a" * 64)
        self.assertFalse(failed["validation"]["passed"])
        self.assertFalse(failed["primary_claim_gate"]["eligible"])
        self.assertEqual(failed["scope_summaries"]["pooled"]["best_known"][
            "layout_raw_counts"]["parity_failed_layout_count"], 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is supplied on the scoring host")
    def test_equal_area_displacement_fails_spatial_error_gate(self):
        target = np.zeros((32, 32), dtype=np.uint8)
        target[8:16, 8:16] = 1
        aerial = np.zeros((32, 32), dtype=np.float32)
        aerial[8:16, 16:24] = 0.5
        metrics, _ = evaluator.hard_metrics_from_aerial(
            __import__("torch").as_tensor(aerial), target,
            doses=[0.98, 1.0, 1.02], threshold=0.225, steepness=50.0,
        )
        self.assertEqual(metrics["printed_positive_pixels_by_dose"], [64, 64, 64])
        self.assertEqual(metrics["nominal_absolute_relative_area_error"], 0.0)
        self.assertEqual(metrics["nominal_error_pixels"], 128)
        self.assertEqual(metrics["nominal_error_fraction_of_target"], 2.0)

        layout = {"layout_id": "metal:displaced", "family": "StdMetal271", "cell_group": "A"}
        candidate = dict(metrics, pvband_pixels=0, pvband_fraction_of_target=0.0,
                         worst_dose_error_pixels=128, worst_dose_error_fraction_of_target=2.0,
                         worst_dose_absolute_relative_area_error=0.0,
                         no_blank_positive_target_any_dose=True)
        reference = dict(candidate, pvband_pixels=16, pvband_fraction_of_target=0.25,
                         nominal_error_pixels=64, nominal_error_fraction_of_target=1.0,
                         worst_dose_error_pixels=64, worst_dose_error_fraction_of_target=1.0,
                         nominal_absolute_relative_area_error=0.1,
                         worst_dose_absolute_relative_area_error=0.1)
        records = []
        for source, method_metrics in (("event_candidate", candidate), ("reference", reference),
                                       ("best_known", reference)):
            records.append({
                "status": "scored", "source": source, "layout_id": layout["layout_id"],
                "metrics": {"canonical_cpu_1thread_basis": method_metrics}, "parity_passed": True,
            })
        with tempfile.TemporaryDirectory() as temporary:
            record_file = Path(temporary) / "records.jsonl"
            record_file.write_text("", encoding="utf-8")
            summary = evaluator._summarize(
                {"dataset": {"layouts": [layout]}}, records, [], "complete", None,
                record_file, "b" * 64,
            )
        pooled = summary["scope_summaries"]["pooled"]
        candidate_delta = pooled["event_candidate_minus_reference"]
        self.assertLess(
            candidate_delta["nominal_absolute_relative_area_error"]["equal_group_mean_delta"], 0.0
        )
        self.assertFalse(summary["primary_claim_gate"]["mean_nominal_spatial_error_nonregression"])
        self.assertFalse(summary["primary_claim_gate"]["eligible"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is supplied on the scoring host")
    def test_layout_basis_is_prepared_once_for_multiple_sources(self):
        from unittest.mock import patch
        import torch

        mask = np.zeros((32, 32), dtype=np.float32)
        mask[8:24, 8:24] = 1.0
        weights_a = (np.ones(49, dtype=np.float64) / 49.0).tolist()
        weights_b = [0.0] * 49
        weights_b[24] = 1.0
        weights_c = [0.0] * 49
        weights_c[16] = 1.0
        calls = {"count": 0}
        intensity = torch.as_tensor(mask * 0.5, dtype=torch.float32).expand(49, -1, -1).clone()
        shared_basis = type("SyntheticBasis", (), {"intensities": intensity.unsqueeze(0)})()

        class SyntheticModel:
            def __init__(self, weights):
                self.weights = torch.as_tensor(weights, dtype=torch.float32)

            def prepare_basis(self, *args, **kwargs):
                calls["count"] += 1
                return shared_basis

            def evaluate_basis(self, basis, defocus_nm=None):
                return self._contract(basis.intensities)

            def __call__(self, input_mask, defocus_nm=0.0):
                return self._contract(shared_basis.intensities)

            def _contract(self, intensities):
                return torch.einsum("bnhw,n->bhw", intensities, self.weights).unsqueeze(1)

        with patch.object(evaluator, "_source_model", side_effect=lambda weights, device: SyntheticModel(weights)):
            result = evaluator.evaluate_layout_methods(
                mask, mask.astype(np.uint8),
                {"reference": {"weights": weights_a}, "candidate": {"weights": weights_b},
                 "best_known": {"weights": weights_c}},
                "cpu",
            )
        self.assertEqual(calls["count"], 1)
        self.assertEqual(set(result), {"reference", "candidate", "best_known"})
        self.assertTrue(all(row["parity_passed"] for row in result.values()))


if __name__ == "__main__":
    unittest.main()
