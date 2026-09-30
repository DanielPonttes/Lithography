import math
import unittest
from unittest import mock

import torch

from light_source import (
    DifferentiableAbbeLitho,
    FixedMaskSourceBasis,
    PixelatedLightSource,
)


class LightSourceContractTests(unittest.TestCase):
    def make_sim(self, *, chunk=3, refractive_index=None, cache_max_bytes=None):
        source = PixelatedLightSource(grid_size=5, sigma_inner=0.25)
        kwargs = {
            "numerical_aperture": 0.9,
            "wavelength_nm": 193.0,
            "pixel_size_nm": 16.0,
            "source_chunk_size": chunk,
            "refractive_index": refractive_index,
        }
        if cache_max_bytes is not None:
            kwargs["cache_max_bytes"] = cache_max_bytes
        return DifferentiableAbbeLitho(source, **kwargs)

    @staticmethod
    def mask(dtype=torch.float32, requires_grad=False):
        torch.manual_seed(4)
        return torch.rand(2, 32, 32, dtype=dtype, requires_grad=requires_grad)

    def test_direct_basis_match_and_source_mask_gradients_float64(self):
        sim = self.make_sim()
        sim.double()
        mask = self.mask(torch.float64, requires_grad=True)
        direct = sim(mask)
        basis = sim.prepare_basis(mask.detach())
        evaluated = sim.evaluate_basis(basis)
        self.assertEqual(evaluated.shape, (2, 1, 32, 32))
        self.assertTrue(torch.allclose(direct, evaluated, rtol=1e-10, atol=1e-10))

        probe = torch.linspace(0.2, 1.1, 32 * 32, dtype=torch.float64).reshape(
            1, 1, 32, 32
        )
        (direct * probe).sum().backward()
        self.assertTrue(torch.isfinite(mask.grad).all())
        self.assertIsNotNone(sim.source.logits.grad)
        self.assertGreater(float(sim.source.logits.grad.abs().sum()), 0.0)
        direct_gradient = sim.source.logits.grad.detach().clone()

        sim.source.logits.grad.zero_()
        (evaluated * probe).sum().backward()
        self.assertGreater(float(sim.source.logits.grad.abs().sum()), 0.0)
        torch.testing.assert_close(sim.source.logits.grad, direct_gradient, rtol=1e-10, atol=1e-10)

    def test_source_gradient_matches_finite_difference_without_fft(self):
        sim = self.make_sim(refractive_index=1.2).double()
        basis = sim.prepare_basis(self.mask(torch.float64), defocus_nm=25)
        probe = torch.linspace(0.1, 1.2, 1024, dtype=torch.float64).reshape(1, 1, 32, 32)
        with mock.patch("torch.fft.fft2", side_effect=AssertionError("FFT during contraction")), \
             mock.patch("torch.fft.ifft2", side_effect=AssertionError("IFFT during contraction")):
            (sim.evaluate_basis(basis) * probe).sum().backward()
            active = sim.source.logits.grad.abs().argmax().item()
            analytic = sim.source.logits.grad.flatten()[active].item()
            original = sim.source.logits.flatten()[active].item()
            epsilon = 1e-5
            with torch.no_grad():
                sim.source.logits.flatten()[active] = original + epsilon
                plus = (sim.evaluate_basis(basis) * probe).sum().item()
                sim.source.logits.flatten()[active] = original - epsilon
                minus = (sim.evaluate_basis(basis) * probe).sum().item()
                sim.source.logits.flatten()[active] = original
            self.assertAlmostEqual(analytic, (plus - minus) / (2 * epsilon), delta=2e-7)

    def test_multiple_focus_bases_remain_valid(self):
        sim = self.make_sim(refractive_index=1.2)
        mask = self.mask()
        bases = {z: sim.prepare_basis(mask, defocus_nm=z) for z in (0, -25, 25)}
        for z, basis in bases.items():
            torch.testing.assert_close(sim.evaluate_basis(basis, defocus_nm=z), sim(mask, defocus_nm=z))

    def test_defocus_phase_matches_analytic_expression_and_constant_flux(self):
        sim = self.make_sim(refractive_index=1.2).double()
        transfer = sim._make_optical_transfer(32, 32, torch.zeros(1, 2, dtype=torch.float64),
                                            torch.float64, torch.device("cpu"), 25)
        frequency = 1 / (32 * 16)
        phase = 2 * math.pi * 25 / 193 * (math.sqrt(1.2 ** 2 - (193 * frequency) ** 2) - 1.2)
        expected = complex(math.cos(phase), math.sin(phase))
        self.assertLess(abs(transfer[0, 0, 1].item() - expected), 1e-12)
        self.assertEqual(transfer[0, 0, 0].item(), 1 + 0j)
        image = sim(torch.ones(32, 32, dtype=torch.float64), defocus_nm=25)
        torch.testing.assert_close(image, torch.ones_like(image), rtol=1e-12, atol=1e-12)

    def test_mask_gradient_matches_finite_difference(self):
        sim = self.make_sim(chunk=2)
        sim.double()
        mask = self.mask(torch.float64, requires_grad=True)
        output = sim(mask)
        probe = torch.zeros_like(output)
        probe[0, 0, 7, 11] = 1.0
        (output * probe).sum().backward()
        analytic = float(mask.grad[0, 7, 11])

        epsilon = 1.0e-5
        plus = mask.detach().clone()
        minus = mask.detach().clone()
        plus[0, 7, 11] += epsilon
        minus[0, 7, 11] -= epsilon
        finite_difference = (
            (
                ((sim(plus) * probe).sum() - (sim(minus) * probe).sum())
                / (2.0 * epsilon)
            )
            .detach()
            .item()
        )
        self.assertAlmostEqual(analytic, finite_difference, delta=2.0e-6)

    def test_chunk_invariance_and_zero_constant_inputs(self):
        mask = self.mask()
        sim_one = self.make_sim(chunk=1)
        sim_many = self.make_sim(chunk=64)
        sim_many.source.load_state_dict(sim_one.source.state_dict())
        self.assertTrue(torch.allclose(sim_one(mask), sim_many(mask), atol=1e-6))

        zero = torch.zeros(32, 32)
        self.assertTrue(torch.equal(sim_one(zero), torch.zeros_like(sim_one(zero))))
        constant = torch.ones(32, 32)
        self.assertTrue(torch.isfinite(sim_one(constant)).all())

    def test_source_flux_is_zero_outside_support(self):
        source = PixelatedLightSource(grid_size=7)
        weights = source.weight_map()
        self.assertTrue(torch.equal(weights[~source._pupil_support], torch.zeros_like(
            weights[~source._pupil_support]
        )))
        self.assertAlmostEqual(float(weights.sum().detach()), 1.0, places=6)

    def test_cache_budget_clear_to_and_load_state_invalidation(self):
        sim = self.make_sim(cache_max_bytes=32 * 1024**2)
        mask = self.mask()
        sim(mask)
        self.assertGreater(len(sim._optical_cache), 0)
        self.assertLessEqual(sim._cache_bytes, sim.cache_max_bytes)
        sim.clear_cache()
        self.assertEqual(sim._cache_bytes, 0)
        self.assertEqual(len(sim._optical_cache), 0)

        sim(mask)
        sim.double()
        self.assertEqual(len(sim._optical_cache), 0)
        state = {key: value.clone() for key, value in sim.state_dict().items()}
        sim(mask.double())
        self.assertGreater(len(sim._optical_cache), 0)
        sim.load_state_dict(state)
        self.assertEqual(len(sim._optical_cache), 0)

        small_cache = self.make_sim(cache_max_bytes=1)
        small_cache(mask)
        self.assertEqual(len(small_cache._optical_cache), 0)

    def test_basis_budget_checked_before_fft(self):
        sim = self.make_sim()
        mask = self.mask()
        with mock.patch.object(torch.fft, "fft2", side_effect=AssertionError):
            with self.assertRaises(ValueError):
                sim.prepare_basis(mask, max_bytes=1)
        with self.assertRaises(ValueError):
            sim.prepare_basis(mask.requires_grad_())

    def test_defocus_validation_phase_and_no_nans(self):
        mask = self.mask()
        no_index = self.make_sim()
        with self.assertRaises(ValueError):
            no_index(mask, defocus_nm=20.0)

        low_index = self.make_sim(refractive_index=0.8)
        with self.assertRaises(ValueError):
            low_index(mask, defocus_nm=20.0)

        sim = self.make_sim(refractive_index=1.2)
        defocused = sim(mask, defocus_nm=20.0)
        self.assertTrue(torch.isfinite(defocused).all())
        self.assertTrue(torch.isfinite(sim(mask, defocus_nm=0.0)).all())

    def test_basis_stale_optics_coordinates_and_defocus_rejected(self):
        sim = self.make_sim(refractive_index=1.2)
        mask = self.mask()
        basis = sim.prepare_basis(mask, defocus_nm=0.0)

        sim.numerical_aperture += 0.01
        with self.assertRaises(ValueError):
            sim.evaluate_basis(basis)
        sim.numerical_aperture -= 0.01
        sim(mask, defocus_nm=10.0)
        torch.testing.assert_close(sim.evaluate_basis(basis), sim(mask, defocus_nm=0.0))
        with self.assertRaises(ValueError):
            sim.evaluate_basis(basis, defocus_nm=10.0)

        fresh = self.make_sim()
        fresh_basis = fresh.prepare_basis(mask)
        active_index = torch.nonzero(fresh.source._pupil_support, as_tuple=False)[0]
        with torch.no_grad():
            fresh.source._coordinates[active_index[0], active_index[1], 0] += 1e-3
        with self.assertRaises(ValueError):
            fresh.evaluate_basis(fresh_basis)

    def test_basis_state_dict_roundtrip_and_dtype_move(self):
        sim = self.make_sim()
        mask = self.mask()
        basis = sim.prepare_basis(mask)
        state = basis.state_dict()
        restored = FixedMaskSourceBasis(
            torch.zeros_like(basis.intensities),
            basis.source_coordinates,
            {},
            source_support=basis.source_support,
        )
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(restored.intensities, basis.intensities))
        self.assertEqual(restored.metadata, basis.metadata)
        self.assertTrue(torch.allclose(sim.evaluate_basis(restored), sim(mask)))

        moved = FixedMaskSourceBasis.from_basis(basis).double()
        sim.double()
        self.assertTrue(torch.allclose(sim.evaluate_basis(moved), sim(mask.double())))

    def test_forward_input_shapes_and_finite_validation(self):
        sim = self.make_sim()
        image = self.mask()[0]
        self.assertEqual(sim(image).shape, (1, 1, 32, 32))
        self.assertEqual(sim(image.unsqueeze(0).unsqueeze(1)).shape, (1, 1, 32, 32))
        with self.assertRaises(ValueError):
            sim(torch.full((32, 32), float("nan")))
        with self.assertRaises(ValueError):
            sim(torch.zeros(2, 2, 32, 32))


if __name__ == "__main__":
    unittest.main()
