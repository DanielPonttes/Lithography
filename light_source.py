"""Differentiable pixelated illumination source and scalar Abbe simulator.

This module intentionally implements the scalar, source-aware part of the
model only.  It does not claim parity with SOCS or with polarized imaging.
"""

from collections import OrderedDict
import copy
import math

import torch
import torch.nn as nn


def _finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _primitive_metadata(value, path="metadata"):
    """Copy metadata while rejecting objects that are unsafe in state_dict."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = _primitive_metadata(item, f"{path}.{key}")
        return result
    raise TypeError(f"{path} must contain only primitive values")


def _tensor_signature(tensor):
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    return (
        int(getattr(tensor, "_version", 0)),
        str(cpu_tensor.dtype),
        tuple(cpu_tensor.shape),
        tuple(cpu_tensor.reshape(-1).tolist()),
    )


class PixelatedLightSource(nn.Module):
    """A nonnegative, unit-flux 2D source map with trainable pixel weights.

    The initial distribution is uniform over an annulus. Pixels inside the
    source disk but outside that initial annulus start with a small weight, so
    training can move toward another source shape. Pixels outside
    ``sigma_outer`` remain excluded.

    ``sigma_inner`` and ``sigma_outer`` are normalized partial-coherence
    coordinates, not wafer distances. The annular initialization is a
    starting prior; it is not a claim that the original paper's source was
    annular.
    """

    def __init__(
        self,
        grid_size=9,
        sigma_inner=0.3,
        sigma_outer=0.9,
        background_logit=-6.0,
    ):
        super(PixelatedLightSource, self).__init__()
        if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 5 or grid_size % 2 == 0:
            raise ValueError("grid_size must be an odd integer of at least 5")
        if not (0.0 <= sigma_inner < sigma_outer <= 1.0):
            raise ValueError("require 0 <= sigma_inner < sigma_outer <= 1")
        sigma_inner = _finite_float(sigma_inner, "sigma_inner")
        sigma_outer = _finite_float(sigma_outer, "sigma_outer")
        background_logit = _finite_float(background_logit, "background_logit")

        # Keep geometry in double precision even though trainable logits start
        # in float32. A later .double() converts both consistently.
        axis = torch.linspace(
            -sigma_outer, sigma_outer, int(grid_size), dtype=torch.float64
        )
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        pupil_support = radius <= sigma_outer
        initial_annulus = pupil_support & (radius >= sigma_inner)
        if not torch.any(initial_annulus):
            raise ValueError("source grid has no samples in the initial annulus")

        logits = torch.full(
            (int(grid_size), int(grid_size)),
            background_logit,
            dtype=torch.float32,
        )
        logits[initial_annulus] = 0.0

        self.logits = nn.Parameter(logits)
        self.register_buffer("_coordinates", torch.stack((xx, yy), dim=-1))
        self.register_buffer("_pupil_support", pupil_support)
        self.grid_size = int(grid_size)
        self.sigma_inner = sigma_inner
        self.sigma_outer = sigma_outer

    def weight_map(self):
        """Return the normalized source tensor, shaped ``(grid, grid)``."""
        floor = torch.finfo(self.logits.dtype).min
        masked_logits = self.logits.masked_fill(~self._pupil_support, floor)
        return torch.softmax(masked_logits.reshape(-1), dim=0).reshape(
            self.grid_size, self.grid_size
        )

    def distribution(self):
        """Return active source coordinates ``(N, 2)`` and weights."""
        weights = self.weight_map()
        coordinates = self._coordinates[self._pupil_support]
        return coordinates, weights[self._pupil_support]


class FixedMaskSourceBasis(nn.Module):
    """Detached per-source aerial intensities for one fixed mask.

    A basis contains no reference to the mask tensor. Construct it with
    :meth:`DifferentiableAbbeLitho.prepare_basis`, passing ``mask.detach()``
    when the caller started with a tensor that requires gradients. The basis
    is safe to move and serialize through ``state_dict``; its metadata is a
    dictionary of primitive values stored as module extra state.
    """

    def __init__(
        self,
        intensities,
        source_coordinates,
        metadata,
        source_support=None,
    ):
        super(FixedMaskSourceBasis, self).__init__()
        if not isinstance(intensities, torch.Tensor) or intensities.ndim != 4:
            raise ValueError("intensities must have shape (B,N,H,W)")
        if not torch.is_floating_point(intensities) or intensities.is_complex():
            raise ValueError("intensities must be a real floating tensor")
        if (
            not isinstance(source_coordinates, torch.Tensor)
            or source_coordinates.ndim != 2
            or source_coordinates.shape[1] != 2
        ):
            raise ValueError("source_coordinates must have shape (N,2)")
        if source_support is None:
            source_support = torch.empty(0, dtype=torch.bool)
        if (
            not isinstance(source_support, torch.Tensor)
            or source_support.dtype != torch.bool
            or source_support.ndim < 1
        ):
            raise ValueError("source_support must be a bool tensor")
        if source_support.numel() and intensities.shape[1] != int(
            source_support.sum().item()
        ):
            raise ValueError("source_support does not match source count")

        self.register_buffer("intensities", intensities.detach().clone())
        self.register_buffer(
            "source_coordinates", source_coordinates.detach().clone()
        )
        self.register_buffer("source_support", source_support.detach().clone())
        self._metadata = _primitive_metadata(metadata)

    @property
    def metadata(self):
        """Return a detached copy of basis metadata."""
        return copy.deepcopy(self._metadata)

    @classmethod
    def from_basis(cls, basis):
        """Clone a basis without retaining references to its buffers."""
        if not isinstance(basis, cls):
            raise TypeError("basis must be a FixedMaskSourceBasis")
        return cls(
            basis.intensities,
            basis.source_coordinates,
            basis.metadata,
            source_support=basis.source_support,
        )

    def get_extra_state(self):
        return self.metadata

    def set_extra_state(self, state):
        if state is None:
            self._metadata = {}
            return
        self._metadata = _primitive_metadata(state)


class DifferentiableAbbeLitho(nn.Module):
    """Scalar Abbe aerial-image simulator with source and mask gradients.

    The source weights are summed incoherently. For source point ``s``, the
    coherent field is the inverse FFT of the mask spectrum multiplied by the
    shifted circular pupil ``P(f + s)``. The source is shared across the
    batch.

    Args:
        source: A trainable :class:`PixelatedLightSource` module.
        numerical_aperture: Image-side NA used to set the pupil cutoff.
        wavelength_nm: Vacuum wavelength in nanometers.
        pixel_size_nm: Wafer-plane spacing between adjacent mask pixels.
        source_chunk_size: Source samples evaluated at once to limit memory.
        refractive_index: Explicit medium index required for nonzero defocus.
        cache_max_bytes: Maximum memory retained by the optical transfer cache.
    """

    def __init__(
        self,
        source,
        numerical_aperture=1.35,
        wavelength_nm=193.0,
        pixel_size_nm=1.0,
        source_chunk_size=8,
        refractive_index=None,
        cache_max_bytes=32 * 1024**2,
    ):
        super(DifferentiableAbbeLitho, self).__init__()
        if not isinstance(source, PixelatedLightSource):
            raise TypeError("source must be a PixelatedLightSource")
        numerical_aperture = _finite_float(
            numerical_aperture, "numerical_aperture"
        )
        wavelength_nm = _finite_float(wavelength_nm, "wavelength_nm")
        pixel_size_nm = _finite_float(pixel_size_nm, "pixel_size_nm")
        if numerical_aperture <= 0.0 or wavelength_nm <= 0.0:
            raise ValueError("numerical_aperture and wavelength_nm must be positive")
        if pixel_size_nm <= 0.0 or int(source_chunk_size) < 1:
            raise ValueError("pixel_size_nm and source_chunk_size must be positive")
        if refractive_index is not None:
            refractive_index = _finite_float(refractive_index, "refractive_index")
            if refractive_index <= 0.0:
                raise ValueError("refractive_index must be positive")
        try:
            cache_max_bytes = int(cache_max_bytes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("cache_max_bytes must be non-negative") from exc
        if cache_max_bytes < 0:
            raise ValueError("cache_max_bytes must be non-negative")

        self.source = source
        self.numerical_aperture = numerical_aperture
        self.wavelength_nm = wavelength_nm
        self.pixel_size_nm = pixel_size_nm
        self.source_chunk_size = int(source_chunk_size)
        self.refractive_index = refractive_index
        self.cache_max_bytes = cache_max_bytes
        self._optical_cache = OrderedDict()
        self._cache_bytes = 0

    def optical_config(self):
        """Return optical parameters as primitive values for logs and checks."""
        return {
            "numerical_aperture": self.numerical_aperture,
            "wavelength_nm": self.wavelength_nm,
            "pixel_size_nm": self.pixel_size_nm,
            "refractive_index": self.refractive_index,
        }

    def clear_cache(self):
        """Drop all cached frequency and pupil/phase tensors."""
        self._optical_cache.clear()
        self._cache_bytes = 0

    def _apply(self, fn):
        result = super()._apply(fn)
        self.clear_cache()
        return result

    def load_state_dict(self, *args, **kwargs):
        self.clear_cache()
        result = super().load_state_dict(*args, **kwargs)
        self.clear_cache()
        return result

    def _validate_defocus(self, defocus_nm):
        defocus_nm = _finite_float(defocus_nm, "defocus_nm")
        if defocus_nm != 0.0:
            if self.refractive_index is None:
                raise ValueError(
                    "nonzero defocus requires explicit refractive_index"
                )
            if self.refractive_index < self.numerical_aperture:
                raise ValueError(
                    "refractive_index must be at least numerical_aperture"
                )
        return defocus_nm

    @staticmethod
    def _validate_mask(mask, require_no_grad=False):
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor")
        if mask.is_complex():
            raise ValueError("mask must be real")
        if mask.ndim == 2:
            normalized_shape = (1, 1, mask.shape[0], mask.shape[1])
        elif mask.ndim == 3:
            normalized_shape = (mask.shape[0], 1, mask.shape[1], mask.shape[2])
        elif mask.ndim == 4 and mask.shape[1] == 1:
            normalized_shape = tuple(mask.shape)
        else:
            raise ValueError("mask must have shape (H,W), (B,H,W), or (B,1,H,W)")
        if any(int(size) < 1 for size in normalized_shape):
            raise ValueError("mask dimensions must be non-empty")
        if require_no_grad and mask.requires_grad:
            raise ValueError(
                "prepare_basis requires a non-gradient mask; pass mask.detach()"
            )
        if not bool(torch.isfinite(mask).all().item()):
            raise ValueError("mask must contain only finite values")
        return mask.reshape(normalized_shape)

    def _work_dtype(self, mask):
        if mask.dtype == torch.float64 or self.source.logits.dtype == torch.float64:
            return torch.float64
        return torch.float32

    def _source_geometry_key(self):
        return (
            _tensor_signature(self.source._coordinates),
            _tensor_signature(self.source._pupil_support),
        )

    def _cache_key(self, height, width, work_dtype, device, defocus_nm):
        return (
            int(height),
            int(width),
            self.pixel_size_nm,
            self.numerical_aperture,
            self.wavelength_nm,
            self.refractive_index,
            defocus_nm,
            str(device),
            str(self.source.logits.device),
            str(work_dtype),
            self._source_geometry_key(),
        )

    def _make_optical_transfer(
        self, height, width, coordinates, work_dtype, device, defocus_nm
    ):
        fx = torch.fft.fftfreq(
            width, d=self.pixel_size_nm, device=device, dtype=work_dtype
        )
        fy = torch.fft.fftfreq(
            height, d=self.pixel_size_nm, device=device, dtype=work_dtype
        )
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        cutoff = self.numerical_aperture / self.wavelength_nm
        fx_normalized = fx_grid / cutoff
        fy_normalized = fy_grid / cutoff
        sx = coordinates[:, 0, None, None]
        sy = coordinates[:, 1, None, None]
        shifted_x = fx_normalized[None] + sx
        shifted_y = fy_normalized[None] + sy
        pupil = shifted_x.square() + shifted_y.square() <= 1.0
        complex_dtype = (
            torch.complex128 if work_dtype == torch.float64 else torch.complex64
        )
        if defocus_nm == 0.0:
            return pupil.to(dtype=complex_dtype)

        # Clamp before sqrt and mask the result so out-of-pupil samples can
        # never create NaNs, even when the square root argument is negative.
        fshift_x = shifted_x * cutoff
        fshift_y = shifted_y * cutoff
        lambda_f_squared = self.wavelength_nm**2 * (
            fshift_x.square() + fshift_y.square()
        )
        radicand = (self.refractive_index**2 - lambda_f_squared).clamp_min(0.0)
        phase = (
            2.0
            * math.pi
            * defocus_nm
            / self.wavelength_nm
            * (torch.sqrt(radicand) - self.refractive_index)
        )
        phase = torch.where(pupil, phase, torch.zeros_like(phase))
        complex_phase = torch.complex(torch.zeros_like(phase), phase)
        return pupil.to(dtype=complex_dtype) * torch.exp(complex_phase)

    def _cached_transfer(
        self, height, width, coordinates, work_dtype, device, defocus_nm
    ):
        key = self._cache_key(height, width, work_dtype, device, defocus_nm)
        transfer = self._optical_cache.pop(key, None)
        if transfer is not None:
            self._optical_cache[key] = transfer
            return transfer

        complex_bytes = 16 if work_dtype == torch.float64 else 8
        required_bytes = (
            int(coordinates.shape[0]) * int(height) * int(width) * complex_bytes
        )
        if required_bytes > self.cache_max_bytes:
            # The caller will build only source chunks, avoiding a large cache
            # allocation while retaining the normal chunked execution path.
            return None
        transfer = self._make_optical_transfer(
            height, width, coordinates, work_dtype, device, defocus_nm
        )
        self._optical_cache[key] = transfer
        self._cache_bytes += required_bytes
        while self._cache_bytes > self.cache_max_bytes and self._optical_cache:
            _, old_transfer = self._optical_cache.popitem(last=False)
            self._cache_bytes -= old_transfer.numel() * old_transfer.element_size()
        return transfer

    def _active_coordinates(self):
        return self.source._coordinates[self.source._pupil_support]

    def forward(self, mask, defocus_nm=0.0):
        mask = self._validate_mask(mask)
        defocus_nm = self._validate_defocus(defocus_nm)
        work_dtype = self._work_dtype(mask)
        mask = mask[:, 0].to(dtype=work_dtype)
        if not bool(torch.isfinite(mask).all().item()):
            raise ValueError("mask must remain finite after dtype conversion")
        batch_size, height, width = mask.shape
        mask_spectrum = torch.fft.fft2(mask, dim=(-2, -1), norm="backward")

        coordinates, source_weights = self.source.distribution()
        coordinates = coordinates.detach().to(device=mask.device, dtype=work_dtype)
        source_weights = source_weights.to(device=mask.device, dtype=work_dtype)
        transfer = self._cached_transfer(
            height,
            width,
            coordinates,
            work_dtype,
            mask.device,
            defocus_nm,
        )

        aerial = torch.zeros(
            (batch_size, height, width), dtype=work_dtype, device=mask.device
        )
        for start in range(0, coordinates.shape[0], self.source_chunk_size):
            stop = min(start + self.source_chunk_size, coordinates.shape[0])
            if transfer is None:
                transfer_chunk = self._make_optical_transfer(
                    height,
                    width,
                    coordinates[start:stop],
                    work_dtype,
                    mask.device,
                    defocus_nm,
                )
            else:
                transfer_chunk = transfer[start:stop]
            fields = torch.fft.ifft2(
                mask_spectrum[:, None] * transfer_chunk[None],
                dim=(-2, -1),
                norm="backward",
            )
            intensity = fields.real.square() + fields.imag.square()
            aerial = aerial + (
                intensity * source_weights[None, start:stop, None, None]
            ).sum(dim=1)

        return aerial.unsqueeze(1)

    def _basis_budget_bytes(self, batch_size, source_count, height, width, work_dtype):
        real_bytes = 8 if work_dtype == torch.float64 else 4
        complex_bytes = 2 * real_bytes
        chunk_count = min(self.source_chunk_size, max(source_count, 1))
        output_bytes = batch_size * source_count * height * width * real_bytes
        spectrum_bytes = batch_size * height * width * complex_bytes
        transfer_bytes = chunk_count * height * width * complex_bytes
        field_bytes = batch_size * chunk_count * height * width * complex_bytes
        # Include the detached output clone, live chunk buffers and intensity
        # temporaries. FFT library workspaces and allocator overhead vary by
        # device and remain outside this estimate.
        intensity_bytes = batch_size * chunk_count * height * width * real_bytes
        return (2 * output_bytes + spectrum_bytes + 2 * transfer_bytes
                + 2 * field_bytes + 3 * intensity_bytes)

    def prepare_basis(self, mask, defocus_nm=0.0, max_bytes=512 * 1024**2):
        """Prepare detached per-source intensities for a fixed, non-grad mask.

        This method does not retain ``mask`` and runs under ``no_grad``. A
        mask that requires gradients is rejected; callers should pass
        ``mask.detach()`` explicitly when that is intentional. The memory
        budget is checked before FFTs or the basis output is allocated.
        """
        mask = self._validate_mask(mask, require_no_grad=True)
        defocus_nm = self._validate_defocus(defocus_nm)
        try:
            max_bytes = int(max_bytes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_bytes must be non-negative") from exc
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")

        work_dtype = self._work_dtype(mask)
        batch_size, _, height, width = mask.shape
        coordinates_snapshot = self._active_coordinates()
        source_count = int(coordinates_snapshot.shape[0])
        required_bytes = self._basis_budget_bytes(
            batch_size, source_count, height, width, work_dtype
        )
        if required_bytes > max_bytes:
            raise ValueError(
                "basis exceeds max_bytes before FFT/allocation: "
                f"requires about {required_bytes} bytes"
            )

        with torch.no_grad():
            mask = mask[:, 0].to(dtype=work_dtype)
            if not bool(torch.isfinite(mask).all().item()):
                raise ValueError("mask must remain finite after dtype conversion")
            mask_spectrum = torch.fft.fft2(mask, dim=(-2, -1), norm="backward")
            intensities = torch.empty(
                (batch_size, source_count, height, width),
                dtype=work_dtype,
                device=mask.device,
            )
            coordinates = coordinates_snapshot.detach().to(
                device=mask.device, dtype=work_dtype
            )
            for start in range(0, source_count, self.source_chunk_size):
                stop = min(start + self.source_chunk_size, source_count)
                transfer = self._make_optical_transfer(
                    height,
                    width,
                    coordinates[start:stop],
                    work_dtype,
                    mask.device,
                    defocus_nm,
                )
                fields = torch.fft.ifft2(
                    mask_spectrum[:, None] * transfer[None],
                    dim=(-2, -1),
                    norm="backward",
                )
                intensities[:, start:stop] = (
                    fields.real.square() + fields.imag.square()
                )

            metadata = {
                "version": 1,
                "optical_config": self.optical_config(),
                "raster": {
                    "height": int(height),
                    "width": int(width),
                    "pixel_size_nm": self.pixel_size_nm,
                },
                "defocus_nm": defocus_nm,
                "source_count": source_count,
                "dtype": str(intensities.dtype),
            }
            basis = FixedMaskSourceBasis(
                intensities,
                coordinates_snapshot,
                metadata,
                source_support=self.source._pupil_support,
            )
        return basis

    def _validate_basis(self, basis):
        if not isinstance(basis, FixedMaskSourceBasis):
            raise TypeError("basis must be a FixedMaskSourceBasis")
        metadata = basis.metadata
        if metadata.get("optical_config") != self.optical_config():
            raise ValueError("basis optical configuration is stale")
        defocus_nm = self._validate_defocus(metadata.get("defocus_nm"))
        raster = metadata.get("raster")
        if not isinstance(raster, dict):
            raise ValueError("basis raster metadata is missing")
        if raster.get("pixel_size_nm") != self.pixel_size_nm:
            raise ValueError("basis raster pixel size is stale")
        if basis.intensities.ndim != 4:
            raise ValueError("basis intensities must have shape (B,N,H,W)")
        if (
            basis.intensities.shape[2] != int(raster.get("height", -1))
            or basis.intensities.shape[3] != int(raster.get("width", -1))
        ):
            raise ValueError("basis raster shape metadata is invalid")

        coordinates, source_weights = self.source.distribution()
        if not torch.equal(
            basis.source_coordinates.detach().to(device="cpu"),
            coordinates.detach().to(device="cpu"),
        ):
            raise ValueError("basis source coordinates are stale")
        if basis.source_support.numel() == 0 or not torch.equal(
            basis.source_support.detach().to(device="cpu"),
            self.source._pupil_support.detach().to(device="cpu"),
        ):
            raise ValueError("basis source support is stale")
        if basis.intensities.shape[1] != source_weights.shape[0]:
            raise ValueError("basis source count is stale")
        return source_weights, defocus_nm

    def evaluate_basis(self, basis, defocus_nm=None):
        """Contract source weights without FFT; each basis retains its focus.

        Bases for multiple process corners can coexist. An optional explicit
        defocus checks that the caller selected the intended corner.
        """
        source_weights, basis_defocus = self._validate_basis(basis)
        if defocus_nm is not None and self._validate_defocus(defocus_nm) != basis_defocus:
            raise ValueError("basis defocus does not match the requested corner")
        weights = source_weights.to(
            device=basis.intensities.device, dtype=basis.intensities.dtype
        )
        aerial = torch.einsum("bnhw,n->bhw", basis.intensities, weights)
        return aerial.unsqueeze(1)


def resist_image(aerial, threshold=0.225, steepness=50.0, dose=1.0):
    """Apply LithoBench-style sigmoid resist threshold to aerial intensity.

    Dose scales intensity here, as exposure energy. This differs from the
    current SOCS implementation, which scales mask amplitude before squaring.
    """
    if not all(math.isfinite(float(v)) for v in (dose, steepness, threshold)):
        raise ValueError("resist parameters must be finite")
    if dose < 0.0 or steepness <= 0.0:
        raise ValueError("dose must be non-negative and steepness must be positive")
    return torch.sigmoid(steepness * (dose * aerial - threshold))
