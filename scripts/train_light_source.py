"""Fit a shared source from two explicit, disjoint fixed-mask datasets."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from light_source import DifferentiableAbbeLitho, PixelatedLightSource
from source_training import SourceDataset, SourceFitConfig, fit_source, process_grid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--validation-data", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("work/light_source"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--band-weight", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--sigma-inner", type=float, default=0.3)
    parser.add_argument("--sigma-outer", type=float, default=0.9)
    parser.add_argument("--na", type=float, default=1.35)
    parser.add_argument("--wavelength-nm", type=float, default=193.0)
    parser.add_argument("--refractive-index", type=float, default=None,
                        help="required when a nonzero defocus is requested")
    parser.add_argument("--defocus-nm", type=float, nargs="+", default=[0.0])
    parser.add_argument("--doses", type=float, nargs="+", default=[0.98, 1.0, 1.02],
                        help="intensity multipliers; not SOCS amplitude factors")
    parser.add_argument("--source-chunk-size", type=int, default=8)
    parser.add_argument("--max-basis-mib", type=int, default=512)
    parser.add_argument("--max-total-basis-mib", type=int, default=512)
    parser.add_argument("--max-device-basis-mib", type=int, default=256,
                        help="maximum retained basis bytes per layout, including all focuses")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    train = SourceDataset.load(args.train_data)
    validation = SourceDataset.load(args.validation_data)
    source = PixelatedLightSource(args.grid_size, args.sigma_inner, args.sigma_outer)
    sim = DifferentiableAbbeLitho(
        source, numerical_aperture=args.na, wavelength_nm=args.wavelength_nm,
        pixel_size_nm=train.pixel_size_nm, source_chunk_size=args.source_chunk_size,
        refractive_index=args.refractive_index,
    ).to(args.device)
    report = fit_source(
        sim, train, validation, corners=process_grid(args.doses, args.defocus_nm),
        config=SourceFitConfig(
            steps=args.steps, learning_rate=args.learning_rate, band_weight=args.band_weight,
            max_basis_bytes=args.max_basis_mib * 1024 ** 2,
            max_total_basis_bytes=args.max_total_basis_mib * 1024 ** 2,
            max_device_basis_bytes=args.max_device_basis_mib * 1024 ** 2,
            seed=args.seed,
        ), output_dir=args.output,
    )
    print("Held-out Abbe metrics (%s; not calibrated to SOCS):" % report["band_type"])
    print("Before:", report["before"]["validation"]["mean"])
    print("After: ", report["after"]["validation"]["mean"])
    print("Saved:", args.output.resolve())
    return report


if __name__ == "__main__":
    main()
