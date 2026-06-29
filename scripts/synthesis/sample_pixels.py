#!/usr/bin/env python3
"""Stratified pixel sampling by |B_LOS| magnitude for NICOLE synthesis testing.

This is "step 0" of the MUISCA→NICOLE bridge: given a cropped MODEST region,
get a full-region |B_LOS| map at the deepest level from the SPINOR inversion
already bundled with the MODEST data (model-independent), stratify pixels
into magnitude bins, sample representative pixels per bin, and output both
machine-readable JSON and ready-to-paste CLI snippets for feeding into
export_predictions.py. Because the binning is model-independent, running this
script once per --model-type with the same --seed/--crop-bounds produces an
identical pixel selection across model variants, which is what makes
compare_models.py's cross-model comparison fair.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.pixel_sampling import (
    sample_pixels_by_abs_bz,
    write_pixel_selection_outputs,
)
from utils.synthesis import SynthesisConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment-root",
        required=True,
        help="Folder name under output/experiments/",
    )
    p.add_argument(
        "--model-type",
        required=True,
        help=(
            "Single model variation, e.g. 'wfa_only'. Only affects the output "
            "path -- the pixel selection itself is model-independent (SPINOR-sourced), "
            "so re-running with a different --model-type and the same --seed/"
            "--crop-bounds yields an identical pixel selection."
        ),
    )
    p.add_argument(
        "--region-label",
        default="whole",
        help="Label used in the output path (e.g. 'negative_region')",
    )
    p.add_argument(
        "--crop-bounds",
        type=int,
        nargs=4,
        default=None,
        metavar=("Y0", "Y1", "X0", "X1"),
        help="Region bounds passed to ModestData.load_all (Y0, Y1, X0, X1 order)",
    )
    p.add_argument(
        "--modest-cache-dir",
        type=Path,
        default=Path("/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache"),
    )
    p.add_argument(
        "--no-modest-cache",
        action="store_true",
        help="Disable caching of MODEST data",
    )
    p.add_argument(
        "--polarization-mask",
        action="store_true",
        help="Apply circular-polarization mask in MODEST loader",
    )
    p.add_argument(
        "--polarization-threshold",
        type=float,
        default=1e-2,
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis"),
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of magnitude bins for |B_LOS| stratification",
    )
    p.add_argument(
        "--n-per-bin",
        type=int,
        default=3,
        help="Target number of pixels per bin",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling",
    )
    args = p.parse_args()

    cfg = SynthesisConfig(
        source="modest",
        experiment_root=args.experiment_root,
        model_type=args.model_type,
        region_label=args.region_label,
        crop_bounds=tuple(args.crop_bounds) if args.crop_bounds is not None else None,
        modest_cache_dir=args.modest_cache_dir,
        use_modest_cache=not args.no_modest_cache,
        apply_polarization_mask=args.polarization_mask,
        polarization_threshold=args.polarization_threshold,
        output_root=args.output_root,
    )

    # Run stratified sampling (SPINOR-sourced, no model loading needed)
    print(f"\nSampling {args.n_bins} bins × {args.n_per_bin} pixels (seed={args.seed})...")
    result = sample_pixels_by_abs_bz(
        cfg=cfg,
        n_bins=args.n_bins,
        n_per_bin=args.n_per_bin,
        seed=args.seed,
    )

    print(f"Selected {len(result.pixels)} pixels across {len(result.bin_edges)-1} bins")
    for b in range(len(result.bin_edges) - 1):
        pixels_in_bin = [p for p, bb in result.bin_of_pixel.items() if bb == b]
        if pixels_in_bin:
            bin_lo = float(result.bin_edges[b])
            bin_hi = float(result.bin_edges[b + 1])
            print(f"  bin {b}: {len(pixels_in_bin)} pixels in [{bin_lo:.1f}, {bin_hi:.1f}] G")

    # Write outputs
    out_dir = cfg.out_dir() / "pixel_selection"
    outputs = write_pixel_selection_outputs(
        result=result,
        out_dir=out_dir,
        experiment_root=cfg.experiment_root,
        model_type=cfg.model_type,
        crop_bounds=cfg.crop_bounds,
    )

    print(f"\nOutputs:")
    for key, path in outputs.items():
        print(f"  {key} -> {path}")

    # Print ready-to-use snippets
    print("\n" + "=" * 80)
    with open(outputs["text"]) as f:
        print(f.read())
    print("=" * 80)


if __name__ == "__main__":
    main()
