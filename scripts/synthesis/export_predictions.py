#!/usr/bin/env python3
"""Export MUISCA predictions + matched observed Stokes to HDF5 for NICOLE synthesis.

CLI mirrors the conventions of scripts/analysis/modest_analysis.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.synthesis import PredictionExporter, SynthesisConfig


def parse_pixel(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--pixel expects 'ix,iy' (got {value!r})"
        )
    return int(parts[0]), int(parts[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["modest"], default="modest",
                   help="Prediction source (only 'modest' in v1)")
    p.add_argument("--experiment-root", required=True,
                   help="Folder name under output/experiments/")
    p.add_argument("--model-type", required=True,
                   help="Single model variation, e.g. 'wfa_only'")
    p.add_argument("--region-label", default="whole",
                   help="Label used in the output path (e.g. 'negative_region')")
    p.add_argument("--crop-bounds", type=int, nargs=4, default=None,
                   metavar=("Y0", "Y1", "X0", "X1"),
                   help="Region bounds passed to ModestData.load_all "
                        "(matches modest_analysis.py crop_bounds tuple order)")
    p.add_argument("--pixel", type=parse_pixel, action="append", required=True,
                   help="Pixel 'ix,iy' to export (repeatable)")
    p.add_argument("--polarization-mask", action="store_true",
                   help="Apply circular-polarization mask in MODEST loader")
    p.add_argument("--polarization-threshold", type=float, default=1e-2)
    p.add_argument("--modest-cache-dir", type=Path,
                   default=Path("/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache"))
    p.add_argument("--no-modest-cache", action="store_true")
    p.add_argument("--output-root", type=Path,
                   default=Path("/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis"))
    p.add_argument("--inference-batch-size", type=int, default=4096)
    args = p.parse_args()

    cfg = SynthesisConfig(
        source=args.source,
        experiment_root=args.experiment_root,
        model_type=args.model_type,
        region_label=args.region_label,
        crop_bounds=tuple(args.crop_bounds) if args.crop_bounds is not None else None,
        modest_cache_dir=args.modest_cache_dir,
        use_modest_cache=not args.no_modest_cache,
        apply_polarization_mask=args.polarization_mask,
        polarization_threshold=args.polarization_threshold,
        output_root=args.output_root,
        inference_batch_size=args.inference_batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exporter = PredictionExporter(cfg, device)
    out_path = exporter.export(pixels=args.pixel)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
