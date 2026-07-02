#!/usr/bin/env python3
"""Compare NICOLE-synthesized Stokes against observed Stokes; emit PNG overlays and chi2.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.synthesis import SynthesisComparator, SynthesisConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions-h5", type=Path, required=True)
    p.add_argument("--syntheses-h5", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <predictions_h5 parent>/comparison/")
    p.add_argument("--sigma-i", type=float, default=1e-3)
    p.add_argument("--sigma-v", type=float, default=1e-3)
    args = p.parse_args()

    out_dir = args.output_dir or args.predictions_h5.parent / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.predictions_h5, "r") as h5:
        source = h5.attrs.get("source", "modest")
        experiment_root = h5.attrs.get("experiment_root", "")
        model_type = h5.attrs.get("model_type", "")
        region_label = h5.attrs.get("region_label", "whole")
        muram_step = h5.attrs.get("muram_step", None)
        add_gt_pressure = h5.attrs.get("add_gt_pressure", False)
        # output_root: read directly when present (added alongside muram support).
        # Previously this was never passed through at all, silently relying on
        # SynthesisConfig's default matching whatever --output-root the caller
        # actually used upstream -- reading the attr closes that latent gap too.
        output_root_attr = h5.attrs.get("output_root", None)

    cfg_kwargs = dict(
        source=str(source),
        experiment_root=str(experiment_root),
        model_type=str(model_type),
        region_label=str(region_label),
        muram_step=int(muram_step) if muram_step is not None else None,
        add_gt_pressure=bool(add_gt_pressure),
    )
    if output_root_attr is not None:
        cfg_kwargs["output_root"] = Path(str(output_root_attr))
    cfg = SynthesisConfig(**cfg_kwargs)
    comp = SynthesisComparator(cfg)
    comp.load(args.predictions_h5, args.syntheses_h5)

    chi2 = comp.chi_square(sigma_I=args.sigma_i, sigma_V=args.sigma_v)
    chi2_path = out_dir / "chi2.json"
    with open(chi2_path, "w") as f:
        json.dump(chi2, f, indent=2)
    print(f"chi2 -> {chi2_path}")

    with h5py.File(args.syntheses_h5, "r") as h5:
        pixels = h5["pixels"][...]
    for ix, iy in pixels.tolist():
        png = comp.plot_overlay(int(ix), int(iy), out_dir)
        print(f"  overlay -> {png}")


if __name__ == "__main__":
    main()
