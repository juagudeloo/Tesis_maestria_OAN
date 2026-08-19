#!/usr/bin/env python3
"""Drive NICOLE in synthesis mode on a set of pre-exported MUISCA predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.hinode_wavelengths import N_WL_OBSERVED
from utils.synthesis import NicoleRunner, SynthesisConfig


def parse_pixel(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--pixel expects 'ix,iy' (got {value!r})")
    return int(parts[0]), int(parts[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions-h5", type=Path, required=True,
                   help="HDF5 produced by scripts/synthesis/export_predictions.py")
    p.add_argument("--pixel", type=parse_pixel, action="append",
                   help="Pixel to synthesize (repeatable). Omit to run all exported pixels.")
    p.add_argument("--nicole-root", type=Path,
                   default=Path("/scratchsan/observatorio/juagudeloo/NICOLE"))
    p.add_argument("--nicole-assets", type=Path,
                   default=Path("/scratchsan/observatorio/juagudeloo/MUISCA/data/nicole_assets"))
    # Default None -> SynthesisConfig derives the grid from the MODEST FITS header
    # (utils.hinode_wavelengths). Pass these only to deliberately override it; the
    # old hardcoded 6300.796 / 21.5 defaults were 0.0776 A (~3.7 km/s) off the
    # observed axis.
    p.add_argument("--wl-first", type=float, default=None)
    p.add_argument("--wl-step-mA", type=float, default=None)
    p.add_argument("--n-wl", type=int, default=N_WL_OBSERVED)
    p.add_argument("--v-mic-cms", type=float, default=1.0e5)
    p.add_argument("--v-mac-cms", type=float, default=0.0)
    p.add_argument("--el-p-seed", type=float, default=1.0)
    p.add_argument(
        "--keep-workdirs",
        action="store_true",
        help=(
            "Keep each pixel's pix_<ix>_<iy>/ working directory (NICOLE's ASCII/binary "
            "inputs, logs, output model) after syntheses.h5 is written. Default is to "
            "delete them -- the synthesized profile is already captured in syntheses.h5, "
            "and the workdirs are regenerable by re-running this script. Pass this to "
            "keep them around for debugging a specific pixel."
        ),
    )
    args = p.parse_args()

    with h5py.File(args.predictions_h5, "r") as h5:
        source = h5.attrs.get("source", "modest")
        experiment_root = h5.attrs.get("experiment_root", "")
        model_type = h5.attrs.get("model_type", "")
        region_label = h5.attrs.get("region_label", "whole")
        muram_step = h5.attrs.get("muram_step", None)
        add_gt_pressure = h5.attrs.get("add_gt_pressure", False)
        # output_root is read directly when present (added alongside muram
        # support) rather than climbing a fixed number of parents from
        # predictions_h5 -- out_dir()'s depth now varies by source (muram
        # inserts a muram/step-N segment, modest doesn't), so a fixed climb
        # is no longer reliable. The climb stays only as a fallback for
        # predictions.h5 files written before this attr existed.
        output_root_attr = h5.attrs.get("output_root", None)

    output_root = (
        Path(str(output_root_attr))
        if output_root_attr is not None
        else args.predictions_h5.resolve().parent.parent.parent.parent
    )

    cfg = SynthesisConfig(
        source=str(source),
        experiment_root=str(experiment_root),
        model_type=str(model_type),
        region_label=str(region_label),
        muram_step=int(muram_step) if muram_step is not None else None,
        add_gt_pressure=bool(add_gt_pressure),
        output_root=output_root,
        nicole_root=args.nicole_root,
        nicole_assets=args.nicole_assets,
        wl_first=args.wl_first,
        wl_step_mA=args.wl_step_mA,
        n_wl=args.n_wl,
        v_mic_cms=args.v_mic_cms,
        v_mac_cms=args.v_mac_cms,
        el_p_seed=args.el_p_seed,
    )

    runner = NicoleRunner(cfg)
    out = runner.run_pixels_from_h5(
        args.predictions_h5, pixels=args.pixel, cleanup_workdirs=not args.keep_workdirs
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
