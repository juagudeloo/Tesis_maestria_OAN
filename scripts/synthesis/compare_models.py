#!/usr/bin/env python3
"""Compare NICOLE-synthesized Stokes across multiple trained model variants.

Given two or more --model-type variants already run through export_predictions.py
and run_nicole_synthesis.py for the same --experiment-root/--region-label (sharing
a pixel list -- see scripts/synthesis/sample_pixels.py, whose pixel selection is
model-independent), this script:

  1. Reuses the existing, unmodified SynthesisComparator.chi_square() per model
     (so chi2 numbers are guaranteed identical to running compare_synthesis.py
     on each model alone).
  2. Joins the per-model chi2 by pixel into cross_model_chi2.json.
  3. Aggregates chi2 by mean and median per |B_LOS| bin per model into
     bin_summary.json (the bin assignment comes from sample_pixels.py's
     selected_pixels.json).
  4. Plots one combined overlay PNG per pixel: one observed Stokes I/V curve
     plus one synthesized curve per model variant, on the same axes.

Output goes to output_root/experiment_root/region_label/cross_model_comparison/,
a path that isn't owned by any single model variant.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.synthesis import SynthesisComparator, SynthesisConfig

DEFAULT_COLORS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


def _format_bin_range(lo: float, hi: float) -> str:
    return f"{lo:.4g}-{hi:.4g}"


def _read_raw_arrays(predictions_h5: Path, syntheses_h5: Path) -> dict:
    """Read predictions.h5/syntheses.h5 directly for joined plotting use.

    Mirrors the read pattern compare_synthesis.py already uses (it reads
    syntheses.h5's "pixels" dataset directly rather than going through the
    comparator) -- SynthesisComparator has no public accessor for the pixel
    arrays it loaded, only private ones, so reading the files directly here
    avoids depending on those.
    """
    with h5py.File(predictions_h5, "r") as h5:
        pred_pixels = h5["pixels"][...]
        stokes_obs = h5["stokes_obs"][...]
        wavelengths = h5["wavelengths"][...]
    with h5py.File(syntheses_h5, "r") as h5:
        synth_pixels = h5["pixels"][...]
        stokes_synth = h5["stokes_synth"][...]
    return {
        "pred_index": {(int(ix), int(iy)): i for i, (ix, iy) in enumerate(pred_pixels.tolist())},
        "stokes_obs": stokes_obs,
        "wavelengths": wavelengths,
        "synth_index": {(int(ix), int(iy)): i for i, (ix, iy) in enumerate(synth_pixels.tolist())},
        "stokes_synth": stokes_synth,
    }


def plot_combined_overlay(
    ix: int,
    iy: int,
    model_types: list[str],
    raw_by_model: dict[str, dict],
    model_colors: dict[str, str],
    out_dir: Path,
    bin_range: tuple[float, float] | None,
    abs_b_los: float | None,
) -> Path:
    target_dir = out_dir / _format_bin_range(*bin_range) if bin_range is not None else out_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    # Observed Stokes is shared across models (same MODEST data) -- take it from
    # the first model, but sanity-check it agrees with the others.
    first = model_types[0]
    wl = raw_by_model[first]["wavelengths"]
    i_obs = raw_by_model[first]["stokes_obs"][raw_by_model[first]["pred_index"][(ix, iy)], 0]
    v_obs = raw_by_model[first]["stokes_obs"][raw_by_model[first]["pred_index"][(ix, iy)], 1]
    for mt in model_types[1:]:
        other_i_obs = raw_by_model[mt]["stokes_obs"][raw_by_model[mt]["pred_index"][(ix, iy)], 0]
        if not np.allclose(i_obs, other_i_obs, atol=1e-6, rtol=1e-4, equal_nan=True):
            print(
                f"  WARNING: observed Stokes I for pixel ({ix},{iy}) differs between "
                f"{first!r} and {mt!r} predictions.h5 -- expected identical (same MODEST data)"
            )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(wl, i_obs, color="k", label="Observed")
    axes[1].plot(wl, v_obs, color="k", label="Observed")

    for mt in model_types:
        raw = raw_by_model[mt]
        row = raw["stokes_synth"][raw["synth_index"][(ix, iy)]]
        i_syn, v_syn = row[0], row[3]
        axes[0].plot(wl, i_syn, color=model_colors[mt], ls="--", label=f"NICOLE ({mt})")
        axes[1].plot(wl, v_syn, color=model_colors[mt], ls="--", label=f"NICOLE ({mt})")

    axes[0].set_title(f"Stokes I  pix=({ix},{iy})")
    axes[0].set_xlabel("Wavelength [Å]")
    axes[0].set_ylabel("I / I_c")
    axes[0].legend()

    axes[1].set_title(f"Stokes V  pix=({ix},{iy})")
    axes[1].set_xlabel("Wavelength [Å]")
    axes[1].set_ylabel("V / I_c")
    axes[1].legend()

    suptitle = f"|B_LOS| = {abs_b_los:.3g} G" if abs_b_los is not None else f"pix=({ix},{iy})"
    fig.suptitle(suptitle)
    plt.tight_layout()
    out_path = target_dir / f"overlay_pix_{ix:05d}_{iy:05d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", required=True)
    p.add_argument("--region-label", default="whole")
    p.add_argument(
        "--model-type",
        action="append",
        required=True,
        help="Repeat for each model variant to compare, e.g. --model-type wfa_only --model-type no_physics",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis"),
    )
    p.add_argument("--sigma-i", type=float, default=1e-3)
    p.add_argument("--sigma-v", type=float, default=1e-3)
    args = p.parse_args()

    model_types = args.model_type
    if len(model_types) < 2:
        sys.exit(
            f"Cross-model comparison requires at least 2 --model-type values, got {model_types!r}"
        )

    chi2_by_model: dict[str, dict] = {}
    raw_by_model: dict[str, dict] = {}
    cfg_by_model: dict[str, SynthesisConfig] = {}

    for model_type in model_types:
        cfg_m = SynthesisConfig(
            source="modest",
            experiment_root=args.experiment_root,
            model_type=model_type,
            region_label=args.region_label,
            output_root=args.output_root,
        )
        cfg_by_model[model_type] = cfg_m
        pred_h5, synth_h5 = cfg_m.predictions_h5(), cfg_m.syntheses_h5()
        if not pred_h5.exists():
            sys.exit(f"Missing {pred_h5} -- run export_predictions.py for --model-type {model_type} first")
        if not synth_h5.exists():
            sys.exit(f"Missing {synth_h5} -- run run_nicole_synthesis.py for --model-type {model_type} first")

        comp = SynthesisComparator(cfg_m)
        comp.load(pred_h5, synth_h5)
        chi2_by_model[model_type] = comp.chi_square(sigma_I=args.sigma_i, sigma_V=args.sigma_v)
        raw_by_model[model_type] = _read_raw_arrays(pred_h5, synth_h5)

    # Pixel sets should be identical across models if sample_pixels.py (SPINOR-sourced,
    # model-independent) was used consistently -- intersect and warn if not.
    pixel_sets = {mt: set(raw_by_model[mt]["synth_index"].keys()) for mt in model_types}
    common_pixels = set.intersection(*pixel_sets.values())
    for mt in model_types:
        missing = pixel_sets[mt] - common_pixels
        if missing:
            print(
                f"  WARNING: {mt!r} has {len(missing)} pixel(s) not shared by all models "
                f"(dropped from comparison): {sorted(missing)}"
            )
    common_pixels = sorted(common_pixels)
    print(f"Comparing {len(model_types)} models on {len(common_pixels)} common pixels")

    # Bin info (model-independent -- read from the first model's selection, sanity-check
    # against the others if present).
    bin_of_pixel: dict[tuple[int, int], int] = {}
    abs_bz_of_pixel: dict[tuple[int, int], float] = {}
    bin_edges: list[float] = []
    for i, mt in enumerate(model_types):
        selection_json = cfg_by_model[mt].out_dir() / "pixel_selection" / "selected_pixels.json"
        if not selection_json.exists():
            continue
        with open(selection_json) as f:
            selection = json.load(f)
        if i == 0:
            bin_edges = selection["bin_edges_gauss"]
            for entry in selection["pixels"]:
                bin_of_pixel[(entry["ix"], entry["iy"])] = entry["bin"]
                abs_bz_of_pixel[(entry["ix"], entry["iy"])] = entry["abs_bz_gauss"]
        else:
            if selection["bin_edges_gauss"] != bin_edges:
                print(
                    f"  WARNING: {mt!r}'s selected_pixels.json has different bin_edges_gauss "
                    f"than {model_types[0]!r}'s -- pixel sampling may not have been run "
                    "consistently across models"
                )

    out_dir = args.output_root / args.experiment_root / args.region_label / "cross_model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # cross_model_chi2.json
    cross_chi2 = {
        "experiment_root": args.experiment_root,
        "region_label": args.region_label,
        "model_types": model_types,
        "sigma_I": args.sigma_i,
        "sigma_V": args.sigma_v,
        "pixels": {
            f"{ix},{iy}": {mt: chi2_by_model[mt][f"{ix},{iy}"] for mt in model_types}
            for ix, iy in common_pixels
        },
    }
    cross_chi2_path = out_dir / "cross_model_chi2.json"
    with open(cross_chi2_path, "w") as f:
        json.dump(cross_chi2, f, indent=2)
    print(f"cross_model_chi2 -> {cross_chi2_path}")

    # bin_summary.json
    bins_summary = []
    if bin_edges:
        n_bins = len(bin_edges) - 1
        for b in range(n_bins):
            pixels_in_bin = [
                (ix, iy) for (ix, iy) in common_pixels if bin_of_pixel.get((ix, iy)) == b
            ]
            if not pixels_in_bin:
                continue
            models_summary = {}
            for mt in model_types:
                chi2_I_vals = [chi2_by_model[mt][f"{ix},{iy}"]["chi2_I"] for ix, iy in pixels_in_bin]
                chi2_V_vals = [chi2_by_model[mt][f"{ix},{iy}"]["chi2_V"] for ix, iy in pixels_in_bin]
                models_summary[mt] = {
                    "n_pixels": len(pixels_in_bin),
                    "chi2_I_mean": float(np.nanmean(chi2_I_vals)),
                    "chi2_I_median": float(np.nanmedian(chi2_I_vals)),
                    "chi2_V_mean": float(np.nanmean(chi2_V_vals)),
                    "chi2_V_median": float(np.nanmedian(chi2_V_vals)),
                }
            bins_summary.append(
                {"bin": b, "bin_lo": bin_edges[b], "bin_hi": bin_edges[b + 1], "models": models_summary}
            )
    bin_summary_data = {
        "experiment_root": args.experiment_root,
        "region_label": args.region_label,
        "model_types": model_types,
        "bin_edges_gauss": bin_edges,
        "bins": bins_summary,
    }
    bin_summary_path = out_dir / "bin_summary.json"
    with open(bin_summary_path, "w") as f:
        json.dump(bin_summary_data, f, indent=2)
    print(f"bin_summary -> {bin_summary_path}")

    # Print a readable table to stdout.
    print("\nPer-bin chi2 summary (mean / median):")
    for entry in bins_summary:
        print(f"  bin {entry['bin']}: [{entry['bin_lo']:.3g}, {entry['bin_hi']:.3g}] G")
        for mt in model_types:
            ms = entry["models"][mt]
            print(
                f"    {mt:>20s}: n={ms['n_pixels']:<3d} "
                f"chi2_I mean={ms['chi2_I_mean']:.3g} median={ms['chi2_I_median']:.3g}  "
                f"chi2_V mean={ms['chi2_V_mean']:.3g} median={ms['chi2_V_median']:.3g}"
            )

    # Combined overlay PNGs.
    model_colors = dict(zip(model_types, cycle(DEFAULT_COLORS)))
    print(f"\nWriting {len(common_pixels)} combined overlay PNGs...")
    for ix, iy in common_pixels:
        b = bin_of_pixel.get((ix, iy))
        bin_range = (bin_edges[b], bin_edges[b + 1]) if b is not None and bin_edges else None
        abs_b_los = abs_bz_of_pixel.get((ix, iy))
        png = plot_combined_overlay(
            ix, iy, model_types, raw_by_model, model_colors, out_dir, bin_range, abs_b_los
        )
        print(f"  overlay -> {png}")


if __name__ == "__main__":
    main()
