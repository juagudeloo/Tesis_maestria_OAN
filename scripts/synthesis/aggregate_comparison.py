#!/usr/bin/env python3
"""Aggregate chi2 distribution comparison across model variants, for posters/figures.

Given two or more --model-type variants already run through export_predictions.py
and run_nicole_synthesis.py on the full violin/aggregate tier (sample_pixels.py's
--n-per-bin pixels, not just the small overlay-tier subset compare_models.py
plots individual PNGs for), this script:

  1. Reuses the existing, unmodified SynthesisComparator.chi_square() per model
     (chi2 computation is cheap pure-numpy, so this script deliberately
     recomputes it independently rather than depending on compare_models.py
     having been run first -- both scripts stay independently runnable).
  2. Builds a long-format table (one row per pixel x model) of chi2_I/chi2_V
     with bin labels, and writes it to aggregate_chi2_long.json.
  3. Plots seaborn violin plots faceted by |B_LOS| bin (col) and colored by
     model variant (hue), log-y, one for Stokes I and one for Stokes V.

Output goes to SynthesisConfig.region_dir()/aggregate_plots/ -- for modest,
that's output_root/experiment_root/modest/region_label/aggregate_plots/; for muram,
output_root/experiment_root/muram/step-N[-gt-pressure]/aggregate_plots/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.synthesis import SynthesisComparator, SynthesisConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["modest", "muram"], default="modest")
    p.add_argument("--experiment-root", required=True)
    p.add_argument("--region-label", default="whole",
                    help="modest only -- ignored when --source muram")
    p.add_argument(
        "--model-type",
        action="append",
        required=True,
        help="Repeat for each model variant to compare, e.g. --model-type wfa_only --model-type no_physics",
    )
    p.add_argument("--muram-step", type=int, default=None,
                    help="MURaM simulation step number (required when --source muram)")
    p.add_argument("--add-gt-pressure", action="store_true",
                    help="Compare the ground-truth-pressure run (--source muram only)")
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis"),
    )
    p.add_argument("--sigma-i", type=float, default=1e-3)
    p.add_argument("--sigma-v", type=float, default=1e-3)
    args = p.parse_args()

    if args.source == "muram" and args.muram_step is None:
        p.error("--muram-step is required when --source muram")
    if args.add_gt_pressure and args.source != "muram":
        p.error("--add-gt-pressure requires --source muram")

    model_types = args.model_type
    if len(model_types) < 2:
        sys.exit(
            f"Aggregate comparison requires at least 2 --model-type values, got {model_types!r}"
        )

    chi2_by_model: dict[str, dict] = {}
    cfg_by_model: dict[str, SynthesisConfig] = {}

    for model_type in model_types:
        cfg_m = SynthesisConfig(
            source=args.source,
            experiment_root=args.experiment_root,
            model_type=model_type,
            region_label=args.region_label,
            muram_step=args.muram_step,
            add_gt_pressure=args.add_gt_pressure,
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

    # Pixel sets should be identical across models if sample_pixels.py (SPINOR-sourced,
    # model-independent) was used consistently -- intersect and warn if not.
    pixel_sets = {mt: set(chi2_by_model[mt].keys()) for mt in model_types}
    common_keys = set.intersection(*pixel_sets.values())
    for mt in model_types:
        missing = pixel_sets[mt] - common_keys
        if missing:
            print(
                f"  WARNING: {mt!r} has {len(missing)} pixel(s) not shared by all models "
                f"(dropped from comparison): {sorted(missing)}"
            )
    common_pixels = sorted(
        (int(ix), int(iy)) for ix, iy in (k.split(",") for k in common_keys)
    )
    print(f"Comparing {len(model_types)} models on {len(common_pixels)} common pixels (violin tier)")

    # Bin info -- read from the first model's selection (model-independent).
    bin_of_pixel: dict[tuple[int, int], int] = {}
    abs_bz_of_pixel: dict[tuple[int, int], float] = {}
    bin_edges: list[float] = []
    selection_json = cfg_by_model[model_types[0]].sampling_dir() / "pixel_selection" / "selected_pixels.json"
    if not selection_json.exists():
        sys.exit(f"Missing {selection_json} -- run sample_pixels.py for --model-type {model_types[0]} first")
    with open(selection_json) as f:
        selection = json.load(f)
    bin_edges = selection["bin_edges_gauss"]
    for entry in selection["pixels"]:
        bin_of_pixel[(entry["ix"], entry["iy"])] = entry["bin"]
        abs_bz_of_pixel[(entry["ix"], entry["iy"])] = entry["abs_bz_gauss"]

    out_dir = cfg_by_model[model_types[0]].region_dir() / "aggregate_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Long-format records: one row per (pixel, model_type).
    records = []
    for ix, iy in common_pixels:
        b = bin_of_pixel.get((ix, iy))
        if b is None:
            continue
        bin_lo, bin_hi = bin_edges[b], bin_edges[b + 1]
        for mt in model_types:
            c2 = chi2_by_model[mt][f"{ix},{iy}"]
            records.append({
                "ix": ix,
                "iy": iy,
                "bin": b,
                "bin_lo": bin_lo,
                "bin_hi": bin_hi,
                "abs_bz_gauss": abs_bz_of_pixel.get((ix, iy)),
                "model_type": mt,
                "chi2_I": c2["chi2_I"],
                "chi2_V": c2["chi2_V"],
            })

    long_data = {
        "experiment_root": args.experiment_root,
        "region_label": args.region_label,
        "model_types": model_types,
        "sigma_I": args.sigma_i,
        "sigma_V": args.sigma_v,
        "bin_edges_gauss": bin_edges,
        "records": records,
    }
    long_path = out_dir / "aggregate_chi2_long.json"
    with open(long_path, "w") as f:
        json.dump(long_data, f, indent=2)
    print(f"aggregate_chi2_long -> {long_path} ({len(records)} records)")

    # Violin plots, faceted by bin (col), colored by model (hue), log-y.
    df = pd.DataFrame(records)
    df["bin_label"] = df.apply(lambda r: f"{r['bin_lo']:.3g}-{r['bin_hi']:.3g} G", axis=1)
    bin_order = [
        f"{bin_edges[b]:.3g}-{bin_edges[b+1]:.3g} G"
        for b in sorted(df["bin"].unique())
    ]

    sns.set_theme(style="whitegrid")
    for col in ("chi2_I", "chi2_V"):
        g = sns.catplot(
            data=df, x="model_type", y=col, col="bin_label", col_order=bin_order,
            hue="model_type", kind="violin", col_wrap=min(len(bin_order), 5),
            height=4, aspect=0.9, cut=0, legend=False,
        )
        g.set(yscale="log")
        g.set_titles("{col_name}")
        for ax in g.axes.flat:
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30)
        g.set_ylabels(f"{col} (log scale)")
        png_path = out_dir / f"violin_{col}.png"
        g.fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"violin_{col} -> {png_path}")


if __name__ == "__main__":
    main()
