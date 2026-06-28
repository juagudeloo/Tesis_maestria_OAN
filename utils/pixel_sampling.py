"""Stratified pixel sampling for NICOLE synthesis testing.

Given a cropped MODEST region and a trained MUISCA model, stratifies pixels
by predicted |B_LOS| at the deepest optical-depth level (logτ = 0), samples
representative pixels per magnitude bin, and produces a diagnostic PNG showing
where on the map those pixels sit.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.synthesis import PredictionExporter, SynthesisConfig, WholeRegionPrediction


@dataclass
class PixelSamplingResult:
    """Result of stratified pixel sampling over a region."""

    pixels: list[tuple[int, int]]       # Selected (ix, iy) tuples
    bin_of_pixel: dict[tuple[int, int], int]  # Maps each pixel to its bin index
    bin_edges: np.ndarray               # (n_bins+1,) quantile boundaries in Gauss
    bin_values: list[float]             # Per-bin |B_LOS| values for selected pixels
    tau_index: int                      # Deepest level (-1)
    logtau_value: float                 # logtau[tau_index]
    abs_bz_map: np.ndarray              # (pred_nx, pred_ny) |B_LOS| map at tau_index
    pred_nx: int
    pred_ny: int


def sample_pixels_by_abs_bz(
    cfg: SynthesisConfig,
    device: torch.device,
    n_bins: int = 5,
    n_per_bin: int = 3,
    seed: int = 0,
) -> PixelSamplingResult:
    """Stratify a region by |B_LOS| at the deepest level and sample pixels per bin.

    Bin edges are log-spaced (equal-width in log₁₀(|B_LOS|) space) to give each bin
    a physically meaningful order-of-magnitude field regime. Pixel counts per bin will
    vary (high-field bins may be sparse or empty on quiet-Sun crops).

    Parameters
    ----------
    cfg : SynthesisConfig
        Configuration specifying experiment, model, region, crop bounds.
    device : torch.device
        Torch device for model inference.
    n_bins : int, default 5
        Number of log-spaced bins for |B_LOS|.
    n_per_bin : int, default 3
        Target number of pixels per bin.
    seed : int, default 0
        Random seed for reproducible sampling.

    Returns
    -------
    PixelSamplingResult
        Sampled pixels, bin assignments, log-spaced bin edges, and the |B_LOS| map.
    """
    rng = np.random.default_rng(seed)

    # Run inference over the whole region
    exporter = PredictionExporter(cfg, device)
    region = exporter.predict_whole_region()

    # Extract the |B_LOS| map at the deepest optical-depth level (logtau=0, index -1)
    tau_index = -1
    logtau_value = float(region.logtau[tau_index])
    bz_deep = region.pred_mhd["Bz"][:, :, tau_index]
    abs_bz_map = np.abs(bz_deep)

    # Extract finite values and derive log-space floor
    valid_vals = abs_bz_map[np.isfinite(abs_bz_map)]
    if len(valid_vals) == 0:
        raise RuntimeError(
            "No finite |B_LOS| values in the region; cannot stratify. Check for NaN/inf in predictions."
        )

    # For log-spacing, need a floor for log(0). Use the minimum positive value as the floor,
    # fallback to 1e-3 if all values are exactly 0.
    positive_vals = valid_vals[valid_vals > 0]
    if len(positive_vals) > 0:
        floor = float(np.min(positive_vals))
    else:
        floor = 1e-3
    ceil = float(np.max(valid_vals))

    # Build log-spaced bin edges (equal-width in log10 space)
    bin_edges = np.logspace(np.log10(floor), np.log10(ceil), n_bins + 1)

    # For binning, clip |B_LOS| to floor so that any 0.0 G pixels can be assigned
    # (they'll go to the first bin; their true value stays recorded as 0.0)
    abs_bz_clipped = np.maximum(abs_bz_map, floor)
    bin_indices = np.digitize(abs_bz_clipped.ravel(), bin_edges) - 1
    # Clip to valid bin range (handles floating-point edge case where val == ceil)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    bin_indices = bin_indices.reshape(abs_bz_map.shape)

    # Collect pixels per bin
    selected_pixels = []
    bin_of_pixel = {}
    bin_values = []
    n_actual_bins = len(bin_edges) - 1

    for b in range(n_actual_bins):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        # Get (ix, iy) coordinates of pixels in this bin
        ixs, iys = np.where(mask)
        candidates = list(zip(ixs.tolist(), iys.tolist()))
        # Sample up to n_per_bin from this bin
        n_sample = min(n_per_bin, len(candidates))
        if n_sample > 0:
            sampled = rng.choice(len(candidates), size=n_sample, replace=False)
            for idx in sampled:
                ix, iy = candidates[idx]
                selected_pixels.append((ix, iy))
                bin_of_pixel[(ix, iy)] = b
                # Record the true unclipped value (may be exactly 0.0)
                bin_values.append(float(abs_bz_map[ix, iy]))

    return PixelSamplingResult(
        pixels=selected_pixels,
        bin_of_pixel=bin_of_pixel,
        bin_edges=bin_edges,
        bin_values=bin_values,
        tau_index=tau_index,
        logtau_value=logtau_value,
        abs_bz_map=abs_bz_map,
        pred_nx=region.pred_nx,
        pred_ny=region.pred_ny,
    )


def plot_pixel_selection(
    result: PixelSamplingResult,
    out_path: Path,
    experiment_root: str,
    model_type: str,
) -> Path:
    """Plot the |B_LOS| map with selected pixel locations overlaid.

    Uses log-scale coloring to match the log-spaced bin edges, ensuring that
    the visual color gradation reflects the physical field-strength stratification.

    Parameters
    ----------
    result : PixelSamplingResult
        Result from sample_pixels_by_abs_bz().
    out_path : Path
        Output PNG file path.
    experiment_root : str
        Experiment name for the figure title.
    model_type : str
        Model type for the figure title.

    Returns
    -------
    Path
        The output PNG path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up the figure and colormap for |B_LOS| on a log scale
    fig, ax = plt.subplots(figsize=(12, 8))

    # Derive floor matching the bin edges (minimum positive value, or fallback)
    valid_vals = result.abs_bz_map[np.isfinite(result.abs_bz_map)]
    positive_vals = valid_vals[valid_vals > 0]
    if len(positive_vals) > 0:
        floor = float(np.min(positive_vals))
    else:
        floor = 1e-3
    ceil = float(np.max(valid_vals))

    # Plot the |B_LOS| map with LogNorm for log-scale color mapping
    # (axis 0 = row = y, axis 1 = col = x)
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=floor, vmax=ceil)
    im = ax.imshow(
        result.abs_bz_map.T,
        origin="lower",
        cmap="viridis",
        norm=norm,
        extent=[0, result.pred_nx, 0, result.pred_ny],
    )
    cbar = plt.colorbar(im, ax=ax, label="|B_LOS| [Gauss] (log scale)")

    # Overlay sampled pixels with one marker per bin
    n_bins = len(result.bin_edges) - 1
    cmap_scatter = plt.cm.get_cmap("tab10", max(n_bins, 1))
    bin_colors = {b: cmap_scatter(b % 10) for b in range(n_bins)}

    # Group pixels by bin for legend
    pixels_by_bin = {}
    for pix, b in result.bin_of_pixel.items():
        if b not in pixels_by_bin:
            pixels_by_bin[b] = []
        pixels_by_bin[b].append(pix)

    for b in sorted(pixels_by_bin.keys()):
        pixs = pixels_by_bin[b]
        ixs, iys = zip(*pixs)
        bin_lo = float(result.bin_edges[b])
        bin_hi = float(result.bin_edges[b + 1])
        # Use scientific notation for readable labels across 0.01-10000 G range
        label = f"bin {b}: {bin_lo:.2g}-{bin_hi:.2g} G"
        ax.plot(
            ixs,
            iys,
            "o",
            color=bin_colors[b],
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=8,
            label=label,
            alpha=0.8,
        )

    ax.set_xlabel("ix (pixel column)")
    ax.set_ylabel("iy (pixel row)")
    ax.set_title(
        f"|B_LOS| stratified sampling (log-spaced bins): {experiment_root}/{model_type}\n"
        f"logtau={result.logtau_value:.2f}, {len(result.pixels)} pixels in {len(pixels_by_bin)} non-empty bins"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def write_pixel_selection_outputs(
    result: PixelSamplingResult,
    out_dir: Path,
    experiment_root: str,
    model_type: str,
    crop_bounds: Optional[tuple[int, int, int, int]] = None,
) -> dict[str, Path]:
    """Write JSON and text snippets for the selected pixels.

    Parameters
    ----------
    result : PixelSamplingResult
        Result from sample_pixels_by_abs_bz().
    out_dir : Path
        Output directory for JSON and text files.
    experiment_root : str
        Experiment name (included in JSON).
    model_type : str
        Model type (included in JSON).
    crop_bounds : tuple[int, int, int, int], optional
        Crop bounds (y0, y1, x0, x1) to include in JSON.

    Returns
    -------
    dict[str, Path]
        Dict with keys "json", "text", "png" pointing to the output files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare JSON data
    json_data = {
        "experiment_root": experiment_root,
        "model_type": model_type,
        "n_pixels": len(result.pixels),
        "n_bins": len(result.bin_edges) - 1,
        "tau_index": result.tau_index,
        "logtau": result.logtau_value,
        "crop_bounds": crop_bounds,
        "bin_edges_gauss": result.bin_edges.tolist(),
        "pixels": [{"ix": ix, "iy": iy, "bin": result.bin_of_pixel[(ix, iy)],
                    "abs_bz_gauss": val}
                   for (ix, iy), val in zip(result.pixels, result.bin_values)],
    }

    # Write JSON
    json_path = out_dir / "selected_pixels.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Prepare text snippets for CLI usage
    pixel_args = " ".join([f"--pixel {ix},{iy}" for ix, iy in result.pixels])
    bash_array = "PIXELS=(" + " ".join([f'"{ix},{iy}"' for ix, iy in result.pixels]) + ")"

    text_content = f"""# Pixel selection for {experiment_root}/{model_type}
# logtau = {result.logtau_value:.2f}, |B_LOS| stratified sampling
# {len(result.pixels)} pixels in {len(result.bin_edges)-1} bins

# For export_predictions.py or run_nicole_synthesis.py (repeatable --pixel args):
python scripts/synthesis/export_predictions.py \\
    --experiment-root {experiment_root} \\
    --model-type {model_type} \\
    {pixel_args}

# For tools/run_nicole_synthesis.sh (bash array):
{bash_array}
"""

    text_path = out_dir / "pixel_selection_snippets.txt"
    with open(text_path, "w") as f:
        f.write(text_content)

    # Also write the PNG if result includes the map
    png_path = out_dir / "abs_bz_map_selected_pixels.png"
    plot_pixel_selection(result, png_path, experiment_root, model_type)

    return {
        "json": json_path,
        "text": text_path,
        "png": png_path,
    }
