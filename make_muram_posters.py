#!/usr/bin/env python3
"""
Compose per-tau MURaM poster images by assembling existing imshow, histogram,
and scatterplot PNGs for each trained weight w and for both noise conditions.

Inputs (discovered automatically):
- images/paper_experiment/
    - imshow_comparisons/{with_noise,without_noise}/tau_*/
    - histograms/{with_noise,without_noise}/tau_*/
    - scatterplots/{with_noise,without_noise}/tau_*/

Outputs:
- images/Compends/MURaM/tau_<val>/tau_<val>_MURaM.svg
- images/Compends/MURaM/tau_<val>/tau_<val>_MURaM.png

Usage:
    python make_muram_posters.py               # build posters for all tau folders found
    python make_muram_posters.py --tau 0.00    # only for a specific tau value
    python make_muram_posters.py --dry-run     # just print what would be done

Notes:
- The script discovers available weights by parsing filenames in each tau folder.
- Missing images are handled gracefully and shown as gray placeholders.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------- Configuration ----------
BASE_IMAGES_DIR = Path("images/paper_experiment")
OUT_BASE_DIR = Path("images/Compends/MURaM")
CONDITIONS = ["with_noise", "without_noise"]


def list_tau_dirs() -> List[str]:
    """List tau directory names (e.g., 'tau_-0.10', 'tau_0.00') present for both histograms/with_noise and without_noise.
    Falls back to union across any category/condition found.
    """
    tau_set = set()
    for category in ["histograms", "imshow_comparisons", "scatterplots"]:
        for cond in CONDITIONS:
            root = BASE_IMAGES_DIR / category / cond
            if not root.exists():
                continue
            for child in root.iterdir():
                if child.is_dir() and child.name.startswith("tau_"):
                    tau_set.add(child.name)
    # Sort numerically by tau value if possible
    def tau_key(tname: str) -> float:
        try:
            return float(tname.split("tau_")[-1])
        except Exception:
            return 0.0
    return sorted(tau_set, key=tau_key)


def parse_weight_from_filename(p: Path) -> Optional[float]:
    """Extract numeric weight value from filename by capturing the last 'w_<num>.png'.
    Examples:
      '087000_hist_T_tau_0.00_w_with_noise_w_0.0001.png' -> 0.0001
      '087000_atm_params_comparison_tau_0.00_w_without_noise_w_0.9.png' -> 0.9
    Returns None if no parse.
    """
    m = re.search(r"w_([0-9]*\.?[0-9]+)\.(?:png|svg)$", p.name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def discover_images_for_tau(tau_dir_name: str, condition: str) -> Tuple[List[float], Dict[str, Dict[float, Path]]]:
    """Discover available images for a given tau and condition.
    Returns:
      - weights: sorted list of unique numeric weights discovered across all categories
      - images: dict mapping keys to per-weight paths.
        keys: 'imshow', 'hist_T', 'hist_v', 'hist_B', 'scat_T', 'scat_v', 'scat_B'
    """
    images: Dict[str, Dict[float, Path]] = {
        'imshow': {},
        'hist_T': {}, 'hist_v': {}, 'hist_B': {},
        'scat_T': {}, 'scat_v': {}, 'scat_B': {},
    }

    # Directories
    im_dir = BASE_IMAGES_DIR / "imshow_comparisons" / condition / tau_dir_name
    hi_dir = BASE_IMAGES_DIR / "histograms" / condition / tau_dir_name
    sc_dir = BASE_IMAGES_DIR / "scatterplots" / condition / tau_dir_name

    # Collect imshow files
    if im_dir.exists():
        for p in im_dir.glob("*atm_params_comparison*_*_w_*.png"):
            w = parse_weight_from_filename(p)
            if w is not None:
                images['imshow'][w] = p

    # Collect histogram files
    if hi_dir.exists():
        for p in hi_dir.glob("*_hist_*_w_*.png"):
            w = parse_weight_from_filename(p)
            if w is None:
                continue
            # Identify which quantity
            if "_hist_T_" in p.name:
                images['hist_T'][w] = p
            elif "_hist_v_" in p.name:
                images['hist_v'][w] = p
            elif "_hist_B_" in p.name or "_hist_blos_" in p.name:
                images['hist_B'][w] = p

    # Collect scatter files
    if sc_dir.exists():
        for p in sc_dir.glob("*_scatter_*_w_*.png"):
            w = parse_weight_from_filename(p)
            if w is None:
                continue
            # Identify which quantity
            # file names include e.g. ..._scatter_v_... or ..._scatter_B_... or ..._scatter_T_...
            fname = p.name
            if "_scatter_v_" in fname:
                images['scat_v'][w] = p
            elif "_scatter_B_" in fname or "_scatter_blos_" in fname:
                images['scat_B'][w] = p
            else:
                # Fallback: if contains '_scatter_T_' or lacks clear marker assume T
                images['scat_T'][w] = p

    # Union of weights
    weight_set = set()
    for d in images.values():
        weight_set.update(d.keys())
    weights = sorted(weight_set)
    return weights, images


def draw_cell_image(ax, path: Optional[Path]):
    ax.axis('off')
    if path and path.exists():
        try:
            img = plt.imread(str(path))
            ax.imshow(img)
            return
        except Exception:
            pass
    # Fallback placeholder
    ax.set_facecolor('#f0f0f0')
    ax.text(0.5, 0.5, 'missing', ha='center', va='center', fontsize=9, color='gray', transform=ax.transAxes)


def build_poster_for_tau(tau_dir_name: str, tau_label: str, out_dir: Path, dry_run: bool = False):
    """Build a poster figure for a single tau across both conditions."""
    weights_by_cond: Dict[str, List[float]] = {}
    images_by_cond: Dict[str, Dict[str, Dict[float, Path]]] = {}

    for cond in CONDITIONS:
        weights, images = discover_images_for_tau(tau_dir_name, cond)
        weights_by_cond[cond] = weights
        images_by_cond[cond] = images

    # Global columns = union of weights across both conditions
    all_weights = sorted(set().union(*weights_by_cond.values()))
    if not all_weights:
        print(f"[skip] No images found for {tau_dir_name}")
        return

    # Layout: per condition we use 7 rows: 1 imshow + 3 histogram (T,v,B) + 3 scatter (T,v,B)
    rows_per_cond = 7
    nrows = rows_per_cond * len(CONDITIONS)
    ncols = len(all_weights)

    if dry_run:
        print(f"Would create poster for {tau_label} with {ncols} columns (weights: {all_weights}) and {nrows} rows")
        return

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.0 * ncols, 2.2 * nrows))
    if nrows == 1:
        axes = np.array([axes])  # normalize shape

    # Helper to compute row index for each section
    def row_index(cond_idx: int, section: str, q_idx: Optional[int] = None) -> int:
        base = cond_idx * rows_per_cond
        if section == 'imshow':
            return base + 0
        if section == 'hist':
            return base + 1 + (q_idx or 0)
        if section == 'scat':
            return base + 4 + (q_idx or 0)
        raise ValueError('invalid section')

    # Map order for quantities
    q_keys = ['T', 'v', 'B']

    # Titles for left-most cells
    left_titles = {
        'imshow': 'Imshow comparisons',
        'hist_T': 'Histograms (T)',
        'hist_v': 'Histograms (v)',
        'hist_B': 'Histograms (B_LOS)',
        'scat_T': 'Scatter (T)',
        'scat_v': 'Scatter (v)',
        'scat_B': 'Scatter (B_LOS)',
    }

    # Fill figure
    for cond_idx, cond in enumerate(CONDITIONS):
        images = images_by_cond[cond]
        for col_idx, w in enumerate(all_weights):
            # Top row (per condition): imshow
            ax = axes[row_index(cond_idx, 'imshow'), col_idx]
            draw_cell_image(ax, images['imshow'].get(w))
            if cond_idx == 0:
                ax.set_title(f"w={w:g}", fontsize=10)

            # Hist rows per quantity
            for qi, q in enumerate(q_keys):
                ax_h = axes[row_index(cond_idx, 'hist', qi), col_idx]
                draw_cell_image(ax_h, images[f'hist_{q}'].get(w))
            # Scatter rows per quantity
            for qi, q in enumerate(q_keys):
                ax_s = axes[row_index(cond_idx, 'scat', qi), col_idx]
                draw_cell_image(ax_s, images[f'scat_{q}'].get(w))

        # Add left-side row labels for this condition
        first_col = 0
        axes[row_index(cond_idx, 'imshow'), first_col].text(-0.02, 0.5, f"{cond.replace('_', ' ').title()}",
                                                            va='center', ha='right', rotation=90,
                                                            transform=axes[row_index(cond_idx, 'imshow'), first_col].transAxes,
                                                            fontsize=11, fontweight='bold')
        # Section labels
        axes[row_index(cond_idx, 'imshow'), first_col].text(-0.02, 1.05, left_titles['imshow'], va='bottom', ha='right',
                                                            transform=axes[row_index(cond_idx, 'imshow'), first_col].transAxes,
                                                            fontsize=10)
        for qi, q in enumerate(q_keys):
            axes[row_index(cond_idx, 'hist', qi), first_col].text(-0.02, 0.5, left_titles[f'hist_{q}'], va='center', ha='right',
                                                                  transform=axes[row_index(cond_idx, 'hist', qi), first_col].transAxes,
                                                                  fontsize=9)
        for qi, q in enumerate(q_keys):
            axes[row_index(cond_idx, 'scat', qi), first_col].text(-0.02, 0.5, left_titles[f'scat_{q}'], va='center', ha='right',
                                                                  transform=axes[row_index(cond_idx, 'scat', qi), first_col].transAxes,
                                                                  fontsize=9)

    # Global title
    fig.suptitle(f"MuRAM Results – {tau_label}", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0.04, 0.02, 0.995, 0.98])

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{tau_dir_name}_MURaM.svg"
    png_path = out_dir / f"{tau_dir_name}_MURaM.png"
    pdf_path = out_dir / f"{tau_dir_name}_MURaM.pdf"
    fig.savefig(svg_path, dpi=200)
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path, dpi=200)
    plt.close(fig)
    print(f"Saved {svg_path}, {png_path}, and {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Create MURaM per-tau poster compendiums")
    parser.add_argument('--tau', type=str, default=None,
                        help="Specific tau value to process, e.g., 0.00 or -1.50; matches folder name 'tau_<val>'")
    parser.add_argument('--dry-run', action='store_true', help="List planned posters without creating them")
    args = parser.parse_args()

    # Determine tau directories
    tau_dirs = list_tau_dirs()
    if args.tau is not None:
        tau_dir_name = f"tau_{args.tau}"
        tau_dirs = [t for t in tau_dirs if t == tau_dir_name]
        if not tau_dirs:
            print(f"No tau folder found for {tau_dir_name}")
            return

    # Ensure output base exists
    OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Build posters
    for tname in tau_dirs:
        # Label for title
        tau_val = tname.split('tau_')[-1]
        tau_label = f"log τ = {tau_val}"
        out_dir = OUT_BASE_DIR / tname
        build_poster_for_tau(tname, tau_label, out_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
