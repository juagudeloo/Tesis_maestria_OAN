#!/usr/bin/env python3
"""
Compose per-tau MODEST poster images by assembling existing imshow, histogram,
and scatter PNGs for each trained weight w and for both noise conditions.

Inputs (discovered automatically):
- images/paper_experiment/Hinode_MODEST/
    - modest_nn_predictions_with_noise/{histograms,imshow,scatter}/tau_*/
    - modest_nn_predictions_without_noise_smoothed/{histograms,imshow,scatter}/tau_*/

Outputs:
- images/Compends/MODEST/tau_<val>/tau_<val>_MODEST.svg
- images/Compends/MODEST/tau_<val>/tau_<val>_MODEST.png

Usage:
    python make_modest_posters.py               # build posters for all tau folders found
    python make_modest_posters.py --tau 0.0     # only for a specific tau value
    python make_modest_posters.py --dry-run     # just print what would be done
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------- Configuration ----------
BASE_MODEST_DIR = Path("images/paper_experiment/Hinode_MODEST")
PRED_DIRS = [
    ("with_noise", BASE_MODEST_DIR / "modest_nn_predictions_with_noise"),
    ("without_noise_smoothed", BASE_MODEST_DIR / "modest_nn_predictions_without_noise_smoothed"),
]
OUT_BASE_DIR = Path("images/Compends/MODEST")
QUANTITIES = ["Temperature", "Velocity", "B_LOS"]


def list_tau_dirs() -> List[str]:
    """List tau directory names (e.g., 'tau_-0.8', 'tau_0.0') present under any of the MODEST prediction folders.
    Sorted numerically by tau value when possible.
    """
    tau_set = set()
    for _, pred_base in PRED_DIRS:
        for category in ["histograms", "imshow", "scatter"]:
            root = pred_base / category
            if not root.exists():
                continue
            for child in root.iterdir():
                if child.is_dir() and child.name.startswith("tau_"):
                    tau_set.add(child.name)
    def tau_key(name: str) -> float:
        try:
            return float(name.split("tau_")[-1])
        except Exception:
            return 0.0
    return sorted(tau_set, key=tau_key)


def parse_weight_from_filename(p: Path) -> Optional[float]:
    """Extract numeric weight value from filename by capturing the last 'w_<num>.png'.
    Examples:
      'Temperature_imshow_with_noise_w_0.0001.png' -> 0.0001
      'B_LOS_hist_without_noise_smoothed_w_0.9.png' -> 0.9
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
        keys: 'imshow_T', 'imshow_Velocity', 'imshow_B',
              'hist_T', 'hist_Velocity', 'hist_B',
              'scat_T', 'scat_Velocity', 'scat_B'
    """
    # Map human-friendly keys
    images: Dict[str, Dict[float, Path]] = {
        'imshow_Temperature': {}, 'imshow_Velocity': {}, 'imshow_B_LOS': {},
        'hist_Temperature': {}, 'hist_Velocity': {}, 'hist_B_LOS': {},
        'scat_Temperature': {}, 'scat_Velocity': {}, 'scat_B_LOS': {},
    }

    # Choose base dir for this condition
    pred_base = None
    for cond_name, base in PRED_DIRS:
        if cond_name == condition:
            pred_base = base
            break
    if pred_base is None:
        return [], images

    for category in ["imshow", "histograms", "scatter"]:
        cat_dir = pred_base / category / tau_dir_name
        if not cat_dir.exists():
            continue
        for p in cat_dir.glob("*.png"):
            w = parse_weight_from_filename(p)
            if w is None:
                continue
            name = p.name
            # Identify quantity and category by filename prefix
            if name.startswith("Temperature_"):
                q = "Temperature"
            elif name.startswith("Velocity_"):
                q = "Velocity"
            elif name.startswith("B_LOS_"):
                q = "B_LOS"
            else:
                # Unknown; skip to avoid misplacement
                continue
            if category == "imshow":
                images[f"imshow_{q}"][w] = p
            elif category == "histograms":
                images[f"hist_{q}"][w] = p
            else:
                images[f"scat_{q}"][w] = p

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


def condition_label(cond: str) -> str:
    return {
        'with_noise': 'with noise',
        'without_noise_smoothed': 'without noise (smoothed)'
    }.get(cond, cond)


def build_poster_for_tau(tau_dir_name: str, tau_label: str, out_dir: Path, dry_run: bool = False):
    weights_by_cond: Dict[str, List[float]] = {}
    images_by_cond: Dict[str, Dict[str, Dict[float, Path]]] = {}

    for cond, _ in PRED_DIRS:
        weights, images = discover_images_for_tau(tau_dir_name, cond)
        weights_by_cond[cond] = weights
        images_by_cond[cond] = images

    all_weights = sorted(set().union(*weights_by_cond.values()))
    if not all_weights:
        print(f"[skip] No MODEST images found for {tau_dir_name}")
        return

    # Layout per condition: 9 rows (3 imshow + 3 hist + 3 scatter)
    rows_per_cond = 9
    nrows = rows_per_cond * len(PRED_DIRS)
    ncols = len(all_weights)

    if dry_run:
        print(f"Would create MODEST poster for {tau_label} with {ncols} columns (weights: {all_weights}) and {nrows} rows")
        return

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.0 * ncols, 2.0 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    # Helper to compute row index for each section and quantity
    def row_index(cond_idx: int, section: str, q_idx: int) -> int:
        base = cond_idx * rows_per_cond
        if section == 'imshow':
            return base + q_idx  # 0..2
        if section == 'hist':
            return base + 3 + q_idx  # 3..5
        if section == 'scat':
            return base + 6 + q_idx  # 6..8
        raise ValueError('invalid section')

    # Titles for left-most cells
    left_titles = {
        'imshow_Temperature': 'Imshow (Temperature)',
        'imshow_Velocity': 'Imshow (Velocity)',
        'imshow_B_LOS': 'Imshow (B_LOS)',
        'hist_Temperature': 'Histograms (Temperature)',
        'hist_Velocity': 'Histograms (Velocity)',
        'hist_B_LOS': 'Histograms (B_LOS)',
        'scat_Temperature': 'Scatter (Temperature)',
        'scat_Velocity': 'Scatter (Velocity)',
        'scat_B_LOS': 'Scatter (B_LOS)',
    }

    for cond_idx, (cond, _) in enumerate(PRED_DIRS):
        images = images_by_cond[cond]
        for col_idx, w in enumerate(all_weights):
            # 3 imshow rows
            for qi, q in enumerate(QUANTITIES):
                ax = axes[row_index(cond_idx, 'imshow', qi), col_idx]
                draw_cell_image(ax, images[f'imshow_{q}'].get(w))
                if cond_idx == 0:
                    ax.set_title(f"w={w:g}", fontsize=10)
            # 3 hist rows
            for qi, q in enumerate(QUANTITIES):
                axh = axes[row_index(cond_idx, 'hist', qi), col_idx]
                draw_cell_image(axh, images[f'hist_{q}'].get(w))
            # 3 scatter rows
            for qi, q in enumerate(QUANTITIES):
                axs = axes[row_index(cond_idx, 'scat', qi), col_idx]
                draw_cell_image(axs, images[f'scat_{q}'].get(w))

        # Condition label at the top of this condition block (left-most column)
        first_col = 0
        axes[row_index(cond_idx, 'imshow', 0), first_col].text(
            -0.02, 0.5, condition_label(cond), va='center', ha='right', rotation=90,
            transform=axes[row_index(cond_idx, 'imshow', 0), first_col].transAxes,
            fontsize=11, fontweight='bold')

        # Section labels at left of each row in the block
        for qi, q in enumerate(QUANTITIES):
            axes[row_index(cond_idx, 'imshow', qi), first_col].text(
                -0.02, 1.05, left_titles[f'imshow_{q}'], va='bottom', ha='right',
                transform=axes[row_index(cond_idx, 'imshow', qi), first_col].transAxes,
                fontsize=9)
        for qi, q in enumerate(QUANTITIES):
            axes[row_index(cond_idx, 'hist', qi), first_col].text(
                -0.02, 0.5, left_titles[f'hist_{q}'], va='center', ha='right',
                transform=axes[row_index(cond_idx, 'hist', qi), first_col].transAxes,
                fontsize=9)
        for qi, q in enumerate(QUANTITIES):
            axes[row_index(cond_idx, 'scat', qi), first_col].text(
                -0.02, 0.5, left_titles[f'scat_{q}'], va='center', ha='right',
                transform=axes[row_index(cond_idx, 'scat', qi), first_col].transAxes,
                fontsize=9)

    fig.suptitle(f"MODEST Predictions – {tau_label}", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0.05, 0.02, 0.995, 0.98])

    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{tau_dir_name}_MODEST.svg"
    png_path = out_dir / f"{tau_dir_name}_MODEST.png"
    pdf_path = out_dir / f"{tau_dir_name}_MODEST.pdf"
    fig.savefig(svg_path, dpi=200)
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path, dpi=200)
    plt.close(fig)
    print(f"Saved {svg_path}, {png_path}, and {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Create MODEST per-tau poster compendiums")
    parser.add_argument('--tau', type=str, default=None,
                        help="Specific tau value to process, e.g., 0.0 or -0.8; matches folder name 'tau_<val>'")
    parser.add_argument('--dry-run', action='store_true', help="List planned posters without creating them")
    args = parser.parse_args()

    tau_dirs = list_tau_dirs()
    if args.tau is not None:
        tau_dir_name = f"tau_{args.tau}"
        tau_dirs = [t for t in tau_dirs if t == tau_dir_name]
        if not tau_dirs:
            print(f"No tau folder found for {tau_dir_name}")
            return

    OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    for tname in tau_dirs:
        tau_val = tname.split('tau_')[-1]
        tau_label = f"log τ = {tau_val}"
        out_dir = OUT_BASE_DIR / tname
        build_poster_for_tau(tname, tau_label, out_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
