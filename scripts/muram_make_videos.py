import os
import sys
import io
import warnings
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

plt.rcParams.update({
  'axes.titlesize': 'x-large',  # heading 1
  'axes.labelsize': 'large',   # heading 2
  'xtick.labelsize': 'medium',         # fontsize of the ticks
  'ytick.labelsize': 'medium',         # fontsize of the ticks
  'font.family': 'serif',        # Font family
  'text.usetex': False,          # Do not use LaTeX for text rendering
  'figure.figsize': (10, 8),     # Default figure size
  'savefig.dpi': 300,            # High resolution for saving figures
  'savefig.format': 'png',       # Default format for saving figures
  'legend.fontsize': 'small',  # Font size for legends
  'lines.linewidth': 2,          # Line width for plots
  'lines.markersize': 8,         # Marker size for plots,
  'axes.formatter.useoffset': False,  # Disable offset
  'axes.formatter.use_mathtext': True,  # Use scientific notation
  'axes.formatter.limits': (-3, 4),  # Use scientific notation for values over 10^2
  'axes.labelsize': 'large',     # Font size for axes labels
  'figure.titlesize': 'xx-large', # Font size for suptitles (heading 1)
  'axes.formatter.use_locale': False  # Do not use locale settings
})

# Ensure utils are importable
ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from muram_data import MhdData  # noqa: E402


def figure_to_image(fig, dpi=100):
    """Convert matplotlib figure to a PIL Image without writing to disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    buf.close()
    return img.copy()


class VideoWriter:
    """Collects frames and saves a video with fallbacks.

    Saves MP4 using matplotlib's ffmpeg writer if available; otherwise
    falls back to saving individual PNG frames.
    """

    def __init__(self, output_path: Path, fps: int = 10, dpi: int = 100):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.dpi = dpi
        self.frames: List[Image.Image] = []

    def add_figure(self, fig):
        try:
            img = figure_to_image(fig, dpi=self.dpi)
            self.frames.append(img)
        except Exception as e:
            print(f"    Warning: Failed to convert frame: {e}")
        finally:
            plt.close(fig)

    def save(self) -> bool:
        if not self.frames:
            print(f"  ERROR: No frames to save for {self.output_path}")
            return False

        print(f"  Saving {len(self.frames)} frames to {self.output_path}...")

        # Try matplotlib's ffmpeg writer
        try:
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=self.fps, metadata=dict(artist="MURaM"), bitrate=1800)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.axis("off")
            with writer.saving(fig, str(self.output_path), dpi=self.dpi):
                for img in self.frames:
                    ax.imshow(img)
                    ax.axis("off")
                    writer.grab_frame()
                    ax.clear()
            plt.close(fig)
            print(f"  ✓ Saved video (ffmpeg): {self.output_path}")
            return True
        except Exception as e:
            print(f"  ERROR: ffmpeg writer failed ({e}). Saving PNG frames instead...")
            for i, img in enumerate(self.frames):
                frame_path = self.output_path.parent / f"{self.output_path.stem}_frame_{i:03d}.png"
                try:
                    img.save(frame_path)
                except Exception as e2:
                    print(f"    Failed to save frame {i}: {e2}")
            print(f"  Saved {len(self.frames)} PNG frames to {self.output_path.parent}")
            return False


def find_available_steps(data_path: Path, min_step: int, max_step: int) -> List[int]:
    """Find steps that have required MURaM files present."""
    required = ["eos", "result_0", "result_2", "result_6"]
    available = []
    for i in range(min_step, max_step + 1):
        suffix = f".{i:03d}000"
        missing = [name for name in required if not (data_path / f"{name}{suffix}").exists()]
        if not missing:
            available.append(i)
    return available


def render_xz_slice(
    arr3d: np.ndarray,
    y_idx: int,
    quantity: str,
    height_levels: np.ndarray | None = None,
    step: int | None = None,
    logtau: np.ndarray | None = None,
):
    """Render an x–z slice (arr[:, y_idx, :]) as a matplotlib figure.

    Parameters
    ----------
    arr3d : np.ndarray
        Data array of shape (nx, ny, nz)
    y_idx : int
        Fixed y index for the slice
    quantity : str
        Quantity name for labeling
    height_levels : np.ndarray | None
        Optional vector of height labels in km (length nz)
    step : int | None
        Optional step number for labeling
    """
    slice_q = arr3d[:, y_idx, :]  # (nx, nz)

    # Colormap and limits
    cmap_map = {"T": "hot", "Vz": "bwr_r", "Bz": "PiYG"}
    cmap_name = cmap_map.get(quantity, "viridis")
    cmap = plt.get_cmap(cmap_name) if cmap_name in plt.colormaps() else plt.get_cmap("viridis")

    # Symmetric limits for signed fields
    if quantity in ("Vz", "Bz"):
        vmin, vmax = np.quantile(slice_q, [0.01, 0.99])
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
    else:
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(8, 6))
    # Transpose so x runs horizontally and z vertically
    im = ax.imshow(slice_q.T, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("x index")
    title = f"{quantity} x–z slice at y={y_idx}"
    if step is not None:
        title += f" (step {step})"
    ax.set_title(title)

    # Height labels
    # After transpose, vertical axis corresponds to z (length nz)
    nz = slice_q.shape[1]
    n_ticks = min(8, nz)
    tick_positions = np.linspace(0, nz - 1, n_ticks)
    if height_levels is not None and len(height_levels) == nz:
        tick_labels = [f"{int(height_levels[int(p)])}" for p in tick_positions]
        ax.set_ylabel("Height (km)")
    else:
        tick_labels = [f"{int(p)}" for p in tick_positions]
        ax.set_ylabel("z index")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    # Overlay tau=1 (log10 tau = 0) contour if provided
    try:
        if logtau is not None and logtau.shape == arr3d.shape:
            tau_slice = logtau[:, y_idx, :]  # (nx, nz)
            # Transpose to match the image orientation (x horizontal, z vertical)
            CS = ax.contour(tau_slice.T, levels=[0.0], colors="white", linewidths=1.2)
            # Optional label near the longest segment
            if CS.allsegs and CS.allsegs[0]:
                longest = max(CS.allsegs[0], key=lambda s: s.shape[0])
                mid = longest[len(longest) // 2]
                x_lab, y_lab = mid[0], mid[1]
                ax.text(
                    x_lab + 1,
                    y_lab + 2,
                    r"$\tau = 1$",
                    color="white",
                    fontsize=10,
                    ha="left",
                    va="bottom",
                    bbox=dict(facecolor="black", alpha=0.4, boxstyle="round,pad=0.2"),
                )
        elif logtau is not None:
            print("  Warning: logtau shape does not match data shape; skipping tau contour.")
    except Exception as e:
        print(f"  Warning: failed to draw tau contour: {e}")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def generate_all_videos(
    mhd: MhdData,
    steps: List[int],
    output_dir: Path,
    data_path: Path,
    y_idx: int = 100,
    quantities: List[str] = ["T", "Vz", "Bz"],
    tau_levels: List[float] = [0.0, -0.8, -2.0],
    z_max: int | None = 250,
    fps: int = 10,
    dpi: int = 100,
    kappa_path: Path | None = None,
):
    """Generate both slice and OD surface videos in a single pass per step."""
    print("\n=== Generating Videos (Slices and OD Surfaces) ===")
    
    # Setup output directories
    slices_dir = output_dir / "slices_xz"
    slices_dir.mkdir(parents=True, exist_ok=True)
    od_dir = output_dir / "surfaces_od"
    od_dir.mkdir(parents=True, exist_ok=True)

    # OD grid setup
    new_logtau = np.arange(-2.0, 0.1, 0.1)
    tau_to_idx = {tau: int(round((tau - new_logtau[0]) / 0.1)) for tau in tau_levels}

    # Initialize writers for slices
    slice_writers: Dict[str, VideoWriter] = {}
    for q in quantities:
        slice_writers[q] = VideoWriter(slices_dir / f"{q}_xz_y{y_idx}.mp4", fps=fps, dpi=dpi)

    # Initialize writers for OD surfaces
    od_writers: Dict[str, Dict[float, VideoWriter]] = {}
    for q in quantities:
        od_writers[q] = {}
        for tau in tau_levels:
            od_writers[q][tau] = VideoWriter(od_dir / f"{q}_surface_logtau_{tau:+.1f}.mp4", fps=fps, dpi=dpi)

    # OD surface rendering limits
    limit_values = {
        "T": (5500, 7000),
        "Vz": (-5, 5),
        "Bz": (-100, 100),
    }

    # Load kappa once if provided
    if kappa_path is None:
        kappa_path = data_path / "csv" / "kappa.0.dat"
    
    print(f"Using kappa path: {kappa_path}")

    # Process each step once
    for step in steps:
        print(f"Processing step {step}...")
        
        # Load step data
        try:
            mhd.load_step(step, z_max=z_max)
        except Exception as e:
            print(f"  Error loading step {step}: {e}")
            continue

        # Prepare optical depth once (needed for tau=1 contour and OD surfaces)
        try:
            mhd.load_opacity_table(str(kappa_path))
            mhd.compute_optical_depth()
        except Exception as e:
            print(f"  Error computing optical depth for step {step}: {e}")
            # Continue with slices without tau contour

        # Generate slice videos from geometric data (overlay tau contour if available)
        for q in quantities:
            if q not in mhd.data:
                print(f"  Skipping {q}: not in data")
                continue

            try:
                arr_q = mhd.data[q].value if hasattr(mhd.data[q], "value") else mhd.data[q]
                logtau = mhd.logtau if hasattr(mhd, "logtau") else None
                fig = render_xz_slice(
                    arr_q,
                    y_idx=y_idx,
                    quantity=q,
                    height_levels=mhd.height_levels,
                    step=step,
                    logtau=logtau,
                )
                slice_writers[q].add_figure(fig)
            except Exception as e:
                print(f"  Error generating slice for {q}: {e}")

        # Generate OD surface videos from OD-remapped data
        try:
            mhd.remap_to_optical_depth(new_logtau, quantities=quantities)

            for tau in tau_levels:
                z_idx = tau_to_idx[tau]
                try:
                    figs = mhd.plot_surfaces(
                        z_idx=z_idx,
                        data_source="optical_depth",
                        zero_height=False,
                        quantities=quantities,
                        limit_values=limit_values,
                        return_figs=True,
                        second_plot="histogram"
                    )
                    for q, fig in figs:
                        if q in od_writers and tau in od_writers[q]:
                            # Add step number to title
                            if fig.axes:
                                current_title = fig.axes[0].get_title()
                                fig.axes[0].set_title(f"{current_title} (step {step})")
                            od_writers[q][tau].add_figure(fig)
                except Exception as e:
                    print(f"  Error plotting OD surfaces at tau={tau:+.1f}: {e}")

        except Exception as e:
            print(f"  Error preparing OD for step {step}: {e}")

    # Save all videos
    print("\nSaving slice videos...")
    for q, w in slice_writers.items():
        w.save()

    print("\nSaving OD surface videos...")
    for q in quantities:
        for tau in tau_levels:
            od_writers[q][tau].save()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate MURaM videos using MhdData")
    parser.add_argument("--data-path", type=str, default="/scratchsan/observatorio/juagudeloo/data/",
                        help="Base path to data directory (contains muram-simulation/ and csv/ subdirectories)")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "images" / "muram_videos"),
                        help="Directory to save videos")
    parser.add_argument("--min-step", type=int, default=60, help="Minimum step number to process")
    parser.add_argument("--max-step", type=int, default=200, help="Maximum step number to process")
    parser.add_argument("--y-idx", type=int, default=100, help="y index for x–z slice videos")
    parser.add_argument("--nx", type=int, default=480, help="Grid size X")
    parser.add_argument("--ny", type=int, default=480, help="Grid size Y")
    parser.add_argument("--nz", type=int, default=256, help="Grid size Z")
    parser.add_argument("--z-max", type=int, default=250, help="Optional max z layer to keep")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for videos")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for rendered frames")
    parser.add_argument("--kappa", type=str, default=None,
                        help="Path to opacity table (kappa.0.dat). If not provided, uses data_path/csv/kappa.0.dat")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    muram_sim_path = data_path / "muram-simulation"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MURaM Video Generation (MhdData)")
    print("=" * 60)
    print(f"Base data path: {data_path}")
    print(f"MURaM simulation path: {muram_sim_path}")
    print(f"Output dir: {output_dir}")
    print(f"Steps: {args.min_step}..{args.max_step}")
    print(f"Slice y index: {args.y_idx}")
    print(f"DPI: {args.dpi}, FPS: {args.fps}\n")

    steps = find_available_steps(muram_sim_path, args.min_step, args.max_step)
    print(f"Found {len(steps)} available steps")
    if not steps:
        print("No steps with required files found. Exiting.")
        return

    mhd = MhdData(data_path=str(muram_sim_path), nx=args.nx, ny=args.ny, nz=args.nz)

    # Generate both slice and OD surface videos in a single pass per step
    generate_all_videos(
        mhd,
        steps=steps,
        output_dir=output_dir,
        data_path=data_path,
        y_idx=args.y_idx,
        tau_levels=[0.0, -0.8, -2.0],
        z_max=args.z_max,
        fps=args.fps,
        dpi=args.dpi,
        kappa_path=Path(args.kappa) if args.kappa else None,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
