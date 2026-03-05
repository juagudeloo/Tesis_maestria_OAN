import os
import sys
sys.path.append("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/")
import argparse
from pathlib import Path

import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from utils.modest_data import ModestData
from utils.normalizer import MhdNormalizer, StokesNormalizer
from utils.analysis import AnalysisModelPipeline, DiagnosticPlots
from scripts.base_training import TrainingConfig

PLAGE_CROP_BOUNDS = (0,100,400, 600)  # X_MIN, X_MAX, Y_MIN, Y_MAX

class ModestDiagnosticPlots:
    def __init__(
        self,
        pipeline: AnalysisModelPipeline,
        modest_output_dir: Path,
        mhd_normalizer: MhdNormalizer,
        stokes_normalizer: StokesNormalizer,
        modest: ModestData,
        args,
    ):
        self.pipeline = pipeline
        self.modest_output_dir = modest_output_dir
        self.mhd_normalizer = mhd_normalizer
        self.stokes_normalizer = stokes_normalizer
        self.modest = modest
        self.args = args

        self.modest_data = None
        self.modest_logtau = None
        self.modest_mhd_data = None
        self.modest_stokes_input = None
        self.pred_nx = None
        self.pred_ny = None
        self.n_tau_eff = None
        self.tau_indices = None

    def _resolve_tau_indices(self, indices: list[int] | None, n_tau: int) -> list[int]:
        if not indices:
            return sorted(set([0, n_tau // 2, n_tau - 1]))
        out = []
        for idx in indices:
            ridx = idx if idx >= 0 else n_tau + idx
            if 0 <= ridx < n_tau:
                out.append(ridx)
        return sorted(set(out))

    def prepare_snapshot(self, n_tau: int):
        self.modest_data = self.modest.load_all(
            region_bounds=tuple(self.args.crop_bounds) if self.args.cropped_region else None,
            apply_mask=self.args.polarization_mask,
        )

        self.modest_logtau = list(
            self.modest_data.get("tau_values", sorted(self.modest_data["spinor_atm"]["T"].keys()))
        )
        self.n_tau_eff = min(n_tau, len(self.modest_logtau))

        self.modest_mhd_data = {
            "T": np.stack(
                [self.modest_data["spinor_atm"]["T"][t] for t in self.modest_logtau[:self.n_tau_eff]], axis=-1
            ).astype(np.float32),
            "Vz": np.stack(
                [self.modest_data["spinor_atm"]["Vlos"][t] for t in self.modest_logtau[:self.n_tau_eff]], axis=-1
            ).astype(np.float32),
            "Bz": np.stack(
                [self.modest_data["spinor_atm"]["Blos"][t] for t in self.modest_logtau[:self.n_tau_eff]], axis=-1
            ).astype(np.float32),
        }

        self.pred_nx, self.pred_ny = self.modest_data["smoothed_stokes"]["I"].shape[:2]
        norm_stokes = self.stokes_normalizer.transform(self.modest_data["smoothed_stokes"])
        I_flat = norm_stokes["I"].reshape(self.pred_nx * self.pred_ny, -1)
        V_flat = norm_stokes["V"].reshape(self.pred_nx * self.pred_ny, -1)
        self.modest_stokes_input = np.stack([I_flat, V_flat], axis=1).astype(np.float32)

        self.tau_indices = self._resolve_tau_indices(self.args.tau_indices, self.n_tau_eff)
        print(f"\nMODEST optical depth nodes: {[float(v) for v in self.modest_logtau]}")

    def _filter_matches(self, matches):
        if not self.tau_indices:
            return matches
        allowed = set(self.tau_indices)
        return [m for m in matches if m[1] in allowed]

    def _plot_imshows(
        self,
        true_2d: np.ndarray,
        pred_2d: np.ndarray,
        title: str,
        save_path: Path,
        param: str,
        transpose: bool = True,
        transpose_pred: bool | None = None,
    ):
        gt = true_2d.T if transpose else true_2d
        pred_do_transpose = transpose if transpose_pred is None else transpose_pred
        pr = pred_2d.T if pred_do_transpose else pred_2d

        param_cmaps = {"T": "hot", "Vz": "bwr_r", "Bz": "PiYG"}
        cmap = param_cmaps.get(param, "viridis")

        vals = (
            np.concatenate([gt[np.isfinite(gt)], pr[np.isfinite(pr)]])
            if (np.isfinite(gt).any() and np.isfinite(pr).any())
            else np.array([0.0, 1.0])
        )
        vmin, vmax = np.quantile(vals, [0.01, 0.99])
        if np.nanmin(vals) < 0 < np.nanmax(vals):
            vmax_abs = max(abs(vmin), abs(vmax))
            vmin, vmax = -vmax_abs, vmax_abs

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        im0 = axes[0].imshow(gt, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Ground truth ({gt.shape[0]}x{gt.shape[1]})")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(pr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Prediction ({pr.shape[0]}x{pr.shape[1]})")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        if gt.shape == pr.shape:
            er = pr - gt
            emax = np.quantile(np.abs(er[np.isfinite(er)]), 0.99) if np.isfinite(er).any() else 1.0
            im2 = axes[2].imshow(er, origin="lower", cmap="RdBu_r", vmin=-emax, vmax=emax)
            axes[2].set_title("Error (Pred-GT)")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        else:
            axes[2].text(
                0.5, 0.5,
                f"Error skipped\nshape mismatch\nGT={gt.shape}\nPred={pr.shape}",
                ha="center", va="center", transform=axes[2].transAxes
            )
            axes[2].set_axis_off()

        fig.suptitle(title)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_jointplot(
        self,
        true_2d: np.ndarray,
        pred_2d: np.ndarray,
        title: str,
        save_path: Path,
        max_points: int = 50000,
    ):
        x = true_2d[np.isfinite(true_2d)].ravel()
        y = pred_2d[np.isfinite(pred_2d)].ravel()
        if x.size == 0 or y.size == 0:
            return

        n = min(max_points, x.size, y.size)
        q = np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 / n
        xq = np.quantile(x, q)
        yq = np.quantile(y, q)

        lo = float(min(np.min(xq), np.min(yq)))
        hi = float(max(np.max(xq), np.max(yq)))

        g = sns.jointplot(x=xq, y=yq, kind="scatter", s=8, alpha=0.25, height=6)
        g.ax_joint.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        g.ax_joint.set_xlim(lo, hi)
        g.ax_joint.set_ylim(lo, hi)
        g.ax_joint.set_xlabel("Ground truth (quantiles)")
        g.ax_joint.set_ylabel("Prediction (quantiles)")
        g.fig.suptitle(title, y=1.02)
        g.fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(g.fig)

    def run(self, model_configs, models):
        for name, model in models.items():
            model_type = model_configs[name]["experiment_key"]

            pred_tau = self.pipeline.get_model_logtau_values(model_configs[name])
            if pred_tau is None:
                print(f"[{model_type}] Predicted optical depth nodes: not found in experiment_results/config.")
                continue

            matches = self.pipeline.common_tau_matches([float(v) for v in self.modest_logtau], pred_tau)
            matches = self._filter_matches(matches)

            print(f"[{model_type}] Predicted optical depth nodes: {pred_tau}")
            print(f"[{model_type}] Common optical depth values (MODEST ∩ predicted): {[m[0] for m in matches]}")

            if not matches:
                print(f"[{model_type}] No common optical depths. Skipping plots.")
                continue

            pred_mhd = self.pipeline.predict_and_denormalize(
                model=model,
                stokes_input=self.modest_stokes_input,
                mhd_normalizer=self.mhd_normalizer,
                pred_nx=self.pred_nx,
                pred_ny=self.pred_ny,
                batch_size=self.args.inference_batch_size,
            )

            out_root = self.modest_output_dir / model_type
            surface_dir = out_root / "surface"
            joint_dir = out_root / "jointplots"
            surface_dir.mkdir(parents=True, exist_ok=True)
            joint_dir.mkdir(parents=True, exist_ok=True)
            saved_any = False

            for param in ("T", "Vz", "Bz"):
                pred_cube = pred_mhd[param]
                true_cube = self.modest_mhd_data[param]

                for tau_val, i_mod, i_pred in matches:
                    if i_mod >= true_cube.shape[2] or i_pred >= pred_cube.shape[2]:
                        print(f"[{model_type}] Skip {param} @ tau={tau_val}: index out of bounds (mod={i_mod}, pred={i_pred}).")
                        continue

                    true_map = true_cube[:, :, i_mod]
                    pred_map = pred_cube[:, :, i_pred]

                    self._plot_imshows(
                        true_2d=true_map,
                        pred_2d=pred_map,
                        title=f"{model_type} | {param} | matched log(tau)={tau_val:.2f}",
                        save_path=surface_dir / f"{param}_tau_{tau_val:+.2f}_imshow.png",
                        param=param,
                        transpose=True,  # keep GT behavior
                        transpose_pred=not self.args.cropped_region,  # cropped: do NOT transpose generated data
                    )
                    self._plot_jointplot(
                        true_2d=true_map,
                        pred_2d=pred_map,
                        title=f"{model_type} | {param} | matched log(tau)={tau_val:.2f}",
                        save_path=joint_dir / f"{param}_tau_{tau_val:+.2f}_jointplot.png",
                    )
                    saved_any = True

            if saved_any:
                print(f"[{model_type}] Surface images saved to: {surface_dir}")
                print(f"[{model_type}] Jointplots saved to: {joint_dir}")
            else:
                print(f"[{model_type}] No images were saved.")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modest_base_dir = Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/images/analysis/modest")
    if args.cropped_region:
        crop_label = args.crop_label.strip()
        if not crop_label:
            raise ValueError("--crop-label must be a non-empty string when --cropped-region is set.")
        modest_output_dir = modest_base_dir / "cropped" / crop_label
    else:
        modest_output_dir = modest_base_dir / "whole"

    pipeline = AnalysisModelPipeline(
        device=device,
        output_dir=modest_output_dir,
    )
    model_configs, models, n_tau = pipeline.prepare_models(args.model_types)
    print(f"Using device: {device}")
    print(f"Number of log(tau) points: {n_tau}")

    print("Selected model configs:")
    for _, cfg in model_configs.items():
        print(f"  - {cfg['label']} ({cfg['experiment_key']})")

    mhd_normalizer = MhdNormalizer()
    stokes_normalizer = StokesNormalizer()
    default_cfg = TrainingConfig()
    mhd_normalizer.load(filepath=default_cfg.data_path / default_cfg.mhd_normalizer_path)
    stokes_normalizer.load(filepath=default_cfg.data_path / default_cfg.stokes_normalizer_path)

    modest = ModestData(
        circular_polarization_threshold=args.polarization_threshold if args.polarization_mask else None
    )

    diagnostics = ModestDiagnosticPlots(
        pipeline=pipeline,
        modest_output_dir=modest_output_dir,
        mhd_normalizer=mhd_normalizer,
        stokes_normalizer=stokes_normalizer,
        modest=modest,
        args=args,
    )
    diagnostics.prepare_snapshot(n_tau=n_tau)
    diagnostics.run(model_configs=model_configs, models=models)

    print(f"\nFinished analysis for {modest_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PINN MSCNN model")
    parser.add_argument(
        '--cropped-region', 
        action='store_true',
        help='whether to use cropped region (default: False)')   
    parser.add_argument(
        '--crop-bounds', 
        nargs=4, 
        type=int, 
        default=PLAGE_CROP_BOUNDS,
        metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX'),
        help=f'bounds for cropping the region (default plage bounds: {PLAGE_CROP_BOUNDS})'
    )
    parser.add_argument(
        '--polarization-mask',
        action='store_true',
        help='whether to apply circular polarization mask to the data (default: False)'
    )
    parser.add_argument(
        '--polarization-threshold',
        type=float,
        default=1e-2,
        help='threshold for circular polarization mask (default: 0.01)'
    )
    parser.add_argument(
        '--model-types', '--model_types',
        nargs='+',
        default=['all'],
        choices=['all', 'no_physics', 'wfa_only', 'doppler_only', 'black_body_only', 'all_physics_terms'],
        help="Which trained model types to load (default: all). Example: --model-types no_physics wfa_only",
    )
    parser.add_argument(
        '--crop-label',
        type=str,
        default='plage',
        help='name of the cropped region subfolder (used only with --cropped-region), e.g. "plage"',
    )
    parser.add_argument(
        '--tau-indices',
        nargs='*',
        type=int,
        default=None,
        help='Tau indices to plot (default: 0, mid, last)'
    )
    parser.add_argument(
        '--surface-stride',
        type=int,
        default=4,
        help='Downsampling stride for surface plots (default: 4)'
    )
    parser.add_argument(
        '--inference-batch-size',
        type=int,
        default=4096,
        help='Batch size for MODEST inference (default: 4096)'
    )
    args = parser.parse_args()
    main(args)