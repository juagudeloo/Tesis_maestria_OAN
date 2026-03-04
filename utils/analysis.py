from pathlib import Path
import json
import torch
import torch.nn.functional as F
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.base_training import TrainingConfig
from models.pinn_mscnn_model import PhysicsInformedMSCNN

try:
    from torchinfo import summary as torch_summary
except Exception:
    torch_summary = None

class AnalysisModelPipeline:
    """Reusable pipeline for config selection, runtime cfg building, and model loading."""

    def __init__(self, device, output_dir: Path | None = None):
        self.device = device
        self.output_dir = output_dir

    def get_model_configs(self) -> dict:
        base_model_path = Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments/")
        experiment_dir = base_model_path / "experiment_80_to_113"

        def _exp_cfg(exp_name: str, label: str, color: str, lw: float, ld: float, lt: float, use_physics):
            return {
                "path": experiment_dir / exp_name / "final_model.pth",
                "config_path": experiment_dir / exp_name / "experiment_config.json",
                "results_path": experiment_dir / "experiment_results.json",
                "experiment_key": exp_name,
                "output_subdir": exp_name,
                "use_physics": use_physics,
                "lambda_wfa": lw,
                "lambda_doppler": ld,
                "lambda_temp": lt,
                "label": label,
                "color": color,
            }

        return {
            "no_physics_80_to_113": _exp_cfg("no_physics", "No Physics 80 to 113", "blue", 0.0, 0.0, 0.0, None),
            "wfa_only_80_to_113": _exp_cfg("wfa_only", "WFA Only 80 to 113", "orange", 1.0, 0.0, 0.0, ["wfa"]),
            "doppler_only_80_to_113": _exp_cfg("doppler_only", "Doppler Only 80 to 113", "green", 0.0, 1.0, 0.0, ["doppler"]),
            "black_body_only_80_to_113": _exp_cfg("black_body_only", "BlackBody Only 80 to 113", "red", 0.0, 0.0, 1.0, ["temperature"]),
            "all_physics_terms_80_to_113": _exp_cfg("all_physics_terms", "All Physics Terms 80 to 113", "purple", 1.0, 1.0, 1.0, ["wfa", "doppler", "temperature"]),
        }

    @staticmethod
    def select_model_configs(model_configs: dict, selected_types: list[str]) -> dict:
        if not selected_types or "all" in selected_types:
            return model_configs
        selected_set = set(selected_types)
        filtered = {
            name: cfg for name, cfg in model_configs.items()
            if cfg.get("experiment_key") in selected_set
        }
        if not filtered:
            available = sorted({cfg.get("experiment_key") for cfg in model_configs.values()})
            raise ValueError(
                f"No models matched --model-types={selected_types}. "
                f"Available types: {available}"
            )
        return filtered

    def build_runtime_training_config(self, model_cfg: dict) -> TrainingConfig:
        cfg_kwargs = {
            "device": str(self.device),
            "batch_size": 512,
            "enable_epoch_plots": True,
        }

        cfg_path = model_cfg.get("config_path")
        if cfg_path is not None and Path(cfg_path).exists():
            with open(cfg_path, "r") as f:
                raw = json.load(f)

            train_cfg = raw.get("training_config", {})
            phys_cfg = raw.get("physics_config", {})
            data_cfg = raw.get("data_config", {})

            cfg_kwargs.update({
                "data_path": raw.get("data_path", TrainingConfig().data_path),
                "learning_rate": train_cfg.get("learning_rate", 1e-3),
                "batch_size": int(train_cfg.get("batch_size", 512)),
                "lambda_wfa": float(phys_cfg.get("lambda_wfa", model_cfg["lambda_wfa"])),
                "lambda_doppler": float(phys_cfg.get("lambda_doppler", model_cfg["lambda_doppler"])),
                "lambda_temp": float(phys_cfg.get("lambda_temp", model_cfg["lambda_temp"])),
                "blos_physics_mode": phys_cfg.get("blos_physics_mode", "tau_averaged"),
                "blos_target_logtau": phys_cfg.get("blos_target_logtau", None),
                "vlos_physics_mode": phys_cfg.get("vlos_physics_mode", "single_height"),
                "vlos_target_logtau": phys_cfg.get("vlos_target_logtau", -1.0),
                "temp_physics_mode": phys_cfg.get("temp_physics_mode", "single_height"),
                "temp_target_logtau": phys_cfg.get("temp_target_logtau", 0.0),
                "logtau_values": data_cfg.get("logtau_values", None),
            })

        cfg = TrainingConfig(**cfg_kwargs)
        cfg.log_dir = self.output_dir
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    def load_model(self, config: dict) -> tuple[PhysicsInformedMSCNN, int]:
        checkpoint = torch.load(config["path"], map_location=self.device)
        output_features = self._infer_output_features_from_checkpoint(checkpoint)
        if output_features % 3 != 0:
            raise ValueError(f"Invalid output_features={output_features}; expected multiple of 3")
        n_tau = output_features // 3

        model = PhysicsInformedMSCNN(
            scales=[1, 2, 3],
            in_channels=2,
            c1_filters=16,
            c2_filters=32,
            kernel_size=5,
            pool_size=2,
            n_linear_layers=4,
            output_features=output_features,
            input_length=112,
            lambda_wfa=config["lambda_wfa"],
            lambda_doppler=config["lambda_doppler"],
            lambda_temp=config["lambda_temp"],
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, n_tau

    @staticmethod
    def _infer_output_features_from_checkpoint(checkpoint: dict) -> int:
        state = checkpoint["model_state_dict"]
        key = "linear_block.output_layer.bias"
        if key not in state:
            raise KeyError(f"Missing '{key}' in checkpoint state_dict")
        return int(state[key].shape[0])

    def load_all_models(self, model_configs: dict) -> tuple[dict, int]:
        models = {}
        n_tau_ref = None
        for name, config in model_configs.items():
            print(f"Loading {config['label']}...")
            model, n_tau = self.load_model(config)
            if n_tau_ref is None:
                n_tau_ref = n_tau
            elif n_tau != n_tau_ref:
                raise ValueError(f"Inconsistent n_tau across models: {name} has {n_tau}, expected {n_tau_ref}")
            models[name] = model
            print(f"  ✓ Model loaded successfully (n_tau={n_tau})")
            self._print_model_summary(model, config["label"], self.device)
        print(f"\n✓ All {len(models)} models loaded\n")
        return models, int(n_tau_ref)

    def prepare_models(self, selected_types: list[str]) -> tuple[dict, dict, int]:
        model_configs = self.get_model_configs()
        model_configs = self.select_model_configs(model_configs, selected_types)
        models, n_tau = self.load_all_models(model_configs)
        return model_configs, models, n_tau

    @staticmethod
    def _print_model_summary(model: PhysicsInformedMSCNN, model_label: str, device) -> None:
        print(f"\n--- Torch summary: {model_label} ---")
        if torch_summary is not None:
            try:
                torch_summary(
                    model,
                    input_size=(1, 2, 112),
                    device=str(device),
                    col_names=("input_size", "output_size", "num_params"),
                    depth=3,
                    verbose=1,
                )
                return
            except Exception as e:
                print(f"torchinfo summary failed: {e}")

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print(f"Total params: {n_params:,}")
        print(f"Trainable params: {n_trainable:,}")

    def get_model_logtau_values(self, model_cfg: dict) -> list[float] | None:
        """Read model log(tau) nodes from experiment_results.json, fallback to experiment_config.json."""
        exp_key = model_cfg.get("experiment_key")
        results_path = model_cfg.get("results_path")

        if exp_key and results_path and Path(results_path).exists():
            try:
                with open(results_path, "r") as f:
                    raw = json.load(f)
                vals = raw.get(exp_key, {}).get("config", {}).get("logtau_values", None)
                if vals is not None:
                    return [float(v) for v in vals]
            except Exception:
                pass

        cfg_path = model_cfg.get("config_path")
        if cfg_path and Path(cfg_path).exists():
            try:
                with open(cfg_path, "r") as f:
                    raw = json.load(f)
                vals = raw.get("data_config", {}).get("logtau_values", None)
                if vals is not None:
                    return [float(v) for v in vals]
            except Exception:
                pass

        return None

    def predict_and_denormalize(
        self,
        model: PhysicsInformedMSCNN,
        stokes_input: np.ndarray,
        mhd_normalizer,
        pred_nx: int,
        pred_ny: int,
        batch_size: int = 4096,
    ) -> dict[str, np.ndarray]:
        """Predict and denormalize model outputs on the native prediction grid (no resizing)."""
        model.eval()
        all_pred = []

        with torch.no_grad():
            for i in range(0, stokes_input.shape[0], batch_size):
                x = torch.from_numpy(stokes_input[i:i + batch_size]).float().to(self.device)
                y = model(x).detach().cpu().numpy()
                all_pred.append(y)

        pred = np.concatenate(all_pred, axis=0)  # (N, 3*n_tau)
        n_tau_local = pred.shape[1] // 3
        pred = pred.reshape(-1, n_tau_local, 3)

        T_denorm = mhd_normalizer.denormalize(pred[:, :, 0], param="T").reshape(pred_nx, pred_ny, n_tau_local)
        Vz_denorm = mhd_normalizer.denormalize(pred[:, :, 1], param="Vz").reshape(pred_nx, pred_ny, n_tau_local)
        Bz_denorm = mhd_normalizer.denormalize(pred[:, :, 2], param="Bz").reshape(pred_nx, pred_ny, n_tau_local)
        return {"T": T_denorm, "Vz": Vz_denorm, "Bz": Bz_denorm}

    @staticmethod
    def common_tau_matches(
        modest_tau: list[float],
        pred_tau: list[float],
        atol: float = 1e-6,
    ) -> list[tuple[float, int, int]]:
        """Return (tau_value, idx_modest, idx_pred) for optical depths present in both grids."""
        matches: list[tuple[float, int, int]] = []
        for i_m, mt in enumerate(modest_tau):
            for i_p, pt in enumerate(pred_tau):
                if np.isclose(float(mt), float(pt), atol=atol, rtol=0.0):
                    matches.append((float(mt), i_m, i_p))
                    break
        return matches

class DiagnosticPlots:
    """Data-agnostic diagnostic plot generator."""

    def __init__(
        self,
        config: TrainingConfig,
        model_name: str,
        label: str | None = None,
        step: int | None = None,
        output_dir: str | Path | None = Path('./images'),
    ):
        self.config = config
        self.model_name = model_name
        self.label = label if label is not None else (f"step_{step}" if step is not None else "snapshot")

        self.logtau = config.get_logtau_values()
        self.ods = config.epoch_plot_ods if config.epoch_plot_ods is not None else [-1.0, -0.8, 0.0]
        self.params = config.epoch_plot_params if config.epoch_plot_params is not None else ["T", "Vz", "Bz"]
        self.n_sample = int(config.epoch_plot_scatter_samples)

        self.param_cmaps = {"T": "hot", "Vz": "bwr_r", "Bz": "PiYG"}
        self.error_cmap = "RdBu_r"
        
        base_out_dir = Path(output_dir)
        self.out_dir = base_out_dir / "final" / model_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def predict(self, model: PhysicsInformedMSCNN, stokes_input: np.ndarray) -> np.ndarray:
        n_pixels = stokes_input.shape[0]
        all_pred = []
        with torch.no_grad():
            for i in range(0, n_pixels, self.config.batch_size):
                x = torch.from_numpy(stokes_input[i:i + self.config.batch_size]).float().to(self.config.device)
                y = model(x).detach().cpu().numpy()
                all_pred.append(y)
        return np.concatenate(all_pred, axis=0)

    def denormalize_maps(
        self,
        pred_norm: np.ndarray,
        gt_values: np.ndarray,
        nx_pred: int,
        ny_pred: int,
        nx_gt: int,
        ny_gt: int,
        denormalize_param: Callable[[np.ndarray, str], np.ndarray],
        param_names: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        n_tau = int(len(self.logtau))

        pred_reshaped = pred_norm.reshape(pred_norm.shape[0], n_tau, len(param_names)) if pred_norm.ndim == 2 else pred_norm
        gt_reshaped = gt_values.reshape(gt_values.shape[0], n_tau, len(param_names)) if gt_values.ndim == 2 else gt_values

        pred_den, gt_den = {}, {}
        for param_idx, param_name in enumerate(param_names):
            pred_param_norm = pred_reshaped[:, :, param_idx]
            gt_param_values = gt_reshaped[:, :, param_idx]

            # Only predictions are denormalized
            pred_param_den = denormalize_param(pred_param_norm, param_name)

            pred_den[param_name] = pred_param_den.reshape(nx_pred, ny_pred, n_tau)
            gt_den[param_name] = gt_param_values.reshape(nx_gt, ny_gt, n_tau)

        return pred_den, gt_den

    def plot_image_panel(self, true_map: np.ndarray, pred_map: np.ndarray, p: str, od_eff: float) -> None:
        err_map = pred_map - true_map
        both = np.concatenate([true_map.ravel(), pred_map.ravel()])

        if p in ("Vz", "Bz"):
            vmax = np.nanquantile(np.abs(both), 0.99)
            vmin = -vmax
        else:
            vmin, vmax = np.nanquantile(both, [0.01, 0.99])

        emax = np.nanquantile(np.abs(err_map.ravel()), 0.99)
        param_cmap = self.param_cmaps.get(p, "viridis")

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        im0 = ax[0].imshow(true_map.T, origin="lower", cmap=param_cmap, vmin=vmin, vmax=vmax)
        ax[0].set_title(f"GT {p}")
        ax[0].axis("off")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(pred_map.T, origin="lower", cmap=param_cmap, vmin=vmin, vmax=vmax)
        ax[1].set_title(f"Pred {p}")
        ax[1].axis("off")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        im2 = ax[2].imshow(err_map.T, origin="lower", cmap=self.error_cmap, vmin=-emax, vmax=emax)
        ax[2].set_title(f"Error {p}")
        ax[2].axis("off")
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        fig.suptitle(f"Final Model | {self.model_name} | Snapshot {self.label} | {p} @ log(tau)={od_eff:.2f}")
        fig.tight_layout()
        fig.savefig(self.out_dir / f"{p}_logtau_{od_eff:.2f}_images.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    def plot_jointplot(self, true_map: np.ndarray, pred_map: np.ndarray, p: str, od_eff: float, tau_idx: int) -> None:
        x = true_map.ravel()
        y = pred_map.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size == 0:
            return

        if x.size > self.n_sample > 0:
            rng = np.random.default_rng(seed=tau_idx + 7)
            idx = rng.choice(x.size, size=self.n_sample, replace=False)
            x, y = x[idx], y[idx]

        rmse = np.sqrt(np.mean((y - x) ** 2))
        rrmse = rmse / (np.mean(np.abs(x)) + 1e-10)
        corr = np.corrcoef(x, y)[0, 1] if x.size > 1 else np.nan

        lo, hi = np.nanquantile(np.concatenate([x, y]), [0.01, 0.99])
        g = sns.jointplot(
            x=x,
            y=y,
            kind="scatter",
            height=6,
            s=8,
            alpha=0.25,
            marginal_kws={"bins": 50, "fill": True},
        )
        g.ax_joint.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        g.ax_joint.set_xlim(lo, hi)
        g.ax_joint.set_ylim(lo, hi)
        g.ax_joint.set_xlabel("Ground truth")
        g.ax_joint.set_ylabel("Prediction")
        g.fig.suptitle(
            f"Final Model | {self.model_name} | Snapshot {self.label} | {p} @ log(tau)={od_eff:.2f}\n"
            f"Corr={corr:.3f}, RRMSE={rrmse:.3f}",
            y=1.02,
        )
        g.fig.tight_layout()
        g.fig.savefig(self.out_dir / f"{p}_logtau_{od_eff:.2f}_jointplot.png", dpi=170, bbox_inches="tight")
        plt.close(g.fig)

    def generate(
        self,
        model: PhysicsInformedMSCNN,
        stokes_input: np.ndarray,
        gt_values: np.ndarray,
        nx_pred: int,
        ny_pred: int,
        denormalize_param: Callable[[np.ndarray, str], np.ndarray],
        param_names: list[str] | None = None,
        nx_gt: int | None = None,
        ny_gt: int | None = None,
    ) -> None:
        param_names = param_names or ["T", "Vz", "Bz"]
        nx_gt = nx_pred if nx_gt is None else nx_gt
        ny_gt = ny_pred if ny_gt is None else ny_gt

        was_training = model.training
        model.eval()

        pred_norm = self.predict(model=model, stokes_input=stokes_input)
        pred_den, gt_den = self.denormalize_maps(
            pred_norm=pred_norm,
            gt_values=gt_values,
            nx_pred=nx_pred,
            ny_pred=ny_pred,
            nx_gt=nx_gt,
            ny_gt=ny_gt,
            denormalize_param=denormalize_param,
            param_names=param_names,
        )

        for od in self.ods:
            tau_idx = int(np.argmin(np.abs(self.logtau - od)))
            od_eff = float(self.logtau[tau_idx])

            for p in self.params:
                true_map = gt_den[p][:, :, tau_idx]
                pred_map = pred_den[p][:, :, tau_idx]

                # Align GT to pred resolution for fair pixel-wise diagnostics
                if true_map.shape != pred_map.shape:
                    true_map = self._resize_map_to_shape(true_map, pred_map.shape)

                self.plot_image_panel(true_map=true_map, pred_map=pred_map, p=p, od_eff=od_eff)
                self.plot_jointplot(true_map=true_map, pred_map=pred_map, p=p, od_eff=od_eff, tau_idx=tau_idx)

        if was_training:
            model.train()

    @staticmethod
    def _resize_map_to_shape(arr2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        if arr2d.shape == target_shape:
            return arr2d
        t = torch.from_numpy(arr2d).float().unsqueeze(0).unsqueeze(0)
        out = F.interpolate(t, size=target_shape, mode="bilinear", align_corners=False)
        return out.squeeze(0).squeeze(0).cpu().numpy()
