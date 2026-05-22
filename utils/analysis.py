from pathlib import Path
import json
import csv
import torch
import torch.nn.functional as F
from typing import Callable, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.base_training import TrainingConfig
from models.pinn_mscnn_model import PhysicsInformedMSCNN
from utils.modest_data import transform_modest_stokes_profiles

try:
    from torchinfo import summary as torch_summary
except Exception:
    torch_summary = None


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute scalar quality metrics for prediction vs ground truth arrays."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]

    if y_true.size == 0:
        return {
            "n_points": 0,
            "rmse": np.nan,
            "rrmse": np.nan,
            "mae": np.nan,
            "nmae": np.nan,
            "bias": np.nan,
            "corr": np.nan,
            "r2": np.nan,
        }

    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    denom = float(np.mean(np.abs(y_true)) + 1e-10)

    if y_true.size > 1 and np.nanstd(y_true) > 0 and np.nanstd(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = np.nan

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    return {
        "n_points": int(y_true.size),
        "rmse": rmse,
        "rrmse": float(rmse / denom),
        "mae": mae,
        "nmae": float(mae / denom),
        "bias": bias,
        "corr": corr,
        "r2": r2,
    }


def format_metric(value: float, decimals: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{decimals}f}"


def compute_distribution_similarity_metrics(x: np.ndarray, y: np.ndarray, bins: int = 128) -> dict[str, float]:
    """Compute unpaired distribution-similarity metrics between two arrays."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size == 0 or y.size == 0:
        return {
            "n_true": int(x.size),
            "n_pred": int(y.size),
            "ks": np.nan,
            "w1_quantile": np.nan,
            "jsd": np.nan,
            "overlap": np.nan,
        }

    # KS statistic (two-sample) using ECDFs
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    grid = np.sort(np.concatenate([x_sorted, y_sorted]))
    cdf_x = np.searchsorted(x_sorted, grid, side="right") / x_sorted.size
    cdf_y = np.searchsorted(y_sorted, grid, side="right") / y_sorted.size
    ks = float(np.max(np.abs(cdf_x - cdf_y)))

    # 1-Wasserstein approximation via quantile functions
    n_q = min(20000, x.size, y.size)
    q = np.linspace(0.0, 1.0, n_q, endpoint=False) + 0.5 / n_q
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    w1_quantile = float(np.mean(np.abs(xq - yq)))

    # Histogram-based Jensen-Shannon divergence + overlap coefficient
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        jsd = np.nan
        overlap = np.nan
    else:
        hist_x, edges = np.histogram(x, bins=bins, range=(lo, hi), density=False)
        hist_y, _ = np.histogram(y, bins=edges, density=False)
        px = hist_x.astype(np.float64)
        py = hist_y.astype(np.float64)
        px /= max(px.sum(), 1.0)
        py /= max(py.sum(), 1.0)

        eps = 1e-12
        m = 0.5 * (px + py)
        kl_pm = np.sum(px * (np.log(px + eps) - np.log(m + eps)))
        kl_qm = np.sum(py * (np.log(py + eps) - np.log(m + eps)))
        jsd = float(0.5 * (kl_pm + kl_qm))
        overlap = float(np.sum(np.minimum(px, py)))

    return {
        "n_true": int(x.size),
        "n_pred": int(y.size),
        "ks": ks,
        "w1_quantile": w1_quantile,
        "jsd": jsd,
        "overlap": overlap,
    }


class AnalysisModelPipeline:
    """Reusable pipeline for config selection, runtime cfg building, and model loading."""

    def __init__(
        self,
        device,
        output_dir: Path | None = None,
        experiments_base_dir: Path | str | None = None,
        experiment_root: str = "experiment_80_to_113",
    ):
        self.device = device
        self.output_dir = output_dir
        self.experiments_base_dir = Path(experiments_base_dir) if experiments_base_dir is not None else Path(
            "/scratchsan/observatorio/juagudeloo/MUISCA/output/experiments/"
        )
        self.experiment_root = str(experiment_root)

    def get_model_configs(self) -> dict:
        experiment_dir = self.experiments_base_dir / self.experiment_root
        results_path = experiment_dir / "experiment_results.json"
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment root not found: {experiment_dir}")

        model_configs = {}
        for subdir in sorted([p for p in experiment_dir.iterdir() if p.is_dir()]):
            model_path = subdir / "final_model.pth"
            config_path = subdir / "experiment_config.json"
            if not model_path.exists():
                continue

            experiment_key = subdir.name
            lambda_wfa = 0.0
            lambda_doppler = 0.0
            lambda_temp = 0.0

            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        raw_cfg = json.load(f)
                    experiment_key = raw_cfg.get("experiment_name", experiment_key)
                    phys_cfg = raw_cfg.get("physics_config", {})
                    lambda_wfa = float(phys_cfg.get("lambda_wfa", 0.0))
                    lambda_doppler = float(phys_cfg.get("lambda_doppler", 0.0))
                    lambda_temp = float(phys_cfg.get("lambda_temp", 0.0))
                except Exception:
                    pass

            color = "gray"
            if experiment_key.startswith("no_physics"):
                color = "blue"
            elif experiment_key.startswith("wfa_only"):
                color = "orange"
            elif experiment_key.startswith("doppler_only"):
                color = "green"
            elif experiment_key.startswith("black_body_only"):
                color = "red"
            elif experiment_key.startswith("all_physics_terms"):
                color = "purple"

            key_name = experiment_key if experiment_key not in model_configs else f"{experiment_key}__{subdir.name}"
            model_configs[key_name] = {
                "path": model_path,
                "config_path": config_path,
                "results_path": results_path,
                "experiment_key": experiment_key,
                "output_subdir": subdir.name,
                "use_physics": None,
                "lambda_wfa": lambda_wfa,
                "lambda_doppler": lambda_doppler,
                "lambda_temp": lambda_temp,
                "label": f"{experiment_key} ({self.experiment_root})",
                "color": color,
            }

        if not model_configs:
            raise FileNotFoundError(
                f"No models with final_model.pth found under {experiment_dir}"
            )

        return model_configs

    @staticmethod
    def select_model_configs(model_configs: dict, selected_types: list[str]) -> dict:
        if not selected_types or "all" in selected_types:
            return model_configs

        def _normalize_selected_type(value: str) -> str:
            if value.startswith("wfa-lambda-"):
                return value.replace("wfa-lambda-", "wfa_only-lambda-", 1)
            if value.startswith("doppler-lambda-"):
                return value.replace("doppler-lambda-", "doppler_only-lambda-", 1)
            if value.startswith("black-body-lambda-"):
                return value.replace("black-body-lambda-", "black_body_only-lambda-", 1)
            if value.startswith("black_body-lambda-"):
                return value.replace("black_body-lambda-", "black_body_only-lambda-", 1)
            if value == "wfa":
                return "wfa_only"
            if value == "doppler":
                return "doppler_only"
            if value in {"black-body", "black_body"}:
                return "black_body_only"
            if value == "all_physics":
                return "all_physics_terms"
            return value

        selected_set = {_normalize_selected_type(s) for s in selected_types}
        filtered = {
            name: cfg for name, cfg in model_configs.items()
            if (
                cfg.get("experiment_key") in selected_set
                or any(
                    cfg.get("experiment_key", "").startswith(f"{sel}-lambda-")
                    for sel in selected_set
                )
            )
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

        pred = np.concatenate(all_pred, axis=0)  # (N, 3*n_tau) block order [T|Vz|Bz]
        if pred.ndim != 2 or pred.shape[1] % 3 != 0:
            raise ValueError(f"Expected prediction shape (N, 3*n_tau), got {pred.shape}")
        n_tau_local = pred.shape[1] // 3

        T_denorm = mhd_normalizer.denormalize(pred[:, :n_tau_local], param="T").reshape(pred_nx, pred_ny, n_tau_local)
        Vz_denorm = mhd_normalizer.denormalize(pred[:, n_tau_local:2 * n_tau_local], param="Vz").reshape(pred_nx, pred_ny, n_tau_local)
        Bz_denorm = mhd_normalizer.denormalize(pred[:, 2 * n_tau_local:3 * n_tau_local], param="Bz").reshape(pred_nx, pred_ny, n_tau_local)
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

class MuramDiagnosticPlots:
    """Data-agnostic diagnostic plot generator."""

    def __init__(
        self,
        config: TrainingConfig,
        model_name: str,
        label: str | None = None,
        step: int | None = None,
        output_dir: str | Path | None = Path('./images'),
        stokes_normalizer=None,
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
        self.stokes_normalizer = stokes_normalizer
        
        base_out_dir = Path(output_dir)
        step_folder = str(step) if step is not None else "snapshot"
        self.out_dir = base_out_dir / "final" / step_folder / model_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_rows: list[dict[str, float | str | int]] = []

    def plot_stokes_mean_std(
        self,
        stokes_input: np.ndarray,
        wavelengths: np.ndarray | None = None,
    ) -> None:
        """Plot denormalized mean ± std Stokes I/V profiles from final model inputs.
        
        wavelengths must not be None. Always use metadata-derived wavelength arrays.
        """
        arr = np.asarray(stokes_input, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1] != 2:
            return

        if wavelengths is None:
            raise ValueError(
                "wavelengths parameter is required and must come from MODEST metadata. "
                "Do not use hardcoded Hinode wavelength defaults."
            )
        
        wl = np.asarray(wavelengths, dtype=np.float64)
        if wl.ndim != 1 or wl.size != arr.shape[2]:
            raise ValueError(
                f"wavelengths shape mismatch: expected (n_wavelengths={arr.shape[2]},), got {wl.shape}. "
                "Ensure wavelengths are properly extracted from MODEST metadata."
            )

        I_norm = arr[:, 0, :]
        V_norm = arr[:, 1, :]
        I_mean_norm = np.mean(I_norm, axis=0)
        I_std_norm = np.std(I_norm, axis=0)
        V_mean_norm = np.mean(V_norm, axis=0)
        V_std_norm = np.std(V_norm, axis=0)

        I_mean = I_mean_norm
        I_std = I_std_norm
        V_mean = V_mean_norm
        V_std = V_std_norm
        if self.stokes_normalizer is not None and getattr(self.stokes_normalizer, "final_stats", None) is not None:
            stats_i = self.stokes_normalizer.final_stats.get("I", {})
            stats_v = self.stokes_normalizer.final_stats.get("V", {})
            mu_i = float(stats_i.get("mean", 0.0))
            sd_i = float(stats_i.get("std", 1.0))
            mu_v = float(stats_v.get("mean", 0.0))
            sd_v = float(stats_v.get("std", 1.0))
            I_mean = I_mean_norm * sd_i + mu_i
            I_std = I_std_norm * abs(sd_i)
            V_mean = V_mean_norm * sd_v + mu_v
            V_std = V_std_norm * abs(sd_v)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(f"{self.model_name} | Final Stokes Profiles: mean ± 1σ", fontsize=13, fontweight="bold")

        axes[0].plot(wl, I_mean, color="tab:orange", linewidth=1.8, label="Mean I")
        axes[0].fill_between(wl, I_mean - I_std, I_mean + I_std, color="tab:orange", alpha=0.25, label="±1σ")
        axes[0].set_ylabel("Stokes I")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best", fontsize=9)

        axes[1].plot(wl, V_mean, color="tab:purple", linewidth=1.8, label="Mean V")
        axes[1].fill_between(wl, V_mean - V_std, V_mean + V_std, color="tab:purple", alpha=0.25, label="±1σ")
        axes[1].set_xlabel("Wavelength [Angstrom]")
        axes[1].set_ylabel("Stokes V")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best", fontsize=9)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(self.out_dir / "stokes_mean_std_profiles.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    def _write_metrics_csv(self) -> None:
        if not self.metrics_rows:
            return
        metrics_path = self.out_dir / "metrics_summary.csv"
        fieldnames = [
            "model", "snapshot", "param", "logtau", "pairing", "n_points",
            "corr", "r2", "rmse", "rrmse", "mae", "nmae", "bias",
        ]
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_rows)
        print(f"[{self.model_name}] Metrics saved to: {metrics_path}")

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

    def plot_jointplot(self, true_map: np.ndarray, pred_map: np.ndarray, p: str, od_eff: float, tau_idx: int) -> dict[str, float] | None:
        x = true_map.ravel()
        y = pred_map.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size == 0:
            return None

        metrics = compute_regression_metrics(x, y)

        if x.size > self.n_sample > 0:
            rng = np.random.default_rng(seed=tau_idx + 7)
            idx = rng.choice(x.size, size=self.n_sample, replace=False)
            x, y = x[idx], y[idx]

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
            f"Corr={format_metric(metrics['corr'])}, R²={format_metric(metrics['r2'])}, "
            f"RRMSE={format_metric(metrics['rrmse'])}, NMAE={format_metric(metrics['nmae'])}",
            y=1.02,
        )
        g.fig.tight_layout()
        g.fig.savefig(self.out_dir / f"{p}_logtau_{od_eff:.2f}_jointplot.png", dpi=170, bbox_inches="tight")
        plt.close(g.fig)
        return metrics

    def generate(
        self,
        pred_den: dict[str, np.ndarray],
        gt_den: dict[str, np.ndarray],
        stokes_input: np.ndarray | None = None,
        wavelengths: np.ndarray | None = None,
    ) -> None:
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
                metrics = self.plot_jointplot(true_map=true_map, pred_map=pred_map, p=p, od_eff=od_eff, tau_idx=tau_idx)
                if metrics is not None:
                    self.metrics_rows.append({
                        "model": self.model_name,
                        "snapshot": self.label,
                        "param": p,
                        "logtau": od_eff,
                        "pairing": "paired",
                        "n_points": int(metrics["n_points"]),
                        "corr": float(metrics["corr"]),
                        "r2": float(metrics["r2"]),
                        "rmse": float(metrics["rmse"]),
                        "rrmse": float(metrics["rrmse"]),
                        "mae": float(metrics["mae"]),
                        "nmae": float(metrics["nmae"]),
                        "bias": float(metrics["bias"]),
                    })

        if stokes_input is not None:
            self.plot_stokes_mean_std(stokes_input=stokes_input, wavelengths=wavelengths)

        self._write_metrics_csv()

    @staticmethod
    def _resize_map_to_shape(arr2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        if arr2d.shape == target_shape:
            return arr2d
        t = torch.from_numpy(arr2d).float().unsqueeze(0).unsqueeze(0)
        out = F.interpolate(t, size=target_shape, mode="bilinear", align_corners=False)
        return out.squeeze(0).squeeze(0).cpu().numpy()

class ModestDiagnosticPlots:
    def __init__(
        self,
        pipeline: AnalysisModelPipeline,
        modest_output_dir: Path,
        mhd_normalizer,
        stokes_normalizer,
        modest,
        modest_cache,
        args,
    ):
        self.pipeline = pipeline
        self.modest_output_dir = modest_output_dir
        self.mhd_normalizer = mhd_normalizer
        self.stokes_normalizer = stokes_normalizer
        self.modest = modest
        self.modest_cache = modest_cache
        self.args = args

        self.modest_data = None
        self.modest_logtau = None
        self.modest_mhd_data = None
        self.modest_stokes_input = None
        self.pred_nx = None
        self.pred_ny = None
        self.n_tau_eff = None
        self.tau_indices = None
        self.metrics_rows: list[dict[str, float | str | int]] = []
        self.modest_wavelength: np.ndarray | None = None

    @staticmethod
    def _resize_map_to_shape(arr2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        if arr2d.shape == target_shape:
            return arr2d
        t = torch.from_numpy(arr2d).float().unsqueeze(0).unsqueeze(0)
        out = F.interpolate(t, size=target_shape, mode="bilinear", align_corners=False)
        return out.squeeze(0).squeeze(0).cpu().numpy()

    def _temperature_calibration_mode(self) -> str:
        mode = str(getattr(self.args, "temp_calibration_mode", "off")).lower()
        if mode not in {"off", "apply_fit"}:
            return "off"
        return mode

    def _temperature_calibration_clip_quantiles(self) -> tuple[float, float] | None:
        q = getattr(self.args, "temp_calibration_clip_quantiles", None)
        if q is None:
            return None
        if len(q) != 2:
            return None
        qlo = float(q[0])
        qhi = float(q[1])
        if not (0.0 <= qlo < qhi <= 1.0):
            return None
        return (qlo, qhi)

    def _temperature_calibration_min_samples(self) -> int:
        return int(max(10, int(getattr(self.args, "temp_calibration_min_samples", 500))))

    def _temperature_calibration_path(self, model_type: str, out_root: Path) -> Path:
        base_dir_raw = getattr(self.args, "temp_calibration_dir", None)
        if base_dir_raw:
            return Path(base_dir_raw) / model_type / "temperature_calibration.json"
        return out_root / "temperature_calibration.json"

    def _fit_temperature_affine_by_tau(
        self,
        model_type: str,
        matches: list[tuple[float, int, int]],
        true_cube: np.ndarray,
        pred_cube: np.ndarray,
    ) -> dict[str, Any]:
        min_samples = self._temperature_calibration_min_samples()
        clip_q = self._temperature_calibration_clip_quantiles()
        coefficients: dict[str, dict[str, float | int]] = {}

        for tau_val, i_mod, i_pred in matches:
            if i_mod >= true_cube.shape[2] or i_pred >= pred_cube.shape[2]:
                continue
            true_map = np.asarray(true_cube[:, :, i_mod], dtype=np.float32)
            pred_map = np.asarray(pred_cube[:, :, i_pred], dtype=np.float32)
            if true_map.shape != pred_map.shape:
                true_map = self._resize_map_to_shape(true_map, pred_map.shape)

            x = true_map.ravel()  # GT
            y = pred_map.ravel()  # Pred
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]

            if x.size < min_samples:
                continue

            if clip_q is not None:
                qlo, qhi = clip_q
                x_lo, x_hi = np.quantile(x, [qlo, qhi])
                y_lo, y_hi = np.quantile(y, [qlo, qhi])
                m2 = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
                if np.count_nonzero(m2) >= min_samples:
                    x = x[m2]
                    y = y[m2]

            if x.size < min_samples:
                continue

            a = 1.0
            b = float(np.nanmean(x - y))

            metrics_before = compute_regression_metrics(x, y)
            metrics_after = compute_regression_metrics(x, y + b)
            coefficients[f"{float(tau_val):.6f}"] = {
                "tau_value": float(tau_val),
                "a": float(a),
                "b": float(b),
                "n_points": int(x.size),
                "corr_before": float(metrics_before["corr"]),
                "corr_after": float(metrics_after["corr"]),
                "rrmse_before": float(metrics_before["rrmse"]),
                "rrmse_after": float(metrics_after["rrmse"]),
            }

        return {
            "mode": "bias_per_tau",
            "model_type": model_type,
            "min_samples": int(min_samples),
            "clip_quantiles": list(clip_q) if clip_q is not None else None,
            "coefficients": coefficients,
        }

    @staticmethod
    def _save_temperature_calibration(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _load_temperature_calibration(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _find_tau_coeff(coefficients: dict[str, dict[str, Any]], tau_val: float) -> dict[str, Any] | None:
        for key, coeff in coefficients.items():
            try:
                if np.isclose(float(key), float(tau_val), atol=1e-6, rtol=0.0):
                    return coeff
            except Exception:
                continue
        return None

    def _apply_temperature_calibration(
        self,
        pred_cube: np.ndarray,
        matches: list[tuple[float, int, int]],
        payload: dict[str, Any],
    ) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
        coefficients = payload.get("coefficients", {}) if isinstance(payload, dict) else {}
        calibrated = np.array(pred_cube, copy=True)
        applied_by_pred_idx: dict[int, dict[str, float]] = {}

        for tau_val, _i_mod, i_pred in matches:
            if i_pred >= calibrated.shape[2]:
                continue
            coeff = self._find_tau_coeff(coefficients, tau_val)
            if coeff is None:
                continue
            b = float(coeff.get("b", 0.0))
            calibrated[:, :, i_pred] = calibrated[:, :, i_pred] + b
            applied_by_pred_idx[i_pred] = {
                "a": 1.0,
                "b": b,
                "n_points": int(coeff.get("n_points", 0)),
            }

        return calibrated, applied_by_pred_idx

    def _write_metrics_csv(self, model_type: str, out_root: Path) -> None:
        rows = [r for r in self.metrics_rows if r.get("model") == model_type]
        if not rows:
            return
        metrics_path = out_root / "metrics_summary.csv"
        fieldnames = [
            "model", "param", "logtau", "comparison", "n_points", "n_true", "n_pred",
            "corr", "r2", "rmse", "rrmse", "mae", "nmae", "bias",
            "ks", "w1_quantile", "jsd", "overlap",
            "calibration_mode", "calibration_applied", "calibration_source",
            "calibration_a", "calibration_b", "calibration_n_fit",
        ]
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[{model_type}] Metrics saved to: {metrics_path}")

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
        prediction_input_mode = "downsampled" if getattr(self.args, "downsample_prediction_input", False) else "upsampled"
        self.modest_data = self.modest.load_all(
            region_bounds=tuple(self.args.crop_bounds) if self.args.cropped_region else None,
            apply_mask=self.args.polarization_mask,
            cache=self.modest_cache,
            use_cache=not self.args.no_modest_cache,
            prediction_input_mode=prediction_input_mode,
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
        # MODEST Stokes profiles are already continuum-normalized in the loader path.
        # Keep them as-is here and only apply the standard Stokes normalizer expected by the model.
        prediction_stokes = self.modest_data.get("prediction_stokes", self.modest_data["smoothed_stokes"])
        wl_raw = self.modest_data.get("wl", np.arange(prediction_stokes["I"].shape[-1]))
        self.modest_wavelength = np.asarray(
            wl_raw.value if hasattr(wl_raw, "value") else wl_raw,
            dtype=np.float64,
        )
        prediction_stokes = transform_modest_stokes_profiles(
            stokes=prediction_stokes,
            wavelength_angstrom=self.modest_wavelength,
            shift_positions=float(getattr(self.args, "modest_stokes_shift_positions", 0.0)),
            i_scale=float(getattr(self.args, "modest_stokes_i_scale", 1.0)),
            v_scale=float(getattr(self.args, "modest_stokes_v_scale", 1.0)),
            invert_direction=bool(getattr(self.args, "modest_stokes_invert_direction", False)),
        )
        self.pred_nx, self.pred_ny = prediction_stokes["I"].shape[:2]
        norm_stokes = self.stokes_normalizer.transform(prediction_stokes)
        I_flat = norm_stokes["I"].reshape(self.pred_nx * self.pred_ny, -1)
        V_flat = norm_stokes["V"].reshape(self.pred_nx * self.pred_ny, -1)
        self.modest_stokes_input = np.stack([I_flat, V_flat], axis=1).astype(np.float32)
        self.tau_indices = self._resolve_tau_indices(self.args.tau_indices, self.n_tau_eff)
        print(f"\nMODEST optical depth nodes: {[float(v) for v in self.modest_logtau]}")
        gt_nx, gt_ny = self.modest_mhd_data["T"].shape[:2]
        print(f"MODEST prediction input mode: {prediction_input_mode}")
        print(f"MODEST prediction input grid: {self.pred_nx}x{self.pred_ny} (GT: {gt_nx}x{gt_ny})")

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
        if param in ("Vz", "Bz"):
            vmax = np.quantile(np.abs(vals), 0.99)
            vmin = -vmax
        else:
            vmin, vmax = np.quantile(vals, [0.01, 0.99])

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
                0.5,
                0.5,
                f"Error skipped\nshape mismatch\nGT={gt.shape}\nPred={pr.shape}",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
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
    ) -> tuple[dict[str, float], str] | None:
        if true_2d.shape == pred_2d.shape:
            x = true_2d.ravel()
            y = pred_2d.ravel()
            m = np.isfinite(x) & np.isfinite(y)
            x_pair, y_pair = x[m], y[m]
            if x_pair.size == 0:
                return None

            if x_pair.size > max_points:
                rng = np.random.default_rng(seed=17)
                idx = rng.choice(x_pair.size, size=max_points, replace=False)
                x_plot, y_plot = x_pair[idx], y_pair[idx]
            else:
                x_plot, y_plot = x_pair, y_pair

            metrics_reg = compute_regression_metrics(x_pair, y_pair)
            metrics_dist = compute_distribution_similarity_metrics(x_pair, y_pair)
            comparison = "paired_pixel"
            title_metrics = (
                f"paired | Corr={format_metric(metrics_reg['corr'])}, R²={format_metric(metrics_reg['r2'])}, "
                f"RRMSE={format_metric(metrics_reg['rrmse'])}, NMAE={format_metric(metrics_reg['nmae'])}"
            )
        else:
            x = true_2d[np.isfinite(true_2d)].ravel()
            y = pred_2d[np.isfinite(pred_2d)].ravel()
            if x.size == 0 or y.size == 0:
                return None

            n = min(max_points, x.size, y.size)
            q = np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 / n
            x_plot = np.quantile(x, q)
            y_plot = np.quantile(y, q)
            metrics_reg = {
                "n_points": np.nan,
                "corr": np.nan,
                "r2": np.nan,
                "rmse": np.nan,
                "rrmse": np.nan,
                "mae": np.nan,
                "nmae": np.nan,
                "bias": np.nan,
            }
            metrics_dist = compute_distribution_similarity_metrics(x, y)
            comparison = "distribution_quantile"
            title_metrics = (
                f"distribution | KS={format_metric(metrics_dist['ks'])}, "
                f"W1q={format_metric(metrics_dist['w1_quantile'])}, "
                f"JSD={format_metric(metrics_dist['jsd'])}, OVL={format_metric(metrics_dist['overlap'])}"
            )

        lo = float(min(np.min(x_plot), np.min(y_plot)))
        hi = float(max(np.max(x_plot), np.max(y_plot)))
        g = sns.jointplot(x=x_plot, y=y_plot, kind="scatter", s=8, alpha=0.25, height=6)
        g.ax_joint.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        g.ax_joint.set_xlim(lo, hi)
        g.ax_joint.set_ylim(lo, hi)
        if comparison == "paired_pixel":
            g.ax_joint.set_xlabel("Ground truth")
            g.ax_joint.set_ylabel("Prediction")
        else:
            g.ax_joint.set_xlabel("Ground truth (quantiles)")
            g.ax_joint.set_ylabel("Prediction (quantiles)")
        g.fig.suptitle(
            f"{title}\n"
            f"{title_metrics}",
            y=1.02,
        )
        g.fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(g.fig)
        out_metrics = {
            **metrics_reg,
            **metrics_dist,
        }
        return out_metrics, comparison

    def _plot_stokes_mean_std(self, model_type: str, out_root: Path) -> None:
        """Plot denormalized mean ± std Stokes I/V profiles from MODEST model-input stokes.
        
        Wavelength data is always extracted from MODEST metadata. No hardcoded defaults are used.
        """
        if self.modest_stokes_input is None:
            return

        arr = np.asarray(self.modest_stokes_input, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1] != 2:
            return

        if self.modest_wavelength is None:
            raise ValueError(
                "MODEST wavelength metadata is required and must be set during prepare_snapshot(). "
                "Cannot use hardcoded defaults."
            )
        
        wl = np.asarray(self.modest_wavelength, dtype=np.float64)
        if wl.ndim != 1 or wl.size != arr.shape[2]:
            raise ValueError(
                f"MODEST wavelength shape mismatch: expected (n_wavelengths={arr.shape[2]},), got {wl.shape}. "
                "Ensure wavelength arrays are properly extracted from MODEST metadata."
            )

        I_norm = arr[:, 0, :]
        V_norm = arr[:, 1, :]
        I_mean_norm = np.mean(I_norm, axis=0)
        I_std_norm = np.std(I_norm, axis=0)
        V_mean_norm = np.mean(V_norm, axis=0)
        V_std_norm = np.std(V_norm, axis=0)

        I_mean = I_mean_norm
        I_std = I_std_norm
        V_mean = V_mean_norm
        V_std = V_std_norm
        stats = getattr(self.stokes_normalizer, "final_stats", None)
        if isinstance(stats, dict):
            stats_i = stats.get("I", {})
            stats_v = stats.get("V", {})
            mu_i = float(stats_i.get("mean", 0.0))
            sd_i = float(stats_i.get("std", 1.0))
            mu_v = float(stats_v.get("mean", 0.0))
            sd_v = float(stats_v.get("std", 1.0))
            I_mean = I_mean_norm * sd_i + mu_i
            I_std = I_std_norm * abs(sd_i)
            V_mean = V_mean_norm * sd_v + mu_v
            V_std = V_std_norm * abs(sd_v)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(f"{model_type} | MODEST Stokes Profiles: mean ± 1σ", fontsize=13, fontweight="bold")

        axes[0].plot(wl, I_mean, color="tab:orange", linewidth=1.8, label="Mean I")
        axes[0].fill_between(wl, I_mean - I_std, I_mean + I_std, color="tab:orange", alpha=0.25, label="±1σ")
        axes[0].set_ylabel("Stokes I")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best", fontsize=9)

        axes[1].plot(wl, V_mean, color="tab:purple", linewidth=1.8, label="Mean V")
        axes[1].fill_between(wl, V_mean - V_std, V_mean + V_std, color="tab:purple", alpha=0.25, label="±1σ")
        axes[1].set_xlabel("Wavelength [Angstrom]")
        axes[1].set_ylabel("Stokes V")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best", fontsize=9)

        out_root.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_root / "stokes_mean_std_profiles.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    def run(self, model_configs, models):
        calibration_mode = self._temperature_calibration_mode()
        print(f"Temperature calibration mode: {calibration_mode}")
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
            if bool(getattr(self.args, "modest_pred_mhd_invert_sign", False)):
                pred_mhd["Vz"] = -pred_mhd["Vz"]
                pred_mhd["Bz"] = -pred_mhd["Bz"]
                print(f"[{model_type}] Applied predicted MHD sign inversion for Vz and Bz.")
            out_root = self.modest_output_dir / model_type
            self._plot_stokes_mean_std(model_type=model_type, out_root=out_root)
            cal_path = self._temperature_calibration_path(model_type=model_type, out_root=out_root)

            applied_by_tau_idx: dict[int, dict[str, float]] = {}
            calibration_source = ""
            if calibration_mode == "apply_fit":
                cal_payload = self._load_temperature_calibration(cal_path)
                if cal_payload is None:
                    print(f"[{model_type}] No calibration file found — fitting now.")
                    cal_payload = self._fit_temperature_affine_by_tau(
                        model_type=model_type,
                        matches=matches,
                        true_cube=self.modest_mhd_data["T"],
                        pred_cube=pred_mhd["T"],
                    )
                    n_coeff = len(cal_payload.get("coefficients", {}))
                    if n_coeff > 0:
                        self._save_temperature_calibration(cal_path, cal_payload)
                        print(f"[{model_type}] Fitted and saved calibration ({n_coeff} taus) to: {cal_path}")
                    else:
                        print(f"[{model_type}] No calibration coefficients fitted (insufficient finite points). Proceeding without calibration.")
                        cal_payload = None
                else:
                    print(f"[{model_type}] Loaded existing calibration from: {cal_path}")
                if cal_payload is not None:
                    pred_mhd["T"], applied_by_tau_idx = self._apply_temperature_calibration(
                        pred_cube=pred_mhd["T"],
                        matches=matches,
                        payload=cal_payload,
                    )
                    calibration_source = str(cal_path)

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
                    plot_title = f"{model_type} | {param} | matched log(tau)={tau_val:.2f}"
                    if param == "T" and calibration_mode == "apply_fit":
                        if applied_by_tau_idx.get(i_pred) is not None:
                            plot_title += " | post-calibrated (apply_fit)"
                        else:
                            plot_title += " | raw prediction (apply_fit requested, no coeff)"
                    self._plot_imshows(
                        true_2d=true_map,
                        pred_2d=pred_map,
                        title=plot_title,
                        save_path=surface_dir / f"{param}_tau_{tau_val:+.2f}_imshow.png",
                        param=param,
                        transpose=True,
                    )
                    metrics_out = self._plot_jointplot(
                        true_2d=true_map,
                        pred_2d=pred_map,
                        title=plot_title,
                        save_path=joint_dir / f"{param}_tau_{tau_val:+.2f}_jointplot.png",
                    )
                    if metrics_out is not None:
                        metrics, comparison = metrics_out
                        cal_info = applied_by_tau_idx.get(i_pred) if param == "T" else None
                        self.metrics_rows.append({
                            "model": model_type,
                            "param": param,
                            "logtau": float(tau_val),
                            "comparison": comparison,
                            "n_points": int(metrics["n_points"]) if np.isfinite(metrics["n_points"]) else np.nan,
                            "n_true": int(metrics["n_true"]),
                            "n_pred": int(metrics["n_pred"]),
                            "corr": float(metrics["corr"]),
                            "r2": float(metrics["r2"]),
                            "rmse": float(metrics["rmse"]),
                            "rrmse": float(metrics["rrmse"]),
                            "mae": float(metrics["mae"]),
                            "nmae": float(metrics["nmae"]),
                            "bias": float(metrics["bias"]),
                            "ks": float(metrics["ks"]),
                            "w1_quantile": float(metrics["w1_quantile"]),
                            "jsd": float(metrics["jsd"]),
                            "overlap": float(metrics["overlap"]),
                            "calibration_mode": calibration_mode,
                            "calibration_applied": bool(cal_info is not None),
                            "calibration_source": calibration_source,
                            "calibration_a": float(cal_info["a"]) if cal_info is not None else np.nan,
                            "calibration_b": float(cal_info["b"]) if cal_info is not None else np.nan,
                            "calibration_n_fit": int(cal_info["n_points"]) if cal_info is not None else np.nan,
                        })
                    saved_any = True
            if saved_any:
                print(f"[{model_type}] Surface images saved to: {surface_dir}")
                print(f"[{model_type}] Jointplots saved to: {joint_dir}")
                self._write_metrics_csv(model_type=model_type, out_root=out_root)
            else:
                print(f"[{model_type}] No images were saved.")
