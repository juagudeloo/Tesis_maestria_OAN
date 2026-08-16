"""
Physics-Informed Neural Network Training Script
================================================

Trains the PINN MSCNN model using interleaved epoch training across multiple
MURaM simulation steps. Implements proper mini-batch handling, checkpoint saving,
and physics regularization.

Usage:
    python train_pinn_model.py --config config.yaml
    python train_pinn_model.py --resume checkpoint_epoch_50.pth
"""

import sys
import os
import argparse
import random
import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from typing import Any
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure utils and models are importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.muram_data import (
    MhdData,
    StokesData,
    MuramStepDataset,
    build_granulation_polarization_masks,
    build_balanced_region_indices,
    build_bz_strength_balanced_indices,
)
from utils.modest_data import ModestData
from utils.normalizer import MhdNormalizer, StokesNormalizer
from utils.cache_manage import MuramDataCache, ModestDataCache, BalancedTrainDataCache
from models.pinn_mscnn_model import PhysicsInformedMSCNN
from utils.physics_utils import ApproxInversions


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Data paths
    data_path: str = "/scratchsan/observatorio/juagudeloo/MUISCA/data/"
    mhd_normalizer_path: str = "normalization_stats/mhd_normalization.json"
    stokes_normalizer_path: str = "normalization_stats/stokes_normalization.json"
    kappa_path: str = "csv/kappa.0.dat"
    lsf_path: str = "hinode-MODEST/PSFs/hinode_sp.spline.psf"
    
    # Simulation parameters
    nx: int = 480
    ny: int = 480
    nz: int = 256
    z_max: int = 250
    min_step: int = 60
    max_step: int = 200
    dz_km: float = 10.0
    step_size: int = 1  # Step size between simulation steps

    # Optical depth remapping grid (used by MURaM -> tau mapping).
    # Defaults are the tau_500 grid NICOLE-generated data is fixed to
    # (scripts/synthesis/tau500_multi_step_regen.py: NEW_LOGTAU_GRID, 45 levels).
    # If logtau_values is provided, it overrides min/max/step
    logtau_values: list[float] | None = None
    logtau_min: float = -3.0
    logtau_max: float = 1.4
    logtau_step: float = 0.1

    # Stokes continuum normalization policy
    stokes_cont_indices: list[int] | None = None
    stokes_ic_mode: str = "fixed_global"  # 'per_step' or 'fixed_global'
    stokes_fixed_ic: float | None = None
    stokes_mult_factor: float = 1.0

    # Training data source: 'nicole_tau500' (NICOLE-synthesized on tau_500,
    # stokes_{step}_nicole_tau500.npy) or 'muram_legacy' (Rosseland-tau grid,
    # stokes_{step}.npy -- kept for reference/old checkpoints, not the default)
    data_source: str = "nicole_tau500"

    # Training parameters
    n_epochs: int = 20
    batch_size: int = 512  # Spatial batch size (512 pixels per batch)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Model architecture
    scales: list[int] | None = None  # [1, 2, 3]
    in_channels: int = 2
    c1_filters: int = 16
    c2_filters: int = 32
    kernel_size: int = 5
    pool_size: int = 2
    n_linear_layers: int = 4
    
    # Physics parameters
    central_wavelength: float = 6301.5  # Angstroms
    lande_factor: float = 1.67
    wl_range: tuple[int, int] = (15, 60)
    lambda_wfa: float = 0.01      # WFA term weight
    lambda_doppler: float = 0.01  # Doppler term weight
    lambda_temp: float = 0.01     # Temperature term weight
    wfa_gate_mode: str = "off"  # 'off', 'threshold', or 'plateau'
    wfa_gate_threshold: float = 0.0  # Activate WFA when epoch train MSE <= threshold
    wfa_gate_patience: int = 5  # Plateau epochs before activating WFA
    wfa_gate_min_delta: float = 1e-4  # Minimum MSE improvement to reset plateau counter
    wfa_gate_warmup_epochs: int = 0  # Minimum epochs before gate can activate
    blos_physics_mode: str = "tau_averaged"  # 'tau_averaged' or 'single_height'
    blos_target_logtau: float | None = None  # Target log(tau) for B_LOS single_height mode
    vlos_physics_mode: str = "single_height"  # 'tau_averaged' or 'single_height'
    vlos_target_logtau: float | None = -1.0  # Target log(tau) for V_LOS single_height mode
    temp_physics_mode: str = "single_height"  # 'tau_averaged' or 'single_height'
    temp_target_logtau: float | None = 0.0  # Target log(tau) for temperature single_height mode
    temp_reference_temperature: float = 6000.0  # Reference temperature for blackbody (Kelvin)
    temp_continuum_indices: list[int] | None = None  # Continuum indices for temperature estimation
    temp_continuum_wavelength: float = 6300.5  # Angstroms
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10  # Save checkpoint every N epochs
    resume_from: str | None = None
    
    # Logging
    log_dir: str = "logs"
    log_every: int = 10  # Log metrics every N batches within an epoch
    
    # Device and performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # New: Caching parameters
    use_cache: bool = True
    cache_dir: str = "/scratchsan/observatorio/juagudeloo/MUISCA/.muram_cache"

    # Post-balancing cache (stores final train-ready tensors)
    use_balanced_cache: bool = False
    balanced_cache_dir: str = "/scratchsan/observatorio/juagudeloo/MUISCA/.muram_balanced_cache"
    clear_balanced_cache: bool = False
    balanced_cache_strategy: str = "auto"  # 'auto', 'preload', or 'disk'
    balanced_cache_ram_budget_gb: float = 32.0
    balanced_cache_ram_fraction: float = 0.75

    # Region-mask balancing (training only)
    apply_region_mask: bool = True
    log_region_mask_stats: bool = True

    # Bz histogram balancing (training only)
    apply_bz_bin_balance: bool = False
    log_bz_bin_balance_stats: bool = True
    bz_balance_mode: str = "mean_abs"  # 'mean_abs', 'max_abs', or 'tau_index'
    bz_balance_bins: int = 12
    bz_balance_tau_idx: int | None = None
    bz_balance_scope: str = "global"  # 'global' or 'per_step'
    bz_balance_seed: int = 42
    
    # Epoch diagnostics (image + scatter evolution)
    enable_epoch_plots: bool = True
    epoch_plot_step: int | None = None  # If None, use first validation step
    epoch_plot_ods: list[float] | None = None
    epoch_plot_params: list[str] | None = None
    epoch_plot_scatter_samples: int = 5000
    enable_epoch_videos: bool = True
    epoch_plot_video_fps: int = 4

    # MODEST snapshot diagnostics per epoch
    enable_modest_epoch_plots: bool = False
    modest_cache_dir: str = "/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache"
    no_modest_cache: bool = False
    clear_modest_cache: bool = False
    modest_downsample_prediction_input: bool = True
    modest_stokes_v_multiplier: float = -1.0
    modest_polarization_mask: bool = False
    modest_polarization_threshold: float = 1e-2
    modest_crop_bounds: list[int] | tuple[int, int, int, int] | None = None
    modest_epoch_plot_ods: list[float] | None = None
    modest_epoch_plot_params: list[str] | None = None
    modest_epoch_plot_scatter_samples: int | None = None
    modest_temp_calibration_mode: str = "off"  # 'off', 'fit_only', 'apply_only'
    modest_temp_calibration_file: str | None = None
    modest_temp_calibration_min_samples: int = 500
    modest_temp_calibration_clip_quantiles: list[float] | tuple[float, float] | None = None

    def __post_init__(self):
        if self.scales is None:
            self.scales = [1, 2, 3]
        if self.stokes_cont_indices is None:
            self.stokes_cont_indices = [0, 1, 2, 3]

        valid_data_sources = {"muram_legacy", "nicole_tau500"}
        if self.data_source not in valid_data_sources:
            raise ValueError(
                f"data_source must be one of {sorted(valid_data_sources)}, got {self.data_source!r}"
            )
        # Non-legacy sources get isolated normalizer-stats paths by default, so a
        # fresh compute_normalization_stats.py run never overwrites the legacy files.
        if self.data_source != "muram_legacy":
            default_mhd_norm_path = "normalization_stats/mhd_normalization.json"
            default_stokes_norm_path = "normalization_stats/stokes_normalization.json"
            if self.mhd_normalizer_path == default_mhd_norm_path:
                self.mhd_normalizer_path = f"normalization_stats/{self.data_source}/mhd_normalization.json"
            if self.stokes_normalizer_path == default_stokes_norm_path:
                self.stokes_normalizer_path = f"normalization_stats/{self.data_source}/stokes_normalization.json"

        if self.data_source == "muram_legacy" and self.stokes_ic_mode == "fixed_global" and self.stokes_fixed_ic is None:
            ic_stats_path = Path(self.data_path) / "normalization_stats" / "ic_reference_stats.json"
            if ic_stats_path.exists():
                with open(ic_stats_path, "r", encoding="utf-8") as f:
                    ic_payload = json.load(f)
                fixed_ic = ic_payload.get("fixed_ic")
                if fixed_ic is not None:
                    self.stokes_fixed_ic = float(fixed_ic)

        if self.data_source != "muram_legacy" and self.stokes_ic_mode == "fixed_global" and self.stokes_fixed_ic is None:
            # fixed_ic is meaningless for pre-normalized sources (already continuum-
            # normalized by NICOLE); skip it instead of requiring an unrelated legacy
            # ic_reference_stats.json.
            self.stokes_ic_mode = "per_step"

        if not np.isfinite(float(self.stokes_mult_factor)) or float(self.stokes_mult_factor) <= 0:
            raise ValueError(f"stokes_mult_factor must be finite and > 0, got {self.stokes_mult_factor}")

        valid_ic_modes = {"per_step", "fixed_global"}
        if self.stokes_ic_mode not in valid_ic_modes:
            raise ValueError(
                f"stokes_ic_mode must be one of {sorted(valid_ic_modes)}, got {self.stokes_ic_mode!r}"
            )
        if self.stokes_ic_mode == "fixed_global":
            if self.stokes_fixed_ic is None:
                raise ValueError("stokes_fixed_ic is required when stokes_ic_mode='fixed_global'")
            if not np.isfinite(float(self.stokes_fixed_ic)) or float(self.stokes_fixed_ic) <= 0:
                raise ValueError(f"stokes_fixed_ic must be finite and > 0, got {self.stokes_fixed_ic}")
        if self.temp_continuum_indices is None:
            self.temp_continuum_indices = [0, 1, 2, 3]
        valid_wfa_gate_modes = {"off", "threshold", "plateau"}
        if self.wfa_gate_mode not in valid_wfa_gate_modes:
            raise ValueError(
                f"wfa_gate_mode must be one of {sorted(valid_wfa_gate_modes)}, got {self.wfa_gate_mode!r}"
            )
        if self.wfa_gate_patience < 1:
            raise ValueError("wfa_gate_patience must be >= 1")
        if self.wfa_gate_warmup_epochs < 0:
            raise ValueError("wfa_gate_warmup_epochs must be >= 0")
        if self.logtau_values is not None and len(self.logtau_values) == 0:
            self.logtau_values = None
        # Normalize cache dir (allow shared override via env)
        default_cache = "/scratchsan/observatorio/juagudeloo/MUISCA/.muram_cache"
        if self.data_source != "muram_legacy" and self.cache_dir == default_cache:
            self.cache_dir = f"{default_cache}_{self.data_source}"
        if (not self.cache_dir or self.cache_dir == default_cache) and os.environ.get("MURAM_CACHE_DIR"):
            self.cache_dir = os.environ["MURAM_CACHE_DIR"]
        self.cache_dir = str(Path(self.cache_dir).expanduser().resolve())

        default_balanced_cache = "/scratchsan/observatorio/juagudeloo/MUISCA/.muram_balanced_cache"
        if self.data_source != "muram_legacy" and self.balanced_cache_dir == default_balanced_cache:
            self.balanced_cache_dir = f"{default_balanced_cache}_{self.data_source}"
        if (not self.balanced_cache_dir or self.balanced_cache_dir == default_balanced_cache) and os.environ.get("MURAM_BALANCED_CACHE_DIR"):
            self.balanced_cache_dir = os.environ["MURAM_BALANCED_CACHE_DIR"]
        self.balanced_cache_dir = str(Path(self.balanced_cache_dir).expanduser().resolve())

        valid_balanced_cache_strategies = {"auto", "preload", "disk"}
        self.balanced_cache_strategy = str(self.balanced_cache_strategy).lower()
        if self.balanced_cache_strategy not in valid_balanced_cache_strategies:
            raise ValueError(
                "balanced_cache_strategy must be one of "
                f"{sorted(valid_balanced_cache_strategies)}, got {self.balanced_cache_strategy!r}"
            )
        self.balanced_cache_ram_budget_gb = float(self.balanced_cache_ram_budget_gb)
        if self.balanced_cache_ram_budget_gb <= 0:
            raise ValueError("balanced_cache_ram_budget_gb must be > 0")
        self.balanced_cache_ram_fraction = float(self.balanced_cache_ram_fraction)
        if self.balanced_cache_ram_fraction <= 0 or self.balanced_cache_ram_fraction > 1:
            raise ValueError("balanced_cache_ram_fraction must be in (0, 1]")

        default_modest_cache = "/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache"
        if (not self.modest_cache_dir or self.modest_cache_dir == default_modest_cache) and os.environ.get("MODEST_CACHE_DIR"):
            self.modest_cache_dir = os.environ["MODEST_CACHE_DIR"]
        self.modest_cache_dir = str(Path(self.modest_cache_dir).expanduser().resolve())

        if self.modest_crop_bounds is not None:
            if len(self.modest_crop_bounds) != 4:
                raise ValueError("modest_crop_bounds must contain exactly 4 integers: [y_start, y_end, x_start, x_end]")
            self.modest_crop_bounds = tuple(int(v) for v in self.modest_crop_bounds)

        valid_cal_modes = {"off", "fit_only", "apply_only"}
        self.modest_temp_calibration_mode = str(self.modest_temp_calibration_mode).lower()
        if self.modest_temp_calibration_mode not in valid_cal_modes:
            raise ValueError(
                f"modest_temp_calibration_mode must be one of {sorted(valid_cal_modes)}, "
                f"got {self.modest_temp_calibration_mode!r}"
            )
        self.modest_temp_calibration_min_samples = int(max(10, int(self.modest_temp_calibration_min_samples)))

        if self.modest_temp_calibration_clip_quantiles is not None:
            if len(self.modest_temp_calibration_clip_quantiles) != 2:
                raise ValueError("modest_temp_calibration_clip_quantiles must contain exactly two values [q_low, q_high]")
            q_low = float(self.modest_temp_calibration_clip_quantiles[0])
            q_high = float(self.modest_temp_calibration_clip_quantiles[1])
            if not (0.0 <= q_low < q_high <= 1.0):
                raise ValueError(
                    "modest_temp_calibration_clip_quantiles must satisfy 0 <= q_low < q_high <= 1"
                )
            self.modest_temp_calibration_clip_quantiles = [q_low, q_high]

        if self.modest_temp_calibration_file:
            self.modest_temp_calibration_file = str(Path(self.modest_temp_calibration_file).expanduser().resolve())

        if not np.isfinite(float(self.modest_stokes_v_multiplier)) or float(self.modest_stokes_v_multiplier) == 0.0:
            raise ValueError(
                f"modest_stokes_v_multiplier must be finite and non-zero, got {self.modest_stokes_v_multiplier}"
            )

        valid_bz_balance_modes = {"mean_abs", "max_abs", "tau_index"}
        self.bz_balance_mode = str(self.bz_balance_mode).lower()
        if self.bz_balance_mode not in valid_bz_balance_modes:
            raise ValueError(
                f"bz_balance_mode must be one of {sorted(valid_bz_balance_modes)}, got {self.bz_balance_mode!r}"
            )
        self.bz_balance_bins = int(self.bz_balance_bins)
        if self.bz_balance_bins < 2:
            raise ValueError("bz_balance_bins must be >= 2")
        if self.bz_balance_tau_idx is not None:
            self.bz_balance_tau_idx = int(self.bz_balance_tau_idx)
            if self.bz_balance_tau_idx < 0:
                raise ValueError("bz_balance_tau_idx must be >= 0")
        valid_bz_balance_scopes = {"global", "per_step"}
        self.bz_balance_scope = str(self.bz_balance_scope).lower()
        if self.bz_balance_scope not in valid_bz_balance_scopes:
            raise ValueError(
                f"bz_balance_scope must be one of {sorted(valid_bz_balance_scopes)}, got {self.bz_balance_scope!r}"
            )
        self.bz_balance_seed = int(self.bz_balance_seed)

        # Convert paths to Path objects
        self.data_path = Path(self.data_path)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, path: Path):
        """Save configuration to JSON."""
        with open(path, 'w') as f:
            config_dict = asdict(self)
            # Convert Path objects to strings for JSON serialization
            for key, value in config_dict.items():
                if isinstance(value, Path):
                    config_dict[key] = str(value)
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        valid_keys = {field.name for field in fields(cls)}
        filtered_config = {key: value for key, value in config_dict.items() if key in valid_keys}
        return cls(**filtered_config)

    def get_logtau_values(self) -> np.ndarray:
        """Resolve optical-depth grid for MURaM remapping/physics context."""
        if self.logtau_values is not None:
            logtau = np.asarray(self.logtau_values, dtype=np.float32)
        else:
            if self.logtau_step <= 0:
                raise ValueError(f"logtau_step must be > 0, got {self.logtau_step}")
            # include endpoint robustly. Accumulate in float64 (numpy's default)
            # then cast down -- computing the arange directly in float32 accrues
            # visible step-to-step rounding drift (~4e-6 by the last of 45
            # steps), enough to fail exact-grid-match checks against externally
            # generated data (e.g. the tau500 atmos_*.npz files) that use the
            # same min/max/step but arange's float64 default.
            logtau = np.arange(
                self.logtau_min,
                self.logtau_max + 0.5 * self.logtau_step,
                self.logtau_step,
            ).astype(np.float32)

        if logtau.ndim != 1 or logtau.size < 2:
            raise ValueError("logtau grid must be 1D with at least 2 points")
        if not np.all(np.diff(logtau) > 0):
            raise ValueError("logtau grid must be strictly increasing")

        return np.round(logtau, 6)

    def get_n_logtau(self) -> int:
        """Number of optical-depth levels used by mapping/model output."""
        return int(self.get_logtau_values().shape[0])

class MetricsLogger:
    """Tracks and logs training metrics."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.epoch_losses = []
        self.batch_losses = []
        self.lr_history = []
        
        # File handlers
        self.epoch_log = open(log_dir / "epoch_log.csv", 'w')
        self.batch_log = open(log_dir / "batch_log.csv", 'w')
        
        # Write headers
        self.epoch_log.write("epoch,train_loss,val_loss,lr\n")
        self.batch_log.write("epoch,step,batch,loss,mse_loss,physics_loss,wfa_loss,doppler_loss,temperature_loss\n")
    
    def log_batch(self, epoch: int, step: int, batch: int, loss_dict: dict[str, float]):
        """Log batch-level metrics."""
        self.batch_log.write(
            f"{epoch},{step},{batch},"
            f"{loss_dict.get('total_loss', 0.0)},"
            f"{loss_dict.get('mse_loss', 0.0)},"
            f"{loss_dict.get('physics_loss', 0.0)},"
            f"{loss_dict.get('wfa_loss', 0.0)},"
            f"{loss_dict.get('doppler_loss', 0.0)},"
            f"{loss_dict.get('temperature_loss', 0.0)}\n"
        )
        self.batch_log.flush()
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Log epoch-level metrics."""
        self.epoch_log.write(f"{epoch},{train_loss},{val_loss},{lr}\n")
        self.epoch_log.flush()
        
        self.epoch_losses.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})
        self.lr_history.append(lr)
    
    def close(self):
        """Close file handlers."""
        self.epoch_log.close()
        self.batch_log.close()
    
    def __del__(self):
        self.close()

def build_cache_config_signature(config: TrainingConfig) -> dict:
    """Shared cache-signature contract across training/ablation/analysis."""
    return {
        'data_source': str(config.data_source),
        'nx': config.nx,
        'ny': config.ny,
        'nz': config.nz,
        'z_max': config.z_max,
        'dz_km': config.dz_km,
        'central_wavelength': config.central_wavelength,
        'wl_range': config.wl_range,
        'logtau_values': tuple(float(x) for x in config.get_logtau_values().tolist()),
        'stokes_cont_indices': tuple(int(x) for x in (config.stokes_cont_indices or [0, 1, 2, 3])),
        'stokes_ic_mode': str(config.stokes_ic_mode),
        'stokes_fixed_ic': None if config.stokes_fixed_ic is None else float(config.stokes_fixed_ic),
        'stokes_mult_factor': float(config.stokes_mult_factor),
    }


def build_balanced_cache_signature(config: TrainingConfig, train_steps: list[int]) -> dict:
    """Signature for post-balancing cache validity."""
    return {
        "version": 1,
        "data_source": str(config.data_source),
        "steps": [int(s) for s in sorted(train_steps)],
        "apply_region_mask": bool(config.apply_region_mask),
        "apply_bz_bin_balance": bool(config.apply_bz_bin_balance),
        "bz_balance_scope": str(config.bz_balance_scope),
        "bz_balance_mode": str(config.bz_balance_mode),
        "bz_balance_bins": int(config.bz_balance_bins),
        "bz_balance_tau_idx": None if config.bz_balance_tau_idx is None else int(config.bz_balance_tau_idx),
        "bz_balance_seed": int(config.bz_balance_seed),
        "logtau_values": [float(x) for x in config.get_logtau_values().tolist()],
        "stokes_cont_indices": [int(x) for x in (config.stokes_cont_indices or [0, 1, 2, 3])],
        "stokes_ic_mode": str(config.stokes_ic_mode),
        "stokes_fixed_ic": None if config.stokes_fixed_ic is None else float(config.stokes_fixed_ic),
        "stokes_mult_factor": float(config.stokes_mult_factor),
    }


def estimate_balanced_cache_sample_bytes(dataset: MuramStepDataset) -> int:
    """Estimate in-memory bytes needed to preload one balanced sample set."""
    return (
        int(dataset.stokes_input.nbytes)
        + int(dataset.mhd_targets.nbytes)
        + int(dataset.spatial_indices.nbytes)
    )


class BalancedStepTensorDataset(Dataset):
    """Dataset wrapping cached balanced tensors (already normalized)."""

    def __init__(self, stokes_input: np.ndarray, mhd_targets: np.ndarray, spatial_indices: np.ndarray):
        if stokes_input.ndim != 3:
            raise ValueError(f"Expected stokes_input as 3D array, got shape {stokes_input.shape}")
        if mhd_targets.ndim != 2:
            raise ValueError(f"Expected mhd_targets as 2D array, got shape {mhd_targets.shape}")
        if spatial_indices.ndim != 2 or spatial_indices.shape[1] != 2:
            raise ValueError(f"Expected spatial_indices as (N,2), got shape {spatial_indices.shape}")
        n = int(stokes_input.shape[0])
        if int(mhd_targets.shape[0]) != n or int(spatial_indices.shape[0]) != n:
            raise ValueError("Balanced tensors have inconsistent sample dimension")

        self.stokes_input = np.asarray(stokes_input, dtype=np.float32)
        self.mhd_targets = np.asarray(mhd_targets, dtype=np.float32)
        self.spatial_indices = np.asarray(spatial_indices, dtype=np.int64)

    def __len__(self):
        return int(self.stokes_input.shape[0])

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.stokes_input[idx]).float(),
            torch.from_numpy(self.mhd_targets[idx]).float(),
            torch.from_numpy(self.spatial_indices[idx]).long(),
        )


def build_or_refresh_balanced_cache(
    train_steps: list[int],
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    raw_cache: MuramDataCache | None,
    global_bz_selection_indices: dict[int, np.ndarray] | None,
    global_bz_balance_metadata: dict[str, Any] | None,
) -> tuple[BalancedTrainDataCache, str, dict[str, Any]]:
    """Build balanced-cache entries for all train steps if needed.

    Returns cache object, signature hash, and summary report.
    """
    balanced_cache = BalancedTrainDataCache(cache_dir=config.balanced_cache_dir, compression="lzf")
    signature = build_balanced_cache_signature(config=config, train_steps=train_steps)
    signature_hash = BalancedTrainDataCache.make_signature_hash(signature)

    if config.clear_balanced_cache:
        balanced_cache.reset(signature=signature, signature_hash=signature_hash)

    if not balanced_cache.ensure_signature(signature=signature, signature_hash=signature_hash):
        print("Balanced cache signature mismatch detected; rebuilding balanced cache.")
        balanced_cache.reset(signature=signature, signature_hash=signature_hash)

    built_steps = 0
    reused_steps = 0
    skipped_steps = 0
    preload_bytes = 0

    for step in tqdm(train_steps, desc="Build balanced cache"):
        if balanced_cache.has_step(step=step, signature_hash=signature_hash):
            reused_steps += 1
            step_entry = balanced_cache.manifest.get("steps", {}).get(str(step), {})
            preload_bytes += int(step_entry.get("metadata", {}).get("preload_bytes", 0))
            continue

        result = load_and_prepare_step(
            step=step,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            cache=raw_cache,
            apply_balanced_masks=config.apply_region_mask,
            log_region_stats=False,
            apply_bz_balance=(config.apply_bz_bin_balance and config.bz_balance_scope == "per_step"),
            global_bz_selection_indices=global_bz_selection_indices,
            global_bz_balance_metadata=global_bz_balance_metadata,
            ignore_missing_files=True,
        )
        if result is None:
            skipped_steps += 1
            continue

        dataset, approx_data = result
        step_preload_bytes = estimate_balanced_cache_sample_bytes(dataset)
        preload_bytes += step_preload_bytes

        balanced_cache.save_step(
            step=step,
            signature_hash=signature_hash,
            stokes_input=dataset.stokes_input,
            mhd_targets=dataset.mhd_targets,
            spatial_indices=dataset.spatial_indices,
            approx_data=approx_data,
            extra_metadata={
                "preload_bytes": int(step_preload_bytes),
                "n_selected": int(len(dataset)),
            },
        )
        built_steps += 1

    stats = balanced_cache.get_stats()
    report = {
        "signature_hash": signature_hash,
        "built_steps": int(built_steps),
        "reused_steps": int(reused_steps),
        "skipped_steps": int(skipped_steps),
        "total_steps_cached": int(stats.get("total_steps", 0)),
        "total_selected": int(stats.get("total_selected", 0)),
        "total_disk_bytes": int(stats.get("total_bytes", 0)),
        "total_disk_mb": float(stats.get("total_size_mb", 0.0)),
        "single_file_estimated_mb": float(stats.get("total_size_mb", 0.0)),
        "multi_file_recommended": True,
        "estimated_preload_bytes": int(preload_bytes),
        "estimated_preload_gb": float(preload_bytes) / (1024**3),
    }
    return balanced_cache, signature_hash, report


def choose_balanced_cache_runtime_mode(config: TrainingConfig, estimated_preload_bytes: int) -> str:
    """Select runtime mode for balanced cache: preload or disk."""
    requested = str(config.balanced_cache_strategy).lower()
    if requested in {"preload", "disk"}:
        return requested

    allowed_bytes = int(config.balanced_cache_ram_budget_gb * config.balanced_cache_ram_fraction * (1024**3))
    if estimated_preload_bytes <= allowed_bytes:
        return "preload"
    return "disk"


def preload_balanced_steps_from_cache(
    train_steps: list[int],
    balanced_cache: BalancedTrainDataCache,
    signature_hash: str,
) -> dict[int, tuple[BalancedStepTensorDataset, dict[str, np.ndarray]]]:
    """Load all cached balanced steps into RAM once."""
    loaded: dict[int, tuple[BalancedStepTensorDataset, dict[str, np.ndarray]]] = {}
    for step in tqdm(train_steps, desc="Preload balanced cache"):
        if not balanced_cache.has_step(step=step, signature_hash=signature_hash):
            continue
        stokes_input, mhd_targets, spatial_indices, approx_data = balanced_cache.load_step(
            step=step,
            signature_hash=signature_hash,
        )
        loaded[step] = (
            BalancedStepTensorDataset(
                stokes_input=stokes_input,
                mhd_targets=mhd_targets,
                spatial_indices=spatial_indices,
            ),
            approx_data,
        )
    return loaded


def initialize_wfa_gate_state(config: TrainingConfig) -> dict[str, Any]:
    """Create runtime state for train-time WFA activation gate."""
    gate_mode = str(config.wfa_gate_mode).lower()
    enabled = gate_mode == 'off' or config.lambda_wfa <= 0
    return {
        'mode': gate_mode,
        'enabled': enabled,
        'best_metric': None,
        'plateau_epochs': 0,
        'last_metric': None,
        'trigger_epoch': None,
        'trigger_reason': 'always_on' if enabled and gate_mode == 'off' else None,
    }


def update_wfa_gate_state(
    gate_state: dict[str, Any],
    config: TrainingConfig,
    epoch: int,
    epoch_mse_loss: float,
) -> tuple[dict[str, Any], bool, str | None]:
    """Update train-time WFA gate using epoch train MSE and return transition info."""
    gate_mode = str(gate_state.get('mode', config.wfa_gate_mode)).lower()
    if gate_mode == 'off' or config.lambda_wfa <= 0:
        gate_state['enabled'] = True
        gate_state['last_metric'] = float(epoch_mse_loss)
        return gate_state, False, None

    if bool(gate_state.get('enabled', False)):
        gate_state['last_metric'] = float(epoch_mse_loss)
        best_metric = gate_state.get('best_metric')
        if best_metric is None or epoch_mse_loss < float(best_metric):
            gate_state['best_metric'] = float(epoch_mse_loss)
        return gate_state, False, None

    gate_state['last_metric'] = float(epoch_mse_loss)
    current_epoch = epoch + 1
    if current_epoch <= config.wfa_gate_warmup_epochs:
        best_metric = gate_state.get('best_metric')
        if best_metric is None or epoch_mse_loss < float(best_metric):
            gate_state['best_metric'] = float(epoch_mse_loss)
        return gate_state, False, None

    trigger_reason = None
    if gate_mode == 'threshold':
        if epoch_mse_loss <= config.wfa_gate_threshold:
            trigger_reason = f"threshold(train_mse={epoch_mse_loss:.6f} <= {config.wfa_gate_threshold:.6f})"
    elif gate_mode == 'plateau':
        best_metric = gate_state.get('best_metric')
        if best_metric is None:
            gate_state['best_metric'] = float(epoch_mse_loss)
        else:
            improvement = float(best_metric) - float(epoch_mse_loss)
            if improvement > config.wfa_gate_min_delta:
                gate_state['best_metric'] = float(epoch_mse_loss)
                gate_state['plateau_epochs'] = 0
            else:
                gate_state['plateau_epochs'] = int(gate_state.get('plateau_epochs') or 0) + 1

            if int(gate_state.get('plateau_epochs') or 0) >= config.wfa_gate_patience:
                trigger_reason = (
                    f"plateau(train_mse={epoch_mse_loss:.6f}, patience={config.wfa_gate_patience}, "
                    f"min_delta={config.wfa_gate_min_delta:.6f})"
                )
    else:
        raise ValueError(f"Unsupported wfa_gate_mode: {gate_mode}")

    if trigger_reason is not None:
        gate_state['enabled'] = True
        gate_state['trigger_epoch'] = current_epoch
        gate_state['trigger_reason'] = trigger_reason
        return gate_state, True, trigger_reason

    return gate_state, False, None

def load_source_arrays(
    step: int,
    config: TrainingConfig,
    ignore_missing_files: bool = False,
) -> tuple[StokesData, dict[str, np.ndarray]] | None:
    """
    Load raw, per-step MHD + Stokes arrays for `config.data_source`.

    Returns a (stokes, mhd_data) pair with the same downstream contract
    regardless of source: `stokes` is a StokesData instance with `.data`
    (fine wavelength grid, I/Q/U/V, pre-LSF/resample) and `.mean_continuum`
    populated; `mhd_data` maps {'T', 'Vz', 'Bz'} to (nx, ny, n_logtau)
    Quantities on config.get_logtau_values(). Callers still need to run
    load_hinode_lsf/apply_spectral_convolution/resample_to_hinode/
    spectropolarimetry on the returned `stokes` -- those steps are
    source-agnostic and are not duplicated here.

    Returns None (instead of raising) when required files are missing and
    ignore_missing_files=True.
    """
    new_logtau = config.get_logtau_values()

    if config.data_source == "muram_legacy":
        mhd = MhdData(
            data_path=config.data_path / "muram-simulation",
            nx=config.nx, ny=config.ny, nz=config.nz
        )
        try:
            mhd.load_step(step=step, z_max=config.z_max)
        except FileNotFoundError as exc:
            if ignore_missing_files:
                print(f"  ⚠ Skipping step {step} because required files are missing: {exc}")
                return None
            raise
        mhd.load_opacity_table(kappa_path=config.data_path / config.kappa_path)
        mhd.compute_optical_depth(dz=config.dz_km * u.km)
        mhd.remap_to_optical_depth(new_logtau, quantities=["T", "Vz", "Bz"])

        stokes = StokesData(
            data_dir=config.data_path / "muram-simulation/",
            step=step,
            wavelength_range=(6300.5, 6303.5),
            wavelength_step=0.01
        )
        try:
            stokes.load_stokes()
        except FileNotFoundError as exc:
            if ignore_missing_files:
                print(f"  ⚠ Skipping step {step} because required files are missing: {exc}")
                return None
            raise
        stokes_cont_indices = config.stokes_cont_indices or [0, 1, 2, 3]
        if config.stokes_ic_mode == "fixed_global":
            if config.stokes_fixed_ic is None:
                raise ValueError("stokes_fixed_ic must be set for fixed_global mode")
            fixed_ic = float(config.stokes_fixed_ic)
        else:
            fixed_ic = None
        stokes.continuum_normalization(cont_indices=stokes_cont_indices, fixed_ic=fixed_ic)
        if config.stokes_mult_factor != 1.0:
            stokes.data["I"] = stokes.data["I"] * config.stokes_mult_factor
            stokes.data["V"] = stokes.data["V"] * config.stokes_mult_factor

        return stokes, mhd.od_data

    elif config.data_source == "nicole_tau500":
        sim_dir = config.data_path / "muram-simulation"
        stokes_path = sim_dir / f"stokes_{step}_nicole_tau500.npy"
        atmos_path = sim_dir / f"atmos_{step}_tau500.npz"
        if not stokes_path.exists() or not atmos_path.exists():
            missing = stokes_path if not stokes_path.exists() else atmos_path
            if ignore_missing_files:
                print(f"  ⚠ Skipping step {step} because required files are missing: {missing}")
                return None
            raise FileNotFoundError(
                f"nicole_tau500 data not found for step {step}: expected {stokes_path} and {atmos_path}"
            )

        atmos = np.load(atmos_path)
        saved_logtau = np.round(np.asarray(atmos["logtau"], dtype=np.float32), 6)
        if saved_logtau.shape != new_logtau.shape or not np.allclose(saved_logtau, new_logtau, atol=1e-6):
            raise ValueError(
                f"atmos_{step}_tau500.npz was generated on a different log(tau) grid than the "
                f"active config. Saved: {saved_logtau.tolist()} | Requested: {new_logtau.tolist()}. "
                "Set logtau_min=-3.0, logtau_max=1.4, logtau_step=0.1 (the tau500-generation grid) "
                "to use this data source."
            )

        mhd_data = {
            "T": np.asarray(atmos["T"], dtype=np.float64) * u.K,
            "Vz": np.asarray(atmos["Vz"], dtype=np.float64) * u.km / u.s,
            "Bz": np.asarray(atmos["Bz"], dtype=np.float64) * u.G,
        }

        # Stokes cube is (nx, ny, nwl, 4) = I,Q,U,V, already NICOLE-normalized
        # (Continuum reference=1) -- fixed_ic / stokes_mult_factor do not apply.
        stokes_cube = np.load(stokes_path)
        stokes = StokesData(
            data_dir=sim_dir,
            step=step,
            wavelength_range=(6300.5, 6303.5),
            wavelength_step=0.01,
        )
        stokes.data = {
            "I": stokes_cube[:, :, :, 0],
            "Q": stokes_cube[:, :, :, 1],
            "U": stokes_cube[:, :, :, 2],
            "V": stokes_cube[:, :, :, 3],
        }
        stokes.nx, stokes.ny, stokes.nwl = stokes.data["I"].shape
        stokes_cont_indices = config.stokes_cont_indices or [0, 1, 2, 3]
        stokes.mean_continuum = stokes.data["I"][:, :, stokes_cont_indices].mean(axis=2)

        return stokes, mhd_data

    raise ValueError(f"Unknown data_source: {config.data_source!r}")


def load_and_prepare_step(
    step: int,
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    cache: MuramDataCache | None = None,
    apply_balanced_masks: bool = False,
    log_region_stats: bool = False,
    apply_bz_balance: bool = False,
    global_bz_selection_indices: dict[int, np.ndarray] | None = None,
    global_bz_balance_metadata: dict[str, Any] | None = None,
    ignore_missing_files: bool = False,
) -> tuple[MuramStepDataset, dict[str, np.ndarray]] | None:
    """
    Load and prepare a single simulation step for training.
    
    Uses cache if available to significantly speed up repeated runs.
    
    Parameters
    ----------
    step : int
        Simulation step number
    config : TrainingConfig
        Training configuration
    mhd_normalizer : MhdNormalizer
        MHD data normalizer
    stokes_normalizer : StokesNormalizer
        Stokes data normalizer
    cache : MuramDataCache, optional
        Cache manager for loading/saving processed data
    ignore_missing_files : bool
        If True, return None instead of raising when required input files are missing.
    
    Returns
    -------
    dataset : MuramStepDataset
        Dataset containing normalized inputs/targets
    approx_data : dict
        Physics approximations {'blos': (nx, ny), 'vlos': (nx, ny), 'temp': (nx, ny)}
    """
    # Compute configuration hash for cache validation
    config_for_hash = build_cache_config_signature(config)
    config_hash = MuramDataCache.make_config_hash(config_for_hash) if cache else None
    new_logtau = config.get_logtau_values()

    region_sampling_info = None
    bz_balance_info = None

    # Try to load from cache (strict first, then relaxed hash fallback)
    allow_relaxed_cache_fallback = (
        config.stokes_ic_mode == "per_step" and config.stokes_fixed_ic is None
    )

    if cache is not None:
        exact_hit = cache.exists(step, config_hash, logtau_values=new_logtau)
        relaxed_hit = False
        if not exact_hit and allow_relaxed_cache_fallback:
            try:
                relaxed_hit = cache.exists(step, None, logtau_values=new_logtau)
            except Exception:
                relaxed_hit = False

        if exact_hit or relaxed_hit:
            try:
                stokes_cached, mhd_cached, approx_cached = cache.load_raw(step=step, verbose=True)
                required_keys = {"blos", "vlos", "temp"}
                if not isinstance(approx_cached, dict) or not required_keys.issubset(approx_cached.keys()):
                    raise KeyError(f"Cache step {step} missing keys {required_keys}")

                # Enforce valid ApproxInversions-like payload
                for k in required_keys:
                    v = approx_cached[k]
                    if not isinstance(v, np.ndarray) or v.ndim != 2:
                        raise KeyError(f"Cache step {step} key '{k}' invalid (expected 2D np.ndarray)")

                # New cache contract: circular polarization must be present
                if "circular_polarization" not in stokes_cached:
                    raise KeyError(
                        f"Cache step {step} missing 'circular_polarization'. "
                        "Reprocessing with updated pipeline."
                    )
                if "hinode_wl" not in stokes_cached:
                    raise KeyError(
                        f"Cache step {step} missing 'hinode_wl'. "
                        "Reprocessing with updated pipeline."
                    )

                selected_indices = None
                if apply_balanced_masks:
                    mean_cont = stokes_cached.get("mean_continuum")
                    if mean_cont is None:
                        mean_cont = np.mean(stokes_cached["I"][:, :, :4], axis=2)

                    mask_data = build_granulation_polarization_masks(
                        mean_continuum=mean_cont,
                        circular_polarization=stokes_cached["circular_polarization"],
                    )
                    selected_indices, balance_stats = build_balanced_region_indices(mask_data["masks"])
                    region_sampling_info = {
                        "thresholds": {
                            "continuum": float(mask_data["continuum_threshold"]),
                            "circular_polarization": float(mask_data["polarization_threshold"]),
                        },
                        "counts_before": balance_stats["counts_before"],
                        "counts_after": balance_stats["counts_after"],
                    }

                if global_bz_selection_indices is not None and step in global_bz_selection_indices:
                    selected_indices = np.asarray(global_bz_selection_indices[step], dtype=np.int64)
                    step_stats = {}
                    if isinstance(global_bz_balance_metadata, dict):
                        step_stats = global_bz_balance_metadata.get("per_step_counts", {}).get(str(step), {})
                    bz_balance_info = {
                        "scope": "global",
                        "reference_tau_idx": None if not isinstance(global_bz_balance_metadata, dict) else global_bz_balance_metadata.get("reference_tau_idx"),
                        "reference_logtau": None if not isinstance(global_bz_balance_metadata, dict) else global_bz_balance_metadata.get("reference_logtau"),
                        "counts_before_step": step_stats.get("before"),
                        "counts_after_step": step_stats.get("after"),
                        "counts_after": {"total_selected": int(selected_indices.size)},
                    }
                elif apply_bz_balance:
                    selected_indices, bz_balance_info = build_bz_strength_balanced_indices(
                        mhd_data=mhd_cached,
                        base_selected_indices=selected_indices,
                        n_bins=config.bz_balance_bins,
                        score_mode=config.bz_balance_mode,
                        tau_idx=config.bz_balance_tau_idx,
                    )

                dataset_cached = MuramStepDataset(
                    stokes_data=stokes_cached,
                    mhd_data=mhd_cached,
                    stokes_normalizer=stokes_normalizer,
                    mhd_normalizer=mhd_normalizer,
                    selected_flat_indices=selected_indices,
                    region_sampling_info=region_sampling_info,
                    bz_balance_info=bz_balance_info,
                )

                if apply_balanced_masks and log_region_stats and dataset_cached.region_sampling_info is not None:
                    stats = dataset_cached.region_sampling_info
                    b = stats["counts_before"]
                    a = stats["counts_after"]
                    print(
                        f"  Region counts (step {step}) before balance: "
                        f"GS={b['granular_strong']}, IS={b['intergranular_strong']}, "
                        f"GW={b['granular_weak']}, IW={b['intergranular_weak']}"
                    )
                    print(
                        f"  Region counts (step {step}) after balance:  "
                        f"GS={a['granular_strong']}, IS={a['intergranular_strong']}, "
                        f"GW={a['granular_weak']}, IW={a['intergranular_weak']} "
                        f"(total={a['total_selected']})"
                    )
                if (apply_bz_balance or global_bz_selection_indices is not None) and config.log_bz_bin_balance_stats and dataset_cached.bz_balance_info is not None:
                    stats = dataset_cached.bz_balance_info
                    if stats.get("scope") == "global":
                        print(
                            f"  Bz bin balance (step {step}): scope=global, "
                            f"tau_idx={stats.get('reference_tau_idx')}, selected={stats['counts_after']['total_selected']}"
                        )
                    else:
                        print(
                            f"  Bz bin balance (step {step}): mode={stats['score_mode']}, bins={stats['n_bins']}, "
                            f"target/bin={stats['target_per_bin']}, selected={stats['counts_after']['total_selected']}"
                        )

                return dataset_cached, approx_cached
            except Exception as e:
                print(f"  ⚠ Cache load failed for step {step}: {e}")
                print(f"  Reprocessing step {step}...")
    
    # Check normalizer/tau-grid compatibility before doing any I/O.
    if hasattr(mhd_normalizer, "n_tau") and len(new_logtau) != mhd_normalizer.n_tau:
        raise ValueError(
            f"logtau grid has {len(new_logtau)} levels, but mhd_normalizer expects "
            f"{mhd_normalizer.n_tau}. Recompute normalizer stats or adjust logtau grid."
        )

    result = load_source_arrays(step=step, config=config, ignore_missing_files=ignore_missing_files)
    if result is None:
        return None
    stokes, mhd_od_data = result

    stokes.load_hinode_lsf(config.data_path / config.lsf_path)
    stokes.apply_spectral_convolution()
    stokes.resample_to_hinode()
    stokes.spectropolarimetry()

    # Keep derived 2D maps in stokes payload for cache/users that need region masking.
    stokes.data["mean_continuum"] = np.asarray(stokes.mean_continuum, dtype=np.float32)
    stokes.data["circular_polarization"] = np.asarray(stokes.circular_polarization, dtype=np.float32)
    stokes.data["hinode_wl"] = np.asarray(stokes.hinode_wl, dtype=np.float32)

    selected_indices = None
    if apply_balanced_masks:
        mask_data = build_granulation_polarization_masks(
            mean_continuum=stokes.data["mean_continuum"],
            circular_polarization=stokes.data["circular_polarization"],
        )
        selected_indices, balance_stats = build_balanced_region_indices(mask_data["masks"])
        region_sampling_info = {
            "thresholds": {
                "continuum": float(mask_data["continuum_threshold"]),
                "circular_polarization": float(mask_data["polarization_threshold"]),
            },
            "counts_before": balance_stats["counts_before"],
            "counts_after": balance_stats["counts_after"],
        }

    if global_bz_selection_indices is not None and step in global_bz_selection_indices:
        selected_indices = np.asarray(global_bz_selection_indices[step], dtype=np.int64)
        step_stats = {}
        if isinstance(global_bz_balance_metadata, dict):
            step_stats = global_bz_balance_metadata.get("per_step_counts", {}).get(str(step), {})
        bz_balance_info = {
            "scope": "global",
            "reference_tau_idx": None if not isinstance(global_bz_balance_metadata, dict) else global_bz_balance_metadata.get("reference_tau_idx"),
            "reference_logtau": None if not isinstance(global_bz_balance_metadata, dict) else global_bz_balance_metadata.get("reference_logtau"),
            "counts_before_step": step_stats.get("before"),
            "counts_after_step": step_stats.get("after"),
            "counts_after": {"total_selected": int(selected_indices.size)},
        }
    elif apply_bz_balance:
        selected_indices, bz_balance_info = build_bz_strength_balanced_indices(
            mhd_data=mhd_od_data,
            base_selected_indices=selected_indices,
            n_bins=config.bz_balance_bins,
            score_mode=config.bz_balance_mode,
            tau_idx=config.bz_balance_tau_idx,
        )

    # Create dataset
    dataset = MuramStepDataset(
        stokes_data=stokes.data,
        mhd_data=mhd_od_data,
        stokes_normalizer=stokes_normalizer,
        mhd_normalizer=mhd_normalizer,
        selected_flat_indices=selected_indices,
        region_sampling_info=region_sampling_info,
        bz_balance_info=bz_balance_info,
    )

    if apply_balanced_masks and log_region_stats and dataset.region_sampling_info is not None:
        stats = dataset.region_sampling_info
        b = stats["counts_before"]
        a = stats["counts_after"]
        print(
            f"  Region counts (step {step}) before balance: "
            f"GS={b['granular_strong']}, IS={b['intergranular_strong']}, "
            f"GW={b['granular_weak']}, IW={b['intergranular_weak']}"
        )
        print(
            f"  Region counts (step {step}) after balance:  "
            f"GS={a['granular_strong']}, IS={a['intergranular_strong']}, "
            f"GW={a['granular_weak']}, IW={a['intergranular_weak']} "
            f"(total={a['total_selected']})"
        )
    if (apply_bz_balance or global_bz_selection_indices is not None) and config.log_bz_bin_balance_stats and dataset.bz_balance_info is not None:
        stats = dataset.bz_balance_info
        if stats.get("scope") == "global":
            print(
                f"  Bz bin balance (step {step}): scope=global, "
                f"tau_idx={stats.get('reference_tau_idx')}, selected={stats['counts_after']['total_selected']}"
            )
        else:
            print(
                f"  Bz bin balance (step {step}): mode={stats['score_mode']}, bins={stats['n_bins']}, "
                f"target/bin={stats['target_per_bin']}, selected={stats['counts_after']['total_selected']}"
            )
    
    # Compute physics approximations (unnormalized)
    inv = ApproxInversions(
        stokes=stokes.data,
        wavelength=stokes.wl,
        central_wavelength=config.central_wavelength * u.Angstrom,
        lande_factor=config.lande_factor
    )
    
    blos_approx = inv.compute_blos_wfa(wl_range=config.wl_range).value  # (nx, ny)
    vlos_approx = inv.compute_vlos_doppler(wl_range=config.wl_range).value  # (nx, ny)
    
    # Compute temperature approximation using blackbody method
    temp_approx = inv.compute_temperature_blackbody(
        cont_indices=config.temp_continuum_indices,
        reference_temperature=config.temp_reference_temperature * u.K,
        continuum_wavelength=config.temp_continuum_wavelength * u.Angstrom
    ).value  # (nx, ny)
    
    approx_data = {
        'blos': blos_approx,
        'vlos': vlos_approx,
        'temp': temp_approx,
    }
    
    # Save to cache if enabled
    if cache is not None:
        try:
            cache.save(
                step=step,
                stokes_data=stokes.data,
                mhd_data=mhd_od_data,
                approx_data=approx_data,
                config_hash=config_hash,
                logtau_values=new_logtau,
                verbose=True,
            )
        except Exception as e:
            print(f"  ⚠ Failed to save cache for step {step}: {e}")
    
    return dataset, approx_data


def compute_global_bz_balancing_indices(
    train_steps: list[int],
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    cache: MuramDataCache | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Build global Bz-bin balancing indices across all training steps.

    The balancing score is |Bz| evaluated at the deepest optical-depth level
    unless bz_balance_tau_idx is explicitly provided.
    """
    if not config.apply_bz_bin_balance:
        return {}, {}

    n_tau = int(config.get_n_logtau())
    ref_tau_idx = int(config.bz_balance_tau_idx) if config.bz_balance_tau_idx is not None else int(n_tau - 1)
    if ref_tau_idx < 0 or ref_tau_idx >= n_tau:
        raise ValueError(f"bz_balance_tau_idx must be within [0, {n_tau - 1}], got {ref_tau_idx}")

    rng = np.random.default_rng(config.bz_balance_seed)

    all_scores: list[np.ndarray] = []
    all_step_ids: list[np.ndarray] = []
    all_flat_idx: list[np.ndarray] = []
    per_step_before: dict[str, int] = {}

    print("\nPrecomputing global Bz balancing indices from ready-for-training data...")
    for step in tqdm(train_steps, desc="Global Bz balance scan"):
        result = load_and_prepare_step(
            step=step,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            cache=cache,
            apply_balanced_masks=config.apply_region_mask,
            log_region_stats=False,
            apply_bz_balance=False,
            global_bz_selection_indices=None,
            global_bz_balance_metadata=None,
            ignore_missing_files=True,
        )

        if result is None:
            per_step_before[str(step)] = 0
            continue

        dataset, _ = result

        targets = np.asarray(dataset.mhd_targets, dtype=np.float32)
        bz_norm = targets[:, 2 * n_tau:3 * n_tau]
        bz_denorm = np.asarray(mhd_normalizer.denormalize(bz_norm, param="Bz"), dtype=np.float32)
        scores = np.abs(bz_denorm[:, ref_tau_idx])

        flat_idx = (
            dataset.spatial_indices[:, 0].astype(np.int64) * int(dataset.ny)
            + dataset.spatial_indices[:, 1].astype(np.int64)
        )

        finite_mask = np.isfinite(scores)
        scores = scores[finite_mask]
        flat_idx = flat_idx[finite_mask]

        per_step_before[str(step)] = int(flat_idx.size)
        if flat_idx.size == 0:
            continue

        all_scores.append(scores.astype(np.float32, copy=False))
        all_flat_idx.append(flat_idx.astype(np.int64, copy=False))
        all_step_ids.append(np.full(flat_idx.shape[0], int(step), dtype=np.int64))

    if len(all_scores) == 0:
        print("⚠ Global Bz balancing found no candidate pixels across train steps; continuing without global balancing.")
        return {}, {
            "reference_tau_idx": int(ref_tau_idx),
            "reference_logtau": float(config.get_logtau_values()[ref_tau_idx]),
            "counts_before": {},
            "counts_after": {},
            "per_step_counts": {step: {"before": count, "after": 0} for step, count in per_step_before.items()},
            "n_selected": 0,
            "skipped": True,
        }

    scores_global = np.concatenate(all_scores, axis=0)
    step_ids_global = np.concatenate(all_step_ids, axis=0)
    flat_idx_global = np.concatenate(all_flat_idx, axis=0)

    score_min = float(np.min(scores_global))
    score_max = float(np.max(scores_global))
    n_bins = int(max(2, config.bz_balance_bins))

    if np.isclose(score_min, score_max):
        selected_positions = np.arange(scores_global.size, dtype=np.int64)
        bin_edges = np.array([score_min, score_max], dtype=np.float32)
        counts_before = {"bin_0": int(scores_global.size)}
        target_per_bin = int(scores_global.size)
    else:
        bin_edges = np.linspace(score_min, score_max, n_bins + 1, dtype=np.float32)
        bin_ids = np.digitize(scores_global, bin_edges[1:-1], right=False)
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

        counts_before = {
            f"bin_{bin_idx}": int(np.sum(bin_ids == bin_idx))
            for bin_idx in range(n_bins)
        }
        occupied = [c for c in counts_before.values() if c > 0]
        if not occupied:
            raise RuntimeError("Global Bz balancing found no occupied bins.")
        target_per_bin = int(min(occupied))

        selected_chunks = []
        for bin_idx in range(n_bins):
            idx_bin = np.flatnonzero(bin_ids == bin_idx)
            if idx_bin.size == 0:
                continue
            if idx_bin.size > target_per_bin:
                chosen = rng.choice(idx_bin, size=target_per_bin, replace=False)
            else:
                chosen = idx_bin
            selected_chunks.append(chosen.astype(np.int64, copy=False))

        selected_positions = np.concatenate(selected_chunks, axis=0)
        rng.shuffle(selected_positions)

    selected_by_step: dict[int, np.ndarray] = {}
    per_step_after: dict[str, int] = {}
    for step in train_steps:
        step_mask = step_ids_global[selected_positions] == int(step)
        step_selected = flat_idx_global[selected_positions][step_mask]
        if step_selected.size == 0:
            # Keep training robust: if a step receives no selected pixels, fallback to one candidate.
            fallback = flat_idx_global[step_ids_global == int(step)]
            if fallback.size > 0:
                step_selected = fallback[:1]
        selected_by_step[int(step)] = np.asarray(step_selected, dtype=np.int64)
        per_step_after[str(step)] = int(step_selected.size)

    counts_after_total = int(sum(per_step_after.values()))
    metadata = {
        "scope": "global",
        "reference_tau_idx": int(ref_tau_idx),
        "reference_logtau": float(config.get_logtau_values()[ref_tau_idx]),
        "score_min": score_min,
        "score_max": score_max,
        "n_bins": int(len(bin_edges) - 1),
        "bin_edges": [float(v) for v in bin_edges.tolist()],
        "counts_before": counts_before,
        "counts_after": {"total_selected": counts_after_total},
        "target_per_bin": int(target_per_bin),
        "per_step_counts": {
            str(step): {
                "before": int(per_step_before.get(str(step), 0)),
                "after": int(per_step_after.get(str(step), 0)),
            }
            for step in train_steps
        },
        "seed": int(config.bz_balance_seed),
    }

    print(
        "  ✓ Global Bz balancing ready: "
        f"tau_idx={metadata['reference_tau_idx']} (logtau={metadata['reference_logtau']:.3f}), "
        f"bins={metadata['n_bins']}, target/bin={metadata['target_per_bin']}, "
        f"total_selected={metadata['counts_after']['total_selected']}"
    )

    return selected_by_step, metadata

def train_one_step(
    model: PhysicsInformedMSCNN,
    dataloader: DataLoader,
    approx_data: dict[str, np.ndarray],
    mhd_normalizer: MhdNormalizer,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    step_num: int,
    logger: MetricsLogger | None,
    enable_wfa: bool = True,
) -> dict[str, float]:
    """
    Train on one simulation step (one epoch through that step's data).
    
    Parameters
    ----------
    model : PhysicsInformedMSCNN
        Model with integrated physics computation
    dataloader : DataLoader
        DataLoader for this step
    approx_data : Dict[str, np.ndarray]
        Physics approximations for this step (blos, vlos, temp)
    mhd_normalizer : MhdNormalizer
        Normalizer for denormalization
    optimizer : torch.optim.Optimizer
        Optimizer
    config : TrainingConfig
        Training configuration
    epoch : int
        Current epoch number
    step_num : int
        Current simulation step number
    logger : MetricsLogger
        Metrics logger
    Returns
    -------
    step_metrics : Dict[str, float]
        Dictionary containing average loss components for this step
    """
    model.train()
    
    # Set physics context once for this step (including temperature)
    model.set_physics_context(
        mhd_normalizer=mhd_normalizer,
        logtau_values=config.get_logtau_values(),
        blos_approx=approx_data.get('blos'),
        vlos_approx=approx_data.get('vlos'),
        temp_approx=approx_data.get('temp'),
    )
    
    # Initialize accumulators for all loss components
    step_metrics = {
        'total_loss': 0.0,
        'mse_loss': 0.0,
        'physics_loss': 0.0,
        'wfa_loss': 0.0,
        'doppler_loss': 0.0,
        'temperature_loss': 0.0,
    }
    
    n_batches = 0
    
    for batch_idx, (stokes_batch, mhd_batch, spatial_idx_batch) in enumerate(dataloader):
        # Move to device
        stokes_batch = stokes_batch.to(config.device)
        mhd_batch = mhd_batch.to(config.device)
        spatial_idx_batch = spatial_idx_batch.to(config.device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(stokes_batch)
        
        loss_dict = model.compute_loss(
            predictions=predictions,
            targets=mhd_batch,
            spatial_indices=spatial_idx_batch,
            enable_wfa=enable_wfa,
        )
        
        total_loss = loss_dict['loss']
        
        # Backward pass
        total_loss.backward()
        
        # Accumulate loss components
        step_metrics['mse_loss'] += loss_dict['mse'].item()
        step_metrics['physics_loss'] += loss_dict['physics'].item()
        step_metrics['wfa_loss'] += float(loss_dict.get('wfa', 0.0))
        step_metrics['doppler_loss'] += float(loss_dict.get('doppler', 0.0))
        step_metrics['temperature_loss'] += float(loss_dict.get('temperature', 0.0))
        
        # Gradient clipping
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        
        step_metrics['total_loss'] += total_loss.item()
        n_batches += 1
    
    # Average metrics over all batches
    for key in step_metrics.keys():
        step_metrics[key] /= n_batches

    # Mark physics fields as NaN when WFA gate is closed so the CSV reflects
    # that no physics constraint was active (rather than a misleading 0.0).
    if not enable_wfa:
        for key in ('physics_loss', 'wfa_loss', 'doppler_loss', 'temperature_loss'):
            step_metrics[key] = float('nan')

    # Log metrics
    if logger is not None:
        logger.log_batch(epoch=epoch, step=step_num, batch=0, loss_dict=step_metrics)
    
    return step_metrics

def validate(
    model: PhysicsInformedMSCNN,
    val_steps: list[int],
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    cache: MuramDataCache | None = None,
) -> float:
    """
    Validate on a subset of steps.
    
    Parameters
    ----------
    model : PhysicsInformedMSCNN
        Model to validate
    val_steps : List[int]
        Validation step numbers
    config : TrainingConfig
        Training configuration
    mhd_normalizer : MhdNormalizer
        MHD normalizer
    stokes_normalizer : StokesNormalizer
        Stokes normalizer
    cache : MuramDataCache, optional
        Cache manager for data loading
    
    Returns
    -------
    avg_val_loss : float
        Average validation loss across all validation steps
    """
    model.eval()
    n_val_samples = 0
    total_val_loss = 0.0

    with torch.no_grad():
        for step in val_steps:
            try:
                result = load_and_prepare_step(
                    step=step,
                    config=config,
                    mhd_normalizer=mhd_normalizer,
                    stokes_normalizer=stokes_normalizer,
                    cache=cache,
                    ignore_missing_files=True,
                )

                if result is None:
                    continue

                dataset, approx_data = result
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )
                
                # Set physics context for validation (including temperature)
                model.set_physics_context(
                    mhd_normalizer=mhd_normalizer,
                    logtau_values=config.get_logtau_values(),
                    blos_approx=approx_data.get('blos'),
                    vlos_approx=approx_data.get('vlos'),
                    temp_approx=approx_data.get('temp'),
                )
                
                for stokes_batch, mhd_batch, spatial_idx_batch in dataloader:
                    stokes_batch = stokes_batch.to(config.device)
                    mhd_batch = mhd_batch.to(config.device)
                    spatial_idx_batch = spatial_idx_batch.to(config.device)
                    
                    predictions = model(stokes_batch)
                    
                    loss_dict = model.compute_loss(
                        predictions=predictions,
                        targets=mhd_batch,
                        spatial_indices=spatial_idx_batch,
                        enable_wfa=True,
                    )
                    
                    total_loss = loss_dict['loss']
                    total_val_loss += total_loss.item() * stokes_batch.size(0)
                    n_val_samples += stokes_batch.size(0)
            
            except Exception as e:
                print(f"  Warning: Failed to validate on step {step}: {e}")
                continue
    
    if n_val_samples == 0:
        print("  Warning: validation found no usable steps; returning NaN.")
        return float("nan")

    return total_val_loss / n_val_samples

def save_checkpoint(
    model: PhysicsInformedMSCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: TrainingConfig,
    wfa_gate_state: dict[str, Any] | None = None,
    is_best: bool = False,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': asdict(config),
        'wfa_gate_state': wfa_gate_state,
    }
    
    # Save regular checkpoint
    checkpoint_path = config.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = config.checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"  Saved best model: {best_path}")

def load_checkpoint(
    checkpoint_path: Path,
    model: PhysicsInformedMSCNN,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float, float, dict[str, Any] | None]:
    """
    Load training checkpoint.
    
    Returns
    -------
    start_epoch : int
    train_loss : float
    val_loss : float
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint.get('train_loss', 0.0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    wfa_gate_state = checkpoint.get('wfa_gate_state')
    
    print(f"  Resumed from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
    if isinstance(wfa_gate_state, dict):
        print(
            "  WFA gate state: "
            f"enabled={wfa_gate_state.get('enabled')}, "
            f"mode={wfa_gate_state.get('mode')}, "
            f"trigger_epoch={wfa_gate_state.get('trigger_epoch')}"
        )
    
    return start_epoch, train_loss, val_loss, wfa_gate_state

def train_epoch(
    model: PhysicsInformedMSCNN,
    train_steps: list[int],
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    logger: MetricsLogger | None = None,
    n_steps_per_epoch: int = -1,
    cache: MuramDataCache | None = None,
    enable_wfa: bool = True,
    global_bz_selection_indices: dict[int, np.ndarray] | None = None,
    global_bz_balance_metadata: dict[str, Any] | None = None,
    balanced_cache: BalancedTrainDataCache | None = None,
    balanced_cache_signature_hash: str | None = None,
    preloaded_balanced_steps: dict[int, tuple[BalancedStepTensorDataset, dict[str, np.ndarray]]] | None = None,
) -> dict[str, float]:
    """
    Train for one epoch across multiple simulation steps.
    
    Parameters
    ----------
    model : PhysicsInformedMSCNN
        Model to train
    train_steps : List[int]
        List of simulation steps to use for training
    config : TrainingConfig
        Training configuration
    mhd_normalizer : MhdNormalizer
        MHD data normalizer
    stokes_normalizer : StokesNormalizer
        Stokes data normalizer
    optimizer : torch.optim.Optimizer
        Optimizer
    epoch : int
        Current epoch number (for logging)
    logger : MetricsLogger, optional
        Logger for batch-level metrics
    n_steps_per_epoch : int, optional
        Maximum number of steps to use per epoch (-1 for all steps)
    cache : MuramDataCache, optional
        Cache manager for data loading
    
    Returns
    -------
    epoch_metrics : Dict[str, float]
        Aggregated metrics for the epoch
    """
    model.train()
    
    # Shuffle and limit steps
    steps_to_use = train_steps.copy()
    random.shuffle(steps_to_use)
    if n_steps_per_epoch > 0:
        steps_to_use = steps_to_use[:n_steps_per_epoch]
    
    # Initialize metrics (including temperature)
    epoch_metrics = {
        'total_loss': 0.0,
        'mse_loss': 0.0,
        'physics_loss': 0.0,
        'wfa_loss': 0.0,
        'doppler_loss': 0.0,
        'temperature_loss': 0.0,
        'n_steps': 0,
        'n_pixels_used': 0,
    }
    
    # Progress bar
    step_pbar = tqdm(steps_to_use, desc=f"Epoch {epoch + 1}", unit="step", leave=False)
    
    for step in step_pbar:
        try:
            if preloaded_balanced_steps is not None:
                preloaded = preloaded_balanced_steps.get(step)
                if preloaded is None:
                    continue
                dataset, approx_data = preloaded
            elif balanced_cache is not None and balanced_cache_signature_hash is not None:
                stokes_input, mhd_targets, spatial_indices, approx_data = balanced_cache.load_step(
                    step=step,
                    signature_hash=balanced_cache_signature_hash,
                )
                dataset = BalancedStepTensorDataset(
                    stokes_input=stokes_input,
                    mhd_targets=mhd_targets,
                    spatial_indices=spatial_indices,
                )
            else:
                # Load and prepare step (uses raw cache if available)
                result = load_and_prepare_step(
                    step=step,
                    config=config,
                    mhd_normalizer=mhd_normalizer,
                    stokes_normalizer=stokes_normalizer,
                    cache=cache,
                    apply_balanced_masks=config.apply_region_mask,
                    log_region_stats=(config.apply_region_mask and config.log_region_mask_stats),
                    apply_bz_balance=(config.apply_bz_bin_balance and config.bz_balance_scope == "per_step"),
                    global_bz_selection_indices=global_bz_selection_indices,
                    global_bz_balance_metadata=global_bz_balance_metadata,
                    ignore_missing_files=True,
                )

                if result is None:
                    continue

                dataset, approx_data = result
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=False,
            )
            
            # Train on this step and get all loss components
            step_metrics = train_one_step(
                model=model,
                dataloader=dataloader,
                approx_data=approx_data,
                mhd_normalizer=mhd_normalizer,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                step_num=step,
                logger=logger,
                enable_wfa=enable_wfa,
            )
            
            # Accumulate step metrics (including temperature)
            epoch_metrics['total_loss'] += step_metrics['total_loss']
            epoch_metrics['mse_loss'] += step_metrics['mse_loss']
            epoch_metrics['physics_loss'] += step_metrics['physics_loss']
            epoch_metrics['wfa_loss'] += step_metrics['wfa_loss']
            epoch_metrics['doppler_loss'] += step_metrics['doppler_loss']
            epoch_metrics['temperature_loss'] += step_metrics['temperature_loss']
            epoch_metrics['n_steps'] += 1
            epoch_metrics['n_pixels_used'] += int(len(dataset))
            
            # Update progress bar
            step_pbar.set_postfix({
                'loss': f'{step_metrics["total_loss"]:.6f}',
            })
            
            # Clean up
            del dataset, dataloader
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n  Error processing step {step}: {e}")
            continue
    
    if epoch_metrics['n_steps'] == 0:
        raise RuntimeError("No usable training steps were found after skipping missing files.")

    # Compute averages
    n_steps = epoch_metrics['n_steps']
    if n_steps > 0:
        for key in epoch_metrics:
            if key not in {'n_steps', 'n_pixels_used'}:
                epoch_metrics[key] /= n_steps
    
    return epoch_metrics

def generate_epoch_diagnostic_plots(
    model: PhysicsInformedMSCNN,
    epoch: int,
    step: int,
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    cache: MuramDataCache | None = None,
) -> None:
    """
    Save per-epoch image + jointplot diagnostics for one monitoring step.
    """
    logtau = config.get_logtau_values()
    ods = config.epoch_plot_ods if config.epoch_plot_ods is not None else [-1.0, -0.8, 0.0]
    params = config.epoch_plot_params if config.epoch_plot_params is not None else ["T", "Vz", "Bz"]
    n_sample = int(config.epoch_plot_scatter_samples)

    # Colormaps aligned with utils/analysis_functions.py
    param_cmaps = {"T": "hot", "Vz": "bwr_r", "Bz": "PiYG"}
    error_cmap = "RdBu_r"

    base_out_dir = config.log_dir / "epoch_diagnostics" / f"step_{step}"
    out_dir = base_out_dir / f"epoch_{epoch+1:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, float | int | str]] = []

    was_training = model.training
    model.eval()

    result = load_and_prepare_step(
        step=step,
        config=config,
        mhd_normalizer=mhd_normalizer,
        stokes_normalizer=stokes_normalizer,
        cache=cache,
        ignore_missing_files=True,
    )

    if result is None:
        if was_training:
            model.train()
        print(f"  Warning: skipping diagnostics for missing step {step}.")
        return

    dataset, _ = result

    n_pixels = dataset.stokes_input.shape[0]
    all_pred = []
    with torch.no_grad():
        for i in range(0, n_pixels, config.batch_size):
            x = torch.from_numpy(dataset.stokes_input[i:i + config.batch_size]).float().to(config.device)
            y = model(x).detach().cpu().numpy()
            all_pred.append(y)

    pred_norm = np.concatenate(all_pred, axis=0)
    gt_norm = dataset.mhd_targets

    # Loop-based denormalization pipeline (same style as other scripts)
    n_tau = int(len(logtau))

    if pred_norm.ndim != 2 or pred_norm.shape[1] != 3 * n_tau:
        raise ValueError(
            f"Expected pred_norm shape (N, {3 * n_tau}), got {pred_norm.shape}"
        )
    if gt_norm.ndim != 2 or gt_norm.shape[1] != 3 * n_tau:
        raise ValueError(
            f"Expected gt_norm shape (N, {3 * n_tau}), got {gt_norm.shape}"
        )

    pred_split = {
        "T": pred_norm[:, :n_tau],
        "Vz": pred_norm[:, n_tau:2 * n_tau],
        "Bz": pred_norm[:, 2 * n_tau:3 * n_tau],
    }
    gt_split = {
        "T": gt_norm[:, :n_tau],
        "Vz": gt_norm[:, n_tau:2 * n_tau],
        "Bz": gt_norm[:, 2 * n_tau:3 * n_tau],
    }

    pred_den = {}
    gt_den = {}
    nx, ny = dataset.nx, dataset.ny

    for param_name in ["T", "Vz", "Bz"]:
        pred_param_norm = pred_split[param_name]
        gt_param_norm = gt_split[param_name]

        pred_param_den = mhd_normalizer.denormalize(pred_param_norm, param=param_name)
        gt_param_den = mhd_normalizer.denormalize(gt_param_norm, param=param_name)

        pred_den[param_name] = pred_param_den.reshape(nx, ny, n_tau)
        gt_den[param_name] = gt_param_den.reshape(nx, ny, n_tau)

    for od in ods:
        tau_idx = int(np.argmin(np.abs(logtau - od)))
        od_eff = float(logtau[tau_idx])

        for p in params:
            true_map = gt_den[p][:, :, tau_idx]
            pred_map = pred_den[p][:, :, tau_idx]
            err_map = pred_map - true_map

            x_all = true_map.ravel()
            y_all = pred_map.ravel()
            valid = np.isfinite(x_all) & np.isfinite(y_all)
            x_all = x_all[valid]
            y_all = y_all[valid]
            if x_all.size == 0:
                continue

            rmse = float(np.sqrt(np.mean((y_all - x_all) ** 2)))
            rrmse = float(rmse / (np.mean(np.abs(x_all)) + 1e-10))
            corr = float(np.corrcoef(x_all, y_all)[0, 1]) if x_all.size > 1 else float("nan")

            metrics_rows.append(
                {
                    "epoch": int(epoch + 1),
                    "step": int(step),
                    "param": str(p),
                    "logtau": float(od_eff),
                    "n_points": int(x_all.size),
                    "corr": corr,
                    "rrmse": rrmse,
                }
            )

            both = np.concatenate([true_map.ravel(), pred_map.ravel()])
            if p in ("Vz", "Bz"):
                vmax = np.nanquantile(np.abs(both), 0.99)
                vmin = -vmax
            else:
                vmin, vmax = np.nanquantile(both, [0.01, 0.99])

            emax = np.nanquantile(np.abs(err_map.ravel()), 0.99)

            param_cmap = param_cmaps.get(p, "viridis")

            # Image panel
            fig, ax = plt.subplots(1, 3, figsize=(14, 4))
            im0 = ax[0].imshow(true_map.T, origin="lower", cmap=param_cmap, vmin=vmin, vmax=vmax)
            ax[0].set_title(f"GT {p}")
            ax[0].axis("off")
            plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

            im1 = ax[1].imshow(pred_map.T, origin="lower", cmap=param_cmap, vmin=vmin, vmax=vmax)
            ax[1].set_title(f"Pred {p}")
            ax[1].axis("off")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

            im2 = ax[2].imshow(err_map.T, origin="lower", cmap=error_cmap, vmin=-emax, vmax=emax)
            ax[2].set_title(f"Error {p}")
            ax[2].axis("off")
            plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

            fig.suptitle(
                f"Epoch {epoch+1} | Step {step} | {p} @ log(tau)={od_eff:.2f} | "
                f"Corr={corr:.3f}, RRMSE={rrmse:.3f}"
            )
            fig.tight_layout()
            fig.savefig(
                out_dir / f"{p}_logtau_{od_eff:.2f}_images.png",
                dpi=170,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Jointplot (seaborn): scatter + marginal histograms
            x = x_all
            y = y_all

            if x.size > n_sample > 0:
                rng = np.random.default_rng(seed=epoch + tau_idx + 7)
                idx = rng.choice(x.size, size=n_sample, replace=False)
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
                f"Epoch {epoch+1} | Step {step} | {p} @ log(tau)={od_eff:.2f}\n"
                f"Corr={corr:.3f}, RRMSE={rrmse:.3f}",
                y=1.02
            )
            g.fig.tight_layout()
            g.fig.savefig(
                out_dir / f"{p}_logtau_{od_eff:.2f}_jointplot.png",
                dpi=170,
                bbox_inches="tight",
            )
            plt.close(g.fig)

    if metrics_rows:
        metrics_path = out_dir / "plot_metrics.csv"
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "step", "param", "logtau", "n_points", "corr", "rrmse"],
            )
            writer.writeheader()
            writer.writerows(metrics_rows)

    if was_training:
        model.train()


def _resize_map_to_shape(arr2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if arr2d.shape == target_shape:
        return arr2d
    t = torch.from_numpy(np.asarray(arr2d, dtype=np.float32)).float().unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=target_shape, mode="bilinear", align_corners=False)
    return out.squeeze(0).squeeze(0).cpu().numpy()


def generate_epoch_metric_trend_plots(
    config: TrainingConfig,
    step: int,
) -> None:
    """
    Build trend plots (Corr and RRMSE) from per-epoch diagnostic CSVs.
    """
    step_dir = config.log_dir / "epoch_diagnostics" / f"step_{step}"
    if not step_dir.exists():
        return

    trend_records: dict[tuple[str, float], list[tuple[int, float, float]]] = {}
    epoch_dirs = sorted([d for d in step_dir.glob("epoch_*") if d.is_dir()], key=lambda p: p.name)

    for e_dir in epoch_dirs:
        metrics_path = e_dir / "plot_metrics.csv"
        if not metrics_path.exists():
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epoch = int(float(row.get("epoch", "nan")))
                    param = str(row.get("param", ""))
                    logtau = float(row.get("logtau", "nan"))
                    corr = float(row.get("corr", "nan"))
                    rrmse = float(row.get("rrmse", "nan"))
                except Exception:
                    continue

                if not param or not np.isfinite(logtau):
                    continue

                key = (param, float(logtau))
                trend_records.setdefault(key, []).append((epoch, corr, rrmse))

    if not trend_records:
        return

    trends_dir = step_dir / "trends"
    trends_dir.mkdir(parents=True, exist_ok=True)

    def _save_metric_plot(metric_name: str, value_index: int, out_name: str, y_label: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        for (param, logtau), values in sorted(trend_records.items(), key=lambda item: (item[0][0], item[0][1])):
            values_sorted = sorted(values, key=lambda x: x[0])
            epochs = [v[0] for v in values_sorted]
            metric_vals = [v[value_index] for v in values_sorted]
            ax.plot(epochs, metric_vals, marker="o", linewidth=1.6, markersize=4, label=f"{param} @ {logtau:.2f}")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_label)
        ax.set_title(f"Step {step} | {metric_name} vs Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(trends_dir / out_name, dpi=170, bbox_inches="tight")
        plt.close(fig)

    _save_metric_plot(metric_name="Correlation", value_index=1, out_name="corr_vs_epoch.png", y_label="Correlation")
    _save_metric_plot(metric_name="RRMSE", value_index=2, out_name="rrmse_vs_epoch.png", y_label="RRMSE")


def _default_modest_temp_calibration_path(config: TrainingConfig) -> Path:
    if config.modest_temp_calibration_file:
        return Path(config.modest_temp_calibration_file)
    return config.log_dir / "epoch_diagnostics_modest" / "temperature_calibration.json"


def _fit_temperature_affine_per_tau(
    config: TrainingConfig,
    matches: list[tuple[float, int, int]],
    true_cube: np.ndarray,
    pred_cube: np.ndarray,
) -> dict[str, Any]:
    min_samples = int(config.modest_temp_calibration_min_samples)
    clip_q = config.modest_temp_calibration_clip_quantiles
    coefficients: dict[str, dict[str, float | int]] = {}

    for tau_val, i_mod, i_pred in matches:
        if i_mod >= true_cube.shape[2] or i_pred >= pred_cube.shape[2]:
            continue
        true_map = np.asarray(true_cube[:, :, i_mod], dtype=np.float32)
        pred_map = np.asarray(pred_cube[:, :, i_pred], dtype=np.float32)
        if true_map.shape != pred_map.shape:
            true_map = _resize_map_to_shape(true_map, pred_map.shape)

        x = true_map.ravel()
        y = pred_map.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < min_samples:
            continue

        if clip_q is not None:
            q_low, q_high = float(clip_q[0]), float(clip_q[1])
            x_lo, x_hi = np.quantile(x, [q_low, q_high])
            y_lo, y_hi = np.quantile(y, [q_low, q_high])
            m2 = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
            if np.count_nonzero(m2) >= min_samples:
                x = x[m2]
                y = y[m2]

        if x.size < min_samples:
            continue

        if np.nanstd(y) < 1e-10:
            a = 1.0
            b = float(np.nanmean(x) - np.nanmean(y))
        else:
            a, b = np.polyfit(y, x, deg=1)
            a = float(a)
            b = float(b)

        rmse_before = float(np.sqrt(np.mean((y - x) ** 2)))
        y_cal = (a * y) + b
        rmse_after = float(np.sqrt(np.mean((y_cal - x) ** 2)))
        coefficients[f"{float(tau_val):.6f}"] = {
            "tau_value": float(tau_val),
            "a": float(a),
            "b": float(b),
            "n_points": int(x.size),
            "rmse_before": rmse_before,
            "rmse_after": rmse_after,
        }

    return {
        "mode": "affine_per_tau",
        "min_samples": int(min_samples),
        "clip_quantiles": list(clip_q) if clip_q is not None else None,
        "coefficients": coefficients,
    }


def _load_temperature_calibration(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_temperature_calibration(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _find_tau_calibration_coeff(coefficients: dict[str, dict[str, Any]], tau_val: float) -> dict[str, Any] | None:
    for key, coeff in coefficients.items():
        try:
            if np.isclose(float(key), float(tau_val), atol=1e-6, rtol=0.0):
                return coeff
        except Exception:
            continue
    return None


def _apply_temperature_affine_per_tau(
    pred_cube: np.ndarray,
    matches: list[tuple[float, int, int]],
    payload: dict[str, Any],
) -> np.ndarray:
    coefficients = payload.get("coefficients", {}) if isinstance(payload, dict) else {}
    out = np.array(pred_cube, copy=True)
    for tau_val, _i_mod, i_pred in matches:
        if i_pred >= out.shape[2]:
            continue
        coeff = _find_tau_calibration_coeff(coefficients, tau_val)
        if coeff is None:
            continue
        a = float(coeff.get("a", 1.0))
        b = float(coeff.get("b", 0.0))
        out[:, :, i_pred] = (a * out[:, :, i_pred]) + b
    return out

def prepare_modest_epoch_snapshot(
    config: TrainingConfig,
    stokes_normalizer: StokesNormalizer,
) -> dict[str, object]:
    """Load and normalize one MODEST snapshot for repeated per-epoch diagnostics."""
    modest = ModestData(
        circular_polarization_threshold=config.modest_polarization_threshold,
        stokes_v_multiplier=config.modest_stokes_v_multiplier,
    )
    modest_cache = ModestDataCache(cache_dir=config.modest_cache_dir)

    print(f"MODEST cache directory: {config.modest_cache_dir}")
    if config.clear_modest_cache and not config.no_modest_cache:
        modest_cache.clear(confirm=False)
    modest_cache.print_cache_info()

    region_bounds = tuple(config.modest_crop_bounds) if config.modest_crop_bounds is not None else None
    prediction_input_mode = "downsampled" if config.modest_downsample_prediction_input else "upsampled"
    modest_data = modest.load_all(
        region_bounds=region_bounds,
        apply_mask=config.modest_polarization_mask,
        cache=modest_cache,
        use_cache=not config.no_modest_cache,
        prediction_input_mode=prediction_input_mode,
    )
    print(f"MODEST prediction input mode: {prediction_input_mode}")

    modest_logtau = list(
        modest_data.get("tau_values", sorted(modest_data["spinor_atm"]["T"].keys()))
    )

    gt_den = {
        "T": np.stack([modest_data["spinor_atm"]["T"][t] for t in modest_logtau], axis=-1).astype(np.float32),
        "Vz": np.stack([modest_data["spinor_atm"]["Vlos"][t] for t in modest_logtau], axis=-1).astype(np.float32),
        "Bz": np.stack([modest_data["spinor_atm"]["Blos"][t] for t in modest_logtau], axis=-1).astype(np.float32),
    }

    prediction_stokes = modest_data.get("prediction_stokes", modest_data["smoothed_stokes"])
    pred_nx, pred_ny = prediction_stokes["I"].shape[:2]
    cont_indices = [0, 1, 2, 3]
    I_c_modest = float(np.nanmean(prediction_stokes["I"][:, :, cont_indices]))
    if np.isfinite(I_c_modest) and I_c_modest > 10.0:
        print(f"MODEST continuum appears unnormalized (I_c={I_c_modest:.6e}); applying I/I_c scaling.")
        stokes_for_norm = {k: v / I_c_modest for k, v in prediction_stokes.items()}
    else:
        print(f"MODEST continuum appears already normalized (I_c={I_c_modest:.6e}); skipping extra I/I_c scaling.")
        stokes_for_norm = prediction_stokes

    norm_stokes = stokes_normalizer.transform(stokes_for_norm)
    I_flat = norm_stokes["I"].reshape(pred_nx * pred_ny, -1)
    V_flat = norm_stokes["V"].reshape(pred_nx * pred_ny, -1)
    stokes_input = np.stack([I_flat, V_flat], axis=1).astype(np.float32)

    return {
        "stokes_input": stokes_input,
        "pred_nx": int(pred_nx),
        "pred_ny": int(pred_ny),
        "modest_logtau": np.asarray(modest_logtau, dtype=np.float32),
        "gt_den": gt_den,
    }

def generate_epoch_modest_diagnostic_plots(
    model: PhysicsInformedMSCNN,
    epoch: int,
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    modest_snapshot: dict[str, object],
) -> None:
    """Save per-epoch image + jointplot diagnostics for one MODEST snapshot."""
    pred_logtau = config.get_logtau_values()
    modest_logtau = np.asarray(modest_snapshot["modest_logtau"], dtype=np.float32)

    # Match optical-depth nodes common to both grids
    common_matches: list[tuple[float, int, int]] = []
    for i_mod, tau_mod in enumerate(modest_logtau):
        pred_idx = np.where(np.isclose(pred_logtau, float(tau_mod), atol=1e-6, rtol=0.0))[0]
        if pred_idx.size > 0:
            common_matches.append((float(tau_mod), i_mod, int(pred_idx[0])))

    if not common_matches:
        warnings.warn("No common optical-depth nodes between MODEST and model grid; skipping MODEST epoch diagnostics.")
        return

    requested_ods = (
        config.modest_epoch_plot_ods
        if config.modest_epoch_plot_ods is not None
        else (config.epoch_plot_ods if config.epoch_plot_ods is not None else [m[0] for m in common_matches])
    )
    params = (
        config.modest_epoch_plot_params
        if config.modest_epoch_plot_params is not None
        else (config.epoch_plot_params if config.epoch_plot_params is not None else ["T", "Vz", "Bz"])
    )
    n_sample = int(
        config.modest_epoch_plot_scatter_samples
        if config.modest_epoch_plot_scatter_samples is not None
        else config.epoch_plot_scatter_samples
    )

    selected_matches: list[tuple[float, int, int]] = []
    for od in requested_ods:
        best = min(common_matches, key=lambda x: abs(x[0] - float(od)))
        if best not in selected_matches:
            selected_matches.append(best)

    param_cmaps = {"T": "hot", "Vz": "bwr_r", "Bz": "PiYG"}
    error_cmap = "RdBu_r"

    out_dir = config.log_dir / "epoch_diagnostics_modest" / f"epoch_{epoch+1:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    stokes_input = np.asarray(modest_snapshot["stokes_input"], dtype=np.float32)
    pred_nx = int(modest_snapshot["pred_nx"])
    pred_ny = int(modest_snapshot["pred_ny"])
    gt_den = modest_snapshot["gt_den"]

    n_pixels = stokes_input.shape[0]
    all_pred = []
    with torch.no_grad():
        for i in range(0, n_pixels, config.batch_size):
            x = torch.from_numpy(stokes_input[i:i + config.batch_size]).float().to(config.device)
            y = model(x).detach().cpu().numpy()
            all_pred.append(y)

    pred_norm = np.concatenate(all_pred, axis=0)
    n_tau_pred = int(len(pred_logtau))
    if pred_norm.ndim != 2 or pred_norm.shape[1] != 3 * n_tau_pred:
        raise ValueError(
            f"Expected pred_norm shape (N, {3 * n_tau_pred}), got {pred_norm.shape}"
        )

    pred_den = {
        "T": mhd_normalizer.denormalize(pred_norm[:, :n_tau_pred], param="T").reshape(pred_nx, pred_ny, n_tau_pred),
        "Vz": mhd_normalizer.denormalize(pred_norm[:, n_tau_pred:2 * n_tau_pred], param="Vz").reshape(pred_nx, pred_ny, n_tau_pred),
        "Bz": mhd_normalizer.denormalize(pred_norm[:, 2 * n_tau_pred:3 * n_tau_pred], param="Bz").reshape(pred_nx, pred_ny, n_tau_pred),
    }

    cal_mode = str(config.modest_temp_calibration_mode).lower()
    cal_path = _default_modest_temp_calibration_path(config)
    if cal_mode == "fit_only":
        fit_payload = _fit_temperature_affine_per_tau(
            config=config,
            matches=selected_matches,
            true_cube=np.asarray(gt_den["T"], dtype=np.float32),
            pred_cube=np.asarray(pred_den["T"], dtype=np.float32),
        )
        n_coeff = len(fit_payload.get("coefficients", {}))
        if n_coeff > 0:
            fit_payload["epoch"] = int(epoch + 1)
            _save_temperature_calibration(cal_path, fit_payload)
            print(f"Saved MODEST temperature calibration ({n_coeff} taus) to {cal_path}")
        else:
            warnings.warn("MODEST temperature calibration fit produced no coefficients (insufficient finite points).")
        if was_training:
            model.train()
        return

    if cal_mode == "apply_only":
        payload = _load_temperature_calibration(cal_path)
        if payload is None:
            warnings.warn(f"MODEST temperature calibration file not found: {cal_path}. Skipping MODEST epoch diagnostics.")
            if was_training:
                model.train()
            return
        pred_den["T"] = _apply_temperature_affine_per_tau(
            pred_cube=np.asarray(pred_den["T"], dtype=np.float32),
            matches=selected_matches,
            payload=payload,
        )
        print(f"Applied MODEST temperature calibration from {cal_path}")

    for tau_val, tau_idx_mod, tau_idx_pred in selected_matches:
        for p in params:
            true_map = np.asarray(gt_den[p][:, :, tau_idx_mod], dtype=np.float32)
            pred_map = np.asarray(pred_den[p][:, :, tau_idx_pred], dtype=np.float32)

            vals = (
                np.concatenate([true_map[np.isfinite(true_map)], pred_map[np.isfinite(pred_map)]])
                if (np.isfinite(true_map).any() and np.isfinite(pred_map).any())
                else np.array([0.0, 1.0])
            )
            if p in ("Vz", "Bz"):
                vmax = np.quantile(np.abs(vals), 0.99)
                vmin = -vmax
            else:
                vmin, vmax = np.quantile(vals, [0.01, 0.99])
            param_cmap = param_cmaps.get(p, "viridis")

            fig, ax = plt.subplots(1, 3, figsize=(14, 4))
            im0 = ax[0].imshow(true_map.T, origin="lower", cmap=param_cmap, vmin=vmin, vmax=vmax)
            ax[0].set_title(f"GT {p}")
            ax[0].axis("off")
            plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

            im1 = ax[1].imshow(pred_map.T, origin="lower", cmap=param_cmap, vmin=vmin, vmax=vmax)
            ax[1].set_title(f"Pred {p}")
            ax[1].axis("off")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

            if true_map.shape == pred_map.shape:
                err_map = pred_map - true_map
                emax = np.quantile(np.abs(err_map[np.isfinite(err_map)]), 0.99) if np.isfinite(err_map).any() else 1.0
                im2 = ax[2].imshow(err_map.T, origin="lower", cmap=error_cmap, vmin=-emax, vmax=emax)
                ax[2].set_title(f"Error {p}")
                ax[2].axis("off")
                plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
            else:
                ax[2].text(
                    0.5,
                    0.5,
                    f"Error skipped\nshape mismatch\nGT={true_map.shape}\nPred={pred_map.shape}",
                    ha="center",
                    va="center",
                    transform=ax[2].transAxes,
                )
                ax[2].set_axis_off()

            fig.suptitle(f"Epoch {epoch+1} | MODEST snapshot | {p} @ log(tau)={tau_val:.2f}")
            fig.tight_layout()
            fig.savefig(
                out_dir / f"{p}_logtau_{tau_val:.2f}_images.png",
                dpi=170,
                bbox_inches="tight",
            )
            plt.close(fig)

            x = true_map.ravel()
            y = pred_map.ravel()
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if x.size == 0:
                continue

            if x.size > n_sample > 0:
                rng = np.random.default_rng(seed=epoch + tau_idx_pred + 13)
                idx = rng.choice(x.size, size=n_sample, replace=False)
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
                f"Epoch {epoch+1} | MODEST snapshot | {p} @ log(tau)={tau_val:.2f}\n"
                f"Corr={corr:.3f}, RRMSE={rrmse:.3f}",
                y=1.02
            )
            g.fig.tight_layout()
            g.fig.savefig(
                out_dir / f"{p}_logtau_{tau_val:.2f}_jointplot.png",
                dpi=170,
                bbox_inches="tight",
            )
            plt.close(g.fig)

    if was_training:
        model.train()

def generate_epoch_diagnostic_videos(
    config: TrainingConfig,
    step: int | None = None,
) -> None:
    """
    Build MP4 videos from per-epoch diagnostic PNGs.
    Creates one video per plot type (param + logtau + style) for each monitored step.
    """
    try:
        import imageio.v2 as imageio
    except Exception as e:
        warnings.warn(f"Skipping epoch diagnostic videos (imageio unavailable): {e}")
        return

    base_dir = config.log_dir / "epoch_diagnostics"
    if not base_dir.exists():
        return

    step_dirs = [base_dir / f"step_{step}"] if step is not None else sorted(base_dir.glob("step_*"))
    for step_dir in step_dirs:
        if not step_dir.exists():
            continue

        epoch_dirs = sorted([d for d in step_dir.glob("epoch_*") if d.is_dir()], key=lambda p: p.name)
        if len(epoch_dirs) < 2:
            continue

        grouped: dict[str, list[Path]] = {}
        for e_dir in epoch_dirs:
            for img_path in e_dir.glob("*.png"):
                grouped.setdefault(img_path.name, []).append(img_path)

        if not grouped:
            continue

        video_dir = step_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        for img_name, img_paths in grouped.items():
            ordered_paths = sorted(img_paths, key=lambda p: p.parent.name)
            if len(ordered_paths) < 2:
                continue

            frames = []
            max_h, max_w = 0, 0
            for pth in ordered_paths:
                frame = imageio.imread(pth)

                # Ensure RGB uint8 frames
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.ndim == 3 and frame.shape[-1] == 4:
                    frame = frame[..., :3]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                frames.append(frame)
                max_h = max(max_h, frame.shape[0])
                max_w = max(max_w, frame.shape[1])

            padded_frames = []
            for fr in frames:
                h, w = fr.shape[:2]
                if h == max_h and w == max_w:
                    padded_frames.append(fr)
                    continue
                canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                y0 = (max_h - h) // 2
                x0 = (max_w - w) // 2
                canvas[y0:y0+h, x0:x0+w] = fr
                padded_frames.append(canvas)

            stem = Path(img_name).stem
            out_mp4 = video_dir / f"{stem}.mp4"

            try:
                with imageio.get_writer(
                    out_mp4,
                    format="FFMPEG",   # force ffmpeg backend (avoid tifffile)
                    mode="I",
                    fps=max(1, int(config.epoch_plot_video_fps)),
                    codec="libx264",
                    macro_block_size=16,
                ) as writer:
                    for fr in padded_frames:
                        writer.append_data(fr)
                print(f"  ✓ Video saved: {out_mp4}")
            except Exception as e:
                out_gif = video_dir / f"{stem}.gif"
                warnings.warn(
                    f"MP4 generation failed for {out_mp4.name} ({e}). "
                    f"Falling back to GIF: {out_gif.name}"
                )
                imageio.mimsave(out_gif, padded_frames, fps=max(1, int(config.epoch_plot_video_fps)))
                print(f"  ✓ GIF saved: {out_gif}")

    return

def generate_epoch_modest_diagnostic_videos(config: TrainingConfig) -> None:
    """Build MP4 videos from MODEST per-epoch diagnostic PNGs."""
    try:
        import imageio.v2 as imageio
    except Exception as e:
        warnings.warn(f"Skipping MODEST epoch diagnostic videos (imageio unavailable): {e}")
        return

    base_dir = config.log_dir / "epoch_diagnostics_modest"
    if not base_dir.exists():
        return

    epoch_dirs = sorted([d for d in base_dir.glob("epoch_*") if d.is_dir()], key=lambda p: p.name)
    if len(epoch_dirs) < 2:
        return

    grouped: dict[str, list[Path]] = {}
    for e_dir in epoch_dirs:
        for img_path in e_dir.glob("*.png"):
            grouped.setdefault(img_path.name, []).append(img_path)

    if not grouped:
        return

    video_dir = base_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    for img_name, img_paths in grouped.items():
        ordered_paths = sorted(img_paths, key=lambda p: p.parent.name)
        if len(ordered_paths) < 2:
            continue

        frames = []
        max_h, max_w = 0, 0
        for pth in ordered_paths:
            frame = imageio.imread(pth)
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.ndim == 3 and frame.shape[-1] == 4:
                frame = frame[..., :3]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            frames.append(frame)
            max_h = max(max_h, frame.shape[0])
            max_w = max(max_w, frame.shape[1])

        padded_frames = []
        for fr in frames:
            h, w = fr.shape[:2]
            if h == max_h and w == max_w:
                padded_frames.append(fr)
                continue
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            y0 = (max_h - h) // 2
            x0 = (max_w - w) // 2
            canvas[y0:y0+h, x0:x0+w] = fr
            padded_frames.append(canvas)

        stem = Path(img_name).stem
        out_mp4 = video_dir / f"{stem}.mp4"
        try:
            with imageio.get_writer(
                out_mp4,
                format="FFMPEG",
                mode="I",
                fps=max(1, int(config.epoch_plot_video_fps)),
                codec="libx264",
                macro_block_size=16,
            ) as writer:
                for fr in padded_frames:
                    writer.append_data(fr)
            print(f"  ✓ MODEST video saved: {out_mp4}")
        except Exception as e:
            out_gif = video_dir / f"{stem}.gif"
            warnings.warn(
                f"MODEST MP4 generation failed for {out_mp4.name} ({e}). "
                f"Falling back to GIF: {out_gif.name}"
            )
            imageio.mimsave(out_gif, padded_frames, fps=max(1, int(config.epoch_plot_video_fps)))
            print(f"  ✓ MODEST GIF saved: {out_gif}")

def train_pinn_model(config: TrainingConfig):
    """Main training loop with interleaved epoch training."""
    
    print("=" * 70)
    print("PINN MSCNN Training".center(70))
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Data path: {config.data_path}")
    print(f"Steps: {config.min_step} to {config.max_step} (step size: {config.step_size})")
    print(f"Epochs: {config.n_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Use cache: {config.use_cache}")
    if config.use_cache:
        print(f"Cache dir: {config.cache_dir}")
    print(f"Use balanced cache: {config.use_balanced_cache}")
    if config.use_balanced_cache:
        print(f"Balanced cache dir: {config.balanced_cache_dir}")
        print(f"Balanced cache strategy: {config.balanced_cache_strategy}")
        print(
            "Balanced cache RAM budget: "
            f"{config.balanced_cache_ram_budget_gb:.1f} GB x {config.balanced_cache_ram_fraction:.2f}"
        )
    print(f"Learning rate: {config.learning_rate}")
    print(f"Lambda WFA: {config.lambda_wfa}")
    print(f"Lambda Doppler: {config.lambda_doppler}")
    print(f"Lambda Temperature: {config.lambda_temp}")
    print(f"Apply Bz bin balance: {config.apply_bz_bin_balance}")
    if config.apply_bz_bin_balance:
        print(
            f"Bz balance scope/mode/bins: {config.bz_balance_scope}/{config.bz_balance_mode}/{config.bz_balance_bins}"
        )
        print(f"Bz balance tau idx: {config.bz_balance_tau_idx} (None -> deepest)")
    print(f"WFA gate mode: {config.wfa_gate_mode}")
    if config.wfa_gate_mode == 'threshold':
        print(f"WFA gate threshold (train MSE): {config.wfa_gate_threshold}")
    elif config.wfa_gate_mode == 'plateau':
        print(
            f"WFA gate plateau patience/min_delta: "
            f"{config.wfa_gate_patience}/{config.wfa_gate_min_delta}"
        )
    if config.wfa_gate_mode != 'off':
        print(f"WFA gate warmup epochs: {config.wfa_gate_warmup_epochs}")
    print(f"B_LOS physics mode: {config.blos_physics_mode}")
    if config.blos_physics_mode == "single_height":
        print(f"B_LOS target log(tau): {config.blos_target_logtau}")
    print(f"V_LOS physics mode: {config.vlos_physics_mode}")
    if config.vlos_physics_mode == "single_height":
        print(f"V_LOS target log(tau): {config.vlos_target_logtau}")
    print(f"Temperature physics mode: {config.temp_physics_mode}")
    if config.temp_physics_mode == "single_height":
        print(f"Temperature target log(tau): {config.temp_target_logtau}")
    print(f"Temperature reference: {config.temp_reference_temperature} K")
    print(f"Epoch plots: {config.enable_epoch_plots}")
    if config.enable_epoch_plots:
        print(f"Epoch plot step: {config.epoch_plot_step if config.epoch_plot_step is not None else 'first val step'}")
        print(f"Epoch plot ODs: {config.epoch_plot_ods}")
        print(f"Epoch plot params: {config.epoch_plot_params}")
        print(f"Epoch videos: {config.enable_epoch_videos} (fps={config.epoch_plot_video_fps})")
    print(f"MODEST epoch plots: {config.enable_modest_epoch_plots}")
    if config.enable_modest_epoch_plots:
        print(f"MODEST cache dir: {config.modest_cache_dir} (enabled={not config.no_modest_cache})")
        print(f"MODEST prediction input mode: {'downsampled' if config.modest_downsample_prediction_input else 'upsampled'}")
        print(f"MODEST polarization mask: {config.modest_polarization_mask} (thr={config.modest_polarization_threshold})")
        print(f"MODEST crop bounds: {config.modest_crop_bounds}")
        print(f"MODEST epoch ODs: {config.modest_epoch_plot_ods}")
        print(f"MODEST epoch params: {config.modest_epoch_plot_params}")
        print(f"MODEST temperature calibration mode: {config.modest_temp_calibration_mode}")
        print(f"MODEST temperature calibration file: {config.modest_temp_calibration_file}")
        print(f"MODEST temperature calibration min samples: {config.modest_temp_calibration_min_samples}")
        print(f"MODEST temperature calibration clip quantiles: {config.modest_temp_calibration_clip_quantiles}")
    print("=" * 70)
    
    # Load normalizers
    print("\nLoading normalizers...")
    mhd_normalizer = MhdNormalizer()
    mhd_normalizer.load(filepath=config.data_path / config.mhd_normalizer_path)
    stokes_normalizer = StokesNormalizer()
    stokes_normalizer.load(filepath=config.data_path / config.stokes_normalizer_path)
    print("  ✓ Normalizers loaded")
    
    # Initialize model (including temperature parameters)
    print("\nInitializing model...")
    n_logtau = config.get_n_logtau()
    model = PhysicsInformedMSCNN(
        scales=config.scales,
        in_channels=config.in_channels,
        c1_filters=config.c1_filters,
        c2_filters=config.c2_filters,
        kernel_size=config.kernel_size,
        pool_size=config.pool_size,
        n_linear_layers=config.n_linear_layers,
        output_features=3 * n_logtau,
        lambda_wfa=config.lambda_wfa,
        lambda_doppler=config.lambda_doppler,
        lambda_temp=config.lambda_temp,
        blos_physics_mode=config.blos_physics_mode,
        blos_target_logtau=config.blos_target_logtau,
        vlos_physics_mode=config.vlos_physics_mode,
        vlos_target_logtau=config.vlos_target_logtau,
        temp_physics_mode=config.temp_physics_mode,
        temp_target_logtau=config.temp_target_logtau,
    ).to(config.device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model initialized with {n_params:,} trainable parameters")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Logger
    logger = MetricsLogger(config.log_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    wfa_gate_state = initialize_wfa_gate_state(config)
    
    if config.resume_from is not None:
        start_epoch, _, best_val_loss, loaded_wfa_gate_state = load_checkpoint(
            Path(config.resume_from),
            model=model,
            optimizer=optimizer,
        )
        if isinstance(loaded_wfa_gate_state, dict):
            wfa_gate_state.update(loaded_wfa_gate_state)
    
    # Prepare step list
    all_steps = list(range(config.min_step, config.max_step + 1))
    
    # Split into train and validation (e.g., 90-10 split)
    n_val = max(1, len(all_steps) // 10)
    val_steps = random.sample(all_steps, n_val)
    train_steps = [s for s in all_steps if s not in val_steps]
    
    print(f"\nTrain steps: {len(train_steps)}")
    print(f"Validation steps: {len(val_steps)}")
    monitor_step_for_epoch_plots = (
        config.epoch_plot_step if config.epoch_plot_step is not None else val_steps[0]
    )
    
    # Save configuration
    config.save(config.checkpoint_dir / "config.json")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training".center(70))
    print("=" * 70)
    
    # Initialize cache if enabled
    cache = None
    if config.use_cache:
        cache = MuramDataCache(cache_dir=config.cache_dir, compression='gzip')
        print("\nCache Information:")
        cache.print_cache_info()

    global_bz_selection_indices = None
    global_bz_balance_metadata = None
    if config.apply_bz_bin_balance and config.bz_balance_scope == "global":
        global_bz_selection_indices, global_bz_balance_metadata = compute_global_bz_balancing_indices(
            train_steps=train_steps,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            cache=cache,
        )
        global_meta_path = config.log_dir / "global_bz_balance_metadata.json"
        with open(global_meta_path, "w") as f:
            json.dump(global_bz_balance_metadata, f, indent=2)
        print(f"Global Bz balance metadata saved to: {global_meta_path}")

    balanced_cache = None
    balanced_cache_signature_hash = None
    balanced_cache_report = None
    balanced_runtime_mode = None
    preloaded_balanced_steps = None
    if config.use_balanced_cache:
        print("\nPreparing balanced training cache...")
        balanced_cache, balanced_cache_signature_hash, balanced_cache_report = build_or_refresh_balanced_cache(
            train_steps=train_steps,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            raw_cache=cache,
            global_bz_selection_indices=global_bz_selection_indices,
            global_bz_balance_metadata=global_bz_balance_metadata,
        )
        balanced_runtime_mode = choose_balanced_cache_runtime_mode(
            config=config,
            estimated_preload_bytes=int(balanced_cache_report["estimated_preload_bytes"]),
        )
        balanced_cache_report["runtime_mode"] = balanced_runtime_mode
        balanced_cache_report["ram_budget_gb"] = float(config.balanced_cache_ram_budget_gb)
        balanced_cache_report["ram_fraction"] = float(config.balanced_cache_ram_fraction)

        balanced_report_path = config.log_dir / "balanced_cache_report.json"
        with open(balanced_report_path, "w") as f:
            json.dump(balanced_cache_report, f, indent=2)
        print(f"Balanced cache report saved to: {balanced_report_path}")
        print(
            "Balanced cache summary: "
            f"steps={balanced_cache_report['total_steps_cached']}, "
            f"selected={balanced_cache_report['total_selected']}, "
            f"disk={balanced_cache_report['total_disk_mb']:.1f} MB, "
            f"preload_est={balanced_cache_report['estimated_preload_gb']:.2f} GB, "
            f"mode={balanced_runtime_mode}"
        )

        if balanced_runtime_mode == "preload":
            preloaded_balanced_steps = preload_balanced_steps_from_cache(
                train_steps=train_steps,
                balanced_cache=balanced_cache,
                signature_hash=balanced_cache_signature_hash,
            )
            print(f"Preloaded balanced steps: {len(preloaded_balanced_steps)}/{len(train_steps)}")

    modest_snapshot = None
    if config.enable_modest_epoch_plots:
        print("\nPreparing MODEST snapshot for per-epoch diagnostics...")
        try:
            modest_snapshot = prepare_modest_epoch_snapshot(
                config=config,
                stokes_normalizer=stokes_normalizer,
            )
            print("  ✓ MODEST snapshot prepared")
        except Exception as e:
            warnings.warn(f"Failed to prepare MODEST snapshot diagnostics: {e}")
            modest_snapshot = None

    total_training_pixels = 0

    for epoch in range(start_epoch, config.n_epochs):
        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")
        print("-" * 70)
        train_wfa_enabled = bool(wfa_gate_state.get('enabled', True))
        print(f"  Train-time WFA enabled: {train_wfa_enabled}")
        
        # Train for one epoch using the extracted function
        epoch_metrics = train_epoch(
            model=model,
            train_steps=train_steps,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            optimizer=optimizer,
            epoch=epoch,
            logger=logger,
            n_steps_per_epoch=-1,  # Use all training steps
            cache=cache,
            enable_wfa=train_wfa_enabled,
            global_bz_selection_indices=global_bz_selection_indices,
            global_bz_balance_metadata=global_bz_balance_metadata,
            balanced_cache=balanced_cache if balanced_runtime_mode == "disk" else None,
            balanced_cache_signature_hash=balanced_cache_signature_hash if balanced_runtime_mode == "disk" else None,
            preloaded_balanced_steps=preloaded_balanced_steps,
        )
        
        avg_train_loss = epoch_metrics['total_loss']
        total_training_pixels += int(epoch_metrics.get('n_pixels_used', 0))
        wfa_gate_state, wfa_gate_triggered, wfa_gate_reason = update_wfa_gate_state(
            gate_state=wfa_gate_state,
            config=config,
            epoch=epoch,
            epoch_mse_loss=float(epoch_metrics['mse_loss']),
        )
        
        # Validation
        print("\nValidating...")
        avg_val_loss = validate(
            model=model,
            val_steps=val_steps,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            cache=cache,
        )

        if config.enable_epoch_plots:
            generate_epoch_diagnostic_plots(
                model=model,
                epoch=epoch,
                step=monitor_step_for_epoch_plots,
                config=config,
                mhd_normalizer=mhd_normalizer,
                stokes_normalizer=stokes_normalizer,
                cache=cache,
            )

        if config.enable_modest_epoch_plots and modest_snapshot is not None:
            generate_epoch_modest_diagnostic_plots(
                model=model,
                epoch=epoch,
                config=config,
                mhd_normalizer=mhd_normalizer,
                modest_snapshot=modest_snapshot,
            )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch metrics
        logger.log_epoch(epoch + 1, avg_train_loss, avg_val_loss, current_lr)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  LR:         {current_lr:.2e}")
        
        # Print detailed loss breakdown
        print(f"  Loss Components:")
        print(f"    ├─ MSE Loss:         {epoch_metrics['mse_loss']:.6f}")
        print(f"    └─ Physics Loss:     {epoch_metrics['physics_loss']:.6f}")
        print(f"        ├─ WFA Loss:         {epoch_metrics['wfa_loss']:.6f}")
        print(f"        ├─ Doppler Loss:     {epoch_metrics['doppler_loss']:.6f}")
        print(f"        └─ Temperature Loss: {epoch_metrics['temperature_loss']:.6f}")
        print(
            f"  WFA gate state (next epoch): enabled={bool(wfa_gate_state.get('enabled', True))}, "
            f"mode={wfa_gate_state.get('mode')}"
        )
        if wfa_gate_state.get('mode') == 'plateau':
            print(
                f"    plateau_epochs={int(wfa_gate_state.get('plateau_epochs', 0))}, "
                f"best_train_mse={wfa_gate_state.get('best_metric')}"
            )
        if wfa_gate_triggered:
            print(f"  ★ WFA gate activated for subsequent epochs: {wfa_gate_reason}")
        print(f"  Pixels used this epoch (balanced): {epoch_metrics.get('n_pixels_used', 0)}")
        
        # Save checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            print(f"  ★ New best validation loss: {best_val_loss:.6f}")
        
        if (epoch + 1) % config.save_every == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                config=config,
                wfa_gate_state=wfa_gate_state,
                is_best=is_best,
            )
    
    print("\n" + "=" * 70)
    print("Training Complete!".center(70))
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total training pixels used (all epochs): {total_training_pixels}")

    training_metadata = {
        "total_training_pixels_used": int(total_training_pixels),
        "epochs": int(config.n_epochs),
        "step_size": int(config.step_size),
        "batch_size": int(config.batch_size),
        "wfa_gate_state": wfa_gate_state,
        "bz_balance": {
            "enabled": bool(config.apply_bz_bin_balance),
            "scope": str(config.bz_balance_scope),
            "mode": str(config.bz_balance_mode),
            "bins": int(config.bz_balance_bins),
            "tau_idx": None if config.bz_balance_tau_idx is None else int(config.bz_balance_tau_idx),
            "global_metadata_file": str(config.log_dir / "global_bz_balance_metadata.json")
            if config.apply_bz_bin_balance and config.bz_balance_scope == "global"
            else None,
        },
        "balanced_cache": {
            "enabled": bool(config.use_balanced_cache),
            "dir": str(config.balanced_cache_dir),
            "strategy": str(config.balanced_cache_strategy),
            "runtime_mode": None if balanced_runtime_mode is None else str(balanced_runtime_mode),
            "signature_hash": balanced_cache_signature_hash,
            "report_file": str(config.log_dir / "balanced_cache_report.json") if config.use_balanced_cache else None,
        },
    }
    metadata_path = config.log_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(training_metadata, f, indent=2)
    print(f"Training metadata saved to: {metadata_path}")

    if config.enable_epoch_plots:
        print("Generating final epoch-metric trend plots...")
        generate_epoch_metric_trend_plots(
            config=config,
            step=monitor_step_for_epoch_plots,
        )
    
    # Build epoch-diagnostic videos at end of training
    if config.enable_epoch_plots and config.enable_epoch_videos:
        print("\nBuilding epoch diagnostic videos...")
        generate_epoch_diagnostic_videos(config=config, step=monitor_step_for_epoch_plots)

    if config.enable_modest_epoch_plots and config.enable_epoch_videos:
        print("\nBuilding MODEST epoch diagnostic videos...")
        generate_epoch_modest_diagnostic_videos(config=config)

    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Train PINN MSCNN model")
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--c1-filters', type=int, default=None,
                       help='Number of filters in first conv layer (overrides config)')
    parser.add_argument('--stokes-ic-mode', '--stokes_ic_mode', dest='stokes_ic_mode',
                       type=str, choices=['per_step', 'fixed_global'], default='fixed_global',
                       help='Continuum normalization mode for Stokes data')
    parser.add_argument('--stokes-mult-factor', '--stokes_mult_factor', dest='stokes_mult_factor',
                       type=float, default=1.0,
                       help='Scalar multiplier applied to normalized Stokes I and V before training')
    parser.add_argument('--data-source', '--data_source', dest='data_source',
                       type=str, choices=['muram_legacy', 'nicole_tau500'], default='nicole_tau500',
                       help='Training data source (default: nicole_tau500)')
    parser.add_argument('--wfa-gate-mode', '--wfa_gate_mode', dest='wfa_gate_mode',
                       type=str, choices=['off', 'threshold', 'plateau'], default=None,
                       help='Train-time WFA activation gate mode')
    parser.add_argument('--wfa-gate-threshold', '--wfa_gate_threshold', dest='wfa_gate_threshold',
                       type=float, default=None,
                       help='Enable WFA once epoch train MSE is <= this threshold')
    parser.add_argument('--wfa-gate-patience', '--wfa_gate_patience', dest='wfa_gate_patience',
                       type=int, default=None,
                       help='Plateau epochs before enabling WFA')
    parser.add_argument('--wfa-gate-min-delta', '--wfa_gate_min_delta', dest='wfa_gate_min_delta',
                       type=float, default=None,
                       help='Minimum epoch train MSE improvement to reset WFA plateau counter')
    parser.add_argument('--wfa-gate-warmup-epochs', '--wfa_gate_warmup_epochs', dest='wfa_gate_warmup_epochs',
                       type=int, default=None,
                       help='Minimum number of epochs before WFA gate can activate')
    
    # Add cache-related arguments
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable data caching')
    parser.add_argument('--cache-dir', '--cache_dir', type=str,
                       default=None,
                       help='Directory for cached data (or set MURAM_CACHE_DIR). Defaults to the '
                            'standard .muram_cache dir, suffixed with the data source for non-legacy sources.')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear cache before training')
    parser.add_argument('--balanced-cache', '--balanced_cache', dest='use_balanced_cache', action='store_true',
                       help='Enable post-balancing train-data cache')
    parser.add_argument('--balanced-cache-dir', '--balanced_cache_dir', dest='balanced_cache_dir', type=str,
                       default=None,
                       help='Directory for balanced training cache (or set MURAM_BALANCED_CACHE_DIR). '
                            'Defaults to the standard dir, suffixed with the data source for non-legacy sources.')
    parser.add_argument('--clear-balanced-cache', '--clear_balanced_cache', dest='clear_balanced_cache', action='store_true',
                       help='Clear balanced training cache before training')
    parser.add_argument('--balanced-cache-strategy', '--balanced_cache_strategy', dest='balanced_cache_strategy',
                       type=str, choices=['auto', 'preload', 'disk'], default='auto',
                       help='Balanced cache runtime strategy')
    parser.add_argument('--balanced-cache-ram-budget-gb', '--balanced_cache_ram_budget_gb',
                       dest='balanced_cache_ram_budget_gb', type=float, default=32.0,
                       help='RAM budget in GB used to decide balanced-cache preload feasibility')
    parser.add_argument('--balanced-cache-ram-fraction', '--balanced_cache_ram_fraction',
                       dest='balanced_cache_ram_fraction', type=float, default=0.75,
                       help='Fraction of RAM budget allowed for balanced-cache preload')
    
    # Epoch diagnostics CLI
    parser.add_argument('--no-epoch-plots', '--no_epoch_plots', dest='no_epoch_plots', action='store_true',
                       help='Disable per-epoch diagnostic plots')
    parser.add_argument('--no-epoch-videos', '--no_epoch_videos', dest='no_epoch_videos', action='store_true',
                       help='Disable per-epoch diagnostic videos')
    parser.add_argument('--epoch-plot-step', '--epoch_plot_step', dest='epoch_plot_step', type=int, default=None,
                       help='Monitoring step for epoch diagnostics')
    parser.add_argument('--epoch-plot-ods', '--epoch_plot_ods', dest='epoch_plot_ods', type=float, nargs='+', default=None,
                       help='Optical-depth values for epoch diagnostics')
    parser.add_argument('--epoch-plot-params', '--epoch_plot_params', dest='epoch_plot_params', type=str, nargs='+',
                       choices=['T', 'Vz', 'Bz'], default=None,
                       help='Parameters for epoch diagnostics')
    parser.add_argument('--epoch-plot-scatter-samples', '--epoch_plot_scatter_samples',
                       dest='epoch_plot_scatter_samples', type=int, default=None,
                       help='Max sampled points per scatter plot')

    # MODEST epoch diagnostics CLI
    parser.add_argument('--modest-epoch-plots', '--modest_epoch_plots', dest='modest_epoch_plots', action='store_true',
                       help='Enable per-epoch diagnostics on MODEST snapshot')
    parser.add_argument('--modest-cache-dir', '--modest_cache_dir', dest='modest_cache_dir', type=str,
                       default=os.environ.get(
                           "MODEST_CACHE_DIR",
                           "/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache",
                       ),
                       help='MODEST cache directory (or set MODEST_CACHE_DIR)')
    parser.add_argument('--no-modest-cache', '--no_modest_cache', dest='no_modest_cache', action='store_true',
                       help='Disable MODEST cache usage for per-epoch diagnostics')
    parser.add_argument('--clear-modest-cache', '--clear_modest_cache', dest='clear_modest_cache', action='store_true',
                       help='Clear MODEST cache before preparing per-epoch snapshot')
    modest_input_group = parser.add_mutually_exclusive_group()
    modest_input_group.add_argument('--modest-downsample-prediction-input', '--modest_downsample_prediction_input',
                       dest='modest_downsample_prediction_input', action='store_true',
                       help='Use downsampled MODEST prediction input for per-epoch diagnostics')
    modest_input_group.add_argument('--modest-upsample-prediction-input', '--modest_upsample_prediction_input',
                       dest='modest_downsample_prediction_input', action='store_false',
                       help='Use upsampled MODEST prediction input for per-epoch diagnostics')
    parser.set_defaults(modest_downsample_prediction_input=None)
    parser.add_argument('--modest-polarization-mask', '--modest_polarization_mask', dest='modest_polarization_mask',
                       action='store_true',
                       help='Apply circular polarization mask to MODEST snapshot for diagnostics')
    parser.add_argument('--modest-polarization-threshold', '--modest_polarization_threshold',
                       dest='modest_polarization_threshold', type=float, default=None,
                       help='Circular polarization threshold for MODEST mask')
    parser.add_argument('--modest-stokes-v-multiplier', '--modest_stokes_v_multiplier',
                       dest='modest_stokes_v_multiplier', type=float, default=None,
                       help='Scale factor applied to MODEST Stokes V (default: -1.0 to match MURaM polarity)')
    parser.add_argument('--modest-crop-bounds', '--modest_crop_bounds', dest='modest_crop_bounds',
                       nargs=4, type=int, default=None,
                       metavar=('Y_MIN', 'Y_MAX', 'X_MIN', 'X_MAX'),
                       help='Crop bounds for MODEST snapshot diagnostics')
    parser.add_argument('--modest-epoch-plot-ods', '--modest_epoch_plot_ods', dest='modest_epoch_plot_ods',
                       type=float, nargs='+', default=None,
                       help='Optical-depth values for MODEST epoch diagnostics')
    parser.add_argument('--modest-epoch-plot-params', '--modest_epoch_plot_params', dest='modest_epoch_plot_params',
                       type=str, nargs='+', choices=['T', 'Vz', 'Bz'], default=None,
                       help='Parameters for MODEST epoch diagnostics')
    parser.add_argument('--modest-epoch-plot-scatter-samples', '--modest_epoch_plot_scatter_samples',
                       dest='modest_epoch_plot_scatter_samples', type=int, default=None,
                       help='Max sampled points per MODEST scatter plot')
    parser.add_argument('--modest-temp-calibration-mode', '--modest_temp_calibration_mode',
                       dest='modest_temp_calibration_mode',
                       type=str,
                       choices=['off', 'fit_only', 'apply_only'],
                       default=None,
                       help='MODEST per-epoch temperature calibration mode')
    parser.add_argument('--modest-temp-calibration-file', '--modest_temp_calibration_file',
                       dest='modest_temp_calibration_file',
                       type=str,
                       default=None,
                       help='Path to temperature calibration JSON for MODEST diagnostics')
    parser.add_argument('--modest-temp-calibration-min-samples', '--modest_temp_calibration_min_samples',
                       dest='modest_temp_calibration_min_samples',
                       type=int,
                       default=None,
                       help='Minimum paired finite samples per log(tau) for calibration fitting')
    parser.add_argument('--modest-temp-calibration-clip-quantiles', '--modest_temp_calibration_clip_quantiles',
                       dest='modest_temp_calibration_clip_quantiles',
                       type=float,
                       nargs=2,
                       default=None,
                       metavar=('Q_LOW', 'Q_HIGH'),
                       help='Optional quantile clipping for MODEST temperature calibration fitting')
    
    # Optical depth remapping grid (RESTORED)
    parser.add_argument(
        '--logtau_values', '--logtau-values',
        type=float,
        nargs='+',
        default=None,
        help='Explicit log(tau) grid values (overrides min/max/step), e.g. --logtau_values -2.0 -1.9 ... 0.0'
    )
    parser.add_argument(
        '--logtau_min', '--logtau-min',
        type=float,
        default=None,
        help='Minimum log(tau) for range mode (if --logtau_values is not provided)'
    )
    parser.add_argument(
        '--logtau_max', '--logtau-max',
        type=float,
        default=None,
        help='Maximum log(tau) for range mode (if --logtau_values is not provided)'
    )
    parser.add_argument(
        '--logtau_step', '--logtau-step',
        type=float,
        default=None,
        help='Step in log(tau) for range mode (if --logtau_values is not provided)'
    )

    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = TrainingConfig.load(Path(args.config))
    else:
        config = TrainingConfig()
    
    # Override with command-line arguments
    if args.resume:
        config.resume_from = args.resume
    if args.epochs:
        config.n_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.c1_filters is not None:
        config.c1_filters = args.c1_filters
    config.stokes_ic_mode = args.stokes_ic_mode
    config.stokes_mult_factor = args.stokes_mult_factor
    config.data_source = args.data_source
    if config.stokes_ic_mode == 'fixed_global' and config.stokes_fixed_ic is None:
        ic_stats_path = Path(config.data_path) / "normalization_stats" / "ic_reference_stats.json"
        if ic_stats_path.exists():
            with open(ic_stats_path, "r", encoding="utf-8") as f:
                ic_payload = json.load(f)
            fixed_ic = ic_payload.get("fixed_ic")
            if fixed_ic is not None:
                config.stokes_fixed_ic = float(fixed_ic)
    if args.wfa_gate_mode is not None:
        config.wfa_gate_mode = args.wfa_gate_mode
    if args.wfa_gate_threshold is not None:
        config.wfa_gate_threshold = args.wfa_gate_threshold
    if args.wfa_gate_patience is not None:
        config.wfa_gate_patience = args.wfa_gate_patience
    if args.wfa_gate_min_delta is not None:
        config.wfa_gate_min_delta = args.wfa_gate_min_delta
    if args.wfa_gate_warmup_epochs is not None:
        config.wfa_gate_warmup_epochs = args.wfa_gate_warmup_epochs
    
    # Non-legacy sources get isolated cache dirs and normalizer-stats paths by
    # default, mirroring TrainingConfig.__post_init__ (which already ran with
    # data_source='muram_legacy' before the override above).
    default_cache_dir = "/scratchsan/observatorio/juagudeloo/MUISCA/.muram_cache"
    default_balanced_cache_dir = "/scratchsan/observatorio/juagudeloo/MUISCA/.muram_balanced_cache"
    if args.cache_dir is None:
        args.cache_dir = os.environ.get(
            "MURAM_CACHE_DIR",
            default_cache_dir if args.data_source == "muram_legacy" else f"{default_cache_dir}_{args.data_source}",
        )
    if args.balanced_cache_dir is None:
        args.balanced_cache_dir = os.environ.get(
            "MURAM_BALANCED_CACHE_DIR",
            default_balanced_cache_dir if args.data_source == "muram_legacy" else f"{default_balanced_cache_dir}_{args.data_source}",
        )
    if args.data_source != "muram_legacy":
        default_mhd_norm_path = "normalization_stats/mhd_normalization.json"
        default_stokes_norm_path = "normalization_stats/stokes_normalization.json"
        if config.mhd_normalizer_path == default_mhd_norm_path:
            config.mhd_normalizer_path = f"normalization_stats/{args.data_source}/mhd_normalization.json"
        if config.stokes_normalizer_path == default_stokes_norm_path:
            config.stokes_normalizer_path = f"normalization_stats/{args.data_source}/stokes_normalization.json"

    # Apply cache CLI overrides
    config.use_cache = not args.no_cache
    config.cache_dir = str(Path(args.cache_dir).expanduser().resolve())
    config.use_balanced_cache = args.use_balanced_cache
    config.balanced_cache_dir = str(Path(args.balanced_cache_dir).expanduser().resolve())
    config.clear_balanced_cache = args.clear_balanced_cache
    config.balanced_cache_strategy = args.balanced_cache_strategy
    config.balanced_cache_ram_budget_gb = args.balanced_cache_ram_budget_gb
    config.balanced_cache_ram_fraction = args.balanced_cache_ram_fraction

    # Apply epoch diagnostics CLI overrides
    config.enable_epoch_plots = not args.no_epoch_plots
    if args.no_epoch_videos:
        config.enable_epoch_videos = False
    if args.epoch_plot_step is not None:
        config.epoch_plot_step = args.epoch_plot_step
    if args.epoch_plot_ods is not None:
        config.epoch_plot_ods = args.epoch_plot_ods
    if args.epoch_plot_params is not None:
        config.epoch_plot_params = args.epoch_plot_params
    if args.epoch_plot_scatter_samples is not None:
        config.epoch_plot_scatter_samples = args.epoch_plot_scatter_samples

    # Apply MODEST epoch diagnostics CLI overrides
    config.enable_modest_epoch_plots = args.modest_epoch_plots
    config.modest_cache_dir = str(Path(args.modest_cache_dir).expanduser().resolve())
    config.no_modest_cache = args.no_modest_cache
    config.clear_modest_cache = args.clear_modest_cache
    if args.modest_downsample_prediction_input is not None:
        config.modest_downsample_prediction_input = args.modest_downsample_prediction_input
    config.modest_polarization_mask = args.modest_polarization_mask
    if args.modest_polarization_threshold is not None:
        config.modest_polarization_threshold = args.modest_polarization_threshold
    if args.modest_stokes_v_multiplier is not None:
        config.modest_stokes_v_multiplier = args.modest_stokes_v_multiplier
    if args.modest_crop_bounds is not None:
        config.modest_crop_bounds = tuple(args.modest_crop_bounds)
    if args.modest_epoch_plot_ods is not None:
        config.modest_epoch_plot_ods = args.modest_epoch_plot_ods
    if args.modest_epoch_plot_params is not None:
        config.modest_epoch_plot_params = args.modest_epoch_plot_params
    if args.modest_epoch_plot_scatter_samples is not None:
        config.modest_epoch_plot_scatter_samples = args.modest_epoch_plot_scatter_samples
    if args.modest_temp_calibration_mode is not None:
        config.modest_temp_calibration_mode = args.modest_temp_calibration_mode
    if args.modest_temp_calibration_file is not None:
        config.modest_temp_calibration_file = str(Path(args.modest_temp_calibration_file).expanduser().resolve())
    if args.modest_temp_calibration_min_samples is not None:
        config.modest_temp_calibration_min_samples = args.modest_temp_calibration_min_samples
    if args.modest_temp_calibration_clip_quantiles is not None:
        config.modest_temp_calibration_clip_quantiles = list(args.modest_temp_calibration_clip_quantiles)

    # Handle cache clearing
    if args.clear_cache and config.use_cache:
        cache = MuramDataCache(cache_dir=config.cache_dir)
        cache.clear(step=None, confirm=False)
        print("✓ Cache cleared\n")
    if args.clear_balanced_cache and config.use_balanced_cache:
        balanced_cache = BalancedTrainDataCache(cache_dir=config.balanced_cache_dir)
        balanced_cache.clear()
        print("✓ Balanced cache cleared\n")
    
    # Apply optical-depth CLI overrides (RESTORED)
    if args.logtau_values is not None:
        config.logtau_values = args.logtau_values
    if args.logtau_min is not None:
        config.logtau_min = args.logtau_min
    if args.logtau_max is not None:
        config.logtau_max = args.logtau_max
    if args.logtau_step is not None:
        config.logtau_step = args.logtau_step

    # Run training
    train_pinn_model(config)


if __name__ == "__main__":
    main()