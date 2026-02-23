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
import argparse
import random
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import astropy.units as u
from tqdm import tqdm
from utils.grad_norm import GradNormScheduler, log_gradient_norms_by_task

# Ensure utils and models are importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.muram_data import MhdData, StokesData, MuramStepDataset
from utils.normalizer import MhdNormalizer, StokesNormalizer
from utils.cache_manage import DataCache
from models.pinn_mscnn_model import PhysicsInformedMSCNN
from utils.physics_utils import ApproxInversions


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Data paths
    data_path: str = "/scratchsan/observatorio/juagudeloo/data/"
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
    lambda_wfa: float = 0.01      # WFA term weight (if not using GradNorm)
    lambda_doppler: float = 0.01  # Doppler term weight (if not using GradNorm)
    lambda_temp: float = 0.01     # Temperature term weight (if not using GradNorm)
    use_gradnorm: bool = True  # Enable GradNorm automatic balancing
    gradnorm_alpha: float = 1.5  # GradNorm alpha parameter
    gradnorm_update_freq: int = 100  # Update GradNorm weights every N batches
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
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "plateau"  # 'plateau' or 'cosine'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # New: Caching parameters
    use_cache: bool = True
    cache_dir: str = "/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.data_cache"
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [1, 2, 3]
        if self.temp_continuum_indices is None:
            self.temp_continuum_indices = [0, 1, 2, 3]
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
        return cls(**config_dict)

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
        self.gradnorm_log = open(log_dir / "gradnorm_log.csv", 'w')
        
        # Write headers
        self.epoch_log.write("epoch,train_loss,val_loss,lr\n")
        self.batch_log.write("epoch,step,batch,loss,mse_loss,physics_loss,wfa_loss,doppler_loss,temperature_loss\n")
        self.gradnorm_log.write("epoch,step,batch,grad_norm_loss,mse_grad,wfa_grad,doppler_grad,temp_grad,mse_weight,wfa_weight,doppler_weight,temp_weight\n")
    
    def log_batch(self, epoch: int, step: int, batch: int, loss_dict: dict[str, float]):
        """Log batch-level metrics."""
        self.batch_log.write(
            f"{epoch},{step},{batch},"
            f"{loss_dict.get('total', 0.0)},"
            f"{loss_dict.get('mse', 0.0)},"
            f"{loss_dict.get('physics', 0.0)},"
            f"{loss_dict.get('wfa', 0.0)},"
            f"{loss_dict.get('doppler', 0.0)},"
            f"{loss_dict.get('temperature', 0.0)}\n"
        )
        self.batch_log.flush()
    
    def log_gradnorm(self, epoch: int, step: int, batch: int, gradnorm_dict: dict[str, float]):
        """Log GradNorm metrics."""
        self.gradnorm_log.write(
            f"{epoch},{step},{batch},"
            f"{gradnorm_dict.get('grad_norm_loss', 0.0)},"
            f"{gradnorm_dict.get('mse_grad_norm', 0.0)},"
            f"{gradnorm_dict.get('wfa_grad_norm', 0.0)},"
            f"{gradnorm_dict.get('doppler_grad_norm', 0.0)},"
            f"{gradnorm_dict.get('temperature_grad_norm', 0.0)},"
            f"{gradnorm_dict.get('mse_weight', 1.0)},"
            f"{gradnorm_dict.get('wfa_weight', 1.0)},"
            f"{gradnorm_dict.get('doppler_weight', 1.0)},"
            f"{gradnorm_dict.get('temperature_weight', 1.0)}\n"
        )
        self.gradnorm_log.flush()
    
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
        self.gradnorm_log.close()
    
    def __del__(self):
        self.close()

def build_cache_config_signature(config: TrainingConfig) -> dict:
    """Shared cache-signature contract across training/ablation/analysis."""
    return {
        'nx': config.nx,
        'ny': config.ny,
        'nz': config.nz,
        'z_max': config.z_max,
        'dz_km': config.dz_km,
        'central_wavelength': config.central_wavelength,
        'wl_range': config.wl_range,
    }

def load_and_prepare_step(
    step: int,
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    cache: DataCache | None = None,
) -> tuple[MuramStepDataset, dict[str, np.ndarray]]:
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
    cache : DataCache, optional
        Cache manager for loading/saving processed data
    
    Returns
    -------
    dataset : MuramStepDataset
        Dataset containing normalized inputs/targets
    approx_data : dict
        Physics approximations {'blos': (nx, ny), 'vlos': (nx, ny), 'temp': (nx, ny)}
    """
    # Compute configuration hash for cache validation
    config_for_hash = build_cache_config_signature(config)
    config_hash = DataCache.make_config_hash(config_for_hash) if cache else None
    
    # Try to load from cache
    if cache is not None and cache.exists(step, config_hash):
        try:
            return cache.load(
                step=step,
                stokes_normalizer=stokes_normalizer,
                mhd_normalizer=mhd_normalizer,
                verbose=True,
            )
        except Exception as e:
            print(f"  ⚠ Cache load failed for step {step}: {e}")
            print(f"  Reprocessing step {step}...")
    
    # Load MHD data
    mhd = MhdData(
        data_path=config.data_path / "muram-simulation",
        nx=config.nx, ny=config.ny, nz=config.nz
    )
    mhd.load_step(step=step, z_max=config.z_max)
    mhd.load_opacity_table(kappa_path=config.data_path / config.kappa_path)
    mhd.compute_optical_depth(dz=config.dz_km * u.km)
    
    # Remap to optical depth
    new_logtau = np.arange(-2.0, 0.1, 0.1)
    mhd.remap_to_optical_depth(new_logtau, quantities=["T", "Vz", "Bz"])
    
    # Load Stokes data
    stokes = StokesData(
        data_dir=config.data_path / "muram-simulation/",
        step=step,
        wavelength_range=(6300.5, 6303.5),
        wavelength_step=0.01
    )
    stokes.load_stokes()
    stokes.continuum_normalization(cont_indices=[0, 1, 2, 3])
    stokes.load_hinode_lsf(config.data_path / config.lsf_path)
    stokes.apply_spectral_convolution()
    stokes.resample_to_hinode()
    
    # Create dataset
    dataset = MuramStepDataset(
        stokes_data=stokes.data,
        mhd_data=mhd.od_data,
        stokes_normalizer=stokes_normalizer,
        mhd_normalizer=mhd_normalizer,
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
                mhd_data=mhd.od_data,
                approx_data=approx_data,
                config_hash=config_hash,
                verbose=True,
            )
        except Exception as e:
            print(f"  ⚠ Failed to save cache for step {step}: {e}")
    
    return dataset, approx_data

def train_one_step(
    model: PhysicsInformedMSCNN,
    dataloader: DataLoader,
    approx_data: dict[str, np.ndarray],
    mhd_normalizer: MhdNormalizer,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    step_num: int,
    logger: MetricsLogger,
    gradnorm_scheduler: GradNormScheduler | None = None,
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
    gradnorm_scheduler : Optional[GradNormScheduler]
        GradNorm scheduler
        
    Returns
    -------
    step_metrics : Dict[str, float]
        Dictionary containing average loss components for this step
    """
    model.train()
    
    # Set physics context once for this step (including temperature)
    model.set_physics_context(
        mhd_normalizer=mhd_normalizer,
        logtau_values=np.arange(-2.0, 0.1, 0.1),
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
    
    # GradNorm tracking
    if gradnorm_scheduler is not None:
        step_metrics.update({
            'mse_grad_norm': 0.0,
            'wfa_grad_norm': 0.0,
            'doppler_grad_norm': 0.0,
            'temperature_grad_norm': 0.0,
            'mse_weight': 0.0,
            'wfa_weight': 0.0,
            'doppler_weight': 0.0,
            'temperature_weight': 0.0,
            'grad_norm_loss': 0.0,
        })
    
    n_batches = 0
    batch_count = 0
    
    for batch_idx, (stokes_batch, mhd_batch, spatial_idx_batch) in enumerate(dataloader):
        # Move to device
        stokes_batch = stokes_batch.to(config.device)
        mhd_batch = mhd_batch.to(config.device)
        spatial_idx_batch = spatial_idx_batch.to(config.device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(stokes_batch)
        
        if config.use_gradnorm and gradnorm_scheduler is not None:
            # Compute individual unweighted losses for GradNorm
            loss_dict = model.compute_loss(
                predictions=predictions,
                targets=mhd_batch,
                spatial_indices=spatial_idx_batch,
                return_individual=True,
            )
            
            individual_losses = loss_dict['individual']
            
            # Compute weighted loss using GradNorm
            total_loss = gradnorm_scheduler.compute_weighted_loss(individual_losses)
            
            # Backward pass
            total_loss.backward()
            
            # Update GradNorm weights every N batches
            if batch_count % config.gradnorm_update_freq == 0:
                current_losses = {k: v.item() for k, v in individual_losses.items()}
                gradnorm_diagnostics = gradnorm_scheduler.step(
                    individual_losses,
                    model,
                    current_losses
                )
                
                # Log GradNorm metrics
                if logger is not None:
                    logger.log_gradnorm(epoch, step_num, batch_idx, gradnorm_diagnostics)
                
                # Accumulate GradNorm metrics
                for key, value in gradnorm_diagnostics.items():
                    metric_key = key
                    if metric_key in step_metrics:
                        step_metrics[metric_key] += value
            
            # Track individual loss components for logging
            step_metrics['mse_loss'] += loss_dict['mse'].item()
            step_metrics['physics_loss'] += loss_dict['physics'].item()
            step_metrics['wfa_loss'] += loss_dict.get('wfa', 0.0)
            step_metrics['doppler_loss'] += loss_dict.get('doppler', 0.0)
            step_metrics['temperature_loss'] += loss_dict.get('temperature', 0.0)
            
        else:
            # Standard training with manual lambda weights
            loss_dict = model.compute_loss(
                predictions=predictions,
                targets=mhd_batch,
                spatial_indices=spatial_idx_batch,
                return_individual=False,
            )
            
            total_loss = loss_dict['loss']
            
            # Backward pass
            total_loss.backward()
            
            # Accumulate loss components
            step_metrics['mse_loss'] += loss_dict['mse'].item()
            step_metrics['physics_loss'] += loss_dict['physics'].item()
            step_metrics['wfa_loss'] += loss_dict.get('wfa', 0.0)
            step_metrics['doppler_loss'] += loss_dict.get('doppler', 0.0)
            step_metrics['temperature_loss'] += loss_dict.get('temperature', 0.0)
        
        # Gradient clipping
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        
        step_metrics['total_loss'] += total_loss.item()
        n_batches += 1
        batch_count += 1
    
    # Average metrics over all batches
    for key in step_metrics.keys():
        step_metrics[key] /= n_batches
    
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
    cache: DataCache | None = None,
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
    cache : DataCache, optional
        Cache manager for data loading
    
    Returns
    -------
    avg_val_loss : float
        Average validation loss across all validation steps
    """
    model.eval()
    total_val_loss = 0.0
    n_val_samples = 0
    
    with torch.no_grad():
        for step in val_steps:
            try:
                dataset, approx_data = load_and_prepare_step(
                    step=step,
                    config=config,
                    mhd_normalizer=mhd_normalizer,
                    stokes_normalizer=stokes_normalizer,
                    cache=cache,
                )
                
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
                    logtau_values=np.arange(-2.0, 0.1, 0.1),
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
                    )
                    
                    total_loss = loss_dict['loss']
                    total_val_loss += total_loss.item() * stokes_batch.size(0)
                    n_val_samples += stokes_batch.size(0)
            
            except Exception as e:
                print(f"  Warning: Failed to validate on step {step}: {e}")
                continue
    
    return total_val_loss / n_val_samples if n_val_samples > 0 else float('inf')

def save_checkpoint(
    model: PhysicsInformedMSCNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: TrainingConfig,
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
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
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
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> tuple[int, float, float]:
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
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint.get('train_loss', 0.0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"  Resumed from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
    
    return start_epoch, train_loss, val_loss

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
    gradnorm_scheduler: GradNormScheduler | None = None,
    cache: DataCache | None = None,
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
    gradnorm_scheduler : Optional[GradNormScheduler]
        GradNorm scheduler
    cache : DataCache, optional
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
        'n_steps': 0
    }
    
    # Progress bar
    step_pbar = tqdm(steps_to_use, desc=f"Epoch {epoch + 1}", unit="step", leave=False)
    
    for step in step_pbar:
        try:
            # Load and prepare step (uses cache if available)
            dataset, approx_data = load_and_prepare_step(
                step=step,
                config=config,
                mhd_normalizer=mhd_normalizer,
                stokes_normalizer=stokes_normalizer,
                cache=cache,
            )
            
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
                gradnorm_scheduler=gradnorm_scheduler,
            )
            
            # Accumulate step metrics (including temperature)
            epoch_metrics['total_loss'] += step_metrics['total_loss']
            epoch_metrics['mse_loss'] += step_metrics['mse_loss']
            epoch_metrics['physics_loss'] += step_metrics['physics_loss']
            epoch_metrics['wfa_loss'] += step_metrics['wfa_loss']
            epoch_metrics['doppler_loss'] += step_metrics['doppler_loss']
            epoch_metrics['temperature_loss'] += step_metrics['temperature_loss']
            epoch_metrics['n_steps'] += 1
            
            # Update progress bar
            step_pbar.set_postfix({'loss': f'{step_metrics["total_loss"]:.6f}'})
            
            # Clean up
            del dataset, dataloader
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n  Error processing step {step}: {e}")
            continue
    
    # Compute averages
    n_steps = epoch_metrics['n_steps']
    if n_steps > 0:
        for key in epoch_metrics:
            if key != 'n_steps':
                epoch_metrics[key] /= n_steps
    
    return epoch_metrics

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
    print(f"Learning rate: {config.learning_rate}")
    print(f"Lambda WFA: {config.lambda_wfa}")
    print(f"Lambda Doppler: {config.lambda_doppler}")
    print(f"Lambda Temperature: {config.lambda_temp}")
    print(f"Use GradNorm: {config.use_gradnorm}")
    if config.use_gradnorm:
        print(f"GradNorm alpha: {config.gradnorm_alpha}")
        print(f"GradNorm update frequency: {config.gradnorm_update_freq}")
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
    model = PhysicsInformedMSCNN(
        scales=config.scales,
        in_channels=config.in_channels,
        c1_filters=config.c1_filters,
        c2_filters=config.c2_filters,
        kernel_size=config.kernel_size,
        pool_size=config.pool_size,
        n_linear_layers=config.n_linear_layers,
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
    
    # GradNorm scheduler
    gradnorm_scheduler = None
    if config.use_gradnorm and any([config.lambda_wfa > 0, config.lambda_doppler > 0, config.lambda_temp > 0]):
        print("\nInitializing GradNorm scheduler...")
        initial_weights = [1.0, 1.0, 1.0, 1.0]  # MSE, WFA, Doppler, Temp
        gradnorm_scheduler = GradNormScheduler(
            num_tasks=4,
            alpha=config.gradnorm_alpha,
            initial_weights=initial_weights,
            device=config.device
        )
        print(f"  ✓ GradNorm initialized with alpha={config.gradnorm_alpha}")
    
    # Scheduler
    scheduler = None
    if config.use_scheduler:
        if config.scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                verbose=True
            )
        elif config.scheduler_type == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
        else:
            raise ValueError(
                f"Invalid scheduler_type='{config.scheduler_type}'. Use 'plateau' or 'cosine'."
            )
    # Logger
    logger = MetricsLogger(config.log_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.resume_from is not None:
        start_epoch, _, best_val_loss = load_checkpoint(
            Path(config.resume_from),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
    
    # Prepare step list
    all_steps = list(range(config.min_step, config.max_step + 1))
    
    # Split into train and validation (e.g., 90-10 split)
    n_val = max(1, len(all_steps) // 10)
    val_steps = random.sample(all_steps, n_val)
    train_steps = [s for s in all_steps if s not in val_steps]
    
    print(f"\nTrain steps: {len(train_steps)}")
    print(f"Validation steps: {len(val_steps)}")
    
    # Save configuration
    config.save(config.checkpoint_dir / "config.json")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training".center(70))
    print("=" * 70)
    
    # Initialize cache if enabled
    cache = None
    if config.use_cache:
        cache = DataCache(cache_dir=config.cache_dir, compression='gzip')
        print("\nCache Information:")
        cache.print_cache_info()

    for epoch in range(start_epoch, config.n_epochs):
        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")
        print("-" * 70)
        
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
            gradnorm_scheduler=gradnorm_scheduler,
            cache=cache,
        )
        
        avg_train_loss = epoch_metrics['total_loss']
        
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
        
        # Update scheduler
        if scheduler is not None:
            if config.scheduler_type == 'plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
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
        
        # Print GradNorm weights if enabled
        if gradnorm_scheduler is not None:
            weights = gradnorm_scheduler.task_weights.detach().cpu().numpy()
            print(f"  GradNorm Weights:")
            print(f"    MSE: {weights[0]:.4f}, WFA: {weights[1]:.4f}, "
                  f"Doppler: {weights[2]:.4f}, Temp: {weights[3]:.4f}")
        
        # Save checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            print(f"  ★ New best validation loss: {best_val_loss:.6f}")
        
        if (epoch + 1) % config.save_every == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                config=config,
                is_best=is_best,
            )
    
    print("\n" + "=" * 70)
    print("Training Complete!".center(70))
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Train PINN MSCNN model")
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')

    # Scheduler arguments (missing before)
    parser.add_argument('--no-scheduler', action='store_true',
                       help='Disable learning rate scheduler (fixed LR)')
    parser.add_argument('--scheduler-type', type=str, choices=['plateau', 'cosine', 'none'],
                       help="Scheduler type ('none' disables scheduler)")
    
    # Add cache-related arguments
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable data caching')
    parser.add_argument('--cache-dir', type=str, 
                       default='/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.data_cache',
                       help='Directory for cached data')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear cache before training')
    
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

    # Apply scheduler CLI overrides
    if args.scheduler_type:
        if args.scheduler_type == 'none':
            config.use_scheduler = False
        else:
            config.use_scheduler = True
            config.scheduler_type = args.scheduler_type
    if args.no_scheduler:
        config.use_scheduler = False

    # Handle cache clearing
    if args.clear_cache and config.use_cache:
        cache = DataCache(cache_dir=config.cache_dir)
        cache.clear(step=None, confirm=False)
        print("✓ Cache cleared\n")
    
    # Run training
    train_pinn_model(config)


if __name__ == "__main__":
    main()