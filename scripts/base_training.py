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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import astropy.units as u
from tqdm import tqdm

# Ensure utils and models are importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.muram_data import MhdData, StokesData  # noqa: E402
from utils.normalizer import MhdNormalizer, StokesNormalizer  # noqa: E402
from models.pinn_mscnn_model import PhysicsInformedMSCNN  # noqa: E402
from utils.physics_utils import ApproxInversions  # noqa: E402


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
    
    # Training parameters
    n_epochs: int = 20
    batch_size: int = 512  # Spatial batch size (512 pixels per batch)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Model architecture
    scales: List[int] = None  # [1, 2, 3]
    in_channels: int = 2
    c1_filters: int = 16
    c2_filters: int = 32
    kernel_size: int = 5
    pool_size: int = 2
    n_linear_layers: int = 4
    dropout_rate: float = 0.2
    
    # Physics parameters
    central_wavelength: float = 6301.5  # Angstroms
    lande_factor: float = 1.67
    wl_range: Tuple[int, int] = (15, 60)
    lambda_physics: float = 0.001  # Physics regularization weight
    lambda_wfa: float = 0.01      # WFA term weight
    lambda_doppler: float = 0.01  # Doppler term weight
    use_physics: str = "wfa"  # 'wfa', 'doppler', 'both', or None
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10  # Save checkpoint every N epochs
    resume_from: Optional[str] = None
    
    # Logging
    log_dir: str = "logs"
    log_every: int = 10  # Log metrics every N batches within an epoch
    
    # Device and performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Scheduler
    scheduler_type: str = "plateau"  # 'plateau' or 'cosine'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [1, 2, 3]
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


class MuramStepDataset(Dataset):
    """
    Dataset for a single MURaM simulation step.
    
    Returns normalized Stokes profiles and MHD targets for each spatial pixel.
    Also provides spatial indices for physics regularization.
    """
    
    def __init__(
        self,
        stokes_data: Dict[str, np.ndarray],
        mhd_data: Dict[str, np.ndarray],
        stokes_normalizer: StokesNormalizer,
        mhd_normalizer: MhdNormalizer,
    ):
        """
        Parameters
        ----------
        stokes_data : dict
            Raw Stokes data {'I': (nx, ny, nλ), 'V': (nx, ny, nλ)}
        mhd_data : dict
            Raw MHD data {'T': (nx, ny, nτ), 'Vz': ..., 'Bz': ...}
        stokes_normalizer : StokesNormalizer
            Fitted normalizer for Stokes data
        mhd_normalizer : MhdNormalizer
            Fitted normalizer for MHD data
        """
        self.nx, self.ny = stokes_data['I'].shape[:2]
        self.n_pixels = self.nx * self.ny
        
        # Normalize data
        norm_stokes = stokes_normalizer.transform(stokes_data)
        norm_mhd = mhd_normalizer.transform(mhd_data)
        
        # Flatten spatial dimensions: (nx, ny, ...) -> (nx*ny, ...)
        # Stokes input: (n_pixels, 2, nλ)
        I_flat = norm_stokes['I'].reshape(self.n_pixels, -1)  # (n_pixels, 112)
        V_flat = norm_stokes['V'].reshape(self.n_pixels, -1)
        self.stokes_input = np.stack([I_flat, V_flat], axis=1)  # (n_pixels, 2, 112)
        
        # MHD targets: concatenate T, Vz, Bz along feature dimension
        # Each has shape (n_pixels, 21) -> concatenate to (n_pixels, 63)
        T_flat = norm_mhd['T'].reshape(self.n_pixels, -1)
        Vz_flat = norm_mhd['Vz'].reshape(self.n_pixels, -1)
        Bz_flat = norm_mhd['Bz'].reshape(self.n_pixels, -1)
        self.mhd_targets = np.concatenate([T_flat, Vz_flat, Bz_flat], axis=1)
        
        # Store spatial indices for physics regularization
        ix, iy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        self.spatial_indices = np.stack([ix.ravel(), iy.ravel()], axis=1)  # (n_pixels, 2)
        
    def __len__(self):
        return self.n_pixels
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.stokes_input[idx]).float(),
            torch.from_numpy(self.mhd_targets[idx]).float(),
            torch.from_numpy(self.spatial_indices[idx]).long(),
        )


def load_and_prepare_step(
    step: int,
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
) -> Tuple[MuramStepDataset, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load and prepare a single simulation step for training.
    
    Returns
    -------
    dataset : MuramStepDataset
        Dataset containing normalized inputs/targets
    physics_data : dict
        Unnormalized MHD data for physics regularization
        {'Bz': (nx, ny, 21), 'Vz': (nx, ny, 21)}
    approx_data : dict
        Physics approximations {'blos': (nx, ny), 'vlos': (nx, ny)}
    """
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
    vlos_approx = inv.compute_vlos_doppler(wl_range=config.wl_range).value
    
    approx_data = {
        'blos': blos_approx,
        'vlos': vlos_approx,
    }
    
    # Keep unnormalized OD data for physics regularization
    physics_data = {
        'Bz': mhd.od_data['Bz'],
        'Vz': mhd.od_data['Vz'],
    }
    
    return dataset, physics_data, approx_data


class PhysicsLossComputer:
    """
    Computes physics-informed loss for mini-batches.
    
    Handles indexing into full-resolution physics approximations
    and predicted OD cubes.
    """
    
    def __init__(
        self,
        physics_data: Dict[str, np.ndarray],
        approx_data: Dict[str, np.ndarray],
        logtau_values: np.ndarray,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device
        self.logtau_values = logtau_values
        
        # Convert to tensors (keep on CPU for indexing, move to GPU when needed)
        self.blos_full = torch.from_numpy(approx_data['blos']).float()
        self.vlos_full = torch.from_numpy(approx_data['vlos']).float()
        self.Bz_od_full = torch.from_numpy(physics_data['Bz']).float()
        self.Vz_od_full = torch.from_numpy(physics_data['Vz']).float()
        
        self.nx, self.ny = self.blos_full.shape
        self.n_tau = len(logtau_values)
        
    def compute_batch_physics_loss(
        self,
        predictions: torch.Tensor,
        spatial_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics regularization for a mini-batch.
        
        Parameters
        ----------
        predictions : Tensor, shape (batch_size, 63)
            Model predictions (T, Vz, Bz concatenated)
        spatial_indices : Tensor, shape (batch_size, 2)
            Spatial (ix, iy) coordinates for each sample in batch
        
        Returns
        -------
        total_physics_loss : Tensor
            Total physics regularization loss
        loss_components : dict
            Individual loss components for logging
        """
        batch_size = predictions.shape[0]
        
        # Extract predictions (already normalized)
        # predictions: (batch_size, 63) = (batch_size, 21 + 21 + 21)
        pred_T = predictions[:, :21]      # (batch_size, 21)
        pred_Vz = predictions[:, 21:42]   # (batch_size, 21)
        pred_Bz = predictions[:, 42:63]   # (batch_size, 21)
        
        # Get spatial coordinates
        ix = spatial_indices[:, 0]  # (batch_size,)
        iy = spatial_indices[:, 1]
        
        # Index into full-resolution data (on CPU, then move to device)
        batch_blos = self.blos_full.to(self.device)[ix, iy]  # (batch_size,)
        batch_vlos = self.vlos_full.to(self.device)[ix, iy]
        batch_Bz_od = self.Bz_od_full.to(self.device)[ix, iy, :]  # (batch_size, 21)
        batch_Vz_od = self.Vz_od_full.to(self.device)[ix, iy, :]
        
        loss_components = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 1. Weak Field Approximation (WFA) loss
        if self.config.use_physics in ['wfa', 'both']:
            # Compute Bz_LOS from predictions using trapezoidal integration
            # B_LOS = ∫ Bz(τ) dτ / ∫ dτ (τ in linear scale)
            tau_linear = 10 ** self.logtau_values  # (21,) on CPU
            tau_linear_tensor = torch.from_numpy(tau_linear).float().to(self.device)
            
            # Trapezoidal rule: integral ≈ Σ[(y_i + y_{i+1})/2 * (x_{i+1} - x_i)]
            dtau = tau_linear_tensor[1:] - tau_linear_tensor[:-1]  # (20,)
            
            # Average Bz between consecutive points
            Bz_avg = (pred_Bz[:, :-1] + pred_Bz[:, 1:]) / 2  # (batch_size, 20)
            
            # Compute integral of Bz
            integral_Bz = torch.sum(Bz_avg * dtau.unsqueeze(0), dim=1)  # (batch_size,)
            
            # Compute integral of dτ (just the range)
            integral_dtau = tau_linear_tensor[-1] - tau_linear_tensor[0]
            
            # Predicted B_LOS
            pred_blos = integral_Bz / integral_dtau  # (batch_size,)
            
            # WFA loss
            wfa_loss = nn.MSELoss()(pred_blos, batch_blos)
            loss_components['wfa'] = wfa_loss.item()
            total_loss += self.config.lambda_wfa * wfa_loss
        
        # 2. Doppler Shift loss
        if self.config.use_physics in ['doppler', 'both']:
            # Compute V_LOS from predictions using similar integration
            Vz_avg = (pred_Vz[:, :-1] + pred_Vz[:, 1:]) / 2  # (batch_size, 20)
            integral_Vz = torch.sum(Vz_avg * dtau.unsqueeze(0), dim=1)
            pred_vlos = integral_Vz / integral_dtau  # (batch_size,)
            
            doppler_loss = nn.MSELoss()(pred_vlos, batch_vlos)
            loss_components['doppler'] = doppler_loss.item()
            total_loss += self.config.lambda_doppler * doppler_loss
        
        # 3. Gradient consistency (optional smoothness constraint)
        # Penalize large changes in Bz and Vz along optical depth
        if self.config.lambda_physics > 0:
            # Compute finite differences
            dBz_dtau = torch.diff(pred_Bz, dim=1)  # (batch_size, 20)
            dVz_dtau = torch.diff(pred_Vz, dim=1)
            
            # L2 penalty on gradients (smoothness)
            gradient_loss = torch.mean(dBz_dtau ** 2) + torch.mean(dVz_dtau ** 2)
            loss_components['gradient'] = gradient_loss.item()
            total_loss += self.config.lambda_physics * gradient_loss
        
        return total_loss, loss_components


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
        self.batch_log.write("epoch,step,batch,loss,mse_loss,physics_loss,wfa_loss,doppler_loss,gradient_loss\n")
    
    def log_batch(self, epoch: int, step: int, batch: int, loss_dict: Dict[str, float]):
        """Log batch-level metrics."""
        self.batch_log.write(
            f"{epoch},{step},{batch},"
            f"{loss_dict.get('total', 0.0)},"
            f"{loss_dict.get('mse', 0.0)},"
            f"{loss_dict.get('physics', 0.0)},"
            f"{loss_dict.get('wfa', 0.0)},"
            f"{loss_dict.get('doppler', 0.0)},"
            f"{loss_dict.get('gradient', 0.0)}\n"
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


def train_one_step(
    model: PhysicsInformedMSCNN,
    dataloader: DataLoader,
    physics_computer: PhysicsLossComputer,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    step_num: int,
    logger: MetricsLogger,
) -> float:
    """
    Train on one simulation step (one epoch through that step's data).
    
    Returns
    -------
    avg_loss : float
        Average loss across all batches in this step
    """
    model.train()
    step_loss = 0.0
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
        
        # Supervised MSE loss
        mse_loss = nn.MSELoss()(predictions, mhd_batch)
        
        # Physics regularization
        physics_loss, physics_components = physics_computer.compute_batch_physics_loss(
            predictions=predictions,
            spatial_indices=spatial_idx_batch,
        )
        
        # Total loss
        total_loss = mse_loss + physics_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        
        # Accumulate loss
        step_loss += total_loss.item()
        n_batches += 1
        
        # Log batch metrics
        if batch_idx % config.log_every == 0:
            loss_dict = {
                'total': total_loss.item(),
                'mse': mse_loss.item(),
                'physics': physics_loss.item(),
                **physics_components
            }
            logger.log_batch(epoch, step_num, batch_idx, loss_dict)
    
    return step_loss / n_batches if n_batches > 0 else 0.0


def validate(
    model: PhysicsInformedMSCNN,
    val_steps: List[int],
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
) -> float:
    """
    Validate on a subset of steps.
    
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
                dataset, physics_data, approx_data = load_and_prepare_step(
                    step=step,
                    config=config,
                    mhd_normalizer=mhd_normalizer,
                    stokes_normalizer=stokes_normalizer,
                )
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=0,  # No multiprocessing for validation
                    pin_memory=False,
                )
                
                physics_computer = PhysicsLossComputer(
                    physics_data=physics_data,
                    approx_data=approx_data,
                    logtau_values=np.arange(-2.0, 0.1, 0.1),
                    config=config,
                    device=config.device,
                )
                
                for stokes_batch, mhd_batch, spatial_idx_batch in dataloader:
                    stokes_batch = stokes_batch.to(config.device)
                    mhd_batch = mhd_batch.to(config.device)
                    spatial_idx_batch = spatial_idx_batch.to(config.device)
                    
                    predictions = model(stokes_batch)
                    mse_loss = nn.MSELoss()(predictions, mhd_batch)
                    physics_loss, _ = physics_computer.compute_batch_physics_loss(
                        predictions, spatial_idx_batch
                    )
                    
                    total_loss = mse_loss + physics_loss
                    total_val_loss += total_loss.item() * stokes_batch.size(0)
                    n_val_samples += stokes_batch.size(0)
            
            except Exception as e:
                print(f"  Warning: Failed to validate on step {step}: {e}")
                continue
    
    return total_val_loss / n_val_samples if n_val_samples > 0 else float('inf')


def save_checkpoint(
    model: PhysicsInformedMSCNN,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
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
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[int, float, float]:
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


def train_pinn_model(config: TrainingConfig):
    """Main training loop with interleaved epoch training."""
    
    print("=" * 70)
    print("PINN MSCNN Training".center(70))
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Data path: {config.data_path}")
    print(f"Steps: {config.min_step} to {config.max_step}")
    print(f"Epochs: {config.n_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Physics regularization: {config.use_physics}")
    print("=" * 70)
    
    # Load normalizers
    print("\nLoading normalizers...")
    mhd_normalizer = MhdNormalizer()
    mhd_normalizer.load(filepath=config.data_path / config.mhd_normalizer_path)
    stokes_normalizer = StokesNormalizer()
    stokes_normalizer.load(filepath=config.data_path / config.stokes_normalizer_path)
    print("  ✓ Normalizers loaded")
    
    # Initialize model
    print("\nInitializing model...")
    model = PhysicsInformedMSCNN(
        scales=config.scales,
        in_channels=config.in_channels,
        c1_filters=config.c1_filters,
        c2_filters=config.c2_filters,
        kernel_size=config.kernel_size,
        pool_size=config.pool_size,
        n_linear_layers=config.n_linear_layers,
        dropout_rate=config.dropout_rate,
    ).to(config.device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model initialized with {n_params:,} trainable parameters")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
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
        scheduler = None
    
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
    
    for epoch in range(start_epoch, config.n_epochs):
        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")
        print("-" * 70)
        
        # Shuffle training steps each epoch
        random.shuffle(train_steps)
        
        epoch_train_loss = 0.0
        n_successful_steps = 0
        
        # Progress bar for steps
        step_pbar = tqdm(train_steps, desc=f"Epoch {epoch + 1}", unit="step")
        
        for step in step_pbar:
            try:
                # Load and prepare step
                dataset, physics_data, approx_data = load_and_prepare_step(
                    step=step,
                    config=config,
                    mhd_normalizer=mhd_normalizer,
                    stokes_normalizer=stokes_normalizer,
                )
                
                # Create dataloader
                dataloader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                    pin_memory=config.pin_memory,
                )
                
                # Create physics computer
                physics_computer = PhysicsLossComputer(
                    physics_data=physics_data,
                    approx_data=approx_data,
                    logtau_values=np.arange(-2.0, 0.1, 0.1),
                    config=config,
                    device=config.device,
                )
                
                # Train on this step
                step_loss = train_one_step(
                    model=model,
                    dataloader=dataloader,
                    physics_computer=physics_computer,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    step_num=step,
                    logger=logger,
                )
                
                epoch_train_loss += step_loss
                n_successful_steps += 1
                
                # Update progress bar
                step_pbar.set_postfix({'loss': f'{step_loss:.6f}'})
                
                # Clean up to free memory
                del dataset, dataloader, physics_computer
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n  Error processing step {step}: {e}")
                continue
        
        # Average training loss
        avg_train_loss = epoch_train_loss / n_successful_steps if n_successful_steps > 0 else float('inf')
        
        # Validation
        print("\nValidating...")
        avg_val_loss = validate(
            model=model,
            val_steps=val_steps,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
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
    
    # Run training
    train_pinn_model(config)


if __name__ == "__main__":
    main()