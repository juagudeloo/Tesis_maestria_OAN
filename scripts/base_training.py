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

from utils.muram_data import MhdData, StokesData, MuramStepDataset
from utils.normalizer import MhdNormalizer, StokesNormalizer
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

def load_and_prepare_step(
    step: int,
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
) -> Tuple[MuramStepDataset, Dict[str, np.ndarray]]:
    """
    Load and prepare a single simulation step for training.
    
    Returns
    -------
    dataset : MuramStepDataset
        Dataset containing normalized inputs/targets
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
    
    return dataset, approx_data

def train_one_step(
    model: PhysicsInformedMSCNN,
    dataloader: DataLoader,
    approx_data: Dict[str, np.ndarray],
    mhd_normalizer: MhdNormalizer,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    step_num: int,
    logger: MetricsLogger,
) -> Dict[str, float]:
    """
    Train on one simulation step (one epoch through that step's data).
    
    Parameters
    ----------
    model : PhysicsInformedMSCNN
        Model with integrated physics computation
    dataloader : DataLoader
        DataLoader for this step
    approx_data : Dict[str, np.ndarray]
        Physics approximations for this step
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
    
    # Set physics context once for this step
    model.set_physics_context(
        mhd_normalizer=mhd_normalizer,
        logtau_values=np.arange(-2.0, 0.1, 0.1),
        blos_approx=approx_data.get('blos'),
        vlos_approx=approx_data.get('vlos'),
    )
    
    # Initialize accumulators for all loss components
    step_metrics = {
        'total_loss': 0.0,
        'mse_loss': 0.0,
        'physics_loss': 0.0,
        'wfa_loss': 0.0,
        'doppler_loss': 0.0,
        'gradient_loss': 0.0,
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
        
        # Compute total loss (MSE + physics)
        loss_dict = model.compute_loss(
            predictions=predictions,
            targets=mhd_batch,
            spatial_indices=spatial_idx_batch,
        )
        
        total_loss = loss_dict['loss']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        
        # Accumulate all loss components
        step_metrics['total_loss'] += loss_dict['loss'].item()
        step_metrics['mse_loss'] += loss_dict['mse'].item()
        step_metrics['physics_loss'] += loss_dict['physics'].item()
        step_metrics['wfa_loss'] += loss_dict.get('wfa', 0.0)
        step_metrics['doppler_loss'] += loss_dict.get('doppler', 0.0)
        step_metrics['gradient_loss'] += loss_dict.get('gradient', 0.0)
        n_batches += 1
        
        # Log batch metrics
        if logger is not None and batch_idx % config.log_every == 0:
            log_dict = {
                'total': loss_dict['loss'].item(),
                'mse': loss_dict['mse'].item(),
                'physics': loss_dict['physics'].item(),
                'wfa': loss_dict.get('wfa', 0.0),
                'doppler': loss_dict.get('doppler', 0.0),
                'gradient': loss_dict.get('gradient', 0.0),
            }
            logger.log_batch(epoch, step_num, batch_idx, log_dict)
    
    # Average all metrics
    if n_batches > 0:
        for key in step_metrics:
            step_metrics[key] /= n_batches
    
    return step_metrics

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
                dataset, approx_data = load_and_prepare_step(
                    step=step,
                    config=config,
                    mhd_normalizer=mhd_normalizer,
                    stokes_normalizer=stokes_normalizer,
                )
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )
                
                # Set physics context for validation
                model.set_physics_context(
                    mhd_normalizer=mhd_normalizer,
                    logtau_values=np.arange(-2.0, 0.1, 0.1),
                    blos_approx=approx_data.get('blos'),
                    vlos_approx=approx_data.get('vlos'),
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

def train_epoch(
    model: PhysicsInformedMSCNN,
    train_steps: List[int],
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    logger: Optional[MetricsLogger] = None,
    n_steps_per_epoch: int = -1,
) -> Dict[str, float]:
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
    
    # Initialize metrics
    epoch_metrics = {
        'total_loss': 0.0,
        'mse_loss': 0.0,
        'physics_loss': 0.0,
        'wfa_loss': 0.0,
        'doppler_loss': 0.0,
        'smoothness_loss': 0.0,
        'n_steps': 0
    }
    
    # Progress bar
    step_pbar = tqdm(steps_to_use, desc=f"Epoch {epoch + 1}", unit="step", leave=False)
    
    for step in step_pbar:
        try:
            # Load and prepare step
            dataset, approx_data = load_and_prepare_step(
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
            )
            
            # Accumulate step metrics
            epoch_metrics['total_loss'] += step_metrics['total_loss']
            epoch_metrics['mse_loss'] += step_metrics['mse_loss']
            epoch_metrics['physics_loss'] += step_metrics['physics_loss']
            epoch_metrics['wfa_loss'] += step_metrics['wfa_loss']
            epoch_metrics['doppler_loss'] += step_metrics['doppler_loss']
            epoch_metrics['smoothness_loss'] += step_metrics['gradient_loss']
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
        use_physics=config.use_physics,
        lambda_wfa=config.lambda_wfa,
        lambda_doppler=config.lambda_doppler,
        lambda_physics=config.lambda_physics,
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