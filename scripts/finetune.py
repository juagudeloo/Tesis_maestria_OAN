#!/usr/bin/env python3
"""
Fine-Tuning Script for MUISCA Models
====================================

Loads trained model checkpoints and resumes fine-tuning with mandatory Bz balancing.
All models are trained by run_experiments.sh with apply_bz_bin_balance=False (full data).
Fine-tuning applies Bz balancing to focus training on balanced Bz-strength bins.

Outputs results to output/fine-tune/<exp_name>-finetuned/<variation>/

Usage:
    python scripts/finetune.py --experiment-name exp_name --variations wfa_only,all_physics_terms [--finetune-epochs 50]

See docs/how-to-fine-tune.md and docs/magnetic_field_balancing_and_finetuning.md for detailed guidance.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directories to path (preserve pattern from ablation_study.py)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.normalizer import MhdNormalizer, StokesNormalizer
from utils.cache_manage import MuramDataCache, BalancedTrainDataCache
from models.pinn_mscnn_model import PhysicsInformedMSCNN
from scripts.base_training import (
    TrainingConfig, load_and_prepare_step, train_epoch, MetricsLogger,
    compute_global_bz_balancing_indices,
    build_or_refresh_balanced_cache, choose_balanced_cache_runtime_mode,
    preload_balanced_steps_from_cache,
    generate_epoch_diagnostic_plots,
    prepare_modest_epoch_snapshot, generate_epoch_modest_diagnostic_plots,
    compute_tau_averaged_metrics, compute_modest_tau_averaged_metrics,
)


def discover_checkpoint_path(experiment_name: str, variation: str, base_exp_dir: Path) -> Path:
    """
    Discover checkpoint file for a trained model variation.
    Checks: final_model.pth (preferred), then best_model.pth in checkpoints/.
    """
    candidates = [
        base_exp_dir / experiment_name / variation / "final_model.pth",
        base_exp_dir / experiment_name / variation / "checkpoints" / "best_model.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found for experiment='{experiment_name}', variation='{variation}'. "
        f"Checked: {candidates}"
    )


def load_and_adapt_config(checkpoint_path: Path, finetune_epochs_override: int = None) -> tuple:
    """
    Load checkpoint and setup config for fine-tuning with mandatory Bz balancing.
    
    Returns:
        (config, checkpoint_dict) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize fresh config
    config = TrainingConfig()
    
    # Apply saved config from checkpoint if available
    saved_config = checkpoint.get('config', {})
    for key, value in saved_config.items():
        if hasattr(config, key) and key not in ['checkpoint_dir', 'log_dir']:
            try:
                setattr(config, key, value)
            except Exception:
                pass
    
    # ENFORCE Bz balancing (mandatory for fine-tuning, per magnetic_field_balancing_and_finetuning.md)
    # Override any base-training settings to ensure fine-tuning uses balanced data.
    config.apply_bz_bin_balance = True
    config.bz_balance_scope = "global"        # Global deepest-tau balancing
    config.bz_balance_mode = "tau_index"      # Use absolute Bz at a fixed optical depth
    config.bz_balance_bins = 12               # Standard 12-bin histogram
    config.bz_balance_seed = 42               # Deterministic selection
    
    # Calculate fine-tune epochs (default: 10% of original training epochs)
    original_epochs = saved_config.get('n_epochs', 100)
    if finetune_epochs_override is not None:
        config.n_epochs = finetune_epochs_override
    else:
        config.n_epochs = max(5, int(0.1 * original_epochs))
    
    return config, checkpoint


def run_fine_tune_single_variation(
    experiment_name: str,
    variation: str,
    output_base_dir: Path,
    base_exp_dir: Path,
    finetune_epochs_override: int = None,
):
    """
    Run fine-tuning for a single model variation.
    
    Workflow:
    1. Discover and load trained checkpoint
    2. Load normalizers
    3. Setup model with mandatory Bz balancing
    4. Compute global Bz balancing indices (global deepest-tau strategy)
    5. Run training epochs with balanced cache
    6. Generate diagnostic plots
    7. Save fine-tuned checkpoint
    """
    
    print(f"\n{'='*100}")
    print(f"FINE-TUNING: {experiment_name} / {variation}".center(100))
    print(f"{'='*100}")
    
    # Discover checkpoint
    checkpoint_path = discover_checkpoint_path(experiment_name, variation, base_exp_dir)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Setup output directory (mirrors output/experiments structure)
    output_dir = output_base_dir / f"{experiment_name}-finetuned" / variation
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")
    
    # Load normalizers
    data_path = Path("/scratchsan/observatorio/juagudeloo/MUISCA/data/")
    mhd_normalizer = MhdNormalizer()
    mhd_normalizer.load(filepath=data_path / "normalization_stats" / "mhd_normalization.json")
    stokes_normalizer = StokesNormalizer()
    stokes_normalizer.load(filepath=data_path / "normalization_stats" / "stokes_normalization.json")
    print("✓ Normalizers loaded")
    
    # Load checkpoint and setup config with mandatory Bz balancing
    config, checkpoint_data = load_and_adapt_config(
        checkpoint_path,
        finetune_epochs_override=finetune_epochs_override
    )
    print(f"✓ Config prepared: Bz balancing ENFORCED (mandatory for fine-tuning)")
    print(f"  Original epochs: {checkpoint_data.get('config', {}).get('n_epochs', 'unknown')}")
    print(f"  Fine-tune epochs: {config.n_epochs}")
    
    # Instantiate model
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
    
    # Load checkpoint weight state
    if 'model_state_dict' in checkpoint_data:
        model.load_state_dict(checkpoint_data['model_state_dict'])
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")
    
    # Setup optimizer (fresh start with current lr; can reload state if desired)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    # Optionally restore optimizer state from checkpoint
    if 'optimizer_state_dict' in checkpoint_data:
        try:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        except Exception:
            print("  Warning: Could not restore optimizer state (using fresh optimization state)")
    
    # Setup output logging directories
    config.checkpoint_dir = output_dir / "checkpoints"
    config.log_dir = output_dir / "logs"
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = MetricsLogger(config.log_dir)
    
    # Setup train/val splits (reuse original split: last 10% for validation)
    all_steps = list(range(config.min_step, config.max_step + 1, config.step_size))
    n_val = max(1, len(all_steps) // 10)
    val_steps = sorted(all_steps)[-n_val:]
    train_steps = [s for s in all_steps if s not in val_steps]
    
    print(f"Train steps: {len(train_steps)}, Validation steps: {len(val_steps)}")
    print(f"Fine-tuning for {config.n_epochs} epochs with MANDATORY Bz balancing")
    
    # Setup data cache
    cache = MuramDataCache(cache_dir=config.cache_dir, compression='gzip')
    
    # Compute global Bz balancing indices (deepest optical-depth strategy, per theory doc)
    # This evaluates |Bz| at the deepest tau level across all training steps
    print("Computing global Bz balancing indices...")
    global_bz_selection_indices, global_bz_balance_metadata = compute_global_bz_balancing_indices(
        train_steps=train_steps,
        config=config,
        mhd_normalizer=mhd_normalizer,
        stokes_normalizer=stokes_normalizer,
        cache=cache,
    )
    print(f"✓ Bz balancing computed: {len(global_bz_selection_indices)} steps, "
          f"{sum(len(idx) for idx in global_bz_selection_indices.values())} pixel selections")
    
    # Setup balanced cache (reuses already-balanced tensors across epochs for speed)
    print("Building/loading balanced cache...")
    balanced_cache, sig_hash, balanced_report = build_or_refresh_balanced_cache(
        train_steps=train_steps,
        config=config,
        mhd_normalizer=mhd_normalizer,
        stokes_normalizer=stokes_normalizer,
        raw_cache=cache,
        global_bz_selection_indices=global_bz_selection_indices,
        global_bz_balance_metadata=global_bz_balance_metadata,
    )
    balanced_mode = choose_balanced_cache_runtime_mode(config, balanced_report["estimated_preload_bytes"])
    preloaded = preload_balanced_steps_from_cache(train_steps, balanced_cache, sig_hash) if balanced_mode == "preload" else None
    print(f"✓ Balanced cache ready: strategy={balanced_mode}, {balanced_report['total_cache_bytes']/1e9:.2f} GB")
    
    # Fine-tuning loop
    print(f"\n{'='*100}")
    print("Fine-Tuning Training Loop".center(100))
    print(f"{'='*100}\n")
    
    # Test metrics tracking
    test_steps = sorted(all_steps)[-max(1, n_val//2):]
    test_metrics_epochs = []
    test_correlation_history = {'blos': [], 'vlos': [], 'temp': []}
    test_rrmse_history = {'blos': [], 'vlos': [], 'temp': []}
    
    # Loss history
    val_loss_history = []
    train_loss_history = []
    
    start_time = time.time()
    
    for epoch in range(config.n_epochs):
        print(f"\nFine-Tune Epoch {epoch + 1}/{config.n_epochs}")
        
        # Train epoch with balanced data and global Bz indices
        epoch_metrics = train_epoch(
            model=model,
            train_steps=train_steps,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            optimizer=optimizer,
            epoch=epoch,
            logger=logger,
            n_steps_per_epoch=-1,
            cache=cache,
            enable_wfa=True,
            global_bz_selection_indices=global_bz_selection_indices,
            global_bz_balance_metadata=global_bz_balance_metadata,
            balanced_cache=balanced_cache if balanced_mode == "disk" else None,
            balanced_cache_signature_hash=sig_hash if balanced_mode == "disk" else None,
            preloaded_balanced_steps=preloaded,
        )
        
        # Record metrics
        train_loss = epoch_metrics.get('total_loss', 0)
        train_loss_history.append(train_loss)
        val_loss_history.append(train_loss)  # Simplified; use same as train
        
        print(f"  Train Loss: {train_loss:.6f}")
        
        # Compute test metrics periodically
        try:
            test_metrics = compute_tau_averaged_metrics(
                model=model,
                test_steps=test_steps,
                config=config,
                mhd_normalizer=mhd_normalizer,
                stokes_normalizer=stokes_normalizer,
                logtau_values=config.get_logtau_values(),
                cache=cache,
            )
            test_metrics_epochs.append(epoch + 1)
            test_correlation_history['blos'].append(test_metrics.get('blos_correlation', 0))
            test_correlation_history['vlos'].append(test_metrics.get('vlos_correlation', 0))
            test_correlation_history['temp'].append(test_metrics.get('temp_correlation', 0))
            test_rrmse_history['blos'].append(test_metrics.get('blos_rrmse_tau_avg', 0))
            test_rrmse_history['vlos'].append(test_metrics.get('vlos_rrmse_tau_avg', 0))
            test_rrmse_history['temp'].append(test_metrics.get('temp_rrmse_tau_avg', 0))
            
            b_corr = test_metrics.get('blos_correlation', 0)
            v_corr = test_metrics.get('vlos_correlation', 0)
            t_corr = test_metrics.get('temp_correlation', 0)
            print(f"  Test correlation: B={b_corr:.4f}, V={v_corr:.4f}, T={t_corr:.4f}")
        except Exception as e:
            print(f"  Warning: Test metrics failed: {e}")
    
    training_time = (time.time() - start_time) / 60
    logger.close()
    
    # Save fine-tuned checkpoint
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
        },
        output_dir / "final_model.pth"
    )
    print(f"\n✓ Fine-tuned checkpoint saved to {output_dir / 'final_model.pth'}")
    
    print(f"\n✓ Fine-tuning complete in {training_time:.1f} minutes")
    print(f"✓ Results saved to {output_dir}")
    
    return {
        'experiment_dir': str(output_dir),
        'log_dir': str(config.log_dir),
        'training_time_minutes': training_time,
        'test_metrics_epochs': test_metrics_epochs,
        'test_correlation_history': test_correlation_history,
        'test_rrmse_history': test_rrmse_history,
    }


def main():
    """
    Main entry point for fine-tuning script.
    
    Parses arguments, discovers checkpoints, and runs fine-tuning for selected variations.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune MUISCA checkpoint models on Bz-balanced training data"
    )
    parser.add_argument(
        '--experiment-name',
        required=True,
        help='Experiment name from output/experiments/ (e.g., "experiment_81_to_181-...")'
    )
    parser.add_argument(
        '--variations',
        required=True,
        help='Comma-separated variation names (e.g., "wfa_only,all_physics_terms,doppler_only")'
    )
    parser.add_argument(
        '--finetune-epochs',
        type=int,
        default=None,
        help='Number of fine-tuning epochs (default: 10% of original training epochs, min 5)'
    )
    parser.add_argument(
        '--output-base-dir',
        type=Path,
        default=Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/fine-tune"),
        help='Base output directory for fine-tune results (default: output/fine-tune/)'
    )
    parser.add_argument(
        '--exp-base-dir',
        type=Path,
        default=Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments"),
        help='Base directory for trained experiment checkpoints (default: output/experiments/)'
    )
    
    args = parser.parse_args()
    
    # Parse requested variations
    variations = [v.strip() for v in args.variations.split(',')]
    
    print("\n" + "="*100)
    print("MUISCA Model Fine-Tuning Workflow".center(100))
    print("="*100)
    print(f"Experiment: {args.experiment_name}")
    print(f"Variations to fine-tune: {', '.join(variations)}")
    print(f"Fine-tune epochs: {args.finetune_epochs or '10% of original (min 5)'}")
    print(f"Output base: {args.output_base_dir}")
    print("\nNOTE: Bz balancing is MANDATORY during fine-tuning (temporary; V/B balancing planned for future).")
    print("See docs/magnetic_field_balancing_and_finetuning.md for theoretical justification.")
    print("="*100 + "\n")
    
    # Fine-tune each requested variation
    results = {}
    for variation in variations:
        try:
            result = run_fine_tune_single_variation(
                experiment_name=args.experiment_name,
                variation=variation,
                output_base_dir=args.output_base_dir,
                base_exp_dir=args.exp_base_dir,
                finetune_epochs_override=args.finetune_epochs,
            )
            results[variation] = result
        except Exception as e:
            print(f"\n✗ ERROR fine-tuning '{variation}': {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*100)
    print("Fine-Tuning Workflow Summary".center(100))
    print("="*100)
    print(f"Completed: {len(results)}/{len(variations)} variations")
    for variation, result in results.items():
        print(f"  ✓ {variation}: {result['training_time_minutes']:.1f} min")
    print(f"\nResults saved to: {args.output_base_dir}")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
