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
)
# These live in ablation_study, not base_training -- importing them from base_training made
# this script fail at import time. ablation_study guards its entrypoint under __main__, so
# importing it here is side-effect free.
from scripts.experiments.ablation_study import compute_tau_averaged_metrics


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


# Keys present in experiment_config.json that describe what a run DID rather than how it was
# configured -- they have no TrainingConfig counterpart and must not be replayed.
_EXPERIMENT_CONFIG_REPORT_ONLY_KEYS = frozenset({
    "n_steps_per_epoch",
    "test_steps",
    "total_training_pixels_used",
})

# experiment_config.json records a couple of switches under result-style names that do not
# match their TrainingConfig attribute. They must still be replayed: apply_region_mask
# defaults to True, so skipping balanced_region_training left region masking ON for a base
# run that had it OFF, shrinking the balancing pool from 460,800 to 31,444 pixels.
_EXPERIMENT_CONFIG_KEY_ALIASES = {
    "balanced_region_training": "apply_region_mask",
    "balanced_bz_training": "apply_bz_bin_balance",
}

_EXPERIMENT_CONFIG_SECTIONS = ("data_config", "training_config", "model_config", "physics_config")


def source_step_exists(step: int, config: TrainingConfig) -> bool:
    """Whether a MURaM step has source data on disk for the config's data source.

    Mirrors the filenames load_source_arrays() resolves (base_training.py): nicole_tau500
    needs both stokes_<step>_nicole_tau500.npy and atmos_<step>_tau500.npz; muram_legacy
    needs stokes_<step>.npy.
    """
    sim_dir = Path(config.data_path) / "muram-simulation"
    if config.data_source == "nicole_tau500":
        return (sim_dir / f"stokes_{step}_nicole_tau500.npy").exists() and (
            sim_dir / f"atmos_{step}_tau500.npz"
        ).exists()
    return (sim_dir / f"stokes_{step}.npy").exists()


def available_source_steps(config: TrainingConfig) -> list:
    """Steps that do have source data on disk, for error messages."""
    sim_dir = Path(config.data_path) / "muram-simulation"
    if config.data_source == "nicole_tau500":
        pattern, strip = "stokes_*_nicole_tau500.npy", "_nicole_tau500"
    else:
        pattern, strip = "stokes_*.npy", ""
    steps = []
    for path in sim_dir.glob(pattern):
        token = path.stem.replace("stokes_", "").replace(strip, "")
        if token.isdigit():
            steps.append(int(token))
    return sorted(steps)


def load_experiment_config_json(checkpoint_path: Path) -> dict:
    """Load the experiment_config.json written alongside a trained checkpoint.

    Base training saves the run's real settings here, NOT inside the .pth (final_model.pth
    holds only model_state_dict plus metric history). Without this, fine-tuning silently
    falls back to TrainingConfig() defaults -- which means min_step=60/max_step=200/
    step_size=1, i.e. iterating 141 steps when only a handful were ever synthesized.

    Looks next to the checkpoint first, then one level up (for checkpoints/best_model.pth).
    Raises rather than returning empty: a silent fallback here produces a wrong run.
    """
    candidates = [
        checkpoint_path.parent / "experiment_config.json",
        checkpoint_path.parent.parent / "experiment_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "r") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No experiment_config.json found next to checkpoint {checkpoint_path}.\n"
        f"Checked: {[str(c) for c in candidates]}\n"
        "Fine-tuning needs the base run's real settings (step range, logtau grid, "
        "bz_balance_tau_idx). Without it the run would fall back to TrainingConfig() "
        "defaults and iterate steps that were never synthesized."
    )


def apply_experiment_config(config: TrainingConfig, experiment_config: dict) -> dict:
    """Replay a base run's experiment_config.json onto a TrainingConfig. Returns applied keys."""
    applied = {}
    for section in _EXPERIMENT_CONFIG_SECTIONS:
        for key, value in (experiment_config.get(section) or {}).items():
            if key in _EXPERIMENT_CONFIG_REPORT_ONLY_KEYS or key in ("checkpoint_dir", "log_dir"):
                continue
            attr = _EXPERIMENT_CONFIG_KEY_ALIASES.get(key, key)
            if not hasattr(config, attr):
                continue
            try:
                setattr(config, attr, value)
                applied[attr] = value
            except Exception:
                pass
    return applied


def load_and_adapt_config(
    checkpoint_path: Path,
    finetune_epochs_override: int = None,
    step_overrides: dict = None,
    balance_overrides: dict = None,
) -> tuple:
    """
    Load checkpoint and setup config for fine-tuning with mandatory Bz balancing.

    Returns:
        (config, checkpoint_dict) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Initialize fresh config
    config = TrainingConfig()

    # Primary source of truth: the base run's experiment_config.json on disk.
    experiment_config = load_experiment_config_json(checkpoint_path)
    applied = apply_experiment_config(config, experiment_config)
    print(f"✓ Restored {len(applied)} settings from experiment_config.json")

    # Secondary: a config embedded in the checkpoint, if this one happens to carry it.
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
    # Bin shape: quantile edges under a cap, equalized upward. Overridable via CLI.
    balance_overrides = balance_overrides or {}
    config.bz_balance_bins = balance_overrides.get("bz_balance_bins", config.bz_balance_bins)
    config.bz_balance_cap = balance_overrides.get("bz_balance_cap", config.bz_balance_cap)
    config.bz_balance_oversample = balance_overrides.get(
        "bz_balance_oversample", config.bz_balance_oversample
    )

    # tau_index mode needs the depth to score at. It comes from the base run (restored
    # above); if it is missing the balancer would silently fall back to n_tau // 2 -- a
    # different height than the one the base model was trained to balance on.
    if getattr(config, "bz_balance_tau_idx", None) is None:
        raise ValueError(
            "bz_balance_mode='tau_index' is enforced for fine-tuning, but bz_balance_tau_idx "
            "is unset. It should come from experiment_config.json's data_config; without it "
            "the balancer would silently score at n_tau//2 instead of the base run's depth."
        )

    # Optional step-range override (fine-tune on a different, e.g. stronger-field, range).
    # Normalizers are deliberately NOT recomputed for this -- see run_fine_tune_single_variation.
    if step_overrides:
        for key, value in step_overrides.items():
            if key == "steps":
                continue  # explicit list, consumed directly when building the split
            setattr(config, key, value)

    # Calculate fine-tune epochs (default: 10% of original training epochs)
    original_epochs = (
        saved_config.get('n_epochs')
        or (experiment_config.get('training_config') or {}).get('n_epochs')
        or 100
    )
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
    step_overrides: dict = None,
    balance_overrides: dict = None,
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
    
    # Load checkpoint and setup config with mandatory Bz balancing. This must happen BEFORE
    # the normalizers are loaded, since their paths are data-source dependent and come from
    # the config.
    config, checkpoint_data = load_and_adapt_config(
        checkpoint_path,
        finetune_epochs_override=finetune_epochs_override,
        step_overrides=step_overrides,
        balance_overrides=balance_overrides,
    )
    print(f"✓ Config prepared: Bz balancing ENFORCED (mandatory for fine-tuning)")
    print(f"  Data source:      {config.data_source}")
    print(f"  Step range:       {config.min_step}..{config.max_step} step {config.step_size}")
    print(f"  Bz balance depth: tau_idx={config.bz_balance_tau_idx}")
    print(f"  Fine-tune epochs: {config.n_epochs}")

    # Load the SAME normalizers the base run used, resolved from the config so they follow
    # the data source (mirrors base_training.py). Fine-tuning must never refit these: the Bz
    # asinh scale B0_transform_per_tau is baked into the pretrained weights' output space
    # (physical Bz = B0 * sinh(pred * std + mean)), so changing it would silently reinterpret
    # every prediction. This holds even when --steps selects a different range.
    data_path = Path(config.data_path)
    mhd_norm_path = data_path / config.mhd_normalizer_path
    stokes_norm_path = data_path / config.stokes_normalizer_path
    mhd_normalizer = MhdNormalizer()
    mhd_normalizer.load(filepath=mhd_norm_path)
    stokes_normalizer = StokesNormalizer()
    stokes_normalizer.load(filepath=stokes_norm_path)
    print(f"✓ Normalizers loaded from {mhd_norm_path.parent}")

    # The normalizer and the checkpoint must agree on the optical-depth grid, or every
    # denormalized prediction is meaningless. Mismatches here are silent otherwise.
    if int(mhd_normalizer.n_tau) != int(config.get_n_logtau()):
        raise ValueError(
            f"Normalizer/config optical-depth mismatch: {mhd_norm_path} has n_tau="
            f"{mhd_normalizer.n_tau}, but the config's logtau grid has {config.get_n_logtau()} "
            f"levels (data_source={config.data_source}). Recompute normalization stats for "
            "this data source, or point data_source at the one these stats were built for."
        )

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
    explicit_steps = (step_overrides or {}).get("steps")
    if explicit_steps:
        all_steps = sorted(int(s) for s in explicit_steps)
    else:
        all_steps = list(range(config.min_step, config.max_step + 1, config.step_size))

    # Fail before training rather than partway through it: only a subset of MURaM steps has
    # ever been synthesized for a given data source.
    missing = [s for s in all_steps if not source_step_exists(s, config)]
    if missing:
        raise FileNotFoundError(
            f"Requested fine-tuning steps have no {config.data_source} Stokes on disk: {missing}\n"
            f"Available under {Path(config.data_path) / 'muram-simulation'}: "
            f"{available_source_steps(config)}\n"
            "Synthesize them first, or pass --steps with ones that exist."
        )

    n_val = max(1, len(all_steps) // 10)
    val_steps = sorted(all_steps)[-n_val:]
    train_steps = [s for s in all_steps if s not in val_steps]

    print(f"Steps: {all_steps}")
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
    # build_or_refresh_balanced_cache reports 'total_disk_bytes'; the old key name here
    # ('total_cache_bytes') never existed and raised KeyError on every run.
    print(
        f"✓ Balanced cache ready: strategy={balanced_mode}, "
        f"{balanced_report.get('total_disk_bytes', 0)/1e9:.2f} GB, "
        f"{balanced_report.get('total_selected', 0)} pixels selected"
    )
    
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
    # Stringify Paths before saving: config carries checkpoint_dir/log_dir as PosixPath, and
    # torch.load's weights_only=True default (which the analysis pipeline uses) refuses to
    # unpickle them, making the resulting checkpoint unreadable by scripts/analysis/*.
    serializable_config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in config.__dict__.items()
    }
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': serializable_config,
        },
        output_dir / "final_model.pth"
    )
    print(f"\n✓ Fine-tuned checkpoint saved to {output_dir / 'final_model.pth'}")

    # Mirror the base run's experiment_config.json into the fine-tune output. The analysis
    # pipeline reads the model's optical-depth grid from data_config.logtau_values here
    # (utils/analysis.py: get_model_logtau_values); without it every fine-tuned model is
    # skipped with "optical depth nodes: not found". Start from the base run's file so the
    # rest of the provenance carries over, then overwrite what fine-tuning changed.
    finetuned_config = load_experiment_config_json(checkpoint_path)
    finetuned_config.setdefault("data_config", {}).update({
        "logtau_values": [float(v) for v in config.get_logtau_values().tolist()],
        "min_step": int(config.min_step),
        "max_step": int(config.max_step),
        "step_size": int(config.step_size),
        "bz_balance_tau_idx": int(config.bz_balance_tau_idx),
        "bz_balance_bins": int(config.bz_balance_bins),
        "bz_balance_scope": str(config.bz_balance_scope),
        "bz_balance_mode": str(config.bz_balance_mode),
        "bz_balance_bin_scale": str(config.bz_balance_bin_scale),
        "bz_balance_oversample": bool(config.bz_balance_oversample),
        "balanced_bz_training": True,
        "balanced_region_training": bool(config.apply_region_mask),
    })
    finetuned_config.setdefault("training_config", {})["n_epochs"] = int(config.n_epochs)
    # Leave experiment_name alone: the analysis pipeline uses it as the model-variation key
    # (it holds "wfa_only", not the experiment folder), so renaming it makes --model-types
    # stop matching. Record the provenance in a separate field instead.
    finetuned_config["finetuned_from"] = str(checkpoint_path)
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(finetuned_config, f, indent=2)
    print(f"✓ Config written to {output_dir / 'experiment_config.json'}")
    
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
        default=ROOT / "output" / "fine-tune",
        help='Base output directory for fine-tune results (default: output/fine-tune/)'
    )
    parser.add_argument(
        '--exp-base-dir',
        type=Path,
        default=ROOT / "output" / "experiments",
        help='Base directory for trained experiment checkpoints (default: output/experiments/)'
    )
    # Step selection. Default (both unset) replays the base run's range from
    # experiment_config.json. Overriding does NOT refit the normalizers -- see
    # run_fine_tune_single_variation.
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=None,
        help='Explicit step list to fine-tune on, e.g. --steps 120 130 (overrides the base '
             'run range; preferred, since only some steps are synthesized)'
    )
    parser.add_argument('--min-step', type=int, default=None,
                        help='Override the base run\'s minimum step (inclusive)')
    parser.add_argument('--max-step', type=int, default=None,
                        help='Override the base run\'s maximum step (inclusive)')
    parser.add_argument('--step-size', type=int, default=None,
                        help='Override the base run\'s step stride')
    # Bz balancing shape. Defaults match TrainingConfig: quantile bins capped at the
    # scores' p99.9, equalized upward by oversampling rare bins.
    parser.add_argument('--bz-balance-bins', type=int, default=None,
                        help='Number of |B_LOS| bins to balance across (default: 12)')
    parser.add_argument('--bz-balance-cap', type=float, default=None,
                        help='|B_LOS| ceiling in G above which pixels share the top bin '
                             '(default: p99.9 of the scores). Bounds balancing into the '
                             'tail, where too few distinct pixels exist to learn from.')
    parser.add_argument('--no-bz-oversample', action='store_true',
                        help='Equalize bins downward to the rarest one (original behavior). '
                             'On MURaM this keeps ~0.01%% of the pool -- almost never what '
                             'you want.')

    args = parser.parse_args()

    # Parse requested variations
    variations = [v.strip() for v in args.variations.split(',')]

    # Collect only the step settings actually passed; anything left out keeps the base run's
    # value from experiment_config.json.
    step_overrides = {}
    if args.steps:
        step_overrides["steps"] = args.steps
    if args.min_step is not None:
        step_overrides["min_step"] = args.min_step
    if args.max_step is not None:
        step_overrides["max_step"] = args.max_step
    if args.step_size is not None:
        step_overrides["step_size"] = args.step_size

    balance_overrides = {}
    if args.bz_balance_bins is not None:
        balance_overrides["bz_balance_bins"] = args.bz_balance_bins
    if args.bz_balance_cap is not None:
        balance_overrides["bz_balance_cap"] = args.bz_balance_cap
    if args.no_bz_oversample:
        balance_overrides["bz_balance_oversample"] = False

    print("\n" + "="*100)
    print("MUISCA Model Fine-Tuning Workflow".center(100))
    print("="*100)
    print(f"Experiment: {args.experiment_name}")
    print(f"Variations to fine-tune: {', '.join(variations)}")
    print(f"Fine-tune epochs: {args.finetune_epochs or '10% of original (min 5)'}")
    print(f"Steps: {'from base run config' if not step_overrides else step_overrides}")
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
                step_overrides=step_overrides,
                balance_overrides=balance_overrides,
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
