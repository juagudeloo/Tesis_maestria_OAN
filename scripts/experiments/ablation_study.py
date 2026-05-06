"""
Physics Regularization Ablation Study
======================================

Compares PINN training with different regularization configurations:
1. No physics (pure supervised learning)
2. WFA only
3. All physics terms (WFA + Gradient smoothness)

Usage:
    python first_experiment.py --n_epochs 50 --device cuda
"""

import sys
import os
import json
import time
from pathlib import Path
import warnings
import contextlib
from scipy.optimize import OptimizeWarning

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directories to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.normalizer import MhdNormalizer, StokesNormalizer
from models.pinn_mscnn_model import PhysicsInformedMSCNN
from utils.cache_manage import MuramDataCache, BalancedTrainDataCache
from scripts.base_training import (
    TrainingConfig,
    load_and_prepare_step, validate, train_epoch, MetricsLogger,
    initialize_wfa_gate_state, update_wfa_gate_state,
    compute_global_bz_balancing_indices,
    build_or_refresh_balanced_cache, choose_balanced_cache_runtime_mode,
    preload_balanced_steps_from_cache,
    generate_epoch_diagnostic_plots, generate_epoch_diagnostic_videos,
    prepare_modest_epoch_snapshot, generate_epoch_modest_diagnostic_plots,
    generate_epoch_modest_diagnostic_videos,
)


def _series_for_log_plot(values) -> np.ndarray:
    """Convert non-positive values to NaN for stable log-scale plotting."""
    arr = np.asarray(values, dtype=float)
    return np.where(arr > 0, arr, np.nan)


def _get_active_range(values) -> int:
    """
    Find the first index where a loss value becomes non-zero.
    Used to avoid plotting zero values as a continuous line on log scale.
    
    Returns the first index where value > 0, or None if all values are <= 0.
    """
    arr = np.asarray(values, dtype=float)
    nonzero_indices = np.where(arr > 0)[0]
    return nonzero_indices[0] if len(nonzero_indices) > 0 else None


def _denormalize_targets_from_stats(
    normalized_targets: np.ndarray,
    param: str,
    mhd_normalizer: MhdNormalizer,
) -> np.ndarray:
    """Denormalize normalized targets using stored stats without clipping."""
    if param not in {'T', 'Vz', 'Bz'}:
        raise ValueError(f"Unsupported parameter: {param}")

    if not mhd_normalizer.finalized or mhd_normalizer.final_stats is None:
        raise RuntimeError("MHD normalizer must be finalized before denormalization")

    denorm = np.zeros_like(normalized_targets, dtype=np.float32)
    param_stats = mhd_normalizer.final_stats[param]

    for tau_idx in range(normalized_targets.shape[1]):
        stats_tau = param_stats[tau_idx]
        std = float(stats_tau['std'])
        mean = float(stats_tau['mean'])

        if stats_tau.get('type') == 'centered':
            values_tau = normalized_targets[:, tau_idx] * std
        else:
            values_tau = (normalized_targets[:, tau_idx] * std) + mean

        if param == 'Bz':
            B0_tau = float(mhd_normalizer.B0_transform_per_tau[tau_idx])
            values_tau = B0_tau * np.sinh(values_tau)

        denorm[:, tau_idx] = values_tau.astype(np.float32, copy=False)

    return denorm


def generate_training_data_histograms(
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    train_steps: list[int],
    output_dir: Path,
    cache: MuramDataCache | None = None,
    global_bz_selection_indices: dict[int, np.ndarray] | None = None,
    global_bz_balance_metadata: dict | None = None,
    bins: int = 120,
    max_samples_per_param: int = 400000,
    force_recompute: bool = False,
) -> tuple[Path, Path]:
    """Generate per-τ histograms and summary stats for training-target value ranges."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_index_path = output_dir / "training_data_histograms_train_split_per_tau_index.json"
    stats_path = output_dir / "training_data_histograms_train_split_stats.json"
    stokes_plot_path = output_dir / "training_data_histograms_train_split_stokes_mean_std_profiles.png"
    hist_paths = {
        'T': output_dir / "training_data_histograms_train_split_per_tau_T.png",
        'Vz': output_dir / "training_data_histograms_train_split_per_tau_Vz.png",
        'Bz': output_dir / "training_data_histograms_train_split_per_tau_Bz.png",
    }

    if (
        all(p.exists() for p in hist_paths.values())
        and stokes_plot_path.exists()
        and stats_path.exists()
        and hist_index_path.exists()
        and not force_recompute
    ):
        print(f"Training-data per-τ histograms already exist, reusing: {hist_index_path}")
        return hist_index_path, stats_path

    rng = np.random.default_rng(42)
    sample_per_step = max(2_000, min(max_samples_per_param, 20_000))
    n_tau = config.get_n_logtau()
    logtau_vals = np.asarray(config.get_logtau_values(), dtype=float)
    hinode_wl = None

    buckets = {param: [[] for _ in range(n_tau)] for param in ['T', 'Vz', 'Bz']}
    totals = {param: np.zeros(n_tau, dtype=np.int64) for param in ['T', 'Vz', 'Bz']}
    mins = {param: np.full(n_tau, np.inf, dtype=np.float64) for param in ['T', 'Vz', 'Bz']}
    maxs = {param: np.full(n_tau, -np.inf, dtype=np.float64) for param in ['T', 'Vz', 'Bz']}
    stokes_buckets = {'I': [], 'V': []}
    stokes_totals = {'I': 0, 'V': 0}

    print("\nCollecting training-value samples for histogram diagnostics...")
    for step in tqdm(train_steps, desc="Histogram data (train split)"):
        result = load_and_prepare_step(
            step=step,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            cache=cache,
            apply_balanced_masks=config.apply_region_mask,
            log_region_stats=False,
            apply_bz_balance=(config.apply_bz_bin_balance and config.bz_balance_scope == "per_step"),
            global_bz_selection_indices=global_bz_selection_indices,
            global_bz_balance_metadata=global_bz_balance_metadata,
            ignore_missing_files=True,
        )
        if result is None:
            continue

        dataset, _ = result
        if hinode_wl is None and getattr(dataset, "hinode_wl", None) is not None:
            hinode_wl = np.asarray(dataset.hinode_wl, dtype=np.float64)

        targets = dataset.mhd_targets
        T_norm = targets[:, :n_tau]
        Vz_norm = targets[:, n_tau:2 * n_tau]
        Bz_norm = targets[:, 2 * n_tau:3 * n_tau]

        recovered = {
            'T': _denormalize_targets_from_stats(T_norm, 'T', mhd_normalizer),
            'Vz': _denormalize_targets_from_stats(Vz_norm, 'Vz', mhd_normalizer),
            'Bz': _denormalize_targets_from_stats(Bz_norm, 'Bz', mhd_normalizer),
        }

        stokes_input = np.asarray(dataset.stokes_input, dtype=np.float32)
        if stokes_input.ndim == 3 and stokes_input.shape[1] == 2 and stokes_input.shape[2] == 112:
            stokes_step = {
                'I': stokes_input[:, 0, :],
                'V': stokes_input[:, 1, :],
            }
            for key in ('I', 'V'):
                values = stokes_step[key]
                stokes_totals[key] += int(values.shape[0])
                if values.shape[0] > sample_per_step:
                    idx = rng.choice(values.shape[0], size=sample_per_step, replace=False)
                    stokes_buckets[key].append(values[idx])
                else:
                    stokes_buckets[key].append(values)

        for param, values in recovered.items():
            values = np.asarray(values, dtype=np.float32)
            for tau_idx in range(n_tau):
                values_tau = values[:, tau_idx]
                totals[param][tau_idx] += int(values_tau.size)
                mins[param][tau_idx] = min(mins[param][tau_idx], float(np.min(values_tau)))
                maxs[param][tau_idx] = max(maxs[param][tau_idx], float(np.max(values_tau)))

                if values_tau.size > sample_per_step:
                    idx = rng.choice(values_tau.size, size=sample_per_step, replace=False)
                    sampled_tau = values_tau[idx]
                else:
                    sampled_tau = values_tau
                buckets[param][tau_idx].append(sampled_tau)

    sampled_values = {param: [] for param in ['T', 'Vz', 'Bz']}
    for param in ['T', 'Vz', 'Bz']:
        for tau_idx in range(n_tau):
            parts = buckets[param][tau_idx]
            if len(parts) == 0:
                sampled_values[param].append(np.empty((0,), dtype=np.float32))
                continue
            merged = np.concatenate(parts)
            if merged.size > max_samples_per_param:
                idx = rng.choice(merged.size, size=max_samples_per_param, replace=False)
                merged = merged[idx]
            sampled_values[param].append(merged)

    sampled_stokes = {}
    for key in ('I', 'V'):
        parts = stokes_buckets[key]
        if len(parts) == 0:
            sampled_stokes[key] = np.empty((0, 112), dtype=np.float32)
            continue
        merged = np.concatenate(parts, axis=0)
        if merged.shape[0] > max_samples_per_param:
            idx = rng.choice(merged.shape[0], size=max_samples_per_param, replace=False)
            merged = merged[idx]
        sampled_stokes[key] = merged.astype(np.float32, copy=False)

    labels = {
        'T': 'Temperature T [K]',
        'Vz': 'Velocity Vz [km/s]',
        'Bz': 'Magnetic field Bz [G]',
    }
    colors = {'T': 'tab:red', 'Vz': 'tab:blue', 'Bz': 'tab:green'}

    n_cols = int(min(7, n_tau))
    n_rows = int(np.ceil(n_tau / n_cols))

    stats_payload = {
        'train_steps': [int(s) for s in train_steps],
        'apply_region_mask': bool(config.apply_region_mask),
        'ready_for_training_data': True,
        'preprocessing_stage': (
            'histograms computed from load_and_prepare_step outputs after normalization '
            'and final pixel selection used by training'
        ),
        'apply_bz_bin_balance': bool(config.apply_bz_bin_balance),
        'bz_balance_scope': str(config.bz_balance_scope),
        'bz_balance_mode': str(config.bz_balance_mode),
        'bz_balance_bins': int(config.bz_balance_bins),
        'bz_balance_tau_idx': None if config.bz_balance_tau_idx is None else int(config.bz_balance_tau_idx),
        'bz_balance_global_reference_logtau': (
            None if not isinstance(global_bz_balance_metadata, dict)
            else global_bz_balance_metadata.get('reference_logtau')
        ),
        'n_logtau': int(n_tau),
        'logtau_values': [float(v) for v in logtau_vals.tolist()],
        'max_samples_per_param': int(max_samples_per_param),
        'sample_per_step': int(sample_per_step),
        'parameters': {},
    }

    for param in ['T', 'Vz', 'Bz']:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 2.6), squeeze=False)
        fig.suptitle(
            f"Training Dataset Distribution per optical depth - {labels[param]} (train split)",
            fontsize=13,
            fontweight='bold',
        )

        per_tau_stats = []
        for tau_idx in range(n_tau):
            row = tau_idx // n_cols
            col = tau_idx % n_cols
            ax = axes[row, col]
            values = sampled_values[param][tau_idx]
            tau_label = f"logτ={logtau_vals[tau_idx]:.2f}"

            if values.size == 0:
                ax.set_title(f"{tau_label}\n(no samples)", fontsize=9)
                ax.axis('off')
                per_tau_stats.append({
                    'tau_idx': int(tau_idx),
                    'logtau': float(logtau_vals[tau_idx]),
                    'n_values_total': 0,
                    'n_values_sampled': 0,
                })
                continue

            p01, p05, p50, p95, p99 = np.percentile(values, [1, 5, 50, 95, 99])
            ax.hist(values, bins=bins, color=colors[param], alpha=0.75, edgecolor='black', linewidth=0.2)
            ax.axvline(p01, color='k', linestyle='--', linewidth=0.8, alpha=0.75)
            ax.axvline(p99, color='k', linestyle='--', linewidth=0.8, alpha=0.75)
            ax.axvline(p50, color='goldenrod', linestyle='-', linewidth=0.9, alpha=0.9)
            ax.set_title(tau_label, fontsize=9)
            ax.grid(True, alpha=0.2)
            if row == n_rows - 1:
                ax.set_xlabel('Value', fontsize=8)
            if col == 0:
                ax.set_ylabel('Count', fontsize=8)
            ax.tick_params(axis='both', labelsize=7)

            per_tau_stats.append({
                'tau_idx': int(tau_idx),
                'logtau': float(logtau_vals[tau_idx]),
                'n_values_total': int(totals[param][tau_idx]),
                'n_values_sampled': int(values.size),
                'min': float(mins[param][tau_idx]),
                'max': float(maxs[param][tau_idx]),
                'p1_sampled': float(p01),
                'p5_sampled': float(p05),
                'p50_sampled': float(p50),
                'p95_sampled': float(p95),
                'p99_sampled': float(p99),
            })

        for k in range(n_tau, n_rows * n_cols):
            row = k // n_cols
            col = k % n_cols
            axes[row, col].axis('off')

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(hist_paths[param], dpi=170, bbox_inches='tight')
        plt.close(fig)

        stats_payload['parameters'][param] = {
            'plot_file': str(hist_paths[param]),
            'per_tau': per_tau_stats,
        }

    stokes_stats = stokes_normalizer.final_stats if stokes_normalizer.final_stats is not None else {}
    mean_i = float(stokes_stats.get('I', {}).get('mean', 0.0))
    std_i = float(stokes_stats.get('I', {}).get('std', 1.0))
    mean_v = float(stokes_stats.get('V', {}).get('mean', 0.0))
    std_v = float(stokes_stats.get('V', {}).get('std', 1.0))

    I_norm = sampled_stokes['I']
    V_norm = sampled_stokes['V']
    if hinode_wl is None:
        raise RuntimeError(
            "Hinode wavelength grid is unavailable in dataset payload. "
            "Clear stale cache and rerun to populate stokes_data['hinode_wl']."
        )
    if I_norm.shape[0] > 0 and V_norm.shape[0] > 0:
        I_mean_norm = np.mean(I_norm, axis=0)
        I_std_norm = np.std(I_norm, axis=0)
        V_mean_norm = np.mean(V_norm, axis=0)
        V_std_norm = np.std(V_norm, axis=0)

        I_mean_den = I_mean_norm * std_i + mean_i
        I_std_den = I_std_norm * abs(std_i)
        V_mean_den = V_mean_norm * std_v + mean_v
        V_std_den = V_std_norm * abs(std_v)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(
            "Training Stokes Profiles (final preprocessing stage): mean ± 1σ",
            fontsize=13,
            fontweight='bold',
        )

        axes[0].plot(hinode_wl, I_mean_den, color='tab:orange', linewidth=1.8, label='Mean I')
        axes[0].fill_between(
            hinode_wl,
            I_mean_den - I_std_den,
            I_mean_den + I_std_den,
            color='tab:orange',
            alpha=0.25,
            label='±1σ',
        )
        axes[0].set_ylabel('Stokes I')
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc='best', fontsize=9)

        axes[1].plot(hinode_wl, V_mean_den, color='tab:purple', linewidth=1.8, label='Mean V')
        axes[1].fill_between(
            hinode_wl,
            V_mean_den - V_std_den,
            V_mean_den + V_std_den,
            color='tab:purple',
            alpha=0.25,
            label='±1σ',
        )
        axes[1].set_xlabel('Wavelength [Angstrom]')
        axes[1].set_ylabel('Stokes V')
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc='best', fontsize=9)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(stokes_plot_path, dpi=170, bbox_inches='tight')
        plt.close(fig)

        stats_payload['stokes_profiles'] = {
            'plot_file': str(stokes_plot_path),
            'wavelength_angstrom': [float(v) for v in hinode_wl.tolist()],
            'n_values_total': {
                'I': int(stokes_totals['I']),
                'V': int(stokes_totals['V']),
            },
            'n_values_sampled': {
                'I': int(I_norm.shape[0]),
                'V': int(V_norm.shape[0]),
            },
            'mean_denormalized': {
                'I': [float(v) for v in I_mean_den.tolist()],
                'V': [float(v) for v in V_mean_den.tolist()],
            },
            'std_denormalized': {
                'I': [float(v) for v in I_std_den.tolist()],
                'V': [float(v) for v in V_std_den.tolist()],
            },
        }
    else:
        stats_payload['stokes_profiles'] = {
            'plot_file': str(stokes_plot_path),
            'wavelength_angstrom': [float(v) for v in hinode_wl.tolist()],
            'n_values_total': {
                'I': int(stokes_totals['I']),
                'V': int(stokes_totals['V']),
            },
            'n_values_sampled': {
                'I': int(I_norm.shape[0]),
                'V': int(V_norm.shape[0]),
            },
            'warning': 'No Stokes samples were available to build mean±std profiles.',
        }

    with open(stats_path, 'w') as f:
        json.dump(stats_payload, f, indent=2)

    with open(hist_index_path, 'w') as f:
        index_payload = {k: str(v) for k, v in hist_paths.items()}
        index_payload['stokes_mean_std'] = str(stokes_plot_path)
        json.dump(index_payload, f, indent=2)

    print(f"Saved per-τ histogram index:      {hist_index_path}")
    for param in ['T', 'Vz', 'Bz']:
        print(f"Saved per-τ histogram ({param}):   {hist_paths[param]}")
    print(f"Saved Stokes mean±std profile:      {stokes_plot_path}")
    print(f"Saved histogram stats:           {stats_path}")
    return hist_index_path, stats_path


class ExperimentTracker:
    """Tracks metrics across different experimental conditions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results = {}
    def add_experiment(self, name: str, metrics: dict):
        """Add results from one experimental condition."""
        self.results[name] = metrics
    def save_results(self):
        """Save all results to JSON."""
        results_path = self.output_dir / "experiment_results.json"
        training_pixels_by_experiment = {
            name: int(metrics.get('total_training_pixels_used', 0))
            for name, metrics in self.results.items()
        }
        payload = dict(self.results)
        payload['__metadata__'] = {
            'total_training_pixels_used': int(sum(training_pixels_by_experiment.values())),
            'training_pixels_used_by_experiment': training_pixels_by_experiment,
            'n_experiments': int(len(self.results)),
        }
        with open(results_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {results_path}")
    
    def print_summary_table(self):
        """Print a formatted summary table."""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "=" * 140)
        print("PHYSICS REGULARIZATION ABLATION STUDY - SUMMARY".center(140))
        print("=" * 140)
        
        # Header
        header = f"{'Experiment':<25} {'Val Loss':<12} {'Time (min)':<12} {'B_LOS RRMSE':<15} {'V_LOS RRMSE':<15} {'Temp RRMSE':<15} {'Best?':<8}"
        print(header)
        print("-" * 140)
        
        # Find best model
        best_exp = min(self.results.keys(), 
                      key=lambda x: self.results[x]['final_val_loss'])
        
        # Print rows
        for exp_name in self.results.keys():
            metrics = self.results[exp_name]
            is_best = "★ YES" if exp_name == best_exp else ""
            
            row = (f"{exp_name:<25} "
                  f"{metrics['final_val_loss']:<12.6f} "
                  f"{metrics['training_time_minutes']:<12.1f} "
                  f"{metrics['test_metrics']['blos_rrmse_tau_avg']:<15.6f} "
                  f"{metrics['test_metrics']['vlos_rrmse_tau_avg']:<15.6f} "
                  f"{metrics['test_metrics']['temp_rrmse_tau_avg']:<15.6f} "
                  f"{is_best:<8}")
            print(row)
        
        print("=" * 140)
        
        # Print key findings
        print("\n📊 KEY FINDINGS:")
        print("-" * 140)
        
        baseline_name = 'no_physics'
        if baseline_name in self.results and best_exp != baseline_name:
            baseline_loss = self.results[baseline_name]['final_val_loss']
            best_loss = self.results[best_exp]['final_val_loss']
            improvement = (baseline_loss - best_loss) / baseline_loss * 100
            
            print(f"✓ Best configuration: {best_exp}")
            print(f"✓ Improvement over no physics: {improvement:.2f}%")
            print(f"✓ Validation loss: {best_loss:.6f} vs {baseline_loss:.6f} (baseline)")
        elif best_exp == baseline_name:
            print("⚠ Physics regularization did NOT improve performance")
            print("  Recommendation: Check lambda values or physics approximation quality")
        
        print("=" * 140 + "\n")
    
    def plot_individual_loss_curves(self):
        """Generate individual plots for each experiment showing all loss components."""
        for exp_name, results in self.results.items():
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'Loss Components - {exp_name}', fontsize=14, fontweight='bold')
            
            epochs = range(1, len(results['train_loss_history']) + 1)
            
            # Total loss
            ax1 = axes[0, 0]
            ax1.plot(epochs, _series_for_log_plot(results['train_loss_history']), 'b-o', label='Total Loss', linewidth=2)
            _activation_idx = _get_active_range(results.get('wfa_loss_history') or results.get('physics_loss_history') or [])
            if _activation_idx is not None:
                ax1.axvline(x=list(epochs)[_activation_idx], color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Physics on')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Total Training Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # MSE loss
            ax2 = axes[0, 1]
            if 'mse_loss_history' in results:
                ax2.plot(epochs, _series_for_log_plot(results['mse_loss_history']), 'g-s', label='MSE Loss', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('MSE Loss Component')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            # Physics loss breakdown
            ax3 = axes[1, 0]
            if 'physics_loss_history' in results and any(l > 0 for l in results['physics_loss_history']):
                ax3.plot(epochs, _series_for_log_plot(results['physics_loss_history']), 'r-^', label='Total Physics', linewidth=2)
            if 'wfa_loss_history' in results and any(l > 0 for l in results['wfa_loss_history']):
                wfa_start = _get_active_range(results['wfa_loss_history'])
                if wfa_start is not None:
                    ax3.plot(epochs[wfa_start:], _series_for_log_plot(results['wfa_loss_history'][wfa_start:]), 'm--', label='WFA', linewidth=1.5)
            if 'doppler_loss_history' in results and any(l > 0 for l in results['doppler_loss_history']):
                doppler_start = _get_active_range(results['doppler_loss_history'])
                if doppler_start is not None:
                    ax3.plot(epochs[doppler_start:], _series_for_log_plot(results['doppler_loss_history'][doppler_start:]), 'c--', label='Doppler', linewidth=1.5)
            if 'temperature_loss_history' in results and any(l > 0 for l in results['temperature_loss_history']):
                temp_start = _get_active_range(results['temperature_loss_history'])
                if temp_start is not None:
                    ax3.plot(epochs[temp_start:], _series_for_log_plot(results['temperature_loss_history'][temp_start:]), 'y--', label='Temperature', linewidth=1.5)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Physics Loss Components')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            ax3.set_xlim(min(epochs), max(epochs))
            
            # Validation loss
            ax4 = axes[1, 1]
            ax4.plot(epochs, _series_for_log_plot(results['val_loss_history']), 'orange', marker='o', label='Validation Loss', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Validation Loss')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')

            # Empty panels kept for layout symmetry
            axes[0, 2].axis('off')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Save individual plot
            plot_path = self.output_dir / f"{exp_name}-loss_curves.png"
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
            
        print(f"Individual loss curve plots saved to {self.output_dir}")

    def plot_testset_correlation_and_rrmse(self):
        """Generate per-experiment test-set correlation/RRMSE evolution plots and persist CSV metrics."""
        labels = ['B_LOS', 'V_LOS', 'Temp']

        for exp_name, results in self.results.items():
            test_metrics = results.get('test_metrics', {})
            if not test_metrics:
                continue

            corr_history = results.get('test_correlation_history', {})
            rrmse_history = results.get('test_rrmse_history', {})
            epochs_history = np.asarray(results.get('test_metrics_epochs', []), dtype=int)

            modest_corr_history = results.get('modest_test_correlation_history', {})
            modest_rrmse_history = results.get('modest_test_rrmse_history', {})
            modest_epochs_history = np.asarray(results.get('modest_test_metrics_epochs', []), dtype=int)

            has_history = (
                epochs_history.size > 0
                and isinstance(corr_history, dict)
                and isinstance(rrmse_history, dict)
                and all(k in corr_history for k in ('blos', 'vlos', 'temp'))
                and all(k in rrmse_history for k in ('blos', 'vlos', 'temp'))
            )

            corr_values = [
                float(test_metrics.get('blos_correlation', np.nan)),
                float(test_metrics.get('vlos_correlation', np.nan)),
                float(test_metrics.get('temp_correlation', np.nan)),
            ]
            rrmse_values = [
                float(test_metrics.get('blos_rrmse_tau_avg', np.nan)),
                float(test_metrics.get('vlos_rrmse_tau_avg', np.nan)),
                float(test_metrics.get('temp_rrmse_tau_avg', np.nan)),
            ]

            exp_dir = Path(results.get('experiment_dir', self.output_dir / exp_name))
            logs_dir = Path(results.get('log_dir', exp_dir / 'logs'))
            logs_dir.mkdir(parents=True, exist_ok=True)

            csv_path = logs_dir / 'test_set_metrics.csv'
            with open(csv_path, 'w') as f:
                f.write('experiment,metric,blos,vlos,temp\n')
                f.write(
                    f"{exp_name},correlation,{corr_values[0]:.10f},{corr_values[1]:.10f},{corr_values[2]:.10f}\n"
                )
                f.write(
                    f"{exp_name},rrmse_tau_avg,{rrmse_values[0]:.10f},{rrmse_values[1]:.10f},{rrmse_values[2]:.10f}\n"
                )

            if has_history:
                epoch_csv_path = logs_dir / 'test_set_epoch_log.csv'
                with open(epoch_csv_path, 'w') as f:
                    f.write(
                        'epoch,blos_correlation,vlos_correlation,temp_correlation,'
                        'blos_rrmse_tau_avg,vlos_rrmse_tau_avg,temp_rrmse_tau_avg\n'
                    )
                    for i, ep in enumerate(epochs_history.tolist()):
                        f.write(
                            f"{ep},"
                            f"{float(corr_history['blos'][i]):.10f},"
                            f"{float(corr_history['vlos'][i]):.10f},"
                            f"{float(corr_history['temp'][i]):.10f},"
                            f"{float(rrmse_history['blos'][i]):.10f},"
                            f"{float(rrmse_history['vlos'][i]):.10f},"
                            f"{float(rrmse_history['temp'][i]):.10f}\n"
                        )

            has_modest_history = (
                modest_epochs_history.size > 0
                and isinstance(modest_corr_history, dict)
                and isinstance(modest_rrmse_history, dict)
                and all(k in modest_corr_history for k in ('blos', 'vlos', 'temp'))
                and all(k in modest_rrmse_history for k in ('blos', 'vlos', 'temp'))
            )

            if has_modest_history:
                modest_epoch_csv_path = logs_dir / 'modest_test_set_epoch_log.csv'
                with open(modest_epoch_csv_path, 'w') as f:
                    f.write(
                        'epoch,blos_correlation,vlos_correlation,temp_correlation,'
                        'blos_rrmse_tau_avg,vlos_rrmse_tau_avg,temp_rrmse_tau_avg\n'
                    )
                    for i, ep in enumerate(modest_epochs_history.tolist()):
                        f.write(
                            f"{ep},"
                            f"{float(modest_corr_history['blos'][i]):.10f},"
                            f"{float(modest_corr_history['vlos'][i]):.10f},"
                            f"{float(modest_corr_history['temp'][i]):.10f},"
                            f"{float(modest_rrmse_history['blos'][i]):.10f},"
                            f"{float(modest_rrmse_history['vlos'][i]):.10f},"
                            f"{float(modest_rrmse_history['temp'][i]):.10f}\n"
                        )

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Test Set Metric Evolution - {exp_name}', fontsize=13, fontweight='bold')

            ax1 = axes[0]
            if has_history:
                ax1.plot(epochs_history, np.asarray(corr_history['blos'], dtype=float), marker='o', linewidth=1.8, label='B_LOS')
                ax1.plot(epochs_history, np.asarray(corr_history['vlos'], dtype=float), marker='s', linewidth=1.8, label='V_LOS')
                ax1.plot(epochs_history, np.asarray(corr_history['temp'], dtype=float), marker='^', linewidth=1.8, label='Temp')
                ax1.set_xlabel('Epoch')
            else:
                bars_corr = ax1.bar(labels, corr_values, color=['#1f77b4', '#2ca02c', '#ff7f0e'], alpha=0.85)
                for bar, val in zip(bars_corr, corr_values):
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + (0.03 if np.isfinite(val) and val >= 0 else -0.05),
                        f'{val:.4f}',
                        ha='center',
                        va='bottom' if np.isfinite(val) and val >= 0 else 'top',
                        fontsize=9,
                    )
            ax1.set_title('Correlation (Pearson)')
            ax1.set_ylabel('Correlation')
            ax1.set_ylim(-1.0, 1.0)
            ax1.grid(True, axis='y', alpha=0.3)
            if has_history:
                ax1.legend()

            ax2 = axes[1]
            if has_history:
                ax2.plot(epochs_history, np.asarray(rrmse_history['blos'], dtype=float), marker='o', linewidth=1.8, label='B_LOS')
                ax2.plot(epochs_history, np.asarray(rrmse_history['vlos'], dtype=float), marker='s', linewidth=1.8, label='V_LOS')
                ax2.plot(epochs_history, np.asarray(rrmse_history['temp'], dtype=float), marker='^', linewidth=1.8, label='Temp')
                ax2.set_xlabel('Epoch')
                ax2.legend()
            else:
                bars_rrmse = ax2.bar(labels, rrmse_values, color=['#9467bd', '#8c564b', '#17becf'], alpha=0.85)
                for bar, val in zip(bars_rrmse, rrmse_values):
                    y = val if np.isfinite(val) else 0.0
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        y + 0.02 * (abs(y) + 1.0),
                        f'{val:.4f}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                    )
            ax2.set_title('RRMSE (Tau-Averaged)')
            ax2.set_ylabel('RRMSE')
            ax2.grid(True, axis='y', alpha=0.3)

            plt.tight_layout(rect=(0, 0, 1, 0.95))

            plot_path = self.output_dir / f"{exp_name}-correlation_and_rrmse-test_set.png"
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()

            if has_modest_history:
                fig_m, axes_m = plt.subplots(1, 2, figsize=(12, 5))
                fig_m.suptitle(f'MODEST Test Metric Evolution - {exp_name}', fontsize=13, fontweight='bold')

                axes_m[0].plot(modest_epochs_history, np.asarray(modest_corr_history['blos'], dtype=float), marker='o', linewidth=1.8, label='B_LOS')
                axes_m[0].plot(modest_epochs_history, np.asarray(modest_corr_history['vlos'], dtype=float), marker='s', linewidth=1.8, label='V_LOS')
                axes_m[0].plot(modest_epochs_history, np.asarray(modest_corr_history['temp'], dtype=float), marker='^', linewidth=1.8, label='Temp')
                axes_m[0].set_title('Correlation (Pearson)')
                axes_m[0].set_xlabel('Epoch')
                axes_m[0].set_ylabel('Correlation')
                axes_m[0].set_ylim(-1.0, 1.0)
                axes_m[0].grid(True, axis='y', alpha=0.3)
                axes_m[0].legend()

                axes_m[1].plot(modest_epochs_history, np.asarray(modest_rrmse_history['blos'], dtype=float), marker='o', linewidth=1.8, label='B_LOS')
                axes_m[1].plot(modest_epochs_history, np.asarray(modest_rrmse_history['vlos'], dtype=float), marker='s', linewidth=1.8, label='V_LOS')
                axes_m[1].plot(modest_epochs_history, np.asarray(modest_rrmse_history['temp'], dtype=float), marker='^', linewidth=1.8, label='Temp')
                axes_m[1].set_title('RRMSE (Tau-Averaged)')
                axes_m[1].set_xlabel('Epoch')
                axes_m[1].set_ylabel('RRMSE')
                axes_m[1].grid(True, axis='y', alpha=0.3)
                axes_m[1].legend()

                plt.tight_layout(rect=(0, 0, 1, 0.95))
                modest_plot_path = self.output_dir / f"{exp_name}-correlation_and_rrmse-modest_test_set.png"
                plt.savefig(modest_plot_path, dpi=200, bbox_inches='tight')
                plt.close(fig_m)

            if exp_name == 'no_physics':
                alias_plot = self.output_dir / 'non_physics-correlation_and_rrmse-test_set.png'
                fig_alias, axes_alias = plt.subplots(1, 2, figsize=(12, 5))
                fig_alias.suptitle('Test Set Metric Evolution - non_physics', fontsize=13, fontweight='bold')
                if has_history:
                    axes_alias[0].plot(epochs_history, np.asarray(corr_history['blos'], dtype=float), marker='o', linewidth=1.8, label='B_LOS')
                    axes_alias[0].plot(epochs_history, np.asarray(corr_history['vlos'], dtype=float), marker='s', linewidth=1.8, label='V_LOS')
                    axes_alias[0].plot(epochs_history, np.asarray(corr_history['temp'], dtype=float), marker='^', linewidth=1.8, label='Temp')
                    axes_alias[0].legend()
                    axes_alias[0].set_xlabel('Epoch')
                    axes_alias[1].plot(epochs_history, np.asarray(rrmse_history['blos'], dtype=float), marker='o', linewidth=1.8, label='B_LOS')
                    axes_alias[1].plot(epochs_history, np.asarray(rrmse_history['vlos'], dtype=float), marker='s', linewidth=1.8, label='V_LOS')
                    axes_alias[1].plot(epochs_history, np.asarray(rrmse_history['temp'], dtype=float), marker='^', linewidth=1.8, label='Temp')
                    axes_alias[1].legend()
                    axes_alias[1].set_xlabel('Epoch')
                else:
                    axes_alias[0].bar(labels, corr_values, color=['#1f77b4', '#2ca02c', '#ff7f0e'], alpha=0.85)
                    axes_alias[1].bar(labels, rrmse_values, color=['#9467bd', '#8c564b', '#17becf'], alpha=0.85)
                axes_alias[0].set_title('Correlation (Pearson)')
                axes_alias[0].set_ylabel('Correlation')
                axes_alias[0].set_ylim(-1.0, 1.0)
                axes_alias[0].grid(True, axis='y', alpha=0.3)
                axes_alias[1].set_title('RRMSE (Tau-Averaged)')
                axes_alias[1].set_ylabel('RRMSE')
                axes_alias[1].grid(True, axis='y', alpha=0.3)
                plt.tight_layout(rect=(0, 0, 1, 0.95))
                plt.savefig(alias_plot, dpi=200, bbox_inches='tight')
                plt.close(fig_alias)

        print(f"Per-experiment MURaM/MODEST test-set correlation/RRMSE evolution plots saved to {self.output_dir}")
        
    def generate_comparison_plots(self):
        """Generate comparison visualizations."""
        if not self.results:
            print("No results to plot")
            return
        
        fig = plt.figure(figsize=(20, 24))
        
        # Extract data
        experiments = list(self.results.keys())
        
        # 1. Validation Loss Comparison
        ax1 = plt.subplot(4, 3, 1)
        val_losses = [self.results[exp]['final_val_loss'] for exp in experiments]
        bars1 = ax1.bar(range(len(experiments)), val_losses, color='steelblue', alpha=0.7)
        ax1.set_xticks(range(len(experiments)))
        ax1.set_xticklabels(experiments, rotation=45, ha='right')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title('Final Validation Loss')
        ax1.grid(True, alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars1, val_losses)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Training Time Comparison
        ax2 = plt.subplot(4, 3, 2)
        train_times = [self.results[exp]['training_time_minutes'] for exp in experiments]
        bars2 = ax2.bar(range(len(experiments)), train_times, color='coral', alpha=0.7)
        ax2.set_xticks(range(len(experiments)))
        ax2.set_xticklabels(experiments, rotation=45, ha='right')
        ax2.set_ylabel('Time (minutes)')
        ax2.set_title('Training Time')
        ax2.grid(True, alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars2, train_times)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. B_LOS RRMSE (Tau-averaged)
        ax3 = plt.subplot(4, 3, 3)
        blos_rrmse = [self.results[exp]['test_metrics']['blos_rrmse_tau_avg'] 
                      for exp in experiments]
        bars3 = ax3.bar(range(len(experiments)), blos_rrmse, color='forestgreen', alpha=0.7)
        ax3.set_xticks(range(len(experiments)))
        ax3.set_xticklabels(experiments, rotation=45, ha='right')
        ax3.set_ylabel('RRMSE')
        ax3.set_title('B_LOS RRMSE (Tau-Averaged)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=min(blos_rrmse), color='red', linestyle='--', alpha=0.5, label='Best')
        ax3.legend()
        
        for i, (bar, val) in enumerate(zip(bars3, blos_rrmse)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 4. V_LOS RRMSE (Tau-averaged)
        ax4 = plt.subplot(4, 3, 4)
        vlos_rrmse = [self.results[exp]['test_metrics']['vlos_rrmse_tau_avg'] 
                      for exp in experiments]
        bars4 = ax4.bar(range(len(experiments)), vlos_rrmse, color='purple', alpha=0.7)
        ax4.set_xticks(range(len(experiments)))
        ax4.set_xticklabels(experiments, rotation=45, ha='right')
        ax4.set_ylabel('RRMSE')
        ax4.set_title('V_LOS RRMSE (Tau-Averaged)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=min(vlos_rrmse), color='red', linestyle='--', alpha=0.5, label='Best')
        ax4.legend()
        
        for i, (bar, val) in enumerate(zip(bars4, vlos_rrmse)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 5. Temperature RRMSE (Tau-averaged) - NEW
        ax5 = plt.subplot(4, 3, 5)
        temp_rrmse = [self.results[exp]['test_metrics']['temp_rrmse_tau_avg'] 
                      for exp in experiments]
        bars5 = ax5.bar(range(len(experiments)), temp_rrmse, color='darkorange', alpha=0.7)
        ax5.set_xticks(range(len(experiments)))
        ax5.set_xticklabels(experiments, rotation=45, ha='right')
        ax5.set_ylabel('RRMSE')
        ax5.set_title('Temperature RRMSE (Tau-Averaged)')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=min(temp_rrmse), color='red', linestyle='--', alpha=0.5, label='Best')
        ax5.legend()
        
        for i, (bar, val) in enumerate(zip(bars5, temp_rrmse)):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Convergence curves (validation loss)
        ax6 = plt.subplot(4, 3, 6)
        for exp in experiments:
            val_history = self.results[exp]['val_loss_history']
            epochs = range(1, len(val_history) + 1)
            ax6.plot(epochs, _series_for_log_plot(val_history), marker='o', label=exp, linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Validation Loss')
        ax6.set_title('Validation Loss Convergence')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')
        
        # 7. Relative improvement matrix
        ax7 = plt.subplot(4, 3, 7)
        baseline_name = 'no_physics'
        if baseline_name in self.results:
            baseline_val = self.results[baseline_name]['final_val_loss']
            baseline_blos = self.results[baseline_name]['test_metrics']['blos_rrmse_tau_avg']
            baseline_vlos = self.results[baseline_name]['test_metrics']['vlos_rrmse_tau_avg']
            baseline_temp = self.results[baseline_name]['test_metrics']['temp_rrmse_tau_avg']
            
            improvements = []
            for exp in experiments:
                if exp == baseline_name:
                    improvements.append([0, 0, 0, 0])
                else:
                    val_imp = (baseline_val - self.results[exp]['final_val_loss']) / baseline_val * 100
                    blos_imp = (baseline_blos - self.results[exp]['test_metrics']['blos_rrmse_tau_avg']) / baseline_blos * 100
                    vlos_imp = (baseline_vlos - self.results[exp]['test_metrics']['vlos_rrmse_tau_avg']) / baseline_vlos * 100
                    temp_imp = (baseline_temp - self.results[exp]['test_metrics']['temp_rrmse_tau_avg']) / baseline_temp * 100
                    improvements.append([val_imp, blos_imp, vlos_imp, temp_imp])
            
            improvements = np.array(improvements)
            
            im = ax7.imshow(improvements.T, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
            ax7.set_xticks(range(len(experiments)))
            ax7.set_xticklabels(experiments, rotation=45, ha='right')
            ax7.set_yticks([0, 1, 2, 3])
            ax7.set_yticklabels(['Val Loss', 'B_LOS RRMSE', 'V_LOS RRMSE', 'Temp RRMSE'])
            ax7.set_title('% Improvement over Baseline\n(Positive = Better)')
            
            for i in range(len(experiments)):
                for j in range(4):
                    ax7.text(i, j, f'{improvements[i, j]:.1f}%',
                           ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=ax7, label='% Improvement')
        
        # 8. Total Loss Curves
        ax8 = plt.subplot(4, 3, 8)
        for exp in experiments:
            if 'train_loss_history' in self.results[exp]:
                loss_history = self.results[exp]['train_loss_history']
                epochs = range(1, len(loss_history) + 1)
                ax8.plot(epochs, _series_for_log_plot(loss_history), marker='o', label=exp, linewidth=2, markersize=4)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Total Loss')
        ax8.set_title('Total Loss Convergence (Training)')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
        ax8.set_yscale('log')
        
        # 9. MSE Loss Component
        ax9 = plt.subplot(4, 3, 9)
        for exp in experiments:
            if 'mse_loss_history' in self.results[exp]:
                loss_history = self.results[exp]['mse_loss_history']
                epochs = range(1, len(loss_history) + 1)
                ax9.plot(epochs, _series_for_log_plot(loss_history), marker='s', label=exp, linewidth=2, markersize=4)
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('MSE Loss')
        ax9.set_title('MSE Loss Component')
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3)
        ax9.set_yscale('log')
        
        # 10. Physics Loss Components
        ax10 = plt.subplot(4, 3, 10)
        for exp in experiments:
            if 'physics_loss_history' in self.results[exp]:
                loss_history = self.results[exp]['physics_loss_history']
                if len(loss_history) > 0 and any(l > 0 for l in loss_history):
                    epochs = range(1, len(loss_history) + 1)
                    ax10.plot(epochs, _series_for_log_plot(loss_history), marker='^', label=exp, linewidth=2, markersize=4)
        ax10.set_xlabel('Epoch')
        ax10.set_ylabel('Physics Loss')
        ax10.set_title('Physics Loss Components')
        ax10.legend(fontsize=8)
        ax10.grid(True, alpha=0.3)
        ax10.set_yscale('log')
        
        # 11. Correlation Comparison - NEW
        ax11 = plt.subplot(4, 3, 11)
        blos_corr = [self.results[exp]['test_metrics']['blos_correlation'] for exp in experiments]
        vlos_corr = [self.results[exp]['test_metrics']['vlos_correlation'] for exp in experiments]
        temp_corr = [self.results[exp]['test_metrics']['temp_correlation'] for exp in experiments]
        
        x = np.arange(len(experiments))
        width = 0.25
        ax11.bar(x - width, blos_corr, width, label='B_LOS', color='forestgreen', alpha=0.7)
        ax11.bar(x, vlos_corr, width, label='V_LOS', color='purple', alpha=0.7)
        ax11.bar(x + width, temp_corr, width, label='Temperature', color='darkorange', alpha=0.7)
        ax11.set_xticks(x)
        ax11.set_xticklabels(experiments, rotation=45, ha='right')
        ax11.set_ylabel('Pearson Correlation')
        ax11.set_title('Correlation Coefficients')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        ax11.set_ylim([0, 1])

        # 12. Empty panel kept for layout symmetry
        ax12 = plt.subplot(4, 3, 12)
        ax12.axis('off')
        
        plt.suptitle('Physics Regularization Ablation Study', fontsize=16, y=0.997)
        plt.tight_layout()
        
        plot_path = self.output_dir / "comparison_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to {plot_path}")
        plt.show()

def _tau_average(values: np.ndarray, logtau_values: np.ndarray) -> np.ndarray:
    # 1D integration in tau; independent from spatial plotting choices.
    tau_linear = 10 ** logtau_values
    dtau = np.diff(tau_linear)
    denom = (tau_linear[-1] - tau_linear[0]) + 1e-12
    vals_mid = (values[:, :-1] + values[:, 1:]) / 2
    return np.sum(vals_mid * dtau[np.newaxis, :], axis=1) / denom


def _normalize_lambda_values(values) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    return [float(values)]


def _lambda_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "_")


def _variant_names(base_key: str, folder_prefix: str, lambda_value: float, is_multi: bool) -> tuple[str, str]:
    if not is_multi:
        return base_key, base_key
    token = _lambda_token(lambda_value)
    return f"{base_key}-lambda-{token}", f"{folder_prefix}-lambda-{token}"

def compute_tau_averaged_metrics(
    model: PhysicsInformedMSCNN,
    test_steps: list[int],
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    logtau_values: np.ndarray,
    cache: MuramDataCache | None = None,
) -> dict[str, float]:
    """Evaluate model on test steps using tau-averaged physics metrics."""
    from scipy.stats import pearsonr

    model.eval()
    device = config.device
    
    all_pred_blos = []
    all_true_blos = []
    all_pred_vlos = []
    all_true_vlos = []
    all_pred_temp = []
    all_true_temp = []
    
    n_tau = int(len(logtau_values))

    with torch.no_grad():
        for step in tqdm(test_steps, desc="Evaluating test steps"):
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

            # Strictly require ApproxInversions outputs
            required_keys = {"blos", "vlos", "temp"}
            if not isinstance(approx_data, dict) or not required_keys.issubset(approx_data.keys()):
                raise KeyError(
                    f"Step {step} missing required approximation keys {required_keys}. "
                    "These must come from ApproxInversions. "
                    "Clear stale cache and reprocess."
                )

            dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

            true_blos = approx_data["blos"].flatten()
            true_vlos = approx_data["vlos"].flatten()
            true_temp = approx_data["temp"].flatten()
            
            step_pred_blos = []
            step_pred_vlos = []
            step_pred_temp = []
            
            model.set_physics_context(
                mhd_normalizer=mhd_normalizer,
                logtau_values=logtau_values,
                blos_approx=approx_data['blos'],
                vlos_approx=approx_data['vlos'],
                temp_approx=approx_data['temp'],
            )

            for stokes_batch, _, spatial_idx_batch in dataloader:
                stokes_batch = stokes_batch.to(device)
                predictions = model(stokes_batch)

                # Convert predictions to numpy for denormalization
                predictions_np = predictions.cpu().numpy()

                if predictions_np.ndim != 2 or predictions_np.shape[1] != 3 * n_tau:
                    raise ValueError(
                        f"Expected predictions shape (batch, {3 * n_tau}), got {predictions_np.shape}"
                    )

                # Block-concatenated layout: [T(τ...), Vz(τ...), Bz(τ...)]
                T_norm = predictions_np[:, :n_tau]
                Vz_norm = predictions_np[:, n_tau:2 * n_tau]
                Bz_norm = predictions_np[:, 2 * n_tau:3 * n_tau]

                # Denormalize each parameter individually
                T_denorm = mhd_normalizer.denormalize(T_norm, param='T')
                Vz_denorm = mhd_normalizer.denormalize(Vz_norm, param='Vz')
                Bz_denorm = mhd_normalizer.denormalize(Bz_norm, param='Bz')

                pred_blos_batch = _tau_average(Bz_denorm, logtau_values)
                pred_vlos_batch = _tau_average(Vz_denorm, logtau_values)
                pred_temp_batch = _tau_average(T_denorm, logtau_values)

                step_pred_blos.append(pred_blos_batch)
                step_pred_vlos.append(pred_vlos_batch)
                step_pred_temp.append(pred_temp_batch)
            
            all_pred_blos.append(np.concatenate(step_pred_blos))
            all_true_blos.append(true_blos)
            all_pred_vlos.append(np.concatenate(step_pred_vlos))
            all_true_vlos.append(true_vlos)
            all_pred_temp.append(np.concatenate(step_pred_temp))
            all_true_temp.append(true_temp)

    if len(all_pred_blos) == 0:
        print("  Warning: no usable test steps were found; returning NaN metrics.")
        return {
            'blos_rrmse_tau_avg': np.nan,
            'vlos_rrmse_tau_avg': np.nan,
            'temp_rrmse_tau_avg': np.nan,
            'blos_correlation': np.nan,
            'vlos_correlation': np.nan,
            'temp_correlation': np.nan,
            'blos_rmse': np.nan,
            'vlos_rmse': np.nan,
            'temp_rmse': np.nan,
        }

    all_pred_blos = np.concatenate(all_pred_blos)
    all_true_blos = np.concatenate(all_true_blos)
    all_pred_vlos = np.concatenate(all_pred_vlos)
    all_true_vlos = np.concatenate(all_true_vlos)
    all_pred_temp = np.concatenate(all_pred_temp)
    all_true_temp = np.concatenate(all_true_temp)
    
    rmse_blos = np.sqrt(np.mean((all_pred_blos - all_true_blos) ** 2))
    rrmse_blos = rmse_blos / (np.mean(np.abs(all_true_blos)) + 1e-10)
    
    rmse_vlos = np.sqrt(np.mean((all_pred_vlos - all_true_vlos) ** 2))
    rrmse_vlos = rmse_vlos / (np.mean(np.abs(all_true_vlos)) + 1e-10)
    
    rmse_temp = np.sqrt(np.mean((all_pred_temp - all_true_temp) ** 2))
    rrmse_temp = rmse_temp / (np.mean(np.abs(all_true_temp)) + 1e-10)
    
    corr_blos, _ = pearsonr(all_pred_blos, all_true_blos)
    corr_vlos, _ = pearsonr(all_pred_vlos, all_true_vlos)
    corr_temp, _ = pearsonr(all_pred_temp, all_true_temp)
    
    return {
        'blos_rrmse_tau_avg': float(rrmse_blos),
        'vlos_rrmse_tau_avg': float(rrmse_vlos),
        'temp_rrmse_tau_avg': float(rrmse_temp),
        'blos_correlation': float(corr_blos),
        'vlos_correlation': float(corr_vlos),
        'temp_correlation': float(corr_temp),
        'blos_rmse': float(rmse_blos),
        'vlos_rmse': float(rmse_vlos),
        'temp_rmse': float(rmse_temp),
    }


def compute_modest_tau_averaged_metrics(
    model: PhysicsInformedMSCNN,
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    modest_snapshot: dict[str, object],
) -> dict[str, float]:
    """Evaluate model on MODEST snapshot using tau-averaged metrics over common optical-depth nodes."""
    pred_logtau = config.get_logtau_values()
    modest_logtau = np.asarray(modest_snapshot["modest_logtau"], dtype=np.float32)

    matches: list[tuple[int, int]] = []
    for i_mod, tau_mod in enumerate(modest_logtau):
        pred_idx = np.where(np.isclose(pred_logtau, float(tau_mod), atol=1e-6, rtol=0.0))[0]
        if pred_idx.size > 0:
            matches.append((i_mod, int(pred_idx[0])))

    if not matches:
        return {
            'blos_rrmse_tau_avg': np.nan,
            'vlos_rrmse_tau_avg': np.nan,
            'temp_rrmse_tau_avg': np.nan,
            'blos_correlation': np.nan,
            'vlos_correlation': np.nan,
            'temp_correlation': np.nan,
            'blos_rmse': np.nan,
            'vlos_rmse': np.nan,
            'temp_rmse': np.nan,
        }

    stokes_input = np.asarray(modest_snapshot["stokes_input"], dtype=np.float32)
    pred_nx = int(modest_snapshot["pred_nx"])
    pred_ny = int(modest_snapshot["pred_ny"])
    gt_den = modest_snapshot["gt_den"]

    n_pixels = stokes_input.shape[0]
    n_tau_pred = int(len(pred_logtau))

    all_pred = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(0, n_pixels, config.batch_size):
            x = torch.from_numpy(stokes_input[i:i + config.batch_size]).float().to(config.device)
            y = model(x).detach().cpu().numpy()
            all_pred.append(y)
    if was_training:
        model.train()

    pred_norm = np.concatenate(all_pred, axis=0)
    pred_den = {
        "T": mhd_normalizer.denormalize(pred_norm[:, :n_tau_pred], param="T").reshape(pred_nx, pred_ny, n_tau_pred),
        "Vz": mhd_normalizer.denormalize(pred_norm[:, n_tau_pred:2 * n_tau_pred], param="Vz").reshape(pred_nx, pred_ny, n_tau_pred),
        "Bz": mhd_normalizer.denormalize(pred_norm[:, 2 * n_tau_pred:3 * n_tau_pred], param="Bz").reshape(pred_nx, pred_ny, n_tau_pred),
    }

    def _metric_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        yt = y_true[m]
        yp = y_pred[m]
        if yt.size == 0:
            return np.nan, np.nan, np.nan
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        rrmse = float(rmse / (np.mean(np.abs(yt)) + 1e-10))
        corr = float(np.corrcoef(yp, yt)[0, 1]) if yt.size > 1 else np.nan
        return rmse, rrmse, corr

    metrics = {}
    for p, key in (("Bz", "blos"), ("Vz", "vlos"), ("T", "temp")):
        true_cube = np.asarray(gt_den[p], dtype=np.float32)
        pred_cube = np.asarray(pred_den[p], dtype=np.float32)

        true_stack = np.stack([true_cube[:, :, i_mod] for i_mod, _ in matches], axis=-1)
        pred_stack = np.stack([pred_cube[:, :, i_pred] for _, i_pred in matches], axis=-1)

        true_tau_avg = np.mean(true_stack, axis=-1).ravel()
        pred_tau_avg = np.mean(pred_stack, axis=-1).ravel()

        rmse, rrmse, corr = _metric_pair(true_tau_avg, pred_tau_avg)
        metrics[f"{key}_rmse"] = rmse
        metrics[f"{key}_rrmse_tau_avg"] = rrmse
        metrics[f"{key}_correlation"] = corr

    return metrics

@contextlib.contextmanager
def timer():
    start = time.time()
    
    try:
        yield None
    finally:
        end = time.time()
        print("Elapsed time: {:.2f} minutes".format((end - start) / 60))

def run_single_experiment(
    experiment_name: str,
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    test_steps: list[int],
    n_steps_per_epoch: int = 20,
    min_step: int = 60,
    max_step: int = 200,
    step_size: int = 1,
    cache: MuramDataCache | None = None,
    plot_training_data_histograms: bool = True,
    training_hist_bins: int = 120,
    training_hist_max_samples: int = 400000,
) -> dict:
    """Run a single training experiment."""
    print("\n" + "=" * 100)
    print(f"EXPERIMENT: {experiment_name}".center(100))
    print("=" * 100)
    print(f"Device: {config.device}")
    print(f"Number of epochs: {config.n_epochs}")
    print(f"Training step range: {min_step} to {max_step} (step size: {step_size})")
    print(f"Lambda WFA: {config.lambda_wfa}")
    print(f"Lambda Doppler: {config.lambda_doppler}")
    print(f"Lambda Temperature: {config.lambda_temp}")
    print(f"B_LOS physics mode: {config.blos_physics_mode}")
    if config.blos_physics_mode == 'single_height':
        print(f"B_LOS target log(tau): {config.blos_target_logtau}")
    print(f"V_LOS physics mode: {config.vlos_physics_mode}")
    if config.vlos_physics_mode == 'single_height':
        print(f"V_LOS target log(tau): {config.vlos_target_logtau}")
    print(f"Temperature physics mode: {config.temp_physics_mode}")
    if config.temp_physics_mode == 'single_height':
        print(f"Temperature target log(tau): {config.temp_target_logtau}")
    print(f"WFA gate mode: {config.wfa_gate_mode}")
    if config.wfa_gate_mode == 'threshold':
        print(f"WFA gate threshold: {config.wfa_gate_threshold}")
    elif config.wfa_gate_mode == 'plateau':
        print(
            f"WFA gate plateau: patience={config.wfa_gate_patience}, "
            f"min_delta={config.wfa_gate_min_delta}, warmup={config.wfa_gate_warmup_epochs}"
        )
    print(f"Learning rate: {config.learning_rate}")
    print(f"Weight decay: {config.weight_decay}")
    print(f"Gradient clip: {config.gradient_clip}")
    print("=" * 100)
    
    # Save experiment configuration
    config_dict = {
        'experiment_name': experiment_name,
        'training_config': {
            'n_epochs': config.n_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'gradient_clip': config.gradient_clip,
        },
        'physics_config': {
            'lambda_wfa': config.lambda_wfa,
            'lambda_doppler': config.lambda_doppler,
            'lambda_temp': config.lambda_temp,
            'stokes_mult_factor': config.stokes_mult_factor,
            'wfa_gate_mode': config.wfa_gate_mode,
            'wfa_gate_threshold': config.wfa_gate_threshold,
            'wfa_gate_patience': config.wfa_gate_patience,
            'wfa_gate_min_delta': config.wfa_gate_min_delta,
            'wfa_gate_warmup_epochs': config.wfa_gate_warmup_epochs,
            'blos_physics_mode': config.blos_physics_mode,
            'blos_target_logtau': config.blos_target_logtau,
            'vlos_physics_mode': config.vlos_physics_mode,
            'vlos_target_logtau': config.vlos_target_logtau,
            'temp_physics_mode': config.temp_physics_mode,
            'temp_target_logtau': config.temp_target_logtau,
        },
        'data_config': {
            'min_step': min_step,
            'max_step': max_step,
            'step_size': step_size,
            'test_steps': test_steps,
            'n_steps_per_epoch': n_steps_per_epoch,
            'logtau_values': [float(x) for x in config.get_logtau_values().tolist()],
            'balanced_region_training': bool(config.apply_region_mask),
            'balanced_bz_training': bool(config.apply_bz_bin_balance),
            'bz_balance_scope': str(config.bz_balance_scope),
            'bz_balance_mode': str(config.bz_balance_mode),
            'bz_balance_bins': int(config.bz_balance_bins),
            'bz_balance_tau_idx': None if config.bz_balance_tau_idx is None else int(config.bz_balance_tau_idx),
        },
        'model_config': {
            'scales': [1, 2, 3],
            'in_channels': 2,
            'c1_filters': config.c1_filters,
            'c2_filters': 32,
            'kernel_size': 5,
            'pool_size': 2,
            'n_linear_layers': 4,
        },
        'device': config.device,
        'data_path': str(config.data_path),
    }
    
    # Create experiment directory and save config
    exp_dir = config.checkpoint_dir.parent
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    # Initialize model
    n_logtau = config.get_n_logtau()
    model = PhysicsInformedMSCNN(
        scales=[1, 2, 3],
        in_channels=2,
        c1_filters=config.c1_filters,
        c2_filters=32,
        kernel_size=5,
        pool_size=2,
        n_linear_layers=4,
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
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Prepare train/val split
    step_size = getattr(config, "step_size", 1)
    all_steps = list(range(min_step, max_step + 1, step_size))
    train_steps = [s for s in all_steps if s not in test_steps]

    import random
    random.seed(42)
    n_val = max(1, len(train_steps) // 10)
    val_steps = random.sample(train_steps, n_val)
    train_steps = [s for s in train_steps if s not in val_steps]

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
        global_meta_path = config.checkpoint_dir.parent / "global_bz_balance_metadata.json"
        with open(global_meta_path, "w") as f:
            json.dump(global_bz_balance_metadata, f, indent=2)
        print(f"Global Bz balance metadata saved to: {global_meta_path}")

    balanced_cache = None
    balanced_cache_signature_hash = None
    balanced_runtime_mode = None
    preloaded_balanced_steps = None
    if config.use_balanced_cache:
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
        balanced_cache_report_path = config.log_dir / "balanced_cache_report.json"
        with open(balanced_cache_report_path, "w") as f:
            json.dump(balanced_cache_report, f, indent=2)
        print(f"Balanced cache report saved to: {balanced_cache_report_path}")
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

    if plot_training_data_histograms:
        hist_output_dir = config.checkpoint_dir.parent.parent
        try:
            generate_training_data_histograms(
                config=config,
                mhd_normalizer=mhd_normalizer,
                stokes_normalizer=stokes_normalizer,
                train_steps=train_steps,
                output_dir=hist_output_dir,
                cache=cache,
                global_bz_selection_indices=global_bz_selection_indices,
                global_bz_balance_metadata=global_bz_balance_metadata,
                bins=training_hist_bins,
                max_samples_per_param=training_hist_max_samples,
            )
        except Exception as e:
            print(f"⚠ Failed to generate training-data histograms: {e}")
    
    # Initialize logger
    logger = MetricsLogger(config.log_dir)
    monitor_step_for_epoch_plots = (
        config.epoch_plot_step if config.epoch_plot_step is not None else val_steps[0]
    )

    modest_snapshot = None
    print("\nPreparing MODEST test snapshot...")
    try:
        modest_snapshot = prepare_modest_epoch_snapshot(
            config=config,
            stokes_normalizer=stokes_normalizer,
        )
        print("  ✓ MODEST test snapshot prepared")
    except Exception as e:
        print(f"  ⚠ Failed to prepare MODEST test snapshot: {e}")
        modest_snapshot = None

    # Training loop
    start_time = time.time()
    val_loss_history = []
    train_loss_history = []
    mse_loss_history = []
    physics_loss_history = []
    wfa_loss_history = []
    doppler_loss_history = []
    temperature_loss_history = []
    test_metrics_epochs = []
    test_correlation_history = {'blos': [], 'vlos': [], 'temp': []}
    test_rrmse_history = {'blos': [], 'vlos': [], 'temp': []}
    modest_test_metrics_epochs = []
    modest_test_correlation_history = {'blos': [], 'vlos': [], 'temp': []}
    modest_test_rrmse_history = {'blos': [], 'vlos': [], 'temp': []}
    train_wfa_enabled_history = []
    total_training_pixels = 0
    wfa_gate_state = initialize_wfa_gate_state(config)
    wfa_gate_trigger_epoch = None
    wfa_gate_trigger_reason = None

    epoch_test_log_path = Path(config.log_dir) / "test_set_epoch_log.csv"
    with open(epoch_test_log_path, 'w') as f:
        f.write(
            'epoch,blos_correlation,vlos_correlation,temp_correlation,'
            'blos_rrmse_tau_avg,vlos_rrmse_tau_avg,temp_rrmse_tau_avg\n'
        )
    modest_epoch_test_log_path = Path(config.log_dir) / "modest_test_set_epoch_log.csv"
    with open(modest_epoch_test_log_path, 'w') as f:
        f.write(
            'epoch,blos_correlation,vlos_correlation,temp_correlation,'
            'blos_rrmse_tau_avg,vlos_rrmse_tau_avg,temp_rrmse_tau_avg\n'
        )
    
    for epoch in range(config.n_epochs):
        with timer():
            print(f"\nEpoch {epoch + 1}/{config.n_epochs}")
            train_wfa_enabled = bool(wfa_gate_state.get('enabled', True))
            train_wfa_enabled_history.append(train_wfa_enabled)
            print(f"  Train-time WFA enabled: {train_wfa_enabled}")
            
            # Use the shared train_epoch function with cache
            epoch_metrics = train_epoch(
                model=model,
                train_steps=train_steps,
                config=config,
                mhd_normalizer=mhd_normalizer,
                stokes_normalizer=stokes_normalizer,
                optimizer=optimizer,
                epoch=epoch,
                logger=logger,
                n_steps_per_epoch=n_steps_per_epoch,
                cache=cache,
                enable_wfa=train_wfa_enabled,
                global_bz_selection_indices=global_bz_selection_indices,
                global_bz_balance_metadata=global_bz_balance_metadata,
                balanced_cache=balanced_cache if balanced_runtime_mode == "disk" else None,
                balanced_cache_signature_hash=balanced_cache_signature_hash if balanced_runtime_mode == "disk" else None,
                preloaded_balanced_steps=preloaded_balanced_steps,
            )
            
            # Extract metrics
            avg_train_loss = epoch_metrics['total_loss']
            avg_mse_loss = epoch_metrics['mse_loss']
            avg_physics_loss = epoch_metrics['physics_loss']
            avg_wfa_loss = epoch_metrics['wfa_loss']
            avg_doppler_loss = epoch_metrics['doppler_loss']
            avg_temperature_loss = epoch_metrics['temperature_loss']
            
            # Store histories
            train_loss_history.append(avg_train_loss)
            mse_loss_history.append(avg_mse_loss)
            physics_loss_history.append(avg_physics_loss)
            wfa_loss_history.append(avg_wfa_loss)
            doppler_loss_history.append(avg_doppler_loss)
            temperature_loss_history.append(avg_temperature_loss)
            total_training_pixels += int(epoch_metrics.get('n_pixels_used', 0))

            wfa_gate_state, wfa_gate_triggered, wfa_gate_reason = update_wfa_gate_state(
                gate_state=wfa_gate_state,
                config=config,
                epoch=epoch,
                epoch_mse_loss=float(avg_mse_loss),
            )
            if wfa_gate_triggered:
                wfa_gate_trigger_epoch = int(wfa_gate_state.get('trigger_epoch') or (epoch + 1))
                wfa_gate_trigger_reason = wfa_gate_reason
                print(f"  WFA gate triggered at epoch {wfa_gate_trigger_epoch}: {wfa_gate_reason}")
            
            # Validation
            avg_val_loss = validate(
                model=model,
                val_steps=val_steps[:5],
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

            val_loss_history.append(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            epoch_test_metrics = compute_tau_averaged_metrics(
                model=model,
                test_steps=test_steps,
                config=config,
                mhd_normalizer=mhd_normalizer,
                stokes_normalizer=stokes_normalizer,
                logtau_values=config.get_logtau_values(),
                cache=cache,
            )

            test_metrics_epochs.append(epoch + 1)
            test_correlation_history['blos'].append(float(epoch_test_metrics['blos_correlation']))
            test_correlation_history['vlos'].append(float(epoch_test_metrics['vlos_correlation']))
            test_correlation_history['temp'].append(float(epoch_test_metrics['temp_correlation']))
            test_rrmse_history['blos'].append(float(epoch_test_metrics['blos_rrmse_tau_avg']))
            test_rrmse_history['vlos'].append(float(epoch_test_metrics['vlos_rrmse_tau_avg']))
            test_rrmse_history['temp'].append(float(epoch_test_metrics['temp_rrmse_tau_avg']))

            with open(epoch_test_log_path, 'a') as f:
                f.write(
                    f"{epoch + 1},"
                    f"{float(epoch_test_metrics['blos_correlation']):.10f},"
                    f"{float(epoch_test_metrics['vlos_correlation']):.10f},"
                    f"{float(epoch_test_metrics['temp_correlation']):.10f},"
                    f"{float(epoch_test_metrics['blos_rrmse_tau_avg']):.10f},"
                    f"{float(epoch_test_metrics['vlos_rrmse_tau_avg']):.10f},"
                    f"{float(epoch_test_metrics['temp_rrmse_tau_avg']):.10f}\n"
                )

            epoch_modest_metrics = None
            if modest_snapshot is not None:
                epoch_modest_metrics = compute_modest_tau_averaged_metrics(
                    model=model,
                    config=config,
                    mhd_normalizer=mhd_normalizer,
                    modest_snapshot=modest_snapshot,
                )
                modest_test_metrics_epochs.append(epoch + 1)
                modest_test_correlation_history['blos'].append(float(epoch_modest_metrics['blos_correlation']))
                modest_test_correlation_history['vlos'].append(float(epoch_modest_metrics['vlos_correlation']))
                modest_test_correlation_history['temp'].append(float(epoch_modest_metrics['temp_correlation']))
                modest_test_rrmse_history['blos'].append(float(epoch_modest_metrics['blos_rrmse_tau_avg']))
                modest_test_rrmse_history['vlos'].append(float(epoch_modest_metrics['vlos_rrmse_tau_avg']))
                modest_test_rrmse_history['temp'].append(float(epoch_modest_metrics['temp_rrmse_tau_avg']))

                with open(modest_epoch_test_log_path, 'a') as f:
                    f.write(
                        f"{epoch + 1},"
                        f"{float(epoch_modest_metrics['blos_correlation']):.10f},"
                        f"{float(epoch_modest_metrics['vlos_correlation']):.10f},"
                        f"{float(epoch_modest_metrics['temp_correlation']):.10f},"
                        f"{float(epoch_modest_metrics['blos_rrmse_tau_avg']):.10f},"
                        f"{float(epoch_modest_metrics['vlos_rrmse_tau_avg']):.10f},"
                        f"{float(epoch_modest_metrics['temp_rrmse_tau_avg']):.10f}\n"
                    )
            
            print("=" * 100)
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Total Loss:      {avg_train_loss:.6f}")
            print(f"  MSE Loss:        {avg_mse_loss:.6f}")
            print(f"  Physics Loss:    {avg_physics_loss:.6f}")
            print(f"    ├─ WFA Loss:         {avg_wfa_loss:.6f}")
            print(f"    ├─ Doppler Loss:     {avg_doppler_loss:.6f}")
            print(f"    └─ Temperature Loss: {avg_temperature_loss:.6f}")
            print(f"  Validation Loss: {avg_val_loss:.6f}")
            print(f"  Learning Rate:   {current_lr:.2e}")
            print(
                "  Test Corr (B,V,T): "
                f"{epoch_test_metrics['blos_correlation']:.4f}, "
                f"{epoch_test_metrics['vlos_correlation']:.4f}, "
                f"{epoch_test_metrics['temp_correlation']:.4f}"
            )
            print(
                "  Test RRMSE (B,V,T): "
                f"{epoch_test_metrics['blos_rrmse_tau_avg']:.4f}, "
                f"{epoch_test_metrics['vlos_rrmse_tau_avg']:.4f}, "
                f"{epoch_test_metrics['temp_rrmse_tau_avg']:.4f}"
            )
            if epoch_modest_metrics is not None:
                print(
                    "  MODEST Corr (B,V,T): "
                    f"{epoch_modest_metrics['blos_correlation']:.4f}, "
                    f"{epoch_modest_metrics['vlos_correlation']:.4f}, "
                    f"{epoch_modest_metrics['temp_correlation']:.4f}"
                )
                print(
                    "  MODEST RRMSE (B,V,T): "
                    f"{epoch_modest_metrics['blos_rrmse_tau_avg']:.4f}, "
                    f"{epoch_modest_metrics['vlos_rrmse_tau_avg']:.4f}, "
                    f"{epoch_modest_metrics['temp_rrmse_tau_avg']:.4f}"
                )
            print(f"  Pixels used this epoch (balanced): {epoch_metrics.get('n_pixels_used', 0)}")
            
            print("=" * 100)
    
    training_time = (time.time() - start_time) / 60

    if config.enable_epoch_plots and config.enable_epoch_videos:
        print("\nBuilding epoch diagnostic videos...")
        generate_epoch_diagnostic_videos(
            config=config,
            step=monitor_step_for_epoch_plots,
        )

    if config.enable_modest_epoch_plots and config.enable_epoch_videos:
        print("\nBuilding MODEST epoch diagnostic videos...")
        generate_epoch_modest_diagnostic_videos(config=config)

    if test_metrics_epochs:
        test_metrics = {
            'blos_rrmse_tau_avg': float(test_rrmse_history['blos'][-1]),
            'vlos_rrmse_tau_avg': float(test_rrmse_history['vlos'][-1]),
            'temp_rrmse_tau_avg': float(test_rrmse_history['temp'][-1]),
            'blos_correlation': float(test_correlation_history['blos'][-1]),
            'vlos_correlation': float(test_correlation_history['vlos'][-1]),
            'temp_correlation': float(test_correlation_history['temp'][-1]),
            'blos_rmse': np.nan,
            'vlos_rmse': np.nan,
            'temp_rmse': np.nan,
        }
    else:
        print("\nEvaluating on test set...")
        test_metrics = compute_tau_averaged_metrics(
            model=model,
            test_steps=test_steps,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            logtau_values=config.get_logtau_values(),
            cache=cache,
        )
    
    # Save model
    if config.checkpoint_dir:
        model_path = config.checkpoint_dir.parent / "final_model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'test_metrics': test_metrics,
            'test_metrics_epochs': test_metrics_epochs,
            'test_correlation_history': test_correlation_history,
            'test_rrmse_history': test_rrmse_history,
            'modest_test_metrics_epochs': modest_test_metrics_epochs,
            'modest_test_correlation_history': modest_test_correlation_history,
            'modest_test_rrmse_history': modest_test_rrmse_history,
            'wfa_gate_state': wfa_gate_state,
            'wfa_gate_trigger_epoch': wfa_gate_trigger_epoch,
            'wfa_gate_trigger_reason': wfa_gate_trigger_reason,
            'train_wfa_enabled_history': train_wfa_enabled_history,
        }
        
        torch.save(checkpoint_data, model_path)
    
    logger.close()

    config_dict['data_config']['total_training_pixels_used'] = int(total_training_pixels)
    config_dict['runtime'] = {
        'wfa_gate_state': wfa_gate_state,
        'wfa_gate_trigger_epoch': wfa_gate_trigger_epoch,
        'wfa_gate_trigger_reason': wfa_gate_trigger_reason,
        'train_wfa_enabled_history': train_wfa_enabled_history,
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    return {
        'experiment_dir': str(config.checkpoint_dir.parent),
        'log_dir': str(config.log_dir),
        'final_val_loss': val_loss_history[-1],
        'val_loss_history': val_loss_history,
        'train_loss_history': train_loss_history,
        'mse_loss_history': mse_loss_history,
        'physics_loss_history': physics_loss_history,
        'wfa_loss_history': wfa_loss_history,
        'doppler_loss_history': doppler_loss_history,
        'temperature_loss_history': temperature_loss_history,
        'train_wfa_enabled_history': train_wfa_enabled_history,
        'wfa_gate_state': wfa_gate_state,
        'wfa_gate_trigger_epoch': wfa_gate_trigger_epoch,
        'wfa_gate_trigger_reason': wfa_gate_trigger_reason,
        'training_time_minutes': training_time,
        'total_training_pixels_used': int(total_training_pixels),
        'test_metrics': test_metrics,
        'test_metrics_epochs': test_metrics_epochs,
        'test_correlation_history': test_correlation_history,
        'test_rrmse_history': test_rrmse_history,
        'modest_test_metrics': {
            'blos_rrmse_tau_avg': float(modest_test_rrmse_history['blos'][-1]) if modest_test_rrmse_history['blos'] else np.nan,
            'vlos_rrmse_tau_avg': float(modest_test_rrmse_history['vlos'][-1]) if modest_test_rrmse_history['vlos'] else np.nan,
            'temp_rrmse_tau_avg': float(modest_test_rrmse_history['temp'][-1]) if modest_test_rrmse_history['temp'] else np.nan,
            'blos_correlation': float(modest_test_correlation_history['blos'][-1]) if modest_test_correlation_history['blos'] else np.nan,
            'vlos_correlation': float(modest_test_correlation_history['vlos'][-1]) if modest_test_correlation_history['vlos'] else np.nan,
            'temp_correlation': float(modest_test_correlation_history['temp'][-1]) if modest_test_correlation_history['temp'] else np.nan,
        },
        'modest_test_metrics_epochs': modest_test_metrics_epochs,
        'modest_test_correlation_history': modest_test_correlation_history,
        'modest_test_rrmse_history': modest_test_rrmse_history,
        'config': {
            'lambda_wfa': config.lambda_wfa,
            'lambda_doppler': config.lambda_doppler,
            'lambda_temp': config.lambda_temp,
            'stokes_mult_factor': config.stokes_mult_factor,
            'wfa_gate_mode': config.wfa_gate_mode,
            'wfa_gate_threshold': config.wfa_gate_threshold,
            'wfa_gate_patience': config.wfa_gate_patience,
            'wfa_gate_min_delta': config.wfa_gate_min_delta,
            'wfa_gate_warmup_epochs': config.wfa_gate_warmup_epochs,
            'blos_physics_mode': config.blos_physics_mode,
            'blos_target_logtau': config.blos_target_logtau,
            'vlos_physics_mode': config.vlos_physics_mode,
            'vlos_target_logtau': config.vlos_target_logtau,
            'temp_physics_mode': config.temp_physics_mode,
            'temp_target_logtau': config.temp_target_logtau,
            'device': config.device,
            'data_path': str(config.data_path),
            'logtau_values': [float(x) for x in config.get_logtau_values().tolist()],
        }
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Physics regularization ablation study")
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--n_steps', type=int, default=-1, help='Number of training steps per epoch (-1 for all steps)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--min_step', type=int, default=112, help='Minimum training step (inclusive)')
    parser.add_argument('--max_step', type=int, default=113, help='Maximum training step (exclusive)')
    parser.add_argument('--step_size', type=int, default=1,
                       help='Step size between simulation steps (default: 1)')
    parser.add_argument('--experiment_name', type=str, default='physics_regularization_ablation',
                       help='Name for the experiment folder')
    parser.add_argument('--output_dir', type=str, 
                       default='/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments',
                       help='Base output directory')
    
    # Learning rate
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    
    # Model architecture
    parser.add_argument('--c1-filters', '--c1_filter', dest='c1_filters', type=int, default=16,
                       help='Number of filters in first conv layer (default: 16)')
    parser.add_argument('--stokes-mult-factor', '--stokes_mult_factor', dest='stokes_mult_factor',
                       type=float, default=1.0,
                       help='Scalar multiplier applied to normalized Stokes I and V before training')
    
    # Lambda values for physics terms
    parser.add_argument('--lambda_wfa', type=float, nargs='+', default=[0.01],
                       help='Weight(s) for WFA B_LOS loss. Example: --lambda_wfa 0.1 0.01 0.001')
    parser.add_argument('--lambda_doppler', type=float, nargs='+', default=[0.01],
                       help='Weight(s) for Doppler V_LOS loss. Example: --lambda_doppler 0.1 0.01')
    parser.add_argument('--lambda_temp', type=float, nargs='+', default=[0.01],
                       help='Weight(s) for temperature loss. Example: --lambda_temp 2.0 1.0 0.5')
    parser.add_argument('--no-training-data-histograms', '--no_training_data_histograms',
                       dest='no_training_data_histograms', action='store_true',
                       help='Disable train-split histogram diagnostics (T, Vz, Bz)')
    parser.add_argument('--training-hist-bins', '--training_hist_bins', dest='training_hist_bins',
                       type=int, default=120,
                       help='Number of bins for train-split histograms (default: 120)')
    parser.add_argument('--training-hist-max-samples', '--training_hist_max_samples',
                       dest='training_hist_max_samples', type=int, default=400000,
                       help='Max sampled values per parameter for histogram diagnostics (default: 400000)')
    
    # Physics modes
    parser.add_argument('--blos_physics_mode', type=str, default='tau_averaged',
                       choices=['tau_averaged', 'single_height'],
                       help='B_LOS physics comparison mode')
    parser.add_argument('--blos_target_logtau', type=float, default=None,
                       help='Target log(tau) for B_LOS single_height mode (e.g., -1.0)')
    parser.add_argument('--vlos_physics_mode', type=str, default='tau_averaged',
                       choices=['tau_averaged', 'single_height'],
                       help='V_LOS physics comparison mode')
    parser.add_argument('--vlos_target_logtau', type=float, default=None,
                       help='Target log(tau) for V_LOS single_height mode (e.g., -0.5)')
    parser.add_argument('--temp_physics_mode', type=str, default='single_height',
                       choices=['tau_averaged', 'single_height'],
                       help='Temperature physics comparison mode')
    parser.add_argument('--temp_target_logtau', type=float, default=0.0,
                       help='Target log(tau) for temperature single_height mode (default: 0.0 for photosphere)')
    
    # Experiment selection
    parser.add_argument('--experiments', type=str, nargs='+',
                       choices=['all_physics_terms', 'wfa_only', 'doppler_only', 'black_body_only', 'no_physics', 'all'],
                       default=['all'],
                       help='Which experiments to run (default: all)')
    
    # Cache-related arguments
    default_cache_dir = os.environ.get(
        'MURAM_CACHE_DIR',
        '/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.muram_cache'
    )
    parser.add_argument('--stokes-ic-mode', '--stokes_ic_mode', dest='stokes_ic_mode',
                       type=str, choices=['per_step', 'fixed_global'], default='fixed_global',
                       help='Continuum normalization mode for Stokes data')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable data caching')
    parser.add_argument('--cache-dir', '--cache_dir', dest='cache_dir', type=str, default=default_cache_dir,
                       help='Directory for cached MURaM data (or set MURAM_CACHE_DIR)')
    parser.add_argument('--balanced-cache', '--balanced_cache', dest='use_balanced_cache', action='store_true',
                       help='Enable post-balancing train-data cache')
    parser.add_argument('--balanced-cache-dir', '--balanced_cache_dir', dest='balanced_cache_dir', type=str,
                       default=os.environ.get(
                           'MURAM_BALANCED_CACHE_DIR',
                           '/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.muram_balanced_cache'
                       ),
                       help='Directory for balanced training cache')
    parser.add_argument('--clear-balanced-cache', '--clear_balanced_cache', dest='clear_balanced_cache', action='store_true',
                       help='Clear balanced training cache before running experiments')
    parser.add_argument('--balanced-cache-strategy', '--balanced_cache_strategy', dest='balanced_cache_strategy',
                       type=str, choices=['auto', 'preload', 'disk'], default='auto',
                       help='Balanced cache runtime strategy')
    parser.add_argument('--balanced-cache-ram-budget-gb', '--balanced_cache_ram_budget_gb',
                       dest='balanced_cache_ram_budget_gb', type=float, default=32.0,
                       help='RAM budget in GB used to decide balanced-cache preload feasibility')
    parser.add_argument('--balanced-cache-ram-fraction', '--balanced_cache_ram_fraction',
                       dest='balanced_cache_ram_fraction', type=float, default=0.75,
                       help='Fraction of RAM budget allowed for balanced-cache preload')

    # Region masking toggle (training only)
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        '--apply-region-mask', '--apply_region_mask',
        dest='apply_region_mask',
        action='store_true',
        help='Apply balanced 4-region mask during training (gran/intergran x strong/weak polarization).'
    )
    mask_group.add_argument(
        '--no-region-mask', '--no_region_mask',
        dest='apply_region_mask',
        action='store_false',
        help='Disable region masking and train with all available pixels.'
    )
    parser.set_defaults(apply_region_mask=True)

    bz_group = parser.add_mutually_exclusive_group()
    bz_group.add_argument(
        '--apply-bz-bin-balance', '--apply_bz_bin_balance',
        dest='apply_bz_bin_balance',
        action='store_true',
        help='Balance training pixels across Bz-strength bins by downsampling dense bins.'
    )
    bz_group.add_argument(
        '--no-bz-bin-balance', '--no_bz_bin_balance',
        dest='apply_bz_bin_balance',
        action='store_false',
        help='Disable Bz-strength balancing and keep the selected pixel set unchanged.'
    )
    parser.set_defaults(apply_bz_bin_balance=False)

    parser.add_argument(
        '--bz-balance-mode', '--bz_balance_mode',
        dest='bz_balance_mode',
        type=str,
        choices=['mean_abs', 'max_abs', 'tau_index'],
        default='mean_abs',
        help='Summary statistic used to compute the per-pixel Bz balancing score.'
    )
    parser.add_argument(
        '--bz-balance-bins', '--bz_balance_bins',
        dest='bz_balance_bins',
        type=int,
        default=12,
        help='Number of uniform Bz-strength bins used for balancing.'
    )
    parser.add_argument(
        '--bz-balance-scope', '--bz_balance_scope',
        dest='bz_balance_scope',
        type=str,
        choices=['global', 'per_step'],
        default='global',
        help='Apply Bz balancing globally across train steps or independently per step.'
    )
    parser.add_argument(
        '--bz-balance-seed', '--bz_balance_seed',
        dest='bz_balance_seed',
        type=int,
        default=42,
        help='Random seed used when selecting balanced Bz pixels.'
    )
    parser.add_argument(
        '--bz-balance-tau-idx', '--bz_balance_tau_idx',
        dest='bz_balance_tau_idx',
        type=int,
        default=None,
        help='Optical-depth index used when --bz-balance-mode=tau_index.'
    )
    parser.add_argument(
        '--bz-balance-logtau', '--bz_balance_logtau',
        dest='bz_balance_logtau',
        type=float,
        default=None,
        help='Optical-depth log(tau) value used when --bz-balance-mode=tau_index (alternative to tau index).'
    )
    
    # Optical depth remapping grid
    parser.add_argument(
        '--logtau_values',
        type=float,
        nargs='+',
        default=None,
        help='Explicit log(tau) grid values (overrides min/max/step), e.g. --logtau_values -2.0 -1.9 ... 0.0'
    )
    parser.add_argument('--logtau_min', type=float, default=-2.0,
                       help='Minimum log(tau) for range mode (default: -2.0)')
    parser.add_argument('--logtau_max', type=float, default=0.0,
                       help='Maximum log(tau) for range mode (default: 0.0)')
    parser.add_argument('--logtau_step', type=float, default=0.1,
                       help='Step in log(tau) for range mode (default: 0.1)')

    # Epoch diagnostics arguments (missing before)
    parser.add_argument('--no-epoch-plots', '--no_epoch_plots', dest='no_epoch_plots', action='store_true',
                       help='Disable per-epoch diagnostic plots')
    parser.add_argument('--no-epoch-videos', '--no_epoch_videos', dest='no_epoch_videos', action='store_true',
                       help='Disable per-epoch diagnostic videos')
    parser.add_argument('--epoch-plot-video-fps', '--epoch_plot_video_fps', dest='epoch_plot_video_fps', type=int, default=4,
                       help='FPS for epoch diagnostic videos')
    parser.add_argument('--epoch-plot-step', '--epoch_plot_step', dest='epoch_plot_step', type=int, default=None,
                       help='Monitoring step for epoch diagnostics')
    parser.add_argument('--epoch-plot-ods', '--epoch_plot_ods', dest='epoch_plot_ods', type=float, nargs='+', default=None,
                       help='Optical-depth values for epoch diagnostics')
    parser.add_argument('--epoch-plot-params', '--epoch_plot_params', dest='epoch_plot_params', type=str, nargs='+',
                       choices=['T', 'Vz', 'Bz'], default=['T', 'Vz', 'Bz'],
                       help='Parameters to include in epoch diagnostics')
    parser.add_argument('--epoch-plot-scatter-samples', '--epoch_plot_scatter_samples',
                       dest='epoch_plot_scatter_samples', type=int, default=5000,
                       help='Max sampled points per scatter plot')

    # MODEST epoch diagnostics
    parser.add_argument('--modest-epoch-plots', '--modest_epoch_plots', dest='modest_epoch_plots', action='store_true',
                       help='Enable per-epoch diagnostics on MODEST snapshot')
    parser.add_argument('--modest-cache-dir', '--modest_cache_dir', dest='modest_cache_dir', type=str,
                       default=os.environ.get(
                           "MODEST_CACHE_DIR",
                           "/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.modest_cache",
                       ),
                       help='MODEST cache directory (or set MODEST_CACHE_DIR)')
    parser.add_argument('--no-modest-cache', '--no_modest_cache', dest='no_modest_cache', action='store_true',
                       help='Disable MODEST cache usage for per-epoch diagnostics')
    parser.add_argument('--clear-modest-cache', '--clear_modest_cache', dest='clear_modest_cache', action='store_true',
                       help='Clear MODEST cache before preparing per-epoch snapshot')
    modest_input_group = parser.add_mutually_exclusive_group()
    modest_input_group.add_argument('--modest-downsample-prediction-input', '--modest_downsample_prediction_input',
                       dest='modest_downsample_prediction_input', action='store_true',
                       help='Use downsampled MODEST prediction input for per-epoch diagnostics (pixel-by-pixel)')
    modest_input_group.add_argument('--modest-upsample-prediction-input', '--modest_upsample_prediction_input',
                       dest='modest_downsample_prediction_input', action='store_false',
                       help='Use upsampled MODEST prediction input for per-epoch diagnostics')
    parser.set_defaults(modest_downsample_prediction_input=True)
    parser.add_argument('--modest-polarization-mask', '--modest_polarization_mask', dest='modest_polarization_mask',
                       action='store_true',
                       help='Apply circular polarization mask to MODEST snapshot for diagnostics')
    parser.add_argument('--modest-polarization-threshold', '--modest_polarization_threshold',
                       dest='modest_polarization_threshold', type=float, default=1e-2,
                       help='Circular polarization threshold for MODEST mask')
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
    parser.add_argument('--wfa-gate-mode', '--wfa_gate_mode', dest='wfa_gate_mode',
                       type=str, choices=['off', 'threshold', 'plateau'], default='off',
                       help='Train-time WFA activation gate mode')
    parser.add_argument('--wfa-gate-threshold', '--wfa_gate_threshold', dest='wfa_gate_threshold',
                       type=float, default=0.0,
                       help='Enable WFA once epoch train MSE is <= this threshold')
    parser.add_argument('--wfa-gate-patience', '--wfa_gate_patience', dest='wfa_gate_patience',
                       type=int, default=5,
                       help='Plateau epochs before enabling WFA')
    parser.add_argument('--wfa-gate-min-delta', '--wfa_gate_min_delta', dest='wfa_gate_min_delta',
                       type=float, default=1e-4,
                       help='Minimum epoch train MSE improvement to reset WFA plateau counter')
    parser.add_argument('--wfa-gate-warmup-epochs', '--wfa_gate_warmup_epochs', dest='wfa_gate_warmup_epochs',
                       type=int, default=0,
                       help='Minimum number of epochs before WFA gate can activate')

    args = parser.parse_args()

    if args.bz_balance_tau_idx is not None and args.bz_balance_logtau is not None:
        raise ValueError("Use either --bz-balance-tau-idx or --bz-balance-logtau, not both.")

    if args.logtau_values is not None:
        resolved_logtau = np.asarray(args.logtau_values, dtype=np.float32)
    else:
        if args.logtau_step <= 0:
            raise ValueError(f"logtau_step must be > 0, got {args.logtau_step}")
        resolved_logtau = np.arange(
            args.logtau_min,
            args.logtau_max + 0.5 * args.logtau_step,
            args.logtau_step,
            dtype=np.float32,
        )
    resolved_logtau = np.round(resolved_logtau, 6)

    if args.bz_balance_logtau is not None:
        target_logtau = float(np.round(args.bz_balance_logtau, 6))
        match_idx = np.where(np.isclose(resolved_logtau, target_logtau, atol=1e-6))[0]
        if match_idx.size == 0:
            raise ValueError(
                "Requested --bz-balance-logtau is not in the active logtau grid. "
                f"Requested: {target_logtau}. Grid: {resolved_logtau.tolist()}"
            )
        args.bz_balance_tau_idx = int(match_idx[0])
    args.cache_dir = str(Path(args.cache_dir).expanduser().resolve())
    args.balanced_cache_dir = str(Path(args.balanced_cache_dir).expanduser().resolve())
    args.modest_cache_dir = str(Path(args.modest_cache_dir).expanduser().resolve())
    args.lambda_wfa = _normalize_lambda_values(args.lambda_wfa)
    args.lambda_doppler = _normalize_lambda_values(args.lambda_doppler)
    args.lambda_temp = _normalize_lambda_values(args.lambda_temp)

    # Base configuration
    data_path = Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/data/")
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    test_steps = list(range(198, 201))
    
    # Load normalizers
    print("Loading normalizers...")
    mhd_normalizer = MhdNormalizer()
    mhd_normalizer.load(data_path / "normalization_stats/mhd_normalization.json")
    stokes_normalizer = StokesNormalizer()
    stokes_normalizer.load(data_path / "normalization_stats/stokes_normalization.json")
    print("  ✓ Normalizers loaded")
    
    tracker = ExperimentTracker(output_dir)
    
    # Print configuration
    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION".center(80))
    print("=" * 80)
    print(f"Learning rate:      {args.learning_rate:.2e}")
    print(f"Lambda WFA:         {args.lambda_wfa}")
    print(f"Lambda Doppler:     {args.lambda_doppler}")
    print(f"Lambda Temperature: {args.lambda_temp}")
    print(f"Stokes mult factor: {args.stokes_mult_factor}")
    print(f"Stokes I_c mode:    {args.stokes_ic_mode}")
    print(f"MODEST pred input:  {'downsampled' if args.modest_downsample_prediction_input else 'upsampled'}")
    print(f"WFA gate mode:      {args.wfa_gate_mode}")
    if args.wfa_gate_mode == 'threshold':
        print(f"WFA gate threshold: {args.wfa_gate_threshold}")
    elif args.wfa_gate_mode == 'plateau':
        print(
            f"WFA gate plateau:   patience={args.wfa_gate_patience}, "
            f"min_delta={args.wfa_gate_min_delta}, warmup={args.wfa_gate_warmup_epochs}"
        )
    print(f"Apply region mask:  {args.apply_region_mask}")
    print(f"Apply Bz balance:   {args.apply_bz_bin_balance}")
    print(f"Use balanced cache: {args.use_balanced_cache}")
    if args.use_balanced_cache:
        print(f"Balanced cache dir: {args.balanced_cache_dir}")
        print(
            "Balanced cache mode: "
            f"{args.balanced_cache_strategy} (RAM budget={args.balanced_cache_ram_budget_gb} GB x {args.balanced_cache_ram_fraction})"
        )
    if args.apply_bz_bin_balance:
        print(
            f"Bz balance scope:   {args.bz_balance_scope}"
        )
        print(
            f"Bz balance mode:    {args.bz_balance_mode} | bins={args.bz_balance_bins} | "
            f"tau_idx={args.bz_balance_tau_idx} | "
            f"logtau={(None if args.bz_balance_tau_idx is None else float(resolved_logtau[args.bz_balance_tau_idx]))} | "
            f"seed={args.bz_balance_seed}"
        )
    print(f"Train steps mode:  range [{args.min_step}, {args.max_step}] step={args.step_size}")
    if args.logtau_values is not None:
        print(f"log(tau) values:    {args.logtau_values}")
    else:
        print(f"log(tau) range:     [{args.logtau_min}, {args.logtau_max}] step={args.logtau_step}")
    print("=" * 80 + "\n")
    
    selected_experiments = set(args.experiments if 'all' not in args.experiments else [
        'all_physics_terms', 'wfa_only', 'doppler_only', 'black_body_only', 'no_physics'
    ])
    if 'all_physics_terms' in selected_experiments:
        if len(args.lambda_wfa) > 1 or len(args.lambda_doppler) > 1 or len(args.lambda_temp) > 1:
            raise ValueError(
                "all_physics_terms currently supports only one value per physics lambda. "
                "Provide single values for --lambda_wfa, --lambda_doppler and --lambda_temp "
                "when running all_physics_terms."
            )

    common_epoch_plot_kwargs = dict(
        enable_epoch_plots=not args.no_epoch_plots,
        epoch_plot_step=args.epoch_plot_step,
        epoch_plot_ods=args.epoch_plot_ods,
        epoch_plot_params=args.epoch_plot_params,
        epoch_plot_scatter_samples=args.epoch_plot_scatter_samples,
        enable_epoch_videos=not args.no_epoch_videos,
        epoch_plot_video_fps=args.epoch_plot_video_fps,
        enable_modest_epoch_plots=args.modest_epoch_plots,
        modest_cache_dir=args.modest_cache_dir,
        no_modest_cache=args.no_modest_cache,
        clear_modest_cache=args.clear_modest_cache,
        modest_downsample_prediction_input=args.modest_downsample_prediction_input,
        modest_polarization_mask=args.modest_polarization_mask,
        modest_polarization_threshold=args.modest_polarization_threshold,
        modest_crop_bounds=args.modest_crop_bounds,
        modest_epoch_plot_ods=args.modest_epoch_plot_ods,
        modest_epoch_plot_params=args.modest_epoch_plot_params,
        modest_epoch_plot_scatter_samples=args.modest_epoch_plot_scatter_samples,
        wfa_gate_mode=args.wfa_gate_mode,
        wfa_gate_threshold=args.wfa_gate_threshold,
        wfa_gate_patience=args.wfa_gate_patience,
        wfa_gate_min_delta=args.wfa_gate_min_delta,
        wfa_gate_warmup_epochs=args.wfa_gate_warmup_epochs,
        use_balanced_cache=args.use_balanced_cache,
        balanced_cache_dir=args.balanced_cache_dir,
        clear_balanced_cache=args.clear_balanced_cache,
        balanced_cache_strategy=args.balanced_cache_strategy,
        balanced_cache_ram_budget_gb=args.balanced_cache_ram_budget_gb,
        balanced_cache_ram_fraction=args.balanced_cache_ram_fraction,
    )

    def _build_cfg(folder_name: str, lambda_wfa: float, lambda_doppler: float, lambda_temp: float) -> TrainingConfig:
        return TrainingConfig(
            data_path=str(data_path),
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            lambda_wfa=lambda_wfa,
            lambda_doppler=lambda_doppler,
            lambda_temp=lambda_temp,
            blos_physics_mode=args.blos_physics_mode,
            blos_target_logtau=args.blos_target_logtau,
            vlos_physics_mode=args.vlos_physics_mode,
            vlos_target_logtau=args.vlos_target_logtau,
            temp_physics_mode=args.temp_physics_mode,
            temp_target_logtau=args.temp_target_logtau,
            device=args.device,
            checkpoint_dir=output_dir / folder_name / "checkpoints",
            log_dir=output_dir / folder_name / "logs",
            step_size=args.step_size,
            logtau_values=args.logtau_values,
            logtau_min=args.logtau_min,
            logtau_max=args.logtau_max,
            logtau_step=args.logtau_step,
            apply_region_mask=args.apply_region_mask,
            apply_bz_bin_balance=args.apply_bz_bin_balance,
            bz_balance_scope=args.bz_balance_scope,
            bz_balance_mode=args.bz_balance_mode,
            bz_balance_bins=args.bz_balance_bins,
            bz_balance_tau_idx=args.bz_balance_tau_idx,
            bz_balance_seed=args.bz_balance_seed,
            c1_filters=args.c1_filters,
            stokes_mult_factor=args.stokes_mult_factor,
            stokes_ic_mode=args.stokes_ic_mode,
            **common_epoch_plot_kwargs,
        )

    all_experiment_configs: dict[str, TrainingConfig] = {}

    all_experiment_configs['all_physics_terms'] = _build_cfg(
        folder_name="all_physics_terms",
        lambda_wfa=args.lambda_wfa[0],
        lambda_doppler=args.lambda_doppler[0],
        lambda_temp=args.lambda_temp[0],
    )

    all_experiment_configs['no_physics'] = _build_cfg(
        folder_name="no_physics",
        lambda_wfa=0.0,
        lambda_doppler=0.0,
        lambda_temp=0.0,
    )

    wfa_multi = len(args.lambda_wfa) > 1
    for lw in args.lambda_wfa:
        key_name, folder_name = _variant_names(
            base_key="wfa_only",
            folder_prefix="wfa",
            lambda_value=lw,
            is_multi=wfa_multi,
        )
        all_experiment_configs[key_name] = _build_cfg(
            folder_name=folder_name,
            lambda_wfa=lw,
            lambda_doppler=0.0,
            lambda_temp=0.0,
        )

    doppler_multi = len(args.lambda_doppler) > 1
    for ld in args.lambda_doppler:
        key_name, folder_name = _variant_names(
            base_key="doppler_only",
            folder_prefix="doppler",
            lambda_value=ld,
            is_multi=doppler_multi,
        )
        all_experiment_configs[key_name] = _build_cfg(
            folder_name=folder_name,
            lambda_wfa=0.0,
            lambda_doppler=ld,
            lambda_temp=0.0,
        )

    temp_multi = len(args.lambda_temp) > 1
    for lt in args.lambda_temp:
        key_name, folder_name = _variant_names(
            base_key="black_body_only",
            folder_prefix="black-body",
            lambda_value=lt,
            is_multi=temp_multi,
        )
        all_experiment_configs[key_name] = _build_cfg(
            folder_name=folder_name,
            lambda_wfa=0.0,
            lambda_doppler=0.0,
            lambda_temp=lt,
        )
    
    # Parse experiments to run
    if 'all' in args.experiments:
        experiments_to_run = list(all_experiment_configs.keys())
    else:
        experiments_to_run = []
        for selected in args.experiments:
            if selected in all_experiment_configs:
                experiments_to_run.append(selected)
                continue
            matched = [
                key for key in all_experiment_configs.keys()
                if key == selected or key.startswith(f"{selected}-lambda-")
            ]
            experiments_to_run.extend(matched)
        experiments_to_run = list(dict.fromkeys(experiments_to_run))
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS TO RUN".center(80))
    print("=" * 80)
    for i, exp_name in enumerate(experiments_to_run, 1):
        print(f"  {i}. {exp_name}")
    print("=" * 80 + "\n")
    
    # Initialize shared cache for all experiments
    cache = None
    if not args.no_cache:
        cache = MuramDataCache(cache_dir=args.cache_dir, compression='gzip')
        print(f"Shared MURaM data cache: {args.cache_dir}")
        print("\nInitial Cache Status:")
        cache.print_cache_info()

    if args.clear_balanced_cache and args.use_balanced_cache:
        balanced_cache = BalancedTrainDataCache(cache_dir=args.balanced_cache_dir)
        balanced_cache.clear()
        print(f"Cleared balanced cache: {args.balanced_cache_dir}")
    
    # Run selected experiments with shared cache
    for name in experiments_to_run:
        if name not in all_experiment_configs:
            print(f"⚠ Warning: Unknown experiment '{name}', skipping...")
            continue
        
        config = all_experiment_configs[name]
        config.use_cache = not args.no_cache
        config.cache_dir = args.cache_dir
        config.use_balanced_cache = args.use_balanced_cache
        config.balanced_cache_dir = args.balanced_cache_dir
        config.clear_balanced_cache = False
        config.balanced_cache_strategy = args.balanced_cache_strategy
        config.balanced_cache_ram_budget_gb = args.balanced_cache_ram_budget_gb
        config.balanced_cache_ram_fraction = args.balanced_cache_ram_fraction
        
        results = run_single_experiment(
            experiment_name=name,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            test_steps=test_steps,
            n_steps_per_epoch=args.n_steps,
            min_step=args.min_step,
            max_step=args.max_step,
            step_size=args.step_size,
            cache=cache,  # Share cache across experiments
            plot_training_data_histograms=not args.no_training_data_histograms,
            training_hist_bins=args.training_hist_bins,
            training_hist_max_samples=args.training_hist_max_samples,
        )
        
        tracker.add_experiment(name, results)
    
    tracker.save_results()
    tracker.print_summary_table()
    tracker.generate_comparison_plots()
    tracker.plot_individual_loss_curves()
    tracker.plot_testset_correlation_and_rrmse()
    
    print("\n✓ Experiment complete!")
    print(f"Results saved to: {output_dir}")
    
if __name__ == "__main__":
    main()