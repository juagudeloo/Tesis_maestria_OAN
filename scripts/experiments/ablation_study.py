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
from utils.cache_manage import MuramDataCache
from scripts.base_training import (
    TrainingConfig,
    load_and_prepare_step, validate, train_epoch, MetricsLogger,
    initialize_wfa_gate_state, update_wfa_gate_state,
    generate_epoch_diagnostic_plots, generate_epoch_diagnostic_videos,
    prepare_modest_epoch_snapshot, generate_epoch_modest_diagnostic_plots,
    generate_epoch_modest_diagnostic_videos,
)


def _series_for_log_plot(values) -> np.ndarray:
    """Convert non-positive values to NaN for stable log-scale plotting."""
    arr = np.asarray(values, dtype=float)
    return np.where(arr > 0, arr, np.nan)


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
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
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
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Loss Components - {exp_name}', fontsize=14, fontweight='bold')
            
            epochs = range(1, len(results['train_loss_history']) + 1)
            
            # Total loss
            ax1 = axes[0, 0]
            ax1.plot(epochs, _series_for_log_plot(results['train_loss_history']), 'b-o', label='Total Loss', linewidth=2)
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
                ax3.plot(epochs, _series_for_log_plot(results['wfa_loss_history']), 'm--', label='WFA', linewidth=1.5)
            if 'doppler_loss_history' in results and any(l > 0 for l in results['doppler_loss_history']):
                ax3.plot(epochs, _series_for_log_plot(results['doppler_loss_history']), 'c--', label='Doppler', linewidth=1.5)
            if 'temperature_loss_history' in results and any(l > 0 for l in results['temperature_loss_history']):
                ax3.plot(epochs, _series_for_log_plot(results['temperature_loss_history']), 'y--', label='Temperature', linewidth=1.5)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Physics Loss Components')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            
            # Validation loss
            ax4 = axes[1, 1]
            ax4.plot(epochs, results['val_loss_history'], 'orange', marker='o', label='Validation Loss', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Validation Loss')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save individual plot
            plot_path = self.output_dir / f"{exp_name}-loss_curves.png"
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
            
        print(f"Individual loss curve plots saved to {self.output_dir}")
        
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
            ax6.plot(epochs, val_history, marker='o', label=exp, linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Validation Loss')
        ax6.set_title('Validation Loss Convergence')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
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
            dataset, approx_data = load_and_prepare_step(
                step=step,
                config=config,
                mhd_normalizer=mhd_normalizer,
                stokes_normalizer=stokes_normalizer,
                cache=cache,
            )

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
            'test_steps': test_steps,
            'n_steps_per_epoch': n_steps_per_epoch,
            'logtau_values': [float(x) for x in config.get_logtau_values().tolist()],
            'balanced_region_training': bool(config.apply_region_mask),
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
    
    # Initialize logger
    logger = MetricsLogger(config.log_dir)
    monitor_step_for_epoch_plots = (
        config.epoch_plot_step if config.epoch_plot_step is not None else val_steps[0]
    )

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
            print(f"  ⚠ Failed to prepare MODEST snapshot diagnostics: {e}")
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
    train_wfa_enabled_history = []
    total_training_pixels = 0
    wfa_gate_state = initialize_wfa_gate_state(config)
    wfa_gate_trigger_epoch = None
    wfa_gate_trigger_reason = None
    
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
        'config': {
            'lambda_wfa': config.lambda_wfa,
            'lambda_doppler': config.lambda_doppler,
            'lambda_temp': config.lambda_temp,
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
    
    # Lambda values for physics terms
    parser.add_argument('--lambda_wfa', type=float, nargs='+', default=[0.01],
                       help='Weight(s) for WFA B_LOS loss. Example: --lambda_wfa 0.1 0.01 0.001')
    parser.add_argument('--lambda_doppler', type=float, nargs='+', default=[0.01],
                       help='Weight(s) for Doppler V_LOS loss. Example: --lambda_doppler 0.1 0.01')
    parser.add_argument('--lambda_temp', type=float, nargs='+', default=[0.01],
                       help='Weight(s) for temperature loss. Example: --lambda_temp 2.0 1.0 0.5')
    
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
    args.cache_dir = str(Path(args.cache_dir).expanduser().resolve())
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
    print(f"Stokes I_c mode:    {args.stokes_ic_mode}")
    print(f"WFA gate mode:      {args.wfa_gate_mode}")
    if args.wfa_gate_mode == 'threshold':
        print(f"WFA gate threshold: {args.wfa_gate_threshold}")
    elif args.wfa_gate_mode == 'plateau':
        print(
            f"WFA gate plateau:   patience={args.wfa_gate_patience}, "
            f"min_delta={args.wfa_gate_min_delta}, warmup={args.wfa_gate_warmup_epochs}"
        )
    print(f"Apply region mask:  {args.apply_region_mask}")
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
            c1_filters=args.c1_filters,
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
    
    # Run selected experiments with shared cache
    for name in experiments_to_run:
        if name not in all_experiment_configs:
            print(f"⚠ Warning: Unknown experiment '{name}', skipping...")
            continue
        
        config = all_experiment_configs[name]
        config.use_cache = not args.no_cache
        config.cache_dir = args.cache_dir
        
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
        )
        
        tracker.add_experiment(name, results)
    
    tracker.save_results()
    tracker.print_summary_table()
    tracker.generate_comparison_plots()
    tracker.plot_individual_loss_curves()
    
    print("\n✓ Experiment complete!")
    print(f"Results saved to: {output_dir}")
    
if __name__ == "__main__":
    main()