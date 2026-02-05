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
import json
import time
from pathlib import Path
from typing import Dict, List
import warnings
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
from utils.grad_norm import GradNormScheduler
from scripts.base_training import (
    TrainingConfig,
    load_and_prepare_step, validate, train_epoch
)


class ExperimentTracker:
    """Tracks metrics across different experimental conditions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results = {}
        
    def add_experiment(self, name: str, metrics: Dict):
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
        
        print("\n" + "=" * 120)
        print("PHYSICS REGULARIZATION ABLATION STUDY - SUMMARY".center(120))
        print("=" * 120)
        
        # Header
        header = f"{'Experiment':<25} {'Val Loss':<12} {'Time (min)':<12} {'B_LOS RRMSE':<15} {'V_LOS RRMSE':<15} {'Best?':<8}"
        print(header)
        print("-" * 120)
        
        # Find best model
        best_exp = min(self.results.keys(), 
                      key=lambda x: self.results[x]['final_val_loss'])
        
        # Print rows
        for exp_name in self.results.keys():
            metrics = self.results[exp_name]
            is_best = "★ YES" if exp_name == best_exp else "";
            
            row = (f"{exp_name:<25} "
                  f"{metrics['final_val_loss']:<12.6f} "
                  f"{metrics['training_time_minutes']:<12.1f} "
                  f"{metrics['test_metrics']['blos_rrmse_tau_avg']:<15.6f} "
                  f"{metrics['test_metrics']['vlos_rrmse_tau_avg']:<15.6f} "
                  f"{is_best:<8}")
            print(row)
        
        print("=" * 120)
        
        # Print key findings
        print("\n📊 KEY FINDINGS:")
        print("-" * 120)
        
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
        
        print("=" * 120 + "\n")
    
    def plot_individual_loss_curves(self):
        """Generate individual plots for each experiment showing all loss components."""
        for exp_name, results in self.results.items():
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Loss Components - {exp_name}', fontsize=14, fontweight='bold')
            
            epochs = range(1, len(results['train_loss_history']) + 1)
            
            # Total loss
            ax1 = axes[0, 0]
            ax1.plot(epochs, results['train_loss_history'], 'b-o', label='Total Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Total Training Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # MSE loss
            ax2 = axes[0, 1]
            if 'mse_loss_history' in results:
                ax2.plot(epochs, results['mse_loss_history'], 'g-s', label='MSE Loss', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('MSE Loss Component')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            # Physics loss breakdown
            ax3 = axes[1, 0]
            if 'physics_loss_history' in results and any(l > 0 for l in results['physics_loss_history']):
                ax3.plot(epochs, results['physics_loss_history'], 'r-^', label='Total Physics', linewidth=2)
            if 'wfa_loss_history' in results and any(l > 0 for l in results['wfa_loss_history']):
                ax3.plot(epochs, results['wfa_loss_history'], 'm--', label='WFA', linewidth=1.5)
            if 'doppler_loss_history' in results and any(l > 0 for l in results['doppler_loss_history']):
                ax3.plot(epochs, results['doppler_loss_history'], 'c--', label='Doppler', linewidth=1.5)
            if 'temperature_loss_history' in results and any(l > 0 for l in results['temperature_loss_history']):
                ax3.plot(epochs, results['temperature_loss_history'], 'y--', label='Temperature', linewidth=1.5)
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
            plot_path = self.output_dir / f"{exp_name}_loss_curves.png"
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
            
        print(f"Individual loss curve plots saved to {self.output_dir}")
        
    def generate_comparison_plots(self):
        """Generate comparison visualizations."""
        if not self.results:
            print("No results to plot")
            return
        
        fig = plt.figure(figsize=(20, 18))
        
        # Extract data
        experiments = list(self.results.keys())
        
        # 1. Validation Loss Comparison
        ax1 = plt.subplot(3, 3, 1)
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
        ax2 = plt.subplot(3, 3, 2)
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
        ax3 = plt.subplot(3, 3, 3)
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
        ax4 = plt.subplot(3, 3, 4)
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
        
        # 5. Convergence curves (validation loss)
        ax5 = plt.subplot(3, 3, 5)
        for exp in experiments:
            val_history = self.results[exp]['val_loss_history']
            epochs = range(1, len(val_history) + 1)
            ax5.plot(epochs, val_history, marker='o', label=exp, linewidth=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Validation Loss')
        ax5.set_title('Validation Loss Convergence')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Relative improvement matrix
        ax6 = plt.subplot(3, 3, 6)
        baseline_name = 'no_physics'
        if baseline_name in self.results:
            baseline_val = self.results[baseline_name]['final_val_loss']
            baseline_blos = self.results[baseline_name]['test_metrics']['blos_rrmse_tau_avg']
            baseline_vlos = self.results[baseline_name]['test_metrics']['vlos_rrmse_tau_avg']
            
            improvements = []
            for exp in experiments:
                if exp == baseline_name:
                    improvements.append([0, 0, 0])
                else:
                    val_imp = (baseline_val - self.results[exp]['final_val_loss']) / baseline_val * 100
                    blos_imp = (baseline_blos - self.results[exp]['test_metrics']['blos_rrmse_tau_avg']) / baseline_blos * 100
                    vlos_imp = (baseline_vlos - self.results[exp]['test_metrics']['vlos_rrmse_tau_avg']) / baseline_vlos * 100
                    improvements.append([val_imp, blos_imp, vlos_imp])
            
            improvements = np.array(improvements)
            
            im = ax6.imshow(improvements.T, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
            ax6.set_xticks(range(len(experiments)))
            ax6.set_xticklabels(experiments, rotation=45, ha='right')
            ax6.set_yticks([0, 1, 2])
            ax6.set_yticklabels(['Val Loss', 'B_LOS RRMSE', 'V_LOS RRMSE'])
            ax6.set_title('% Improvement over Baseline\n(Positive = Better)')
            
            for i in range(len(experiments)):
                for j in range(3):
                    ax6.text(i, j, f'{improvements[i, j]:.1f}%',
                           ha="center", va="center", color="black", fontsize=9)
            
            plt.colorbar(im, ax=ax6, label='% Improvement')
        
        # 7. Total Loss Curves
        ax7 = plt.subplot(3, 3, 7)
        for exp in experiments:
            if 'train_loss_history' in self.results[exp]:
                loss_history = self.results[exp]['train_loss_history']
                epochs = range(1, len(loss_history) + 1)
                ax7.plot(epochs, loss_history, marker='o', label=exp, linewidth=2, markersize=4)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Total Loss')
        ax7.set_title('Total Loss Convergence (Training)')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        ax7.set_yscale('log')
        
        # 8. MSE Loss Component
        ax8 = plt.subplot(3, 3, 8)
        for exp in experiments:
            if 'mse_loss_history' in self.results[exp]:
                loss_history = self.results[exp]['mse_loss_history']
                epochs = range(1, len(loss_history) + 1)
                ax8.plot(epochs, loss_history, marker='s', label=exp, linewidth=2, markersize=4)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('MSE Loss')
        ax8.set_title('MSE Loss Component')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
        ax8.set_yscale('log')
        
        # 9. Physics Loss Components
        ax9 = plt.subplot(3, 3, 9)
        for exp in experiments:
            if 'physics_loss_history' in self.results[exp]:
                loss_history = self.results[exp]['physics_loss_history']
                if len(loss_history) > 0 and any(l > 0 for l in loss_history):
                    epochs = range(1, len(loss_history) + 1)
                    ax9.plot(epochs, loss_history, marker='^', label=exp, linewidth=2, markersize=4)
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('Physics Loss')
        ax9.set_title('Physics Loss Components')
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3)
        ax9.set_yscale('log')
        
        plt.suptitle('Physics Regularization Ablation Study', fontsize=16, y=0.997)
        plt.tight_layout()
        
        plot_path = self.output_dir / "comparison_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to {plot_path}")
        plt.show()

def compute_tau_averaged_metrics(
    model: PhysicsInformedMSCNN,
    test_steps: List[int],
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    logtau_values: np.ndarray,
) -> Dict[str, float]:
    """Evaluate model on test steps using tau-averaged physics metrics."""
    from scipy.stats import pearsonr
    
    model.eval()
    device = config.device
    
    all_pred_blos = []
    all_true_blos = []
    all_pred_vlos = []
    all_true_vlos = []
    
    with torch.no_grad():
        for step in tqdm(test_steps, desc="Evaluating test steps"):
            try:
                dataset, approx_data = load_and_prepare_step(
                    step=step,
                    config=config,
                    mhd_normalizer=mhd_normalizer,
                    stokes_normalizer=stokes_normalizer,
                )
                
                dataloader = DataLoader(
                    dataset, batch_size=512, shuffle=False, num_workers=0
                )
                
                true_blos = approx_data['blos'].flatten()
                true_vlos = approx_data['vlos'].flatten()
                
                step_pred_blos = []
                step_pred_vlos = []
                
                for stokes_batch, _, spatial_idx_batch in dataloader:
                    stokes_batch = stokes_batch.to(device)
                    predictions = model(stokes_batch)
                    
                    # Convert predictions to numpy for denormalization
                    predictions_np = predictions.cpu().numpy()
                    
                    # Denormalize using inverse_transform (returns dict with 'T', 'Vz', 'Bz')
                    pred_denorm = mhd_normalizer.inverse_transform(
                        predictions_np, param_order=['T', 'Vz', 'Bz']
                    )
                    
                    # Compute tau-averaged B_LOS
                    tau_linear = 10 ** logtau_values
                    dtau = np.diff(tau_linear)
                    integral_dtau = tau_linear[-1] - tau_linear[0]
                    
                    # Bz shape: (batch_size, n_tau=21)
                    Bz_avg = (pred_denorm["Bz"][:, :-1] + pred_denorm["Bz"][:, 1:]) / 2
                    integral_Bz = np.sum(Bz_avg * dtau[np.newaxis, :], axis=1)
                    pred_blos_batch = integral_Bz / integral_dtau
                    
                    # Compute tau-averaged V_LOS
                    # Vz shape: (batch_size, n_tau=21)
                    Vz_avg = (pred_denorm["Vz"][:, :-1] + pred_denorm["Vz"][:, 1:]) / 2
                    integral_Vz = np.sum(Vz_avg * dtau[np.newaxis, :], axis=1)
                    pred_vlos_batch = integral_Vz / integral_dtau
                    
                    step_pred_blos.append(pred_blos_batch)
                    step_pred_vlos.append(pred_vlos_batch)
                
                all_pred_blos.append(np.concatenate(step_pred_blos))
                all_true_blos.append(true_blos)
                all_pred_vlos.append(np.concatenate(step_pred_vlos))
                all_true_vlos.append(true_vlos)
                
            except Exception as e:
                print(f"Warning: Failed to evaluate step {step}: {e}")
                continue
    
    all_pred_blos = np.concatenate(all_pred_blos)
    all_true_blos = np.concatenate(all_true_blos)
    all_pred_vlos = np.concatenate(all_pred_vlos)
    all_true_vlos = np.concatenate(all_true_vlos)
    
    rmse_blos = np.sqrt(np.mean((all_pred_blos - all_true_blos) ** 2))
    rrmse_blos = rmse_blos / (np.mean(np.abs(all_true_blos)) + 1e-10)
    
    rmse_vlos = np.sqrt(np.mean((all_pred_vlos - all_true_vlos) ** 2))
    rrmse_vlos = rmse_vlos / (np.mean(np.abs(all_true_vlos)) + 1e-10)
    
    corr_blos, _ = pearsonr(all_pred_blos, all_true_blos)
    corr_vlos, _ = pearsonr(all_pred_vlos, all_true_vlos)
    
    return {
        'blos_rrmse_tau_avg': float(rrmse_blos),
        'vlos_rrmse_tau_avg': float(rrmse_vlos),
        'blos_correlation': float(corr_blos),
        'vlos_correlation': float(corr_vlos),
        'blos_rmse': float(rmse_blos),
        'vlos_rmse': float(rmse_vlos),
    }

def run_single_experiment(
    experiment_name: str,
    config: TrainingConfig,
    mhd_normalizer: MhdNormalizer,
    stokes_normalizer: StokesNormalizer,
    test_steps: List[int],
    n_steps_per_epoch: int = 20,
    min_step: int = 60,
    max_step: int = 200,
) -> Dict:
    """Run a single training experiment."""
    print("\n" + "=" * 100)
    print(f"EXPERIMENT: {experiment_name}".center(100))
    print("=" * 100)
    print(f"Device: {config.device}")
    print(f"Number of epochs: {config.n_epochs}")
    print(f"Training step range: {min_step} to {max_step}")
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
    print(f"Use GradNorm: {config.use_gradnorm}")
    if config.use_gradnorm:
        print(f"GradNorm alpha: {config.gradnorm_alpha}")
        print(f"GradNorm update freq: {config.gradnorm_update_freq}")
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
            'scheduler_factor': config.scheduler_factor,
            'scheduler_patience': config.scheduler_patience,
        },
        'physics_config': {
            'lambda_wfa': config.lambda_wfa,
            'lambda_doppler': config.lambda_doppler,
            'lambda_temp': config.lambda_temp,
            'use_gradnorm': config.use_gradnorm,
            'gradnorm_alpha': config.gradnorm_alpha,
            'gradnorm_update_freq': config.gradnorm_update_freq,
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
        },
        'model_config': {
            'scales': [1, 2, 3],
            'in_channels': 2,
            'c1_filters': 16,
            'c2_filters': 32,
            'kernel_size': 5,
            'pool_size': 2,
            'n_linear_layers': 4,
            'dropout_rate': 0.2,
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
    model = PhysicsInformedMSCNN(
        scales=[1, 2, 3],
        in_channels=2,
        c1_filters=16,
        c2_filters=32,
        kernel_size=5,
        pool_size=2,
        n_linear_layers=4,
        dropout_rate=0.2,
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
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        verbose=True
    )
    
    # Initialize GradNorm scheduler if enabled
    gradnorm_scheduler = None
    if config.use_gradnorm and any([config.lambda_wfa > 0, config.lambda_doppler > 0, config.lambda_temp > 0]):
        print("\nInitializing GradNorm scheduler for experiment...")
        initial_weights = [1.0, 1.0, 1.0, 1.0]  # MSE, WFA, Doppler, Temp
        gradnorm_scheduler = GradNormScheduler(
            num_tasks=4,
            alpha=config.gradnorm_alpha,
            initial_weights=initial_weights,
            device=config.device
        )
        print(f"  ✓ GradNorm initialized with alpha={config.gradnorm_alpha}")
    
    # Prepare train/val split
    all_steps = list(range(min_step, max_step+1))
    train_steps = [s for s in all_steps if s not in test_steps]
    
    import random
    random.seed(42)
    n_val = max(1, len(train_steps) // 10)
    val_steps = random.sample(train_steps, n_val)
    train_steps = [s for s in train_steps if s not in val_steps]
    
    # Training loop
    start_time = time.time()
    val_loss_history = []
    train_loss_history = []
    mse_loss_history = []
    physics_loss_history = []
    wfa_loss_history = []
    doppler_loss_history = []
    temperature_loss_history = []
    
    for epoch in range(config.n_epochs):
        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")
        
        # Use the shared train_epoch function with GradNorm
        epoch_metrics = train_epoch(
            model=model,
            train_steps=train_steps,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            optimizer=optimizer,
            epoch=epoch,
            logger=None,
            n_steps_per_epoch=n_steps_per_epoch,
            gradnorm_scheduler=gradnorm_scheduler,
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
        
        # Validation
        avg_val_loss = validate(
            model=model,
            val_steps=val_steps[:5],
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
        )
        
        val_loss_history.append(avg_val_loss)
        scheduler.step(avg_val_loss)
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
        
        # Print GradNorm weights if enabled
        if gradnorm_scheduler is not None:
            weights = gradnorm_scheduler.task_weights.detach().cpu().numpy()
            print(f"  GradNorm Weights:")
            print(f"    MSE: {weights[0]:.4f}, WFA: {weights[1]:.4f}, "
                  f"Doppler: {weights[2]:.4f}, Temp: {weights[3]:.4f}")
        
        print("=" * 100)
    
    training_time = (time.time() - start_time) / 60
    
    print("\nEvaluating on test set...")
    test_metrics = compute_tau_averaged_metrics(
        model=model,
        test_steps=test_steps,
        config=config,
        mhd_normalizer=mhd_normalizer,
        stokes_normalizer=stokes_normalizer,
        logtau_values=np.arange(-2.0, 0.1, 0.1),
    )
    
    # Save model
    if config.checkpoint_dir:
        model_path = config.checkpoint_dir.parent / "final_model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Include GradNorm state if used
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'test_metrics': test_metrics,
        }
        
        if gradnorm_scheduler is not None:
            checkpoint_data['gradnorm_state'] = gradnorm_scheduler.state_dict()
        
        torch.save(checkpoint_data, model_path)
    
    return {
        'final_val_loss': val_loss_history[-1],
        'val_loss_history': val_loss_history,
        'train_loss_history': train_loss_history,
        'mse_loss_history': mse_loss_history,
        'physics_loss_history': physics_loss_history,
        'wfa_loss_history': wfa_loss_history,
        'doppler_loss_history': doppler_loss_history,
        'temperature_loss_history': temperature_loss_history,
        'training_time_minutes': training_time,
        'test_metrics': test_metrics,
        'config': {
            'lambda_wfa': config.lambda_wfa,
            'lambda_doppler': config.lambda_doppler,
            'lambda_temp': config.lambda_temp,
            'use_gradnorm': config.use_gradnorm,
            'gradnorm_alpha': config.gradnorm_alpha,
            "gradnorm_update_freq": config.gradnorm_update_freq,
            'blos_physics_mode': config.blos_physics_mode,
            'blos_target_logtau': config.blos_target_logtau,
            'vlos_physics_mode': config.vlos_physics_mode,
            'vlos_target_logtau': config.vlos_target_logtau,
            'temp_physics_mode': config.temp_physics_mode,
            'temp_target_logtau': config.temp_target_logtau,
            'device': config.device,
            'data_path': str(config.data_path),
        }
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Physics regularization ablation study")
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--n_steps', type=int, default=-1, help='Number of training steps per epoch (-1 for all steps)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--min_step', type=int, default=60, help='Minimum training step (inclusive)')
    parser.add_argument('--max_step', type=int, default=201, help='Maximum training step (exclusive)')
    parser.add_argument('--experiment_name', type=str, default='physics_regularization_ablation',
                       help='Name for the experiment folder')
    parser.add_argument('--output_dir', type=str, 
                       default='/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments',
                       help='Base output directory')
    
    # Learning rate
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    
    # Lambda values for physics terms
    parser.add_argument('--lambda_wfa', type=float, default=0.01,
                       help='Weight for WFA B_LOS loss (default: 0.01, use 0.0 to disable)')
    parser.add_argument('--lambda_doppler', type=float, default=0.01,
                       help='Weight for Doppler V_LOS loss (default: 0.01, use 0.0 to disable)')
    parser.add_argument('--lambda_temp', type=float, default=0.01,
                       help='Weight for temperature loss (default: 0.01, use 0.0 to disable)')
    
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
    
    # GradNorm (optional)
    parser.add_argument('--use_gradnorm', action='store_true',
                       help='Enable GradNorm automatic loss balancing')
    parser.add_argument('--gradnorm_alpha', type=float, default=1.5,
                       help='GradNorm alpha parameter (default: 1.5)')
    
    args = parser.parse_args()
    
    # Base configuration
    data_path = Path("/scratchsan/observatorio/juagudeloo/data/")
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
    print(f"Use GradNorm:       {args.use_gradnorm}")
    if args.use_gradnorm:
        print(f"GradNorm alpha:     {args.gradnorm_alpha}")
    print("=" * 80 + "\n")
    
    # Define experiments using TrainingConfig with command-line arguments
    experiments = [
        # 1. All physics terms with specified lambdas
        TrainingConfig(
            data_path=str(data_path),
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            lambda_wfa=args.lambda_wfa,
            lambda_doppler=args.lambda_doppler,
            lambda_temp=args.lambda_temp,
            use_gradnorm=args.use_gradnorm,
            gradnorm_alpha=args.gradnorm_alpha,
            blos_physics_mode=args.blos_physics_mode,
            blos_target_logtau=args.blos_target_logtau,
            vlos_physics_mode=args.vlos_physics_mode,
            vlos_target_logtau=args.vlos_target_logtau,
            temp_physics_mode=args.temp_physics_mode,
            temp_target_logtau=args.temp_target_logtau,
            device=args.device,
            checkpoint_dir=output_dir / "all_physics_terms" / "checkpoints",
            log_dir=output_dir / "all_physics_terms" / "logs",
        ),
        # 2. WFA only
        TrainingConfig(
            data_path=str(data_path),
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            lambda_wfa=args.lambda_wfa,
            lambda_doppler=0.0,
            lambda_temp=0.0,
            use_gradnorm=False,  # No GradNorm for single-term experiments
            blos_physics_mode=args.blos_physics_mode,
            blos_target_logtau=args.blos_target_logtau,
            vlos_physics_mode=args.vlos_physics_mode,
            vlos_target_logtau=args.vlos_target_logtau,
            temp_physics_mode=args.temp_physics_mode,
            temp_target_logtau=args.temp_target_logtau,
            device=args.device,
            checkpoint_dir=output_dir / "wfa_only" / "checkpoints",
            log_dir=output_dir / "wfa_only" / "logs",
        ),
        # 3. Doppler only
        TrainingConfig(
            data_path=str(data_path),
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            lambda_wfa=0.0,
            lambda_doppler=args.lambda_doppler,
            lambda_temp=0.0,
            use_gradnorm=False,  # No GradNorm for single-term experiments
            blos_physics_mode=args.blos_physics_mode,
            blos_target_logtau=args.blos_target_logtau,
            vlos_physics_mode=args.vlos_physics_mode,
            vlos_target_logtau=args.vlos_target_logtau,
            temp_physics_mode=args.temp_physics_mode,
            temp_target_logtau=args.temp_target_logtau,
            device=args.device,
            checkpoint_dir=output_dir / "doppler_only" / "checkpoints",
            log_dir=output_dir / "doppler_only" / "logs",
        ),
        # 4. Black-body temperature only
        TrainingConfig(
            data_path=str(data_path),
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            lambda_wfa=0.0,
            lambda_doppler=0.0,
            lambda_temp=args.lambda_temp,
            use_gradnorm=False,  # No GradNorm for single-term experiments
            blos_physics_mode=args.blos_physics_mode,
            blos_target_logtau=args.blos_target_logtau,
            vlos_physics_mode=args.vlos_physics_mode,
            vlos_target_logtau=args.vlos_target_logtau,
            temp_physics_mode=args.temp_physics_mode,
            temp_target_logtau=args.temp_target_logtau,
            device=args.device,
            checkpoint_dir=output_dir / "black_body_only" / "checkpoints",
            log_dir=output_dir / "black_body_only" / "logs",
        ),
        # 5. No physics (baseline)
        TrainingConfig(
            data_path=str(data_path),
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            lambda_wfa=0.0,
            lambda_doppler=0.0,
            lambda_temp=0.0,
            use_gradnorm=False,
            blos_physics_mode=args.blos_physics_mode,
            blos_target_logtau=args.blos_target_logtau,
            vlos_physics_mode=args.vlos_physics_mode,
            vlos_target_logtau=args.vlos_target_logtau,
            temp_physics_mode=args.temp_physics_mode,
            temp_target_logtau=args.temp_target_logtau,
            device=args.device,
            checkpoint_dir=output_dir / "no_physics" / "checkpoints",
            log_dir=output_dir / "no_physics" / "logs",
        ),
    ]
    
    experiment_names = [
        'all_physics_terms',
        'wfa_only',
        'doppler_only',
        'black_body_only',
        'no_physics'
    ]
    
    # Run experiments
    for name, config in zip(experiment_names, experiments):
        results = run_single_experiment(
            experiment_name=name,
            config=config,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            test_steps=test_steps,
            n_steps_per_epoch=args.n_steps,
            min_step=args.min_step,
            max_step=args.max_step,
        )
        
        tracker.add_experiment(name, results)
            
    tracker.save_results()
    tracker.print_summary_table()
    tracker.generate_comparison_plots()
    tracker.plot_individual_loss_curves()
    
    print("\n✓ Experiment complete!")
    print(f"Results saved to: {output_dir}")
    
    if args.use_gradnorm:
        print("\n📊 GradNorm was enabled for physics experiments")
        print(f"   Alpha parameter: {args.gradnorm_alpha}")


if __name__ == "__main__":
    main()