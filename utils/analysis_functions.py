"""Analysis functions for comparing model predictions against ground truth.

This module provides comprehensive analysis tools for evaluating atmospheric
parameter predictions from PINN-MSCNN models against SPINOR ground truth data.
Includes spatial maps, statistical metrics, uncertainty quantification, and
error analysis across optical depths.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import binned_statistic, pearsonr

from utils.normalizer import MhdNormalizer


def _save_metrics_json(metrics: dict, save_dir: Path, filename: str):
    """Save metrics dictionary to JSON file."""
    json_filename = Path(filename).stem + ".json"
    json_path = save_dir / json_filename
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved to {json_path}")


def plot_prediction_comparison(
    mean_atm: dict,
    std_atm: dict,
    ground_truth: dict,
    mag_to_plot: str = "T",
    od_to_plot: float = 0.0,
    logtau: np.ndarray = None,
    model_label: str = "Model",
    figsize: tuple = (14, 18),
    save_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None
):
    """
    Plot comprehensive comparison between predicted mean, ground truth, and uncertainty.
    
    Parameters
    ----------
    mean_atm : dict
        Dictionary with keys ['T', 'Vz', 'Bz'] containing mean predictions (H, W, n_heights)
    std_atm : dict
        Dictionary with keys ['T', 'Vz', 'Bz'] containing standard deviations (H, W, n_heights)
    ground_truth : dict
        Dictionary with keys ['T', 'Vlos', 'Blos'] containing ground truth data
    mag_to_plot : str
        Parameter to plot: 'T', 'Vz', or 'Bz'
    od_to_plot : float
        Optical depth value to plot
    logtau : np.ndarray, optional
        Array of log(tau) values
    model_label : str
        Label for the model
    figsize : tuple
        Figure size
    save_dir : str or Path, optional
        Base directory to save figures. A subfolder 'prediction_comparison' will be created.
    filename : str, optional
        Filename for the saved figure. If None, displays the plot instead.
    """
    if logtau is None:
        logtau = np.arange(-2, 0.1, 0.1)
    
    logtau_idx = np.argmin(np.abs(logtau - od_to_plot))
    
    title_map = {"T": "Temperature", "Vz": "Line-of-sight Velocity", "Bz": "Line-of-sight Magnetic Field"}
    units_map = {"T": "K", "Vz": "km/s", "Bz": "G"}
    modest_key_mapping = {"T": "T", "Vz": "Vlos", "Bz": "Blos"}
    color_mapping = {"T": "inferno", "Vz": "bwr_r", "Bz": "PiYG"}
    uncertainty_cmap = {"T": "YlOrRd", "Vz": "viridis", "Bz": "viridis"}
    
    pred_mean = mean_atm[mag_to_plot][:, :, logtau_idx]
    pred_std = std_atm[mag_to_plot][:, :, logtau_idx]
    gt = ground_truth[modest_key_mapping[mag_to_plot]][od_to_plot]
    
    difference = pred_mean - gt
    
    q0, q99 = np.percentile(gt, [1, 99])
    if mag_to_plot in ["Vz", "Bz"]:
        vmax = max(np.abs(q0), np.abs(q99))
        vmin = -vmax
    else:
        vmin, vmax = q0, q99
    
    diff_max = np.percentile(np.abs(difference), 99)
    diff_vmin, diff_vmax = -diff_max, diff_max
    
    rmse = np.sqrt(np.mean(difference**2))
    bias = np.mean(difference)
    corr, p_value = pearsonr(pred_mean.flatten(), gt.flatten())
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    ax = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    
    fig.suptitle(f"{model_label}: {title_map[mag_to_plot]} at log(τ)={logtau[logtau_idx]:.1f}",
                 fontsize=18, fontweight='bold')
    
    # 1. Predicted Mean
    im0 = ax[0].imshow(pred_mean, cmap=color_mapping[mag_to_plot], vmax=vmax, vmin=vmin)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im0, cax=cax, label=units_map[mag_to_plot])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("Predicted Mean", fontsize=14, fontweight='bold')
    
    # 2. Ground Truth
    im1 = ax[1].imshow(gt, cmap=color_mapping[mag_to_plot], vmax=vmax, vmin=vmin)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im1, cax=cax, label=units_map[mag_to_plot])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
    
    # 3. Difference
    im2 = ax[2].imshow(difference, cmap='RdBu_r', vmax=diff_vmax, vmin=diff_vmin)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax, label=f"Δ {units_map[mag_to_plot]}")
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title(f"Difference (Pred - GT)\nRMSE: {rmse:.2f} | Bias: {bias:.2f} {units_map[mag_to_plot]}",
                   fontsize=14, fontweight='bold')
    
    # 4. Predicted Std Dev
    im3 = ax[3].imshow(pred_std, cmap=uncertainty_cmap[mag_to_plot])
    divider = make_axes_locatable(ax[3])
    cax2 = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im3, cax=cax2, label=f"σ {units_map[mag_to_plot]}")
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_title(f"Predicted Std Dev\nMean σ: {np.mean(pred_std):.2f} | Median σ: {np.median(pred_std):.2f}",
                   fontsize=14, fontweight='bold')
    
    # 5. Distribution Comparison
    ax[4].hist(pred_mean.flatten(), bins=100, color='red', edgecolor='darkred',
              alpha=0.6, label="Predicted Mean", density=True)
    ax[4].hist(gt.flatten(), bins=100, color='blue', edgecolor='darkblue',
              alpha=0.5, label="Ground Truth", density=True)
    ax[4].set_xlim(vmin, vmax)
    ax[4].set_xlabel(units_map[mag_to_plot], fontsize=12)
    ax[4].set_ylabel("Density", fontsize=12)
    ax[4].legend(fontsize=11)
    ax[4].set_title("Distribution Comparison", fontsize=14, fontweight='bold')
    ax[4].grid(alpha=0.3)
    
    # 6. Scatter Plot
    n_samples = min(10000, pred_mean.size)
    indices = np.random.choice(pred_mean.size, n_samples, replace=False)
    gt_flat = gt.flatten()[indices]
    pred_flat = pred_mean.flatten()[indices]
    
    ax[5].scatter(gt_flat, pred_flat, alpha=0.3, s=2, c='black', edgecolors='none')
    lims = [np.min([ax[5].get_xlim(), ax[5].get_ylim()]),
            np.max([ax[5].get_xlim(), ax[5].get_ylim()])]
    ax[5].plot(lims, lims, 'r--', alpha=0.75, linewidth=2, zorder=0, label='1:1 line')
    ax[5].set_xlim(lims)
    ax[5].set_ylim(lims)
    ax[5].set_xlabel(f"Ground Truth ({units_map[mag_to_plot]})", fontsize=12)
    ax[5].set_ylabel(f"Predicted ({units_map[mag_to_plot]})", fontsize=12)
    ax[5].set_title(f"Scatter Plot\nPearson R = {corr:.4f} (p < {p_value:.1e})",
                   fontsize=14, fontweight='bold')
    ax[5].legend(fontsize=11, loc='lower right')
    ax[5].grid(alpha=0.3)
    ax[5].set_aspect('equal', adjustable='box')
    ax[5].text(0.05, 0.95, f"R = {corr:.4f}\nN = {n_samples:,}",
              transform=ax[5].transAxes, fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Prepare metrics dictionary
    metrics = {
        "model_label": model_label,
        "parameter": mag_to_plot,
        "parameter_name": title_map[mag_to_plot],
        "units": units_map[mag_to_plot],
        "optical_depth": float(logtau[logtau_idx]),
        "pearson_r": float(corr),
        "p_value": float(p_value),
        "rmse": float(rmse),
        "bias": float(bias),
        "mean_uncertainty": float(np.mean(pred_std)),
        "median_uncertainty": float(np.median(pred_std)),
        "n_pixels": int(pred_mean.size)
    }
    
    # --- NEW: build output path ---
    if save_dir is not None and model_label is not None:
        model_name = model_label.lower().replace(" ", "_")
        out_dir = os.path.join(
            str(save_dir),
            "prediction_comparison",
            model_name,
            str(od_to_plot)
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        _save_metrics_json(metrics, out_dir, filename)
    else:
        plt.show()
    
    print(f"\n{'='*60}")
    print(f"Summary for {model_label}: {title_map[mag_to_plot]} at log(τ)={logtau[logtau_idx]:.1f}")
    print(f"{'='*60}")
    print(f"Pearson R:   {corr:.4f} (p-value: {p_value:.2e})")
    print(f"RMSE:        {rmse:.3f} {units_map[mag_to_plot]}")
    print(f"Bias:        {bias:.3f} {units_map[mag_to_plot]}")
    print(f"Mean σ:      {np.mean(pred_std):.3f} {units_map[mag_to_plot]}")
    print(f"Median σ:    {np.median(pred_std):.3f} {units_map[mag_to_plot]}")
    print(f"{'='*60}\n")


def compare_models_at_optical_depth(
    all_predictions: Dict,
    ground_truth: dict,
    mag_to_plot: str = "T",
    od_to_plot: float = 0.0,
    logtau: np.ndarray = None,
    figsize: tuple = (18, 12),
    save_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None
):
    """
    Compare multiple models side-by-side at a specific optical depth.
    
    Parameters
    ----------
    all_predictions : Dict
        Dictionary of model predictions
    ground_truth : dict
        Ground truth SPINOR atmosphere
    mag_to_plot : str
        Parameter to plot: 'T', 'Vz', or 'Bz'
    od_to_plot : float
        Optical depth value
    logtau : np.ndarray, optional
        Optical depth grid
    figsize : tuple
        Figure size
    save_dir : str or Path, optional
        Base directory to save figures. A subfolder 'model_comparison' will be created.
    filename : str, optional
        Filename for the saved figure. If None, displays the plot instead.
    """
    if logtau is None:
        logtau = np.arange(-2, 0.1, 0.1)
    
    logtau_idx = np.argmin(np.abs(logtau - od_to_plot))
    
    title_map = {"T": "Temperature", "Vz": "Line-of-sight Velocity", "Bz": "Line-of-sight Magnetic Field"}
    units_map = {"T": "K", "Vz": "km/s", "Bz": "G"}
    modest_key_mapping = {"T": "T", "Vz": "Vlos", "Bz": "Blos"}
    color_mapping = {"T": "hot", "Vz": "bwr_r", "Bz": "PiYG"}
    
    gt = ground_truth[modest_key_mapping[mag_to_plot]][od_to_plot]
    q0, q99 = np.percentile(gt, [1, 99])
    
    if mag_to_plot in ["Vz", "Bz"]:
        vmax = max(np.abs(q0), np.abs(q99))
        vmin = -vmax
    else:
        vmin, vmax = q0, q99
    
    n_models = len(all_predictions)
    fig, axes = plt.subplots(2, n_models + 1, figsize=figsize)
    
    # Prepare metrics dictionary
    metrics = {
        "parameter": mag_to_plot,
        "parameter_name": title_map[mag_to_plot],
        "units": units_map[mag_to_plot],
        "optical_depth": float(logtau[logtau_idx]),
        "models": {}
    }
    
    # Ground Truth (first column)
    im_gt = axes[0, 0].imshow(gt, cmap=color_mapping[mag_to_plot], vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_gt, cax=cax, label=units_map[mag_to_plot])
    
    axes[1, 0].hist(gt.flatten(), bins=50, color='gray', alpha=0.7, label='GT', density=True)
    axes[1, 0].set_xlabel(units_map[mag_to_plot])
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_xlim(vmin, vmax)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Models
    for i, (model_name, pred_data) in enumerate(all_predictions.items(), start=1):
        mean_atm = pred_data['mean']
        pred_mean = mean_atm[mag_to_plot][:, :, logtau_idx]
        difference = pred_mean - gt
        rmse = np.sqrt(np.mean(difference**2))
        corr, p_value = pearsonr(pred_mean.flatten(), gt.flatten())
        bias = np.mean(difference)
        
        # Store metrics for this model
        metrics["models"][model_name] = {
            "label": pred_data['label'],
            "rmse": float(rmse),
            "pearson_r": float(corr),
            "p_value": float(p_value),
            "bias": float(bias)
        }
        
        im = axes[0, i].imshow(pred_mean, cmap=color_mapping[mag_to_plot], vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"{pred_data['label']}\nRMSE={rmse:.2f}, R={corr:.3f}",
                            fontsize=12, fontweight='bold')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        divider = make_axes_locatable(axes[0, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, label=units_map[mag_to_plot])
        
        axes[1, i].hist(pred_mean.flatten(), bins=50, color=pred_data['color'],
                       alpha=0.7, label=pred_data['label'], density=True)
        axes[1, i].hist(gt.flatten(), bins=50, color='gray', alpha=0.3,
                       label='GT', density=True)
        axes[1, i].set_xlabel(units_map[mag_to_plot])
        axes[1, i].set_ylabel("Density")
        axes[1, i].set_xlim(vmin, vmax)
        axes[1, i].legend()
        axes[1, i].grid(alpha=0.3)
    
    fig.suptitle(f"Model Comparison: {title_map[mag_to_plot]} at log(τ)={logtau[logtau_idx]:.1f}",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_dir is not None and filename is not None:
        save_dir = Path(save_dir) / "model_comparison"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        _save_metrics_json(metrics, save_dir, filename)
    else:
        plt.show()


def plot_mean_vs_optical_depth(
    mean_atm, 
    std_atm, 
    logtau=None, 
    figsize=(18, 6),
    log_scale=None,
    ylims=None,
    ground_truth=None,
    save_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None
):
    """
    Plot mean atmospheric parameters with uncertainties across optical depths.
    
    Parameters
    ----------
    mean_atm : dict
        Mean predictions for each parameter
    std_atm : dict
        Standard deviations for each parameter
    logtau : np.ndarray, optional
        Optical depth grid
    figsize : tuple
        Figure size
    log_scale : dict, optional
        Log scale settings for each parameter
    ylims : dict, optional
        Y-axis limits for each parameter
    ground_truth : dict, optional
        Ground truth atmosphere data
    save_dir : str or Path, optional
        Base directory to save figures. A subfolder 'mean_vs_optical_depth' will be created.
    filename : str, optional
        Filename for the saved figure. If None, displays the plot instead.
    """
    if logtau is None:
        logtau = np.arange(-2, 0.1, 0.1)
    
    if log_scale is None:
        log_scale = {'T': False, 'Vz': False, 'Bz': False}
    
    if ylims is None:
        ylims = {'T': None, 'Vz': None, 'Bz': None}
    
    params = ['T', 'Vz', 'Bz']
    titles = {
        'T': 'Temperature',
        'Vz': 'Line-of-sight Velocity',
        'Bz': 'Line-of-sight Magnetic Field'
    }
    units = {'T': 'K', 'Vz': 'km/s', 'Bz': 'G'}
    colors = {'T': 'red', 'Vz': 'blue', 'Bz': 'green'}
    gt_key_mapping = {'T': 'T', 'Vz': 'Vlos', 'Bz': 'Blos'}
    
    gt_optical_depths = [-2.0, -0.8, 0.0]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Prepare metrics dictionary
    metrics = {
        "logtau_grid": logtau.tolist(),
        "parameters": {}
    }
    
    for idx, param in enumerate(params):
        mean_spatial = np.mean(mean_atm[param], axis=(0, 1))
        std_spatial = np.mean(std_atm[param], axis=(0, 1))
        
        # Store parameter metrics
        metrics["parameters"][param] = {
            "name": titles[param],
            "units": units[param],
            "model_mean_profile": mean_spatial.tolist(),
            "model_std_profile": std_spatial.tolist(),
            "ground_truth_comparisons": {}
        }
        
        axes[idx].plot(logtau, mean_spatial, color=colors[param], 
                      linewidth=2, label='Model Mean', zorder=3)
        axes[idx].fill_between(
            logtau,
            mean_spatial - std_spatial,
            mean_spatial + std_spatial,
            color=colors[param],
            alpha=0.3,
            label='±1σ uncertainty',
            zorder=2
        )
        
        if ground_truth is not None:
            gt_means = []
            gt_od_values = []
            
            for od_val in gt_optical_depths:
                if od_val in ground_truth[gt_key_mapping[param]]:
                    gt_data = ground_truth[gt_key_mapping[param]][od_val]
                    gt_mean = np.mean(gt_data)
                    gt_std = np.std(gt_data)
                    gt_means.append(gt_mean)
                    gt_od_values.append(od_val)
                    
                    od_idx_closest = np.argmin(np.abs(logtau - od_val))
                    pred_mean_at_od = mean_spatial[od_idx_closest]
                    pred_std_at_od = std_spatial[od_idx_closest]
                    
                    within_uncertainty = (gt_mean >= pred_mean_at_od - pred_std_at_od and 
                                        gt_mean <= pred_mean_at_od + pred_std_at_od)
                    
                    diff = pred_mean_at_od - gt_mean
                    relative_diff = (diff / gt_mean) * 100 if gt_mean != 0 else 0
                    
                    # Store GT comparison metrics
                    metrics["parameters"][param]["ground_truth_comparisons"][str(od_val)] = {
                        "gt_mean": float(gt_mean),
                        "gt_std": float(gt_std),
                        "pred_mean": float(pred_mean_at_od),
                        "pred_std": float(pred_std_at_od),
                        "difference": float(diff),
                        "relative_difference_percent": float(relative_diff),
                        "within_1sigma": bool(within_uncertainty)
                    }
            
            if gt_means:
                axes[idx].scatter(gt_od_values, gt_means, 
                                color='black', s=100, marker='o',
                                edgecolors='white', linewidths=2,
                                label='Ground Truth (SPINOR)', zorder=4)
                
                for od_val, gt_mean in zip(gt_od_values, gt_means):
                    od_idx_closest = np.argmin(np.abs(logtau - od_val))
                    pred_mean_at_od = mean_spatial[od_idx_closest]
                    pred_std_at_od = std_spatial[od_idx_closest]
                    
                    within_uncertainty = (gt_mean >= pred_mean_at_od - pred_std_at_od and 
                                        gt_mean <= pred_mean_at_od + pred_std_at_od)
                    
                    marker = '✓' if within_uncertainty else '✗'
                    color_marker = 'green' if within_uncertainty else 'red'
                    axes[idx].annotate(marker, 
                                     xy=(od_val, gt_mean),
                                     xytext=(10, 10), textcoords='offset points',
                                     fontsize=14, fontweight='bold',
                                     color=color_marker,
                                     bbox=dict(boxstyle='round,pad=0.3', 
                                             facecolor='white', 
                                             edgecolor=color_marker, 
                                             alpha=0.8))
        
        if log_scale.get(param, False):
            axes[idx].set_yscale('log')
        
        if ylims.get(param) is not None:
            axes[idx].set_ylim(ylims[param])
        
        axes[idx].set_xlabel('log(τ)', fontsize=12)
        ylabel = f'{titles[param]} ({units[param]})'
        if log_scale.get(param, False):
            ylabel += ' [log scale]'
        axes[idx].set_ylabel(ylabel, fontsize=12)
        axes[idx].set_title(titles[param], fontsize=14, fontweight='bold')
        axes[idx].grid(alpha=0.3)
        axes[idx].legend(loc='best', fontsize=10)
        axes[idx].invert_xaxis()
    
    plt.tight_layout()
    
    if save_dir is not None and filename is not None:
        save_dir = Path(save_dir) / "mean_vs_optical_depth"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        _save_metrics_json(metrics, save_dir, filename)
    else:
        plt.show()
    
    if ground_truth is not None:
        print("\n" + "="*70)
        print("Ground Truth Comparison at Available Optical Depths")
        print("="*70)
        
        for param in params:
            print(f"\n{titles[param]} ({units[param]}):")
            print("-" * 70)
            
            for od_val in gt_optical_depths:
                if od_val in ground_truth[gt_key_mapping[param]]:
                    gt_data = ground_truth[gt_key_mapping[param]][od_val]
                    gt_mean = np.mean(gt_data)
                    gt_std = np.std(gt_data)
                    
                    od_idx_closest = np.argmin(np.abs(logtau - od_val))
                    pred_data = mean_atm[param][:, :, od_idx_closest]
                    pred_mean_spatial = np.mean(pred_data)
                    pred_std_spatial = np.mean(std_atm[param][:, :, od_idx_closest])
                    
                    diff = pred_mean_spatial - gt_mean
                    relative_diff = (diff / gt_mean) * 100 if gt_mean != 0 else 0
                    
                    within_uncertainty = abs(diff) <= pred_std_spatial
                    status = "✓ WITHIN" if within_uncertainty else "✗ OUTSIDE"
                    
                    print(f"  log(τ) = {od_val:5.1f}:")
                    print(f"    Ground Truth:  {gt_mean:10.2f} ± {gt_std:8.2f} {units[param]}")
                    print(f"    Model Pred:    {pred_mean_spatial:10.2f} ± {pred_std_spatial:8.2f} {units[param]}")
                    print(f"    Difference:    {diff:10.2f} ({relative_diff:+.1f}%)")
                    print(f"    Status:        {status} ±1σ uncertainty")
        
        print("\n" + "="*70)


def analyze_error_by_magnitude(
    all_predictions: Dict,
    ground_truth: dict,
    mag_to_analyze: str = "Bz",
    logtau: np.ndarray = None,
    od_val: float = 0.0,
    figsize: tuple = (18, 6),
    n_bins: int = 10,
    plot_counts: bool = True,
    percentile_lims: tuple = None,
    use_absolute: bool = False,
    rrmse_ylim: tuple = None,
    save_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None
):
    """
    Analyze prediction error as a function of magnitude strength using bar plots.
    
    Parameters
    ----------
    all_predictions : Dict
        Dictionary of model predictions
    ground_truth : dict
        Ground truth SPINOR atmosphere
    mag_to_analyze : str
        Magnitude to analyze: 'T', 'Vz', or 'Bz'
    logtau : np.ndarray, optional
        Optical depth grid
    od_val : float
        Optical depth value to analyze
    figsize : tuple
        Figure size
    n_bins : int
        Number of bins for magnitude ranges
    plot_counts : bool
        Whether to plot sample counts on bars
    percentile_lims : tuple, optional
        Percentile limits for x-axis
    use_absolute : bool
        If True, bin by absolute values
    rrmse_ylim : tuple, optional
        Y-axis limits for RRMSE percentage
    save_dir : str or Path, optional
        Base directory to save figures. A subfolder 'error_by_magnitude' will be created.
    filename : str, optional
        Filename for the saved figure. If None, displays the plot instead.
    """
    from scipy.stats import binned_statistic
    
    title_map = {"T": "Temperature", "Vz": "Line-of-sight Velocity", "Bz": "Line-of-sight Magnetic Field"}
    units_map = {"T": "K", "Vz": "km/s", "Bz": "G"}
    gt_key_mapping = {"T": "T", "Vz": "Vlos", "Bz": "Blos"}
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    if logtau is None:
        logtau = np.arange(-2, 0.1, 0.1)
    
    od_idx = np.argmin(np.abs(logtau - od_val))
    gt_data = ground_truth[gt_key_mapping[mag_to_analyze]][od_val]
    
    # Prepare metrics dictionary
    binning_type = "absolute" if use_absolute else "signed"
    metrics = {
        "parameter": mag_to_analyze,
        "parameter_name": title_map[mag_to_analyze],
        "units": units_map[mag_to_analyze],
        "optical_depth": float(od_val),
        "n_bins": n_bins,
        "binning_type": binning_type,
        "models": {}
    }
    
    for i, (model_name, pred_data) in enumerate(all_predictions.items()):
        pred_mag = pred_data['mean'][mag_to_analyze][:, :, od_idx]
        error = np.abs(pred_mag - gt_data)
        
        # Flatten and filter NaNs
        gt_flat = gt_data.flatten()
        pred_flat = pred_mag.flatten()
        error_flat = error.flatten()
        valid = ~np.isnan(gt_flat) & ~np.isnan(pred_flat) & ~np.isnan(error_flat)
        
        # Use signed or absolute values for binning
        if use_absolute:
            gt_binning = np.abs(gt_flat[valid])
            xlabel_prefix = "|Ground Truth"
            xlabel_suffix = "|"
        else:
            gt_binning = gt_flat[valid]
            xlabel_prefix = "Ground Truth"
            xlabel_suffix = ""
        
        pred_valid = pred_flat[valid]
        error_valid = error_flat[valid]
        gt_valid = gt_flat[valid]
        
        # Compute binned statistics
        bin_means, bin_edges, bin_number = binned_statistic(
            gt_binning, error_valid, statistic='mean', bins=n_bins
        )
        bin_stds, _, _ = binned_statistic(
            gt_binning, error_valid, statistic='std', bins=n_bins
        )
        bin_counts, _, _ = binned_statistic(
            gt_binning, error_valid, statistic='count', bins=n_bins
        )
        
        # Compute RRMSE per bin
        bin_rrmse = np.zeros(n_bins)
        bin_metrics = []
        for j in range(n_bins):
            in_bin = (gt_binning >= bin_edges[j]) & (gt_binning < bin_edges[j+1])
            if np.sum(in_bin) > 0:
                gt_bin = gt_valid[in_bin]
                pred_bin = pred_valid[in_bin]
                error_bin = error_valid[in_bin]
                
                rmse_bin = np.sqrt(np.mean(error_bin**2))
                gt_mean_bin = np.mean(np.abs(gt_bin))
                
                if gt_mean_bin > 1e-10:
                    bin_rrmse[j] = rmse_bin / gt_mean_bin
                else:
                    bin_rrmse[j] = np.nan
                
                bin_metrics.append({
                    "bin_low": float(bin_edges[j]),
                    "bin_high": float(bin_edges[j+1]),
                    "n_samples": int(np.sum(in_bin)),
                    "mae": float(bin_means[j]) if not np.isnan(bin_means[j]) else None,
                    "mae_std": float(bin_stds[j]) if not np.isnan(bin_stds[j]) else None,
                    "rmse": float(rmse_bin),
                    "rrmse": float(bin_rrmse[j]) if not np.isnan(bin_rrmse[j]) else None
                })
            else:
                bin_rrmse[j] = np.nan
                bin_metrics.append({
                    "bin_low": float(bin_edges[j]),
                    "bin_high": float(bin_edges[j+1]),
                    "n_samples": 0,
                    "mae": None,
                    "mae_std": None,
                    "rmse": None,
                    "rrmse": None
                })
        
        # Compute bin centers and widths
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        
        # Overall statistics
        overall_mae = np.mean(error_valid)
        overall_rmse = np.sqrt(np.mean(error_valid**2))
        overall_rrmse = overall_rmse / np.mean(np.abs(gt_valid)) if np.mean(np.abs(gt_valid)) > 1e-10 else np.nan
        
        # Store model metrics
        metrics["models"][model_name] = {
            "label": pred_data['label'],
            "overall_mae": float(overall_mae),
            "overall_rmse": float(overall_rmse),
            "overall_rrmse": float(overall_rrmse) if not np.isnan(overall_rrmse) else None,
            "bins": bin_metrics
        }
        
        # Create bar plot with RRMSE
        bars = axes[i].bar(bin_centers, bin_rrmse * 100, width=bin_widths * 0.9,
                          color=pred_data['color'], alpha=0.7, 
                          edgecolor='black', linewidth=1.5)
        
        # Add count labels on top of bars
        if plot_counts:
            for j, (center, rrmse_val, count) in enumerate(zip(bin_centers, bin_rrmse, bin_counts)):
                if not np.isnan(rrmse_val) and count > 0:
                    label_y = rrmse_val * 100
                    if rrmse_ylim is not None and label_y > rrmse_ylim[1]:
                        label_y = rrmse_ylim[1] - (rrmse_ylim[1] - rrmse_ylim[0]) * 0.05
                        axes[i].text(center, label_y, f'n={int(count)}',
                                ha='center', va='top', fontsize=8,
                                bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='yellow', alpha=0.7),
                                color='red')
                    else:
                        axes[i].text(center, label_y, f'n={int(count)}',
                                ha='center', va='bottom', fontsize=8,
                                bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='white', alpha=0.7))
            
        # Formatting
        axes[i].set_xlabel(f"{xlabel_prefix} {title_map[mag_to_analyze]}{xlabel_suffix} [{units_map[mag_to_analyze]}]", 
                          fontsize=11, fontweight='bold')
        axes[i].set_ylabel("RRMSE [%]", fontsize=11, fontweight='bold')
        axes[i].set_title(f"{pred_data['label']}", fontsize=12, fontweight='bold')
        axes[i].grid(alpha=0.3, axis='y')
        
        if rrmse_ylim is not None:
            axes[i].set_ylim(rrmse_ylim)
        
        if percentile_lims is not None:
            axes[i].set_xlim([np.percentile(gt_binning, percentile_lims[0]), 
                             np.percentile(gt_binning, percentile_lims[1])])
        else:
            axes[i].set_xlim([bin_edges[0], bin_edges[-1]])
        
        if not use_absolute and mag_to_analyze in ['Vz', 'Bz']:
            axes[i].axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero')
        
        if not np.isnan(overall_rrmse):
            stats_text = f"MAE: {overall_mae:.2f} {units_map[mag_to_analyze]}\nRMSE: {overall_rmse:.2f} {units_map[mag_to_analyze]}\nRRMSE: {overall_rrmse*100:.2f}%"
        else:
            stats_text = f"MAE: {overall_mae:.2f} {units_map[mag_to_analyze]}\nRMSE: {overall_rmse:.2f} {units_map[mag_to_analyze]}\nRRMSE: N/A"
        
        axes[i].text(0.98, 0.98, stats_text,
                    transform=axes[i].transAxes,
                    fontsize=10, verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    binning_type_display = "Absolute" if use_absolute else "Signed"
    plt.suptitle(f"Prediction RRMSE vs {binning_type_display} {title_map[mag_to_analyze]} at log(τ)={od_val:.1f}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir is not None and filename is not None:
        save_dir_path = Path(save_dir) / "error_by_magnitude"
        save_dir_path.mkdir(parents=True, exist_ok=True)
        save_path = save_dir_path / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        _save_metrics_json(metrics, save_dir_path, filename)
    else:
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print(f"Error Analysis by {binning_type_display} {title_map[mag_to_analyze]} at log(τ)={od_val:.1f}")
    print("="*70)
    
    for model_name, pred_data in all_predictions.items():
        model_metrics = metrics["models"][model_name]
        print(f"\n{pred_data['label']}:")
        print("-" * 70)
        
        for j, bin_data in enumerate(model_metrics["bins"]):
            if bin_data["n_samples"] > 0 and bin_data["mae"] is not None:
                rrmse_str = f"{bin_data['rrmse']*100:.2f}%" if bin_data["rrmse"] is not None else "N/A"
                print(f"  Bin {j+1}: {bin_data['bin_low']:8.2f} - {bin_data['bin_high']:8.2f} {units_map[mag_to_analyze]}  |  "
                      f"N: {bin_data['n_samples']:,}  |  "
                      f"MAE: {bin_data['mae']:8.2f}  |  "
                      f"RRMSE: {rrmse_str}")
        
        rrmse_str = f"{model_metrics['overall_rrmse']*100:.2f}%" if model_metrics['overall_rrmse'] is not None else "N/A"
        print(f"\n  Overall Statistics:")
        print(f"    Mean Absolute Error (MAE):  {model_metrics['overall_mae']:.2f} {units_map[mag_to_analyze]}")
        print(f"    Root Mean Square Error:     {model_metrics['overall_rmse']:.2f} {units_map[mag_to_analyze]}")
        print(f"    Relative RMSE (RRMSE):      {rrmse_str}")
    
    print("\n" + "="*70)


def plot_uncertainty_vs_error(
    all_predictions: Dict,
    ground_truth: dict,
    mag_to_plot: str = "Bz",
    od_val: float = 0.0,
    logtau: np.ndarray = None,
    figsize: tuple = (18, 6),
    save_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None
):
    """
    Plot predicted uncertainty vs actual error to validate calibration.
    
    Parameters
    ----------
    all_predictions : Dict
        Dictionary of model predictions
    ground_truth : dict
        Ground truth SPINOR atmosphere
    mag_to_plot : str
        Parameter to plot: 'T', 'Vz', or 'Bz'
    od_val : float
        Optical depth value
    logtau : np.ndarray, optional
        Optical depth grid
    figsize : tuple
        Figure size
    save_dir : str or Path, optional
        Base directory to save figures. A subfolder 'uncertainty_vs_error' will be created.
    filename : str, optional
        Filename for the saved figure. If None, displays the plot instead.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    modest_key_mapping = {"T": "T", "Vz": "Vlos", "Bz": "Blos"}
    title_map = {"T": "Temperature", "Vz": "Line-of-sight Velocity", "Bz": "Line-of-sight Magnetic Field"}
    units_map = {"T": "K", "Vz": "km/s", "Bz": "G"}
    
    if logtau is None:
        logtau = np.arange(-2, 0.1, 0.1)
    
    od_idx = np.argmin(np.abs(logtau - od_val))
    gt = ground_truth[modest_key_mapping[mag_to_plot]][od_val]
    
    # Prepare metrics dictionary
    metrics = {
        "parameter": mag_to_plot,
        "parameter_name": title_map[mag_to_plot],
        "units": units_map[mag_to_plot],
        "optical_depth": float(od_val),
        "models": {}
    }
    
    for i, (model_name, pred_data) in enumerate(all_predictions.items()):
        pred_mean = pred_data['mean'][mag_to_plot][:, :, od_idx]
        pred_std = pred_data['std'][mag_to_plot][:, :, od_idx]
        
        error = np.abs(pred_mean - gt)
        
        # Flatten
        std_flat = pred_std.flatten()
        error_flat = error.flatten()
        valid = ~np.isnan(std_flat) & ~np.isnan(error_flat)
        
        # Compute calibration metric
        corr_calib, p_value = pearsonr(std_flat[valid], error_flat[valid])
        
        # Store metrics
        metrics["models"][model_name] = {
            "label": pred_data['label'],
            "calibration_correlation": float(corr_calib),
            "calibration_p_value": float(p_value),
            "mean_predicted_std": float(np.mean(std_flat[valid])),
            "mean_actual_error": float(np.mean(error_flat[valid])),
            "n_valid_samples": int(np.sum(valid))
        }
        
        # Scatter plot
        axes[i].scatter(std_flat[valid], error_flat[valid],
                       alpha=0.3, s=1, c=pred_data['color'])
        axes[i].set_xlabel("Predicted Std Dev", fontsize=11)
        axes[i].set_ylabel("Absolute Error", fontsize=11)
        axes[i].set_title(f"{pred_data['label']}", fontsize=12, fontweight='bold')
        axes[i].grid(alpha=0.3)
        
        # Add 1:1 line (perfect calibration)
        max_val = max(axes[i].get_xlim()[1], axes[i].get_ylim()[1])
        axes[i].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect calibration')
        
        axes[i].text(0.05, 0.95, f"R = {corr_calib:.3f}",
                    transform=axes[i].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[i].legend()
    
    plt.suptitle(f"Uncertainty Calibration for {mag_to_plot}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir is not None and filename is not None:
        save_dir = Path(save_dir) / "uncertainty_vs_error"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        _save_metrics_json(metrics, save_dir, filename)
    else:
        plt.show()


def plot_jointplot_comparison(
    all_predictions: Dict,
    ground_truth: dict,
    mag_to_plot: str = "Bz",
    od_val: float = 0.0,
    logtau: np.ndarray = None,
    n_samples: int = 10000,
    kind: str = 'scatter',
    save_dir: Optional[Union[str, Path]] = None,
    filename_prefix: Optional[str] = None
):
    """
    Create jointplots comparing predictions vs ground truth for all models.
    
    Parameters
    ----------
    all_predictions : Dict
        Dictionary of model predictions
    ground_truth : dict
        Ground truth SPINOR atmosphere
    mag_to_plot : str
        Parameter to plot: 'T', 'Vz', or 'Bz'
    od_val : float
        Optical depth value to analyze
    logtau : np.ndarray, optional
        Optical depth grid
    n_samples : int
        Number of samples to plot (for performance)
    kind : str
        Type of plot: 'scatter', 'hex', 'kde', or 'reg'
    save_dir : str or Path, optional
        Base directory to save figures. A subfolder 'jointplot_comparison' will be created.
    filename_prefix : str, optional
        Prefix for filenames. Each model gets a file named '{prefix}_{model_name}.png'.
        If None, displays plots instead.
    """
    if logtau is None:
        logtau = np.arange(-2, 0.1, 0.1)
    
    # Configuration
    title_map = {"T": "Temperature", "Vz": "Line-of-sight Velocity", "Bz": "Line-of-sight Magnetic Field"}
    units_map = {"T": "K", "Vz": "km/s", "Bz": "G"}
    gt_key_mapping = {"T": "T", "Vz": "Vlos", "Bz": "Blos"}
    
    od_idx = np.argmin(np.abs(logtau - od_val))
    gt_data = ground_truth[gt_key_mapping[mag_to_plot]][od_val]
    
    # Create jointplots for each model (each gets its own figure)
    for idx, (model_name, pred_data) in enumerate(all_predictions.items()):
        pred_mean = pred_data['mean'][mag_to_plot][:, :, od_idx]
        
        # Flatten and sample
        gt_flat = gt_data.flatten()
        pred_flat = pred_mean.flatten()
        
        # Remove NaNs
        valid = ~np.isnan(gt_flat) & ~np.isnan(pred_flat)
        gt_valid = gt_flat[valid]
        pred_valid = pred_flat[valid]
        
        # Subsample for performance
        if len(gt_valid) > n_samples:
            indices = np.random.choice(len(gt_valid), n_samples, replace=False)
            gt_valid = gt_valid[indices]
            pred_valid = pred_valid[indices]
        
        # Create DataFrame for seaborn
        df_plot = pd.DataFrame({
            'Ground Truth': gt_valid,
            'Prediction': pred_valid
        })
        
        # Compute metrics
        corr, p_value = pearsonr(gt_valid, pred_valid)
        rmse = np.sqrt(np.mean((pred_valid - gt_valid)**2))
        bias = np.mean(pred_valid - gt_valid)
        
        # Prepare metrics dictionary for this model
        metrics = {
            "model_name": model_name,
            "model_label": pred_data['label'],
            "parameter": mag_to_plot,
            "parameter_name": title_map[mag_to_plot],
            "units": units_map[mag_to_plot],
            "optical_depth": float(od_val),
            "pearson_r": float(corr),
            "p_value": float(p_value),
            "rmse": float(rmse),
            "bias": float(bias),
            "n_samples": int(len(gt_valid))
        }
        
        # Create jointplot using JointGrid (creates its own figure)
        g = sns.JointGrid(data=df_plot, x='Ground Truth', y='Prediction', 
                         height=6, ratio=5, space=0.2)
        
        # Main plot
        if kind == 'scatter':
            g.plot_joint(sns.scatterplot, alpha=0.3, s=10, color=pred_data['color'])
        elif kind == 'hex':
            g.plot_joint(plt.hexbin, gridsize=30, cmap='Blues', mincnt=1)
        elif kind == 'kde':
            g.plot_joint(sns.kdeplot, cmap='Blues', fill=True, levels=10)
        elif kind == 'reg':
            g.plot_joint(sns.regplot, scatter_kws={'alpha': 0.3, 's': 10}, 
                        color=pred_data['color'])
        
        # Marginal histograms
        g.plot_marginals(sns.histplot, kde=True, color=pred_data['color'], alpha=0.6, bins=30)
        
        # Add 1:1 line
        lims = [min(g.ax_joint.get_xlim()[0], g.ax_joint.get_ylim()[0]),
                max(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])]
        g.ax_joint.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, zorder=0, label='1:1 line')
        
        # Limit axes to percentiles 1-99
        combined_data = np.concatenate([gt_valid, pred_valid])
        p1, p99 = np.percentile(combined_data, [1, 99])
        g.ax_joint.set_xlim(p1, p99)
        g.ax_joint.set_ylim(p1, p99)
        
        # Labels
        g.set_axis_labels(f'Ground Truth ({units_map[mag_to_plot]})', 
                         f'Prediction ({units_map[mag_to_plot]})',
                         fontsize=12)
        
        # Title with metrics
        g.fig.suptitle(f"{pred_data['label']}\n"
                      f"R = {corr:.4f} | RMSE = {rmse:.2f} | Bias = {bias:.2f} {units_map[mag_to_plot]}",
                      fontsize=12, fontweight='bold', y=1.02)
        
        # Add statistics box
        stats_text = (f"N = {len(gt_valid):,}\n"
                     f"R = {corr:.4f}\n"
                     f"p < {p_value:.1e}\n"
                     f"RMSE = {rmse:.2f}\n"
                     f"Bias = {bias:.2f}")
        
        g.ax_joint.text(0.05, 0.95, stats_text,
                       transform=g.ax_joint.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        g.ax_joint.legend(loc='lower right', fontsize=10)
        g.ax_joint.grid(alpha=0.3)
        
        # --- NEW: build output path ---
        if save_dir is not None and filename_prefix is not None:
            out_dir = os.path.join(
                str(save_dir),
                "jointplot_comparison",
                model_name,
                str(od_val)
            )
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir,
                f"{filename_prefix}_{model_name}.png"
            )
            g.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(g.fig)
            _save_metrics_json(metrics, out_dir, out_path)
        else:
            # ...existing fallback...
            pass
        
        # Print summary
        print(f"\n{pred_data['label']} - {title_map[mag_to_plot]} at log(τ)={od_val:.1f}")
        print(f"  Pearson R:   {corr:.4f} (p={p_value:.2e})")
        print(f"  RMSE:        {rmse:.2f} {units_map[mag_to_plot]}")
        print(f"  Bias:        {bias:.2f} {units_map[mag_to_plot]}")
        print(f"  Samples:     {len(gt_valid):,}")


def plot_combined_jointplot(
    all_predictions: Dict,
    ground_truth: dict,
    mag_to_plot: str = "Bz",
    od_val: float = 0.0,
    logtau: np.ndarray = None,
    n_samples: int = 5000,
    save_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None
):
    """
    Create a single jointplot with all models overlaid.
    
    Parameters
    ----------
    all_predictions : Dict
        Dictionary of model predictions
    ground_truth : dict
        Ground truth SPINOR atmosphere
    mag_to_plot : str
        Parameter to plot: 'T', 'Vz', or 'Bz'
    od_val : float
        Optical depth value to analyze
    logtau : np.ndarray, optional
        Optical depth grid
    n_samples : int
        Number of samples per model
    save_dir : str or Path, optional
        Base directory to save figures. A subfolder 'combined_jointplot' will be created.
    filename : str, optional
        Filename for the saved figure. If None, displays the plot instead.
    """
    if logtau is None:
        logtau = np.arange(-2, 0.1, 0.1)
    
    # Configuration
    title_map = {"T": "Temperature", "Vz": "Line-of-sight Velocity", "Bz": "Line-of-sight Magnetic Field"}
    units_map = {"T": "K", "Vz": "km/s", "Bz": "G"}
    gt_key_mapping = {"T": "T", "Vz": "Vlos", "Bz": "Blos"}
    
    od_idx = np.argmin(np.abs(logtau - od_val))
    gt_data = ground_truth[gt_key_mapping[mag_to_plot]][od_val]
    
    # Prepare combined DataFrame and metrics
    all_data = []
    metrics = {
        "parameter": mag_to_plot,
        "parameter_name": title_map[mag_to_plot],
        "units": units_map[mag_to_plot],
        "optical_depth": float(od_val),
        "models": {}
    }
    
    for model_name, pred_data in all_predictions.items():
        pred_mean = pred_data['mean'][mag_to_plot][:, :, od_idx]
        
        # Flatten and sample
        gt_flat = gt_data.flatten()
        pred_flat = pred_mean.flatten()
        
        # Remove NaNs
        valid = ~np.isnan(gt_flat) & ~np.isnan(pred_flat)
        gt_valid = gt_flat[valid]
        pred_valid = pred_flat[valid]
        
        # Subsample
        if len(gt_valid) > n_samples:
            indices = np.random.choice(len(gt_valid), n_samples, replace=False)
            gt_valid = gt_valid[indices]
            pred_valid = pred_valid[indices]
        
        # Compute metrics for this model
        corr, p_value = pearsonr(gt_valid, pred_valid)
        rmse = np.sqrt(np.mean((pred_valid - gt_valid)**2))
        bias = np.mean(pred_valid - gt_valid)
        
        metrics["models"][model_name] = {
            "label": pred_data['label'],
            "pearson_r": float(corr),
            "p_value": float(p_value),
            "rmse": float(rmse),
            "bias": float(bias),
            "n_samples": int(len(gt_valid))
        }
        
        # Create DataFrame
        df_model = pd.DataFrame({
            'Ground Truth': gt_valid,
            'Prediction': pred_valid,
            'Model': pred_data['label']
        })
        
        all_data.append(df_model)
    
    # Combine all models
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Create jointplot
    g = sns.JointGrid(data=df_combined, x='Ground Truth', y='Prediction', 
                     hue='Model', height=8, ratio=5, space=0.2)
    
    # Main scatter plot with different colors per model
    g.plot_joint(sns.scatterplot, alpha=0.4, s=10, 
                palette={pred_data['label']: pred_data['color'] 
                        for pred_data in all_predictions.values()})
    
    # Marginal histograms separated by model
    g.plot_marginals(sns.histplot, kde=True, alpha=0.5,
                    palette={pred_data['label']: pred_data['color'] 
                            for pred_data in all_predictions.values()})
    
    # Add 1:1 line
    lims = [min(g.ax_joint.get_xlim()[0], g.ax_joint.get_ylim()[0]),
            max(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])]
    g.ax_joint.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, zorder=0, label='1:1 line')
    
    # Limit axes to percentiles 1-99
    all_gt = df_combined['Ground Truth'].values
    all_pred = df_combined['Prediction'].values
    combined_data = np.concatenate([all_gt, all_pred])
    p1, p99 = np.percentile(combined_data, [1, 99])
    g.ax_joint.set_xlim(p1, p99)
    g.ax_joint.set_ylim(p1, p99)
    
    # Labels
    g.set_axis_labels(f'Ground Truth ({units_map[mag_to_plot]})', 
                     f'Prediction ({units_map[mag_to_plot]})',
                     fontsize=13, fontweight='bold')
    
    # Title
    g.fig.suptitle(f"Model Comparison: {title_map[mag_to_plot]} at log(τ)={od_val:.1f}",
                  fontsize=14, fontweight='bold', y=1.02)
    
    g.ax_joint.grid(alpha=0.3)
    g.ax_joint.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # --- NEW: build output path ---
    if save_dir is not None:
        out_dir = os.path.join(
            str(save_dir),
            "combined_jointplot",
            str(od_val)
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        g.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(g.fig)
        _save_metrics_json(metrics, out_dir, filename)
    else:
        # ...existing fallback...
        pass
    
    # Print summary for each model
    print("\n" + "="*70)
    print(f"Combined Analysis: {title_map[mag_to_plot]} at log(τ)={od_val:.1f}")
    print("="*70)
    
    for model_name, pred_data in all_predictions.items():
        model_metrics = metrics["models"][model_name]
        print(f"\n{pred_data['label']}:")
        print(f"  Pearson R:   {model_metrics['pearson_r']:.4f} (p={model_metrics['p_value']:.2e})")
        print(f"  RMSE:        {model_metrics['rmse']:.2f} {units_map[mag_to_plot]}")
        print(f"  Bias:        {model_metrics['bias']:.2f} {units_map[mag_to_plot]}")
        print(f"  Samples:     {model_metrics['n_samples']:,}")