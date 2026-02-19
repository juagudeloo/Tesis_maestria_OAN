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

class ModestAnalysis:
    """Analysis tools for MODEST/SPINOR-style ground truth comparisons."""

    def __init__(self, default_logtau: Optional[np.ndarray] = None, default_save_dir: Optional[Union[str, Path]] = None):
        self.default_logtau = default_logtau
        self.default_save_dir = default_save_dir
        # MODEST uses different naming: T, Vlos, Blos instead of T, Vz, Bz
        self.modest_key_mapping = {"T": "T", "Vz": "Vlos", "Bz": "Blos"}

    def _resolve_logtau(self, logtau: Optional[np.ndarray]) -> np.ndarray:
        return logtau if logtau is not None else (self.default_logtau if self.default_logtau is not None else np.arange(-2, 0.1, 0.1))

    def _resolve_save_dir(self, save_dir: Optional[Union[str, Path]]) -> Optional[Union[str, Path]]:
        return save_dir if save_dir is not None else self.default_save_dir

    @staticmethod
    def _save_metrics_json(metrics: dict, save_dir: Path, filename: str):
        json_filename = Path(filename).stem + ".json"
        json_path = save_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Metrics saved to {json_path}")
    
    def _get_ground_truth_slice(self, ground_truth: dict, param: str, od_val: float, logtau: np.ndarray) -> np.ndarray:
        """Extract 2D slice from MODEST dictionary-based ground truth at specified optical depth.
        
        Parameters
        ----------
        ground_truth : dict
            MODEST ground truth with structure: ground_truth[param][od_val] -> array (H, W)
            where param is 'T', 'Vlos', or 'Blos'
        param : str
            Physical parameter ('T', 'Vz', or 'Bz') - will be mapped to MODEST naming
        od_val : float
            Target optical depth value
        logtau : np.ndarray
            Optical depth grid (not used in MODEST, kept for compatibility)
            
        Returns
        -------
        np.ndarray
            2D slice (H, W) at the requested optical depth
        """
        # Map model parameter names to MODEST keys
        modest_key = self.modest_key_mapping[param]
        return ground_truth[modest_key][od_val]

    @staticmethod
    def _format_od_folder(od_val: float) -> str:
        return f"{od_val:.2f}"

    def plot_prediction_comparison(
        self,
        mean_atm: dict,
        ground_truth: dict,
        mag_to_plot: str = "T",
        od_to_plot: float = 0.0,
        logtau: np.ndarray = None,
        model_label: str = "Model",
        figsize: tuple = (14, 12),
        save_dir: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None
    ):
        logtau = self._resolve_logtau(logtau)
        
        logtau_idx = np.argmin(np.abs(logtau - od_to_plot))
        
        title_map = {"T": "Temperature", "Vz": "Line-of-sight Velocity", "Bz": "Line-of-sight Magnetic Field"}
        units_map = {"T": "K", "Vz": "km/s", "Bz": "G"}
        color_mapping = {"T": "inferno", "Vz": "bwr_r", "Bz": "PiYG"}
        
        pred = mean_atm[mag_to_plot][:, :, logtau_idx]
        gt = self._get_ground_truth_slice(ground_truth, mag_to_plot, od_to_plot, logtau)
        
        difference = pred - gt
        
        q0, q99 = np.percentile(gt, [1, 99])
        if mag_to_plot in ["Vz", "Bz"]:
            vmax = max(np.abs(q0), np.abs(q99))
            vmin = -vmax
        else:
            vmin, vmax = q0, q99
        
        diff_max = np.percentile(np.abs(difference), 99)
        diff_vmin, diff_vmax = -diff_max, diff_max
        
        rmse = np.sqrt(np.mean(difference**2))
        rrmse = rmse / np.mean(np.abs(gt)) if np.mean(np.abs(gt)) > 1e-10 else np.nan
        bias = np.mean(difference)
        corr, p_value = pearsonr(pred.flatten(), gt.flatten())
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        
        fig.suptitle(f"{model_label}: {title_map[mag_to_plot]} at log(τ)={logtau[logtau_idx]:.1f}",
                     fontsize=18, fontweight='bold')
        
        # 1. Predicted
        im0 = ax[0].imshow(pred, cmap=color_mapping[mag_to_plot], vmax=vmax, vmin=vmin)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im0, cax=cax, label=units_map[mag_to_plot])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title("Predicted", fontsize=14, fontweight='bold')
        
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
        rrmse_str = f"{rrmse*100:.2f}%" if not np.isnan(rrmse) else "N/A"
        ax[2].set_title(f"Difference (Pred - GT)\nRRMSE: {rrmse_str} | Bias: {bias:.2f} {units_map[mag_to_plot]}",
                     fontsize=14, fontweight='bold')
        
        # 4. Distribution Comparison
        ax[3].hist(pred.flatten(), bins=100, color='red', edgecolor='darkred',
                     alpha=0.6, label="Prediction", density=True)
        ax[3].hist(gt.flatten(), bins=100, color='blue', edgecolor='darkblue',
                     alpha=0.5, label="Ground Truth", density=True)
        ax[3].set_xlim(vmin, vmax)
        ax[3].set_xlabel(units_map[mag_to_plot], fontsize=12)
        ax[3].set_ylabel("Density", fontsize=12)
        ax[3].legend(fontsize=11)
        ax[3].set_title("Distribution Comparison", fontsize=14, fontweight='bold')
        ax[3].grid(alpha=0.3)
        
        plt.tight_layout()
        
        metrics = {
            "model_label": model_label,
            "parameter": mag_to_plot,
            "parameter_name": title_map[mag_to_plot],
            "units": units_map[mag_to_plot],
            "optical_depth": float(logtau[logtau_idx]),
            "pearson_r": float(corr),
            "p_value": float(p_value),
            "rrmse": float(rrmse) if not np.isnan(rrmse) else None,
            "rmse": float(rmse),
            "bias": float(bias),
            "n_pixels": int(pred.size)
        }
        
        if save_dir is not None and model_label is not None:
            model_name = model_label.lower().replace(" ", "_")
            out_dir = os.path.join(str(save_dir), "prediction_comparison", model_name, self._format_od_folder(od_to_plot))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, filename)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self._save_metrics_json(metrics, Path(out_dir), filename)
        else:
            plt.show()
        
        print(f"\n{'='*60}")
        print(f"Summary for {model_label}: {title_map[mag_to_plot]} at log(τ)={logtau[logtau_idx]:.1f}")
        print(f"{'='*60}")
        print(f"Pearson R:   {corr:.4f} (p-value: {p_value:.2e})")
        rrmse_str = f"{rrmse*100:.2f}%" if not np.isnan(rrmse) else "N/A"
        print(f"RRMSE:       {rrmse_str}")
        print(f"RMSE:        {rmse:.3f} {units_map[mag_to_plot]}")
        print(f"Bias:        {bias:.3f} {units_map[mag_to_plot]}")
        print(f"{'='*60}\n")

    def compare_models_at_optical_depth(
        self,
        all_predictions: Dict,
        ground_truth: dict,
        mag_to_plot: str = "T",
        od_to_plot: float = 0.0,
        logtau: np.ndarray = None,
        figsize: tuple = (18, 12),
        save_dir: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None
    ):
        logtau = self._resolve_logtau(logtau)
        logtau_idx = np.argmin(np.abs(logtau - od_to_plot))
        
        title_map = {"T": "Temperature", "Vz": "Line-of-sight Velocity", "Bz": "Line-of-sight Magnetic Field"}
        units_map = {"T": "K", "Vz": "km/s", "Bz": "G"}
        color_mapping = {"T": "hot", "Vz": "bwr_r", "Bz": "PiYG"}
        
        gt = self._get_ground_truth_slice(ground_truth, mag_to_plot, od_to_plot, logtau)
        q0, q99 = np.percentile(gt, [1, 99])
        
        if mag_to_plot in ["Vz", "Bz"]:
            vmax = max(np.abs(q0), np.abs(q99))
            vmin = -vmax
        else:
            vmin, vmax = q0, q99
        
        n_models = len(all_predictions)
        fig, axes = plt.subplots(2, n_models + 1, figsize=figsize)
        
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
            prediction_atm = pred_data['prediction']
            pred = prediction_atm[mag_to_plot][:, :, logtau_idx]
            difference = pred - gt
            rmse = np.sqrt(np.mean(difference**2))
            rrmse = rmse / np.mean(np.abs(gt)) if np.mean(np.abs(gt)) > 1e-10 else np.nan
            corr, p_value = pearsonr(pred.flatten(), gt.flatten())
            bias = np.mean(difference)
            
            metrics["models"][model_name] = {
                "label": pred_data['label'],
                "rrmse": float(rrmse) if not np.isnan(rrmse) else None,
                "rmse": float(rmse),
                "pearson_r": float(corr),
                "p_value": float(p_value),
                "bias": float(bias)
            }
            
            rrmse_str = f"{rrmse*100:.2f}%" if not np.isnan(rrmse) else "N/A"
            im = axes[0, i].imshow(pred, cmap=color_mapping[mag_to_plot], vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f"{pred_data['label']}\nRRMSE={rrmse_str}, R={corr:.3f}",
                                fontsize=12, fontweight='bold')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            divider = make_axes_locatable(axes[0, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, label=units_map[mag_to_plot])
            
            axes[1, i].hist(pred.flatten(), bins=50, color=pred_data['color'],
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
            self._save_metrics_json(metrics, save_dir, filename)
        else:
            plt.show()

    def plot_mean_vs_optical_depth(
        self,
        mean_atm,
        logtau=None,
        figsize=(18, 6),
        log_scale=None,
        ylims=None,
        ground_truth=None,
        save_dir: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None
    ):
        logtau = self._resolve_logtau(logtau)
        
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
        
        gt_optical_depths = [-2.0, -0.8, 0.0]
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        metrics = {
            "logtau_grid": logtau.tolist(),
            "parameters": {}
        }
        
        for idx, param in enumerate(params):
            mean_spatial = np.mean(mean_atm[param], axis=(0, 1))
            
            metrics["parameters"][param] = {
                "name": titles[param],
                "units": units[param],
                "model_mean_profile": mean_spatial.tolist(),
                "ground_truth_comparisons": {}
            }
            
            axes[idx].plot(logtau, mean_spatial, color=colors[param], 
                          linewidth=2, label='Model Mean', zorder=3)
            
            if ground_truth is not None:
                gt_means = []
                gt_od_values = []
                
                # Use MODEST key mapping
                modest_key = self.modest_key_mapping[param]
                
                for od_val in gt_optical_depths:
                    if od_val in ground_truth[modest_key]:
                        gt_data = ground_truth[modest_key][od_val]
                        gt_mean = np.mean(gt_data)
                        gt_std = np.std(gt_data)
                        gt_means.append(gt_mean)
                        gt_od_values.append(od_val)
                        
                        od_idx_closest = np.argmin(np.abs(logtau - od_val))
                        pred_mean_at_od = mean_spatial[od_idx_closest]
                        
                        diff = pred_mean_at_od - gt_mean
                        relative_diff = (diff / gt_mean) * 100 if gt_mean != 0 else 0
                        
                        # Store GT comparison metrics
                        metrics["parameters"][param]["ground_truth_comparisons"][str(od_val)] = {
                            "gt_mean": float(gt_mean),
                            "gt_std": float(gt_std),
                            "pred_mean": float(pred_mean_at_od),
                            "difference": float(diff),
                            "relative_difference_percent": float(relative_diff)
                        }
                
                if gt_means:
                    axes[idx].scatter(gt_od_values, gt_means, 
                                    color='black', s=100, marker='o',
                                    edgecolors='white', linewidths=2,
                                    label='Ground Truth (SPINOR)', zorder=4)
                    
                    for od_val, gt_mean in zip(gt_od_values, gt_means):
                        od_idx_closest = np.argmin(np.abs(logtau - od_val))
                        pred_mean_at_od = mean_spatial[od_idx_closest]
                        
                        within_uncertainty = False  # No longer checking uncertainty
                        
                        marker = '●' if within_uncertainty else '○'
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
            self._save_metrics_json(metrics, save_dir, filename)
        else:
            plt.show()
        
        if ground_truth is not None:
            print("\n" + "="*70)
            print("Ground Truth Comparison at Available Optical Depths")
            print("="*70)
            
            for param in params:
                print(f"\n{titles[param]} ({units[param]}):")
                print("-" * 70)
                
                modest_key = self.modest_key_mapping[param]
                mean_spatial = np.mean(mean_atm[param], axis=(0, 1))
                
                for od_val in gt_optical_depths:
                    if od_val in ground_truth[modest_key]:
                        gt_data = ground_truth[modest_key][od_val]
                        gt_mean = np.mean(gt_data)
                        gt_std = np.std(gt_data)
                        
                        od_idx_closest = np.argmin(np.abs(logtau - od_val))
                        pred_mean_at_od = mean_spatial[od_idx_closest]
                        
                        diff = pred_mean_at_od - gt_mean
                        relative_diff = (diff / gt_mean) * 100 if gt_mean != 0 else 0
                        
                        print(f"  log(τ)={od_val:.1f}: GT={gt_mean:.2f}±{gt_std:.2f}, "
                              f"Pred={pred_mean_at_od:.2f}, Diff={diff:.2f} ({relative_diff:.1f}%)")
            
            print("\n" + "="*70)

    def analyze_error_by_magnitude(
        self,
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
        logtau = self._resolve_logtau(logtau)
        
        title_map = {"T": "Temperature", "Vz": "Line-of-sight Velocity", "Bz": "Line-of-sight Magnetic Field"}
        units_map = {"T": "K", "Vz": "km/s", "Bz": "G"}
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        od_idx = np.argmin(np.abs(logtau - od_val))
        gt_data = self._get_ground_truth_slice(ground_truth, mag_to_analyze, od_val, logtau)
        
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
            pred_mag = pred_data['prediction'][mag_to_analyze][:, :, od_idx]
            error = np.abs(pred_mag - gt_data)
            
            gt_flat = gt_data.flatten()
            pred_flat = pred_mag.flatten()
            error_flat = error.flatten()
            valid = ~np.isnan(gt_flat) & ~np.isnan(pred_flat) & ~np.isnan(error_flat)
            
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
            
            bin_means, bin_edges, bin_number = binned_statistic(
                gt_binning, error_valid, statistic='mean', bins=n_bins
            )
            bin_stds, _, _ = binned_statistic(
                gt_binning, error_valid, statistic='std', bins=n_bins
            )
            bin_counts, _, _ = binned_statistic(
                gt_binning, error_valid, statistic='count', bins=n_bins
            )
            
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
            
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_widths = bin_edges[1:] - bin_edges[:-1]
            
            overall_mae = np.mean(error_valid)
            overall_rmse = np.sqrt(np.mean(error_valid**2))
            overall_rrmse = overall_rmse / np.mean(np.abs(gt_valid)) if np.mean(np.abs(gt_valid)) > 1e-10 else np.nan
            
            metrics["models"][model_name] = {
                "label": pred_data['label'],
                "overall_mae": float(overall_mae),
                "overall_rmse": float(overall_rmse),
                "overall_rrmse": float(overall_rrmse) if not np.isnan(overall_rrmse) else None,
                "bins": bin_metrics
            }
            
            bars = axes[i].bar(bin_centers, bin_rrmse * 100, width=bin_widths * 0.9,
                          color=pred_data['color'], alpha=0.7, 
                          edgecolor='black', linewidth=1.5)
            
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
            self._save_metrics_json(metrics, save_dir_path, filename)
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

    def plot_jointplot_comparison(
        self,
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
        logtau = self._resolve_logtau(logtau)
        
        # Configuration
        title_map = {"T": "Temperature", "Vz": "Line-of-sight Velocity", "Bz": "Line-of-sight Magnetic Field"}
        units_map = {"T": "K", "Vz": "km/s", "Bz": "G"}
        
        od_idx = np.argmin(np.abs(logtau - od_val))
        gt_data = self._get_ground_truth_slice(ground_truth, mag_to_plot, od_val, logtau)
        
        for idx, (model_name, pred_data) in enumerate(all_predictions.items()):
            pred = pred_data['prediction'][mag_to_plot][:, :, od_idx]
            
            gt_flat = gt_data.flatten()
            pred_flat = pred.flatten()
            
            valid = ~np.isnan(gt_flat) & ~np.isnan(pred_flat)
            gt_valid = gt_flat[valid]
            pred_valid = pred_flat[valid]
            
            if len(gt_valid) > n_samples:
                indices = np.random.choice(len(gt_valid), n_samples, replace=False)
                gt_valid = gt_valid[indices]
                pred_valid = pred_valid[indices]
            
            df_plot = pd.DataFrame({
                'Ground Truth': gt_valid,
                'Prediction': pred_valid
            })
            
            corr, p_value = pearsonr(gt_valid, pred_valid)
            rmse = np.sqrt(np.mean((pred_valid - gt_valid)**2))
            gt_mean_val = np.mean(np.abs(gt_valid))
            rrmse = rmse / gt_mean_val if gt_mean_val > 1e-10 else np.nan
            bias = np.mean(pred_valid - gt_valid)
            
            metrics = {
                "model_name": model_name,
                "model_label": pred_data['label'],
                "parameter": mag_to_plot,
                "parameter_name": title_map[mag_to_plot],
                "units": units_map[mag_to_plot],
                "optical_depth": float(od_val),
                "pearson_r": float(corr),
                "p_value": float(p_value),
                "rrmse": float(rrmse) if not np.isnan(rrmse) else None,
                "rmse": float(rmse),
                "bias": float(bias),
                "n_samples": int(len(gt_valid))
            }
            
            g = sns.JointGrid(data=df_plot, x='Ground Truth', y='Prediction', 
                           height=6, ratio=5, space=0.2)
            
            if kind == 'scatter':
                g.plot_joint(sns.scatterplot, alpha=0.3, s=10, color=pred_data['color'])
            elif kind == 'hex':
                g.plot_joint(plt.hexbin, gridsize=30, cmap='Blues', mincnt=1)
            elif kind == 'kde':
                g.plot_joint(sns.kdeplot, cmap='Blues', fill=True, levels=10)
            elif kind == 'reg':
                g.plot_joint(sns.regplot, scatter_kws={'alpha': 0.3, 's': 10}, 
                            color=pred_data['color'])
            
            g.plot_marginals(sns.histplot, kde=True, color=pred_data['color'], alpha=0.6, bins=30)
            
            lims = [min(g.ax_joint.get_xlim()[0], g.ax_joint.get_ylim()[0]),
                    max(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])]
            g.ax_joint.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, zorder=0, label='1:1 line')
            
            combined_data = np.concatenate([gt_valid, pred_valid])
            p1, p99 = np.percentile(combined_data, [1, 99])
            g.ax_joint.set_xlim(p1, p99)
            g.ax_joint.set_ylim(p1, p99)
            
            g.set_axis_labels(f'Ground Truth ({units_map[mag_to_plot]})', 
                           f'Prediction ({units_map[mag_to_plot]})',
                           fontsize=12)
            
            rrmse_str = f"{rrmse*100:.2f}%" if not np.isnan(rrmse) else "N/A"
            g.fig.suptitle(f"{pred_data['label']}\n"
                          f"R = {corr:.4f} | RRMSE = {rrmse_str} | Bias = {bias:.2f} {units_map[mag_to_plot]}",
                          fontsize=12, fontweight='bold', y=1.02)
            
            stats_text = (f"N = {len(gt_valid):,}\n"
                         f"R = {corr:.4f}\n"
                         f"p < {p_value:.1e}\n"
                         f"RRMSE = {rrmse_str}\n"
                         f"Bias = {bias:.2f}")
            
            g.ax_joint.text(0.05, 0.95, stats_text,
                        transform=g.ax_joint.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            g.ax_joint.legend(loc='lower right', fontsize=10)
            g.ax_joint.grid(alpha=0.3)
            
            if save_dir is not None and filename_prefix is not None:
                out_dir = os.path.join(
                    str(save_dir),
                    "jointplot_comparison",
                    model_name,
                    self._format_od_folder(od_val)
                )
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(
                    out_dir,
                    f"{filename_prefix}_{model_name}.png"
                )
                g.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close(g.fig)
                self._save_metrics_json(metrics, Path(out_dir), out_path)
            
            print(f"\n{pred_data['label']} - {title_map[mag_to_plot]} at log(τ)={od_val:.1f}")
            print(f"  Pearson R:   {corr:.4f} (p={p_value:.2e})")
            print(f"  RRMSE:       {rrmse_str}")
            print(f"  RMSE:        {rmse:.2f} {units_map[mag_to_plot]}")
            print(f"  Bias:        {bias:.2f} {units_map[mag_to_plot]}")
            print(f"  Samples:     {len(gt_valid):,}")

    def plot_combined_jointplot(
        self,
        all_predictions: Dict,
        ground_truth: dict,
        mag_to_plot: str = "Bz",
        od_val: float = 0.0,
        logtau: np.ndarray = None,
        n_samples: int = 5000,
        save_dir: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None
    ):
        logtau = self._resolve_logtau(logtau)
        
        # Configuration
        title_map = {"T": "Temperature", "Vz": "Line-of-sight Velocity", "Bz": "Line-of-sight Magnetic Field"}
        units_map = {"T": "K", "Vz": "km/s", "Bz": "G"}
        
        od_idx = np.argmin(np.abs(logtau - od_val))
        gt_data = self._get_ground_truth_slice(ground_truth, mag_to_plot, od_val, logtau)
        
        all_data = []
        metrics = {
            "parameter": mag_to_plot,
            "parameter_name": title_map[mag_to_plot],
            "units": units_map[mag_to_plot],
            "optical_depth": float(od_val),
            "models": {}
        }
        
        for model_name, pred_data in all_predictions.items():
            pred = pred_data['prediction'][mag_to_plot][:, :, od_idx]
            
            gt_flat = gt_data.flatten()
            pred_flat = pred.flatten()
            
            valid = ~np.isnan(gt_flat) & ~np.isnan(pred_flat)
            gt_valid = gt_flat[valid]
            pred_valid = pred_flat[valid]
            
            # Compute per-model metrics before subsampling
            corr, p_value = pearsonr(gt_valid, pred_valid)
            rmse = np.sqrt(np.mean((pred_valid - gt_valid)**2))
            gt_mean_val = np.mean(np.abs(gt_valid))
            rrmse = rmse / gt_mean_val if gt_mean_val > 1e-10 else np.nan
            bias = np.mean(pred_valid - gt_valid)
            
            metrics["models"][model_name] = {
                "label": pred_data['label'],
                "pearson_r": float(corr),
                "p_value": float(p_value),
                "rrmse": float(rrmse) if not np.isnan(rrmse) else None,
                "rmse": float(rmse),
                "bias": float(bias),
                "n_samples": int(len(gt_valid))
            }
            
            if len(gt_valid) > n_samples:
                indices = np.random.choice(len(gt_valid), n_samples, replace=False)
                gt_valid = gt_valid[indices]
                pred_valid = pred_valid[indices]
            
            df_model = pd.DataFrame({
                'Ground Truth': gt_valid,
                'Prediction': pred_valid,
                'Model': pred_data['label']
            })
            
            all_data.append(df_model)
        
        df_combined = pd.concat(all_data, ignore_index=True)
        
        g = sns.JointGrid(data=df_combined, x='Ground Truth', y='Prediction', 
                           hue='Model', height=8, ratio=5, space=0.2)
        
        g.plot_joint(sns.scatterplot, alpha=0.4, s=10, 
                    palette={pred_data['label']: pred_data['color'] 
                            for pred_data in all_predictions.values()})
        
        g.plot_marginals(sns.histplot, kde=True, alpha=0.5,
                    palette={pred_data['label']: pred_data['color'] 
                            for pred_data in all_predictions.values()})
        
        lims = [min(g.ax_joint.get_xlim()[0], g.ax_joint.get_ylim()[0]),
                max(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])]
        g.ax_joint.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, zorder=0, label='1:1 line')
        
        all_gt = df_combined['Ground Truth'].values
        all_pred = df_combined['Prediction'].values
        combined_data = np.concatenate([all_gt, all_pred])
        p1, p99 = np.percentile(combined_data, [1, 99])
        g.ax_joint.set_xlim(p1, p99)
        g.ax_joint.set_ylim(p1, p99)
        
        g.set_axis_labels(f'Ground Truth ({units_map[mag_to_plot]})', 
                           f'Prediction ({units_map[mag_to_plot]})',
                           fontsize=13, fontweight='bold')
        
        g.fig.suptitle(f"Model Comparison: {title_map[mag_to_plot]} at log(τ)={od_val:.1f}",
                     fontsize=14, fontweight='bold', y=1.02)
        
        g.ax_joint.grid(alpha=0.3)
        g.ax_joint.legend(loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        if save_dir is not None:
            out_dir = os.path.join(
                str(save_dir),
                "combined_jointplot",
                self._format_od_folder(od_val)
            )
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, filename)
            g.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(g.fig)
            self._save_metrics_json(metrics, Path(out_dir), filename)
        
        print("\n" + "="*70)
        print(f"Combined Analysis: {title_map[mag_to_plot]} at log(τ)={od_val:.1f}")
        print("="*70)
        
        for model_name, pred_data in all_predictions.items():
            model_metrics = metrics["models"][model_name]
            print(f"\n{pred_data['label']}:")
            print(f"  Pearson R:   {model_metrics['pearson_r']:.4f} (p={model_metrics['p_value']:.2e})")
            rrmse_str = f"{model_metrics['rrmse']*100:.2f}%" if model_metrics['rrmse'] is not None else "N/A"
            print(f"  RRMSE:       {rrmse_str}")
            print(f"  RMSE:        {model_metrics['rmse']:.2f} {units_map[mag_to_plot]}")
            print(f"  Bias:        {model_metrics['bias']:.2f} {units_map[mag_to_plot]}")
            print(f"  Samples:     {model_metrics['n_samples']:,}")


class MuramAnalysis(ModestAnalysis):
    """Analysis tools for MURAM array-based ground truth comparisons.
    
    Inherits from ModestAnalysis but overrides ground truth access to handle
    MURAM's array-based format: ground_truth[param][H, W, n_logtau]
    MURAM uses T, Vz, Bz directly (no mapping needed).
    """

    def __init__(self, default_logtau: Optional[np.ndarray] = None, default_save_dir: Optional[Union[str, Path]] = None):
        super().__init__(default_logtau, default_save_dir)
        # MURAM uses T, Vz, Bz directly - override parent's mapping
        self.modest_key_mapping = {"T": "T", "Vz": "Vz", "Bz": "Bz"}
    
    def _get_ground_truth_slice(self, ground_truth: dict, param: str, od_val: float, logtau: np.ndarray) -> np.ndarray:
        """Extract 2D slice from MURAM array-based ground truth at specified optical depth.
        
        Parameters
        ----------
        ground_truth : dict
            MURAM ground truth with keys 'T', 'Vz', 'Bz' -> arrays (H, W, n_logtau)
        param : str
            Physical parameter ('T', 'Vz', or 'Bz')
        od_val : float
            Target optical depth value
        logtau : np.ndarray
            Optical depth grid for finding nearest index
            
        Returns
        -------
        np.ndarray
            2D slice (H, W) at the requested optical depth
        """
        od_idx = np.argmin(np.abs(logtau - od_val))
        gt_array = ground_truth[param]
        
        # Handle astropy units if present
        if hasattr(gt_array, 'value'):
            return gt_array.value[:, :, od_idx]
        return gt_array[:, :, od_idx]


__all__ = ["ModestAnalysis", "MuramAnalysis"]