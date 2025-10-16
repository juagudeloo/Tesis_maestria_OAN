import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from modules.train_utils import get_model, load_data
from modules.charge_data import DataCharger, ModestDataLoader
import scipy.stats
from skimage.metrics import structural_similarity as ssim
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
import astropy.units as u
from scipy.ndimage import uniform_filter

# Import all functions from modules_2.results_plot
from modules.results_plot import (
    load_modest_region,
    plot_modest_continuum,
    load_and_deconvolve_stokes,
    plot_modest_stokes_profiles,
    plot_modest_stokes_images,
    plot_modest_fitted_stokes,
    plot_modest_spinor_atmos,
    load_modest_region_full_scene,
    plot_modest_continuum_full_scene,
    load_and_deconvolve_stokes_full_scene,
    plot_modest_stokes_profiles_full_scene,
    plot_modest_stokes_images_full_scene,
    plot_modest_fitted_stokes_full_scene,
    plot_modest_spinor_atmos_full_scene,
    plot_per_tau_metrics,
    plot_model_summary_metrics,
    plot_overall_summary_metrics
)

def smooth_stokes_profiles(stokes_data, kernel_size=5):
    """
    Apply uniform smoothing to Stokes profiles along the wavelength axis.
    
    Args:
        stokes_data (np.ndarray): Stokes data of shape (NY, NX, n_stokes, n_wl)
        kernel_size (int): Size of the smoothing kernel along wavelength axis (default: 5)
    
    Returns:
        np.ndarray: Smoothed Stokes data
    """
    smoothed_stokes = np.zeros_like(stokes_data)
    
    for stokes_idx in range(stokes_data.shape[2]):  # Loop over Stokes parameters
        for y in range(stokes_data.shape[0]):  # Loop over Y spatial dimension
            for x in range(stokes_data.shape[1]):  # Loop over X spatial dimension
                # Apply 1D uniform filter along the wavelength axis (axis=-1)
                smoothed_stokes[y, x, stokes_idx, :] = uniform_filter(
                    stokes_data[y, x, stokes_idx, :], 
                    size=kernel_size, 
                    mode='nearest'
                )
    
    return smoothed_stokes

def rescale_muram_to_physical_units(muram_data):
    """
    Rescale normalized MuRAM data back to physical units.
    
    Args:
        muram_data (np.ndarray): Normalized MuRAM data of shape (nx, ny, n_logtau, n_params)
        
    Returns:
        np.ndarray: MuRAM data in physical units
    """
    # Physical scaling factors from DataCharger
    temp_max, temp_min = 2e4, 0
    vel_max, vel_min = 1e6, -1e6
    blos_max, blos_min = 3e3, -3e3
    
    muram_phys = np.zeros_like(muram_data)
    
    # Temperature (index 0): denormalize from [0,1] to [temp_min, temp_max]
    muram_phys[..., 0] = muram_data[..., 0] * (temp_max - temp_min) + temp_min
    
    # Velocity (index 1): denormalize from [0,1] to [vel_min, vel_max], then convert to km/s
    muram_phys[..., 1] = (muram_data[..., 1] * (vel_max - vel_min) + vel_min) / 1e5  # Convert to km/s
    
    # B_LOS (index 2): denormalize from [0,1] to [blos_min, blos_max]
    muram_phys[..., 2] = muram_data[..., 2] * (blos_max - blos_min) + blos_min
    
    return muram_phys

def main():
    # Configuration
    experiment_name = "paper_experiment"
    smoothing_kernel_size = 5  # Configurable smoothing kernel size
    
    # Setup paths
    models_dir = Path(f"models/{experiment_name}")
    images_dir = Path(f"images/{experiment_name}")
    images_dir.mkdir(parents=True, exist_ok=True)
    scatter_dir = images_dir / "scatterplots"
    imshow_dir = images_dir / "imshow_comparisons"
    hist_dir = images_dir / "histograms"
    modest_images_dir = Path("images/paper_experiment/Hinode_MODEST/")
    scatter_dir.mkdir(parents=True, exist_ok=True)
    imshow_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)
    modest_images_dir.mkdir(parents=True, exist_ok=True)
    
    test_files = ["087000"]

    # Find all trained model weights for both conditions
    noise_conditions = ["spectral_with_noise", "spectral_without_noise"]
    all_model_weights = {}
    
    for condition in noise_conditions:
        condition_dir = models_dir / condition
        if condition_dir.exists():
            # Find all w_* subdirectories
            w_dirs = sorted([d for d in condition_dir.iterdir() if d.is_dir() and d.name.startswith("w_")])
            all_model_weights[condition] = []
            
            for w_dir in w_dirs:
                # Find model files in each w directory
                model_files = list(w_dir.glob("final_model_*.pth"))
                if model_files:
                    w_value = w_dir.name.replace("w_", "")
                    all_model_weights[condition].extend([(model_file, w_value, condition) for model_file in model_files])
            
            print(f"Found {len(all_model_weights[condition])} models for {condition}")

    # Load test data
    data_charger = DataCharger(
        data_path="/scratchsan/observatorio/juagudeloo/data",
        filenames=test_files,
        nx=480,
        ny=480,
        nz=256
    )
    data_charger.charge_all_files()
    test_data = data_charger.reshape_for_training()
    n_logtau = data_charger.n_logtau
    
    modest_loader = ModestDataLoader(psf_path = "/scratchsan/observatorio/juagudeloo/data/hinode-MODEST/PSFs/hinode_psf_bin.0.16.fits")
    modest_loader.load_all()
    
    # MODEST full scene processing
    (sample_pixels, x_center_fov_pixel, y_center_fov_pixel, x_disk_center, y_disk_center,
     arrow_start_x, arrow_start_y, arrow_dx, arrow_dy) = load_modest_region_full_scene(modest_loader)
    
    plot_modest_continuum_full_scene(modest_loader, sample_pixels, x_center_fov_pixel, y_center_fov_pixel,
                                   x_disk_center, y_disk_center, arrow_start_x, arrow_start_y,
                                   arrow_dx, arrow_dy, modest_images_dir)
    
    deconvolved_obs_stokes_full = load_and_deconvolve_stokes_full_scene(modest_loader)
    
    wl, sample_wavelength_indices, stokes_labels = plot_modest_stokes_profiles_full_scene(modest_loader, deconvolved_obs_stokes_full, sample_pixels, modest_images_dir)
    
    plot_modest_stokes_images_full_scene(modest_loader, deconvolved_obs_stokes_full, sample_pixels, wl, sample_wavelength_indices, stokes_labels, modest_images_dir)
    
    plot_modest_fitted_stokes_full_scene(modest_loader, sample_pixels, modest_images_dir)
    
    spinor_atm_dict = plot_modest_spinor_atmos_full_scene(modest_loader, sample_pixels, modest_images_dir)
    
    print("MODEST atm params shape:", modest_loader.inverted_atmos.shape)
    
    # # Plot MuRAM sample pixels
    # from modules.results_plot import plot_muram_sample_pixels
    # muram_sample_pixels = plot_muram_sample_pixels(data_charger, test_data, test_files[0], modest_images_dir)
    # print(f"MuRAM sample pixels identified: {list(muram_sample_pixels.keys())}")

    # Extract MODEST full scene Stokes data (I and V only)
    modest_stokes = modest_loader.obs_stokes  # shape: (NY, NX, 2, n_wl)
    NY, NX, n_stokes, n_wl = modest_stokes.shape
    
    # Get the mask to identify valid pixels
    mask = modest_loader.mask  # shape: (NY, NX)
    valid_pixels = np.sum(mask)  # Count of valid (unmasked) pixels
    print(f"Valid pixels for neural network processing: {valid_pixels} out of {mask.size} ({100*valid_pixels/mask.size:.2f}%)")
    
    # Process both noise conditions
    for condition in noise_conditions:
        if condition not in all_model_weights or not all_model_weights[condition]:
            print(f"No models found for {condition}, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing {condition.upper()}")
        print(f"{'='*60}")
        
        # Prepare MODEST data based on condition
        if condition == "spectral_with_noise":
            # Use original observed stokes (already has noise)
            processed_modest_stokes = modest_stokes.copy()
            condition_suffix = "with_noise"
        else:  # spectral_without_noise
            # Apply smoothing to reduce noise
            processed_modest_stokes = smooth_stokes_profiles(modest_stokes, kernel_size=smoothing_kernel_size)
            condition_suffix = "without_noise_smoothed"
            print(f"Applied smoothing with kernel size {smoothing_kernel_size}")
        
        # Extract only valid (unmasked) pixels for neural network processing
        # Find indices of valid pixels
        valid_y_indices, valid_x_indices = np.where(mask)
        
        # Extract Stokes data only for valid pixels
        valid_stokes_data = processed_modest_stokes[valid_y_indices, valid_x_indices, :, :]  # shape: (n_valid, n_stokes, n_wl)
        valid_stokes_data = valid_stokes_data.astype(np.float32)
        valid_stokes_tensor = torch.tensor(valid_stokes_data, dtype=torch.float32)
        
        print(f"Processing {len(valid_stokes_data)} valid pixels (skipping {mask.size - len(valid_stokes_data)} masked pixels)")
        
        # Prepare output directory for this condition
        condition_pred_dir = modest_images_dir / f"modest_nn_predictions_{condition_suffix}"
        condition_pred_dir.mkdir(parents=True, exist_ok=True)

        # Loop over all model weights for this condition
        for weight_path, w_str, _ in all_model_weights[condition]:
            print(f"Processing model: w={w_str}")
            
            model = get_model(device="cpu")
            model.load_state_dict(torch.load(weight_path, map_location="cpu"))
            model.eval()

            with torch.no_grad():
                # Apply neural network only to valid pixels
                valid_pred = model(valid_stokes_tensor).numpy()  # shape: (n_valid, n_logtau*n_params)
                valid_pred = valid_pred.reshape(-1, n_logtau, 3)  # shape: (n_valid, n_logtau, 3)
                
                # Reconstruct full scene predictions with NaN for masked pixels
                pred_full_scene = np.full((NY, NX, 21, 3), np.nan, dtype=np.float32)
                
                # Place valid predictions back in their original positions
                pred_full_scene[valid_y_indices, valid_x_indices, :, :] = valid_pred

            # --- Rescale predicted values to physical units (only for valid pixels) ---
            # Scaling: [max, min]
            temp_max, temp_min = 2e4, 0
            vel_max, vel_min = 1e6, -1e6
            blos_max, blos_min = 3e3, -3e3

            pred_phys = np.full_like(pred_full_scene, np.nan)
            
            # Apply scaling only to valid (non-NaN) pixels
            valid_mask_3d = ~np.isnan(pred_full_scene)
            
            # Temperature
            temp_mask = valid_mask_3d[:, :, :, 0]
            pred_phys[:, :, :, 0][temp_mask] = pred_full_scene[:, :, :, 0][temp_mask] * (temp_max - temp_min) + temp_min
            
            # Velocity (convert to km/s after scaling)
            vel_mask = valid_mask_3d[:, :, :, 1]
            pred_phys[:, :, :, 1][vel_mask] = (pred_full_scene[:, :, :, 1][vel_mask] * (vel_max - vel_min) + vel_min) / 1e5
            
            # B_LOS
            blos_mask = valid_mask_3d[:, :, :, 2]
            pred_phys[:, :, :, 2][blos_mask] = pred_full_scene[:, :, :, 2][blos_mask] * (blos_max - blos_min) + blos_min

            # --- Plot comparisons for the three nodes ---
            modest_tau_values = [-2.0, -0.8, 0.0]
            modest_tau_indices = [0, 12, 20]
            param_names = ["Temperature", "Velocity", "B_LOS"]
            param_units = ["K", "km/s", "G"]
            param_spinor_keys = ["temperature_tau_", "velocity_tau_", "blos_tau_"]
            param_pred_indices = [0, 1, 2]

            for tau_idx, tau_val in zip(modest_tau_indices, modest_tau_values):
                tau_str = f"tau_{tau_val:.1f}"
                # Create output folders
                tau_imshow_dir = condition_pred_dir / "imshow" / tau_str
                tau_hist_dir = condition_pred_dir / "histograms" / tau_str
                tau_scatter_dir = condition_pred_dir / "scatter" / tau_str
                tau_imshow_dir.mkdir(parents=True, exist_ok=True)
                tau_hist_dir.mkdir(parents=True, exist_ok=True)
                tau_scatter_dir.mkdir(parents=True, exist_ok=True)

                for i, (param_name, param_unit, spinor_key, pred_idx) in enumerate(zip(param_names, param_units, param_spinor_keys, param_pred_indices)):
                    # Get predicted and spinor data (now in physical units)
                    pred_data = pred_phys[:, :, tau_idx, pred_idx]
                    spinor_data = spinor_atm_dict[f"{spinor_key}{tau_val}"]
                    
                    # Calculate color scale from valid (non-NaN) values only
                    valid_spinor = spinor_data[~np.isnan(spinor_data)]
                    if len(valid_spinor) > 0:
                        vmin = np.quantile(valid_spinor, 0.05) if param_name == "Temperature" else np.quantile(valid_spinor, 0.01)
                        vmax = np.quantile(valid_spinor, 0.95) if param_name == "Temperature" else np.quantile(valid_spinor, 0.99)
                        if param_name == "Velocity" or param_name == "B_LOS":
                            if np.abs(vmin) > np.abs(vmax):
                                vmax = np.abs(vmin)
                            else:
                                vmin = -np.abs(vmax)
                    else:
                        vmin, vmax = 0, 1
                    cmap = "hot" if param_name == "Temperature" else ("RdBu" if param_name == "Velocity" else "PiYG")
                    
                    # --- Imshow comparison ---
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    im0 = axes[0].imshow(spinor_data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
                    axes[0].set_title(f"SPINOR 2D {param_name}\nlogτ={tau_val:.1f} (Masked)")
                    axes[0].set_xticks([]); axes[0].set_yticks([])
                    divider0 = make_axes_locatable(axes[0])
                    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im0, cax=cax0)
                    
                    im1 = axes[1].imshow(pred_data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
                    axes[1].set_title(f"DL Model {param_name}\nlogτ={tau_val:.1f} (Masked)")
                    axes[1].set_xticks([]); axes[1].set_yticks([])
                    divider1 = make_axes_locatable(axes[1])
                    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im1, cax=cax1)
                    
                    plt.suptitle(f"{param_name} Comparison at logτ={tau_val:.1f} ({condition_suffix}, w={w_str})", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(tau_imshow_dir / f"{param_name}_imshow_{condition_suffix}_w_{w_str}.png", dpi=200)
                    plt.close(fig)
                    
                    # --- Histogram comparison (only valid pixels) ---
                    valid_spinor_flat = valid_spinor
                    valid_pred_flat = pred_data[~np.isnan(pred_data)]
                    
                    if len(valid_spinor_flat) > 0 and len(valid_pred_flat) > 0:
                        fig = plt.figure(figsize=(7, 4))
                        
                        # Create histograms and calculate integrals
                        n_spinor, bins_spinor, patches_spinor = plt.hist(valid_spinor_flat, bins=50, alpha=0.7, label="SPINOR 2D", color="blue", histtype="step", linewidth=2)
                        n_pred, bins_pred, patches_pred = plt.hist(valid_pred_flat, bins=50, alpha=0.7, label="DL Model", color="red", histtype="step", linewidth=2, linestyle="--")
                        
                        # Calculate histogram integrals (total counts)
                        integral_spinor = np.sum(n_spinor)
                        integral_pred = np.sum(n_pred)
                        
                        plt.xlabel(f"{param_name} [{param_unit}]")
                        plt.ylabel("Frequency")
                        plt.title(f"{param_name} Distribution at logτ={tau_val:.1f} ({condition_suffix}, w={w_str}) - Valid Pixels Only")
                        plt.legend()
                        
                        # Add text box with histogram integrals
                        textstr = f'SPINOR counts: {integral_spinor:.0f}\nDL Model counts: {integral_pred:.0f}\nRatio: {integral_pred/integral_spinor:.3f}'
                        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
                        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10, 
                                verticalalignment='top', bbox=props)
                        
                        plt.tight_layout()
                        plt.savefig(tau_hist_dir / f"{param_name}_hist_{condition_suffix}_w_{w_str}.png", dpi=200)
                        plt.close(fig)
                    
                    # --- Scatter plot comparison (only valid pixels) ---
                    # Find pixels that are valid in both SPINOR and prediction data
                    both_valid_mask = (~np.isnan(spinor_data)) & (~np.isnan(pred_data))
                    valid_spinor_scatter = spinor_data[both_valid_mask]
                    valid_pred_scatter = pred_data[both_valid_mask]
                    
                    if len(valid_spinor_scatter) > 0 and len(valid_pred_scatter) > 0:
                        fig = plt.figure(figsize=(5, 5))
                        plt.scatter(valid_spinor_scatter, valid_pred_scatter, s=1, alpha=0.5, color="blue")
                        plt.plot([valid_spinor_scatter.min(), valid_spinor_scatter.max()], 
                                [valid_spinor_scatter.min(), valid_spinor_scatter.max()], 'r--', lw=1)
                        plt.xlabel(f"SPINOR 2D {param_name} [{param_unit}]")
                        plt.ylabel(f"DL Model {param_name} [{param_unit}]")
                        
                        # Calculate statistics only on valid pixels
                        corr, _ = scipy.stats.pearsonr(valid_spinor_scatter, valid_pred_scatter)
                        rmse_val = np.sqrt(np.mean((valid_pred_scatter - valid_spinor_scatter) ** 2))
                        mean_spinor = np.mean(np.abs(valid_spinor_scatter))
                        rrmse_val = (rmse_val / mean_spinor * 100) if mean_spinor != 0 else 0
                        
                        plt.title(f"{param_name} Scatter (r={corr:.3f}, RRMSE={rrmse_val:.1f}%) - Valid Pixels Only")
                        plt.tight_layout()
                        plt.savefig(tau_scatter_dir / f"{param_name}_scatter_{condition_suffix}_w_{w_str}.png", dpi=200)
                        plt.close(fig)
                        
                        # Print summary statistics
                        print(f"MODEST {param_name} logτ={tau_val:.1f} {condition_suffix} w={w_str}: r={corr:.3f}, RMSE={rmse_val:.3f}, RRMSE={rrmse_val:.2f}% (n_valid={len(valid_spinor_scatter)})")
                    else:
                        print(f"MODEST {param_name} logτ={tau_val:.1f} {condition_suffix} w={w_str}: No valid pixels for comparison")

    # Process MuRAM test data for both conditions
    all_rrmse = []
    all_pearson = []
    all_ssim = []
    all_wstr = []
    all_conditions = []
    all_tau_vals = None

    for condition in noise_conditions:
        if condition not in all_model_weights or not all_model_weights[condition]:
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing MuRAM test data for {condition.upper()}")
        print(f"{'='*60}")
        
        # Load appropriate test data for this condition
        add_noise = (condition == "spectral_with_noise")
        condition_test_data = load_data(test_files, apply_spectral_conditions=True, add_noise=add_noise)
        
        for weight_path, w_str, _ in all_model_weights[condition]:
            model = get_model(device="cpu")
            model.load_state_dict(torch.load(weight_path, map_location="cpu"))
            model.eval()

            for fname in test_files:
                if fname not in condition_test_data:
                    continue
                    
                stokes = condition_test_data[fname]["stokes_reshaped"]
                muram = condition_test_data[fname]["muram_reshaped"]
                nx, ny = 480, 480
                n_params = muram.shape[1] // n_logtau

                # Generate predictions
                with torch.no_grad():
                    stokes_tensor = torch.tensor(stokes, dtype=torch.float32)
                    pred = model(stokes_tensor).numpy()
                    pred = pred.reshape(nx, ny, n_logtau, n_params)

                # Rescale both predictions and MuRAM data to physical units
                pred_phys = rescale_predictions_to_physical_units(pred)
                muram_reshaped = muram.reshape(nx, ny, n_logtau, n_params)
                true_phys = rescale_muram_to_physical_units(muram_reshaped)

                quantities = [
                    (0, r"$T$", "K", "hot"),
                    (1, r"$v_{\text{LOS}}$", "km/s", "bwr_r"),
                    (2, r"$B_{\text{LOS}}$", "G", "PiYG"),
                ]

                # Prepare arrays to store metrics for summary plots
                rrmse_arr = np.zeros((n_logtau, 3))
                pearson_arr = np.zeros((n_logtau, 3))
                ssim_arr = np.zeros((n_logtau, 3))
                tau_vals = np.array([data_charger.new_logtau[tau_idx] for tau_idx in range(n_logtau)])
                all_tau_vals = tau_vals

                # Create condition-specific directories
                condition_suffix = "with_noise" if add_noise else "without_noise"
                condition_scatter_dir = scatter_dir / condition_suffix
                condition_imshow_dir = imshow_dir / condition_suffix
                condition_hist_dir = hist_dir / condition_suffix
                condition_scatter_dir.mkdir(parents=True, exist_ok=True)
                condition_imshow_dir.mkdir(parents=True, exist_ok=True)
                condition_hist_dir.mkdir(parents=True, exist_ok=True)

                # Loop over each optical depth
                for tau_idx in range(n_logtau):
                    tau_val = tau_vals[tau_idx]
                    tau_str = f"tau_{tau_val:.2f}"
                    tau_scatter_dir = condition_scatter_dir / tau_str
                    tau_imshow_dir = condition_imshow_dir / tau_str
                    tau_hist_dir = condition_hist_dir / tau_str
                    tau_scatter_dir.mkdir(parents=True, exist_ok=True)
                    tau_imshow_dir.mkdir(parents=True, exist_ok=True)
                    tau_hist_dir.mkdir(parents=True, exist_ok=True)
                    
                    plot_per_tau_metrics(pred_phys, true_phys, tau_idx, tau_val, tau_str, quantities, fname, f"{condition_suffix}_w_{w_str}", tau_imshow_dir, tau_scatter_dir, tau_hist_dir, rrmse_arr, pearson_arr, ssim_arr)

                # Save metrics for summary plots
                all_rrmse.append(rrmse_arr)
                all_pearson.append(pearson_arr)
                all_ssim.append(ssim_arr)
                all_wstr.append(f"{condition_suffix}_w_{w_str}")
                all_conditions.append(condition_suffix)
                
                plot_model_summary_metrics(tau_vals, rrmse_arr, pearson_arr, ssim_arr, quantities, fname, f"{condition_suffix}_w_{w_str}", condition_hist_dir, condition_scatter_dir, condition_imshow_dir)

    # Generate overall summary plots
    plot_overall_summary_metrics(all_rrmse, all_pearson, all_ssim, all_wstr, all_tau_vals, quantities, hist_dir, scatter_dir, imshow_dir)
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETED!")
    print(f"Images saved in {images_dir}")
    print(f"MODEST predictions saved for both noise conditions")
    print(f"Smoothing kernel size used for no-noise case: {smoothing_kernel_size}")
    print(f"{'='*60}")

def rescale_predictions_to_physical_units(pred_region):
    """
    Rescale normalized predictions back to physical units.
    
    Args:
        pred_region (np.ndarray): Predicted values in normalized form
        
    Returns:
        np.ndarray: Predictions in physical units
    """
    # Scaling: [max, min]
    temp_max, temp_min = 2e4, 0
    vel_max, vel_min = 1e6, -1e6
    blos_max, blos_min = 3e3, -3e3

    pred_phys = np.zeros_like(pred_region)
    # Temperature
    pred_phys[..., 0] = pred_region[..., 0] * (temp_max - temp_min) + temp_min
    # Velocity (convert to km/s after scaling)
    pred_phys[..., 1] = (pred_region[..., 1] * (vel_max - vel_min) + vel_min) / 1e5
    # B_LOS
    pred_phys[..., 2] = pred_region[..., 2] * (blos_max - blos_min) + blos_min
    
    return pred_phys

if __name__ == "__main__":
    main()