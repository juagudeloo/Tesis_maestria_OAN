import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from modules_2.train_utils import get_model, load_data
from modules_2.charge_data import DataCharger, ModestDataLoader
import scipy.stats
from skimage.metrics import structural_similarity as ssim
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
import astropy.units as u

# Import all functions from modules_2.results_plot
from modules.results_plot import (
    load_modest_region,
    plot_modest_continuum,
    load_and_deconvolve_stokes,
    plot_modest_stokes_profiles,
    plot_modest_stokes_images,
    plot_modest_fitted_stokes,
    plot_modest_spinor_atmos,
    plot_per_tau_metrics,
    plot_model_summary_metrics,
    plot_overall_summary_metrics
)

def main():
    # 1. Setup paths and test files
    experiment_name = "paper_experiment"
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
    test_files = ["087000",
                #   "095000", "107000"
                  ]

    # 2. Find all trained model weights
    model_weights = sorted(models_dir.glob("final_model_w_*.pth"))

    # 3. Load test data
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
    # MODEST region processing
    (_, _, y_start, y_end, x_start, x_end, sample_x, sample_y,
     x_center_fov_pixel, y_center_fov_pixel, x_disk_center, y_disk_center,
     arrow_start_x, arrow_start_y, arrow_dx, arrow_dy, region_continuum) = load_modest_region(modest_loader)
    plot_modest_continuum(modest_loader, y_start, y_end, x_start, x_end, sample_x, sample_y,
                          x_center_fov_pixel, y_center_fov_pixel, x_disk_center, y_disk_center,
                          arrow_start_x, arrow_start_y, arrow_dx, arrow_dy, region_continuum, modest_images_dir)
    deconvolved_obs_stokes_region = load_and_deconvolve_stokes(modest_loader, y_start, y_end, x_start, x_end)
    wl, sample_wavelength_idx, stokes_labels = plot_modest_stokes_profiles(modest_loader, deconvolved_obs_stokes_region, sample_x, sample_y, modest_images_dir)
    plot_modest_stokes_images(modest_loader, deconvolved_obs_stokes_region, sample_x, sample_y, wl, sample_wavelength_idx, stokes_labels, modest_images_dir)
    plot_modest_fitted_stokes(modest_loader, y_start, y_end, x_start, x_end, sample_x, sample_y, modest_images_dir)
    spinor_atm_dict = plot_modest_spinor_atmos(modest_loader, y_start, y_end, x_start, x_end, sample_x, sample_y, modest_images_dir)
    print("MODEST atm params shape:", modest_loader.inverted_atmos.shape)

    # Extract MODEST region Stokes data (I and V only)
    modest_loader.load_all()
    modest_stokes = modest_loader.obs_stokes[y_start:y_end, x_start:x_end, :, :]  # shape: (NY, NX, 2, n_wl)
    NY, NX, n_stokes, n_wl = modest_stokes.shape
    modest_stokes_reshaped = modest_stokes.reshape(-1, n_stokes, n_wl)  # shape: (NY*NX, 2, n_wl)
    modest_stokes_reshaped = modest_stokes_reshaped.astype(np.float32)
    stokes_tensor = torch.tensor(modest_stokes_reshaped, dtype=torch.float32)
        
    # Prepare output directory for MODEST predictions
    modest_pred_dir = Path("images/paper_experiment/Hinode_MODEST/modest_nn_predictions")
    modest_pred_dir.mkdir(parents=True, exist_ok=True)

    # Loop over all model weights and generate predictions for MODEST region
    for weight_path in model_weights:
        w_str = weight_path.stem.split("_w_")[1].replace(".pth", "")
        model = get_model(device="cpu")
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        model.eval()

        with torch.no_grad():
            pred = model(stokes_tensor).numpy()  # shape: (NY*NX, n_logtau, n_params)
            pred_region = pred.reshape(NY, NX, 21, 3)

        # --- Rescale predicted values to physical units ---
        # Scaling: [max, min]
        temp_max, temp_min = 2e4, 0
        vel_max, vel_min = 1e6, -1e6
        blos_max, blos_min = 3e3, -3e3

        pred_phys = np.zeros_like(pred_region)
        # Temperature
        pred_phys[:, :, :, 0] = pred_region[:, :, :, 0] * (temp_max - temp_min) + temp_min
        # Velocity (convert to km/s after scaling)
        pred_phys[:, :, :, 1] = (pred_region[:, :, :, 1] * (vel_max - vel_min) + vel_min) / 1e5
        # B_LOS
        pred_phys[:, :, :, 2] = pred_region[:, :, :, 2] * (blos_max - blos_min) + blos_min

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
            tau_imshow_dir = modest_pred_dir / "imshow" / tau_str
            tau_hist_dir = modest_pred_dir / "histograms" / tau_str
            tau_scatter_dir = modest_pred_dir / "scatter" / tau_str
            tau_imshow_dir.mkdir(parents=True, exist_ok=True)
            tau_hist_dir.mkdir(parents=True, exist_ok=True)
            tau_scatter_dir.mkdir(parents=True, exist_ok=True)

            for i, (param_name, param_unit, spinor_key, pred_idx) in enumerate(zip(param_names, param_units, param_spinor_keys, param_pred_indices)):
                # Get predicted and spinor data (now in physical units)
                pred_data = pred_phys[:, :, tau_idx, pred_idx]
                spinor_data = spinor_atm_dict[f"{spinor_key}{tau_val}"]
                # --- Imshow comparison ---
                vmin = np.quantile(spinor_data, 0.05) if param_name == "Temperature" else np.quantile(spinor_data, 0.01)
                vmax = np.quantile(spinor_data, 0.95) if param_name == "Temperature" else np.quantile(spinor_data, 0.99)
                if param_name == "Velocity" or param_name == "B_LOS":
                    if np.abs(vmin) > np.abs(vmax):
                        vmax = np.abs(vmin)
                    else:
                        vmin = -np.abs(vmax)
                cmap = "hot" if param_name == "Temperature" else ("RdBu_r" if param_name == "Velocity" else "PiYG")
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                im0 = axes[0].imshow(spinor_data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
                axes[0].set_title(f"SPINOR 2D {param_name}\nlogτ={tau_val:.1f}")
                axes[0].set_xticks([]); axes[0].set_yticks([])
                divider0 = make_axes_locatable(axes[0])
                cax0 = divider0.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im0, cax=cax0)
                im1 = axes[1].imshow(pred_data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
                axes[1].set_title(f"DL Model {param_name}\nlogτ={tau_val:.1f}")
                axes[1].set_xticks([]); axes[1].set_yticks([])
                divider1 = make_axes_locatable(axes[1])
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax1)
                plt.suptitle(f"{param_name} Comparison at logτ={tau_val:.1f} (w={w_str})", fontsize=14)
                plt.tight_layout()
                plt.savefig(tau_imshow_dir / f"{param_name}_imshow_w_{w_str}.png", dpi=200)
                plt.close(fig)
                # --- Histogram comparison ---
                fig = plt.figure(figsize=(7, 4))
                plt.hist(spinor_data.flatten(), bins=50, alpha=0.7, label="SPINOR 2D", color="blue", histtype="step", linewidth=2)
                plt.hist(pred_data.flatten(), bins=50, alpha=0.7, label="DL Model", color="red", histtype="step", linewidth=2, linestyle="--")
                plt.xlabel(f"{param_name} [{param_unit}]")
                plt.ylabel("Frequency")
                plt.title(f"{param_name} Distribution at logτ={tau_val:.1f} (w={w_str})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(tau_hist_dir / f"{param_name}_hist_w_{w_str}.png", dpi=200)
                plt.close(fig)
                # --- Scatter plot comparison ---
                fig = plt.figure(figsize=(5, 5))
                plt.scatter(spinor_data.flatten(), pred_data.flatten()/3, s=1, alpha=0.5, color="blue")
                plt.plot([spinor_data.min(), spinor_data.max()], [spinor_data.min(), spinor_data.max()], 'r--', lw=1)
                plt.xlabel(f"SPINOR 2D {param_name} [{param_unit}]")
                plt.ylabel(f"DL Model {param_name} [{param_unit}]")
                corr, _ = scipy.stats.pearsonr(spinor_data.flatten(), pred_data.flatten())
                rmse_val = np.sqrt(np.mean((pred_data.flatten() - spinor_data.flatten()) ** 2))
                mean_spinor = np.mean(np.abs(spinor_data.flatten()))
                rrmse_val = (rmse_val / mean_spinor * 100) if mean_spinor != 0 else 0
                plt.title(f"{param_name} Scatter (r={corr:.3f}, RRMSE={rrmse_val:.1f}%)")
                plt.tight_layout()
                plt.savefig(tau_scatter_dir / f"{param_name}_scatter_w_{w_str}.png", dpi=200)
                plt.close(fig)
                # Print summary statistics
                print(f"MODEST {param_name} logτ={tau_val:.1f} w={w_str}: r={corr:.3f}, RMSE={rmse_val:.3f}, RRMSE={rrmse_val:.2f}%")

    # 4. Loop over models and test files
    # Store metrics for all weights
    all_rrmse = []
    all_pearson = []
    all_ssim = []
    all_wstr = []
    all_tau_vals = None

    for weight_path in model_weights:
        w_str = weight_path.stem.split("_w_")[1].replace(".pth", "")
        model = get_model(device="cpu")
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        model.eval()

        for fname in test_files:
            stokes = test_data[fname]["stokes_reshaped"]  # shape: (nx*ny, n_stokes, n_wl)
            muram = test_data[fname]["muram_reshaped"]    # shape: (nx*ny, n_logtau*n_params)
            nx, ny = 480, 480
            n_params = muram.shape[1] // n_logtau

            # Generate predictions
            with torch.no_grad():
                stokes_tensor = torch.tensor(stokes, dtype=torch.float32)
                pred = model(stokes_tensor).numpy()
                pred = pred.reshape(nx, ny, n_logtau, n_params)

            quantities = [
                (0, r"$T$", "K", "hot"),
                (1, r"$v_{\text{LOS}}$", "km/s", "bwr_r"),
                (2, r"$B_{\text{LOS}}$", "G", "PiYG"),
            ]
            true = muram.reshape(nx, ny, n_logtau, n_params)

            # Prepare arrays to store metrics for summary plots
            rrmse_arr = np.zeros((n_logtau, 3))
            pearson_arr = np.zeros((n_logtau, 3))
            ssim_arr = np.zeros((n_logtau, 3))
            tau_vals = np.array([data_charger.new_logtau[tau_idx] for tau_idx in range(n_logtau)])
            all_tau_vals = tau_vals  # Save for summary plots

            # Loop over each optical depth
            for tau_idx in range(n_logtau):
                tau_val = tau_vals[tau_idx]
                tau_str = f"tau_{tau_val:.2f}"
                tau_scatter_dir = scatter_dir / tau_str
                tau_imshow_dir = imshow_dir / tau_str
                tau_hist_dir = hist_dir / tau_str
                tau_scatter_dir.mkdir(parents=True, exist_ok=True)
                tau_imshow_dir.mkdir(parents=True, exist_ok=True)
                tau_hist_dir.mkdir(parents=True, exist_ok=True)
                plot_per_tau_metrics(pred, true, tau_idx, tau_val, tau_str, quantities, fname, w_str, tau_imshow_dir, tau_scatter_dir, tau_hist_dir, rrmse_arr, pearson_arr, ssim_arr)
            # Save metrics for summary plots
            all_rrmse.append(rrmse_arr)
            all_pearson.append(pearson_arr)
            all_ssim.append(ssim_arr)
            all_wstr.append(w_str)
            plot_model_summary_metrics(tau_vals, rrmse_arr, pearson_arr, ssim_arr, quantities, fname, w_str, hist_dir, scatter_dir, imshow_dir)
    plot_overall_summary_metrics(all_rrmse, all_pearson, all_ssim, all_wstr, all_tau_vals, quantities, hist_dir, scatter_dir, imshow_dir)
    print(f"Images saved in {images_dir}")
    

if __name__ == "__main__":
    main()