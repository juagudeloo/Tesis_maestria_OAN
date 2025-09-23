import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
import astropy.units as u
from scipy.fft import fft2, ifft2, fftshift
from skimage.metrics import structural_similarity as ssim
import scipy.stats
from pathlib import Path


def load_modest_region(modest_loader):
    # --- 1. Define region boundaries and constants ---
    y_start, y_end = 0, 100
    x_start, x_end = 500, 700
    NX = x_end - x_start
    NY = y_end - y_start

    # --- 2. Load continuum data using ModestDataLoader ---
    continuum = modest_loader.load_continuum()
    # If you need the header, you can load it separately if required
    with fits.open(modest_loader.modest_dir / "continuum.fits") as hdul:
        cont_header = hdul[0].header

    # --- 3. Calculate disk center and arrow parameters ---
    y_center_fov_arcsec = -126.0
    x_center_fov_arcsec = 16.0
    pixel_scale = 0.32
    x_center_fov_pixel = continuum.shape[1] / 2
    y_center_fov_pixel = continuum.shape[0] / 2
    x_disk_offset_arcsec = 0 - x_center_fov_arcsec
    y_disk_offset_arcsec = 0 - y_center_fov_arcsec
    x_disk_offset_pixels = x_disk_offset_arcsec / pixel_scale
    y_disk_offset_pixels = y_disk_offset_arcsec / pixel_scale
    x_disk_center = x_center_fov_pixel + x_disk_offset_pixels
    y_disk_center = y_center_fov_pixel + y_disk_offset_pixels
    arrow_start_x = 800
    arrow_start_y = 400
    arrow_dx = x_disk_offset_pixels/10
    arrow_dy = y_disk_offset_pixels/10

    # --- 4. Select region and sample pixel ---
    region_continuum = continuum[y_start:y_end, x_start:x_end]
    min_idx = np.unravel_index(np.argmin(region_continuum), region_continuum.shape)
    sample_y, sample_x = min_idx

    return (continuum, cont_header, y_start, y_end, x_start, x_end, sample_x, sample_y,
            x_center_fov_pixel, y_center_fov_pixel, x_disk_center, y_disk_center,
            arrow_start_x, arrow_start_y, arrow_dx, arrow_dy, region_continuum)

def plot_modest_continuum(modest_loader, y_start, y_end, x_start, x_end, sample_x, sample_y,
                          x_center_fov_pixel, y_center_fov_pixel, x_disk_center, y_disk_center,
                          arrow_start_x, arrow_start_y, arrow_dx, arrow_dy, region_continuum, modest_images_dir):
    continuum = modest_loader.continuum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(continuum, cmap='gray', origin='lower')
    rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, linewidth=2, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    ax1.plot(x_center_fov_pixel, y_center_fov_pixel, 'b+', markersize=15, markeredgewidth=3, label=f'FOV Center\n({16.0}, {-126.0})')
    if 0 <= x_disk_center <= continuum.shape[1] and 0 <= y_disk_center <= continuum.shape[0]:
        ax1.plot(x_disk_center, y_disk_center, 'r+', markersize=15, markeredgewidth=3, label='Solar Disk Center\n(0, 0)')
    else:
        ax1.arrow(arrow_start_x, arrow_start_y, arrow_dx, arrow_dy, head_width=20, head_length=20, fc='blue', ec='blue', linewidth=3, label='Solar Disk Center')
    ax1.legend()
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_title("AR11967 - Full view")
    ax2.imshow(region_continuum, cmap='gray', origin='lower')
    textstr = f'AR11967\nFOV Center\n(16.0, -126.0)\nZürich: Fkc\nMagnetic: βγδ\nμ=0.99'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    ax1.text(0.05, 0.05, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='bottom', bbox=props)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_title("AR11967 - Selected region")
    ax2.plot(sample_x, sample_y, 'rx', markersize=10, markeredgewidth=2, label='Sample pixel')
    ax2.legend()
    con1 = ConnectionPatch(xyA=(0, 0), coordsA=ax2.transData, xyB=(x_start, y_start), coordsB=ax1.transData, color='r', linestyle='-', linewidth=2.0, alpha=0.8)
    con2 = ConnectionPatch(xyA=(0, y_end-y_start), coordsA=ax2.transData, xyB=(x_start, y_end), coordsB=ax1.transData, color='r', linestyle='-', linewidth=2.0, alpha=0.8)
    fig.add_artist(con1); fig.add_artist(con2)
    plt.tight_layout()
    fig.savefig(modest_images_dir / "continuum_with_zoom_and_center.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def load_and_deconvolve_stokes(modest_loader, y_start, y_end, x_start, x_end):
    obs_stokes = modest_loader.obs_stokes
    hinode_psf = modest_loader.load_psf()
    def wiener_deconv(image, psf, noise_level=0.01):
        psf_padded = np.zeros_like(image)
        psf_center = (psf.shape[0] // 2, psf.shape[1] // 2)
        img_center = (image.shape[0] // 2, image.shape[1] // 2)
        y_start_ = img_center[0] - psf_center[0]
        y_end_ = y_start_ + psf.shape[0]
        x_start_ = img_center[1] - psf_center[1]
        x_end_ = x_start_ + psf.shape[1]
        psf_padded[y_start_:y_end_, x_start_:x_end_] = psf
        psf_padded = fftshift(psf_padded)
        img_fft = fft2(image)
        psf_fft = fft2(psf_padded)
        psf_conj = np.conj(psf_fft)
        psf_abs_sq = np.abs(psf_fft)**2
        wiener_filter = psf_conj / (psf_abs_sq + noise_level)
        result_fft = img_fft * wiener_filter
        result = np.real(ifft2(result_fft))
        return result
    def deconvolve_stokes_data(stokes_data, psf):
        ny, nx, n_stokes, n_wavelength = stokes_data.shape
        deconvolved_stokes = np.zeros_like(stokes_data)
        for wave_idx in range(n_wavelength):
            for stokes_idx in range(n_stokes):
                image_2d = stokes_data[:, :, stokes_idx, wave_idx]
                deconvolved_image = wiener_deconv(image_2d, psf)
                deconvolved_stokes[:, :, stokes_idx, wave_idx] = deconvolved_image
        return deconvolved_stokes
    deconvolved_obs_stokes = deconvolve_stokes_data(obs_stokes, hinode_psf)
    deconvolved_obs_stokes_region = deconvolved_obs_stokes[y_start:y_end, x_start:x_end, :, :]
    return deconvolved_obs_stokes_region

def plot_modest_stokes_profiles(modest_loader, deconvolved_obs_stokes_region, sample_x, sample_y, modest_images_dir):
    wl = modest_loader.wl
    stokes_labels = ['Stokes I', 'Stokes V']
    sample_pixel_stokes = deconvolved_obs_stokes_region[sample_y, sample_x, :, :]
    sample_wavelength_idx = np.argmin(sample_pixel_stokes[0, :])-6
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for col, label in enumerate(stokes_labels):
        ax = axes[col]
        ax.plot(wl, sample_pixel_stokes[col, :], 'b-', linewidth=1.5)
        ax.plot(wl[sample_wavelength_idx], sample_pixel_stokes[col, sample_wavelength_idx], 'ro', markersize=6)
        ax.set_title(f'{label}', fontsize=12)
        ax.set_ylabel('Intensity')
        ax.set_xlabel('Wavelength [Å]')
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'MODEST Stokes Profiles - Sample Pixel ({sample_x}, {sample_y}) [no_spatial]', fontsize=15)
    plt.tight_layout()
    plt.savefig(modest_images_dir / "modest_sample_pixel_stokes_profiles_nospatial.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    return wl, sample_wavelength_idx, stokes_labels

def plot_modest_stokes_images(modest_loader, deconvolved_obs_stokes_region, sample_x, sample_y, wl, sample_wavelength_idx, stokes_labels, modest_images_dir):
    continuum = modest_loader.continuum
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    stokes_region_sample = deconvolved_obs_stokes_region[:, :, :, sample_wavelength_idx]
    cont_vmin = np.quantile(continuum, 0.05)
    cont_vmax = np.quantile(continuum, 0.95)
    for col, label in enumerate(stokes_labels):
        ax = axes[col]
        if col == 0:
            vmin, vmax = cont_vmin, cont_vmax
        else:
            data_sample = stokes_region_sample[:, :, col]
            abs_max = np.max(np.abs([np.quantile(data_sample, 0.01), np.quantile(data_sample, 0.99)]))
            vmin, vmax = -abs_max, abs_max
        im = ax.imshow(stokes_region_sample[:, :, col], cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=12)
        ax.set_ylabel('Y [pixels]')
        ax.set_xlabel('X [pixels]')
        ax.set_xticks([]); ax.set_yticks([])
        ax.plot(sample_x, sample_y, 'rx', markersize=8, markeredgewidth=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    plt.suptitle(f'MODEST Stokes Parameters at λ = {wl[sample_wavelength_idx]:.3f} Å [no_spatial]', fontsize=15)
    plt.tight_layout()
    plt.savefig(modest_images_dir / "modest_stokes_parameters_nospatial.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_modest_fitted_stokes(modest_loader, y_start, y_end, x_start, x_end, sample_x, sample_y, modest_images_dir):
    inverted_profs = modest_loader.inverted_profs
    wl = modest_loader.wl
    wl_inv = modest_loader.wl_inv
    inverted_profs_region = inverted_profs[y_start:y_end, x_start:x_end, :, :]
    sample_pixel_inverted = inverted_profs_region[sample_y, sample_x, :, :]
    sample_wavelength_idx_inv = np.argmin(sample_pixel_inverted[0, :])-50
    stokes_labels = ['Stokes I', 'Stokes V']
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, (ax, label) in enumerate(zip(axes, stokes_labels)):
        ax.plot(wl_inv, sample_pixel_inverted[i, :], 'r-', linewidth=1.5)
        ax.plot(wl_inv[sample_wavelength_idx_inv], sample_pixel_inverted[i, sample_wavelength_idx_inv], 'bo', markersize=6)
        ax.set_title(f'{label} profile (Inverted)')
        if i == 0:
            ax.legend(['Fitted Profile', f'Sample wavelength\n(λ = {wl_inv[sample_wavelength_idx_inv]:.3f})'], loc='lower right')
        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel('Intensity' if i == 0 else 'Polarization Signal')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(modest_images_dir / "sample_pixel_inverted_stokes_profiles.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    # Plot fitted Stokes images at sample wavelength
    inverted_profs_region_sample = inverted_profs_region[:, :, :, sample_wavelength_idx_inv]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'Inverted Stokes Parameters at λ = {wl_inv[sample_wavelength_idx_inv]:.3f} - Selected Region', fontsize=16)
    for i, (ax, label) in enumerate(zip(axes, stokes_labels)):
        im = ax.imshow(inverted_profs_region_sample[:, :, i], cmap='gray', origin='lower')
        ax.set_title(label)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.plot(sample_x, sample_y, 'bx', markersize=8, markeredgewidth=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(modest_images_dir / "inverted_stokes_parameters_selected_region.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_modest_spinor_atmos(modest_loader, y_start, y_end, x_start, x_end, sample_x, sample_y, modest_images_dir):
    inverted_atm = modest_loader.inverted_atmos
    tau_values = [-2.0, -0.8, 0.0]
    temp_indices = [8, 6, 7]
    vel_indices = [20, 18, 19]
    mag_field_indices = [11, 9, 10]
    gamma_indices = [14, 12, 13]
    # Temperature
    temp_nodes = inverted_atm[temp_indices, :, :]
    temp_region = temp_nodes[:, y_start:y_end, x_start:x_end]
    fig, axes = plt.subplots(1, 3, figsize=(18, 3.5))
    temp_labels = ['log τ = -2.0', 'log τ = -0.8', 'log τ = 0.0']
    for i, (ax, label) in enumerate(zip(axes, temp_labels)):
        vmin = np.quantile(temp_region[i], 0.05)
        vmax = np.quantile(temp_region[i], 0.95)
        im = ax.imshow(temp_region[i], cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.plot(sample_x, sample_y, 'cx', markersize=8, markeredgewidth=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('[K]')
    plt.suptitle('SPINOR 2D Inverted Temperature at Different Atmospheric Heights', fontsize=16)
    plt.tight_layout()
    plt.savefig(modest_images_dir / "spinor_temperature_three_nodes.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    # Velocity
    vel_nodes = inverted_atm[vel_indices, :, :]
    vel_region = vel_nodes[:, y_start:y_end, x_start:x_end]
    fig, axes = plt.subplots(1, 3, figsize=(18, 3.5))
    vel_labels = ['log τ = -2.0', 'log τ = -0.8', 'log τ = 0.0']
    for i, (ax, label) in enumerate(zip(axes, vel_labels)):
        vmin = np.quantile(vel_region[i], 0.01)
        vmax = np.quantile(vel_region[i], 0.99)
        im = ax.imshow(vel_region[i], cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.plot(sample_x, sample_y, 'kx', markersize=8, markeredgewidth=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('[Km/s]')
    plt.suptitle(r'SPINOR 2D Inverted $v_\text{LOS}$ at Different Atmospheric Heights', fontsize=16)
    plt.tight_layout()
    plt.savefig(modest_images_dir / "spinor_velocity_three_nodes.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    # B_LOS
    bcos_gamma_nodes = []
    for i in range(3):
        b_field = inverted_atm[mag_field_indices[i], :, :]
        gamma_deg = inverted_atm[gamma_indices[i], :, :]
        gamma_rad = np.deg2rad(gamma_deg)
        bcos_gamma = b_field * np.cos(gamma_rad)
        bcos_gamma_nodes.append(bcos_gamma)
    bcos_gamma_nodes = np.array(bcos_gamma_nodes)
    bcos_gamma_region = bcos_gamma_nodes[:, y_start:y_end, x_start:x_end]
    fig, axes = plt.subplots(1, 3, figsize=(18, 3.5))
    bcos_gamma_labels = ['log τ = -2.0', 'log τ = -0.8', 'log τ = 0.0']
    for i, (ax, label) in enumerate(zip(axes, bcos_gamma_labels)):
        vmin = np.quantile(bcos_gamma_region[i], 0.01)
        vmax = np.quantile(bcos_gamma_region[i], 0.99)
        if np.abs(vmin) > np.abs(vmax): vmax = np.abs(vmin)
        else: vmin = -np.abs(vmax)
        im = ax.imshow(bcos_gamma_region[i], cmap='PiYG', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.plot(sample_x, sample_y, 'kx', markersize=8, markeredgewidth=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('[G]')
    plt.suptitle(r'SPINOR 2D Inverted $B_{\text{LOS}}$  at Different Atmospheric Heights', fontsize=16)
    plt.tight_layout()
    plt.savefig(modest_images_dir / "spinor_bcos_gamma_three_nodes.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    # Save all data for further use
    spinor_atm_dict = {}
    for i, tau in enumerate(tau_values):
        spinor_atm_dict[f"temperature_tau_{tau}"] = inverted_atm[temp_indices[i], y_start:y_end, x_start:x_end]
        spinor_atm_dict[f"velocity_tau_{tau}"] = inverted_atm[vel_indices[i], y_start:y_end, x_start:x_end]
        b_field = inverted_atm[mag_field_indices[i], y_start:y_end, x_start:x_end]
        gamma_deg = inverted_atm[gamma_indices[i], y_start:y_end, x_start:x_end]
        gamma_rad = np.deg2rad(gamma_deg)
        blos = b_field * np.cos(gamma_rad)
        spinor_atm_dict[f"blos_tau_{tau}"] = blos
    return spinor_atm_dict

def plot_per_tau_metrics(pred, true, tau_idx, tau_val, tau_str, quantities, fname, w_str, tau_imshow_dir, tau_scatter_dir, tau_hist_dir, rrmse_arr, pearson_arr, ssim_arr):
    # Imshow comparison for each parameter
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for i in range(3):
        cmap = quantities[i][-1]
        # Predicted
        vmin_pred, vmax_pred = np.quantile(pred[..., tau_idx, i], [0.05, 0.95])
        im_pred = ax[0, i].imshow(pred[..., tau_idx, i], cmap=cmap, vmin=vmin_pred, vmax=vmax_pred)
        ax[0, i].set_title(f"Predicted {quantities[i][1]}")
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        fig.colorbar(im_pred, ax=ax[0, i], fraction=0.046, pad=0.04)
        ax[0, i].text(0.02, 0.95, f"$\\tau={tau_val:.2f}$", transform=ax[0, i].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), verticalalignment='top', horizontalalignment='left')
        # True
        vmin_true, vmax_true = np.quantile(true[..., tau_idx, i], [0.05, 0.95])
        im_true = ax[1, i].imshow(true[..., tau_idx, i], cmap=cmap, vmin=vmin_true, vmax=vmax_true)
        ax[1, i].set_title(f"True {quantities[i][1]}")
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        fig.colorbar(im_true, ax=ax[1, i], fraction=0.046, pad=0.04)
        ax[1, i].text(0.02, 0.95, f"$\\tau={tau_val:.2f}$", transform=ax[1, i].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), verticalalignment='top', horizontalalignment='left')
        # SSIM calculation
        pred_img = pred[..., tau_idx, i]
        true_img = true[..., tau_idx, i]
        pred_norm = (pred_img - vmin_pred) / (vmax_pred - vmin_pred + 1e-8)
        true_norm = (true_img - vmin_true) / (vmax_true - vmin_true + 1e-8)
        ssim_val = ssim(true_norm, pred_norm, data_range=1.0)
        ssim_arr[tau_idx, i] = ssim_val
    plt.tight_layout()
    fig.savefig(tau_imshow_dir / f"{fname}_atm_params_comparison_{tau_str}_w_{w_str}.png")
    plt.close(fig)
    # Scatter plots for each parameter
    for i in range(3):
        true_flat = true[..., tau_idx, i].flatten()
        pred_flat = pred[..., tau_idx, i].flatten()
        pearson_corr, _ = scipy.stats.pearsonr(true_flat, pred_flat)
        pearson_arr[tau_idx, i] = pearson_corr
        plt.figure(figsize=(5, 5))
        plt.scatter(true_flat, pred_flat, s=1, alpha=0.5)
        plt.xlabel(f"True {quantities[i][1]} [{quantities[i][2]}]")
        plt.ylabel(f"Predicted {quantities[i][1]} [{quantities[i][2]}]")
        plt.title(f"{quantities[i][1]} (Pearson r = {pearson_corr:.3f})")
        plt.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], 'r--', lw=1)
        plt.text(0.02, 0.95, f"$\\tau={tau_val:.2f}$", transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), verticalalignment='top', horizontalalignment='left')
        plt.tight_layout()
        if i == 1:
            scatter_name = f"v_{tau_str}_w_{w_str}.png"
        elif i == 2:
            scatter_name = f"B_{tau_str}_w_{w_str}.png"
        else:
            scatter_name = f"{quantities[i][1].replace('$','')}_{tau_str}_w_{w_str}.png"
        plt.savefig(tau_scatter_dir / f"{fname}_scatter_{scatter_name}")
        plt.close()
    # Histogram comparison for each parameter
    for i in range(3):
        true_flat = true[..., tau_idx, i].flatten()
        pred_flat = pred[..., tau_idx, i].flatten()
        rrmse = np.sqrt(np.mean((pred_flat - true_flat) ** 2)) / (np.mean(np.abs(true_flat)) + 1e-8)
        rrmse_arr[tau_idx, i] = rrmse
        plt.figure(figsize=(6, 4))
        plt.hist(true_flat, bins=100, alpha=0.5, label="True", color="blue")
        plt.hist(pred_flat, bins=100, alpha=0.5, label="Predicted", color="orange")
        plt.xlabel(f"{quantities[i][1]} [{quantities[i][2]}]")
        plt.ylabel("Count")
        plt.title(f"{quantities[i][1]} (RRMSE = {rrmse:.3f})")
        plt.legend()
        plt.text(0.02, 0.95, f"$\\tau={tau_val:.2f}$", transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), verticalalignment='top', horizontalalignment='left')
        plt.tight_layout()
        if i == 1:
            hist_name = f"v_{tau_str}_w_{w_str}.png"
        elif i == 2:
            hist_name = f"B_{tau_str}_w_{w_str}.png"
        else:
            hist_name = f"{quantities[i][1].replace('$','')}_{tau_str}_w_{w_str}.png"
        plt.savefig(tau_hist_dir / f"{fname}_hist_{hist_name}")
        plt.close()

def plot_model_summary_metrics(tau_vals, rrmse_arr, pearson_arr, ssim_arr, quantities, fname, w_str, hist_dir, scatter_dir, imshow_dir):
    # RRMSE vs optical depth
    plt.figure(figsize=(7, 5))
    for i in range(3):
        plt.plot(tau_vals, rrmse_arr[:, i], label=quantities[i][1])
    plt.xlabel("Optical Depth (log τ)")
    plt.ylabel("RRMSE")
    plt.title(f"RRMSE vs Optical Depth ({fname}, w={w_str})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_dir / f"{fname}_RRMSE_vs_tau_w_{w_str}.png")
    plt.close()
    # Pearson correlation vs optical depth
    plt.figure(figsize=(7, 5))
    for i in range(3):
        plt.plot(tau_vals, pearson_arr[:, i], label=quantities[i][1])
    plt.xlabel("Optical Depth (log τ)")
    plt.ylabel("Pearson Correlation")
    plt.title(f"Pearson Correlation vs Optical Depth ({fname}, w={w_str})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_dir / f"{fname}_Pearson_vs_tau_w_{w_str}.png")
    plt.close()
    # SSIM vs optical depth
    plt.figure(figsize=(7, 5))
    for i in range(3):
        plt.plot(tau_vals, ssim_arr[:, i], label=quantities[i][1])
    plt.xlabel("Optical Depth (log τ)")
    plt.ylabel("SSIM")
    plt.title(f"SSIM vs Optical Depth ({fname}, w={w_str})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(imshow_dir / f"{fname}_SSIM_vs_tau_w_{w_str}.png")
    plt.close()

def plot_overall_summary_metrics(all_rrmse, all_pearson, all_ssim, all_wstr, all_tau_vals, quantities, hist_dir, scatter_dir, imshow_dir):
    metrics = [
        ("RRMSE", all_rrmse, hist_dir, "RRMSE", "RRMSE"),
        ("Pearson Correlation", all_pearson, scatter_dir, "Pearson Correlation", "Pearson"),
        ("SSIM", all_ssim, imshow_dir, "SSIM", "SSIM"),
    ]
    for i in range(3):
        for metric_name, all_metric, out_dir, ylabel, fname_prefix in metrics:
            plt.figure(figsize=(7, 5))
            for arr, w_str in zip(all_metric, all_wstr):
                vals = arr[:, i]
                plt.plot(all_tau_vals, vals, marker='o', label=f"w={w_str}")
            plt.xlabel("Optical Depth (log τ)")
            plt.ylabel(ylabel)
            plt.title(f"{metric_name} vs Optical Depth for {quantities[i][1]}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{quantities[i][1].replace('$','')}_{fname_prefix}_vs_tau_allw.png")
            plt.close()
            
    for i in range(3):
        for metric_name, all_metric, out_dir, ylabel, fname_prefix in metrics:
            plt.figure(figsize=(7, 5))
            for arr, w_str in zip(all_metric, all_wstr):
                vals = arr[:, i]
                plt.plot(all_tau_vals, vals, marker='o', label=f"w={w_str}")
            plt.xlabel("Optical Depth (log τ)")
            plt.ylabel(ylabel)
            plt.title(f"{metric_name} vs Optical Depth for {quantities[i][1]}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{quantities[i][1].replace('$','')}_{fname_prefix}_vs_tau_allw.png")
            plt.close()
            plt.close()
