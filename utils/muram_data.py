import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from scipy.ndimage import convolve1d
from scipy.interpolate import RegularGridInterpolator, interp1d
from utils.hinode_wavelengths import hinode_wavelength_grid
from scipy.integrate import cumulative_trapezoid
import torch
from torch.utils.data import Dataset

from utils.normalizer import MhdNormalizer, StokesNormalizer


class MhdData:
    """
    Class for loading, processing, and visualizing MURAM MHD simulation data.
    
    Handles:
    - Loading simulation steps
    - Computing optical depth stratification
    - Remapping to optical depth coordinates
    - Visualization (surfaces and slices)
    """
    
    def __init__(self, data_path: str, nx: int = 480, ny: int = 480, nz: int = 256):
        """
        Initialize MURAM MHD data processor.
        
        Parameters
        ----------
        data_path : str
            Path to the MURAM simulation data directory
        nx, ny, nz : int
            Grid dimensions
        """
        self.data_path = Path(data_path)
        self.nx = nx
        self.ny = ny
        self.nz = nz  # Original grid z dimension
        self.z_max: Optional[int] = None  # Trimmed z dimension (if z_max was applied during load_step)
        
        # Data containers
        self.data: Dict[str, np.ndarray] = {}
        self.logtau: Optional[np.ndarray] = None
        self.od_data: Dict[str, np.ndarray] = {}
        
        # Reference layers
        self.z0: Optional[int] = None  # Reference height for T ~ 5780 K
        self.height_levels: Optional[np.ndarray] = None
        
        # Opacity interpolator
        self.kappa_interp: Optional[RegularGridInterpolator] = None
    
    @staticmethod
    def to_value(quantity, target_unit):
        """
        Extract numeric value from an Astropy Quantity with explicit unit conversion.
        If quantity is already numeric (no .to() method), return as-is.
        
        Parameters
        ----------
        quantity : Quantity or float
            The value to extract.
        target_unit : Quantity
            The unit to convert to (e.g., u.cm).
        
        Returns
        -------
        float
            The numeric value in the target unit.
        """
        if hasattr(quantity, 'to'):
            return quantity.to(target_unit).value
        return quantity
        
    def load_step(self, step: Union[int, str], z_max: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Load a simulation step.
        
        Parameters
        ----------
        step : int or str
            Step number (e.g., 52 or '052' or '052000')
        z_max : int, optional
            Keep only z indices 0..z_max-1 (drop top layers)
            
        Returns
        -------
        dict
            Dictionary with arrays: T, P, rho, Vz, Bz
        """
        # Normalize step -> suffix "NNN000"
        if isinstance(step, int):
            suffix = f"{step:03d}000"
        else:
            s = str(step)
            if s.isdigit() and len(s) == 3:
                suffix = f"{int(s):03d}000"
            elif s.isdigit() and len(s) == 6:
                suffix = s
            else:
                suffix = s
        
        def _path(name):
            return self.data_path / f"{name}.{suffix}"
        
        # Check required files
        required = ["eos", "result_0", "result_2", "result_6"]
        missing = [r for r in required if not _path(r).exists()]
        if missing:
            raise FileNotFoundError(f"Missing files for step {suffix}: {missing}")
        
        print(f"Loading step {suffix}")
        
        # Read EOS (2, nx, nz, ny)
        mtpr = np.fromfile(str(_path("eos")), dtype=np.float32)
        mtpr = mtpr.reshape((2, self.nx, self.nz, self.ny), order="C")
        mtpr = mtpr.transpose(0, 1, 3, 2)
        T = mtpr[0, :, :, :]
        P = mtpr[1, :, :, :]
        
        # Helper to read 3D fields (nx, nz, ny)
        def _read3(name):
            arr = np.fromfile(str(_path(name)), dtype=np.float32)
            arr = arr.reshape((self.nx, self.nz, self.ny), order="C")
            return arr.transpose(0, 2, 1)
        
        rho = _read3("result_0")
        rho_Vz = _read3("result_2")
        Bz = _read3("result_6")
        
        # Derived fields
        Vz = rho_Vz / rho / 1e5  # convert to km/s
        coef = np.sqrt(4.0 * 3.14159265)
        Bz = Bz * coef
        
        # Optionally trim top z layers
        # Keep self.nz unchanged for file I/O. Track trimmed size in self.z_max.
        if z_max is not None:
            if not (0 < z_max <= self.nz):
                raise ValueError(f"z_max must be in 1..{self.nz}, got {z_max}")
            T = T[:, :, :z_max]
            P = P[:, :, :z_max]
            rho = rho[:, :, :z_max]
            Vz = Vz[:, :, :z_max]
            Bz = Bz[:, :, :z_max]
            self.z_max = z_max
            print(f"  Trimmed z axis to first {z_max} layers (0..{z_max-1})")
        else:
            self.z_max = self.nz
        
        self.data = {
            "T": T*u.K,
            "P": P*u.dyn/u.cm**2,
            "rho": rho*u.g/u.cm**3,
            "Vz": Vz*u.km/u.s,
            "Bz": Bz*u.G,
        }
        
        # Compute reference layer (z0) and height levels
        self._compute_reference_layer()
        
        print("  Done.")
        return self.data
    
    def _compute_reference_layer(self, height_step_km: float = 10.0):
        """Compute reference z-layer where T ~ 5780 K and height levels."""
        if "T" not in self.data:
            raise ValueError("Temperature data not loaded")
        
        mean_T_z = np.nanmean(self.data["T"], axis=(0, 1))
        self.z0 = int(np.argmin(np.abs(mean_T_z - 5780.0*u.K)))
        # Use actual data z-size (self.z_max) for height levels, not original nz
        nz_actual = self.data["T"].shape[2] if "T" in self.data else self.z_max
        self.height_levels = np.arange(0, nz_actual, 1) * height_step_km - self.z0 * height_step_km
        print(f"  Reference layer (T ~ 5780 K) at z-index {self.z0}")
    
    def load_opacity_table(self, kappa_path: str):
        """
        Load and prepare opacity interpolator.
        
        Parameters
        ----------
        kappa_path : str
            Path to kappa.0.dat file
        """
        # Temperature and pressure grid from opacity table
        tab_T = np.array([3.32, 3.34, 3.36, 3.38, 3.40, 3.42, 3.44, 3.46, 3.48, 3.50,
                          3.52, 3.54, 3.56, 3.58, 3.60, 3.62, 3.64, 3.66, 3.68, 3.70,
                          3.73, 3.76, 3.79, 3.82, 3.85, 3.88, 3.91, 3.94, 3.97, 4.00,
                          4.05, 4.10, 4.15, 4.20, 4.25, 4.30, 4.35, 4.40, 4.45, 4.50,
                          4.55, 4.60, 4.65, 4.70, 4.75, 4.80, 4.85, 4.90, 4.95, 5.00,
                          5.05, 5.10, 5.15, 5.20, 5.25, 5.30])  # log(T) in K
        
        tab_p = np.array([-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5,
                          3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.])  # log(P) in dyne/cm^2
        
        df_kappa = pd.read_csv(kappa_path, sep=r'\s+', header=None)
        df_kappa.columns = ["Temperature index", "Pressure index", "Opacity value"]
        
        temp_indices = df_kappa["Temperature index"].unique()
        press_indices = df_kappa["Pressure index"].unique()
        opacity_values = df_kappa.pivot(
            index="Pressure index",
            columns="Temperature index",
            values="Opacity value"
        ).values
        
        Tk = tab_T[temp_indices]
        Pk = tab_p[press_indices]
        K = opacity_values * u.cm**2 / u.g  # Rosseland mean opacity in cm²/g
        
        self.kappa_interp = RegularGridInterpolator((Pk, Tk), K, method="linear")
        print(f"Opacity interpolator loaded from {kappa_path}")
        print(f"  T range: [{Tk.min():.3f}, {Tk.max():.3f}] (log10 K)")
        print(f"  P range: [{Pk.min():.3f}, {Pk.max():.3f}] (log10 dyne/cm²)")
    
    def compute_optical_depth(self, dz=10*u.km, baseline: float = 1e-5) -> np.ndarray:
        """
        Compute log10(optical depth) for loaded data.
        
        Parameters
        ----------
        dz : Quantity or float
            Height step. Can be a Quantity (e.g., 10*u.km) or numeric value in cm.
            Default is 10 km. Will be converted to cm internally.
        baseline : float
            Minimum tau value for clipping
            
        Returns
        -------
        np.ndarray
            log10(tau) array with shape (nx, ny, nz)
        """
        if self.kappa_interp is None:
            raise ValueError("Opacity interpolator not loaded. Call load_opacity_table() first.")
        
        if "T" not in self.data or "P" not in self.data or "rho" not in self.data:
            raise ValueError("T, P, and rho must be loaded")
        
        print("Computing optical depth...")
        
        # Convert dz to cm if it's a Quantity
        dz_cm = self.to_value(dz, u.cm)
        
        # Prepare T and P in log10, clipped to interpolation bounds
        T_log = np.log10(self.data["T"].value)
        P_log = np.log10(self.data["P"].value)
        
        # Get interpolator bounds
        Pk_min, Pk_max = self.kappa_interp.grid[0].min(), self.kappa_interp.grid[0].max()
        Tk_min, Tk_max = self.kappa_interp.grid[1].min(), self.kappa_interp.grid[1].max()
        
        T_log = np.clip(T_log, Tk_min + 1e-5, Tk_max - 1e-5)
        P_log = np.clip(P_log, Pk_min + 1e-5, Pk_max - 1e-5)
        
        # Stack for interpolation
        PT_log = np.stack((P_log.flatten(), T_log.flatten()), axis=-1)
        
        # Interpolate opacity and multiply by density with units
        kappa_rho_interp = self.kappa_interp(PT_log)
        kappa_rho_interp = kappa_rho_interp.reshape(self.data["T"].shape)*(u.cm**2/u.g)
        # κ*ρ: (cm²/g) * (g/cm³) = cm⁻¹ (dimensionless for optical depth integration)
        # Extract numeric value for numerical integration
        kappa_rho = (kappa_rho_interp * self.data["rho"]).to(u.cm**-1).value
        
        # Integrate from top to bottom (reverse z, integrate, reverse back)
        kappa_rev = kappa_rho[:, :, ::-1]
        tau_rev = cumulative_trapezoid(kappa_rev, dx=dz_cm, axis=2, initial=0.0)
        tau = tau_rev[:, :, ::-1]
        tau = np.clip(tau, a_min=1e-30, a_max=None)
        
        # Check for non-positive values
        nonpos_mask = tau <= 0
        if np.any(nonpos_mask):
            cnt = int(np.count_nonzero(nonpos_mask))
            raise ValueError(f"Non-positive tau encountered: {cnt} elements <= 0")
        
        self.logtau = np.log10(tau)
        print("  Optical depth computed.")
        return self.logtau
    
    def remap_to_optical_depth(self, new_logtau: np.ndarray, 
                                quantities: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Remap quantities from geometric height to optical depth coordinates.
        
        Parameters
        ----------
        new_logtau : np.ndarray
            New optical depth grid (e.g., np.arange(-2.0, 0.1, 0.1))
        quantities : list of str, optional
            Quantities to remap (default: ["T", "Vz", "Bz"])
            
        Returns
        -------
        dict
            Remapped quantities with shape (nx, ny, n_logtau)
        """
        if self.logtau is None:
            raise ValueError("Optical depth not computed. Call compute_optical_depth() first.")
        
        if quantities is None:
            quantities = ["T", "Vz", "Bz"]
        
        n_logtau = new_logtau.shape[0]
        self.od_data = {}
        
        print(f"Remapping to optical depth coordinates ({n_logtau} levels)...")
        
        for q in quantities:
            if q not in self.data:
                print(f"  Skipping {q}: not in loaded data")
                continue
            
            print(f"  Processing {q}")
            geom_atm = self.data[q]
            new_quantity = np.zeros((self.nx, self.ny, n_logtau))
            
            for ix in range(self.nx):
                for iy in range(self.ny):
                    mapper = interp1d(
                        x=self.logtau[ix, iy, :],
                        y=geom_atm.value[ix, iy, :],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate"
                    )
                    new_quantity[ix, iy, :] = mapper(new_logtau)
            
            self.od_data[q] = new_quantity*u.Unit(geom_atm.unit)
        
        print("  Remapping complete.")
        return self.od_data
    
    def plot_surfaces(self, z_idx: int, data_source: str = "geometric",
                      zero_height: bool = True, pixel_size_km: float = 25,
                      quantities: Optional[List[str]] = None,
                      second_plot: str = "profile",
                      default_cmap: str = "viridis",
                      limit_values: Optional[Dict[str, Tuple[float, float]]] = None,
                      save_dir: Optional[str] = None, dpi: int = 150,
                      return_figs: bool = False):
        """
        Plot 2D surfaces and vertical profiles or histograms for each quantity.
        
        Parameters
        ----------
        z_idx : int
            Z-index for surface plot
        data_source : str
            "geometric" or "optical_depth"
        zero_height : bool
            Show zero height marker (T ~ 5780 K)
        pixel_size_km : float
            Pixel size for scale bar
        quantities : list of str, optional
            Quantities to plot
        second_plot : str
            Type of second plot: "profile" or "histogram"
        default_cmap : str
            Default colormap
        limit_values : dict, optional
            {quantity: (vmin, vmax)} for color limits
        save_dir : str, optional
            Directory to save figures
        dpi : int
            Figure DPI
        return_figs : bool
            If True, return list of (quantity, figure) tuples
            
        Returns
        -------
        list of (str, Figure) or None
        """
        if data_source == "geometric":
            data = self.data
            height_levels = self.height_levels
        elif data_source == "optical_depth":
            data = self.od_data
            height_levels = None  # Will be set from caller
        else:
            raise ValueError("data_source must be 'geometric' or 'optical_depth'")
        
        if quantities is None:
            quantities = list(data.keys())
        
        cmap_map = {
            "T": "hot", "P": "summer", "rho": "spring",
            "Bx": "PRGn", "By": "PRGn", "Bz": "PiYG",
            "Vx": "managua", "Vy": "managua", "Vz": "bwr_r",
        }
        
        latex_map = {
            "T": r"$T$", "P": r"$P$", "rho": r"$\rho$",
            "Vx": r"$v_x$", "Vy": r"$v_y$", "Vz": r"$v_z$",
            "Bx": r"$B_x$", "By": r"$B_y$", "Bz": r"$B_z$",
        }
        
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = None
        
        figs = [] if return_figs else None
        
        for q in quantities:
            if q not in data:
                print(f"  Skipping {q}: not in data")
                continue
            
            arr = data[q]
            q_unit = arr.unit.to_string("latex") if hasattr(arr, 'unit') else ""
            arr = arr.value if hasattr(arr, 'value') else arr
            if arr.ndim != 3:
                print(f"  Skipping {q}: not 3D")
                continue
            
            surf = arr[:, :, z_idx]
            profile = np.nanmean(arr, axis=(0, 1))
            
            cmap_name = cmap_map.get(q, default_cmap)
            cmap = plt.get_cmap(cmap_name) if cmap_name in plt.colormaps() else plt.get_cmap(default_cmap)
            
            # Determine color limits
            if limit_values and q in limit_values:
                vmin, vmax = limit_values[q]
            elif q in ["Bx", "By", "Bz", "Vz", "Vx", "Vy"]:
                vmin, vmax = np.quantile(surf, [0.01, 0.99])
                maxabs = max(abs(vmin), abs(vmax))
                vmin, vmax = -maxabs, maxabs
            else:
                vmin, vmax = None, None
            
            q_label = latex_map.get(q, q)
            
            # Log transform for rho and P
            if q in ["rho", "P"]:
                profile = np.log10(profile)
                surf = np.log10(surf)
                q_label = q_label + " (log10)"
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                
                im = axs[0].imshow(surf.T, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
                axs[0].set_title(rf"{q_label} surface (z={z_idx})", fontsize=12)
                axs[0].set_axis_off()
                
                scalebar = ScaleBar(pixel_size_km, 'km', length_fraction=0.25, location='lower left', pad=0.5)
                axs[0].add_artist(scalebar)
                plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04, label=f"{q_unit}")
                
                if second_plot == "profile":
                    line_color = cmap(0.6)
                    
                    if height_levels is not None:
                        axs[1].plot(height_levels, profile, "-o", ms=3, color=line_color)
                        if self.z0 is not None and zero_height and data_source == "geometric":
                            axs[1].axvline(z_idx * 10 - self.z0 * 10, color="gray", lw=1, ls="--")
                        axs[1].set_xlabel("Height (km)")
                    else:
                        x_vals = np.arange(len(profile))
                        axs[1].plot(x_vals, profile, "-o", ms=3, color=line_color)
                        axs[1].set_xlabel("Z-index")
                    
                    if zero_height and data_source == "geometric":
                        y_min, y_max = axs[1].get_ylim()
                        y_mid = 0.5 * (y_min + y_max)
                        axs[1].text(0, y_mid, r"$T \approx 5780$ K",
                                    color='red', fontsize=9, ha='center', va='center',
                                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                        axs[1].axvline(0, color='red', lw=1.5, ls='-')
                    
                    axs[1].set_ylabel(f"mean {q_label} (x,y)")
                    axs[1].set_title(rf"{q_label} profile along z", fontsize=12)
                
                elif second_plot == "histogram":
                    axs[1].hist(surf.ravel(), bins=50, color="black", alpha=0.7)
                    axs[1].set_xlabel(f"{q_label}")
                    axs[1].set_ylabel("Counts")
                    axs[1].set_title(rf"{q_label} histogram at z={z_idx}", fontsize=12)
                
                else:
                    raise ValueError(f"Unknown second_plot option: {second_plot}. Must be 'profile' or 'histogram'")
                
                plt.tight_layout()
                
                if return_figs:
                    figs.append((q, fig))
                else:
                    if save_path is not None:
                        fname = f"{q}_surface_z{z_idx}.png"
                        fig.savefig(save_path / fname, dpi=dpi, bbox_inches='tight')
                        print(f"Saved: {save_path / fname}")
                        plt.close(fig)
                    else:
                        plt.show()
                        plt.close(fig)
        
        if return_figs:
            return figs
    
    def plot_slices(self, x_idx: int, pixel_size_km: float = 25,
                    quantities: Optional[List[str]] = None,
                    default_cmap: str = "viridis", dz_km: float = 10,
                    figsize: Tuple[int, int] = (7, 8),
                    save_dir: Optional[str] = None, dpi: int = 150,
                    return_figs: bool = False):
        """
        Plot vertical slices (y, z) with optical depth contour.
        
        Parameters
        ----------
        x_idx : int
            X-index for slice
        pixel_size_km : float
            Pixel size for scale bar
        quantities : list of str, optional
            Quantities to plot
        default_cmap : str
            Default colormap
        dz_km : float
            Height step in km
        figsize : tuple
            Figure size
        save_dir : str, optional
            Directory to save figures
        dpi : int
            Figure DPI
        return_figs : bool
            If True, return list of (quantity, figure) tuples
            
        Returns
        -------
        list of (str, Figure) or None
        """
        if quantities is None:
            quantities = list(self.data.keys())
        
        cmap_map = {
            "T": "hot", "P": "summer", "rho": "spring",
            "Bx": "PRGn", "By": "PRGn", "Bz": "PiYG",
            "Vx": "managua", "Vy": "managua", "Vz": "bwr_r",
        }
        
        latex_map = {
            "T": r"$T$", "P": r"$P$", "rho": r"$\rho$",
            "Vx": r"$v_x$", "Vy": r"$v_y$", "Vz": r"$v_z$",
            "Bx": r"$B_x$", "By": r"$B_y$", "Bz": r"$B_z$",
        }
        
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = None
        
        figs = [] if return_figs else None
        
        for q in quantities:
            if q not in self.data:
                print(f"  Skipping {q}: not in data")
                continue
            
            arr = self.data[q]
            q_unit = arr.unit.to_string("latex") if hasattr(arr, 'unit') else ""
            arr = arr.value if hasattr(arr, 'value') else arr
            if arr.ndim != 3 or not (0 <= x_idx < arr.shape[0]):
                print(f"  Skipping {q}: invalid shape or x_idx")
                continue
            
            slice_q = arr[x_idx, :, :]
            
            display_slice = slice_q.copy()
            q_label = latex_map.get(q, q)
            
            if q in ["rho", "P"]:
                display_slice = np.log10(np.clip(display_slice, 1e-40, None))
                q_label = q_label + " (log10)"
            
            cmap_name = cmap_map.get(q, default_cmap)
            cmap = plt.get_cmap(cmap_name) if cmap_name in plt.colormaps() else plt.get_cmap(default_cmap)
            
            if q in ["Bx", "By", "Bz", "Vz", "Vx", "Vy"]:
                vmin, vmax = np.quantile(display_slice, [0.01, 0.99])
                maxabs = max(abs(vmin), abs(vmax))
                vmin, vmax = -maxabs, maxabs
            else:
                vmin, vmax = None, None
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                fig, ax = plt.subplots(figsize=figsize)
                im = ax.imshow(display_slice.T, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
                
                ax.set_xticks([])
                
                scalebar = ScaleBar(pixel_size_km, 'km', length_fraction=0.25, location='lower left', pad=0.5)
                ax.add_artist(scalebar)
                
                # Overlay logtau = 0 contour
                if self.logtau is not None and self.logtau.shape == arr.shape:
                    tau_slice = self.logtau[x_idx, :, :]
                    ys = np.arange(slice_q.shape[0])
                    zs = np.arange(slice_q.shape[1])
                    CS = ax.contour(ys, zs, tau_slice.T, levels=[0.0], colors='white', linewidths=1.5)
                    
                    if len(CS.allsegs[0]) != 0:
                        segs = CS.allsegs[0]
                        longest = max(segs, key=lambda s: s.shape[0])
                        mid = longest[len(longest) // 2]
                        x_lab, y_lab = mid[0], mid[1]
                        ax.text(x_lab + 1, y_lab + 10, r"$\log \tau = 0$",
                                color="white", fontsize=10, ha="left", va="bottom",
                                bbox=dict(facecolor="black", alpha=0.4, boxstyle="round,pad=0.2"))
                
                # Reference layer marker
                if self.z0 is not None and 0 <= self.z0 < slice_q.shape[1]:
                    ax.axhline(self.z0, color='yellow', lw=1.5, ls='--')
                    ax.text(2, self.z0 + 2, r'$T\approx 5780$', color='yellow', fontsize=9,
                            bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.2'))
                
                # Height ticks
                n_z = slice_q.shape[1]
                n_ticks = 7
                tick_positions = np.linspace(0, n_z - 1, n_ticks)
                
                if self.height_levels is not None and len(self.height_levels) == n_z:
                    tick_labels = [f"{int(self.height_levels[int(p)])}" for p in tick_positions]
                else:
                    tick_labels = [f"{int((p - self.z0) * dz_km):+d}" for p in tick_positions]
                
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels)
                ax.set_ylabel("Height (km)")
                
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"{q_unit}")
                ax.set_title(f"{q_label} slice at x={x_idx}")
                plt.tight_layout()
                
                if return_figs:
                    figs.append((q, fig))
                else:
                    if save_path is not None:
                        fname = f"{q}_slice_x{x_idx}.png"
                        fig.savefig(save_path / fname, dpi=dpi, bbox_inches='tight')
                        print(f"Saved: {save_path / fname}")
                        plt.close(fig)
                    else:
                        plt.show()
                        plt.close(fig)
        
        if return_figs:
            return figs
    
    def plot_medians_vs_logtau(self, new_logtau: np.ndarray,
                                quantities: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (5, 4),
                                save_dir: Optional[str] = None,
                                dpi: int = 150,
                                return_figs: bool = False):
        """
        Plot median values of optical depth data versus log(tau) levels.
        
        Parameters
        ----------
        new_logtau : np.ndarray
            Optical depth grid values
        quantities : list of str, optional
            Quantities to plot (default: all in od_data)
        figsize : tuple
            Figure size
        save_dir : str, optional
            Directory to save figures
        dpi : int
            Figure DPI
        return_figs : bool
            If True, return list of (quantity, figure) tuples
            
        Returns
        -------
        list of (str, Figure) or None
        """
        if not self.od_data:
            raise ValueError("No optical depth data available. Call remap_to_optical_depth() first.")
        
        if quantities is None:
            quantities = list(self.od_data.keys())
        
        latex_map = {
            "T": r"$T$",
            "P": r"$P$",
            "rho": r"$\rho$",
            "Bx": r"$B_x$",
            "By": r"$B_y$",
            "Bz": r"$B_z$",
            "Vx": r"$v_x$",
            "Vy": r"$v_y$",
            "Vz": r"$v_z$"
        }
        
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = None
        
        figs = [] if return_figs else None
        
        for q in quantities:
            if q not in self.od_data:
                print(f"  Skipping {q}: not in optical depth data")
                continue
            
            data_q = self.od_data[q]
            q_unit = data_q.unit.to_string("latex") if hasattr(data_q, 'unit') else ""
            median_values = np.median(data_q, axis=(0, 1))
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(new_logtau, median_values, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel(r'$\log \tau$', fontsize=12)
            ax.set_ylabel(f'Median {latex_map.get(q, q)} ({q_unit})', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Median {latex_map.get(q, q)} vs Optical Depth', fontsize=12)
            plt.tight_layout()
            
            if return_figs:
                figs.append((q, fig))
            else:
                if save_path is not None:
                    fname = f"{q}_median_vs_logtau.png"
                    fig.savefig(save_path / fname, dpi=dpi, bbox_inches='tight')
                    print(f"Saved: {save_path / fname}")
                    plt.close(fig)
                else:
                    plt.show()
                    plt.close(fig)
        
        if return_figs:
            return figs


# ============================================================================
# EXAMPLE USAGE (MHD)
# ============================================================================
"""
from pathlib import Path
import numpy as np

# Initialize
mhd = MhdData(
    data_path="/path/to/muram-simulation/",
    nx=480, ny=480, nz=256
)

# Load a simulation step
mhd.load_step(step=223, z_max=250)

# Load opacity table and compute optical depth
mhd.load_opacity_table(kappa_path="../csv/kappa.0.dat")
mhd.compute_optical_depth(dz=1e6)

# Remap to optical depth coordinates
new_logtau = np.arange(-2.0, 0.1, 0.1)
mhd.remap_to_optical_depth(new_logtau, quantities=["T", "Vz", "Bz"])

# Plot geometric height surfaces
mhd.plot_surfaces(
    z_idx=190,
    data_source="geometric",
    save_dir="./output/surfaces"
)

# Plot vertical slices with optical depth contours
mhd.plot_slices(
    x_idx=120,
    save_dir="./output/slices"
)

# Plot optical depth surfaces
limit_values = {
    "T": (5500, 7000),
    "Vz": (-5, 5),
    "Bz": (-100, 100),
}
mhd.plot_surfaces(
    z_idx=20,
    data_source="optical_depth",
    quantities=["T", "Vz", "Bz"],
    limit_values=limit_values,
    save_dir="./output/surfaces_od"
)

# Plot median values vs optical depth
mhd.plot_medians_vs_logtau(
    new_logtau=new_logtau,
    quantities=["T", "Bz", "Vz"],
    save_dir="./output/medians"
)

# Access processed data
T_data = mhd.data["T"]  # Geometric height data
T_od = mhd.od_data["T"]  # Optical depth data
logtau = mhd.logtau  # Optical depth map
"""

# ============================================================================
# StokesData class
# ============================================================================
class StokesData:
    """
    Class for loading, processing, and visualizing Stokes polarimetric data from MURaM simulations.
    
    Handles:
    - Loading Stokes cubes (I, Q, U, V)
    - Continuum normalization
    - Spectropolarimetry calculations
    - Spectral convolution with LSF
    - Resampling to Hinode/SP wavelengths
    - Optional: Adding realistic Hinode noise
    - Visualization of spectra and polarimetric maps
    
    The final stokes array (self.data) is progressively processed and adapted for Hinode.
    """
    
    def __init__(self, data_dir: str, step: Union[int, str], wavelength_range: Tuple[float, float] = (6300.5, 6303.5), 
                 wavelength_step: float = 0.01):
        self.data_dir = Path(data_dir)
        self.step = self._normalize_step(step)
        self.wl = np.arange(wavelength_range[0], wavelength_range[1], wavelength_step)
        self.data: Dict[str, np.ndarray] = {}
        self.mean_continuum: Optional[np.ndarray] = None
        self.circular_polarization: Optional[np.ndarray] = None
        self.hinode_wl: Optional[np.ndarray] = None
        self.lsf_kernel: Optional[np.ndarray] = None
        self.nx: Optional[int] = None
        self.ny: Optional[int] = None
        self.nwl: Optional[int] = None

    def _normalize_step(self, step: Union[int, str]) -> str:
        if isinstance(step, int):
            return f"{step:03d}000"
        else:
            s = str(step)
            if s.isdigit() and len(s) == 3:
                return f"{int(s):03d}000"
            elif s.isdigit() and len(s) == 6:
                return s
            else:
                return s

    def load_stokes(self) -> Dict[str, np.ndarray]:
        stokes_path = self.data_dir / f"stokes_{self.step}.npy"
        if not stokes_path.exists():
            raise FileNotFoundError(f"Stokes file not found: {stokes_path}")
        print(f"Loading Stokes data from {stokes_path}")
        stokes_cube = np.load(stokes_path)
        self.data = {
            "I": stokes_cube[:, :, :, 0],
            "Q": stokes_cube[:, :, :, 1],
            "U": stokes_cube[:, :, :, 2],
            "V": stokes_cube[:, :, :, 3]
        }
        for key in self.data.keys():
            self.data[key] = np.flip(np.rot90(self.data[key], k=1, axes=(0,1)), axis=0)
        self.nx, self.ny, self.nwl = self.data["I"].shape
        for key in self.data.keys():
            print(f"  {key} shape: {self.data[key].shape}")
        print("  Done.")
        return self.data

    def continuum_normalization(
        self,
        cont_indices: Optional[List[int]] = None,
        fixed_ic: Optional[float] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        if not self.data:
            raise ValueError("Stokes data not loaded. Call load_stokes() first.")
        if cont_indices is None:
            cont_indices = [0, 1, 2, 3]
        cont_indices = np.asarray(cont_indices, dtype=int)
        if cont_indices.ndim != 1 or cont_indices.size == 0:
            raise ValueError("cont_indices must be a 1D non-empty array-like of indices")

        print("Applying continuum normalization...")
        if fixed_ic is None:
            I_c = float(self.data["I"][:, :, cont_indices].mean(axis=2).mean())
            ic_mode = "per-step"
        else:
            I_c = float(fixed_ic)
            if not np.isfinite(I_c) or I_c <= 0:
                raise ValueError(f"fixed_ic must be finite and > 0, got {fixed_ic}")
            ic_mode = "fixed-global"

        for key in self.data.keys():
            self.data[key] = self.data[key] / I_c
        mean_continuum = self.data["I"][:, :, cont_indices].mean(axis=2)
        self.mean_continuum = mean_continuum
        print(f"  Continuum mode: {ic_mode}")
        print(f"  Global continuum used: I_c = {I_c:.6e}")
        print("  Done.")
        return self.data, mean_continuum

    def spectropolarimetry(self, stokes_data: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        if stokes_data is None:
            stokes_data = self.data
        if "V" not in stokes_data or "I" not in stokes_data:
            raise ValueError("Stokes V and I required for circular polarization")
        nwl_points = stokes_data["I"].shape[-1]
        epsilon = 1e-10
        I_safe = np.maximum(stokes_data["I"], epsilon)
        circular_pol = np.sum(np.abs(stokes_data["V"]) / I_safe, axis=-1) / nwl_points
        if np.isnan(circular_pol).any() or np.isinf(circular_pol).any():
            print("  WARNING: NaN or Inf values in circular polarization")
        self.circular_polarization = circular_pol
        return circular_pol

    def rms_calculation(self, data: np.ndarray) -> float:
        return np.sqrt(np.mean((data - data.mean())**2)) / data.mean()

    def load_hinode_lsf(self, lsf_path: str) -> np.ndarray:
        print(f"Loading Hinode LSF from {lsf_path}")
        lsf_data = pd.read_csv(lsf_path, sep=r"\s+", header=None)
        lsf_wl_orig = lsf_data[0].values
        lsf_values_orig = lsf_data[1].values / lsf_data[1].values.sum()
        lsf_interp = interp1d(lsf_wl_orig, lsf_values_orig, kind="cubic")
        lsf_wl_min = np.round(lsf_wl_orig.min() + 0.010, 3)
        lsf_wl_max = np.round(lsf_wl_orig.max() - 0.010, 3)
        step = self.wl[1] - self.wl[0]
        new_kernel_wl = np.arange(lsf_wl_min, lsf_wl_max + step, step)
        new_kernel_values = lsf_interp(new_kernel_wl)
        new_kernel_values = new_kernel_values / new_kernel_values.sum()
        self.lsf_kernel = new_kernel_values
        print(f"  LSF wavelength range: {lsf_wl_min:.3f} - {lsf_wl_max:.3f} Å")
        print(f"  LSF kernel size: {len(self.lsf_kernel)}")
        print("  Done.")
        return self.lsf_kernel

    def apply_spectral_convolution(self) -> Dict[str, np.ndarray]:
        if not self.data:
            raise ValueError("Stokes data not loaded. Call load_stokes() first.")
        if self.lsf_kernel is None:
            raise ValueError("LSF kernel not loaded. Call load_hinode_lsf() first.")
        print("Applying spectral convolution with LSF...")
        for key in self.data.keys():
            self.data[key] = convolve1d(
                self.data[key],
                weights=self.lsf_kernel,
                axis=-1,
                mode='wrap'
            )
            print(f"  {key}: convolved")
        print("  Done.")
        return self.data

    def resample_to_hinode(self, hinode_wl: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Resample the synthesized profiles onto the Hinode/SOT-SP spectral grid.

        The grid defaults to utils.hinode_wavelengths.hinode_wavelength_grid(), i.e.
        it is derived from the MODEST FITS header -- the same axis the real
        observations are loaded on. Pass `hinode_wl` only to override with an
        explicit array.

        This used to build the grid from a hardcoded
        CRVAL1=6302.0/CDELT1=0.0215/CRPIX1=57 triple, which yielded
        [6300.7960, 6303.1825] A while the observations were loaded on
        [6300.8736, 6303.2576] A. That 0.0776 A disagreement (~3.6 spectral
        pixels) made every model trained here read a spurious ~3.7 km/s
        blueshift when applied to real MODEST data.
        """
        if not self.data:
            raise ValueError("No Stokes data available. Call load_stokes() first.")
        if hinode_wl is None:
            hinode_wl = hinode_wavelength_grid()
        self.hinode_wl = np.asarray(hinode_wl, dtype=np.float64)
        NAXIS1 = int(self.hinode_wl.size)

        # Guard the invariant that was silently violated before: the axis training
        # data is resampled onto MUST be the FITS-derived axis the observations are
        # loaded on. Any reintroduced hardcoded grid fails loudly here instead of
        # quietly biasing V_LOS by ~3.7 km/s.
        canonical = hinode_wavelength_grid(NAXIS1)
        assert np.allclose(self.hinode_wl, canonical, atol=1e-9), (
            "Hinode resample grid does not match the MODEST FITS-derived axis.\n"
            f"  got       : [{self.hinode_wl[0]:.4f}, {self.hinode_wl[-1]:.4f}] ({NAXIS1} pts)\n"
            f"  canonical : [{canonical[0]:.4f}, {canonical[-1]:.4f}] ({canonical.size} pts)\n"
            f"  max offset: {np.max(np.abs(self.hinode_wl - canonical)):.4e} A "
            f"(~{3e5 * np.max(np.abs(self.hinode_wl - canonical)) / 6302.0:.3f} km/s of spurious Doppler shift)\n"
            "Get the axis from utils.hinode_wavelengths, never from hardcoded "
            "CRVAL1/CDELT1/CRPIX1 values."
        )
        # The synthesis grid must bracket the target grid, or the cubic interpolation
        # below would extrapolate (silently, for out-of-range points).
        src = np.asarray(self.wl, dtype=np.float64)
        assert src[0] <= self.hinode_wl[0] and src[-1] >= self.hinode_wl[-1], (
            f"Synthesis grid [{src[0]:.4f}, {src[-1]:.4f}] A does not cover the target "
            f"Hinode grid [{self.hinode_wl[0]:.4f}, {self.hinode_wl[-1]:.4f}] A; "
            "resampling would extrapolate. Widen the synthesis wavelength range."
        )

        print(f"Resampling to Hinode/SP wavelengths...")
        print(f"  {NAXIS1} wavelength points from {self.hinode_wl[0]:.4f} to {self.hinode_wl[-1]:.4f} Å")
        for key in self.data.keys():
            stokes_reshaped = self.data[key].reshape(-1, self.data[key].shape[2])
            interp_func = interp1d(self.wl, stokes_reshaped, kind="cubic", axis=1)
            stokes_interp = interp_func(self.hinode_wl).reshape(self.nx, self.ny, NAXIS1)
            self.data[key] = stokes_interp
            print(f"  {key}: resampled to shape {stokes_interp.shape}")
        self.wl = self.hinode_wl
        self.nwl = NAXIS1
        print("  Done.")
        return self.data

    def add_hinode_noise(self, noise_level: float = 5.9e-4, 
                        continuum_intensity: Optional[float] = None) -> Dict[str, np.ndarray]:
        if not self.data:
            raise ValueError("Stokes data not available. Call resample_to_hinode() first.")
        if continuum_intensity is None:
            continuum_intensity = self.data["I"].flatten().mean()
        sigma = noise_level * continuum_intensity
        print(f"Adding Hinode noise...")
        print(f"  Continuum intensity: {continuum_intensity:.6e}")
        print(f"  Noise level: σ = {sigma:.6e}")
        for key in self.data.keys():
            noise = sigma * np.random.randn(self.nx, self.ny, self.nwl)
            self.data[key] = self.data[key] + noise
            print(f"  {key}: noise added")
        print("  Done.")
        return self.data

    def plot_selected_pixel(self, x: int, y: int, 
                           save_dir: Optional[str] = None, dpi: int = 300) -> None:
        if "I" not in self.data:
            raise ValueError("Stokes I not loaded. Call load_stokes() first.")
        fig, ax = plt.subplots(figsize=(5, 5))
        data = self.data["I"][..., 0]
        im = ax.imshow(data, cmap='inferno', origin="lower")
        rect = patches.Rectangle((y - 1.5, x - 1.5), 3*4, 3*4, linewidth=3, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Intensity with Highlighted Pixel")
        plt.tight_layout()
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / "selected_pixel.png", dpi=dpi)
            print(f"Saved: {save_path / 'selected_pixel.png'}")
        else:
            plt.show()
        plt.close(fig)

    def plot_stokes(self, ax, x: int, y: int,
                    wl_index_range: Optional[Tuple[int, int]] = None,
                    scatter: bool = False,
                    **kwargs):
        """Plot Stokes I and V for a given pixel on provided axes."""
        if "I" not in self.data or "V" not in self.data:
            raise ValueError("Stokes I and V must be loaded. Call load_stokes() first.")

        stokes_titles = [r"$I$", r"$V/I_c$"]
        for i, key in enumerate(["I", "V"]):
            wl_to_plot = self.wl
            if wl_index_range is not None:
                wl_to_plot = self.wl[wl_index_range[0]:wl_index_range[1]]
                stokes_to_plot = self.data[key][x, y, wl_index_range[0]:wl_index_range[1]]
            else:
                stokes_to_plot = self.data[key][x, y, :]

            if scatter:
                ax[i].scatter(wl_to_plot, stokes_to_plot, **kwargs)
            else:
                ax[i].plot(wl_to_plot, stokes_to_plot, **kwargs)

            ax[i].set_title(stokes_titles[i])
            ax[i].set_xlabel("Wavelength (Å)")
            ax[i].legend()

        return ax

    def plot_polarizations(
        self,
        continuum_map: Optional[np.ndarray] = None,
        polarization_map: Optional[np.ndarray] = None,
        titles: Tuple[str, str] = ("Continuum", "Circular Pol."),
        rms_values: Optional[Tuple[float, float]] = None,
        quantiles: Tuple[float, float] = (0.05, 0.95),
        save_dir: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Plot continuum and circular polarization maps with optional RMS labels."""
        continuum = continuum_map if continuum_map is not None else self.mean_continuum
        polarization = polarization_map if polarization_map is not None else self.circular_polarization

        if continuum is None or polarization is None:
            raise ValueError("Continuum and polarization data must be available to plot.")

        if rms_values is None:
            rms_values = (
                self.rms_calculation(continuum),
                self.rms_calculation(polarization),
            )

        q_low, q_high = np.quantile(polarization, quantiles)

        fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.3))

        im1 = ax[0].imshow(continuum, cmap="inferno", origin="lower")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        divider1 = make_axes_locatable(ax[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax1)

        im2 = ax[1].imshow(polarization, cmap="PiYG", vmin=q_low, vmax=q_high, origin="lower")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        divider2 = make_axes_locatable(ax[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax2)

        ax[0].set_title(f"{titles[0]} (rms = {rms_values[0]:.2f})")
        ax[1].set_title(f"{titles[1]} (rms = {rms_values[1]:.2f})")

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            outfile = save_path / "polarizations.png"
            fig.savefig(outfile, dpi=dpi)
            print(f"Saved: {outfile}")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)


def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Compute Otsu threshold for finite values using numpy only."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot compute Otsu threshold on empty/non-finite array.")
    if np.allclose(arr.min(), arr.max()):
        return float(arr.min())

    hist, bin_edges = np.histogram(arr, bins=int(n_bins))
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    prob = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    sigma_b2[~np.isfinite(sigma_b2)] = -np.inf
    idx = int(np.argmax(sigma_b2))
    return float(bin_centers[idx])


def build_granulation_polarization_masks(
    mean_continuum: np.ndarray,
    circular_polarization: np.ndarray,
    continuum_threshold: Optional[float] = None,
    polarization_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build granular/intergranular and strong/weak circular-polarization masks.

    Notebook-consistent rules:
    - granular: mean_continuum > Otsu(mean_continuum)
    - weak polarization: circular_polarization < Otsu(circular_polarization)
    """
    mean_continuum = np.asarray(mean_continuum)
    circular_polarization = np.asarray(circular_polarization)

    if mean_continuum.shape != circular_polarization.shape:
        raise ValueError(
            "mean_continuum and circular_polarization must have the same shape. "
            f"Got {mean_continuum.shape} vs {circular_polarization.shape}."
        )

    t_cont = float(continuum_threshold) if continuum_threshold is not None else _otsu_threshold(mean_continuum)
    t_pol = float(polarization_threshold) if polarization_threshold is not None else _otsu_threshold(circular_polarization)

    granular = mean_continuum > t_cont
    intergranular = ~granular
    weak_pol = circular_polarization < t_pol
    strong_pol = ~weak_pol

    masks = {
        "granular_strong": granular & strong_pol,
        "intergranular_strong": intergranular & strong_pol,
        "granular_weak": granular & weak_pol,
        "intergranular_weak": intergranular & weak_pol,
    }
    counts = {k: int(v.sum()) for k, v in masks.items()}

    return {
        "continuum_threshold": t_cont,
        "polarization_threshold": t_pol,
        "masks": masks,
        "counts": counts,
    }


def build_balanced_region_indices(
    region_masks: Dict[str, np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    """Return flattened indices with equal samples in the 4 region masks."""
    required = [
        "granular_strong",
        "intergranular_strong",
        "granular_weak",
        "intergranular_weak",
    ]
    missing = [k for k in required if k not in region_masks]
    if missing:
        raise KeyError(f"Missing region masks: {missing}")

    if rng is None:
        rng = np.random.default_rng()

    idx_by_region = {name: np.flatnonzero(region_masks[name].ravel()) for name in required}
    counts_before = {name: int(idx.size) for name, idx in idx_by_region.items()}
    target = min(counts_before.values()) if counts_before else 0

    if target <= 0:
        raise ValueError(f"At least one region has zero pixels: {counts_before}")

    selected_chunks = []
    for name in required:
        chosen = rng.choice(idx_by_region[name], size=target, replace=False)
        selected_chunks.append(chosen)

    selected = np.concatenate(selected_chunks)
    rng.shuffle(selected)

    counts_after = {name: int(target) for name in required}
    counts_after["total_selected"] = int(selected.size)
    return selected.astype(np.int64), {
        "counts_before": counts_before,
        "counts_after": counts_after,
    }


def build_bz_strength_balanced_indices(
    mhd_data: Dict[str, np.ndarray],
    base_selected_indices: Optional[np.ndarray] = None,
    n_bins: int = 12,
    score_mode: str = "mean_abs",
    tau_idx: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return flattened indices balanced across Bz-strength bins."""
    if "Bz" not in mhd_data:
        raise KeyError("mhd_data must contain a 'Bz' array")

    if rng is None:
        rng = np.random.default_rng()

    bz = np.asarray(mhd_data["Bz"], dtype=np.float32)
    if bz.ndim != 3:
        raise ValueError(f"Expected Bz to be 3D (nx, ny, n_tau), got shape {bz.shape}")

    valid_score_modes = {"mean_abs", "max_abs", "tau_index"}
    if score_mode not in valid_score_modes:
        raise ValueError(f"score_mode must be one of {sorted(valid_score_modes)}, got {score_mode!r}")

    if score_mode == "mean_abs":
        pixel_scores = np.mean(np.abs(bz), axis=2)
    elif score_mode == "max_abs":
        pixel_scores = np.max(np.abs(bz), axis=2)
    else:
        if tau_idx is None:
            tau_idx = bz.shape[2] // 2
        if tau_idx < 0 or tau_idx >= bz.shape[2]:
            raise ValueError(f"tau_idx must be within [0, {bz.shape[2] - 1}], got {tau_idx}")
        pixel_scores = np.abs(bz[:, :, int(tau_idx)])

    score_flat = np.asarray(pixel_scores, dtype=np.float32).ravel()
    if base_selected_indices is None:
        candidate_indices = np.arange(score_flat.size, dtype=np.int64)
    else:
        candidate_indices = np.asarray(base_selected_indices, dtype=np.int64).ravel()

    if candidate_indices.size == 0:
        raise ValueError("base_selected_indices is empty")
    if candidate_indices.min() < 0 or candidate_indices.max() >= score_flat.size:
        raise ValueError("base_selected_indices contains out-of-range values")

    candidate_scores = score_flat[candidate_indices]
    finite_mask = np.isfinite(candidate_scores)
    candidate_indices = candidate_indices[finite_mask]
    candidate_scores = candidate_scores[finite_mask]

    if candidate_indices.size == 0:
        raise ValueError("No finite Bz scores available for balancing")

    score_min = float(np.min(candidate_scores))
    score_max = float(np.max(candidate_scores))
    if np.isclose(score_min, score_max):
        counts_before = {"bin_0": int(candidate_indices.size)}
        return candidate_indices.astype(np.int64), {
            "score_mode": score_mode,
            "tau_idx": None if tau_idx is None else int(tau_idx),
            "n_bins": 1,
            "score_min": score_min,
            "score_max": score_max,
            "bin_edges": [score_min, score_max],
            "counts_before": counts_before,
            "counts_after": {"bin_0": int(candidate_indices.size), "total_selected": int(candidate_indices.size)},
            "total_candidates": int(candidate_indices.size),
        }

    bin_edges = np.linspace(score_min, score_max, int(max(2, n_bins)) + 1, dtype=np.float32)
    bin_ids = np.digitize(candidate_scores, bin_edges[1:-1], right=False)
    bin_ids = np.clip(bin_ids, 0, bin_edges.size - 2)

    counts_before = {
        f"bin_{bin_idx}": int(np.sum(bin_ids == bin_idx))
        for bin_idx in range(bin_edges.size - 1)
    }
    occupied_bins = [count for count in counts_before.values() if count > 0]
    if not occupied_bins:
        raise ValueError("No occupied Bz bins found during balancing")

    target_per_bin = min(occupied_bins)
    selected_chunks = []
    counts_after = {}
    for bin_idx in range(bin_edges.size - 1):
        mask = bin_ids == bin_idx
        bin_indices = candidate_indices[mask]
        bin_count = int(bin_indices.size)
        counts_after[f"bin_{bin_idx}"] = int(min(bin_count, target_per_bin))
        if bin_count == 0:
            continue
        if bin_count > target_per_bin:
            chosen = rng.choice(bin_indices, size=target_per_bin, replace=False)
        else:
            chosen = bin_indices
        selected_chunks.append(chosen)

    if not selected_chunks:
        raise ValueError("Bz balancing produced no selected indices")

    selected = np.concatenate(selected_chunks)
    rng.shuffle(selected)
    counts_after["total_selected"] = int(selected.size)

    return selected.astype(np.int64), {
        "score_mode": score_mode,
        "tau_idx": None if tau_idx is None else int(tau_idx),
        "n_bins": int(bin_edges.size - 1),
        "score_min": score_min,
        "score_max": score_max,
        "bin_edges": [float(v) for v in bin_edges.tolist()],
        "counts_before": counts_before,
        "counts_after": counts_after,
        "total_candidates": int(candidate_indices.size),
        "target_per_bin": int(target_per_bin),
    }


# ============================================================================
# MuramStepDataset class - PyTorch Dataset for training
# ============================================================================
class MuramStepDataset(Dataset):
    """
    PyTorch Dataset for a single MURaM simulation step.
    
    Returns normalized Stokes profiles and MHD targets for each spatial pixel.
    Also provides spatial indices for physics regularization.
    
    Parameters
    ----------
    stokes_data : dict
        Raw Stokes data {'I': (nx, ny, nλ), 'V': (nx, ny, nλ)}
    mhd_data : dict
        Raw MHD data {'T': (nx, ny, nτ), 'Vz': ..., 'Bz': ...}
    stokes_normalizer : StokesNormalizer
        Fitted normalizer for Stokes data
    mhd_normalizer : MhdNormalizer
        Fitted normalizer for MHD data
        
    Returns
    -------
    tuple
        (stokes_input, mhd_targets, spatial_indices)
        - stokes_input: (2, nλ) normalized Stokes I and V
        - mhd_targets: (3 * nτ,) normalized concatenated T, Vz, Bz
        - spatial_indices: (2,) [x, y] pixel coordinates
        
    Example
    -------
    >>> from torch.utils.data import DataLoader
    >>> dataset = MuramStepDataset(stokes_data, mhd_data, stokes_norm, mhd_norm)
    >>> dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    >>> for stokes, mhd, indices in dataloader:
    ...     predictions = model(stokes)
    ...     loss = criterion(predictions, mhd)
    """
    
    def __init__(
        self,
        stokes_data: Dict[str, np.ndarray],
        mhd_data: Dict[str, np.ndarray],
        stokes_normalizer: StokesNormalizer,
        mhd_normalizer: MhdNormalizer,
        selected_flat_indices: Optional[np.ndarray] = None,
        region_sampling_info: Optional[Dict[str, Any]] = None,
        bz_balance_info: Optional[Dict[str, Any]] = None,
    ):
        self.nx, self.ny = stokes_data['I'].shape[:2]
        self.n_pixels_full = self.nx * self.ny
        self.region_sampling_info = region_sampling_info
        self.bz_balance_info = bz_balance_info
        self.hinode_wl = None
        if "hinode_wl" in stokes_data:
            self.hinode_wl = np.asarray(stokes_data["hinode_wl"], dtype=np.float32)
        
        # Normalize data
        norm_stokes = stokes_normalizer.transform(stokes_data)
        norm_mhd = mhd_normalizer.transform(mhd_data)
        
        # Flatten spatial dimensions: (nx, ny, ...) -> (nx*ny, ...)
        # Stokes input: (n_pixels, 2, nλ)
        I_flat = norm_stokes['I'].reshape(self.n_pixels_full, -1)  # (n_pixels, 112)
        V_flat = norm_stokes['V'].reshape(self.n_pixels_full, -1)
        self.stokes_input = np.stack([I_flat, V_flat], axis=1)  # (n_pixels, 2, 112)
        
        # MHD targets: concatenate T, Vz, Bz along feature dimension
        # Each has shape (n_pixels, 21) -> concatenate to (n_pixels, 63)
        T_flat = norm_mhd['T'].reshape(self.n_pixels_full, -1)
        Vz_flat = norm_mhd['Vz'].reshape(self.n_pixels_full, -1)
        Bz_flat = norm_mhd['Bz'].reshape(self.n_pixels_full, -1)
        self.mhd_targets = np.concatenate([T_flat, Vz_flat, Bz_flat], axis=1)
        
        # Store spatial indices for physics regularization
        ix, iy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        self.spatial_indices = np.stack([ix.ravel(), iy.ravel()], axis=1)  # (n_pixels_full, 2)

        if selected_flat_indices is not None:
            selected_flat_indices = np.asarray(selected_flat_indices, dtype=np.int64)
            if selected_flat_indices.ndim != 1:
                raise ValueError("selected_flat_indices must be 1D")
            if selected_flat_indices.size == 0:
                raise ValueError("selected_flat_indices is empty")
            if selected_flat_indices.min() < 0 or selected_flat_indices.max() >= self.n_pixels_full:
                raise ValueError(
                    "selected_flat_indices contains out-of-range values. "
                    f"Valid range is [0, {self.n_pixels_full - 1}]"
                )

            self.stokes_input = self.stokes_input[selected_flat_indices]
            self.mhd_targets = self.mhd_targets[selected_flat_indices]
            self.spatial_indices = self.spatial_indices[selected_flat_indices]

        self.n_pixels = int(self.stokes_input.shape[0])
        
    def __len__(self):
        return self.n_pixels
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.stokes_input[idx]).float(),
            torch.from_numpy(self.mhd_targets[idx]).float(),
            torch.from_numpy(self.spatial_indices[idx]).long(),
        )


__all__ = [
    "MhdData",
    "StokesData",
    "MuramStepDataset",
    "build_granulation_polarization_masks",
    "build_balanced_region_indices",
    "build_bz_strength_balanced_indices",
]
