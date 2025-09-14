import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision import transforms
from astropy.io import fits
from astropy.constants import c, e, m_e
from astropy import units as u
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import simpson
from scipy.ndimage import convolve1d
from scipy.signal import fftconvolve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error


class DataCharger:
    """
    Class to charge and process MuRAM simulation data for neural network training.
    Handles multiple files and applies all Hinode/SOT-SP adaptations.
    """
    
    def __init__(self, data_path: str, filenames: list, nx: int = 480, ny: int = 480, nz: int = 256):
        """
        Initialize the DataCharger.
        
        Args:
            data_path (str): Path to the data directory
            filenames (list): List of filename strings to process
            nx (int): Width axis dimension
            ny (int): Height axis dimension  
            nz (int): Geometrical height axis dimension
        """
        self.data_path = Path(data_path)
        self.filenames = filenames
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.n_files = len(filenames)
        
        # Hinode/SP spectral parameters
        self.NAXIS1 = 112
        self.CRVAL1 = 6302.0
        self.CDELT1 = 0.0215
        self.CRPIX1 = 57
        
        # Physical scaling factors
        self.phys_maxmin = {
            "T": [2e4, 0],
            "V": [1e6, -1e6],
            "B": [3e3, -3e3]
        }
        
        # Noise level for Hinode
        self.noise_level = 5.9e-4
        
        # Load opacity data
        self._load_opacity_data()
        
        # Load PSF and LSF
        self._load_psf_lsf()
        
        # Initialize data containers
        self.muram_box = None
        self.stokes_data = None
        self.new_logtau = np.arange(-2.0, 0.1, 0.1)
        self.n_logtau = len(self.new_logtau)
        
    def _load_opacity_data(self):
        """Load opacity interpolation data."""
        # Temperature and pressure grids
        self.tab_T = np.array([3.32, 3.34, 3.36, 3.38, 3.40, 3.42, 3.44, 3.46, 3.48, 3.50,
                              3.52, 3.54, 3.56, 3.58, 3.60, 3.62, 3.64, 3.66, 3.68, 3.70,
                              3.73, 3.76, 3.79, 3.82, 3.85, 3.88, 3.91, 3.94, 3.97, 4.00,
                              4.05, 4.10, 4.15, 4.20, 4.25, 4.30, 4.35, 4.40, 4.45, 4.50,
                              4.55, 4.60, 4.65, 4.70, 4.75, 4.80, 4.85, 4.90, 4.95, 5.00,
                              5.05, 5.10, 5.15, 5.20, 5.25, 5.30])
        
        self.tab_p = np.array([-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5,
                              3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.])
        
        # Load opacity table
        df_kappa = pd.read_csv(self.data_path / 'csv' / 'kappa.0.dat', 
                              sep='\s+', header=None)
        df_kappa.columns = ["Temperature index", "Pressure index", "Opacity value"]
        temp_indices = df_kappa["Temperature index"].unique()
        press_indices = df_kappa["Pressure index"].unique()
        opacity_values = df_kappa.pivot(index="Pressure index", 
                                       columns="Temperature index", 
                                       values="Opacity value").values
        
        Tk = self.tab_T[temp_indices]
        Pk = self.tab_p[press_indices]
        self.kappa_interp = RegularGridInterpolator((Pk, Tk), opacity_values, method="linear")
        
    def _load_psf_lsf(self):
        """Load PSF and LSF data."""
        # Load Hinode PSF
        with fits.open(self.data_path / "hinode-MODEST" / "PSFs" / "hinode_psf_bin.0.16.fits") as hdul:
            psf_header = hdul[0].header
            original_psf = hdul[0].data
            
        original_psf = np.ascontiguousarray(original_psf, dtype=np.float32)
        
        # Calculate PSF scaling
        psf_pixel_scale = psf_header.get('CDELT1', 0.16)
        muram_pixel_scale_arcsec = 25 / 725
        psf_scaling_factor = psf_pixel_scale / muram_pixel_scale_arcsec
        target_psf_size = int(original_psf.shape[0] * psf_scaling_factor)
        
        # Resample PSF
        psf_tensor = torch.tensor(original_psf).unsqueeze(0).unsqueeze(0).float()
        resize_psf = transforms.Resize((target_psf_size, target_psf_size), 
                                      interpolation=transforms.InterpolationMode.BICUBIC)
        resampled_psf = resize_psf(psf_tensor).squeeze().numpy()
        self.psf = resampled_psf / resampled_psf.sum()
        
        # Load LSF
        lsf_hinode = pd.read_csv(self.data_path / "hinode-MODEST" / "PSFs" / "hinode_sp.spline.psf", 
                                sep="\s+", header=None)
        lsf_wavelengths = lsf_hinode[0].values
        lsf_values = lsf_hinode[1].values
        lsf_interp = interp1d(lsf_wavelengths, lsf_values, kind="cubic")
        
        # Create LSF kernel
        lsf_wl_min = np.round(lsf_wavelengths.min() + 0.010, 3)
        lsf_wl_max = np.round(lsf_wavelengths.max() - 0.010, 3)
        step = 0.01
        new_kernel_wl = np.arange(lsf_wl_min, lsf_wl_max + step, step)
        new_kernel_values = lsf_interp(new_kernel_wl)
        self.lsf_kernel = new_kernel_values / new_kernel_values.sum()
        
        # Wavelength arrays
        self.wl_original = np.arange(6300.5, 6303.5, 0.01)
        self.wl_hinode = self.CRVAL1 + (np.arange(1, self.NAXIS1 + 1) - self.CRPIX1) * self.CDELT1
        
    def _limit_values(self, data, min_val, max_val):
        """Limit values to avoid interpolation issues."""
        return np.clip(data, min_val + 0.00001, max_val - 0.00001)
        
    def _calculate_optical_depth(self, muram_box, mpre, filename):
        """Calculate optical depth for atmospheric stratification."""
        geom_path = self.data_path / "geom_height"
        logtau_name = f"logtau_{filename}.npy"
        
        if os.path.exists(geom_path / logtau_name):
            return np.load(geom_path / logtau_name)
            
        # Calculate optical depth
        T_log = np.log10(muram_box[..., 0])
        P_log = np.log10(mpre)
        T_log = self._limit_values(T_log, self.tab_T.min(), self.tab_T.max())
        P_log = self._limit_values(P_log, self.tab_p.min(), self.tab_p.max())
        PT_log = np.stack((P_log.flatten(), T_log.flatten()), axis=-1)
        
        kappa_rho = self.kappa_interp(PT_log).reshape(muram_box[..., 0].shape)
        kappa_rho = np.multiply(kappa_rho, muram_box[..., 1])
        
        tau = np.zeros_like(kappa_rho)
        dz = 1e6
        tau[:, :, self.nz - 1] = 1e-5
        
        print(f"Calculating optical depth for {filename}...")
        for iz in tqdm(range(1, self.nz)):
            for ix in range(self.nx):
                for iy in range(self.ny):
                    kpz = kappa_rho[ix, iy, self.nz - 1 - iz:]
                    tau[ix, iy, self.nz - 1 - iz] = simpson(y=kpz, dx=dz)
                    
        muram_logtau = np.log10(tau)
        np.save(geom_path / logtau_name, muram_logtau)
        return muram_logtau
        
    def _stratify_to_optical_depth(self, muram_box, muram_logtau):
        """Stratify atmospheric parameters to optical depth nodes."""
        atm_to_logtau = np.zeros((self.nx, self.ny, self.n_logtau, muram_box.shape[-1]))
        
        for imur in range(muram_box.shape[-1]):
            geom_atm = muram_box[..., imur]
            new_muram_quantity = np.zeros((self.nx, self.ny, self.n_logtau))
            
            for ix in tqdm(range(self.nx), desc=f"Mapping to optical depth atm parameter {imur}"):
                for iy in range(self.ny):
                    mapper = interp1d(
                        x=muram_logtau[ix, iy, :],
                        y=geom_atm[ix, iy, :],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate"
                    )
                    new_muram_quantity[ix, iy, :] = mapper(self.new_logtau)
            atm_to_logtau[..., imur] = new_muram_quantity
            
        return atm_to_logtau
        
    def _normalize_atmosphere(self, muram_box):
        """Apply min-max normalization to atmospheric parameters."""
        scaled_atm = np.zeros_like(muram_box)
        scaled_atm[..., 0] = self._norm_func(muram_box[..., 0], self.phys_maxmin["T"])
        scaled_atm[..., 1] = self._norm_func(muram_box[..., 1], self.phys_maxmin["V"])
        scaled_atm[..., 2] = self._norm_func(muram_box[..., 2], self.phys_maxmin["B"])
        return scaled_atm
        
    def _norm_func(self, arr, maxmin):
        """Min-max normalization function."""
        max_val, min_val = maxmin
        return (arr - min_val) / (max_val - min_val)
        
    def _continuum_normalization(self, spectral_cube):
        """Normalize spectral cube by continuum."""
        cont_indices = [0, 1, 2, 3]
        I_c = spectral_cube[:, :, cont_indices, 0].mean(axis=2).flatten().mean()
        
        norm_spectral_cube = spectral_cube / I_c
        mean_continuum_image = spectral_cube[:, :, cont_indices, 0].mean(axis=2) / I_c
        
        return norm_spectral_cube, mean_continuum_image
        
    def _apply_psf_convolution(self, stokes):
        """Apply PSF convolution to Stokes parameters."""
        stokes_convolved = np.zeros_like(stokes)
        
        for i in range(stokes.shape[-1]):
            for j in tqdm(range(stokes.shape[2]), desc=f"Applying PSF to stokes {i}"):
                stokes_convolved[:, :, j, i] = fftconvolve(stokes[:, :, j, i], self.psf, mode='same')
                
        return stokes_convolved
        
    def _apply_lsf_convolution(self, stokes):
        """Apply LSF convolution to Stokes parameters."""
        stokes_lsf_convolved = np.zeros_like(stokes)
        
        for stk in tqdm(range(stokes.shape[-1]), desc="Applying LSF"):
            stokes_lsf_convolved[..., stk] = convolve1d(
                stokes[..., stk],
                weights=self.lsf_kernel,
                axis=-1,
                mode='wrap'
            )
            
        return stokes_lsf_convolved
        
    def _resample_to_hinode_wavelengths(self, stokes):
        """Resample Stokes parameters to Hinode wavelengths."""
        stokes_resampled = np.zeros((self.nx, self.ny, self.NAXIS1, stokes.shape[-1]))
        
        for stk in tqdm(range(stokes.shape[-1]), desc="Resampling wavelengths"):
            stokes_reshaped = stokes[..., stk].reshape(-1, stokes.shape[2])
            stokes_interp = interp1d(self.wl_original, stokes_reshaped, kind="cubic", axis=1)
            stokes_resampled[..., stk] = stokes_interp(self.wl_hinode).reshape(self.nx, self.ny, self.NAXIS1)
            
        return stokes_resampled
        
    def _add_noise(self, norm_stokes, mean_continuum_image):
        """Add Gaussian noise to normalized Stokes parameters."""
        # Calculate I_c (continuum intensity) like in the notebook
        I_c = mean_continuum_image.flatten().mean()
        level_of_noise = self.noise_level * I_c  # 5.9e-4 * I_c
        
        norm_stokes_with_noise = np.zeros_like(norm_stokes)
        mean_continuum_with_noise = np.zeros_like(mean_continuum_image)
        
        for jx in tqdm(range(norm_stokes.shape[0]), desc="Adding noise"):
            for jy in range(norm_stokes.shape[1]):
                for stk in range(norm_stokes.shape[-1]):
                    norm_stokes_with_noise[jx, jy, :, stk] = (
                        norm_stokes[jx, jy, :, stk] + 
                        level_of_noise * np.random.randn(self.NAXIS1)
                    )
                # Add noise to continuum image as well
                mean_continuum_with_noise[jx, jy] = (
                    mean_continuum_image[jx, jy] + 
                    level_of_noise * np.random.randn(1)[0]
                )
                    
        return norm_stokes_with_noise, mean_continuum_with_noise
        
    def charge_single_file(self, filename, normalize_atmosphere=True):
        """Charge and process data for a single file.
        
        Args:
            filename (str): Filename to process
            normalize_atmosphere (bool): Whether to normalize atmospheric parameters
        """
        print(f"Processing file: {filename}")
        
        # Load atmospheric data
        geom_path = self.data_path / "geom_height"
        mtpr = np.load(geom_path / f"mtpr_{filename}.npy").flatten()
        mrho = np.load(geom_path / f"mrho_{filename}.npy")
        mvzz = np.load(geom_path / f"mvzz_{filename}.npy")
        mvzz = mvzz / mrho  # Convert momentum to velocity
        mbzz = np.load(geom_path / f"mbzz_{filename}.npy")
        
        # Convert magnetic field to Gauss
        coef = np.sqrt(4.0 * np.pi)
        mbzz = mbzz * coef
        
        # Arrange atmospheric data
        muram_box = np.array([mtpr, mvzz, mbzz])
        muram_box = np.moveaxis(muram_box, 0, -1)
        muram_box = np.reshape(muram_box, (self.nx, self.nz, self.ny, muram_box.shape[-1]))
        muram_box = np.moveaxis(muram_box, 1, 2)
        
        # Load pressure for optical depth calculation
        eos = np.fromfile(os.path.join(geom_path, f"eos.{filename}"), dtype=np.float32)
        eos = eos.reshape((2, self.nx, self.nz, self.ny), order="C")
        mpre = eos[1]
        mpre = np.moveaxis(mpre, 1, 2)
        
        # Calculate optical depth and stratify
        muram_logtau = self._calculate_optical_depth(muram_box, mpre, filename)
        muram_box = self._stratify_to_optical_depth(muram_box, muram_logtau)
        if normalize_atmosphere:
            muram_box = self._normalize_atmosphere(muram_box)
        
        # Load and process Stokes data
        stokes = np.load(self.data_path / "stokes" / f"{filename}_prof.npy")
        stokes = np.array([stokes[..., 0], stokes[..., 3]])  # I and V only
        stokes = np.transpose(stokes, (1, 2, 3, 0))
        
        # Apply Hinode adaptations
        # stokes = self._apply_psf_convolution(stokes)
        stokes = self._apply_lsf_convolution(stokes)
        stokes = self._resample_to_hinode_wavelengths(stokes)
        
        # Normalize and add noise
        norm_stokes, mean_continuum_image = self._continuum_normalization(stokes)
        norm_stokes, mean_continuum_with_noise = self._add_noise(norm_stokes, mean_continuum_image)
        
        # Initialize the scaler
        scaler = MinMaxScaler()
        start_wl = 20
        end_wl = 60
        wfa_B_LOS = B_LOS_from_stokes(stokes = norm_stokes, 
                                   ll = self.wl_hinode*u.Angstrom, 
                                   start_ll = start_wl, 
                                   end_ll = end_wl, 
                                   llambda0 = 6301.5*u.Angstrom, 
                                   g = 1.67,
                                   stokes_v_index = 1)
        # Convert wfa_B_LOS to numpy array for normalization
        wfa_B_LOS_gauss = wfa_B_LOS.value

        # Normalize wfa_B_LOS
        wfa_B_LOS_minmax = scaler.fit_transform(wfa_B_LOS_gauss.reshape(-1, 1)).reshape(wfa_B_LOS_gauss.shape)
        
        # Calculate RRMSE for each optical depth height with normalized data
        rrmse_values_normalized = []

        for height_idx in range(self.n_logtau):
            # Get magnetic field at this optical depth height (index 2 is B_LOS)
            muram_B_height = muram_box[:, :, height_idx, 2]
            
            # Normalize MuRAM B_LOS at this height
            muram_B_height_normalized = scaler.fit_transform(muram_B_height.reshape(-1, 1)).reshape(muram_B_height.shape)
            
            # Calculate RRMSE with normalized data
            rrmse = relative_rmse(wfa_B_LOS_minmax, muram_B_height_normalized)
            rrmse_values_normalized.append(rrmse)

        # Find the height with minimum RRMSE
        min_rrmse_idx_normalized = np.argmin(rrmse_values_normalized)
        min_rrmse_logtau = self.new_logtau[min_rrmse_idx_normalized]
        best_muram_B_normalized = muram_box[:, :, min_rrmse_idx_normalized, 2]

        # Fit the scaler on best_muram_B_normalized for future use
        best_muram_B_minmax_scaler = MinMaxScaler()
        best_muram_B_minmax = best_muram_B_minmax_scaler.fit_transform(
            best_muram_B_normalized.reshape(-1, 1)
        ).reshape(best_muram_B_normalized.shape)

        # Return all relevant info for dictionary storage
        return (
            muram_box,
            norm_stokes,
            wfa_B_LOS_minmax,
            best_muram_B_minmax,
            best_muram_B_minmax_scaler,
            min_rrmse_idx_normalized,
            min_rrmse_logtau
        )

    def charge_all_files(self, normalize_atmosphere=True):
        """Charge and process all files, returning a dictionary per file."""
        print(f"Charging {self.n_files} files...")

        self.data_per_file = {}
        for filename in self.filenames:
            (
                muram_box,
                stokes,
                wfa_BLOS_minmax,
                best_muram_B_minmax,
                best_muram_B_minmax_scaler,
                min_rrmse_idx_normalized,
                min_rrmse_logtau
            ) = self.charge_single_file(filename, normalize_atmosphere=normalize_atmosphere)

            self.data_per_file[filename] = {
                "muram_box": muram_box,
                "stokes": stokes,
                "wfa_BLOS_minmax": wfa_BLOS_minmax,
                "best_muram_B_minmax": best_muram_B_minmax,
                "best_muram_B_minmax_scaler": best_muram_B_minmax_scaler,
                "min_rrmse_idx_normalized": min_rrmse_idx_normalized,
                "min_rrmse_logtau": min_rrmse_logtau
            }

        print("Data charging completed!")
        return self.data_per_file

    def reshape_for_training(self):
        """
        Reshape data for PyTorch neural network training for all files.

        Returns:
            dict: Dictionary keyed by filename, each value is a dict with keys:
                'stokes_reshaped', 'muram_reshaped', 'wfa_BLOS_reshaped', 'best_muram_B_reshaped',
                'best_muram_B_minmax_scaler', 'min_rrmse_idx_normalized', 'min_rrmse_logtau'
        """
        if not hasattr(self, "data_per_file"):
            raise ValueError("Data not charged yet. Call charge_all_files() first.")

        reshaped_data = {}
        for filename, file_data in self.data_per_file.items():
            muram_box = file_data["muram_box"]
            stokes_data = file_data["stokes"]
            wfa_BLOS_minmax = file_data["wfa_BLOS_minmax"]
            best_muram_B_minmax = file_data["best_muram_B_minmax"]
            best_muram_B_minmax_scaler = file_data["best_muram_B_minmax_scaler"]
            min_rrmse_idx_normalized = file_data["min_rrmse_idx_normalized"]
            min_rrmse_logtau = file_data["min_rrmse_logtau"]

            # Reshape Stokes: (nx, ny, n_wl, n_stokes) -> (nx*ny, n_stokes, n_wl)
            stokes_reshaped = stokes_data.reshape(-1, self.NAXIS1, 2)
            stokes_reshaped = np.transpose(stokes_reshaped, (0, 2, 1))

            # Reshape MuRAM: (nx, ny, n_logtau, n_params) -> (nx*ny, n_logtau*n_params)
            muram_reshaped = muram_box.reshape(-1, self.n_logtau * 3)

            wfa_BLOS_reshaped = wfa_BLOS_minmax.reshape(-1, 1)
            best_muram_B_reshaped = best_muram_B_minmax.reshape(-1, 1)

            print(f"[{filename}] Stokes reshaped: {stokes_reshaped.shape}")
            print(f"[{filename}] MuRAM reshaped: {muram_reshaped.shape}")
            print(f"[{filename}] WFA B_LOS reshaped: {wfa_BLOS_reshaped.shape}")
            print(f"[{filename}] Best MuRAM B reshaped: {best_muram_B_reshaped.shape}")

            reshaped_data[filename] = {
                "stokes_reshaped": stokes_reshaped,
                "muram_reshaped": muram_reshaped,
                "wfa_BLOS_reshaped": wfa_BLOS_reshaped,
                "best_muram_B_reshaped": best_muram_B_reshaped,
                "best_muram_B_minmax_scaler": best_muram_B_minmax_scaler,
                "min_rrmse_idx_normalized": min_rrmse_idx_normalized,
                "min_rrmse_logtau": min_rrmse_logtau
            }

        return reshaped_data

    def get_data_info(self):
        """Get information about the loaded data."""
        if self.muram_box is None:
            return "No data loaded"
            
        info = {
            'n_files': self.n_files,
            'spatial_dims': (self.nx, self.ny),
            'n_optical_depths': self.n_logtau,
            'n_wavelengths': self.NAXIS1,
            'n_stokes': 2,
            'n_atm_params': 3,
            'muram_shape': self.muram_box.shape,
            'stokes_shape': self.stokes_data.shape,
            "wfa_BLOS_shape": self.wfa_BLOS_minmax.shape,
            "best_muram_B_shape": self.best_muram_B_minmax.shape,
            'optical_depth_range': (self.new_logtau.min(), self.new_logtau.max()),
            'wavelength_range': (self.wl_hinode.min(), self.wl_hinode.max())
        }
        
        return info

############################################################################
# WFA B_LOS utils
############################################################################


def relative_rmse(predicted, actual):
    """Calculate relative root mean square error using scikit-learn"""
    rmse = root_mean_squared_error(actual.flatten(), predicted.flatten())
    mean_actual = np.mean(np.abs(actual))
    return rmse / mean_actual if mean_actual != 0 else np.inf

def B_LOS_from_stokes(stokes:np.ndarray,
                      ll:np.ndarray,
                      start_ll:int,
                      end_ll:int,
                      llambda0:float,
                      g:float,
                      stokes_v_index:int):
    """
    Estimate the line-of-sight magnetic field from a data cube of Stokes I profiles.
    Args:
    stokes: numpy.ndarray
        3D data cube of Stokes I profiles
    ll: numpy.ndarray
        Wavelength axis in amstrongs
    start_ll: int
        Index of the starting wavelength range
    end_ll: int
        Index of the ending wavelength range
    llambda0: float
        Rest wavelength in amstrongs
    g: float
        Landé factor
    stokes_v_index: int
        Index of the Stokes V profile in the data cube
    Returns:
    numpy.ndarray
        2D map of the line-of-sight magnetic field in Gauss
    """
    wfa_constant = e.si / (4 * np.pi) / m_e / c
    wfa_constant = wfa_constant.to(1 / u.G / u.Angstrom )
    def estimate_B(dI_dl, V):
        ND = len(V)
        a = np.zeros([ND, 2])
        a[:, 0] = dI_dl[:]
        a[:, 1] = 1.0
        b = V[:]

        # Correct least-squares solution
        p = np.linalg.pinv(a) @ b / dI_dl.unit
        
        # Compute B_LOS
        B = -p[0]*u.Angstrom / (wfa_constant * (llambda0)**2.0 * g)
        return B
        
    NX = stokes.shape[0]
    NY = stokes.shape[1]

    B = np.zeros([NX,NY])

    for i in range (0,NX):
        for j in range(0,NY):
            dI_dl =  np.gradient(stokes[i,j,start_ll:end_ll,0]) / np.gradient(ll[start_ll:end_ll])
            local_B = estimate_B(dI_dl = dI_dl,
                                V = stokes[i,j,start_ll:end_ll,stokes_v_index])
            B[i,j] = local_B.value
    B = B * u.G
    
    return B