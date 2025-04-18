import numpy as np

import pandas as pd

from astropy.io import fits

from skimage import filters

from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import simpson
from scipy.signal import fftconvolve
from scipy.ndimage import convolve1d

from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

import os

from tqdm import tqdm 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pathlib import Path

class MURaM:
    """
    A class to handle MURaM simulation data.

    Attributes:
    -----------
    ptm : Path
        Path to the data directory.
    filename : str
        Name of the file to be processed.
    nx : int
        Number of grid points in the x-direction.
    nz : int
        Number of grid points in the y-direction.
    ny : int
        Number of grid points in the z-direction.
    atm_quant : np.ndarray
        Array to store atmospheric quantities.
    stokes : np.ndarray
        Array to store Stokes parameters.
    mags_names : list
        List of atmospheric quantities names.
    phys_maxmin : dict
        Dictionary to store physical max and min values for scaling.
    new_wl : np.ndarray
        Array to store new wavelength values after spectral resolution degradation.
    I_63005 : np.ndarray
        Intensity map used for balancing intergranular and granular regions.
    """

    def __init__(self, filename: str, verbose: bool = False, ptm: str = "./data"):
        """
        Initialize the MURaM object with the given filename.

        Parameters:
        -----------
        filename : str
            Name of the file to be processed.
        """

        
        self.ptm = Path(ptm)
        self.filename = filename
        self.verbose = verbose
        
        self.nlam = 300  # this parameter is useful when managing the self.stokes parameters
        self.nx = 480
        self.nz = 256
        self.ny = 480
    def charge_quantities(self) -> None:
        """
        Load and process the atmospheric and Stokes quantities from the data files.
        """
        print(f"""
                ######################## 
                Reading {self.filename} MuRAM data...
                ######################## 
                      """)
        
        quantities_path = "geom_height"
        
        print("Charging temperature ...")
        eos = np.fromfile(self.ptm / quantities_path / f"eos.{self.filename}", dtype=np.float32)
        eos = eos.reshape((2, self.nx*self.nz*self.ny), order="C")
        mtpr = eos[0]
        mpre = eos[1]
        if self.verbose:
            print("mtpr shape:", mtpr.shape)
        
        
        if self.verbose:
            print("Charging density...")
        mrho = np.load(self.ptm / quantities_path / f"mrho_{self.filename}.npy")
        if self.verbose:
            print("mrho shape:", mrho.shape)
        
        print("Charge velocity...")
        mvzz = np.load(self.ptm / quantities_path / f"mvzz_{self.filename}.npy")
        if self.verbose:
            print("mvyy shape:", mvzz.shape)
        
        mvzz = mvzz / mrho

        #########################################
        mrho = np.log10(mrho) #For better learning process
        ########################################

        print("Charging magnetic field vector...")
        mbxx = np.load(self.ptm / quantities_path / f"mbxx_{self.filename}.npy")
        mbyy = np.load(self.ptm / quantities_path / f"mbyy_{self.filename}.npy")
        mbzz = np.load(self.ptm / quantities_path / f"mbzz_{self.filename}.npy")
        
        coef = np.sqrt(4.0 * np.pi)  # cgs units conversion
        
        mbxx = mbxx * coef
        mbyy = mbyy * coef
        mbzz = mbzz * coef
        if self.verbose:
            print("mbxx shape:", mbxx.shape)
            print("mbzz shape:", mbzz.shape)
            print("mbyy shape:", mbyy.shape)
        
        print(f"""
                ######################## 
                Finished!
                ######################## 
                      """)

        print("Creating atmosphere quantities array...")
        self.mags_names = [r"$T$", r"$\log(\rho)$", r"$v_{\text{LOS}}$", r"$B_x$", r"$B_y$", r"$B_z$"]
        self.output_names = ["mtpr","mrho", "mvzz", "mbxx", "mbyy", "mbzz"]
        self.atm_quant = np.array([mtpr, mrho, mvzz, mbxx, mbyy, mbzz])
        self.atm_quant = np.moveaxis(self.atm_quant, 0, 1)
        self.atm_quant = np.reshape(self.atm_quant, (self.nx, self.nz, self.ny, self.atm_quant.shape[-1]))
        self.atm_quant = np.moveaxis(self.atm_quant, 1, 2)
        
        plot_atmosphere_quantities(atm_quant=self.atm_quant, 
                                   titles = self.mags_names,
                                   image_name=f"{self.filename}_atm_quantities",
                                   height_index=180)
        print("Created!")
        if self.verbose:
            print("atm geom height shape:", self.atm_quant.shape)

        print("Charging stokes vectors...")
        self.stokes = np.load(self.ptm / "stokes" / f"{self.filename}_prof.npy")
        self.I_63005 = self.stokes[:, :, 0, 0]  # Intensity map that is going to be used to balance intergranular and granular regions.
        print("Charged!")
        if self.verbose:
            print("stokes shape", self.stokes.shape)
    def just_LOS_components(self) -> None:
        """
        Keep only the LOS component of the magnetic field.
        """
        print("Keeping only the LOS component of the magnetic field...")
        self.mags_names = [r"$T$", r"$\log(\rho$)", r"$v_{z}$", r"$B_\text{LOS}$"]
        self.atm_quant[..., 3] = self.atm_quant[..., -1] # Bz
        self.atm_quant = self.atm_quant[...,0:4]
        self.just_LOS_flag = True
        print("atm_quant shape:", self.atm_quant.shape)
    def optical_depth_stratification(self, new_logtau: np.ndarray[float]) -> None:
        
        # Geom data path
        geom_path = self.ptm / "geom_height"
        logtau_name = f"logtau_{self.filename}.npy"
        
        if not os.path.exists(geom_path / logtau_name):
            print("Calculating optical depth stratification...")
            muram_logtau = calculate_logtau(muram = self, 
                                    save_path = geom_path,
                                    save_name=logtau_name)
            print(f"Saved to {geom_path / logtau_name}")
        else:
            print("Loading optical depth stratification...")
            muram_logtau = np.load(geom_path / logtau_name)
            print(f"Loaded from {geom_path / logtau_name}")
        
        if self.verbose:
            print("muram logtau shape", muram_logtau.shape)
        print("Done!")
        
        # Plotting the optical depth stratification
        image_path = Path("images/first_experiment/atmosphere")
        if not image_path.exists():
            image_path.mkdir(parents=True)
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(muram_logtau[:,:,180], cmap = "gist_gray")
        ax[1].plot(muram_logtau.mean(axis = (0,1)),self.atm_quant[...,0].mean(axis = (0,1)))
        fig.savefig("images/first_experiment/atmosphere/optical_depth.png")
        
         # New optical depth stratification array.
        self.n_logtau = new_logtau.shape[0]

        # Mapping to the new optical depth stratification
        atm_to_logtau = np.zeros((self.nx,self.ny,self.n_logtau,self.atm_quant.shape[-1]))
        print(f"Mapping to new optical depth stratification...")
        for imur in range(self.atm_quant.shape[-1]):
            # Calculate the new optical depth stratification
            muram_quantity = self.atm_quant[..., imur]
            
            new_muram_quantity = map_to_logtau(muram = self, 
                                                geom_atm=muram_quantity,
                                                geom_logtau=muram_logtau,
                                                new_logtau=new_logtau)
            atm_to_logtau[...,imur] = new_muram_quantity
            
        print("Loaded!")

        self.atm_quant = atm_to_logtau
        del atm_to_logtau
        
        if self.verbose:
            print("atm logtau shape:", self.atm_quant.shape)               
    def modified_components(self) -> None:
        if not r"$B_x$" in self.mags_names or not r"$B_y$" in self.mags_names:
            raise ValueError("All magnetic field components must be loaded before modifying.")
        self.mags_names = [r"$T$", r"$\rho$", r"$v_{z}$", r"$B$", r"$\varphi$", r"$\gamma$"]
        
        # Magnetic field components
        mbxx = self.atm_quant[..., 3]
        mbyy = self.atm_quant[..., 4]
        mbzz = self.atm_quant[..., 5]
        
        print("Modifying magnetic field components to fight azimuth ambiguity...")
        # Modified magnetic field components
        def opposite_angle(angle):
            opposite = (angle + np.pi) % (2 * np.pi)
            return opposite
        magnetic_field_strength = np.sqrt(mbxx**2 + mbyy**2 + mbzz**2)
        azimuth = np.arctan(mbyy/mbxx)
        azimuth = np.where(azimuth > np.pi, opposite_angle(azimuth), azimuth) # azimuth ambiguity
        azimuth = np.where(azimuth < 0, opposite_angle(azimuth), azimuth) # azimuth ambiguity
        azimuth = np.rad2deg(azimuth)
        zenith = np.rad2deg(np.arctan(mbzz/np.sqrt(mbxx**2 + mbyy**2)))
        print("Quantities modified!")

        print("Creating atmosphere quantities array...")
        # Saving modified quantities replacing the x,y,z components
        self.atm_quant[..., 3] = magnetic_field_strength
        self.atm_quant[..., 4] = azimuth
        self.atm_quant[..., 5] = zenith
        
        plot_atmosphere_quantities(atm_quant=self.atm_quant, 
                                   titles = self.mags_names,
                                   image_name=f"{self.filename}_modified_atm_quantities")
        print("Created!")
        if self.verbose:
            print("atm modified components shape:", self.atm_quant.shape)
            
        self.modified_flag = True
    def spatial_convolution(self) -> None:
        """
        Degrade the spectral resolution of the Stokes parameters.

        Parameters:
        -----------
        new_points : int, optional
            Number of new spectral points after degradation (default is 36).
        """

        ###########################################
        # Spatial convolution
        ###########################################
        with fits.open(self.ptm / "hinode-MODEST" / "PSFs" / "hinode_psf_bin.0.16.fits") as hdul:
            psf = hdul[0].data #This PSF is already normalized.

        psf = psf / psf.sum()

        print("Spatially convolving the Stokes parameters...")
        stokes_spatially_convolved = np.zeros_like(self.stokes)
        for i in range(self.stokes.shape[-1]):
            for j in tqdm(range(self.stokes.shape[2])):
                stokes_spatially_convolved[:, :, j, i] = fftconvolve(self.stokes[:, :, j, i], psf, mode='same')

        self.stokes = stokes_spatially_convolved    
    def spectral_convolution(self) -> None:

        wl = (np.arange(300) * 0.01  + 6300) # angstroms

        lsf_hinode = pd.read_csv(self.ptm / "hinode-MODEST" / "PSFs" / "hinode_sp.spline.psf", sep = "\s+", header = None)

        lsf_wavelengths = lsf_hinode[0].values
        lsf_values = lsf_hinode[1].values
        lsf_interp = interp1d(lsf_wavelengths, lsf_values, kind = "cubic") 

        lsf_wl_min = np.round(lsf_wavelengths.min()+0.010, 3)
        lsf_wl_max = np.round(lsf_wavelengths.max()-0.010, 3)

        step = wl[1] - wl[0]
        new_kernel_wl = np.arange(lsf_wl_min, lsf_wl_max+step, step) #We are assuming that the values of the LSF are presented in amstrongs.
        new_kernel_values = lsf_interp(new_kernel_wl)

        lsf_values = lsf_values / lsf_values.sum()
        new_kernel_values = new_kernel_values / new_kernel_values.sum()


        stokes_sptrl_convolved = np.zeros_like(self.stokes)
        print("Spectrally convolving the Stokes parameters...")
        # Apply convolution for each Stokes parameter efficiently
        for stk in tqdm(range(self.stokes.shape[-1])):
            stokes_sptrl_convolved[..., stk] = convolve1d(
                self.stokes[..., stk],               # Input: (nx, ny, n_lambda)
                weights=new_kernel_values,     # Kernel
                axis=-1,                       # Apply along wavelength axis
                mode='wrap'                    # Mimic periodic boundary as before
            )

        self.stokes = stokes_sptrl_convolved
    def spectral_sampling(self) -> None:
        # Parameters from Hinode/SP FITS header (typical values)
        NAXIS1 = 112
        CRVAL1 = 6302.0    # Ångstroms
        CDELT1 = 0.0215    # Å/pixel
        CRPIX1 = 57        # Reference pixel

        # Wavelength array calculation
        self.wl = (np.arange(300) * 0.01  + 6300.5) # angstroms
        self.new_wl = CRVAL1 + (np.arange(1, NAXIS1 + 1) - CRPIX1) * CDELT1

        print("Interpolating Stokes parameters to Hinode/SP wavelengths...")
        # Reshape stokes array for vectorized interpolation
        stokes_reshaped = self.stokes.reshape(-1, self.stokes.shape[2], self.stokes.shape[3])

        # Create an array to store the interpolated values
        stokes_to_hinode_wl = np.zeros((self.nx, self.ny, NAXIS1, self.stokes.shape[-1]))

        # Perform vectorized interpolation
        for stk in tqdm(range(self.stokes.shape[-1])):
            stokes_sptrl_interp = interp1d(self.wl, stokes_reshaped[:, :, stk], kind="cubic", axis=1)
            stokes_to_hinode_wl[..., stk] = stokes_sptrl_interp(self.new_wl).reshape(self.nx, self.ny, NAXIS1)
        
        self.stokes = stokes_to_hinode_wl
    def surface_pixel_sampling(self) -> None:

        # Target pixel size:
        target_pixel_size_km = 116
        current_pixel_size_km = 25
        scaling_factor = current_pixel_size_km / target_pixel_size_km
        
        target_size = int(480 * scaling_factor)  # ~103 pixels

        resize = transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC)
        new_image_dimensions = resize.size
        self.new_nx = new_image_dimensions[0]
        self.new_ny = new_image_dimensions[1]
        print("atm surface pixel sampling...")
        resampled_atm = np.zeros((self.new_nx, self.new_ny, self.atm_quant.shape[2], self.atm_quant.shape[3]))
        for atm_mag in tqdm(range(self.atm_quant.shape[-1])):
            for itau in range(self.atm_quant.shape[2]):
                resampled_atm[..., itau, atm_mag] = resize(torch.tensor(self.atm_quant[..., itau, atm_mag]).unsqueeze(0).unsqueeze(0)).numpy().squeeze()
        print("stokes surface pixel sampling...")
        resampled_stokes = np.zeros((self.new_nx, self.new_ny, self.stokes.shape[2], self.stokes.shape[3]))
        for stk in tqdm(range(self.stokes.shape[-1])):
            for iwl in range(self.stokes.shape[2]):
                resampled_stokes[..., iwl, stk] = resize(torch.tensor(self.stokes[..., iwl, stk]).unsqueeze(0).unsqueeze(0)).numpy().squeeze()

        self.atm_quant = resampled_atm
        self.stokes = resampled_stokes
        print("surface pixel sampling done!")
        print(f"atm shape: {self.atm_quant.shape}")
        print(f"stokes shape: {self.stokes.shape}")
    def scale_quantities(self, stokes_weigths: list[int]) -> None:
        """
        Scale the atmospheric and Stokes quantities.
        """
        if self.verbose:
            print(f""" self.stokes:
            I_max = {np.max(self.stokes[:, :, :, 0])}
            Q_max = {np.max(self.stokes[:, :, :, 1])}
            U_max = {np.max(self.stokes[:, :, :, 2])}
            V_max = {np.max(self.stokes[:, :, :, 3])}
            I_min = {np.min(self.stokes[:, :, :, 0])}
            Q_min = {np.min(self.stokes[:, :, :, 1])}
            U_min = {np.min(self.stokes[:, :, :, 2])}
            V_min = {np.min(self.stokes[:, :, :, 3])}
            """)
            
            print(f"""
            MAX VALUES:
            mtpr max = {np.max(self.atm_quant[:, :, :, 0])}
            mrho max = {np.max(self.atm_quant[:, :, :, 1])}
            mvzz max = {np.max(self.atm_quant[:, :, :, 2])}
            mbzz max = {np.max(self.atm_quant[:, :, :, 3])}
                """)
            
            print(f"""
            MIN VALUES:
            mtpr min = {np.min(self.atm_quant[:, :, :, 0])}
            mrho min = {np.min(self.atm_quant[:, :, :, 1])}
            mvzz min = {np.min(self.atm_quant[:, :, :, 2])}
            mbzz min = {np.min(self.atm_quant[:, :, :, 3])}
                """) 
        
        print("Scaling the quantities...")
        # Atmosphere magnitudes scale factors
        self.phys_maxmin = {
            "T": [8e3, 4e3], # K
            "B": [2e3, -2e3], # G
            "Rho": [-7, -6], # g/cm^3 log 
            "V": [1e6, -1e6], #cm/s
        }

        # maxmin normalization function
        def norm_func(arr: np.ndarray, maxmin: list[float]) -> np.ndarray:
            max_val = maxmin[0]
            min_val = maxmin[1]
            return (arr - min_val) / (max_val - min_val)

        # Atmosphere magnitudes normalization
        self.atm_quant[:, :, :, 0] = norm_func(self.atm_quant[:, :, :, 0], self.phys_maxmin["T"])
        self.atm_quant[:, :, :, 1] = norm_func(self.atm_quant[:, :, :, 1], self.phys_maxmin["Rho"])
        self.atm_quant[:, :, :, 2] = norm_func(self.atm_quant[:, :, :, 2], self.phys_maxmin["V"])
        self.atm_quant[:, :, :, 3] = norm_func(self.atm_quant[:, :, :, 3], self.phys_maxmin["B"])
        
        print("Normalizing the Stokes parameters by the continuum...")
        # Continuum calculation
        scaled_stokes = np.ones_like(self.stokes)

        # Took the first 10 values in the continuum and calculated the mean value to obtain the I_c
        print("calculating the continuum...")
        wl_cont_values = self.new_wl[:10]  
        cont_values = self.stokes[:, :, :10, 0]  # Extract continuum values for all spatial points
        cont_model = interp1d(wl_cont_values, cont_values, kind="linear", axis=-1, bounds_error=False, fill_value="extrapolate")  # Vectorized interpolation
        Ic = cont_model(self.new_wl).flatten().mean()  # Interpolated continuum values
        scaled_stokes = self.stokes / Ic # Apply continuum normalization
        self.mean_continuum = cont_values.mean(axis = 2)

        self.stokes = scaled_stokes
        del scaled_stokes
        # Stokes parameter weighting
        for i in range(len(stokes_weigths)):
            self.stokes[:, :, :, i] = self.stokes[:, :, :, i] * stokes_weigths[i]
            
        print("Scaled!")

        if self.verbose:
            print(f""" self.stokes:
            I_max = {np.max(self.stokes[:, :, :, 0])}
            Q_max = {np.max(self.stokes[:, :, :, 1])}
            U_max = {np.max(self.stokes[:, :, :, 2])}
            V_max = {np.max(self.stokes[:, :, :, 3])}
            I_min = {np.min(self.stokes[:, :, :, 0])}
            Q_min = {np.min(self.stokes[:, :, :, 1])}
            U_min = {np.min(self.stokes[:, :, :, 2])}
            V_min = {np.min(self.stokes[:, :, :, 3])}
            """)
            
            print(f"""
            MAX VALUES:
            mtpr max = {np.max(self.atm_quant[:, :, :, 0])}
            mrho max = {np.max(self.atm_quant[:, :, :, 1])}
            mvzz max = {np.max(self.atm_quant[:, :, :, 2])}
            mbzz max = {np.max(self.atm_quant[:, :, :, 3])}
                """)
            
            print(f"""
            MIN VALUES:
            mtpr min = {np.min(self.atm_quant[:, :, :, 0])}
            mrho min = {np.min(self.atm_quant[:, :, :, 1])}
            mvzz min = {np.min(self.atm_quant[:, :, :, 2])}
            mbzz min = {np.min(self.atm_quant[:, :, :, 3])}
                """) 
    def spectral_noise(self, level_of_noise: float) -> None:
        norm_stokes_with_noise = np.zeros_like(self.stokes)


        for jx in tqdm(range(self.stokes.shape[0])):
            for jy in range(self.stokes.shape[1]):
                for stk in range(self.stokes.shape[-1]):
                    new_points = int(np.random.poisson(1))
                    norm_stokes_with_noise[jx, jy, :, stk] = self.stokes[jx, jy, :, stk] + level_of_noise * np.random.randn(len(self.new_wl)) 
        
        self.stokes = norm_stokes_with_noise
    def intergran_gran_polariz_balance(self) -> None:
        """
        Balance the quantities of data from the granular and intergranular zones.
        """
        # Balancing the quantity of data from the four new masks by randomly dropping elements from the greater zone.
        print("Balancing data...")

        # Continuum mask
        thresh1 = filters.threshold_otsu(self.mean_continuum)
        mean_continuum_mask = self.mean_continuum<thresh1

        # Circular polarization mask
        circular_polarimetry = np.sum(np.abs(self.stokes[...,3]), axis = -1)/len(self.new_wl)
        denoised = filters.gaussian(circular_polarimetry, sigma = 2)
        thresh1 = filters.threshold_otsu(denoised)
        circular_polarization_mask = denoised < thresh1

        # Create composite masks
        foreground_mean_continuum = mean_continuum_mask == 1
        background_mean_continuum = mean_continuum_mask == 0
        foreground_circular_polarization = circular_polarization_mask == 1
        background_circular_polarization = circular_polarization_mask == 0

        # Combine masks to get the four categories
        combined_foreground = foreground_mean_continuum & foreground_circular_polarization
        combined_background = background_mean_continuum & background_circular_polarization
        combined_foreground_mean_background_circular = foreground_mean_continuum & background_circular_polarization
        combined_background_mean_foreground_circular = background_mean_continuum & foreground_circular_polarization

        # Function to balance data between two masks
        def balance_data(mask1, mask2, data):
            len1 = np.sum(mask1)
            len2 = np.sum(mask2)
            np.random.seed(50)
            if len1 < len2:
                index_select = np.random.choice(range(len2), size=(len1,), replace=False)
                balanced_data1 = data[mask1]
                balanced_data2 = data[mask2][index_select]
            elif len1 > len2:
                index_select = np.random.choice(range(len1), size=(len2,), replace=False)
                balanced_data1 = data[mask1][index_select]
                balanced_data2 = data[mask2]
            else:
                balanced_data1 = data[mask1]
                balanced_data2 = data[mask2]

            return balanced_data1, balanced_data2

        # Balance data for each pair of masks
        muram_box_balanced_1, muram_box_balanced_2 = balance_data(combined_foreground, combined_foreground_mean_background_circular, self.atm_quant)
        stokes_balanced_1, stokes_balanced_2 = balance_data(combined_foreground, combined_foreground_mean_background_circular, self.stokes)

        muram_box_balanced_3, muram_box_balanced_4 = balance_data(combined_background_mean_foreground_circular, combined_background, self.atm_quant)
        stokes_balanced_3, stokes_balanced_4 = balance_data(combined_background_mean_foreground_circular, combined_background, self.stokes)

        # Combine balanced data
        self.atm_quant = np.concatenate((muram_box_balanced_1, muram_box_balanced_2, muram_box_balanced_3, muram_box_balanced_4), axis=0)
        self.stokes = np.concatenate((stokes_balanced_1, stokes_balanced_2, stokes_balanced_3, stokes_balanced_4), axis=0)


        print("Done")
                    
        if self.verbose:
            print(f"""
        Shape after granular and intergranular balance:")
            atm shape: {self.atm_quant.shape}
            stokes shape: {self.stokes.shape}
            """)
        print("Done")

##############################################################
# Preprocessing utils
##############################################################
def calculate_logtau(muram:MURaM, save_path: str, save_name: str, verbose: bool) -> np.ndarray:
    """
    Calculate the optical depth stratification for the MURaM simulation data.

    Parameters:
    -----------
    muram : MURaM
        The MURaM object containing the simulation data.
    save_path : str
        The path to save the calculated optical depth data.
    save_name : str
        The name of the file to save the calculated optical depth data.

    Returns:
    --------
    np.ndarray
        The calculated optical depth stratification array.
    """
    # Data path
    geom_path = muram.ptm / "geom_height"
    
    # Load the pressure
    eos = np.fromfile(os.path.join(geom_path,  f"eos.{muram.filename}"), dtype=np.float32)
    eos = eos.reshape((2, muram.nx, muram.nz, muram.ny), order = "C")
    mpre = eos[1]
    mpre = np.moveaxis(mpre, 1, 2)  # Pressure array to be used in the calculation of the optical depth
    del eos
    
    #######################################
    #Calculate optical depth
    #######################################
    
    # Upload the opacity data
    tab_T = np.array([3.32, 3.34, 3.36, 3.38, 3.40, 3.42, 3.44, 3.46, 3.48, 3.50,
                    3.52, 3.54, 3.56, 3.58, 3.60, 3.62, 3.64, 3.66, 3.68, 3.70,
                    3.73, 3.76, 3.79, 3.82, 3.85, 3.88, 3.91, 3.94, 3.97, 4.00,
                    4.05, 4.10, 4.15, 4.20, 4.25, 4.30, 4.35, 4.40, 4.45, 4.50,
                    4.55, 4.60, 4.65, 4.70, 4.75, 4.80, 4.85, 4.90, 4.95, 5.00,
                    5.05, 5.10, 5.15, 5.20, 5.25, 5.30])

    tab_p = np.array([-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5,
                    3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.])
    
    df_kappa = pd.read_csv('./csv/kappa.0.dat', sep='\s+', header=None)
    df_kappa.columns = ["Temperature index", "Pressure index", "Opacity value"]
    temp_indices = df_kappa["Temperature index" ].unique()
    press_indices = df_kappa["Pressure index"].unique()
    opacity_values = df_kappa.pivot(index = "Pressure index", columns = "Temperature index", values = "Opacity value").values

    Tk = tab_T[temp_indices]
    Pk = tab_p[press_indices]
    K = opacity_values
    
    # Interpolation of the opacity values
    print("Interpolating opacity values...")
    print(np.max(Tk), np.min(Tk))
    print(np.max(Pk), np.min(Pk))
    kappa_interp = RegularGridInterpolator((Pk,Tk), K, method="linear")
    
    def limit_values(data, min_val, max_val):
        new_data = np.clip(data, min_val+0.00001, max_val-0.00001)
        return new_data
            
    T_log = np.log10(muram.atm_quant[..., 0]) 
    P_log = np.log10(mpre) 
    T_log = limit_values(T_log, Tk.min(), Tk.max())
    P_log = limit_values(P_log, Pk.min(), Pk.max())
    # Add print statements to check the range of T_log and P_log
    if verbose:
        print("T_log min:", T_log.min(), "T_log max:", T_log.max())
        print("P_log min:", P_log.min(), "P_log max:", P_log.max())
        print("Tk min:", Tk.min(), "Tk max:", Tk.max())
        print("Pk min:", Pk.min(), "Pk max:", Pk.max())
    PT_log = np.array(list(zip(P_log.flatten(), T_log.flatten())))
    
    # Check for out-of-bounds values in PT_log
    out_of_bounds = np.any((PT_log[:, 0] < Pk.min()) | (PT_log[:, 0] > Pk.max()) | 
                           (PT_log[:, 1] < Tk.min()) | (PT_log[:, 1] > Tk.max()))
    if out_of_bounds:
        print("Out-of-bounds values found in PT_log")
        print("PT_log min:", PT_log.min(axis=0))
        print("PT_log max:", PT_log.max(axis=0))
        out_of_bounds_indices = np.where((PT_log[:, 0] < Pk.min()) | (PT_log[:, 0] > Pk.max()) | 
                                         (PT_log[:, 1] < Tk.min()) | (PT_log[:, 1] > Tk.max()))
        print("Out-of-bounds indices:", out_of_bounds_indices)
        print("Out-of-bounds values:", PT_log[out_of_bounds_indices])
        raise ValueError("PT_log contains out-of-bounds values")

    # Check for nan or inf values in PT_log
    if np.any(np.isnan(PT_log)) or np.any(np.isinf(PT_log)):
        print("nan or inf values found in PT_log")
        print("PT_log:", PT_log)
        raise ValueError("PT_log contains nan or inf values")
    
    print("no errors...")
    kappa_rho = np.zeros_like(muram.atm_quant[..., 0])
    kappa_rho = kappa_interp(PT_log)
    kappa_rho = kappa_rho.reshape(muram.atm_quant[...,0].shape)
    kappa_rho = np.multiply(kappa_rho, muram.atm_quant[..., 1])
    
    #Optical depth calculation
    tau = np.zeros_like(kappa_rho)
    dz = 1e6 # 10 km -> 1e6 cm
    tau[:,:,muram.nz-1] = 1e-5

    print("Calculating optical depth...")
    for iz in tqdm(range(1,muram.nz)):
        for ix in range(muram.nx):
            for iy in range(muram.ny):
                kpz = kappa_rho[ix,iy,muram.nz-1-iz:]
                tau[ix,iy,muram.nz-1-iz] = simpson(y = kpz, 
                                    dx = dz)
                
    logtau = np.log10(tau)
    
    np.save(save_path / save_name, logtau)
    print("Done!")
    return logtau
def map_to_logtau(muram: MURaM,
                  geom_atm: np.ndarray,
                  geom_logtau: np.ndarray,
                  new_logtau: np.ndarray):
    """
    Map quantities from geometrical height to optical depth.
    
    Args:
        muram (MURaM): MURaM simulation object.
        geom_atm (np.ndarray): Original array distributed along geometrical height.
        geom_logtau (np.ndarray): Distribution of optical depth for the original array.
        new_logtau (np.ndarray): Target optical depth grid for mapping.
    
    Returns:
        np.ndarray: Array containing the mapped quantity to the new optical depth grid.
    """
    new_muram_quantity = np.zeros((muram.nx, muram.ny, muram.n_logtau))
    
    for ix in tqdm(range(muram.nx)):
        for iy in range(muram.ny):
            # Create and apply interpolation function directly
            mapper = interp1d(
                x=geom_logtau[ix, iy, :], 
                y=geom_atm[ix, iy, :], 
                kind="linear", 
                bounds_error=False, 
                fill_value="extrapolate"
            )
            new_muram_quantity[ix, iy, :] = mapper(new_logtau)
    
    return new_muram_quantity
def continuum_normalization(spectral_cube: np.ndarray, wl_array: np.ndarray) -> np.ndarray:
    """
    Function to normalize the spectral cube by continuum normalization.
    Args:
        spectral_cube(np.ndarray): Spectral cube to be normalized.
        wl_array(np.ndarray): Wavelength array.
    Returns:
        (np.ndarray) Normalized spectral
    """

    # Vectorized interpolation for speedup
    # cont_values = spectral_cube[:, :, cont_indices, 0].mean(axis = 2).flatten().mean()  # corresponding intensity values to the selected continuum indices
    # cont_model = interp1d(wl_cont_values, cont_values, kind="linear", axis=2)  # Interpolation applied over the assumed continuum values
    I_c =  spectral_cube[:, :, :10, 0].mean(axis = 2).flatten().mean()
    # Apply the normalization
    norm_spectral_cube = np.zeros_like(spectral_cube)
    #mean_continuum_image = cont_model(wl_array).mean(axis=2)
    mean_continuum_image = spectral_cube[:, :, :10, 0].mean(axis = 2)
    for i in range(spectral_cube.shape[-1]):
        norm_spectral_cube[..., i] = spectral_cube[..., i]/ I_c

    mean_continuum_image /= I_c
    return norm_spectral_cube, mean_continuum_image
def spectropolarimetry(stokes: np.ndarray) -> np.ndarray:
    """
    Function to calculate the polarization degree and angle from the Stokes parameters.
    Args:
        stokes(np.ndarray): Array containing the Stokes parameters.
    Returns:
        (np.ndarray) Array containing the polarization degree and angle.
    """
    Q, U, V = stokes[..., 1], stokes[..., 2], stokes[..., 3]
    nwl_points = stokes.shape[-2]

    mean_continuum = np.mean(stokes[..., 0], axis = -1)
    linear_polarization = np.sum(np.sqrt(Q**2 + U**2), axis = -1) / nwl_points
    circular_polarization = np.sum(np.abs(V),axis=-1) / nwl_points
    return mean_continuum, linear_polarization, circular_polarization

##############################################################
# Plot utils
##############################################################
def plot_stokes(stokes: np.ndarray, 
                wl_points: np.ndarray,
                stokes_titles: list[str], 
                image_name: str,
                images_dir: str = "images",
                stokes_subdir: str = "stokes") -> None:
    """
    Plots the Stokes parameters (I, Q, U, V) and saves the plot as an image file.
    Parameters:
        stokes (np.ndarray): A 2D array where each column represents a Stokes parameter (I, Q, U, V) and each row represents a wavelength point.
        new_wl (np.ndarray): A 1D array of wavelength points corresponding to the Stokes parameters.
        image_name (str): The name of the image file to save.
        images_dir (str, optional): The directory where the image will be saved. Default is "images".
        stokes_subdir (str, optional): The subdirectory within images_dir where the image will be saved. Default is "stokes".
    Returns:
    None
    Saves:
    A plot of the Stokes parameters as an image file in the specified directory.
    """
    
    fig, ax = plt.subplots(1, 4, figsize=(6*4,4))
    step_value = wl_points[1] - wl_points[0]
    fig.suptitle(f'Stokes Parameters (Step: {step_value:.2f} nm)', fontsize=16)
    for i in range(len(stokes_titles)):
        ax[i].scatter(wl_points, stokes[:,i], color = "red", s = 10)
        ax[i].plot(wl_points, stokes[:,i], "k", alpha=0.5)
        ax[i].set_title(stokes_titles[i])
    
    images_dir = os.path.join(images_dir, stokes_subdir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    image_path = os.path.join(images_dir, f"{image_name}.pdf")
    fig.savefig(image_path)

    print(f"Saved image to: {image_path}")
def plot_atmosphere_quantities(atm_quant: np.ndarray, 
                               titles:list[str], 
                               image_name: str, 
                               height_index: int = -1,
                               images_dir: str = "images", 
                               atm_subdir: str = "atmosphere") -> None:
    """
    Plots the atmospheric quantities and saves the plot as an image file.
    
    Parameters:
        atm_quant (np.ndarray): A 4D array where the last dimension represents different atmospheric quantities.
        image_name (str): The name of the image file to save.
        images_dir (str, optional): The directory where the image will be saved. Default is "images".
        atm_subdir (str, optional): The subdirectory within images_dir where the image will be saved. Default is "atmosphere".
    
    Returns:
    None
    Saves:
    A plot of the atmospheric quantities as an image file in thWWe specified directory.
    """
    
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Atmospheric Quantities', fontsize=16)
    cmaps = ['inferno', 'spring', 'seismic_r', 'PuOr', 'PuOr', 'PuOr']

    for i in range(6):
        ax[i // 3, i % 3].imshow(atm_quant[:, :, height_index , i], cmap=cmaps[i])
        ax[i // 3, i % 3].set_title(titles[i])
        ax[i // 3, i % 3].axis('off')
    
    images_dir = os.path.join(images_dir, atm_subdir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    image_path = os.path.join(images_dir, f"{image_name}.pdf")
    fig.savefig(image_path)

    print(f"Saved image to: {image_path}")
def plot_polarizations(stokes: np.ndarray, wl: np.ndarray,image_path: str = "images", image_name: str = "polarizations.pdf") -> None:
    """
    Function to plot the mean continuum, linear polarization, and circular polarization.
    Args:
        mean_continuum_data(np.ndarray): Array containing the mean continuum.
        linear_polarization_data(np.ndarray): Array containing the linear polarization.
        circular_polarization_data(np.ndarray): Array containing the circular polarization.
        filename(str): Name of the file to save the plot.
    Returns:
        None
    """

    mean_continuum_data, linear_polarization_data, circular_polarization_data = spectropolarimetry(stokes)
    # Calculate the root mean square (RMS) of the data
    def rms_calculation(data: np.ndarray) -> float:
        return np.sqrt(np.mean((data - data.mean())**2))/data.mean()
    
    rms = [rms_calculation(mean_continuum_data),
           rms_calculation(linear_polarization_data),
           rms_calculation(circular_polarization_data)]

    fig, ax = plt.subplots(1, 3, figsize=(3*4.3, 4.3))

    # Plot mean_continuum
    im1 = ax[0].imshow(mean_continuum_data, cmap='inferno')
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax1)

    # Plot old_norm_linear_polarization
    q_low, q_high = np.percentile(linear_polarization_data, [5, 95])
    im2 = ax[1].imshow(linear_polarization_data, cmap='RdYlGn_r', vmin=q_low, vmax=q_high)
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax2)

    # Plot old_norm_circular_polarization
    q_low, q_high= np.percentile(circular_polarization_data, [5, 95])
    im3 = ax[2].imshow(circular_polarization_data, cmap="PiYG", vmin=q_low, vmax=q_high)
    divider3 = make_axes_locatable(ax[2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax3)

    if rms:
        ax[0].set_title(f'Continuum (rms = {rms[0]:.2f})')
        ax[1].set_title(f'Linear Pol.(rms = {rms[1]:.2f})')
        ax[2].set_title(f'Circular Pol. (rms = {rms[2]:.2f})')
    else:
        ax[0].set_title('Continuum')
        ax[1].set_title('Linear Pol.')
        ax[2].set_title('Circular Pol.')
        

    plt.tight_layout()
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    images_output = os.path.join(image_path, image_name)
    fig.savefig(images_output)
##############################################################
# loading utils
##############################################################
def load_training_data(filenames: list[str], 
                       ptm: str = "./data",
                       noise_level: float = 5.9e-4, #Hinode noise
                       new_logtau: np.ndarray[float] = np.linspace(-2.5, 0, 20),
                       stokes_weights: list[int] = [1,10,10,10],
                       verbose: bool = False,
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess training data from a list of filenames.
    This function reads data from multiple files, processes it, and returns the processed data.
    Parameters:
        filenames (list[str]): List of file paths to load data from.
        n_spectral_points (int): Number of new points for spectral resolution degradation.
    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - atm_data (np.ndarray): Array of atmospheric quantities.
        - stokes_data (np.ndarray): Array of Stokes parameters.
        - mags_names (np.ndarray): Array of magnetic field names.
    """
    #Arrays for saving the whole dataset
    atm_data = []
    stokes_data = []

    for fln in filenames:
        #Creation of the MURaM object for each filename for charging the data.
        muram = MURaM(filename=fln, verbose = verbose, ptm = ptm)
        muram.charge_quantities()
        muram.just_LOS_components()
        muram.optical_depth_stratification(new_logtau=new_logtau)
        muram.spatial_convolution()
        muram.spectral_convolution()
        muram.spectral_sampling()
        muram.surface_pixel_sampling()
        muram.scale_quantities(stokes_weigths=stokes_weights)
        muram.spectral_noise(level_of_noise=noise_level)
        # Plot
        plot_polarizations(muram.stokes, muram.new_wl,
                            image_path = "images/fifth_experiment/stokes",
                            image_name= f"{muram.filename}_polarization_with_noise")
        
        pixel_x = 282
        pixel_y = 432
        resampled_pixel_x = int(pixel_x * muram.stokes.shape[0] / muram.nx)
        resampled_pixel_y = int(pixel_y * muram.stokes.shape[1] / muram.ny)
        plot_stokes(muram.stokes[resampled_pixel_x,resampled_pixel_y], muram.new_wl, 
                    stokes_titles = [r"$I/I_c$", r"$Q/I_c$", r"$U/I_c$", r"$V/I_c$"],
                    image_name=f"{muram.filename}_stokes_with_noise", 
                    images_dir="images/fifth_experiment")
        muram.intergran_gran_polariz_balance()

        atm_data.append(muram.atm_quant)
        stokes_data.append(muram.stokes)

    atm_data = np.concatenate(atm_data, axis=0)
    stokes_data = np.concatenate(stokes_data, axis=0)
    
    return atm_data, stokes_data, muram.new_wl
def load_data_cubes(filenames: list[str], 
                    ptm = "./data",
                    noise_level: float = 5.9e-4, #Hinode noise
                    new_logtau: np.ndarray[float] = np.linspace(-2.5, 0, 20),
                    stokes_weights: list[int] = [1,10,10,10],
                    verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads data cubes from a list of filenames and processes them using the MURaM class.
    Args:
        filenames (list[str]): List of file paths to load data from.
    Returns:
        tuple: A tuple containing:
            - atm_data (list[np.ndarray]): List of atmospheric data arrays.
            - stokes_data (list[np.ndarray]): List of Stokes parameter data arrays.
            - mags_names (np.ndarray): Array of magnetic field names.
            - phys_maxmin (np.ndarray): Array of physical maximum and minimum values.
    """
    #Arrays for saving the whole dataset
    atm_data = []
    stokes_data = []

    print("new_logtau:", new_logtau)
    for fln in filenames:
        #Creation of the MURaM object for each filename for charging the data.
        muram = MURaM(filename=fln, verbose = verbose, ptm = ptm)
        muram.charge_quantities()
        muram.just_LOS_components()
        muram.optical_depth_stratification(new_logtau=new_logtau)
        muram.spatial_convolution()
        muram.spectral_convolution()
        muram.spectral_sampling()
        muram.surface_pixel_sampling()
        muram.scale_quantities(stokes_weigths=stokes_weights)
        muram.spectral_noise(level_of_noise=noise_level)
        # Plot
        plot_polarizations(muram.stokes, muram.new_wl,
                            image_path = "images/fifth_experiment/stokes",
                            image_name= f"{muram.filename}_polarization_with_noise")
        
        pixel_x = 282
        pixel_y = 432
        resampled_pixel_x = int(pixel_x * muram.stokes.shape[0] / muram.nx)
        resampled_pixel_y = int(pixel_y * muram.stokes.shape[1] / muram.ny)
        plot_stokes(muram.stokes[resampled_pixel_x,resampled_pixel_y], muram.new_wl, 
                    stokes_titles = [r"$I/I_c$", r"$Q/I_c$", r"$U/I_c$", r"$V/I_c$"],
                    image_name=f"{muram.filename}_stokes_with_noise", 
                    images_dir="images/fifth_experiment")

        atm_data.append(muram.atm_quant)
        stokes_data.append(muram.stokes)
    
    return atm_data, stokes_data, muram.mags_names, muram.phys_maxmin, muram.new_nx, muram.new_ny
def create_dataloaders(stokes_data: np.ndarray,
                       atm_data: np.ndarray,
                       device: str,
                       batch_size: int = 80,
                       linear: bool = False,
                       stokes_as_channels: bool = False) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Creates PyTorch DataLoaders for training and testing datasets.
    Parameters:
    -----------
    stokes_data : np.ndarray
        Input data representing Stokes parameters.
    atm_data : np.ndarray
        Output data representing atmospheric parameters.
    device : str
        Device to which tensors will be moved (e.g., 'cpu' or 'cuda').
    batch_size : int, optional
        Number of samples per batch (default is 80).
    linear : bool, optional
        If True, flattens the input data along the external axes (default is False).
    stokes_as_channels : bool, optional
        If True, moves the Stokes parameter axis to the channel dimension (default is False).
    Returns:
    --------
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        A tuple containing the training DataLoader and the testing DataLoader.
    Notes:
    ------
    - The function splits the input and output data into training and testing sets.
    - Converts the numpy arrays to PyTorch tensors and moves them to the specified device.
    - Optionally reshapes the input data based on the `linear` and `stokes_as_channels` flags.
    - Creates TensorDataset objects for training and testing data.
    - Initializes DataLoader objects for both training and testing datasets.
    - Prints the shapes of the datasets and the lengths of the DataLoaders.
    """
    
    # Data splitting
    in_train, in_test, out_train, out_test = train_test_split(stokes_data, atm_data, test_size=0.33, random_state=42)

    #Converting to tensors
    in_train = torch.from_numpy(in_train).to(device).float()
    in_test = torch.from_numpy(in_test).to(device).float()
    out_train = torch.from_numpy(out_train).to(device).float()
    out_test = torch.from_numpy(out_test).to(device).float()
    
    #Flattening of the output external axes
    if linear:
        in_train = torch.reshape(in_train, (in_train.size()[0], in_train.size()[1]*in_train.size()[2]))
        in_test = torch.reshape(in_test, (in_test.size()[0], in_test.size()[1]*in_test.size()[2]))
    if stokes_as_channels:
        in_train = torch.moveaxis(in_train, 1,2)
        in_test = torch.moveaxis(in_test, 1,2)
    out_train = torch.reshape(out_train, (out_train.size()[0], out_train.size()[1]*out_train.size()[2]))
    out_test = torch.reshape(out_test, (out_test.size()[0], out_test.size()[1]*out_test.size()[2]))
    
    print("in_train shape:", in_train.size())
    print("out_train shape:", out_train.size())
    print("in_test shape:", in_test.size())
    print("out_test shape:", out_test.size())
    
    #Train and test datasets
    train_dataset = TensorDataset(in_train.to(device), out_train.to(device))
    test_dataset = TensorDataset(in_test.to(device), out_test.to(device))

    #Train and test dataloaders
    train_dataloader = DataLoader(train_dataset,
            batch_size=batch_size, # how manz samples per batch? 
            shuffle=True # shuffle data every epoch?
    )

    test_dataloader = DataLoader(test_dataset,
        batch_size=batch_size,
        shuffle=False # don't necessarily have to shuffle the testing data
    )    

    print(f"Length of train dataloader: {len(train_dataloader)} batches of {batch_size}")
    print(f"Length of test dataloader: {len(test_dataloader)} batches of {batch_size}")
    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    print(f"""
    Shape of each batch input and output:
    train input batch shape: {train_features_batch.shape}, 
    train output batch shape: {train_labels_batch.shape}
            """ )
    
    return train_dataloader, test_dataloader
