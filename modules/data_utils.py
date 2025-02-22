import numpy as np

from skimage import filters

from scipy.interpolate import interp1d

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader

import os

from tqdm import tqdm 

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    ny : int
        Number of grid points in the y-direction.
    nz : int
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

    def __init__(self, filename: str):
        """
        Initialize the MURaM object with the given filename.

        Parameters:
        -----------
        filename : str
            Name of the file to be processed.
        """
        self.ptm = Path("/scratchsan/observatorio/juagudeloo/data")
        self.filename = filename
        
        self.nlam = 300  # this parameter is useful when managing the self.stokes parameters
        self.nx = 480
        self.ny = 256
        self.od = 20  # height axis
        self.nz = 480

    def charge_quantities(self) -> None:
        """
        Load and process the atmospheric and Stokes quantities from the data files.
        """
        print(f"""
                ######################## 
                Reading {self.filename} MuRAM data...
                ######################## 
                      """)
        
        print("Charging temperature ...")
        mtpr = np.load(self.ptm / "opt_depth" / f"mtpr_logtau_{self.filename}.npy").flatten()
        print("mtpr shape:", mtpr.shape)
        
        print("Charging magnetic field vector...")
        mbxx = np.load(self.ptm / "opt_depth" / f"mbxx_logtau_{self.filename}.npy")
        mbyy = np.load(self.ptm / "opt_depth" / f"mbyy_logtau_{self.filename}.npy")
        mbzz = np.load(self.ptm / "opt_depth" / f"mbzz_logtau_{self.filename}.npy")
        
        coef = np.sqrt(4.0 * np.pi)  # cgs units conversion
        
        mbxx = mbxx * coef
        mbyy = mbyy * coef
        mbzz = mbzz * coef
        print("mbxx shape:", mbxx.shape)
        print("mbyy shape:", mbyy.shape)
        print("mbzz shape:", mbzz.shape)
        
        print("Charging density...")
        mrho = np.load(self.ptm / "opt_depth" / f"mrho_logtau_{self.filename}.npy")
        print("mrho shape:", mrho.shape)
        
        print("Charge velocity...")
        mvxx = np.load(self.ptm / "opt_depth" / f"mvxx_logtau_{self.filename}.npy")
        mvyy = np.load(self.ptm / "opt_depth" / f"mvyy_logtau_{self.filename}.npy")
        mvzz = np.load(self.ptm / "opt_depth" / f"mvzz_logtau_{self.filename}.npy")
        print("mvxx shape:", mvxx.shape)
        print("mvyy shape:", mvyy.shape)
        print("mvzz shape:", mbzz.shape)
        
        mvxx = mvxx / mrho
        mvyy = mvyy / mrho
        mvzz = mvzz / mrho
        
        print(f"""
                ######################## 
                Finished!
                ######################## 
                      """)

        print("Modifying magnetic field components to fight azimuth ambiguity...")
        mbqq = np.sign(mbxx**2 - mbzz**2) * np.sqrt(np.abs(mbxx**2 - mbzz**2))
        mbuu = np.sign(mbxx * mbzz) * np.sqrt(np.abs(mbxx * mbzz))
        mbvv = mbyy
        print("Quantities modified!")

        print("Creating atmosphere quantities array...")
        self.atm_quant = np.array([mtpr, mrho, mbqq, mbuu, mbvv, mvyy])
        self.atm_quant = np.moveaxis(self.atm_quant, 0, 1)
        self.atm_quant = np.reshape(self.atm_quant, (self.nx, self.od, self.nz, self.atm_quant.shape[-1]))
        self.atm_quant = np.moveaxis(self.atm_quant, 1, 2)
        print("Created!")
        print("atm_quant shape:", self.atm_quant.shape)

        print("Charging stokes vectors...")
        self.stokes = np.load(self.ptm / "stokes" / f"{self.filename}_prof.npy")
        self.I_63005 = self.stokes[:, :, 0, 0]  # Intensity map that is going to be used to balance intergranular and granular regions.
        print("Charged!")
        print("self.stokes shape", self.stokes.shape)

    def degrade_spec_resol(self, new_points: int) -> None:
        """
        Degrade the spectral resolution of the Stokes parameters.

        Parameters:
        -----------
        new_points : int, optional
            Number of new spectral points after degradation (default is 36).
        """
        # New spectral resolution arrays
        new_resol = np.linspace(0, 288, new_points, dtype=np.int64)
        new_resol = np.add(new_resol, 6)
        # File to save the degraded self.stokes
        new_stokes_out = self.ptm / "resampled_stokes" / f"resampled_self.stokes_f{self.filename}_sr{new_points}_wl_points.npy"
        
        # Degradation process
        if not os.path.exists(new_stokes_out):
            # Gaussian LSF kernel definition
            N_kernel_points = 13  # number of points of the kernel.
            def gauss(n: int = N_kernel_points, sigma: float = 1) -> np.ndarray:
                r = range(-int(n / 2), int(n / 2) + 1)
                return np.array([1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-float(x)**2 / (2 * sigma**2)) for x in r])
            g = gauss()
            
            # Convolution
            print("Degrading...")
            new_stokes = np.zeros((self.nx, self.nz, new_points, self.stokes.shape[-1]))
            
            for s in range(self.stokes.shape[-1]):
                for jx in tqdm(range(self.nx)):
                    for jz in range(self.nz):
                        spectrum = self.stokes[jx, jz, :, s]
                        resampled_spectrum = np.zeros(new_points)
                        i = 0
                        for center_wl in new_resol:
                            low_limit = center_wl - 6
                            upper_limit = center_wl + 7

                            if center_wl == 6:
                                shorten_spect = spectrum[0:13]
                            elif center_wl == 294:
                                shorten_spect = spectrum[-14:-1]
                            else:
                                shorten_spect = spectrum[low_limit:upper_limit]

                            resampled_spectrum[i] = np.sum(np.multiply(shorten_spect, g))
                            i += 1
                        new_stokes[jx, jz, :, s] = resampled_spectrum
            np.save(new_stokes_out, new_stokes)
        else:
            new_stokes = np.load(new_stokes_out)
            print("self.stokes degraded!")
        self.stokes = new_stokes
        print("The new self.stokes shape is:", self.stokes.shape)

        # These are the new wavelength values for the degraded resolution
        self.new_wl = (new_resol * 0.01) + 6300.5

    def scale_quantities(self) -> None:
        """
        Scale the atmospheric and Stokes quantities.
        """
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
        mbqq max = {np.max(self.atm_quant[:, :, :, 2])}
        mbuu max = {np.max(self.atm_quant[:, :, :, 3])}
        mbvv max = {np.max(self.atm_quant[:, :, :, 4])}
        mvyy max = {np.max(self.atm_quant[:, :, :, 5])}
            """)
        
        print(f"""
        MIN VALUES:
        mtpr min = {np.min(self.atm_quant[:, :, :, 0])}
        mrho min = {np.min(self.atm_quant[:, :, :, 1])}
        mbqq min = {np.min(self.atm_quant[:, :, :, 2])}
        mbuu min = {np.min(self.atm_quant[:, :, :, 3])}
        mbvv min = {np.min(self.atm_quant[:, :, :, 4])}
        mvyy min = {np.min(self.atm_quant[:, :, :, 5])}
            """) 
        
        print("Scaling the quantities...")
        # Atmosphere magnitudes scale factors
        self.phys_maxmin = {
            "T": [2e4, 0],
            "B": [3e3, -3e3],
            "Rho": [1e-5, 1e-10],
            "V": [1e6, -1e6]
        }

        # maxmin normalization function
        def norm_func(arr: np.ndarray, maxmin: list[float]) -> np.ndarray:
            max_val = maxmin[0]
            min_val = maxmin[1]
            return (arr - min_val) / (max_val - min_val)

        # Atmosphere magnitudes normalization
        self.atm_quant[:, :, :, 0] = norm_func(self.atm_quant[:, :, :, 0], self.phys_maxmin["T"])
        self.atm_quant[:, :, :, 1] = norm_func(self.atm_quant[:, :, :, 1], self.phys_maxmin["Rho"])
        self.atm_quant[:, :, :, 2] = norm_func(self.atm_quant[:, :, :, 2], self.phys_maxmin["B"])
        self.atm_quant[:, :, :, 3] = norm_func(self.atm_quant[:, :, :, 3], self.phys_maxmin["B"])
        self.atm_quant[:, :, :, 4] = norm_func(self.atm_quant[:, :, :, 4], self.phys_maxmin["B"])
        self.atm_quant[:, :, :, 5] = norm_func(self.atm_quant[:, :, :, 5], self.phys_maxmin["V"])
        
        # Stokes parameter normalization by the continuum
        scaled_stokes = np.ones_like(self.stokes)
        cont_indices = [0, 1, int(len(self.new_wl) / 2) - 1, int(len(self.new_wl) / 2), int(len(self.new_wl) / 2) + 1, -2, -1]
        wl_cont_values = self.new_wl[cont_indices]  # corresponding wavelength values to the selected continuum indices
        print("calculating the continuum...")
        for jx in tqdm(range(self.nx)):
            for jz in range(self.nz):
                for i in range(self.stokes.shape[-1]):
                    cont_values = self.stokes[jx, jz, cont_indices, 0]  # corresponding intensity values to the selected continuum indices
                    cont_model = interp1d(wl_cont_values, cont_values, kind="cubic")  # Interpolation applied over the assumed continuum values
                    scaled_stokes[jx, jz, :, i] = self.stokes[jx, jz, :, i] / cont_model(self.new_wl)
        self.stokes = scaled_stokes
        
        scaling_importance = [1, 10, 10, 10]  # Stokes parameters importance levels -> mapping Q, U and V to 0.1 of the intensity scale
        for i in range(len(scaling_importance)):
            self.stokes[:, :, :, i] = self.stokes[:, :, :, i] * scaling_importance[i]
            
        print("Scaled!")

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
        mbqq max = {np.max(self.atm_quant[:, :, :, 2])}
        mbuu max = {np.max(self.atm_quant[:, :, :, 3])}
        mbvv max = {np.max(self.atm_quant[:, :, :, 4])}
        mvyy max = {np.max(self.atm_quant[:, :, :, 5])}
            """)
        
        print(f"""
        MIN VALUES:
        mtpr min = {np.min(self.atm_quant[:, :, :, 0])}
        mrho min = {np.min(self.atm_quant[:, :, :, 1])}
        mbqq min = {np.min(self.atm_quant[:, :, :, 2])}
        mbuu min = {np.min(self.atm_quant[:, :, :, 3])}
        mbvv min = {np.min(self.atm_quant[:, :, :, 4])}
        mvyy min = {np.min(self.atm_quant[:, :, :, 5])}
            """)

    def gran_intergran_balance(self) -> None:
        """
        Balance the quantities of data from the granular and intergranular zones.
        """
        # Threshold definition
        thresh1 = filters.threshold_otsu(self.I_63005)
        
        # Mask extraction
        im_bin = self.I_63005 < thresh1
        gran_mask = np.ma.masked_array(self.I_63005, mask=im_bin).mask
        inter_mask = np.ma.masked_array(self.I_63005, mask=~im_bin).mask

        # Mask application
        atm_quant_gran = self.atm_quant[gran_mask]
        atm_quant_inter = self.atm_quant[inter_mask]
        stokes_gran = self.stokes[gran_mask]
        stokes_inter = self.stokes[inter_mask]
        len_inter = atm_quant_inter.shape[0]
        len_gran = atm_quant_gran.shape[0]

        #Leveraging the quantity of data from the granular and intergranular zones by a random dropping of elements of the greater zone.
        print("leveraging...")
        index_select  = []
        np.random.seed(50)
        if len_inter < len_gran:
            index_select = np.random.choice(range(len_gran), size = (len_inter,), replace = False)
            self.atm_quant = np.concatenate((atm_quant_gran[index_select], atm_quant_inter), axis = 0)
            self.stokes = np.concatenate((stokes_gran[index_select], stokes_inter), axis = 0)
        elif len_inter > len_gran:
            index_select = np.random.choice(range(len_inter), size = (len_gran,), replace = False)
            self.atm_quant = np.concatenate((atm_quant_gran, atm_quant_inter[index_select]), axis = 0)
            self.stokes = np.concatenate((stokes_gran, stokes_inter[index_select]), axis = 0)
            
        print(f"Shape after granular and intergranular balance:")
        print(f"atm_quant shape: {self.atm_quant.shape}")
        print(f"stokes shape: {self.stokes.shape}")
        print("Done")


##############################################################
# Plot utils
##############################################################

def plot_stokes(stokes: np.ndarray, 
                wl_points: np.ndarray, 
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
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    step_value = wl_points[1] - wl_points[0]
    fig.suptitle(f'Stokes Parameters (Step: {step_value:.2f} nm)', fontsize=16)
    ax[0].plot(wl_points, stokes[:, 0])
    ax[0].set_title("I")
    ax[1].plot(wl_points, stokes[:, 1])
    ax[1].set_title("Q")
    ax[2].plot(wl_points, stokes[:, 2])
    ax[2].set_title("U")
    ax[3].plot(wl_points, stokes[:, 3])
    ax[3].set_title("V")
    
    images_dir = os.path.join(images_dir, stokes_subdir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    image_path = os.path.join(images_dir, f"{image_name}_{len(wl_points)}_wl_points.pdf")
    fig.savefig(image_path)

    print(f"Saved image to: {image_path}")

def plot_atmosphere_quantities(atm_quant: np.ndarray, image_name: str, images_dir: str = "images", atm_subdir: str = "atmosphere") -> None:
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
    A plot of the atmospheric quantities as an image file in the specified directory.
    """
    
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Atmospheric Quantities', fontsize=16)
    
    titles = ["Temperature", "Density", "Magnetic Field QQ", "Magnetic Field UU", "Magnetic Field VV", "Velocity YY"]
    
    for i in range(6):
        ax[i // 3, i % 3].imshow(atm_quant[:, atm_quant.shape[2] // 2, : , i], cmap='viridis')
        ax[i // 3, i % 3].set_title(titles[i])
        ax[i // 3, i % 3].axis('off')
    
    images_dir = os.path.join(images_dir, atm_subdir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    image_path = os.path.join(images_dir, f"{image_name}_atm_quantities.pdf")
    fig.savefig(image_path)

    print(f"Saved image to: {image_path}")

##############################################################
# loading utils
##############################################################
def load_training_data(filenames: list[str], n_spectral_points: int = 36) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        muram = MURaM(filename=fln)
        muram.charge_quantities()
        muram.degrade_spec_resol(new_points=n_spectral_points)
        muram.scale_quantities()
        muram.gran_intergran_balance()

        atm_data.append(muram.atm_quant)
        stokes_data.append(muram.stokes)

    atm_data = np.concatenate(atm_data, axis=0)
    stokes_data = np.concatenate(stokes_data, axis=0)
    
    return atm_data, stokes_data, muram.mags_names, muram.new_wl

def load_data_cubes(filenames: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    for fln in filenames:
        #Creation of the MURaM object for each filename for charging the data.
        muram = MURaM(filename=fln)
        muram.charge_quantities()
        muram.degrade_spec_resol()
        muram.scale_quantities()

        atm_data.append(muram.atm_quant)
        stokes_data.append(muram.stokes)
    
    return atm_data, stokes_data, muram.mags_names, muram.phys_maxmin

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
            batch_size=batch_size, # how many samples per batch? 
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