import numpy as np

from skimage import filters

from scipy.interpolate import interp1d

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset

import os

from tqdm import tqdm 

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path

class MURaM():
    def __init__(self, filename):
        self.ptm = Path("/scratchsan/observatorio/juagudeloo/data")
        self.filename = filename
        
        nlam = 300 #this parameter is useful when managing the self.stokes parameters #wavelenght interval - its from 6300 amstroengs in steps of 10 amstroengs
        self.nx = 480
        self.ny = 256 #height axis
        self.nz = 480
    def charge_quantities(self):
        print(f"""
                ######################## 
                Reading {self.filename} MuRAM data...
                ######################## 
                      """)
        
        print("Charging temperature ...")
        mtpr = np.load(self.ptm / f"mtpr_{self.filename}.npy").flatten()
        print("mtpr shape:", mtpr.shape)
        
        print("Charging magnetic field vector...")
        mbxx = np.load(self.ptm / f"mbxx_{self.filename}.npy")
        mbyy = np.load(self.ptm / f"mbyy_{self.filename}.npy")
        mbzz = np.load(self.ptm / f"mbzz_{self.filename}.npy")
        
        coef = np.sqrt(4.0*np.pi) #cgs units conversion300
        
        mbxx=mbxx*coef
        mbyy=mbyy*coef
        mbzz=mbzz*coef
        print("mbxx shape:", mbxx.shape)
        print("mbyy shape:", mbyy.shape)
        print("mbzz shape:", mbzz.shape)
        
        print("Charging density...")
        mrho = np.load(self.ptm / f"mrho_{self.filename}.npy")
        print("mrho shape:", mrho.shape)
        
        print("Charge velocity...")
        mvxx = np.load(self.ptm / f"mvxx_{self.filename}.npy")
        mvyy = np.load(self.ptm / f"mvyy_{self.filename}.npy")
        mvzz = np.load(self.ptm / f"mvzz_{self.filename}.npy")
        print("mvxx shape:", mvxx.shape)
        print("mvyy shape:", mvyy.shape)
        print("mvzz shape:", mbzz.shape)
        
        mvxx = mvxx/mrho
        mvyy = mvyy/mrho
        mvzz = mvzz/mrho
        
        print(f"""
                ######################## 
                Finished!
                ######################## 
                      """)

        print("Modifying magnetic field components to fight azimuth ambiguity...")
        mbqq = np.sign(mbxx**2 - mbzz**2)*np.sqrt(np.abs(mbxx**2 - mbzz**2))
        mbuu = np.sign(mbxx*mbzz)*np.sqrt(np.abs(mbxx*mbzz))
        mbvv = mbyy
        print("Quantities modified!")

        print("Creating atmosphere quantities array...")
        self.atm_quant = np.array([mtpr, mrho, mbqq, mbuu, mbvv, mvyy])
        self.atm_quant = np.moveaxis(self.atm_quant, 0, 1)
        self.atm_quant = np.reshape(self.atm_quant, (self.nx,self.ny,self.nz,self.atm_quant.shape[-1]))
        self.atm_quant = np.moveaxis(self.atm_quant, 1, 2)
        print("Created!")
        print("self.atm_quant shape:", self.atm_quant.shape)

        print("Charging self.stokes vectors...")
        self.stokes = np.load(self.ptm / f"{self.filename}_prof.npy")
        self.I_63005 = self.stokes[:,:,0,0] ## Intensity map that is going to be used to balance intergranular and granular regions.
        print("Charged!")
        print("self.stokes shape", self.stokes.shape)

    def optical_depth_stratification(self):
        opt_len = 20 #Number of optical depth nodes
        mags_names = ["T", "rho", "Bq", "Bu", "Bv", "vy"] # atm quantities
        print("Applying optical depth stratification...")
        opt_depth = np.load(self.ptm / f"optical_depth_{self.filename}.npy")
        #optical depth points
        tau_out = self.ptm / f"array_of_tau_{self.filename}_{opt_len}_depth_points.npy"
        tau = np.linspace(-3, 1, opt_len)
        np.save(tau_out, tau)
        
        #optical stratification
        opt_mags_interp = {}
        opt_mags = np.zeros((self.nx, self.nz, opt_len, self.atm_quant.shape[-1]))
        opt_mags_out =self.ptm / f"optical_stratified_atm_modified_mbvuq_{self.filename}_{opt_len}_depth_points_{self.atm_quant.shape[-1]}_components.npy"
        if not os.path.exists(opt_mags_out):
            for jx in tqdm(range(self.nx)):
                    for jz in range(self.nz):
                        for i in range(len(mags_names)):
                            opt_mags_interp[mags_names[i]] = interp1d(opt_depth[jx,:, jz], self.atm_quant[jx, jz,:,i])
                            opt_mags[jx, jz,:,i] = opt_mags_interp[mags_names[i]](tau)
            np.save(opt_mags_out, opt_mags)
        else:
            opt_mags = np.load(opt_mags_out)
        self.atm_quant = opt_mags
        print(opt_mags.shape)
    def degrade_spec_resol(self):
        #Number of spectral points
        new_points = 36
        # New spectral resolution arrays
        new_resol = np.linspace(0,288,new_points, dtype=np.int64)
        new_resol = np.add(new_resol, 6)
        #File to save the degraded self.stokes
        new_stokes_out = self.ptm / f"resampled_self.stokes_f{self.filename}_sr{new_points}_wl_points.npy"
        
        #Degradation process
        if not os.path.exists(new_stokes_out):
        
            # Gaussian LSF kernel definition
            N_kernel_points = 13 # number of points of the kernel.
            def gauss(n=N_kernel_points,sigma=1):
                r = range(-int(n/2),int(n/2)+1)
                return np.array([1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in r])
            g = gauss()
        
            
        
            #Convolution
            print("Degrading...")
            new_stokes = np.zeros((self.nx, self.nz, new_points, self.stokes.shape[-1]))
            
            for s in range(self.stokes.shape[-1]):
                for jx in tqdm(range(self.nx)):
                    for jz in range(self.nz):
                        spectrum = self.stokes[jx, jz,:,s]
                        resampled_spectrum = np.zeros(new_points)
                        i = 0
                        for center_wl in new_resol:
                            low_limit = center_wl-6
                            upper_limit = center_wl+7
        
                            if center_wl == 6:
                                shorten_spect = spectrum[0:13]
                            elif center_wl == 294:
                                shorten_spect = spectrum[-14:-1]
                            else:
                                shorten_spect = spectrum[low_limit:upper_limit]
        
                            resampled_spectrum[i] = np.sum(np.multiply(shorten_spect,g))
                            i += 1
                        new_stokes[jx, jz,:,s] = resampled_spectrum
            np.save(new_stokes_out, new_stokes)
        else:
            new_stokes = np.load(new_stokes_out)
            print("self.stokes degraded!")
        self.stokes = new_stokes
        print("The new self.stokes shape is:", self.stokes.shape)

    def scale_quantities(self):

        print(f""" self.stokes:
        I_max = {np.max(self.stokes[:,:,:,0])}
        Q_max = {np.max(self.stokes[:,:,:,1])}
        U_max = {np.max(self.stokes[:,:,:,2])}
        V_max = {np.max(self.stokes[:,:,:,3])}
        I_min = {np.min(self.stokes[:,:,:,0])}
        Q_min = {np.min(self.stokes[:,:,:,1])}
        U_min = {np.min(self.stokes[:,:,:,2])}
        V_min = {np.min(self.stokes[:,:,:,3])}
        """)
        
        print(f"""
        MAX VALUES:
        mtpr max = {np.max(self.atm_quant[:,:,:,0])}
        mrho max = {np.max(self.atm_quant[:,:,:,1])}
        mbqq max = {np.max(self.atm_quant[:,:,:,2])}
        mbuu max = {np.max(self.atm_quant[:,:,:,3])}
        mbvv max = {np.max(self.atm_quant[:,:,:,4])}
        mvyy max = {np.max(self.atm_quant[:,:,:,5])}
            """)
        
        print(f"""
        MIN VALUES:
        mtpr min = {np.min(self.atm_quant[:,:,:,0])}
        mrho min = {np.min(self.atm_quant[:,:,:,1])}
        mbqq min = {np.min(self.atm_quant[:,:,:,2])}
        mbuu min = {np.min(self.atm_quant[:,:,:,3])}
        mbvv min = {np.min(self.atm_quant[:,:,:,4])}
        mvyy min = {np.min(self.atm_quant[:,:,:,5])}
            """) 
        
        print("Scaling the quantities...")
        #Atmosphere magnitudes scale factors
        phys_maxmin = {}
        phys_maxmin["T"] = [2e4, 0]
        phys_maxmin["B"] = [3e3, -3e3]
        phys_maxmin["Rho"] = [1e-5, 1e-10]
        phys_maxmin["V"] = [1e6, -1e6]

        #maxmin normalization function
        def norm_func(arr, maxmin):
            max_val = maxmin[0]
            min_val = maxmin[1]
            return (arr-min_val)/(max_val-min_val)

        #Atmosphere magnitudes normalization
        self.atm_quant[:,:,:,0] = norm_func(self.atm_quant[:,:,:,0], phys_maxmin["T"])
        
        self.atm_quant[:,:,:,1] = norm_func(self.atm_quant[:,:,:,1], phys_maxmin["Rho"])
        
        self.atm_quant[:,:,:,2] = norm_func(self.atm_quant[:,:,:,2], phys_maxmin["B"])
        self.atm_quant[:,:,:,3] = norm_func(self.atm_quant[:,:,:,3], phys_maxmin["B"])
        self.atm_quant[:,:,:,4] = norm_func(self.atm_quant[:,:,:,4], phys_maxmin["B"])
        
        self.atm_quant[:,:,:,5] = norm_func(self.atm_quant[:,:,:,5], phys_maxmin["V"])
        
        #self.stokes parameter normalization by the continuum
        scaled_stokes = np.ones_like(self.stokes)
        for jx in range(self.nx):
            for jz in range(self.nz):
                for i in range(self.stokes.shape[-1]):
                    cont_val = np.mean(self.stokes[jx, jz,:,0])
                    scaled_stokes[jx, jz,:,i] = self.stokes[jx, jz,:,i]/cont_val
        self.stokes = scaled_stokes
        print("Scaled!")

        print(f""" self.stokes:
        I_max = {np.max(self.stokes[:,:,:,0])}
        Q_max = {np.max(self.stokes[:,:,:,1])}
        U_max = {np.max(self.stokes[:,:,:,2])}
        V_max = {np.max(self.stokes[:,:,:,3])}
        I_min = {np.min(self.stokes[:,:,:,0])}
        Q_min = {np.min(self.stokes[:,:,:,1])}
        U_min = {np.min(self.stokes[:,:,:,2])}
        V_min = {np.min(self.stokes[:,:,:,3])}
        """)
        
        print(f"""
        MAX VALUES:
        mtpr max = {np.max(self.atm_quant[:,:,:,0])}
        mrho max = {np.max(self.atm_quant[:,:,:,1])}
        mbqq max = {np.max(self.atm_quant[:,:,:,2])}
        mbuu max = {np.max(self.atm_quant[:,:,:,3])}
        mbvv max = {np.max(self.atm_quant[:,:,:,4])}
        mvyy max = {np.max(self.atm_quant[:,:,:,5])}
            """)
        
        print(f"""
        MIN VALUES:
        mtpr min = {np.min(self.atm_quant[:,:,:,0])}
        mrho min = {np.min(self.atm_quant[:,:,:,1])}
        mbqq min = {np.min(self.atm_quant[:,:,:,2])}
        mbuu min = {np.min(self.atm_quant[:,:,:,3])}
        mbvv min = {np.min(self.atm_quant[:,:,:,4])}
        mvyy min = {np.min(self.atm_quant[:,:,:,5])}
            """) 
    def gran_intergran_balance(self):
        #Threshold definition
        thresh1 = filters.threshold_otsu(self.I_63005)
        
        #Mask extraction
        im_bin = self.I_63005<thresh1
        gran_mask =  np.ma.masked_array(self.I_63005, mask=im_bin).mask
        inter_mask = np.ma.masked_array(self.I_63005, mask=~im_bin).mask

        #Mask application
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
        print("Done")