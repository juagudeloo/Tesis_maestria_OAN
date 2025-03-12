
from pathlib import Path
import torch
import numpy as np

#MODULES IMPORT
import sys
sys.path.append("../modules")
from modules.data_utils import load_data_cubes
from modules.nn_models import InversionModel
from modules.train_test_utils import charge_weights, generate_results, descale_atm, plot_surface_generated_atm, plot_od_generated_atm, plot_density_bars, plot_correlation


def main():
    
    
    #########################################################################################
    # Preprocessing
    #########################################################################################
    
    # Filenames of the snapshots to be calculated
    filenames = ["080000", 
                 "175000"
                 ]
    
    #########################################################################################
    # Models and weights paths
    #########################################################################################
    
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")
    
    # Weights paths
    target_dir = Path("models")      
    #########################################################################################
    # Generation
    #########################################################################################
    model_type = "linear"
    nx = 480
    ny = 480
    n_spec_points = 114
    new_logtau = np.array([-2.0, -0.8, 0.0])
    stokes_weights = [1,7,7,2]
    
    # Load data
    atm_data, stokes_data, mags_names, phys_maxmin = load_data_cubes(filenames, 
                                                                    ptm = "/scratchsan/observatorio/juagudeloo/data",
                                                                    n_spectral_points=n_spec_points,
                                                                    new_logtau=new_logtau,
                                                                    stokes_weights=stokes_weights)
    
    # Descale atm data
    atm_data_original = np.reshape(np.copy(atm_data[0]), (nx, ny, new_logtau.shape[0],6))
    atm_data_original = descale_atm(atm_data_original, phys_maxmin)
    
    # Load model and charge corresponding stokes data
    stokes_data =  np.reshape(np.copy(stokes_data[0]), (stokes_data[0].shape[0]*stokes_data[0].shape[1], stokes_data[0].shape[2],stokes_data[0].shape[3]))
    stokes_data = np.moveaxis(stokes_data, 1, 2)
    
    scales = [1,2,4] #Coarse-grain scales
    # ----------------- Thermodynamic model -----------------
    thermody_experiment_name = f"thermodynamic_unique"
    print("Running experiment: ", thermody_experiment_name)
    thermody_model = InversionModel(scales=scales, 
                           nwl_points=n_spec_points,
                           n_outputs=3*len(new_logtau)).to(device).float()
    thermody_model.name = "thermodynamic"
    
    thermody_weights_name = thermody_experiment_name + ".pth"
    #Charge weights
    print(f"Charging weights from {thermody_weights_name}...")
    charge_weights(model = thermody_model,
                    target_dir = target_dir,
                    weights_name = thermody_weights_name
                )
    
    #Generate results
    print(f"Generating results for {thermody_experiment_name}...")
    thermody_generated = generate_results(model = thermody_model,
                                      stokes_data = stokes_data,
                                      atm_shape=(nx,ny,new_logtau.shape[0],3),
                                      maxmin = phys_maxmin,
                                      type_of_quantity=1,
                                      device = device
                                    )      
    
    # ----------------- Magnetic model -----------------
    magn_experiment_name = f"magnetic_field_unique"
    magn_model = InversionModel(scales=scales, 
                           nwl_points=n_spec_points,
                           n_outputs=3*len(new_logtau)).to(device).float()
    magn_model.name = "magnetic_field"
    magn_weights_name = magn_experiment_name + ".pth"
    
    #Charge weights
    print(f"Charging weights from {magn_experiment_name}...")
    charge_weights(model = magn_model,
                    target_dir = target_dir,
                    weights_name = magn_weights_name
                )
    
    
    #Generate results
    print(f"Generating results for {magn_experiment_name}...")
    magnetic_generated = generate_results(model = magn_model,
                                      stokes_data = stokes_data,
                                      atm_shape=(nx,ny,new_logtau.shape[0],3),
                                      maxmin = phys_maxmin,
                                      type_of_quantity=2,
                                      device = device
                                    )      
    # ----------------------------------------------------
    
    # Convert velocity component from cm/s to km/s
    
    atm_generated = np.concatenate([thermody_generated, magnetic_generated], axis = -1)
    print("atm generated shape:", atm_generated.shape)
    
    
    atm_generated[..., 2] /= 1e5
    atm_data_original[..., 2] /= 1e5
  
    ##################################
    # Plot generated atmospheres  
    ##################################
    
    print("Plotting generated atmospheres...")
    #Suface plots
    model_subdir = "Two models"
    #OD plots
    plot_od_generated_atm(
                      atm_generated = atm_generated,
                      atm_original = atm_data_original,
                      model_subdir = model_subdir,
                      image_name = "mean_OD.png",
                      titles = mags_names,
                      tau=new_logtau
                      )
    
    #Density bars
    tau_indices = range(0,new_logtau.shape[0])
    for itau in tau_indices:
      plot_surface_generated_atm(
                        atm_generated = atm_generated,
                        atm_original = atm_data_original,
                        model_subdir = model_subdir,
                        image_name = f"OD_surface_{itau}.png",
                        titles = mags_names,
                        tau = new_logtau,
                        itau = 2
                      )
      plot_density_bars(
              atm_generated = atm_generated,
              atm_original = atm_data_original,
              dense_diag_subdir= "density_plots",
              model_subdir = model_subdir,
              image_name = "OD_density.png",
              tau_index = itau,
              tau=new_logtau,
              titles = mags_names)
      plot_correlation(
              atm_generated = atm_generated,
              atm_original = atm_data_original,
              dense_diag_subdir= "density_plots",
              model_subdir = model_subdir,
              image_name = "OD_density.png",
              tau=new_logtau,
              tau_index = itau,
              titles = mags_names)
    
print("Done!")

if __name__ == "__main__":
    main()