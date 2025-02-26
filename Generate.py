
from pathlib import Path
import torch
import numpy as np

#MODULES IMPORT
import sys
sys.path.append("../modules")
from modules.data_utils import load_data_cubes
from modules.nn_models import LinearModel, CNN1DModel
from modules.train_test_utils import charge_weights, generate_results, descale_atm, plot_surface_generated_atm, plot_od_generated_atm, plot_density_bars


def main():
    
    
    #########################################################################################
    # Preprocessing
    #########################################################################################
    
    # Filenames of the snapshots to be calculated
    filenames = ["175000"]
    
    # Load data
    test_spectral_res = [36, 58, 90, 114
                         ]
    
    #########################################################################################
    # Models and weights paths
    #########################################################################################
    
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")
    
    # Load models
    nn_models = {}

    # Linear model
    # Weights paths
    target_dir = Path("models")      
    #########################################################################################
    # Generation
    #########################################################################################
    models_types = [
      "linear", 
      "cnn1d_4channels"
      ]
    
    for model_type in models_types:
      for n_spec_points in test_spectral_res:
        # Load data
        atm_data, stokes_data, mags_names, phys_maxmin = load_data_cubes(filenames, n_spectral_points=n_spec_points)
        
        # Descale atm data
        atm_data_original = np.reshape(np.copy(atm_data[0]), (480,480,20,6))
        atm_data_original = descale_atm(atm_data_original, phys_maxmin)
        
        # Load model and charge corresponding stokes data
        if model_type == "linear":
          stokes_data =  np.reshape(np.copy(stokes_data[0]), (stokes_data[0].shape[0]*stokes_data[0].shape[1], stokes_data[0].shape[2]*stokes_data[0].shape[3]))
          model = LinearModel(n_spec_points*4,6*20,hidden_units=2048).to(device)
        if model_type == "cnn1d_4channels":
          stokes_data =  np.reshape(np.copy(stokes_data[0]), (stokes_data[0].shape[0]*stokes_data[0].shape[1], stokes_data[0].shape[2],stokes_data[0].shape[3]))
          stokes_data =  np.moveaxis(stokes_data, 2, 1)
          model = CNN1DModel(4,6*20,hidden_units=72, signal_length=n_spec_points).to(device)
        
        model_name = model_type+"_"+str(n_spec_points)
        weights_name = model_name + "_spectral_points.pth"
                
        #Charge weights
        print(f"Charging weights from {weights_name}...")
        charge_weights(model = model,
                        target_dir = target_dir,
                        weights_name = weights_name
                    )
        
        #Generate results
        print(f"Generating results for {model_name}...")
        atm_generated = generate_results(model = model,
                                          stokes_data = stokes_data,
                                          maxmin = phys_maxmin,
                                          device = device
                                        )      
        
        # Convert velocity component from cm/s to km/s
        atm_generated[..., 5] /= 1e5
        atm_data_original[..., 5] /= 1e5
      
        ##################################
        # Plot generated atmospheres  
        ##################################
        
        print("Plotting generated atmospheres...")
        #Suface plots
        plot_surface_generated_atm(
                            atm_generated = atm_generated,
                            atm_original = atm_data_original,
                            model_subdir = model_name,
                            image_name = "low_atm_surface.png",
                            titles = mags_names,
                            itau = 19
                          )19
        plot_surface_generated_atm(
                            atm_generated = atm_generated,
                            atm_original = atm_data_original,
                            model_subdir = model_name,
                            image_name = "high_atm_surface.png",
                            titles = mags_names,
                            itau = 8
                          )
      
        #OD plots
        
        plot_od_generated_atm(
                          atm_generated = atm_generated,
                          atm_original = atm_data_original,
                          model_subdir = model_name,
                          image_name = "mean_OD.png",
                          titles = mags_names
                          )
        
        #Density bars
        tau_indices = range(0,20)
        for itau in tau_indices:
          plot_density_bars(
                  atm_generated = atm_generated,
                  atm_original = atm_data_original,
                  dense_diag_subdir= "density_plots",
                  model_subdir = model_name,
                  image_name = "OD_density.png",
                  tau_index = itau,
                  titles = mags_names,
                  num_bars = 100)
        
    print("Done!")
    
if __name__ == "__main__":
    main()