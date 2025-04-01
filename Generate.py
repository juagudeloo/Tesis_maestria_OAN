
from pathlib import Path
import torch
import numpy as np

#MODULES IMPORT
import sys
sys.path.append("../modules")
from modules.data_utils import load_data_cubes
from modules.nn_models import SimpleLinearModel, SimpleCNN1DModel
from modules.train_test_utils import charge_weights, generate_results, descale_atm, plot_surface_generated_atm, plot_od_generated_atm, plot_density_bars


def main():
    
    
    #########################################################################################
    # Preprocessing
    #########################################################################################
    
    # Filenames of the snapshots to be calculated
    filenames = ["087000"]
    
    # Load data
    atm_data, stokes_data, mags_names, phys_maxmin = load_data_cubes(filenames)
    
    stokes_data_linear = np.reshape(np.copy(stokes_data[0]), (stokes_data[0].shape[0]*stokes_data[0].shape[1], stokes_data[0].shape[2]*stokes_data[0].shape[3]))
    #----------------------------------------------
    stokes_data_cnn_36channels = np.reshape(np.copy(stokes_data[0]), (stokes_data[0].shape[0]*stokes_data[0].shape[1], stokes_data[0].shape[2],stokes_data[0].shape[3]))
    #----------------------------------------------
    stokes_data_cnn_4channels = np.moveaxis(stokes_data_cnn_36channels, 2, 1)
    #----------------------------------------------
    atm_data_original = np.reshape(np.copy(atm_data[0]), (480,480,20,6))
    atm_data_original = descale_atm(atm_data_original, phys_maxmin)
    #----------------------------------------------
    
    # Model subdirectories names for the plots
    models_names = ["simple_linear", "simple_cnn1d_36channels", "simple_cnn1d_4channels"]
    
    stokes_data = {
      models_names[0]: stokes_data_linear,
      models_names[1]: stokes_data_cnn_36channels,
      models_names[2]: stokes_data_cnn_4channels,
      
    }
    
    #########################################################################################
    # Models and weights paths
    #########################################################################################
    
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")
    
    # Load models
    nn_models = {models_names[0]: SimpleLinearModel(36*4,6*20,hidden_units=2048).to(device), 
                 models_names[1]: SimpleCNN1DModel(36,6*20,hidden_units=1024, signal_length=4).to(device),
                 models_names[2]: SimpleCNN1DModel(4,6*20,hidden_units=72, signal_length=36).to(device),
                 }
    
    models_names = list(nn_models.keys())
    
    # Weights paths
    target_dir = Path("models/first_experiment")
    weight_names = [models_names[0] + "_2048_hidden_units_10_epochs.pth", 
                   models_names[1] + "_1024_hidden_units_10_epochs.pth",
                   models_names[2] + "_72_hidden_units_10_epochs.pth",
                   ]
    
    #########################################################################################
    # Generation
    #########################################################################################
    
    for i in range(len(nn_models.keys())):
            
      model = nn_models[models_names[i]]
      weights_name = weight_names[i]
      model_name = models_names[i]
      
      #Charge weights
      print(f"Charging weights from {weights_name}...")
      charge_weights(model = model,
                      target_dir = target_dir,
                      weights_name = weights_name
                   )
      
      #Generate results
      print(f"Generating results for {model_name}...")
      atm_generated = generate_results(model = model,
                                        stokes_data = stokes_data[model_name],
                                        maxmin = phys_maxmin,
                                        device = device
                                      )      
    
      ##################################
      # Plot generated atmospheres  
      ##################################
      
      print("Plotting generated atmospheres...")
      #Suface plots
      plot_surface_generated_atm(
                          atm_generated = atm_generated,
                          atm_original = atm_data_original,
                          images_dir = "images/first_experiment",
                          model_subdir = model_name,
                          image_name = "low_atm_surface.png",
                          titles = mags_names,
                          itau = 19
                        )
      plot_surface_generated_atm(
                          atm_generated = atm_generated,
                          atm_original = atm_data_original,
                          images_dir = "images/first_experiment",
                          model_subdir = model_name,
                          image_name = "high_atm_surface.png",
                          titles = mags_names,
                          itau = 8
                        )
    
      #OD plots
      
      plot_od_generated_atm(
                        atm_generated = atm_generated,
                        atm_original = atm_data_original,
                        images_dir = "images/first_experiment",
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
                images_dir = "images/first_experiment",
                model_subdir = model_name,
                image_name = "OD_density.png",
                tau_index = itau,
                titles = mags_names,
                num_bars = 100)
        
        
      print("Done!")
    
if __name__ == "__main__":
    main()
