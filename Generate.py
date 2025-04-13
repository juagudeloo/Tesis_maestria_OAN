
from pathlib import Path
import torch
import numpy as np

#MODULES IMPORT
import sys
sys.path.append("../modules")
from modules.data_utils import load_data_cubes
from modules.nn_models import *
from modules.train_test_utils import charge_weights, generate_results, descale_atm, plot_surface_generated_atm, plot_od_generated_atm, plot_density_bars, plot_correlation


def main():
    
    
    #########################################################################################
    # Preprocessing
    #########################################################################################
    
    # Filenames of the snapshots to be calculated
    filenames = [
                 "087000"
                 ]
    
    #########################################################################################
    # Models and weights paths
    #########################################################################################
    
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")
    
    # Weights paths
    target_dir = Path("models/fourth_experiment")      
    #########################################################################################
    # Generation
    #########################################################################################
    nx = 480
    ny = 480
    n_spec_points = 112
    new_logtau = np.arange(-2.0, 0+0.1, 0.1)
    stokes_weights = [1,100,100,10]
    
    # Load data
    atm_data, stokes_data, mags_names, phys_maxmin = load_data_cubes(filenames, 
                                                                      n_spectral_points=n_spec_points,
                                                                      new_logtau=new_logtau,
                                                                      stokes_weights=stokes_weights)
    
    images_path = Path("images/fourth_experiment")
    if not images_path.exists():
        images_path.mkdir(parents=True)
    #Suface plots
    main_experiment = "One models"

    model_types = ["mscnn",
                   "cnn1d_4channels"]
    for i, filename in enumerate(filenames):
      # Descale atm data
      atm_data_original = np.reshape(np.copy(atm_data[i]), (nx, ny, new_logtau.shape[0],6))
      atm_data_original = descale_atm(atm_data_original, phys_maxmin)
      atm_data_original[..., 2] /= 1e5
      
      # Load model and charge corresponding stokes data
      stokes_data_original =  np.reshape(np.copy(stokes_data[i]), (stokes_data[i].shape[0]*stokes_data[i].shape[1], stokes_data[i].shape[2],stokes_data[i].shape[3]))
      stokes_data_original = np.moveaxis(stokes_data_original, 1, 2)
      
      for model_type in model_types:
        #3. Create models
        if model_type == "mscnn":
            scales = [1,2,4]
            model = MSCNN(scales=scales, 
                                nwl_points=n_spec_points,
                                n_outputs=6*len(new_logtau),
                                c1_filters=32,
                                c2_filters=64).to(device).float()
            model.name = "MSCNN"
            
        elif model_type == "cnn1d_4channels":
            hu = 1024
            model = CNN1D(4,6*len(new_logtau),hidden_units=hu, signal_length=n_spec_points).to(device)
            model.name = "CNN1D_4CHANNELS"

        print(f"Generating results for {model.name}...")
        weights_name = f"{model.name}.pth"
        charge_weights(model = model,
                        target_dir = target_dir,
                        weights_name = weights_name
                    )
        atm_generated = generate_results(model = model,
                                          stokes_data = stokes_data_original,
                                          atm_shape=(nx,ny,new_logtau.shape[0],6),
                                          maxmin = phys_maxmin,
                                          type_of_quantity=3,
                                          device = device
                                        )      
        
        print("atm generated shape:", atm_generated.shape)
        
        
        atm_generated[..., 2] /= 1e5
      
        ##################################
        # Plot generated atmospheres  
        ##################################
        
        print("Plotting generated atmospheres...")
        
        #OD plots

        model_subdir = main_experiment + "_" + model.name

        plot_od_generated_atm(
                          atm_generated = atm_generated,
                          atm_original = atm_data_original,
                          images_dir=images_path,
                          filename=filename,
                          model_subdir = model_subdir,
                          image_name = "mean_OD.png",
                          titles = mags_names,
                          tau=new_logtau,
                          )
        
        #Density bars
        tau_indices = range(0,new_logtau.shape[0])
        for itau in tau_indices:
          plot_surface_generated_atm(
                            atm_generated = atm_generated,
                            atm_original = atm_data_original,
                            images_dir=images_path,
                            filename=filename,
                            model_subdir = model_subdir,
                            surface_subdir= "surface_plots",
                            image_name = f"OD_surface.png",
                            titles = mags_names,
                            tau = new_logtau,
                            itau = itau
                          )
          plot_density_bars(
                  atm_generated = atm_generated,
                  atm_original = atm_data_original,
                  images_dir=images_path,
                  filename=filename,
                  dense_diag_subdir= "density_plots",
                  model_subdir = model_subdir,
                  image_name = "OD_density.png",
                  tau_index = itau,
                  tau=new_logtau,
                  titles = mags_names)
          plot_correlation(
                  atm_generated = atm_generated,
                  atm_original = atm_data_original,
                  images_dir=images_path,
                  filename=filename,
                  corr_diag_subdir= "correlation_plots",
                  model_subdir = model_subdir,
                  image_name = "OD_correlation.png",
                  tau=new_logtau,
                  tau_index = itau,
                  titles = mags_names)
    
print("Done!")

if __name__ == "__main__":
    main()