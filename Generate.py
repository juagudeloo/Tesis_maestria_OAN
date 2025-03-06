
from pathlib import Path
import torch
import numpy as np

#MODULES IMPORT
import sys
sys.path.append("../modules")
from modules.data_utils import load_data_cubes
from modules.nn_models import LinearModel, CNN1DModel
from modules.train_test_utils import charge_weights, generate_results, descale_atm, plot_surface_generated_atm, plot_od_generated_atm, plot_density_bars, plot_correlation


def main():
    
    
    #########################################################################################
    # Preprocessing
    #########################################################################################
    
    # Filenames of the snapshots to be calculated
    filenames = ["085000", "175000"]
    
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
    n_spec_points = 114
    new_logtau = np.array([-2.0, -0.8, 0.0])
    
    test_stokes_weights = [[1,1,1,1],
                           [1,10,10,10],
                           [1,20,20,10],
                           [1,40,40,10],
                           [1,80,80,10],
                           [1,160,160,10]]
    
    for stokes_weights in test_stokes_weights:
        print(f"Stokes weights: {stokes_weights}")
        # Load data
        atm_data, stokes_data, mags_names, phys_maxmin = load_data_cubes(filenames, 
                                                                  n_spectral_points=n_spec_points,
                                                                  new_logtau=new_logtau,
                                                                  stokes_weights=stokes_weights)
        for i in range(len(filenames)):
          filename = filenames[i]
          # Descale atm data
          atm_data_original = np.reshape(np.copy(atm_data[i]), (480,480,len(new_logtau),atm_data[i].shape[-1]))
          atm_data_original = descale_atm(atm_data_original, phys_maxmin)
          
          # Load model and charge corresponding stokes data
          stokes_original =  np.reshape(np.copy(stokes_data[i]), (stokes_data[i].shape[i]*stokes_data[i].shape[1], stokes_data[i].shape[2]*stokes_data[i].shape[3]))
          model = LinearModel(in_shape=n_spec_points*4,
                              out_shape=new_logtau.shape[0]*atm_data_original.shape[-1],
                              hidden_units=2048).to(device)
          
          experiment_name = f"{stokes_weights[0]}_{stokes_weights[1]}_{stokes_weights[2]}_{stokes_weights[3]}_stokes_weights"
          weights_name = experiment_name + ".pth"
                  
          #Charge weights
          print(f"Charging weights from {experiment_name}...")
          charge_weights(model = model,
                          target_dir = target_dir,
                          weights_name = weights_name
                      )
          
          #Generate results
          print(f"Generating results for {experiment_name}...")
          atm_generated = generate_results(model = model,
                                            stokes_data = stokes_original,
                                            atm_shape=atm_data_original.shape,
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
        
          #OD plots
          
          plot_od_generated_atm(
                            atm_generated = atm_generated,
                            atm_original = atm_data_original,
                            tau=new_logtau,
                            model_subdir = experiment_name,
                            image_name = "mean_OD.png",
                            titles = mags_names,
                            filename=filename
                            )
          
          #Density bars
          tau_indices = range(0,3)
          for itau in tau_indices:
            #Surface plots
            plot_surface_generated_atm(
                              atm_generated = atm_generated,
                              atm_original = atm_data_original,
                              tau=new_logtau,
                              filename=filename,
                              model_subdir = experiment_name,
                              surface_subdir="surface_plots",
                              image_name = "surface.png",
                              titles = mags_names,
                              itau = itau
                            )
            #Density bars
            plot_density_bars(
                    atm_generated = atm_generated,
                    atm_original = atm_data_original,
                    tau=new_logtau,
                    filename=filename,
                    dense_diag_subdir= "density_plots",
                    model_subdir = experiment_name,
                    image_name = "OD_density.png",
                    tau_index = itau,
                    titles = mags_names)
            #Correlation plots
            plot_correlation(
                    atm_generated = atm_generated,
                    atm_original = atm_data_original,
                    tau=new_logtau,
                    filename=filename,
                    corr_diag_subdir = "correlation_plots",
                    model_subdir = experiment_name,
                    image_name = "correlation.png",
                    titles = mags_names,
                    tau_index = itau)
        
            print(f"Done {filename}!")
    
if __name__ == "__main__":
    main()
