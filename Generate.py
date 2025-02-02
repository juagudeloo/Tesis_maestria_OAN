
from pathlib import Path
import torch
import numpy as np

#MODULES IMPORT
import sys
sys.path.append("../modules")
from modules.data_utils import load_data_cubes
from modules.nn_models import SimpleLinearModel, SimpleCNN1DModel
from modules.train_test_utils import charge_weights, generate_results, descale_atm, plot_surface_generated_atm, plot_od_generated_atm


def main():
    
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")
    
    # Load models
    nn_models = {"SimpleLinear": SimpleLinearModel(36*4,6*20,hidden_units=2048).to(device), 
                 "SimpleCNN1D": SimpleCNN1DModel(36,6*20,hidden_units=1024).to(device)}
    
    # Filenames of the snapshots to be calculated
    filenames = ["175000"]
    
    # Load weights
    target_dir = Path("models")
    model_names = ["SimpleLinear_2048_hidden_units_10_epochs.pth", 
                   "SimpleCNN1D_1024_hidden_units_10_epochs.pth"
                   ]
    
    charge_weights(model = nn_models["SimpleLinear"],
                     target_dir = target_dir,
                     model_name = model_names[0]
                   )
                   
    charge_weights(model = nn_models["SimpleCNN1D"],
                     target_dir = target_dir,
                     model_name = model_names[1]
                   )
    
    # Load data
    atm_data, stokes_data, mags_names, phys_maxmin = load_data_cubes(filenames)
    
    stokes_data_plot = np.reshape(np.copy(stokes_data[0]), (stokes_data[0].shape[0],stokes_data[0].shape[1], stokes_data[0].shape[2],stokes_data[0].shape[3]))
    stokes_data_linear = np.reshape(np.copy(stokes_data[0]), (stokes_data[0].shape[0]*stokes_data[0].shape[1], stokes_data[0].shape[2]*stokes_data[0].shape[3]))
    stokes_data_cnn = np.reshape(np.copy(stokes_data[0]), (stokes_data[0].shape[0]*stokes_data[0].shape[1], stokes_data[0].shape[2],stokes_data[0].shape[3]))
    atm_data_original = np.reshape(np.copy(atm_data[0]), (480,480,20,6))
    atm_data_original = descale_atm(atm_data_original, phys_maxmin)
    # Generate results
    atm_generated = {"SimpleLinear": generate_results(model = nn_models["SimpleLinear"],
                                                      stokes_data = stokes_data_linear,
                                                      maxmin = phys_maxmin,
                                                      device = device
                                                     ),
                     "SimpleCNN1D": generate_results(model = nn_models["SimpleCNN1D"],
                                                      stokes_data = stokes_data_cnn,
                                                      maxmin = phys_maxmin,
                                                      device = device
                                                     )
                    }
    
    ##################################
    # Plot generated atmospheres  
    ##################################
    
    #Suface plots
    plot_surface_generated_atm(
                       atm_generated = atm_generated["SimpleLinear"],
                       atm_original = atm_data_original,
                       images_name = "atm_surface_SimpleLinear.png"
                      )

    plot_surface_generated_atm(
                        atm_generated = atm_generated["SimpleCNN1D"],
                        atm_original = atm_data_original,
                        images_name = "atm_surface_SimpleCNN1D.png"
                        )
    
    #OD plots
    
    plot_od_generated_atm(
                        stokes_data = stokes_data_plot,
                       atm_generated = atm_generated["SimpleLinear"],
                       atm_original = atm_data_original,
                       images_name = "atm_od_SimpleLinear_intergranular.png",
                       ix = 130,
                       iy = 50
                      )

    plot_od_generated_atm(
                        stokes_data = stokes_data_plot,
                        atm_generated = atm_generated["SimpleCNN1D"],
                        atm_original = atm_data_original,
                        images_name = "atm_od_SimpleCNN1D_intergranular.png",
                        ix = 50,
                        iy = 130
                        )
    
    
    plot_od_generated_atm(
                        stokes_data = stokes_data_plot,
                       atm_generated = atm_generated["SimpleLinear"],
                       atm_original = atm_data_original,
                       images_name = "atm_od_SimpleLinear_granular.png",
                       ix = 250,
                       iy = 250
                      )

    plot_od_generated_atm(
                        stokes_data = stokes_data_plot,
                        atm_generated = atm_generated["SimpleCNN1D"],
                        atm_original = atm_data_original,
                        images_name = "atm_od_SimpleCNN1D_granular.png",
                        ix = 250,
                        iy = 250
                        )
    
if __name__ == "__main__":
    main()