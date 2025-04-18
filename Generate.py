
from pathlib import Path
import torch
import numpy as np

#MODULES IMPORT
import sys
sys.path.append("../modules")
from modules.data_utils import load_data_cubes
from modules.nn_models import InversionModel
from modules.train_test_utils import charge_weights, generate_results, descale_atm, plot_surface_generated_atm, plot_od_generated_atm, plot_density_bars, plot_correlation, plot_correlation_along_od, plot_rmse_along_od


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
    target_dir = Path("models/fifth_experiment")      
    #########################################################################################
    # Generation
    #########################################################################################
    nx = 480
    ny = 480
    noise_level = 5.9e-4 #Hinode noise level (2016 - A. Lagg)
    new_logtau = np.arange(-2.0, 0+0.1, 0.1)
    stokes_weights = [1,100,100,10]
    
    # Load data
    atm_data, stokes_data, mags_names, phys_maxmin, new_nx, new_ny = load_data_cubes(filenames, 
                                                                      ptm = "/scratchsan/observatorio/juagudeloo/data",
                                                                      noise_level=noise_level,
                                                                      new_logtau=new_logtau,
                                                                      stokes_weights=stokes_weights)
    
    images_path = Path("images/fifth_experiment")
    if not images_path.exists():
        images_path.mkdir(parents=True)
    for i, filename in enumerate(filenames):
      # Descale atm data
      atm_data_original = np.reshape(np.copy(atm_data[i]), (new_nx, new_ny, new_logtau.shape[0],4))
      atm_data_original = descale_atm(atm_data_original, phys_maxmin)
      atm_data_original[..., 2] /= 1e5
      
      # Load model and charge corresponding stokes data
      stokes_data_original =  np.reshape(np.copy(stokes_data[i]), (stokes_data[i].shape[0]*stokes_data[i].shape[1], stokes_data[i].shape[2],stokes_data[i].shape[3]))
      stokes_data_original = np.moveaxis(stokes_data_original, 1, 2)
      
      #2. Create models
      scales = [1,2,4]
      los_model = InversionModel(scales=scales, 
                            nwl_points=stokes_data_original.shape[-1],
                            n_outputs=atm_data_original.shape[-1]*len(new_logtau)).to(device).float()
      los_model.name = "only_LOS"

      experiment_name = "Hinode_"+los_model.name
      weights_name = f"{experiment_name}.pth"

      print(f"Charging weights from {experiment_name}...")
      charge_weights(model = los_model,
                      target_dir = target_dir,
                      weights_name = weights_name
                  )
      
      #Generate results
      print(f"Generating results for {experiment_name}...")
      atm_generated = generate_results(model = los_model,
                                        stokes_data = stokes_data_original,
                                        atm_shape=(new_nx,new_ny,new_logtau.shape[0],atm_data_original.shape[-1]),
                                        maxmin = phys_maxmin,
                                        type_of_quantity=1,
                                        device = device
                                      )      
      
      atm_generated[..., 2] /= 1e5
    
      ##################################
      # Plot generated atmospheres  
      ##################################
      
      print("Plotting generated atmospheres...")
      
      #OD plots

      model_subdir = experiment_name

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
      
      plot_correlation_along_od(
                        atm_generated = atm_generated,
                        atm_original = atm_data_original,
                        images_dir=images_path,
                        corr_diag_subdir= "metrics_in_OD_plots",
                        filename=filename,
                        model_subdir = model_subdir,
                        image_name = "mean_OD_correlation.png",
                        titles = mags_names,
                        tau=new_logtau,
                        )
      plot_rmse_along_od(
                        atm_generated = atm_generated,
                        atm_original = atm_data_original,
                        images_dir=images_path,
                        rmse_diag_subdir= "metrics_in_OD_plots",
                        filename=filename,
                        model_subdir = model_subdir,
                        image_name = "mean_OD_rmse.png",
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