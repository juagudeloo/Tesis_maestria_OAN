import sys
from pathlib import Path

from timeit import default_timer as timer 

import numpy as np
import pandas as pd

import torch
from torch import nn

#MODULES IMPORT
sys.path.append("../modules")
from modules.data_utils import load_training_data, create_dataloaders, plot_stokes
from modules.nn_models import InversionModel
from modules.train_test_utils import train, set_seeds, create_writer, save_model
from modules.physics_constraints import WFAConstrainedLoss  # Import the WFA loss class

def main():
    
    # DATA LOADING ###
    #filenames to be readed for creating the dataset
    filenames = []
    for i in range(80, 97):
        if i in [87, 94, 97, 98]:
            continue
        else:
            filenames.append(f"0{i}000")
    filenames.append("099000")
    # for i in range(100, 111):
    #     filenames.append(f"{i}000")
    # filenames = ["080000", 
                #  "085000", "090000"
                #  ] #for initital testings
        
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")
    
    ### Checking linear vs convolutional 1d models ###
    
    m_type = "hybrid"
    epochs = 40
    lr = 1e-3
    new_logtau = np.arange(-2.0, 0+0.1, 0.1)
    noise_level = 5.9e-4 #Hinode noise level (2016 - A. Lagg)
    
    stokes_weights = [1,100,100,10]
    
    #1. Loop through stokes weights
    atm_data, stokes_data, wl_points = load_training_data(filenames, 
                                                          ptm = "/scratchsan/observatorio/juagudeloo/data",
                                                            noise_level=noise_level,
                                                            new_logtau=new_logtau,
                                                            stokes_weights=stokes_weights)
    stokes_data = np.moveaxis(stokes_data, 1, 2)

    #5. Create dataloaders
    train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                atm_data = atm_data,
                device = "cpu",  # Load data to CPU first to avoid GPU memory overflow
                batch_size = 80,
                linear = False)
    
    
    wfa_weigths = [0,1e-4, 1e-3, 1e-2, 0.1, 0.5]
    for epochs in [20,40]:
        for wfa_weight in wfa_weigths:
            #2. Create models
            scales = [1,2,4]
            los_model = InversionModel(scales=scales, 
                                nwl_points=len(wl_points),
                                n_outputs=atm_data.shape[-1]*len(new_logtau)).to(device).float()
            los_model.name = "only_LOS"

            #3. Set hyperparameters
            set_seeds()
            #Loss function
            # Set parameters for the Fe I 6301.5 spectral line
            wavelengths = torch.tensor(wl_points)
            blos_threshold = 1000.0 # Threshold for WFA constraint (in Gauss). Lower values than the threshold do not use it
            base_loss_threshold = 0.0004  # Threshold for base loss (in Gauss)

            # Create the WFA loss function
            wfa_loss = WFAConstrainedLoss(
                wavelengths=wavelengths,
                lambda0=6301.5,  # Rest wavelength in Angstroms
                g_factor=1.67,   # Landé factor for Fe I 6301.5
                wfa_weight=wfa_weight,
                wavelength_range=(20, 60),  # Indices for line region (adjust based on your data)
                base_loss="mse",
                blos_threshold=blos_threshold,  # Threshold for WFA constraint
                base_loss_threshold=base_loss_threshold  # Threshold for base loss
            )
            loss_fn = wfa_loss.to(device)  # Move the loss function to the same device as the model
            
            #Optimizers
            optimizer = torch.optim.Adam(params=los_model.parameters(), lr=lr)
            
            #4. Train models
            
            # ----------------- General model -----------------
            experiment_name = f"Hinode_"+los_model.name+"_WFA_constraint"+f"_wfa_weight_{wfa_weight}_base_loss_threshold_{base_loss_threshold}_epochs_{epochs}"
            print(f"Running experiment: {experiment_name}")
            
            # ----------------- Training -----------------
            print("Running experiment: ", experiment_name)
            results = train(model=los_model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader, 
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=epochs,
                device=device,
                writer=create_writer(experiment_name=experiment_name,
                                    model_name=los_model.name,
                                    extra=f""))
            
            metrics_path = Path(f"csv/fifth_experiment/{experiment_name}/")
            if not metrics_path.exists():
                metrics_path.mkdir(parents=True, exist_ok=True)
            # Save the metrics to CSV files
            loss_metrics_df = pd.DataFrame(results)
            loss_metrics_df.to_csv(metrics_path / f"loss_metrics.csv", index=False)


            #Save the model to file so we can get back the best model
            save_filepath = experiment_name + ".pth"
            save_model(model=los_model,
                    target_dir="models/fifth_experiment",
                    model_name=save_filepath)
            print("-"*50 + "\n")

if __name__ == "__main__":
    main()

