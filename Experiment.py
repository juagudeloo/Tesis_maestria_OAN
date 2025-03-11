import sys

from timeit import default_timer as timer 

import numpy as np

import torch
from torch import nn

#MODULES IMPORT
sys.path.append("../modules")
from modules.data_utils import load_training_data, create_dataloaders, plot_stokes
from modules.nn_models import InversionModel
from modules.train_test_utils import train, set_seeds, create_writer, save_model

#
def main():
    
    ### DATA LOADING ###
    #filenames to be readed for creating the dataset
    filenames = ["080000", 
                 "085000", "090000"
                 ]
        
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")
    
    ### Checking linear vs convolutional 1d models ###
    
    m_type = "hybrid"
    epochs = 10
    lr = 1e-3
    n_spec_points = 114
    new_logtau = np.array([-2.0, -0.8, 0.0])
    noise_level = 0.1
    
    stokes_weights = [1,7,7,2]
    
    #1. Loop through stokes weights
    atm_data, stokes_data, wl_points = load_training_data(filenames, 
                                                          ptm = "/scratchsan/observatorio/juagudeloo/data",
                                                            noise_level=noise_level,
                                                            n_spectral_points=n_spec_points,
                                                            new_logtau=new_logtau,
                                                            stokes_weights=stokes_weights)
    
    plot_stokes(stokes=stokes_data[0], 
                wl_points = wl_points,
                image_name = "example_stokes",)
    
    #2. Create dataloaders
    thermody_train_dataloader, thermody_test_dataloader = create_dataloaders(stokes_data = stokes_data,
                atm_data = atm_data[...,:3],
                device = device,
                batch_size = 80,
                linear = False)
    
    magn_train_dataloader, magn_test_dataloader = create_dataloaders(stokes_data = stokes_data,
                atm_data = atm_data[...,3:],
                device = device,
                batch_size = 80,
                linear = False)
    
    #3. Create models
    scales = [1,2,4]
    thermody_model = InversionModel(scales=scales, 
                           nwl_points=len(wl_points),
                           n_outputs=3*len(new_logtau)).to(device).float()
    thermody_model.name = "thermodynamic"
    
    magn_model = InversionModel(scales=scales, 
                           nwl_points=len(wl_points),
                           n_outputs=3*len(new_logtau)).to(device).float()
    magn_model.name = "magnetic_field"

    #4. Set hyperparameters
    set_seeds()
    #Loss function
    loss_fn = nn.MSELoss() # this is also called "criterion"/"cost function" in some places
    #Optimizers
    thermody_optimizer = torch.optim.Adam(params=thermody_model.parameters(), lr=lr)
    magn_optimizer = torch.optim.Adam(params=magn_model.parameters(), lr=lr)
    
    #5. Train models
    
    # ----------------- Thermodynamic model -----------------
    thermody_experiment_name = f"thermodynamic_unique"
    print("Running experiment: ", thermody_experiment_name)
    train(model=thermody_model,
        train_dataloader=thermody_train_dataloader,
        test_dataloader=thermody_test_dataloader, 
        optimizer=thermody_optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
        writer=create_writer(experiment_name=thermody_experiment_name,
                            model_name=thermody_model.name,
                            extra=f""))
    
    #Save the model to file so we can get back the best model
    save_filepath = thermody_experiment_name + ".pth"
    save_model(model=thermody_model,
            target_dir="models",
            model_name=save_filepath)
    print("-"*50 + "\n")
    
    # ----------------- Magnetic field model -----------------
    print("Running experiment: ", magn_experiment_name)
    magn_experiment_name = f"magnetic_field_unique"
    train(model=magn_model,
        train_dataloader=magn_train_dataloader,
        test_dataloader=magn_test_dataloader, 
        optimizer=magn_optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
        writer=create_writer(experiment_name=magn_experiment_name,
                            model_name=magn_model.name,
                            extra=f""))
    
    #Save the model to file so we can get back the best model
    save_filepath = thermody_experiment_name + ".pth"
    save_model(model=magn_model,
            target_dir="models",
            model_name=save_filepath)
    print("-"*50 + "\n")
    

           

if __name__ == "__main__":
    main()

