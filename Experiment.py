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
    
    ## DATA LOADING ###
    #filenames to be readed for creating the dataset
    filenames = []
    for i in range(80, 97):
       filenames.append(f"0{i}000")
    filenames.append("099000")
    for i in range(100, 113):
        filenames.append(f"{i}000")
    #filenames = ["080000", "085000", "090000"] #for initital testings
        
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")
    
    ### Checking linear vs convolutional 1d models ###
    
    m_type = "hybrid"
    epochs = 20
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
    
    #2. Create models
    scales = [1,2,4]
    los_model = InversionModel(scales=scales, 
                           nwl_points=len(wl_points),
                           n_outputs=atm_data.shape[-1]*len(new_logtau)).to(device).float()
    los_model.name = "LOS_model"

    #3. Set hyperparameters
    set_seeds()
    #Loss function
    loss_fn = nn.MSELoss() # this is also called "criterion"/"cost function" in some places
    #Optimizers
    optimizer = torch.optim.Adam(params=los_model.parameters(), lr=lr)
    
    #4. Train models
    
    # ----------------- General model -----------------
    experiment_name = f"only_LOS"

    #5. Create dataloaders
    train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                atm_data = atm_data,
                device = "cpu",  # Load data to CPU first to avoid GPU memory overflow
                batch_size = 80,
                linear = False)
    
    # ----------------- Training -----------------
    print("Running experiment: ", experiment_name)
    train(model=los_model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader, 
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
        writer=create_writer(experiment_name=experiment_name,
                            model_name=los_model.name,
                            extra=f""))
    
    #Save the model to file so we can get back the best model
    save_filepath = experiment_name + ".pth"
    save_model(model=los_model,
            target_dir="models/fifth_experiment",
            model_name=save_filepath)
    print("-"*50 + "\n")

if __name__ == "__main__":
    main()

