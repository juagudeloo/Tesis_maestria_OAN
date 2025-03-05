import sys

from timeit import default_timer as timer 

import numpy as np

import torch
from torch import nn

#MODULES IMPORT
sys.path.append("../modules")
from modules.data_utils import load_training_data, create_dataloaders, plot_stokes
from modules.nn_models import LinearModel, CNN1DModel
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
    
    m_type = "linear"
    epochs = 10
    lr = 1e-3
    n_spec_points = 114
    new_logtau = np.array([-2.0, -0.8, 0.0])
    
    test_stokes_weights = [[1,1,1,1],
                           [1,10,10,10],
                           [1,20,20,10],
                           [1,40,40,10],
                           [1,80,80,10],
                           [1,160,160,10]]
    
    #1. Loop through stokes weights
    for stokes_weights in test_stokes_weights:
        atm_data, stokes_data, wl_points = load_training_data(filenames, 
                                                                n_spectral_points=n_spec_points,
                                                                new_logtau=new_logtau,
                                                                stokes_weights=stokes_weights)
        
        plot_stokes(stokes=stokes_data[0], 
                    wl_points = wl_points,
                    image_name = "example_stokes",)
        
        # Create dataset and dataloaders
        train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                        atm_data = atm_data,
                        device = device,
                        batch_size = 80)
        

        print(f"Training {m_type} model with {n_spec_points} spectral points")
        train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                    atm_data = atm_data,
                    device = device,
                    batch_size = 80,
                    linear = True)
        
        hu = 2048
        model = LinearModel(n_spec_points*4,6*20,hidden_units=hu).to(device)
        model = model.float()
        #Loss function
        loss_fn = nn.MSELoss() # this is also called "criterion"/"cost function" in some places

        #Optimizer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        #Train model        
        
        experiment_name = f"{stokes_weights[0]}_{stokes_weights[1]}_{stokes_weights[2]}_{stokes_weights[3]}_stokes_weights"
        train(model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader, 
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            device=device,
            writer=create_writer(experiment_name=experiment_name,
                                model_name=m_type,
                                extra=f""))
        
        #Save the model to file so we can get back the best model
        save_filepath = experiment_name + ".pth"
        save_model(model=model,
                target_dir="models",
                model_name=save_filepath)
        print("-"*50 + "\n")

           

if __name__ == "__main__":
    main()

