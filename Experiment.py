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
    
    model_types = ["linear",
                   "cnn1d_4channels"]
    epochs = 10
    lr = 1e-3
    n_spec_points = 112
    new_logtau = np.array([-2.0, -0.8, 0.0])
    
    test_stokes_weights = [[1,1,1,1],
                           [1,10,10,10],
                           [1,100,100,10],]
    
    #1. Loop through stokes weights
    for stokes_weights in test_stokes_weights:
        atm_data, stokes_data, wl_points = load_training_data(filenames, 
                                                                    n_spectral_points=n_spec_points,
                                                                    new_logtau=new_logtau,
                                                                    stokes_weights=stokes_weights)
        plot_stokes(stokes=stokes_data[0], 
                        wl_points = wl_points,
                        image_name = "example_stokes",)
        
        for m_type in model_types:
            print(f"Training {m_type} model with {n_spec_points} spectral points")
            if m_type == "linear":
                train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                            atm_data = atm_data,
                            device = device,
                            batch_size = 80,
                            linear = True)
                hu = 1024
                model = LinearModel(n_spec_points*4,6*3,hidden_units=hu).to(device)
            elif m_type == "cnn1d_4channels":
                train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                            atm_data = atm_data,
                            device = device,
                            batch_size = 80,
                            stokes_as_channels=True,
                            linear = False)
                hu = 1024
                model = CNN1DModel(4,6*3,hidden_units=hu, signal_length=n_spec_points).to(device)            
            
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
            save_filepath = m_type + "_" + experiment_name + ".pth"
            save_model(model=model,
                    target_dir="models/third_experiment/",
                    model_name=save_filepath)
            print("-"*50 + "\n")

           

if __name__ == "__main__":
    main()

