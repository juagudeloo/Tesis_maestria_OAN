import sys

from timeit import default_timer as timer 

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
    
    model_types = [
        "linear",
        "cnn1d_4channels",
        "cnn1d_36channels"
        ]
    epochs = 10
    lr = 1e-3
    test_spectral_res = [36, 75, 112]
    
    #1. Loop through spectral resolutions
    for n_spec_points in test_spectral_res:
        atm_data, stokes_data, wl_points = load_training_data(filenames, n_spectral_points=n_spec_points)
        
        plot_stokes(stokes=stokes_data[0], 
                    wl_points = wl_points,
                    image_name = "example_stokes",)
        
        # Create dataset and dataloaders
        train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                        atm_data = atm_data,
                        device = device,
                        batch_size = 80)
        
        #2. Loop through model types
        for m_type in model_types:
            print(f"Training {m_type} models")
            if m_type == "linear":
                train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                            atm_data = atm_data,
                            device = device,
                            batch_size = 80,
                            linear = True)
                hu = 1024
                model = LinearModel(n_spec_points*4,6*20,hidden_units=hu).to(device)
            elif m_type == "cnn1d_4channels":
                train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                            atm_data = atm_data,
                            device = device,
                            batch_size = 80,
                            stokes_as_channels=True,
                            linear = False)
                hu = 1024
                model = CNN1DModel(4,6*20,hidden_units=hu, signal_length=n_spec_points).to(device)
            elif m_type == "cnn1d_36channels":
                train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                            atm_data = atm_data,
                            device = device,
                            batch_size = 80,
                            stokes_as_channels=False,
                            linear = False)
                hu = 2048
                model = CNN1DModel(n_spec_points,6*20,hidden_units=hu, signal_length=4).to(device)
            #Creating the model
            model = model.float()
            #Loss function
            loss_fn = nn.MSELoss() # this is also called "criterion"/"cost function" in some places

            #Optimizer
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
            #Train model
            train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader, 
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=epochs,
                device=device,
                writer=create_writer(experiment_name=str(n_spec_points)+"_spectral_points",
                                    model_name=m_type,
                                    extra=f""))
            
            #Save the model to file so we can get back the best model
            save_filepath = f"{m_type}_{n_spec_points}_spectral_points.pth"
            save_model(model=model,
                    target_dir="models/second_experiment",
                    model_name=save_filepath)
            print("-"*50 + "\n")

           

if __name__ == "__main__":
    main()

