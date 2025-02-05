import sys

from timeit import default_timer as timer 

import torch
from torch import nn

#MODULES IMPORT
sys.path.append("../modules")
from modules.data_utils import load_training_data, create_dataloaders
from modules.nn_models import SimpleLinearModel, SimpleCNN1DModel
from modules.train_test_utils import train, set_seeds, create_writer, save_model


def main():
    
    ### DATA LOADING ###
    #filenames to be readed for creating the dataset
    filenames = ["080000", 
                 #"085000", "090000"
                 ]
    
    #Load data
    atm_data, stokes_data, mags_names = load_training_data(filenames)
    
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")
    
    # Create dataset and dataloaders
    train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                       atm_data = atm_data,
                       device = device,
                       batch_size = 80)
    
    ### MODEL CREATION ###
    hidden_units = [72, 1024, 2048, 4096]
    
    ### Checking linear vs convolutional 1d models ###
    
    test_epochs = [10, 20]
    model_types = [
        #"simple_linear", "simple_cnn1d_36channels", 
                   "simple_cnn1d_4channels"]
    lr = 1e-3
    
    
    
    #1. Loop through model types
    for m_type in model_types:
        print(f"Training {m_type} models")
        if m_type == "simple_linear":
            train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                        atm_data = atm_data,
                        device = device,
                        batch_size = 80,
                        linear = True)
        elif m_type == "simple_cnn1d_36channels":
            train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                        atm_data = atm_data,
                        device = device,
                        batch_size = 80,
                        stokes_as_channels=False,
                        linear = False)
        elif m_type == "simple_cnn1d_4channels":
            train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                        atm_data = atm_data,
                        device = device,
                        batch_size = 80,
                        stokes_as_channels=True,
                        linear = False)
        #2. Loop through hidden units
        for hu in hidden_units:
            #3. Loop throuch epochs
            for epochs in test_epochs:
                #Creating the model
                if m_type == "simple_linear":
                    model = SimpleLinearModel(36*4,6*20,hidden_units=hu).to(device)
                elif m_type == "simple_cnn1d_36channels":
                    model = SimpleCNN1DModel(36,6*20,hidden_units=hu, signal_length=4).to(device)
                elif m_type == "simple_cnn1d_4channels":
                    model = SimpleCNN1DModel(4,6*20,hidden_units=hu, signal_length=36).to(device)
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
                    writer=create_writer(experiment_name=str(hu)+"_hidden_units",
                                        model_name=m_type,
                                        extra=f"{epochs}_epochs"))
                
                #Save the model to file so we can get back the best model
                save_filepath = f"{m_type}_{hu}_hidden_units_{epochs}_epochs.pth"
                save_model(model=model,
                        target_dir="models",
                        model_name=save_filepath)
                print("-"*50 + "\n")

            


if __name__ == "__main__":
    main()

