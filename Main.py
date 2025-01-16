import os
from pathlib import Path
import sys

import time
from timeit import default_timer as timer 
import datetime

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

#MODULES IMPORT
sys.path.append("../modules")
from modules.data_utils import load_data, create_dataloaders
from modules.nn_models import SimpleLinearModel, SimpleCNN1DModel
from modules.train_test_utils import train, set_seeds, create_writer


def main():
    
    ### DATA LOADING ###
    #filenames to be readed for creating the dataset
    filenames = ["080000", "085000", "090000"]
    
    #Load data
    atm_data, stokes_data, mags_names = load_data(filenames)
    
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")
    
    # Create dataset and dataloaders
    train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                       atm_data = atm_data,
                       device = device,
                       batch_size = 80)
    
    ### MODEL CREATION ###
    set_seeds()
    simple_linear_models = {
        "simple_linear_72": SimpleLinearModel(36*4,6*20,hidden_units=72).to(device),
        "simple_linear_1024": SimpleLinearModel(36*4,6*20,hidden_units=1024).to(device),
        "simple_linear_2048": SimpleLinearModel(36*4,6*20,hidden_units=2048).to(device),
        "simple_linear_4096": SimpleLinearModel(36*4,6*20,hidden_units=4096).to(device)
        
    }
    
    simple_cnn1d_models = {
        "simple_cnn1d_72": SimpleCNN1DModel(36,6*20,hidden_units=72).to(device),
        "simple_cnn1d_1024": SimpleCNN1DModel(36,6*20,hidden_units=1024).to(device),
        "simple_cnn1d_2048": SimpleCNN1DModel(36,6*20,hidden_units=2048).to(device),
        "simple_cnn1d_4096": SimpleCNN1DModel(36,6*20,hidden_units=4096).to(device)
    }
    
    hidden_units = [72, 1024, 2048, 4096]
    
    ### Checking linear vs convolutional 1d models ###
    
    test_epochs = [10, 20]
    model_types = ["simple_linear", "simple_cnn1d"]
    lr = 1e-3
    
    
    
    #1. Loop through model types
    for m_type in model_types:
        if m_type == "simple_linear":
            train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data.float(),
                        atm_data = atm_data.float(),
                        device = device,
                        batch_size = 80,
                        linear = True)
        else:
            train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data.float(),
                        atm_data = atm_data.float(),
                        device = device,
                        batch_size = 80,
                        linear = False)
        #2. Loop through hidden units
        for hu in hidden_units:
            #3. Loop throuch epochs
            for epochs in test_epochs:
                #Creating the model
                if m_type == "simple_linear":
                    model = SimpleLinearModel(36*4,6*20,hidden_units=hu).to(device)
                else:
                    model = SimpleCNN1DModel(36,6*20,hidden_units=hu).to(device)
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
                                        model_name=model.name,
                                        extra=f"{epochs}_epochs"))
                
                #Save the model to file so we can get back the best model
                save_filepath = f"{model.name}_{hu}_hidden_units_{epochs}_epochs.pth"
                save_model(model=model,
                        target_dir="models",
                        model_name=save_filepath)
                print("-"*50 + "\n")

            


if __name__ == "__main__":
    main()

