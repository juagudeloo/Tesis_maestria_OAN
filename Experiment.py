import sys

from timeit import default_timer as timer 

import numpy as np

import torch
from torch import nn

#MODULES IMPORT
sys.path.append("../modules")
from modules.data_utils import load_training_data, create_dataloaders, plot_stokes
from modules.nn_models import *
from modules.train_test_utils import train, set_seeds, create_writer, save_model

#
def main():
    
    ### DATA LOADING ###
    #filenames to be readed for creating the dataset
    #filenames = []
    #for i in range(80, 97):
    #   filenames.append(f"0{i}000")
    #filenames.append("099000")
    #for i in range(100, 113):
    #    filenames.append(f"{i}000")
    filenames = ["080000", 
                 "085000", "090000"
                 ] #for initital testings
        
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensors stored in: {device}")
    
    ### Checking linear vs convolutional 1d models ###
    
    m_type = "hybrid"
    epochs = 10
    lr = 1e-3
    n_spec_points = 112
    new_logtau = np.arange(-2.0, 0+0.1, 0.1)
    noise_level = 0
    
    stokes_weights = [1,100,100,10]
    
    #1. Loop through stokes weights
    atm_data, stokes_data, wl_points = load_training_data(filenames, 
                                                            n_spectral_points=n_spec_points,
                                                            new_logtau=new_logtau,
                                                            stokes_weights=stokes_weights)
    stokes_data = np.moveaxis(stokes_data, 1, 2)
    
    plot_stokes(stokes=stokes_data[0], 
                wl_points = wl_points,
                images_dir="images/fourth_experiment",
                image_name = "example_stokes",)
    
    #2. Create dataloaders
    train_dataloader, test_dataloader = create_dataloaders(stokes_data = stokes_data,
                atm_data = atm_data,
                device = device,
                batch_size = 80,
                linear = False)
    
    model_types = [
                    "mscnn",
                   "cnn1d_4channels"]
    for model_type in model_types:
        #3. Create models
        if model_type == "mscnn":
            scales = [1,2,4]
            model = MSCNN(scales=scales, 
                                nwl_points=len(wl_points),
                                n_outputs=6*len(new_logtau),
                                c1_filters=32,
                                c2_filters=64).to(device).float()
            model.name = "MSCNN"
            
        elif model_type == "cnn1d_4channels":
            hu = 1024
            model = CNN1D(4,6*len(new_logtau),hidden_units=hu, signal_length=n_spec_points).to(device)
            model.name = "CNN1D_4CHANNELS"
        
        #4. Set hyperparameters
        set_seeds()
        #Loss function
        loss_fn = nn.MSELoss()

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        
        #5. Train models
        
        # ----------------- Thermodynamic model -----------------
        experiment_name = model.name
        print("Running experiment: ", experiment_name)
        train(model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader, 
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            device=device,
            writer=create_writer(experiment_name=experiment_name,
                                model_name=model.name,
                                extra=f""))
        
        #Save the model to file so we can get back the best model
        save_filepath = experiment_name + ".pth"
        save_model(model=model,
                target_dir="models/fourth_experiment/",
                model_name=save_filepath)
        print("-"*50 + "\n")
    

    

           

if __name__ == "__main__":
    main()

