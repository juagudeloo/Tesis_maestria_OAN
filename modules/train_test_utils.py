import os
from pathlib import Path
import sys

import time
from timeit import default_timer as timer 
import datetime

import numpy as np

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


### TRAINING AND TESTING UTILITIES ###
def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
  """
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      train_acc += (y_pred == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> tuple[float, float]:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval() 

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)

          # 1. Forward pass
          test_pred = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(test_pred, y)
          test_loss += loss.item()

          # Calculate and accumulate accuracy
          test_acc += ((test_pred == y).sum().item()/len(test_pred))

  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device, 
          writer: torch.utils.tensorboard.writer.SummaryWriter # new parameter to take in a writer
          ) -> dict[str, list]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer log_dir if present.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: A SummaryWriter() instance to log model results to.

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} 
      For example if training for epochs=2: 
              {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                               global_step=epoch)

            # Close the writer
            writer.close()
        else:
            pass

    # Return the filled results at the end of the epochs
    return results

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)


### MODEL UTILITIES ###

def charge_weights(model: torch.nn.Module,
                 target_dir: str,
                 weights_name: str):
  
  target_dir_path = Path(target_dir)
  model_path = target_dir_path / weights_name
  
  print(f"[INFO] Loading model from: {model_path}")
  
  model.load_state_dict(torch.load(model_path, weights_only=True))

def descale_atm(atm_generated: np.ndarray,
                maxmin: dict[str, list[float]]) -> np.ndarray:
    def denorm_func(arr, maxmin):
        max_val = maxmin[0]
        min_val = maxmin[1]
        return arr*(max_val-min_val)+min_val
    
    atm_generated[:,:,:,0] = denorm_func(atm_generated[:,:,:,0], maxmin["T"])
    atm_generated[:,:,:,1] = denorm_func(atm_generated[:,:,:,1], maxmin["Rho"])
    atm_generated[:,:,:,2] = denorm_func(atm_generated[:,:,:,2], maxmin["B"])
    atm_generated[:,:,:,3] = denorm_func(atm_generated[:,:,:,3], maxmin["B"])
    atm_generated[:,:,:,4] = denorm_func(atm_generated[:,:,:,4], maxmin["B"])
    atm_generated[:,:,:,5] = denorm_func(atm_generated[:,:,:,5], maxmin["V"])
    
    return atm_generated

def generate_results(model: torch.nn.Module,
                     stokes_data: np.ndarray,
                     maxmin: dict[str, list[float]],
                     device: torch.device
                     ) -> np.ndarray:
  
  stokes_data = torch.tensor(stokes_data).float()
  
  print(f"stokes data shape for generation:", stokes_data.size())
  
  # Reduce batch size
  batch_size = 1024  # Adjust this value based on your GPU memory
  atm_generated = []
  for i in range(0, stokes_data.shape[0], batch_size):
      batch_data = stokes_data[i:i+batch_size].to(device)
      with torch.no_grad():
          atm_generated_batch = model(batch_data)
      atm_generated.append(atm_generated_batch.cpu())
      torch.cuda.empty_cache()  # Clear cache to free up memory
  
  atm_generated = torch.cat(atm_generated, dim=0)
  atm_generated = atm_generated.numpy()
  atm_generated = np.reshape(atm_generated, (480,480,20,6))
  
  print("atm generated data shape :", atm_generated.shape)
  
  atm_generated = descale_atm(atm_generated, maxmin)
  
  return atm_generated

### VISUALIZATION UTILITIES ###

tau = np.linspace(-2.5, 0, 20)

def plot_surface_generated_atm(atm_generated: np.ndarray,
             atm_original: np.ndarray,
             model_subdir: str,
             image_name: str,
             images_dir: str = "images",
             itau: int = 10
             ):

  print("atm_generated shape:", atm_generated.shape)
  print("atm_original shape:", atm_original.shape)
  fig, axs = plt.subplots(2, 6, figsize=(30, 10))
  

  itau = 17  # Variable for selecting the value of the tau axis
  tau_value = tau[itau]
  fig.suptitle(f'Atmospheric Parameters at tau = {tau_value:.2f}', fontsize=16)

  # Define colorbar limits based on atm_original
  vmin_T, vmax_T = atm_original[:, :, itau, 0].min(), atm_original[:, :, itau, 0].max()
  vmin_Rho, vmax_Rho = atm_original[:, :, itau, 1].min(), atm_original[:, :, itau, 1].max()
  vmin_Bq, vmax_Bq = atm_original[:, :, itau, 2].min(), atm_original[:, :, itau, 2].max()
  vmin_Bu, vmax_Bu = atm_original[:, :, itau, 3].min(), atm_original[:, :, itau, 3].max()
  vmin_Bv, vmax_Bv = atm_original[:, :, itau, 4].min(), atm_original[:, :, itau, 4].max()
  vmin_v, vmax_v = atm_original[:, :, itau, 5].min(), atm_original[:, :, itau, 5].max()

  # Plot generated atmosphere
  im = axs[0, 0].imshow(atm_generated[:, :, itau, 0], cmap='inferno', interpolation='nearest', vmin=vmin_T, vmax=vmax_T)
  axs[0, 0].set_title('Generated Temperature')
  axs[0, 0].axis('off')
  fig.colorbar(im, ax=axs[0, 0])

  im = axs[0, 1].imshow(atm_generated[:, :, itau, 1], cmap='spring', interpolation='nearest', vmin=vmin_Rho, vmax=vmax_Rho)
  axs[0, 1].set_title('Generated Density')
  axs[0, 1].axis('off')
  fig.colorbar(im, ax=axs[0, 1])

  im = axs[0, 2].imshow(atm_generated[:, :, itau, 2], cmap='PuOr', interpolation='nearest', vmin=vmin_Bq, vmax=vmax_Bq)
  axs[0, 2].set_title('Generated Bq')
  axs[0, 2].axis('off')
  fig.colorbar(im, ax=axs[0, 2])

  im = axs[0, 3].imshow(atm_generated[:, :, itau, 3], cmap='PuOr', interpolation='nearest', vmin=vmin_Bu, vmax=vmax_Bu)
  axs[0, 3].set_title('Generated Bu')
  axs[0, 3].axis('off')
  fig.colorbar(im, ax=axs[0, 3])
  
  im = axs[0, 4].imshow(atm_generated[:, :, itau, 4], cmap='PuOr', interpolation='nearest', vmin=vmin_Bv, vmax=vmax_Bv)
  axs[0, 4].set_title('Generated Bv')
  axs[0, 4].axis('off')
  fig.colorbar(im, ax=axs[0, 4])

  im = axs[0, 5].imshow(atm_generated[:, :, itau, 5], cmap='seismic_r', interpolation='nearest', vmin=vmin_v, vmax=vmax_v)
  axs[0, 5].set_title('Generated v')
  axs[0, 5].axis('off')
  fig.colorbar(im, ax=axs[0, 5])

  # Plot original atmosphere
  im = axs[1, 0].imshow(atm_original[:, :, itau, 0], cmap='inferno', interpolation='nearest', vmin=vmin_T, vmax=vmax_T)
  axs[1, 0].set_title('Original Temperature')
  axs[1, 0].axis('off')
  fig.colorbar(im, ax=axs[1, 0])
  
  im = axs[1, 1].imshow(atm_original[:, :, itau, 1], cmap='spring', interpolation='nearest', vmin=vmin_Rho, vmax=vmax_Rho)
  axs[1, 1].set_title('Original Density')
  axs[1, 1].axis('off')
  fig.colorbar(im, ax=axs[1, 1])

  im = axs[1, 2].imshow(atm_original[:, :, itau, 2], cmap='PuOr', interpolation='nearest', vmin=vmin_Bq, vmax=vmax_Bq)
  axs[1, 2].set_title('Original Bq')
  axs[1, 2].axis('off')
  fig.colorbar(im, ax=axs[1, 2])

  im = axs[1, 3].imshow(atm_original[:, :, itau, 3], cmap='PuOr', interpolation='nearest', vmin=vmin_Bu, vmax=vmax_Bu)
  axs[1, 3].set_title('Original Bu')
  axs[1, 3].axis('off')
  fig.colorbar(im, ax=axs[1, 3])

  im = axs[1, 4].imshow(atm_original[:, :, itau, 4], cmap='PuOr', interpolation='nearest', vmin=vmin_Bv, vmax=vmax_Bv)
  axs[1, 4].set_title('Original Bv')
  axs[1, 4].axis('off')
  fig.colorbar(im, ax=axs[1, 4])

  im = axs[1, 5].imshow(atm_original[:, :, itau, 5], cmap='seismic_r', interpolation='nearest', vmin=vmin_v, vmax=vmax_v)
  axs[1, 5].set_title('Original v')
  axs[1, 5].axis('off')
  fig.colorbar(im, ax=axs[1, 5])

  
  
  images_dir = os.path.join(images_dir, model_subdir)
  if not os.path.exists(images_dir):
    os.makedirs(images_dir)
  image_path = os.path.join(images_dir, image_name)
  fig.savefig(image_path)
  
  print(f"Saved image to: {image_path}")
     
def plot_od_generated_atm(
                       atm_generated: np.ndarray,
                       atm_original: np.ndarray,
                       ix: int,
                       iy: int,
                       model_subdir: str,
                       image_name: str,
                       images_dir: str = "images"
                       ):

    print("atm_generated shape:", atm_generated.shape)
    print("atm_original shape:", atm_original.shape)
    fig, axs = plt.subplots(1, 7, figsize=(30, 5))
    
    # Plot Temperature Surface
    im = axs[0].imshow(atm_original[:,:,10,0], cmap='inferno', interpolation='nearest')
    axs[0].scatter(iy, ix, color='green', s=50, edgecolor='black')
    axs[0].set_title('Temperature Surface')
    axs[0].axis('on')
    fig.colorbar(im, ax=axs[0])

    # Plot generated and original atmosphere
    axs[1].plot(tau, atm_generated[:,:, :, 0].mean(axis=(0,2)), color='orangered', label='Generated')
    axs[1].plot(tau, atm_original[:,:, :, 0].mean(axis=(0,2)), color='navy', label='Original')
    axs[1].set_title('Temperature')
    axs[1].set_xlabel('Tau')
    axs[1].axis('on')

    axs[2].plot(tau, atm_generated[:,:, :, 1].mean(axis=(0,2)), color='orangered', label='Generated')
    axs[2].plot(tau, atm_original[:,:, :, 1].mean(axis=(0,2)), color='navy', label='Original')
    axs[2].set_title('Density')
    axs[2].set_xlabel('Tau')
    axs[2].axis('on')

    axs[3].plot(tau, atm_generated[:,:, :, 2].mean(axis=(0,2)), color='orangered', label='Generated')
    axs[3].plot(tau, atm_original[:,:, :, 2].mean(axis=(0,2)), color='navy', label='Original')
    axs[3].set_title('Bq')
    axs[3].set_xlabel('Tau')
    axs[3].axis('on')

    axs[4].plot(tau, atm_generated[:,:, :, 3].mean(axis=(0,2)), color='orangered', label='Generated')
    axs[4].plot(tau, atm_original[:,:, :, 3].mean(axis=(0,2)), color='navy', label='Original')
    axs[4].set_title('Bu')
    axs[4].set_xlabel('Tau')
    axs[4].axis('on')

    axs[5].plot(tau, atm_generated[:,:, :, 4].mean(axis=(0,2)), color='orangered', label='Generated')
    axs[5].plot(tau, atm_original[:,:, :, 4].mean(axis=(0,2)), color='navy', label='Original')
    axs[5].set_title('Bv')
    axs[5].set_xlabel('Tau')
    axs[5].axis('on')

    axs[6].plot(tau, atm_generated[:,:, :, 5].mean(axis=(0,2)), color='orangered', label='Generated')
    axs[6].plot(tau, atm_original[:,:, :, 5].mean(axis=(0,2)), color='navy', label='Original')
    axs[6].set_title('v')
    axs[6].set_xlabel('Tau')
    axs[6].axis('on')

    # Add legend
    axs[1].legend(loc='upper right')
    
    images_dir = os.path.join(images_dir, model_subdir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    image_path = os.path.join(images_dir, image_name)
    fig.savefig(image_path)
    
    print(f"Saved image to: {image_path}")
    

def plot_density_bars(atm_generated: np.ndarray,
  atm_original: np.ndarray,
  model_subdir: str,
  dense_diag_subdir: str,
  image_name: str,
  titles: list,
  tau_index: int,
  images_dir: str = "images",
  num_bars: int = 10):
  """
  Plots the density of values of the atm_generated and atm_original for a specific optical depth index.
  The plot is composed of bars.

  Args:
  atm_generated (np.ndarray): Generated atmospheric data.
  atm_original (np.ndarray): Original atmospheric data.
  model_subdir (str): Subdirectory for saving the plot.
  image_name (str): Name of the image file.
  titles (list): List of titles for each subplot.
  tau_index (int): Index of the optical depth to plot.
  images_dir (str, optional): Directory to save the images. Defaults to "images".
  num_bars (int, optional): Number of bars in the plot. Defaults to 10.
  """
  
  def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between two arrays.

    Args:
      y_true (np.ndarray): Array of true values.
      y_pred (np.ndarray): Array of predicted values.

    Returns:
      float: The SMAPE value as a percentage.
    """
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
  
  fig, axs = plt.subplots(1, atm_generated.shape[3], figsize=(5 * atm_generated.shape[3], 5))
  fig.suptitle('xlim based on 5th and 95th quantiles', fontsize=16)

  for j in range(atm_generated.shape[3]):
    gen_values = atm_generated[:, :, tau_index, j].flatten()
    orig_values = atm_original[:, :, tau_index, j].flatten()

    # Calculate quantiles for xlim
    gen_q5, gen_q95 = np.quantile(gen_values, [0.05, 0.95])
    orig_q5, orig_q95 = np.quantile(orig_values, [0.05, 0.95])
    xlim_min = min(gen_q5, orig_q5)
    xlim_max = max(gen_q95, orig_q95)

    # Create histogram bins
    bins = np.linspace(xlim_min, xlim_max, num_bars + 1)
    smape_res = smape(gen_values, orig_values)
    # Plot histograms
    axs[j].hist(gen_values, bins=bins, alpha=0.5, label='Generated', color='orangered')
    axs[j].hist(orig_values, bins=bins, alpha=0.5, label='Original', color='navy')
    axs[j].set_title(f"{titles[j]} (tau={tau[tau_index]:.2f}) (smape = {smape_res:.2f})")
    axs[j].set_xlabel('Value')
    axs[j].set_ylabel('Density')
    axs[j].legend(loc='upper right')
    axs[j].set_xlim([xlim_min, xlim_max])  # Set xlim based on quantiles

  images_dir = os.path.join(images_dir, model_subdir, dense_diag_subdir)
  if not os.path.exists(images_dir):
    os.makedirs(images_dir)
  image_path = os.path.join(images_dir, f"{tau[tau_index]:.2f}_{image_name}")
  fig.savefig(image_path)

  print(f"Saved image to: {image_path}")

# Example usage:
# plot_density_bars(atm_generated, atm_original, "model_subdir", "density_bars.png", ["Title1", "Title2", "Title3", "Title4", "Title5", "Title6"])