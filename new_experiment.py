import sys
import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
sys.path.append('./modules_2')
from train_utils import set_seed, get_device, load_data, get_model, custom_loss, train_model, train_model_with_logging


def main():
    experiment_name = "paper_experiment"  # <-- Set your experiment folder name here
    set_seed(42)
    device = get_device()
    # Generate filenames from 090000 to 100000, excluding NaN files
    exclude_nan_files = {'094000', '097000', '098000'}
    filenames = [f"{i:06d}" for i in range(80000, 112100, 1000) if f"{i:06d}" not in exclude_nan_files]
    valid_filenames = []
    valid_data_per_file = {}
    skipped_filenames = []
    print("Checking available files...")
    for fname in filenames:
        try:
            data = load_data([fname])
            # Check for NaNs or infs in loaded data
            for key, value in data.items():
                if hasattr(value, 'numpy'):
                    arr = value.numpy()
                else:
                    arr = value
                if (hasattr(arr, 'dtype') and
                    (np.isnan(arr).any() or np.isinf(arr).any())):
                    print(f"Warning: NaN or Inf detected in file {fname}, key {key}")
            # If data loads successfully, add to valid list
            if fname in data:
                valid_filenames.append(fname)
                valid_data_per_file[fname] = data[fname]
                print(f"Loaded: {fname}")
            else:
                skipped_filenames.append(fname)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            skipped_filenames.append(fname)
    print(f"Total valid files: {len(valid_filenames)}")
    if skipped_filenames:
        print("Skipped filenames:")
        for fname in skipped_filenames:
            print(fname)
    
    # Experiment loop over different w values
    w_values = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9]
    num_epochs = 30
    batch_size = 128
    save_dir = os.path.join("models", experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    for w in w_values:
        print(f"\n==== Training with w = {w} ====")
        model = get_model(device=device)
        writer = SummaryWriter(log_dir=os.path.join(save_dir, f"tensorboard_w_{w}"))
        total_loss_history, loss1_history, loss2_history, runtime = train_model_with_logging(
            model, valid_data_per_file, valid_filenames, num_epochs=num_epochs, batch_size=batch_size, device=device, w=w, writer=writer)
        writer.close()
        # Save model
        save_path = os.path.join(save_dir, f"final_model_w_{w}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        # Save loss histories and runtime
        log_data = {
            "w": w,
            "total_loss_history": total_loss_history,
            "loss1_history": loss1_history,
            "loss2_history": loss2_history,
            "runtime": runtime
        }
        with open(os.path.join(save_dir, f"loss_history_w_{w}.pkl"), "wb") as f:
            pickle.dump(log_data, f)
        print(f"Loss history and runtime saved for w={w}")

if __name__ == "__main__":
    main()