import sys
import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
sys.path.append('./modules')
from train_utils import set_seed, get_device, load_data, get_model, custom_loss, train_model, train_model_with_logging


def main():
    experiment_name = "paper_experiment"  # <-- Set your experiment folder name here
    set_seed(42)
    device = get_device()
    
    # Experiment configurations
    # Spectral conditions are ALWAYS applied (mandatory)
    apply_spectral_conditions = True
    noise_conditions = [True, False]     # With and without noise (optional)
    w_values = [0, 1e-3, 1e-2, 0.1, 0.5, 0.9]
    num_epochs = 20
    # Set batch size based on available GPU memory
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(device).total_memory // (1024 ** 2)  # in MB
        if total_mem >= 24000:
            batch_size = 512
        elif total_mem >= 12000:
            batch_size = 256
        elif total_mem >= 6000:
            batch_size = 128
        else:
            batch_size = 64
    else:
        batch_size = 32  # Fallback for CPU
    print(f"Using batch size: {batch_size}")
    
    # Generate filenames from 080000 to 112000, excluding NaN files
    exclude_nan_files = {'094000', '097000', '098000'}
    filenames = [f"{i:06d}" for i in range(80000, 112100, 1000) if f"{i:06d}" not in exclude_nan_files]
    
    # Main experiment loop over noise conditions only
    for add_noise in noise_conditions:
        noise_name = "with_noise" if add_noise else "without_noise"
        condition_name = f"spectral_{noise_name}"
        
        print(f"\n{'='*60}")
        print(f"Running experiment {condition_name.upper()}")
        print(f"Spectral conditions: {apply_spectral_conditions} (mandatory), Noise: {add_noise}")
        print(f"{'='*60}")
        
        # Create base directory for this configuration
        base_save_dir = os.path.join("models", experiment_name, condition_name)
        os.makedirs(base_save_dir, exist_ok=True)
        
        # Load and validate data for this configuration
        valid_filenames = []
        valid_data_per_file = {}
        skipped_filenames = []
        
        print(f"Checking available files for {condition_name}...")
        for fname in filenames:
            try:
                data = load_data([fname], apply_spectral_conditions=apply_spectral_conditions, add_noise=add_noise)
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
        
        print(f"Total valid files for {condition_name}: {len(valid_filenames)}")
        if skipped_filenames:
            print("Skipped filenames:")
            for fname in skipped_filenames:
                print(fname)
        
        # Experiment loop over different w values for this configuration
        for w in w_values:
            print(f"\n---- Training {condition_name} with w = {w} ----")
            
            # Create subfolder for this w value
            w_save_dir = os.path.join(base_save_dir, f"w_{w}")
            os.makedirs(w_save_dir, exist_ok=True)
            
            model = get_model(device=device)
            writer = SummaryWriter(log_dir=os.path.join(w_save_dir, "tensorboard"))
            
            total_loss_history, loss1_history, loss2_history, runtime = train_model_with_logging(
                model, valid_data_per_file, valid_filenames, 
                num_epochs=num_epochs, batch_size=batch_size, 
                device=device, w=w, writer=writer
            )
            writer.close()
            
            # Save model
            save_path = os.path.join(w_save_dir, f"final_model_{condition_name}_w_{w}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            
            # Save loss histories and runtime
            log_data = {
                "spectral_conditions": apply_spectral_conditions,
                "add_noise": add_noise,
                "condition_name": condition_name,
                "w": w,
                "total_loss_history": total_loss_history,
                "loss1_history": loss1_history,
                "loss2_history": loss2_history,
                "runtime": runtime,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "valid_filenames": valid_filenames
            }
            with open(os.path.join(w_save_dir, f"experiment_log_{condition_name}_w_{w}.pkl"), "wb") as f:
                pickle.dump(log_data, f)
            print(f"Experiment log saved for {condition_name} w={w}")
        
        print(f"Completed all w values for {condition_name}")
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved in: models/{experiment_name}/")
    print("Experiment configurations:")
    print("- spectral_without_noise/")
    print("- spectral_with_noise/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()