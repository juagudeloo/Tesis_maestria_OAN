import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(filenames, nx=480, ny=480, nz=256, data_path="/scratchsan/observatorio/juagudeloo/data", apply_spectral_conditions=True, add_noise=True):
    sys.path.append('./modules_2')
    from charge_data import DataCharger
    data_charger = DataCharger(
        data_path=data_path,
        filenames=filenames,
        nx=nx,
        ny=ny,
        nz=nz
    )
    data_charger.charge_all_files(
        normalize_atmosphere=True, 
        apply_spectral_conditions=apply_spectral_conditions,
        add_noise=add_noise
    )
    data_per_file = data_charger.reshape_for_training()
    return data_per_file

def get_model(scales=[1,2,3], in_channels=2, c1_filters=16, c2_filters=32, kernel_size=5, stride=1, padding=0, pool_size=2, n_linear_layers=4, output_features=3*21, device=None):
    sys.path.append('./modules')
    from nn_inversion_model import MSCNNInversionModel
    model = MSCNNInversionModel(
        scales=scales,
        in_channels=in_channels,
        c1_filters=c1_filters,
        c2_filters=c2_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        pool_size=pool_size,
        n_linear_layers=n_linear_layers,
        output_features=output_features
    )
    if device:
        model = model.to(device)
    return model

def custom_loss(output, target, blos_pred, blos_true, w, wfa_mask=None):
    mse = nn.MSELoss()
    loss1 = mse(output, target)
    
    if wfa_mask is not None and wfa_mask.sum() > 0:
        # Only compute WFA loss for pixels where mask is True
        masked_blos_pred = blos_pred[wfa_mask]
        masked_blos_true = blos_true[wfa_mask]
        loss2 = mse(masked_blos_pred, masked_blos_true)
        # Scale the WFA loss by the fraction of pixels that use it
        wfa_fraction = wfa_mask.float().mean()
        total_loss = (1-w) * loss1 + w * loss2 * wfa_fraction
    else:
        loss2 = torch.tensor(0.0, device=output.device)
        total_loss = loss1
    
    return total_loss, loss1, loss2

def train_model(model, data_per_file, filenames, num_epochs=10, batch_size=128, w=0.5, device=None):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for filename in filenames:
            file_data = data_per_file[filename]
            stokes = torch.tensor(file_data["stokes_reshaped"], dtype=torch.float32).to(device)
            muram = torch.tensor(file_data["muram_reshaped"], dtype=torch.float32).to(device)
            wfa_blos = torch.tensor(file_data["wfa_BLOS_reshaped"], dtype=torch.float32).to(device)
            best_muram_b = torch.tensor(file_data["best_muram_B_reshaped"], dtype=torch.float32).to(device)
            scaler = file_data["best_muram_B_minmax_scaler"]
            min_rrmse_idx = file_data["min_rrmse_idx_normalized"]

            num_samples = stokes.shape[0]
            for i in range(0, num_samples, batch_size):
                batch_stokes = stokes[i:i+batch_size]
                batch_muram = muram[i:i+batch_size]
                batch_wfa_blos = wfa_blos[i:i+batch_size]
                batch_best_muram_b = best_muram_b[i:i+batch_size]

                # Create mask for WFA BLOS < 200 Gauss
                wfa_mask = torch.abs(batch_wfa_blos.squeeze()) < 200.0

                optimizer.zero_grad()
                output = model(batch_stokes)
                output_reshaped = output.view(-1, 3, 21)
                target = batch_muram
                pred_blos = output_reshaped[:, 2, min_rrmse_idx].unsqueeze(1)
                pred_blos_np = pred_blos.detach().cpu().numpy()
                pred_blos_scaled = torch.tensor(scaler.transform(pred_blos_np), dtype=torch.float32).to(device)
                loss, _, _ = custom_loss(output, target, pred_blos_scaled, batch_wfa_blos, w=w, wfa_mask=wfa_mask)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_stokes.size(0)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(filenames)/num_samples:.6f}")
    print("Training finished.")

def train_model_with_logging(model, data_per_file, filenames, num_epochs=10, batch_size=128, device=None, w=0.5, writer=None):
    import numpy as np
    import torch.nn as nn
    import torch.optim as optim
    import time
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    total_loss_history = []
    loss1_history = []
    loss2_history = []
    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        for filename in filenames:
            file_data = data_per_file[filename]
            stokes = torch.tensor(file_data["stokes_reshaped"], dtype=torch.float32).to(device)
            muram = torch.tensor(file_data["muram_reshaped"], dtype=torch.float32).to(device)
            wfa_blos = torch.tensor(file_data["wfa_BLOS_reshaped"], dtype=torch.float32).to(device)
            best_muram_b = torch.tensor(file_data["best_muram_B_reshaped"], dtype=torch.float32).to(device)
            scaler = file_data["best_muram_B_minmax_scaler"]
            min_rrmse_idx = file_data["min_rrmse_idx_normalized"]

            num_samples = stokes.shape[0]
            for i in range(0, num_samples, batch_size):
                batch_stokes = stokes[i:i+batch_size]
                batch_muram = muram[i:i+batch_size]
                batch_wfa_blos = wfa_blos[i:i+batch_size]
                batch_best_muram_b = best_muram_b[i:i+batch_size]

                # Create mask for WFA BLOS < 200 Gauss
                wfa_mask = torch.abs(batch_wfa_blos.squeeze()) < 200.0

                optimizer.zero_grad()
                output = model(batch_stokes)
                output_reshaped = output.view(-1, 3, 21)
                target = batch_muram
                pred_blos = output_reshaped[:, 2, min_rrmse_idx].unsqueeze(1)
                pred_blos_np = pred_blos.detach().cpu().numpy()
                pred_blos_scaled = torch.tensor(scaler.transform(pred_blos_np), dtype=torch.float32).to(device)
                loss, loss1, loss2 = custom_loss(output, target, pred_blos_scaled, batch_wfa_blos, w=w, wfa_mask=wfa_mask)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_stokes.size(0)
                total_loss1 += loss1.item() * batch_stokes.size(0)
                total_loss2 += loss2.item() * batch_stokes.size(0)
        avg_loss = total_loss / (len(filenames) * num_samples)
        avg_loss1 = total_loss1 / (len(filenames) * num_samples)
        avg_loss2 = total_loss2 / (len(filenames) * num_samples)
        total_loss_history.append(avg_loss)
        loss1_history.append(avg_loss1)
        loss2_history.append(avg_loss2)
        if writer:
            writer.add_scalar(f"Loss/Total", avg_loss, epoch)
            writer.add_scalar(f"Loss/Loss1", avg_loss1, epoch)
            writer.add_scalar(f"Loss/Loss2", avg_loss2, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Loss1: {avg_loss1:.6f}, Loss2: {avg_loss2:.6f}")
    runtime = time.time() - start_time
    print("Training finished. Runtime: {:.2f} seconds".format(runtime))
    return total_loss_history, loss1_history, loss2_history, runtime
