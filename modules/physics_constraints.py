import torch
import torch.nn as nn
import numpy as np
from astropy import units as u
from astropy.constants import c, e, m_e

class WFAConstrainedLoss(nn.Module):
    """
    Loss function that adds a constraint based on the Weak Field Approximation (WFA).
    This enforces consistency between Stokes V and derivative of Stokes I profiles.
    
    The WFA states: V ≈ -e/(4πm_e*c) * λ₀² * g_eff * B_LOS * dI/dλ
    """
    
    def __init__(self, wavelengths, lambda0, g_factor, wfa_weight=0.5, stokes_v_index=3,
         wavelength_range=None, base_loss="mae", blos_threshold=100.0, 
         base_loss_threshold=0.0004):  # Add this parameter
        """
        Initialize the WFA constrained loss function.
        
        Args:
            wavelengths (torch.Tensor): Wavelength array in Angstroms
            lambda0 (float): Rest wavelength of the spectral line in Angstroms
            g_factor (float): Landé factor of the spectral line
            wfa_weight (float): Weight for the WFA constraint [0-1]
            stokes_v_index (int): Index of Stokes V in the data (default: 3)
            wavelength_range (tuple): (start_idx, end_idx) for fitting the WFA
            base_loss (str): Base loss function ("mse" or "mae")
            blos_threshold (float): Threshold (in Gauss) below which WFA constraint is not applied
        """
        super(WFAConstrainedLoss, self).__init__()
        
        self.wavelengths = wavelengths.clone().detach()
        self.lambda0 = lambda0
        self.g_factor = g_factor
        self.wfa_weight = wfa_weight
        self.stokes_v_index = stokes_v_index
        self.blos_threshold = blos_threshold
        self.base_loss_threshold = base_loss_threshold  # Store the threshold
        
        # Calculate WFA constant with proper unit conversion
        wfa_constant_si = e.si / (4 * np.pi) / m_e / c
        # Convert to 1/(G·Å) - needed for proper units
        self.wfa_constant = float(wfa_constant_si.to(1 / u.G / u.Angstrom).value)
        
        # Store wavelength range for fitting
        if wavelength_range is None:
            # Default to full range
            self.start_idx = 0
            self.end_idx = len(wavelengths) - 1
        else:
            self.start_idx, self.end_idx = wavelength_range
            
        # Set base loss
        if base_loss.lower() == "mse":
            self.base_criterion = nn.MSELoss(reduction='none')
        elif base_loss.lower() == "mae":
            self.base_criterion = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
            
    def compute_wfa_blos(self, stokes):
        """
        Compute B_LOS using the WFA for each pixel in the batch.
        
        Args:
            stokes (torch.Tensor): Tensor of shape [batch_size, n_wavelengths, 4]
                                  containing the Stokes parameters
        
        Returns:
            torch.Tensor: Estimated B_LOS values
        """
        batch_size = stokes.shape[0]
        device = stokes.device
        
        # Extract wavelength range of interest
        wl_range = self.wavelengths[self.start_idx:self.end_idx].to(device)
        
        # Initialize B_LOS tensor
        B_LOS = torch.zeros(batch_size, device=device)
        
        # For each sample in the batch
        for b in range(batch_size):
            # Extract Stokes I and V within the wavelength range
            I = stokes[b, 0, self.start_idx:self.end_idx]  # Stokes I
            V = stokes[b, self.stokes_v_index, self.start_idx:self.end_idx, ]  # Stokes V
            
            # Calculate dI/dλ using finite differences
            dI_dl = torch.gradient(I)[0]/torch.gradient(wl_range)[0]
            
            # Set up least-squares problem: V = -C * λ₀² * g * B_LOS * dI/dλ
            # Design matrix for least squares
            A = torch.zeros((self.end_idx - self.start_idx, 2), device=device)
            A[:, 0] = dI_dl
            A[:, 1] = 1.0  # Constant term to account for offsets
            
            b_vec = V
            
            # Solve the least squares problem using QR decomposition for stability
            result = torch.linalg.lstsq(A, b_vec, rcond=None)
            solution = result[0] if isinstance(result, tuple) else result.solution
            
            # Extract the coefficient and calculate B_LOS
            coef = solution[0]
            B_LOS[b] = -coef / (self.wfa_constant * self.lambda0**2 * self.g_factor)
        
        return B_LOS
    
    def forward(self, input_stokes, predicted_atm, target_atm, target_blos=None,
            itau_bottom:int = 0, itau_top:int = 3, 
            blos_index:int = -1,
            n_logtau:int = 21,
            n_mags:int = 4):
        """
        Calculate the combined loss with thresholding for the WFA constraint.
        
        Args:
            input_stokes (torch.Tensor): Input Stokes parameters
            predicted_atm (torch.Tensor): Predicted atmospheric parameters
            target_atm (torch.Tensor): Target atmospheric parameters
            target_blos (torch.Tensor, optional): Target B_LOS values for supervision
        Returns:
            torch.Tensor: Loss value
        """
        # Basic MSE/MAE loss between predicted and target atmospheric parameters
        base_loss = torch.mean(self.base_criterion(predicted_atm, target_atm))
        # Check if base_loss is below threshold
        if base_loss >= self.base_loss_threshold:
            # If not below threshold, just return base_loss with zero WFA contribution
            return base_loss, base_loss, torch.tensor(0.0, device=base_loss.device)
        # Mean B_LOS for optical depths associated with the generation of the absorption line
        reshaped_predicted_atm = predicted_atm.view(predicted_atm.shape[0], n_logtau, n_mags)
        predicted_blos = reshaped_predicted_atm[:,itau_bottom:itau_top,blos_index].mean(dim=1) 
        
        # Calculate WFA-based B_LOS from Stokes profiles
        wfa_blos = self.compute_wfa_blos(input_stokes)
        # Create threshold mask - only apply WFA constraint for stronger fields
        # Using abs() since we care about field strength, not polarity
        
        threshold_mask = (torch.abs(wfa_blos) < self.blos_threshold)
        # If no pixels exceed threshold, just return base loss
        if not torch.any(threshold_mask):
            return base_loss, base_loss, torch.tensor(0.0, device=base_loss.device)
        
        # Convert to log scale to handle order of magnitude differences
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_abs_predicted_blos = torch.log10(torch.abs(predicted_blos) + epsilon)
        log_abs_wfa_blos = torch.log10(torch.abs(wfa_blos) + epsilon)
        
        # Apply mask to only consider values above threshold
        masked_predicted_blos = log_abs_predicted_blos[threshold_mask]
        masked_wfa_blos = log_abs_wfa_blos[threshold_mask]
        # Calculate WFA loss on the masked values
        wfa_loss = torch.mean(self.base_criterion(masked_predicted_blos, masked_wfa_blos))
        
        # Combine losses 
        total_loss = (1 - self.wfa_weight) * base_loss + self.wfa_weight * wfa_loss
        
        return total_loss, base_loss, wfa_loss