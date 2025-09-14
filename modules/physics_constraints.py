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
                 base_loss_threshold=0.0004, tau_indices=None, n_logtau=21, n_mags=4, blos_index=-1):
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
            base_loss_threshold (float): Threshold for base loss below which WFA is applied
            tau_indices (int, list, or tuple): Tau level(s) to use for B_LOS calculation
            n_logtau (int): Number of tau levels in the atmosphere model
            n_mags (int): Number of magnetic parameters in the atmosphere model
            blos_index (int): Index of B_LOS in the magnetic parameters
        """
        super(WFAConstrainedLoss, self).__init__()

        self.wavelengths = wavelengths.clone().detach()
        self.lambda0 = lambda0
        self.g_factor = g_factor
        self.wfa_weight = wfa_weight
        self.stokes_v_index = stokes_v_index
        self.blos_threshold = blos_threshold
        self.base_loss_threshold = base_loss_threshold

        self.n_logtau = n_logtau
        self.n_mags = n_mags
        self.blos_index = blos_index

        if tau_indices is None:
            self.tau_indices = list(range(0, 3))
        elif isinstance(tau_indices, int):
            self.tau_indices = [tau_indices]
        elif isinstance(tau_indices, (list, tuple)):
            self.tau_indices = list(tau_indices)
        else:
            raise ValueError("tau_indices must be int, list, tuple, or None")

        # Calculate WFA constant with proper unit conversion
        wfa_constant_si = e.si / (4 * np.pi) / m_e / c
        self.wfa_constant = wfa_constant_si.to(1 / u.G / u.Angstrom)

        if wavelength_range is None:
            self.start_idx = 0
            self.end_idx = len(wavelengths) - 1
        else:
            self.start_idx, self.end_idx = wavelength_range

        if base_loss.lower() == "mse":
            self.base_criterion = nn.MSELoss(reduction='none')
        elif base_loss.lower() == "mae":
            self.base_criterion = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")

    def compute_wfa_blos(self, stokes):
        """
        Compute B_LOS using the WFA for each pixel in the batch, with units.

        Args:
            stokes (torch.Tensor): Tensor of shape [batch_size, n_wavelengths, 4]
                                  containing the Stokes parameters

        Returns:
            astropy.units.Quantity: Estimated B_LOS values with units of Gauss
        """
        batch_size = stokes.shape[0]
        device = stokes.device

        # Extract wavelength range of interest and attach units
        wl_range = self.wavelengths[self.start_idx:self.end_idx].to(device).cpu().numpy() * u.Angstrom

        B_LOS = np.zeros(batch_size) * u.G

        for b in range(batch_size):
            I = stokes[b, 0, self.start_idx:self.end_idx].cpu().numpy()
            V = stokes[b, self.stokes_v_index, self.start_idx:self.end_idx].cpu().numpy()

            # Calculate dI/dλ using finite differences, with units
            dI_dl = np.gradient(I) / np.gradient(wl_range.value)
            dI_dl = dI_dl / u.Angstrom

            # Design matrix for least squares
            A = np.zeros((len(dI_dl), 2))
            A[:, 0] = dI_dl.value
            A[:, 1] = 1.0
            b_vec = V

            # Least squares solution (pseudo-inverse)
            p = np.linalg.pinv(A) @ b_vec

            # Compute B_LOS with units
            B = -p[0] * u.Angstrom / (self.wfa_constant * (self.lambda0 * u.Angstrom)**2 * self.g_factor)
            B_LOS[b] = B.to(u.G)

        return B_LOS

    def forward(self, input_stokes, predicted_atm, target_atm, target_blos=None):
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
        base_loss = torch.mean(self.base_criterion(predicted_atm, target_atm))
        if base_loss >= self.base_loss_threshold:
            return base_loss, base_loss, torch.tensor(0.0, device=base_loss.device)

        reshaped_predicted_atm = predicted_atm.view(predicted_atm.shape[0], self.n_logtau, self.n_mags)
        predicted_blos = reshaped_predicted_atm[:, self.tau_indices, self.blos_index].mean(dim=1)

        # Calculate WFA-based B_LOS from Stokes profiles (with units)
        wfa_blos = self.compute_wfa_blos(input_stokes)
        wfa_blos_tensor = torch.tensor(wfa_blos.value, device=base_loss.device)

        threshold_mask = (torch.abs(wfa_blos_tensor) < self.blos_threshold)
        if not torch.any(threshold_mask):
            return base_loss, base_loss, torch.tensor(0.0, device=base_loss.device)

        epsilon = 1e-10
        log_abs_predicted_blos = torch.log10(torch.abs(predicted_blos) + epsilon)
        log_abs_wfa_blos = torch.log10(torch.abs(wfa_blos_tensor) + epsilon)

        masked_predicted_blos = log_abs_predicted_blos[threshold_mask]
        masked_wfa_blos = log_abs_wfa_blos[threshold_mask]
        wfa_loss = torch.mean(self.base_criterion(masked_predicted_blos, masked_wfa_blos))

        total_loss = (1 - self.wfa_weight) * base_loss + self.wfa_weight * wfa_loss

        return total_loss, base_loss, wfa_loss