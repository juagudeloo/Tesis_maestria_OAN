"""Physics-informed MSCNN model definition."""

from typing import Dict, Optional, Tuple, Any, Literal
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import astropy.units as u

from models.mscnn_model import MSCNNInversionModel


class PhysicsInformedMSCNN(MSCNNInversionModel):
    """MSCNN inversion model with integrated physics-based regularization computed in physical units.
    
    Parameters
    ----------
    use_physics : {None, 'wfa', 'doppler', 'temperature', 'both', 'all'}, optional
        Controls which physics regularization to apply:
        - None: No physics regularization (pure supervised learning)
        - 'wfa': Only WFA B_LOS regularization
        - 'doppler': Only Doppler V_LOS regularization
        - 'temperature': Only temperature regularization
        - 'both': Both WFA and Doppler regularization
        - 'all': All physics terms (WFA, Doppler, and Temperature)
    lambda_wfa : float
        Weight for WFA B_LOS loss term
    lambda_doppler : float
        Weight for Doppler V_LOS loss term
    lambda_temp : float
        Weight for temperature loss term
    blos_physics_mode : {'tau_averaged', 'single_height'}, optional
        Mode for computing B_LOS comparison
    blos_target_logtau : float, optional
        Target log(tau) value for B_LOS single_height mode
    vlos_physics_mode : {'tau_averaged', 'single_height'}, optional
        Mode for computing V_LOS comparison
    vlos_target_logtau : float, optional
        Target log(tau) value for V_LOS single_height mode
    temp_physics_mode : {'tau_averaged', 'single_height'}, optional
        Mode for computing temperature comparison (default: 'single_height')
    temp_target_logtau : float, optional
        Target log(tau) for temperature single_height mode (default: 0.0, photosphere)
    """

    def __init__(
        self,
        *,
        use_physics: Optional[str] = None,
        lambda_wfa: float = 0.01,
        lambda_doppler: float = 0.01,
        lambda_temp: float = 0.01,
        blos_physics_mode: Literal['tau_averaged', 'single_height'] = 'tau_averaged',
        blos_target_logtau: Optional[float] = None,
        vlos_physics_mode: Literal['tau_averaged', 'single_height'] = 'tau_averaged',
        vlos_target_logtau: Optional[float] = None,
        temp_physics_mode: Literal['tau_averaged', 'single_height'] = 'single_height',
        temp_target_logtau: Optional[float] = 0.0,
        dropout_rate: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(dropout_rate=dropout_rate, **kwargs)
        
        # Validate and store physics regularization mode
        valid_modes = [None, 'wfa', 'doppler', 'temperature', 'both', 'all']
        if use_physics not in valid_modes:
            raise ValueError(f"use_physics must be one of {valid_modes}, got {use_physics}")
        
        # Validate physics_mode for each quantity
        valid_physics_modes = ['tau_averaged', 'single_height']
        if blos_physics_mode not in valid_physics_modes:
            raise ValueError(f"blos_physics_mode must be one of {valid_physics_modes}, got {blos_physics_mode}")
        if vlos_physics_mode not in valid_physics_modes:
            raise ValueError(f"vlos_physics_mode must be one of {valid_physics_modes}, got {vlos_physics_mode}")
        if temp_physics_mode not in valid_physics_modes:
            raise ValueError(f"temp_physics_mode must be one of {valid_physics_modes}, got {temp_physics_mode}")
        
        self.use_physics = use_physics
        self.lambda_wfa = float(lambda_wfa)
        self.lambda_doppler = float(lambda_doppler)
        self.lambda_temp = float(lambda_temp)
        self.blos_physics_mode = blos_physics_mode
        self.blos_target_logtau = blos_target_logtau
        self.vlos_physics_mode = vlos_physics_mode
        self.vlos_target_logtau = vlos_target_logtau
        self.temp_physics_mode = temp_physics_mode
        self.temp_target_logtau = temp_target_logtau
        
        # Physics computation state (set via set_physics_context)
        self.mhd_normalizer = None
        self.logtau_values = None
        self.blos_approx = None
        self.vlos_approx = None
        self.temp_approx = None
        self._tau_linear = None
        self._dtau = None
        self._integral_dtau = None
        self._blos_target_logtau_idx = None
        self._vlos_target_logtau_idx = None
        self._temp_target_logtau_idx = None

    def set_physics_context(
        self,
        mhd_normalizer: 'MhdNormalizer',
        logtau_values: np.ndarray,
        blos_approx: Optional[np.ndarray] = None,
        vlos_approx: Optional[np.ndarray] = None,
        temp_approx: Optional[np.ndarray] = None,
        blos_physics_mode: Optional[Literal['tau_averaged', 'single_height']] = None,
        blos_target_logtau: Optional[float] = None,
        vlos_physics_mode: Optional[Literal['tau_averaged', 'single_height']] = None,
        vlos_target_logtau: Optional[float] = None,
        temp_physics_mode: Optional[Literal['tau_averaged', 'single_height']] = None,
        temp_target_logtau: Optional[float] = None,
    ):
        """
        Set physics computation context for the current training step.
        
        Parameters
        ----------
        mhd_normalizer : MhdNormalizer
            Normalizer for denormalizing predictions to physical units
        logtau_values : np.ndarray
            Log optical depth values (e.g., np.arange(-2.0, 0.1, 0.1))
        blos_approx : np.ndarray, optional
            WFA B_LOS approximation map (nx, ny) in Gauss
        vlos_approx : np.ndarray, optional
            Doppler V_LOS approximation map (nx, ny) in km/s
        temp_approx : np.ndarray, optional
            Black-body temperature approximation map (nx, ny) in Kelvin
        blos_physics_mode : {'tau_averaged', 'single_height'}, optional
            Override the blos_physics_mode set at initialization
        blos_target_logtau : float, optional
            Override the blos_target_logtau set at initialization
        vlos_physics_mode : {'tau_averaged', 'single_height'}, optional
            Override the vlos_physics_mode set at initialization
        vlos_target_logtau : float, optional
            Override the vlos_target_logtau set at initialization
        temp_physics_mode : {'tau_averaged', 'single_height'}, optional
            Override the temp_physics_mode set at initialization
        temp_target_logtau : float, optional
            Override the temp_target_logtau set at initialization
        """
        self.mhd_normalizer = mhd_normalizer
        self.logtau_values = logtau_values
        
        # Update physics modes if provided
        valid_physics_modes = ['tau_averaged', 'single_height']
        if blos_physics_mode is not None:
            if blos_physics_mode not in valid_physics_modes:
                raise ValueError(f"blos_physics_mode must be one of {valid_physics_modes}")
            self.blos_physics_mode = blos_physics_mode
        if vlos_physics_mode is not None:
            if vlos_physics_mode not in valid_physics_modes:
                raise ValueError(f"vlos_physics_mode must be one of {valid_physics_modes}")
            self.vlos_physics_mode = vlos_physics_mode
        if temp_physics_mode is not None:
            if temp_physics_mode not in valid_physics_modes:
                raise ValueError(f"temp_physics_mode must be one of {valid_physics_modes}")
            self.temp_physics_mode = temp_physics_mode
        
        # Update target logtau values if provided
        if blos_target_logtau is not None:
            self.blos_target_logtau = blos_target_logtau
        if vlos_target_logtau is not None:
            self.vlos_target_logtau = vlos_target_logtau
        if temp_target_logtau is not None:
            self.temp_target_logtau = temp_target_logtau
        
        # Pre-compute tau integration quantities
        self._tau_linear = 10 ** logtau_values
        self._dtau = np.diff(self._tau_linear)
        self._integral_dtau = self._tau_linear[-1] - self._tau_linear[0]
        
        # Compute target logtau indices for single_height mode
        if self.blos_physics_mode == 'single_height':
            if self.blos_target_logtau is None:
                self._blos_target_logtau_idx = len(logtau_values) // 2
                self.blos_target_logtau = logtau_values[self._blos_target_logtau_idx]
            else:
                self._blos_target_logtau_idx = int(np.argmin(np.abs(logtau_values - self.blos_target_logtau)))
                self.blos_target_logtau = logtau_values[self._blos_target_logtau_idx]
        
        if self.vlos_physics_mode == 'single_height':
            if self.vlos_target_logtau is None:
                self._vlos_target_logtau_idx = len(logtau_values) // 2
                self.vlos_target_logtau = logtau_values[self._vlos_target_logtau_idx]
            else:
                self._vlos_target_logtau_idx = int(np.argmin(np.abs(logtau_values - self.vlos_target_logtau)))
                self.vlos_target_logtau = logtau_values[self._vlos_target_logtau_idx]
        
        if self.temp_physics_mode == 'single_height':
            if self.temp_target_logtau is None:
                # Default to log(tau) = 0.0 (photosphere)
                self._temp_target_logtau_idx = int(np.argmin(np.abs(logtau_values - 0.0)))
                self.temp_target_logtau = logtau_values[self._temp_target_logtau_idx]
            else:
                self._temp_target_logtau_idx = int(np.argmin(np.abs(logtau_values - self.temp_target_logtau)))
                self.temp_target_logtau = logtau_values[self._temp_target_logtau_idx]
        
        # Move approximations to device
        device = self._get_device()
        if blos_approx is not None:
            self.blos_approx = torch.tensor(blos_approx, dtype=torch.float32, device=device)
        if vlos_approx is not None:
            self.vlos_approx = torch.tensor(vlos_approx, dtype=torch.float32, device=device)
        if temp_approx is not None:
            self.temp_approx = torch.tensor(temp_approx, dtype=torch.float32, device=device)

    def _get_device(self) -> torch.device:
        """Get device of model parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _denormalize_predictions(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Denormalize model predictions to physical units.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Normalized predictions (batch_size, 63)
            
        Returns
        -------
        denorm_dict : Dict[str, torch.Tensor]
            Dictionary with 'T', 'Vz', 'Bz' in physical units
        """
        if self.mhd_normalizer is None:
            raise RuntimeError("MHD normalizer not set. Call set_physics_context() first.")
        
        # Convert to numpy for denormalization
        predictions_np = predictions.detach().cpu().numpy()
        
        # Denormalize using the normalizer
        denorm_dict = self.mhd_normalizer.inverse_transform(
            predictions_np, param_order=['T', 'Vz', 'Bz']
        )
        
        # Convert back to torch tensors on correct device
        device = self._get_device()
        denorm_torch = {
            'T': torch.tensor(denorm_dict['T'], dtype=torch.float32, device=device),
            'Vz': torch.tensor(denorm_dict['Vz'], dtype=torch.float32, device=device),
            'Bz': torch.tensor(denorm_dict['Bz'], dtype=torch.float32, device=device),
        }
        
        return denorm_torch

    def _compute_tau_averaged_blos(self, bz: torch.Tensor) -> torch.Tensor:
        """
        Compute tau-averaged B_LOS from Bz profile.
        
        Parameters
        ----------
        bz : torch.Tensor
            Bz values in physical units (batch_size, n_tau=21)
            
        Returns
        -------
        blos : torch.Tensor
            Tau-averaged B_LOS (batch_size,)
        """
        device = self._get_device()
        
        # Trapezoidal integration
        bz_avg = (bz[:, :-1] + bz[:, 1:]) / 2  # (batch_size, 20)
        dtau_tensor = torch.tensor(self._dtau, dtype=torch.float32, device=device)
        integral_bz = torch.sum(bz_avg * dtau_tensor[None, :], dim=1)  # (batch_size,)
        
        return integral_bz / self._integral_dtau

    def _compute_tau_averaged_vlos(self, vz: torch.Tensor) -> torch.Tensor:
        """
        Compute tau-averaged V_LOS from Vz profile.
        
        Parameters
        ----------
        vz : torch.Tensor
            Vz values in physical units (batch_size, n_tau=21)
            
        Returns
        -------
        vlos : torch.Tensor
            Tau-averaged V_LOS (batch_size,)
        """
        device = self._get_device()
        
        # Trapezoidal integration
        vz_avg = (vz[:, :-1] + vz[:, 1:]) / 2  # (batch_size, 20)
        dtau_tensor = torch.tensor(self._dtau, dtype=torch.float32, device=device)
        integral_vz = torch.sum(vz_avg * dtau_tensor[None, :], dim=1)  # (batch_size,)
        
        return integral_vz / self._integral_dtau

    def _compute_tau_averaged_temperature(self, temp: torch.Tensor) -> torch.Tensor:
        """
        Compute tau-averaged temperature from T profile.
        
        Parameters
        ----------
        temp : torch.Tensor
            Temperature values in physical units (batch_size, n_tau=21)
            
        Returns
        -------
        temp_avg : torch.Tensor
            Tau-averaged temperature (batch_size,)
        """
        device = self._get_device()
        
        # Trapezoidal integration
        temp_avg = (temp[:, :-1] + temp[:, 1:]) / 2  # (batch_size, 20)
        dtau_tensor = torch.tensor(self._dtau, dtype=torch.float32, device=device)
        integral_temp = torch.sum(temp_avg * dtau_tensor[None, :], dim=1)  # (batch_size,)
        
        return integral_temp / self._integral_dtau

    def _extract_at_logtau(self, values: torch.Tensor, target_idx: int) -> torch.Tensor:
        """
        Extract values at the specified optical depth index.
        
        Parameters
        ----------
        values : torch.Tensor
            Values across optical depths (batch_size, n_tau)
        target_idx : int
            Target optical depth index
            
        Returns
        -------
        torch.Tensor
            Values at target optical depth (batch_size,)
        """
        return values[:, target_idx]

    def _compute_predicted_blos(self, denorm_pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute B_LOS from predictions based on blos_physics_mode.
        
        Parameters
        ----------
        denorm_pred : Dict[str, torch.Tensor]
            Denormalized predictions with 'Bz' (batch_size, n_tau)
            
        Returns
        -------
        torch.Tensor
            B_LOS values (batch_size,)
        """
        if self.blos_physics_mode == 'tau_averaged':
            return self._compute_tau_averaged_blos(denorm_pred['Bz'])
        else:  # single_height
            if self._blos_target_logtau_idx is None:
                raise RuntimeError("B_LOS target logtau index not set. Call set_physics_context() first.")
            return self._extract_at_logtau(denorm_pred['Bz'], self._blos_target_logtau_idx)

    def _compute_predicted_vlos(self, denorm_pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute V_LOS from predictions based on vlos_physics_mode.
        
        Parameters
        ----------
        denorm_pred : Dict[str, torch.Tensor]
            Denormalized predictions with 'Vz' (batch_size, n_tau)
            
        Returns
        -------
        torch.Tensor
            V_LOS values (batch_size,)
        """
        if self.vlos_physics_mode == 'tau_averaged':
            return self._compute_tau_averaged_vlos(denorm_pred['Vz'])
        else:  # single_height
            if self._vlos_target_logtau_idx is None:
                raise RuntimeError("V_LOS target logtau index not set. Call set_physics_context() first.")
            return self._extract_at_logtau(denorm_pred['Vz'], self._vlos_target_logtau_idx)

    def _compute_predicted_temperature(self, denorm_pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute temperature from predictions based on temp_physics_mode.
        
        Parameters
        ----------
        denorm_pred : Dict[str, torch.Tensor]
            Denormalized predictions with 'T' (batch_size, n_tau)
            
        Returns
        -------
        torch.Tensor
            Temperature values (batch_size,)
        """
        if self.temp_physics_mode == 'tau_averaged':
            return self._compute_tau_averaged_temperature(denorm_pred['T'])
        else:  # single_height
            if self._temp_target_logtau_idx is None:
                raise RuntimeError("Temperature target logtau index not set. Call set_physics_context() first.")
            return self._extract_at_logtau(denorm_pred['T'], self._temp_target_logtau_idx)

    def _compute_wfa_loss(
        self,
        denorm_pred: Dict[str, torch.Tensor],
        spatial_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute WFA-based B_LOS loss in physical units.
        
        Parameters
        ----------
        denorm_pred : Dict[str, torch.Tensor]
            Denormalized predictions with 'Bz' (batch_size, n_tau)
        spatial_indices : torch.Tensor
            Spatial coordinates (batch_size, 2) as [y, x]
            
        Returns
        -------
        loss : torch.Tensor
            WFA loss value
        """
        if self.blos_approx is None:
            raise RuntimeError("B_LOS approximation not set. Call set_physics_context() first.")
        
        # Compute B_LOS from predictions based on physics_mode
        pred_blos = self._compute_predicted_blos(denorm_pred)
        
        # Get corresponding WFA approximations
        y_idx = spatial_indices[:, 0].long()
        x_idx = spatial_indices[:, 1].long()
        approx_blos = self.blos_approx[y_idx, x_idx]  # (batch_size,)
        
        # MSE loss in physical units
        return torch.mean((pred_blos - approx_blos) ** 2)

    def _compute_doppler_loss(
        self,
        denorm_pred: Dict[str, torch.Tensor],
        spatial_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Doppler-based V_LOS loss in physical units.
        
        Parameters
        ----------
        denorm_pred : Dict[str, torch.Tensor]
            Denormalized predictions with 'Vz' (batch_size, n_tau)
        spatial_indices : torch.Tensor
            Spatial coordinates (batch_size, 2) as [y, x]
            
        Returns
        -------
        loss : torch.Tensor
            Doppler loss value
        """
        if self.vlos_approx is None:
            raise RuntimeError("V_LOS approximation not set. Call set_physics_context() first.")
        
        # Compute V_LOS from predictions based on physics_mode
        pred_vlos = self._compute_predicted_vlos(denorm_pred)
        
        # Get corresponding Doppler approximations
        y_idx = spatial_indices[:, 0].long()
        x_idx = spatial_indices[:, 1].long()
        approx_vlos = self.vlos_approx[y_idx, x_idx]  # (batch_size,)
        
        # Handle NaN values in approximations (from failed Gaussian fits)
        valid_mask = ~torch.isnan(approx_vlos)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self._get_device())
        
        # MSE loss in physical units (only valid pixels)
        return torch.mean((pred_vlos[valid_mask] - approx_vlos[valid_mask]) ** 2)

    def _compute_temperature_loss(
        self,
        denorm_pred: Dict[str, torch.Tensor],
        spatial_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temperature loss in physical units.
        
        Parameters
        ----------
        denorm_pred : Dict[str, torch.Tensor]
            Denormalized predictions with 'T' (batch_size, n_tau)
        spatial_indices : torch.Tensor
            Spatial coordinates (batch_size, 2) as [y, x]
            
        Returns
        -------
        loss : torch.Tensor
            Temperature loss value
        """
        if self.temp_approx is None:
            raise RuntimeError("Temperature approximation not set. Call set_physics_context() first.")
        
        # Compute temperature from predictions based on physics_mode
        pred_temp = self._compute_predicted_temperature(denorm_pred)
        
        # Get corresponding temperature approximations
        y_idx = spatial_indices[:, 0].long()
        x_idx = spatial_indices[:, 1].long()
        approx_temp = self.temp_approx[y_idx, x_idx]  # (batch_size,)
        
        # MSE loss in physical units (Kelvin)
        return torch.mean((pred_temp - approx_temp) ** 2)

    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        spatial_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-based regularization losses in physical units.
        
        This method computes all enabled physics losses by:
        1. Denormalizing predictions to physical units
        2. Computing tau-averaged or single-height quantities (B_LOS, V_LOS, T)
        3. Comparing against physics approximations
        
        Parameters
        ----------
        predictions : torch.Tensor
            Normalized model predictions (batch_size, 63)
        spatial_indices : torch.Tensor
            Spatial coordinates (batch_size, 2) as [y, x]
            
        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of physics losses
        loss_components : Dict[str, float]
            Individual loss components for logging
        """
        # Return zero if physics is disabled
        if self.use_physics is None:
            device = self._get_device()
            return torch.tensor(0.0, device=device), {}
        
        # Denormalize predictions to physical units
        denorm_pred = self._denormalize_predictions(predictions)
        
        device = self._get_device()
        total_loss = torch.tensor(0.0, device=device)
        loss_components = {}
        
        # Add physics mode info to components
        loss_components['blos_physics_mode'] = self.blos_physics_mode
        loss_components['vlos_physics_mode'] = self.vlos_physics_mode
        loss_components['temp_physics_mode'] = self.temp_physics_mode
        if self.blos_physics_mode == 'single_height':
            loss_components['blos_target_logtau'] = self.blos_target_logtau
        if self.vlos_physics_mode == 'single_height':
            loss_components['vlos_target_logtau'] = self.vlos_target_logtau
        if self.temp_physics_mode == 'single_height':
            loss_components['temp_target_logtau'] = self.temp_target_logtau
        
        # WFA B_LOS loss (if enabled)
        if self.use_physics in ['wfa', 'both', 'all'] and self.lambda_wfa > 0:
            wfa_loss = self._compute_wfa_loss(denorm_pred, spatial_indices)
            loss_components['wfa'] = wfa_loss.item()
            total_loss = total_loss + self.lambda_wfa * wfa_loss
        
        # Doppler V_LOS loss (if enabled)
        if self.use_physics in ['doppler', 'both', 'all'] and self.lambda_doppler > 0:
            doppler_loss = self._compute_doppler_loss(denorm_pred, spatial_indices)
            loss_components['doppler'] = doppler_loss.item()
            total_loss = total_loss + self.lambda_doppler * doppler_loss
        
        # Temperature loss (if enabled)
        if self.use_physics in ['temperature', 'all'] and self.lambda_temp > 0:
            temp_loss = self._compute_temperature_loss(denorm_pred, spatial_indices)
            loss_components['temperature'] = temp_loss.item()
            total_loss = total_loss + self.lambda_temp * temp_loss
        
        return total_loss, loss_components

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        spatial_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with optional physics regularization.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions (batch_size, 63)
        targets : torch.Tensor
            Ground truth targets (batch_size, 63)
        spatial_indices : torch.Tensor, optional
            Spatial coordinates (batch_size, 2) for physics losses
            
        Returns
        -------
        loss_dict : Dict[str, torch.Tensor]
            Dictionary containing 'loss', 'mse', 'physics', and components
        """
        # Supervised MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Physics regularization
        if self.use_physics is not None and spatial_indices is not None:
            physics_loss, physics_components = self.compute_physics_loss(
                predictions, spatial_indices
            )
        else:
            device = self._get_device()
            physics_loss = torch.tensor(0.0, device=device)
            physics_components = {}
        
        # Total loss
        total_loss = mse_loss + physics_loss

        return {
            "loss": total_loss,
            "mse": mse_loss,
            "physics": physics_loss,
            **physics_components
        }