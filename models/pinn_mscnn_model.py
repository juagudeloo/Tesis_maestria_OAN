"""Physics-informed MSCNN model definition."""

from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import astropy.units as u

from models.mscnn_model import MSCNNInversionModel


class PhysicsInformedMSCNN(MSCNNInversionModel):
    """MSCNN inversion model with integrated physics-informed loss computation.

    This model extends the base MSCNN architecture with physics-based regularization
    computed in physical units (after denormalization). All physics computations are
    self-contained within the model.
    
    Parameters
    ----------
    use_physics : {None, 'wfa', 'doppler', 'both'}, optional
        Controls which physics regularization to apply:
        - None: No physics regularization (pure supervised learning)
        - 'wfa': Only WFA B_LOS regularization
        - 'doppler': Only Doppler V_LOS regularization  
        - 'both': Both WFA and Doppler regularization
    lambda_wfa : float
        Weight for WFA B_LOS loss term
    lambda_doppler : float
        Weight for Doppler V_LOS loss term
    lambda_physics : float
        Weight for gradient smoothness regularization
    """

    def __init__(
        self,
        *,
        use_physics: Optional[str] = None,
        lambda_wfa: float = 0.01,
        lambda_doppler: float = 0.01,
        lambda_physics: float = 0.001,
        dropout_rate: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(dropout_rate=dropout_rate, **kwargs)
        
        # Validate and store physics regularization mode
        valid_modes = [None, 'wfa', 'doppler', 'both']
        if use_physics not in valid_modes:
            raise ValueError(f"use_physics must be one of {valid_modes}, got {use_physics}")
        
        self.use_physics = use_physics
        self.lambda_wfa = float(lambda_wfa)
        self.lambda_doppler = float(lambda_doppler)
        self.lambda_physics = float(lambda_physics)
        
        # Physics computation state (set via set_physics_context)
        self.mhd_normalizer = None
        self.logtau_values = None
        self.blos_approx = None
        self.vlos_approx = None
        self._tau_linear = None
        self._dtau = None
        self._integral_dtau = None

    def set_physics_context(
        self,
        mhd_normalizer: 'MhdNormalizer',
        logtau_values: np.ndarray,
        blos_approx: Optional[np.ndarray] = None,
        vlos_approx: Optional[np.ndarray] = None,
    ):
        """
        Set physics computation context for the current training step.
        
        This should be called once per simulation step with the relevant
        physics approximations and normalizer.
        
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
        """
        self.mhd_normalizer = mhd_normalizer
        self.logtau_values = logtau_values
        
        # Pre-compute tau integration quantities
        self._tau_linear = 10 ** logtau_values
        self._dtau = np.diff(self._tau_linear)
        self._integral_dtau = self._tau_linear[-1] - self._tau_linear[0]
        
        # Move approximations to device
        device = self._get_device()
        if blos_approx is not None:
            self.blos_approx = torch.tensor(blos_approx, dtype=torch.float32, device=device)
        if vlos_approx is not None:
            self.vlos_approx = torch.tensor(vlos_approx, dtype=torch.float32, device=device)

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
        
        # Compute tau-averaged B_LOS from predictions
        pred_blos = self._compute_tau_averaged_blos(denorm_pred['Bz'])
        
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
        
        # Compute tau-averaged V_LOS from predictions
        pred_vlos = self._compute_tau_averaged_vlos(denorm_pred['Vz'])
        
        # Get corresponding Doppler approximations
        y_idx = spatial_indices[:, 0].long()
        x_idx = spatial_indices[:, 1].long()
        approx_vlos = self.vlos_approx[y_idx, x_idx]  # (batch_size,)
        
        # MSE loss in physical units
        return torch.mean((pred_vlos - approx_vlos) ** 2)

    def _compute_gradient_smoothness(
        self,
        denorm_pred: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute gradient smoothness regularization in physical units.
        
        Penalizes large second derivatives (non-smooth profiles) across optical depth.
        
        Parameters
        ----------
        denorm_pred : Dict[str, torch.Tensor]
            Denormalized predictions (batch_size, n_tau) for each parameter
            
        Returns
        -------
        loss : torch.Tensor
            Smoothness loss value (average over T, Vz, Bz)
        """
        device = self._get_device()
        total_loss = torch.tensor(0.0, device=device)
        
        for param in ['T', 'Vz', 'Bz']:
            values = denorm_pred[param]  # (batch_size, n_tau)
            
            # First derivative (central differences)
            grad1 = values[:, 2:] - values[:, :-2]  # (batch_size, n_tau-2)
            
            # Second derivative approximation
            grad2 = grad1[:, 1:] - grad1[:, :-1]  # (batch_size, n_tau-3)
            
            # L2 penalty on second derivatives
            smoothness = torch.mean(grad2 ** 2)
            total_loss = total_loss + smoothness
        
        return total_loss / 3  # Average over three parameters

    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        spatial_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-based regularization losses in physical units.
        
        This method computes all enabled physics losses by:
        1. Denormalizing predictions to physical units
        2. Computing tau-averaged quantities (B_LOS, V_LOS)
        3. Comparing against physics approximations
        4. Adding gradient smoothness regularization
        
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
        
        # WFA B_LOS loss (if enabled)
        if self.use_physics in ['wfa', 'both'] and self.lambda_wfa > 0:
            wfa_loss = self._compute_wfa_loss(denorm_pred, spatial_indices)
            loss_components['wfa'] = wfa_loss.item()
            total_loss = total_loss + self.lambda_wfa * wfa_loss
        
        # Doppler V_LOS loss (if enabled)
        if self.use_physics in ['doppler', 'both'] and self.lambda_doppler > 0:
            doppler_loss = self._compute_doppler_loss(denorm_pred, spatial_indices)
            loss_components['doppler'] = doppler_loss.item()
            total_loss = total_loss + self.lambda_doppler * doppler_loss
        
        # Gradient smoothness loss (if enabled)
        if self.lambda_physics > 0:
            grad_loss = self._compute_gradient_smoothness(denorm_pred)
            loss_components['gradient'] = grad_loss.item()
            total_loss = total_loss + self.lambda_physics * grad_loss
        
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