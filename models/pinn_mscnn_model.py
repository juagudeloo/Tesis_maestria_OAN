"""Physics-informed MSCNN model definition."""

from typing import Dict, Optional, Tuple, Any, Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import astropy.units as u

from models.mscnn_model import MSCNNInversionModel
from utils.physics_utils import ApproxInversions


class PhysicsInformedMSCNN(MSCNNInversionModel):
    """MSCNN inversion model with optional physics-informed loss.

    The loss is MSE(predictions, targets) plus an optional regularization term built from
    the WFA-based B_LOS and/or Doppler-based V_LOS approximations. The
    regularization selects the optical-depth height with the lowest RRMSE by
    comparing each approximation to the MHD cubes, and then compares the
    approximation against the model's predicted optical-depth values (Bz or Vz)
    at that best height. If model-predicted optical-depth data is not provided,
    it falls back to comparing against the MHD cubes (previous behavior).
    
    Parameters
    ----------
    use_physics : {None, 'wfa', 'doppler', 'both'}, optional
        Controls which physics regularization to apply:
        - None: No physics regularization (pure supervised learning)
        - 'wfa': Only WFA B_LOS regularization
        - 'doppler': Only Doppler V_LOS regularization  
        - 'both': Both WFA and Doppler regularization (default)
    
    Notes
    -----
    Height selection (lowest RRMSE) is performed against the MHD optical-depth
    cubes, but the regularization mismatch can be computed against either the
    MHD cubes or the model's own predicted optical-depth cubes when provided
    via `predicted_od_data`.
    """

    def __init__(
        self,
        *,
        central_wavelength: u.Quantity = 6301.5 * u.Angstrom,
        lande_factor: float = 1.67,
        wl_range: Tuple[int, int] = (15, 60),
        lambda_reg: float = 0.1,
        use_physics: Optional[str] = 'both',
        dropout_rate: float = 0.2,  # Add dropout parameter
        **kwargs: Any,
    ) -> None:
        # Pass dropout_rate to parent MSCNNInversionModel
        super().__init__(dropout_rate=dropout_rate, **kwargs)
        
        self.central_wavelength = central_wavelength
        self.lande_factor = float(lande_factor)
        self.wl_range = [int(wl_range[0]), int(wl_range[1])]
        self.lambda_reg = float(lambda_reg)
        
        # Validate and store physics regularization mode
        valid_modes = [None, 'wfa', 'doppler', 'both']
        if use_physics not in valid_modes:
            raise ValueError(f"use_physics must be one of {valid_modes}, got {use_physics}")
        self.use_physics = use_physics

    def _get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _get_stokes_and_wavelength(
        self,
        stokes_input: Any,
        wavelength: Optional[np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        if isinstance(stokes_input, dict):
            stokes_dict = stokes_input
            wl_array = wavelength
        elif hasattr(stokes_input, "data"):
            stokes_dict = stokes_input.data
            wl_array = wavelength if wavelength is not None else getattr(stokes_input, "wl", None)
        else:
            raise ValueError("stokes_input must be a dict or expose 'data' and optionally 'wl'.")

        if wl_array is None:
            raise ValueError("Provide wavelength or ensure stokes_input has 'wl'.")
        return stokes_dict, np.asarray(wl_array)

    @staticmethod
    def _best_rrmse_index(approx_map: np.ndarray, mhd_cube: np.ndarray) -> Tuple[int, np.ndarray]:
        rrmse_values = []
        for k in range(mhd_cube.shape[2]):
            mhd_slice = mhd_cube[:, :, k]
            rmse = np.sqrt(np.mean((approx_map - mhd_slice) ** 2))
            denom = np.mean(np.abs(mhd_slice)) + 1e-8
            rrmse_values.append(rmse / denom)
        rrmse_values = np.asarray(rrmse_values)
        return int(np.argmin(rrmse_values)), rrmse_values

    def physics_regularization(
        self,
        *,
        mhd_od_data: Dict[str, Any],
        predicted_od_data: Optional[Dict[str, Any]] = None,
        blos_approx: Optional[np.ndarray] = None,
        vlos_approx: Optional[np.ndarray] = None,
        logtau_values: Optional[np.ndarray] = None,
        blos_best_index: Optional[int] = None,
        vlos_best_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        """Compute physics regularization using pre-computed approximations.
        
        Parameters
        ----------
        mhd_od_data : Dict[str, Any]
            MHD optical-depth cubes for height selection via RRMSE
        predicted_od_data : Dict[str, Any], optional
            Model-predicted optical-depth cubes for regularization mismatch
        blos_approx : np.ndarray, optional
            Pre-computed WFA B_LOS approximation map (ny, nx)
        vlos_approx : np.ndarray, optional
            Pre-computed Doppler V_LOS approximation map (ny, nx)
        logtau_values : np.ndarray, optional
            Log(tau) values for height reporting
        blos_best_index : int, optional
            Precomputed best-height index for B_LOS (global scene)
        vlos_best_index : int, optional
            Precomputed best-height index for V_LOS (global scene)
            
        Returns
        -------
        reg_tensor : torch.Tensor
            Total regularization value
        reg_components : Dict[str, torch.Tensor]
            Individual regularization components
        height_info : Dict[str, Any]
            Information about selected heights
        """
        # Return zero regularization if physics is disabled
        if self.use_physics is None:
            device = self._get_device()
            zero_tensor = torch.tensor(0.0, dtype=torch.float32, device=device)
            return zero_tensor, {}, {}

        device = self._get_device()
        reg_value = 0.0
        reg_components = {}
        height_info = {}
        
        # Compute WFA regularization if requested
        if self.use_physics in ['wfa', 'both']:
            if blos_approx is None:
                raise ValueError("blos_approx must be provided for WFA regularization when use_physics='wfa' or 'both'")
            blos = blos_approx
            mhd_bz = mhd_od_data.get("Bz")
            if mhd_bz is None:
                raise ValueError("mhd_od_data must contain 'Bz' for WFA regularization.")
            mhd_bz = mhd_bz.value if hasattr(mhd_bz, "value") else mhd_bz
            
            bz_rrmse = None
            if blos_best_index is not None:
                if blos_best_index < 0 or blos_best_index >= mhd_bz.shape[2]:
                    raise ValueError("blos_best_index is out of bounds for provided MHD cube")
                bz_idx = int(blos_best_index)
            else:
                bz_idx, bz_rrmse = self._best_rrmse_index(blos, mhd_bz)
            # Prefer model-predicted optical-depth data for mismatch, if available
            if predicted_od_data is not None and predicted_od_data.get("Bz") is not None:
                pred_bz = predicted_od_data["Bz"]
                pred_bz = pred_bz.value if hasattr(pred_bz, "value") else pred_bz
                bz_target = pred_bz[:, :, bz_idx]
            else:
                raise ValueError("predicted data must contain 'Bz' for WFA regularization.")
            reg_b = np.mean((blos - bz_target) ** 2)
            reg_value += reg_b
            
            reg_components["blos_mse"] = torch.as_tensor(reg_b, dtype=torch.float32, device=device)
            height_info.update({
                "blos_best_logtau": None if logtau_values is None else logtau_values[bz_idx],
                "blos_rrmse": bz_rrmse,
                "blos_best_index": bz_idx,
            })
        
        # Compute Doppler regularization if requested
        if self.use_physics in ['doppler', 'both']:
            if vlos_approx is None:
                raise ValueError("vlos_approx must be provided for Doppler regularization when use_physics='doppler' or 'both'")
            vlos = vlos_approx
            mhd_vz = mhd_od_data.get("Vz")
            if mhd_vz is None:
                raise ValueError("mhd_od_data must contain 'Vz' for Doppler regularization.")
            mhd_vz = mhd_vz.value if hasattr(mhd_vz, "value") else mhd_vz
            
            vz_rrmse = None
            if vlos_best_index is not None:
                if vlos_best_index < 0 or vlos_best_index >= mhd_vz.shape[2]:
                    raise ValueError("vlos_best_index is out of bounds for provided MHD cube")
                vz_idx = int(vlos_best_index)
            else:
                vz_idx, vz_rrmse = self._best_rrmse_index(vlos, mhd_vz)
            # Prefer model-predicted optical-depth data for mismatch, if available
            if predicted_od_data is not None and predicted_od_data.get("Vz") is not None:
                pred_vz = predicted_od_data["Vz"]
                pred_vz = pred_vz.value if hasattr(pred_vz, "value") else pred_vz
                vz_target = pred_vz[:, :, vz_idx]
            else:
                raise ValueError("predicted data must contain 'Vz' for Doppler regularization.")
            reg_v = np.mean((vlos - vz_target) ** 2)
            reg_value += reg_v
            
            reg_components["vlos_mse"] = torch.as_tensor(reg_v, dtype=torch.float32, device=device)
            height_info.update({
                "vlos_best_logtau": None if logtau_values is None else logtau_values[vz_idx],
                "vlos_rrmse": vz_rrmse,
                "vlos_best_index": vz_idx,
            })

        reg_tensor = torch.as_tensor(reg_value, dtype=torch.float32, device=device)
        return reg_tensor, reg_components, height_info

    def compute_loss(
        self,
        *,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mhd_od_data: Dict[str, Any],
        predicted_od_data: Optional[Dict[str, Any]] = None,
        blos_approx: Optional[np.ndarray] = None,
        vlos_approx: Optional[np.ndarray] = None,
        logtau_values: Optional[np.ndarray] = None,
        blos_best_index: Optional[int] = None,
        vlos_best_index: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss with optional physics regularization.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions (batch_size, n_outputs)
        targets : torch.Tensor
            Ground truth targets (batch_size, n_outputs)
        mhd_od_data : Dict[str, Any]
            MHD optical-depth cubes for height selection
        predicted_od_data : Dict[str, Any], optional
            Model-predicted optical-depth cubes for regularization
        blos_approx : np.ndarray, optional
            Pre-computed WFA B_LOS approximation
        vlos_approx : np.ndarray, optional
            Pre-computed Doppler V_LOS approximation
        logtau_values : np.ndarray, optional
            Log(tau) values for height reporting
        blos_best_index : int, optional
            Precomputed best-height index for B_LOS (global scene)
        vlos_best_index : int, optional
            Precomputed best-height index for V_LOS (global scene)
            
        Returns
        -------
        loss_dict : Dict[str, torch.Tensor]
            Dictionary containing 'loss', 'mse', 'regularization', etc.
        """
        base_mse = F.mse_loss(predictions, targets)
        reg_tensor, reg_components, height_info = self.physics_regularization(
            mhd_od_data=mhd_od_data,
            predicted_od_data=predicted_od_data,
            blos_approx=blos_approx,
            vlos_approx=vlos_approx,
            logtau_values=logtau_values,
            blos_best_index=blos_best_index,
            vlos_best_index=vlos_best_index,
        )

        total_loss = base_mse + self.lambda_reg * reg_tensor

        return {
            "loss": total_loss,
            "mse": base_mse,
            "regularization": reg_tensor,
            "regularization_parts": reg_components,
            "height_info": height_info,
        }