import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Optional, List
import torch

class MhdNormalizer:
    """Compute normalization statistics incrementally with per-optical-depth normalization."""
    
    def __init__(self, n_tau: int = 21, epsilon: float = 1e-8):
        """
        Initialize normalizer with per-τ statistics.
        
        Args:
            n_tau: number of optical depth levels (default: 21)
            epsilon: small value to avoid division by zero
        """
        self.n_tau = n_tau
        self.epsilon = epsilon
        self.finalized = False
        self.final_stats = None
        
        # Initialize statistics for each parameter at each τ level
        self.stats = {}
        for param in ['T', 'Vz', 'Bz']:
            self.stats[param] = []
            for tau_idx in range(n_tau):
                self.stats[param].append({
                    'n': 0,
                    'mean': 0.0,
                    'M2': 0.0,
                    'tau_idx': tau_idx
                })
        
    def update(self, od_data: dict):
        """
        Update running statistics per optical depth level.
        
        Args:
            od_data: dict with keys 'T', 'Vz', 'Bz' 
                     containing (nx, ny, n_tau) arrays
        """
        if self.finalized:
            raise RuntimeError("Cannot update finalized normalizer.")
        
        for param in ['T', 'Vz', 'Bz']:
            data = od_data[param].value if hasattr(od_data[param], 'value') else od_data[param]
            
            # Check shape
            if data.shape[2] != self.n_tau:
                raise ValueError(f"Expected {self.n_tau} τ levels, got {data.shape[2]}")
            
            # Process each τ level independently
            for tau_idx in range(self.n_tau):
                # Extract data at this τ level (nx, ny)
                data_tau = data[:, :, tau_idx]
                
                # Apply log transform for Bz
                if param == 'Bz':
                    sign = np.sign(data_tau)
                    data_tau = sign * np.log10(np.abs(data_tau) + 1.0)
                
                # Flatten to 1D
                data_flat = data_tau.ravel()
                
                # Welford's algorithm for this τ level
                for x in data_flat:
                    self.stats[param][tau_idx]['n'] += 1
                    delta = x - self.stats[param][tau_idx]['mean']
                    self.stats[param][tau_idx]['mean'] += delta / self.stats[param][tau_idx]['n']
                    delta2 = x - self.stats[param][tau_idx]['mean']
                    self.stats[param][tau_idx]['M2'] += delta * delta2
    
    def finalize(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert accumulated statistics to mean/std for each τ level.
        
        Returns:
            final_stats: dict with normalization parameters per τ level
        """
        if self.finalized:
            return self.final_stats
        
        self.final_stats = {}
        for param in ['T', 'Vz', 'Bz']:
            self.final_stats[param] = []
            
            for tau_idx in range(self.n_tau):
                n = self.stats[param][tau_idx]['n']
                mean = self.stats[param][tau_idx]['mean']
                variance = self.stats[param][tau_idx]['M2'] / n if n > 1 else 0.0
                std = np.sqrt(variance)
                
                self.final_stats[param].append({
                    'tau_idx': tau_idx,
                    'mean': float(mean),
                    'std': float(std),
                    'n_samples': int(n),
                    'type': 'signum_log' if param == 'Bz' else ('standard' if param == 'T' else 'centered')
                })
        
        self.finalized = True
        return self.final_stats
    
    def transform(self, od_data: dict) -> dict:
        """
        Normalize atmospheric parameters per optical depth level.
        
        Args:
            od_data: dict with keys 'T', 'Vz', 'Bz'
            
        Returns:
            normalized_data: dict with normalized arrays
        """
        if not self.finalized:
            raise RuntimeError("Normalizer not finalized. Call finalize() first.")
        
        normalized = {}
        
        for param in ['T', 'Vz', 'Bz']:
            data = od_data[param].value if hasattr(od_data[param], 'value') else od_data[param]
            
            # Initialize normalized array
            normalized_data = np.zeros_like(data, dtype=np.float32)
            
            # Normalize each τ level independently
            for tau_idx in range(self.n_tau):
                data_tau = data[:, :, tau_idx]
                
                # Apply log transform for Bz
                if param == 'Bz':
                    sign = np.sign(data_tau)
                    data_tau = sign * np.log10(np.abs(data_tau) + 1.0)
                
                # Get statistics for this τ level
                stats_tau = self.final_stats[param][tau_idx]
                
                # Normalize
                if stats_tau['type'] == 'standard':
                    normalized_data[:, :, tau_idx] = (data_tau - stats_tau['mean']) / (stats_tau['std'] + self.epsilon)
                elif stats_tau['type'] == 'centered':
                    normalized_data[:, :, tau_idx] = data_tau / (stats_tau['std'] + self.epsilon)
                elif stats_tau['type'] == 'signum_log':
                    normalized_data[:, :, tau_idx] = (data_tau - stats_tau['mean']) / (stats_tau['std'] + self.epsilon)
            
            normalized[param] = normalized_data
        
        return normalized
    
    def denormalize(self, normalized_data: np.ndarray, param: str) -> np.ndarray:
        """
        Denormalize predictions back to physical units (per-τ).
        
        Parameters
        ----------
        normalized_data : np.ndarray
            Normalized predictions, shape (n_pixels, n_tau) or (n_samples, n_tau)
        param : str
            Parameter name ('T', 'Vz', or 'Bz')
            
        Returns
        -------
        denormalized : np.ndarray
            Denormalized values in physical units, same shape as input
        """
        if not self.finalized:
            raise RuntimeError("Normalizer not finalized. Call finalize() first.")
        
        if param not in ['T', 'Vz', 'Bz']:
            raise ValueError(f"Unknown parameter: {param}")
        
        # Initialize output array
        denormalized = np.zeros_like(normalized_data, dtype=np.float32)
        
        # Denormalize each τ level independently
        for tau_idx in range(self.n_tau):
            stats_tau = self.final_stats[param][tau_idx]
            mean = stats_tau['mean']
            std = stats_tau['std']
            
            # Get normalized data for this τ level
            normalized_tau = normalized_data[:, tau_idx]
            
            # Denormalize: x = (x_norm * std) + mean
            denormalized_tau = (normalized_tau * std) + mean
            
            # For Bz, reverse the log transform
            if param == 'Bz':
                sign = np.sign(denormalized_tau)
                denormalized_tau = sign * (10.0 ** np.abs(denormalized_tau) - 1.0)
            
            # Clip to reasonable ranges to avoid extreme outliers
            if param == 'T':
                denormalized_tau = np.clip(denormalized_tau, 3000, 15000)  # Temperature in K
            elif param == 'Vz':
                denormalized_tau = np.clip(denormalized_tau, -20, 20)  # Velocity in km/s
            elif param == 'Bz':
                denormalized_tau = np.clip(denormalized_tau, -3000, 3000)  # Magnetic field in G
            
            denormalized[:, tau_idx] = denormalized_tau
        
        return denormalized
    
    def save(self, filepath: str):
        """Save finalized normalization statistics."""
        if not self.finalized:
            print("Warning: Normalizer not finalized. Finalizing...")
            self.finalize()
        
        save_dict = {
            'final_stats': self.final_stats,
            'n_tau': self.n_tau,
            'epsilon': self.epsilon,
            'finalized': self.finalized,
            'version': '2.0_per_tau'
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"Per-τ normalization statistics saved to {filepath}")
        print(f"\nStatistics summary (first 3 τ levels):")
        for param in ['T', 'Vz', 'Bz']:
            print(f"\n{param}:")
            for tau_idx in [0, 10, 20]:  # Show beginning, middle, end
                stats = self.final_stats[param][tau_idx]
                print(f"  τ={tau_idx:2d}: mean={stats['mean']:8.4f}, std={stats['std']:7.4f}, n={stats['n_samples']:,}")
    
    def load(self, filepath: str):
        """Load normalization statistics from JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Not found: {filepath}")
        
        with open(filepath, 'r') as f:
            save_dict = json.load(f)
        
        # Check version
        version = save_dict.get('version', '1.0_global')
        if version == '1.0_global':
            raise ValueError(
                "Loaded statistics use global normalization (old version). "
                "Per-τ normalization required. Please recompute statistics."
            )
        
        self.final_stats = save_dict['final_stats']
        self.n_tau = save_dict['n_tau']
        self.epsilon = save_dict.get('epsilon', 1e-8)
        self.finalized = save_dict.get('finalized', True)
        
        print(f"Per-τ normalization statistics loaded from {filepath}")
        print(f"Optical depth levels: {self.n_tau}")
        
        return self
    
    def save_state(self, filepath: str):
        """Save intermediate state for resumption."""
        if self.finalized:
            raise RuntimeError("Cannot save state of finalized normalizer.")
        
        save_dict = {
            'stats': self.stats,
            'n_tau': self.n_tau,
            'epsilon': self.epsilon,
            'finalized': self.finalized,
            'version': '2.0_per_tau'
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"Normalizer state saved to {filepath}")
        print(f"Progress: τ=0 processed {self.stats['T'][0]['n']:,} samples")
    
    def load_state(self, filepath: str):
        """Load intermediate state to resume computation."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Not found: {filepath}")
        
        with open(filepath, 'r') as f:
            save_dict = json.load(f)
        
        self.stats = save_dict['stats']
        self.n_tau = save_dict['n_tau']
        self.epsilon = save_dict.get('epsilon', 1e-8)
        self.finalized = save_dict.get('finalized', False)
        
        print(f"Normalizer state loaded from {filepath}")
        print(f"Progress: τ=0 processed {self.stats['T'][0]['n']:,} samples")
        
        return self

class StokesNormalizer:
    def __init__(self):
        self.stats = {
            'I': {'n': 0, 'mean': 0.0, 'M2': 0.0},
            'V': {'n': 0, 'mean': 0.0, 'M2': 0.0}
        }
        self.finalized = False
        self.final_stats = None
    
    def update(self, stokes_dict):
        """Update statistics with new data from one simulation step.
        
        Args:
            stokes_dict: {'I': (nx, ny, nλ), 'V': (nx, ny, nλ)}
        """
        if self.finalized:
            raise RuntimeError("Cannot update finalized normalizer.")
        
        for key in ['I', 'V']:
            data = stokes_dict[key].flatten()
            
            for x in data:
                self.stats[key]['n'] += 1
                delta = x - self.stats[key]['mean']
                self.stats[key]['mean'] += delta / self.stats[key]['n']
                delta2 = x - self.stats[key]['mean']
                self.stats[key]['M2'] += delta * delta2
    
    def finalize(self):
        """Compute final mean and std."""
        if self.finalized:
            return self.final_stats
        
        self.final_stats = {}
        for key in ['I', 'V']:
            n = self.stats[key]['n']
            mean = self.stats[key]['mean']
            std = np.sqrt(self.stats[key]['M2'] / n)
            self.final_stats[key] = {'mean': float(mean), 'std': float(std), 'n_samples': int(n)}
        
        self.finalized = True
        return self.final_stats
    
    def save(self, filepath):
        """Save statistics to JSON."""
        if not self.finalized:
            print("Warning: Normalizer not finalized. Finalizing...")
            self.finalize()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.final_stats, f, indent=2)
        
        print(f"Stokes normalization statistics saved to {filepath}")
        for key in ['I', 'V']:
            stats = self.final_stats[key]
            print(f"  {key}: mean={stats['mean']:8.4f}, std={stats['std']:7.4f}, n={stats['n_samples']:,}")
    
    def load(self, filepath):
        """Load statistics from JSON."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Not found: {filepath}")
        
        with open(filepath, 'r') as f:
            self.final_stats = json.load(f)
        
        self.finalized = True
        print(f"Stokes normalization statistics loaded from {filepath}")
        return self
    
    def transform(self, stokes_dict: dict, epsilon: float = 1e-8) -> dict:
        """
        Normalize Stokes parameters.
        
        Args:
            stokes_dict: dict with keys 'I' and 'V' containing arrays
            epsilon: small value to avoid division by zero
            
        Returns:
            normalized_data: dict with normalized arrays
        """
        if not self.finalized:
            raise RuntimeError("Normalizer not finalized. Call finalize() first.")
        
        normalized = {}
        
        for key in ['I', 'V']:
            data = stokes_dict[key].value if hasattr(stokes_dict[key], 'value') else stokes_dict[key]
            mean = self.final_stats[key]['mean']
            std = self.final_stats[key]['std']
            normalized[key] = (data - mean) / (std + epsilon)
        
        return normalized
    
    def inverse_transform(self, predictions: np.ndarray, param_order: list = ['I', 'V']) -> dict:
        """
        Denormalize model predictions back to physical units.
        
        Args:
            predictions: array from model with shape (n_samples, n_features)
            param_order: order of parameters in predictions (default: ['I', 'V'])
            
        Returns:
            denormalized: dict with keys from param_order in physical units
        """
        if not self.finalized:
            raise RuntimeError("Normalizer not finalized.")
        
        n_params = len(param_order)
        n_features_per_param = predictions.shape[1] // n_params
        
        denormalized = {}
        
        for i, key in enumerate(param_order):
            start_idx = i * n_features_per_param
            end_idx = (i + 1) * n_features_per_param
            pred_param = predictions[:, start_idx:end_idx]
            
            mean = self.final_stats[key]['mean']
            std = self.final_stats[key]['std']
            denormalized[key] = pred_param * std + mean
        
        return denormalized
    
    def save_state(self, filepath: str):
        """Save intermediate state for resumption."""
        if self.finalized:
            raise RuntimeError("Cannot save state of finalized normalizer.")
        
        save_dict = {
            'stats': self.stats,
            'finalized': self.finalized,
            'version': '1.0_stokes'
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"Stokes normalizer state saved to {filepath}")
        print(f"Progress: I processed {self.stats['I']['n']:,} samples")
        print(f"          V processed {self.stats['V']['n']:,} samples")
    
    def load_state(self, filepath: str):
        """Load intermediate state to resume computation."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Not found: {filepath}")
        
        with open(filepath, 'r') as f:
            save_dict = json.load(f)
        
        self.stats = save_dict['stats']
        self.finalized = save_dict.get('finalized', False)
        
        print(f"Stokes normalizer state loaded from {filepath}")
        print(f"Progress: I processed {self.stats['I']['n']:,} samples")
        print(f"          V processed {self.stats['V']['n']:,} samples")
        
        return self