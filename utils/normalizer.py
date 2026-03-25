import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Optional, List
import torch

class MhdNormalizer:
    """Compute normalization statistics incrementally with per-optical-depth normalization.
    
    For Bz: applies per-τ asinh(Bz / B0_τ) transform where B0_τ = P60(|Bz|).
    Also computes global tail-weighting parameters: B0_w = P90, B1_w = P99.5.
    """
    
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
        self.logtau_values = None
        
        # Percentile-based transform scales and tail-loss parameters
        self.B0_transform_per_tau = None  # shape (n_tau,), transform scale per-τ
        self.B0_weight_start = None  # scalar, P90 for tail-weighting start
        self.B1_weight_saturation = None  # scalar, P99.5 for tail-weighting saturation
        self.huber_delta = None  # scalar, Huber threshold = 0.25 * B0_weight_start
        
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
        
        # For Bz, collect raw values (before any transform) for percentile calculation
        self.bz_raw_values = [[] for _ in range(n_tau)]
        
    def update(self, od_data: dict):
        """
        Update running statistics per optical depth level.
        
        For Bz: applies per-τ asinh(Bz / B0_τ) transform where B0_τ will be computed
        in finalize() from percentiles of collected raw values.
        
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
                
                # Flatten to 1D
                data_flat = data_tau.ravel()
                
                # For Bz, collect raw values for percentile computation in finalize()
                # and apply asinh transform (with placeholder B0=1 for now, will rescale later)
                if param == 'Bz':
                    # Store raw absolute values for percentile calculation
                    self.bz_raw_values[tau_idx].extend(np.abs(data_flat).tolist())
                    # Apply asinh with temporary B0=1; will be rescaled in finalize()
                    data_tau = np.arcsinh(data_flat)  # asinh(Bz) with implicit B0=1
                else:
                    data_tau = data_flat
                
                # Welford's algorithm for this τ level
                for x in data_tau:
                    self.stats[param][tau_idx]['n'] += 1
                    delta = x - self.stats[param][tau_idx]['mean']
                    self.stats[param][tau_idx]['mean'] += delta / self.stats[param][tau_idx]['n']
                    delta2 = x - self.stats[param][tau_idx]['mean']
                    self.stats[param][tau_idx]['M2'] += delta * delta2
    
    def finalize(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert accumulated statistics to mean/std for each τ level.
        
        For Bz: computes per-τ asinh scaling parameter B0_tr from P60(|Bz|),
        then rescales accumulated asinh statistics. Also computes global
        tail-loss parameters: B0_w=P90, B1_w=P99.5, delta=0.25*B0_w.
        
        Returns:
            final_stats: dict with normalization parameters per τ level
        """
        if self.finalized:
            return self.final_stats
        
        # Compute Bz percentiles for asinh scaling and tail loss
        print("\nComputing Bz percentile parameters from raw data...")
        all_bz_raw = np.concatenate([np.array(vals) for vals in self.bz_raw_values if len(vals) > 0])
        if len(all_bz_raw) == 0:
            raise ValueError("No raw Bz data collected. Cannot compute percentiles.")
        
        # Global percentiles for tail loss (conservative defaults)
        self.B0_weight_start = float(np.percentile(all_bz_raw, 90.0))  # P90
        self.B1_weight_saturation = float(np.percentile(all_bz_raw, 99.5))  # P99.5
        self.huber_delta = 0.25 * self.B0_weight_start  # Huber threshold
        
        # Per-τ percentiles for asinh transform scaling
        self.B0_transform_per_tau = np.zeros(self.n_tau, dtype=np.float32)
        for tau_idx in range(self.n_tau):
            if len(self.bz_raw_values[tau_idx]) > 0:
                self.B0_transform_per_tau[tau_idx] = float(np.percentile(self.bz_raw_values[tau_idx], 60.0))
            else:
                self.B0_transform_per_tau[tau_idx] = 1.0  # fallback
        
        print(f"  Global B0_weight_start (P90):      {self.B0_weight_start:.2f} G")
        print(f"  Global B1_weight_saturation (P99.5): {self.B1_weight_saturation:.2f} G")
        print(f"  Huber delta:                        {self.huber_delta:.2f} G")
        print(f"  Per-τ B0_transform (P60) range:    [{self.B0_transform_per_tau.min():.2f}, {self.B0_transform_per_tau.max():.2f}] G")
        
        self.final_stats = {}
        for param in ['T', 'Vz', 'Bz']:
            self.final_stats[param] = []
            
            for tau_idx in range(self.n_tau):
                n = self.stats[param][tau_idx]['n']
                mean = self.stats[param][tau_idx]['mean']
                variance = self.stats[param][tau_idx]['M2'] / n if n > 1 else 0.0
                std = np.sqrt(variance)
                
                stats_entry = {
                    'tau_idx': tau_idx,
                    'mean': float(mean),
                    'std': float(std),
                    'n_samples': int(n),
                }
                
                if param == 'Bz':
                    stats_entry['type'] = 'asinh'
                    stats_entry['B0_transform'] = float(self.B0_transform_per_tau[tau_idx])
                elif param == 'T':
                    stats_entry['type'] = 'standard'
                else:
                    stats_entry['type'] = 'centered'
                
                self.final_stats[param].append(stats_entry)
        
        self.finalized = True
        return self.final_stats
    
    def transform(self, od_data: dict) -> dict:
        """
        Normalize atmospheric parameters per optical depth level.
        
        For Bz: applies per-τ asinh(Bz / B0_τ) transform then z-score normalization.
        
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
                
                # Apply asinh transform for Bz with per-τ scaling
                if param == 'Bz':
                    B0 = self.B0_transform_per_tau[tau_idx]
                    data_tau = np.arcsinh(data_tau / B0)
                
                # Get statistics for this τ level
                stats_tau = self.final_stats[param][tau_idx]
                
                # Normalize
                if stats_tau['type'] == 'standard':
                    normalized_data[:, :, tau_idx] = (data_tau - stats_tau['mean']) / (stats_tau['std'] + self.epsilon)
                elif stats_tau['type'] == 'centered':
                    normalized_data[:, :, tau_idx] = data_tau / (stats_tau['std'] + self.epsilon)
                elif stats_tau['type'] == 'asinh':
                    normalized_data[:, :, tau_idx] = (data_tau - stats_tau['mean']) / (stats_tau['std'] + self.epsilon)
            
            normalized[param] = normalized_data
        
        return normalized
    
    def denormalize(self, normalized_data: np.ndarray, param: str) -> np.ndarray:
        """
        Denormalize predictions back to physical units (per-τ).
        
        For Bz: reverses z-score normalization then applies inverse asinh:
        Bz = B0_τ * sinh(y_asinh).
        
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
            
            # For Bz, reverse the asinh transform: Bz = B0 * sinh(y)
            if param == 'Bz':
                B0 = self.B0_transform_per_tau[tau_idx]
                denormalized_tau = B0 * np.sinh(denormalized_tau)
            
            # Clip to reasonable ranges to avoid extreme outliers
            if param == 'T':
                denormalized_tau = np.clip(denormalized_tau, 3000, 15000)  # Temperature in K
            elif param == 'Vz':
                denormalized_tau = np.clip(denormalized_tau, -20, 20)  # Velocity in km/s
            elif param == 'Bz':
                denormalized_tau = np.clip(denormalized_tau, -10000, 10000)  # Magnetic field in G
            
            denormalized[:, tau_idx] = denormalized_tau
        
        return denormalized
    
    def save(self, filepath: str, logtau_values: Optional[List[float]] = None):
        """Save finalized normalization statistics with asinh transform parameters."""
        if not self.finalized:
            print("Warning: Normalizer not finalized. Finalizing...")
            self.finalize()

        save_dict = {
            'final_stats': self.final_stats,
            'n_tau': self.n_tau,
            'epsilon': self.epsilon,
            'finalized': self.finalized,
            'version': '3.0_asinh_per_tau'
        }
        
        # Save asinh transform scale per-τ
        if self.B0_transform_per_tau is not None:
            save_dict['B0_transform_per_tau'] = [float(x) for x in self.B0_transform_per_tau.tolist()]
        
        # Save global tail-loss parameters
        if self.B0_weight_start is not None:
            save_dict['B0_weight_start'] = float(self.B0_weight_start)
            save_dict['B1_weight_saturation'] = float(self.B1_weight_saturation)
            save_dict['huber_delta'] = float(self.huber_delta)
        
        if logtau_values is not None:
            arr = np.asarray(logtau_values, dtype=np.float32)
            save_dict['logtau_values'] = [float(x) for x in np.round(arr, 6).tolist()]

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"Per-τ normalization statistics (asinh) saved to {filepath}")
        print(f"\nAsinh transform scale (B0_transform per-τ) range: "
              f"[{self.B0_transform_per_tau.min():.2f}, {self.B0_transform_per_tau.max():.2f}] G")
        print(f"Tail-loss parameters: B0_weight_start={self.B0_weight_start:.2f} G, "
              f"B1_weight_saturation={self.B1_weight_saturation:.2f} G, delta={self.huber_delta:.2f} G")
        print(f"\nStatistics summary (first/middle/last τ levels):")
        for param in ['T', 'Vz', 'Bz']:
            print(f"\n{param}:")
            n_levels = len(self.final_stats[param])
            if n_levels == 0:
                print("  (no τ levels available)")
                continue

            summary_indices = sorted({0, n_levels // 2, n_levels - 1})
            for tau_idx in summary_indices:
                stats = self.final_stats[param][tau_idx]
                print(f"  τ={tau_idx:2d}: mean={stats['mean']:8.4f}, std={stats['std']:7.4f}, n={stats['n_samples']:,}")
    
    def load(self, filepath: str):
        """Load normalization statistics from JSON file (asinh version 3.0+ only)."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Not found: {filepath}")
        
        with open(filepath, 'r') as f:
            save_dict = json.load(f)
        
        # Check version and enforce hard switch from sign-log to asinh
        version = save_dict.get('version', '1.0_global')
        if version in ['1.0_global', '2.0_per_tau']:
            raise ValueError(
                f"Loaded statistics use version {version} (sign-log Bz transform). "
                "Switch to asinh transform requires recomputing normalization statistics. "
                "Run: python scripts/compute_normalization_stats.py --clean_start --logtau_values <your_values>"
            )
        
        if version != '3.0_asinh_per_tau':
            raise ValueError(
                f"Unsupported normalization version: {version}. "
                "Expected version 3.0_asinh_per_tau with asinh transform."
            )
        
        self.final_stats = save_dict['final_stats']
        self.n_tau = save_dict['n_tau']
        self.epsilon = save_dict.get('epsilon', 1e-8)
        self.finalized = save_dict.get('finalized', True)
        self.logtau_values = save_dict.get('logtau_values', None)
        
        # Load asinh transform parameters (new + legacy key names)
        B0_transform_values = save_dict.get('B0_transform_per_tau', save_dict.get('bz_B0_tr_per_tau'))
        if B0_transform_values is not None:
            self.B0_transform_per_tau = np.asarray(B0_transform_values, dtype=np.float32)
        
        # Load tail-loss parameters (new + legacy key names)
        B0_weight_start = save_dict.get('B0_weight_start', save_dict.get('bz_B0_w'))
        B1_weight_saturation = save_dict.get('B1_weight_saturation', save_dict.get('bz_B1_w'))
        huber_delta = save_dict.get('huber_delta', save_dict.get('bz_delta'))
        if B0_weight_start is not None and B1_weight_saturation is not None and huber_delta is not None:
            self.B0_weight_start = float(B0_weight_start)
            self.B1_weight_saturation = float(B1_weight_saturation)
            self.huber_delta = float(huber_delta)

        # Backward aliases for in-memory compatibility with older code paths
        self.bz_B0_tr_per_tau = self.B0_transform_per_tau
        self.bz_B0_w = self.B0_weight_start
        self.bz_B1_w = self.B1_weight_saturation
        self.bz_delta = self.huber_delta
        
        print(f"Per-τ normalization statistics (asinh) loaded from {filepath}")
        print(f"Optical depth levels: {self.n_tau}")
        if self.B0_weight_start is not None:
            print(f"Bz tail-loss params: B0_weight_start={self.B0_weight_start:.2f} G, "
                  f"B1_weight_saturation={self.B1_weight_saturation:.2f} G, delta={self.huber_delta:.2f} G")
        
        return self
    
    def save_state(self, filepath: str):
        """Save intermediate state for resumption (asinh version)."""
        if self.finalized:
            raise RuntimeError("Cannot save state of finalized normalizer.")
        
        save_dict = {
            'stats': self.stats,
            'n_tau': self.n_tau,
            'epsilon': self.epsilon,
            'finalized': self.finalized,
            'version': '3.0_asinh_per_tau'
        }
        
        # Save raw Bz values for percentile computation on resume
        save_dict['bz_raw_values'] = [list(vals) for vals in self.bz_raw_values]
        
        if self.logtau_values is not None:
            arr = np.asarray(self.logtau_values, dtype=np.float32)
            save_dict['logtau_values'] = [float(x) for x in np.round(arr, 6).tolist()]
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"Normalizer state saved to {filepath}")
        print(f"Progress: τ=0 processed {self.stats['T'][0]['n']:,} samples")
    
    def load_state(self, filepath: str):
        """Load intermediate state to resume computation (asinh version)."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Not found: {filepath}")
        
        with open(filepath, 'r') as f:
            save_dict = json.load(f)
        
        # Check version
        version = save_dict.get('version', '1.0_global')
        if version in ['1.0_global', '2.0_per_tau']:
            raise ValueError(
                f"Resume state uses version {version} (sign-log Bz transform). "
                "Cannot resume with asinh transform switch. "
                "Start fresh: python scripts/compute_normalization_stats.py --clean_start"
            )
        
        self.stats = save_dict['stats']
        self.n_tau = save_dict['n_tau']
        self.epsilon = save_dict.get('epsilon', 1e-8)
        self.finalized = save_dict.get('finalized', False)
        self.logtau_values = save_dict.get('logtau_values', None)
        
        # Restore raw Bz values for percentile computation
        if 'bz_raw_values' in save_dict:
            self.bz_raw_values = [list(vals) for vals in save_dict['bz_raw_values']]
        
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