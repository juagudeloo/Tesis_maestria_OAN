"""
Cache Management System for Processed MURaM Data
=================================================

Caches processed Stokes and MHD data in HDF5 format to avoid reprocessing.
Dramatically reduces training setup time when data has already been processed.

Usage:
    from utils.cache_manage import DataCache
    
    cache = DataCache(cache_dir="/path/to/cache")
    
    # Check if step is cached
    if cache.exists(step=100):
        dataset, approx_data = cache.load(step=100, normalizers=(mhd_norm, stokes_norm))
    else:
        # Process and cache
        dataset, approx_data = load_and_prepare_step(...)
        cache.save(step=100, dataset=dataset, approx_data=approx_data)
"""

import json
import hashlib
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

from utils.muram_data import MuramStepDataset
from utils.normalizer import MhdNormalizer, StokesNormalizer


class DataCache:
    """
    Cache manager for processed MURaM data using HDF5 format.
    
    Stores:
    - Normalized Stokes data (I, V)
    - Normalized MHD data (T, Vz, Bz)
    - Physics approximations (blos, vlos, temp)
    - Configuration metadata
    
    Features:
    - Efficient HDF5 storage with compression
    - Configuration hashing to detect incompatible changes
    - Automatic directory creation
    - Progress reporting
    """
    
    def __init__(self, cache_dir: str = ".data_cache", compression: str = "gzip"):
        """
        Initialize cache manager.
        
        Parameters
        ----------
        cache_dir : str
            Root directory for cached data
        compression : str
            HDF5 compression type ('gzip', 'lzf', or None)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict[str, dict]:
        """Load metadata file tracking cached steps."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    @staticmethod
    def _config_hash(config_dict: dict) -> str:
        """
        Compute hash of configuration parameters.
        
        Parameters
        ----------
        config_dict : dict
            Configuration to hash
            
        Returns
        -------
        str
            Hex hash of configuration
        """
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    @staticmethod
    def make_config_hash(config_dict: dict) -> str:
        """Public helper for shared cache hashing across scripts."""
        return DataCache._config_hash(config_dict)
    
    def _get_cache_path(self, step: int) -> Path:
        """Get cache file path for a step."""
        return self.cache_dir / f"step_{step:06d}.h5"
    
    def exists(self, step: int, config_hash: str | None = None) -> bool:
        """
        Check if processed data exists in cache.
        
        Parameters
        ----------
        step : int
            Simulation step number
        config_hash : str, optional
            Configuration hash to validate. If provided, checks that
            cached config matches.
            
        Returns
        -------
        bool
            True if cache exists and is valid
        """
        cache_path = self._get_cache_path(step)
        if not cache_path.exists():
            return False
        
        # If config_hash provided, validate
        if config_hash is not None:
            step_key = str(step)
            if step_key not in self.metadata:
                return False
            cached_hash = self.metadata[step_key].get('config_hash')
            if cached_hash != config_hash:
                print(f"  ⚠ Cache config mismatch for step {step}")
                print(f"    Expected: {config_hash}, Found: {cached_hash}")
                return False
        
        return True
    
    def save(
        self,
        step: int,
        stokes_data: dict[str, np.ndarray],
        mhd_data: dict[str, np.ndarray],
        approx_data: dict[str, np.ndarray] | None = None,
        config_hash: str | None = None,
        verbose: bool = True,
    ) -> Path:
        """
        Save processed data to cache.
        
        Parameters
        ----------
        step : int
            Simulation step number
        stokes_data : dict
            Processed Stokes data {'I': (nx, ny, nλ), 'V': ...}
        mhd_data : dict
            Processed MHD data {'T': (nx, ny, nτ), 'Vz': ..., 'Bz': ...}
        approx_data : dict
            Physics approximations {'blos': (nx, ny), 'vlos': ..., 'temp': ...}
        config_hash : str, optional
            Configuration hash for validation
        verbose : bool
            Print progress messages
            
        Returns
        -------
        Path
            Path to cache file
        """
        cache_path = self._get_cache_path(step)
        
        try:
            with h5py.File(cache_path, 'w') as f:
                # Store Stokes data
                stokes_group = f.create_group('stokes')
                for key, data in stokes_data.items():
                    stokes_group.create_dataset(
                        key, data=data, compression=self.compression, compression_opts=4
                    )
                
                # Store MHD data
                mhd_group = f.create_group('mhd')
                for key, data in mhd_data.items():
                    mhd_group.create_dataset(
                        key, data=data, compression=self.compression, compression_opts=4
                    )
                
                # Store physics approximations
                approx_group = f.create_group('approximations')
                if approx_data is not None:
                    for key, data in approx_data.items():
                        approx_group.create_dataset(
                            key, data=data, compression=self.compression, compression_opts=4
                        )
                
                # Store metadata
                f.attrs['step'] = step
                if config_hash is not None:
                    f.attrs['config_hash'] = config_hash
            
            # Update metadata
            self.metadata[str(step)] = {
                'config_hash': config_hash or 'unknown',
                'nx': stokes_data['I'].shape[0],
                'ny': stokes_data['I'].shape[1],
                'nwl': stokes_data['I'].shape[2],
                'ntau': mhd_data['T'].shape[2],
                'file_size_mb': cache_path.stat().st_size / (1024**2),
            }
            self._save_metadata()
            
            if verbose:
                file_size_mb = cache_path.stat().st_size / (1024**2)
                print(f"  ✓ Cached step {step} ({file_size_mb:.1f} MB)")
            
            return cache_path
        
        except Exception as e:
            print(f"  ✗ Failed to cache step {step}: {e}")
            # Clean up partial file
            if cache_path.exists():
                cache_path.unlink()
            raise
    
    def load(
        self,
        step: int,
        stokes_normalizer: StokesNormalizer,
        mhd_normalizer: MhdNormalizer,
        verbose: bool = True,
    ) -> tuple[MuramStepDataset, dict[str, np.ndarray]]:
        """
        Load processed data from cache and create dataset.
        
        Parameters
        ----------
        step : int
            Simulation step number
        stokes_normalizer : StokesNormalizer
            Normalizer for Stokes data
        mhd_normalizer : MhdNormalizer
            Normalizer for MHD data
        verbose : bool
            Print progress messages
            
        Returns
        -------
        tuple
            (MuramStepDataset, approx_data_dict)
        """
        cache_path = self._get_cache_path(step)
        
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
        
        try:
            with h5py.File(cache_path, 'r') as f:
                # Load Stokes data
                stokes_data = {
                    key: f['stokes'][key][:] for key in f['stokes'].keys()
                }
                
                # Load MHD data
                mhd_data = {
                    key: f['mhd'][key][:] for key in f['mhd'].keys()
                }
                
                # Load physics approximations
                approx_data = {
                    key: f['approximations'][key][:] for key in f['approximations'].keys()
                }
            
            # Create dataset
            dataset = MuramStepDataset(
                stokes_data=stokes_data,
                mhd_data=mhd_data,
                stokes_normalizer=stokes_normalizer,
                mhd_normalizer=mhd_normalizer,
            )
            
            if verbose:
                print(f"  ✓ Loaded cached step {step}")
            
            return dataset, approx_data
        
        except Exception as e:
            print(f"  ✗ Failed to load cache for step {step}: {e}")
            raise
    
    def load_raw(
        self,
        step: int,
        verbose: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Load raw cached arrays (stokes, mhd, approx) without building dataset.
        Useful for analysis scripts.
        """
        cache_path = self._get_cache_path(step)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        with h5py.File(cache_path, 'r') as f:
            stokes_data = {k: f['stokes'][k][:] for k in f['stokes'].keys()}
            mhd_data = {k: f['mhd'][k][:] for k in f['mhd'].keys()}
            approx_data = {}
            if 'approximations' in f:
                approx_data = {k: f['approximations'][k][:] for k in f['approximations'].keys()}

        if verbose:
            print(f"  ✓ Loaded raw cached step {step}")
        return stokes_data, mhd_data, approx_data
    
    def clear(self, step: int | None = None, confirm: bool = True) -> None:
        """
        Clear cache files.
        
        Parameters
        ----------
        step : int, optional
            If provided, delete only this step. Otherwise, delete all.
        confirm : bool
            If True, ask for confirmation before deleting
        """
        if step is not None:
            cache_path = self._get_cache_path(step)
            if not cache_path.exists():
                print(f"Cache file not found: {cache_path}")
                return
            
            if confirm:
                response = input(f"Delete cache for step {step}? (y/n): ")
                if response.lower() != 'y':
                    return
            
            cache_path.unlink()
            if str(step) in self.metadata:
                del self.metadata[str(step)]
            self._save_metadata()
            print(f"  ✓ Deleted cache for step {step}")
        
        else:
            if confirm:
                response = input(f"Delete all cache files in {self.cache_dir}? (y/n): ")
                if response.lower() != 'y':
                    return
            
            # Delete all .h5 files
            for cache_file in self.cache_dir.glob("step_*.h5"):
                cache_file.unlink()
            
            # Clear metadata
            self.metadata = {}
            self._save_metadata()
            print(f"  ✓ Cleared all cache files")
    
    def get_cache_stats(self) -> dict[str, any]:
        """
        Get statistics about cached data.
        
        Returns
        -------
        dict
            Statistics including total size, number of steps, etc.
        """
        cache_files = list(self.cache_dir.glob("step_*.h5"))
        total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024**2)
        
        return {
            'total_steps_cached': len(cache_files),
            'total_size_mb': total_size_mb,
            'cache_dir': str(self.cache_dir),
            'steps': list(self.metadata.keys()) if self.metadata else [],
        }
    
    def print_cache_info(self):
        """Print formatted cache information."""
        stats = self.get_cache_stats()
        
        print("\n" + "=" * 80)
        print("CACHE STATISTICS".center(80))
        print("=" * 80)
        print(f"Cache directory:      {stats['cache_dir']}")
        print(f"Steps cached:         {stats['total_steps_cached']}")
        print(f"Total size:           {stats['total_size_mb']:.1f} MB")
        
        if stats['steps']:
            steps_sorted = sorted([int(s) for s in stats['steps']])
            print(f"Step range:           {steps_sorted[0]} - {steps_sorted[-1]}")
            print("\nDetailed cache info:")
            print(f"{'Step':<8} {'nx':>6} {'ny':>6} {'nwl':>6} {'ntau':>6} {'Size (MB)':>12} {'Config':<10}")
            print("-" * 80)
            for step_key in sorted(stats['steps']):
                meta = self.metadata[step_key]
                print(f"{step_key:<8} {meta['nx']:>6} {meta['ny']:>6} {meta['nwl']:>6} "
                      f"{meta['ntau']:>6} {meta['file_size_mb']:>12.1f} {meta['config_hash']:<10}")
        
        print("=" * 80 + "\n")
