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


class BalancedTrainDataCache:
    """Cache manager for post-balancing train-ready tensors.

    This cache stores one file per simulation step containing the final balanced
    tensors used by training, so balancing logic is not re-run every epoch.
    """

    def __init__(self, cache_dir: str = ".muram_balanced_cache", compression: str = "lzf"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.manifest_path = self.cache_dir / "balanced_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, object]:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {
            "version": 1,
            "signature_hash": None,
            "signature": {},
            "steps": {},
            "total_steps": 0,
            "total_selected": 0,
            "total_bytes": 0,
        }

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    @staticmethod
    def make_signature_hash(signature: dict) -> str:
        payload = json.dumps(signature, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    def _step_path(self, step: int) -> Path:
        return self.cache_dir / f"balanced_step_{step:06d}.h5"

    def reset(self, signature: dict, signature_hash: str) -> None:
        for fp in self.cache_dir.glob("balanced_step_*.h5"):
            fp.unlink()
        self.manifest = {
            "version": 1,
            "signature_hash": signature_hash,
            "signature": signature,
            "steps": {},
            "total_steps": 0,
            "total_selected": 0,
            "total_bytes": 0,
        }
        self._save_manifest()

    def clear(self) -> None:
        """Remove all balanced cache files and reset manifest."""
        for fp in self.cache_dir.glob("balanced_step_*.h5"):
            fp.unlink()
        self.manifest = {
            "version": 1,
            "signature_hash": None,
            "signature": {},
            "steps": {},
            "total_steps": 0,
            "total_selected": 0,
            "total_bytes": 0,
        }
        self._save_manifest()

    def ensure_signature(self, signature: dict, signature_hash: str) -> bool:
        """Return True if the manifest matches this signature hash."""
        current_hash = self.manifest.get("signature_hash")
        if current_hash is None:
            self.manifest["signature_hash"] = signature_hash
            self.manifest["signature"] = signature
            self._save_manifest()
            return True
        return current_hash == signature_hash

    def has_step(self, step: int, signature_hash: str) -> bool:
        if self.manifest.get("signature_hash") != signature_hash:
            return False
        entry = self.manifest.get("steps", {}).get(str(step))
        if not isinstance(entry, dict):
            return False
        path = self._step_path(step)
        if not path.exists():
            return False
        return bool(entry.get("valid", False))

    def save_step(
        self,
        step: int,
        signature_hash: str,
        stokes_input: np.ndarray,
        mhd_targets: np.ndarray,
        spatial_indices: np.ndarray,
        approx_data: dict[str, np.ndarray],
        extra_metadata: dict[str, object] | None = None,
    ) -> Path:
        if self.manifest.get("signature_hash") != signature_hash:
            raise ValueError("Balanced cache signature mismatch. Reset cache before saving.")

        path = self._step_path(step)
        with h5py.File(path, "w") as f:
            ds = f.create_group("dataset")
            ds.create_dataset(
                "stokes_input",
                data=np.asarray(stokes_input, dtype=np.float32),
                compression=self.compression,
            )
            ds.create_dataset(
                "mhd_targets",
                data=np.asarray(mhd_targets, dtype=np.float32),
                compression=self.compression,
            )
            ds.create_dataset(
                "spatial_indices",
                data=np.asarray(spatial_indices, dtype=np.int64),
                compression=self.compression,
            )

            ap = f.create_group("approximations")
            for key in ("blos", "vlos", "temp"):
                if key not in approx_data:
                    raise KeyError(f"Missing approximation key '{key}' for balanced cache step {step}")
                ap.create_dataset(
                    key,
                    data=np.asarray(approx_data[key], dtype=np.float32),
                    compression=self.compression,
                )

            f.attrs["step"] = int(step)
            f.attrs["signature_hash"] = str(signature_hash)

        entry = {
            "file": path.name,
            "selected": int(stokes_input.shape[0]),
            "bytes": int(path.stat().st_size),
            "valid": True,
        }
        if extra_metadata:
            entry["metadata"] = extra_metadata

        steps = self.manifest.setdefault("steps", {})
        steps[str(step)] = entry

        self.manifest["total_steps"] = int(len(steps))
        self.manifest["total_selected"] = int(sum(int(v.get("selected", 0)) for v in steps.values()))
        self.manifest["total_bytes"] = int(sum(int(v.get("bytes", 0)) for v in steps.values()))
        self._save_manifest()
        return path

    def load_step(
        self,
        step: int,
        signature_hash: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        if not self.has_step(step, signature_hash):
            raise FileNotFoundError(f"Balanced cache miss for step {step}")

        path = self._step_path(step)
        with h5py.File(path, "r") as f:
            stokes_input = np.asarray(f["dataset"]["stokes_input"][:], dtype=np.float32)
            mhd_targets = np.asarray(f["dataset"]["mhd_targets"][:], dtype=np.float32)
            spatial_indices = np.asarray(f["dataset"]["spatial_indices"][:], dtype=np.int64)
            approx_data = {
                "blos": np.asarray(f["approximations"]["blos"][:], dtype=np.float32),
                "vlos": np.asarray(f["approximations"]["vlos"][:], dtype=np.float32),
                "temp": np.asarray(f["approximations"]["temp"][:], dtype=np.float32),
            }
        return stokes_input, mhd_targets, spatial_indices, approx_data

    def get_stats(self) -> dict[str, object]:
        return {
            "cache_dir": str(self.cache_dir),
            "signature_hash": self.manifest.get("signature_hash"),
            "total_steps": int(self.manifest.get("total_steps", 0)),
            "total_selected": int(self.manifest.get("total_selected", 0)),
            "total_bytes": int(self.manifest.get("total_bytes", 0)),
            "total_size_mb": float(self.manifest.get("total_bytes", 0)) / (1024**2),
            "steps": self.manifest.get("steps", {}),
        }

    def print_cache_info(self) -> None:
        stats = self.get_stats()
        print("\n" + "=" * 80)
        print("BALANCED TRAIN CACHE STATISTICS".center(80))
        print("=" * 80)
        print(f"Cache directory:      {stats['cache_dir']}")
        print(f"Signature hash:       {stats['signature_hash']}")
        print(f"Steps cached:         {stats['total_steps']}")
        print(f"Selected samples:     {stats['total_selected']}")
        print(f"Total size:           {stats['total_size_mb']:.1f} MB")
        print("=" * 80 + "\n")


class MuramDataCache:
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
    
    def __init__(self, cache_dir: str = ".muram_cache", compression: str = "gzip"):
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
        self._reserved_keys = {"__cache_info__"}

    
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
        return MuramDataCache._config_hash(config_dict)
    
    def _get_cache_path(self, step: int) -> Path:
        """Get cache file path for a step."""
        return self.cache_dir / f"step_{step:06d}.h5"
    
    def exists(self, step: int, config_hash: str | None = None, logtau_values: np.ndarray | list[float] | None = None) -> bool:
        """
        Check if processed data exists in cache.
        
        Parameters
        ----------
        step : int
            Simulation step number
        config_hash : str, optional
            Configuration hash to validate. If provided, checks that
            cached config matches.
        logtau_values : np.ndarray or list[float], optional
            Optical-depth grid used during MURaM -> tau remapping.
            
        Returns
        -------
        bool
            True if cache exists and is valid
        """
        self._ensure_logtau_compatible(logtau_values)

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

    def _ensure_logtau_compatible(self, logtau_values: np.ndarray | list[float] | None) -> None:
        expected = self._normalize_logtau_values(logtau_values)
        if expected is None:
            return

        cache_info = self.metadata.get("__cache_info__", {})
        cached_global = cache_info.get("logtau_values")
        if cached_global is not None and cached_global != expected:
            raise ValueError(
                "Cache logtau grid mismatch.\n"
                f"Expected: {expected}\n"
                f"Found (cache metadata): {cached_global}\n"
                f"Delete cache at '{self.cache_dir}' and rerun."
            )

        # Also validate per-step metadata if present
        for step_key in self._step_keys():
            cached_step = self.metadata.get(step_key, {}).get("logtau_values")
            if cached_step is not None and cached_step != expected:
                raise ValueError(
                    "Cache logtau grid mismatch at step "
                    f"{step_key}.\nExpected: {expected}\nFound: {cached_step}\n"
                    f"Delete cache at '{self.cache_dir}' and rerun."
                )

        # If missing global metadata, initialize it
        if cached_global is None:
            self.metadata["__cache_info__"] = {"logtau_values": expected}
            self._save_metadata()

    @staticmethod
    def _normalize_logtau_values(logtau_values: np.ndarray | list[float] | None) -> list[float] | None:
        if logtau_values is None:
            return None
        arr = np.asarray(logtau_values, dtype=np.float32)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("logtau_values must be a 1D array with at least 2 values.")
        return [float(x) for x in np.round(arr, 6).tolist()]

    def _step_keys(self) -> list[str]:
        return [k for k in self.metadata.keys() if k.isdigit()]

    def save(
        self,
        step: int,
        stokes_data: dict[str, np.ndarray],
        mhd_data: dict[str, np.ndarray],
        approx_data: dict[str, np.ndarray] | None = None,
        config_hash: str | None = None,
        logtau_values: np.ndarray | list[float] | None = None,
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
        logtau_values : np.ndarray or list[float], optional
            Optical-depth grid used during MURaM -> tau remapping.
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
                if logtau_values is not None:
                    f.attrs['logtau_values'] = np.asarray(logtau_values, dtype=np.float32)

            # Normalize/serialize logtau for json metadata
            logtau_list = self._normalize_logtau_values(logtau_values)

            # Update metadata
            self.metadata[str(step)] = {
                'config_hash': config_hash or 'unknown',
                'nx': stokes_data['I'].shape[0],
                'ny': stokes_data['I'].shape[1],
                'nwl': stokes_data['I'].shape[2],
                'ntau': mhd_data['T'].shape[2],
                'logtau_values': logtau_list,
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
        
        step_keys = self._step_keys()
        return {
            'total_steps_cached': len(cache_files),
            'total_size_mb': total_size_mb,
            'cache_dir': str(self.cache_dir),
            'steps': step_keys if step_keys else [],
            'cache_logtau_values': self.metadata.get("__cache_info__", {}).get("logtau_values"),
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
        if stats.get('cache_logtau_values') is not None:
            print(f"Cache logtau nodes:   {stats['cache_logtau_values']}")
        
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


class ModestDataCache:
    """Cache manager for processed MODEST snapshots (pre-cropping)."""

    def __init__(self, cache_dir: str = ".modest_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, dict]:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    @staticmethod
    def _config_hash(config_dict: dict) -> str:
        payload = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    @staticmethod
    def make_config_hash(config_dict: dict) -> str:
        return ModestDataCache._config_hash(config_dict)

    def _cache_path(self, config_hash: str) -> Path:
        return self.cache_dir / f"modest_{config_hash}.h5"

    def exists(self, config_hash: str) -> bool:
        return self._cache_path(config_hash).exists()

    def save(self, config_hash: str, payload: dict, config: dict | None = None, verbose: bool = True) -> Path:
        cache_path = self._cache_path(config_hash)
        self._save_hdf5(cache_path, payload)

        self.metadata[config_hash] = {
            'file': cache_path.name,
            'file_size_mb': cache_path.stat().st_size / (1024**2),
            'config': config or {},
            'backend': 'hdf5',
        }
        self._save_metadata()

        if verbose:
            print(f"  ✓ Cached MODEST snapshot ({cache_path.name}, {self.metadata[config_hash]['file_size_mb']:.1f} MB)")

        return cache_path

    def load(self, config_hash: str, verbose: bool = True) -> dict:
        cache_path = self._cache_path(config_hash)
        if not cache_path.exists():
            raise FileNotFoundError(f"MODEST cache not found: {cache_path}")

        payload = self._load_hdf5(cache_path)

        if verbose:
            print(f"  ✓ Loaded MODEST cache {cache_path.name}")

        return payload

    def clear(self, config_hash: str | None = None, confirm: bool = True) -> None:
        if config_hash is not None:
            cache_path = self._cache_path(config_hash)
            if not cache_path.exists():
                print(f"MODEST cache file not found: {cache_path}")
                return

            if confirm:
                response = input(f"Delete MODEST cache {cache_path.name}? (y/n): ")
                if response.lower() != 'y':
                    return

            cache_path.unlink()
            self.metadata.pop(config_hash, None)
            self._save_metadata()
            print(f"  ✓ Deleted MODEST cache {cache_path.name}")
            return

        if confirm:
            response = input(f"Delete all MODEST cache files in {self.cache_dir}? (y/n): ")
            if response.lower() != 'y':
                return

        for cache_file in self.cache_dir.glob("modest_*.h5"):
            cache_file.unlink()
        self.metadata = {}
        self._save_metadata()
        print("  ✓ Cleared all MODEST cache files")

    def get_cache_stats(self) -> dict[str, any]:
        cache_files = list(self.cache_dir.glob("modest_*.h5"))
        total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024**2)
        return {
            'cache_dir': str(self.cache_dir),
            'entries': len(cache_files),
            'total_size_mb': total_size_mb,
            'hashes': sorted(self.metadata.keys()),
            'backend': 'hdf5',
        }

    def print_cache_info(self):
        stats = self.get_cache_stats()
        print("\n" + "=" * 80)
        print("MODEST CACHE STATISTICS".center(80))
        print("=" * 80)
        print(f"Cache directory:      {stats['cache_dir']}")
        print(f"Backend:              {stats['backend']}")
        print(f"Entries cached:       {stats['entries']}")
        print(f"Total size:           {stats['total_size_mb']:.1f} MB")
        print("=" * 80 + "\n")

    @staticmethod
    def _array_like(value):
        if value is None:
            return None
        if hasattr(value, 'value'):
            return np.asarray(value.value)
        return np.asarray(value)

    def _save_hdf5(self, cache_path: Path, payload: dict) -> None:
        with h5py.File(cache_path, 'w') as f:
            f.attrs['backend'] = 'hdf5'

            def _save_stokes_dict(name: str, data: dict | None):
                if data is None:
                    return
                grp = f.create_group(name)
                for key, arr in data.items():
                    grp.create_dataset(key, data=np.asarray(arr), compression='lzf')

            arr = self._array_like(payload.get('continuum'))
            if arr is not None:
                f.create_dataset('continuum', data=arr, compression='lzf')

            _save_stokes_dict('obs_stokes', payload.get('obs_stokes'))
            _save_stokes_dict('upsampled_stokes', payload.get('upsampled_stokes'))
            _save_stokes_dict('deconvolved_stokes', payload.get('deconvolved_stokes'))
            _save_stokes_dict('smoothed_stokes', payload.get('smoothed_stokes'))
            _save_stokes_dict('inverted_profs', payload.get('inverted_profs'))

            arr = self._array_like(payload.get('inverted_atmos'))
            if arr is not None:
                f.create_dataset('inverted_atmos', data=arr, compression='lzf')

            spinor = payload.get('spinor_atm')
            if spinor is not None:
                spin_grp = f.create_group('spinor_atm')
                for param, tau_dict in spinor.items():
                    param_grp = spin_grp.create_group(param)
                    tau_values = [float(tau) for tau in sorted(tau_dict.keys())]
                    param_grp.attrs['tau_values'] = np.asarray(tau_values, dtype=np.float32)
                    for idx, tau in enumerate(tau_values):
                        param_grp.create_dataset(f"{idx:03d}", data=np.asarray(tau_dict[tau]), compression='lzf')

            arr = self._array_like(payload.get('wl'))
            if arr is not None:
                f.create_dataset('wl', data=arr)
            arr = self._array_like(payload.get('wl_inv'))
            if arr is not None:
                f.create_dataset('wl_inv', data=arr)

            arr = self._array_like(payload.get('circ_pol'))
            if arr is not None:
                f.create_dataset('circ_pol', data=arr, compression='lzf')

            arr = self._array_like(payload.get('mask'))
            if arr is not None:
                f.create_dataset('mask', data=np.asarray(arr, dtype=np.uint8), compression='lzf')

            sample_pixels = payload.get('sample_pixels')
            if sample_pixels is not None:
                f.attrs['sample_pixels_json'] = json.dumps(sample_pixels)

    def _load_hdf5(self, cache_path: Path) -> dict:
        payload: dict = {}
        with h5py.File(cache_path, 'r') as f:
            def _load_stokes_dict(name: str):
                if name not in f:
                    return None
                grp = f[name]
                return {key: grp[key][:] for key in grp.keys()}

            payload['continuum'] = f['continuum'][:] if 'continuum' in f else None
            payload['obs_stokes'] = _load_stokes_dict('obs_stokes')
            payload['upsampled_stokes'] = _load_stokes_dict('upsampled_stokes')
            payload['deconvolved_stokes'] = _load_stokes_dict('deconvolved_stokes')
            payload['smoothed_stokes'] = _load_stokes_dict('smoothed_stokes')
            payload['inverted_profs'] = _load_stokes_dict('inverted_profs')
            payload['inverted_atmos'] = f['inverted_atmos'][:] if 'inverted_atmos' in f else None

            spinor = None
            if 'spinor_atm' in f:
                spinor = {}
                spin_grp = f['spinor_atm']
                for param in spin_grp.keys():
                    param_grp = spin_grp[param]
                    tau_values = [float(t) for t in param_grp.attrs.get('tau_values', [])]
                    tau_dict = {}
                    for idx, tau in enumerate(tau_values):
                        tau_dict[tau] = param_grp[f"{idx:03d}"][:]
                    spinor[param] = tau_dict
            payload['spinor_atm'] = spinor

            payload['wl'] = f['wl'][:] if 'wl' in f else None
            payload['wl_inv'] = f['wl_inv'][:] if 'wl_inv' in f else None
            payload['circ_pol'] = f['circ_pol'][:] if 'circ_pol' in f else None
            payload['mask'] = (f['mask'][:] > 0) if 'mask' in f else None

            sample_pixels_json = f.attrs.get('sample_pixels_json', None)
            payload['sample_pixels'] = json.loads(sample_pixels_json) if sample_pixels_json is not None else None

        return payload


# Backward-compatible alias; prefer MuramDataCache in new code
DataCache = MuramDataCache
