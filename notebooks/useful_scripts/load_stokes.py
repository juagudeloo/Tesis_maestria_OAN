#!/usr/bin/env python3
"""
Utility script to load pre-processed Stokes data from NumPy arrays.
Much faster than reading from raw NICOLE files.
"""

import os
import numpy as np
from pathlib import Path


def load_stokes(filenum):
    """Load a single stokes array by filenum"""
    stokes_dir = '/scratchsan/observatorio/juagudeloo/data/muram-simulation/stokes_arrays'
    filepath = os.path.join(stokes_dir, f'stokes_{filenum:03d}.npy')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Stokes array not found: {filepath}")
    
    return np.load(filepath)


def load_stokes_range(start_filenum, end_filenum, as_dict=False):
    """
    Load multiple stokes arrays at once.
    
    Parameters:
    -----------
    start_filenum : int
        Starting filenum (inclusive)
    end_filenum : int
        Ending filenum (inclusive)
    as_dict : bool
        If True, return dict with filenum as keys. If False, return list.
    
    Returns:
    --------
    dict or list of numpy arrays
    """
    
    stokes_dir = '/scratchsan/observatorio/juagudeloo/data/muram-simulation/'
    
    if as_dict:
        data = {}
        for filenum in range(start_filenum, end_filenum + 1):
            filepath = os.path.join(stokes_dir, f'stokes_{filenum:03d}000.npy')
            if os.path.exists(filepath):
                data[filenum] = np.load(filepath)
            else:
                print(f"Warning: {filepath} not found")
        return data
    else:
        data = []
        for filenum in range(start_filenum, end_filenum + 1):
            filepath = os.path.join(stokes_dir, f'stokes_{filenum:03d}.npy')
            if os.path.exists(filepath):
                data.append(np.load(filepath))
            else:
                print(f"Warning: {filepath} not found")
        return data


def list_available_stokes():
    """List all available stokes arrays"""
    stokes_dir = '/scratchsan/observatorio/juagudeloo/data/muram-simulation'
    
    if not os.path.exists(stokes_dir):
        print(f"Directory not found: {stokes_dir}")
        return []
    
    files = sorted(Path(stokes_dir).glob('stokes_*.npy'))
    filenums = [int(f.stem.split('_')[1]) for f in files]
    
    print(f"Found {len(filenums)} stokes arrays:")
    for filenum in filenums:
        filepath = os.path.join(stokes_dir, f'stokes_{filenum:03d}.npy')
        size_gb = os.path.getsize(filepath) / 1e9
        print(f"  stokes_{filenum:03d}.npy ({size_gb:.2f} GB)")
    
    return filenums


if __name__ == '__main__':
    # Example usage
    print("Available stokes arrays:")
    list_available_stokes()
    
    print("\nExample: Loading filenum 100")
    try:
        stokes = load_stokes(100)
        print(f"Shape: {stokes.shape}")
        print(f"Data type: {stokes.dtype}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
