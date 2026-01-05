#!/usr/bin/env python3
"""
Verify that saved numpy arrays match original NICOLE .prof files.
If they match, delete the original .prof files to save storage.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add utils to path
sys.path.append('/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/utils')
from model_prof_tools import read_prof


def verify_single_file(filenum, nx=480, ny=480, nlam=300, num_samples=10):
    """
    Verify that a saved numpy array matches the original .prof file.
    
    Parameters:
    -----------
    filenum : int
        File number to verify
    nx, ny, nlam : int
        Dimensions of the data
    num_samples : int
        Number of random positions to sample for verification
        
    Returns:
    --------
    bool : True if arrays match, False otherwise
    """
    
    prof_file = f'/scratchsan/observatorio/juagudeloo/data/muram-simulation/{filenum:03d}000_0000_0000.prof'
    npy_file = f'/scratchsan/observatorio/juagudeloo/data/muram-simulation/stokes_{filenum:03d}000.npy'
    
    # Check both files exist
    if not os.path.exists(prof_file):
        print(f"[{filenum:03d}] ✗ Original .prof file not found")
        return False
    
    if not os.path.exists(npy_file):
        print(f"[{filenum:03d}] ✗ Numpy .npy file not found")
        return False
    
    try:
        # Load the saved numpy array
        saved_array = np.load(npy_file)
        
        # Verify shape
        expected_shape = (nx, ny, nlam, 4)
        if saved_array.shape != expected_shape:
            print(f"[{filenum:03d}] ✗ Shape mismatch: expected {expected_shape}, got {saved_array.shape}")
            return False
        
        # Sample random positions to verify data matches
        np.random.seed(42)  # For reproducibility
        sample_positions = np.random.randint(0, min(nx, ny), size=(num_samples, 2))
        
        all_match = True
        for ix, iy in sample_positions:
            # Read from original file
            prof_data = np.array(read_prof(filename=prof_file,
                                          filetype='nicole',
                                          nx=nx, ny=ny, nlam=nlam,
                                          ix=ix, iy=iy)).reshape((nlam, 4))
            
            # Compare with saved array
            if not np.allclose(saved_array[ix, iy, :, :], prof_data, rtol=1e-9, atol=1e-12):
                print(f"[{filenum:03d}] ✗ Data mismatch at position ({ix}, {iy})")
                all_match = False
                break
        
        if all_match:
            print(f"[{filenum:03d}] ✓ Verified: All {num_samples} samples match")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"[{filenum:03d}] ✗ Error during verification: {str(e)}")
        return False


def get_filenum_pairs():
    """Find all filenum pairs that have both .prof and .npy files"""
    data_dir = '/scratchsan/observatorio/juagudeloo/data/muram-simulation'
    
    # Get all .npy files
    npy_files = sorted(Path(data_dir).glob('stokes_*.npy'))
    filenums = [int(f.stem.split('_')[1][:3]) for f in npy_files]
    
    # Filter to only those that have corresponding .prof files
    pairs = []
    for filenum in filenums:
        prof_file = os.path.join(data_dir, f'{filenum:03d}000_0000_0000.prof')
        if os.path.exists(prof_file):
            pairs.append(filenum)
    
    return pairs


def main():
    """Main verification and cleanup routine"""
    
    print("="*70)
    print("Verifying Numpy Arrays Against Original NICOLE Files")
    print("="*70)
    
    # Find all files to verify
    filenums = get_filenum_pairs()
    
    if not filenums:
        print("\nNo file pairs found to verify!")
        return
    
    print(f"\nFound {len(filenums)} file pairs to verify")
    print(f"File numbers: {min(filenums)} to {max(filenums)}\n")
    
    # Ask for confirmation before proceeding
    print("This script will:")
    print("  1. Verify each numpy array matches the original .prof file")
    print("  2. If verified, DELETE the original .prof file to save storage")
    print(f"\nTotal .prof files to potentially delete: {len(filenums)}")
    
    response = input("\nDo you want to proceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    print("\n" + "-"*70)
    print("Starting verification...")
    print("-"*70 + "\n")
    
    verified = []
    failed = []
    deleted = []
    
    for i, filenum in enumerate(filenums, 1):
        if verify_single_file(filenum):
            verified.append(filenum)
            
            # If verification passed, delete the original .prof file
            prof_file = f'/scratchsan/observatorio/juagudeloo/data/muram-simulation/{filenum:03d}000_0000_0000.prof'
            prof_size_gb = os.path.getsize(prof_file) / 1e9
            
            try:
                os.remove(prof_file)
                deleted.append(filenum)
                print(f"[{filenum:03d}] ✓ Deleted original .prof file ({prof_size_gb:.2f} GB)")
            except Exception as e:
                print(f"[{filenum:03d}] ✗ Failed to delete: {str(e)}")
        else:
            failed.append(filenum)
        
        # Progress update
        if i % 10 == 0:
            print(f"\nProgress: {i}/{len(filenums)} files processed\n")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files processed: {len(filenums)}")
    print(f"Successfully verified: {len(verified)}")
    print(f"Verification failed: {len(failed)}")
    print(f"Original files deleted: {len(deleted)}")
    
    if deleted:
        # Calculate space saved
        avg_size_gb = 2.1  # Average size from ls output
        space_saved_gb = len(deleted) * avg_size_gb
        print(f"Approximate storage saved: {space_saved_gb:.1f} GB")
    
    if failed:
        print(f"\nFailed file numbers: {failed}")
    
    print("="*70)


if __name__ == '__main__':
    main()
