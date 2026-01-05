#!/usr/bin/env python3
"""
Batch script to read and save Stokes data from raw NICOLE files as NumPy arrays.
Processes filenums from 60 to 200, using parallel processing for speed.
"""

import sys
import os
import time
import numpy as np
from joblib import Parallel, delayed

# Add utils to path
sys.path.append('/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/utils')
from model_prof_tools import read_prof


def read_profile(ix, iy, filename, filetype, nx, ny, nlam):
    """Read a single profile and return coordinates with data"""
    prof_data = np.array(read_prof(filename=filename,
                                    filetype=filetype,
                                    nx=nx, ny=ny, nlam=nlam,
                                    ix=ix, iy=iy)).reshape((nlam, 4))
    return ix, iy, prof_data


def process_filenum(filenum, filetype='nicole', nx=480, ny=480, nlam=300):
    """Process a single filenum: read all profiles and save as numpy array"""
    
    filename = f'/scratchsan/observatorio/juagudeloo/data/muram-simulation/{filenum:03d}000_0000_0000.prof'
    output_dir = '/scratchsan/observatorio/juagudeloo/data/muram-simulation/'
    output_path = os.path.join(output_dir, f'stokes_{filenum:03d}000.npy')
    
    # Skip if already exists
    if os.path.exists(output_path):
        file_size_gb = os.path.getsize(output_path) / 1e9
        print(f"[{filenum:03d}] ✓ Already exists: {output_path} ({file_size_gb:.2f} GB)")
        return filenum, True, 0.0
    
    # Check if source file exists
    if not os.path.exists(filename):
        print(f"[{filenum:03d}] ✗ Source file not found: {filename}")
        return filenum, False, 0.0
    
    start_time = time.time()
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Parallel processing with joblib
        stokes = np.zeros((nx, ny, nlam, 4))
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(read_profile)(ix, iy, filename, filetype, nx, ny, nlam)
            for ix in range(nx)
            for iy in range(ny)
        )
        
        # Fill the stokes array
        for ix, iy, prof_data in results:
            stokes[ix, iy, :, :] = prof_data
                
        # Save the array
        np.save(output_path, stokes)
        
        elapsed = time.time() - start_time
        file_size_gb = os.path.getsize(output_path) / 1e9
        
        print(f"[{filenum:03d}] ✓ Saved in {elapsed:.2f}s ({file_size_gb:.2f} GB): {output_path}")
        return filenum, True, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{filenum:03d}] ✗ Error: {str(e)}")
        return filenum, False, elapsed


def main():
    """Process all filenums from 60 to 200"""
    
    print("="*70)
    print("Batch Processing Stokes Data (filenums 60-200)")
    print("="*70)
    
    min_num = 60
    max_num = 201
    
    filenums = list(range(min_num, max_num))
    total_files = len(filenums)
    
    print(f"\nProcessing {total_files} files...\n")
    
    start_time = time.time()
    
    # Process filenums sequentially (each one uses parallel processing internally)
    results = []
    for i, filenum in enumerate(filenums, 1):
        filenum_result, success, elapsed = process_filenum(filenum)
        results.append((filenum_result, success, elapsed))
        
        # Print progress every 10 files
        if i % 10 == 0:
            print(f"Progress: {i}/{total_files} files processed")
    
    total_time = time.time() - start_time
    
    # Summary
    successful = sum(1 for _, success, _ in results if success)
    failed = total_files - successful
    total_elapsed = sum(elapsed for _, _, elapsed in results)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files processed: {total_files}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per file: {total_elapsed/successful:.2f} seconds" if successful > 0 else "")
    print("="*70)


if __name__ == '__main__':
    main()
