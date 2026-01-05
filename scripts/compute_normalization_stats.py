import sys
sys.path.append("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/")

from pathlib import Path
import numpy as np
import astropy.units as u
from tqdm import tqdm

from utils.muram_data import MhdData, StokesData
from utils.normalizer import MhdNormalizer, StokesNormalizer

def compute_normalization_stats(
    min_step: int = 60,
    max_step: int = 223,
    save_interval: int = 20,
    resume_from: str = None
):
    """
    Compute normalization statistics for both MHD and Stokes data.
    
    Args:
        min_step: first simulation step to process
        max_step: last simulation step to process (inclusive)
        save_interval: save intermediate state every N steps
        resume_from: path to resume from saved state (optional)
    """
    # Configuration - using same paths as other scripts
    data_path = Path("/scratchsan/observatorio/juagudeloo/data/")
    mhd_data_dir = data_path / "muram-simulation"
    stokes_data_dir = data_path / "muram-simulation"
    kappa_path = data_path / "csv/kappa.0.dat"
    lsf_path = data_path / "hinode-MODEST/PSFs/hinode_sp.spline.psf"
    output_dir = data_path / "normalization_stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulation parameters
    nx, ny, nz = 480, 480, 256
    z_max = 250
    dz_km = 10.0
    new_logtau = np.arange(-2.0, 0.1, 0.1)  # 21 optical depth levels
    n_tau = len(new_logtau)
    
    # Initialize normalizers
    if resume_from:
        print(f"Resuming from saved state: {resume_from}")
        resume_path = Path(resume_from)
        mhd_normalizer = MhdNormalizer(n_tau=n_tau).load_state(
            resume_path / "mhd_state.json"
        )
        stokes_normalizer = StokesNormalizer().load_state(
            resume_path / "stokes_state.json"
        )
        # Extract starting step from directory name
        if "step_" in resume_path.name:
            start_step = int(resume_path.name.split("_")[1]) + 1
        else:
            start_step = min_step
        print(f"Resuming from step {start_step}")
    else:
        mhd_normalizer = MhdNormalizer(n_tau=n_tau)
        stokes_normalizer = StokesNormalizer()
        start_step = min_step
    
    available_steps = list(range(start_step, max_step + 1))
    
    print(f"\nComputing normalization statistics for both MHD and Stokes data")
    print(f"Processing steps {start_step} to {max_step}")
    print(f"MHD data directory: {mhd_data_dir}")
    print(f"Stokes data directory: {stokes_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Optical depth levels: {n_tau}")
    print(f"Grid dimensions: ({nx}, {ny}, {nz})")
    print(f"Trimming z to: {z_max}")
    print(f"Save interval: every {save_interval} steps\n")
    
    # Track progress
    successful_steps = 0
    failed_steps = []
    
    # Process each simulation step
    for step in tqdm(available_steps, desc="Processing steps"):
        try:
            # ============ Load MHD data ============
            print(f"\n[Step {step}] Loading MHD data...")
            mhd = MhdData(data_path=mhd_data_dir, nx=nx, ny=ny, nz=nz)
            mhd.load_step(step=step, z_max=z_max)
            
            # Compute optical depth
            print(f"[Step {step}] Computing optical depth...")
            mhd.load_opacity_table(kappa_path)
            mhd.compute_optical_depth(dz=dz_km * u.km)
            
            # Remap to optical depth coordinates
            print(f"[Step {step}] Remapping to optical depth...")
            mhd.remap_to_optical_depth(new_logtau, quantities=["T", "Vz", "Bz"])
            
            # Extract MHD data for normalization
            mhd_data = {
                'T': mhd.od_data['T'],
                'Vz': mhd.od_data['Vz'],
                'Bz': mhd.od_data['Bz']
            }
            
            # ============ Load Stokes data ============
            print(f"[Step {step}] Loading Stokes data...")
            stokes = StokesData(
                data_dir=stokes_data_dir,
                step=step,
                wavelength_range=(6300.5, 6303.5),
                wavelength_step=0.01
            )
            stokes.load_stokes()
            stokes.continuum_normalization(cont_indices=[0, 1, 2, 3])
            stokes.load_hinode_lsf(lsf_path)
            stokes.apply_spectral_convolution()
            stokes.resample_to_hinode()
            
            # Extract Stokes data for normalization
            stokes_data = {
                'I': stokes.data['I'],
                'V': stokes.data['V']
            }
            
            # ============ Update normalizers ============
            print(f"[Step {step}] Updating normalizers...")
            mhd_normalizer.update(mhd_data)
            stokes_normalizer.update(stokes_data)
            
            successful_steps += 1
            
            # Free memory
            del mhd, stokes
            
            # Save intermediate state periodically
            if successful_steps % save_interval == 0:
                state_dir = output_dir / "intermediate_states" / f"step_{step:06d}"
                state_dir.mkdir(parents=True, exist_ok=True)
                
                mhd_normalizer.save_state(state_dir / "mhd_state.json")
                stokes_normalizer.save_state(state_dir / "stokes_state.json")
                
                print(f"\n[Step {step}] Saved intermediate state ({successful_steps} steps processed)")
        
        except Exception as e:
            print(f"\n[ERROR] Error processing step {step}: {e}")
            import traceback
            traceback.print_exc()
            failed_steps.append(step)
            
            # Save state before continuing
            error_dir = output_dir / "error_state" / f"step_{step:06d}"
            error_dir.mkdir(parents=True, exist_ok=True)
            mhd_normalizer.save_state(error_dir / "mhd_state.json")
            stokes_normalizer.save_state(error_dir / "stokes_state.json")
            print(f"Saved error state to {error_dir}")
            continue
    
    print("\n" + "="*60)
    print("All steps processed. Finalizing statistics...")
    print("="*60)
    print(f"  Successfully processed: {successful_steps}/{len(available_steps)} steps")
    if failed_steps:
        print(f"  Failed steps: {failed_steps}")
    print()
    
    # Finalize both normalizers
    mhd_stats = mhd_normalizer.finalize()
    stokes_stats = stokes_normalizer.finalize()
    
    # Save final statistics
    mhd_normalizer.save(output_dir / "mhd_normalization.json")
    stokes_normalizer.save(output_dir / "stokes_normalization.json")
    
    print("\n" + "="*60)
    print("Normalization statistics computed successfully!")
    print("="*60)
    print(f"\nFinal files saved:")
    print(f"  MHD:    {output_dir / 'mhd_normalization.json'}")
    print(f"  Stokes: {output_dir / 'stokes_normalization.json'}")
    
    return mhd_normalizer, stokes_normalizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for MHD and Stokes data"
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=60,
        help="First simulation step to process (default: 60)"
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=200,
        help="Last simulation step to process (default: 223)"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=20,
        help="Save intermediate state every N steps (default: 20)"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to directory with saved state to resume from"
    )
    
    args = parser.parse_args()
    
    compute_normalization_stats(
        min_step=args.min_step,
        max_step=args.max_step,
        save_interval=args.save_interval,
        resume_from=args.resume_from
    )
