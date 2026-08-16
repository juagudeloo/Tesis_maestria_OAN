#!/usr/bin/env python3
"""Generate MURaM Stokes from zero with NICOLE on the tau_500 scale (issue #5), one row-chunk.

For a row range [--row-start, --row-end) of a MURaM step, this:
  1. Loads the step's MHD atmosphere and computes log(tau_500) via the H-
     opacity approximation (utils.tau500_opacity, reused through
     scripts.synthesis.tau500_multi_step_regen.compute_logtau500).
  2. Remaps the TRUE T / V_LOS / B_LOS (and geometric height z) onto the fixed
     tau_500 grid -- the same grid the validated 6-step experiment used.
  3. Synthesizes the whole chunk in ONE NICOLE invocation via
     NicoleRunner.run_cube (native binary cube; NICOLE's Fortran loops over
     pixels internally -- no per-pixel Python/subprocess overhead). The cube
     path was validated equivalent to the per-pixel path to machine precision.
  4. Saves the chunk's Stokes (fine 300-wavelength grid matching the raw
     stokes_*.npy, so the existing LSF + resample preprocessing applies
     unchanged) plus the tau_500-remapped atmosphere (for coherent retraining
     labels later).

Output normalization is NICOLE's default HSRA-disk-center (Continuum
reference=1) -- continuum near 1.0. NOTE for the downstream training loader:
this data is ALREADY continuum-normalized, so the fixed_ic step must be
skipped for it (see docs/stokes_synthesis_discrepancy_investigation.md,
conclusion #5) -- that is a separate (retraining) issue, not this script.

Chunks are merged into the full frame by merge_tau500_stokes.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.base_training import TrainingConfig
from scripts.synthesis.tau500_multi_step_regen import NEW_LOGTAU_GRID, compute_logtau500
from utils.muram_data import MhdData
from utils.synthesis import NicoleRunner, SynthesisConfig, _rmtree_retry

# Fine wavelength grid matching the raw stokes_{step}.npy (6300.5 + 10 mA * k,
# k=0..299) so the existing continuum-norm -> LSF -> resample preprocessing
# applies to the generated data exactly as it does to the original.
WL_FIRST = 6300.5
WL_STEP_MA = 10.0
N_WL = 300


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", type=int, required=True, help="MURaM simulation step")
    p.add_argument("--row-start", type=int, required=True, help="First ix row (inclusive)")
    p.add_argument("--row-end", type=int, required=True, help="Last ix row (exclusive)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for the chunk .npy / .npz outputs")
    p.add_argument("--nicole-root", type=Path,
                   default=Path("/scratchsan/observatorio/juagudeloo/NICOLE_v16.06"))
    p.add_argument("--nicole-assets", type=Path,
                   default=Path("/scratchsan/observatorio/juagudeloo/MUISCA/data/nicole_assets"))
    p.add_argument("--keep-workdir", action="store_true",
                   help="Keep the NICOLE cube working directory (debug)")
    args = p.parse_args()

    t0 = time.time()
    train_cfg = TrainingConfig()
    data_path = Path(train_cfg.data_path)
    nx, ny = train_cfg.nx, train_cfg.ny
    r0, r1 = args.row_start, args.row_end
    if not (0 <= r0 < r1 <= nx):
        sys.exit(f"Invalid row range [{r0},{r1}) for nx={nx}")

    print(f"[step {args.step}] rows [{r0},{r1}) of {nx}")
    mhd = MhdData(data_path=data_path / "muram-simulation", nx=nx, ny=ny, nz=train_cfg.nz)
    mhd.load_step(step=args.step, z_max=train_cfg.z_max)

    print("Computing log(tau_500)...")
    logtau500 = compute_logtau500(mhd, dz_km=train_cfg.dz_km)  # (nx, ny, nz_geom)

    G = NEW_LOGTAU_GRID
    nz = len(G)
    T_full = mhd.data["T"].value
    Vz_full = mhd.data["Vz"].to(u.km / u.s).value
    Bz_full = mhd.data["Bz"].to(u.G).value
    nz_geom = T_full.shape[2]
    z_km_1d = np.arange(nz_geom, dtype=np.float64) * train_cfg.dz_km  # geometric height per z-index

    nrows = r1 - r0
    T_c = np.empty((nrows, ny, nz), dtype=np.float64)
    Vz_c = np.empty((nrows, ny, nz), dtype=np.float64)
    Bz_c = np.empty((nrows, ny, nz), dtype=np.float64)
    Z_c = np.empty((nrows, ny, nz), dtype=np.float64)

    print(f"Remapping {nrows * ny} pixels onto the tau_500 grid...")
    for i, ix in enumerate(range(r0, r1)):
        for iy in range(ny):
            lt = logtau500[ix, iy, :]
            T_c[i, iy] = interp1d(lt, T_full[ix, iy], bounds_error=False, fill_value="extrapolate")(G)
            Vz_c[i, iy] = interp1d(lt, Vz_full[ix, iy], bounds_error=False, fill_value="extrapolate")(G)
            Bz_c[i, iy] = interp1d(lt, Bz_full[ix, iy], bounds_error=False, fill_value="extrapolate")(G)
            Z_c[i, iy] = interp1d(lt, z_km_1d, bounds_error=False, fill_value="extrapolate")(G)

    cfg = SynthesisConfig(
        source="muram", experiment_root="tau500_generation", model_type="gt",
        muram_step=args.step,
        output_root=args.output_dir,
        nicole_root=args.nicole_root, nicole_assets=args.nicole_assets,
    )
    runner = NicoleRunner(cfg)

    workdir = args.output_dir / f"_cube_wd_{args.step}_{r0:04d}_{r1:04d}"
    print(f"Synthesizing chunk with NICOLE (one cube invocation)...")
    iquv = runner.run_cube(workdir, G, T_c, Vz_c, Bz_c, N_WL,
                           wl_first=WL_FIRST, wl_step_mA=WL_STEP_MA)  # (nrows, ny, 4, N_WL)
    if not args.keep_workdir:
        _rmtree_retry(workdir)

    # Store Stokes as (nrows, ny, N_WL, 4) [I,Q,U,V last] to match stokes_*.npy layout.
    stokes_chunk = np.transpose(iquv, (0, 1, 3, 2)).astype(np.float64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"chunk_{args.step}_{r0:04d}_{r1:04d}"
    stokes_path = args.output_dir / f"{stem}_stokes.npy"
    atmos_path = args.output_dir / f"{stem}_atmos.npz"
    np.save(stokes_path, stokes_chunk)
    np.savez(
        atmos_path,
        logtau=G, T=T_c, Vz=Vz_c, Bz=Bz_c, z_km=Z_c,
        row_start=r0, row_end=r1, step=args.step,
    )

    print(f"  stokes -> {stokes_path}  shape {stokes_chunk.shape}")
    print(f"  atmos  -> {atmos_path}")
    cont = np.mean([stokes_chunk[..., 0][..., :5].mean(), stokes_chunk[..., 0][..., -5:].mean()])
    print(f"  mean continuum I ~ {cont:.4f} (expect ~1.0, Continuum reference=1)")
    print(f"  NaNs in Stokes: {int(np.isnan(stokes_chunk).sum())}")
    print(f"  elapsed: {time.time() - t0:.1f} s for {nrows * ny} pixels "
          f"({(time.time() - t0) / (nrows * ny) * 1000:.1f} ms/pixel)")


if __name__ == "__main__":
    main()
