#!/usr/bin/env python3
"""Merge per-chunk tau_500 NICOLE Stokes into the full-frame cube (issue #5).

Assembles the row-chunks written by generate_tau500_stokes.py for one step
into the full (nx, ny, N_WL, 4) Stokes cube and the full (nx, ny, nz)
tau_500 atmosphere, validating that the chunks tile the frame exactly (no
gaps, no overlaps) and contain no NaNs before writing the merged outputs.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.base_training import TrainingConfig

CHUNK_RE = re.compile(r"chunk_(\d+)_(\d+)_(\d+)_stokes\.npy$")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--chunk-dir", type=Path, required=True,
                   help="Directory holding chunk_{step}_{r0}_{r1}_stokes.npy / _atmos.npz")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write the merged stokes_{step}_nicole_tau500.npy / atmos")
    args = p.parse_args()

    train_cfg = TrainingConfig()
    nx, ny = train_cfg.nx, train_cfg.ny

    chunks = []
    for f in sorted(args.chunk_dir.glob(f"chunk_{args.step}_*_stokes.npy")):
        m = CHUNK_RE.search(f.name)
        if not m or int(m.group(1)) != args.step:
            continue
        r0, r1 = int(m.group(2)), int(m.group(3))
        chunks.append((r0, r1, f))
    if not chunks:
        sys.exit(f"No chunks found for step {args.step} in {args.chunk_dir}")
    chunks.sort()

    # Coverage check: chunks must tile [0, nx) exactly.
    covered = np.zeros(nx, dtype=int)
    for r0, r1, _ in chunks:
        covered[r0:r1] += 1
    if not np.all(covered == 1):
        gaps = np.where(covered == 0)[0]
        overlaps = np.where(covered > 1)[0]
        sys.exit(f"Chunks do not tile [0,{nx}) exactly. gaps={gaps.tolist()[:10]} "
                 f"overlaps={overlaps.tolist()[:10]}")

    n_wl = np.load(chunks[0][2], mmap_mode="r").shape[2]
    stokes_full = np.empty((nx, ny, n_wl, 4), dtype=np.float64)
    logtau = None
    T_full = Vz_full = Bz_full = Z_full = None

    for r0, r1, f in chunks:
        s = np.load(f)
        if s.shape != (r1 - r0, ny, n_wl, 4):
            sys.exit(f"Chunk {f.name} has shape {s.shape}, expected {(r1 - r0, ny, n_wl, 4)}")
        stokes_full[r0:r1] = s
        a = np.load(f.with_name(f.name.replace("_stokes.npy", "_atmos.npz")))
        if logtau is None:
            logtau = a["logtau"]
            nz = len(logtau)
            T_full = np.empty((nx, ny, nz), dtype=np.float64)
            Vz_full = np.empty((nx, ny, nz), dtype=np.float64)
            Bz_full = np.empty((nx, ny, nz), dtype=np.float64)
            Z_full = np.empty((nx, ny, nz), dtype=np.float64)
        T_full[r0:r1] = a["T"]; Vz_full[r0:r1] = a["Vz"]
        Bz_full[r0:r1] = a["Bz"]; Z_full[r0:r1] = a["z_km"]

    n_nan = int(np.isnan(stokes_full).sum())
    if n_nan:
        sys.exit(f"Merged Stokes contains {n_nan} NaNs -- refusing to write")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stokes_out = args.output_dir / f"stokes_{args.step}_nicole_tau500.npy"
    atmos_out = args.output_dir / f"atmos_{args.step}_tau500.npz"
    np.save(stokes_out, stokes_full)
    np.savez(atmos_out, logtau=logtau, T=T_full, Vz=Vz_full, Bz=Bz_full, z_km=Z_full,
             step=args.step)

    cont = np.mean([stokes_full[..., 0][..., :5].mean(), stokes_full[..., 0][..., -5:].mean()])
    print(f"Merged {len(chunks)} chunks for step {args.step}")
    print(f"  stokes -> {stokes_out}  shape {stokes_full.shape}")
    print(f"  atmos  -> {atmos_out}   (T/Vz/Bz/z_km on {len(logtau)}-level tau_500 grid)")
    print(f"  mean continuum I ~ {cont:.4f}   NaNs: {n_nan}")


if __name__ == "__main__":
    main()
