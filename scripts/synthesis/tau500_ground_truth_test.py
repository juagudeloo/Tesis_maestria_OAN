#!/usr/bin/env python3
"""Ground-truth-only tau_500 validation experiment (tau-scale-experiment branch).

Bypasses the MUISCA CNN entirely: remaps MURaM's OWN true T/Vz/Bz onto a
log(tau_500) grid computed via utils/tau500_opacity.py's H- opacity
approximation (instead of the pipeline's usual Rosseland-mean kappa.0.dat
grid), writes NICOLE .model files directly from that ground truth, runs
NICOLE, and compares the synthesized Stokes I/V against MURaM's OWN true
synthesized Stokes profiles for the same pixels.

This isolates the tau-scale hypothesis from model-prediction error: if the
Rosseland-vs-tau_500 mismatch explains the line-depth discrepancy seen in
the full MUISCA->NICOLE bridge, feeding NICOLE a genuinely tau_500-scaled
ground-truth atmosphere should reproduce MURaM's true Stokes profiles much
more closely than the existing Rosseland-tau pipeline does (see the
step-198 pixel_comparison overlays for that baseline).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.base_training import TrainingConfig
from utils.muram_data import MhdData, StokesData
from utils.synthesis import NicoleRunner, SynthesisConfig
from utils.tau500_opacity import kappa_500

MURAM_STEP = 198
NEW_LOGTAU_GRID = np.arange(-3.0, 1.41, 0.1)  # stays within HSRA's validated monotonic branch

# Reuse the existing model-independent |B_LOS|-stratified pixel sample (10
# bins x 15/bin) already computed for the step-198 baseline run, so the
# old-vs-new comparison below is a direct, paired, apples-to-apples diff on
# the exact same pixels rather than a fresh (and non-comparable) sample.
PIXEL_SELECTION_JSON = Path(
    "/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/"
    "experiment_81_to_181-step_size_5-normal/muram/step-198/wfa_only/"
    "pixel_selection/selected_pixels.json"
)
OLD_PREDICTIONS_H5 = Path(
    "/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/"
    "experiment_81_to_181-step_size_5-normal/muram/step-198/wfa_only/predictions.h5"
)
OLD_SYNTHESES_H5 = Path(
    "/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/"
    "experiment_81_to_181-step_size_5-normal/muram/step-198/wfa_only/syntheses.h5"
)


def compute_logtau500(mhd: MhdData, dz_km: float) -> np.ndarray:
    """log10(tau_500), integrated the same way MhdData.compute_optical_depth
    integrates Rosseland tau, but using tau500_opacity.kappa_500(T, P, rho)
    in place of the Rosseland kappa.0.dat interpolator."""
    T = mhd.data["T"].value
    P = mhd.data["P"].value
    rho = mhd.data["rho"].to(u.g / u.cm**3).value

    kappa_rho = kappa_500(T, P, rho) * rho  # cm^-1

    dz_cm = (dz_km * u.km).to(u.cm).value
    kappa_rev = kappa_rho[:, :, ::-1]
    tau_rev = cumulative_trapezoid(kappa_rev, dx=dz_cm, axis=2, initial=0.0)
    tau = tau_rev[:, :, ::-1]
    tau = np.clip(tau, a_min=1e-30, a_max=None)
    return np.log10(tau)


def remap_to_logtau500(logtau500: np.ndarray, quantity: np.ndarray, new_grid: np.ndarray,
                        nx: int, ny: int) -> np.ndarray:
    """Per-pixel interpolation from the computed logtau500(z) onto new_grid,
    mirroring MhdData.remap_to_optical_depth's own per-pixel interp1d loop."""
    out = np.zeros((nx, ny, len(new_grid)))
    for ix in range(nx):
        for iy in range(ny):
            mapper = interp1d(
                logtau500[ix, iy, :], quantity[ix, iy, :],
                kind="linear", bounds_error=False, fill_value="extrapolate",
            )
            out[ix, iy, :] = mapper(new_grid)
    return out


def main():
    train_cfg = TrainingConfig()
    data_path = Path(train_cfg.data_path)

    with open(PIXEL_SELECTION_JSON) as f:
        selection = json.load(f)
    test_pixels = [(int(p["ix"]), int(p["iy"])) for p in selection["pixels"]]
    print(f"Loaded {len(test_pixels)} pixels from {PIXEL_SELECTION_JSON.name} "
          f"(model-independent |B_LOS|-stratified sample, reused from the step-198 baseline)")

    print(f"\nLoading MURaM step {MURAM_STEP} ground truth...")
    mhd = MhdData(
        data_path=data_path / "muram-simulation",
        nx=train_cfg.nx, ny=train_cfg.ny, nz=train_cfg.nz,
    )
    mhd.load_step(step=MURAM_STEP, z_max=train_cfg.z_max)

    print("Computing log(tau_500) via H- opacity approximation...")
    logtau500 = compute_logtau500(mhd, dz_km=train_cfg.dz_km)
    print(f"  logtau_500 range: [{logtau500.min():.2f}, {logtau500.max():.2f}]")

    nx, ny = train_cfg.nx, train_cfg.ny
    print(f"Remapping T, Vz, Bz onto the new tau_500 grid for {len(test_pixels)} pixels...")
    T_full = mhd.data["T"].value
    Vz_full = mhd.data["Vz"].to(u.km / u.s).value
    Bz_full = mhd.data["Bz"].to(u.G).value

    results = {}
    for (ix, iy) in test_pixels:
        mapper_T = interp1d(logtau500[ix, iy, :], T_full[ix, iy, :], kind="linear",
                             bounds_error=False, fill_value="extrapolate")
        mapper_Vz = interp1d(logtau500[ix, iy, :], Vz_full[ix, iy, :], kind="linear",
                              bounds_error=False, fill_value="extrapolate")
        mapper_Bz = interp1d(logtau500[ix, iy, :], Bz_full[ix, iy, :], kind="linear",
                              bounds_error=False, fill_value="extrapolate")
        results[(ix, iy)] = {
            "T": mapper_T(NEW_LOGTAU_GRID),
            "Vz": mapper_Vz(NEW_LOGTAU_GRID),
            "Bz": mapper_Bz(NEW_LOGTAU_GRID),
        }

    print("Loading MURaM's own true Stokes profiles for the same pixels...")
    stokes = StokesData(
        data_dir=data_path / "muram-simulation",
        step=MURAM_STEP,
        wavelength_range=(6300.5, 6303.5),
        wavelength_step=0.01,
    )
    stokes.load_stokes()
    stokes_cont_indices = train_cfg.stokes_cont_indices or [0, 1, 2, 3]
    fixed_ic = float(train_cfg.stokes_fixed_ic) if train_cfg.stokes_ic_mode == "fixed_global" else None
    stokes.continuum_normalization(cont_indices=stokes_cont_indices, fixed_ic=fixed_ic)
    if train_cfg.stokes_mult_factor != 1.0:
        stokes.data["I"] = stokes.data["I"] * train_cfg.stokes_mult_factor
        stokes.data["V"] = stokes.data["V"] * train_cfg.stokes_mult_factor
    stokes.load_hinode_lsf(data_path / train_cfg.lsf_path)
    stokes.apply_spectral_convolution()
    stokes.resample_to_hinode()

    print("Setting up NicoleRunner (ground-truth logtau_500, plain hydrostatic seed)...")
    cfg = SynthesisConfig(
        source="muram",
        experiment_root="tau500_ground_truth_experiment",
        model_type="ground_truth",
        muram_step=MURAM_STEP,
        output_root=Path("/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis"),
    )
    runner = NicoleRunner(cfg)

    from utils.synthesis import _rmtree_retry

    synth_out = {}
    n = len(test_pixels)
    for k, (ix, iy) in enumerate(test_pixels):
        r = results[(ix, iy)]
        workdir = runner.prepare_workdir(
            ix=ix, iy=iy,
            logtau=NEW_LOGTAU_GRID,
            T=r["T"], v_los_kms=r["Vz"], b_long_G=r["Bz"],
        )
        iquv = runner.run_pixel(workdir)
        synth_out[(ix, iy)] = iquv
        _rmtree_retry(workdir)
        if (k + 1) % 10 == 0 or (k + 1) == n:
            print(f"  {k + 1}/{n} pixels synthesized")

    sigma_I = 1e-3

    def _metrics(I_true, I_synth):
        wing_true = np.mean([I_true[:5].mean(), I_true[-5:].mean()])
        wing_synth = np.mean([I_synth[:5].mean(), I_synth[-5:].mean()])
        I_true_n = I_true / wing_true
        I_synth_n = I_synth / wing_synth
        chi2 = np.sum(((I_true_n - I_synth_n) / sigma_I) ** 2)
        return wing_synth / wing_true, I_true_n.min(), I_synth_n.min(), chi2

    new_ratio, new_true_depth, new_synth_depth, new_chi2 = [], [], [], []
    for (ix, iy) in test_pixels:
        r, td, sd, c2 = _metrics(stokes.data["I"][ix, iy, :], synth_out[(ix, iy)][0])
        new_ratio.append(r)
        new_true_depth.append(td)
        new_synth_depth.append(sd)
        new_chi2.append(c2)
    new_ratio = np.array(new_ratio)
    new_depth_err = np.array(new_synth_depth) - np.array(new_true_depth)
    new_chi2 = np.array(new_chi2)

    print(f"\n=== NEW (tau_500 ground truth): n={n} pixels ===")
    print(f"  continuum ratio: mean={new_ratio.mean():.3f} median={np.median(new_ratio):.3f} std={new_ratio.std():.3f}")
    print(f"  line-core-depth error (synth-true): mean={new_depth_err.mean():+.4f} "
          f"median={np.median(new_depth_err):+.4f} mean|err|={np.abs(new_depth_err).mean():.4f}")
    print(f"  fraction synth TOO SHALLOW (err>0): {(new_depth_err > 0).mean()*100:.1f}%")
    print(f"  chi2_I(locally-normalized): mean={new_chi2.mean():.3e} median={np.median(new_chi2):.3e}")

    # Paired old (Rosseland tau, MUISCA-model-predicted atmosphere) baseline,
    # already computed and on disk for these exact same pixels.
    if OLD_PREDICTIONS_H5.exists() and OLD_SYNTHESES_H5.exists():
        with h5py.File(OLD_PREDICTIONS_H5) as f:
            old_pred_pixels = f["pixels"][...]
            old_stokes_obs = f["stokes_obs"][...]
        with h5py.File(OLD_SYNTHESES_H5) as f:
            old_synth_pixels = f["pixels"][...]
            old_stokes_synth = f["stokes_synth"][...]
        idx_obs = {tuple(p): i for i, p in enumerate(old_pred_pixels.tolist())}
        idx_synth = {tuple(p): i for i, p in enumerate(old_synth_pixels.tolist())}

        old_ratio, old_true_depth, old_synth_depth, old_chi2 = [], [], [], []
        for (ix, iy) in test_pixels:
            key = (ix, iy)
            if key not in idx_obs or key not in idx_synth:
                continue
            I_obs = old_stokes_obs[idx_obs[key], 0]
            I_synth = old_stokes_synth[idx_synth[key], 0]
            r, td, sd, c2 = _metrics(I_obs, I_synth)
            old_ratio.append(r)
            old_true_depth.append(td)
            old_synth_depth.append(sd)
            old_chi2.append(c2)
        old_ratio = np.array(old_ratio)
        old_depth_err = np.array(old_synth_depth) - np.array(old_true_depth)
        old_chi2 = np.array(old_chi2)

        print(f"\n=== OLD (Rosseland tau, MUISCA-predicted atmosphere): n={len(old_ratio)} pixels ===")
        print(f"  continuum ratio: mean={old_ratio.mean():.3f} median={np.median(old_ratio):.3f} std={old_ratio.std():.3f}")
        print(f"  line-core-depth error (synth-true): mean={old_depth_err.mean():+.4f} "
              f"median={np.median(old_depth_err):+.4f} mean|err|={np.abs(old_depth_err).mean():.4f}")
        print(f"  fraction synth TOO SHALLOW (err>0): {(old_depth_err > 0).mean()*100:.1f}%")
        print(f"  chi2_I(locally-normalized): mean={old_chi2.mean():.3e} median={np.median(old_chi2):.3e}")

        print(f"\n=== SUMMARY: OLD -> NEW ===")
        print(f"  mean continuum ratio:      {old_ratio.mean():.3f} -> {new_ratio.mean():.3f}")
        print(f"  mean|line-core-depth err|: {np.abs(old_depth_err).mean():.4f} -> {np.abs(new_depth_err).mean():.4f}")
        print(f"  %pixels too-shallow:       {(old_depth_err > 0).mean()*100:.1f}% -> {(new_depth_err > 0).mean()*100:.1f}%")
        print(f"  median chi2_I:             {np.median(old_chi2):.3e} -> {np.median(new_chi2):.3e}")
    else:
        print("\n(Old baseline predictions.h5/syntheses.h5 not found -- skipping paired comparison)")

    out_path = Path("/tmp/claude-3062/-scratchsan-observatorio-juagudeloo-MUISCA/"
                     "63f867f4-06e9-4ee8-9216-bf90db88858e/scratchpad/tau500_gt_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "pixels": test_pixels,
            "new_ratio": new_ratio.tolist(),
            "new_depth_err": new_depth_err.tolist(),
            "new_chi2": new_chi2.tolist(),
        }, f, indent=1)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
