#!/usr/bin/env python3
"""Multi-step NICOLE-consistent Stokes regeneration (tau-scale-experiment branch).

Extends tau500_ground_truth_test.py's single-step (step 198) validation
across several MURaM steps (training-range + step 198): for each step,
stratified-sample pixels by |B_LOS| (model-independent), remap MURaM's own
true T/Vz/Bz onto a log(tau_500) grid (utils/tau500_opacity.py's H- opacity
approximation), run NICOLE forward synthesis directly on that ground-truth
atmosphere, and compare against MURaM's OWN pre-existing stokes_*.npy Stokes
profiles for the same pixels.

Motivation (from conversation with the user and their co-advisor): nobody
currently working on this project is sure how MURaM's stokes_*.npy files
were originally synthesized -- only the underlying MHD atmosphere (T, Vz,
Bz, P from the simulation itself) is trusted. This script produces a
NICOLE-synthesized alternative "truth" Stokes dataset, built with known,
inspectable assumptions (this repo's H- opacity approximation + NICOLE
itself), so it can be compared directly against both the mystery-code
stokes_*.npy files and (separately, already done for step 198) the MUISCA
model's own predictions -- all under one consistent, understood pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.base_training import TrainingConfig
from utils.muram_data import MhdData, StokesData
from utils.pixel_sampling import sample_pixels_by_abs_bz
from utils.synthesis import NicoleRunner, SynthesisConfig, _rmtree_retry
from utils.tau500_opacity import kappa_500

# Training range is steps 81-181 (step_size=5); step 198 is the already-
# validated out-of-training step from earlier in this session.
STEPS = [81, 106, 131, 156, 181, 198]
NEW_LOGTAU_GRID = np.arange(-3.0, 1.41, 0.1)  # stays within HSRA's validated monotonic branch
N_BINS = 8
N_PER_BIN = 10
SAMPLING_SEED = 0

OUTPUT_ROOT = Path("/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/tau500_nicole_regen")


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


def _metrics(I_true, I_synth, sigma_I=1e-3):
    wing_true = np.mean([I_true[:5].mean(), I_true[-5:].mean()])
    wing_synth = np.mean([I_synth[:5].mean(), I_synth[-5:].mean()])
    I_true_n = I_true / wing_true
    I_synth_n = I_synth / wing_synth
    chi2 = np.sum(((I_true_n - I_synth_n) / sigma_I) ** 2)
    return wing_synth / wing_true, I_true_n.min(), I_synth_n.min(), chi2


def run_one_step(step: int, train_cfg: TrainingConfig, data_path: Path) -> dict:
    print(f"\n{'='*70}\nStep {step}\n{'='*70}")

    print(f"Loading MURaM step {step} ground truth...")
    mhd = MhdData(
        data_path=data_path / "muram-simulation",
        nx=train_cfg.nx, ny=train_cfg.ny, nz=train_cfg.nz,
    )
    mhd.load_step(step=step, z_max=train_cfg.z_max)

    print("Sampling pixels by |B_LOS| (model-independent)...")
    sample_cfg = SynthesisConfig(
        source="muram", experiment_root="tau500_nicole_regen",
        model_type="ground_truth", muram_step=step,
    )
    selection = sample_pixels_by_abs_bz(sample_cfg, n_bins=N_BINS, n_per_bin=N_PER_BIN, seed=SAMPLING_SEED)
    test_pixels = selection.pixels
    print(f"  {len(test_pixels)} pixels selected")

    print("Computing log(tau_500) via H- opacity approximation...")
    logtau500 = compute_logtau500(mhd, dz_km=train_cfg.dz_km)

    T_full = mhd.data["T"].value
    Vz_full = mhd.data["Vz"].to(u.km / u.s).value
    Bz_full = mhd.data["Bz"].to(u.G).value

    print(f"Remapping T, Vz, Bz onto the new tau_500 grid for {len(test_pixels)} pixels...")
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

    print("Loading MURaM's own true (mystery-code) Stokes profiles...")
    stokes = StokesData(
        data_dir=data_path / "muram-simulation",
        step=step,
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

    cfg = SynthesisConfig(
        source="muram", experiment_root="tau500_nicole_regen",
        model_type="ground_truth", muram_step=step,
        output_root=OUTPUT_ROOT,
    )
    runner = NicoleRunner(cfg)

    print("Running NICOLE synthesis...")
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
        if (k + 1) % 20 == 0 or (k + 1) == n:
            print(f"  {k + 1}/{n} pixels synthesized")

    ratio, true_depth, synth_depth, chi2 = [], [], [], []
    I_true_all, I_synth_all, V_true_all, V_synth_all = [], [], [], []
    for (ix, iy) in test_pixels:
        I_true = stokes.data["I"][ix, iy, :]
        I_synth = synth_out[(ix, iy)][0]
        r, td, sd, c2 = _metrics(I_true, I_synth)
        ratio.append(r)
        true_depth.append(td)
        synth_depth.append(sd)
        chi2.append(c2)
        I_true_all.append(I_true)
        I_synth_all.append(I_synth)
        V_true_all.append(stokes.data["V"][ix, iy, :])
        V_synth_all.append(synth_out[(ix, iy)][3])

    ratio = np.array(ratio)
    depth_err = np.array(synth_depth) - np.array(true_depth)
    chi2 = np.array(chi2)

    print(f"\n--- Step {step}: NICOLE(tau_500 ground truth) vs MURaM's own stokes_*.npy, n={n} ---")
    print(f"  continuum ratio: mean={ratio.mean():.3f} median={np.median(ratio):.3f} std={ratio.std():.3f}")
    print(f"  line-core-depth error (synth-true): mean={depth_err.mean():+.4f} mean|err|={np.abs(depth_err).mean():.4f}")
    print(f"  fraction synth TOO SHALLOW (err>0): {(depth_err > 0).mean()*100:.1f}%")
    print(f"  chi2_I(locally-normalized): mean={chi2.mean():.3e} median={np.median(chi2):.3e}")

    out_dir = OUTPUT_ROOT / f"step-{step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.json", "w") as f:
        json.dump({
            "step": step,
            "pixels": test_pixels,
            "ratio": ratio.tolist(),
            "depth_err": depth_err.tolist(),
            "chi2": chi2.tolist(),
        }, f, indent=1)

    import h5py
    with h5py.File(out_dir / "nicole_truth_stokes.h5", "w") as f:
        f.create_dataset("pixels", data=np.asarray(test_pixels, dtype=np.int64))
        f.create_dataset("I_nicole_truth", data=np.asarray(I_synth_all))
        f.create_dataset("V_nicole_truth", data=np.asarray(V_synth_all))
        f.create_dataset("I_muram_own", data=np.asarray(I_true_all))
        f.create_dataset("V_muram_own", data=np.asarray(V_true_all))
        f.create_dataset("logtau_500", data=NEW_LOGTAU_GRID)
        f.attrs["muram_step"] = step

    return {
        "step": step,
        "n_pixels": n,
        "ratio_mean": float(ratio.mean()),
        "ratio_median": float(np.median(ratio)),
        "depth_err_mean": float(depth_err.mean()),
        "depth_err_abs_mean": float(np.abs(depth_err).mean()),
        "frac_too_shallow": float((depth_err > 0).mean()),
        "chi2_median": float(np.median(chi2)),
    }


def main():
    train_cfg = TrainingConfig()
    data_path = Path(train_cfg.data_path)

    summaries = []
    for step in STEPS:
        summaries.append(run_one_step(step, train_cfg, data_path))

    print(f"\n{'='*70}\nSUMMARY ACROSS ALL {len(STEPS)} STEPS\n{'='*70}")
    print(f"{'step':>6} {'n_px':>6} {'ratio_mean':>11} {'depth_err':>10} {'%shallow':>9} {'chi2_med':>11}")
    for s in summaries:
        print(f"{s['step']:6d} {s['n_pixels']:6d} {s['ratio_mean']:11.3f} "
              f"{s['depth_err_mean']:+10.4f} {s['frac_too_shallow']*100:8.1f}% {s['chi2_median']:11.3e}")

    out_path = OUTPUT_ROOT / "all_steps_summary.json"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=1)
    print(f"\nSummary saved -> {out_path}")


if __name__ == "__main__":
    main()
