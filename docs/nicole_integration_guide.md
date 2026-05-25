# Integrating NICOLE Spectral Synthesis into the MUISCA Inversion Workflow

## What is NICOLE?

NICOLE (NLTE Optimal Inversion Code for Lines of the Earth-sun) is a Fortran-based radiative transfer code
developed by Héctor Socas-Navarro (IAC). It solves the polarized radiative transfer equation under both
LTE and NLTE conditions to either:

- **Synthesize** Stokes profiles (I, Q, U, V) from a given atmospheric model, or
- **Invert** observed Stokes profiles back into an atmospheric model using Levenberg–Marquardt optimization.

Reference: Socas-Navarro et al. 2015, A&A 577, A7.

In the MUISCA workflow, you would use NICOLE in **synthesis mode only** — as a post-hoc consistency
check on your CNN inversion output.

---

## The Proposed Workflow

```
Observed Stokes (I, V)
        │
        ▼
  MUISCA PINN-MSCNN
        │
        ▼
Predicted MHD parameters        ← T(τ), V_LOS(τ), B_LOS(τ) across 21 log(τ) levels
        │
        ▼
  Build NICOLE .model file       ← convert units, add required columns
        │
        ▼
  Run NICOLE in synthesis mode
        │
        ▼
Synthesized Stokes (I, Q, U, V)
        │
        ▼
Compare synthesized vs. observed ← residuals, χ², visual plots
```

The idea is: if the CNN inversion is physically self-consistent, running NICOLE forward on its predicted
atmosphere should reproduce the original observation to within noise.

---

## Understanding the NICOLE Atmospheric Model Format

NICOLE reads atmospheres in ASCII (`.model`) format. Each pixel is one file. The format is:

```
Format version: 1.0
  v_mac    stray_light
  ltau_500   T      El_p     v_mic    B_long   v_los    B_x    B_y
  ...one row per optical depth level...
```

### Column meanings

| Column    | Symbol       | Units            | Description                                   |
|-----------|--------------|------------------|-----------------------------------------------|
| `ltau_500`| log(τ₅₀₀)   | dimensionless    | Log base-10 of continuum optical depth at 500 nm |
| `T`       | T            | K                | Temperature                                   |
| `El_p`    | Pₑ           | dyn cm⁻²         | Electron pressure                             |
| `v_mic`   | ξ_mic        | cm s⁻¹           | Microturbulent velocity                       |
| `B_long`  | B_LOS        | Gauss            | Line-of-sight (longitudinal) magnetic field   |
| `v_los`   | V_LOS        | cm s⁻¹           | Line-of-sight velocity                        |
| `B_x`     | Bₓ           | Gauss            | Horizontal field component (x)                |
| `B_y`     | B_y          | Gauss            | Horizontal field component (y)                |

**Header line:** `v_mac  stray_light` where `v_mac` is macroturbulence in cm s⁻¹ (a single value for the
whole atmosphere), and `stray_light` is the stray-light fraction (0–1).

### Example model file (HSRA with 100 G field)

```
Format version: 1.0
  1e5    0.
  1.40  9560.0 5.98E+03 6.00E+04 100.00  0.00E+00  100.00   100.00
  1.30  9390.0 5.00E+03 6.00E+04 100.00  0.00E+00  100.00   100.00
  ...
 -2.00  4900.0 2.10E+00 6.00E+04 100.00  0.00E+00  100.00   100.00
```

---

## What MUISCA Predicts vs. What NICOLE Needs

Your CNN predicts three quantities across 21 log(τ) levels (log τ = −2.0 to 0.0 in steps of 0.1):

| MUISCA output | Physical unit | NICOLE column | Unit needed   | Conversion                       |
|---------------|---------------|---------------|---------------|----------------------------------|
| T(τ)          | K             | `T`           | K             | None (direct)                    |
| V_LOS(τ)      | km s⁻¹        | `v_los`       | cm s⁻¹        | × 1e5                            |
| B_LOS(τ)      | Gauss         | `B_long`      | Gauss         | None (direct)                    |

### Missing quantities you must supply externally

NICOLE requires columns that MUISCA does not predict. You have two options for each:

#### 1. Electron pressure `El_p`
This is the hardest missing piece. It depends on temperature and gas pressure via the equation of state.
**Option A (recommended):** Use `Impose hydrostatic equilibrium = Y` in `NICOLE.input` — NICOLE will
integrate the hydrostatic equilibrium equation itself and compute pressure from temperature alone.
You must also set `Input density = Pel` and provide a seed value at the top of the atmosphere (the
outermost row of the model). Use a standard photospheric value, e.g., Pₑ ~ 2–10 dyn cm⁻² at log τ = −2.

**Option B:** Interpolate from a reference atmosphere (e.g., HSRA) at each τ level.

#### 2. Microturbulence `v_mic`
MUISCA does not recover microturbulence. Use a constant value, e.g., `v_mic = 1.0 km/s = 1e5 cm/s`, which
is a standard photospheric assumption. This affects line broadening but is a second-order effect for your
main parameters.

#### 3. Transverse field `B_x`, `B_y`
MUISCA only predicts the LOS component. NICOLE needs the full field vector.
- For MODEST observations, polarization signal in Q and U is typically below noise, so set `B_x = B_y = 0`.
- For MURaM verification, you could supply the known transverse components from the simulation.

#### 4. Macroturbulence `v_mac` and stray light
Use `v_mac = 0` (or a typical value ~1–2 km/s converted to cm s⁻¹) and `stray_light = 0`.

---

## Step-by-Step Integration Plan

### Step 1 — Build NICOLE (if not already compiled)

```bash
cd /scratchsan/observatorio/juagudeloo/NICOLE/main
make
# Produces the executable: ../main/nicole
```

The Makefile uses gfortran with `-O3 -fdefault-real-8`. Confirm the build with:
```bash
./nicole --help    # or run with no args to see usage
```

### Step 2 — Create a synthesis directory per pixel (or batch)

For each spatial pixel you want to verify, create a working directory:
```
synthesis_run/
├── NICOLE.input      ← synthesis configuration
├── LINES             ← spectral line atomic data
├── pixel.model       ← atmosphere from MUISCA output (generated by your script)
└── pixel.pro         ← output profiles (created by NICOLE)
```

### Step 3 — Write the NICOLE configuration file

```ini
# NICOLE.input for synthesis verification
Command=../../main/nicole
Mode = Synthesis

Input model= pixel.model
Output profiles= pixel.pro
Output model= pixel_out.mod

Heliocentric angle= 1.0          # cos(μ) = 1 for disk center (adjust for MODEST)
Impose hydrostatic equilibrium= Y
Input density= Pel               # NICOLE integrates pressure from T

Formal solution method= 0        # Auto-select (Hermite or WPM)
Printout detail= 1

[Region 1]
  First wavelength= 6300.        # Angstroms — adjust to your wavelength grid
  Wavelength step= 20 mA         # Match your observed wavelength sampling
  Number of wavelengths= 112     # Must match your Stokes profile length

[Line 1]
  Line=FeI 6301.5

[Nodes]
Temperature=0
Velocity=0
Bz=0
Bx=0
By=0
Microturbulence=0
Macroturbulence=0
```

> **Note on wavelength grid:** The MODEST observations have 112 wavelength samples. You need to verify
> the exact first wavelength and step from your `ModestData` loader to match NICOLE's output to your
> observed grid.

### Step 4 — Python script to convert MUISCA output to NICOLE model

```python
import numpy as np

def write_nicole_model(filepath, logtau, T_K, V_LOS_kms, B_LOS_G,
                       v_mic_cms=1e5, v_mac_cms=1e5, stray=0.0,
                       B_x_G=0.0, B_y_G=0.0, El_p_seed=5.0):
    """
    Write a NICOLE ASCII model file from MUISCA predictions.

    Parameters
    ----------
    logtau      : 1D array, log10(tau_500) values, e.g. np.arange(-2.0, 0.1, 0.1)
    T_K         : 1D array, temperature in Kelvin at each tau level
    V_LOS_kms   : 1D array, LOS velocity in km/s (positive = redshift)
    B_LOS_G     : 1D array, LOS magnetic field in Gauss
    v_mic_cms   : microturbulence in cm/s (default 1 km/s)
    v_mac_cms   : macroturbulence in cm/s (default 1 km/s)
    stray       : stray light fraction 0-1
    B_x_G, B_y_G : transverse field components in Gauss
    El_p_seed   : electron pressure at the top (outermost) level in dyn/cm²
                  (only used as seed; NICOLE recomputes via hydrostatic equilibrium)
    """
    # MUISCA predicts from log(τ) = -2.0 (top, low opacity) to 0.0 (bottom, high opacity)
    # NICOLE .model files conventionally go from large tau (deep) to small tau (top)
    # i.e., descending order in logtau. Check your specific run.
    idx = np.argsort(logtau)[::-1]  # sort descending
    logtau_sorted = logtau[idx]
    T_sorted      = T_K[idx]
    v_sorted      = V_LOS_kms[idx] * 1e5  # km/s → cm/s
    B_sorted      = B_LOS_G[idx]

    n_levels = len(logtau_sorted)
    # Approximate electron pressure profile (seed for hydrostatic mode)
    # Use a simple exponential decay from the top
    El_p = El_p_seed * np.exp(logtau_sorted - logtau_sorted[0])

    with open(filepath, 'w') as f:
        f.write("Format version: 1.0\n")
        f.write(f"  {v_mac_cms:.2e}    {stray:.2f}\n")
        for i in range(n_levels):
            f.write(
                f"  {logtau_sorted[i]:.2f}  {T_sorted[i]:.1f}  "
                f"{El_p[i]:.2E}  {v_mic_cms:.2E}  "
                f"{B_sorted[i]:.2f}  {v_sorted[i]:.2E}  "
                f"{B_x_G:.2f}  {B_y_G:.2f}\n"
            )
```

### Step 5 — Run NICOLE

```python
import subprocess, os

def run_nicole_synthesis(run_dir):
    """Run NICOLE synthesis in the given directory."""
    result = subprocess.run(
        ['python', 'run_nicole.py'],
        cwd=run_dir,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"NICOLE failed:\n{result.stderr}")
    return result.stdout
```

Or directly from the shell:
```bash
cd synthesis_run/
python /scratchsan/observatorio/juagudeloo/NICOLE/test/syn1/run_nicole.py
```

### Step 6 — Read the output profiles

NICOLE writes the synthesized Stokes profiles in ASCII:
```
wavelength_angstrom   I   Q   U   V
...
```
All Stokes parameters are normalized to the continuum intensity.

```python
import numpy as np

def read_nicole_profile(filepath):
    """Read NICOLE ASCII output profile."""
    data = np.loadtxt(filepath)
    wl    = data[:, 0]   # wavelength in Angstroms
    I     = data[:, 1]   # Stokes I (continuum normalized)
    Q     = data[:, 2]   # Stokes Q
    U     = data[:, 3]   # Stokes U
    V     = data[:, 4]   # Stokes V
    return wl, I, Q, U, V
```

### Step 7 — Compare synthesized vs. observed profiles

```python
import matplotlib.pyplot as plt

def compare_profiles(wl_obs, I_obs, V_obs, wl_syn, I_syn, V_syn, pixel_label=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(wl_obs, I_obs, label='Observed', color='k')
    axes[0].plot(wl_syn, I_syn, label='NICOLE (MUISCA input)', color='r', ls='--')
    axes[0].set_title(f'Stokes I  {pixel_label}')
    axes[0].legend()

    axes[1].plot(wl_obs, V_obs, label='Observed', color='k')
    axes[1].plot(wl_syn, V_syn, label='NICOLE (MUISCA input)', color='r', ls='--')
    axes[1].set_title(f'Stokes V  {pixel_label}')
    axes[1].legend()

    plt.tight_layout()
    return fig
```

---

## Caveats and Known Limitations

### 1. Missing electron pressure is the biggest blocker
MUISCA predicts T, V_LOS, and B_LOS but not the full thermodynamic state (electron/gas pressure or
density). With `Impose hydrostatic equilibrium = Y`, NICOLE integrates pressure from temperature, which
is physically reasonable but imperfect — especially if the MHD model has strong velocity fields that
violate strict hydrostatic balance. MURaM simulations are convective, so there will be systematic
discrepancies at deep layers.

### 2. MUISCA's log(τ) grid may not match NICOLE's natural grid
Your model uses log τ ∈ [−2.0, 0.0] in 0.1 steps (21 levels). NICOLE is happy with this range, but
note that the HSRA model (NICOLE's reference atmosphere) extends to log τ = +1.4 deep into the atmosphere.
If you only provide 21 levels from −2 to 0, NICOLE will extrapolate beyond your deepest point when it
needs to integrate hydrostatic equilibrium. This is generally safe but adds uncertainty.
**Recommendation:** Pad the model by appending a few deeper levels using a reference atmosphere like HSRA
for log τ = 0.1 to 0.5, using your CNN prediction at log τ = 0 as the boundary condition.

### 3. Transverse field components (B_x, B_y) are unknown
MUISCA only inverts Stokes I and V. This means you cannot recover Q and U, and therefore cannot constrain
the transverse field. Setting B_x = B_y = 0 is physically unrealistic for non-vertical fields, and NICOLE
will synthesize Q = U = 0 in that case. For comparing with MODEST observations (where Q, U are typically
noise-dominated for the Fe I 6301/6302 lines at disk center), this is an acceptable approximation.

### 4. Microturbulence and macroturbulence are fixed
These parameters broaden the spectral line. If your assumed values (e.g., ξ_mic = 1 km/s) do not match
the actual data, the synthesized line widths will differ from the observations even if T and V are
correct. This is a degeneracy that can be partially broken by inverting line depth ratios of two lines
with different gf values — a step beyond the current MUISCA scope.

### 5. NLTE effects
The Fe I 6301.5 and 6302.5 lines form under near-LTE conditions in the photosphere. NICOLE defaults to
LTE for these lines. For Ca II 8542 (a chromospheric line), you would need `Mode=NLTE` in the LINES file
and a model atom — significantly more complex. Stick with LTE for the Fe I lines.

### 6. Single-pixel runs are slow to set up
Running NICOLE per pixel requires creating a directory and file for each pixel. For a 50×50 pixel map
this means 2500 NICOLE runs. Consider:
- Selecting a representative subset of pixels (e.g., one per class: quiet Sun, plage, sunspot umbra).
- Parallelizing with GNU Parallel or SLURM job arrays.
- Writing a batch script that loops over pixels and submits each as a separate job.

### 7. `Cycles > 1` not supported
The NICOLE configuration option `Cycles=N` (for running multi-cycle inversions) is explicitly noted in
the source as unsupported in the current version. In synthesis mode this is irrelevant — you only ever
need one cycle.

---

## Connecting to the SLURM Environment

For batch synthesis on the Maxwell cluster, a SLURM array job is the cleanest approach:

```bash
#!/bin/bash
#SBATCH --job-name=nicole_synthesis
#SBATCH --array=0-2499          # one task per pixel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00

conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter

PIXEL_IDX=$SLURM_ARRAY_TASK_ID
python scripts/run_nicole_pixel.py --pixel_idx $PIXEL_IDX
```

Where `run_nicole_pixel.py` would:
1. Load the saved MUISCA inversion output (e.g., from an HDF5 or `.npy` array).
2. Extract T, V_LOS, B_LOS for the given pixel index.
3. Write a NICOLE model file, run NICOLE, read the output profile.
4. Compute and save the residual.

---

## Summary of Files You Need to Create/Adapt

| File | Purpose | Action needed |
|------|---------|---------------|
| `scripts/nicole_synthesis.py` | Main integration script | Write from scratch using the snippets above |
| `NICOLE.input` (template) | NICOLE synthesis config | Adapt `test/syn1/NICOLE.input`, set wavelength grid to match MODEST |
| `LINES` | Spectral line atomic data | Copy from `/NICOLE/test/syn1/LINES` (Fe I 6301.5 already there) |
| `utils/model_prof_tools.py` | Already in MUISCA repo | This is the NICOLE I/O library — it already exists, but uses older API |

The `utils/model_prof_tools.py` already in your MUISCA repository is a copy of NICOLE's own I/O tools.
You can use its `write_model()` function directly instead of the custom writer shown above — but check
that it writes the 1.0 ASCII format and not one of the binary formats.

---

## Physical Interpretation of the Residuals

Once you have synthesized profiles from MUISCA's predictions, the residuals tell you different things
depending on which Stokes parameter:

| Large residual in... | Likely cause |
|----------------------|-------------|
| Stokes I line depth  | Temperature error at τ ≈ 0 (continuum-forming layer) or wrong microturbulence |
| Stokes I line width  | Microturbulence or macroturbulence mismatch |
| Stokes I line shift  | V_LOS error |
| Stokes V amplitude   | B_LOS error in the line-forming region (τ ≈ −0.5 to −1.5) |
| Stokes V asymmetry   | Velocity gradient across τ — a property MUISCA cannot recover with uniform τ weighting |

A systematic blue-shift of the synthesized V profile relative to the observation would indicate that
MUISCA is overestimating V_LOS in the upper layers where Stokes V forms.
