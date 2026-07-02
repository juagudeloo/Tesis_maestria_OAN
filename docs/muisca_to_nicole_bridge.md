# MUISCA → NICOLE Forward-Synthesis Bridge

This document explains the synthesis layer that lets you take the atmosphere
that a trained MUISCA model predicts for a given Stokes profile and feed it
back into NICOLE for a forward radiative-transfer calculation. The goal is a
post-hoc consistency check: if the inversion is physically self-consistent,
the Stokes profiles re-synthesized from MUISCA's predicted atmosphere should
match the originally observed Stokes profiles.

The bridge is purely additive. No training, analysis, or fine-tuning code is
touched.

---
## Why a bridge is needed

MUISCA's analysis pipeline never persists its predictions. It runs the model
in `scripts/analysis/{muram,modest}_analysis.py`, denormalizes the output to
Kelvin / km·s⁻¹ / Gauss, and consumes the cubes in-memory to produce PNG
diagnostics. Nothing on disk is ready for a downstream synthesis step.

NICOLE, in turn, expects a per-pixel ASCII `.model` file with eight columns
in a strict order (log τ, T, Pₑ, ξ_mic, B_long, v_los, B_x, B_y), arranged
**top-to-bottom of atmosphere** (descending log τ), with `v_los` in **cm·s⁻¹**
and a header line carrying macroturbulence and stray light.

The gaps between what MUISCA produces and what NICOLE needs are:

1. The model's denormalized predictions are never written to disk.
2. `utils/model_prof_tools.py` already understands NICOLE's binary formats
   but ships no writer for any of them — not even the simple ASCII format.
3. MUISCA's optical-depth grid is ascending (top → bottom of the atmosphere is
   τ = 0 → τ = 1 in log-space); NICOLE expects descending.
4. v_los is in km·s⁻¹; NICOLE wants cm·s⁻¹.
5. MUISCA predicts only three columns (T, V_LOS, B_LOS). NICOLE needs five
   more (Pₑ, ξ_mic, ξ_mac, B_x, B_y, stray light).
6. NICOLE needs a `LINES` database and a `NICOLE.input` configuration in the
   working directory; neither lives inside MUISCA.

This bridge closes all six gaps without touching anything upstream.

---

## What was added

```
MUISCA/
├── data/
│   └── nicole_assets/                       ← (new) checked-in NICOLE templates
│       ├── LINES                            ← Fe I 6301.5 / 6302.5 line database
│       ├── NICOLE.input.template            ← synthesis-mode config with placeholders
│       └── README.md
├── docs/
│   └── muisca_to_nicole_bridge.md           ← (new) this document
├── scripts/
│   └── synthesis/                           ← (new) CLI drivers
│       ├── __init__.py
│       ├── sample_pixels.py                 ← step 0 (optional): stratify a region by model-independent |B_LOS|, pick test pixels
│       ├── export_predictions.py            ← step 1: run model → predictions.h5 (source=modest or muram)
│       ├── run_nicole_synthesis.py          ← step 2: predictions.h5 → NICOLE → syntheses.h5
│       ├── compare_synthesis.py             ← step 3: overlay PNGs + χ² (single model)
│       ├── compare_models.py                ← step 4: cross-model comparison: joined χ², bin summary, combined overlays
│       └── aggregate_comparison.py          ← step 5: statistical view (violin plots) across the full sampled tier
├── tools/
│   └── run_nicole_synthesis.sh              ← (new) sbatch front for the whole flow; --source/--step/--add-gt-pressure CLI flags
└── utils/
    ├── model_prof_tools.py                  ← (modified) added write_ascii_model()
    ├── pixel_sampling.py                    ← (new) stratified-by-|B_LOS| pixel sampler + diagnostic plot (modest- or muram-sourced)
    └── synthesis.py                         ← (new) SynthesisConfig / WholeRegionPrediction / Exporter / Runner / Comparator
```

The data flow is three sequential steps, plus an optional step 0 for picking
which pixels to run when you want more than one hand-picked coordinate. The
same flow works from **two sources**: `--source modest` (real Hinode/SOT-SP
observations, region-cropped) or `--source muram --muram-step N` (a MURaM
simulation step, e.g. one outside the model's training window, for an
out-of-distribution generalization check):

```
[Trained checkpoint] + [MODEST observation | MURaM simulation step]
        │
        ▼   scripts/synthesis/sample_pixels.py            (optional)
selected_pixels.json + abs_bz_map_selected_pixels.png
        │
        ▼   scripts/synthesis/export_predictions.py
predictions.h5 (denormalized T, V_LOS, B_LOS + matched observed Stokes
                [+ ground-truth gas pressure, muram --add-gt-pressure only])
        │
        ▼   scripts/synthesis/run_nicole_synthesis.py
        │   (one workdir per pixel: model.model, NICOLE.input, LINES)
        │   subprocess → /NICOLE/run/run_nicole.py → /NICOLE/main/nicole
syntheses.h5 (NICOLE-synthesized IQUV per pixel)
        │
        ▼   scripts/synthesis/compare_synthesis.py
overlay PNGs + chi2.json
```

Each script is a thin wrapper around the classes in
[utils/synthesis.py](../utils/synthesis.py) and
[utils/pixel_sampling.py](../utils/pixel_sampling.py), so the same workflow
can be driven from a notebook or a SLURM array if you prefer.

---

## The core writer: `write_ascii_model`

File: [utils/model_prof_tools.py](../utils/model_prof_tools.py)

```python
def write_ascii_model(
    filepath,
    logtau,                    # MUISCA convention: ascending log τ
    T,                         # K
    v_los_kms,                 # km/s — converted to cm/s inside
    b_long_G,                  # Gauss
    *,
    v_mic_cms=1.0e5,           # 1 km/s — fixed standard photospheric value
    v_mac_cms=0.0,             # 0 — no macroturbulent broadening assumed
    stray_light=0.0,           # 0 — no stray-light fraction assumed
    b_x_G=0.0,                 # 0 — no transverse field (see "Assumption 5" below)
    b_y_G=0.0,                 # 0
    el_p=None,                 # optional per-tau array; default = constant seed
    el_p_seed=1.0,             # dyn/cm^2 — overwritten by NICOLE under hydro eq
)
```

The writer is deliberately simple. It receives a MUISCA-style atmosphere
(ascending log τ, V_LOS in km·s⁻¹) and emits the NICOLE Format-version-1.0
ASCII layout in a single pass.

**Verification.** A round-trip test against
`/scratchsan/observatorio/juagudeloo/NICOLE/test/syn1/hsra_mag.model` was run:
read the canonical HSRA atmosphere via the existing `check_model` + `read_model`,
write it back with `write_ascii_model`, re-read the new file, then diff
all eight columns plus the header. Maximum relative error = **0.0e+00** for
log τ, T, Pₑ, ξ_mic, B_long, v_los, B_x, B_y, v_mac, and stray light. The
writer matches NICOLE's own reader exactly.

What the writer does internally:

1. Promotes scalar inputs (e.g. `v_mic_cms=1e5`, `b_x_G=0`) to per-tau arrays
   via NumPy broadcasting.
2. Sorts the level list by descending log τ (so the deepest layer comes
   first — NICOLE's convention).
3. Multiplies `v_los_kms` by `1e5` to convert to cm·s⁻¹.
4. If `el_p` is `None`, fills the column with a constant `el_p_seed`. NICOLE
   replaces this profile when `Impose hydrostatic equilibrium = Y` is set in
   `NICOLE.input.template`, so the seed is only the boundary condition for
   the integrator (see "Assumption 1" below).
5. Writes the header (`Format version: 1.0` + `v_mac stray_light`) and one
   data row per τ-level.

---

## The runtime layer: `utils/synthesis.py`

This single file holds four classes that own the three workflow steps plus
their shared configuration.

### `SynthesisConfig`

A dataclass that pins all the knobs of one synthesis run: which experiment
and model variation to use, where MODEST/MURaM data lives, the wavelength
grid that NICOLE should sample, the missing-column defaults, and the output
root. It is passed to all three of the other classes so the flow stays
internally consistent. The interesting fields:

| Field                | Default              | Why this default                                                                                                        |
|----------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------|
| `source`             | (required)           | `"modest"` (real Hinode/SOT-SP observations) or `"muram"` (a MURaM simulation step).                                    |
| `muram_step`         | `None`               | MURaM simulation step number; required when `source="muram"`. Ignored for `"modest"`.                                  |
| `add_gt_pressure`    | `False`              | `source="muram"` only — feed NICOLE the true MURaM gas pressure instead of a hydrostatic-equilibrium seed (see "Ground-truth pressure" below). Raises `ValueError` if set together with `source="modest"`. |
| `region_label`       | `"whole"`             | MODEST region-cropping label. Ignored when `source="muram"` — `step-N` occupies that position in the output path instead (see "Output layout" below). |
| `wl_first`           | `6300.796`           | First wavelength of the MODEST/Hinode SP Fe I 6301.5/6302.5 window.                                                     |
| `wl_step_mA`         | `21.5`               | Hinode SP spectral sampling (≈21.5 mÅ per pixel).                                                                       |
| `n_wl`               | `112`                | Number of MODEST wavelength samples per Stokes profile.                                                                 |
| `v_mic_cms`          | `1.0e5` (= 1 km·s⁻¹) | Standard photospheric value (see "Assumption 2").                                                                       |
| `v_mac_cms`          | `0.0`                | No macroturbulent broadening assumed (see "Assumption 3").                                                               |
| `stray_light`        | `0.0`                | No stray-light fraction assumed (see "Assumption 4").                                                                   |
| `el_p_seed`          | `1.0`                | dyn·cm⁻² seed for NICOLE's hydrostatic integration; close to HSRA at log τ = −2 (see "Assumption 1"). Superseded per-pixel when `add_gt_pressure=True`.  |
| `nicole_assets`      | `data/nicole_assets` | Where the `LINES` and `NICOLE.input.template` live.                                                                     |
| `nicole_root`        | `../NICOLE_v16.06`   | Location of the NICOLE Fortran tree (`main/nicole`, `run/run_nicole.py`). See "Operational prerequisite" below.          |
| `python_for_nicole`  | `"python2"`          | Interpreter used to invoke `run_nicole.py`. v16.06's driver is Python 2; set to `"python3"` or `sys.executable` if you switch to a Python-3 NICOLE. |

#### Output layout

`out_dir()`/`region_dir()` branch on `source`:

- `source="modest"`: `output_root/experiment_root/model_type/region_label/...`
  (unchanged from the original design — no `"modest"` path segment).
- `source="muram"`: `output_root/experiment_root/muram/step-N[-gt-pressure]/model_type/...`
  — `step-N` (optionally suffixed `-gt-pressure`) occupies the same path
  *level* `region_label` occupies for MODEST, not an extra nested layer on
  top of it, since MURaM has no region-cropping concept. `region_dir()`
  (the level `compare_models.py`/`aggregate_comparison.py` write into, shared
  across model variants) is `.../region_label/` for modest and `.../step-N/`
  for muram.

### `PredictionExporter`

This is step 1. It reuses the existing analysis machinery so the bridge
cannot drift out of sync with how the rest of MUISCA produces predictions.
Internally, `export(pixels)` first calls `predict_whole_region()` — a
method that runs inference once over the full cropped region and returns a
`WholeRegionPrediction` (the denormalized `pred_mhd` dict, `prediction_stokes`,
`wavelength`, `logtau`, the region's pixel-grid shape, and an optional
`gt_pressure` cube) — then subselects the requested pixels out of that cube
and writes them to HDF5. `predict_whole_region()` is also the method
[`utils/pixel_sampling.py`](../utils/pixel_sampling.py) calls to drive
stratified sampling (see "Stratified pixel sampling" below); factoring it out
means the sampler and the exporter share one inference pass and one
`(ix, iy)` indexing convention, instead of each script loading the model and
indexing pixels independently.

`predict_whole_region()` branches internally on `cfg.source` for the
data-loading step only — everything else (`AnalysisModelPipeline`, normalizer
loading, the actual `predict_and_denormalize()` inference call, the returned
`WholeRegionPrediction` shape) is shared, unbranched code:

- **`source="modest"`** (`PredictionExporter._load_modest_stokes`): the
  original MODEST path, unchanged.
- **`source="muram"`** (`PredictionExporter._load_muram_stokes`): loads the
  requested `cfg.muram_step` via the exact raw `MhdData`+`StokesData`
  sequence used elsewhere in the codebase for MURaM ingestion (`load_step` →
  `load_opacity_table` → `compute_optical_depth` → `remap_to_optical_depth`
  for `T`/`Vz`/`Bz`; `load_stokes` → `continuum_normalization` →
  `load_hinode_lsf` → `apply_spectral_convolution` → `resample_to_hinode`),
  using `pipeline.build_runtime_training_config(model_cfg)` so the LSF path,
  continuum-normalization mode, and `stokes_mult_factor` match what the
  checkpoint was actually trained with (the same config
  `scripts/analysis/muram_analysis.py` builds). Deliberately does **not**
  reuse `scripts.base_training.load_and_prepare_step()`/`MuramStepDataset`,
  since that helper normalizes Stokes I/V *inside* its own `__init__` and
  never retains the raw (continuum-normalized-only) profile the bridge needs
  for the `stokes_obs`/comparison-target field. MURaM's own spatial grid
  already matches the grid the model's Stokes input is built on, so — unlike
  MODEST — no upsample-to-a-finer-prediction-grid step is needed.

Reused symbols:

- `utils.analysis.AnalysisModelPipeline.prepare_models()` — loads the
  checkpoint and figures out `n_tau` (the number of optical-depth nodes the
  model was actually trained on).
- `AnalysisModelPipeline.predict_and_denormalize()` — the canonical inference
  routine. This is the single source of truth for unit conversion (Kelvin,
  km·s⁻¹, Gauss); the bridge does not re-implement the denormalizer.
- `AnalysisModelPipeline.build_runtime_training_config()` — muram only; the
  same per-model `TrainingConfig` reconstruction `muram_analysis.py` uses.
- `utils.normalizer.{MhdNormalizer, StokesNormalizer}` — loaded from the
  default `data/normalization_stats/` JSONs configured by `TrainingConfig`.
- `utils.modest_data.ModestData.load_all()` and `ModestDataCache` — modest
  only; the same MODEST loader that `scripts/analysis/modest_analysis.py`
  uses, so the observed Stokes that gets stored next to the predictions in
  `predictions.h5` is identical to what the model actually saw at inference
  time.
- `utils.muram_data.{MhdData, StokesData}` — muram only.

#### Ground-truth pressure (`--add-gt-pressure`, muram only)

When `cfg.add_gt_pressure=True`, `predict_whole_region()` does a **second**
`mhd.remap_to_optical_depth(...)` call — this time for `"P"` (MURaM's gas
pressure, already dyn·cm⁻²) — onto `pred_tau_arr`, the *model's own predicted
tau grid* (`pipeline.get_model_logtau_values(model_cfg)`), **not** the fixed
canonical grid `sample_pixels_by_abs_bz()` uses for stratification (those are
two unrelated grids serving unrelated purposes; mixing them up would
misalign the pressure column against the T/Vz/Bz predictions it needs to sit
alongside in one `.model` file per pixel). The result is stored as
`WholeRegionPrediction.gt_pressure` and, when present, exported per-pixel to
`predictions.h5` as an additional `Pgas_gt` dataset — see "Assumption 1"
below for how `NicoleRunner` consumes it.

What the exporter writes (`predictions.h5`):

| dataset       | shape                  | units / meaning                          |
|---------------|------------------------|------------------------------------------|
| `T`           | `(n_pixels, n_tau)`    | Kelvin                                   |
| `Vz`          | `(n_pixels, n_tau)`    | km·s⁻¹ (MUISCA convention)               |
| `Bz`          | `(n_pixels, n_tau)`    | Gauss                                    |
| `pixels`      | `(n_pixels, 2)`        | `(ix, iy)` of each row in the prediction grid |
| `logtau`      | `(n_tau,)`             | ascending log τ (MUISCA convention)      |
| `wavelengths` | `(112,)`               | Å                                        |
| `stokes_obs`  | `(n_pixels, 2, 112)`   | matched observed/reference Stokes I and V (continuum-normalized) |
| `Pgas_gt`     | `(n_pixels, n_tau)`    | *(only when `add_gt_pressure=True`)* true MURaM gas pressure, dyn·cm⁻², on the model's own tau grid |

HDF5 attrs `source`, `experiment_root`, `model_type`, `region_label`,
`pred_grid`, `output_root` (added so `run_nicole_synthesis.py`/
`compare_synthesis.py` can reconstruct the config without assuming a fixed
path depth, since muram's path is one segment deeper than modest's) and, for
`source="muram"`, `muram_step` and `add_gt_pressure`, let the downstream
steps reconstruct the original configuration from the file alone.

### `NicoleRunner`

Step 2. For each pixel, it materializes a self-contained working directory
under `<out_dir()>/pix_<ix>_<iy>/` (see "Output layout" above for how
`out_dir()` differs between modest and muram) containing exactly the three
files NICOLE needs:

- `model.model` — written via `write_ascii_model`. When `run_pixels_from_h5()`
  finds a `Pgas_gt` dataset in `predictions.h5` (i.e. the export step ran with
  `--add-gt-pressure`), it passes that pixel's row through `prepare_workdir()`
  as `write_ascii_model(..., el_p=gt_pressure_row)` instead of the default
  `el_p_seed=cfg.el_p_seed` — using the writer's pre-existing full-array
  `el_p=` parameter, which was already supported but unused before this
  feature.
- `NICOLE.input` — built from `data/nicole_assets/NICOLE.input.template` by
  substituting eight placeholders (`{NICOLE_COMMAND}`, `{INPUT_MODEL}`,
  `{OUTPUT_PROFILE}`, `{OUTPUT_MODEL}`, `{WL_FIRST}`, `{WL_STEP_MA}`,
  `{N_WL}`, and two governing the pressure treatment: `{HYDROSTATIC_EQ}` /
  `{INPUT_DENSITY}`, `"Y"`/`"Pel"` by default or `"N"`/`"PGas"` when a
  ground-truth pressure row is present — see "Assumption 1" below).
- `LINES` — copied verbatim from `data/nicole_assets/LINES`.

It then runs NICOLE by invoking its own driver:
`subprocess.run([python, "/NICOLE/run/run_nicole.py"], cwd=workdir)`. This is
the same entry point a manual NICOLE user would use; the bridge does not
re-implement NICOLE's Fortran wrappers.

The synthesized Stokes profile is read back through the existing
`utils.model_prof_tools.read_prof()`. NICOLE writes one record of four floats
per wavelength (I, Q, U, V), which `read_prof` already parses; the runner
just reshapes the result to `(4, n_wl)` and stores it in `syntheses.h5`:

| dataset        | shape                  | meaning                          |
|----------------|------------------------|----------------------------------|
| `pixels`       | `(n_pixels, 2)`        | which pixels were synthesized    |
| `stokes_synth` | `(n_pixels, 4, n_wl)`  | NICOLE I/Q/U/V, continuum-normalized |
| `wavelengths`  | `(n_wl,)`              | Å                                |

### `SynthesisComparator`

Step 3. Loads both HDF5 files, joins them by pixel coordinates, and emits:

- `chi2.json` — per-pixel χ² for Stokes I and V using user-supplied noise
  scales (`sigma_I`, `sigma_V`).
- `overlay_pix_<ix>_<iy>.png` — Stokes I and V observed vs. NICOLE-synthesized
  on the same wavelength axis.

Q and U are not compared because MUISCA only predicts the line-of-sight
field, so the transverse components written to the `.model` file are zero;
NICOLE will therefore synthesize Q ≈ U ≈ 0. The comparator includes them in
the PNG only for completeness.

---

## Assumptions, and why each was made

These are the seven decisions that bridge the conceptual gap between
MUISCA's three-quantity inversion and NICOLE's eight-column atmospheric
model.

### Assumption 1 — Electron pressure (Pₑ) is reconstructed by NICOLE, not predicted

MUISCA does not predict pressure or density. NICOLE needs an Pₑ column.

The chosen approach is to fill the column with a constant seed (default
`el_p_seed = 1.0 dyn·cm⁻²`, comparable to HSRA's value at log τ = −2) and
let NICOLE recompute the full pressure stratification under the hydrostatic
equilibrium assumption. This is enabled by two keywords in
`NICOLE.input.template`:

```
Impose hydrostatic equilibrium= Y
Input density= Pel
```

Under this configuration, NICOLE integrates dP/dτ = ρg upward using the
predicted temperature stratification and a boundary condition derived from
the seed. The seed value is therefore *not* a load-bearing prediction — it
is only the initial guess for the integrator. This is the option recommended
in the original integration guide (`docs/nicole_integration_guide.md`) and
matches what HSRA itself does in the canonical `test/syn1` example.

The limitation is that NICOLE's hydrostatic integration assumes negligible
inertial terms, while the MURaM atmospheres MUISCA was trained on are fully
convective. There will be systematic discrepancies in the deepest layers
(τ ≳ 0). This is documented but not mitigated in v1.

#### Overriding this assumption: `--add-gt-pressure` (muram only)

For MURaM sources the true gas pressure is directly available (`MhdData`
already loads it via the EOS files), so this assumption can be tested
directly rather than debated: `export_predictions.py --add-gt-pressure`
remaps MURaM's real `P` onto the model's own tau grid and feeds it to NICOLE
verbatim, flipping the two keywords above to

```
Impose hydrostatic equilibrium= N
Input density= PGas
```

— a configuration NICOLE already ships and tests itself
(`NICOLE_v16.06/test/conv2/NICOLE.input`), not a new capability being relied
on unverified. `Input density` accepts `Pgas|Pel|Nel|Dens`; MURaM's `P` is
already gas pressure in the exact unit (dyn·cm⁻²) `PGas` expects, so no
conversion is needed beyond stripping the `astropy` unit. Runs land in a
sibling `step-N-gt-pressure/` output tree (see "Output layout" above) rather
than overwriting the plain run, specifically so the two can be diffed
directly — `chi2.json`/`bin_summary.json` before vs. after — to see whether
real pressure support changes the synthesis match at all. See "Verification
result" below for the round-trip check and a first data point.

### Assumption 2 — Microturbulence is a fixed constant (1 km·s⁻¹)

MUISCA does not invert ξ_mic. Setting it to a single per-pixel value of
1 km·s⁻¹ matches the standard photospheric assumption used in essentially
every Fe I 6301/6302 inversion in the literature. This affects line width
(broadening) but not depth or shift, so it does not interact with MUISCA's
predictions for T or V_LOS. The default is overridable via `--v-mic-cms` on
the runner CLI; if you want a τ-dependent ξ_mic you can pass an array
directly to `write_ascii_model`.

### Assumption 3 — Macroturbulence is zero

For the same reason as ξ_mic. Macroturbulence adds Gaussian convolution to
the synthesized profile. Hinode/SP's instrumental profile (the convolution
that MODEST data already has baked in) is a more dominant broadening source
than typical macroturbulence (≲ 2 km·s⁻¹). Leaving ξ_mac = 0 keeps the
synthesized profile interpretable without conflating two broadening
mechanisms. If a single-pixel verification shows systematically narrower
synthesized lines than observed, raising this is the first knob to try.

### Assumption 4 — Stray light is zero

A non-zero stray-light fraction is a way of representing instrumental light
leakage that fills in the line core. MODEST observations have a deconvolution
step (Hinode's instrumental PSF is removed before MUISCA sees the data), so
the residual stray-light contamination is small. We do not model it.

### Assumption 5 — Transverse field components are zero

MUISCA inverts Stokes I and V only and predicts B_LOS only. Stokes Q and U
encode the transverse field, but they are not inputs to the model. We
therefore set B_x = B_y = 0 in the synthesized atmosphere. NICOLE will then
synthesize Q = U = 0 by construction; the bridge only compares Stokes I and V.

This is correct for verification (we are checking how well MUISCA's I/V
inversion accounts for the I/V it was given) but it means the bridge cannot
diagnose any failure mode that would manifest in Q or U.

### Assumption 6 — The optical-depth grid is reversed, not re-binned

MUISCA's convention: log τ ascending (`logtau_values[0]` = top of the
atmosphere, lowest log τ). NICOLE's convention: descending (deepest layer
first). The writer does the reversal internally via `np.argsort(logtau)[::-1]`.

The number of τ-levels and the spacing are taken verbatim from the
checkpoint that the exporter loads (via
`AnalysisModelPipeline.get_model_logtau_values`). This means:

- The example checkpoint at
  `output/experiments/experiment_81_to_181-step_size_5-normal/wfa_only/`
  has 95 levels spanning log τ ∈ [−8.0, +1.4]. The bridge writes all 95.
- A 21-level checkpoint (log τ ∈ [−2.0, 0.0], 0.1 step) would write 21 levels
  spanning that range only.

We do not interpolate. If the model was trained on a sparse grid, NICOLE
will receive that same sparse grid. The "21 τ levels" wording in
`CLAUDE.md` reflects the older default; the bridge is grid-agnostic.

### Assumption 7 — NICOLE samples the MODEST wavelength grid exactly

NICOLE allows arbitrary `[Region 1]` parameters. The template fixes them to
the MODEST/Hinode SP sampling — 6300.796 Å + 21.5 mÅ × 112 — so that the
synthesized output and the observed Stokes share a wavelength axis without
any post-hoc interpolation. Post-hoc interpolation was rejected because it
would smear the Stokes V zero-crossing, which is exactly the feature most
sensitive to the B_LOS prediction we are trying to validate.

If you want to compare against a different observation (e.g. a coarser-
sampled dataset), pass `--wl-first / --wl-step-mA / --n-wl` to
`run_nicole_synthesis.py` and the runner will re-render the template.

---

## Stratified pixel sampling (step 0)

A single hand-picked pixel can't tell you whether a finding (e.g. "Stokes V
amplitude is too small") is a local quirk or systematic across field-strength
regimes. [`scripts/synthesis/sample_pixels.py`](../scripts/synthesis/sample_pixels.py)
and [`utils/pixel_sampling.py`](../utils/pixel_sampling.py) add an optional
step 0 that runs inference once over a cropped region, stratifies pixels by
predicted `|B_LOS|` at the deepest optical-depth level, and samples a few
pixels per field-strength bin — guaranteeing the test set spans quiet-Sun,
plage, and strong-field (pore/sunspot) regimes instead of leaving that to
chance. NICOLE synthesis is fast (≈1 s/pixel), so 15–20 pixels is cheap.

`sample_pixels_by_abs_bz(cfg, n_bins=5, n_per_bin=3, seed=0)`:

1. Gets a model-independent `|B_LOS|` map. The source depends on `cfg.source`
   (`_abs_bz_map_from_modest`/`_abs_bz_map_from_muram` in
   `utils/pixel_sampling.py`), but the principle is the same for both: bin on
   ground truth, never on either candidate model's own prediction — stratifying
   by a model's own `|B_LOS|` prediction is circular when the goal is comparing
   that model against another one, since whichever model defines the bins gets
   a built-in home-field advantage (confirmed on the `negative_region` crop:
   the two model variants' own predictions produced *completely different* bin
   edges and pixel sets for the identical seed/crop, before this fix). Running
   this script once per `--model-type` with the same `--seed` therefore
   produces an *identical* pixel list across model variants, which is the
   prerequisite for `compare_models.py` (below) to be a fair comparison.

   - **`source="modest"`**: loads MODEST data via `ModestData.load_all()` (no
     model/GPU needed) and takes `|spinor_atm["Blos"][deepest_tau]|` — the
     **SPINOR inversion already bundled with the MODEST data**. SPINOR's `Blos`
     is the same ground-truth reference already used for MODEST comparisons in
     `utils/analysis.py`. Its tau grid is coarser than a model's prediction
     grid (3 levels at log τ = −2.0, −0.8, 0.0 by default vs. e.g. 95 levels
     for the example checkpoint) — "deepest level" here means SPINOR's own
     deepest level (0.0), not the model's. SPINOR's native pixel grid is also
     coarser than the model's (un-upsampled — e.g. `(80,200)` vs. the model's
     `(160,400)` for the same crop under the default 2× upsampling), so the
     map is upsampled via integer repeat (`np.repeat` along both axes, factor
     derived dynamically from the two grids' shapes, not hardcoded) before
     binning — no transpose is needed since both grids share the same axis
     order, just differing by the upsampling factor.
   - **`source="muram"`**: loads `MhdData` for `cfg.muram_step` (again, no
     model/GPU needed — this is a pure data-processing step) and remaps `Bz`
     onto a **fixed, model-independent canonical tau grid**
     (`np.arange(-2.0, 0.05, 0.1)`, `TrainingConfig`'s own default — the same
     grid used for the poster's MURaM atmosphere figures), taking the deepest
     level (log τ = 0). This grid is deliberately *not* the model's own
     predicted tau grid — using each model's own grid would reintroduce the
     exact circularity issue above, since two model variants can have
     different predicted grids. It is also *not* the grid `--add-gt-pressure`
     remaps pressure onto (which *does* need to be the model's own grid) —
     these are two independent grids used for two independent purposes; mixing
     them up would be a real bug. MURaM's own spatial grid already matches the
     model's prediction grid, so — unlike MODEST/SPINOR — no upsample step is
     needed here either.
2. Builds **log-spaced bin edges** — `np.logspace(log10(floor), log10(max), n_bins+1)`,
   i.e. equal-width intervals in log₁₀(Gauss) space — rather than
   percentile/quantile edges. The two strategies were both tried:
   - *Quantile edges* (`np.quantile(vals, linspace(0,1,n_bins+1))`) guarantee
     equal **pixel count** per bin. On the `negative_region` test crop this
     produced `[0, 5.9, 11.2, 22.0, 79.8, 10000]` G — the first four bins
     squeezed into 0–80 G and the last bin spanning 80–10000 G. That's the
     expected signature of quantile binning on a heavily right-skewed
     |B_LOS| distribution (quiet-Sun pixels dominate the pixel count), but it
     meant a 90 G plage pixel and a 788 G sunspot pixel landed in the same
     bin — poor resolution exactly where the model's WFA accuracy is most in
     question.
   - **Log-spaced edges** (current behavior) instead guarantee equal
     **decade width** per bin, so each bin is a physically meaningful
     order-of-magnitude regime (e.g. ~0.03–0.7 G, 0.7–16 G, 16–400 G,
     400–10000 G). Pixel *counts* per bin are no longer equal — a strong-field
     bin may have few or zero candidates on a quiet-Sun-dominated crop — but
     the sampling loop already tolerates sparse/empty bins.
   - The floor for the log scale is derived from the data (`min` of the
     positive values in the map, not a hardcoded constant), since `|B_LOS|`
     legitimately hits exact `0.0` G for some pixels and `log(0)` is
     undefined. Pixels are clipped to this floor only for bin *assignment*;
     the recorded `|B_LOS|` value for a zero pixel stays `0.0`, not the floor.
3. Samples up to `n_per_bin` pixels per non-empty bin via
   `np.random.default_rng(seed).choice(..., replace=False)` — this is the
   **violin/aggregate tier**, used for χ² statistics and distribution plots
   (see "Aggregate distribution comparison" below). It's deliberately a
   large *bounded* sample, not literally every pixel in the region: past a
   few hundred pixels a violin/box plot stops getting meaningfully sharper,
   while NICOLE's ~0.1–1s/pixel sequential synthesis cost would balloon into
   hours for true exhaustive coverage of a heavily right-skewed `|B_LOS|`
   population.
4. `mark_overlay_subset(result, n_overlay_per_bin, seed)` then draws a
   *second, smaller* sample of `n_overlay_per_bin` pixels per bin (default 3)
   — but drawn from *within* step 3's already-selected pixels for that bin,
   not a fresh independent draw, so this **overlay tier** is always a strict
   subset of the violin tier. It's used only for individual per-pixel Stokes
   I/V overlay PNGs (`compare_models.py`), which stay visually manageable at
   a handful per bin even though the violin tier might have dozens.

`write_pixel_selection_outputs(...)` writes, under
`<region output dir>/pixel_selection/`:

- `selected_pixels.json` — pixels, bin index, `|B_LOS|` value, an
  `is_overlay_example` flag per pixel (marking the overlay-tier subset),
  log-spaced bin edges, log τ value, crop bounds.
- `pixel_selection_snippets.txt` — ready-to-paste `--pixel ix,iy ...` args for
  `export_predictions.py`/`run_nicole_synthesis.py` (covering the full violin
  tier), and a `PIXELS=(...)` bash array matching
  `tools/run_nicole_synthesis.sh`'s format.
- `abs_bz_map_selected_pixels.png` — the `|B_LOS|` map for the whole crop,
  colored with `LogNorm` (so the color resolution matches the log-spaced
  bins; a linear scale would wash out everything below ~100 G), with sampled
  pixels overlaid as scatter points color-coded by bin.

```bash
# modest
python scripts/synthesis/sample_pixels.py \
    --experiment-root experiment_81_to_181-step_size_5-normal \
    --model-type wfa_only \
    --region-label negative_region \
    --crop-bounds 0 80 0 200 \
    --n-bins 5 --n-per-bin 20 --n-overlay-per-bin 3 --seed 0
# → output/synthesis/.../wfa_only/negative_region/pixel_selection/{...}

# muram (no --crop-bounds/--region-label -- MURaM has no cropping concept)
python scripts/synthesis/sample_pixels.py \
    --source muram --muram-step 198 \
    --experiment-root experiment_81_to_181-step_size_5-normal \
    --model-type wfa_only \
    --n-bins 5 --n-per-bin 20 --n-overlay-per-bin 3 --seed 0
# → output/synthesis/.../muram/step-198/wfa_only/pixel_selection/{...}
```

Both write `{selected_pixels.json, pixel_selection_snippets.txt,
abs_bz_map_selected_pixels.png}`. The printed snippet's `--pixel` args feed
directly into step 1 (`export_predictions.py`), which accepts the flag
repeatably.

---

## Cross-model comparison

[`scripts/synthesis/compare_models.py`](../scripts/synthesis/compare_models.py)
compares two or more trained model variants (e.g. `wfa_only` vs `no_physics`)
against each other on the same pixels, given they were each run through
steps 0–2 independently (sharing a pixel list, per the model-independent
step 0 above). It does **not** duplicate any chi² logic — it instantiates one
`SynthesisComparator` per `--model-type` and calls the existing, unmodified
`chi_square()` on each (verified bit-for-bit identical to running
`compare_synthesis.py` on each model alone), then joins the results by
pixel and by bin:

```bash
# modest
python scripts/synthesis/compare_models.py \
    --experiment-root experiment_81_to_181-step_size_5-normal \
    --region-label negative_region \
    --model-type wfa_only --model-type no_physics
# → output/synthesis/<experiment_root>/<region_label>/pixel_comparison/

# muram
python scripts/synthesis/compare_models.py \
    --source muram --muram-step 198 \
    --experiment-root experiment_81_to_181-step_size_5-normal \
    --model-type wfa_only --model-type no_physics
# → output/synthesis/<experiment_root>/muram/step-198/pixel_comparison/
```

This output directory (`SynthesisConfig.region_dir()/pixel_comparison`) sits
one level up from any single model's directory since the comparison isn't
owned by any one model variant — for modest that's
`output/synthesis/<experiment_root>/<region_label>/...` (`step-198`
occupying the `region_label` position for muram, so `region_dir()` for
muram is `output/synthesis/<experiment_root>/muram/step-198/...` directly,
not nested one level deeper under a region label). It contains:

- `cross_model_chi2.json` — per-pixel χ² for every requested model, keyed
  `"ix,iy"` → `{model_type: {chi2_I, chi2_V, n_wl}}`. Covers the full
  violin/aggregate tier (all of `sample_pixels.py`'s `--n-per-bin` pixels),
  which only makes this aggregate stronger.
- `bin_summary.json` — χ² aggregated by **mean and median** per `|B_LOS|`
  bin per model (not raw sum, since bins can have unequal pixel counts; not
  profile averaging, since Stokes V is sign-sensitive to B_LOS polarity and
  a bin can mix polarities, which would partially cancel a literal average
  of V profiles). Also covers the full violin tier.
- `<bin_lo>-<bin_hi>/overlay_pix_<ix>_<iy>.png` — one PNG per pixel, but only
  for the small **overlay tier** (`selected_pixels.json`'s
  `is_overlay_example` pixels — a strict subset of the violin tier, see step
  0 above), so the PNG count stays a handful per bin even when the violin
  tier has dozens. Each PNG shows the observed Stokes I/V curve (solid
  black, taken from one model's `predictions.h5` after a sanity check that
  it agrees with the others — it should, since they all reference the same
  underlying MODEST observation) plus one dashed, distinctly-colored
  synthesized curve per model variant, on the same axes. Mirrors
  `SynthesisComparator.plot_overlay`'s layout (`figsize=(12,4)`, same
  titles/labels/save convention) for visual consistency with the
  single-model overlays. If `selected_pixels.json` predates the overlay-tier
  feature (no pixel has `is_overlay_example`), the script warns and falls
  back to plotting every common pixel instead of producing zero PNGs.

If the requested models' pixel sets don't actually match (e.g. step 0 was
run with mismatched seeds, or wasn't re-run after this fix), the script
warns and proceeds on the intersection rather than failing outright — useful
as a diagnostic that step 0 wasn't applied consistently.

### Aggregate distribution comparison

[`scripts/synthesis/aggregate_comparison.py`](../scripts/synthesis/aggregate_comparison.py)
takes the same inputs as `compare_models.py` but, instead of per-pixel
overlays, builds a statistical view across the **full violin tier** —
useful for a conference-poster figure where a handful of example spectra
isn't convincing evidence of a representative comparison. It independently
recomputes χ² via `SynthesisComparator.chi_square()` (cheap pure-numpy, so
this duplication costs nothing and keeps the two scripts runnable in either
order or alone):

```bash
python scripts/synthesis/aggregate_comparison.py \
    --experiment-root experiment_81_to_181-step_size_5-normal \
    --region-label negative_region \
    --model-type wfa_only --model-type no_physics
# → output/synthesis/<experiment_root>/<region_label>/aggregate_plots/
```

Same `--source`/`--muram-step`/`--add-gt-pressure` flags as `compare_models.py`
apply here too, writing to the `region_dir()`-relative sibling
`.../aggregate_plots/` directory.

- `aggregate_chi2_long.json` — long-format table, one record per
  `(pixel, model_type)`: `{ix, iy, bin, bin_lo, bin_hi, abs_bz_gauss,
  model_type, chi2_I, chi2_V}`. Ready to load straight into a
  `pandas.DataFrame`.
- `violin_chi2_I.png` / `violin_chi2_V.png` — seaborn `catplot` violin plots,
  faceted by `|B_LOS|` bin (`col`) and colored by model variant (`hue`),
  log-y (χ² spans orders of magnitude), `cut=0` (clamps each violin's KDE to
  the observed data range — χ² is strictly non-negative, and an unclamped
  KDE can visually extend below zero, misrepresenting the metric).

---

## How to use it

Three steps (plus the optional step 0 above, and steps 4-5 when comparing
2+ model variants). Each step can be run independently.

```bash
# 1. Run the trained model on a region of MODEST and persist the predictions
python scripts/synthesis/export_predictions.py \
    --source modest \
    --experiment-root experiment_81_to_181-step_size_5-normal \
    --model-type wfa_only \
    --region-label negative_region \
    --crop-bounds 0 80 0 200 \
    --pixel 40,100
# → output/synthesis/.../predictions.h5

# 2. Build per-pixel NICOLE workdirs and invoke NICOLE in synthesis mode
python scripts/synthesis/run_nicole_synthesis.py \
    --predictions-h5 output/synthesis/.../predictions.h5 \
    --nicole-root /scratchsan/observatorio/juagudeloo/NICOLE_v16.06
# → output/synthesis/.../syntheses.h5
# → output/synthesis/.../pix_00040_00100/{model.model,NICOLE.input,LINES,run_nicole.py,profile.pro,model_out.mod,logfile_1}

# 3. Compare and produce diagnostics
python scripts/synthesis/compare_synthesis.py \
    --predictions-h5 output/synthesis/.../predictions.h5 \
    --syntheses-h5 output/synthesis/.../syntheses.h5
# → output/synthesis/.../comparison/{chi2.json, overlay_pix_*.png}
```

For a MURaM source (e.g. a step outside the training window, for an
out-of-distribution generalization check), step 1 becomes:

```bash
python scripts/synthesis/export_predictions.py \
    --source muram --muram-step 198 \
    --experiment-root experiment_81_to_181-step_size_5-normal \
    --model-type wfa_only \
    --pixel 152,449
# → output/synthesis/<experiment_root>/muram/step-198/wfa_only/predictions.h5
```

— no `--region-label`/`--crop-bounds` (MURaM has no cropping concept in this
feature); steps 2-3 are unchanged (they're fully source-agnostic, driven by
attrs already in `predictions.h5`). Add `--add-gt-pressure` to feed NICOLE
MURaM's real gas pressure instead of a hydrostatic-equilibrium seed (see
"Ground-truth pressure" above) — output lands in a sibling
`muram/step-198-gt-pressure/` tree so it can be diffed against the plain run.

Or use the sbatch front, which chains steps 0-3 for one or more `MODEL_TYPES`
(running step 4, `compare_models.py`, and step 5, `aggregate_comparison.py`,
automatically at the end if 2 or more are listed):

```bash
sbatch tools/run_nicole_synthesis.sh                                     # modest (edit-in-file config)
sbatch tools/run_nicole_synthesis.sh --source muram --step 198           # muram
sbatch tools/run_nicole_synthesis.sh --source muram --step 198 --add-gt-pressure
```

Edit the constants at the top of [tools/run_nicole_synthesis.sh](../tools/run_nicole_synthesis.sh)
to pick the experiment, model variations, region, and pixel list; `--source`/
`--step`/`--add-gt-pressure` are the only settings exposed on the command
line (everything else stays edit-in-file, deliberately — see the script's
own `--help`). Listing multiple `MODEL_TYPES` is the easiest way to run a
full cross-model comparison end-to-end: step 0's model-independent sampling
gives every listed variant the same pixel list, steps 1-3 run for each
variant in turn, and steps 4-5 join them.

---

## Operational prerequisite: building NICOLE v16.06

NICOLE's Fortran binary is *not* part of the MUISCA tree. The bridge is
configured against **NICOLE v16.06** living at
`/scratchsan/observatorio/juagudeloo/NICOLE_v16.06/`. The build was done once
on this machine with gfortran 10.2:

```bash
cd /scratchsan/observatorio/juagudeloo/NICOLE_v16.06/main

# Regenerate the makefile for gfortran (the shipped one targets Intel Fortran).
# create_makefile.py is Python 2 in v16.06.
python2 create_makefile.py \
    --compiler=gfortran \
    --otherflags="-O3 -fdefault-real-8 -fallow-argument-mismatch -fallow-invalid-boz" \
    --recl=8 \
    -y

make
```

The `-fallow-argument-mismatch` flag is needed under gfortran ≥ 10 because
v16.06's `nicole.f90` calls a routine with a rank-1 array where the
declaration expects a scalar — older compilers silently accepted it. Without
the flag, the build fails at the final link step. `-fallow-invalid-boz` is a
similar leniency for legacy `BOZ` constants.

This produces `/scratchsan/observatorio/juagudeloo/NICOLE_v16.06/main/nicole`
(a 1.3 MB ELF executable). The bridge checks for this file at
`NicoleRunner.__init__` time and refuses to proceed if it is missing.

If you cannot build NICOLE locally (missing gfortran, etc.), the bridge will
still complete steps 1 and 2-up-to-`prepare_workdir`: the trained model will
run, `predictions.h5` will be written, and the per-pixel working directories
will be fully populated. Only the `subprocess.run` of NICOLE itself will be
skipped.

### Why v16.06 instead of the source already at `/NICOLE/`

The `/scratchsan/observatorio/juagudeloo/NICOLE/` tree on this machine is
incomplete: 14 of the 17 expected source files in `numerical_recipes/` are
missing (`nrtype.f90`, `nrutil.f90`, `nr.f90`, and the standard Press et al.
helpers). Those files come from *Numerical Recipes in Fortran 90* (Cambridge
University Press) and are not freely redistributable, so a fresh git clone
typically does not include them. v16.06 was uploaded with the full
`numerical_recipes/` directory included, which is why it builds where the
existing `/NICOLE/` does not. The two trees are otherwise installed
side-by-side; the existing `/NICOLE/` is left untouched.

### v16.06 quirks the bridge adapts to

Five small adaptations live in [utils/synthesis.py](../utils/synthesis.py)
to drive v16.06 specifically. They are not bridge bugs — they are
version-specific entry-point details.

| Quirk                                                                                | Where it's handled                                                                                            |
|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `run_nicole.py` is Python 2 (uses `print "..."`, `<>` operator).                     | `SynthesisConfig.python_for_nicole = "python2"`; runner invokes via that interpreter.                         |
| `run_nicole.py` only initializes `inputmodel` / `outprof` when CLI flags are passed. | `NicoleRunner.run_pixel` always passes `--modelin=model.model --profout=profile.pro`.                          |
| `run_nicole.py` opens itself via relative `open('run_nicole.py')`.                   | `NicoleRunner.prepare_workdir` symlinks the driver into each per-pixel workdir (canonical `test/syn1` pattern). |
| `run_nicole.py` needs `--nicolecommand` to point at the Fortran binary.              | Runner passes `--nicolecommand=<nicole_root>/main/nicole` derived from `SynthesisConfig.nicole_root`.           |
| Shipped `check_prof` opens the binary output in text mode and `UnicodeDecodeError`s under Python 3. | Runner skips the probe and calls `read_prof` directly with `("nicole2.3", 1, 1, n_wl)`, which are known a priori. |

If you later switch to a NICOLE version with a Python-3 `run_nicole.py`,
flip `python_for_nicole = "python3"` (or `sys.executable`) in
`SynthesisConfig`. None of the other quirks are version-specific in a
problematic way: the `--modelin`/`--profout` flags are documented in NICOLE
itself, and the symlink pattern matches all in-tree `test/*` examples.

---

## Verification result

### Single-pixel verification

A single MODEST pixel ((ix, iy) = (40, 100) of the `negative_region` crop of
`experiment_81_to_181-step_size_5-normal/wfa_only`) was driven end-to-end
through all three steps. NICOLE wall time per pixel is roughly one second.
The comparator wrote:

- `chi2.json` containing per-pixel χ² for Stokes I and V.
- `overlay_pix_00040_00100.png` showing observed vs. NICOLE-synthesized
  Stokes I and V on the same axis.

Findings from the overlay (these are statements about the trained model,
not about the bridge):

- Fe I 6301.5 and 6302.5 absorption lines are recovered in the right place
  with the right shape.
- The synthesized line cores are deeper than observed (≈ 0.34 vs ≈ 0.45),
  consistent with the predicted line-forming-region temperature being too
  low or the continuum-normalization mismatch documented in
  "Current limitations" item 3.
- The synthesized lines are slightly redshifted relative to observed
  (≈ 0.1 Å), consistent with the predicted V_LOS at line-forming optical
  depth being too positive.
- The synthesized Stokes V doublet has the correct antisymmetric shape but
  amplitudes ≈ 10× smaller than observed (±0.01 vs ±0.12). This is the
  bridge surfacing exactly the kind of finding it was designed to: at the
  line-forming layer (log τ ≈ −1), the WFA-only model under-predicts B_LOS
  despite the deep-layer Bz reaching 2.6 kG in this pixel.
- Stokes Q and U are numerically zero in the synthesized output by
  construction (assumption 5), and the comparator does not attempt to
  match them.

### Multi-pixel, stratified verification

The same crop was re-tested using `sample_pixels.py` (`--n-bins 5
--n-per-bin 3 --seed 0`), which selected 15 pixels across 5 log-spaced
`|B_LOS|` bins (deepest level, log τ ≈ 1.40):

| bin | `|B_LOS|` range (G) | pixels (examples) |
|-----|----------------------|--------------------|
| 0   | 0.001 – 0.026        | weakest quiet-Sun  |
| 1   | 0.026 – 0.65         | quiet-Sun          |
| 2   | 0.65 – 16            | weak plage         |
| 3   | 16 – 402             | plage / pore       |
| 4   | 402 – 10000          | sunspot-level      |

All 15 were run through `export_predictions.py` → `run_nicole_synthesis.py`
→ `compare_synthesis.py` without errors, producing 15 `overlay_pix_*.png`
files and a `chi2.json` with 15 entries. The χ² pattern confirms the
single-pixel finding is systematic, not local:

- **χ²(Stokes V) is flat at ~10²–10⁴ across bins 0–3** (weak through
  moderately-strong field, `|B_LOS|` up to ~80 G).
- **χ²(Stokes V) jumps ~2 orders of magnitude in bin 4** (`|B_LOS|` ≈
  440–850 G): from ~10²–10⁴ to ~2×10⁵–9×10⁵. The WFA-only model's Stokes V
  reconstruction degrades sharply specifically in the strong-field regime —
  exactly where the single-pixel run's "10× too small" Stokes V amplitude
  was first spotted, now confirmed across multiple independent pixels rather
  than one coordinate.
- χ²(Stokes I) does not show the same bin-4 jump (it ranges ~3×10⁶–1.8×10⁷
  across all bins with no clear trend with field strength), suggesting the
  degradation is specific to the magnetic-sensitive Stokes V channel and
  not a general synthesis-quality issue that scales with field strength.

### MURaM source verification

The full pipeline (steps 0-5) was run end-to-end against MURaM step 198 —
**outside** the model's training window (`experiment_81_to_181-step_size_5-normal`
trains on steps 81-181), so this is a genuine out-of-distribution
generalization check rather than a held-out-but-similar test set — for both
`wfa_only` and `no_physics`:

- Step 0 confirmed the ground-truth-Bz stratification is model-independent:
  both model variants produced an **identical** pixel list for the same
  seed (verified programmatically, not just by inspection), the same
  guarantee already established for the SPINOR/MODEST case.
- Step 1's export produced real, physically sensible field-strength bins
  (e.g. `[0.0003, 0.05]`, `[0.05, 11]`, `[11, 2129]` G at log τ = 0 for one
  run), and `predictions.h5`'s `pred_grid` came out as MURaM's native
  `(480, 480)` — confirming no upsampling step was mistakenly triggered
  (MURaM's grid already matches the model's prediction grid, unlike MODEST).
- A single-pixel overlay showed the expected Fe I 6301.5/6302.5 line shapes
  and, notably, a **much tighter** Stokes V match than the MODEST
  single-pixel case above (this makes sense: the comparison target here is
  MURaM's own synthesized Stokes, which the model has at least partially
  learned the structure of, vs. real independently-observed Hinode data).
- Steps 4-5 correctly wrote `pixel_comparison/`/`aggregate_plots/` directly
  under `muram/step-198/` (sibling to the per-model-type directories, not
  nested under an extra region-label level), and "Comparing 2 models on N
  common pixels" confirmed the pixel sets matched exactly between the two
  model variants.
- One incidental finding worth flagging: on this out-of-distribution step,
  `wfa_only` and `no_physics` came out much closer to each other in χ²(V)
  than on real MODEST data (where WFA showed a clear regularization
  advantage — see "Multi-pixel, stratified verification" above). This is an
  open generalization question, not something this bridge investigates
  further on its own.

### Ground-truth pressure verification (`--add-gt-pressure`)

Run against the same MURaM step 198, `wfa_only`:

- The remapped `Pgas_gt` dataset was cross-checked against an independent
  recomputation of MURaM's `P` on the same pixel and the same tau grid
  (`pred_tau_arr`, not the stratification grid) — **bit-for-bit identical**
  (`np.allclose(..., rtol=1e-8)` → `True`, max relative difference `0.0`).
- The generated `NICOLE.input` for a ground-truth-pressure pixel correctly
  substituted `Impose hydrostatic equilibrium= N` / `Input density= PGas`;
  the plain (no-flag) run correctly kept `Y`/`Pel`.
- The generated `model.model`'s pressure column matched the corresponding
  `Pgas_gt` value exactly (e.g. deepest level, log τ = 1.40: `2.087E+05`
  dyn·cm⁻² in the file vs. `208685.46` dyn·cm⁻² in the HDF5 dataset).
- On the one pixel tested, the ground-truth-pressure run produced
  measurably different χ² than the plain run at the same pixel
  (χ²_I: 3.64×10⁶ → 3.44×10⁶; χ²_V: 62.3 → 67.1) — confirming the effect
  propagates all the way through NICOLE rather than being silently ignored,
  but this is a single-pixel smoke test, not a claim about whether
  ground-truth pressure systematically helps or hurts. That comparison is
  exactly what running the aggregate/cross-model steps with and without
  `--add-gt-pressure` and diffing `bin_summary.json` is for.

---

## Current limitations and open follow-ups

These are not bugs — they are tradeoffs documented for awareness:

1. **No region-cropping for MURaM.** `--source muram` always runs on the
   full simulation domain (matching `scripts/analysis/muram_analysis.py`'s
   own behavior) — there is no MODEST-style `--crop-bounds` equivalent.
   `region_label` stays a `SynthesisConfig` field for dataclass simplicity
   but is silently ignored for `source="muram"`; `step-N` occupies that
   position in the output path instead.
2. **Explicit pixel list required.** `export_predictions.py` still expects
   an explicit `--pixel` list — `sample_pixels.py` (step 0) picks a
   representative handful for you, but there is no driver that fans out
   across *every* pixel in a region. A SLURM-array driver for full-region
   coverage is the natural next extension.
3. **Stokes-I normalization mismatch.** The MODEST loader returns
   "continuum-normalized" Stokes I, but the empirical range in a sample
   pixel is ≈ 0.44–0.80 rather than ≈ 0.0–1.0 (the line core sits at 0.44,
   the continuum at 0.80). NICOLE's output normalizes to `I_continuum = 1`.
   The comparator overlays them as-is in v1; if the visual mismatch is
   distracting, rescale the observed I to `I / max(I)` before comparison.
4. **No HSRA padding.** If a checkpoint's deepest level (largest log τ)
   stops well above log τ = +1, NICOLE's hydrostatic integration will
   extrapolate the temperature stratification deeper into the atmosphere on
   its own. For the 95-level example checkpoint this is moot (max log τ =
   +1.4 already), but for the canonical 21-level [−2.0, 0.0] grid it may
   matter. Padding with HSRA was discussed in the plan and rejected for v1
   on the basis that NICOLE's extrapolation is generally well-behaved. This
   also does not apply when `--add-gt-pressure` is used, since NICOLE no
   longer integrates hydrostatic equilibrium in that mode at all.
5. **`data/` is gitignored.** The bridge places its templates at
   `data/nicole_assets/`, which lives inside the gitignored `data/` tree.
   The files work at runtime, but to commit them add an override to
   `.gitignore`:

   ```
   !data/nicole_assets/
   !data/nicole_assets/*
   ```

   or move the directory under `scripts/synthesis/templates/` and update the
   default `nicole_assets` path in `SynthesisConfig`.
6. **`--add-gt-pressure` is a targeted diagnostic, not a general feature.**
   It only remaps and feeds the MURaM `P` column that already exists on
   disk; it does not attempt to derive or predict pressure for MODEST
   sources (which have none), and it does not change any other assumption
   (microturbulence, macroturbulence, stray light, transverse field all stay
   as in Assumptions 2–5). Treat it as a one-off "does real pressure change
   anything" test, not a new default mode.
