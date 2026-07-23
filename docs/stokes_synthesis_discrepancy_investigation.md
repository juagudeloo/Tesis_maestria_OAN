# Stokes Synthesis Discrepancy Investigation

This document summarizes an investigation into why NICOLE-synthesized Stokes
profiles (from the [MUISCA → NICOLE bridge](muisca_to_nicole_bridge.md))
don't match observed/true Stokes profiles as closely as expected — both for
real MODEST observations and for MURaM ground truth. It records what was
tested, what was ruled out, what was confirmed, and what's still open. All
of this work lives on the `tau-scale-experiment` git branch.

---

## Origin

The MUISCA→NICOLE bridge's initial verification (see
`muisca_to_nicole_bridge.md`, "Verification result") already documented two
symptoms on real MODEST data:

- A continuum-normalization mismatch: NICOLE's synthesized Stokes I doesn't
  sit at the same continuum level as the observed profile.
- A slight wavelength shift and a Stokes V amplitude mismatch.

A co-advisor raised the concern that MODEST's *observed* Stokes profiles
might have been normalized by SPINOR (the inversion code used to produce
MODEST's independent B_LOS ground truth) in a way that isn't compatible
with NICOLE's own continuum convention, and proposed re-inverting MODEST
with NICOLE directly using un-normalized ("raw counts") data.

## Hypotheses tested, in order

### 1. Did SPINOR normalize MODEST's observed Stokes? — Ruled out

Read `utils/modest_data.py::ModestData.load_obs_stokes` in full: it loads
`inverted_obs.1.fits` and applies zero rescaling beyond a Stokes-V sign
flip. `continuum.fits` (the local continuum) is loaded but never divided
into the Stokes cube anywhere in the codebase. SPINOR only appears in this
project as the source of `inverted_atmos.fits` (the *inverted atmosphere*,
parsed by `build_spinor_atmosphere`) — never as something that touches the
input intensities. The FITS files themselves are already continuum-scaled
to ≈0.84 (not exactly 1.0) before this repo ever loads them, most likely
from an upstream Hinode/SP calibration step (a quiet-Sun-reference
convention, per a collaborator's independent confirmation), not from
SPINOR.

### 2. Does locally renormalizing MODEST's observed I fix the comparison? — No, mixed result

Divided each observed pixel's Stokes I by its own local `continuum.fits`
value before comparing against the existing NICOLE synthesis. Result:
χ²(I) got *worse* for 6 of 9 test pixels, not uniformly better. This ruled
out "just renormalize the observed side" as a fix and motivated checking
whether the same offset appears even without any MODEST-specific
calibration involved at all.

### 3. Does the same continuum offset appear on MURaM ground truth (no SPINOR, no MODEST)? — Confirmed, still present

Compared NICOLE-resynthesized Stokes (from the MUISCA model's *prediction*
on MURaM step 198) against MURaM's own synthetic "truth" Stokes — a case
with zero SPINOR or MODEST calibration involved. The same systematic
continuum deficit appeared: mean ratio ≈0.91 across a 146-pixel
|B_LOS|-stratified sample (10 bins × 15/bin), present in 97–99% of pixels.
This ruled out MODEST-specific calibration as the cause — whatever's
happening is inside the MUISCA→NICOLE synthesis chain itself.

### 4. Is the hydrostatic-equilibrium pressure seed the cause? — Ruled out (small effect)

Added `--add-gt-pressure` to the bridge (`utils/synthesis.py`,
`data/nicole_assets/NICOLE.input.template`'s `{HYDROSTATIC_EQ}`/
`{INPUT_DENSITY}` placeholders): feeds NICOLE the true MURaM gas pressure
(`Input density=PGas`, `Impose hydrostatic equilibrium=N`) instead of the
default flat electron-pressure seed integrated via hydrostatic equilibrium
(`el_p_seed=1.0`, `Impose hydrostatic equilibrium=Y`). Across the same
146-pixel sample, the continuum ratio barely moved (0.907 → 0.911 for
`wfa_only`). This is still part of the bridge today (`--add-gt-pressure`
flag, documented in `muisca_to_nicole_bridge.md`).

### 5. Is predicted temperature biased cool at the continuum-forming layer? — Ruled out

Compared MUISCA's predicted T against MURaM's true T at log τ=0 across the
same 146 pixels: mean difference −29 K out of ≈6390 K (≈0.5%), with only
58.9% of pixels predicted cooler than truth — essentially no directional
bias, and far too small to explain a ~9% continuum deficit.

### 6. Is NICOLE's default continuum-normalization convention the cause? — No (but this test found the real issue)

NICOLE's `Continuum reference` keyword (never previously set in
`NICOLE.input.template`, so defaulting to option 1: normalize to *HSRA's*
continuum intensity at disk center, a fixed reference **independent of the
input atmosphere**) looked like a strong candidate, since MURaM/MODEST
data is normalized to something tied to the *actual* atmosphere instead.
Tested `Continuum reference=4` (local normalization to each region's own
first wavelength point). Result: **no effect on line shape** — because
switching normalization convention is mathematically just an overall
rescale, and χ²(I) computed after manually locally-renormalizing the
*default* output was bit-for-bit identical to the `Continuum reference=4`
run. This is a real, useful negative result — it corrected an earlier
over-claim.

However, this test surfaced the actual finding: once both profiles are
compared on their own local continuum (removing the scale question
entirely), **NICOLE's line core is shallower than MURaM's true line core in
all 15 of 15 test pixels**, with real per-pixel magnitude (e.g. true
core/Ic 0.130 vs. NICOLE 0.234 at one pixel). This is a genuine
line-formation discrepancy, not a normalization artifact.

### 7. Is the tau-scale (Rosseland vs. τ₅₀₀) mismatch the cause? — Confirmed, largely explains the continuum offset

Traced the tau grid MUISCA is trained and predicts on back to its source:
`utils/muram_data.py::MhdData.load_opacity_table` loads
`data/csv/kappa.0.dat`, explicitly commented `# Rosseland mean opacity in
cm²/g`, confirmed independently by `notebooks/1-muram_mhd_data.ipynb`
(cell 14, markdown): *"we'll do the spline interpolation of the values
corresponding to the Rosseland opacity"* — an intentional choice from the
very start of this project, for MURaM's own radiative-MHD energy-balance
purposes.

NICOLE's ASCII `.model` format, by contrast, requires **log(τ₅₀₀₀)** — the
continuum optical depth at 500 nm — stated three times in
`NICOLE_v16.06/manual/manual.tex` (lines 798, 1151, 1175). Rosseland-mean
opacity is a harmonic mean over all wavelengths (dominated by the most
transparent ones), a physically different quantity from the monochromatic
opacity at 500 nm. `docs/nicole_integration_guide.md` (lines 55–63, 205)
already *assumed* the tau column was τ₅₀₀ — the mismatch between that
assumption and what the code actually computes had gone unnoticed.

**Attempted fix #1 — let NICOLE compute τ₅₀₀ internally (dead end):**
NICOLE's `Height scale=z` option (feed geometric height instead of tau,
let NICOLE derive τ₅₀₀ from its own opacity routines) looked promising and
required no new opacity physics. Implemented it end-to-end (per-pixel
z(τ_Rosseland) inversion, `write_ascii_model`'s `depth_override` param,
`{HEIGHT_SCALE}` template placeholder) — but NICOLE crashed:
*"Something is wrong with the Z scale in the model, it spans from 0.0 to
0.0."* Traced into NICOLE's Fortran source (`main/nicole.f90:936-947`,
`misc/z_to_tau.f90`): the ASCII `.model` reader only ever populates
`Atmo%ltau_500` from column 1, regardless of the `Height scale` setting —
`Height scale=z` is only wired up for NICOLE's separate IDL-savefile input
format, which this bridge doesn't use. This entire implementation was
**reverted** — none of it exists on the branch anymore.

**Revisiting attempted fix #1 via an IDL-savefile writer (not attempted,
but re-assessed as feasible):** After adopting fix #2, we re-examined
whether the IDL-savefile route is actually viable, since it would let
NICOLE compute τ₅₀₀ using its own *complete* opacity physics (H⁻ + He⁻ +
H₂⁻ + metals + scattering, via `Wittmann_opac`) instead of the H⁻-only
approximation in fix #2 — which would directly test whether the residual
line-depth discrepancy (Conclusion #2) is caused by fix #2's missing
opacity terms. Three findings changed the assessment from "dead end" to
"real but nontrivial option":

1. No real IDL/GDL installation is needed to *use* this path. Per NICOLE's
   manual (`manual.tex:268-281`): IDL-savefile input *"will then be
   transformed by the Python wrapper into NICOLE's own native binary
   format"* — `run_nicole.py` (the same script this bridge already calls)
   reads the savefile and converts it before the Fortran executable ever
   runs. Grepping NICOLE's Fortran source confirms zero mentions of
   "savefile" anywhere in `main/` or `forward/` — the Fortran side never
   sees it. (Confirmed separately: no IDL/GDL is installed on this system
   anyway, so this matters.)
2. NICOLE ships `run/idlsave.py`, a ~600-line pure-Python IDL SAVE-format
   *reader* (the basis for `scipy.io.readsav`) — but it only reads, no
   writer exists anywhere in this codebase or a well-maintained third-party
   package. Building one means mirroring the reader's byte-level logic
   (types, structures, arrays, alignment) into a writer — real, bounded
   work, with `run_nicole.py` itself usable as an iterative validator
   (write candidate bytes, try feeding them through, see if it parses).
3. We would **not** need to build our own equation-of-state solve to
   supply the savefile's required `nH`/`nHminus`/`nHplus`/`nH2`/`nH2plus`/
   `el_p` arrays. `Fill_densities` (`forward/forward.f90:258`, called from
   `main/nicole.f90:933`) already derives all of these from just
   (T, gas pressure) when `Input density=Pgas` — the exact same code path
   `--add-gt-pressure` already exercises successfully today. Supplying
   `z` (geometric height) and `gas_p` in the savefile, with
   `Height scale=z` and `Input density=Pgas`, should be enough for NICOLE
   to derive τ₅₀₀ from its own full opacity self-consistently.

Not built. If pursued, the scope is comparable to `tau500_opacity.py`
(fix #2) but as a binary-serialization task rather than a physics one.

**Attempted fix #2 — approximate τ₅₀₀ directly (adopted):** Built
`utils/tau500_opacity.py`: an H⁻ bound-free + free-free opacity
approximation, ported line-for-line from NICOLE's own
`forward/wittmann_opac.f90` (not from a remembered textbook formula, to
avoid transcription risk), combined with an empirical electron-pressure
fit Pe(T) built from the real HSRA reference atmosphere
(`data/csv/hsra_pel.model` / `hsra_pgas.model`, copied from
`NICOLE_v16.06/run/hsra.model` and `test/conv2/hsra_pg.model`). This
deliberately skips He⁻, H₂⁻, and metal-line opacity contributions (H⁻
dominates through most of the photosphere; the skipped terms matter most
in the coolest upper layers) — see the module docstring for the full list
of approximations and their justification. Sanity-checked against the
existing Rosseland table and literature values before use (κ₅₀₀ ≈ 0.69
cm²/g at the classic τ=1, T≈6390 K point — in the expected literature
range).

## Ground-truth τ₅₀₀ validation experiments

Two scripts (`scripts/synthesis/tau500_ground_truth_test.py`,
`tau500_multi_step_regen.py`) bypass the MUISCA model entirely: remap
MURaM's *own true* T/Vz/Bz onto the new τ₅₀₀ grid, run NICOLE forward
synthesis directly, and compare against MURaM's own `stokes_*.npy`
profiles. This isolates the tau-scale hypothesis from model-prediction
error.

**Step 198, 146 pixels** (paired against the existing Rosseland-tau,
model-predicted baseline on the same pixels):

| metric | OLD (Rosseland, model-predicted) | NEW (τ₅₀₀, ground truth) |
|---|---|---|
| mean continuum ratio | 0.907 | 0.988 |
| mean line-core-depth error | +0.0596 (too shallow) | +0.0352 (too shallow) |
| % pixels too-shallow | 94.5% | 82.9% |
| median χ²(I), shape-only | 1.247e6 | 1.235e6 |

**Six independent MURaM steps** (81, 106, 131, 156, 181 — spanning the
full training range — plus 198; ~80 pixels/step, model-independent
|B_LOS|-stratified sampling):

| step | continuum ratio | line-depth error | % too-shallow |
|---|---|---|---|
| 81 | 1.016 | +0.0336 | 81.2% |
| 106 | 1.087 | +0.0442 | 78.8% |
| 131 | 1.013 | +0.0367 | 89.2% |
| 156 | 1.004 | +0.0283 | 75.9% |
| 181 | 1.011 | +0.0361 | 85.0% |
| 198 | 1.009 | +0.0305 | 71.2% |

Output for each step (NICOLE's resynthesis, MURaM's own Stokes, and
per-pixel metrics) is saved at
`output/synthesis/tau500_nicole_regen/step-{N}/`.

## Conclusions

1. **The continuum-brightness mismatch is explained by the tau-scale bug**,
   and is essentially closed by the τ₅₀₀ correction — consistently across
   6 independent simulation snapshots (ratio 0.86–0.91 → 1.00–1.09), not a
   single-step fluke.
2. **A separate, smaller line-core-depth discrepancy survives the fix**,
   and is *also* consistent and reproducible across all 6 steps (NICOLE
   shallower than MURaM's own Stokes in 72–89% of pixels, every time, by a
   similar magnitude each time). Because this persists even against
   ground-truth atmospheres with a corrected tau scale, it is not
   explained by model-prediction error or by the tau-scale bug — it's a
   genuine, reproducible difference between NICOLE's own synthesis physics
   and whatever code originally produced MURaM's `stokes_*.npy` files
   (which nobody currently working on this project can identify with
   certainty).
3. **SPINOR normalization, the hydrostatic pressure seed, and a
   temperature bias were each tested directly and ruled out** as
   explanations for the original discrepancy.
4. NICOLE's default continuum-normalization reference (HSRA disk-center)
   and MURaM's own `fixed_ic` reference (`data/normalization_stats/
   ic_reference_stats.json`, averaged over simulation steps 70–80) are
   two independently-derived conventions that happen to agree to within
   **0.79%** (empirically measured by comparing NICOLE's `Continuum
   reference=0` raw output against its default-normalized output for the
   same pixel) — not a source of bias in the findings above, but worth
   stating explicitly since it's easy to misread as "the same
   normalization" when it isn't.
5. The CNN's input-side continuum normalization (`fixed_ic`) and NICOLE's
   synthesis-side normalization operate on two disjoint parts of the
   pipeline today (input Stokes vs. the forward-synthesis comparison
   output) — no redundancy currently exists. This *would* become a real
   redundancy if NICOLE-regenerated Stokes ever replace `stokes_*.npy` as
   the model's training/ground-truth source without adjusting the loader
   (see "Open questions" below).

## What's on the `tau-scale-experiment` branch

- `utils/tau500_opacity.py` — the H⁻ opacity approximation (see its
  docstring for the full derivation and every approximation made).
- `scripts/synthesis/tau500_ground_truth_test.py` — single-step (198)
  validation, with a paired old-vs-new comparison against the existing
  baseline.
- `scripts/synthesis/tau500_multi_step_regen.py` — the 6-step
  regeneration described above.
- `data/csv/hsra_pel.model`, `data/csv/hsra_pgas.model` — the HSRA
  reference atmosphere (electron-pressure and gas-pressure conventions),
  now tracked in git (previously `data/` was entirely gitignored; a
  narrow carve-out was added for these plus `data/nicole_assets/*` and
  `data/normalization_stats/*.json`, since those are small files scripts
  actually load at runtime — see `.gitignore`'s trailing block).
- Commits: `51f970c` (opacity module + experiment scripts), `dff03f5`
  (the `.gitignore` carve-out). Pushed to `origin/tau-scale-experiment`.
- The `--use-height-scale` z-scale experiment (attempted fix #1 above) was
  fully implemented, tested, found non-viable, and reverted — it does not
  exist on the branch. This document is the record of why it was tried
  and why it doesn't work, in case it comes up again.

## Open questions / possible next steps

These are proposals, not committed work — flagged for discussion, not yet
built:

1. **Invert real MODEST data with NICOLE directly** (instead of relying
   on SPINOR for the independent B_LOS ground truth), so every stage of
   the pipeline — MURaM "truth," MUISCA's forward-synthesis comparison,
   and MODEST's own inversion — goes through one consistent code and one
   consistent set of physics assumptions. Not yet built: this needs
   NICOLE's inversion mode (`Mode=I` or similar, node/initial-guess
   configuration), which is new plumbing this bridge doesn't have yet
   (everything so far is forward-synthesis only), and inversions run
   substantially slower per pixel than synthesis.
2. **Investigate the residual line-depth discrepancy** (conclusion #2
   above). Candidate causes, roughly in order of suspected likelihood:
   the H⁻-only opacity approximation's missing terms (metals, He⁻, H₂⁻,
   scattering), the fixed 1 km/s microturbulence assumption used
   everywhere in this bridge, or a genuine difference between NICOLE's
   atomic line data and whatever produced MURaM's original Stokes. A
   direct way to test the first candidate: build an IDL-savefile writer
   so `Height scale=z` works (see "Revisiting attempted fix #1" above) —
   this would route τ₅₀₀ through NICOLE's own complete opacity physics
   instead of the H⁻-only approximation, isolating whether that
   incompleteness explains the residual discrepancy.
3. **If `stokes_*.npy` is ever replaced by NICOLE-regenerated Stokes** as
   the training/ground-truth source (extending today's 6-step, ~80
   pixels/step regeneration to full-frame, many-step coverage — full-frame
   is not practical without real parallelization, at ~1s/pixel × 230,400
   pixels/step), the loader must skip the `fixed_ic` continuum-
   normalization step for that data (see conclusion #5) rather than
   stacking it on top of NICOLE's own normalization.
