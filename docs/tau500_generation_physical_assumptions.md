# Physical Assumptions in the τ₅₀₀ Stokes Profile Generation

## Bottom line

The training Stokes profiles regenerated through NICOLE on the τ₅₀₀ scale
(issue #5) are not a first-principles forward synthesis of MURaM's
atmospheres — they rest on a specific, deliberately scoped chain of
physical approximations, in continuum opacity, line formation, and the
atmospheric structure handed to NICOLE. None of these were chosen
carelessly (most are documented, with reasoning, at the point in the code
where they're made), but they are approximations, and it matters for
interpreting the trained model's results — and for anyone trying to
reproduce this dataset — to have them listed in one place rather than
scattered across module docstrings. This document is that list, ordered
roughly by how much each assumption is expected to matter physically.

---

## 1. Continuum opacity for the τ₅₀₀ grid

Source: `utils/tau500_opacity.py`, `scripts/synthesis/tau500_multi_step_regen.py::compute_logtau500`

- **Only H⁻ bound-free + free-free opacity is included.** He⁻, H₂⁻,
  metals, and scattering are all omitted. The opacity formula itself is
  not a textbook approximation — it's ported line-for-line from NICOLE's
  own `wittmann_opac.f90` (lines 209–251), evaluated at a fixed
  λ = 5000 Å — so it is exactly the H⁻ term NICOLE itself would compute.
  The *omission* of the other terms is the approximation, and the module's
  own docstring notes this matters most in the coolest, uppermost layers.
  This is the leading suspect for the smaller, still-unresolved line-core
  depth discrepancy documented in
  `docs/stokes_synthesis_discrepancy_investigation.md`.
- **Electron pressure Pe(T) is not solved self-consistently.** It comes
  from an empirical fit to the real HSRA reference atmosphere's
  deep/monotonic branch (log τ₅₀₀ ∈ [−4.0, +1.4], T ∈ [4170, 9560] K —
  HSRA's temperature-minimum/chromospheric branch is excluded because
  it's non-monotonic in T and outside MURaM's box anyway). This sidesteps
  a full multi-species ionization-equilibrium solve, at the cost of
  assuming HSRA's Pe(T) relation is representative of every MURaM pixel's
  actual conditions, rather than being computed from that pixel's own
  density.
- **H ionization and H⁻ formation are each a single Saha equation, pure
  hydrogen.** Standard atomic constants (χ_H = 13.5984 eV,
  χ_H⁻ = 0.754 eV), no coupling to metal ionization.
- **P_Htot = Pgas − Pe treats everything that isn't an electron as
  hydrogen** — helium (~10% of particles by number) and trace metals are
  folded in as if they were hydrogen for the final mass-opacity
  conversion.
- **τ₅₀₀ itself is a trapezoidal integral of κρ along MURaM's own fixed
  geometric grid** (dz = 10 km, same grid used for the original
  Rosseland-τ pipeline), not an independently-resolved depth grid.

## 2. Line formation inside NICOLE

Source: `data/nicole_assets/NICOLE.input.template`, `data/nicole_assets/LINES`

- **LTE, explicitly** (`Mode='LTE'`) for both Fe I 6301.5 and Fe I 6302.5
  — no NLTE population departures.
- **Collisional (van der Waals) broadening uses fixed Barklem/ABO
  constants** (damping σ, α tabulated per line) rather than being
  computed from each point's local density.
- **Microturbulence is fixed at 1 km/s** (`v_mic_cms = 1.0e5`, NICOLE's
  own default) — not derived from MURaM's turbulent velocity field.
- **Macroturbulence = 0 and stray light = 0.**
- **Disk-center geometry**: `Heliocentric angle = 1.0` (μ = 1). No limb
  darkening, no foreshortening.
- **Default (solar) elemental abundances** — not MURaM's own composition.
- **NICOLE's default radiative-transfer solver and depth-grid handling**
  (`Formal solution method=0`, `Optimize Grid=0`) — the fixed τ₅₀₀ grid is
  used as given, with no adaptive remeshing.

## 3. Atmospheric structure and magnetic field passed to NICOLE

Source: `utils/synthesis.py::NicoleRunner.run_cube`, `utils/model_prof_tools.py::write_binary_model_cube`

- **Hydrostatic equilibrium is imposed** (`Impose hydrostatic
  equilibrium=Y`): NICOLE integrates its own gas pressure/density from T
  plus a seed boundary condition, rather than using MURaM's true
  (magnetized, dynamic, non-hydrostatic) gas pressure. This generation run
  does not use the `--add-gt-pressure` path (which supplies MURaM's real
  Pgas instead) — that path exists and was validated earlier in the
  investigation, but was found not to change the continuum/line-depth
  mismatch enough to justify the added complexity for this dataset.
- **Only the line-of-sight components of the field and velocity are
  used.** B_LOS (MURaM's Bz) and V_LOS (MURaM's Vz) are the only vector
  quantities passed to NICOLE; the transverse field components (Bx, By)
  and the full vector field/velocity slots in the binary model cube are
  left at zero. In practice this means the synthesized linear
  polarization (Stokes Q, U) reflects a purely longitudinal field
  geometry, not MURaM's real 3D field vector — any genuine transverse-field
  contribution to Q/U (or to the effective Zeeman splitting under field
  inclination) is absent from this dataset by construction.

## 4. Interpolation onto the fixed τ₅₀₀ grid

Source: `scripts/synthesis/generate_tau500_stokes.py`

- T, V_LOS, and B_LOS are remapped per-pixel, by linear interpolation
  (`scipy.interpolate.interp1d`), from MURaM's native geometric-height
  grid onto the shared 45-level log(τ₅₀₀) grid (−3.0 to +1.4, step 0.1).
  Where a pixel's actual τ₅₀₀ range doesn't cover the full grid, values
  are **linearly extrapolated** (`bounds_error=False,
  fill_value="extrapolate"`) rather than left undefined.

## 5. Geometric truncation upstream of all of this

Source: `utils.muram_data.MhdData.load_step` (via `TrainingConfig.z_max`)

- The MURaM cube is trimmed to the first 250 of 256 vertical layers
  before any opacity, τ₅₀₀, or synthesis calculation — implicitly
  assuming the topmost ~6 layers contribute negligibly to Fe I
  6301.5/6302.5 formation.

## 6. A numerical artifact, not a physical assumption, but worth flagging

Source: `utils/synthesis.py::NicoleRunner.run_cube`

- Every synthesized chunk is prepended with a synthetic "warm-up" pixel
  (a copy of the chunk's first real pixel, with T scaled ×1.05) purely to
  force NICOLE's internal cube-mode state into a reproducible
  "different-predecessor" regime. This is bookkeeping to make chunked,
  parallelized generation reproducible and chunk-independent — without it,
  a pixel synthesized as the first pixel of a cube differs by ~3–9×10⁻³
  from the same pixel synthesized elsewhere in a cube. It has no physical
  content of its own and is discarded from every output.

## 7. Spectral window

Source: `scripts/synthesis/generate_tau500_stokes.py`

- Both lines are synthesized together in one window, 6300.50–6303.49 Å,
  10 mÅ step, 300 points — chosen to exactly match the original
  `stokes_*.npy` convention, so the existing LSF-convolution and
  resample-to-Hinode-grid pipeline downstream applies unchanged.

---

## Open question this doesn't resolve

Even after correcting the optical-depth scale (the core issue this
generation run addresses — see
`docs/stokes_synthesis_discrepancy_investigation.md`), a smaller
line-core-depth discrepancy remains between NICOLE-resynthesized and true
MURaM Stokes profiles on ground truth. The leading candidates are the
opacity simplifications in §1 (H⁻-only, HSRA-derived Pe) or a genuine
difference between NICOLE's physics and whatever code originally produced
MURaM's `stokes_*.npy`. This has not been isolated further and should be
treated as an open uncertainty in this dataset, not a resolved one.
