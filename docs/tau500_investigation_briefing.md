# Why NICOLE-Synthesized Stokes Profiles Didn't Match Observations, and What We're Doing About It

## Bottom line

The mismatch between NICOLE-synthesized Stokes profiles and both real
MODEST observations and MURaM ground truth is **not** caused by SPINOR
normalization, the hydrostatic-pressure boundary condition, or a
temperature bias in the model's predictions — we tested each of these
directly and ruled them out. The actual cause, confirmed with real
numbers across six independent MURaM simulation snapshots, is that
**MUISCA's pipeline has been using a different optical-depth scale than
the one NICOLE expects**: it uses the Rosseland-mean opacity (needed for
MURaM's own radiative-MHD energy balance) as the vertical coordinate, but
NICOLE's input format requires the monochromatic optical depth at 500 nm
— a physically different quantity. This matches exactly what was
suspected going in; we built an approximation for the correct scale,
validated it against ground truth, and it closes almost the entire
continuum-brightness discrepancy. A smaller, separate line-depth
discrepancy remains and is not yet explained. Given this, we're moving
forward with regenerating MURaM's training Stokes profiles through NICOLE
on the corrected depth scale, so the model is trained and verified under
one consistent, physically well-defined convention.

---

## 1. The original symptom

The MUISCA→NICOLE bridge takes an atmosphere (predicted by the trained
CNN, or MURaM's own ground truth) and re-synthesizes Stokes I and V from
it using NICOLE, as a consistency check: if the atmosphere is right, the
re-synthesized profile should match the profile that atmosphere was
supposed to have produced. It didn't, in two ways: the continuum level of
the synthesized profile was systematically dimmer than expected, and the
line core was shallower than expected. This showed up both on real Hinode/
MODEST observations and on MURaM simulation data.

## 2. Hypotheses we tested and ruled out

**Did SPINOR contaminate the observed MODEST Stokes profiles?** This was
the original concern — that SPINOR (used only to produce an independent
B_LOS estimate for MODEST) had normalized the observed intensities in a
way inconsistent with NICOLE's convention. We read the data-loading code
in full: the observed Stokes profiles are loaded from the FITS files with
zero rescaling; SPINOR is never in that code path at all, it only
produces the separate inverted-atmosphere file used elsewhere for
ground-truth B_LOS. Ruled out directly.

**Would locally renormalizing the observed continuum fix it?** We tried
dividing each observed pixel's intensity by its own local continuum value
before comparing. This made the match *worse* for most pixels, not
better — so a simple rescale wasn't the answer, and it pushed us to check
whether the mismatch exists independent of any MODEST-specific
calibration at all.

**Is it MODEST-specific?** We reran the same comparison entirely on MURaM
synthetic data — no SPINOR, no real-instrument calibration involved at
all. The same continuum deficit appeared, at the same magnitude (~9%
dim). This ruled out MODEST/SPINOR as the cause outright: whatever's
wrong is inside the MUISCA→NICOLE synthesis chain itself.

**Is it the pressure boundary condition?** By default, NICOLE doesn't use
the atmosphere's real gas pressure — it integrates pressure from
temperature via hydrostatic equilibrium, seeded by an essentially
arbitrary boundary value at the top of the atmosphere. This seemed like a
plausible source of error, so we fed NICOLE the true MURaM pressure
directly instead. The continuum mismatch barely moved (well under 1%
change). Ruled out as the primary cause, though we kept this capability
since real pressure is more physically correct regardless.

**Is the model predicting temperature too cool?** A cooler continuum-
forming layer would produce a dimmer continuum. We compared the model's
predicted temperature against MURaM's true temperature at the relevant
depth: the average difference was under 30 K out of ~6400 K (about
0.5%), with no consistent direction. Far too small to explain a 9%
brightness deficit. Ruled out.

**Is it just a different normalization convention?** NICOLE, by default,
normalizes its output to a fixed reference — the continuum intensity of
the HSRA reference atmosphere at disk center — rather than to the actual
local continuum of whatever atmosphere it was given. This looked
promising, since MURaM/MODEST data is normalized differently. We tested
switching NICOLE to local normalization instead. It made **no difference
to the line shape at all** — which makes sense in hindsight: changing a
normalization reference is just an overall rescaling, it can't change the
relative shape of a profile. This was a useful negative result, and it's
what led us to the real finding, described next.

## 3. The real cause: two different optical-depth scales

Once we controlled for normalization entirely (comparing both profiles on
their own local continuum), a cleaner signal emerged: NICOLE's line core
was consistently shallower than the true line core, in every single test
pixel. That's not a scaling artifact — it's a genuine difference in how
much the line profile departs from the continuum, which points to
something structural in how the atmosphere is being interpreted by
NICOLE.

Here's the physical issue we found. **MUISCA's entire pipeline — the
temperature/velocity/field grid the model is trained on and predicts —
uses the Rosseland-mean opacity to define its vertical coordinate.**
Rosseland-mean opacity is a frequency-*averaged* opacity, weighted toward
whichever wavelengths are most transparent; it's the correct quantity for
computing radiative energy transport in an MHD simulation like MURaM,
where what matters is the total (all-wavelengths) radiative flux, not any
single color. This was a deliberate, documented choice from the very
start of this project — the opacity table used throughout was explicitly
built as a Rosseland-mean table.

**NICOLE, on the other hand, requires the depth column to be the
optical depth at exactly 500 nm** (τ₅₀₀) — this is the standard
convention essentially every spectropolarimetric inversion code uses
(NICOLE, SIR, SPINOR all report atmospheric stratifications on this
scale), because it corresponds to something directly observable: the
depth at which continuum light near that wavelength is actually emitted.
This is stated unambiguously, multiple times, in NICOLE's own
documentation.

These are genuinely different physical quantities. At a given real height
in the atmosphere, the Rosseland-mean opacity and the monochromatic
opacity at 500 nm are, in general, different numbers — so "the layer
where Rosseland τ = 1" and "the layer where τ₅₀₀ = 1" are not the same
physical depth. Every time this bridge handed NICOLE a temperature
stratification labeled by our Rosseland-based τ grid, NICOLE took that
label at face value as τ₅₀₀ — silently misassigning each temperature to
the wrong physical depth relative to where the spectral line actually
forms. That's exactly the kind of error that would shallow a line core
without necessarily producing an obvious temperature bias at any single
nominal depth, which matches what we observed.

## 4. What we did about it: approximating the correct depth scale

To fix this, we needed to independently compute τ₅₀₀ for MURaM's
atmospheres — not just reuse the existing Rosseland table.

The dominant source of continuum opacity in the visible, for a star like
the Sun, is the **negative hydrogen ion (H⁻)**: a hydrogen atom carrying
a second, very loosely bound electron. It's the historically well-known
explanation (Wildt, 1939) for why the solar continuum is opaque at all in
visible light, since neutral H and He alone are nearly transparent there.
We built an opacity approximation using only the H⁻ contribution
(bound-free and free-free), taking the exact formulas from NICOLE's own
source code rather than a remembered textbook formula, to avoid
transcription errors. This deliberately excludes smaller contributors —
He⁻, molecular hydrogen ions, and photoionization of trace metals — which
matter more only in the coolest upper layers of the photosphere. We also
needed an estimate of electron pressure (H⁻ formation requires a free
electron, which mostly comes from easily-ionized trace metals like Na,
Mg, Fe — not from hydrogen itself, whose ionization potential is too high
at these temperatures) — we got this from an empirical fit to the real
HSRA reference solar atmosphere, rather than building a full multi-
species ionization solver from scratch.

We validated this approximation against literature values and the
existing Rosseland table before using it (same order of magnitude, same
qualitative trend with depth).

## 5. Results: this explains most of the discrepancy

We ran a controlled test that completely bypasses the trained model:
take MURaM's own true atmosphere, remap it onto our corrected τ₅₀₀ grid,
synthesize with NICOLE, and compare directly against MURaM's own
synthetic Stokes profiles. This isolates the depth-scale question from
any question about whether the model predicts well.

Across six independent MURaM simulation snapshots (five spanning the
training range, plus one entirely outside it):

| | before (Rosseland scale) | after (τ₅₀₀ correction) |
|---|---|---|
| continuum brightness ratio | ~0.86–0.91 (9–14% too dim) | ~1.00–1.09 (essentially correct) |
| line-core-depth error | consistently too shallow | reduced, but not eliminated |

The continuum fix is consistent and robust — not a fluke of one
simulation snapshot. The line-depth discrepancy shrank meaningfully but
is still present and still systematic (the line remains somewhat too
shallow in roughly three-quarters of tested pixels, across all six
snapshots, by a similar amount each time). Because this persists even on
ground truth with the correct depth scale, it isn't a model-prediction
error and it isn't the depth-scale mismatch — it's a separate, still-open
question, most likely related to the opacity terms our approximation
deliberately left out, or a difference in assumptions (microturbulence,
atomic line data) between NICOLE and whatever code originally produced
MURaM's synthetic Stokes profiles.

## 6. Decision and path forward

Given the size and robustness of the improvement, and that the remaining
gap is well-characterized rather than mysterious, we're proceeding with
this approach: **regenerating MURaM's Stokes profiles through NICOLE
itself, on the corrected τ₅₀₀ scale, and using that as the new,
internally consistent training and verification data for the PINN-MSCNN
model** — rather than continuing to rely on the original (undocumented-
provenance) synthesis code the training data currently comes from. This
doesn't just fix the depth-scale mismatch; it also means every stage of the
pipeline — training data, model verification, and (eventually) real-data
inversion — would use one code and one consistent set of physical
assumptions, instead of implicitly mixing several.

We want to be precise about what this claims and doesn't claim: it
resolves the confirmed, large, and reproducible continuum-brightness
discrepancy. It does not yet resolve the smaller residual line-depth
discrepancy, which remains an open question for further work (most
directly testable by extending NICOLE's opacity treatment beyond H⁻
alone, using NICOLE's own complete opacity physics rather than our
approximation — a follow-up we've scoped but not yet built).

Full technical detail, including exact numbers, code references, and the
dead ends we ruled out along the way, is in
`docs/stokes_synthesis_discrepancy_investigation.md`.
