# Four Defects That Invalidated Our Previous Results, and What They Turned Out to Be

## Bottom line

Following up on the τ₅₀₀ work, we retrained on the NICOLE-synthesized profiles and
went looking for why the magnetic field kept coming out wrong against MODEST. That
investigation turned up **four separate defects, each of which independently
invalidates conclusions we had been drawing**. Three of them were silent: they
produced numbers that looked plausible and were never flagged by any error.

The most consequential is that **the physics-informed loss terms were never training
anything**. A round-trip through NumPy in the denormalization step severed the
autograd graph, so the weak-field-approximation, Doppler and temperature terms
produced a value that was reported and summed into the total loss, but contributed
exactly zero gradient. This means every ablation we have run comparing "with physics"
against "no physics" was comparing two runs of the *same* objective. The differences
we reported between them were initialization noise — which was itself unconstrained,
because the random seed was never fixed either.

The second is that the Bz normalization was mis-scaled by a factor of ~3.5, which made
the network's magnetic-field output violently unstable: a prediction only half a
standard deviation past its training range produced tens of kilogauss. That is the
blow-up the ±10 kG clip had been added to contain, and it explains why ~13% of
predictions on the sunspot crops sat pinned at exactly that clip.

The third — a spectral-axis misalignment — we found first and have already confirmed
fixed, with a large measured improvement in line-of-sight velocity.

Separately, and not a defect: quantifying the training data's coverage shows that
**MURaM simply does not contain sunspot-strength fields**. No amount of code fixing
changes that, and it bounds what we can claim.

All four are fixed and committed. A retrain is running now. Until it finishes, no
previously reported ablation result should be treated as meaningful.

---

## 1. The spectral axis was offset by 0.078 Å

**Symptom.** Predicted line-of-sight velocities on MODEST were systematically offset
by −3.3 to −6.8 km/s depending on optical depth, while the same model reproduced
MURaM's velocities almost exactly (correlation 0.986, bias 0.03 km/s on the held-out
snapshot). A model that has learned velocity correctly but misreads it only on
observations points at the observations' calibration, not the network.

**Cause.** The wavelength axis was defined independently in two places. Training
profiles were resampled onto an axis built from assumed values
(CRVAL1 = 6302.0, CDELT1 = 0.0215, CRPIX1 = 57), giving [6300.7960, 6303.1825] Å,
while the MODEST observations were loaded on the axis in their own FITS header
(WLREF/WLMIN/WLMAX), giving [6300.8736, 6303.2576] Å. The 0.0776 Å disagreement is a
rigid shift of ~3.6 spectral pixels — a spurious 3.7 km/s Doppler shift for a model
trained on one axis and applied to the other.

Worth noting what this was *not*: we checked the atomic data first. The rest
wavelengths in the MODEST catalog paper (Castellanos Durán et al. 2024, Table 1)
match our NICOLE line list to 0.0004 Å, which is 0.019 km/s. The line list was never
the problem.

**Fix and result.** The axis is now derived from the FITS header in a single place,
with no fallback — a missing header raises rather than quietly substituting assumed
values. After retraining, V_LOS bias on the negative-polarity crop dropped from
−3.87 / −5.13 / −7.08 km/s to **−1.58 / −0.78 / −0.92 km/s** at log τ = −2.0 / −0.8 /
0.0, and correlation at log τ = 0 went from 0.037 to 0.645. Temperature improved as
well (correlation 0.566 → 0.867 at log τ = 0), which is expected since the whole input
profile was shifted, not just the velocity information.

---

## 2. The physics loss terms produced no gradient

This is the one with the widest consequences.

**Symptom.** In a 432-epoch run, once the WFA gate opened at epoch 41, the weighted
WFA term sat at ~14.7 against a mean-squared error of ~0.038 — dominating the reported
total loss by a factor of 384 — and then stayed flat between 1.45 and 1.52 for the
next 390 epochs while the MSE went on improving from 0.0383 to 0.0277. A term that
dominates the loss and never moves is not being optimized.

**Cause.** To convert normalized network outputs into physical units (Gauss, km/s,
Kelvin) for the physics comparisons, the code converted the predictions to NumPy, ran
the denormalization, and converted back with `torch.tensor(...)`. That last step
creates a new tensor with no gradient history: the chain back to the network weights
was cut. We confirmed this directly — the physics losses reported
`requires_grad=False`, and calling backward on one raised an error.

**What this invalidates.** Weight updates in `wfa_only` and `no_physics` were driven
by identical gradients. Any difference between those arms came from elsewhere. We
checked what "elsewhere" could be: there is no learning-rate scheduler, no early
stopping, and no validation-based checkpoint selection, and the train/validation split
is seeded. The only remaining sources were **weight initialization and batch order**,
neither of which was seeded (see §4). So the ablation was measuring initialization
variance and reporting it as "improvement over no physics".

**Fix.** Denormalization is now done with torch operations, verified numerically
identical to the NumPy path (exact for T and V_LOS, 1.2 × 10⁻⁷ for B_LOS, which is
float32 rounding). All three physics terms now produce non-zero gradients, and the
physics loss decreases under optimization instead of sitting flat.

---

## 3. The magnetic-field normalization was mis-scaled

**Symptom.** ~13% of B_LOS predictions on the sunspot crop sat at exactly ±10,000 G —
the clip that had been added to `denormalize()` to stop the field from running away.
Meanwhile the *median* residual was only +75 G. In other words, the typical pixel was
fine and the aggregate statistics were being destroyed by a minority of railed pixels.

**Cause.** B_LOS is normalized through a per-depth inverse-hyperbolic-sine transform,
`y = asinh(B / B₀) / σ`, with B₀ the 60th percentile of |B| at that depth. The running
mean and standard deviation, however, were accumulated during a first pass that used a
placeholder B₀ = 1, and were never recomputed once the real B₀ was known. The stored σ
therefore belonged to a different transform than the one it scaled, and came out ~3.5×
too large (3.89 instead of 1.18 at log τ = 0, on the current three-snapshot fit).

The consequence is severe because the inverse is exponential, `B = B₀ sinh(y σ)`. With
the inflated σ, the normalized training targets spanned only ±1.15 instead of the
intended ±4, and the mapping became extremely stiff:

| normalized output | implied field, before | implied field, after fix |
|---|---|---|
| 1.15 (old training maximum) | 1,255 G | 52 G |
| 1.50 | **4,899 G** | 82 G |
| 2.00 | **34,301 G** | 151 G |
| 3.91 (new training maximum) | 8 × 10⁷ G | 1,474 G (the real maximum) |

The network only had to overshoot its training range by about half a standard
deviation to emit tens of kilogauss. Sensitivity at y = 1.5 was ~19,000 G per unit of
normalized output; after the fix it is ~103 G, a factor of 185 gentler.

**Fix.** The moments are re-derived against the final B₀. This needs no extra pass over
the data — the per-depth histograms already computed for the percentiles carry the
distribution, and asinh is an odd function on a sign-symmetric field, so the mean is
zero by construction. Verified to reproduce a direct computation exactly. The
normalized target now has unit variance over ±3.9 -- its maximum, 3.91, decodes to
1,474 G, which is exactly the strongest field in the training data -- and the 10 kG clip
is only reached
about three standard deviations outside the training range rather than half a standard
deviation inside it.

---

## 4. Training was not reproducible

`torch.manual_seed` was never called. The ablation variations run sequentially in one
process, so each inherited whatever random state the previous left behind: consecutive
models differed by up to 0.79 per weight, and the data loader shuffled differently too.

This compounds §2 exactly where it hurts. With the physics contributing no gradient,
initialization was the *only* thing distinguishing the arms — and it was uncontrolled.
The summary line the ablation prints, "Improvement over no physics: X%", was reporting
that noise.

Seeding is now explicit and configurable, applied before the model is constructed in
both training and fine-tuning. Verified: two models built with the same seed are
bit-identical; with different seeds they differ as expected.

---

## 5. A separate finding: the training data does not span sunspot fields

While investigating the magnetic-field errors we quantified how much support the
training set actually provides. Counting *distinct* pixels in MURaM snapshots
110/120/130 at the photosphere:

| \|B_LOS\| above | distinct training pixels | assessment |
|---|---|---|
| 100 G | 78,445 | ample |
| 250 G | ~25,000 | ample |
| 352 G | 6,945 | ample |
| 500 G | 2,701 | marginal |
| 950 G | 147 | memorization territory |
| 1,500 G | **0** | absent |

The Hinode sunspot crop, for comparison, has a 90th percentile of 245 G, a 99th of
894 G, and reaches 3,711 G.

So the bulk of the field of view is well supported, and the model should be able to
learn it. The umbral core is not: above ~1 kG there are too few distinct examples to
learn from, and above 1.5 kG the simulation contains nothing at all. This is a
property of the data, not of the code — no reweighting or retraining creates fields a
quiet-Sun simulation does not have.

We tested the reweighting hypothesis directly rather than assuming it. Class
imbalance *is* real (strong fields are ~1% of training pixels but far more common in
the crop), and the repository's Bz-balanced fine-tuning exists for exactly that. We
found and fixed a defect that had been collapsing the balanced selection to 96 pixels
out of 691,200, reworked the binning so it keeps 237,026, and ran it. **It did not
improve the magnetic-field error** — bias at log τ = −0.8 went from −937 to −1112 G
with correlation unchanged. That negative result is informative: the problem was not
imbalance.

The analysis now writes a per-field-strength breakdown alongside the usual metrics, so
this range-of-applicability boundary is visible directly in the results rather than
hidden inside an aggregate dominated by weak-field pixels.

---

## 6. Where things stand

Fixed and committed: the spectral axis, the physics gradient, the Bz normalization
scale, seeding, the balancing collapse, and a set of defects that had made the
fine-tuning script unrunnable (it failed at import). Training runs are now tracked in
MLflow so configurations, metric curves and models are recorded per run.

Running now: a full retrain with all four fixes active simultaneously. This is the
first ablation that measures what it claims to measure — the arms start from identical
weights, differ only in their physics terms, and those terms now actually train.

Open questions we expect this run to settle:

1. **Does the physics term help?** Genuinely unknown. Every prior answer was an
   artifact.
2. **Is the WFA weight right?** It is currently 10. Measured on real data, the WFA
   gradient is ~460× the MSE gradient at the point where the gate opens, and the ratio
   worsens as the MSE converges — because the physics losses are computed in physical
   units and pass through the exponential denormalization, while the MSE lives in
   normalized space. If the MSE degrades when the gate opens, the weight needs to come
   down by one to two orders of magnitude.
3. **How far does the magnetic-field error actually close?** The clipping is gone by
   construction, but the data-coverage limit in §5 is not something the retrain
   addresses.

Still open and not addressed by any of this: the validation loss is currently not a
usable signal — it includes the physics terms in physical units while training has
them gated off, and swings between 68 and 506 between consecutive epochs. That makes
it unsuitable for early stopping, for learning-rate scheduling, and for the
"best variation" comparison the ablation prints. Fixing it is a prerequisite for all
three.
