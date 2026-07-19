# NICOLE Assets (Synthesis Bridge)

Files needed by the MUISCA -> NICOLE forward-synthesis bridge. These are checked
in so the synthesis driver is hermetic and reproducible — it does not depend on
the contents of `/scratchsan/observatorio/juagudeloo/NICOLE/test/syn1/` at runtime.

## Files

- **`LINES`** — Fe I 6301.5 and Fe I 6302.5 line parameters (Barklem
  collisions, LTE). Adapted from `NICOLE/run/LINES` with labels renamed to
  `[FeI 6301.5]` and `[FeI 6302.5]` to match the references in
  `NICOLE.input.template`.

- **`NICOLE.input.template`** — synthesis-mode configuration with placeholders
  substituted at runtime by `utils.synthesis.NicoleRunner.prepare_workdir()`:
  `{NICOLE_COMMAND}`, `{INPUT_MODEL}`, `{OUTPUT_PROFILE}`, `{OUTPUT_MODEL}`,
  `{WL_FIRST}`, `{WL_STEP_MA}`, `{N_WL}`. `Region 1` defaults are filled with
  the MODEST/Hinode SP grid (6300.796 Å + 21.5 mÅ × 112).

## Updating

If NICOLE upgrades its line database or input syntax, refresh these files from
the upstream source (`/scratchsan/observatorio/juagudeloo/NICOLE/run/LINES` and
`NICOLE/test/syn1/NICOLE.input`) and re-run the single-pixel verification in
`scripts/synthesis/run_nicole_synthesis.py`.
