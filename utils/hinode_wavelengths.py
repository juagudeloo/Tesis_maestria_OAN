#!/usr/bin/env python3
"""Single source of truth for the Hinode/SOT-SP wavelength axis.

Every code path that needs the observed spectral grid -- loading real MODEST
observations, resampling NICOLE-synthesized training profiles onto the
instrument grid, or synthesizing from MUISCA predictions for comparison
against observations -- MUST get it from here, so the training and
observation sides can never drift apart again.

Why this module exists
----------------------
The axis used to live in two independent places: a hardcoded
`{"CRVAL1": 6302.0, "CDELT1": 0.0215, "CRPIX1": 57}` dict on the
training/synthesis side (yielding [6300.7960, 6303.1825] A) and the
MODEST FITS header on the observation side (yielding
[6300.8736, 6303.2576] A). They disagreed by +0.0776 A -- a rigid shift of
~3.6 spectral pixels, i.e. a spurious ~3.7 km/s blueshift for a model
trained on one grid and applied to the other. That fully accounted for the
bulk of the observed V_LOS bias against MODEST's SPINOR-2D inversions
(measured -3.32 km/s at log tau = -2.0).

The FITS header is authoritative because it is physically anchored: WLREF
is the Fe I 6302.4936 A laboratory rest wavelength (identical to Table 1 of
Castellanos Duran et al. 2024, A&A 687, A218 -- the MODEST catalog paper),
so a line at rest sits at exactly zero offset. The old hardcoded values
were a generic guess, not tied to the line.

There is deliberately NO fallback. A missing header raises, because a silent
fallback to assumed values is precisely what caused the original bug.
"""

from pathlib import Path
from typing import Union

import numpy as np
from astropy.io import fits

# The MODEST scan whose header defines the axis. Matches ModestData's default.
DEFAULT_MODEST_DIR = Path(
    "/scratchsan/observatorio/juagudeloo/MUISCA/data/hinode-MODEST/INV_560_AR11967/"
)
DEFAULT_PROFS_FILENAME = "inverted_profs.1.fits"

# Hinode/SOT-SP observed Stokes profiles per pixel (spectral points).
N_WL_OBSERVED = 112
# MODEST's inverted-profile files are sampled more finely than the observations.
N_WL_INVERTED = 250


def read_hinode_wavelength_metadata(
    modest_dir: Union[Path, str] = DEFAULT_MODEST_DIR,
    filename: str = DEFAULT_PROFS_FILENAME,
) -> tuple[float, float, float]:
    """Return (wl_min_abs, wl_max_abs, wl_ref) in Angstrom from the MODEST FITS header.

    Raises FileNotFoundError / KeyError / ValueError rather than falling back to
    assumed values -- see the module docstring.
    """
    path = Path(modest_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot derive the Hinode wavelength axis: {path} not found. "
            "The axis is read from the MODEST FITS header (WLREF/WLMIN/WLMAX) and "
            "has no fallback by design -- a hardcoded fallback previously drifted "
            "from the observed axis by 0.0776 A (~3.7 km/s). Point modest_dir at a "
            "MODEST scan directory containing this file."
        )

    with fits.open(path) as hdul:
        hdr = hdul[0].header
    missing = [k for k in ("WLREF", "WLMIN", "WLMAX") if k not in hdr]
    if missing:
        raise KeyError(
            f"{path} is missing required wavelength keyword(s) {missing}. "
            "Cannot derive the Hinode wavelength axis without them."
        )

    wl_ref = float(hdr["WLREF"])
    wl_min_abs = wl_ref + float(hdr["WLMIN"])
    wl_max_abs = wl_ref + float(hdr["WLMAX"])

    if not (np.isfinite(wl_ref) and np.isfinite(wl_min_abs) and np.isfinite(wl_max_abs)):
        raise ValueError(f"{path} has non-finite WLREF/WLMIN/WLMAX: {wl_ref}, {wl_min_abs}, {wl_max_abs}")
    if wl_max_abs <= wl_min_abs:
        raise ValueError(
            f"{path} gives a non-increasing wavelength axis "
            f"[{wl_min_abs}, {wl_max_abs}] (WLREF={wl_ref})"
        )
    return wl_min_abs, wl_max_abs, wl_ref


def hinode_wavelength_grid(
    n_wl: int = N_WL_OBSERVED,
    modest_dir: Union[Path, str] = DEFAULT_MODEST_DIR,
    filename: str = DEFAULT_PROFS_FILENAME,
) -> np.ndarray:
    """Return the (n_wl,) Hinode/SOT-SP wavelength grid in Angstrom.

    Endpoints come from the MODEST FITS header; points are evenly spaced
    between them (the same linspace convention the observation loader has
    always used, so observed profiles are unchanged by this refactor).
    """
    wl_min_abs, wl_max_abs, _ = read_hinode_wavelength_metadata(modest_dir, filename)
    return np.linspace(wl_min_abs, wl_max_abs, int(n_wl), dtype=np.float64)


def hinode_grid_first_and_step_mA(
    n_wl: int = N_WL_OBSERVED,
    modest_dir: Union[Path, str] = DEFAULT_MODEST_DIR,
    filename: str = DEFAULT_PROFS_FILENAME,
) -> tuple[float, float]:
    """Return (first_wavelength_A, step_mA) for the FITS-derived grid.

    For NICOLE, which is configured with a first wavelength plus a constant
    step rather than an explicit array.
    """
    grid = hinode_wavelength_grid(n_wl, modest_dir, filename)
    step_mA = float((grid[-1] - grid[0]) / (grid.size - 1) * 1000.0)
    return float(grid[0]), step_mA
