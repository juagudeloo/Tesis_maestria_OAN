from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fft2, fftshift, ifft2
from scipy.ndimage import uniform_filter1d

from utils.cache_manage import ModestDataCache


def transform_modest_stokes_profiles(
	stokes: Dict[str, np.ndarray],
	wavelength_angstrom: np.ndarray,
	shift_positions: float = 0.0,
	i_scale: float = 1.0,
	v_scale: float = 1.0,
	invert_direction: bool = False,
	edge_extrapolation_points: int = 4,
) -> Dict[str, np.ndarray]:
	"""Apply optional spectral transformations to MODEST Stokes I/V profiles.

	Transform order:
	1) Optional spectral-axis inversion.
	2) Optional spectral shift in sample positions with edge extrapolation.
	3) Channel scaling (I and V independently).
	"""
	if "I" not in stokes or "V" not in stokes:
		raise KeyError("stokes dictionary must contain 'I' and 'V' keys")

	wl = np.asarray(wavelength_angstrom, dtype=np.float64)
	if wl.ndim != 1:
		raise ValueError(f"wavelength_angstrom must be 1D, got shape {wl.shape}")

	I = np.asarray(stokes["I"], dtype=np.float32)
	V = np.asarray(stokes["V"], dtype=np.float32)
	if I.shape != V.shape:
		raise ValueError(f"Stokes I/V shape mismatch: I{I.shape} vs V{V.shape}")
	if I.ndim != 3:
		raise ValueError(f"Expected Stokes cubes with shape (ny, nx, nwl), got {I.shape}")
	if I.shape[2] != wl.size:
		raise ValueError(
			f"Wavelength size mismatch: nwl={I.shape[2]} in Stokes but {wl.size} wavelength points"
		)

	# Work on copies to keep the input dictionary unchanged.
	I_out = np.array(I, copy=True)
	V_out = np.array(V, copy=True)

	if invert_direction:
		I_out = I_out[:, :, ::-1]
		V_out = V_out[:, :, ::-1]

	def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
		if x.size < 2:
			return 0.0, float(y[0]) if y.size > 0 else 0.0
		a, b = np.polyfit(x, y, 1)
		return float(a), float(b)

	def _shift_cube(cube: np.ndarray, delta_positions: float) -> np.ndarray:
		if abs(delta_positions) < 1e-14:
			return cube

		shifted = np.array(cube, copy=True)
		nwl = shifted.shape[2]
		grid = np.arange(nwl, dtype=np.float64)
		query_grid = grid - float(delta_positions)
		grid_min = 0.0
		grid_max = float(nwl - 1)

		ny, nx, _ = shifted.shape
		for iy in range(ny):
			for ix in range(nx):
				spec = shifted[iy, ix, :]
				finite = np.isfinite(spec)
				if np.count_nonzero(finite) < 2:
					continue

				x = grid[finite]
				y = spec[finite].astype(np.float64)

				if x.size < 2:
					continue

				y_interp = np.interp(query_grid, x, y)

				k = max(2, min(int(edge_extrapolation_points), x.size))
				a_left, b_left = _fit_line(x[:k], y[:k])
				a_right, b_right = _fit_line(x[-k:], y[-k:])

				left_mask = query_grid < grid_min
				right_mask = query_grid > grid_max
				if np.any(left_mask):
					y_interp[left_mask] = a_left * query_grid[left_mask] + b_left
				if np.any(right_mask):
					y_interp[right_mask] = a_right * query_grid[right_mask] + b_right

				shifted[iy, ix, :] = y_interp.astype(np.float32)

		return shifted

	I_out = _shift_cube(I_out, shift_positions)
	V_out = _shift_cube(V_out, shift_positions)

	I_out = I_out * float(i_scale)
	V_out = V_out * float(v_scale)

	return {"I": I_out.astype(np.float32, copy=False), "V": V_out.astype(np.float32, copy=False)}


class ModestData:
	"""Load, process, and visualize MODEST Hinode/SP data products.

	The class follows the end-to-end workflow captured in the
	``4-modest_data.ipynb`` notebook: loading continuum and Stokes data,
	performing PSF-based Wiener deconvolution, optional smoothing, computing
	circular polarization masks, loading inverted profiles and SPINOR 2D
	atmospheres, and producing a handful of diagnostic plots.
	"""

	def __init__(
		self,
		modest_dir: Union[Path, str] = "/scratchsan/observatorio/juagudeloo/MUISCA/data/hinode-MODEST/INV_560_AR11967/",
		psf_path: Union[Path, str] = "/scratchsan/observatorio/juagudeloo/MUISCA/data/hinode-MODEST/PSFs/hinode_psf_bin.0.16.fits",
		circular_polarization_threshold: float = 1e-2,
		stokes_v_multiplier: float = -1.0,
	) -> None:
		self.modest_dir = Path(modest_dir)
		self.psf_path = Path(psf_path)
		self.circular_polarization_threshold = circular_polarization_threshold
		self.stokes_v_multiplier = float(stokes_v_multiplier)
		if not np.isfinite(self.stokes_v_multiplier) or self.stokes_v_multiplier == 0.0:
			raise ValueError("stokes_v_multiplier must be finite and non-zero")

		# Data containers
		self.continuum: Optional[np.ndarray] = None
		self.obs_stokes: Optional[Dict[str, np.ndarray]] = None  # (ny, nx, nwl)
		self.upsampled_stokes: Optional[Dict[str, np.ndarray]] = None
		self.deconvolved_stokes: Optional[Dict[str, np.ndarray]] = None
		self.smoothed_stokes: Optional[Dict[str, np.ndarray]] = None
		self.prediction_stokes: Optional[Dict[str, np.ndarray]] = None
		self.inverted_profs: Optional[Dict[str, np.ndarray]] = None
		self.inverted_atmos: Optional[np.ndarray] = None  # (n_params, ny, nx)
		self.spinor_atm: Optional[Dict[str, Dict[float, np.ndarray]]] = None

		# Wavelength axes
		self.wl: Optional[u.Quantity] = None
		self.wl_inv: Optional[u.Quantity] = None

		# Diagnostics and masking
		self.circ_pol: Optional[np.ndarray] = None
		self.mask: Optional[np.ndarray] = None
		self.sample_pixels: Optional[Dict[str, Dict[str, float]]] = None

	# ------------------------------------------------------------------
	# Loading helpers
	# ------------------------------------------------------------------
	def load_continuum(self, filename: str = "continuum.fits") -> np.ndarray:
		path = self.modest_dir / filename
		with fits.open(path) as hdul:
			self.continuum = hdul[0].data
		return self.continuum

	def load_obs_stokes(self, filename: str = "inverted_obs.1.fits") -> Dict[str, np.ndarray]:
		path = self.modest_dir / filename
		with fits.open(path) as hdul:
			cube = hdul[0].data  # (ny, nx, 4, nwl)
		# Keep only I (0) and V (1) channels
		self.obs_stokes = {
			"I": cube[:, :, 0, :].astype(np.float32),
			"V": (cube[:, :, 1, :].astype(np.float32) * self.stokes_v_multiplier),
		}
		return self.obs_stokes

	def load_inverted_profs(self, filename: str = "inverted_profs.1.fits") -> Dict[str, np.ndarray]:
		path = self.modest_dir / filename
		with fits.open(path) as hdul:
			cube = hdul[0].data  # (ny, nx, 4, nwl_inv)
		self.inverted_profs = {
			"I": cube[:, :, 0, :].astype(np.float32),
			"V": (cube[:, :, 1, :].astype(np.float32) * self.stokes_v_multiplier),
		}
		return self.inverted_profs

	def _read_wavelength_metadata_from_inverted_profs(
		self,
		filename: str = "inverted_profs.1.fits",
	) -> Optional[Tuple[float, float, float]]:
		"""Return (wl_min_abs, wl_max_abs, wl_ref) in Angstrom from FITS header if present."""
		path = self.modest_dir / filename
		if not path.exists():
			return None
		try:
			with fits.open(path) as hdul:
				hdr = hdul[0].header
			wl_ref = float(hdr["WLREF"])
			wl_min = float(hdr["WLMIN"])
			wl_max = float(hdr["WLMAX"])
		except Exception:
			return None

		if not (np.isfinite(wl_ref) and np.isfinite(wl_min) and np.isfinite(wl_max)):
			return None
		wl_min_abs = wl_ref + wl_min
		wl_max_abs = wl_ref + wl_max
		if not (np.isfinite(wl_min_abs) and np.isfinite(wl_max_abs) and wl_max_abs > wl_min_abs):
			return None
		return wl_min_abs, wl_max_abs, wl_ref

	def load_inverted_atmos(self, filename: str = "inverted_atmos.fits") -> np.ndarray:
		path = self.modest_dir / filename
		with fits.open(path) as hdul:
			self.inverted_atmos = hdul[0].data.astype(np.float32)
		return self.inverted_atmos

	def _load_psf(self) -> np.ndarray:
		with fits.open(self.psf_path) as hdul:
			psf = hdul[0].data
		psf = np.ascontiguousarray(psf, dtype=np.float32)
		psf /= np.sum(psf)
		return psf

	def compute_wavelength_arrays(self) -> Tuple[u.Quantity, u.Quantity]:
		obs_nwl = 112
		inv_nwl = 250
		# Use MODEST FITS metadata when available to avoid wavelength-offset bias.
		meta = self._read_wavelength_metadata_from_inverted_profs()
		if meta is not None:
			wl_min_abs, wl_max_abs, wl_ref = meta
			wl_obs = np.linspace(wl_min_abs, wl_max_abs, obs_nwl, dtype=np.float64)
			wl_inv = np.linspace(wl_min_abs, wl_max_abs, inv_nwl, dtype=np.float64)
			print(
				f"MODEST wavelength axis from FITS header: [{wl_min_abs:.4f}, {wl_max_abs:.4f}] Å (WLREF={wl_ref:.4f})"
			)
		else:
			crval1 = 6302.0
			cdelt1 = 0.0215
			crpix1 = 57
			wl_obs = crval1 + (np.arange(1, obs_nwl + 1) - crpix1) * cdelt1
			spectral_range = wl_obs[-1] - wl_obs[0]
			cdelt1_inv = spectral_range / (inv_nwl - 1)
			crpix1_inv = 125
			wl_inv = crval1 + (np.arange(1, inv_nwl + 1) - crpix1_inv) * cdelt1_inv
			print(
				"MODEST wavelength metadata missing; using fallback Hinode grid assumptions."
			)

		self.wl = wl_obs * u.Angstrom
		self.wl_inv = wl_inv * u.Angstrom
		return self.wl, self.wl_inv

	# ------------------------------------------------------------------
	# Upsampling for fast-mode observations
	# ------------------------------------------------------------------
	def upsample_stokes(
		self, 
		upsampling_factor: int = 2,
		aperture_m: float = 0.50,
		pixel_scale_arcsec: float = 0.32,
		lambda_angstrom: float = 6302.0,
		seed: Optional[int] = None
	) -> Dict[str, np.ndarray]:
		"""Upsample observed Stokes data using FFT-based method with noise injection."""
		if self.obs_stokes is None:
			raise ValueError("Load observed Stokes before upsampling")

		if seed is not None:
			np.random.seed(seed)

		ny, nx, nwl = self.obs_stokes["I"].shape
		f = upsampling_factor

		# Calculate diffraction limit
		lambda_m = lambda_angstrom * 1e-10
		scale_rad = pixel_scale_arcsec / (np.degrees(1) * 3600)
		lambda_over_d = lambda_m / aperture_m
		q_number = lambda_over_d / scale_rad

		# Dimensions
		dd = np.array([2 * ny, 2 * nx])
		dh = dd // 2
		dn = (dd * f) // 2

		print(f"Upsampling from ({ny}, {nx}) to ({dn[0]}, {dn[1]})")

		upsampled_stokes = {
			"I": np.zeros((dn[0], dn[1], nwl), dtype=np.float32),
			"V": np.zeros((dn[0], dn[1], nwl), dtype=np.float32)
		}

		def ellipse_mask(shape, q):
			lfx = shape[1] / q
			lfy = shape[0] / q
			y, x = np.ogrid[:shape[0], :shape[1]]
			x_norm = (x - shape[1] // 2) / lfx
			y_norm = (y - shape[0] // 2) / lfy
			return np.sqrt(x_norm**2 + y_norm**2)

		aperture = ellipse_mask(dd, q_number)
		idx_outside = aperture > 1.001

		for key in ["I", "V"]:
			print(f"Upsampling Stokes {key}...")
			for iw in range(nwl):
				if iw % 20 == 0:
					print(f"  Wavelength {iw}/{nwl}")

				# Step 1: Pad by mirroring
				im = np.zeros(dd, dtype=np.float32)
				im[:dh[0], :dh[1]] = self.obs_stokes[key][:, :, iw]
				im[:dh[0], dh[1]:] = np.fliplr(self.obs_stokes[key][:, :, iw])
				im[dh[0]:, :dh[1]] = np.flipud(self.obs_stokes[key][:, :, iw])
				im[dh[0]:, dh[1]:] = np.flipud(np.fliplr(self.obs_stokes[key][:, :, iw]))

				# Step 2: FFT and estimate noise from amplitude spectrum
				fc = fftshift(fft2(im))
				power = np.abs(fc)
				if np.sum(idx_outside) > 10:
					noise_power = np.median(power[idx_outside])
				else:
					noise_power = np.percentile(power, 85)

				# Step 3: Create noise field at padded size
				noise = np.random.randn(dd[0], dd[1]).astype(np.float32) * noise_power
				noise_fft = fftshift(fft2(noise))

				# Step 4: Zero-pad noise FFT to upsampled size
				dfc_shape = (dn[0] * 2, dn[1] * 2)
				noise_fft_upsampled = np.zeros(dfc_shape, dtype=np.complex64)
				y_start = (dfc_shape[0] - dd[0]) // 2
				y_end = y_start + dd[0]
				x_start = (dfc_shape[1] - dd[1]) // 2
				x_end = x_start + dd[1]
				noise_fft_upsampled[y_start:y_end, x_start:x_end] = noise_fft

				# Step 5: Replace center with real data
				dfc = noise_fft_upsampled.copy()
				dfc[y_start:y_end, x_start:x_end] = fc

				# Step 6: Inverse FFT
				upsampled = np.real(ifft2(fftshift(dfc)))
				result = upsampled[:dn[0], :dn[1]]

				# Step 7: Normalize to preserve original mean intensity
				# The upsampling and FFT operations distribute energy across more pixels
				# This renormalization ensures the mean intensity is preserved
				original_mean = np.nanmean(self.obs_stokes[key][:, :, iw])
				result_mean = np.nanmean(result)
				if result_mean != 0 and not np.isnan(result_mean):
					result = result * (original_mean / result_mean)

				upsampled_stokes[key][:, :, iw] = result

		return upsampled_stokes

	# ------------------------------------------------------------------
	# Processing: deconvolution, smoothing, polarization, masking
	# ------------------------------------------------------------------
	@staticmethod
	def _wiener_deconv(image: np.ndarray, psf_kernel: np.ndarray, noise_level: float) -> np.ndarray:
		psf_padded = np.zeros_like(image)
		pc = (psf_kernel.shape[0] // 2, psf_kernel.shape[1] // 2)
		ic = (image.shape[0] // 2, image.shape[1] // 2)
		y0, y1 = ic[0] - pc[0], ic[0] - pc[0] + psf_kernel.shape[0]
		x0, x1 = ic[1] - pc[1], ic[1] - pc[1] + psf_kernel.shape[1]
		psf_padded[y0:y1, x0:x1] = psf_kernel
		psf_padded = fftshift(psf_padded)
		img_fft = fft2(image)
		psf_fft = fft2(psf_padded)
		wiener = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + noise_level)
		return np.real(ifft2(img_fft * wiener))

	def deconvolve_stokes(self, noise_level: float = 0.01, use_upsampled: bool = False) -> Dict[str, np.ndarray]:
		"""Apply Wiener deconvolution to Stokes data.

		Parameters
		----------
		noise_level : float
			Noise parameter for Wiener filter
		use_upsampled : bool
			If True, use upsampled Stokes as input (must call upsample_stokes first)
		"""
		if use_upsampled:
			if not hasattr(self, 'upsampled_stokes') or self.upsampled_stokes is None:
				raise ValueError("Call upsample_stokes() before deconvolving upsampled data")
			source = self.upsampled_stokes
		else:
			if self.obs_stokes is None:
				raise ValueError("Load observed Stokes before deconvolution")
			source = self.obs_stokes

		psf = self._load_psf()
		ny, nx, n_wl = source["I"].shape
		self.deconvolved_stokes = {key: np.zeros((ny, nx, n_wl), dtype=np.float32) for key in ["I", "V"]}

		for key in ["I", "V"]:
			for iw in range(n_wl):
				self.deconvolved_stokes[key][:, :, iw] = self._wiener_deconv(
					source[key][:, :, iw], psf, noise_level
				)
		return self.deconvolved_stokes

	def smooth_stokes(self, window_size: int = 3, use_deconvolved: bool = True) -> Dict[str, np.ndarray]:
		source = self.deconvolved_stokes if use_deconvolved else self.obs_stokes
		if source is None:
			raise ValueError("Stokes data not available for smoothing")

		self.smoothed_stokes = {
			key: uniform_filter1d(val, size=window_size, axis=2) for key, val in source.items()
		}
		return self.smoothed_stokes

	def compute_circular_polarization(self, epsilon: float = 1e-10) -> np.ndarray:
		if self.obs_stokes is None:
			raise ValueError("Load observed Stokes before computing circular polarization")

		I_safe = np.maximum(self.obs_stokes["I"], epsilon)
		self.circ_pol = np.mean(np.abs(self.obs_stokes["V"]) / I_safe, axis=-1)
		return self.circ_pol

	def create_mask_from_circular_polarization(self, threshold: Optional[float] = None) -> np.ndarray:
		if self.circ_pol is None:
			self.compute_circular_polarization()

		thr = threshold if threshold is not None else self.circular_polarization_threshold
		self.mask = self.circ_pol <= thr
		return self.mask

	def apply_mask_to_data(self) -> None:
		if self.mask is None:
			self.create_mask_from_circular_polarization()

		if self.continuum is not None:
			self.continuum = self.continuum.copy()
			self.continuum[~self.mask] = np.nan

		def _mask_stokes(stokes: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
			if stokes is None:
				return None
			masked = {k: v.copy() for k, v in stokes.items()}
			for key in masked:
				target_shape = masked[key].shape[:2]
				mask_on_target = self._mask_to_target_shape(self.mask, target_shape)
				masked[key][~mask_on_target] = np.nan
			return masked

		self.obs_stokes = _mask_stokes(self.obs_stokes)
		self.deconvolved_stokes = _mask_stokes(self.deconvolved_stokes)
		self.smoothed_stokes = _mask_stokes(self.smoothed_stokes)
		self.prediction_stokes = _mask_stokes(self.prediction_stokes)

		if self.inverted_profs is not None:
			self.inverted_profs = {k: v.copy() for k, v in self.inverted_profs.items()}
			for key in self.inverted_profs:
				self.inverted_profs[key][~self.mask] = np.nan

		if self.inverted_atmos is not None:
			self.inverted_atmos = self.inverted_atmos.copy()
			self.inverted_atmos[:, ~self.mask] = np.nan

		if self.spinor_atm is not None:
			masked_spinor: Dict[str, Dict[float, np.ndarray]] = {}
			for param, tau_dict in self.spinor_atm.items():
				masked_spinor[param] = {}
				for tau, arr in tau_dict.items():
					masked_arr = arr.copy()
					masked_arr[~self.mask] = np.nan
					masked_spinor[param][tau] = masked_arr
			self.spinor_atm = masked_spinor

	# ------------------------------------------------------------------
	# Sampling and spinor reconstruction
	# ------------------------------------------------------------------
	def find_sample_pixels(self) -> Dict[str, Dict[str, float]]:
		if self.continuum is None:
			raise ValueError("Continuum not loaded")
		if self.circ_pol is None:
			self.compute_circular_polarization()

		max_cont_idx = np.unravel_index(np.nanargmax(self.continuum), self.continuum.shape)
		min_cont_idx = np.unravel_index(np.nanargmin(self.continuum), self.continuum.shape)
		max_pol_idx = np.unravel_index(np.nanargmax(self.circ_pol), self.circ_pol.shape)
		min_pol_idx = np.unravel_index(np.nanargmin(self.circ_pol), self.circ_pol.shape)

		self.sample_pixels = {
			"max_continuum": {"y": float(max_cont_idx[0]), "x": float(max_cont_idx[1]), "label": "Max Continuum"},
			"min_continuum": {"y": float(min_cont_idx[0]), "x": float(min_cont_idx[1]), "label": "Min Continuum"},
			"max_polarization": {"y": float(max_pol_idx[0]), "x": float(max_pol_idx[1]), "label": "Max |Stokes V|"},
			"min_polarization": {"y": float(min_pol_idx[0]), "x": float(min_pol_idx[1]), "label": "Min |Stokes V|"},
		}
		return self.sample_pixels

	def build_spinor_atmosphere(
		self,
		tau_values: Tuple[float, float, float] = (-2.0, -0.8, 0.0),
		temp_indices: Tuple[int, int, int] = (8, 6, 7),
		vel_indices: Tuple[int, int, int] = (20, 18, 19),
		mag_field_indices: Tuple[int, int, int] = (11, 9, 10),
		gamma_indices: Tuple[int, int, int] = (14, 12, 13),
	) -> Dict[str, Dict[float, np.ndarray]]:
		if self.inverted_atmos is None:
			raise ValueError("Load inverted atmosphere first")

		spinor: Dict[str, Dict[float, np.ndarray]] = {"T": {}, "Vlos": {}, "Blos": {}}
		for i, tau in enumerate(tau_values):
			spinor["T"][tau] = self.inverted_atmos[temp_indices[i]]
			spinor["Vlos"][tau] = self.inverted_atmos[vel_indices[i]]
			b_field = self.inverted_atmos[mag_field_indices[i]]
			gamma_deg = self.inverted_atmos[gamma_indices[i]]
			gamma_rad = np.deg2rad(gamma_deg)
			spinor["Blos"][tau] = b_field * np.cos(gamma_rad)

		self.spinor_atm = spinor
		return self.spinor_atm

	def extract_region(
		self, y_start: int, y_end: int, x_start: int, x_end: int
	) -> Dict[str, object]:
		region: Dict[str, object] = {}
		base_bounds = (y_start, y_end, x_start, x_end)

		# Reference grid for user-provided bounds (original resolution)
		if self.continuum is not None:
			ref_shape = self.continuum.shape
		elif self.obs_stokes is not None:
			ref_shape = self.obs_stokes["I"].shape[:2]
		else:
			ref_shape = None

		def _crop_stokes_dict(stokes: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
			if stokes is None:
				return None
			any_key = next(iter(stokes))
			target_shape = stokes[any_key].shape[:2]
			if ref_shape is None:
				b = base_bounds
			else:
				b = self._scale_region_bounds(base_bounds, ref_shape, target_shape)
			return {k: v[b[0]:b[1], b[2]:b[3], :] for k, v in stokes.items()}

		if self.continuum is not None:
			region["continuum"] = self.continuum[y_start:y_end, x_start:x_end]

		# Stokes-like cubes: crop with scaled bounds when needed (e.g., upsampled/deconvolved/smoothed)
		if self.obs_stokes is not None:
			region["obs_stokes"] = _crop_stokes_dict(self.obs_stokes)
		if self.deconvolved_stokes is not None:
			region["deconvolved_stokes"] = _crop_stokes_dict(self.deconvolved_stokes)
		if self.smoothed_stokes is not None:
			region["smoothed_stokes"] = _crop_stokes_dict(self.smoothed_stokes)
		if self.prediction_stokes is not None:
			region["prediction_stokes"] = _crop_stokes_dict(self.prediction_stokes)
		if self.inverted_profs is not None:
			region["inverted_profs"] = _crop_stokes_dict(self.inverted_profs)

		# Non-Stokes products remain in original-resolution bounds
		if self.inverted_atmos is not None:
			region["inverted_atmos"] = self.inverted_atmos[:, y_start:y_end, x_start:x_end]
		if self.spinor_atm is not None:
			sub_spinor: Dict[str, Dict[float, np.ndarray]] = {}
			for param, tau_dict in self.spinor_atm.items():
				sub_spinor[param] = {tau: arr[y_start:y_end, x_start:x_end] for tau, arr in tau_dict.items()}
			region["spinor_atm"] = sub_spinor
		if self.mask is not None:
			region["mask"] = self.mask[y_start:y_end, x_start:x_end]
		return region

	def _scale_region_bounds(
		self,
		region_bounds: Tuple[int, int, int, int],
		source_shape: Tuple[int, int],
		target_shape: Tuple[int, int],
	) -> Tuple[int, int, int, int]:
		"""Scale (y_start, y_end, x_start, x_end) from source grid to target grid."""
		y_start, y_end, x_start, x_end = region_bounds
		sy = target_shape[0] / max(source_shape[0], 1)
		sx = target_shape[1] / max(source_shape[1], 1)

		ty0 = int(np.floor(y_start * sy))
		ty1 = int(np.ceil(y_end * sy))
		tx0 = int(np.floor(x_start * sx))
		tx1 = int(np.ceil(x_end * sx))

		ty0 = max(0, min(ty0, target_shape[0]))
		ty1 = max(0, min(ty1, target_shape[0]))
		tx0 = max(0, min(tx0, target_shape[1]))
		tx1 = max(0, min(tx1, target_shape[1]))
		return ty0, ty1, tx0, tx1

	# ------------------------------------------------------------------
	# Plotting helpers
	# ------------------------------------------------------------------
	def _ensure_samples(self) -> Dict[str, Dict[str, float]]:
		if self.sample_pixels is None:
			self.find_sample_pixels()
		return self.sample_pixels

	def plot_continuum(
		self,
		save_path: Optional[Union[Path, str]] = None,
		x_center_fov_pixel: Optional[float] = None,
		y_center_fov_pixel: Optional[float] = None,
		disk_center: Optional[Tuple[float, float]] = None,
		arrow_params: Optional[Tuple[float, float, float, float]] = None,
	) -> None:
		if self.continuum is None:
			raise ValueError("Continuum not loaded")

		samples = self._ensure_samples()
		fig, ax = plt.subplots(1, 1, figsize=(10, 8))
		im = ax.imshow(self.continuum, cmap="gray", origin="lower")
		ax.set_title("MODEST Continuum")
		ax.set_xlabel("X [pixels]")
		ax.set_ylabel("Y [pixels]")

		if x_center_fov_pixel is not None and y_center_fov_pixel is not None:
			ax.plot(x_center_fov_pixel, y_center_fov_pixel, "b+", markersize=15, markeredgewidth=3, label="FOV center")

		if disk_center is not None:
			x_disk, y_disk = disk_center
			if 0 <= x_disk <= self.continuum.shape[1] and 0 <= y_disk <= self.continuum.shape[0]:
				ax.plot(x_disk, y_disk, "r+", markersize=15, markeredgewidth=3, label="Disk center")
			elif arrow_params is not None:
				ax.arrow(*arrow_params, head_width=20, head_length=20, fc="blue", ec="blue", linewidth=3,
						 label="Disk center")

		colors = ["red", "blue", "green", "orange"]
		markers = ["x", "o", "s", "^"]
		for i, (_, pixel_info) in enumerate(samples.items()):
			ax.plot(pixel_info["x"], pixel_info["y"], markers[i], color=colors[i], markersize=8, markeredgewidth=2,
					label=pixel_info["label"])

		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		plt.colorbar(im, cax=cax)
		ax.legend()
		plt.tight_layout()

		if save_path:
			Path(save_path).parent.mkdir(parents=True, exist_ok=True)
			plt.savefig(save_path, dpi=300, bbox_inches="tight")
			plt.close(fig)
		else:
			plt.show()
			plt.close(fig)

	def plot_stokes_profiles(
		self,
		stokes_data: Dict[str, np.ndarray],
		wl: Union[u.Quantity, np.ndarray],
		sample_wavelength_idx: int,
		stokes_labels: Tuple[str, str] = ("Stokes I", "Stokes V"),
		title: str = "Stokes Profiles",
		save_path: Optional[Union[Path, str]] = None,
	) -> None:
		samples = self._ensure_samples()
		fig, axes = plt.subplots(2, 4, figsize=(16, 8))

		for i, (_, pixel_info) in enumerate(samples.items()):
			sample_x, sample_y = int(pixel_info["x"]), int(pixel_info["y"])
			profiles = [stokes_data["I"][sample_y, sample_x, :], stokes_data["V"][sample_y, sample_x, :]]
			for row, (profile, label) in enumerate(zip(profiles, stokes_labels)):
				ax = axes[row, i]
				ax.plot(wl, profile, "b-", linewidth=1.5)
				if not np.isnan(profile[sample_wavelength_idx]):
					ax.plot(wl[sample_wavelength_idx], profile[sample_wavelength_idx], "ro", markersize=6)
				ax.set_title(f"{label} - {pixel_info['label']}", fontsize=10)
				ax.set_ylabel("Intensity" if row == 0 else "Polarization")
				ax.set_xlabel("Wavelength [Angstrom]")
				ax.grid(True, alpha=0.3)

		plt.suptitle(title, fontsize=16)
		plt.tight_layout()
		if save_path:
			Path(save_path).parent.mkdir(parents=True, exist_ok=True)
			plt.savefig(save_path, dpi=300, bbox_inches="tight")
			plt.close(fig)
		else:
			plt.show()
			plt.close(fig)

	def plot_stokes_images(
		self,
		stokes_data: Dict[str, np.ndarray],
		wl: Union[u.Quantity, np.ndarray],
		sample_wavelength_idx: int,
		stokes_labels: Tuple[str, str] = ("Stokes I", "Stokes V"),
		title: str = "Stokes Parameters",
		save_path: Optional[Union[Path, str]] = None,
	) -> None:
		samples = self._ensure_samples()
		fig, axes = plt.subplots(1, 2, figsize=(18, 6))

		images = [stokes_data["I"][:, :, sample_wavelength_idx], stokes_data["V"][:, :, sample_wavelength_idx]]
		for col, (img, label) in enumerate(zip(images, stokes_labels)):
			ax = axes[col]

			valid = img[~np.isnan(img)]
			if valid.size > 0:
				if col == 0:
					vmin, vmax = np.quantile(valid, [0.05, 0.95])
				else:
					q1, q99 = np.quantile(valid, [0.01, 0.99])
					abs_max = np.max(np.abs([q1, q99]))
					vmin, vmax = -abs_max, abs_max
			else:
				vmin, vmax = (0, 1) if col == 0 else (-1, 1)

			im = ax.imshow(img, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
			ax.set_title(label, fontsize=12)
			ax.set_ylabel("Y [pixels]")
			ax.set_xlabel("X [pixels]")

			colors = ["red", "blue", "green", "orange"]
			markers = ["x", "o", "s", "^"]
			for i, (_, p_info) in enumerate(samples.items()):
				ax.plot(p_info["x"], p_info["y"], markers[i], color=colors[i], markersize=10, markeredgewidth=2,
						label=p_info["label"])

			if col == 0:
				ax.legend(loc="upper right", fontsize=8)

			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="5%", pad=0.05)
			plt.colorbar(im, cax=cax)

		wl_val = wl[sample_wavelength_idx].value if hasattr(wl, "unit") else wl[sample_wavelength_idx]
		plt.suptitle(f"{title} at lambda = {wl_val:.3f} Angstrom", fontsize=14)
		plt.tight_layout()

		if save_path:
			Path(save_path).parent.mkdir(parents=True, exist_ok=True)
			plt.savefig(save_path, dpi=300, bbox_inches="tight")
			plt.close(fig)
		else:
			plt.show()
			plt.close(fig)

	def plot_spinor_atmosphere(
		self,
		tau_values: Tuple[float, float, float] = (-2.0, -0.8, 0.0),
		save_dir: Optional[Union[Path, str]] = None,
	) -> None:
		if self.spinor_atm is None:
			raise ValueError("Build SPINOR atmosphere before plotting")

		samples = self._ensure_samples()
		colors = ["cyan", "magenta", "green", "red"]
		save_dir_path = Path(save_dir) if save_dir is not None else None

		def _plot_parameter(param_key: str, cmap: str, label: str, norm=None, cbar_label: str = "") -> None:
			fig, axes = plt.subplots(1, 3, figsize=(18, 5))
			for i, (ax, tau) in enumerate(zip(axes, tau_values)):
				data = self.spinor_atm[param_key][tau]
				valid = data[~np.isnan(data)]
				if valid.size > 0:
					if param_key == "T":
						vmin, vmax = np.quantile(valid, [0.05, 0.95])
					else:
						q1, q99 = np.quantile(valid, [0.01, 0.99])
						if np.abs(q1) > np.abs(q99):
							vmax = np.abs(q1)
						else:
							vmax = np.abs(q99)
						vmin = -vmax
				else:
					vmin, vmax = (-1, 1) if param_key != "T" else (0, 1)

				im = ax.imshow(data, cmap=cmap, origin="lower", norm=norm, vmin=None if norm else vmin,
							   vmax=None if norm else vmax)
				ax.set_title(f"log tau = {tau}")
				ax.set_xlabel("X [pixels]")
				ax.set_ylabel("Y [pixels]")
				for j, (_, pix) in enumerate(samples.items()):
					ax.plot(pix["x"], pix["y"], "x", color=colors[j], markersize=8, markeredgewidth=2)
				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right", size="5%", pad=0.05)
				cbar = plt.colorbar(im, cax=cax)
				if cbar_label:
					cbar.set_label(cbar_label)
			plt.suptitle(label, fontsize=16)
			plt.tight_layout()
			if save_dir_path is not None:
				save_dir_path.mkdir(parents=True, exist_ok=True)
				plt.savefig(save_dir_path / f"spinor_{param_key}.png", dpi=300, bbox_inches="tight")
				plt.close(fig)
			else:
				plt.show()
				plt.close(fig)

		_plot_parameter("T", cmap="hot", label="SPINOR Temperature", cbar_label="K")
		_plot_parameter("Vlos", cmap="RdBu_r", label="SPINOR v_los", cbar_label="km/s")
		_plot_parameter(
			"Blos",
			cmap="PiYG",
			label="SPINOR B_los",
			norm=SymLogNorm(linthresh=1e-2, linscale=1.0, vmin=-1e3, vmax=1e3),
			cbar_label="G",
		)

	# ------------------------------------------------------------------
	# Orchestration
	# ------------------------------------------------------------------
	def get_sorted_tau_values(self, param_key: str = "T") -> Tuple[float, ...]:
		"""Return sorted optical-depth keys for SPINOR; used for tau-grid overlap checks."""
		if self.spinor_atm is None:
			raise ValueError("SPINOR atmosphere not available")
		if param_key not in self.spinor_atm:
			raise KeyError(f"Unknown SPINOR key: {param_key}")
		return tuple(sorted(self.spinor_atm[param_key].keys()))

	def load_all(
		self, 
		region_bounds: Optional[Tuple[int, int, int, int]] = None, 
		apply_mask: bool = True,
		cache: ModestDataCache | None = None,
		use_cache: bool = True,
		upsampling_factor: int = 2,
		prediction_input_mode: str = "upsampled",
	) -> Dict[str, object]:
		"""Load all MODEST data products with optional upsampling.

		Parameters
		----------
		region_bounds : tuple, optional
			(y_start, y_end, x_start, x_end) for spatial subset
		apply_mask : bool
			Whether to apply circular polarization mask
		cache : ModestDataCache, optional
			If provided and use_cache=True, stores/loads uncropped processed MODEST snapshot
		use_cache : bool
			Enable cache usage when cache instance is provided
		upsampling_factor : int
			Spatial upsampling factor used before deconvolution
		prediction_input_mode : str
			Input grid used later for model prediction: 'upsampled' or 'downsampled'.
			'upsampled' uses smoothed deconvolved upsampled Stokes directly.
			'downsampled' reduces smoothed Stokes back to original observed grid.
		"""
		if prediction_input_mode not in ("upsampled", "downsampled"):
			raise ValueError("prediction_input_mode must be 'upsampled' or 'downsampled'")

		cache_config = {
			"modest_dir": str(self.modest_dir.resolve()),
			"psf_path": str(self.psf_path.resolve()),
			"upsampling_factor": int(upsampling_factor),
			"stokes_v_multiplier": float(self.stokes_v_multiplier),
			"wavelength_axis_source": "inverted_profs_header_or_hinode_fallback",
		}

		cache_hash = None
		if use_cache and cache is not None:
			cache_hash = cache.make_config_hash(cache_config)
			if cache.exists(cache_hash):
				payload = cache.load(cache_hash)
				self._restore_snapshot(payload)
			else:
				# Cache must always store uncropped + unmasked products
				self._compute_all_products(upsampling_factor=upsampling_factor)
				cache.save(cache_hash, self._snapshot_payload(), config=cache_config)
		else:
			self._compute_all_products(upsampling_factor=upsampling_factor)

		# Build prediction-ready Stokes on requested grid (native pipeline option)
		self.prediction_stokes = self._build_prediction_stokes(mode=prediction_input_mode)
		if apply_mask:
			self.apply_mask_to_data()

		if region_bounds is not None:
			return self.extract_region(*region_bounds) | {
				"wl": self.wl,
				"wl_inv": self.wl_inv,
				"prediction_input_mode": prediction_input_mode,
				"tau_values": self.get_sorted_tau_values("T"),
			}

		return {
			"continuum": self.continuum,
			"obs_stokes": self.obs_stokes,
			"upsampled_stokes": self.upsampled_stokes,
			"deconvolved_stokes": self.deconvolved_stokes,
			"smoothed_stokes": self.smoothed_stokes,
			"prediction_stokes": self.prediction_stokes,
			"inverted_profs": self.inverted_profs,
			"inverted_atmos": self.inverted_atmos,
			"spinor_atm": self.spinor_atm,
			"wl": self.wl,
			"wl_inv": self.wl_inv,
			"mask": self.mask,
			"circ_pol": self.circ_pol,
			"sample_pixels": self.sample_pixels,
			"prediction_input_mode": prediction_input_mode,
			"tau_values": self.get_sorted_tau_values("T"),
		}

	def _build_prediction_stokes(self, mode: str) -> Dict[str, np.ndarray]:
		if self.smoothed_stokes is None:
			raise ValueError("Smoothed Stokes are not available to build prediction input")
		if mode == "upsampled":
			return {k: v.copy() for k, v in self.smoothed_stokes.items()}

		if self.obs_stokes is None:
			raise ValueError("Observed Stokes not available for downsampling target grid")

		target_shape = self.obs_stokes["I"].shape[:2]
		return {
			k: self._downsample_cube_to_shape(v, target_shape)
			for k, v in self.smoothed_stokes.items()
		}

	@staticmethod
	def _mask_to_target_shape(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
		"""Map a boolean mask to a target grid using nearest-like integer scaling when possible."""
		sy, sx = mask.shape
		ty, tx = target_shape
		if (sy, sx) == (ty, tx):
			return mask.astype(bool)

		if ty % sy == 0 and tx % sx == 0:
			fy = ty // sy
			fx = tx // sx
			return np.repeat(np.repeat(mask, fy, axis=0), fx, axis=1).astype(bool)

		raised = np.zeros(target_shape, dtype=bool)
		sy_ratio = sy / ty
		sx_ratio = sx / tx
		for iy in range(ty):
			sy_idx = min(int(np.floor(iy * sy_ratio)), sy - 1)
			for ix in range(tx):
				sx_idx = min(int(np.floor(ix * sx_ratio)), sx - 1)
				raised[iy, ix] = bool(mask[sy_idx, sx_idx])
		return raised

	@staticmethod
	def _downsample_cube_to_shape(cube: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
		"""Downsample (ny, nx, nwl) cube to target (ny, nx) using block mean when possible."""
		ny, nx, nwl = cube.shape
		ty, tx = target_shape
		if (ny, nx) == (ty, tx):
			return cube.astype(np.float32)

		if ny % ty == 0 and nx % tx == 0:
			fy = ny // ty
			fx = nx // tx
			reduced = cube.reshape(ty, fy, tx, fx, nwl)
			return np.nanmean(reduced, axis=(1, 3)).astype(np.float32)

		raise ValueError(
			f"Cannot downsample cube from {(ny, nx)} to {(ty, tx)} using integer block mean"
		)

	def _compute_all_products(self, upsampling_factor: int) -> None:
		self.load_continuum()
		self.load_obs_stokes()
		self.compute_wavelength_arrays()

		print("Upsampling Stokes data for PSF compatibility...")
		self.upsampled_stokes = self.upsample_stokes(upsampling_factor=upsampling_factor)
		self.deconvolve_stokes(use_upsampled=True)

		self.compute_circular_polarization()
		self.find_sample_pixels()
		self.smooth_stokes()

		self.load_inverted_profs()
		self.load_inverted_atmos()
		self.build_spinor_atmosphere()

	def _snapshot_payload(self) -> Dict[str, Any]:
		return {
			"continuum": self.continuum,
			"obs_stokes": self.obs_stokes,
			"upsampled_stokes": self.upsampled_stokes,
			"deconvolved_stokes": self.deconvolved_stokes,
			"smoothed_stokes": self.smoothed_stokes,
			"prediction_stokes": self.prediction_stokes,
			"inverted_profs": self.inverted_profs,
			"inverted_atmos": self.inverted_atmos,
			"spinor_atm": self.spinor_atm,
			"wl": self.wl,
			"wl_inv": self.wl_inv,
			"circ_pol": self.circ_pol,
			"mask": self.mask,
			"sample_pixels": self.sample_pixels,
		}

	def _restore_snapshot(self, payload: Dict[str, Any]) -> None:
		self.continuum = payload.get("continuum")
		self.obs_stokes = payload.get("obs_stokes")
		self.upsampled_stokes = payload.get("upsampled_stokes")
		self.deconvolved_stokes = payload.get("deconvolved_stokes")
		self.smoothed_stokes = payload.get("smoothed_stokes")
		self.prediction_stokes = payload.get("prediction_stokes")
		self.inverted_profs = payload.get("inverted_profs")
		self.inverted_atmos = payload.get("inverted_atmos")
		self.spinor_atm = payload.get("spinor_atm")
		self.wl = payload.get("wl")
		self.wl_inv = payload.get("wl_inv")
		self.circ_pol = payload.get("circ_pol")
		self.mask = payload.get("mask")
		self.sample_pixels = payload.get("sample_pixels")