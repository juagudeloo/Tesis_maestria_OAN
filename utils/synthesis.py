"""MUISCA -> NICOLE forward-synthesis bridge.

Implements the components designed in the plan:
  - SynthesisConfig: runtime configuration shared by exporter, runner, comparator.
  - PredictionExporter: runs a trained model on requested pixels and persists
    denormalized T/V_LOS/B_LOS + matched observed Stokes to HDF5.
  - NicoleRunner: builds per-pixel NICOLE working directories from a
    persisted predictions HDF5 file and invokes NICOLE in synthesis mode.
  - SynthesisComparator: overlays synthesized vs observed Stokes and reports
    per-line chi-square.

v1 ships the MODEST source path; MURaM is left for v2.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.base_training import TrainingConfig
from utils.analysis import AnalysisModelPipeline
from utils.cache_manage import ModestDataCache
from utils.modest_data import ModestData, transform_modest_stokes_profiles
from utils.model_prof_tools import check_prof, read_prof, write_ascii_model
from utils.normalizer import MhdNormalizer, StokesNormalizer


@dataclass
class WholeRegionPrediction:
    """Result of running MUISCA inference over a whole cropped/uncropped region."""
    pred_mhd: dict[str, np.ndarray]          # {"T", "Vz", "Bz"}, each (pred_nx, pred_ny, n_tau)
    prediction_stokes: dict[str, np.ndarray] # {"I", "V"}, each (pred_nx, pred_ny, n_wl)
    wavelength: np.ndarray                    # (n_wl,)
    logtau: np.ndarray                        # (n_tau,)
    pred_nx: int
    pred_ny: int


@dataclass
class SynthesisConfig:
    source: str  # "modest" (v1) or "muram" (v2, stub)
    experiment_root: str
    model_type: str
    region_label: str = "whole"
    crop_bounds: Optional[tuple[int, int, int, int]] = None  # passed verbatim to ModestData

    # Filesystem layout
    output_root: Path = Path("/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis")
    nicole_root: Path = Path("/scratchsan/observatorio/juagudeloo/NICOLE_v16.06")
    nicole_assets: Path = Path("/scratchsan/observatorio/juagudeloo/MUISCA/data/nicole_assets")

    # MODEST loader
    modest_cache_dir: Path = Path("/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache")
    use_modest_cache: bool = True
    apply_polarization_mask: bool = False
    polarization_threshold: float = 1e-2
    stokes_v_multiplier: float = -1.0
    prediction_input_mode: str = "upsampled"

    # NICOLE wavelength grid (MODEST/Hinode SP Fe I 6301.5/6302.5 window)
    wl_first: float = 6300.796
    wl_step_mA: float = 21.5
    n_wl: int = 112

    # NICOLE atmospheric-column defaults (for the values MUISCA does not predict)
    v_mic_cms: float = 1.0e5
    v_mac_cms: float = 0.0
    stray_light: float = 0.0
    el_p_seed: float = 1.0

    # NICOLE driver interpreter (v16.06's run_nicole.py is Python 2)
    python_for_nicole: str = "python2"

    # Inference
    inference_batch_size: int = 4096

    def out_dir(self) -> Path:
        return self.output_root / self.experiment_root / self.model_type / self.region_label

    def predictions_h5(self) -> Path:
        return self.out_dir() / "predictions.h5"

    def syntheses_h5(self) -> Path:
        return self.out_dir() / "syntheses.h5"


class PredictionExporter:
    """Loads a trained MUISCA model, runs it on the requested region, and
    persists denormalized predictions + matched observed Stokes to HDF5.
    """

    def __init__(self, cfg: SynthesisConfig, device: torch.device):
        if cfg.source != "modest":
            raise NotImplementedError(
                f"PredictionExporter v1 supports source='modest' only "
                f"(got {cfg.source!r}); MURaM path is on the v2 backlog"
            )
        self.cfg = cfg
        self.device = device

    def predict_whole_region(self) -> WholeRegionPrediction:
        """Run the trained model over the entire cropped/uncropped region.

        Returns a WholeRegionPrediction containing denormalized T/Vz/Bz cubes
        of shape (pred_nx, pred_ny, n_tau) plus matched observed Stokes and
        wavelength/logtau arrays. This is the core inference step that both
        export() and external sampling scripts (e.g., pixel_sampling.py) reuse.
        """
        cfg = self.cfg
        out_dir = cfg.out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        pipeline = AnalysisModelPipeline(
            device=self.device,
            output_dir=out_dir,
            experiment_root=cfg.experiment_root,
        )
        model_configs, models, n_tau = pipeline.prepare_models([cfg.model_type])
        if not models:
            raise RuntimeError(
                f"No model loaded for experiment_root={cfg.experiment_root!r}, "
                f"model_type={cfg.model_type!r}"
            )
        model_name, model = next(iter(models.items()))
        model_cfg = model_configs[model_name]
        pred_tau = pipeline.get_model_logtau_values(model_cfg)
        if pred_tau is None:
            raise RuntimeError(
                f"Could not determine logtau values for model {model_name}"
            )
        pred_tau_arr = np.asarray(pred_tau, dtype=np.float64)

        # Normalizers (loaded once)
        default_cfg = TrainingConfig()
        mhd_normalizer = MhdNormalizer()
        stokes_normalizer = StokesNormalizer()
        mhd_normalizer.load(filepath=str(Path(default_cfg.data_path) / default_cfg.mhd_normalizer_path))
        stokes_normalizer.load(filepath=str(Path(default_cfg.data_path) / default_cfg.stokes_normalizer_path))

        # Load MODEST observations and build the prediction-ready Stokes cube.
        modest = ModestData(
            circular_polarization_threshold=cfg.polarization_threshold,
            stokes_v_multiplier=cfg.stokes_v_multiplier,
        )
        modest_cache = ModestDataCache(cache_dir=cfg.modest_cache_dir)
        modest_data = modest.load_all(
            region_bounds=tuple(cfg.crop_bounds) if cfg.crop_bounds is not None else None,
            apply_mask=cfg.apply_polarization_mask,
            cache=modest_cache,
            use_cache=cfg.use_modest_cache,
            prediction_input_mode=cfg.prediction_input_mode,
        )

        # Wavelength axis (continuum-normalized prediction-ready Stokes).
        prediction_stokes = modest_data.get("prediction_stokes", modest_data["smoothed_stokes"])
        wl_raw = modest_data.get("wl", np.arange(prediction_stokes["I"].shape[-1]))
        wavelength = np.asarray(
            wl_raw.value if hasattr(wl_raw, "value") else wl_raw,
            dtype=np.float64,
        )
        # No spectral transforms in v1 — defaults match the analysis script's
        # no-shift, no-scale, no-invert path.
        prediction_stokes = transform_modest_stokes_profiles(
            stokes=prediction_stokes,
            wavelength_angstrom=wavelength,
            shift_positions=0.0,
            i_scale=1.0,
            v_scale=1.0,
            invert_direction=False,
        )
        pred_nx, pred_ny = prediction_stokes["I"].shape[:2]
        norm_stokes = stokes_normalizer.transform(prediction_stokes)
        I_flat = norm_stokes["I"].reshape(pred_nx * pred_ny, -1)
        V_flat = norm_stokes["V"].reshape(pred_nx * pred_ny, -1)
        stokes_input = np.stack([I_flat, V_flat], axis=1).astype(np.float32)

        # Inference
        pred_mhd = pipeline.predict_and_denormalize(
            model=model,
            stokes_input=stokes_input,
            mhd_normalizer=mhd_normalizer,
            pred_nx=pred_nx,
            pred_ny=pred_ny,
            batch_size=cfg.inference_batch_size,
        )

        return WholeRegionPrediction(
            pred_mhd=pred_mhd,
            prediction_stokes=prediction_stokes,
            wavelength=wavelength,
            logtau=pred_tau_arr,
            pred_nx=pred_nx,
            pred_ny=pred_ny,
        )

    def export(self, pixels: Sequence[tuple[int, int]]) -> Path:
        cfg = self.cfg

        # Run inference over the whole region once
        region = self.predict_whole_region()
        pred_mhd = region.pred_mhd
        prediction_stokes = region.prediction_stokes
        wavelength = region.wavelength
        logtau = region.logtau
        pred_nx = region.pred_nx
        pred_ny = region.pred_ny
        n_tau = len(logtau)

        # Subselect the requested pixels.
        pixels_arr = np.asarray(pixels, dtype=np.int64).reshape(-1, 2)
        for (ix, iy) in pixels_arr:
            if not (0 <= ix < pred_nx and 0 <= iy < pred_ny):
                raise IndexError(
                    f"Pixel (ix={ix}, iy={iy}) out of range for prediction grid "
                    f"{pred_nx}x{pred_ny}"
                )

        n = pixels_arr.shape[0]
        T_out = np.empty((n, n_tau), dtype=np.float64)
        Vz_out = np.empty((n, n_tau), dtype=np.float64)
        Bz_out = np.empty((n, n_tau), dtype=np.float64)
        stokes_obs = np.empty((n, 2, prediction_stokes["I"].shape[-1]), dtype=np.float32)
        for k, (ix, iy) in enumerate(pixels_arr):
            T_out[k] = pred_mhd["T"][ix, iy, :n_tau]
            Vz_out[k] = pred_mhd["Vz"][ix, iy, :n_tau]
            Bz_out[k] = pred_mhd["Bz"][ix, iy, :n_tau]
            stokes_obs[k, 0] = prediction_stokes["I"][ix, iy, :]
            stokes_obs[k, 1] = prediction_stokes["V"][ix, iy, :]

        out_path = cfg.predictions_h5()
        if out_path.exists():
            out_path.unlink()
        with h5py.File(out_path, "w") as h5:
            h5.create_dataset("T", data=T_out)
            h5.create_dataset("Vz", data=Vz_out)
            h5.create_dataset("Bz", data=Bz_out)
            h5.create_dataset("pixels", data=pixels_arr)
            h5.create_dataset("logtau", data=logtau)
            h5.create_dataset("wavelengths", data=wavelength)
            h5.create_dataset("stokes_obs", data=stokes_obs)
            h5.attrs["source"] = cfg.source
            h5.attrs["experiment_root"] = cfg.experiment_root
            h5.attrs["model_type"] = cfg.model_type
            h5.attrs["region_label"] = cfg.region_label
            h5.attrs["pred_grid"] = np.array([pred_nx, pred_ny], dtype=np.int64)
            if cfg.crop_bounds is not None:
                h5.attrs["crop_bounds"] = np.asarray(cfg.crop_bounds, dtype=np.int64)
        return out_path


class NicoleRunner:
    """Builds per-pixel NICOLE working directories from a predictions HDF5 file
    and invokes NICOLE in synthesis mode.
    """

    def __init__(self, cfg: SynthesisConfig):
        self.cfg = cfg
        self.template_path = cfg.nicole_assets / "NICOLE.input.template"
        self.lines_path = cfg.nicole_assets / "LINES"
        if not self.template_path.exists():
            raise FileNotFoundError(self.template_path)
        if not self.lines_path.exists():
            raise FileNotFoundError(self.lines_path)
        nicole_command = cfg.nicole_root / "main" / "nicole"
        if not nicole_command.exists():
            raise FileNotFoundError(
                f"NICOLE binary not found at {nicole_command}. "
                "Build NICOLE first (cd NICOLE/main && make)."
            )
        self.nicole_command = nicole_command
        self.run_nicole_py = cfg.nicole_root / "run" / "run_nicole.py"
        if not self.run_nicole_py.exists():
            raise FileNotFoundError(self.run_nicole_py)

    def _pixel_workdir(self, ix: int, iy: int) -> Path:
        return self.cfg.out_dir() / f"pix_{ix:05d}_{iy:05d}"

    def prepare_workdir(
        self,
        ix: int,
        iy: int,
        logtau: np.ndarray,
        T: np.ndarray,
        v_los_kms: np.ndarray,
        b_long_G: np.ndarray,
    ) -> Path:
        workdir = self._pixel_workdir(ix, iy)
        workdir.mkdir(parents=True, exist_ok=True)

        # Write the per-pixel atmosphere
        model_path = workdir / "model.model"
        write_ascii_model(
            model_path,
            logtau=logtau,
            T=T,
            v_los_kms=v_los_kms,
            b_long_G=b_long_G,
            v_mic_cms=self.cfg.v_mic_cms,
            v_mac_cms=self.cfg.v_mac_cms,
            stray_light=self.cfg.stray_light,
            el_p_seed=self.cfg.el_p_seed,
        )

        # Substitute placeholders in NICOLE.input template
        template = self.template_path.read_text()
        rendered = template.format(
            NICOLE_COMMAND=str(self.nicole_command),
            INPUT_MODEL="model.model",
            OUTPUT_PROFILE="profile.pro",
            OUTPUT_MODEL="model_out.mod",
            WL_FIRST=f"{self.cfg.wl_first:.4f}",
            WL_STEP_MA=f"{self.cfg.wl_step_mA:g}",
            N_WL=int(self.cfg.n_wl),
        )
        (workdir / "NICOLE.input").write_text(rendered)

        # Drop the LINES file in place (copy keeps the workdir self-contained).
        shutil.copyfile(self.lines_path, workdir / "LINES")

        # NICOLE's run_nicole.py validates its own source via a relative
        # open('run_nicole.py'); the canonical pattern (cf. test/syn1) is a
        # symlink into the working directory. Replace any stale link.
        link = workdir / "run_nicole.py"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(self.run_nicole_py)

        return workdir

    def run_pixel(self, workdir: Path) -> np.ndarray:
        """Invoke NICOLE in synthesis mode and return the (4, n_wl) IQUV array."""
        env = os.environ.copy()
        # v16.06's run_nicole.py only initializes inputmodel/outprof when --modelin
        # and --profout are passed explicitly; without them, NameError on line 184.
        result = subprocess.run(
            [
                self.cfg.python_for_nicole,
                "./run_nicole.py",
                "--nicolecommand=" + str(self.nicole_command),
                "--modelin=model.model",
                "--profout=profile.pro",
            ],
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"NICOLE failed in {workdir} (exit {result.returncode}).\n"
                f"--- STDOUT ---\n{result.stdout}\n"
                f"--- STDERR ---\n{result.stderr}"
            )
        prof_path = workdir / "profile.pro"
        if not prof_path.exists():
            raise FileNotFoundError(
                f"NICOLE did not produce expected output {prof_path}.\n"
                f"--- STDOUT ---\n{result.stdout}"
            )
        # v16.06 writes 'nicole2.3bp' binary profiles. The shipped check_prof
        # opens in text mode and fails to decode under Python 3, but we already
        # know the format and shape from the NICOLE.input region we rendered.
        flat = read_prof(
            str(prof_path), "nicole2.3", 1, 1, int(self.cfg.n_wl), 0, 0
        )
        flat = np.asarray(flat, dtype=np.float64).reshape(int(self.cfg.n_wl), 4)
        # NICOLE row layout per wavelength: [I, Q, U, V]
        iquv = np.stack([flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]], axis=0)
        return iquv

    def run_pixels_from_h5(
        self,
        predictions_h5: Path,
        pixels: Optional[Sequence[tuple[int, int]]] = None,
    ) -> Path:
        """For each requested pixel, prepare a workdir, run NICOLE, and stash the
        result into syntheses.h5 alongside predictions.h5.
        """
        with h5py.File(predictions_h5, "r") as h5:
            pred_pixels = h5["pixels"][...]
            logtau = h5["logtau"][...]
            T_all = h5["T"][...]
            Vz_all = h5["Vz"][...]
            Bz_all = h5["Bz"][...]
            wavelengths = h5["wavelengths"][...]

        if pixels is None:
            target_pixels = [tuple(p) for p in pred_pixels.tolist()]
        else:
            target_pixels = [tuple(p) for p in pixels]

        # Index map from pixel tuple -> row in predictions
        index_of = {tuple(p): i for i, p in enumerate(pred_pixels.tolist())}
        for p in target_pixels:
            if p not in index_of:
                raise KeyError(
                    f"Pixel {p} was not exported in {predictions_h5}. "
                    f"Available: {list(index_of)[:10]}{'...' if len(index_of) > 10 else ''}"
                )

        n = len(target_pixels)
        synth = np.full((n, 4, self.cfg.n_wl), np.nan, dtype=np.float64)
        for k, (ix, iy) in enumerate(target_pixels):
            row = index_of[(ix, iy)]
            workdir = self.prepare_workdir(
                ix=int(ix),
                iy=int(iy),
                logtau=logtau,
                T=T_all[row],
                v_los_kms=Vz_all[row],
                b_long_G=Bz_all[row],
            )
            iquv = self.run_pixel(workdir)
            if iquv.shape[1] != self.cfg.n_wl:
                raise RuntimeError(
                    f"NICOLE returned {iquv.shape[1]} wavelengths for pixel "
                    f"({ix},{iy}), expected {self.cfg.n_wl}"
                )
            synth[k] = iquv

        out_path = self.cfg.syntheses_h5()
        if out_path.exists():
            out_path.unlink()
        with h5py.File(out_path, "w") as h5:
            h5.create_dataset(
                "pixels",
                data=np.asarray(target_pixels, dtype=np.int64),
            )
            h5.create_dataset("stokes_synth", data=synth)
            h5.create_dataset("wavelengths", data=wavelengths)
            h5.attrs["source"] = self.cfg.source
            h5.attrs["experiment_root"] = self.cfg.experiment_root
            h5.attrs["model_type"] = self.cfg.model_type
            h5.attrs["region_label"] = self.cfg.region_label
        return out_path


class SynthesisComparator:
    """Overlay synthesized vs observed Stokes I and V, compute chi-square."""

    def __init__(self, cfg: SynthesisConfig):
        self.cfg = cfg
        self.predictions_h5: Optional[Path] = None
        self.syntheses_h5: Optional[Path] = None
        self._pred_pixels = None
        self._synth_pixels = None
        self._stokes_obs = None
        self._stokes_synth = None
        self._wavelengths = None

    def load(self, predictions_h5: Path, syntheses_h5: Path) -> None:
        self.predictions_h5 = predictions_h5
        self.syntheses_h5 = syntheses_h5
        with h5py.File(predictions_h5, "r") as h5:
            self._pred_pixels = h5["pixels"][...]
            self._stokes_obs = h5["stokes_obs"][...]
            self._wavelengths = h5["wavelengths"][...]
        with h5py.File(syntheses_h5, "r") as h5:
            self._synth_pixels = h5["pixels"][...]
            self._stokes_synth = h5["stokes_synth"][...]

    def _pred_row(self, ix: int, iy: int) -> int:
        for i, (a, b) in enumerate(self._pred_pixels.tolist()):
            if a == ix and b == iy:
                return i
        raise KeyError(f"Pixel ({ix},{iy}) not in predictions.h5")

    def _synth_row(self, ix: int, iy: int) -> int:
        for i, (a, b) in enumerate(self._synth_pixels.tolist()):
            if a == ix and b == iy:
                return i
        raise KeyError(f"Pixel ({ix},{iy}) not in syntheses.h5")

    def chi_square(
        self,
        sigma_I: float = 1e-3,
        sigma_V: float = 1e-3,
    ) -> dict:
        out = {}
        for ix, iy in self._synth_pixels.tolist():
            i_obs = self._stokes_obs[self._pred_row(ix, iy), 0]
            v_obs = self._stokes_obs[self._pred_row(ix, iy), 1]
            i_syn = self._stokes_synth[self._synth_row(ix, iy), 0]
            v_syn = self._stokes_synth[self._synth_row(ix, iy), 3]
            chi2_I = float(np.nansum(((i_obs - i_syn) / sigma_I) ** 2))
            chi2_V = float(np.nansum(((v_obs - v_syn) / sigma_V) ** 2))
            out[f"{ix},{iy}"] = {
                "chi2_I": chi2_I,
                "chi2_V": chi2_V,
                "n_wl": int(np.isfinite(i_syn).sum()),
            }
        return out

    def plot_overlay(self, ix: int, iy: int, out_dir: Path) -> Path:
        import matplotlib.pyplot as plt

        out_dir.mkdir(parents=True, exist_ok=True)
        i_obs = self._stokes_obs[self._pred_row(ix, iy), 0]
        v_obs = self._stokes_obs[self._pred_row(ix, iy), 1]
        synth_row = self._stokes_synth[self._synth_row(ix, iy)]
        i_syn, q_syn, u_syn, v_syn = synth_row[0], synth_row[1], synth_row[2], synth_row[3]
        wl = self._wavelengths

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(wl, i_obs, color="k", label="Observed")
        axes[0].plot(wl, i_syn, color="C3", ls="--", label="NICOLE (MUISCA atm.)")
        axes[0].set_title(f"Stokes I  pix=({ix},{iy})")
        axes[0].set_xlabel("Wavelength [Å]")
        axes[0].set_ylabel("I / I_c")
        axes[0].legend()

        axes[1].plot(wl, v_obs, color="k", label="Observed")
        axes[1].plot(wl, v_syn, color="C3", ls="--", label="NICOLE")
        axes[1].set_title(f"Stokes V  pix=({ix},{iy})")
        axes[1].set_xlabel("Wavelength [Å]")
        axes[1].set_ylabel("V / I_c")
        axes[1].legend()

        plt.tight_layout()
        out_path = out_dir / f"overlay_pix_{ix:05d}_{iy:05d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
