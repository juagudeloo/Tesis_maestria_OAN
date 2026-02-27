import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import astropy.units as u
from pathlib import Path
import sys
import pandas as pd
import hashlib
import json
from utils.cache_manage import DataCache
from utils.physics_utils import ApproxInversions
from typing import Dict, Tuple

sys.path.append("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN")
from utils.muram_data import MhdData, StokesData
from models.pinn_mscnn_model import PhysicsInformedMSCNN
from utils.normalizer import MhdNormalizer, StokesNormalizer
from utils.analysis_functions import MuramAnalysis

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


def setup_paths():
    """Setup output directories."""
    images_base_path = Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/images")
    images_save_path = images_base_path / "analysis/muram"
    images_save_path.mkdir(parents=True, exist_ok=True)
    return images_save_path


def _resolve_logtau_values(
    logtau_values: list[float] | None = None,
    logtau_min: float = -2.0,
    logtau_max: float = 0.0,
    logtau_step: float = 0.1,
) -> np.ndarray:
    if logtau_values is None or len(logtau_values) == 0:
        if logtau_step <= 0:
            raise ValueError(f"logtau_step must be > 0, got {logtau_step}")
        arr = np.arange(
            logtau_min,
            logtau_max + 0.5 * logtau_step,
            logtau_step,
            dtype=np.float32,
        )
    else:
        arr = np.asarray(logtau_values, dtype=np.float32)

    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("logtau grid must be 1D with at least 2 values")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("logtau grid must be strictly increasing")
    return np.round(arr, 6)


def _shared_cache_signature(logtau_values: np.ndarray) -> dict:
    # Match scripts/base_training.py defaults used for cache hash.
    return {
        "nx": 480,
        "ny": 480,
        "nz": 256,
        "z_max": 250,
        "dz_km": 10.0,
        "central_wavelength": 6301.5,
        "wl_range": (15, 60),
        "logtau_values": tuple(float(x) for x in np.asarray(logtau_values, dtype=np.float32).tolist()),
    }


def load_muram_data(
    data_path,
    steps=[80, 95, 195],
    cache: DataCache | None = None,
    logtau_values: np.ndarray | None = None,
):
    """Load MURAM MHD and Stokes data for specified steps, using shared DataCache."""
    muram_steps = {}
    new_logtau = (
        np.asarray(logtau_values, dtype=np.float32)
        if logtau_values is not None
        else _resolve_logtau_values()
    )
    config_hash = DataCache.make_config_hash(_shared_cache_signature(new_logtau)) if cache else None

    for step in steps:
        print(f"\nLoading MURAM step {step}...")

        if cache is not None:
            exact_hit = cache.exists(step, config_hash, logtau_values=new_logtau)
            relaxed_hit = False
            if not exact_hit:
                try:
                    # Allow reuse when raw cache is compatible but metadata hash differs.
                    relaxed_hit = cache.exists(step, None, logtau_values=new_logtau)
                except Exception:
                    relaxed_hit = False

            if exact_hit or relaxed_hit:
                stokes_data, mhd_data, _ = cache.load_raw(step=step, verbose=True)
                muram_steps[step] = {'mhd': mhd_data, 'stokes': stokes_data}
                if relaxed_hit and not exact_hit:
                    print(f"✓ Loaded MURAM step {step} from shared cache (relaxed hash match)")
                else:
                    print(f"✓ Loaded MURAM step {step} from shared cache")
                continue

        # Load MHD data
        mhd = MhdData(
            data_path=data_path / "muram-simulation",
            nx=480, ny=480, nz=256
        )
        mhd.load_step(step=step, z_max=250)
        mhd.load_opacity_table(kappa_path=data_path / "csv/kappa.0.dat")
        mhd.compute_optical_depth(dz=1e6)
        mhd.remap_to_optical_depth(new_logtau, quantities=["T", "Vz", "Bz"])

        # Load Stokes data
        stokes = StokesData(
            data_dir=data_path / "muram-simulation/",
            step=step,
            wavelength_range=(6300.5, 6303.5),
            wavelength_step=0.01
        )
        stokes.load_stokes()
        stokes.continuum_normalization(cont_indices=[0, 1, 2, 3])
        stokes.load_hinode_lsf(data_path / "hinode-MODEST/PSFs/hinode_sp.spline.psf")
        stokes.apply_spectral_convolution()
        stokes.resample_to_hinode()
        # stokes.add_hinode_noise()

        muram_steps[step] = {'mhd': mhd.od_data, 'stokes': stokes.data}

        if cache is not None:
            inv = ApproxInversions(
                stokes=stokes.data,
                wavelength=stokes.wl,
                central_wavelength=6301.5 * u.Angstrom,
                lande_factor=1.67,
            )
            approx_data = {
                "blos": inv.compute_blos_wfa(wl_range=(15, 60)).value,
                "vlos": inv.compute_vlos_doppler(wl_range=(15, 60)).value,
                "temp": inv.compute_temperature_blackbody(
                    cont_indices=[0, 1, 2, 3],
                    reference_temperature=6000.0 * u.K,
                    continuum_wavelength=6300.5 * u.Angstrom,
                ).value,
            }
            cache.save(
                step=step,
                stokes_data=stokes.data,
                mhd_data=mhd.od_data,
                approx_data=approx_data,
                config_hash=config_hash,
                logtau_values=new_logtau,
                verbose=True,
            )

        print(f"✓ Loaded MURAM step {step}: MHD shape {mhd.data['T'].shape}, Stokes I shape {stokes.data['I'].shape}")
    return muram_steps


def load_normalizers(data_path, muram_steps):
    """Load and apply normalizers to Stokes data."""
    mhd_normalizer = MhdNormalizer()
    mhd_normalizer.load(data_path / "normalization_stats/mhd_normalization.json")

    stokes_normalizer = StokesNormalizer()
    stokes_normalizer.load(data_path / "normalization_stats/stokes_normalization.json")

    for step_data in muram_steps.values():
        step_data['normalized_stokes'] = stokes_normalizer.transform(step_data['stokes'])

    print("✓ Normalizers loaded and Stokes data normalized for all steps")
    return mhd_normalizer, stokes_normalizer


def get_model_configs():
    """Return model configurations."""
    base_model_path = Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments/")
    experiment_dir = base_model_path / "physics_regularization_ablation_80_to_113-fixed_lambdas"
    results_path = experiment_dir / "experiment_results.json"
    no_physics_weights = "no_physics/final_model.pth"
    wfa_only_weights = "wfa_only/final_model.pth"
    doppler_only_weights = "doppler_only/final_model.pth"
    black_body_only_weights = "black_body_only/final_model.pth"
    all_physics_terms_weights = "all_physics_terms/final_model.pth"
    return {
        'no_physics_80_to_113-fixed_lambdas': {
            'path': experiment_dir / no_physics_weights,
            'results_path': results_path,
            'experiment_key': 'no_physics',
            'use_physics': None,
            'lambda_wfa': 0.0,
            'lambda_doppler': 0.0,
            'lambda_temp': 0.0,
            'label': 'No Physics 80 to 113',
            'color': 'blue'
        },
        'wfa_only_80_to_113-fixed_lambdas': {
            'path': experiment_dir / wfa_only_weights,
            'results_path': results_path,
            'experiment_key': 'wfa_only',
            'use_physics': ['wfa'],
            'lambda_wfa': 1.0,
            'lambda_doppler': 0.0,
            'lambda_temp': 0.0,
            'label': 'WFA Only 80 to 113',
            'color': 'orange'
        },
    }


def _infer_output_features_from_checkpoint(checkpoint: Dict) -> int:
    state = checkpoint["model_state_dict"]
    key = "linear_block.output_layer.bias"
    if key not in state:
        raise KeyError(f"Missing '{key}' in checkpoint state_dict")
    return int(state[key].shape[0])


def _resolve_results_meta(cfg: Dict) -> tuple[Path, str]:
    """Resolve results_path/experiment_key with fallback from checkpoint path."""
    model_path = Path(cfg["path"])
    results_path = Path(cfg.get("results_path", model_path.parent.parent / "experiment_results.json"))
    experiment_key = str(cfg.get("experiment_key", model_path.parent.name))
    return results_path, experiment_key

def _resolve_logtau_from_experiment_results(model_configs: Dict) -> np.ndarray:
    cache: Dict[Path, Dict] = {}
    resolved: Dict[str, np.ndarray] = {}
    for model_name, cfg in model_configs.items():
        results_path, exp_key = _resolve_results_meta(cfg)
        if results_path not in cache:
            with open(results_path, "r") as f:
                cache[results_path] = json.load(f)
        results = cache[results_path]
        if exp_key not in results:
            raise KeyError(f"'{exp_key}' not found in {results_path}")
        vals = results[exp_key].get("config", {}).get("logtau_values", None)
        if vals is None:
            raise KeyError(f"'config.logtau_values' not found for '{exp_key}' in {results_path}")
        resolved[model_name] = np.round(np.asarray(vals, dtype=np.float32), 6)

    ref_name = next(iter(resolved))
    ref = resolved[ref_name]
    for name, arr in resolved.items():
        if arr.shape != ref.shape or not np.allclose(arr, ref, atol=1e-6, rtol=0.0):
            raise ValueError(f"Inconsistent logtau_values between models: {ref_name} vs {name}")
    return ref


def load_model(config, device) -> Tuple[PhysicsInformedMSCNN, int]:
    """Load a trained PINN-MSCNN model with checkpoint-matched output size."""
    checkpoint = torch.load(config['path'], map_location=device)
    output_features = _infer_output_features_from_checkpoint(checkpoint)
    if output_features % 3 != 0:
        raise ValueError(f"Invalid output_features={output_features}; expected multiple of 3")
    n_tau = output_features // 3

    model = PhysicsInformedMSCNN(
        scales=[1, 2, 3],
        in_channels=2,
        c1_filters=16,
        c2_filters=32,
        kernel_size=5,
        pool_size=2,
        n_linear_layers=4,
        output_features=output_features,
        input_length=112,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, n_tau


def load_all_models(model_configs, device):
    """Load all trained models and enforce consistent tau dimension."""
    models = {}
    n_tau_ref = None
    for name, config in model_configs.items():
        print(f"Loading {config['label']}...")
        model, n_tau = load_model(config, device)
        if n_tau_ref is None:
            n_tau_ref = n_tau
        elif n_tau != n_tau_ref:
            raise ValueError(f"Inconsistent n_tau across models: {name} has {n_tau}, expected {n_tau_ref}")
        models[name] = model
    print(f"✓ Loaded {len(models)} models (n_tau={n_tau_ref})\n")
    return models, int(n_tau_ref)


def prepare_input_data(muram_steps, device):
    """Prepare input tensors for each step."""
    muram_inputs = {}
    for step, step_data in muram_steps.items():
        I_t = step_data['normalized_stokes']["I"]
        V_t = step_data['normalized_stokes']["V"]
        inputs = np.stack([I_t, V_t], axis=2)
        H, W, Nstokes, Nlambda = inputs.shape
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).permute(0, 1, 3, 2)
        inputs_tensor = inputs_tensor.reshape(H*W, Nstokes, Nlambda).to(device)
        muram_inputs[step] = dict(
            tensor=inputs_tensor,
            shape=(H, W, Nstokes, Nlambda)
        )
        print(f"Step {step}: input tensor shape {inputs_tensor.shape}, total pixels: {H*W}")
    return muram_inputs


def _cache_fingerprint(model_name, model_config, shape, step: int, logtau: np.ndarray | None = None) -> str:
    model_path = Path(model_config["path"])
    payload = {
        "model": model_name,
        "model_path": str(model_path),
        "model_mtime": model_path.stat().st_mtime if model_path.exists() else None,
        "shape": tuple(int(v) for v in shape),
        "step": int(step),
        "logtau": [float(x) for x in np.asarray(logtau).tolist()] if logtau is not None else None,
        "dataset": "muram_whole_region",
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _prediction_cache_file(step: int, model_name: str, fingerprint: str, cache_dir: Path | None = None) -> Path:
    if cache_dir is not None:
        step_dir = cache_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir / f"{model_name}_{fingerprint}.npz"
    return Path()


def _plot_prefix_exists(save_dir: Path, filename_prefix: str, skip_existing: bool) -> bool:
    """Check if any plot artifact with a given prefix already exists."""
    if not skip_existing:
        return False
    if not save_dir.exists():
        return False
    return any(save_dir.glob(f"{filename_prefix}*.png")) or any(save_dir.glob(f"{filename_prefix}*.json"))


def run_inference(
    muram_inputs,
    muram_steps,
    models,
    model_configs,
    mhd_normalizer,
    logtau,
    prediction_cache_dir: Path | None = None,
    use_prediction_cache: bool = True,
):
    """Run inference for all models and steps."""
    muram_predictions = {}
    n_tau = int(len(logtau))

    for step, muram in muram_inputs.items():
        H, W, Nstokes, Nlambda = muram['shape']
        inputs_tensor = muram['tensor']
        muram_predictions[step] = {}
        print(f"\n=== Running inference for MURAM step {step} ===")

        for model_name, model in models.items():
            print(f"  Model: {model_configs[model_name]['label']}")

            cache_file = None
            if use_prediction_cache and prediction_cache_dir is not None:
                fp = _cache_fingerprint(model_name, model_configs[model_name], muram['shape'], step, logtau=logtau)
                cache_file = _prediction_cache_file(
                    step=step, model_name=model_name, fingerprint=fp,
                    cache_dir=prediction_cache_dir,
                )
                if cache_file.exists():
                    cached = np.load(cache_file)
                    prediction_atm = {"T": cached["T"], "Vz": cached["Vz"], "Bz": cached["Bz"]}
                    muram_predictions[step][model_name] = {
                        "prediction": prediction_atm,
                        "label": model_configs[model_name]["label"],
                        "color": model_configs[model_name]["color"],
                    }
                    print(f"    ↷ Loaded cached predictions: {cache_file}")
                    print(f"    ✓ Done. T range: {prediction_atm['T'].min():.1f}-{prediction_atm['T'].max():.1f} K")
                    continue

            model.eval()
            device = next(model.parameters()).device
            n_pixels = inputs_tensor.shape[0]
            all_predictions_batch = []

            with torch.no_grad():
                for i in range(0, n_pixels, 512):
                    batch_end = min(i + 512, n_pixels)
                    batch_inputs = inputs_tensor[i:batch_end].to(device)
                    batch_predictions = model(batch_inputs)
                    all_predictions_batch.append(batch_predictions.cpu().numpy())

            predictions = np.concatenate(all_predictions_batch, axis=0)
            if predictions.shape[1] != 3 * n_tau:
                raise ValueError(f"Prediction size mismatch: got {predictions.shape[1]}, expected {3*n_tau}")
            predictions_reshaped = predictions.reshape(n_pixels, n_tau, 3)

            prediction_atm = {}
            for param_idx, param_name in enumerate(['T', 'Vz', 'Bz']):
                param_normalized = predictions_reshaped[:, :, param_idx]
                param_denorm = mhd_normalizer.denormalize(param_normalized, param_name)
                prediction_atm[param_name] = param_denorm.reshape(H, W, n_tau)

            muram_predictions[step][model_name] = {
                'prediction': prediction_atm,
                'label': model_configs[model_name]['label'],
                'color': model_configs[model_name]['color']
            }

            if use_prediction_cache and cache_file is not None:
                np.savez_compressed(cache_file, T=prediction_atm["T"], Vz=prediction_atm["Vz"], Bz=prediction_atm["Bz"])
                print(f"    ✓ Cached predictions: {cache_file}")

            print(f"    ✓ Done. T range: {prediction_atm['T'].min():.1f}-{prediction_atm['T'].max():.1f} K")

    return muram_predictions, logtau


def run_analysis(
    muram_predictions,
    muram_steps,
    model_configs,
    images_save_path,
    logtau,
    plot_this_ods=None,
    skip_existing_plots: bool = True,
):
    """Run all analysis and plotting."""
    if plot_this_ods is None:
        plot_this_ods = [-1.0, -0.8, 0.0]

    analysis = MuramAnalysis()

    for step in muram_predictions.keys():
        step_data = muram_steps[step]
        gt = step_data['mhd']
        step_save_dir = images_save_path / f"step_{step}"
        step_save_dir.mkdir(parents=True, exist_ok=True)

        for od_to_plot in plot_this_ods:
            for param in ['T', 'Vz', 'Bz']:
                for model_name, pred_data in muram_predictions[step].items():
                    filename = f"muram{step}_{pred_data['label'].lower().replace(' ', '_')}_comparison_{param}_logtau_{od_to_plot}.png"
                    if skip_existing_plots and (step_save_dir / filename).exists():
                        print(f"↷ Skip existing: {filename}")
                    else:
                        analysis.plot_prediction_comparison(
                            mean_atm=pred_data['prediction'],
                            ground_truth=gt,
                            mag_to_plot=param,
                            od_to_plot=od_to_plot,
                            logtau=logtau,
                            model_label=f"{pred_data['label']} (MURAM {step})",
                            figsize=(14, 12),
                            save_dir=step_save_dir,
                            filename=filename
                        )
                        print(f"✓ Step {step} {param} at log(tau)={od_to_plot} - {pred_data['label']}")

                filename = f"model_comparison_{param}_logtau_{od_to_plot:.1f}.png"
                if skip_existing_plots and (step_save_dir / filename).exists():
                    print(f"↷ Skip existing: {filename}")
                else:
                    analysis.compare_models_at_optical_depth(
                        all_predictions=muram_predictions[step],
                        ground_truth=gt,
                        mag_to_plot=param,
                        od_to_plot=od_to_plot,
                        logtau=logtau,
                        figsize=(20, 10),
                        save_dir=step_save_dir,
                        filename=filename
                    )
                    print(f"✓ Step {step} model comparison {param} at log(tau)={od_to_plot:.1f}")

                filename_prefix = f"jointplot_{param}_logtau_{od_to_plot:.1f}"
                if _plot_prefix_exists(step_save_dir, filename_prefix, skip_existing_plots):
                    print(f"↷ Skip existing prefix: {filename_prefix}")
                else:
                    analysis.plot_jointplot_comparison(
                        all_predictions=muram_predictions[step],
                        ground_truth=gt,
                        mag_to_plot=param,
                        od_val=od_to_plot,
                        logtau=logtau,
                        n_samples=10000,
                        kind='reg',
                        save_dir=step_save_dir,
                        filename_prefix=filename_prefix
                    )
                    print(f"✓ Step {step} jointplot {param} at log(tau)={od_to_plot:.1f}")

                filename = f"combined_jointplot_{param}_logtau_{od_to_plot:.1f}.png"
                if skip_existing_plots and (step_save_dir / filename).exists():
                    print(f"↷ Skip existing: {filename}")
                else:
                    analysis.plot_combined_jointplot(
                        all_predictions=muram_predictions[step],
                        ground_truth=gt,
                        mag_to_plot=param,
                        od_val=od_to_plot,
                        logtau=logtau,
                        n_samples=5000,
                        save_dir=step_save_dir,
                        filename=filename
                    )
                    print(f"✓ Step {step} combined jointplot {param}")

                filename = f"error_analysis_{param}_logtau_{od_to_plot:.1f}.png"
                if skip_existing_plots and (step_save_dir / filename).exists():
                    print(f"↷ Skip existing: {filename}")
                else:
                    analysis.analyze_error_by_magnitude(
                        all_predictions=muram_predictions[step],
                        ground_truth=gt,
                        mag_to_analyze=param,
                        od_val=od_to_plot,
                        logtau=logtau,
                        n_bins=20,
                        plot_counts=False,
                        use_absolute=False,
                        rrmse_ylim=(0, 100),
                        save_dir=step_save_dir,
                        filename=filename
                    )
                    print(f"✓ Step {step} error analysis {param}")

        for model_name, pred_data in muram_predictions[step].items():
            filename = f"{pred_data['label'].lower().replace(' ', '_')}_mean_vs_optical_depth.png"
            if skip_existing_plots and (step_save_dir / filename).exists():
                print(f"↷ Skip existing: {filename}")
                continue
            print(f"\n{'='*80}")
            print(f"Model: {pred_data['label']} (Step {step})")
            print(f"{'='*80}")
            analysis.plot_mean_vs_optical_depth(
                mean_atm=pred_data['prediction'],
                logtau=logtau,
                figsize=(18, 6),
                log_scale={'T': False, 'Vz': False, 'Bz': False},
                ylims={'T': (2000, 7000), 'Vz': (-11, 7), 'Bz': (-2000, 2000)},
                ground_truth=gt,
                save_dir=step_save_dir,
                filename=filename
            )
            print(f"✓ Saved mean vs optical depth")


def main(
    plot_ods=None,
    use_cache=True,
    cache_dir='/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.data_cache',
    use_prediction_cache: bool = True,
    skip_existing_plots: bool = True,
    logtau_values: list[float] | None = None,
    logtau_min: float = -2.0,
    logtau_max: float = 0.0,
    logtau_step: float = 0.1,
):
    """Main analysis pipeline."""
    if plot_ods is None:
        plot_ods = [-1.0, -0.8, 0.0]
    
    print("="*80)
    print("MURAM Analysis Pipeline")
    print("="*80)
    print(f"Optical depths to plot: {plot_ods}\n")
    
    images_save_path = setup_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    data_path = Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/data")
    
    # Use shared cache
    cache = None
    if use_cache:
        cache = DataCache(cache_dir=cache_dir, compression='gzip')
        print(f"Using shared MURaM cache: {cache_dir}\n")
    
    mapping_logtau = _resolve_logtau_values(
        logtau_values=logtau_values,
        logtau_min=logtau_min,
        logtau_max=logtau_max,
        logtau_step=logtau_step,
    )
    print(f"MURaM remap logtau grid: {mapping_logtau.tolist()}")

    muram_steps = load_muram_data(data_path, steps=[90], cache=cache, logtau_values=mapping_logtau)
    mhd_normalizer, stokes_normalizer = load_normalizers(data_path, muram_steps)
    muram_inputs = prepare_input_data(muram_steps, device)

    model_configs = get_model_configs()
    models, n_tau = load_all_models(model_configs, device)
    logtau = _resolve_logtau_from_experiment_results(model_configs)

    if mapping_logtau.shape != logtau.shape or not np.allclose(mapping_logtau, logtau, atol=1e-6, rtol=0.0):
        raise ValueError(
            "Remap logtau grid does not match model logtau grid from experiment_results.json. "
            f"Remap={mapping_logtau.tolist()} | Model={logtau.tolist()}. "
            "Pass matching --logtau-values (or --logtau-min/max/step)."
        )

    if n_tau != len(logtau):
        raise ValueError(f"Checkpoint n_tau={n_tau} but results.json has {len(logtau)} logtau values")
    print(f"Using logtau from experiment_results.json: {logtau.tolist()}")

    prediction_cache_dir = (
        Path(cache_dir) / "prediction_cache" / "muram_whole_region"
        if use_prediction_cache else None
    )
    muram_predictions, logtau = run_inference(
        muram_inputs, muram_steps, models, model_configs, mhd_normalizer, logtau,
        prediction_cache_dir=prediction_cache_dir,
        use_prediction_cache=use_prediction_cache,
    )

    run_analysis(
        muram_predictions, muram_steps, model_configs, images_save_path, logtau, plot_ods,
        skip_existing_plots=skip_existing_plots,
    )

    print("\n✓ All analysis complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MURAM Analysis Pipeline")
    parser.add_argument("--od-values", type=float, nargs="+", default=[-1.0, -0.8, 0.0],
                       help="Optical depth values to plot (default: -1.0 -0.8 0.0)")
    parser.add_argument("--no-cache", action="store_true", help="Disable shared MURaM data cache")
    parser.add_argument("--cache-dir", type=str, 
                       default='/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.data_cache',
                       help="Absolute path to shared cache directory")
    parser.add_argument("--no-pred-cache", action="store_true", help="Disable prediction cache")
    parser.add_argument("--overwrite-plots", action="store_true", help="Regenerate plots even if they already exist")
    parser.add_argument(
        "--logtau-values", type=float, nargs="+", default=None,
        help="Explicit remap log(tau) grid (overrides min/max/step)"
    )
    parser.add_argument("--logtau-min", type=float, default=-2.0, help="Min log(tau) for range mode")
    parser.add_argument("--logtau-max", type=float, default=0.0, help="Max log(tau) for range mode")
    parser.add_argument("--logtau-step", type=float, default=0.1, help="Step in log(tau) for range mode")

    args = parser.parse_args()
    main(
        plot_ods=args.od_values,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        use_prediction_cache=not args.no_pred_cache,
        skip_existing_plots=not args.overwrite_plots,
        logtau_values=args.logtau_values,
        logtau_min=args.logtau_min,
        logtau_max=args.logtau_max,
        logtau_step=args.logtau_step,
    )
