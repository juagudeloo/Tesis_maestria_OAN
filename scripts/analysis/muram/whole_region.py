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
    images_save_path = images_base_path / "muram_analysis"
    images_save_path.mkdir(parents=True, exist_ok=True)
    return images_save_path


def _shared_cache_signature() -> dict:
    # Match scripts/base_training.py defaults used for cache hash.
    return {
        "nx": 480,
        "ny": 480,
        "nz": 256,
        "z_max": 250,
        "dz_km": 10.0,
        "central_wavelength": 6301.5,
        "wl_range": (15, 60),
    }


def load_muram_data(data_path, steps=[80, 95, 195], cache: DataCache | None = None):
    """Load MURAM MHD and Stokes data for specified steps, using shared DataCache."""
    muram_steps = {}
    config_hash = DataCache.make_config_hash(_shared_cache_signature()) if cache else None

    for step in steps:
        print(f"\nLoading MURAM step {step}...")

        if cache is not None and cache.exists(step, config_hash):
            stokes_data, mhd_data, _ = cache.load_raw(step=step, verbose=True)
            muram_steps[step] = {'mhd': mhd_data, 'stokes': stokes_data}
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
        new_logtau = np.arange(-2.0, 0.1, 0.1)
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
    no_physics_weights = "no_physics/final_model.pth"
    wfa_only_weights = "wfa_only/final_model.pth"
    doppler_only_weights = "doppler_only/final_model.pth"
    black_body_only_weights = "black_body_only/final_model.pth"
    all_physics_terms_weights = "all_physics_terms/final_model.pth"
    return {
        'no_physics_80_to_113-fixed_lambdas': {
            'path': base_model_path / 'physics_regularization_ablation_80_to_113-fixed_lambdas' / no_physics_weights,
            'use_physics': None,
            'lambda_wfa': 0.0,
            'lambda_doppler': 0.0,
            'lambda_temp': 0.0,
            'label': 'No Physics 80 to 113',
            'color': 'blue'
        },
        'wfa_only_80_to_113-fixed_lambdas': {
            'path': base_model_path / 'physics_regularization_ablation_80_to_113-fixed_lambdas' / wfa_only_weights,
            'use_physics': ['wfa'],
            'lambda_wfa': 1.0,
            'lambda_doppler': 0.0,
            'lambda_temp': 0.0,
            'label': 'WFA Only 80 to 113',
            'color': 'orange'
        },
    }


def load_model(config, device):
    """Load a trained PINN-MSCNN model."""
    model = PhysicsInformedMSCNN(
        scales=[1, 2, 3],
        in_channels=2,
        c1_filters=16,
        c2_filters=32,
        kernel_size=5,
        pool_size=2,
        n_linear_layers=4,
        output_features=3*21,
        input_length=112,
        dropout_rate=0.2,
    ).to(device)
    checkpoint = torch.load(config['path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_all_models(model_configs, device):
    """Load all trained models."""
    models = {}
    for name, config in model_configs.items():
        print(f"Loading {config['label']}...")
        models[name] = load_model(config, device)
    print(f"✓ Loaded {len(models)} models\n")
    return models


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


def _cache_fingerprint(model_name, model_config, shape, step: int) -> str:
    model_path = Path(model_config["path"])
    payload = {
        "model": model_name,
        "model_path": str(model_path),
        "model_mtime": model_path.stat().st_mtime if model_path.exists() else None,
        "shape": tuple(int(v) for v in shape),
        "step": int(step),
        "dataset": "muram_whole_region",
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _prediction_cache_file(cache_dir: Path | None = None, step: int, model_name: str, fingerprint: str) -> Path:
    if cache_dir is not None:
        step_dir = cache_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir / f"{model_name}_{fingerprint}.npz"
    return Path()


def _plot_exists(save_dir: Path, filename: str, skip_existing: bool) -> bool:
    return skip_existing and (save_dir / filename).exists()


def run_inference(muram_inputs, muram_steps, models, model_configs, mhd_normalizer):
    """Run inference for all models and steps."""
    muram_predictions = {}
    logtau = np.arange(-2, 0.1, 0.1)

    for step, muram in muram_inputs.items():
        H, W, Nstokes, Nlambda = muram['shape']
        inputs_tensor = muram['tensor']
        muram_predictions[step] = {}
        print(f"\n=== Running inference for MURAM step {step} ===")

        for model_name, model in models.items():
            print(f"  Model: {model_configs[model_name]['label']}")
            
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
            predictions_reshaped = predictions.reshape(n_pixels, 21, 3)

            prediction_atm = {}
            param_names = ['T', 'Vz', 'Bz']

            for param_idx, param_name in enumerate(param_names):
                param_normalized = predictions_reshaped[:, :, param_idx]
                param_denorm = mhd_normalizer.denormalize(param_normalized, param_name)
                prediction_atm[param_name] = param_denorm.reshape(H, W, 21)

            muram_predictions[step][model_name] = {
                'prediction': prediction_atm,
                'label': model_configs[model_name]['label'],
                'color': model_configs[model_name]['color']
            }

            print(f"    ✓ Done. T range: {prediction_atm['T'].min():.1f}-{prediction_atm['T'].max():.1f} K")

    return muram_predictions, logtau


def run_analysis(muram_predictions, muram_steps, model_configs, images_save_path, plot_this_ods=None):
    """Run all analysis and plotting."""
    if plot_this_ods is None:
        plot_this_ods = [-1.0, -0.8, 0.0]

    analysis = MuramAnalysis()
    logtau = np.arange(-2, 0.1, 0.1)

    for step in muram_predictions.keys():
        step_data = muram_steps[step]
        gt = step_data['mhd']

        for od_to_plot in plot_this_ods:
            for param in ['T', 'Vz', 'Bz']:
                for model_name, pred_data in muram_predictions[step].items():
                    # Single model comparison
                    filename = f"muram{step}_{pred_data['label'].lower().replace(' ', '_')}_comparison_{param}_logtau_{od_to_plot}.png"
                    analysis.plot_prediction_comparison(
                        mean_atm=pred_data['prediction'],
                        ground_truth=gt,
                        mag_to_plot=param,
                        od_to_plot=od_to_plot,
                        logtau=logtau,
                        model_label=f"{pred_data['label']} (MURAM {step})",
                        figsize=(14, 12),
                        save_dir=images_save_path / f"step_{step}",
                        filename=filename
                    )
                    print(f"✓ Step {step} {param} at log(tau)={od_to_plot} - {pred_data['label']}")
                
                # Multi-model comparison
                filename = f"model_comparison_{param}_logtau_{od_to_plot:.1f}.png"
                analysis.compare_models_at_optical_depth(
                    all_predictions=muram_predictions[step],
                    ground_truth=gt,
                    mag_to_plot=param,
                    od_to_plot=od_to_plot,
                    logtau=logtau,
                    figsize=(20, 10),
                    save_dir=images_save_path / f"step_{step}",
                    filename=filename
                )
                print(f"✓ Step {step} model comparison {param} at log(tau)={od_to_plot:.1f}")
                
                # Joint plots
                filename_prefix = f"jointplot_{param}_logtau_{od_to_plot:.1f}"
                analysis.plot_jointplot_comparison(
                    all_predictions=muram_predictions[step],
                    ground_truth=gt,
                    mag_to_plot=param,
                    od_val=od_to_plot,
                    logtau=logtau,
                    n_samples=10000,
                    kind='reg',
                    save_dir=images_save_path / f"step_{step}",
                    filename_prefix=filename_prefix
                )
                print(f"✓ Step {step} jointplot {param} at log(tau)={od_to_plot:.1f}")
                
                # Combined jointplot
                filename = f"combined_jointplot_{param}_logtau_{od_to_plot:.1f}.png"
                analysis.plot_combined_jointplot(
                    all_predictions=muram_predictions[step],
                    ground_truth=gt,
                    mag_to_plot=param,
                    od_val=od_to_plot,
                    logtau=logtau,
                    n_samples=5000,
                    save_dir=images_save_path / f"step_{step}",
                    filename=filename
                )
                print(f"✓ Step {step} combined jointplot {param}")
                
                # Error analysis
                filename = f"error_analysis_{param}_logtau_{od_to_plot:.1f}.png"
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
                    save_dir=images_save_path / f"step_{step}",
                    filename=filename
                )
                print(f"✓ Step {step} error analysis {param}")
        
        # Vertical profile analysis (once per step)
        for model_name, pred_data in muram_predictions[step].items():
            filename = f"{pred_data['label'].lower().replace(' ', '_')}_mean_vs_optical_depth.png"
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
                save_dir=images_save_path / f"step_{step}",
                filename=filename
            )
            print(f"✓ Saved mean vs optical depth")


def main(plot_ods=None, use_cache=True, cache_dir='/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.data_cache'):
    """Main analysis pipeline.
    
    Parameters
    ----------
    plot_ods : list, optional
        List of optical depth values to plot (default: [-1.0, -0.8, 0.0])
    use_cache : bool
        Enable shared MURaM data cache
    cache_dir : str
        Absolute path to cache directory
    """
    if plot_ods is None:
        plot_ods = [-1.0, -0.8, 0.0]
    
    print("="*80)
    print("MURAM Analysis Pipeline")
    print("="*80)
    print(f"Optical depths to plot: {plot_ods}\n")
    
    images_save_path = setup_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    data_path = Path("/scratchsan/observatorio/juagudeloo/data")
    
    # Use shared cache
    cache = None
    if use_cache:
        cache = DataCache(cache_dir=cache_dir, compression='gzip')
        print(f"Using shared MURaM cache: {cache_dir}\n")
    
    muram_steps = load_muram_data(data_path, steps=[90], cache=cache)
    mhd_normalizer, stokes_normalizer = load_normalizers(data_path, muram_steps)
    
    model_configs = get_model_configs()
    models = load_all_models(model_configs, device)
    
    muram_inputs = prepare_input_data(muram_steps, device)
    muram_predictions, logtau = run_inference(muram_inputs, muram_steps, models, model_configs, mhd_normalizer)
    
    run_analysis(muram_predictions, muram_steps, model_configs, images_save_path, plot_ods)
    
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
    
    args = parser.parse_args()
    main(plot_ods=args.od_values, use_cache=not args.no_cache, cache_dir=args.cache_dir)
