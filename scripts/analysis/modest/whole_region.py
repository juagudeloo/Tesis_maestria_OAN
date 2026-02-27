import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import torch
import astropy.units as u
from pathlib import Path
import sys
from scipy.stats import pearsonr
import pandas as pd
from typing import Dict, Tuple, Optional
import json

sys.path.append("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN")
from utils.modest_data import ModestData
from models.pinn_mscnn_model import PhysicsInformedMSCNN
from utils.normalizer import MhdNormalizer, StokesNormalizer
from utils.analysis_functions import ModestAnalysis

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


def setup_paths():
    """Setup output directories."""
    images_base_path = Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/images")
    images_save_path = images_base_path / "analysis/modest/whole_region"
    images_save_path.mkdir(parents=True, exist_ok=True)
    return images_save_path


def load_modest_data(apply_mask=False):
    """Load MODEST Hinode/SP dataset."""
    modest = ModestData()
    modest.load_all(apply_mask=apply_mask)
    print("✓ MODEST data loaded successfully")
    print(f"  Continuum shape: {modest.continuum.shape}")
    print(f"  Stokes I shape: {modest.obs_stokes['I'].shape}")
    print(f"  Wavelength points: {len(modest.wl)}")
    return modest


def load_normalizers(data_path, modest):
    """Load normalizers and normalize Stokes data."""
    mhd_normalizer = MhdNormalizer()
    mhd_normalizer.load(data_path / "normalization_stats/mhd_normalization.json")
    
    stokes_normalizer = StokesNormalizer()
    stokes_normalizer.load(data_path / "normalization_stats/stokes_normalization.json")
    
    normalized_stokes = stokes_normalizer.transform(modest.obs_stokes)
    print("✓ Normalizers loaded and Stokes data normalized")
    return mhd_normalizer, stokes_normalizer, normalized_stokes


def get_model_configs():
    """Return model configurations."""
    base_model_path = Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments/")
    experiment_dir = base_model_path / "experiment_80_to_113"
    results_path = experiment_dir / "experiment_results.json"
    no_physics_weights = "no_physics/final_model.pth"
    wfa_only_weights = "wfa_only/final_model.pth"
    doppler_only_weights = "doppler_only/final_model.pth"
    black_body_only_weights = "black_body_only/final_model.pth"
    all_physics_terms_weights = "all_physics_terms/final_model.pth"
    return {
        'no_physics_80_to_113': {
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
        'wfa_only_80_to_113': {
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

        arr = np.asarray(vals, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"Invalid logtau_values for '{exp_key}' in {results_path}: {vals}")

        resolved[model_name] = np.round(arr, 6)

    ref_name = next(iter(resolved))
    ref = resolved[ref_name]
    for name, arr in resolved.items():
        if arr.shape != ref.shape or not np.allclose(arr, ref, atol=1e-6, rtol=0.0):
            raise ValueError(f"Inconsistent logtau_values between models: {ref_name} vs {name}")
    return ref

def _get_matching_spinor_ods(spinor_od_values, logtau, tol: float = 1e-6):
    return [
        float(od) for od in sorted(float(v) for v in spinor_od_values)
        if np.any(np.isclose(logtau, float(od), atol=tol, rtol=0.0))
    ]

def load_model(config: Dict, device) -> Tuple[PhysicsInformedMSCNN, int]:
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
        lambda_wfa=config['lambda_wfa'],
        lambda_doppler=config['lambda_doppler'],
        lambda_temp=config['lambda_temp'],
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
        print(f"  ✓ Model loaded successfully (n_tau={n_tau})")
    print(f"\n✓ All {len(models)} models loaded\n")
    return models, int(n_tau_ref)

def prepare_input_data(normalized_stokes, device):
    """Prepare input tensor."""
    I_t = normalized_stokes["I"]
    V_t = normalized_stokes["V"]
    inputs = np.stack([I_t, V_t], axis=2)
    H, W, Nstokes, Nlambda = inputs.shape
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).permute(0, 1, 3, 2)
    inputs_tensor = inputs_tensor.reshape(H*W, Nstokes, Nlambda).to(device)
    print(f"Input tensor shape: {inputs_tensor.shape}")
    print(f"Total pixels: {H*W:,}\n")
    return inputs_tensor, (H, W, Nstokes, Nlambda)


def run_inference(inputs_tensor, shape, models, model_configs, mhd_normalizer, logtau):
    """Run inference for all models."""
    all_predictions = {}
    H, W, Nstokes, Nlambda = shape
    n_tau = int(len(logtau))
    
    print("="*70)
    for model_name, model in models.items():
        print(f"\nRunning inference for {model_configs[model_name]['label']}...")
        
        model.eval()
        device = next(model.parameters()).device
        n_pixels = inputs_tensor.shape[0]
        all_predictions_batch = []
        
        with torch.no_grad():
            for i in range(0, n_pixels, 512):
                batch_end = min(i + 512, n_pixels)
                batch_inputs = inputs_tensor[i:batch_end].to(device)
                batch_predictions = model(batch_inputs)  # Direct forward pass
                all_predictions_batch.append(batch_predictions.cpu().numpy())
        
        # Concatenate predictions
        predictions = np.concatenate(all_predictions_batch, axis=0)  # (n_pixels, n_tau, 3)
        if predictions.shape[1] != 3 * n_tau:
            raise ValueError(f"Prediction size mismatch: got {predictions.shape[1]}, expected {3*n_tau}")
        predictions_reshaped = predictions.reshape(n_pixels, n_tau, 3)
        
        # Reshape and denormalize
        predictions_reshaped = predictions_reshaped.reshape(n_pixels, n_tau, 3)
        
        prediction_atm = {}
        param_names = ['T', 'Vz', 'Bz']
        
        for param_idx, param_name in enumerate(param_names):
            param_normalized = predictions_reshaped[:, :, param_idx]  # (n_pixels, n_tau)
            param_denorm = mhd_normalizer.denormalize(param_normalized, param_name)
            prediction_atm[param_name] = param_denorm.reshape(H, W, n_tau)
        
        all_predictions[model_name] = {
            'prediction': prediction_atm,
            'label': model_configs[model_name]['label'],
            'color': model_configs[model_name]['color']
        }
        
        print(f"  T range: {prediction_atm['T'].min():.1f} - {prediction_atm['T'].max():.1f} K")
        print(f"  Vz range: {prediction_atm['Vz'].min():.2f} - {prediction_atm['Vz'].max():.2f} km/s")
        print(f"  Bz range: {prediction_atm['Bz'].min():.2f} - {prediction_atm['Bz'].max():.2f} G")
    
    print("\n" + "="*70)
    print("✓ All model inferences complete\n")
    
    return all_predictions, logtau


def _plot_prefix_exists(save_dir: Path, filename_prefix: str, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    return any(save_dir.rglob(f"{filename_prefix}*.png")) or any(save_dir.rglob(f"{filename_prefix}*.json"))

def run_analysis(all_predictions, modest, model_configs, images_save_path, logtau, plot_ods=None, skip_existing_plots: bool = True):
    """Run all analysis and plotting."""
    spinor_ods = list(modest.spinor_atm["T"].keys())
    matched_ods = _get_matching_spinor_ods(spinor_ods, logtau)
    if not matched_ods:
        raise ValueError(
            f"No SPINOR optical depths match model logtau grid. "
            f"SPINOR={sorted(float(k) for k in spinor_ods)}, model={logtau.tolist()}"
        )

    if plot_ods is None:
        target_ods = matched_ods
    else:
        target_ods = [
            float(od) for od in plot_ods
            if np.any(np.isclose(logtau, float(od), atol=1e-6, rtol=0.0))
        ]
        if not target_ods:
            raise ValueError(f"Requested ODs {plot_ods} are not in model logtau grid {logtau.tolist()}")

    analysis = ModestAnalysis()
    for od_to_plot in target_ods:
        for param in ['T', 'Vz', 'Bz']:
            # Single model comparisons
            for model_name, pred_data in all_predictions.items():
                filename = f"{pred_data['label'].lower().replace(' ', '_')}_comparison_{param}_logtau_{od_to_plot}.png"
                if skip_existing_plots and (images_save_path / filename).exists():
                    print(f"↷ Skip existing: {filename}")
                    continue
                try:
                    # ...existing code...
                    pass
                except Exception as e:
                    print(f"✗ Failed: {e}")

            # Multi-model comparison
            filename = f"model_comparison_{param}_logtau_{od_to_plot}.png"
            if skip_existing_plots and (images_save_path / filename).exists():
                print(f"↷ Skip existing: {filename}")
            else:
                try:
                    # ...existing code...
                    pass
                except Exception as e:
                    print(f"Error: {e}")

            # Joint plots
            filename_prefix = f"jointplot_{param}_logtau_{od_to_plot}"
            if _plot_prefix_exists(images_save_path, filename_prefix, skip_existing_plots):
                print(f"↷ Skip existing prefix: {filename_prefix}")
            else:
                try:
                    # ...existing code...
                    pass
                except Exception as e:
                    print(f"Error: {e}")

            # Combined jointplot
            filename = f"combined_jointplot_{param}_logtau_{od_to_plot}.png"
            if skip_existing_plots and (images_save_path / filename).exists():
                print(f"↷ Skip existing: {filename}")
            else:
                try:
                    # ...existing code...
                    pass
                except Exception as e:
                    print(f"Error: {e}")

            # Error analysis
            filename = f"error_analysis_{param}_logtau_{od_to_plot}.png"
            if skip_existing_plots and (images_save_path / filename).exists():
                print(f"↷ Skip existing: {filename}")
            else:
                try:
                    # ...existing code...
                    pass
                except Exception as e:
                    print(f"Error: {e}")
            
            # Uncertainty analysis
            # filename = f"uncertainty_vs_error_{param}_logtau_{od_to_plot}.png"
            # try:
            #     analysis.plot_uncertainty_vs_error(
            #         all_predictions=all_predictions,
            #         ground_truth=modest.spinor_atm,
            #         mag_to_plot=param,
            #         od_val=od_to_plot,
            #         logtau=logtau,
            #         save_dir=images_save_path,
            #         filename=filename
            #     )
            #     print(f"✓ Uncertainty vs error {param}")
            # except Exception as e:
            #     print(f"Error: {e}")
    
    # Vertical profile analysis
    for model_name, pred_data in all_predictions.items():
        filename = f"{pred_data['label'].lower().replace(' ', '_')}_mean_vs_optical_depth.png"
        if skip_existing_plots and (images_save_path / filename).exists():
            print(f"↷ Skip existing: {filename}")
            continue
        print(f"\n{'='*80}")
        print(f"Model: {pred_data['label']}")
        print(f"{'='*80}")
        analysis.plot_mean_vs_optical_depth(
            mean_atm=pred_data['prediction'],
            logtau=logtau,
            figsize=(18, 6),
            log_scale={'T': False, 'Vz': False, 'Bz': False},
            ylims={'T': (2000, 7000), 'Vz': (-11, 7), 'Bz': (-2000, 2000)},
            ground_truth=modest.spinor_atm,
            save_dir=images_save_path,
            filename=filename
        )
        print(f"✓ Saved mean vs optical depth")


def main(od_values=None, skip_existing_plots: bool = True):
    """Main analysis pipeline.
    
    Parameters
    ----------
    od_values : list, optional
        List of optical depth values to analyze
    """
    print("="*80)
    print("MODEST Full Region Analysis Pipeline")
    print("="*80 + "\n")
    
    if od_values:
        print(f"Analyzing optical depths: {od_values}\n")
    
    # Setup
    images_save_path = setup_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    data_path = Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/data")
    
    # Load data
    modest = load_modest_data(apply_mask=False)
    mhd_normalizer, stokes_normalizer, normalized_stokes = load_normalizers(data_path, modest)
    
    # Load models
    model_configs = get_model_configs()
    models, n_tau = load_all_models(model_configs, device)
    logtau = _resolve_logtau_from_experiment_results(model_configs)
    if n_tau != len(logtau):
        raise ValueError(f"Checkpoint n_tau={n_tau} but results.json has {len(logtau)} logtau values")
    print(f"Using logtau from experiment_results.json: {logtau.tolist()}")

    # Prepare inputs and run inference
    inputs_tensor, shape = prepare_input_data(normalized_stokes, device)
    all_predictions, logtau = run_inference(inputs_tensor, shape, models, model_configs, mhd_normalizer, logtau)

    run_analysis(
        all_predictions, modest, model_configs, images_save_path, logtau, od_values,
        skip_existing_plots=skip_existing_plots,
    )
    
    print("\n✓ All analysis complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MODEST Full Region Analysis Pipeline")
    parser.add_argument("--od-values", type=float, nargs="+", default=None,
                       help="Optical depth values to analyze (default: all available)")
    parser.add_argument("--overwrite-plots", action="store_true", help="Regenerate plots even if they already exist")
    
    args = parser.parse_args()
    main(od_values=args.od_values, skip_existing_plots=not args.overwrite_plots)
