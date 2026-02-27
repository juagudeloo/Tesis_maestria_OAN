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
    images_save_path = images_base_path / "modest_analysis/whole_region"
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
    no_physics_weights = "no_physics/final_model.pth"
    wfa_only_weights = "wfa_only/final_model.pth"
    doppler_only_weights = "doppler_only/final_model.pth"
    black_body_only_weights = "black_body_only/final_model.pth"
    all_physics_terms_weights = "all_physics_terms/final_model.pth"
    return {
        'no_physics_80_to_113': {
            'path': base_model_path / 'experiment_80_to_113' / no_physics_weights,
            'use_physics': None,
            'lambda_wfa': 0.0,
            'lambda_doppler': 0.0,
            'lambda_temp': 0.0,
            'label': 'No Physics 80 to 113',
            'color': 'blue'
        },
        'wfa_only_80_to_113': {
            'path': base_model_path / 'experiment_80_to_113' / wfa_only_weights,
            'use_physics': ['wfa'],
            'lambda_wfa': 1.0,
            'lambda_doppler': 0.0,
            'lambda_temp': 0.0,
            'label': 'WFA Only 80 to 113',
            'color': 'orange'
        },
    }


def load_model(config: Dict, device) -> PhysicsInformedMSCNN:
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
        lambda_wfa=config['lambda_wfa'],
        lambda_doppler=config['lambda_doppler'],
        lambda_temp=config['lambda_temp'],
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
        print(f"  ✓ Model loaded successfully")
    print(f"\n✓ All {len(models)} models loaded\n")
    return models


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


def run_inference(inputs_tensor, shape, models, model_configs, mhd_normalizer):
    """Run inference for all models."""
    all_predictions = {}
    logtau = np.arange(-2, 0.1, 0.1)
    H, W, Nstokes, Nlambda = shape
    
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
        predictions = np.concatenate(all_predictions_batch, axis=0)  # (n_pixels, 63)
        
        # Reshape and denormalize
        predictions_reshaped = predictions.reshape(n_pixels, 21, 3)  # (n_pixels, 21, T/Vz/Bz)
        
        prediction_atm = {}
        param_names = ['T', 'Vz', 'Bz']
        
        for param_idx, param_name in enumerate(param_names):
            param_normalized = predictions_reshaped[:, :, param_idx]  # (n_pixels, 21)
            param_denorm = mhd_normalizer.denormalize(param_normalized, param_name)
            prediction_atm[param_name] = param_denorm.reshape(H, W, 21)
        
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


def run_analysis(all_predictions, modest, model_configs, images_save_path, plot_ods=None, skip_existing_plots: bool = True):
    """Run all analysis and plotting.
    
    Parameters
    ----------
    plot_ods : list, optional
        Optical depth values to plot. If None, uses all available.
    """
    if plot_ods is None:
        plot_ods = list(modest.spinor_atm["T"].keys())
    
    analysis = ModestAnalysis()
    logtau = np.arange(-2, 0.1, 0.1)
    
    for od_to_plot in plot_ods:
        for param in ['T', 'Vz', 'Bz']:
            # Single model comparisons
            for model_name, pred_data in all_predictions.items():
                try:
                    filename = f"{pred_data['label'].lower().replace(' ', '_')}_comparison_{param}_logtau_{od_to_plot}.png"
                    analysis.plot_prediction_comparison(
                        mean_atm=pred_data['prediction'],
                        ground_truth=modest.spinor_atm,
                        mag_to_plot=param,
                        od_to_plot=od_to_plot,
                        logtau=logtau,
                        model_label=pred_data['label'],
                        figsize=(14, 12),
                        save_dir=images_save_path,
                        filename=filename
                    )
                    print(f"✓ {param} at log(tau)={od_to_plot} - {pred_data['label']}")
                except Exception as e:
                    print(f"✗ Failed: {e}")
            
            # Multi-model comparison
            filename = f"model_comparison_{param}_logtau_{od_to_plot}.png"
            try:
                analysis.compare_models_at_optical_depth(
                    all_predictions=all_predictions,
                    ground_truth=modest.spinor_atm,
                    mag_to_plot=param,
                    od_to_plot=od_to_plot,
                    logtau=logtau,
                    figsize=(20, 10),
                    save_dir=images_save_path,
                    filename=filename
                )
                print(f"✓ Model comparison {param} at log(tau)={od_to_plot}")
            except Exception as e:
                print(f"Error: {e}")
            
            # Joint plots
            try:
                filename_prefix = f"jointplot_{param}_logtau_{od_to_plot}"
                analysis.plot_jointplot_comparison(
                    all_predictions=all_predictions,
                    ground_truth=modest.spinor_atm,
                    mag_to_plot=param,
                    od_val=od_to_plot,
                    logtau=logtau,
                    n_samples=10000,
                    kind='reg',
                    save_dir=images_save_path,
                    filename_prefix=filename_prefix
                )
                print(f"✓ Jointplot {param}")
            except Exception as e:
                print(f"Error: {e}")
            
            # Combined jointplot
            filename = f"combined_jointplot_{param}_logtau_{od_to_plot}.png"
            try:
                analysis.plot_combined_jointplot(
                    all_predictions=all_predictions,
                    ground_truth=modest.spinor_atm,
                    mag_to_plot=param,
                    od_val=od_to_plot,
                    logtau=logtau,
                    n_samples=5000,
                    save_dir=images_save_path,
                    filename=filename
                )
                print(f"✓ Combined jointplot {param}")
            except Exception as e:
                print(f"Error: {e}")
            
            # Error analysis
            filename = f"error_analysis_{param}_logtau_{od_to_plot}.png"
            try:
                analysis.analyze_error_by_magnitude(
                    all_predictions=all_predictions,
                    ground_truth=modest.spinor_atm,
                    mag_to_analyze=param,
                    od_val=od_to_plot,
                    logtau=logtau,
                    n_bins=20,
                    plot_counts=False,
                    use_absolute=False,
                    rrmse_ylim=(0, 100),
                    save_dir=images_save_path,
                    filename=filename
                )
                print(f"✓ Error analysis {param}")
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


def main(od_values=None):
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
    models = load_all_models(model_configs, device)
    
    # Prepare inputs and run inference
    inputs_tensor, shape = prepare_input_data(normalized_stokes, device)
    all_predictions, logtau = run_inference(inputs_tensor, shape, models, model_configs, mhd_normalizer)
    
    run_analysis(all_predictions, modest, model_configs, images_save_path, od_values)
    
    print("\n✓ All analysis complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MODEST Full Region Analysis Pipeline")
    parser.add_argument("--od-values", type=float, nargs="+", default=None,
                       help="Optical depth values to analyze (default: all available)")
    
    args = parser.parse_args()
    main(od_values=args.od_values)
