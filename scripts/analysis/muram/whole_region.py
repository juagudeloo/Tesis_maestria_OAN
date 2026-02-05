import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import sys
import pandas as pd

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


def load_muram_data(data_path, steps=[80, 95, 195]):
    """Load MURAM MHD and Stokes data for specified steps."""
    muram_steps = {}
    for step in steps:
        print(f"\nLoading MURAM step {step}...")
        
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
        stokes.add_hinode_noise()
        
        muram_steps[step] = {
            'mhd': mhd.od_data,
            'stokes': stokes.data
        }
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
    return {
        'no_physics_60_to_100': {
            'path': '/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments/physics_regularization_ablation_60_to_100/no_physics/final_model.pth',
            'use_physics': None,
            'lambda_wfa': 0.0,
            'lambda_doppler': 0.0,
            'lambda_temp': 0.0,
            'lambda_physics': 0.0,
            'label': 'No Physics 60 to 100',
            'color': 'blue'
        },
        'wfa_only_60_to_100': {
            'path': '/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments/physics_regularization_ablation_60_to_100/wfa_only/final_model.pth',
            'use_physics': 'wfa',
            'lambda_wfa': 0.01,
            'lambda_doppler': 0.0,
            'lambda_temp': 0.0,
            'lambda_physics': 0.0,
            'label': 'WFA Only 60 to 100',
            'color': 'orange'
        },
        'all_physics_60_to_100': {
            'path': '/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments/physics_regularization_ablation_60_to_100/all_physics_terms/final_model.pth',
            'use_physics': 'wfa',
            'lambda_wfa': 0.01,
            'lambda_doppler': 0.0,
            'lambda_temp': 0.0,
            'lambda_physics': 0.02,
            'label': 'All Physics 60 to 100',
            'color': 'green'
        },
        "no_physics_100_to_200": {
            'path': '/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments/physics_regularization_ablation_100_to_200/no_physics/final_model.pth',
            'use_physics': None,
            'lambda_wfa': 0.0,
            'lambda_doppler': 0.0,
            'lambda_temp': 0.0,
            'lambda_physics': 0.0,
            'label': 'No Physics 100 to 200',
            'color': 'blue'
        },
        'wfa_only_100_to_200': {
            'path': '/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments/physics_regularization_ablation_100_to_200/wfa_only/final_model.pth',
            'use_physics': 'wfa',
            'lambda_wfa': 0.01,
            'lambda_doppler': 0.0,
            'lambda_temp': 0.0,
            'lambda_physics': 0.0,
            'label': 'WFA Only 100 to 200',
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
            mean_atm, std_atm = model.run_inference_with_uncertainty(
                inputs=inputs_tensor,
                mhd_normalizer=mhd_normalizer,
                batch_size=512,
                stochastic_steps=30,
                H=H,
                W=W,
                n_heights=21,
                verbose=False
            )
            muram_predictions[step][model_name] = {
                'mean': mean_atm,
                'std': std_atm,
                'label': model_configs[model_name]['label'],
                'color': model_configs[model_name]['color']
            }
            print(f"    ✓ Done. T range: {mean_atm['T'].min():.1f}-{mean_atm['T'].max():.1f} K")
    
    return muram_predictions, logtau


def run_analysis(muram_predictions, muram_steps, model_configs, images_save_path, plot_this_ods=None):
    """Run all analysis and plotting.
    
    Parameters
    ----------
    plot_this_ods : list, optional
        Optical depth values to plot (default: [-1.0, -0.8, 0.0])
    """
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
                    try:
                        analysis.plot_prediction_comparison(
                            mean_atm=pred_data['mean'],
                            std_atm=pred_data['std'],
                            ground_truth=gt,
                            mag_to_plot=param,
                            od_to_plot=od_to_plot,
                            logtau=logtau,
                            model_label=f"{pred_data['label']} (MURAM {step})",
                            figsize=(14, 20),
                            save_dir=images_save_path / f"step_{step}",
                            filename=filename
                        )
                        print(f"✓ Step {step} {param} at log(tau)={od_to_plot} - {pred_data['label']}")
                    except Exception as e:
                        print(f"✗ {e}")
                
                # Multi-model comparison
                filename = f"model_comparison_{param}_logtau_{od_to_plot:.1f}.png"
                try:
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
                except Exception as e:
                    print(f"Error: {e}")
                
                # Joint plots
                try:
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
                except Exception as e:
                    print(f"Error: {e}")
                
                # Combined jointplot
                filename = f"combined_jointplot_{param}_logtau_{od_to_plot:.1f}.png"
                try:
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
                except Exception as e:
                    print(f"Error: {e}")
                
                # Error analysis
                filename = f"error_analysis_{param}_logtau_{od_to_plot:.1f}.png"
                try:
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
                except Exception as e:
                    print(f"Error: {e}")
                
                # Uncertainty analysis
                filename = f"uncertainty_vs_error_{param}_logtau_{od_to_plot:.1f}.png"
                try:
                    analysis.plot_uncertainty_vs_error(
                        all_predictions=muram_predictions[step],
                        ground_truth=gt,
                        mag_to_plot=param,
                        od_val=od_to_plot,
                        logtau=logtau,
                        save_dir=images_save_path / f"step_{step}",
                        filename=filename
                    )
                    print(f"✓ Step {step} uncertainty vs error {param}")
                except Exception as e:
                    print(f"Error: {e}")
        
        # Vertical profile analysis (once per step)
        for model_name, pred_data in muram_predictions[step].items():
            filename = f"{pred_data['label'].lower().replace(' ', '_')}_mean_vs_optical_depth.png"
            print(f"\n{'='*80}")
            print(f"Model: {pred_data['label']} (Step {step})")
            print(f"{'='*80}")
            analysis.plot_mean_vs_optical_depth(
                mean_atm=pred_data['mean'],
                std_atm=pred_data['std'],
                logtau=logtau,
                figsize=(18, 6),
                log_scale={'T': False, 'Vz': False, 'Bz': False},
                ylims={'T': (2000, 7000), 'Vz': (-11, 7), 'Bz': (-2000, 2000)},
                ground_truth=gt,
                save_dir=images_save_path / f"step_{step}",
                filename=filename
            )
            print(f"✓ Saved mean vs optical depth")


def main(plot_ods=None):
    """Main analysis pipeline.
    
    Parameters
    ----------
    plot_ods : list, optional
        List of optical depth values to plot (default: [-1.0, -0.8, 0.0])
    """
    if plot_ods is None:
        plot_ods = [-1.0, -0.8, 0.0]
    
    print("="*80)
    print("MURAM Analysis Pipeline")
    print("="*80)
    print(f"Optical depths to plot: {plot_ods}\n")
    
    # Setup
    images_save_path = setup_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    data_path = Path("/scratchsan/observatorio/juagudeloo/data")
    
    # Load data
    muram_steps = load_muram_data(data_path)
    mhd_normalizer, stokes_normalizer = load_normalizers(data_path, muram_steps)
    
    # Load models
    model_configs = get_model_configs()
    models = load_all_models(model_configs, device)
    
    # Prepare inputs and run inference
    muram_inputs = prepare_input_data(muram_steps, device)
    muram_predictions, logtau = run_inference(muram_inputs, muram_steps, models, model_configs, mhd_normalizer)
    
    # Run analysis with specified optical depths
    run_analysis(muram_predictions, muram_steps, model_configs, images_save_path, plot_ods)
    
    print("\n✓ All analysis complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MURAM Analysis Pipeline")
    parser.add_argument(
        "--od-values", 
        type=float, 
        nargs="+", 
        default=[-1.0, -0.8, 0.0],
        help="Optical depth values to plot (default: -1.0 -0.8 0.0)"
    )
    
    args = parser.parse_args()
    main(plot_ods=args.od_values)
