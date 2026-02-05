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


def setup_region(y_start=0, y_end=100, x_start=400, x_end=600, region_name="plage"):
    """Setup region boundaries and output directories."""
    images_base_path = Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/images")
    images_save_path = images_base_path / f"modest_analysis/cut_region_{region_name}"
    images_save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Region boundaries:")
    print(f"  Y: {y_start} to {y_end} (height: {y_end - y_start} pixels)")
    print(f"  X: {x_start} to {x_end} (width: {x_end - x_start} pixels)")
    print(f"  Total pixels: {(y_end - y_start) * (x_end - x_start):,}\n")
    
    return images_save_path, (y_start, y_end, x_start, x_end)


def load_and_extract_region(data_path, region_bounds):
    """Load MODEST data and extract specified region."""
    y_start, y_end, x_start, x_end = region_bounds
    
    modest = ModestData()
    modest.load_all(apply_mask=False)
    print("✓ Full MODEST data loaded")
    print(f"  Full continuum shape: {modest.continuum.shape}")
    
    region_data = modest.extract_region(y_start, y_end, x_start, x_end)
    print(f"\n✓ Region extracted successfully")
    print(f"  Region continuum shape: {region_data['continuum'].shape}")
    print(f"  Region Stokes I shape: {region_data['obs_stokes']['I'].shape}\n")
    
    return modest, region_data


def visualize_region(modest, region_data, region_bounds, images_save_path):
    """Visualize the extracted region."""
    y_start, y_end, x_start, x_end = region_bounds
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(modest.continuum, cmap='gray', origin='lower')
    axes[0].set_title("Full FOV with Selected Region", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("X [pixels]")
    axes[0].set_ylabel("Y [pixels]")
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                    linewidth=3, edgecolor='red', facecolor='none', label='Selected Region')
    axes[0].add_patch(rect)
    axes[0].legend(fontsize=11)
    
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)
    
    im1 = axes[1].imshow(region_data['continuum'], cmap='gray', origin='lower')
    axes[1].set_title(f"Extracted Region ({x_end-x_start}×{y_end-y_start} pixels)", 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel("X [pixels]")
    axes[1].set_ylabel("Y [pixels]")
    
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    plt.tight_layout()
    plt.savefig(images_save_path / "extracted_region_continuum.png", dpi=300)
    plt.close()
    print(f"Region intensity range: {np.nanmin(region_data['continuum']):.1f} - {np.nanmax(region_data['continuum']):.1f}\n")


def load_normalizers(data_path, region_data):
    """Load normalizers and normalize region Stokes data."""
    mhd_normalizer = MhdNormalizer()
    mhd_normalizer.load(data_path / "normalization_stats/mhd_normalization.json")
    
    stokes_normalizer = StokesNormalizer()
    stokes_normalizer.load(data_path / "normalization_stats/stokes_normalization.json")
    
    normalized_stokes = stokes_normalizer.transform(region_data['obs_stokes'])
    print("✓ Normalizers loaded and region Stokes data normalized\n")
    
    return mhd_normalizer, stokes_normalizer, normalized_stokes


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
    """Prepare input tensor for region."""
    I_t = normalized_stokes["I"]
    V_t = normalized_stokes["V"]
    inputs = np.stack([I_t, V_t], axis=2)
    H_region, W_region, Nstokes, Nlambda = inputs.shape
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).permute(0, 1, 3, 2)
    inputs_tensor = inputs_tensor.reshape(H_region*W_region, Nstokes, Nlambda).to(device)
    
    print(f"Region input tensor shape: {inputs_tensor.shape}")
    print(f"Total pixels in region: {H_region*W_region:,}")
    print(f"Wavelength points: {Nlambda}\n")
    
    return inputs_tensor, (H_region, W_region, Nstokes, Nlambda)


def run_inference(inputs_tensor, shape, models, model_configs, mhd_normalizer):
    """Run inference for all models on region."""
    all_predictions_region = {}
    logtau = np.arange(-2, 0.1, 0.1)
    H_region, W_region, Nstokes, Nlambda = shape
    
    print("="*70)
    for model_name, model in models.items():
        print(f"\nRunning inference for {model_configs[model_name]['label']} on region...")
        mean_atm, std_atm = model.run_inference_with_uncertainty(
            inputs=inputs_tensor,
            mhd_normalizer=mhd_normalizer,
            batch_size=512,
            stochastic_steps=30,
            H=H_region,
            W=W_region,
            n_heights=21,
            verbose=True
        )
        all_predictions_region[model_name] = {
            'mean': mean_atm,
            'std': std_atm,
            'label': model_configs[model_name]['label'],
            'color': model_configs[model_name]['color']
        }
    print("\n" + "="*70)
    print("✓ All model inferences complete for the region\n")
    
    return all_predictions_region, logtau


def run_analysis(all_predictions_region, region_data, modest, model_configs, region_bounds, images_save_path):
    """Run all analysis and plotting."""
    analysis = ModestAnalysis()
    modest_logtau = list(modest.spinor_atm["T"].keys())
    logtau = np.arange(-2, 0.1, 0.1)
    
    for od_to_plot in modest_logtau:
        for param in ['T', 'Vz', 'Bz']:
            # Single model comparisons
            for model_name, pred_data in all_predictions_region.items():
                filename = f"{pred_data['label'].lower().replace(' ', '_')}_comparison_{param}_logtau_{od_to_plot}.png"
                analysis.plot_prediction_comparison(
                    mean_atm=pred_data['mean'],
                    std_atm=pred_data['std'],
                    ground_truth=region_data['spinor_atm'],
                    mag_to_plot=param,
                    od_to_plot=od_to_plot,
                    logtau=logtau,
                    model_label=f"{pred_data['label']} (Region)",
                    figsize=(14, 20),
                    save_dir=images_save_path,
                    filename=filename
                )
            
            # Multi-model comparison
            filename = f"model_comparison_{param}_logtau_{od_to_plot}.png"
            analysis.compare_models_at_optical_depth(
                all_predictions=all_predictions_region,
                ground_truth=region_data['spinor_atm'],
                mag_to_plot=param,
                od_to_plot=od_to_plot,
                logtau=logtau,
                figsize=(20, 10),
                save_dir=images_save_path,
                filename=filename
            )
            
            # Joint plots
            filename_prefix = f"jointplot_{param}_logtau_{od_to_plot}"
            analysis.plot_jointplot_comparison(
                all_predictions=all_predictions_region,
                ground_truth=region_data['spinor_atm'],
                mag_to_plot=param,
                od_val=od_to_plot,
                logtau=logtau,
                n_samples=5000,
                kind='reg',
                save_dir=images_save_path,
                filename_prefix=filename_prefix
            )
            
            # Combined jointplot
            filename = f"combined_jointplot_{param}_logtau_{od_to_plot}.png"
            analysis.plot_combined_jointplot(
                all_predictions=all_predictions_region,
                ground_truth=region_data['spinor_atm'],
                mag_to_plot=param,
                od_val=od_to_plot,
                logtau=logtau,
                n_samples=3000,
                save_dir=images_save_path,
                filename=filename
            )
            
            # Error analysis
            filename = f"error_analysis_{param}_logtau_{od_to_plot}.png"
            analysis.analyze_error_by_magnitude(
                all_predictions=all_predictions_region,
                ground_truth=region_data['spinor_atm'],
                mag_to_analyze=param,
                od_val=od_to_plot,
                logtau=logtau,
                n_bins=15,
                plot_counts=True,
                use_absolute=False,
                rrmse_ylim=(0, 100),
                save_dir=images_save_path,
                filename=filename
            )
            
            # Uncertainty analysis
            filename = f"uncertainty_vs_error_{param}_logtau_{od_to_plot}.png"
            analysis.plot_uncertainty_vs_error(
                all_predictions=all_predictions_region,
                ground_truth=region_data['spinor_atm'],
                mag_to_plot=param,
                od_val=od_to_plot,
                logtau=logtau,
                save_dir=images_save_path,
                filename=filename
            )
            print(f"✓ {param} at log(tau)={od_to_plot}")
    
    # Vertical profile analysis
    for model_name, pred_data in all_predictions_region.items():
        print(f"\n{'='*80}")
        print(f"Model: {pred_data['label']} (Region Analysis)")
        print(f"{'='*80}")
        filename = f"{pred_data['label'].lower().replace(' ', '_')}_mean_vs_optical_depth.png"
        analysis.plot_mean_vs_optical_depth(
            mean_atm=pred_data['mean'],
            std_atm=pred_data['std'],
            logtau=logtau,
            figsize=(18, 6),
            log_scale={'T': False, 'Vz': False, 'Bz': False},
            ylims={'T': (2000, 7000), 'Vz': (-11, 7), 'Bz': (-2000, 2000)},
            ground_truth=region_data['spinor_atm'],
            save_dir=images_save_path,
            filename=filename
        )
        print(f"✓ Saved mean vs optical depth")


def print_region_statistics(all_predictions_region, region_data, region_bounds, logtau):
    """Print region-specific statistics."""
    y_start, y_end, x_start, x_end = region_bounds
    H_region = y_end - y_start
    W_region = x_end - x_start
    
    print("\n" + "="*80)
    print(f"REGION SUMMARY STATISTICS")
    print(f"Region: Y=[{y_start}:{y_end}], X=[{x_start}:{x_end}]")
    print(f"Size: {H_region}×{W_region} pixels ({H_region*W_region:,} total)")
    print("="*80)
    
    for param in ['T', 'Vz', 'Bz']:
        print(f"\n{param}:")
        print("-"*80)
        gt_key = {'T': 'T', 'Vz': 'Vlos', 'Bz': 'Blos'}[param]
        for od_val in [-2.0, -0.8, 0.0]:
            if od_val in region_data['spinor_atm'][gt_key]:
                od_idx = np.argmin(np.abs(logtau - od_val))
                gt = region_data['spinor_atm'][gt_key][od_val]
                print(f"\n  log(τ) = {od_val:.1f}:")
                print(f"    Ground Truth:  mean={np.mean(gt):.2f}, std={np.std(gt):.2f}")
                for model_name, pred_data in all_predictions_region.items():
                    pred_mean = pred_data['mean'][param][:, :, od_idx]
                    pred_std_map = pred_data['std'][param][:, :, od_idx]
                    diff = pred_mean - gt
                    rmse = np.sqrt(np.mean(diff**2))
                    bias = np.mean(diff)
                    corr, _ = pearsonr(pred_mean.flatten(), gt.flatten())
                    mean_uncertainty = np.mean(pred_std_map)
                    print(f"    {pred_data['label']:15s}: "
                          f"R={corr:.3f}, RMSE={rmse:.2f}, "
                          f"Bias={bias:+.2f}, σ_pred={mean_uncertainty:.2f}")
    print("\n" + "="*80)


def save_results(all_predictions_region, region_bounds):
    """Save predictions to disk."""
    y_start, y_end, x_start, x_end = region_bounds
    output_dir = Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/region_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    region_name = f"region_y{y_start}-{y_end}_x{x_start}-{x_end}"
    for model_name, pred_data in all_predictions_region.items():
        model_output_dir = output_dir / model_name / region_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        for param in ['T', 'Vz', 'Bz']:
            np.save(model_output_dir / f"{param}_mean.npy", pred_data['mean'][param])
            np.save(model_output_dir / f"{param}_std.npy", pred_data['std'][param])
        print(f"✓ Saved predictions for {pred_data['label']} to {model_output_dir}")
    print(f"\n✓ All region predictions saved to {output_dir}")


def main(y_start=0, y_end=100, x_start=400, x_end=600, region_name="plage", visualization_only=False):
    """Main analysis pipeline.
    
    Parameters
    ----------
    y_start : int
        Starting y coordinate of region
    y_end : int
        Ending y coordinate of region
    x_start : int
        Starting x coordinate of region
    x_end : int
        Ending x coordinate of region
    region_name : str
        Name of the region (used for output directory)
    visualization_only : bool
        If True, only visualize the region and exit without running analysis
    """
    print("="*80)
    print(f"MODEST Regional Analysis Pipeline ({region_name})")
    print("="*80 + "\n")
    
    # Setup
    images_save_path, region_bounds = setup_region(y_start, y_end, x_start, x_end, region_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    data_path = Path("/scratchsan/observatorio/juagudeloo/data")
    
    # Load data
    modest, region_data = load_and_extract_region(data_path, region_bounds)
    visualize_region(modest, region_data, region_bounds, images_save_path)
    
    # Exit early if visualization only
    if visualization_only:
        print("\n" + "="*80)
        print("✓ Region visualization complete")
        print(f"✓ Visualization saved to: {images_save_path}")
        print("="*80)
        print("\nTo run full analysis, set visualization_only=False")
        return
    
    # Continue with full analysis
    print("\nProceeding with full analysis...\n")
    
    mhd_normalizer, stokes_normalizer, normalized_stokes = load_normalizers(data_path, region_data)
    
    # Load models
    model_configs = get_model_configs()
    models = load_all_models(model_configs, device)
    
    # Prepare inputs and run inference
    inputs_tensor, shape = prepare_input_data(normalized_stokes, device)
    all_predictions_region, logtau = run_inference(inputs_tensor, shape, models, model_configs, mhd_normalizer)
    
    # Run analysis
    run_analysis(all_predictions_region, region_data, modest, model_configs, region_bounds, images_save_path)
    
    # Print statistics and save results
    print_region_statistics(all_predictions_region, region_data, region_bounds, logtau)
    save_results(all_predictions_region, region_bounds)
    
    print("\n✓ All analysis complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MODEST Regional Analysis")
    parser.add_argument("--y-start", type=int, default=0, help="Starting y coordinate")
    parser.add_argument("--y-end", type=int, default=100, help="Ending y coordinate")
    parser.add_argument("--x-start", type=int, default=400, help="Starting x coordinate")
    parser.add_argument("--x-end", type=int, default=600, help="Ending x coordinate")
    parser.add_argument("--region-name", type=str, default="plage", help="Region name for output directory")
    parser.add_argument("--visualization-only", action="store_true", 
                        help="Only visualize region without running analysis")
    
    args = parser.parse_args()
    
    main(
        y_start=args.y_start,
        y_end=args.y_end,
        x_start=args.x_start,
        x_end=args.x_end,
        region_name=args.region_name,
        visualization_only=args.visualization_only
    )
