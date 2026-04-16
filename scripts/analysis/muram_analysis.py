import os
import sys
sys.path.append("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/")
import argparse
from pathlib import Path

import torch
from utils.cache_manage import MuramDataCache
from utils.normalizer import MhdNormalizer, StokesNormalizer
from utils.analysis import AnalysisModelPipeline, MuramDiagnosticPlots
from scripts.base_training import TrainingConfig, load_and_prepare_step


# -----------------------------------------------------------------------------
# Main MURaM analysis flow
# - loads trained models
# - loads a single MURaM step
# - runs inference and denormalizes predictions
# - writes diagnostic plots and metrics
# -----------------------------------------------------------------------------
def main(args):
    # Pick GPU when available; otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the reusable analysis pipeline that finds models and manages inference.
    pipeline = AnalysisModelPipeline(
        device=device,
        output_dir=Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/images/analysis/muram"),
        experiment_root=args.experiment_root,
    )
    model_configs, models, n_tau = pipeline.prepare_models(args.model_types)
    
    print(f"Using device: {device}")
    print(f"Number of log(tau) points: {n_tau}")

    # Show which experiment folders were selected for analysis.
    print("Selected model configs:")
    for _, cfg in model_configs.items():
        print(f"  - {cfg['label']} ({cfg['experiment_key']})")

    # Shared cache avoids reloading the same simulation step repeatedly.
    cache = MuramDataCache(cache_dir=args.cache_dir, compression='gzip')
    print(f"Shared MURaM data cache: {args.cache_dir}")
    print("\nInitial Cache Status:")
    cache.print_cache_info()

    # Load the normalization statistics used to convert data back to physical units.
    mhd_normalizer = MhdNormalizer()
    stokes_normalizer = StokesNormalizer()
    default_cfg = TrainingConfig()
    mhd_normalizer.load(filepath=str(Path(default_cfg.data_path) / default_cfg.mhd_normalizer_path))
    stokes_normalizer.load(filepath=str(Path(default_cfg.data_path) / default_cfg.stokes_normalizer_path))

    # Process each trained model independently so each gets its own diagnostics folder.
    for name, model in models.items():
        # Rebuild the training configuration for this model so data loading matches training.
        cfg = pipeline.build_runtime_training_config(model_configs[name])
        model_type = model_configs[name]["experiment_key"]
        print(f"Generating diagnostics for: {model_configs[name]['label']}")

        # Load the requested simulation step and prepare the training-like dataset.
        result = load_and_prepare_step(
            step=args.step_to_plot,
            config=cfg,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            cache=cache,
        )
        if result is None:
            raise RuntimeError(f"Failed to load step {args.step_to_plot} for {model_type}")
        dataset, _ = result
        hinode_wl = getattr(dataset, "hinode_wl", None)

        # Convert model outputs back to physical units for plots and metrics.
        pred_den = pipeline.predict_and_denormalize(
            model=model,
            stokes_input=dataset.stokes_input,
            mhd_normalizer=mhd_normalizer,
            pred_nx=dataset.nx,
            pred_ny=dataset.ny,
        )

        # Denormalize ground-truth labels so prediction-vs-truth plots are comparable.
        gt_norm = dataset.mhd_targets
        if gt_norm.ndim != 2 or gt_norm.shape[1] != 3 * n_tau:
            raise ValueError(
                f"Expected dataset.mhd_targets shape (N, {3 * n_tau}), got {gt_norm.shape}"
            )
        gt_den = {
            "T": mhd_normalizer.denormalize(gt_norm[:, :n_tau], param="T").reshape(dataset.nx, dataset.ny, n_tau),
            "Vz": mhd_normalizer.denormalize(gt_norm[:, n_tau:2 * n_tau], param="Vz").reshape(dataset.nx, dataset.ny, n_tau),
            "Bz": mhd_normalizer.denormalize(gt_norm[:, 2 * n_tau:3 * n_tau], param="Bz").reshape(dataset.nx, dataset.ny, n_tau),
        }

        # The plotter writes the final images and summary metrics for this model.
        plotter = MuramDiagnosticPlots(
            config=cfg,
            model_name=model_type,
            step=args.step_to_plot,
            output_dir=Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/images/analysis/muram"),
            stokes_normalizer=stokes_normalizer,
        )
        plotter.generate(
            pred_den=pred_den,
            gt_den=gt_den,
            stokes_input=dataset.stokes_input,
            wavelengths=hinode_wl,
        )

if __name__ == "__main__":
    # Command-line interface for choosing the model set, experiment root, and step to inspect.
    parser = argparse.ArgumentParser(description="Train PINN MSCNN model")
    parser.add_argument('--cache-dir', '--cache_dir', type=str,
                       default=os.environ.get(
                           "MURAM_CACHE_DIR",
                           "/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.muram_cache",
                       ))
    parser.add_argument('--step-to-plot', type=int, default=90, help="Monitoring step to generate diagnostics for the trained models.")
    parser.add_argument(
        '--model-types', '--model_types',
        nargs='+',
        default=['all'],
        help="Which trained model types to load (default: all). Supports base types and lambda variants.",
    )
    parser.add_argument(
        '--experiment-root', '--experiment_root',
        type=str,
        default='experiment_80_to_113',
        help='Experiment folder under output/experiments (e.g., experiment_112_to_113)',
    )
    args = parser.parse_args()
    main(args)