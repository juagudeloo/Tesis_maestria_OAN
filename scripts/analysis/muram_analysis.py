import os
import sys
sys.path.append("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/")
import argparse
from pathlib import Path

import torch
import numpy as np
from utils.cache_manage import DataCache
from utils.normalizer import MhdNormalizer, StokesNormalizer
from utils.analysis import AnalysisModelPipeline, DiagnosticPlots
from scripts.base_training import TrainingConfig, load_and_prepare_step

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = AnalysisModelPipeline(
        device=device,
        output_dir=Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/images/analysis/muram"),
    )
    model_configs, models, n_tau = pipeline.prepare_models(args.model_types)
    
    print(f"Using device: {device}")
    print(f"Number of log(tau) points: {n_tau}")

    print("Selected model configs:")
    for _, cfg in model_configs.items():
        print(f"  - {cfg['label']} ({cfg['experiment_key']})")

    cache = DataCache(cache_dir=args.cache_dir, compression='gzip')
    print(f"Shared MURaM data cache: {args.cache_dir}")
    print("\nInitial Cache Status:")
    cache.print_cache_info()

    # Load normalizers once
    mhd_normalizer = MhdNormalizer()
    stokes_normalizer = StokesNormalizer()
    default_cfg = TrainingConfig()
    mhd_normalizer.load(filepath=default_cfg.data_path / default_cfg.mhd_normalizer_path)
    stokes_normalizer.load(filepath=default_cfg.data_path / default_cfg.stokes_normalizer_path)

    for name, model in models.items():
        cfg = pipeline.build_runtime_training_config(model_configs[name])
        model_type = model_configs[name]["experiment_key"]
        print(f"Generating diagnostics for: {model_configs[name]['label']}")

        dataset, _ = load_and_prepare_step(
            step=args.step_to_plot,
            config=cfg,
            mhd_normalizer=mhd_normalizer,
            stokes_normalizer=stokes_normalizer,
            cache=cache,
        )

        # Convert normalized GT to physical units once; predictions are denormalized in DiagnosticPlots.
        gt_reshaped = dataset.mhd_targets.reshape(dataset.mhd_targets.shape[0], n_tau, 3)
        gt_values_reshaped = np.empty_like(gt_reshaped)
        for param_idx, param_name in enumerate(["T", "Vz", "Bz"]):
            gt_values_reshaped[:, :, param_idx] = mhd_normalizer.denormalize(
                gt_reshaped[:, :, param_idx], param=param_name
            )
        muram_gt_values = gt_values_reshaped.reshape(dataset.mhd_targets.shape)

        plotter = DiagnosticPlots(
            config=cfg,
            model_name=model_type,
            step=args.step_to_plot,
            output_dir=Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/images/analysis/muram"),
        )
        plotter.generate(
            model=model,
            stokes_input=dataset.stokes_input,
            gt_values=muram_gt_values,
            nx_pred=dataset.nx,
            ny_pred=dataset.ny,
            nx_gt=dataset.nx,
            ny_gt=dataset.ny,
            denormalize_param=lambda arr, param: mhd_normalizer.denormalize(arr, param=param),
            param_names=["T", "Vz", "Bz"],
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PINN MSCNN model")
    parser.add_argument('--cache-dir', '--cache_dir', type=str,
                       default=os.environ.get(
                           "MURAM_CACHE_DIR",
                           "/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.data_cache",
                       ))
    parser.add_argument('--step-to-plot', type=int, default=90, help="Monitoring step to generate diagnostics for the trained models.")
    parser.add_argument(
        '--model-types', '--model_types',
        nargs='+',
        default=['all'],
        choices=['all', 'no_physics', 'wfa_only', 'doppler_only', 'black_body_only', 'all_physics_terms'],
        help="Which trained model types to load (default: all). Example: --model-types no_physics wfa_only",
    )
    args = parser.parse_args()
    main(args)