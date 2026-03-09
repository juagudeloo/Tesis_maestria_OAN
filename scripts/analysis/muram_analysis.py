import os
import sys
sys.path.append("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/")
import argparse
from pathlib import Path

import torch
import numpy as np
from utils.cache_manage import MuramDataCache
from utils.normalizer import MhdNormalizer, StokesNormalizer
from utils.analysis import AnalysisModelPipeline, MuramDiagnosticPlots
from scripts.base_training import TrainingConfig, load_and_prepare_step

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = AnalysisModelPipeline(
        device=device,
        output_dir=Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/images/analysis/muram"),
        experiment_root=args.experiment_root,
    )
    model_configs, models, n_tau = pipeline.prepare_models(args.model_types)
    
    print(f"Using device: {device}")
    print(f"Number of log(tau) points: {n_tau}")

    print("Selected model configs:")
    for _, cfg in model_configs.items():
        print(f"  - {cfg['label']} ({cfg['experiment_key']})")

    cache = MuramDataCache(cache_dir=args.cache_dir, compression='gzip')
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

        # Denormalize predictions using the pipeline
        pred_den = pipeline.predict_and_denormalize(
            model=model,
            stokes_input=dataset.stokes_input,
            mhd_normalizer=mhd_normalizer,
            pred_nx=dataset.nx,
            pred_ny=dataset.ny,
        )

        # Denormalize GT to physical units
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

        plotter = MuramDiagnosticPlots(
            config=cfg,
            model_name=model_type,
            step=args.step_to_plot,
            output_dir=Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/images/analysis/muram"),
        )
        plotter.generate(
            pred_den=pred_den,
            gt_den=gt_den,
        )

if __name__ == "__main__":
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