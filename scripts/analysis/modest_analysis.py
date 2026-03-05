import os
import sys
sys.path.append("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/")
import argparse
from pathlib import Path

import torch

from utils.modest_data import ModestData
from utils.normalizer import MhdNormalizer, StokesNormalizer
from utils.analysis import AnalysisModelPipeline, ModestDiagnosticPlots
from scripts.base_training import TrainingConfig

PLAGE_CROP_BOUNDS = (0,100,400, 600)  # X_MIN, X_MAX, Y_MIN, Y_MAX

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modest_base_dir = Path("/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/images/analysis/modest")
    if args.cropped_region:
        crop_label = args.crop_label.strip()
        if not crop_label:
            raise ValueError("--crop-label must be a non-empty string when --cropped-region is set.")
        modest_output_dir = modest_base_dir / "cropped" / crop_label
    else:
        modest_output_dir = modest_base_dir / "whole"

    pipeline = AnalysisModelPipeline(
        device=device,
        output_dir=modest_output_dir,
    )
    model_configs, models, n_tau = pipeline.prepare_models(args.model_types)
    print(f"Using device: {device}")
    print(f"Number of log(tau) points: {n_tau}")

    print("Selected model configs:")
    for _, cfg in model_configs.items():
        print(f"  - {cfg['label']} ({cfg['experiment_key']})")

    mhd_normalizer = MhdNormalizer()
    stokes_normalizer = StokesNormalizer()
    default_cfg = TrainingConfig()
    mhd_normalizer.load(filepath=default_cfg.data_path / default_cfg.mhd_normalizer_path)
    stokes_normalizer.load(filepath=default_cfg.data_path / default_cfg.stokes_normalizer_path)

    modest = ModestData(
        circular_polarization_threshold=args.polarization_threshold if args.polarization_mask else None
    )

    diagnostics = ModestDiagnosticPlots(
        pipeline=pipeline,
        modest_output_dir=modest_output_dir,
        mhd_normalizer=mhd_normalizer,
        stokes_normalizer=stokes_normalizer,
        modest=modest,
        args=args,
    )
    diagnostics.prepare_snapshot(n_tau=n_tau)
    diagnostics.run(model_configs=model_configs, models=models)

    print(f"\nFinished analysis for {modest_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PINN MSCNN model")
    parser.add_argument(
        '--cropped-region', 
        action='store_true',
        help='whether to use cropped region (default: False)')   
    parser.add_argument(
        '--crop-bounds', 
        nargs=4, 
        type=int, 
        default=PLAGE_CROP_BOUNDS,
        metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX'),
        help=f'bounds for cropping the region (default plage bounds: {PLAGE_CROP_BOUNDS})'
    )
    parser.add_argument(
        '--polarization-mask',
        action='store_true',
        help='whether to apply circular polarization mask to the data (default: False)'
    )
    parser.add_argument(
        '--polarization-threshold',
        type=float,
        default=1e-2,
        help='threshold for circular polarization mask (default: 0.01)'
    )
    parser.add_argument(
        '--model-types', '--model_types',
        nargs='+',
        default=['all'],
        choices=['all', 'no_physics', 'wfa_only', 'doppler_only', 'black_body_only', 'all_physics_terms'],
        help="Which trained model types to load (default: all). Example: --model-types no_physics wfa_only",
    )
    parser.add_argument(
        '--crop-label',
        type=str,
        default='plage',
        help='name of the cropped region subfolder (used only with --cropped-region), e.g. "plage"',
    )
    parser.add_argument(
        '--tau-indices',
        nargs='*',
        type=int,
        default=None,
        help='Tau indices to plot (default: 0, mid, last)'
    )
    parser.add_argument(
        '--inference-batch-size',
        type=int,
        default=4096,
        help='Batch size for MODEST inference (default: 4096)'
    )
    args = parser.parse_args()
    main(args)