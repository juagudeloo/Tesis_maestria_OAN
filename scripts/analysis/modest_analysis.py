import os
import sys
sys.path.append("/scratchsan/observatorio/juagudeloo/MUISCA/")
import argparse
from pathlib import Path

import torch

from utils.modest_data import ModestData
from utils.cache_manage import ModestDataCache
from utils.normalizer import MhdNormalizer, StokesNormalizer
from utils.analysis import AnalysisModelPipeline, ModestDiagnosticPlots
from scripts.base_training import TrainingConfig

PLAGE_CROP_BOUNDS = (0,100,400, 600)  # X_MIN, X_MAX, Y_MIN, Y_MAX


# -----------------------------------------------------------------------------
# Main MODEST analysis flow
# - selects cropped or whole-region output layout
# - loads trained models and MODEST observations
# - prepares model inputs from MODEST data
# - writes region diagnostics, joint plots, and Stokes summaries
# -----------------------------------------------------------------------------
def main(args):
    # Pick GPU when available; otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modest_base_dir = Path("/scratchsan/observatorio/juagudeloo/MUISCA/images/analysis/modest")

    # Choose where MODEST diagnostics should be written based on the crop flag.
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
        experiment_root=args.experiment_root,
        experiments_base_dir=args.experiments_base_dir,
    )

    # Load the trained model checkpoints and determine their shared tau grid.
    model_configs, models, n_tau = pipeline.prepare_models(args.model_types)
    print(f"Using device: {device}")
    print(f"Number of log(tau) points: {n_tau}")
    print(f"Prediction input mode: {'downsampled' if args.downsample_prediction_input else 'upsampled'}")

    # Print the selected experiments so it is obvious which checkpoints are being analyzed.
    print("Selected model configs:")
    for _, cfg in model_configs.items():
        print(f"  - {cfg['label']} ({cfg['experiment_key']})")

    # Normalizers are loaded once and reused for every model.
    mhd_normalizer = MhdNormalizer()
    stokes_normalizer = StokesNormalizer()
    default_cfg = TrainingConfig()
    mhd_normalizer.load(filepath=str(Path(default_cfg.data_path) / default_cfg.mhd_normalizer_path))
    stokes_normalizer.load(filepath=str(Path(default_cfg.data_path) / default_cfg.stokes_normalizer_path))

    # MODEST loader handles observation ingestion, masking, and wavelength metadata.
    modest = ModestData(
        circular_polarization_threshold=args.polarization_threshold,
        stokes_v_multiplier=args.modest_stokes_v_multiplier,
    )
    modest_cache = ModestDataCache(cache_dir=args.modest_cache_dir)
    print(f"MODEST cache directory: {args.modest_cache_dir}")
    if args.clear_modest_cache:
        modest_cache.clear(confirm=False)
    modest_cache.print_cache_info()

    # Diagnostic helper does the per-model plotting and metrics writing.
    diagnostics = ModestDiagnosticPlots(
        pipeline=pipeline,
        modest_output_dir=modest_output_dir,
        mhd_normalizer=mhd_normalizer,
        stokes_normalizer=stokes_normalizer,
        modest=modest,
        modest_cache=modest_cache,
        args=args,
    )
    diagnostics.prepare_snapshot(n_tau=n_tau)
    diagnostics.run(model_configs=model_configs, models=models)

    print(f"\nFinished analysis for {modest_output_dir}")

if __name__ == "__main__":
    # CLI arguments mirror the training and data-loading choices used during analysis.
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
        '--modest-stokes-v-multiplier', '--modest_stokes_v_multiplier',
        dest='modest_stokes_v_multiplier',
        type=float,
        default=-1.0,
        help='scale factor applied to MODEST Stokes V (default: -1.0 to match MURaM polarity)',
    )
    parser.add_argument(
        '--modest-stokes-shift-positions', '--modest_stokes_shift_positions',
        dest='modest_stokes_shift_positions',
        type=float,
        default=0.0,
        help='shift MODEST Stokes profiles by this many spectral sample positions (default: 0.0)',
    )
    parser.add_argument(
        '--modest-stokes-i-scale', '--modest_stokes_i_scale',
        dest='modest_stokes_i_scale',
        type=float,
        default=1.0,
        help='multiply MODEST Stokes I profiles by this factor after optional shift/inversion (default: 1.0)',
    )
    parser.add_argument(
        '--modest-stokes-v-scale', '--modest_stokes_v_scale',
        dest='modest_stokes_v_scale',
        type=float,
        default=1.0,
        help='multiply MODEST Stokes V profiles by this factor after optional shift/inversion (default: 1.0)',
    )
    parser.add_argument(
        '--modest-stokes-invert-direction', '--modest_stokes_invert_direction',
        dest='modest_stokes_invert_direction',
        action='store_true',
        help='invert MODEST Stokes profile direction along the spectral axis before optional shift/scale',
    )
    parser.add_argument(
        '--modest-pred-mhd-invert-sign', '--modest_pred_mhd_invert_sign',
        dest='modest_pred_mhd_invert_sign',
        action='store_true',
        help='invert predicted MODEST V_LOS and B_LOS signs before region/joint plotting',
    )
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
    parser.add_argument(
        '--experiments-base-dir', '--experiments_base_dir', dest='experiments_base_dir',
        type=str, default=None,
        help='Directory holding <experiment-root>/ (default: output/experiments). Point at '
             'output/fine-tune to analyze fine-tuned checkpoints.',
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
    parser.add_argument(
        '--modest-cache-dir', '--modest_cache_dir',
        dest='modest_cache_dir',
        type=str,
        default=os.environ.get(
            "MODEST_CACHE_DIR",
            "/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache",
        ),
        help='MODEST cache directory (or set MODEST_CACHE_DIR)'
    )
    parser.add_argument(
        '--no-modest-cache', '--no_modest_cache',
        dest='no_modest_cache',
        action='store_true',
        help='Disable MODEST cache usage'
    )
    parser.add_argument(
        '--clear-modest-cache', '--clear_modest_cache',
        dest='clear_modest_cache',
        action='store_true',
        help='Clear MODEST cache before running analysis'
    )
    parser.add_argument(
        '--downsample-prediction-input',
        action='store_true',
        help='Downsample deconvolved+smoothed Stokes back to native grid before model inference',
    )
    parser.add_argument(
        '--temp-calibration-mode', '--temp_calibration_mode',
        dest='temp_calibration_mode',
        type=str,
        choices=['off', 'apply_fit'],
        default='off',
        help='Temperature post-calibration mode: off (no calibration) or apply_fit (fit/load per-tau bias b; slope a fixed to 1)',
    )
    parser.add_argument(
        '--temp-calibration-dir', '--temp_calibration_dir',
        dest='temp_calibration_dir',
        type=str,
        default=None,
        help='Optional base directory for temperature calibration JSON files (default: model output folder)',
    )
    parser.add_argument(
        '--temp-calibration-min-samples', '--temp_calibration_min_samples',
        dest='temp_calibration_min_samples',
        type=int,
        default=500,
        help='Minimum paired finite pixels required to fit bias calibration per log(tau)',
    )
    parser.add_argument(
        '--temp-calibration-clip-quantiles', '--temp_calibration_clip_quantiles',
        dest='temp_calibration_clip_quantiles',
        type=float,
        nargs=2,
        default=None,
        metavar=('Q_LOW', 'Q_HIGH'),
        help='Optional quantile clipping for calibration fitting, e.g. --temp-calibration-clip-quantiles 0.01 0.99',
    )
    args = parser.parse_args()
    main(args)