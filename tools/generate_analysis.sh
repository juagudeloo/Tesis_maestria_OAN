#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# ANALYSIS CONFIGURATION
# ==============================================================================

# Shared model selection (space-separated list)
# Base options: all, no_physics, wfa_only, doppler_only, black_body_only, all_physics_terms
# Lambda variants must match experiment keys, e.g. wfa_only-lambda-0_01
MODEL_TYPES="no_physics wfa_only"
EXPERIMENT_ROOT="experiment_110_to_130-step_size_10-normal"

# Runtime control
RUN_TARGET="both"                       # both | muram | modest
DISTRIBUTION_TARGET="both"              # stokes | mhd | both (distributions mode)

# MURaM analysis args
CACHE_DIR="/scratchsan/observatorio/juagudeloo/MUISCA/.muram_cache"
STEP_TO_PLOT="198"

# MODEST analysis args
CROPPED_REGION="0"                      # 1 => --cropped-region
#sunspot
# CROP_BOUNDS=(100 300 250 450)             # X_MIN X_MAX Y_MIN Y_MAX
# CROP_LABEL="sunspot"                      # label for cropped region (used in plot titles and output paths)
#plage
# CROP_BOUNDS=(0 100 400 600)             # X_MIN X_MAX Y_MIN Y_MAX
# CROP_LABEL="plage"                      # label for cropped region (used in plot titles and output paths)
#negative region
CROP_BOUNDS=(0 80 0 200)             # X_MIN X_MAX Y_MIN Y_MAX
CROP_LABEL="negative_region"      
#quiet sun
# CROP_BOUNDS=(0 100 600 700)             # X_MIN X_MAX Y_MIN Y_MAX
# CROP_LABEL="quiet_sun"                      # label for cropped region (used in plot titles and output paths)
POLARIZATION_MASK="0"                   # 1 => --polarization-mask
POLARIZATION_THRESHOLD="1e-2"
MODEST_CACHE_DIR="/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache"
CLEAR_MODEST_CACHE="0"                  # 1 => --clear-modest-cache
DOWNSAMPLE_PREDICTION_INPUT="0"         # 1 => --downsample-prediction-input
MODEST_STOKES_SHIFT_POSITIONS="0.0"
MODEST_STOKES_I_SCALE="1.0"
MODEST_STOKES_V_SCALE="1.0"
MODEST_STOKES_INVERT_DIRECTION="0"      # 1 => --modest-stokes-invert-direction
MODEST_PRED_MHD_INVERT_SIGN="0"         # 1 => --modest-pred-mhd-invert-sign (invert predicted Vz/Bz)

# Temperature calibration args (MODEST only)
TEMP_CALIBRATION_MODE="off"             # off | apply_fit (bias-only: per-tau b, fixed a=1)
TEMP_CALIBRATION_DIR=""                 # optional shared dir for calibration JSON files
TEMP_CALIBRATION_MIN_SAMPLES="500"      # min paired samples required to fit per-tau bias
TEMP_CALIBRATION_CLIP_QUANTILES=""      # e.g. "0.01 0.99" — leave empty to disable

usage() {
  cat <<'EOF'
Usage: tools/generate_analysis.sh [distributions] [options]

Options:
  --run both|muram|modest   Select analyses to run (default: both)
  --distribution-target stokes|mhd|both  distributions mode selector (default: both)
  --step-to-plot STEP         MURaM: simulation step to plot (default: 198)
  --cropped-region 0|1      MODEST only: enable/disable cropped-region output (default: 0)
  --polarization-mask 0|1   MODEST only: enable/disable polarization mask (default: 0)
  --polarization-threshold VALUE  MODEST only: circular polarization threshold (default: 1e-2)
  --experiment-root NAME    Experiment folder under output/experiments (default from script variable)
  --modest-cache-dir PATH   MODEST cache directory
  --clear-modest-cache 0|1  Clear MODEST cache before run (default: 0)
  --downsample-prediction-input 0|1  MODEST: downsample prediction Stokes to native grid
  --modest-stokes-shift-positions VALUE  MODEST: shift Stokes profiles by sample positions
  --modest-stokes-i-scale VALUE  MODEST: multiplicative factor for Stokes I
  --modest-stokes-v-scale VALUE  MODEST: multiplicative factor for Stokes V
  --modest-stokes-invert-direction 0|1  MODEST: invert spectral direction before optional shift/scale
  --modest-pred-mhd-invert-sign 0|1  MODEST: invert predicted V_LOS and B_LOS signs before plotting
  --temp-calibration-mode off|apply_fit  MODEST: calibration mode; apply_fit is bias-only (a=1)
  --temp-calibration-dir PATH   MODEST: shared dir to store/load calibration JSON
  --temp-calibration-min-samples N   MODEST: min samples to fit per-tau bias (default: 500)
  --temp-calibration-clip-quantiles "Q_LOW Q_HIGH"  MODEST: e.g. "0.01 0.99"
  -h, --help                Show this help
EOF
}

if [[ $# -gt 0 && "${1}" != --* ]]; then
  if [[ "${1}" == "distributions" ]]; then
    RUN_TARGET="distributions"
    shift
  else
    echo "Unknown positional mode: ${1} (supported: distributions)" >&2
    usage
    exit 1
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_TARGET="${2:-}"
      shift 2
      ;;
    --step-to-plot)
      STEP_TO_PLOT="${2:-}"
      shift 2
      ;;
    --distribution-target)
      DISTRIBUTION_TARGET="${2:-}"
      shift 2
      ;;
    --cropped-region)
      CROPPED_REGION="${2:-}"
      shift 2
      ;;
    --polarization-mask)
      POLARIZATION_MASK="${2:-}"
      shift 2
      ;;
    --polarization-threshold)
      POLARIZATION_THRESHOLD="${2:-}"
      shift 2
      ;;
    --experiment-root)
      EXPERIMENT_ROOT="${2:-}"
      shift 2
      ;;
    --modest-cache-dir)
      MODEST_CACHE_DIR="${2:-}"
      shift 2
      ;;
    --clear-modest-cache)
      CLEAR_MODEST_CACHE="${2:-}"
      shift 2
      ;;
    --downsample-prediction-input)
      DOWNSAMPLE_PREDICTION_INPUT="${2:-}"
      shift 2
      ;;
    --modest-stokes-shift-positions)
      MODEST_STOKES_SHIFT_POSITIONS="${2:-}"
      shift 2
      ;;
    --modest-stokes-i-scale)
      MODEST_STOKES_I_SCALE="${2:-}"
      shift 2
      ;;
    --modest-stokes-v-scale)
      MODEST_STOKES_V_SCALE="${2:-}"
      shift 2
      ;;
    --modest-stokes-invert-direction)
      MODEST_STOKES_INVERT_DIRECTION="${2:-}"
      shift 2
      ;;
    --modest-pred-mhd-invert-sign)
      MODEST_PRED_MHD_INVERT_SIGN="${2:-}"
      shift 2
      ;;
    --temp-calibration-mode)
      TEMP_CALIBRATION_MODE="${2:-}"
      shift 2
      ;;
    --temp-calibration-dir)
      TEMP_CALIBRATION_DIR="${2:-}"
      shift 2
      ;;
    --temp-calibration-min-samples)
      TEMP_CALIBRATION_MIN_SAMPLES="${2:-}"
      shift 2
      ;;
    --temp-calibration-clip-quantiles)
      TEMP_CALIBRATION_CLIP_QUANTILES="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${RUN_TARGET}" in
  both|muram|modest|distributions) ;;
  *)
    echo "Invalid value for --run: ${RUN_TARGET} (use: both|muram|modest)" >&2
    exit 1
    ;;
esac

case "${DISTRIBUTION_TARGET}" in
  stokes|mhd|both) ;;
  *)
    echo "Invalid value for --distribution-target: ${DISTRIBUTION_TARGET} (use: stokes|mhd|both)" >&2
    exit 1
    ;;
esac

if ! [[ "${STEP_TO_PLOT}" =~ ^[0-9]+$ ]]; then
  echo "Invalid value for --step-to-plot: ${STEP_TO_PLOT} (must be an integer)" >&2
  exit 1
fi

if [[ "${CROPPED_REGION}" != "0" && "${CROPPED_REGION}" != "1" ]]; then
  echo "Invalid value for --cropped-region: ${CROPPED_REGION} (use: 0|1)" >&2
  exit 1
fi

if [[ "${POLARIZATION_MASK}" != "0" && "${POLARIZATION_MASK}" != "1" ]]; then
  echo "Invalid value for --polarization-mask: ${POLARIZATION_MASK} (use: 0|1)" >&2
  exit 1
fi

if ! [[ "${POLARIZATION_THRESHOLD}" =~ ^[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$ ]]; then
  echo "Invalid value for --polarization-threshold: ${POLARIZATION_THRESHOLD} (must be numeric)" >&2
  exit 1
fi

if [[ "${CLEAR_MODEST_CACHE}" != "0" && "${CLEAR_MODEST_CACHE}" != "1" ]]; then
  echo "Invalid value for --clear-modest-cache: ${CLEAR_MODEST_CACHE} (use: 0|1)" >&2
  exit 1
fi

if [[ "${DOWNSAMPLE_PREDICTION_INPUT}" != "0" && "${DOWNSAMPLE_PREDICTION_INPUT}" != "1" ]]; then
  echo "Invalid value for --downsample-prediction-input: ${DOWNSAMPLE_PREDICTION_INPUT} (use: 0|1)" >&2
  exit 1
fi

if ! [[ "${MODEST_STOKES_SHIFT_POSITIONS}" =~ ^[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$ ]]; then
  echo "Invalid value for --modest-stokes-shift-positions: ${MODEST_STOKES_SHIFT_POSITIONS} (must be numeric)" >&2
  exit 1
fi

if ! [[ "${MODEST_STOKES_I_SCALE}" =~ ^[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$ ]]; then
  echo "Invalid value for --modest-stokes-i-scale: ${MODEST_STOKES_I_SCALE} (must be numeric)" >&2
  exit 1
fi

if ! [[ "${MODEST_STOKES_V_SCALE}" =~ ^[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$ ]]; then
  echo "Invalid value for --modest-stokes-v-scale: ${MODEST_STOKES_V_SCALE} (must be numeric)" >&2
  exit 1
fi

if [[ "${MODEST_STOKES_INVERT_DIRECTION}" != "0" && "${MODEST_STOKES_INVERT_DIRECTION}" != "1" ]]; then
  echo "Invalid value for --modest-stokes-invert-direction: ${MODEST_STOKES_INVERT_DIRECTION} (use: 0|1)" >&2
  exit 1
fi

if [[ "${MODEST_PRED_MHD_INVERT_SIGN}" != "0" && "${MODEST_PRED_MHD_INVERT_SIGN}" != "1" ]]; then
  echo "Invalid value for --modest-pred-mhd-invert-sign: ${MODEST_PRED_MHD_INVERT_SIGN} (use: 0|1)" >&2
  exit 1
fi

case "${TEMP_CALIBRATION_MODE}" in
  off|apply_fit) ;;
  *)
    echo "Invalid value for --temp-calibration-mode: ${TEMP_CALIBRATION_MODE} (use: off|apply_fit)" >&2
    exit 1
    ;;
esac



CROPPED_REGION_FLAG=""
if [[ "${CROPPED_REGION}" == "1" ]]; then
  CROPPED_REGION_FLAG="--cropped-region"
fi

POLARIZATION_MASK_FLAG=""
if [[ "${POLARIZATION_MASK}" == "1" ]]; then
  POLARIZATION_MASK_FLAG="--polarization-mask"
fi

CLEAR_MODEST_CACHE_FLAG=""
if [[ "${CLEAR_MODEST_CACHE}" == "1" ]]; then
  CLEAR_MODEST_CACHE_FLAG="--clear-modest-cache"
fi

DOWNSAMPLE_PREDICTION_INPUT_FLAG=""
if [[ "${DOWNSAMPLE_PREDICTION_INPUT}" == "1" ]]; then
  DOWNSAMPLE_PREDICTION_INPUT_FLAG="--downsample-prediction-input"
fi

MODEST_STOKES_INVERT_DIRECTION_FLAG=""
if [[ "${MODEST_STOKES_INVERT_DIRECTION}" == "1" ]]; then
  MODEST_STOKES_INVERT_DIRECTION_FLAG="--modest-stokes-invert-direction"
fi

MODEST_PRED_MHD_INVERT_SIGN_FLAG=""
if [[ "${MODEST_PRED_MHD_INVERT_SIGN}" == "1" ]]; then
  MODEST_PRED_MHD_INVERT_SIGN_FLAG="--modest-pred-mhd-invert-sign"
fi

TEMP_CALIBRATION_FLAGS=""
if [[ "${TEMP_CALIBRATION_MODE}" == "apply_fit" ]]; then
  TEMP_CALIBRATION_FLAGS="--temp-calibration-mode apply_fit"
fi
if [[ -n "${TEMP_CALIBRATION_DIR}" ]]; then
  TEMP_CALIBRATION_FLAGS="${TEMP_CALIBRATION_FLAGS} --temp-calibration-dir ${TEMP_CALIBRATION_DIR}"
fi
if [[ -n "${TEMP_CALIBRATION_MIN_SAMPLES}" && "${TEMP_CALIBRATION_MIN_SAMPLES}" != "500" ]]; then
  TEMP_CALIBRATION_FLAGS="${TEMP_CALIBRATION_FLAGS} --temp-calibration-min-samples ${TEMP_CALIBRATION_MIN_SAMPLES}"
fi
if [[ -n "${TEMP_CALIBRATION_CLIP_QUANTILES}" ]]; then
  TEMP_CALIBRATION_FLAGS="${TEMP_CALIBRATION_FLAGS} --temp-calibration-clip-quantiles ${TEMP_CALIBRATION_CLIP_QUANTILES}"
fi

# ==============================================================================
# RUN ANALYSIS
# ==============================================================================

if [[ "${RUN_TARGET}" == "both" || "${RUN_TARGET}" == "muram" ]]; then
  python3 ./scripts/analysis/muram_analysis.py \
    --cache-dir "${CACHE_DIR}" \
    --step-to-plot "${STEP_TO_PLOT}" \
    --experiment-root "${EXPERIMENT_ROOT}" \
    --model-types ${MODEL_TYPES}
fi

if [[ "${RUN_TARGET}" == "both" || "${RUN_TARGET}" == "modest" ]]; then
  python3 ./scripts/analysis/modest_analysis.py \
    ${CROPPED_REGION_FLAG} \
    --crop-bounds "${CROP_BOUNDS[@]}" \
    ${POLARIZATION_MASK_FLAG} \
    --polarization-threshold "${POLARIZATION_THRESHOLD}" \
    --experiment-root "${EXPERIMENT_ROOT}" \
    --model-types ${MODEL_TYPES} \
    --crop-label "${CROP_LABEL}" \
    --modest-cache-dir "${MODEST_CACHE_DIR}" \
    --modest-stokes-shift-positions "${MODEST_STOKES_SHIFT_POSITIONS}" \
    --modest-stokes-i-scale "${MODEST_STOKES_I_SCALE}" \
    --modest-stokes-v-scale "${MODEST_STOKES_V_SCALE}" \
    ${MODEST_STOKES_INVERT_DIRECTION_FLAG} \
    ${MODEST_PRED_MHD_INVERT_SIGN_FLAG} \
    ${DOWNSAMPLE_PREDICTION_INPUT_FLAG} \
    ${CLEAR_MODEST_CACHE_FLAG} \
    ${TEMP_CALIBRATION_FLAGS}
fi

if [[ "${RUN_TARGET}" == "distributions" ]]; then
  python3 ./scripts/analysis/distributions_analysis.py \
    ${CROPPED_REGION_FLAG} \
    --crop-bounds "${CROP_BOUNDS[@]}" \
    ${POLARIZATION_MASK_FLAG} \
    --polarization-threshold "${POLARIZATION_THRESHOLD}" \
    --crop-label "${CROP_LABEL}" \
    --experiment-root "${EXPERIMENT_ROOT}" \
    --distribution-target "${DISTRIBUTION_TARGET}" \
    --modest-cache-dir "${MODEST_CACHE_DIR}" \
    --modest-stokes-shift-positions "${MODEST_STOKES_SHIFT_POSITIONS}" \
    --modest-stokes-i-scale "${MODEST_STOKES_I_SCALE}" \
    --modest-stokes-v-scale "${MODEST_STOKES_V_SCALE}" \
    ${MODEST_STOKES_INVERT_DIRECTION_FLAG} \
    ${DOWNSAMPLE_PREDICTION_INPUT_FLAG} \
    ${CLEAR_MODEST_CACHE_FLAG}
fi