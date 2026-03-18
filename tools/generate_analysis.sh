#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# ANALYSIS CONFIGURATION
# ==============================================================================

# Shared model selection (space-separated list)
# Base options: all, no_physics, wfa_only, doppler_only, black_body_only, all_physics_terms
# Lambda variants must match experiment keys, e.g. wfa_only-lambda-0_01
MODEL_TYPES="no_physics wfa_only-lambda-1 wfa_only-lambda-0_1 wfa_only-lambda-0_01 wfa_only-lambda-0_001 wfa_only-lambda-0_0001 wfa_only-lambda-1em05"
EXPERIMENT_ROOT="experiment_112_to_113-wfa_plateu_gate-global_Ic"

# Runtime control
RUN_TARGET="both"                       # both | muram | modest

# MURaM analysis args
CACHE_DIR="/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.muram_cache"
STEP_TO_PLOT="198"

# MODEST analysis args
CROPPED_REGION="0"                      # 1 => --cropped-region
CROP_BOUNDS=(0 100 400 600)             # X_MIN X_MAX Y_MIN Y_MAX
POLARIZATION_MASK="0"                   # 1 => --polarization-mask
POLARIZATION_THRESHOLD="1e-2"
CROP_LABEL="plage"
MODEST_CACHE_DIR="/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.modest_cache"
CLEAR_MODEST_CACHE="0"                  # 1 => --clear-modest-cache
DOWNSAMPLE_PREDICTION_INPUT="0"         # 1 => --downsample-prediction-input

# Temperature calibration args (MODEST only)
TEMP_CALIBRATION_MODE="off"             # off | apply_fit (bias-only: per-tau b, fixed a=1)
TEMP_CALIBRATION_DIR=""                 # optional shared dir for calibration JSON files
TEMP_CALIBRATION_MIN_SAMPLES="500"      # min paired samples required to fit per-tau bias
TEMP_CALIBRATION_CLIP_QUANTILES=""      # e.g. "0.01 0.99" — leave empty to disable

usage() {
  cat <<'EOF'
Usage: tools/generate_analysis.sh [options]

Options:
  --run both|muram|modest   Select analyses to run (default: both)
  --step-to-plot STEP         MURaM: simulation step to plot (default: 198)
  --cropped-region 0|1      MODEST only: enable/disable cropped-region output (default: 0)
  --polarization-mask 0|1   MODEST only: enable/disable polarization mask (default: 0)
  --polarization-threshold VALUE  MODEST only: circular polarization threshold (default: 1e-2)
  --experiment-root NAME    Experiment folder under output/experiments (default from script variable)
  --modest-cache-dir PATH   MODEST cache directory
  --clear-modest-cache 0|1  Clear MODEST cache before run (default: 0)
  --downsample-prediction-input 0|1  MODEST: downsample prediction Stokes to native grid
  --temp-calibration-mode off|apply_fit  MODEST: calibration mode; apply_fit is bias-only (a=1)
  --temp-calibration-dir PATH   MODEST: shared dir to store/load calibration JSON
  --temp-calibration-min-samples N   MODEST: min samples to fit per-tau bias (default: 500)
  --temp-calibration-clip-quantiles "Q_LOW Q_HIGH"  MODEST: e.g. "0.01 0.99"
  -h, --help                Show this help
EOF
}

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
  both|muram|modest) ;;
  *)
    echo "Invalid value for --run: ${RUN_TARGET} (use: both|muram|modest)" >&2
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
    ${DOWNSAMPLE_PREDICTION_INPUT_FLAG} \
    ${CLEAR_MODEST_CACHE_FLAG} \
    ${TEMP_CALIBRATION_FLAGS}
fi