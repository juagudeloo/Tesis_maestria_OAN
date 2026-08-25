#!/usr/bin/env bash
#SBATCH --job-name=nicole_syn
#SBATCH --cluster=fisica           #nombre de los cluster a donde envia a procesar
#SBATCH -wmaxwell               #Nombre del nodo a usar (configurable via CLUSTER_NODE variable)
#SBATCH --partition=gpu.cecc            #Particion a usar(puede ser: cpu.cecc o gpu.cecc)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=begin             #Send email when job begins
#SBATCH --mail-type=end               #Send email when job ends
#SBATCH --mail-user=juagudeloo@unal.edu.co
#SBATCH --output=/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/nicole_syn_%j.out
#SBATCH --error=/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/nicole_syn_%j.err

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EXPERIMENT_ROOT="experiment_110_to_130-step_size_10-normal"
MODEL_TYPES=("wfa_only" "no_physics")   # one or more model variations to run.
                                          # Step 0's sampling is model-independent
                                          # (SPINOR- or MURaM-ground-truth-sourced),
                                          # so every variant here gets the SAME pixel
                                          # selection automatically. With 2+ entries,
                                          # steps 4-5 (cross-model comparison) also
                                          # run at the end.

# Source: modest (real Hinode/SOT-SP observations, region-cropped) or muram
# (a MURaM simulation step, e.g. one outside the model's training window, for
# an out-of-distribution generalization check). Overridable via --source/--step/
# --add-gt-pressure on the command line; everything else below stays edit-in-file.
SOURCE="modest"                 # modest | muram
MURAM_STEP=""                   # muram only; e.g. 198
ADD_GT_PRESSURE=false           # muram only -- feed NICOLE the true MURaM gas
                                 # pressure instead of a hydrostatic-equilibrium
                                 # seed. Runs land in a sibling step-N-gt-pressure/
                                 # tree, so a plain run and this one can be diffed.

REGION_LABEL="negative_region"  # modest only -- ignored for muram (step-N plays
                                 # that role in the output path instead)
CROP_BOUNDS=(0 80 0 200)        # modest only -- Y0 Y1 X0 X1 (matches ModestData.extract_region order)

# Pixel selection. Either let step 0 auto-select a stratified-by-|B_LOS| sample
# spanning weak/mid/strong field regimes, or pin an explicit manual list.
USE_STRATIFIED_SAMPLING=true
N_BINS=10                    # number of log-spaced |B_LOS| bins (see utils/pixel_sampling.py)
N_PER_BIN=15                # pixels sampled per bin (violin/aggregate tier -- step 5)
N_OVERLAY_PER_BIN=5          # subset of N_PER_BIN flagged for individual overlay PNGs (step 4 only)
SAMPLING_SEED=0
PIXELS=("40,100")           # used only when USE_STRATIFIED_SAMPLING=false

NICOLE_ROOT="/scratchsan/observatorio/juagudeloo/NICOLE_v16.06"
NICOLE_ASSETS="/scratchsan/observatorio/juagudeloo/MUISCA/data/nicole_assets"
OUTPUT_ROOT="/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis"
MODEST_CACHE_DIR="/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache"

# ==============================================================================
# CLI (overrides the CONFIGURATION defaults above)
# ==============================================================================
usage() {
  cat <<'EOF'
Usage: tools/run_nicole_synthesis.sh [--source modest|muram] [--step N] [--add-gt-pressure]

Options:
  --source modest|muram   Prediction source (default: modest)
  --step N                MURaM simulation step number (required with --source muram)
  --add-gt-pressure       Feed NICOLE the true MURaM gas pressure instead of a
                          hydrostatic seed (--source muram only). Output lands in
                          a sibling step-N-gt-pressure/ tree.
  -h, --help              Show this help

Everything else (EXPERIMENT_ROOT, MODEL_TYPES, N_BINS, ...) is configured by
editing the CONFIGURATION block at the top of this script.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="${2:-}"
      shift 2
      ;;
    --step)
      MURAM_STEP="${2:-}"
      shift 2
      ;;
    --add-gt-pressure)
      ADD_GT_PRESSURE=true
      shift
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

if [[ "${SOURCE}" != "modest" && "${SOURCE}" != "muram" ]]; then
  echo "Invalid --source: ${SOURCE} (use: modest|muram)" >&2
  exit 1
fi
if [[ "${SOURCE}" == "muram" ]]; then
  if [[ -z "${MURAM_STEP}" || ! "${MURAM_STEP}" =~ ^[0-9]+$ ]]; then
    echo "--step N (integer) is required when --source muram" >&2
    exit 1
  fi
else
  if [[ "${ADD_GT_PRESSURE}" == true ]]; then
    echo "--add-gt-pressure requires --source muram" >&2
    exit 1
  fi
fi

# ==============================================================================

MUISCA_ROOT="/scratchsan/observatorio/juagudeloo/MUISCA"
cd "${MUISCA_ROOT}" || exit 1

SOURCE_ARGS=(--source "${SOURCE}")
if [[ "${SOURCE}" == "muram" ]]; then
  SOURCE_ARGS+=(--muram-step "${MURAM_STEP}")
fi
GT_PRESSURE_ARGS=()
if [[ "${ADD_GT_PRESSURE}" == true ]]; then
  GT_PRESSURE_ARGS+=(--add-gt-pressure)
fi

for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
  echo "################################################################"
  echo "# Model variant: ${MODEL_TYPE}  (source=${SOURCE}$( [[ "${SOURCE}" == "muram" ]] && echo ", step=${MURAM_STEP}, gt_pressure=${ADD_GT_PRESSURE}" ))"
  echo "################################################################"

  if [[ "${SOURCE}" == "muram" ]]; then
    STEP_LABEL="step-${MURAM_STEP}"
    [[ "${ADD_GT_PRESSURE}" == true ]] && STEP_LABEL="${STEP_LABEL}-gt-pressure"
    REGION_OUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ROOT}/muram/${STEP_LABEL}/${MODEL_TYPE}"
    # Step 0 (sample_pixels.py) never takes --add-gt-pressure -- pixel
    # stratification is pressure-independent by design, so a plain run and
    # its -gt-pressure sibling must sample the SAME pixels for a fair diff.
    # Its output therefore always lives under the plain step-N/ dir, even
    # when this run's REGION_OUT_DIR (predictions/syntheses) has the suffix.
    SAMPLE_OUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ROOT}/muram/step-${MURAM_STEP}/${MODEL_TYPE}"
  else
    REGION_OUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ROOT}/modest/${MODEL_TYPE}/${REGION_LABEL}"
    SAMPLE_OUT_DIR="${REGION_OUT_DIR}"
  fi
  RUN_PIXELS=("${PIXELS[@]}")

  if [ "${USE_STRATIFIED_SAMPLING}" = true ]; then
    echo "=== Step 0: stratified pixel sampling by |B_LOS| (model-independent) ==="
    SAMPLE_ARGS=(
      "${SOURCE_ARGS[@]}"
      --experiment-root "${EXPERIMENT_ROOT}"
      --model-type "${MODEL_TYPE}"
      --n-bins "${N_BINS}"
      --n-per-bin "${N_PER_BIN}"
      --n-overlay-per-bin "${N_OVERLAY_PER_BIN}"
      --seed "${SAMPLING_SEED}"
      --output-root "${OUTPUT_ROOT}"
    )
    if [[ "${SOURCE}" == "modest" ]]; then
      SAMPLE_ARGS+=(
        --region-label "${REGION_LABEL}"
        --crop-bounds "${CROP_BOUNDS[@]}"
        --modest-cache-dir "${MODEST_CACHE_DIR}"
      )
    fi
    python "${MUISCA_ROOT}/scripts/synthesis/sample_pixels.py" "${SAMPLE_ARGS[@]}"

    SELECTED_JSON="${SAMPLE_OUT_DIR}/pixel_selection/selected_pixels.json"
    mapfile -t RUN_PIXELS < <(python -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for p in data['pixels']:
    print(f\"{p['ix']},{p['iy']}\")
" "${SELECTED_JSON}")
    echo "Selected ${#RUN_PIXELS[@]} pixels: ${RUN_PIXELS[*]}"
    echo
  fi

  PIXEL_ARGS=()
  for px in "${RUN_PIXELS[@]}"; do
    PIXEL_ARGS+=(--pixel "${px}")
  done

  echo "=== Step 1: export predictions ==="
  EXPORT_ARGS=(
    "${SOURCE_ARGS[@]}"
    "${GT_PRESSURE_ARGS[@]}"
    --experiment-root "${EXPERIMENT_ROOT}"
    --model-type "${MODEL_TYPE}"
    "${PIXEL_ARGS[@]}"
    --output-root "${OUTPUT_ROOT}"
  )
  if [[ "${SOURCE}" == "modest" ]]; then
    EXPORT_ARGS+=(
      --region-label "${REGION_LABEL}"
      --crop-bounds "${CROP_BOUNDS[@]}"
      --modest-cache-dir "${MODEST_CACHE_DIR}"
    )
  fi
  python "${MUISCA_ROOT}/scripts/synthesis/export_predictions.py" "${EXPORT_ARGS[@]}"

  PRED_H5="${REGION_OUT_DIR}/predictions.h5"

  echo
  echo "=== Step 2: run NICOLE synthesis ==="
  python "${MUISCA_ROOT}/scripts/synthesis/run_nicole_synthesis.py" \
    --predictions-h5 "${PRED_H5}" \
    --nicole-root "${NICOLE_ROOT}" \
    --nicole-assets "${NICOLE_ASSETS}"

  SYNTH_H5="${REGION_OUT_DIR}/syntheses.h5"

  echo
  echo "=== Step 3: compare (single model) ==="
  python "${MUISCA_ROOT}/scripts/synthesis/compare_synthesis.py" \
    --predictions-h5 "${PRED_H5}" \
    --syntheses-h5 "${SYNTH_H5}"

  echo
done

if [ "${#MODEL_TYPES[@]}" -ge 2 ]; then
  echo "################################################################"
  echo "# Step 4: cross-model comparison (pixel_comparison/, overlay tier)"
  echo "################################################################"
  MODEL_TYPE_ARGS=()
  for mt in "${MODEL_TYPES[@]}"; do
    MODEL_TYPE_ARGS+=(--model-type "${mt}")
  done
  COMPARE_ARGS=(
    "${SOURCE_ARGS[@]}"
    "${GT_PRESSURE_ARGS[@]}"
    --experiment-root "${EXPERIMENT_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    "${MODEL_TYPE_ARGS[@]}"
  )
  [[ "${SOURCE}" == "modest" ]] && COMPARE_ARGS+=(--region-label "${REGION_LABEL}")
  python "${MUISCA_ROOT}/scripts/synthesis/compare_models.py" "${COMPARE_ARGS[@]}"

  echo
  echo "################################################################"
  echo "# Step 5: aggregate distribution comparison (aggregate_plots/, violin tier)"
  echo "################################################################"
  python "${MUISCA_ROOT}/scripts/synthesis/aggregate_comparison.py" "${COMPARE_ARGS[@]}"
fi
