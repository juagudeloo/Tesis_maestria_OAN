#!/usr/bin/env bash
#SBATCH --job-name=nicole_syn
#SBATCH --partition=gpu.cecc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=output/synthesis/nicole_syn_%j.out
#SBATCH --error=output/synthesis/nicole_syn_%j.err

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EXPERIMENT_ROOT="experiment_81_to_181-step_size_5-normal"
MODEL_TYPE="no_physics"
REGION_LABEL="negative_region"
CROP_BOUNDS=(0 80 0 200)   # Y0 Y1 X0 X1 (matches ModestData.extract_region order)

# Pixel selection. Either let step 0 auto-select a stratified-by-|B_LOS| sample
# spanning weak/mid/strong field regimes, or pin an explicit manual list.
USE_STRATIFIED_SAMPLING=true
N_BINS=5                    # number of log-spaced |B_LOS| bins (see utils/pixel_sampling.py)
N_PER_BIN=3                 # pixels sampled per bin
SAMPLING_SEED=0
PIXELS=("40,100")           # used only when USE_STRATIFIED_SAMPLING=false

NICOLE_ROOT="/scratchsan/observatorio/juagudeloo/NICOLE_v16.06"
NICOLE_ASSETS="/scratchsan/observatorio/juagudeloo/MUISCA/data/nicole_assets"
OUTPUT_ROOT="/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis"
MODEST_CACHE_DIR="/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache"

# ==============================================================================

cd /scratchsan/observatorio/juagudeloo/MUISCA

REGION_OUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ROOT}/${MODEL_TYPE}/${REGION_LABEL}"

if [ "${USE_STRATIFIED_SAMPLING}" = true ]; then
  echo "=== Step 0: stratified pixel sampling by |B_LOS| ==="
  python scripts/synthesis/sample_pixels.py \
    --experiment-root "${EXPERIMENT_ROOT}" \
    --model-type "${MODEL_TYPE}" \
    --region-label "${REGION_LABEL}" \
    --crop-bounds "${CROP_BOUNDS[@]}" \
    --n-bins "${N_BINS}" \
    --n-per-bin "${N_PER_BIN}" \
    --seed "${SAMPLING_SEED}" \
    --output-root "${OUTPUT_ROOT}" \
    --modest-cache-dir "${MODEST_CACHE_DIR}"

  SELECTED_JSON="${REGION_OUT_DIR}/pixel_selection/selected_pixels.json"
  mapfile -t PIXELS < <(python -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for p in data['pixels']:
    print(f\"{p['ix']},{p['iy']}\")
" "${SELECTED_JSON}")
  echo "Selected ${#PIXELS[@]} pixels: ${PIXELS[*]}"
  echo
fi

PIXEL_ARGS=()
for px in "${PIXELS[@]}"; do
  PIXEL_ARGS+=(--pixel "${px}")
done

echo "=== Step 1: export predictions ==="
python scripts/synthesis/export_predictions.py \
  --source modest \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --model-type "${MODEL_TYPE}" \
  --region-label "${REGION_LABEL}" \
  --crop-bounds "${CROP_BOUNDS[@]}" \
  "${PIXEL_ARGS[@]}" \
  --output-root "${OUTPUT_ROOT}" \
  --modest-cache-dir "${MODEST_CACHE_DIR}"

PRED_H5="${REGION_OUT_DIR}/predictions.h5"

echo
echo "=== Step 2: run NICOLE synthesis ==="
python scripts/synthesis/run_nicole_synthesis.py \
  --predictions-h5 "${PRED_H5}" \
  --nicole-root "${NICOLE_ROOT}" \
  --nicole-assets "${NICOLE_ASSETS}"

SYNTH_H5="${REGION_OUT_DIR}/syntheses.h5"

echo
echo "=== Step 3: compare ==="
python scripts/synthesis/compare_synthesis.py \
  --predictions-h5 "${PRED_H5}" \
  --syntheses-h5 "${SYNTH_H5}"
