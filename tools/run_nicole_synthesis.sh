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
MODEL_TYPE="wfa_only"
REGION_LABEL="negative_region"
CROP_BOUNDS=(0 80 0 200)   # Y0 Y1 X0 X1 (matches ModestData.extract_region order)
PIXELS=("40,100")           # ix,iy tuples (space-separated list)

NICOLE_ROOT="/scratchsan/observatorio/juagudeloo/NICOLE_v16.06"
NICOLE_ASSETS="/scratchsan/observatorio/juagudeloo/MUISCA/data/nicole_assets"
OUTPUT_ROOT="/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis"
MODEST_CACHE_DIR="/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache"

# ==============================================================================

cd /scratchsan/observatorio/juagudeloo/MUISCA

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

PRED_H5="${OUTPUT_ROOT}/${EXPERIMENT_ROOT}/${MODEL_TYPE}/${REGION_LABEL}/predictions.h5"

echo
echo "=== Step 2: run NICOLE synthesis ==="
python scripts/synthesis/run_nicole_synthesis.py \
  --predictions-h5 "${PRED_H5}" \
  --nicole-root "${NICOLE_ROOT}" \
  --nicole-assets "${NICOLE_ASSETS}"

SYNTH_H5="${OUTPUT_ROOT}/${EXPERIMENT_ROOT}/${MODEL_TYPE}/${REGION_LABEL}/syntheses.h5"

echo
echo "=== Step 3: compare ==="
python scripts/synthesis/compare_synthesis.py \
  --predictions-h5 "${PRED_H5}" \
  --syntheses-h5 "${SYNTH_H5}"
