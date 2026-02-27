#!/bin/bash -l
## "%j" es el JobID, un numero asignado por el sistema a su proceso
#SBATCH --job-name=unified_analysis
#SBATCH --cluster=fisica
#SBATCH -wmaxwell
#SBATCH --partition=gpu.cecc
##SBATCH --time=1-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/scripts/analysis/ua_run%j.out
#SBATCH --error=/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/scripts/analysis/ua_run%j.err
###SBATCH --mail-type=begin
###SBATCH --mail-type=end
###SBATCH --mail-user=juagudeloo@unal.edu.co
#SBATCH --export=SCRATCH_DIR=/scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/$SLURM_JOB_ID

module purge
module load envs/anaconda3
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter

# ==============================================================================
# UNIFIED ANALYSIS CONFIGURATION
# ==============================================================================

SCRIPT_PATH="/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/scripts/analysis/unified_analysis.py"

# Which analyses to run: modest_small modest_whole muram_whole all
ANALYSIS_TARGETS=("all")

# Available presets: no_physics wfa_only doppler_only black_body_only all_physics_terms
MODELS=("no_physics" "wfa_only")

BASE_MODEL_PATH="/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments"
MODEST_EXPERIMENT="experiment_80_to_113"
MURAM_EXPERIMENT="experiment_80_to_113"

# MODEST small region
Y_START=0
Y_END=100
X_START=400
X_END=600
REGION_NAME="plage"
VISUALIZATION_ONLY=false

# MODEST whole region OD list (leave empty to use script default)
MODEST_OD_VALUES=()

# MURaM whole region
MURAM_OD_VALUES=(-1.0 -0.8 0.0)
# Keep remap logtau grid aligned with as_run.batch and cns_run.batch
MURAM_LOGTAU_VALUES=(-1.0 -0.8 0.0)
USE_CACHE=true
CACHE_DIR="/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.data_cache"
export MURAM_CACHE_DIR="${CACHE_DIR}"

# ==============================================================================
# RUN
# ==============================================================================

CMD=(python3 "${SCRIPT_PATH}"
    --analysis "${ANALYSIS_TARGETS[@]}"
    --models "${MODELS[@]}"
    --base-model-path "${BASE_MODEL_PATH}"
    --modest-experiment "${MODEST_EXPERIMENT}"
    --muram-experiment "${MURAM_EXPERIMENT}"
    --y-start "${Y_START}"
    --y-end "${Y_END}"
    --x-start "${X_START}"
    --x-end "${X_END}"
    --region-name "${REGION_NAME}"
    --muram-od-values-to-plot "${MURAM_OD_VALUES[@]}"
    --muram-logtau-values "${MURAM_LOGTAU_VALUES[@]}"
    --cache-dir "${CACHE_DIR}"
)

if [ "${VISUALIZATION_ONLY}" = true ]; then
    CMD+=(--visualization-only)
fi

if [ ${#MODEST_OD_VALUES[@]} -gt 0 ]; then
    CMD+=(--modest-od-values "${MODEST_OD_VALUES[@]}")
fi

if [ "${USE_CACHE}" = false ]; then
    CMD+=(--no-cache)
fi

"${CMD[@]}"
