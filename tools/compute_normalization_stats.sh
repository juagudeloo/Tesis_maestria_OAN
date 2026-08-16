#!/bin/bash -l
## "%j" es el JobID, un numero asignado por el sistema a su proceso
## Runnable either directly (./tools/compute_normalization_stats.sh -- fine
## for a handful of steps, a few minutes) or via `sbatch` for a large step
## range; the #SBATCH lines below are plain comments when run directly.
#SBATCH --job-name=comp_norm         #Nombre del Trabajo
#SBATCH --cluster=fisica           #nombre de los cluster a donde envia a procesar
#SBATCH -wmaxwell               #Nombre del nodo a usar (opcional)
#SBATCH --partition=cpu.cecc            #Particion a usar (no necesita GPU)
##SBATCH --time=1-01:00:00       #Tiempo que usara los recursos(--time=DD-:HH:MM:SS)
#SBATCH --nodes=1                       #Numberode nodos a usar
#SBATCH --ntasks=1               #CPU por tarea >1 si usa multihilado(threads)
#SBATCH --mem=24G                       #Total de memoria RAM por nodo en Gbytes
#SBATCH --output=/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/comp_norm_%j.out
#SBATCH --error=/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/comp_norm_%j.err
###SBATCH --mail-type=begin             #Send email when job begins
###SBATCH --mail-type=end               #Send email when job ends
###SBATCH --mail-user=juagudeloo@unal.edu.co

module purge
module load envs/anaconda3
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter

cd /scratchsan/observatorio/juagudeloo/MUISCA

# ==============================================================================
# NORMALIZATION STATS CONFIGURATION
# ==============================================================================

# 'nicole_tau500' (current default/method) or 'muram_legacy'
DATA_SOURCE="nicole_tau500"

# Explicit step list -- overrides MIN_STEP/MAX_STEP scanning below. Leave
# empty (STEPS=()) to fall back to scanning MIN_STEP..MAX_STEP instead (e.g.
# for a muram_legacy full-range run).
STEPS=(110 120 130 198)
MIN_STEP=60
MAX_STEP=200
SAVE_INTERVAL=20

USE_CACHE=true
# Leave empty to use compute_normalization_stats.py's own data-source-aware
# default (.muram_cache_nicole_tau500 for nicole_tau500, .muram_cache for
# muram_legacy) -- only set this to force a specific directory.
CACHE_DIR=""

# Start from zero (wipes only DATA_SOURCE's own normalization_stats subdir)
CLEAN_START=true
PURGE_CACHE=false

# Resume (optional; muram_legacy only -- disabled for fixed_global Ic mode)
USE_RESUME=false
RESUME_FROM=""

# Optical depth grid: leave USE_EXPLICIT_LOGTAU=false and LOGTAU_MIN/MAX
# empty to use compute_normalization_stats.py's own default, which matches
# the tau500 generation grid (45 levels, -3.0 to 1.4) for nicole_tau500.
# If DATA_SOURCE=muram_legacy, set these explicitly (e.g. -2.0/0.0/0.1) --
# the default no longer matches the legacy grid.
USE_EXPLICIT_LOGTAU=false
LOGTAU_VALUES=()
LOGTAU_MIN=""
LOGTAU_MAX=""
LOGTAU_STEP=""

# Stokes continuum normalization policy (muram_legacy only -- ignored for
# nicole_tau500, which is already continuum-normalized by NICOLE)
STOKES_IC_MODE="fixed_global"
IC_START_STEP=70
IC_END_STEP=80
IC_CONT_INDICES=(0 1 2 3)

# ==============================================================================
# RUN
# ==============================================================================

CMD=(python3 ./scripts/compute_normalization_stats.py
    --data-source "${DATA_SOURCE}"
    --save_interval "${SAVE_INTERVAL}"
)

if [ "${#STEPS[@]}" -gt 0 ]; then
    CMD+=(--steps "${STEPS[@]}")
else
    CMD+=(--min_step "${MIN_STEP}" --max_step "${MAX_STEP}")
fi

if [ -n "${CACHE_DIR}" ]; then
    CMD+=(--cache-dir "${CACHE_DIR}")
fi

if [ "${USE_CACHE}" = false ]; then
    CMD+=(--no_cache)
fi

if [ "${USE_RESUME}" = true ] && [ -n "${RESUME_FROM}" ]; then
    CMD+=(--resume_from "${RESUME_FROM}")
fi

if [ "${USE_EXPLICIT_LOGTAU}" = true ]; then
    CMD+=(--logtau_values "${LOGTAU_VALUES[@]}")
elif [ -n "${LOGTAU_MIN}" ] && [ -n "${LOGTAU_MAX}" ] && [ -n "${LOGTAU_STEP}" ]; then
    CMD+=(--logtau_min "${LOGTAU_MIN}" --logtau_max "${LOGTAU_MAX}" --logtau_step "${LOGTAU_STEP}")
fi

CMD+=(
    --stokes_ic_mode "${STOKES_IC_MODE}"
    --ic_start_step "${IC_START_STEP}"
    --ic_end_step "${IC_END_STEP}"
    --ic_cont_indices "${IC_CONT_INDICES[@]}"
)

if [ "${CLEAN_START}" = true ]; then
    CMD+=(--clean_start)
fi

if [ "${PURGE_CACHE}" = true ]; then
    CMD+=(--purge_cache)
fi

"${CMD[@]}"

