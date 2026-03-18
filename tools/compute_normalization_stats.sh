#!/bin/bash -l
## "%j" es el JobID, un numero asignado por el sistema a su proceso
#SBATCH --job-name=comp_norm         #Nombre del Trabajo
#SBATCH --cluster=fisica           #nombre de los cluster a donde envia a procesar
#SBATCH -wmaxwell               #Nombre del nodo a usar (opcional)
#SBATCH --partition=gpu.cecc            #Particion a usar(puede ser: cpu.cecc o gpu.cecc)
##SBATCH --time=1-01:00:00       #Tiempo que usara los recursos(--time=DD-:HH:MM:SS)
#SBATCH --nodes=1                       #Numberode nodos a usar
#SBATCH --ntasks=1               #CPU por tarea >1 si usa multihilado(threads)
#SBATCH --mem=10G                       #Total de memoria RAM por nodo en Gbytes
#SBATCH --gres=gpu:1              # Numbers of needed GPU.
#SBATCH --output=/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/scripts/comp_norm%j.out      #archivo salida estandar(seguimiento)
#SBATCH --error=/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/scripts/comp_norm%j.err       #archivo de Errores
###SBATCH --mail-type=begin             #Send email when job begins
###SBATCH --mail-type=end               #Send email when job ends
###SBATCH --mail-user=juagudeloo@unal.edu.co
#SBATCH --export=SCRATCH_DIR=/scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/$SLURM_JOB_ID

module purge
module load envs/anaconda3
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter

##cd $SCRATCH_DIR ##ehjecutar en  /scratchsan

# ==============================================================================
# NORMALIZATION STATS CONFIGURATION
# ==============================================================================

MIN_STEP=60
MAX_STEP=200
SAVE_INTERVAL=20

USE_CACHE=true
CACHE_DIR="/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.muram_cache"
export MURAM_CACHE_DIR="${CACHE_DIR}"

# Start from zero
CLEAN_START=true
PURGE_CACHE=true

# Resume (optional)
USE_RESUME=false
RESUME_FROM=""

# Optical depth config:
# If true, uses explicit nodes (same as as_run.batch).
# If false, uses range mode: logtau_min/max/step.
USE_EXPLICIT_LOGTAU=true
LOGTAU_VALUES=(-1.0 -0.8 0.0)

LOGTAU_MIN=-2.0
LOGTAU_MAX=0.0
LOGTAU_STEP=0.1

# Stokes continuum normalization policy
STOKES_IC_MODE="fixed_global"
IC_START_STEP=70
IC_END_STEP=80
IC_CONT_INDICES=(0 1 2 3)

# ==============================================================================
# RUN
# ==============================================================================

CMD=(python3 ./scripts/compute_normalization_stats.py
    --min_step "${MIN_STEP}"
    --max_step "${MAX_STEP}"
    --save_interval "${SAVE_INTERVAL}"
    --cache-dir "${CACHE_DIR}"
)

if [ "${USE_CACHE}" = false ]; then
    CMD+=(--no_cache)
fi

if [ "${USE_RESUME}" = true ] && [ -n "${RESUME_FROM}" ]; then
    CMD+=(--resume_from "${RESUME_FROM}")
fi

if [ "${USE_EXPLICIT_LOGTAU}" = true ]; then
    CMD+=(--logtau_values "${LOGTAU_VALUES[@]}")
else
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

