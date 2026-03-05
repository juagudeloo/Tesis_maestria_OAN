#!/bin/bash -l
## "%j" es el JobID, un numero asignado por el sistema a su proceso
#SBATCH --job-name=experiment         #Nombre del Trabajo
#SBATCH --cluster=fisica           #nombre de los cluster a donde envia a procesar
#SBATCH -wmaxwell               #Nombre del nodo a usar (opcional)
#SBATCH --partition=gpu.cecc            #Particion a usar(puede ser: cpu.cecc o gpu.cecc)
##SBATCH --time=15-01:00:00       #Tiempo que usara los recursos(--time=DD-:HH:MM:SS)
#SBATCH --nodes=1                       #Numberode nodos a usar
#SBATCH --ntasks=1               #CPU por tarea >1 si usa multihilado(threads)
#SBATCH --mem=10G                       #Total de memoria RAM por nodo en Gbytes
#SBATCH --gres=gpu:1              # Numbers of needed GPU.
#SBATCH --output=/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/scripts/experiments/experiment_%j.out      #archivo salida estandar(seguimiento)
#SBATCH --error=/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/scripts/experiments/experiment_%j.err       #archivo de Errores
###SBATCH --mail-type=begin             #Send email when job begins
###SBATCH --mail-type=end               #Send email when job ends
###SBATCH --mail-user=juagudeloo@unal.edu.co
#SBATCH --export=SCRATCH_DIR=/scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/$SLURM_JOB_ID



module purge
module load envs/anaconda3
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter

##cd $SCRATCH_DIR ##ehjecutar en  /scratchsan

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

# Data range
MIN_STEP=112
MAX_STEP=113
STEP_SIZE=1

# Training hyperparameters
LEARNING_RATE=1e-3
N_EPOCHS=5
C1_FILTERS=16

# Physics regularization weights (set to 0.0 to disable)
LAMBDA_WFA=1e-2
LAMBDA_DOPPLER=5e-1
LAMBDA_TEMP=2

# Logtau values to map
LOGTAU_VALUES=(-1.0 -0.8 0.0)

# Physics modes
BLOS_MODE='single_height'        # 'tau_averaged' or 'single_height'
BLOS_TARGET_LOGTAU=-0.8          # Must match one of LOGTAU_VALUES for exact single-height supervision

VLOS_MODE='single_height'       # 'tau_averaged' or 'single_height'
VLOS_TARGET_LOGTAU=-1.0         # Only used if VLOS_MODE='single_height'

TEMP_MODE='single_height'       # 'tau_averaged' or 'single_height'
TEMP_TARGET_LOGTAU=0.0          # Only used if TEMP_MODE='single_height' (0.0 = photosphere)

# Shared cache (same path used by normalization script)
CACHE_DIR="/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.muram_cache"
export MURAM_CACHE_DIR="${CACHE_DIR}"

# MODEST epoch diagnostics (ablation study)
ENABLE_MODEST_EPOCH_PLOTS=1
MODEST_CACHE_DIR="/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.modest_cache"
MODEST_CROP_BOUNDS=(0 100 400 600)   # default plage crop from scripts/analysis/modest_analysis.py

# Experiments to run
# Options: all, all_physics_terms, wfa_only, doppler_only, black_body_only, no_physics
EXPERIMENTS="no_physics wfa_only"

# ==============================================================================
# RUN EXPERIMENT
# ==============================================================================

python3 ./scripts/experiments/ablation_study.py \
    --n_epochs "${N_EPOCHS}" \
    --min_step "${MIN_STEP}" \
    --max_step "${MAX_STEP}" \
    --step_size "${STEP_SIZE}" \
    --n_steps -1 \
    --device cuda \
    --experiment_name "experiment_${MIN_STEP}_to_${MAX_STEP}" \
    --output_dir '/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments' \
    --learning_rate "${LEARNING_RATE}" \
    --c1-filters "${C1_FILTERS}" \
    --cache-dir "${CACHE_DIR}" \
    --lambda_wfa "${LAMBDA_WFA}" \
    --lambda_doppler "${LAMBDA_DOPPLER}" \
    --lambda_temp "${LAMBDA_TEMP}" \
    --blos_physics_mode "${BLOS_MODE}" \
    --blos_target_logtau "${BLOS_TARGET_LOGTAU}" \
    --vlos_physics_mode "${VLOS_MODE}" \
    --vlos_target_logtau "${VLOS_TARGET_LOGTAU}" \
    --temp_physics_mode "${TEMP_MODE}" \
    --temp_target_logtau "${TEMP_TARGET_LOGTAU}" \
    --logtau_values "${LOGTAU_VALUES[@]}" \
    --experiments ${EXPERIMENTS} \
    --modest-cache-dir "${MODEST_CACHE_DIR}" \
    --modest-crop-bounds "${MODEST_CROP_BOUNDS[@]}" \
    $( [[ "${ENABLE_MODEST_EPOCH_PLOTS}" == "1" ]] && echo "--modest-epoch-plots" ) \
    --no_scheduler

