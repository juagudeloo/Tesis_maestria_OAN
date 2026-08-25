#!/bin/bash -l
## "%j" es el JobID, un numero asignado por el sistema a su proceso
#SBATCH --job-name=experiment         #Nombre del Trabajo
#SBATCH --cluster=fisica           #nombre de los cluster a donde envia a procesar
#SBATCH -wmaxwell               #Nombre del nodo a usar (configurable via CLUSTER_NODE variable)
#SBATCH --partition=gpu.cecc            #Particion a usar(puede ser: cpu.cecc o gpu.cecc)
##SBATCH --time=15-01:00:00       #Tiempo que usara los recursos(--time=DD-:HH:MM:SS)
#SBATCH --nodes=1                       #Numberode nodos a usar
#SBATCH --ntasks=1               #CPU por tarea >1 si usa multihilado(threads)
#SBATCH --mem=10G                       #Total de memoria RAM por nodo en Gbytes
#SBATCH --gres=gpu:1              # Numbers of needed GPU.
#SBATCH --output=/scratchsan/observatorio/juagudeloo/MUISCA/scripts/experiments/experiment_%j.out      #archivo salida estandar(seguimiento)
#SBATCH --error=/scratchsan/observatorio/juagudeloo/MUISCA/scripts/experiments/experiment_%j.err       #archivo de Errores
###SBATCH --mail-type=begin             #Send email when job begins
###SBATCH --mail-type=end               #Send email when job ends
###SBATCH --mail-user=juagudeloo@unal.edu.co
#SBATCH --export=SCRATCH_DIR=/scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/$SLURM_JOB_ID



module purge
module load envs/anaconda3
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter

# Absolute project root. Everything below addresses the repo through it, so this script can
# be submitted from anywhere -- including `cd tools && sbatch run_experiments.sh`, which
# keeps SLURM from staging the whole repo just to run one job.
MUISCA_ROOT="/scratchsan/observatorio/juagudeloo/MUISCA"
cd "${MUISCA_ROOT}" || exit 1

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

# Data source: 'nicole_tau500' (current default/method) or 'muram_legacy'
DATA_SOURCE="nicole_tau500"

# Data range
MIN_STEP=110
MAX_STEP=130
STEP_SIZE=10
EXPERIMENT_ROOT="experiment_${MIN_STEP}_to_${MAX_STEP}-step_size_${STEP_SIZE}-normal"

# Training hyperparameters
LEARNING_RATE=1e-3
N_EPOCHS=1000
C1_FILTERS=16

# Physics regularization weights (set to 0.0 to disable)
# Use one value to keep current behavior, or many values to run subexperiments for *_only branches.
LAMBDA_WFA_VALUES=(10)
LAMBDA_DOPPLER_VALUES=(5e-1)
LAMBDA_TEMP_VALUES=(2)

# Logtau values to map. Leave empty to use ablation_study.py's own default,
# which matches the tau500 generation grid (45 levels, -3.0 to 1.4) for
# nicole_tau500. If DATA_SOURCE=muram_legacy, set this explicitly instead
# (e.g. NICOLE's HSRA grid: LOGTAU_VALUES=($(seq -f "%.6f" -8.0 0.1 1.4))).
LOGTAU_VALUES=()

# Physics modes
BLOS_MODE='single_height'        # 'tau_averaged' or 'single_height'
BLOS_TARGET_LOGTAU=-0.8          # Must match one of LOGTAU_VALUES for exact single-height supervision

VLOS_MODE='single_height'       # 'tau_averaged' or 'single_height'
VLOS_TARGET_LOGTAU=-1.0         # Only used if VLOS_MODE='single_height'

TEMP_MODE='single_height'       # 'tau_averaged' or 'single_height'
TEMP_TARGET_LOGTAU=0.0          # Only used if TEMP_MODE='single_height' (0.0 = photosphere)

# Optional train-time WFA gate
WFA_GATE_MODE='plateau'         # 'off', 'threshold', or 'plateau'
WFA_GATE_THRESHOLD=0.0          # Used when WFA_GATE_MODE='threshold'
WFA_GATE_PATIENCE=5             # Used when WFA_GATE_MODE='plateau'
WFA_GATE_MIN_DELTA=5e-4         # Used when WFA_GATE_MODE='plateau'
WFA_GATE_WARMUP_EPOCHS=0

# Stokes continuum normalization mode (muram_legacy only -- ignored for
# nicole_tau500, which is already continuum-normalized by NICOLE)
STOKES_IC_MODE='fixed_global'   # 'per_step' or 'fixed_global'

# Scalar multiplier applied after continuum normalization
STOKES_MULT_FACTOR=1

# Shared cache. Leave empty to use ablation_study.py's own data-source-aware
# default (.muram_cache_nicole_tau500 for nicole_tau500, .muram_cache for
# muram_legacy) -- only set to force a specific directory.
CACHE_DIR=""

# Balanced post-bz cache (final balanced tensors reused across epochs).
# Leave empty for the data-source-aware default (see CACHE_DIR above).
BALANCED_CACHE_DIR=""
ENABLE_BALANCED_CACHE=1
BALANCED_CACHE_STRATEGY='auto'   # auto, preload, or disk
BALANCED_CACHE_RAM_BUDGET_GB=32
BALANCED_CACHE_RAM_FRACTION=0.75
CLEAR_BALANCED_CACHE=0

# MODEST epoch diagnostics (ablation study)
ENABLE_MODEST_EPOCH_PLOTS=1
MODEST_CACHE_DIR="/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache"
MODEST_CROP_BOUNDS=(0 100 400 600)   # default plage crop from scripts/analysis/modest_analysis.py
# MODEST pred input mode for per-epoch diagnostics:
# 1 = downsampled (pixel-by-pixel), 0 = upsampled
MODEST_DOWNSAMPLE_PREDICTION_INPUT=1

# Experiments to run
# Options: all, all_physics_terms, wfa_only, doppler_only, black_body_only, no_physics
# Order matters for how soon you learn something: variations run sequentially, ~14 h each at
# 1000 epochs. wfa_only goes first because it is the one under question -- its WFA gate opens
# around epoch 40, so the physics term's effect on the MSE is visible within the first hour
# instead of after no_physics has finished.
EXPERIMENTS="wfa_only no_physics"

# Region-mask balancing during training (ablation study)
# 1 = apply balanced region mask, 0 = disable mask
APPLY_REGION_MASK=0

# NOTE: Bz-bin balancing is DISABLED for base training. Use tools/fine_tune.sh for fine-tuning on balanced Bz data.
# This ensures base training uses the full unsorted dataset, while fine-tuning can focus on balanced bins.
# See docs/how-to-fine-tune.md for fine-tuning workflow and magnetic_field_balancing_and_finetuning.md for theory.

# Bz-bin balancing during training (ablation study)
# 1 = enable Bz balancing, 0 = disable
APPLY_BZ_BIN_BALANCE=0
# Scope: global (recommended) or per_step
BZ_BALANCE_SCOPE='global'
# Mode: mean_abs, max_abs, tau_index
BZ_BALANCE_MODE='tau_index'
# Number of Bz bins used for balancing
BZ_BALANCE_BINS=12
# log(tau) value used when mode=tau_index.
# Example: 0.0 selects the photospheric node if present in LOGTAU_VALUES.
BZ_BALANCE_LOGTAU=0.0
# Random seed for deterministic balanced pixel selection
BZ_BALANCE_SEED=42
# Seed for weight init + shuffling. Same value across variations keeps the ablation arms
# comparable; change it to measure run-to-run variance.
SEED=42

# Training dataset histogram diagnostics (range-of-applicability)
# 1 = generate train-split histograms for T, Vz, Bz; 0 = disable
ENABLE_TRAINING_DATA_HISTOGRAMS=1
TRAINING_HIST_BINS=120
TRAINING_HIST_MAX_SAMPLES=400000

# ==============================================================================
# RUN EXPERIMENT
# ==============================================================================

python3 "${MUISCA_ROOT}/scripts/experiments/ablation_study.py" \
    --data-source "${DATA_SOURCE}" \
    --n_epochs "${N_EPOCHS}" \
    --min_step "${MIN_STEP}" \
    --max_step "${MAX_STEP}" \
    --step_size "${STEP_SIZE}" \
    --n_steps -1 \
    --device cuda \
    --experiment_name "${EXPERIMENT_ROOT}" \
    --output_dir '/scratchsan/observatorio/juagudeloo/MUISCA/output/experiments' \
    --learning_rate "${LEARNING_RATE}" \
    --c1-filters "${C1_FILTERS}" \
    --stokes_ic_mode "${STOKES_IC_MODE}" \
    --stokes-mult-factor "${STOKES_MULT_FACTOR}" \
    $( [[ -n "${CACHE_DIR}" ]] && echo "--cache-dir ${CACHE_DIR}" ) \
    $( [[ "${ENABLE_BALANCED_CACHE}" == "1" ]] && echo "--balanced-cache" ) \
    $( [[ -n "${BALANCED_CACHE_DIR}" ]] && echo "--balanced-cache-dir ${BALANCED_CACHE_DIR}" ) \
    --balanced-cache-strategy "${BALANCED_CACHE_STRATEGY}" \
    --balanced-cache-ram-budget-gb "${BALANCED_CACHE_RAM_BUDGET_GB}" \
    --balanced-cache-ram-fraction "${BALANCED_CACHE_RAM_FRACTION}" \
    $( [[ "${CLEAR_BALANCED_CACHE}" == "1" ]] && echo "--clear-balanced-cache" ) \
    --lambda_wfa "${LAMBDA_WFA_VALUES[@]}" \
    --lambda_doppler "${LAMBDA_DOPPLER_VALUES[@]}" \
    --lambda_temp "${LAMBDA_TEMP_VALUES[@]}" \
    --training-hist-bins "${TRAINING_HIST_BINS}" \
    --training-hist-max-samples "${TRAINING_HIST_MAX_SAMPLES}" \
    --bz-balance-scope "${BZ_BALANCE_SCOPE}" \
    --bz-balance-mode "${BZ_BALANCE_MODE}" \
    --bz-balance-bins "${BZ_BALANCE_BINS}" \
    --bz-balance-logtau "${BZ_BALANCE_LOGTAU}" \
    --bz-balance-seed "${BZ_BALANCE_SEED}" \
    --seed "${SEED}" \
    --blos_physics_mode "${BLOS_MODE}" \
    --blos_target_logtau "${BLOS_TARGET_LOGTAU}" \
    --vlos_physics_mode "${VLOS_MODE}" \
    --vlos_target_logtau "${VLOS_TARGET_LOGTAU}" \
    --temp_physics_mode "${TEMP_MODE}" \
    --temp_target_logtau "${TEMP_TARGET_LOGTAU}" \
    --wfa-gate-mode "${WFA_GATE_MODE}" \
    --wfa-gate-threshold "${WFA_GATE_THRESHOLD}" \
    --wfa-gate-patience "${WFA_GATE_PATIENCE}" \
    --wfa-gate-min-delta "${WFA_GATE_MIN_DELTA}" \
    --wfa-gate-warmup-epochs "${WFA_GATE_WARMUP_EPOCHS}" \
    $( [[ "${#LOGTAU_VALUES[@]}" -gt 0 ]] && echo "--logtau_values ${LOGTAU_VALUES[*]}" ) \
    --experiments ${EXPERIMENTS} \
    --modest-cache-dir "${MODEST_CACHE_DIR}" \
    --modest-crop-bounds "${MODEST_CROP_BOUNDS[@]}" \
    $( [[ "${MODEST_DOWNSAMPLE_PREDICTION_INPUT}" == "1" ]] && echo "--modest-downsample-prediction-input" || echo "--modest-upsample-prediction-input" ) \
    $( [[ "${ENABLE_MODEST_EPOCH_PLOTS}" == "1" ]] && echo "--modest-epoch-plots" ) \
    $( [[ "${ENABLE_TRAINING_DATA_HISTOGRAMS}" == "1" ]] || echo "--no-training-data-histograms" ) \
    $( [[ "${APPLY_BZ_BIN_BALANCE}" == "1" ]] && echo "--apply-bz-bin-balance" || echo "--no-bz-bin-balance" ) \
    $( [[ "${APPLY_REGION_MASK}" == "1" ]] && echo "--apply-region-mask" || echo "--no-region-mask" )

