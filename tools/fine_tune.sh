#!/bin/bash -l

################################################################################
# Fine-Tuning Launcher Script for MUISCA Models
################################################################################
# 
# Submits fine-tuning jobs via SLURM on Maxwell cluster with mandatory Bz balancing.
#
# Usage:
#   sbatch tools/fine_tune.sh --experiment-name <name> --variations <var1,var2,...>
#   sbatch --time=06:00:00 --mem=16G tools/fine_tune.sh --experiment-name <name> --variations <vars>
#
# Defaults:
#   Job name: finetune
#   GPU: 1× GPU on gpu.cecc partition
#   Memory: 32 GB
#   Wall time: 12 hours
#   CPU cores: 4
#
# Mandatory arguments:
#   --experiment-name NAME              Experiment name from output/experiments/
#   --variations VAR1,VAR2,...          Comma-separated variation names to fine-tune
#
# Optional arguments:
#   --finetune-epochs N                 Override fine-tuning epoch count (default: 10% of original)
#   --output-base-dir PATH              Override output directory (default: output/fine-tune/)
#   --exp-base-dir PATH                 Override experiment checkpoint directory
#   --steps N [N ...]                   Explicit steps to fine-tune on, e.g. --steps 120 130
#   --min-step / --max-step / --step-size   Range form of the same override
#
# Step selection defaults to the base run's range, read from experiment_config.json next to
# the checkpoint. Overriding it does NOT refit the normalizers -- those stay the base run's,
# since the Bz asinh scale is baked into the pretrained weights' output space.
#
################################################################################

#SBATCH --job-name=finetune             # Job name
#SBATCH --cluster=fisica                # Cluster name
#SBATCH -w maxwell                      # Node (configurable via CLUSTER_NODE variable)
#SBATCH --partition=gpu.cecc            # GPU partition
#SBATCH --nodes=1                       # Single node
#SBATCH --ntasks=1                      # Single MPI task
#SBATCH --cpus-per-task=4               # 4 CPU cores
#SBATCH --mem=32G                       # 32 GB RAM
#SBATCH --gres=gpu:1                    # 1× GPU
#SBATCH --time=12:00:00                 # 12 hour wall time
#SBATCH --output=/scratchsan/observatorio/juagudeloo/MUISCA/output/fine-tune/finetune_%j.out
#SBATCH --error=/scratchsan/observatorio/juagudeloo/MUISCA/output/fine-tune/finetune_%j.err
#SBATCH --export=SCRATCH_DIR=/scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/$SLURM_JOB_ID

################################################################################
# PARSE COMMAND-LINE ARGUMENTS
################################################################################

EXPERIMENT_NAME=""
VARIATIONS=""
FINETUNE_EPOCHS=""
OUTPUT_BASE_DIR=""
EXP_BASE_DIR=""
STEPS=()
MIN_STEP=""
MAX_STEP=""
STEP_SIZE=""

# Parse arguments passed via sbatch or command line
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --variations)
            VARIATIONS="$2"
            shift 2
            ;;
        --finetune-epochs)
            FINETUNE_EPOCHS="$2"
            shift 2
            ;;
        --output-base-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --exp-base-dir)
            EXP_BASE_DIR="$2"
            shift 2
            ;;
        --steps)
            # Variadic: consume every following bare number, e.g. --steps 120 130
            shift
            while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do
                STEPS+=("$1")
                shift
            done
            ;;
        --min-step)
            MIN_STEP="$2"
            shift 2
            ;;
        --max-step)
            MAX_STEP="$2"
            shift 2
            ;;
        --step-size)
            STEP_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate mandatory arguments
if [ -z "$EXPERIMENT_NAME" ] || [ -z "$VARIATIONS" ]; then
    echo "ERROR: --experiment-name and --variations are required"
    echo ""
    echo "Usage:"
    echo "  sbatch tools/fine_tune.sh --experiment-name <name> --variations <var1,var2>"
    echo ""
    echo "Examples:"
    echo "  sbatch tools/fine_tune.sh --experiment-name experiment_81_to_181 --variations wfa_only"
    echo "  sbatch tools/fine_tune.sh --experiment-name experiment_81_to_181 --variations wfa_only,all_physics_terms,doppler_only"
    echo "  sbatch --time=06:00:00 tools/fine_tune.sh --experiment-name experiment_81_to_181 --variations all_physics_terms --finetune-epochs 20"
    exit 1
fi

################################################################################
# SETUP ENVIRONMENT
################################################################################

# Setup project paths
PROJECT_ROOT="/scratchsan/observatorio/juagudeloo/MUISCA"
cd "$PROJECT_ROOT" || exit 1

# Set defaults if not provided via arguments
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"$PROJECT_ROOT/output/fine-tune"}
EXP_BASE_DIR=${EXP_BASE_DIR:-"$PROJECT_ROOT/output/experiments"}

# Create output directory for fine-tune results and logs
mkdir -p "$OUTPUT_BASE_DIR"

# Activate conda environment
module purge
module load envs/anaconda3
source /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter/etc/profile.d/conda.sh
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter

################################################################################
# PRINT JOB INFORMATION
################################################################################

echo "======================================================================================================"
echo "MUISCA Model Fine-Tuning Launcher"
echo "======================================================================================================"
echo "SLURM Job Information:"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Job Name: ${SLURM_JOB_NAME}"
echo "  Node: ${SLURM_NODELIST}"
echo "  Partition: ${SLURM_PARTITION}"
echo "  CPUs: ${SLURM_CPUS_PER_TASK}"
echo "  Memory: ${SLURM_MEM_PER_NODE}"
echo "  GPUs: ${SLURM_GPUS}"
echo ""
echo "Fine-Tuning Configuration:"
echo "  Experiment Name: ${EXPERIMENT_NAME}"
echo "  Variations to fine-tune: ${VARIATIONS}"
echo "  Fine-tune epochs: ${FINETUNE_EPOCHS:-'10% of original (min 5)'}"
echo "  Output directory: ${OUTPUT_BASE_DIR}"
echo "  Experiment base: ${EXP_BASE_DIR}"
echo ""
echo "Environment:"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Working directory: $(pwd)"
echo "  Python: $(which python3)"
echo "  Conda env: $(conda info --envs | grep \* )"
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
else
    echo "  GPU: Not available"
fi
echo "======================================================================================================"
echo ""

################################################################################
# RUN FINE-TUNING
################################################################################

# Build Python command with optional arguments
FINETUNE_CMD="python3 scripts/finetune.py \
    --experiment-name '${EXPERIMENT_NAME}' \
    --variations '${VARIATIONS}' \
    --output-base-dir '${OUTPUT_BASE_DIR}' \
    --exp-base-dir '${EXP_BASE_DIR}'"

# Add optional fine-tune epochs if specified
if [ -n "$FINETUNE_EPOCHS" ]; then
    FINETUNE_CMD="$FINETUNE_CMD --finetune-epochs ${FINETUNE_EPOCHS}"
fi

# Optional step selection. Omitted entirely -> finetune.py replays the base run's range
# from experiment_config.json.
if [ "${#STEPS[@]}" -gt 0 ]; then
    FINETUNE_CMD="$FINETUNE_CMD --steps ${STEPS[*]}"
fi
if [ -n "$MIN_STEP" ]; then
    FINETUNE_CMD="$FINETUNE_CMD --min-step ${MIN_STEP}"
fi
if [ -n "$MAX_STEP" ]; then
    FINETUNE_CMD="$FINETUNE_CMD --max-step ${MAX_STEP}"
fi
if [ -n "$STEP_SIZE" ]; then
    FINETUNE_CMD="$FINETUNE_CMD --step-size ${STEP_SIZE}"
fi

# Execute fine-tuning script
echo "Running fine-tuning..."
echo ""
eval "$FINETUNE_CMD"
FINETUNE_EXIT_CODE=$?

################################################################################
# SUMMARY
################################################################################

echo ""
echo "======================================================================================================"
if [ $FINETUNE_EXIT_CODE -eq 0 ]; then
    echo "✓ Fine-Tuning Completed Successfully"
else
    echo "✗ Fine-Tuning Failed (exit code: $FINETUNE_EXIT_CODE)"
fi
echo "======================================================================================================"
echo "Results saved to: ${OUTPUT_BASE_DIR}"
echo "Log files: ${OUTPUT_BASE_DIR}/finetune_${SLURM_JOB_ID}.out (stdout)"
echo "           ${OUTPUT_BASE_DIR}/finetune_${SLURM_JOB_ID}.err (stderr)"
echo "======================================================================================================"
echo ""

exit $FINETUNE_EXIT_CODE
