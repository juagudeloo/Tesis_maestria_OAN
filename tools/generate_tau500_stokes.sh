#!/usr/bin/env bash
#SBATCH --job-name=tau500_gen
#SBATCH --cluster=fisica
#SBATCH -wmaxwell
#SBATCH --partition=gpu.cecc
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7
#SBATCH --mail-type=end
#SBATCH --mail-user=juagudeloo@unal.edu.co
#SBATCH --output=/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/tau500_gen_%A_%a.out
#SBATCH --error=/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/tau500_gen_%A_%a.err
#
# Generate MURaM Stokes from zero with NICOLE on the tau_500 scale (issue #5),
# full-frame, parallelized over row-chunks with a SLURM array.
#
# Each array task synthesizes one row-chunk of one step
# (scripts/synthesis/generate_tau500_stokes.py). The chunks are chunk- and
# neighbour-independent (NicoleRunner.run_cube prepends a warm-up pixel), so
# they can be produced in any order / any split and merged deterministically.
#
# USAGE
#   1. Edit STEPS and N_CHUNKS below.
#   2. Pilot first (STRONGLY recommended -- the "with caution" gate):
#        ./tools/generate_tau500_stokes.sh --pilot
#      runs a single small chunk locally and reports ms/pixel + wall time, so
#      you can size N_CHUNKS before committing the full run.
#   3. Submit everything:   ./tools/generate_tau500_stokes.sh --submit-waves
#      This account's SLURM association caps MaxSubmit at 8 jobs (pending+
#      running combined) -- a single `sbatch` with a big --array is rejected
#      outright even with %-throttling (every array task counts against the
#      cap here). --submit-waves runs on the login node (it is NOT itself an
#      sbatch job) and submits the N_CHUNKS*len(STEPS) chunks in waves of 8,
#      via `sbatch --wait`, blocking between waves. Safe to re-run: chunks
#      whose output already exists are skipped, so a failed chunk (or a
#      Ctrl-C) only costs re-running that one wave.
#   4. After it completes, merge each step:
#        ./tools/generate_tau500_stokes.sh --merge
#
#   (A bare `sbatch tools/generate_tau500_stokes.sh` still works for a single
#   wave of <=8 chunks -- e.g. to manually retry a specific range with
#   --array=lo-hi -- but won't fit a full run through the MaxSubmit=8 cap.)
#
# ==============================================================================
# CONFIGURATION
# ==============================================================================
STEPS=(110 120 130 198) # MURaM steps: 131-133 contiguous for train/val, 198 for the OOD test split
N_CHUNKS=40              # row-chunks per step (480 rows / N_CHUNKS rows each)
NX_ROWS=480             # MURaM frame rows (nx)

MUISCA_ROOT="/scratchsan/observatorio/juagudeloo/MUISCA"
NICOLE_ROOT="/scratchsan/observatorio/juagudeloo/NICOLE_v16.06"
# Per-chunk intermediates are preprocessing, not deliverables -> a gitignored
# cache folder (matches the *_cache/ convention, alongside .muram_cache), NOT
# output/. Only the MERGED files (below) are the real regenerated data.
CHUNK_DIR="${MUISCA_ROOT}/.muram_tau500_cache/chunks"
MERGED_DIR="${MUISCA_ROOT}/data/muram-simulation"   # merged stokes_{step}_nicole_tau500.npy land here
PY="/homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter/bin/python"

# ==============================================================================

usage() { grep '^#   ' "$0" | sed 's/^#   //'; }

cd "${MUISCA_ROOT}"
NSTEPS=${#STEPS[@]}

# chunk i of N_CHUNKS -> row range [r0, r1)
chunk_rows() {
  local ci=$1
  local base=$(( NX_ROWS / N_CHUNKS ))
  local rem=$(( NX_ROWS % N_CHUNKS ))
  local r0 r1
  if [ "${ci}" -lt "${rem}" ]; then
    r0=$(( ci * (base + 1) ));            r1=$(( r0 + base + 1 ))
  else
    r0=$(( rem * (base + 1) + (ci - rem) * base )); r1=$(( r0 + base ))
  fi
  echo "${r0} ${r1}"
}

run_chunk() {  # step chunk_index
  local step=$1 ci=$2
  read -r r0 r1 <<< "$(chunk_rows "${ci}")"
  local r0p r1p out
  r0p=$(printf '%04d' "${r0}"); r1p=$(printf '%04d' "${r1}")
  out="${CHUNK_DIR}/chunk_${step}_${r0p}_${r1p}_stokes.npy"
  if [ -f "${out}" ]; then
    echo "### step ${step} chunk ${ci}/${N_CHUNKS} rows [${r0},${r1}) -- already done, skipping"
    return 0
  fi
  echo "### step ${step} chunk ${ci}/${N_CHUNKS} rows [${r0},${r1})"
  "${PY}" scripts/synthesis/generate_tau500_stokes.py \
    --step "${step}" --row-start "${r0}" --row-end "${r1}" \
    --output-dir "${CHUNK_DIR}" --nicole-root "${NICOLE_ROOT}"
}

case "${1:-}" in
  -h|--help)
    usage; exit 0 ;;

  --pilot)
    # One small chunk locally (rows [100,102) of the first step) + timing.
    step="${STEPS[0]}"
    echo "PILOT: step ${step}, rows [100,102), local run"
    "${PY}" scripts/synthesis/generate_tau500_stokes.py \
      --step "${step}" --row-start 100 --row-end 102 \
      --output-dir "${CHUNK_DIR}/_pilot" --nicole-root "${NICOLE_ROOT}"
    echo "Pilot done. Use the reported ms/pixel to size N_CHUNKS:"
    echo "  wall/chunk ~ 21s + (${NX_ROWS}/N_CHUNKS)*480*ms_per_pixel"
    exit 0 ;;

  --merge)
    for step in "${STEPS[@]}"; do
      echo "### merging step ${step}"
      "${PY}" scripts/synthesis/merge_tau500_stokes.py \
        --step "${step}" --chunk-dir "${CHUNK_DIR}" --output-dir "${MERGED_DIR}"
    done
    exit 0 ;;

  --submit-waves)
    # Login-node orchestrator (NOT an sbatch payload itself): submits the full
    # N_CHUNKS*len(STEPS) array in waves of WAVE_SIZE, respecting this
    # account's MaxSubmit cap. Blocks on `sbatch --wait` per wave.
    WAVE_SIZE=8
    TOTAL=$(( N_CHUNKS * NSTEPS ))
    n_waves=$(( (TOTAL + WAVE_SIZE - 1) / WAVE_SIZE ))
    echo "Submitting ${TOTAL} chunks (${NSTEPS} steps x ${N_CHUNKS} chunks) in ${n_waves} waves of <= ${WAVE_SIZE}"
    failed_waves=()
    for (( w=0; w<n_waves; w++ )); do
      lo=$(( w * WAVE_SIZE ))
      hi=$(( lo + WAVE_SIZE - 1 ))
      if [ "${hi}" -ge "${TOTAL}" ]; then hi=$(( TOTAL - 1 )); fi
      echo
      echo "=== wave $((w+1))/${n_waves}: array=${lo}-${hi} ($(date '+%H:%M:%S')) ==="
      if ! sbatch --wait --array="${lo}-${hi}" "$0"; then
        echo "  ⚠ wave $((w+1)) reported a non-zero exit (a chunk may have failed -- check output/synthesis/tau500_gen_*.err)"
        failed_waves+=("$((w+1))")
      fi
    done
    echo
    if [ "${#failed_waves[@]}" -gt 0 ]; then
      echo "Done, but waves [${failed_waves[*]}] reported failures. Re-run --submit-waves to retry -- completed chunks are skipped automatically."
    else
      echo "All ${TOTAL} chunks done."
    fi
    exit 0 ;;

  "")
    # SLURM array task (or a bare local run of task 0 if not under SLURM).
    tid="${SLURM_ARRAY_TASK_ID:-0}"
    total=$(( N_CHUNKS * NSTEPS ))
    if [ "${tid}" -ge "${total}" ]; then
      echo "task ${tid} >= N_CHUNKS*NSTEPS=${total}; nothing to do"; exit 0
    fi
    step_idx=$(( tid / N_CHUNKS ))
    chunk_idx=$(( tid % N_CHUNKS ))
    run_chunk "${STEPS[${step_idx}]}" "${chunk_idx}"
    exit 0 ;;

  *)
    echo "Unknown argument: $1" >&2; usage; exit 1 ;;
esac
