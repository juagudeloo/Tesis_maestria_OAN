#!/bin/bash -l
#SBATCH --job-name=tau500_gen_single
#SBATCH --cluster=fisica
#SBATCH -wmaxwell
#SBATCH --partition=gpu.cecc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/tau500_gen_single_%j.out
#SBATCH --error=/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis/tau500_gen_single_%j.err
#
# Single-node, single-job variant of generate_tau500_stokes.sh (issue #5).
#
# The array-based approach (--submit-waves) pays this cluster's per-job
# full-directory stage-in/stage-out cost (copies the whole MUISCA checkout,
# several hours observed) once per array TASK -- with 160 chunks that means
# paying it 160 times, which dominated over the actual ~20 min/chunk of
# NICOLE compute and is what filled the shared disk quota. This variant pays
# that cost exactly ONCE (one job) and runs one NICOLE invocation at a time
# (no inter-task I/O contention on the node either). Trade-off: fully
# sequential, so wall time ~= sum of all remaining chunks' compute time.
#
# Same chunk grid and output naming as generate_tau500_stokes.sh, so chunks
# already produced by the array run are detected (by file existence) and
# skipped -- this picks up exactly where that approach left off.
#
# USAGE:  sbatch tools/generate_tau500_stokes_single_node.sh
# After it completes (it merges automatically at the end), nothing else is
# needed -- data/muram-simulation/{stokes,atmos}_{step}_*.{npy,npz} land
# directly.

module purge
module load envs/anaconda3
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter

# ==============================================================================
# CONFIGURATION -- keep in sync with generate_tau500_stokes.sh
# ==============================================================================
STEPS=(110 120 130 198) # MURaM steps: 110-130 contiguous for train/val, 198 for the OOD test split
N_CHUNKS=40              # row-chunks per step (480 rows / N_CHUNKS rows each)
NX_ROWS=480              # MURaM frame rows (nx)

MUISCA_ROOT="/scratchsan/observatorio/juagudeloo/MUISCA"
NICOLE_ROOT="/scratchsan/observatorio/juagudeloo/NICOLE_v16.06"
CHUNK_DIR="${MUISCA_ROOT}/.muram_tau500_cache/chunks"
MERGED_DIR="${MUISCA_ROOT}/data/muram-simulation"
PY="/homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter/bin/python"

# ==============================================================================

cd "${MUISCA_ROOT}"
NSTEPS=${#STEPS[@]}
TOTAL=$(( N_CHUNKS * NSTEPS ))

# chunk i of N_CHUNKS -> row range [r0, r1)  (identical to generate_tau500_stokes.sh)
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
  echo "### step ${step} chunk ${ci}/${N_CHUNKS} rows [${r0},${r1}) ($(date '+%F %H:%M:%S'))"
  "${PY}" scripts/synthesis/generate_tau500_stokes.py \
    --step "${step}" --row-start "${r0}" --row-end "${r1}" \
    --output-dir "${CHUNK_DIR}" --nicole-root "${NICOLE_ROOT}"
}

echo "Processing ${TOTAL} chunks (${NSTEPS} steps x ${N_CHUNKS} chunks) sequentially in one job"
echo "Start: $(date '+%F %H:%M:%S')"
for (( tid=0; tid<TOTAL; tid++ )); do
  step_idx=$(( tid / N_CHUNKS ))
  chunk_idx=$(( tid % N_CHUNKS ))
  run_chunk "${STEPS[${step_idx}]}" "${chunk_idx}"
done

echo
echo "=== merging ($(date '+%F %H:%M:%S')) ==="
for step in "${STEPS[@]}"; do
  echo "### merging step ${step}"
  "${PY}" scripts/synthesis/merge_tau500_stokes.py \
    --step "${step}" --chunk-dir "${CHUNK_DIR}" --output-dir "${MERGED_DIR}"
done
echo "Done: $(date '+%F %H:%M:%S')"
