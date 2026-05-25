# Project Guidelines

## Project Context
MUISCA is a physics-informed multi-scale CNN that inverts solar spectropolarimetric observations (Stokes I and V) into atmospheric stratifications of temperature, line-of-sight velocity, and line-of-sight magnetic field across 21 optical-depth levels.

## Environment And Data Preconditions
- Use the conda environment at `/homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter`.
- This repository is data-dependent and uses hardcoded paths under `/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/data/`.
- Before running training, ensure normalization files exist at `data/normalization_stats/mhd_normalization.json` and `data/normalization_stats/stokes_normalization.json`.
- If missing, run:

```bash
python scripts/compute_normalization_stats.py --min_step 60 --max_step 200
```

## Build And Run
- Main training entry point:

```bash
python scripts/experiments/ablation_study.py --n_epochs 30 --min_step 150 --max_step 155 --learning_rate 1e-3 --lambda_wfa 0.01 --lambda_doppler 0.01 --lambda_temp 0.01
```

- Post-training analysis:

```bash
python scripts/analysis/muram_analysis.py --experiment_dir output/experiments/<name>/all_physics_terms
python scripts/analysis/modest_analysis.py --experiment_dir output/experiments/<name>/all_physics_terms
```

- HPC runs (Maxwell cluster, SLURM):

```bash
sbatch tools/compute_normalization_stats.sh
sbatch tools/run_experiments.sh
sbatch tools/generate_analysis.sh
```

## Architecture Landmarks
- `models/mscnn_model.py`: Base multi-scale CNN inversion model.
- `models/pinn_mscnn_model.py`: Physics-informed extension and combined loss terms.
- `scripts/base_training.py`: `TrainingConfig`, train/validate loops, metrics logging.
- `utils/muram_data.py`: MURaM ingestion, optical-depth remapping, synthetic Stokes pipeline.
- `utils/modest_data.py`: MODEST/Hinode observation loading and preprocessing.
- `utils/normalizer.py`: Normalization logic for MHD and Stokes data.
- `utils/physics_utils.py`: WFA, Doppler, and temperature approximation utilities.

## Conventions Specific To This Repository
- Scripts commonly inject project root into `sys.path`; preserve this pattern when adding new executable scripts.
- Configuration is centered on `TrainingConfig` in `scripts/base_training.py` with CLI overrides.
- Cache directories (`.muram_cache/`, `.modest_cache/`) are part of normal operation; handle cache invalidation explicitly.
- This is a research codebase: there is no formal linting or unit-test suite. Validate changes with targeted script runs and sanity checks.

## Pitfalls To Avoid
- Do not assume relative data paths; verify hardcoded absolute data locations exist before running expensive jobs.
- Do not start training before normalization stats are generated.
- Be careful changing optical-depth or wavelength-grid assumptions; downstream processing expects stable conventions.
- Avoid deleting caches during active training runs.

## Link, Do Not Duplicate
- Project overview and workflow details: `README.md`.
- Additional architecture and command context: `CLAUDE.md`.
- Development notebooks that document the pipeline: `notebooks/`.