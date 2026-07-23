# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MUISCA** — a physics-informed multi-scale CNN that inverts solar spectropolarimetric observations (Stokes I and V profiles) to recover atmospheric stratifications of temperature (T), line-of-sight velocity (V_LOS), and line-of-sight magnetic field (B_LOS) across 21 optical depth levels.

## Environment

```bash
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter
```

Key dependencies: `torch`, `astropy`, `scipy`, `h5py`, `matplotlib`, `tqdm`.

## Common Commands

**Compute normalization stats (required before first training run):**
```bash
python scripts/compute_normalization_stats.py --min_step 60 --max_step 200
```

**Run ablation study (main training entry point):**
```bash
python scripts/experiments/ablation_study.py \
    --n_epochs 30 \
    --min_step 150 \
    --max_step 155 \
    --learning_rate 1e-3 \
    --lambda_wfa 0.01 \
    --lambda_doppler 0.01 \
    --lambda_temp 0.01
```

**Post-training analysis:**
```bash
# On MURaM synthetic data (ground truth available)
python scripts/analysis/muram_analysis.py \
    --experiment-root <experiment_name> \
    --model-types all_physics_terms wfa_only

# On real Hinode/SP observations (MODEST)
python scripts/analysis/modest_analysis.py \
    --experiment-root <experiment_name> \
    --model-types all_physics_terms

# Distribution analysis (Stokes/MHD histograms)
tools/generate_analysis.sh distributions --distribution-target both
```

**Fine-tuning (Bz-balanced, run after ablation study):**
```bash
python scripts/finetune.py \
    --experiment-name <experiment_name> \
    --variations wfa_only,all_physics_terms \
    --finetune-epochs 20
```

**HPC submission (Maxwell cluster):**
```bash
sbatch tools/run_experiments.sh
sbatch tools/compute_normalization_stats.sh
sbatch tools/generate_analysis.sh
sbatch tools/fine_tune.sh --experiment-name <name> --variations wfa_only,all_physics_terms
```

There is no linting or unit test infrastructure in this research codebase.

## Architecture

### Data Flow

```
MURaM 3D simulation cubes (T, Vz, Bz at geometric height)
  → optical depth remapping (utils/muram_data.py: MhdData, StokesData)
  → synthetic Stokes profiles + LSF spectral degradation
  → normalization (utils/normalizer.py: MhdNormalizer, StokesNormalizer)
  → HDF5 cache (utils/cache_manage.py: MuramDataCache)
  → MuramStepDataset → DataLoader → training
```

### Model Architecture

`MSCNNInversionModel` (`models/mscnn_model.py`) takes `(batch, 2, 112)` — Stokes I and V at 112 wavelengths — and passes it through `MultiScaleFeatureMapping` which coarse-grains the input at scales 1×, 2×, 3×, applies Conv1d+ReLU+MaxPool at each scale, concatenates features, then passes through 4 dense layers to output `(batch, 63)` = 3 parameters × 21 τ levels.

`PhysicsInformedMSCNN` (`models/pinn_mscnn_model.py`) extends this base with physics regularization losses.

### Loss Function

```
Total = MSE + λ_WFA × WFA_loss + λ_Doppler × Doppler_loss + λ_Temp × Temp_loss + λ_tail × Huber_tail_loss
```

Physics approximations live in `utils/physics_utils.py` (`ApproxInversions`):
- **WFA**: relates Stokes V amplitude to B_LOS via Landé g-factor
- **Doppler**: line core shift → V_LOS
- **Black-body**: continuum intensity → temperature

### Training Pipeline

`scripts/base_training.py` defines `TrainingConfig` (dataclass with all hyperparameters and data paths) and the training loop functions `train_epoch()`, `train_one_step()`, `validate()`, and `MetricsLogger` (CSV logging). Training interleaves batches across multiple simulation timesteps within each epoch.

`scripts/experiments/ablation_study.py` runs 5 configurations: all physics terms, WFA only, Doppler only, Temperature only, and no physics (baseline).

### Fine-Tuning Pipeline

`scripts/finetune.py` loads a trained checkpoint from `output/experiments/` and resumes training with **mandatory Bz balancing** — pixels are resampled across 12 equal bins of |B_LOS| at the deepest τ level, so that extreme-field regions (which are rare in raw MURaM data) are seen more often. Fine-tuning outputs go to `output/fine-tune/<experiment_name>-finetuned/<variation>/`. See `docs/how-to-fine-tune.md` for the full workflow and `docs/magnetic_field_balancing_and_finetuning.md` for theory.

### Real Observation Pipeline

`utils/modest_data.py` (`ModestData`) handles end-to-end loading of Hinode/SP MODEST observations: deconvolution, optional upsampling/smoothing, and polarization masking before passing to the trained model. Analysis on real data is done via `scripts/analysis/modest_analysis.py` which uses `utils/analysis.py` (`ModestDiagnosticPlots`). Distribution analysis (histograms of Stokes and MHD values) is in `scripts/analysis/distributions_analysis.py`.

### Key Data Paths (hardcoded in TrainingConfig)

- MURaM simulation: `/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/data/muram-simulation/`
- MODEST/Hinode: `/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/data/hinode-MODEST/`
- Normalization stats: `data/normalization_stats/mhd_normalization.json`, `stokes_normalization.json`
- Caches: `.muram_cache/`, `.modest_cache/`, `.muram_balanced_cache/` (Bz-balanced fine-tuning)

### Notebooks

Six notebooks document the development incrementally:

| Notebook | Content |
|----------|---------|
| `1-muram_mhd_data.ipynb` | MHD data loading & optical depth mapping |
| `2-muram_stokes_data.ipynb` | Synthetic Stokes profiles & spectral degradation |
| `3-stokes_mhd_relations.ipynb` | Physics approximations (WFA, Doppler, black-body) |
| `4-modest_data.ipynb` | MODEST/Hinode observation handling |
| `5-mscnn_architecture..ipynb` | Multi-scale CNN design (note: double dot in filename) |
| `6-pinn_mscnn_model.ipynb` | Physics-informed training & loss analysis |

### Other Utilities

- `utils/model_prof_tools.py` — NICOLE model/profile binary I/O (used for synthesis/inversion integration; see `docs/nicole_integration_guide.md`)
- `tools/generate_analysis.sh` — shell wrapper over all analysis scripts; edit `EXPERIMENT_ROOT` and `MODEL_TYPES` at the top before running
