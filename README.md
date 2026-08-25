# MUISCA

**Multi-Scale Convolutional Neural Network for Inverting Stokes Parameters in the Solar Context**

A physics-informed deep learning framework for atmospheric inversion of solar spectropolarimetric observations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Key Components](#key-components)
- [Getting Started](#getting-started) — the `tools/` workflow, in order
- [Training Pipeline](#training-pipeline) — synthesis → training → fine-tuning
- [Notebooks](#notebooks)
- [Configuration](#configuration) — including [Data Sources](#data-sources) and the spectral axis
- [Outputs](#outputs) — metrics CSVs and range of applicability
- [Citation](#citation)

---

## 🌟 Overview

MUISCA is a physics-informed neural network (PINN) that inverts Stokes parameters to recover atmospheric stratifications of:
- **Temperature** (T)
- **Line-of-sight velocity** (V_LOS)
- **Line-of-sight magnetic field** (B_LOS)

The model combines:
1. **Supervised learning** from MURaM 3D MHD simulations
2. **Physics-based regularization** using weak-field approximation (WFA), Doppler shift, and black-body temperature constraints
3. **Multi-scale architecture** for capturing features at different spatial scales

Training data is synthesized by running **NICOLE** forward on MURaM atmospheres remapped onto
the τ₅₀₀ continuum optical-depth scale — the `nicole_tau500` data source, which is the
default. The alternative, `muram_legacy`, uses the older Rosseland-τ Stokes and a different
optical-depth grid; it is kept only for reading old checkpoints. The two are deliberately
isolated from each other (separate caches and normalization statistics), so switching
sources cannot silently mix data. See [Data Sources](#data-sources).

---

## 📁 Repository Structure

```
Tesis_maestria_OAN/
├── models/                          # Neural network architectures
│   ├── mscnn_model.py              # Base multi-scale CNN
│   └── pinn_mscnn_model.py         # Physics-informed extension
│
├── utils/                           # Utility modules
│   ├── muram_data.py               # MURaM data loading, LSF/resample, Bz balancing
│   ├── modest_data.py              # MODEST/Hinode observation loading & SPINOR atmosphere
│   ├── hinode_wavelengths.py       # SINGLE source of truth for the Hinode spectral axis
│   ├── normalizer.py               # MHD (per-τ asinh for Bz) & Stokes normalization
│   ├── physics_utils.py            # Physics approximations (WFA, Doppler, Temperature)
│   ├── synthesis.py                # NICOLE bridge: cube synthesis and inversion
│   ├── pixel_sampling.py           # |B_LOS|-stratified pixel selection for synthesis
│   ├── analysis.py                 # Shared analysis pipelines & diagnostic plot classes
│   ├── cache_manage.py             # HDF5 caches for processed / balanced steps
│   └── model_prof_tools.py         # NICOLE model & profile binary I/O
│
├── scripts/                         # Training & experiment scripts
│   ├── base_training.py            # TrainingConfig + core training loop
│   ├── compute_normalization_stats.py # Builds MHD/Stokes normalization JSONs
│   ├── finetune.py                 # Bz-balanced fine-tuning of a trained checkpoint
│   ├── synthesis/                  # τ₅₀₀ Stokes generation & NICOLE comparison
│   │   ├── generate_tau500_stokes.py  # Chunked NICOLE synthesis per MURaM step
│   │   └── merge_tau500_stokes.py     # Reassembles chunks into stokes_<step>_nicole_tau500.npy
│   └── experiments/
│       └── ablation_study.py       # Physics regularization ablation study (5 configurations)
│
├── scripts/analysis/                # Post-training diagnostics
│   ├── muram_analysis.py           # Diagnostic maps on MURaM steps
│   ├── modest_analysis.py          # Diagnostic comparison on MODEST data
│   └── distributions_analysis.py   # Stokes/MHD histogram comparisons
│
├── tools/                           # Run wrappers (HPC/local)
│   ├── run_experiments.sh          # SLURM-ready ablation launcher
│   ├── compute_normalization_stats.sh # SLURM-ready normalization launcher
│   ├── fine_tune.sh                # SLURM-ready Bz-balanced fine-tuning launcher
│   ├── generate_tau500_stokes.sh   # τ₅₀₀ Stokes generation (multi-step)
│   ├── run_nicole_synthesis.sh     # MUISCA → NICOLE forward-synthesis comparison
│   └── generate_analysis.sh        # Unified analysis launcher (MURaM/MODEST)
│
├── notebooks/                       # Jupyter notebooks (documentation & analysis)
│   ├── 1-muram_mhd_data.ipynb      # MURaM MHD data loading & optical depth mapping
│   ├── 2-muram_stokes_data.ipynb   # Synthetic Stokes profiles & spectral degradation
│   ├── 3-stokes_mhd_relations.ipynb # Physics approximations (WFA, Doppler, Temperature)
│   ├── 4-modest_data.ipynb          # MODEST inversion code outputs (optional)
│   ├── 5-mscnn_architecture.ipynb   # Multi-Scale CNN design & architecture
│   └── 6-pinn_mscnn_model.ipynb     # Physics-Informed training & loss analysis
│
├── data/                            # Data directory (mostly gitignored)
│   ├── muram-simulation/            # MURaM snapshots + synthesized τ₅₀₀ Stokes/atmospheres
│   ├── normalization_stats/         # Per-data-source normalization statistics
│   │   └── nicole_tau500/           #   isolated from muram_legacy's stats
│   ├── nicole_assets/               # NICOLE.input templates + LINES atomic data
│   ├── csv/                         # Opacity tables (kappa.0.dat), etc.
│   └── hinode-MODEST/               # Observations, PSF and spectral response files
│
└── output/                          # Training outputs
    ├── experiments/<experiment_name>/<variation>/   # Ablation results per variation
    │       ├── final_model.pth              # weights (+ config, for fine-tune outputs)
    │       ├── experiment_config.json       # the run's real settings; read by fine-tuning
    │       └── logs/, checkpoints/
    ├── fine-tune/<experiment_name>-finetuned/<variation>/
    └── synthesis/                   # NICOLE forward-synthesis comparisons
```

Caches live at the repo root and are gitignored: `.muram_cache_nicole_tau500/` (raw,
post-resample data — **not** normalized, so it survives a normalization change),
`.muram_balanced_cache_nicole_tau500/` (post-balancing tensors — these *are* normalized and
must be cleared when normalization changes), and `.modest_cache/` (observations).

---

## 🔑 Key Components

### 1. **Models** (`models/`)

#### `mscnn_model.py`
- **MSCNNInversionModel**: Base multi-scale convolutional neural network
- Processes Stokes I and V profiles at multiple scales (1×, 2×, 3×)
- Architecture: Conv blocks → Multi-scale fusion → Dense layers → Atmospheric stratification output
- Output width is `3 × n_logtau`, set by the active optical-depth grid: **135 values (45 per
  T, V_LOS, B_LOS)** for `nicole_tau500`. Older `muram_legacy` checkpoints have 63 (21 each),
  so a checkpoint's final-layer shape tells you which grid it was trained on.

#### `pinn_mscnn_model.py`
- **PhysicsInformedMSCNN**: Extends MSCNN with physics regularization
- Computes physics losses in **physical units** (Gauss, km/s, Kelvin)
- Three physics terms:
  1. **WFA B_LOS loss**: Weak-field approximation comparison
  2. **Doppler V_LOS loss**: Doppler shift comparison
  3. **Temperature loss**: Black-body temperature comparison
- Supports two modes for each physics term:
  - `tau_averaged`: Integrate over optical depth
  - `single_height`: Compare at specific log(τ) level

### 2. **Data Processing & Utilities** (`utils/`)

#### `muram_data.py`
- **MhdData**: Loads MURaM 3D MHD cubes (T, Vz, Bz)
- **StokesData**: Loads/processes synthetic Stokes profiles
- **MuramStepDataset**: PyTorch Dataset for a single MURaM step
- Handles:
  - Optical depth remapping
  - Continuum normalization
  - Hinode/SP LSF convolution
  - Wavelength resampling

#### `normalizer.py`
- **MhdNormalizer**: Per-τ normalization for T, V, B using Welford's online algorithm
- **StokesNormalizer**: Global normalization for Stokes I, Q, U, V
- Ensures stable training and physics computation
- Supports inverse transformation (denormalization) to physical units

#### `physics_utils.py`
- **ApproxInversions**: Computes physics-based approximations
  - `compute_blos_wfa()`: Weak-field approximation for B_LOS
  - `compute_vlos_doppler()`: Doppler shift for V_LOS
  - `compute_temperature_blackbody()`: Black-body temperature from continuum
  - Handles Gaussian fitting, line core identification, and error detection

#### `analysis.py`
- **AnalysisModelPipeline**: Centralized loading of trained experiment models/configs
- Builds runtime `TrainingConfig` from saved experiment metadata
- Produces denormalized model predictions and shared tau-grid alignment helpers
- Includes **MuramDiagnosticPlots** and **ModestDiagnosticPlots** for standardized outputs

#### `cache_manage.py`
- **DataCache**: HDF5 caching layer for processed MURaM/MHD/Stokes/physics tensors
- Enforces cache compatibility with config hash and `logtau_values`
- Reduces repeated preprocessing in normalization, training, and analysis scripts

#### `modest_data.py`
- **ModestData**: End-to-end MODEST/Hinode data loader + processor
- Supports deconvolution, optional upsampling/smoothing, polarization masking, and diagnostics
- Provides FITS-based access to observed/inverted profiles and atmospheric products

#### `model_prof_tools.py`
- Legacy utility set for reading/writing/checking NICOLE model/profile formats
- Includes binary/ASCII format validation helpers used in inversion-oriented workflows

### 3. **Training** (`scripts/`)

#### `base_training.py`
- **Core training pipeline**:
  - `TrainingConfig`: Configuration dataclass with all hyperparameters
  - `train_epoch()`: Epoch-level training over multiple MURaM steps
  - `train_one_step()`: Single simulation step training
  - `validate()`: Validation on held-out steps
  - `MetricsLogger`: CSV logging for loss components
- Supports:
  - Interleaved training across simulation steps
  - Checkpoint saving/resuming
  - Fixed learning rate training (Adam)

#### `experiments/ablation_study.py`
- **Physics regularization ablation study**
- Tests 5 configurations:
  1. All physics terms (WFA + Doppler + Temperature)
  2. WFA only
  3. Doppler only
  4. Temperature only (black-body)
  5. No physics (pure supervised learning)
- **ExperimentTracker**: Aggregates results across experiments
- Generates:
  - Comparison plots and summary tables
  - Individual loss curve plots
  - Improvement matrices over baseline
  - JSON summary files

#### `compute_normalization_stats.py`
- Computes and saves `mhd_normalization.json` and `stokes_normalization.json`
- Supports explicit `logtau_values` and resume from intermediate states
- Can reuse shared cache semantics via `load_and_prepare_step`

### 4. **Analysis Scripts** (`scripts/analysis/`)

#### `analysis/muram_analysis.py`
- Loads trained experiment checkpoints and generates MURaM diagnostic plots
- Uses shared cache and normalizers to compare denormalized predictions vs ground truth

#### `analysis/modest_analysis.py`
- Runs inference and diagnostics on MODEST products (whole FOV or cropped regions)
- Supports polarization masking and configurable model subset comparisons

### 5. **Automation Wrappers** (`tools/`)

#### `run_experiments.sh`
- SLURM-ready launcher for `scripts/experiments/ablation_study.py`
- Centralizes step ranges, lambda weights, physics modes, and selected experiment branches

#### `compute_normalization_stats.sh`
- SLURM-ready launcher for normalization-stat computation script
- Supports cache toggle, resume mode, and explicit/range logtau configuration

#### `generate_analysis.sh`
- Unified entry point to run `muram_analysis.py`, `modest_analysis.py`, or both
- Exposes runtime flags for crop mode and analysis selection

---

## 🚀 Getting Started

### Prerequisites

```bash
# Required packages
conda create -n muisca python=3.9
conda activate muisca
pip install torch torchvision astropy scipy tqdm matplotlib
```

### Quick Start

The `tools/*.sh` wrappers are the intended entry points: each holds its configuration in a
block at the top, resolves every path against an absolute `MUISCA_ROOT`, and can be run
directly or submitted with `sbatch`. Run them in this order.

0. **Synthesize training Stokes** — only needed for MURaM steps that don't have
   `stokes_<step>_nicole_tau500.npy` yet. Expensive (chunked NICOLE runs), so check first.
   ```bash
   ./tools/generate_tau500_stokes.sh --submit-waves   # generate
   ./tools/generate_tau500_stokes.sh --merge          # reassemble chunks
   ```

1. **Compute normalization statistics** — required before the first training run, and again
   whenever the training step list or the normalization code changes.
   ```bash
   sbatch -M fisica tools/compute_normalization_stats.sh
   ```
   Fit these on **training steps only**; including the held-out test step leaks it into the
   preprocessing the model learns in.

2. **Train** (ablation over the physics terms):
   ```bash
   sbatch -M fisica tools/run_experiments.sh
   ```
   Or directly, for a short pilot:
   ```bash
   python scripts/experiments/ablation_study.py \
       --experiment_name pilot --min_step 110 --max_step 130 --step_size 10 \
       --n_epochs 5 --experiments wfa_only
   ```

3. **Fine-tune on Bz-balanced data** (optional; starts from a trained checkpoint):
   ```bash
   sbatch -M fisica tools/fine_tune.sh \
       --experiment-name <experiment_name> --variations wfa_only,no_physics \
       --finetune-epochs 20
   ```

4. **Analyze**:
   ```bash
   ./tools/generate_analysis.sh          # edit EXPERIMENT_ROOT/MODEL_TYPES at the top first
   ```

> **Cluster note:** jobs target `-w maxwell` on `--cluster=fisica`. Use
> `--partition=gpu.cecc` — `cpu.cecc` does not contain maxwell (the job will never schedule)
> and `boltzmann.cpu` rejects submission with `Invalid qos specification`. Verify a job
> actually queued with `squeue -M fisica -u $USER`; a rejected `sbatch` leaves no trace.

---

## 🔄 Training Pipeline

### Training Flow

Offline, once per MURaM step (`scripts/synthesis/`, expensive — cached as `.npy`/`.npz`):

```
MURaM cube (T, Vz, Bz on geometric height)
   ↓  remap onto the τ₅₀₀ continuum optical-depth grid (45 levels, −3.0 → 1.4)
   ↓  NICOLE forward synthesis over the whole frame in one invocation
stokes_<step>_nicole_tau500.npy  +  atmos_<step>_tau500.npz
```

Then per training run:

```
1. Load the synthesized Stokes + atmosphere for a step
   ↓   (log τ grid is checked against the file; a mismatch raises)
2. Apply the Hinode LSF convolution
   ↓
3. Resample onto the FITS-derived Hinode wavelength axis (112 points)
   ↓
4. Compute physics approximations (WFA, Doppler, Temperature)
   ↓
5. Normalize — Stokes standardized; Bz through a per-τ asinh transform
   ↓
6. Optional pixel selection (region mask and/or |B_LOS|-bin balancing)
   ↓
7. Cache to HDF5, then mini-batch over spatial pixels
   ↓
8. Forward pass → losses (MSE + physics) → backward → optimize
   ↓
9. Interleave the next step; repeat per epoch
```

Steps 1–5 are `load_and_prepare_step` in `scripts/base_training.py`; `TrainingConfig` in the
same file holds every hyperparameter and path.

### Loss Function

```
Total Loss = MSE_loss + λ_WFA × WFA_loss + λ_Doppler × Doppler_loss + λ_Temp × Temp_loss
```

- Physics losses are computed in **physical units** (Gauss, km/s, Kelvin), so the λ values
  are not all the same order — see `tools/run_experiments.sh` for the values actually used.
- The WFA term can be gated during training (`--wfa-gate-mode plateau|threshold|off`) so it
  only switches on once the supervised loss has settled.

### Fine-Tuning

`scripts/finetune.py` resumes from a trained checkpoint with **mandatory** |B_LOS| balancing,
to counter how rare strong fields are in MURaM (~1% of pixels above 350 G, while a sunspot
crop is well above that at its 90th percentile). Pixels are binned by |B_LOS| at a fixed
optical depth and the bins are equalized by oversampling the rare ones, with the top edge
capped and the replication factor bounded so a handful of extreme pixels cannot be copied
thousands of times.

It reads the base run's `experiment_config.json` from beside the checkpoint (the `.pth` from
base training carries no config), so the step range, log(τ) grid and balancing depth all come
from the original run. **Normalizers are never refit** — the Bz asinh scale is baked into the
pretrained weights' output space, so `--steps` may change which steps are used but never the
statistics.

---

## 📓 Notebooks

Interactive tutorials and analysis notebooks in `notebooks/`:

### `1-muram_mhd_data.ipynb`
- **Purpose**: Understand MURaM MHD data structure and optical depth mapping
- **Contents**:
  - Loading MHD cubes (T, Vz, Bz) from simulation snapshots
  - Computing optical depth (log τ) from opacity tables using Welford's algorithm
  - Remapping atmospheric quantities from geometric to optical depth coordinates via interpolation
  - Visualization of stratifications and surfaces at constant τ
  - Data normalization (per-τ and global statistics)
  - Checks for out-of-bounds values in opacity interpolation

### `2-muram_stokes_data.ipynb`
- **Purpose**: Working with synthetic Stokes profiles from MURaM
- **Contents**:
  - Loading Stokes I, Q, U, V profiles
  - Spectral degradation (Hinode/SP LSF convolution and resampling)
  - Continuum normalization procedures
  - Stokes profile visualization and statistics
  - Quality checks for synthetic data

### `3-stokes_mhd_relations.ipynb`
- **Purpose**: Physics-based approximations linking Stokes profiles to atmospheric parameters
- **Contents**:
  - Weak-Field Approximation (WFA) for B_LOS estimation
  - Doppler shift analysis for V_LOS via Gaussian fitting
  - Black-body temperature approximation from continuum intensity
  - Error handling and failure detection in physics approximations
  - Comparison with ground truth atmospheric parameters
  - Validation of physics relationships

### `4-modest_data.ipynb`
- **Purpose**: Working with MODEST inversion code outputs (optional)
- **Contents**:
  - Loading MODEST atmospheric inversions (if available)
  - Comparing MODEST results with MURaM ground truth
  - Understanding traditional inversion approaches
  - Benchmarking against physics-based methods

### `5-mscnn_architecture.ipynb`
- **Purpose**: Multi-Scale CNN architecture design and analysis
- **Contents**:
  - Network architecture overview and design rationale
  - Multi-scale feature extraction (1×, 2×, 3× scales)
  - Input/output dimensions and data flow
  - Forward pass visualization
  - Parameter count and memory requirements
  - Ablation studies on architecture components

### `6-pinn_mscnn_model.ipynb`
- **Purpose**: Physics-Informed Neural Network integration and training
- **Contents**:
  - Physics loss computation (WFA, Doppler, Temperature)
  - Training loop with physics regularization
  - Loss component analysis and visualization
  - Training dynamics and convergence analysis

---

## ⚙️ Configuration

### Key Hyperparameters

```python
# Training
n_epochs = 30                  # Number of epochs
batch_size = 512               # Spatial pixels per batch
learning_rate = 1e-3           # Adam learning rate
gradient_clip = 1.0            # Gradient clipping threshold

# Physics Regularization
lambda_wfa = 0.01              # WFA B_LOS weight
lambda_doppler = 0.01          # Doppler V_LOS weight
lambda_temp = 0.01             # Temperature weight

# Physics Modes
blos_physics_mode = 'tau_averaged'      # or 'single_height'
vlos_physics_mode = 'single_height'     # Target log(τ) = -1.0
temp_physics_mode = 'single_height'     # Target log(τ) = 0.0 (photosphere)
```

### Data Configuration

```python
# Data source (TrainingConfig.data_source)
data_source = "nicole_tau500"   # default; or "muram_legacy" for old checkpoints

# MURaM simulation
min_step, max_step, step_size = 110, 130, 10   # only synthesized steps are usable
nx, ny = 480, 480              # Spatial dimensions
nz = 256                       # Vertical layers
z_max = 250                    # Limit vertical extent

# Optical depth grid (nicole_tau500): 45 levels, must match the atmos_*_tau500.npz files
logtau_min, logtau_max, logtau_step = -3.0, 1.4, 0.1
```

Only steps that have been synthesized can be trained on. Check with:

```bash
ls data/muram-simulation/stokes_*_nicole_tau500.npy
```

`load_source_arrays` refuses to run if the config's log(τ) grid does not match the grid
stored in `atmos_<step>_tau500.npz`, so a mismatch fails loudly rather than silently
misaligning the targets.

<a id="data-sources"></a>
### Data Sources

`nicole_tau500` and `muram_legacy` are kept fully separate so they can never be mixed:

| | `nicole_tau500` (default) | `muram_legacy` |
|---|---|---|
| Stokes file | `stokes_<step>_nicole_tau500.npy` | `stokes_<step>.npy` |
| Atmosphere | `atmos_<step>_tau500.npz` | remapped at load time |
| log(τ) grid | 45 levels, −3.0 → 1.4 | 21 levels, −2.0 → 0.0 |
| Cache | `.muram_cache_nicole_tau500/` | `.muram_cache/` |
| Normalization | `normalization_stats/nicole_tau500/` | `normalization_stats/` |

### Spectral Axis

The Hinode/SOT-SP wavelength axis is derived from the MODEST FITS header
(`WLREF`/`WLMIN`/`WLMAX`) by `utils/hinode_wavelengths.py`, which is the **only** place it is
defined. Training profiles are resampled onto exactly the axis the observations are loaded
on. There is deliberately no fallback — a missing header raises — and
`StokesData.resample_to_hinode` asserts that the grid in use matches the canonical one.
Hardcoding `CRVAL1`/`CDELT1`/`CRPIX1` anywhere reintroduces a ~0.078 Å (≈3.7 km/s) offset
between training and observation.

---

## 🧪 Running Experiments

### Ablation Study

Test the contribution of each physics term:

```bash
# All physics terms
python ablation_study.py --lambda_wfa 0.01 --lambda_doppler 0.01 --lambda_temp 0.01

# WFA only
python ablation_study.py --lambda_wfa 0.01 --lambda_doppler 0.0 --lambda_temp 0.0

# Doppler only
python ablation_study.py --lambda_wfa 0.0 --lambda_doppler 0.01 --lambda_temp 0.0

# Temperature only
python ablation_study.py --lambda_wfa 0.0 --lambda_doppler 0.0 --lambda_temp 0.01

# No physics (baseline)
python ablation_study.py --lambda_wfa 0.0 --lambda_doppler 0.0 --lambda_temp 0.0
```

## 📊 Outputs

### Training Outputs

```
output/experiments/<experiment_name>/
├── all_physics_terms/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── checkpoint_epoch_030.pth
│   ├── logs/
│   │   ├── epoch_log.csv         # Epoch-level metrics
│   │   ├── batch_log.csv         # Batch-level metrics
│   ├── all_physics_terms_loss_curves.png
│   └── experiment_config.json
├── wfa_only/
├── doppler_only/
├── black_body_only/
├── no_physics/
├── experiment_results.json        # Summary of all experiments
└── comparison_plots.png           # Side-by-side comparison
```

### Metrics

- **Training loss**: MSE + physics components
- **Validation loss**: Total loss on held-out steps
- **Test metrics**: correlation, RMSE and τ-averaged RRMSE for B_LOS, V_LOS and T

The analysis scripts write two CSVs per model into `images/analysis/.../<variation>/`:

| file | contents |
|---|---|
| `metrics_summary.csv` | one row per parameter × log(τ): `corr`, `r2`, `rmse`, `rrmse`, `mae`, `nmae`, `bias` |
| `metrics_by_field_strength.csv` | the same comparison split into bins of **true \|B_LOS\|** |

The second one exists because aggregate metrics are dominated by the weak-field majority and
by outliers, which can hide both where the model is reliable and where it is extrapolating.
Reading `bias` alongside `rmse` is also how you separate a systematic offset from a heavy
tail: when `bias ≈ mae` the error is almost entirely one-directional, and when the mean and
median disagree in sign a minority of pixels is driving the aggregate.

Range of applicability worth keeping in mind: MURaM steps 110–130 hold plenty of pixels below
~500 G, few above ~950 G, and none above ~1500 G, while the Hinode sunspot crops reach ~3.7 kG.
Rows above that range describe extrapolation, not skill.

---

## 🤝 Contributing

To add new physics terms or modify the architecture:

1. **New physics term**:
   - Add approximation method in `utils/physics_utils.py`
   - Add loss computation in `models/pinn_mscnn_model.py`
   - Update `TrainingConfig` in `scripts/base_training.py`

2. **Architecture changes**:
   - Modify `models/mscnn_model.py` or `models/pinn_mscnn_model.py`
   - Keep the output dimension at `3 × n_logtau` (135 for the current τ₅₀₀ grid), derived
     from the config rather than hardcoded

3. **New experiments**:
   - Create script in `scripts/experiments/`
   - Use `TrainingConfig` for consistency

---

## 📚 Key References

1. **Weak-Field Approximation**: Landi Degl'Innocenti & Landolfi, "Polarization in Spectral Lines", 2004
2. **MURaM**: Vögler et al., "The CO5BOLD/MURaM Code", 2005

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@mastersthesis{agudelo2025muisca,
  title={MUISCA: Multi-Scale Convolutional Neural Network for Inverting Stokes Parameters},
  author={Agudelo, Julian},
  year={2025},
  school={Universidad Nacional de Colombia}
}
```

---

## 📧 Contact

For questions or collaborations:
- **Author**: Juan Esteban Agudelo Ortiz
- **Email**: juagudeloo@unal.edu.co
- **Institution**: Observatorio Astronómico Nacional, Universidad Nacional de Colombia

---

## 📄 License

This project is developed as part of a Master's thesis at Universidad Nacional de Colombia.

---

**Last Updated**: 2026-03-05