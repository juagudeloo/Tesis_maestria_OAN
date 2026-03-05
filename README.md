# MUISCA

**Multi-Scale Convolutional Neural Network for Inverting Stokes Parameters in the Solar Context**

A physics-informed deep learning framework for atmospheric inversion of solar spectropolarimetric observations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Key Components](#key-components)
- [Getting Started](#getting-started)
- [Training Pipeline](#training-pipeline)
- [Notebooks](#notebooks)
- [Configuration](#configuration)
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

---

## 📁 Repository Structure

```
Tesis_maestria_OAN/
├── models/                          # Neural network architectures
│   ├── mscnn_model.py              # Base multi-scale CNN
│   └── pinn_mscnn_model.py         # Physics-informed extension
│
├── utils/                           # Utility modules
│   ├── muram_data.py               # MURaM data loading & preprocessing
│   ├── modest_data.py              # MODEST data loading & processing
│   ├── normalizer.py               # Data normalization utilities
│   ├── physics_utils.py            # Physics approximations (WFA, Doppler, Temperature)
│   ├── grad_norm.py                # GradNorm loss balancing
│   ├── analysis.py                 # Shared analysis pipelines & diagnostic plot classes
│   ├── cache_manage.py             # HDF5 cache for processed steps
│   └── model_prof_tools.py         # Model profiling & performance tools
│
├── scripts/                         # Training & experiment scripts
│   ├── base_training.py            # Core training loop with interleaved epoch training
│   ├── compute_normalization_stats.py # Builds MHD/Stokes normalization JSONs
│   └── experiments/
│       ├── ablation_study.py       # Physics regularization ablation study (5 configurations)
│
├── scripts/analysis/                # Post-training diagnostics
│   ├── muram_analysis.py           # Diagnostic maps on MURaM steps
│   └── modest_analysis.py          # Diagnostic comparison on MODEST data
│
├── tools/                           # Run wrappers (HPC/local)
│   ├── run_experiments.sh          # SLURM-ready ablation launcher
│   ├── compute_normalization_stats.sh # SLURM-ready normalization launcher
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
├── data/                            # Data directory (not in repo)
│   ├── muram-simulation/            # MURaM snapshots (T, V, B, Stokes)
│   ├── normalization_stats/         # Pre-computed normalization statistics
│   ├── csv/                         # Opacity tables (kappa.0.dat), etc.
│   └── hinode-MODEST/               # PSF and spectral response files
│
└── output/                          # Training outputs
    ├── experiments/                 # Ablation study results
    │   ├── <experiment_name>/
    │   ├── all_physics_terms/
    │   ├── wfa_only/
    │   ├── doppler_only/
    │   ├── black_body_only/
    │   └── no_physics/
  └── region_analysis/             # Saved regional diagnostics (.npy)
```

---

## 🔑 Key Components

### 1. **Models** (`models/`)

#### `mscnn_model.py`
- **MSCNNInversionModel**: Base multi-scale convolutional neural network
- Processes Stokes I and V profiles at multiple scales (1×, 2×, 3×)
- Architecture: Conv blocks → Multi-scale fusion → Dense layers → Atmospheric stratification output (63 values: 21 per T, V_LOS, B_LOS)

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

#### `grad_norm.py`
- **GradNormScheduler**: Automatic multi-task loss balancing
- Dynamically adjusts weights for MSE, WFA, Doppler, and Temperature losses
- Based on: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing", ICML 2018
- Includes gradient norm computation per task

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
  - `MetricsLogger`: CSV logging for loss components and GradNorm metrics
- Supports:
  - Interleaved training across simulation steps
  - GradNorm automatic loss balancing
  - Checkpoint saving/resuming
  - Learning rate scheduling (ReduceLROnPlateau, CosineAnnealing)

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

1. **Prepare data**:
   ```bash
   # Ensure MURaM data is in /data/muram-simulation/
   # Compute normalization statistics (see notebooks)
   ```

2. **Train baseline model** (no physics):
   ```bash
   cd scripts/experiments
   python ablation_study.py \
       --n_epochs 30 \
       --learning_rate 1e-3 \
       --lambda_wfa 0.0 \
       --lambda_doppler 0.0 \
       --lambda_temp 0.0 \
       --min_step 150 \
       --max_step 155
   ```

3. **Train with physics regularization**:
   ```bash
   python ablation_study.py \
       --n_epochs 30 \
       --learning_rate 1e-3 \
       --lambda_wfa 0.01 \
       --lambda_doppler 0.01 \
       --lambda_temp 0.01 \
       --min_step 150 \
       --max_step 155
   ```

4. **Submit to HPC**:
   ```bash
  # Edit tools/run_experiments.sh to set hyperparameters
  sbatch tools/run_experiments.sh
   ```

---

## 🔄 Training Pipeline

### Training Flow

```
1. Load MURaM step
   ↓
2. Compute optical depth (log τ)
   ↓
3. Remap T, V, B to optical depth grid
   ↓
4. Generate synthetic Stokes profiles
   ↓
5. Apply LSF convolution & wavelength resampling
   ↓
6. Compute physics approximations (WFA, Doppler, Temperature)
   ↓
7. Normalize inputs/outputs
   ↓
8. Create mini-batches (spatial pixels)
   ↓
9. Forward pass through PINN
   ↓
10. Compute losses:
    - MSE loss (supervised)
    - Physics losses (WFA, Doppler, Temperature)
    ↓
11. Backward pass & optimize
    ↓
12. (Optional) Update GradNorm weights
    ↓
13. Repeat for next batch/step/epoch
```

### Loss Function

```
Total Loss = MSE_loss + λ_WFA × WFA_loss + λ_Doppler × Doppler_loss + λ_Temp × Temp_loss
```

**With GradNorm**:
- Weights (λ) are learned dynamically
- Balances gradient magnitudes across tasks

**Without GradNorm** (naive approach):
- Fixed λ values (e.g., 0.01 for each term)

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
  - GradNorm automatic loss balancing mechanics
  - Training loop with physics regularization
  - Loss component analysis and visualization
  - Gradient norm tracking across tasks
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

# GradNorm (optional)
use_gradnorm = False           # Enable automatic balancing
gradnorm_alpha = 1.5           # Restoring force parameter

# Physics Modes
blos_physics_mode = 'tau_averaged'      # or 'single_height'
vlos_physics_mode = 'single_height'     # Target log(τ) = -1.0
temp_physics_mode = 'single_height'     # Target log(τ) = 0.0 (photosphere)
```

### Data Configuration

```python
# MURaM simulation
min_step = 60                  # First simulation step
max_step = 200                 # Last simulation step
nx, ny = 480, 480              # Spatial dimensions
nz = 256                       # Vertical layers
z_max = 250                    # Limit vertical extent

# Optical depth grid
logtau_values = np.arange(-2.0, 0.1, 0.1)  # 21 levels
```

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

### GradNorm Comparison

```bash
# Naive approach (fixed lambdas)
python ablation_study.py --lambda_wfa 0.01 --lambda_doppler 0.01 --lambda_temp 0.01

# GradNorm (automatic balancing)
python ablation_study.py --use_gradnorm --gradnorm_alpha 1.5
```

---

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
│   │   └── gradnorm_log.csv      # GradNorm weights (if enabled)
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
- **Test metrics**:
  - B_LOS RRMSE (tau-averaged)
  - V_LOS RRMSE (tau-averaged)
  - Correlation coefficients
  - RMSE values

---

## 🤝 Contributing

To add new physics terms or modify the architecture:

1. **New physics term**:
   - Add approximation method in `utils/physics_utils.py`
   - Add loss computation in `models/pinn_mscnn_model.py`
   - Update `TrainingConfig` in `scripts/base_training.py`

2. **Architecture changes**:
   - Modify `models/mscnn_model.py` or `models/pinn_mscnn_model.py`
   - Ensure output dimension remains (batch_size, 63)

3. **New experiments**:
   - Create script in `scripts/experiments/`
   - Use `TrainingConfig` for consistency

---

## 📚 Key References

1. **GradNorm**: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing", ICML 2018
2. **Weak-Field Approximation**: Landi Degl'Innocenti & Landolfi, "Polarization in Spectral Lines", 2004
3. **MURaM**: Vögler et al., "The CO5BOLD/MURaM Code", 2005

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