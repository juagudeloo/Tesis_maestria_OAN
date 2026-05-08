# How to Fine-Tune MUISCA Models

This guide explains how to fine-tune trained MUISCA checkpoint models with mandatory Bz (magnetic field) balancing.

## Overview

Fine-tuning is a two-phase training workflow:

1. **Phase 1 (Base Training)**: `scripts/experiments/ablation_study.py` trains models on the full unsorted dataset (`apply_bz_bin_balance=False`)
2. **Phase 2 (Fine-Tuning)**: `scripts/finetune.py` resumes training from a checkpoint with mandatory Bz balancing enabled (`apply_bz_bin_balance=True`)

### Why Fine-Tune with Bz Balancing?

The Bz (magnetic field) values in MURaM data follow a highly skewed distribution:
- Most pixels have weak fields (center of distribution)
- Extreme fields (positive and negative tails) are rare

**Without balancing**: Model sees few extreme-field examples → sparse gradients from tail regimes → optimization dominated by central bins

**With Bz balancing**: Pixel composition is rebalanced across Bz-strength bins → extreme-field examples appear more often → model learns tail regimes better

For detailed theory, see [docs/magnetic_field_balancing_and_finetuning.md](magnetic_field_balancing_and_finetuning.md).

## Quick Start

### 1. Identify Your Trained Model

List available experiments:
```bash
ls output/experiments/
```

Each experiment has this structure:
```
output/experiments/
  experiment_81_to_181-...        # Experiment directory (unique name)
    all_physics_terms/            # Variation 1
      final_model.pth             # ← The checkpoint we'll fine-tune
      logs/
        metrics_log.csv
    wfa_only/                     # Variation 2
      final_model.pth
      logs/
    doppler_only/                 # Variation 3
    no_physics/                   # Variation 4
```

### 2. Launch Fine-Tuning

#### Option A: Direct Python Invocation (Recommended for Testing)

Fine-tune a single variation:
```bash
python scripts/finetune.py \
    --experiment-name experiment_81_to_181-step_size_5-no_stokes_mult_factor \
    --variations wfa_only
```

Fine-tune multiple variations in sequence:
```bash
python scripts/finetune.py \
    --experiment-name experiment_81_to_181-step_size_5-no_stokes_mult_factor \
    --variations wfa_only,all_physics_terms,doppler_only
```

#### Option B: SLURM Submission on Maxwell Cluster (Recommended for Production)

Submit a fine-tuning job:
```bash
sbatch tools/fine_tune.sh \
    --experiment-name experiment_81_to_181-step_size_5-no_stokes_mult_factor \
    --variations wfa_only
```

Submit multiple variations:
```bash
sbatch tools/fine_tune.sh \
    --experiment-name experiment_81_to_181-step_size_5-no_stokes_mult_factor \
    --variations wfa_only,all_physics_terms,doppler_only
```

Override resource allocation:
```bash
sbatch --time=06:00:00 --mem=16G tools/fine_tune.sh \
    --experiment-name experiment_81_to_181-step_size_5-no_stokes_mult_factor \
    --variations all_physics_terms \
    --finetune-epochs 20
```

Monitor job status:
```bash
squeue -u $(whoami)                    # List your jobs
tail -f output/fine-tune/finetune_*.out  # Stream job output
```

### 3. Inspect Results

Fine-tuning outputs mirror the `output/experiments/` structure:

```
output/fine-tune/
  experiment_81_to_181-...-finetuned/           # Matches base experiment name
    wfa_only/                                   # Matches variation name
      final_model.pth                           # ← Fine-tuned checkpoint
      logs/
        metrics_log.csv                         # Per-epoch loss/metrics
      checkpoints/                              # (intermediate saves, if enabled)
    all_physics_terms/
    doppler_only/
```

Compare base vs. fine-tuned metrics:
```bash
# Base training (last 5 epochs)
tail -5 output/experiments/experiment_81_to_181-...-no_stokes_mult_factor/wfa_only/logs/metrics_log.csv

# Fine-tuned (all epochs)
tail -10 output/fine-tune/experiment_81_to_181-...-finetuned/wfa_only/logs/metrics_log.csv
```

## Command Reference

### scripts/finetune.py

```
usage: python scripts/finetune.py [OPTIONS]

Mandatory Arguments:
  --experiment-name NAME              Experiment name from output/experiments/
  --variations VAR1,VAR2,...          Comma-separated variation names to fine-tune

Optional Arguments:
  --finetune-epochs N                 Number of fine-tuning epochs
                                      (default: 10% of original, minimum 5)
  --output-base-dir PATH              Output directory (default: output/fine-tune/)
  --exp-base-dir PATH                 Checkpoint directory (default: output/experiments/)
```

Examples:
```bash
# Fine-tune one variation, default epochs (10% of original)
python scripts/finetune.py --experiment-name exp_81_to_181 --variations wfa_only

# Fine-tune three variations, explicit epoch count
python scripts/finetune.py \
    --experiment-name exp_81_to_181 \
    --variations wfa_only,doppler_only,all_physics_terms \
    --finetune-epochs 50

# Fine-tune with custom output directory
python scripts/finetune.py \
    --experiment-name exp_81_to_181 \
    --variations all_physics_terms \
    --output-base-dir /custom/output/path
```

### tools/fine_tune.sh (SLURM)

```
usage: sbatch [SBATCH_FLAGS] tools/fine_tune.sh [SCRIPT_ARGS]

SLURM Defaults (override with sbatch --flag):
  --job-name=finetune
  --partition=gpu.cecc
  --gres=gpu:1
  --cpus-per-task=4
  --mem=32G
  --time=12:00:00

Script Arguments (same as finetune.py):
  --experiment-name NAME            Experiment name (required)
  --variations VAR1,VAR2,...        Variations (required)
  --finetune-epochs N               Fine-tune epochs (optional)
```

Examples:
```bash
# Submit with defaults (12 hours, 32GB, 1 GPU)
sbatch tools/fine_tune.sh --experiment-name exp_81_to_181 --variations wfa_only

# Fast run: 6-hour wall time, 16GB memory
sbatch --time=06:00:00 --mem=16G tools/fine_tune.sh \
    --experiment-name exp_81_to_181 \
    --variations wfa_only,doppler_only

# Longer training: 24 hours, 2 GPUs (if supported)
sbatch --time=24:00:00 --gres=gpu:2 tools/fine_tune.sh \
    --experiment-name exp_81_to_181 \
    --variations all_physics_terms
```

## Configuration Details

### Bz Balancing Settings (Hardcoded in finetune.py)

Fine-tuning enforces these Bz-balancing parameters (cannot be overridden):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `apply_bz_bin_balance` | `True` | **Mandatory** fine-tuning feature |
| `bz_balance_scope` | `global` | Balance across all training steps |
| `bz_balance_mode` | `tau_index` | Use absolute Bz at fixed optical depth |
| `bz_balance_bins` | `12` | Divide Bz range into 12 equal bins |
| `bz_balance_seed` | `42` | Deterministic bin assignment |
| `bz_balance_logtau` | (deepest) | Optical-depth level for Bz scoring |

### Fine-Tuning Epochs

- **Default calculation**: `max(5, int(0.1 × original_epochs))`
  - If base training ran 100 epochs → default fine-tuning: 10 epochs
  - If base training ran 30 epochs → default fine-tuning: 5 epochs (minimum)
- **Override**: Use `--finetune-epochs N` to specify custom epoch count
- **Typical range**: 5–50 epochs (usually 10-20% of base training)

### Physics Regularization Weights

Physics loss weights (λ values for WFA, Doppler, Temperature) are **inherited** from the trained checkpoint. They are NOT modified during fine-tuning.

**To vary physics weights during fine-tuning:**
- Currently requires manual checkpoint editing (future enhancement)
- See "Advanced: Checkpoint Inspection" section below

### Learning Rate and Optimizer State

- **Learning rate**: Inherited from base training checkpoint config
- **Optimizer state**: Optionally restored from checkpoint (Adam momentum, etc.)
- **Practical note**: Fresh optimizer state often works well for fine-tuning; checkpoint state can help if retraining on similar data

## Workflow Example

### Scenario: Fine-Tune "all_physics_terms" Model After Base Training

**Step 1: Verify checkpoint exists**
```bash
ls -lh output/experiments/experiment_81_to_181-step_size_5-no_stokes_mult_factor/all_physics_terms/final_model.pth
# Should show a file ~20–100 MB
```

**Step 2: Review base training history**
```bash
head -1 output/experiments/experiment_81_to_181-step_size_5-no_stokes_mult_factor/all_physics_terms/logs/metrics_log.csv
tail -5 output/experiments/experiment_81_to_181-step_size_5-no_stokes_mult_factor/all_physics_terms/logs/metrics_log.csv
# Shows headers and final epoch stats
```

**Step 3: Launch fine-tuning via SLURM**
```bash
sbatch tools/fine_tune.sh \
    --experiment-name experiment_81_to_181-step_size_5-no_stokes_mult_factor \
    --variations all_physics_terms \
    --finetune-epochs 15
```

**Step 4: Monitor progress (in separate terminal)**
```bash
# Check job status
squeue -u $(whoami)

# Stream live output
tail -f output/fine-tune/finetune_*.out | grep -E "(Epoch|Loss|Test correlation|✓)"
```

**Step 5: Compare base vs. fine-tuned metrics**
```bash
echo "=== Base Training (final epoch) ==="
tail -1 output/experiments/experiment_81_to_181-step_size_5-no_stokes_mult_factor/all_physics_terms/logs/metrics_log.csv

echo -e "\n=== Fine-Tuned Training (final epoch) ==="
tail -1 output/fine-tune/experiment_81_to_181-step_size_5-no_stokes_mult_factor-finetuned/all_physics_terms/logs/metrics_log.csv
```

**Step 6: Run analysis on fine-tuned model**
```bash
python scripts/analysis/muram_analysis.py \
    --experiment-dir output/fine-tune/experiment_81_to_181-step_size_5-no_stokes_mult_factor-finetuned/all_physics_terms
```

## Advanced Topics

### Checkpoint Inspection

Inspect checkpoint structure and configuration:
```python
import torch

checkpoint = torch.load('output/experiments/experiment_81_to_181-.../final_model.pth')
print(checkpoint.keys())
# dict_keys(['model_state_dict', 'optimizer_state_dict', 'config'])

config = checkpoint['config']
print(f"Training epochs: {config['n_epochs']}")
print(f"Learning rate: {config['learning_rate']}")
print(f"Physics weights: λ_wfa={config['lambda_wfa']}, λ_doppler={config['lambda_doppler']}, λ_temp={config['lambda_temp']}")
print(f"Bz balancing (base training): {config['apply_bz_bin_balance']}")  # Should be False
```

### Reusing Fine-Tuned Models

Fine-tuned checkpoints can be used for:

1. **Analysis**: Feed to `scripts/analysis/muram_analysis.py` or `scripts/analysis/modest_analysis.py`
   ```bash
   python scripts/analysis/muram_analysis.py \
       --experiment-dir output/fine-tune/experiment-.../wfa_only
   ```

2. **Further refinement**: Load fine-tuned checkpoint as starting point for additional fine-tuning
   - Requires manual script modification (planned enhancement: `--checkpoint-path` flag)
   - Copy fine-tuned `final_model.pth` to new location, treat as checkpoint for next phase

3. **Deployment**: Use fine-tuned weights in production inference pipelines

### Cached Bz Balancing Data

Fine-tuning automatically reuses existing balanced cache from base training (if available):
- Located at `.muram_balanced_cache/` by default
- Speeds up training by pre-computing balanced pixel selections
- Automatically refreshed if training configuration changes

To force cache rebuild:
- Delete `.muram_balanced_cache/` directory
- Re-run fine-tuning (will recompute and cache balanced data)

### Modifying Physics Weights for Re-Fine-Tuning

If you want to fine-tune with different physics regularization weights:

1. Load checkpoint and modify config:
   ```python
   import torch
   checkpoint = torch.load('output/experiments/.../final_model.pth')
   config = checkpoint['config']
   config['lambda_wfa'] = 0.02  # New weight
   config['lambda_doppler'] = 0.001  # New weight
   torch.save(checkpoint, 'modified_checkpoint.pth')
   ```

2. Manually update finetune.py to load this checkpoint:
   - Edit line with `discover_checkpoint_path()` to use hardcoded path
   - Re-run fine-tuning script

(Future enhancement: add `--lambda-override` flags to finetune.py)

## Troubleshooting

### Issue: FileNotFoundError: "No checkpoint found"

**Cause**: Typo in experiment or variation name

**Solution**:
```bash
# List valid experiments
ls output/experiments/

# List valid variations for an experiment
ls output/experiments/experiment_81_to_181-step_size_5-no_stokes_mult_factor/

# Re-run with correct names
python scripts/finetune.py \
    --experiment-name experiment_81_to_181-step_size_5-no_stokes_mult_factor \
    --variations wfa_only
```

### Issue: KeyError: 'mhd_normalization.json' or 'stokes_normalization.json'

**Cause**: Normalizers not computed before fine-tuning

**Solution**:
```bash
python scripts/compute_normalization_stats.py --min_step 60 --max_step 200
```

Then retry fine-tuning.

### Issue: SLURM Job Fails: "GPU out of memory"

**Cause**: GPU insufficient for batch size or model size

**Solution** (in order of preference):
1. Increase wall time and reduce concurrent jobs
2. Override SLURM memory: `sbatch --mem=48G tools/fine_tune.sh ...`
3. Check model size: `scripts/finetune.py` reports parameter count at startup
4. If persistent, reduce batch size (requires code edit in finetune.py; future enhancement)

### Issue: Cache Directory Missing / "No such file"

**Cause**: `.muram_cache/` or `.muram_balanced_cache/` deleted during active training

**Solution**:
- Stop any running fine-tuning jobs
- Re-run compute_normalization_stats.py to rebuild raw cache
- Re-run fine-tuning (will rebuild balanced cache automatically)

### Issue: Test Metrics Not Computed

**Cause**: compute_tau_averaged_metrics() failed (rare)

**Symptom**: Output shows "Warning: Test metrics computation failed"

**Solution**:
- Usually transient; metrics still logged to CSV
- Safe to continue; fine-tuning proceeds without metrics display
- Check CSV directly: `tail output/fine-tune/.../logs/metrics_log.csv`

## Future Enhancements

- [ ] Support V and B (velocity/magnetic field) balancing (currently Bz-only)
- [ ] Add `--lambda-override` flags to vary physics weights per fine-tuning run
- [ ] Parallel fine-tuning of multiple variations (currently sequential)
- [ ] Checkpoint persistence interval during fine-tuning
- [ ] Diagnostic plotting (loss evolution, metric trends summarized in output/)
- [ ] Direct integration with MODEST real-data fine-tuning

## References

- **Base training workflow**: See [CLAUDE.md](../CLAUDE.md) for initial training commands
- **Physics background**: See [docs/magnetic_field_balancing_and_finetuning.md](magnetic_field_balancing_and_finetuning.md)
- **Model architecture**: See [models/pinn_mscnn_model.py](../models/pinn_mscnn_model.py)
- **Physics approximations**: See [utils/physics_utils.py](../utils/physics_utils.py)
- **Data pipeline**: See [utils/muram_data.py](../utils/muram_data.py)
