# Magnetic-Field Balancing and Fine-Tuning Guide

This document explains the Bz balancing change added to the training pipeline, where it is implemented, and how to use it for a fine-tuning pass.

## Why this change exists

The magnetic-field target distribution is strongly skewed. The strongest positive and negative Bz values appear much less often than the central bins. A Huber-style tail loss helps, but it does not fix the sampling problem by itself.

If the model sees very few extreme-field pixels, then:

- the gradients from those regimes stay sparse,
- the optimizer still spends most of its time on the dense mid-range bins,
- and the tail loss becomes a weight on a small number of examples instead of a true balancing mechanism.

The new balancing path therefore does two things:

1. Keeps the current supervised + physics losses intact.
2. Filters pixels so the training set is approximately balanced across Bz-strength bins.

The implementation is conservative: it down-samples dense bins instead of inventing synthetic samples.

## What was added

### 1. Bz balancing utility

File: [utils/muram_data.py](../utils/muram_data.py#L1235)

New function:

- `build_bz_strength_balanced_indices()`

Algorithm:

1. Read the full 3D Bz cube from MHD data.
2. Compute one scalar score per spatial pixel.
3. Split those scores into uniform-width bins.
4. Count how many pixels fall into each bin.
5. Find the smallest occupied bin count.
6. Randomly downsample every other occupied bin to that same count.
7. Shuffle the final selection.

Supported score modes:

- `mean_abs`: mean of `|Bz|` over all optical-depth levels.
- `max_abs`: maximum `|Bz|` across optical depth.
- `tau_index`: `|Bz|` at one chosen optical-depth index.

For the current workflow, global balancing uses `tau_index` with the deepest optical-depth index as reference unless you override `bz_balance_tau_idx`.

### 2. Dataset metadata for traceability

File: [utils/muram_data.py](../utils/muram_data.py#L1356)

`MuramStepDataset.__init__()` now accepts `bz_balance_info` and stores it on the dataset instance.

This does not alter the tensor shapes or the training contract. It just keeps the balancing metadata attached to the dataset for logging and debugging.

### 3. Training config controls

File: [scripts/base_training.py](../scripts/base_training.py#L55)

New `TrainingConfig` fields:

- `apply_bz_bin_balance`
- `log_bz_bin_balance_stats`
- `bz_balance_mode`
- `bz_balance_bins`
- `bz_balance_tau_idx`

Validation is handled in `TrainingConfig.__post_init__()` at [scripts/base_training.py](../scripts/base_training.py#L182).

### 4. Step loading applies the filter

File: [scripts/base_training.py](../scripts/base_training.py#L467)

`load_and_prepare_step()` now supports two Bz balancing paths after the existing region mask:

- `per_step`: compute bins and selection inside each step independently.
- `global`: reuse precomputed step-wise selected indices produced from a global scan over all train steps.

The order is:

1. Load step data or cache.
2. Optionally apply the 4-region mask.
3. Optionally apply Bz bin balancing on the remaining pixel set (per-step or global precomputed).
4. Build `MuramStepDataset` with the selected indices.

The relevant hook points are:

- Region mask construction: [scripts/base_training.py](../scripts/base_training.py#L554)
- Region-balanced index selection: [scripts/base_training.py](../scripts/base_training.py#L558)
- Bz-balanced index selection: [scripts/base_training.py](../scripts/base_training.py#L569)
- Dataset creation with both metadata blocks: [scripts/base_training.py](../scripts/base_training.py#L583) and [scripts/base_training.py](../scripts/base_training.py#L584)

### 5. Training loop remains unchanged

File: [scripts/base_training.py](../scripts/base_training.py#L1035)

`train_epoch()` still delegates sampling to `load_and_prepare_step()`, but now accepts optional global precomputed balancing indices/metadata and forwards them to the loader.

That is the right design choice because the sampler is a data policy, not a model-policy.

### 6. CLI flags were added

File: [scripts/experiments/ablation_study.py](../scripts/experiments/ablation_study.py#L1468)

New flags:

- `--apply-bz-bin-balance` / `--no-bz-bin-balance`
- `--bz-balance-mode`
- `--bz-balance-bins`
- `--bz-balance-scope`
- `--bz-balance-seed`
- `--bz-balance-tau-idx`

They are printed in the experiment summary at [scripts/experiments/ablation_study.py](../scripts/experiments/ablation_study.py#L1637) and passed into `TrainingConfig(...)` at [scripts/experiments/ablation_study.py](../scripts/experiments/ablation_study.py#L1688).

## Global deepest-tau balancing workflow

Implemented in `compute_global_bz_balancing_indices()` in [scripts/base_training.py](../scripts/base_training.py).

Workflow:

1. Scan all train steps (post-preprocessing, post-normalization, post-region-mask if enabled).
2. Extract `|Bz|` at the deepest optical-depth node (or user-selected `bz_balance_tau_idx`).
3. Build one global histogram and bin edges across all train steps.
4. Downsample every occupied bin to the smallest occupied bin count globally.
5. Persist per-step selected indices and reuse them in each epoch.

This ensures balancing is computed globally across train data while preserving step-wise loading and caching.

## Histogram guarantee (ready-for-training)

`generate_training_data_histograms()` now uses `load_and_prepare_step()` with the same balancing path used by training, including global precomputed indices when enabled.

So histograms in `output/experiments` represent the actual tensors fed to the model:

- after preprocessing,
- after normalization,
- after region masking,
- after Bz balancing selection.

## How the algorithm works

### Bz balancing algorithm

Implemented in `build_bz_strength_balanced_indices()`.

The function converts the Bz cube into a single score per pixel:

- `mean_abs`: average of `abs(Bz)` across tau
- `max_abs`: maximum of `abs(Bz)` across tau
- `tau_index`: absolute Bz at a fixed optical depth index

After that, it builds uniform-width bins over the score range, counts samples per bin, and downsamples every bin to match the smallest occupied bin.

This is a filtering strategy, not oversampling.

### Why the Huber term is not enough by itself

The current Huber tail loss in [models/pinn_mscnn_model.py](../models/pinn_mscnn_model.py) already increases the penalty for large Bz errors. That is useful, but it still acts only on the samples the model sees.

If extreme Bz pixels are rare:

- the gradients from those bins remain sparse,
- the model still mostly optimizes the dense center of the distribution,
- and the tail-loss weight is attached to very few examples.

So the better strategy is:

1. use the tail loss to make large-field mistakes more expensive,
2. use pixel filtering to change the sample composition,
3. validate on the original unfiltered distribution so evaluation stays realistic.

## How fine-tuning is implemented here

Fine-tuning means continuing training from an already-trained checkpoint, but with a different training emphasis.

In this repository, the practical meaning is:

1. Train a base model with the normal sampling policy.
2. Resume from a checkpoint.
3. Enable Bz balancing for the second phase.
4. Usually reduce the learning rate for the second phase.

Important: fine-tuning is not a different architecture. It is the same model, trained again from previous weights with updated sampling and/or loss weighting.

### Recommended workflow

Phase 1: base training

- keep `apply_bz_bin_balance=False`
- use the existing loss terms
- train until validation stabilizes

Phase 2: fine-tune on balanced Bz bins

- resume from the best checkpoint using the existing resume mechanism
- set `apply_bz_bin_balance=True`
- lower the learning rate
- keep evaluation on the original validation setup

### What changes during fine-tuning

Changes:

- training pixel distribution becomes flatter across Bz-strength bins,
- extreme-field examples contribute more often,
- the model gets more direct exposure to tail regimes.

What should stay the same:

- output shape,
- physics context,
- normalization statistics,
- validation distribution.

## Where the logic lives

### `utils/muram_data.py`

- `build_balanced_region_indices()` at [utils/muram_data.py](../utils/muram_data.py#L1194)
- `build_bz_strength_balanced_indices()` at [utils/muram_data.py](../utils/muram_data.py#L1235)
- `MuramStepDataset.__init__()` at [utils/muram_data.py](../utils/muram_data.py#L1356)

### `scripts/base_training.py`

- `TrainingConfig` at [scripts/base_training.py](../scripts/base_training.py#L55)
- `TrainingConfig.__post_init__()` at [scripts/base_training.py](../scripts/base_training.py#L182)
- `load_and_prepare_step()` at [scripts/base_training.py](../scripts/base_training.py#L467)
- `train_epoch()` at [scripts/base_training.py](../scripts/base_training.py#L1035)

### `scripts/experiments/ablation_study.py`

- CLI argument parsing for the new flags at [scripts/experiments/ablation_study.py](../scripts/experiments/ablation_study.py#L1468)
- `TrainingConfig(...)` construction in `_build_cfg()` at [scripts/experiments/ablation_study.py](../scripts/experiments/ablation_study.py#L1688)

## Practical recommendation

Use Bz balancing only for the fine-tuning phase unless you have already confirmed that it does not hurt performance on the central bins.

If you want a conservative setup, try:

1. Base run with the current Huber tail loss only.
2. Short fine-tune run with `--apply-bz-bin-balance`.
3. Compare validation metrics and tail-bin diagnostics.

If the extreme-field metrics improve without a major loss in the mid-range bins, keep the balancing step.
