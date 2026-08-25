#!/usr/bin/env python3
"""Helpers for logging MUISCA runs to MLflow.

Deliberately small: the scripts call the MLflow API directly (`mlflow.start_run`,
`mlflow.log_metrics`, ...), and this module only covers the two things MLflow does not do
for us.

Backend: SQLite (see default_tracking_uri). MLflow 3.x deprecated the filesystem store, and
SQLite additionally enables the Model Registry. This is safe while the ablation variations
run sequentially in one job; parallelizing them means moving to a tracking server.

Note on what MLflow is and is not doing here: `output/<experiment>/<variation>/` remains the
functional source of truth. `finetune.py` resolves checkpoints by path and reads the
`experiment_config.json` beside them, and the analysis pipeline walks the same layout.
Anything logged to MLflow is an additional copy for browsing and comparison.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict

DEFAULT_TRACKING_DB = Path("/scratchsan/observatorio/juagudeloo/MUISCA/mlruns.db")

# MLflow rejects parameter values longer than this. Our logtau grid is 45 numbers, which
# blows past it -- and a single oversized value makes the whole log_params call fail, so it
# is summarized rather than left to explode.
_MAX_PARAM_CHARS = 480


def default_tracking_uri() -> str:
    """SQLite tracking URI for the shared run database.

    MLflow 3.x puts the filesystem store in maintenance mode and refuses it outright unless
    MLFLOW_ALLOW_FILE_STORE=true, so SQLite is the supported backend. It also enables the
    Model Registry (named models, versions, stages), which a file store cannot provide.

    IMPORTANT if the ablation is ever parallelized: SQLite over NFS does not tolerate
    concurrent writers. Today the variations run sequentially inside a single SLURM job, so
    there is exactly one writer and this is safe. Turning the variations into a job array
    would put N jobs on the same database file -- at that point run a tracking server
    (`mlflow server --backend-store-uri sqlite:///...`) and point MLFLOW_TRACKING_URI at it,
    rather than letting the jobs share the file.

    Honours MLFLOW_TRACKING_URI when set, so this can be redirected without editing code.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    DEFAULT_TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_TRACKING_DB}"


def _summarize_sequence(value) -> str:
    """Compact description of a sequence too long to log verbatim.

    Reports the spacing as well as the endpoints, so an evenly-spaced grid stays fully
    reconstructible from the parameter: the 45-level logtau grid becomes
    "[45 values: -3.0 ... 1.4, step 0.1]" rather than losing how it was sampled. Falls back
    to endpoints alone when the spacing is not uniform (or the entries are not numeric),
    since quoting a single step would then be misleading.
    """
    if not value:
        return "[]"
    head, tail, n = value[0], value[-1], len(value)
    try:
        nums = [float(v) for v in value]
    except (TypeError, ValueError):
        return f"[{n} values: {head} ... {tail}]"

    if n > 2:
        diffs = [b - a for a, b in zip(nums[:-1], nums[1:])]
        spread = max(diffs) - min(diffs)
        scale = max(abs(d) for d in diffs) or 1.0
        # 1e-4 relative, not something tighter: the logtau grid is built in float32, so its
        # steps disagree by ~2.4e-6 relative purely from rounding. A stricter test rejects a
        # grid that is uniform by construction and silently drops the step from the summary.
        if spread <= 1e-4 * scale:
            step = sum(diffs) / len(diffs)
            return f"[{n} values: {head:g} ... {tail:g}, step {step:g}]"
    return f"[{n} values: {head} ... {tail}]"


def flatten_params(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested config dict into MLflow's flat parameter namespace.

    `experiment_config.json` is grouped ({'data_config': {'min_step': 110}, ...}) but
    `mlflow.log_params` only takes flat key/value pairs, so nesting becomes dotted keys
    ('data_config.min_step'). Long sequences are summarized to stay under MLflow's length
    limit while still recording what was used.
    """
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_params(value, prefix=name))
        elif isinstance(value, (list, tuple)):
            text = ", ".join(str(v) for v in value)
            if len(text) > _MAX_PARAM_CHARS:
                text = _summarize_sequence(value)
            flat[name] = text
        else:
            flat[name] = value
    return flat


def finite_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Keep only finite numeric metrics.

    Physics losses report NaN while the WFA gate is still closed (compute_physics_loss
    returns early without populating the components). MLflow will happily store those, but a
    NaN in a metric series makes the UI's charts unreadable, so they are dropped and the
    series simply starts when the gate opens.
    """
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isfinite(value):
            out[key] = value
    return out
