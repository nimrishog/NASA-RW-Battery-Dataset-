"""Rebuild the charge+discharge patch transformer with rolling differences.

The feature set is the approved signal-derived patch feature set plus one
``dprev_*`` channel per feature. Each ``dprev`` channel is the difference
between the current patch token and the previous patch token inside the same
event. The first token receives zero differences.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import run_charge_discharge_reltime_patch_transformer_xai as patch_run


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = ROOT / "charge_discharge_rebuilt_same_features_rolling_diff"

BASE_FEATURE_NAMES = list(patch_run.PATCH_FEATURE_NAMES)
ROLLING_DIFF_FEATURE_NAMES = [f"dprev_{name}" for name in BASE_FEATURE_NAMES]
ROLLING_FEATURE_NAMES = BASE_FEATURE_NAMES + ROLLING_DIFF_FEATURE_NAMES

_base_build_event_patch_sequence = patch_run.build_event_patch_sequence


def add_previous_patch_differences(sequence: np.ndarray) -> np.ndarray:
    """Append first-order patch-to-patch differences to a token sequence."""
    diffs = np.zeros_like(sequence, dtype=np.float32)
    if len(sequence) > 1:
        diffs[1:] = sequence[1:] - sequence[:-1]
    return np.concatenate([sequence.astype(np.float32), diffs], axis=1).astype(np.float32)


def build_event_patch_sequence_with_diffs(event_type: str, df) -> np.ndarray:
    """Build approved patch features and append rolling differences."""
    base_sequence = _base_build_event_patch_sequence(event_type, df)
    return add_previous_patch_differences(base_sequence)


def main() -> None:
    """Train the patch transformer with approved features plus rolling differences."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    patch_run.OUTDIR = OUTDIR
    patch_run.PATCH_FEATURE_NAMES = ROLLING_FEATURE_NAMES
    patch_run.build_event_patch_sequence = build_event_patch_sequence_with_diffs
    patch_run.set_seed()
    torch.set_num_threads(8)
    _, metrics, _, _ = patch_run.train_model()
    metrics["feature_engineering_addition"] = {
        "rolling_difference_definition": "dprev_feature[k] = feature[k] - feature[k-1] within the same event; first token = 0",
        "base_feature_count": len(BASE_FEATURE_NAMES),
        "rolling_difference_feature_count": len(ROLLING_DIFF_FEATURE_NAMES),
        "total_patch_feature_count": len(ROLLING_FEATURE_NAMES),
    }
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Outputs: {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
