"""Fast diagnostic run for rolling-difference patch transformer features."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

import run_charge_discharge_reltime_patch_transformer_xai as patch_run


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = ROOT / "charge_discharge_rebuilt_same_features_rolling_diff_fast6"
CACHE_SOURCE = ROOT / "charge_discharge_rebuilt_same_features_rolling_diff"

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
    return add_previous_patch_differences(_base_build_event_patch_sequence(event_type, df))


def reuse_sequence_cache() -> None:
    """Reuse previously built rolling-diff sequences if available."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name in ("reltime_patch_event_meta.csv", "reltime_patch_sequences.pkl"):
        source = CACHE_SOURCE / name
        target = OUTDIR / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)


def main() -> None:
    """Run a shorter rolling-difference transformer diagnostic."""
    reuse_sequence_cache()
    patch_run.OUTDIR = OUTDIR
    patch_run.PATCH_FEATURE_NAMES = ROLLING_FEATURE_NAMES
    patch_run.build_event_patch_sequence = build_event_patch_sequence_with_diffs
    patch_run.SPEC = patch_run.Spec(
        name="reltime_patch30_transformer_rolling_diff_fast6",
        d_model=96,
        nhead=4,
        num_layers=3,
        dim_feedforward=192,
        dropout=0.15,
        learning_rate=5.0e-4,
        weight_decay=5.0e-4,
        epochs=6,
        batch_size=8192,
    )
    patch_run.set_seed()
    torch.set_num_threads(8)
    _, metrics, _, _ = patch_run.train_model()
    metrics["feature_engineering_addition"] = {
        "rolling_difference_definition": "dprev_feature[k] = feature[k] - feature[k-1] within the same event; first token = 0",
        "base_feature_count": len(BASE_FEATURE_NAMES),
        "rolling_difference_feature_count": len(ROLLING_DIFF_FEATURE_NAMES),
        "total_patch_feature_count": len(ROLLING_FEATURE_NAMES),
        "diagnostic_run": True,
    }
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Outputs: {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
