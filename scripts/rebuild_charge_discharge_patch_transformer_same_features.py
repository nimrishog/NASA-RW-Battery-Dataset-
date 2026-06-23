"""Rebuild the approved charge+discharge patch transformer without XAI.

This wrapper reuses the feature builder and model code from
``run_charge_discharge_reltime_patch_transformer_xai.py`` but writes to a
separate folder and skips SHAP/LIME so a clean model rebuild finishes quickly.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import run_charge_discharge_reltime_patch_transformer_xai as patch_run


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = ROOT / "charge_discharge_rebuilt_same_features"


def main() -> None:
    """Train the patch transformer with the approved feature set."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    patch_run.OUTDIR = OUTDIR
    patch_run.set_seed()
    torch.set_num_threads(8)
    _, metrics, _, _ = patch_run.train_model()
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Outputs: {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
