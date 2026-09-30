from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

import run_controlled_rnn as rnn
import run_transformer_loco as pipeline


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_validation" / "stage7_architecture_dq_dv"
TRANSFORMER_METRICS = ROOT / "results" / "reviewer_validation" / "stage2_paired_dq_dv_seeds" / "paired_seed_metrics.csv"
SEEDS = (42, 43, 44)
FULL_FEATURES = list(pipeline.FEATURES)
FEATURE_SETS = {
    "full": FULL_FEATURES,
    "dq_dv_removed": [name for name in FULL_FEATURES if name != "dq_dv_max_ahpv"],
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        for condition, features in FEATURE_SETS.items():
            run_dir = OUT / f"seed_{seed}" / f"rnn_{condition}"
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                pipeline.CFG = replace(pipeline.CFG, seed=seed)
                pipeline.FEATURES = features
                rnn.OUT = run_dir
                print(f"Training RNN seed={seed} condition={condition}", flush=True)
                rnn.main()
            result = json.loads(metrics_path.read_text(encoding="utf-8"))["row"]
            rows.append({"seed": seed, "architecture": "rnn", "condition": condition, **result})
            pd.DataFrame(rows).to_csv(OUT / "rnn_architecture_ablation_partial.csv", index=False)

    rnn_metrics = pd.DataFrame(rows)
    rnn_metrics.to_csv(OUT / "rnn_architecture_ablation.csv", index=False)
    transformer = pd.read_csv(TRANSFORMER_METRICS)
    transformer["architecture"] = "transformer"
    combined = pd.concat([transformer, rnn_metrics], ignore_index=True)
    combined.to_csv(OUT / "four_condition_metrics.csv", index=False)
    print(combined[["seed", "architecture", "condition", "event_r2", "event_mae", "event_rmse", "checkpoint_r2", "checkpoint_mae", "checkpoint_rmse"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
