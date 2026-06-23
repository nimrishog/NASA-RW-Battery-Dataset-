from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import run_cycle_level_weighted_transformer as base


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = ROOT / "charge_only_regularized_rw11_transformer"


def save_against_baseline() -> None:
    baseline_metrics = json.loads(
        (ROOT / "cycle_level_weighted_transformer_rw9_rw10_to_rw11" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    new_metrics = json.loads((OUTDIR / "metrics.json").read_text(encoding="utf-8"))
    comparison = pd.DataFrame(
        [
            {
                "model": "baseline_charge_only_h8_drop010",
                "test_r2": baseline_metrics["test_metrics_cycle_level"]["r2"],
                "test_mae": baseline_metrics["test_metrics_cycle_level"]["mae"],
                "test_rmse": baseline_metrics["test_metrics_cycle_level"]["rmse"],
                "checkpoint_r2": baseline_metrics["test_metrics_checkpoint_aggregated"]["r2"],
                "checkpoint_mae": baseline_metrics["test_metrics_checkpoint_aggregated"]["mae"],
                "checkpoint_rmse": baseline_metrics["test_metrics_checkpoint_aggregated"]["rmse"],
            },
            {
                "model": "regularized_charge_only_h4_drop020",
                "test_r2": new_metrics["test_metrics_cycle_level"]["r2"],
                "test_mae": new_metrics["test_metrics_cycle_level"]["mae"],
                "test_rmse": new_metrics["test_metrics_cycle_level"]["rmse"],
                "checkpoint_r2": new_metrics["test_metrics_checkpoint_aggregated"]["r2"],
                "checkpoint_mae": new_metrics["test_metrics_checkpoint_aggregated"]["mae"],
                "checkpoint_rmse": new_metrics["test_metrics_checkpoint_aggregated"]["rmse"],
            },
        ]
    )
    comparison.to_csv(OUTDIR / "comparison_against_charge_only_baseline.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = comparison["model"].tolist()
    x = range(len(labels))
    for ax, col, title, color in zip(
        axes,
        ["test_r2", "test_mae", "checkpoint_r2"],
        ["RW11 cycle R2", "RW11 cycle MAE", "RW11 checkpoint R2"],
        ["#2a9d8f", "#457b9d", "#f4a261"],
    ):
        ax.bar(x, comparison[col], color=color)
        ax.set_title(title)
        ax.set_xticks(list(x), labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "comparison_against_charge_only_baseline.png", dpi=180)
    plt.close(fig)


def main() -> None:
    base.OUTDIR = OUTDIR
    base.TRAIN_BATTERIES = ("RW9", "RW10")
    base.TEST_BATTERIES = ("RW11",)
    base.SPEC = base.WeightedSpec(
        name="charge_only_h4_d80_drop020_wd5e4",
        d_model=80,
        nhead=4,
        num_layers=3,
        dim_feedforward=160,
        dropout=0.20,
        learning_rate=3.0e-4,
        weight_decay=5.0e-4,
        epochs=10,
        batch_size=512,
        max_tokens=11,
    )
    base.main()
    save_against_baseline()


if __name__ == "__main__":
    main()
