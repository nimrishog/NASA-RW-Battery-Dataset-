from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import run_transformer_loco as pipeline


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_validation" / "dq_dv_loco_ablation"
COMPLETE_METRICS = ROOT / "results" / "reviewer_validation" / "full_transformer_loco_metrics.csv"
REMOVED_FEATURE = "dq_dv_max_ahpv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline.OUT = OUT / "folds"
    pipeline.FIGURES = OUT / "figures"
    pipeline.FEATURES = [name for name in pipeline.FEATURES if name != REMOVED_FEATURE]

    events = pipeline.load_events()
    rows: list[dict[str, object]] = []
    for battery in pipeline.BATTERIES:
        metrics_path = pipeline.OUT / f"test_{battery}" / "metrics.json"
        if metrics_path.exists():
            row = json.loads(metrics_path.read_text(encoding="utf-8"))["row"]
            print(f"Using completed dQ/dV-removed fold for held-out {battery}", flush=True)
        else:
            print(f"Training dQ/dV-removed Transformer with held-out {battery}", flush=True)
            row = pipeline.train_one_loco(events, battery)
        row["feature_configuration"] = "dq_dv_removed"
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "dq_dv_removed_loco_metrics_partial.csv", index=False)

    removed = pd.DataFrame(rows).sort_values("test_battery")
    removed.to_csv(OUT / "dq_dv_removed_loco_metrics.csv", index=False)

    complete = pd.read_csv(COMPLETE_METRICS)
    complete["feature_configuration"] = "complete"
    comparison = pd.concat([complete, removed], ignore_index=True)
    comparison.to_csv(OUT / "complete_vs_dq_dv_removed_loco_metrics.csv", index=False)

    summary = (
        comparison.groupby("feature_configuration")
        .agg(
            folds=("test_battery", "count"),
            event_r2_mean=("event_r2", "mean"),
            event_r2_sd=("event_r2", "std"),
            event_mae_mean=("event_mae", "mean"),
            event_mae_sd=("event_mae", "std"),
            event_rmse_mean=("event_rmse", "mean"),
            event_rmse_sd=("event_rmse", "std"),
            checkpoint_r2_mean=("checkpoint_r2", "mean"),
            checkpoint_r2_sd=("checkpoint_r2", "std"),
            checkpoint_mae_mean=("checkpoint_mae", "mean"),
            checkpoint_mae_sd=("checkpoint_mae", "std"),
            checkpoint_rmse_mean=("checkpoint_rmse", "mean"),
            checkpoint_rmse_sd=("checkpoint_rmse", "std"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "complete_vs_dq_dv_removed_loco_summary.csv", index=False)

    pivot = comparison.pivot(index="test_battery", columns="feature_configuration", values="checkpoint_rmse")
    ax = pivot[["complete", "dq_dv_removed"]].plot.bar(
        color=["#2f80ed", "#7aa6df"], edgecolor="#263238", figsize=(9.4, 5.4)
    )
    ax.set_xlabel("Held-out battery")
    ax.set_ylabel("Diagnostic-checkpoint RMSE (SOH percentage points)")
    ax.set_title("LOCO Transformer Ablation of Local dQ/dV", weight="bold")
    ax.legend(["Complete features", "dQ/dV removed"], frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "fig_dq_dv_loco_ablation_rmse.png", dpi=300)
    plt.close()

    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
