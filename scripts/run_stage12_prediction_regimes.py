from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import event_data


ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "results" / "reviewer_validation" / "stage2_paired_dq_dv_seeds"
OUT = ROOT / "results" / "reviewer_validation" / "stage12_prediction_regimes"
SEEDS = (42, 43, 44)


def metric_row(group: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(group),
        "mae": mean_absolute_error(group["y_true_soh"], group["y_pred_soh"]),
        "rmse": np.sqrt(mean_squared_error(group["y_true_soh"], group["y_pred_soh"])),
        "r2": r2_score(group["y_true_soh"], group["y_pred_soh"]) if group["y_true_soh"].nunique() > 1 else np.nan,
        "bias": (group["y_pred_soh"] - group["y_true_soh"]).mean(),
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = event_data.load_events()
    events = events[events["battery"].eq("RW11")].set_index("event_order_chronological")
    frames = []
    for seed in SEEDS:
        for condition in ("full", "dq_dv_removed"):
            path = STAGE2 / f"seed_{seed}" / condition / "test_RW11" / "test_predictions.csv"
            frame = pd.read_csv(path)
            frame["seed"] = seed
            frame["condition"] = condition
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    feature_columns = ["event_type", "duration_s", "current_abs_mean_a", "voltage_pdf_std", "voltage_pdf_q90"]
    data = data.join(events[feature_columns], on="event_order_chronological")
    data["soh_stage"] = pd.cut(data["y_true_soh"], bins=[-np.inf, 60, 80, np.inf], labels=["late_<60", "middle_60_80", "early_>80"], right=False)
    base = data.drop_duplicates(["event_order_chronological"])
    for column, output in (("duration_s", "duration_regime"), ("current_abs_mean_a", "current_regime"), ("voltage_pdf_std", "voltage_spread_regime"), ("voltage_pdf_q90", "voltage_level_regime")):
        edges = np.unique(np.quantile(base[column], [0, 1 / 3, 2 / 3, 1]))
        if len(edges) == 4:
            data[output] = pd.cut(data[column], bins=edges, labels=["low", "medium", "high"], include_lowest=True)

    outputs = []
    regime_columns = ["soh_stage", "event_type", "duration_regime", "current_regime", "voltage_spread_regime", "voltage_level_regime"]
    for regime_type in regime_columns:
        grouped = data.groupby(["seed", "condition", regime_type], observed=True).apply(metric_row, include_groups=False).reset_index()
        grouped = grouped.rename(columns={regime_type: "regime"})
        grouped.insert(0, "regime_type", regime_type)
        outputs.append(grouped)
    metrics = pd.concat(outputs, ignore_index=True)
    metrics.to_csv(OUT / "prediction_metrics_by_regime_and_seed.csv", index=False)
    wide = metrics.pivot(index=["regime_type", "regime", "seed"], columns="condition", values=["mae", "rmse", "bias"]).reset_index()
    wide.columns = [column[0] if not column[1] else f"{column[0]}_{column[1]}" for column in wide.columns]
    effects = wide[["regime_type", "regime", "seed"]].copy()
    for metric in ("mae", "rmse", "bias"):
        effects[f"delta_{metric}"] = wide[f"{metric}_dq_dv_removed"] - wide[f"{metric}_full"]
    effects.to_csv(OUT / "paired_effects_by_regime_and_seed.csv", index=False)
    summary = effects.groupby(["regime_type", "regime"]).agg(
        delta_rmse_mean=("delta_rmse", "mean"), delta_rmse_sd=("delta_rmse", "std"),
        delta_mae_mean=("delta_mae", "mean"), delta_mae_sd=("delta_mae", "std"),
        delta_bias_mean=("delta_bias", "mean"), seeds=("seed", "count")
    ).reset_index().sort_values("delta_rmse_mean")
    summary.to_csv(OUT / "paired_effect_summary_by_regime.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
