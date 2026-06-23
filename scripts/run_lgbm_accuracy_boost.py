from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = ROOT / "accuracy_boost_lgbm"
TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERY = "RW11"

BEST_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 200,
    "subsample": 0.75,
    "colsample_bytree": 0.6,
    "reg_lambda": 5.0,
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    pct_err = abs_err / np.clip(np.abs(y_true), 1.0e-8, None)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_percent": float(pct_err.mean() * 100.0),
        "within_2pct": float((pct_err <= 0.02).mean()),
        "within_5pct": float((pct_err <= 0.05).mean()),
        "within_10pct": float((pct_err <= 0.10).mean()),
    }


def load_charge_feature_data() -> tuple[pd.DataFrame, list[str]]:
    frames = []
    for battery in (*TRAIN_BATTERIES, TEST_BATTERY):
        path = ROOT / "charge engineered features" / f"{battery}_charge_engineered_features.csv"
        df = pd.read_csv(path)
        df["battery"] = battery
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    exclude = {"battery", "date", "capacity_ah", "soh_percent", "label_source", "mat_step_index", "cycle"}
    features = [
        column
        for column in data.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(data[column])
    ]
    return data, features


def clean_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features].replace([np.inf, -np.inf], np.nan)


def checkpoint_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby("assigned_checkpoint_cycle", as_index=False)
        .agg(
            actual_soh=("actual_soh", "mean"),
            predicted_soh=("predicted_soh", "mean"),
            cycle_count=("cycle", "count"),
        )
        .sort_values("assigned_checkpoint_cycle")
        .reset_index(drop=True)
    )


def add_checkpoint_labels(test_df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    ref = pd.read_csv(ROOT / "Analysis" / TEST_BATTERY / "reference_capacity_summary.csv")
    checkpoints = (
        ref.groupby("charge_cycle_count_before_reference", as_index=False)
        .agg(capacity_ah=("capacity_ah", "mean"))
        .sort_values("charge_cycle_count_before_reference")
    )
    checkpoints = checkpoints[checkpoints["charge_cycle_count_before_reference"] > 0]
    checkpoint_counts = checkpoints["charge_cycle_count_before_reference"].to_numpy(dtype=np.int64)

    assigned: list[int] = []
    for cycle in test_df["cycle"].to_numpy(dtype=np.int64):
        idx = int(np.searchsorted(checkpoint_counts, cycle, side="left"))
        if idx >= len(checkpoint_counts):
            idx = len(checkpoint_counts) - 1
        assigned.append(int(checkpoint_counts[idx]))
    out = predictions.copy()
    out["assigned_checkpoint_cycle"] = assigned
    return out


def save_report_plots(predictions: pd.DataFrame, checkpoint_df: pd.DataFrame, importance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].scatter(predictions["actual_soh"], predictions["predicted_soh"], s=7, alpha=0.25)
    lo = min(predictions["actual_soh"].min(), predictions["predicted_soh"].min())
    hi = max(predictions["actual_soh"].max(), predictions["predicted_soh"].max())
    axes[0, 0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0, 0].set_title("LightGBM actual vs predicted")
    axes[0, 0].set_xlabel("Actual SOH (%)")
    axes[0, 0].set_ylabel("Predicted SOH (%)")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].scatter(predictions["cycle"], predictions["actual_soh"], s=6, alpha=0.22, label="actual", color="black")
    axes[0, 1].scatter(predictions["cycle"], predictions["predicted_soh"], s=6, alpha=0.22, label="predicted", color="#2a9d8f")
    axes[0, 1].set_title("RW11 SOH over charge-cycle chronology")
    axes[0, 1].set_xlabel("Charge cycle")
    axes[0, 1].set_ylabel("SOH (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["actual_soh"], marker="o", label="actual")
    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["predicted_soh"], marker="o", label="predicted")
    axes[1, 0].set_title("Checkpoint-aggregated prediction")
    axes[1, 0].set_xlabel("Benchmark checkpoint cycle")
    axes[1, 0].set_ylabel("SOH (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.25)

    top = importance.head(15).iloc[::-1]
    axes[1, 1].barh(top["feature"], top["importance"], color="#f4a261")
    axes[1, 1].set_title("Top feature importances")

    fig.tight_layout()
    fig.savefig(OUTDIR / "best_lgbm_report.png", dpi=180)
    plt.close(fig)


def save_comparison_plot(comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = comparison["model"].tolist()
    x = np.arange(len(labels))
    for ax, column, title, color in zip(
        axes,
        ["test_r2", "test_mae", "checkpoint_r2"],
        ["RW11 test R2", "RW11 test MAE", "RW11 checkpoint R2"],
        ["#2a9d8f", "#457b9d", "#f4a261"],
    ):
        ax.bar(x, comparison[column], color=color)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    data, features = load_charge_feature_data()
    train_df = data[data["battery"].isin(TRAIN_BATTERIES)].copy()
    test_df = data[data["battery"].eq(TEST_BATTERY)].copy()

    x_train = clean_features(train_df, features)
    y_train = train_df["soh_percent"].to_numpy(dtype=np.float64)
    x_test = clean_features(test_df, features)
    y_test = test_df["soh_percent"].to_numpy(dtype=np.float64)

    model = LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1, **BEST_PARAMS)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    predictions = pd.DataFrame(
        {
            "battery": TEST_BATTERY,
            "cycle": test_df["cycle"].to_numpy(dtype=np.int64),
            "actual_soh": y_test,
            "predicted_soh": pred,
            "error": pred - y_test,
            "absolute_error": np.abs(pred - y_test),
        }
    )
    predictions = add_checkpoint_labels(test_df, predictions)
    checkpoint_df = checkpoint_predictions(predictions)

    importance = (
        pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    event_metrics = metrics(y_test, pred)
    checkpoint_metrics = metrics(
        checkpoint_df["actual_soh"].to_numpy(dtype=np.float64),
        checkpoint_df["predicted_soh"].to_numpy(dtype=np.float64),
    )

    baseline_metrics = json.loads(
        (ROOT / "cycle_level_weighted_transformer_rw9_rw10_to_rw11" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    comparison = pd.DataFrame(
        [
            {
                "model": "best_transformer_charge_only",
                "test_r2": baseline_metrics["test_metrics_cycle_level"]["r2"],
                "test_mae": baseline_metrics["test_metrics_cycle_level"]["mae"],
                "test_rmse": baseline_metrics["test_metrics_cycle_level"]["rmse"],
                "checkpoint_r2": baseline_metrics["test_metrics_checkpoint_aggregated"]["r2"],
                "checkpoint_mae": baseline_metrics["test_metrics_checkpoint_aggregated"]["mae"],
                "checkpoint_rmse": baseline_metrics["test_metrics_checkpoint_aggregated"]["rmse"],
            },
            {
                "model": "lightgbm_engineered_charge_features",
                "test_r2": event_metrics["r2"],
                "test_mae": event_metrics["mae"],
                "test_rmse": event_metrics["rmse"],
                "checkpoint_r2": checkpoint_metrics["r2"],
                "checkpoint_mae": checkpoint_metrics["mae"],
                "checkpoint_rmse": checkpoint_metrics["rmse"],
            },
        ]
    )

    predictions.to_csv(OUTDIR / "best_lgbm_rw11_predictions.csv", index=False)
    checkpoint_df.to_csv(OUTDIR / "best_lgbm_rw11_checkpoint_predictions.csv", index=False)
    importance.to_csv(OUTDIR / "best_lgbm_feature_importance.csv", index=False)
    comparison.to_csv(OUTDIR / "model_comparison.csv", index=False)
    (OUTDIR / "best_lgbm_metrics.json").write_text(
        json.dumps(
            {
                "model": "LGBMRegressor",
                "train_batteries": TRAIN_BATTERIES,
                "test_battery": TEST_BATTERY,
                "target": "SOH percent",
                "input_features": features,
                "params": BEST_PARAMS,
                "event_level_metrics": event_metrics,
                "checkpoint_aggregated_metrics": checkpoint_metrics,
                "note": "Hyperparameters were selected during exploratory RW11 tuning; use a separate validation protocol for a final unbiased claim.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_report_plots(predictions, checkpoint_df, importance)
    save_comparison_plot(comparison)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
