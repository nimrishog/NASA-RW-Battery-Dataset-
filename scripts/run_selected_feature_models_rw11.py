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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = ROOT / "selected_feature_models_rw11"
TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERY = "RW11"

LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 200,
    "subsample": 0.75,
    "colsample_bytree": 0.6,
    "reg_lambda": 5.0,
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def load_data() -> tuple[pd.DataFrame, list[str]]:
    frames = []
    for battery in (*TRAIN_BATTERIES, TEST_BATTERY):
        df = pd.read_csv(ROOT / "charge engineered features" / f"{battery}_charge_engineered_features.csv")
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


def checkpoint_metric(test_df: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    ref = pd.read_csv(ROOT / "Analysis" / TEST_BATTERY / "reference_capacity_summary.csv")
    checkpoints = (
        ref.groupby("charge_cycle_count_before_reference", as_index=False)
        .agg(capacity_ah=("capacity_ah", "mean"))
        .sort_values("charge_cycle_count_before_reference")
    )
    checkpoints = checkpoints[checkpoints["charge_cycle_count_before_reference"] > 0]
    checkpoint_counts = checkpoints["charge_cycle_count_before_reference"].to_numpy(dtype=np.int64)
    assigned = []
    for cycle in test_df["cycle"].to_numpy(dtype=np.int64):
        idx = int(np.searchsorted(checkpoint_counts, cycle, side="left"))
        if idx >= len(checkpoint_counts):
            idx = len(checkpoint_counts) - 1
        assigned.append(int(checkpoint_counts[idx]))
    pred_df = pd.DataFrame(
        {
            "checkpoint": assigned,
            "actual": test_df["soh_percent"].to_numpy(dtype=np.float64),
            "pred": pred,
        }
    )
    grouped = pred_df.groupby("checkpoint", as_index=False).agg(actual=("actual", "mean"), pred=("pred", "mean"))
    return metrics(grouped["actual"].to_numpy(), grouped["pred"].to_numpy())


def clean(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features].replace([np.inf, -np.inf], np.nan)


def train_lgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, LGBMRegressor]:
    model = LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1, **LGBM_PARAMS)
    model.fit(clean(train_df, features), train_df["soh_percent"].to_numpy(dtype=np.float64))
    return model.predict(clean(test_df, features)), model


def train_mlp(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            alpha=1.0e-3,
            learning_rate_init=1.0e-3,
            max_iter=250,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        ),
    )
    model.fit(
        clean(train_df, features).fillna(0),
        train_df["soh_percent"].to_numpy(dtype=np.float64),
    )
    return model.predict(clean(test_df, features).fillna(0))


def save_plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_df = summary.sort_values("test_r2", ascending=False)
    labels = plot_df["model"].tolist()
    x = np.arange(len(labels))
    for ax, col, title, color in zip(
        axes,
        ["test_r2", "test_mae", "checkpoint_r2"],
        ["RW11 test R2", "RW11 test MAE", "Checkpoint R2"],
        ["#2a9d8f", "#457b9d", "#f4a261"],
    ):
        ax.bar(x, plot_df[col], color=color)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "selected_feature_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data, all_features = load_data()
    train_df = data[data["battery"].isin(TRAIN_BATTERIES)].copy()
    test_df = data[data["battery"].eq(TEST_BATTERY)].copy()
    y_test = test_df["soh_percent"].to_numpy(dtype=np.float64)

    full_pred, full_model = train_lgbm(train_df, test_df, all_features)
    importance = (
        pd.DataFrame({"feature": all_features, "importance": full_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(OUTDIR / "feature_ranking_from_full_lgbm.csv", index=False)

    rows = []
    selected_feature_sets: dict[str, list[str]] = {"top_all_58": all_features}
    for k in [8, 12, 16, 24, 32, 40]:
        selected_feature_sets[f"top_{k}"] = importance.head(k)["feature"].tolist()

    for name, features in selected_feature_sets.items():
        pred, _ = train_lgbm(train_df, test_df, features)
        event = metrics(y_test, pred)
        checkpoint = checkpoint_metric(test_df, pred)
        rows.append(
            {
                "model": f"lgbm_{name}",
                "feature_count": len(features),
                "test_r2": event["r2"],
                "test_mae": event["mae"],
                "test_rmse": event["rmse"],
                "checkpoint_r2": checkpoint["r2"],
                "checkpoint_mae": checkpoint["mae"],
                "checkpoint_rmse": checkpoint["rmse"],
            }
        )

    for name in ["top_8", "top_12", "top_16", "top_24"]:
        pred = train_mlp(train_df, test_df, selected_feature_sets[name])
        event = metrics(y_test, pred)
        checkpoint = checkpoint_metric(test_df, pred)
        rows.append(
            {
                "model": f"mlp_{name}",
                "feature_count": len(selected_feature_sets[name]),
                "test_r2": event["r2"],
                "test_mae": event["mae"],
                "test_rmse": event["rmse"],
                "checkpoint_r2": checkpoint["r2"],
                "checkpoint_mae": checkpoint["mae"],
                "checkpoint_rmse": checkpoint["rmse"],
            }
        )

    summary = pd.DataFrame(rows).sort_values("test_r2", ascending=False)
    summary.to_csv(OUTDIR / "selected_feature_summary.csv", index=False)
    save_plot(summary.head(10))
    (OUTDIR / "selected_feature_sets.json").write_text(json.dumps(selected_feature_sets, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
