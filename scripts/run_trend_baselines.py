"""Fit the three one-variable ridge baselines reported in the manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import event_data


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_validation" / "trend_baselines"
WINDOW = 32
TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERY = "RW11"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def prepare_endpoints() -> pd.DataFrame:
    frame = event_data.interpolate_labels(event_data.load_events())
    frame = frame.sort_values(["battery", "event_order_chronological"]).reset_index(drop=True)
    frame["elapsed_s"] = frame["end_abs_time"] - frame.groupby("battery")["start_abs_time"].transform("min")
    frame["cumulative_throughput_ah"] = frame.groupby("battery")["throughput_magnitude_ah"].cumsum()
    endpoints = []
    for _, battery_frame in frame.groupby("battery", sort=False):
        endpoints.append(battery_frame.iloc[WINDOW - 1 :].copy())
    return pd.concat(endpoints, ignore_index=True)


def checkpoint_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    table = frame[["battery", "assigned_checkpoint_cycle", "target_next_reference_soh"]].copy()
    table["prediction"] = prediction
    grouped = table.groupby(["battery", "assigned_checkpoint_cycle"], as_index=False).agg(
        measured=("target_next_reference_soh", "first"),
        prediction=("prediction", "mean"),
    )
    result = metrics(grouped["measured"].to_numpy(), grouped["prediction"].to_numpy())
    result["n_checkpoints"] = int(len(grouped))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    endpoints = prepare_endpoints()
    train = endpoints[endpoints["battery"].isin(TRAIN_BATTERIES)].copy()
    test = endpoints[endpoints["battery"].eq(TEST_BATTERY)].copy()
    y_train = train["target_next_reference_soh"].to_numpy(float)
    y_test = test["target_next_reference_soh"].to_numpy(float)
    specifications = {
        "charge_count_ridge": "label_charge_count",
        "elapsed_time_ridge": "elapsed_s",
        "throughput_ridge": "cumulative_throughput_ah",
    }
    rows: list[dict[str, object]] = []
    predictions = test[["battery", "event_order_chronological", "assigned_checkpoint_cycle"]].copy()
    predictions["y_true_soh"] = y_test
    for name, feature in specifications.items():
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train[[feature]], y_train)
        prediction = model.predict(test[[feature]])
        predictions[name] = prediction
        event = metrics(y_test, prediction)
        checkpoint = checkpoint_metrics(test, prediction)
        rows.append(
            {
                "model": name,
                "feature": feature,
                "train_batteries": "+".join(TRAIN_BATTERIES),
                "test_battery": TEST_BATTERY,
                "window": WINDOW,
                "n_train_windows": int(len(train)),
                "n_test_windows": int(len(test)),
                **{f"event_{key}": value for key, value in event.items()},
                **{f"checkpoint_{key}": value for key, value in checkpoint.items()},
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "trend_baseline_metrics.csv", index=False)
    predictions.to_csv(OUT / "trend_baseline_test_predictions.csv", index=False)
    (OUT / "run_metadata.json").write_text(
        json.dumps(
            {
                "target": "next-reference SOH",
                "preprocessing": "StandardScaler fitted on RW9/RW10 only",
                "estimator": "Ridge(alpha=1.0)",
                "evaluation": "RW11 event-window endpoints and diagnostic-checkpoint averages",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
