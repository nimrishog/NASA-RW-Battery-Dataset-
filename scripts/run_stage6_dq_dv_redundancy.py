from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import event_data


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_validation" / "stage6_dq_dv_redundancy"
TARGET = "dq_dv_max_ahpv"
FEATURES = [name for name in event_data.FEATURES if name != TARGET]
VOLTAGE = [name for name in FEATURES if name.startswith("voltage_")]
LOADING = [name for name in FEATURES if name not in VOLTAGE]
GROUPS = {"all_remaining": FEATURES, "voltage_only": VOLTAGE, "loading_only": LOADING}


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    return float(r2_score(y_true, prediction)), float(mean_absolute_error(y_true, prediction))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = event_data.load_events()
    data = data[data["battery"].isin(["RW9", "RW10"])].copy()
    correlations = []
    for feature in FEATURES:
        pearson = data[[TARGET, feature]].corr(method="pearson").iloc[0, 1]
        spearman = spearmanr(data[TARGET], data[feature], nan_policy="omit").statistic
        correlations.append({"feature": feature, "pearson": pearson, "spearman": spearman, "abs_pearson": abs(pearson), "abs_spearman": abs(spearman)})
    correlation_table = pd.DataFrame(correlations).sort_values("abs_spearman", ascending=False)
    correlation_table.to_csv(OUT / "training_only_dq_dv_correlations.csv", index=False)

    rows = []
    for train_cell, test_cell in (("RW9", "RW10"), ("RW10", "RW9")):
        train = data[data["battery"].eq(train_cell)]
        test = data[data["battery"].eq(test_cell)]
        y_train = train[TARGET].to_numpy()
        y_test = test[TARGET].to_numpy()
        for group, features in GROUPS.items():
            x_train = train[features].to_numpy()
            x_test = test[features].to_numpy()
            models = {
                "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.07, max_leaf_nodes=15, l2_regularization=0.02, random_state=42),
            }
            for model_name, model in models.items():
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                r2, mae = metrics(y_test, prediction)
                rows.append({"train_cell": train_cell, "test_cell": test_cell, "feature_group": group, "model": model_name, "r2": r2, "mae_ah_per_v": mae})
    predictability = pd.DataFrame(rows)
    predictability.to_csv(OUT / "training_cell_cross_prediction_of_dq_dv.csv", index=False)
    summary = predictability.groupby(["feature_group", "model"]).agg(r2_mean=("r2", "mean"), r2_min=("r2", "min"), r2_max=("r2", "max"), mae_mean=("mae_ah_per_v", "mean")).reset_index()
    summary.to_csv(OUT / "training_cell_cross_prediction_summary.csv", index=False)
    print("TOP CORRELATIONS\n", correlation_table.head(10).to_string(index=False), flush=True)
    print("\nCROSS-CELL PREDICTABILITY\n", predictability.to_string(index=False), flush=True)
    print("\nSUMMARY\n", summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
