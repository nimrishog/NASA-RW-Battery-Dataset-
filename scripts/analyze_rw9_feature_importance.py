from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
FEATURES_CSV = BASE / "charge engineered features" / "RW9_charge_engineered_features.csv"
OUTDIR = BASE / "rw9" / "feature_importance_80_20"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    pct_err = abs_err / np.clip(np.abs(y_true), 1.0e-8, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": float(pct_err.mean() * 100.0),
        "within_5pct": float((pct_err <= 0.05).mean()),
        "within_10pct": float((pct_err <= 0.10).mean()),
    }


def feature_group(feature: str) -> str:
    if feature in {"sample_count", "duration_s"}:
        return "segment_size"
    if feature.startswith("charge_throughput") or feature.startswith("energy_throughput") or feature.startswith("power_"):
        return "throughput_energy"
    if feature.startswith("voltage_") or feature.startswith("dv_dt"):
        return "voltage_dynamics"
    if feature.startswith("current_"):
        return "current_statistics"
    if feature.startswith("temperature_") or feature.startswith("dtemp_dt"):
        return "temperature_statistics"
    if feature.startswith("time_in_vwin") or feature.startswith("time_cross"):
        return "voltage_window_time"
    if feature.startswith("dq_dv") or feature.startswith("dv_dq"):
        return "ic_dv_features"
    if feature.startswith("voltage_pdf"):
        return "voltage_pdf"
    if feature.startswith("temperature_pdf"):
        return "temperature_pdf"
    return "other"


def save_capacity_split_plot(df: pd.DataFrame, split_idx: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.plot(df["cycle"], df["capacity_ah"], linewidth=1.5, color="#1d3557")
    ax.axvline(df["cycle"].iloc[split_idx - 1], color="#e63946", linestyle="--", linewidth=2, label="80/20 split")
    ax.set_title("RW9 capacity over charge cycle with chronological 80/20 split")
    ax.set_xlabel("Charge cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(OUTDIR / "01_capacity_split.png", dpi=180)
    plt.close(fig)


def save_bar_plot(df: pd.DataFrame, value_col: str, title: str, filename: str, color: str) -> None:
    top = df.head(15).copy()
    top = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.barh(top["feature"], top[value_col], color=color)
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(OUTDIR / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    df = df.sort_values("cycle").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    target_col = "capacity_ah"
    non_feature_cols = {
        "battery",
        "cycle",
        "mat_step_index",
        "date",
        "time_start",
        "time_end",
        "label_source",
        "capacity_ah",
        "soh_percent",
    }
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_df[feature_cols])
    x_test = imputer.transform(test_df[feature_cols])
    y_train = train_df[target_col].to_numpy(dtype=np.float64)
    y_test = test_df[target_col].to_numpy(dtype=np.float64)

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_metrics = regression_metrics(y_train, train_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    spearman = train_df[feature_cols + [target_col]].corr(method="spearman")[target_col].drop(target_col)
    corr_df = (
        pd.DataFrame(
            {
                "feature": spearman.index,
                "spearman_corr": spearman.values,
                "abs_spearman_corr": np.abs(spearman.values),
            }
        )
        .sort_values("abs_spearman_corr", ascending=False)
        .reset_index(drop=True)
    )
    corr_df["group"] = corr_df["feature"].map(feature_group)

    mi = mutual_info_regression(x_train, y_train, random_state=42)
    mi_df = (
        pd.DataFrame({"feature": feature_cols, "mutual_information": mi})
        .sort_values("mutual_information", ascending=False)
        .reset_index(drop=True)
    )
    mi_df["group"] = mi_df["feature"].map(feature_group)

    perm = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=8,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    perm_df = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "permutation_importance_mean": perm.importances_mean,
                "permutation_importance_std": perm.importances_std,
            }
        )
        .sort_values("permutation_importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    perm_df["group"] = perm_df["feature"].map(feature_group)

    group_summary = (
        perm_df.groupby("group", as_index=False)["permutation_importance_mean"]
        .sum()
        .sort_values("permutation_importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    corr_df.to_csv(OUTDIR / "feature_spearman_correlations.csv", index=False)
    mi_df.to_csv(OUTDIR / "feature_mutual_information.csv", index=False)
    perm_df.to_csv(OUTDIR / "feature_permutation_importance.csv", index=False)
    group_summary.to_csv(OUTDIR / "feature_group_importance.csv", index=False)

    split_counts = pd.DataFrame(
        [
            {"split": "train", "count": int(len(train_df))},
            {"split": "test", "count": int(len(test_df))},
        ]
    )
    split_counts.to_csv(OUTDIR / "split_counts.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "cycle": test_df["cycle"].to_numpy(dtype=np.int64),
            "actual_capacity_ah": y_test,
            "predicted_capacity_ah": test_pred,
            "absolute_error_ah": np.abs(test_pred - y_test),
            "percent_error": np.abs(test_pred - y_test) / np.clip(np.abs(y_test), 1.0e-8, None) * 100.0,
        }
    )
    pred_df.to_csv(OUTDIR / "test_predictions.csv", index=False)

    save_capacity_split_plot(df, split_idx)
    save_bar_plot(corr_df, "abs_spearman_corr", "Top RW9 features by absolute Spearman correlation", "02_top_spearman_features.png", "#457b9d")
    save_bar_plot(perm_df, "permutation_importance_mean", "Top RW9 features by test permutation importance", "03_top_permutation_features.png", "#e76f51")
    save_bar_plot(mi_df, "mutual_information", "Top RW9 features by mutual information", "04_top_mutual_information_features.png", "#2a9d8f")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    lo = min(float(y_test.min()), float(test_pred.min()))
    hi = max(float(y_test.max()), float(test_pred.max()))
    axes[0].scatter(y_test, test_pred, s=12, alpha=0.45)
    axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1)
    axes[0].set_title("RW9 test predicted vs actual capacity")
    axes[0].set_xlabel("Actual capacity (Ah)")
    axes[0].set_ylabel("Predicted capacity (Ah)")
    residuals = test_pred - y_test
    axes[1].scatter(y_test, residuals, s=12, alpha=0.45)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("RW9 test residuals")
    axes[1].set_xlabel("Actual capacity (Ah)")
    axes[1].set_ylabel("Prediction error (Ah)")
    fig.savefig(OUTDIR / "05_test_predictions.png", dpi=180)
    plt.close(fig)

    report_lines = [
        "# RW9 Feature Importance Analysis",
        "",
        "Chronological split:",
        f"- train cycles: 1 to {int(train_df['cycle'].iloc[-1])}",
        f"- test cycles: {int(test_df['cycle'].iloc[0])} to {int(test_df['cycle'].iloc[-1])}",
        "",
        "Model used for importance ranking:",
        "- RandomForestRegressor",
        "- n_estimators = 400",
        "- max_depth = 12",
        "- min_samples_leaf = 2",
        "",
        "Performance:",
        f"- train MAE: {train_metrics['mae']:.4f} Ah",
        f"- train RMSE: {train_metrics['rmse']:.4f} Ah",
        f"- train R2: {train_metrics['r2']:.4f}",
        f"- test MAE: {test_metrics['mae']:.4f} Ah",
        f"- test RMSE: {test_metrics['rmse']:.4f} Ah",
        f"- test R2: {test_metrics['r2']:.4f}",
        "",
        "Top permutation-importance features:",
    ]
    for _, row in perm_df.head(10).iterrows():
        report_lines.append(
            f"- {row['feature']}: {row['permutation_importance_mean']:.6f} ({row['group']})"
        )
    report_lines.append("")
    report_lines.append("Top feature groups by summed permutation importance:")
    for _, row in group_summary.iterrows():
        report_lines.append(f"- {row['group']}: {row['permutation_importance_mean']:.6f}")

    (OUTDIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "top_permutation_features": perm_df.head(10).to_dict(orient="records"),
        "top_group_importance": group_summary.to_dict(orient="records"),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
