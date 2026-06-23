from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
FEATURE_DIR = BASE / "charge engineered features"
RW12_RUN_DIR = BASE / "cycle_level_weighted_transformer"
OUTDIR = BASE / "rw12_operating_conditions"
BATTERIES = ("RW9", "RW10", "RW11", "RW12")
SHIFT_FEATURES = [
    "sample_count",
    "duration_s",
    "current_mean_a",
    "current_abs_mean_a",
    "current_std_a",
    "voltage_start",
    "voltage_end",
    "voltage_delta",
    "temperature_mean_c",
    "temperature_delta_c",
    "charge_throughput_ah",
    "energy_throughput_wh",
    "dq_dv_median_ahpv",
]


def load_tables() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for battery in BATTERIES:
        df = pd.read_csv(FEATURE_DIR / f"{battery}_charge_engineered_features.csv")
        df["battery"] = battery
        df["elapsed_days"] = (df["time_end"] - df["time_start"].min()) / 86400.0
        out[battery] = df
    return out


def save_over_time_plots(tables: dict[str, pd.DataFrame]) -> None:
    metrics = [
        ("temperature_mean_c", "Mean Temperature (C)"),
        ("current_abs_mean_a", "Mean |Current| (A)"),
        ("voltage_start", "Voltage Start (V)"),
        ("voltage_delta", "Voltage Delta (V)"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True, sharex=False)
    colors = {"RW9": "#1d3557", "RW10": "#457b9d", "RW11": "#2a9d8f", "RW12": "#d62828"}
    for ax, (col, title) in zip(axes, metrics):
        for battery in BATTERIES:
            df = tables[battery].sort_values("elapsed_days").reset_index(drop=True)
            rolling = df[col].rolling(window=max(75, len(df) // 120), center=True, min_periods=30).median()
            ax.plot(df["elapsed_days"], rolling, linewidth=2, color=colors[battery], label=battery)
        ax.set_title(title)
        ax.set_xlabel("Elapsed experiment days")
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=4)
    fig.savefig(OUTDIR / "01_vct_over_time.png", dpi=180)
    plt.close(fig)


def save_distribution_plots(tables: dict[str, pd.DataFrame]) -> None:
    train = pd.concat([tables["RW9"], tables["RW10"], tables["RW11"]], ignore_index=True)
    rw11 = tables["RW11"]
    rw12 = tables["RW12"]
    features = [
        ("temperature_mean_c", "Mean Temperature (C)"),
        ("current_abs_mean_a", "Mean |Current| (A)"),
        ("voltage_start", "Voltage Start (V)"),
        ("voltage_delta", "Voltage Delta (V)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (feature, title) in zip(axes.flat, features):
        ax.hist(train[feature], bins=60, density=True, histtype="step", linewidth=2, label="train RW9-11")
        ax.hist(rw11[feature], bins=60, density=True, histtype="step", linewidth=1.5, label="RW11")
        ax.hist(rw12[feature], bins=60, density=True, histtype="step", linewidth=1.5, label="RW12")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.savefig(OUTDIR / "02_vct_distribution_shift.png", dpi=180)
    plt.close(fig)


def run_domain_shift_model(tables: dict[str, pd.DataFrame]) -> None:
    train = pd.concat([tables["RW9"], tables["RW10"], tables["RW11"]], ignore_index=True).copy()
    rw12 = tables["RW12"].copy()
    train["is_rw12"] = 0
    rw12["is_rw12"] = 1
    data = pd.concat([train, rw12], ignore_index=True)

    feature_frame = data.loc[:, SHIFT_FEATURES].copy()
    feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
    x = feature_frame.to_numpy(dtype=np.float64)
    y = data["is_rw12"].to_numpy(dtype=np.int64)
    scaler = StandardScaler()
    xz = scaler.fit_transform(x)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(xz, y)
    prob = clf.predict_proba(xz)[:, 1]
    auc = roc_auc_score(y, prob)
    fpr, tpr, _ = roc_curve(y, prob)

    coef = pd.DataFrame(
        {"feature": SHIFT_FEATURES, "coefficient": clf.coef_[0]}
    ).sort_values("coefficient")
    coef.to_csv(OUTDIR / "domain_shift_coefficients.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(fpr, tpr, linewidth=2)
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    axes[0].set_title(f"RW12 vs RW9-11 Domain Classifier ROC\nAUC = {auc:.3f}")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].grid(alpha=0.25)

    show = pd.concat([coef.head(6), coef.tail(6)]).drop_duplicates()
    axes[1].barh(show["feature"], show["coefficient"], color=np.where(show["coefficient"] > 0, "#d62828", "#1d3557"))
    axes[1].set_title("Domain Shift Model Coefficients")
    axes[1].set_xlabel("Standardized logistic coefficient")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(OUTDIR / "03_rw12_domain_classifier.png", dpi=180)
    plt.close(fig)

    # Importance weights for adapting train batteries toward RW12-like conditions.
    train_feature_frame = train.loc[:, SHIFT_FEATURES].copy().fillna(feature_frame.median(numeric_only=True))
    train_prob = clf.predict_proba(scaler.transform(train_feature_frame.to_numpy(dtype=np.float64)))[:, 1]
    ns = len(train)
    nt = len(rw12)
    weights = (ns / nt) * (train_prob / np.clip(1.0 - train_prob, 1.0e-6, None))
    weights = weights / weights.mean()
    weight_df = train.loc[:, ["battery", "cycle", "soh_percent"]].copy()
    weight_df["rw12_condition_weight"] = weights
    weight_df.to_csv(OUTDIR / "rw12_condition_weights_train_cycles.csv", index=False)

    (OUTDIR / "domain_shift_summary.json").write_text(
        json.dumps(
            {
                "auc_rw12_vs_train": float(auc),
                "top_positive_coefficients": coef.tail(5).to_dict(orient="records"),
                "top_negative_coefficients": coef.head(5).to_dict(orient="records"),
                "weight_note": "Weights use RW12 feature distribution and are only valid for domain-adaptation style training, not for strict held-out testing.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def save_rw12_error_feature_plots(tables: dict[str, pd.DataFrame]) -> None:
    pred = pd.read_csv(RW12_RUN_DIR / "test_cycle_predictions.csv")
    rw12 = tables["RW12"]
    merged = pred.merge(
        rw12.loc[:, ["cycle", "actual_soh_percent" if "actual_soh_percent" in rw12.columns else "soh_percent"] + SHIFT_FEATURES].rename(
            columns={"cycle": "charge_cycle", "soh_percent": "feature_soh_percent"}
        ),
        on="charge_cycle",
        how="left",
    )
    merged["signed_error_pp"] = merged["predicted_soh_percent"] - merged["actual_soh_percent"]
    merged["absolute_error_pp"] = merged["signed_error_pp"].abs()

    rows = []
    for feature in ["temperature_mean_c", "current_abs_mean_a", "voltage_start", "voltage_delta", "dq_dv_median_ahpv"]:
        corr = merged[[feature, "signed_error_pp", "absolute_error_pp"]].corr(numeric_only=True)
        rows.append(
            {
                "feature": feature,
                "corr_signed_error": float(corr.loc[feature, "signed_error_pp"]),
                "corr_absolute_error": float(corr.loc[feature, "absolute_error_pp"]),
            }
        )
    pd.DataFrame(rows).sort_values("corr_absolute_error", key=lambda s: s.abs(), ascending=False).to_csv(
        OUTDIR / "rw12_error_feature_correlations.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plot_features = [
        ("temperature_mean_c", "Mean Temperature (C)"),
        ("current_abs_mean_a", "Mean |Current| (A)"),
        ("voltage_start", "Voltage Start (V)"),
        ("voltage_delta", "Voltage Delta (V)"),
    ]
    sample = np.linspace(0, len(merged) - 1, min(len(merged), 5000)).astype(int)
    for ax, (feature, title) in zip(axes.flat, plot_features):
        x = merged[feature].to_numpy(dtype=np.float64)
        y = merged["signed_error_pp"].to_numpy(dtype=np.float64)
        ax.scatter(x[sample], y[sample], s=10, alpha=0.2)
        order = np.argsort(x)
        sx = x[order]
        sy = y[order]
        rolling = pd.Series(sy).rolling(window=max(100, len(sy) // 80), center=True, min_periods=40).mean()
        ax.plot(sx, rolling, color="#d62828", linewidth=2)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"RW12 Error vs {title}")
        ax.set_ylabel("Predicted - Actual (pp)")
        ax.grid(alpha=0.25)
    fig.savefig(OUTDIR / "04_rw12_error_vs_conditions.png", dpi=180)
    plt.close(fig)


def write_report(tables: dict[str, pd.DataFrame]) -> None:
    shift = pd.read_csv(OUTDIR / "domain_shift_coefficients.csv")
    summary = json.loads((OUTDIR / "domain_shift_summary.json").read_text(encoding="utf-8"))
    overview = pd.DataFrame(
        [
            {
                "battery": battery,
                "sample_count_median": float(df["sample_count"].median()),
                "duration_median_s": float(df["duration_s"].median()),
                "current_abs_mean_median": float(df["current_abs_mean_a"].median()),
                "voltage_start_median": float(df["voltage_start"].median()),
                "voltage_delta_median": float(df["voltage_delta"].median()),
                "temperature_mean_median": float(df["temperature_mean_c"].median()),
            }
            for battery, df in tables.items()
        ]
    )
    overview.to_csv(OUTDIR / "battery_condition_overview.csv", index=False)
    rw12 = overview.loc[overview["battery"].eq("RW12")].iloc[0]
    train = overview.loc[overview["battery"].isin(["RW9", "RW10", "RW11"])]

    lines = [
        "# RW12 Operating Conditions Analysis",
        "",
        "Main findings:",
        f"- RW12 has longer charge cycles: median sample count {rw12['sample_count_median']:.0f} versus {train['sample_count_median'].median():.0f} in RW9-RW11.",
        f"- RW12 starts charge from a higher voltage: median voltage_start {rw12['voltage_start_median']:.3f} V versus {train['voltage_start_median'].median():.3f} V in RW9-RW11.",
        f"- RW12 moves through a smaller voltage window per cycle: median voltage_delta {rw12['voltage_delta_median']:.3f} V versus {train['voltage_delta_median'].median():.3f} V.",
        f"- RW12 is cooler than RW9-RW10 and close to RW11 in temperature, so temperature alone does not explain the failure.",
        f"- A logistic domain classifier separates RW12 from RW9-RW11 with AUC {summary['auc_rw12_vs_train']:.3f}, which confirms a measurable operating-condition shift.",
        "",
        "Interpretation:",
        "- RW12 is operated under a different combination of cycle length, starting voltage, voltage traversal, and dQ/dV behavior.",
        "- This means condition-based weighting is plausible.",
        "- But if the weights are learned using RW12 itself, that becomes domain adaptation and should not be treated as a strict held-out test.",
    ]
    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    save_over_time_plots(tables)
    save_distribution_plots(tables)
    run_domain_shift_model(tables)
    save_rw12_error_feature_plots(tables)
    write_report(tables)


if __name__ == "__main__":
    main()
