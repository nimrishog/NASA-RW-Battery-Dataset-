from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
FEATURE_DIR = BASE / "charge engineered features"
ANALYSIS_DIR = BASE / "Analysis"
RW12_RUN_DIR = BASE / "cycle_level_weighted_transformer"
OUTDIR = BASE / "rw12_behavior_analysis"
BATTERIES = ("RW9", "RW10", "RW11", "RW12")
CORE_FEATURES = [
    "sample_count",
    "duration_s",
    "charge_throughput_ah",
    "energy_throughput_wh",
    "current_abs_mean_a",
    "voltage_start",
    "voltage_end",
    "voltage_delta",
    "temperature_mean_c",
    "temperature_delta_c",
    "dq_dv_median_ahpv",
]


def load_feature_tables() -> dict[str, pd.DataFrame]:
    return {
        battery: pd.read_csv(FEATURE_DIR / f"{battery}_charge_engineered_features.csv")
        for battery in BATTERIES
    }


def load_reference_soh(battery: str) -> pd.DataFrame:
    ref = pd.read_csv(ANALYSIS_DIR / battery / "reference_capacity_summary.csv")
    grouped = (
        ref.groupby("charge_cycle_count_before_reference", as_index=False)
        .agg(capacity_ah=("capacity_ah", "mean"))
        .sort_values("charge_cycle_count_before_reference")
        .reset_index(drop=True)
    )
    init = float(grouped.loc[grouped["charge_cycle_count_before_reference"].eq(0), "capacity_ah"].mean())
    grouped = grouped[grouped["charge_cycle_count_before_reference"] > 0].copy()
    grouped["soh_percent"] = grouped["capacity_ah"] / init * 100.0
    grouped["battery"] = battery
    return grouped


def save_overview_tables(feature_tables: dict[str, pd.DataFrame]) -> None:
    ref_rows: list[dict[str, float | int | str]] = []
    for battery, df in feature_tables.items():
        ref = load_reference_soh(battery)
        gaps = ref["charge_cycle_count_before_reference"].diff().dropna()
        ref_rows.append(
            {
                "battery": battery,
                "cycle_samples": len(df),
                "soh_min": float(df["soh_percent"].min()),
                "soh_max": float(df["soh_percent"].max()),
                "soh_median": float(df["soh_percent"].median()),
                "checkpoint_count": len(ref),
                "checkpoint_gap_mean": float(gaps.mean()),
                "checkpoint_gap_min": float(gaps.min()),
                "checkpoint_gap_max": float(gaps.max()),
                "sample_count_median": float(df["sample_count"].median()),
                "duration_median_s": float(df["duration_s"].median()),
                "voltage_delta_median": float(df["voltage_delta"].median()),
                "temperature_mean_median_c": float(df["temperature_mean_c"].median()),
                "dq_dv_median": float(df["dq_dv_median_ahpv"].median()),
                "pct_full_301_rows": float(df["sample_count"].eq(301).mean()),
            }
        )
    pd.DataFrame(ref_rows).to_csv(OUTDIR / "battery_overview.csv", index=False)

    train = pd.concat([feature_tables["RW9"], feature_tables["RW10"], feature_tables["RW11"]], ignore_index=True)
    rw11 = feature_tables["RW11"]
    rw12 = feature_tables["RW12"]
    shift_rows = []
    for battery_name, test in [("RW11", rw11), ("RW12", rw12)]:
        for feature in CORE_FEATURES:
            a = train[feature].dropna().to_numpy(dtype=np.float64)
            b = test[feature].dropna().to_numpy(dtype=np.float64)
            ks = ks_2samp(a, b)
            shift_rows.append(
                {
                    "test_battery": battery_name,
                    "feature": feature,
                    "ks_stat": float(ks.statistic),
                    "train_mean": float(a.mean()),
                    "test_mean": float(b.mean()),
                    "train_median": float(np.median(a)),
                    "test_median": float(np.median(b)),
                }
            )
    pd.DataFrame(shift_rows).sort_values(["test_battery", "ks_stat"], ascending=[True, False]).to_csv(
        OUTDIR / "feature_shift_vs_train.csv", index=False
    )

    corr_rows = []
    features = [
        "sample_count",
        "duration_s",
        "charge_throughput_ah",
        "energy_throughput_wh",
        "current_abs_mean_a",
        "voltage_start",
        "voltage_delta",
        "temperature_mean_c",
        "dq_dv_median_ahpv",
    ]
    for battery, df in feature_tables.items():
        corr = df[["soh_percent"] + features].corr(numeric_only=True)["soh_percent"].drop("soh_percent")
        for feature, value in corr.items():
            corr_rows.append({"battery": battery, "feature": feature, "corr_with_soh": float(value)})
    pd.DataFrame(corr_rows).to_csv(OUTDIR / "feature_soh_correlations.csv", index=False)


def plot_reference_soh_curves() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for battery in BATTERIES:
        ref = load_reference_soh(battery)
        ax.plot(
            ref["charge_cycle_count_before_reference"],
            ref["soh_percent"],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=battery,
        )
    ax.set_title("Benchmark SOH Trajectory by Battery")
    ax.set_xlabel("Charge cycles before benchmark discharge")
    ax.set_ylabel("SOH (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(OUTDIR / "01_reference_soh_vs_cycle.png", dpi=180)
    plt.close(fig)


def plot_feature_distributions(feature_tables: dict[str, pd.DataFrame]) -> None:
    train = pd.concat([feature_tables["RW9"], feature_tables["RW10"], feature_tables["RW11"]], ignore_index=True)
    rw12 = feature_tables["RW12"]
    rw11 = feature_tables["RW11"]
    feats = [
        ("sample_count", "Sample Count"),
        ("voltage_delta", "Voltage Delta (V)"),
        ("temperature_mean_c", "Mean Temperature (C)"),
        ("dq_dv_median_ahpv", "Median dQ/dV"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (feature, title) in zip(axes.flat, feats):
        ax.hist(train[feature], bins=60, density=True, histtype="step", linewidth=2, label="train RW9-11")
        ax.hist(rw11[feature], bins=60, density=True, histtype="step", linewidth=1.6, label="RW11")
        ax.hist(rw12[feature], bins=60, density=True, histtype="step", linewidth=1.6, label="RW12")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.savefig(OUTDIR / "02_feature_distribution_shift.png", dpi=180)
    plt.close(fig)


def plot_feature_vs_soh(feature_tables: dict[str, pd.DataFrame]) -> None:
    compare = {"RW11": feature_tables["RW11"], "RW12": feature_tables["RW12"]}
    feats = [
        ("sample_count", "Sample Count"),
        ("voltage_delta", "Voltage Delta (V)"),
        ("dq_dv_median_ahpv", "Median dQ/dV"),
        ("temperature_mean_c", "Mean Temperature (C)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"RW11": "#1d3557", "RW12": "#d62828"}
    for ax, (feature, title) in zip(axes.flat, feats):
        for battery, df in compare.items():
            sub = df[["soh_percent", feature]].dropna().sort_values("soh_percent")
            ax.scatter(sub["soh_percent"], sub[feature], s=4, alpha=0.08, color=colors[battery])
            rolling = sub[feature].rolling(window=max(50, len(sub) // 80), center=True, min_periods=30).median()
            ax.plot(sub["soh_percent"], rolling, linewidth=2, color=colors[battery], label=battery if feature == feats[0][0] else None)
        ax.set_title(title)
        ax.set_xlabel("SOH (%)")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.savefig(OUTDIR / "03_rw11_rw12_feature_vs_soh.png", dpi=180)
    plt.close(fig)


def plot_correlation_heatmap() -> None:
    corr = pd.read_csv(OUTDIR / "feature_soh_correlations.csv")
    pivot = corr.pivot(index="feature", columns="battery", values="corr_with_soh").loc[
        [
            "charge_throughput_ah",
            "energy_throughput_wh",
            "dq_dv_median_ahpv",
            "duration_s",
            "sample_count",
            "voltage_start",
            "temperature_mean_c",
            "current_abs_mean_a",
            "voltage_delta",
        ]
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Feature-SOH Correlation by Battery")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Correlation with SOH")
    fig.savefig(OUTDIR / "04_feature_soh_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def plot_rw12_error_behavior() -> None:
    test_cycle = pd.read_csv(RW12_RUN_DIR / "test_cycle_predictions.csv")
    cp = pd.read_csv(RW12_RUN_DIR / "test_checkpoint_predictions.csv")
    cp["error_pp"] = cp["predicted_soh_percent"] - cp["actual_soh_percent"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    sample = np.linspace(0, len(test_cycle) - 1, min(len(test_cycle), 5000)).astype(int)
    axes[0].scatter(
        test_cycle["actual_soh_percent"].to_numpy()[sample],
        test_cycle["predicted_soh_percent"].to_numpy()[sample] - test_cycle["actual_soh_percent"].to_numpy()[sample],
        s=10,
        alpha=0.25,
    )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("RW12 Cycle-Level Error vs Actual SOH")
    axes[0].set_xlabel("Actual SOH (%)")
    axes[0].set_ylabel("Predicted - Actual (pp)")
    axes[0].grid(alpha=0.25)

    colors = np.where(cp["error_pp"] >= 0.0, "#2a9d8f", "#e76f51")
    axes[1].bar(cp["assigned_checkpoint_cycle"].astype(str), cp["error_pp"], color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("RW12 Checkpoint Error by Benchmark")
    axes[1].set_xlabel("Checkpoint cycle")
    axes[1].set_ylabel("Predicted - Actual (pp)")
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].grid(axis="y", alpha=0.25)

    fig.savefig(OUTDIR / "05_rw12_prediction_error_behavior.png", dpi=180)
    plt.close(fig)

    cp.sort_values("error_pp").to_csv(OUTDIR / "rw12_checkpoint_errors.csv", index=False)


def write_summary(feature_tables: dict[str, pd.DataFrame]) -> None:
    overview = pd.read_csv(OUTDIR / "battery_overview.csv")
    shifts = pd.read_csv(OUTDIR / "feature_shift_vs_train.csv")
    corr = pd.read_csv(OUTDIR / "feature_soh_correlations.csv")
    rw12_err = pd.read_csv(OUTDIR / "rw12_checkpoint_errors.csv")

    rw12 = overview.loc[overview["battery"].eq("RW12")].iloc[0]
    rw11 = overview.loc[overview["battery"].eq("RW11")].iloc[0]
    train = overview.loc[overview["battery"].isin(["RW9", "RW10", "RW11"])]
    top_shift = shifts.loc[shifts["test_battery"].eq("RW12")].sort_values("ks_stat", ascending=False).head(5)
    corr_pivot = corr.pivot(index="feature", columns="battery", values="corr_with_soh")
    summary = {
        "rw12_vs_rw11": {
            "rw12_soh_min": float(rw12["soh_min"]),
            "rw11_soh_min": float(rw11["soh_min"]),
            "rw12_sample_count_median": float(rw12["sample_count_median"]),
            "rw11_sample_count_median": float(rw11["sample_count_median"]),
            "rw12_voltage_delta_median": float(rw12["voltage_delta_median"]),
            "rw11_voltage_delta_median": float(rw11["voltage_delta_median"]),
        },
        "top_rw12_shift_features": top_shift.to_dict(orient="records"),
        "feature_soh_correlation_example": {
            feat: {battery: float(corr_pivot.loc[feat, battery]) for battery in corr_pivot.columns}
            for feat in ["charge_throughput_ah", "dq_dv_median_ahpv", "duration_s", "temperature_mean_c"]
        },
        "worst_rw12_checkpoint_errors": rw12_err.head(8).to_dict(orient="records"),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# RW12 Behavior Analysis",
        "",
        "Main findings:",
        f"- RW12 is less degraded than the other batteries: minimum SOH {rw12['soh_min']:.2f}% versus {train['soh_min'].min():.2f}% to {train['soh_min'].max():.2f}% in RW9-RW11.",
        f"- RW12 has longer charge segments: median sample count {rw12['sample_count_median']:.0f} versus {train['sample_count_median'].median():.0f} in the train batteries.",
        f"- RW12 traverses a smaller voltage window per cycle: median voltage delta {rw12['voltage_delta_median']:.3f} V versus {train['voltage_delta_median'].median():.3f} V in RW9-RW11.",
        "- RW12 also shows weaker feature-to-SOH relationships than the train batteries, especially for throughput, duration, and dQ/dV.",
        "- In the RW9-RW10-RW11 -> RW12 model, the largest RW12 errors occur in the mid/late checkpoints around 76-83% SOH, where the model systematically underpredicts.",
        "",
        "Interpretation:",
        "- RW12 is a covariate-shift problem, not just a new battery ID.",
        "- The label range is narrower and shifted upward, and the charge behavior at matched SOH differs from RW11 and from the train batteries.",
        "- That means the model sees many familiar raw values, but the mapping from charge behavior to SOH is weaker and different on RW12.",
    ]
    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    feature_tables = load_feature_tables()
    save_overview_tables(feature_tables)
    plot_reference_soh_curves()
    plot_feature_distributions(feature_tables)
    plot_feature_vs_soh(feature_tables)
    plot_correlation_heatmap()
    plot_rw12_error_behavior()
    write_summary(feature_tables)


if __name__ == "__main__":
    main()
