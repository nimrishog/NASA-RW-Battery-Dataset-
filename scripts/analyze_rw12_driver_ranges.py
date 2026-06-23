from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
FEATURE_DIR = BASE / "charge engineered features"
OUTDIR = BASE / "rw12_driver_ranges"

BATTERIES = ("RW9", "RW10", "RW11", "RW12")
FEATURES = [
    ("sample_count", "Sample count"),
    ("duration_s", "Duration (s)"),
    ("voltage_start", "Voltage start (V)"),
    ("voltage_end", "Voltage end (V)"),
    ("voltage_delta", "Voltage delta (V)"),
    ("current_abs_mean_a", "Mean |current| (A)"),
    ("temperature_mean_c", "Mean temperature (C)"),
    ("charge_throughput_ah", "Charge throughput (Ah)"),
    ("energy_throughput_wh", "Energy throughput (Wh)"),
    ("dq_dv_median_ahpv", "Median dQ/dV"),
]
SOH_BINS = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        battery: pd.read_csv(FEATURE_DIR / f"{battery}_charge_engineered_features.csv")
        for battery in BATTERIES
    }


def range_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> float:
    inter = max(0.0, min(a_high, b_high) - max(a_low, b_low))
    union = max(a_high, b_high) - min(a_low, b_low)
    if union <= 1.0e-12:
        return 1.0
    return inter / union


def save_range_tables(tables: dict[str, pd.DataFrame]) -> None:
    rows = []
    for battery, df in tables.items():
        for feature, _ in FEATURES:
            s = df[feature].dropna().to_numpy(dtype=np.float64)
            rows.append(
                {
                    "battery": battery,
                    "feature": feature,
                    "min": float(np.min(s)),
                    "p05": float(np.percentile(s, 5)),
                    "p25": float(np.percentile(s, 25)),
                    "median": float(np.percentile(s, 50)),
                    "p75": float(np.percentile(s, 75)),
                    "p95": float(np.percentile(s, 95)),
                    "max": float(np.max(s)),
                }
            )
    ranges = pd.DataFrame(rows)
    ranges.to_csv(OUTDIR / "battery_feature_ranges.csv", index=False)

    train = ranges[ranges["battery"].isin(["RW9", "RW10", "RW11"])]
    train_agg = (
        train.groupby("feature", as_index=False)
        .agg(
            train_p05=("p05", "median"),
            train_p25=("p25", "median"),
            train_median=("median", "median"),
            train_p75=("p75", "median"),
            train_p95=("p95", "median"),
        )
    )
    rw11 = ranges[ranges["battery"].eq("RW11")].copy()
    rw12 = ranges[ranges["battery"].eq("RW12")].copy()
    merged = (
        train_agg.merge(rw11, on="feature", suffixes=("", "_rw11"))
        .merge(rw12, on="feature", suffixes=("_rw11", "_rw12"))
    )
    merged["rw11_vs_rw12_p05_p95_overlap"] = [
        range_overlap(a, b, c, d)
        for a, b, c, d in zip(merged["p05_rw11"], merged["p95_rw11"], merged["p05_rw12"], merged["p95_rw12"])
    ]
    merged["train_vs_rw12_p05_p95_overlap"] = [
        range_overlap(a, b, c, d)
        for a, b, c, d in zip(merged["train_p05"], merged["train_p95"], merged["p05_rw12"], merged["p95_rw12"])
    ]
    merged.to_csv(OUTDIR / "rw11_rw12_range_overlap.csv", index=False)


def plot_range_bands() -> None:
    ranges = pd.read_csv(OUTDIR / "battery_feature_ranges.csv")
    chosen = ["sample_count", "voltage_start", "voltage_delta", "temperature_mean_c", "charge_throughput_ah", "dq_dv_median_ahpv"]
    labels = {f: lbl for f, lbl in FEATURES}
    colors = {"RW9": "#1d3557", "RW10": "#457b9d", "RW11": "#2a9d8f", "RW12": "#d62828"}

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for ax, feature in zip(axes.flat, chosen):
        sub = ranges[ranges["feature"].eq(feature)].copy()
        y = np.arange(len(sub))
        for idx, row in enumerate(sub.itertuples(index=False)):
            ax.hlines(idx, row.p05, row.p95, color=colors[row.battery], linewidth=4, alpha=0.8)
            ax.hlines(idx, row.p25, row.p75, color=colors[row.battery], linewidth=9, alpha=0.45)
            ax.plot(row.median, idx, "o", color=colors[row.battery], markersize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["battery"])
        ax.set_title(labels[feature])
        ax.grid(axis="x", alpha=0.25)
    fig.savefig(OUTDIR / "01_feature_range_bands.png", dpi=180)
    plt.close(fig)


def save_matched_soh_driver_table(tables: dict[str, pd.DataFrame]) -> None:
    rw11 = tables["RW11"]
    rw12 = tables["RW12"]
    rows = []
    for feature, label in FEATURES:
        weighted_abs_d = 0.0
        weighted_n = 0.0
        signed_d_sum = 0.0
        detail = []
        for lo, hi in SOH_BINS:
            a = rw11[(rw11["soh_percent"] >= lo) & (rw11["soh_percent"] < hi)][feature].dropna().to_numpy(dtype=np.float64)
            b = rw12[(rw12["soh_percent"] >= lo) & (rw12["soh_percent"] < hi)][feature].dropna().to_numpy(dtype=np.float64)
            if len(a) < 20 or len(b) < 20:
                continue
            na, nb = len(a), len(b)
            sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
            pooled = np.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / max(na + nb - 2, 1))
            effect = (np.mean(b) - np.mean(a)) / pooled if pooled > 1.0e-8 else 0.0
            weight = min(na, nb)
            weighted_abs_d += abs(effect) * weight
            signed_d_sum += effect * weight
            weighted_n += weight
            detail.append(
                {
                    "soh_bin": f"{lo}-{hi}",
                    "rw11_median": float(np.median(a)),
                    "rw12_median": float(np.median(b)),
                    "effect_size": float(effect),
                    "rw11_n": int(na),
                    "rw12_n": int(nb),
                }
            )
        rows.append(
            {
                "feature": feature,
                "label": label,
                "avg_abs_effect_size": weighted_abs_d / weighted_n if weighted_n else np.nan,
                "avg_signed_effect_size": signed_d_sum / weighted_n if weighted_n else np.nan,
                "detail_json": json.dumps(detail),
            }
        )
    out = pd.DataFrame(rows).sort_values("avg_abs_effect_size", ascending=False).reset_index(drop=True)
    out.to_csv(OUTDIR / "rw11_rw12_matched_soh_driver_scores.csv", index=False)


def plot_driver_scores() -> None:
    scores = pd.read_csv(OUTDIR / "rw11_rw12_matched_soh_driver_scores.csv").head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    colors = np.where(scores["avg_signed_effect_size"] > 0, "#d62828", "#1d3557")
    ax.barh(scores["label"], scores["avg_abs_effect_size"], color=colors)
    ax.set_title("RW12 vs RW11 Driver Strength Within Matched SOH")
    ax.set_xlabel("Average absolute effect size across SOH bins")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(OUTDIR / "02_driver_strength_bar.png", dpi=180)
    plt.close(fig)


def plot_matched_soh_profiles(tables: dict[str, pd.DataFrame]) -> None:
    rw11 = tables["RW11"].copy()
    rw12 = tables["RW12"].copy()
    bins = pd.IntervalIndex.from_tuples(SOH_BINS, closed="left")
    top = pd.read_csv(OUTDIR / "rw11_rw12_matched_soh_driver_scores.csv").head(4)
    chosen = top["feature"].tolist()
    labels = {f: lbl for f, lbl in FEATURES}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, feature in zip(axes.flat, chosen):
        xs = []
        y11 = []
        y12 = []
        for lo, hi in SOH_BINS:
            m11 = rw11[(rw11["soh_percent"] >= lo) & (rw11["soh_percent"] < hi)][feature].dropna()
            m12 = rw12[(rw12["soh_percent"] >= lo) & (rw12["soh_percent"] < hi)][feature].dropna()
            if len(m11) == 0 or len(m12) == 0:
                continue
            xs.append((lo + hi) / 2)
            y11.append(float(m11.median()))
            y12.append(float(m12.median()))
        ax.plot(xs, y11, marker="o", linewidth=2, label="RW11")
        ax.plot(xs, y12, marker="o", linewidth=2, label="RW12")
        ax.set_title(labels[feature])
        ax.set_xlabel("SOH bin center (%)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.savefig(OUTDIR / "03_matched_soh_feature_profiles.png", dpi=180)
    plt.close(fig)


def write_report() -> None:
    overlap = pd.read_csv(OUTDIR / "rw11_rw12_range_overlap.csv")
    scores = pd.read_csv(OUTDIR / "rw11_rw12_matched_soh_driver_scores.csv")
    top = scores.head(5)

    lines = [
        "# RW12 Driver Range Analysis",
        "",
        "This report compares RW12 against RW11 and the train batteries using explicit feature ranges and matched-SOH separation.",
        "",
        "Main findings:",
    ]

    for feature in ["sample_count", "voltage_start", "voltage_delta", "charge_throughput_ah", "dq_dv_median_ahpv"]:
        row = overlap[overlap["feature"].eq(feature)].iloc[0]
        lines.append(
            f"- {feature}: RW11 vs RW12 5-95 overlap = {row['rw11_vs_rw12_p05_p95_overlap']:.3f}, "
            f"train vs RW12 5-95 overlap = {row['train_vs_rw12_p05_p95_overlap']:.3f}."
        )

    lines.extend(
        [
            "",
            "Top separating features after matching RW11 and RW12 by SOH:",
        ]
    )

    for row in top.itertuples(index=False):
        direction = "higher in RW12" if row.avg_signed_effect_size > 0 else "lower in RW12"
        lines.append(
            f"- {row.label}: driver score {row.avg_abs_effect_size:.3f}, typically {direction}."
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- Temperature is shifted, but RW11 is already cool, so temperature alone is not the main explanation.",
            "- The more important structural differences are longer full-length cycles, higher starting voltage, smaller voltage traversal, and weaker throughput progression at the same SOH.",
            "- This is an inference from the feature ranges and matched-SOH separations; it is not proof of physical causality.",
        ]
    )
    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "top_driver_features": top.loc[:, ["feature", "label", "avg_abs_effect_size", "avg_signed_effect_size"]].to_dict(orient="records")
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    save_range_tables(tables)
    plot_range_bands()
    save_matched_soh_driver_table(tables)
    plot_driver_scores()
    plot_matched_soh_profiles(tables)
    write_report()


if __name__ == "__main__":
    main()
