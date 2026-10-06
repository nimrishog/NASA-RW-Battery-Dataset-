"""Build data-driven descriptive and dQ/dV figures used in the manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "modeling" / "event_features_rw9_rw11.csv"
SUMMARY = ROOT / "data" / "raw_extraction_summary" / "summary.json"
RESULTS = ROOT / "results" / "reviewer_validation"
OUT = ROOT / "results" / "manuscript_figures"
BATTERIES = ["RW9", "RW10", "RW11"]
COLORS = {"RW9": "#2f80ed", "RW10": "#f2994a", "RW11": "#27ae60"}
TEAL = "#1f9d8a"
ORANGE = "#f2994a"
GRID = "#d8e1e8"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)


def finish(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=340, bbox_inches="tight")
    plt.close(fig)


def style(axis: plt.Axes) -> None:
    axis.grid(True, color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)


def panel(axis: plt.Axes, letter: str) -> None:
    axis.text(0.01, 0.98, letter, transform=axis.transAxes, va="top", ha="left", weight="bold")


def load_events() -> pd.DataFrame:
    frame = pd.read_csv(EVENTS)
    return frame[frame["battery"].isin(BATTERIES)].sort_values(["battery", "event_order_chronological"])


def reference_blocks(events: pd.DataFrame) -> pd.DataFrame:
    return events.groupby(["battery", "assigned_checkpoint_cycle"], as_index=False).agg(
        capacity_ah=("capacity_ah", "first"),
        soh_percent=("soh_percent", "first"),
        label_charge_count=("label_charge_count", "max"),
        current_abs_mean_a=("current_abs_mean_a", "median"),
        voltage_pdf_std=("voltage_pdf_std", "median"),
        duration_s=("duration_s", "median"),
        dq_dv_max_ahpv=("dq_dv_max_ahpv", "median"),
    )


def plot_dataset_counts(events: pd.DataFrame) -> None:
    raw = json.loads(SUMMARY.read_text(encoding="utf-8"))["batteries"]
    counts = events.groupby(["battery", "event_type"]).size().unstack(fill_value=0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    axes[0].bar(BATTERIES, [raw[b]["sample_rows"] / 1e6 for b in BATTERIES], color="#2f80ed")
    axes[0].set_ylabel("Million rows")
    x = np.arange(3)
    width = 0.36
    axes[1].bar(x - width / 2, counts.loc[BATTERIES, "charge"], width, label="Charge", color=TEAL)
    axes[1].bar(x + width / 2, counts.loc[BATTERIES, "discharge"], width, label="Discharge", color=ORANGE)
    axes[1].set_xticks(x, BATTERIES)
    axes[1].set_ylabel("Events")
    axes[1].legend(frameon=False)
    axes[2].bar(BATTERIES, [raw[b]["reference_discharge_steps"] for b in BATTERIES], color=ORANGE)
    axes[2].set_ylabel("Reference-discharge tests")
    for axis, letter in zip(axes, "abc"):
        panel(axis, letter)
        style(axis)
    fig.tight_layout()
    finish(fig, "fig_dataset_counts_rw9_rw11.png")


def plot_capacity_and_features(blocks: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
    for battery in BATTERIES:
        part = blocks[blocks["battery"].eq(battery)]
        axes[0].plot(part["label_charge_count"], part["capacity_ah"], marker="o", ms=4, label=battery, color=COLORS[battery])
        axes[1].plot(part["label_charge_count"], part["soh_percent"], marker="o", ms=4, label=battery, color=COLORS[battery])
    axes[0].set_ylabel("Capacity (Ah)")
    axes[1].set_ylabel("SOH (%)")
    for axis, letter in zip(axes, "ab"):
        axis.set_xlabel("Charge random-walk count before reference test")
        axis.legend(frameon=False)
        panel(axis, letter)
        style(axis)
    fig.tight_layout()
    finish(fig, "fig_capacity_soh_rw9_rw11.png")

    specs = [
        ("current_abs_mean_a", "Block-median current magnitude", "A"),
        ("voltage_pdf_std", "Voltage spread", "V"),
        ("duration_s", "Event duration", "s"),
        ("dq_dv_max_ahpv", "Maximum local dQ/dV", "Ah/V"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for axis, (column, title, unit), letter in zip(axes.ravel(), specs, "abcd"):
        for battery in BATTERIES:
            part = blocks[blocks["battery"].eq(battery)]
            axis.plot(part["label_charge_count"], part[column], marker="o", ms=3, label=battery, color=COLORS[battery])
        axis.set_title(title)
        axis.set_xlabel("Reference-checkpoint charge-event count")
        axis.set_ylabel(unit)
        panel(axis, letter)
        style(axis)
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    finish(fig, "fig_engineered_feature_behavior_rw9_rw11.png")


def plot_operating_ranges(events: pd.DataFrame) -> None:
    labels, current, voltage, box_colors = [], [], [], []
    for battery in BATTERIES:
        for event_type, color in (("charge", TEAL), ("discharge", ORANGE)):
            part = events[(events["battery"].eq(battery)) & (events["event_type"].eq(event_type))]
            labels.append(f"{battery}\n{event_type}")
            current.append(part["current_mean_a"].dropna().to_numpy())
            voltage.append(part["voltage_delta_signed"].dropna().to_numpy())
            box_colors.append(color)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for axis, values, ylabel, letter in zip(axes, (current, voltage), ("Signed mean current (A)", "Voltage change (V)"), "ab"):
        boxes = axis.boxplot(values, patch_artist=True, tick_labels=labels, widths=0.55, showfliers=False)
        for patch, color in zip(boxes["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axis.axhline(0, color="#526d82", linewidth=1)
        axis.set_ylabel(ylabel)
        panel(axis, letter)
        style(axis)
    fig.tight_layout()
    finish(fig, "fig_current_voltage_ranges_rw9_rw11.png")


def plot_dq_dv_ablation() -> None:
    paired = pd.read_csv(RESULTS / "stage2_paired_dq_dv_seeds" / "paired_seed_effects.csv")
    loco = pd.read_csv(RESULTS / "dq_dv_loco_ablation" / "complete_vs_dq_dv_removed_loco_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    x = np.arange(len(paired))
    width = 0.36
    axes[0].bar(x - width / 2, paired["delta_event_rmse"], width, label="Event-window", color="#7faee8")
    axes[0].bar(x + width / 2, paired["delta_checkpoint_rmse"], width, label="Diagnostic-checkpoint", color="#2f80ed")
    axes[0].set_xticks(x, [f"Seed {seed}" for seed in paired["seed"]])
    axes[0].set_ylabel("RMSE difference: removed minus complete")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].legend(frameon=False)
    pivot = loco.pivot(index="test_battery", columns="feature_configuration", values="checkpoint_rmse").loc[BATTERIES]
    x = np.arange(len(pivot))
    axes[1].bar(x - width / 2, pivot["complete"], width, label="Complete", color="#7faee8")
    axes[1].bar(x + width / 2, pivot["dq_dv_removed"], width, label="Without dQ/dV", color="#2f80ed")
    axes[1].set_xticks(x, pivot.index)
    axes[1].set_ylabel("Diagnostic-checkpoint RMSE")
    axes[1].legend(frameon=False)
    for axis, letter in zip(axes, "ab"):
        panel(axis, letter)
        style(axis)
    fig.tight_layout()
    finish(fig, "fig_dq_dv_ablation_seeds_cells.png")


def plot_dq_dv_dependence() -> None:
    correlations = pd.read_csv(RESULTS / "stage6_dq_dv_redundancy" / "training_only_dq_dv_correlations.csv").nlargest(8, "abs_spearman").iloc[::-1]
    prediction = pd.read_csv(RESULTS / "stage6_dq_dv_redundancy" / "training_cell_cross_prediction_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(correlations["feature"], correlations["abs_spearman"], color=TEAL)
    axes[0].set_xlabel("Absolute Spearman association")
    groups = ["all_remaining", "loading_only", "voltage_only"]
    models = ["ridge", "hist_gradient_boosting"]
    x = np.arange(len(groups))
    width = 0.36
    for offset, model, color in ((-width / 2, models[0], "#7faee8"), (width / 2, models[1], "#2f80ed")):
        values = [float(prediction[(prediction["feature_group"].eq(group)) & (prediction["model"].eq(model))]["r2_mean"].iloc[0]) for group in groups]
        axes[1].bar(x + offset, values, width, label=model.replace("_", " "), color=color)
    axes[1].set_xticks(x, ["All remaining", "Loading only", "Voltage only"])
    axes[1].set_ylabel("Cross-cell dQ/dV prediction R2")
    axes[1].legend(frameon=False)
    for axis, letter in zip(axes, "ab"):
        panel(axis, letter)
        style(axis)
    fig.tight_layout()
    finish(fig, "fig_dq_dv_training_dependence.png")


def plot_shap_redistribution() -> None:
    table = pd.read_csv(RESULTS / "stage4_global_shap_redistribution" / "global_shap_redistribution_common_features.csv")
    table = table.sort_values("full_share", ascending=False).head(12).iloc[::-1]
    y = np.arange(len(table))
    height = 0.38
    fig, axis = plt.subplots(figsize=(11, 7.5))
    axis.barh(y - height / 2, table["full_share"], height, label="Complete", color="#7faee8")
    axis.barh(y + height / 2, table["ablated_share"], height, label="Without dQ/dV", color="#2f80ed")
    axis.set_yticks(y, table["feature"])
    axis.set_xlabel("Normalized mean absolute Gradient SHAP share")
    axis.legend(frameon=False)
    style(axis)
    fig.tight_layout()
    finish(fig, "fig_dq_dv_shap_redistribution.png")


def main() -> None:
    events = load_events()
    blocks = reference_blocks(events)
    plot_dataset_counts(events)
    plot_capacity_and_features(blocks)
    plot_operating_ranges(events)
    plot_dq_dv_ablation()
    plot_dq_dv_dependence()
    plot_shap_redistribution()
    print(f"Wrote manuscript figures to {OUT}")


if __name__ == "__main__":
    main()
