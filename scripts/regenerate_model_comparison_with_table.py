from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
SUITE_DIR = ROOT / "charge_discharge_full_cycle_suite"
SUMMARY_CSV = SUITE_DIR / "combined_charge_vs_charge_discharge_summary.csv"
OUT_PNG = SUITE_DIR / "combined_charge_vs_charge_discharge_summary_with_table.png"
OUT_CSV = SUITE_DIR / "combined_charge_vs_charge_discharge_summary_table.csv"


MODEL_ORDER = [
    "patch-only transformer",
    "pure full-cycle token transformer",
    "feature-token-only pure transformer",
    "hybrid transformer",
    "LightGBM",
]
MODEL_LABELS = {
    "patch-only transformer": "Patch-only\ntransformer",
    "pure full-cycle token transformer": "Pure full-cycle\ntoken transformer",
    "feature-token-only pure transformer": "Feature-token\npure transformer",
    "hybrid transformer": "Hybrid\ntransformer",
    "LightGBM": "LightGBM",
}
SCOPE_ORDER = ["charge only", "charge + discharge"]
SCOPE_COLORS = {"charge only": "#2a9d8f", "charge + discharge": "#e76f51"}


def get_metric(summary: pd.DataFrame, model: str, scope: str, metric: str) -> float:
    row = summary[(summary["model"].eq(model)) & (summary["input_scope"].eq(scope))]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][metric])


def add_bar_labels(ax: plt.Axes, bars, metric: str) -> None:
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        label = f"{height:.3f}" if "r2" in metric.lower() else f"{height:.2f}"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def build_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        charge = summary[(summary["model"].eq(model)) & (summary["input_scope"].eq("charge only"))]
        cd = summary[(summary["model"].eq(model)) & (summary["input_scope"].eq("charge + discharge"))]
        charge = charge.iloc[0] if not charge.empty else None
        cd = cd.iloc[0] if not cd.empty else None
        rows.append(
            {
                "Model": MODEL_LABELS[model].replace("\n", " "),
                "Charge R2": "" if charge is None else f"{float(charge['test_r2']):.3f}",
                "Charge MAE": "" if charge is None else f"{float(charge['test_mae']):.2f}",
                "Charge+Disch R2": "" if cd is None else f"{float(cd['test_r2']):.3f}",
                "Charge+Disch MAE": "" if cd is None else f"{float(cd['test_mae']):.2f}",
                "Checkpoint R2 CD": "" if cd is None else f"{float(cd['checkpoint_r2']):.3f}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    table_df = build_table(summary)
    table_df.to_csv(OUT_CSV, index=False)

    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[3.1, 1.45], hspace=0.46, wspace=0.26)
    metrics = [
        ("test_r2", "Event-level R²", (0.84, 0.985)),
        ("test_mae", "Event-level MAE", (2.35, 4.05)),
        ("checkpoint_r2", "Checkpoint R²", (0.84, 0.985)),
    ]
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    handles = []
    labels = []
    for ax_idx, (metric, title, ylim) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, ax_idx])
        for scope_idx, scope in enumerate(SCOPE_ORDER):
            values = [get_metric(summary, model, scope, metric) for model in MODEL_ORDER]
            bars = ax.bar(
                x + (scope_idx - 0.5) * width,
                values,
                width=width,
                color=SCOPE_COLORS[scope],
                label=scope,
                edgecolor="white",
                linewidth=0.8,
            )
            add_bar_labels(ax, bars, metric)
            if ax_idx == 0:
                handles.append(bars[0])
                labels.append(scope)
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=0, ha="center", fontsize=9)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig.suptitle("Charge-only vs Charge+Discharge Model Performance", fontsize=18, weight="bold", y=0.995)

    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.02, 0.05, 0.96, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#172b4d")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "#eef2f7")
    ax_table.set_title("Metrics table", fontsize=13, weight="bold", pad=8)

    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
