from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
RESULT_DIR = BASE / "cycle_level_weighted_transformer_rw9_rw10_to_rw11"


def save_training_dashboard(history: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    axes[0].plot(history["epoch"], history["train_loss"], label="train", linewidth=2)
    axes[0].plot(history["epoch"], history["test_loss"], label="test", linewidth=2)
    axes[0].set_title("Weighted Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_mae"], label="train", linewidth=2)
    axes[1].plot(history["epoch"], history["test_mae"], label="test", linewidth=2)
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("SOH pp")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(history["epoch"], history["train_r2"], label="train", linewidth=2)
    axes[2].plot(history["epoch"], history["test_r2"], label="test", linewidth=2)
    axes[2].set_title("R2")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R2")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.savefig(outdir / "training_dashboard.png", dpi=180)
    plt.close(fig)


def save_cycle_test_plots(test_df: pd.DataFrame, outdir: Path) -> None:
    plot_df = test_df.sort_values("charge_cycle").reset_index(drop=True)
    y_true = plot_df["actual_soh_percent"].to_numpy(dtype=np.float64)
    y_pred = plot_df["predicted_soh_percent"].to_numpy(dtype=np.float64)
    residual = y_pred - y_true
    abs_err = np.abs(residual)
    sample_idx = np.linspace(0, len(plot_df) - 1, min(len(plot_df), 6000)).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(plot_df["charge_cycle"], y_true, linewidth=1.5, label="actual")
    axes[0, 0].plot(plot_df["charge_cycle"], y_pred, linewidth=1.2, label="predicted")
    axes[0, 0].set_title("RW11 Test SOH Over Charge Cycles")
    axes[0, 0].set_xlabel("Charge cycle")
    axes[0, 0].set_ylabel("SOH (%)")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    hb = axes[0, 1].hexbin(y_true, y_pred, gridsize=50, cmap="YlOrRd", mincnt=1)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    axes[0, 1].plot([lo, hi], [lo, hi], color="black", linewidth=1)
    axes[0, 1].set_title("RW11 Actual vs Predicted SOH")
    axes[0, 1].set_xlabel("Actual SOH (%)")
    axes[0, 1].set_ylabel("Predicted SOH (%)")
    fig.colorbar(hb, ax=axes[0, 1], label="cycle count")

    axes[1, 0].scatter(plot_df["charge_cycle"].to_numpy()[sample_idx], residual[sample_idx], s=10, alpha=0.25)
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    rolling = pd.Series(residual).rolling(window=400, center=True, min_periods=50).mean()
    axes[1, 0].plot(plot_df["charge_cycle"], rolling, color="#d62828", linewidth=2, label="rolling mean residual")
    axes[1, 0].set_title("RW11 Residual Over Charge Cycles")
    axes[1, 0].set_xlabel("Charge cycle")
    axes[1, 0].set_ylabel("Predicted - Actual (pp)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].hist(abs_err, bins=60, color="#457b9d", alpha=0.85, density=True)
    axes[1, 1].set_title("RW11 Absolute Error Distribution")
    axes[1, 1].set_xlabel("Absolute error (pp)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].grid(alpha=0.25)

    fig.savefig(outdir / "test_cycle_report.png", dpi=180)
    plt.close(fig)


def save_checkpoint_plots(cp_df: pd.DataFrame, outdir: Path) -> None:
    plot_df = cp_df.sort_values("assigned_checkpoint_cycle").reset_index(drop=True)
    plot_df["error_pp"] = plot_df["predicted_soh_percent"] - plot_df["actual_soh_percent"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(plot_df["assigned_checkpoint_cycle"], plot_df["actual_soh_percent"], marker="o", linewidth=1.8, label="actual")
    axes[0].plot(plot_df["assigned_checkpoint_cycle"], plot_df["predicted_soh_percent"], marker="o", linewidth=1.8, label="predicted")
    axes[0].set_title("RW11 Checkpoint-Aggregated SOH")
    axes[0].set_xlabel("Assigned benchmark checkpoint cycle")
    axes[0].set_ylabel("SOH (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    colors = np.where(plot_df["error_pp"] >= 0.0, "#2a9d8f", "#e76f51")
    axes[1].bar(plot_df["assigned_checkpoint_cycle"].astype(str), plot_df["error_pp"], color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("RW11 Checkpoint Error")
    axes[1].set_xlabel("Assigned benchmark checkpoint cycle")
    axes[1].set_ylabel("Predicted - Actual (pp)")
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].grid(axis="y", alpha=0.25)

    fig.savefig(outdir / "test_checkpoint_report.png", dpi=180)
    plt.close(fig)


def save_error_cdf(test_df: pd.DataFrame, outdir: Path) -> None:
    abs_err = np.sort(test_df["absolute_error_pp"].to_numpy(dtype=np.float64))
    cdf = np.arange(1, len(abs_err) + 1, dtype=np.float64) / len(abs_err)
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    ax.plot(abs_err, cdf, linewidth=2)
    ax.set_title("RW11 Cycle-Level Absolute Error CDF")
    ax.set_xlabel("Absolute error (pp)")
    ax.set_ylabel("Cumulative fraction")
    ax.grid(alpha=0.25)
    fig.savefig(outdir / "test_error_cdf.png", dpi=180)
    plt.close(fig)


def save_architecture_diagram(metrics: dict, outdir: Path) -> None:
    spec = metrics["spec"]
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str = "#edf6f9", ec: str = "#1d3557", fs: int = 10) -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", linewidth=1.5, edgecolor=ec, facecolor=fc)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    box(0.5, 6.7, 2.5, 1.5, "Input cycle\nup to 11 tokens\n30 rows per token")
    box(0.5, 4.6, 2.5, 1.3, "Token feature vector\n26 dims\nraw + normalized")
    box(3.5, 6.1, 2.6, 1.7, "Linear projection\n26 -> 80\n+ LayerNorm")
    box(3.5, 4.3, 2.6, 1.3, "Add [CLS]\n+ positional encoding")

    box(7.0, 6.1, 3.0, 1.8, "Encoder layer x 3", fc="#fff3b0", ec="#6d597a", fs=11)
    box(7.35, 6.95, 2.3, 0.55, "Multi-head self-attention", fc="#fde68a", ec="#6d597a", fs=9)
    box(7.35, 6.25, 2.3, 0.45, "8 heads, head dim = 10", fc="#fde68a", ec="#6d597a", fs=9)
    box(7.35, 5.55, 2.3, 0.45, "Add & Norm", fc="#fde68a", ec="#6d597a", fs=9)
    box(7.35, 4.90, 2.3, 0.55, "Feed Forward\n80 -> 160 -> 80", fc="#fde68a", ec="#6d597a", fs=9)
    box(7.35, 4.20, 2.3, 0.45, "Add & Norm", fc="#fde68a", ec="#6d597a", fs=9)

    box(11.0, 6.2, 2.1, 1.5, "[CLS] output\npooling", fc="#e9f5db")
    box(13.6, 6.0, 1.9, 1.8, "MLP head\n80 -> 80 -> 1\nSOH (%)", fc="#e9f5db")
    box(11.0, 3.9, 4.5, 1.5, "Training loss: weighted MSE\nsample weight = 1 / benchmark block size", fc="#ffe5d9", ec="#9c6644")
    box(11.0, 1.8, 4.5, 1.5, "Train: RW9 + RW10\nTest: RW11\nBatch size 512, epochs 10\ndropout 0.10, lr 3e-4", fc="#dde5f2", ec="#3d5a80")

    arrows = [
        ((3.0, 7.45), (3.5, 7.45)),
        ((3.0, 5.25), (3.5, 5.25)),
        ((6.1, 7.0), (7.0, 7.0)),
        ((6.1, 5.0), (7.0, 5.0)),
        ((10.0, 7.0), (11.0, 7.0)),
        ((13.1, 7.0), (13.6, 7.0)),
        ((14.55, 6.0), (14.55, 5.4)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#457b9d"))

    ax.text(0.5, 0.55, "Customized from the encoder-side layout style in 'Attention Is All You Need': battery cycle tokens replace word tokens, and the output is cycle SOH.", fontsize=10)
    fig.savefig(outdir / "architecture_diagram.png", dpi=180)
    plt.close(fig)


def main() -> None:
    metrics = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
    history = pd.read_csv(RESULT_DIR / "training_history.csv")
    test_df = pd.read_csv(RESULT_DIR / "test_cycle_predictions.csv")
    cp_df = pd.read_csv(RESULT_DIR / "test_checkpoint_predictions.csv")

    cp_df["absolute_error_pp"] = (cp_df["predicted_soh_percent"] - cp_df["actual_soh_percent"]).abs()
    test_df["signed_error_pp"] = test_df["predicted_soh_percent"] - test_df["actual_soh_percent"]

    save_training_dashboard(history, RESULT_DIR)
    save_cycle_test_plots(test_df, RESULT_DIR)
    save_checkpoint_plots(cp_df, RESULT_DIR)
    save_error_cdf(test_df, RESULT_DIR)
    save_architecture_diagram(metrics, RESULT_DIR)

    worst_cycles = test_df.sort_values("absolute_error_pp", ascending=False).head(200).reset_index(drop=True)
    worst_cycles.to_csv(RESULT_DIR / "worst_test_cycles.csv", index=False)
    cp_df.sort_values("absolute_error_pp", ascending=False).to_csv(RESULT_DIR / "checkpoint_error_table.csv", index=False)

    report_lines = [
        "# RW9+RW10 -> RW11 Weighted Cycle-Level Transformer Report",
        "",
        "## Model",
        f"- d_model: {metrics['spec']['d_model']}",
        f"- heads: {metrics['spec']['nhead']}",
        f"- encoder layers: {metrics['spec']['num_layers']}",
        f"- feedforward dim: {metrics['spec']['dim_feedforward']}",
        f"- dropout: {metrics['spec']['dropout']}",
        f"- batch size: {metrics['spec']['batch_size']}",
        f"- epochs: {metrics['spec']['epochs']}",
        f"- parameters: {metrics['parameter_count']}",
        "",
        "## Test Metrics",
        f"- cycle-level R2: {metrics['test_metrics_cycle_level']['r2']:.4f}",
        f"- cycle-level MAE: {metrics['test_metrics_cycle_level']['mae']:.4f}",
        f"- cycle-level RMSE: {metrics['test_metrics_cycle_level']['rmse']:.4f}",
        f"- checkpoint R2: {metrics['test_metrics_checkpoint_aggregated']['r2']:.4f}",
        f"- checkpoint MAE: {metrics['test_metrics_checkpoint_aggregated']['mae']:.4f}",
        f"- checkpoint RMSE: {metrics['test_metrics_checkpoint_aggregated']['rmse']:.4f}",
    ]
    (RESULT_DIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
