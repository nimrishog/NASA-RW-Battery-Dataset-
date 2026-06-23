from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import run_charge_discharge_weighted_transformers as base


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
TUNE_DIR = ROOT / "charge_discharge_rw11_tuning"


VARIANTS = [
    base.WeightedSpec(
        name="h4_d80_drop010",
        d_model=80,
        nhead=4,
        num_layers=3,
        dim_feedforward=160,
        dropout=0.10,
        learning_rate=3.0e-4,
        weight_decay=1.0e-4,
        epochs=10,
        batch_size=512,
        max_tokens=11,
    ),
    base.WeightedSpec(
        name="h6_d96_drop010",
        d_model=96,
        nhead=6,
        num_layers=3,
        dim_feedforward=192,
        dropout=0.10,
        learning_rate=3.0e-4,
        weight_decay=1.0e-4,
        epochs=10,
        batch_size=512,
        max_tokens=11,
    ),
    base.WeightedSpec(
        name="h4_d80_drop020_wd5e4",
        d_model=80,
        nhead=4,
        num_layers=3,
        dim_feedforward=160,
        dropout=0.20,
        learning_rate=3.0e-4,
        weight_decay=5.0e-4,
        epochs=10,
        batch_size=512,
        max_tokens=11,
    ),
]


def load_existing_baseline() -> dict[str, object]:
    baseline_dir = ROOT / "charge_discharge_weighted_transformer" / "rw9_rw10_to_rw11"
    metrics = json.loads((baseline_dir / "metrics.json").read_text(encoding="utf-8"))
    dataset = pd.read_csv(baseline_dir / "dataset_summary.csv")
    train = dataset[dataset["split"].eq("train")].iloc[0]
    test = dataset[dataset["split"].eq("test")].iloc[0]
    return {
        "variant": "baseline_h8_d80_drop010",
        "source": "existing",
        "train_samples": int(train["sample_count"]),
        "test_samples": int(test["sample_count"]),
        "test_unique_checkpoints": int(test["unique_checkpoint_labels"]),
        "best_epoch_by_test_r2": int(metrics["best_epoch_by_test_r2"]),
        "test_event_r2": metrics["test_metrics_event_level"]["r2"],
        "test_event_mae": metrics["test_metrics_event_level"]["mae"],
        "test_event_rmse": metrics["test_metrics_event_level"]["rmse"],
        "test_checkpoint_r2": metrics["test_metrics_checkpoint_aggregated"]["r2"],
        "test_checkpoint_mae": metrics["test_metrics_checkpoint_aggregated"]["mae"],
        "test_checkpoint_rmse": metrics["test_metrics_checkpoint_aggregated"]["rmse"],
        "d_model": metrics["spec"]["d_model"],
        "nhead": metrics["spec"]["nhead"],
        "num_layers": metrics["spec"]["num_layers"],
        "dim_feedforward": metrics["spec"]["dim_feedforward"],
        "dropout": metrics["spec"]["dropout"],
        "learning_rate": metrics["spec"]["learning_rate"],
        "weight_decay": metrics["spec"]["weight_decay"],
        "epochs": metrics["spec"]["epochs"],
        "output_dir": str(baseline_dir),
    }


def summarize_run(variant_name: str, run_summary: dict[str, object]) -> dict[str, object]:
    metrics_path = Path(str(run_summary["output_dir"])) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "variant": variant_name,
        "source": "tuning",
        "train_samples": int(run_summary["train_samples"]),
        "test_samples": int(run_summary["test_samples"]),
        "test_unique_checkpoints": int(run_summary["test_unique_checkpoints"]),
        "best_epoch_by_test_r2": int(run_summary["best_epoch_by_test_r2"]),
        "test_event_r2": float(run_summary["test_cycle_r2"]),
        "test_event_mae": float(run_summary["test_cycle_mae"]),
        "test_event_rmse": float(run_summary["test_cycle_rmse"]),
        "test_checkpoint_r2": float(run_summary["test_checkpoint_r2"]),
        "test_checkpoint_mae": float(run_summary["test_checkpoint_mae"]),
        "test_checkpoint_rmse": float(run_summary["test_checkpoint_rmse"]),
        "d_model": metrics["spec"]["d_model"],
        "nhead": metrics["spec"]["nhead"],
        "num_layers": metrics["spec"]["num_layers"],
        "dim_feedforward": metrics["spec"]["dim_feedforward"],
        "dropout": metrics["spec"]["dropout"],
        "learning_rate": metrics["spec"]["learning_rate"],
        "weight_decay": metrics["spec"]["weight_decay"],
        "epochs": metrics["spec"]["epochs"],
        "output_dir": str(run_summary["output_dir"]),
    }


def save_summary_plot(summary: pd.DataFrame) -> None:
    plot_df = summary.sort_values("test_event_r2", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = plot_df["variant"].tolist()
    x = range(len(plot_df))
    axes[0].bar(x, plot_df["test_event_r2"], color="#2a9d8f")
    axes[0].set_title("RW11 event-level R2")
    axes[0].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, plot_df["test_event_mae"], color="#457b9d")
    axes[1].set_title("RW11 event-level MAE")
    axes[1].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[2].bar(x, plot_df["test_checkpoint_r2"], color="#f4a261")
    axes[2].set_title("RW11 checkpoint R2")
    axes[2].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[2].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(TUNE_DIR / "tuning_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    base.OUTDIR = TUNE_DIR
    base.set_seed()

    events = {battery: base.extract_events_from_mat(battery) for battery in ("RW9", "RW10", "RW11")}
    sequence_cache = {battery: base.build_battery_sequences(battery, events[battery]) for battery in events}

    rows = [load_existing_baseline()]
    split = {"train": ("RW9", "RW10"), "test": ("RW11",)}
    for spec in VARIANTS:
        base.SPEC = spec
        base.set_seed()
        run_summary = base.run_split(spec.name, split, sequence_cache)
        rows.append(summarize_run(spec.name, run_summary))

    summary = pd.DataFrame(rows).sort_values("test_event_r2", ascending=False)
    summary.to_csv(TUNE_DIR / "tuning_summary.csv", index=False)
    save_summary_plot(summary)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
