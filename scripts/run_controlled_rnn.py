from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import run_transformer_loco as pipeline


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_validation" / "controlled_rnn"


class LazyWindowDataset(Dataset):
    def __init__(self, arrays, battery_ids, ends, targets, weights):
        self.arrays = arrays
        self.battery_ids = battery_ids.astype(np.int8, copy=False)
        self.ends = ends.astype(np.int32, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.weights = weights.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        battery = int(self.battery_ids[index])
        end = int(self.ends[index])
        start = end - pipeline.CFG.window + 1
        return (
            torch.from_numpy(self.arrays[battery][start : end + 1]),
            torch.tensor(self.targets[index]),
            torch.tensor(self.weights[index]),
        )


def prepare_split(events: pd.DataFrame):
    names = ["RW9", "RW10", "RW11"]
    blocks = [
        events[events["battery"].eq(name)]
        .sort_values("event_order_chronological")
        .reset_index(drop=True)
        for name in names
    ]
    raw = [block[pipeline.FEATURES].to_numpy(dtype=np.float32) for block in blocks]
    ends = [np.arange(pipeline.CFG.window - 1, len(block), dtype=np.int32) for block in blocks]
    targets = [block["soh_percent"].to_numpy(dtype=np.float32)[end] for block, end in zip(blocks, ends)]
    weights = [block["sample_weight"].to_numpy(dtype=np.float32)[end] for block, end in zip(blocks, ends)]
    validation_masks = [
        pipeline.stratified_val_mask(targets[index], pipeline.CFG.val_frac, pipeline.CFG.seed + index)
        for index in range(2)
    ]

    low = np.empty(len(pipeline.FEATURES), dtype=np.float32)
    high = np.empty(len(pipeline.FEATURES), dtype=np.float32)
    mean = np.empty(len(pipeline.FEATURES), dtype=np.float32)
    std = np.empty(len(pipeline.FEATURES), dtype=np.float32)
    for feature_index in range(len(pipeline.FEATURES)):
        training_values = []
        for battery_index in range(2):
            view = np.lib.stride_tricks.sliding_window_view(
                raw[battery_index][:, feature_index], pipeline.CFG.window
            )
            training_values.append(view[~validation_masks[battery_index]].reshape(-1))
        values = np.concatenate(training_values).astype(np.float32, copy=False)
        low[feature_index] = np.quantile(values, pipeline.CFG.clip_quantile)
        high[feature_index] = np.quantile(values, 1.0 - pipeline.CFG.clip_quantile)
        clipped = np.clip(values, low[feature_index], high[feature_index])
        mean[feature_index] = clipped.mean()
        std[feature_index] = max(float(clipped.std()), 1.0e-8)
    normalized = [((np.clip(array, low, high) - mean) / std).astype(np.float32) for array in raw]
    training_targets = np.concatenate(
        [targets[index][~validation_masks[index]] for index in range(2)]
    ).astype(np.float32)
    target_mean = float(training_targets.mean())
    target_std = max(float(training_targets.std()), 1.0e-8)

    def combine(indices, masks=None, validation=False):
        battery_ids, selected_ends, selected_targets, selected_weights = [], [], [], []
        for position, battery_index in enumerate(indices):
            mask = np.ones(len(ends[battery_index]), dtype=bool)
            if masks is not None:
                mask = masks[position] if validation else ~masks[position]
            battery_ids.append(np.full(mask.sum(), battery_index, dtype=np.int8))
            selected_ends.append(ends[battery_index][mask])
            selected_targets.append(((targets[battery_index][mask] - target_mean) / target_std).astype(np.float32))
            selected_weights.append(
                np.ones(mask.sum(), dtype=np.float32)
                if validation or battery_index == 2
                else weights[battery_index][mask]
            )
        return LazyWindowDataset(
            normalized,
            np.concatenate(battery_ids),
            np.concatenate(selected_ends),
            np.concatenate(selected_targets),
            np.concatenate(selected_weights),
        )

    split = {
        "train_dataset": combine([0, 1], validation_masks),
        "val_dataset": combine([0, 1], validation_masks, validation=True),
        "test_dataset": combine([2]),
        "y_mean": target_mean,
        "y_std": target_std,
    }
    return split, blocks[2].iloc[ends[2]].reset_index(drop=True)


def make_comparison_figure(rnn_row: dict[str, object]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_path = ROOT / "results" / "reviewer_validation" / "trend_baselines" / "trend_baseline_metrics.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Missing {baseline_path}. Run `python scripts/run_trend_baselines.py` before this script."
        )
    baseline = pd.read_csv(baseline_path).set_index("model")
    baseline_order = ["charge_count_ridge", "elapsed_time_ridge", "throughput_ridge"]
    labels = ["Charge-count\nridge", "Elapsed-time\nridge", "Throughput\nridge", "Vanilla RNN", "Transformer"]
    event_values = [float(baseline.loc[name, "event_rmse"]) for name in baseline_order]
    checkpoint_values = [float(baseline.loc[name, "checkpoint_rmse"]) for name in baseline_order]
    event_values.extend([float(rnn_row["event_rmse"]), 3.0388083069706244])
    checkpoint_values.extend([float(rnn_row["checkpoint_rmse"]), 2.535021103840701])
    colors = ["#2f80ed"] * len(labels)
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    event_bars = ax.bar(x - width / 2, event_values, width, color=colors, alpha=0.60, edgecolor="#263238", linewidth=0.8, label="Event-window")
    checkpoint_bars = ax.bar(x + width / 2, checkpoint_values, width, color=colors, edgecolor="#263238", linewidth=0.8, label="Diagnostic-checkpoint")
    upper = max(event_values + checkpoint_values) * 1.18
    ax.set_ylim(0, upper)
    for bars, values in ((event_bars, event_values), (checkpoint_bars, checkpoint_values)):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + upper * 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE (SOH percentage points)")
    ax.set_title("Controlled Baseline Comparison on Held-Out RW11", weight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_controlled_neural_baseline_rmse.png", dpi=300)
    manuscript_figures = ROOT / "results" / "manuscript_figures"
    manuscript_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(manuscript_figures / "fig_controlled_neural_baseline_rmse.png", dpi=300)
    plt.close(fig)


class EventRNN(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_features, 64), nn.LayerNorm(64))
        self.rnn = nn.RNN(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            nonlinearity="tanh",
            batch_first=True,
            dropout=0.10,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.rnn(self.input(x))
        return self.head(encoded[:, -1]).squeeze(-1)


def main() -> None:
    pipeline.set_seed(pipeline.CFG.seed)
    events = pipeline.load_events()
    split, meta_test = prepare_split(events)
    OUT.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(split["train_dataset"], batch_size=pipeline.CFG.batch_size, shuffle=True)
    val_loader = DataLoader(split["val_dataset"], batch_size=pipeline.CFG.batch_size, shuffle=False)
    test_loader = DataLoader(split["test_dataset"], batch_size=pipeline.CFG.batch_size, shuffle=False)

    model = EventRNN(len(pipeline.FEATURES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=pipeline.CFG.learning_rate, weight_decay=pipeline.CFG.weight_decay)
    total_steps = pipeline.CFG.epochs * max(1, len(train_loader))
    warmup = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best = {"val_r2": -1.0e9, "epoch": -1}
    best_state = None
    history = []
    for epoch in range(1, pipeline.CFG.epochs + 1):
        train_metrics, _, _ = pipeline.run_epoch(
            model, train_loader, optimizer, scheduler, split["y_mean"], split["y_std"]
        )
        with torch.no_grad():
            val_metrics, _, _ = pipeline.run_epoch(
                model, val_loader, None, None, split["y_mean"], split["y_std"]
            )
        history.append(
            {
                "epoch": epoch,
                "train_r2": train_metrics["r2"],
                "val_r2": val_metrics["r2"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
            }
        )
        pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
        print(
            f"epoch {epoch:02d}/{pipeline.CFG.epochs} train_r2={train_metrics['r2']:.4f} "
            f"val_r2={val_metrics['r2']:.4f} val_rmse={val_metrics['rmse']:.3f}",
            flush=True,
        )
        if val_metrics["r2"] > best["val_r2"]:
            best = {"val_r2": val_metrics["r2"], "val_rmse": val_metrics["rmse"], "epoch": epoch}
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No RNN validation checkpoint was produced")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "best": best}, OUT / "best_model.pt")

    with torch.no_grad():
        test_metrics, test_pred, test_true = pipeline.run_epoch(
            model, test_loader, None, None, split["y_mean"], split["y_std"]
        )
    prediction = meta_test[["battery", "event_order_chronological", "assigned_checkpoint_cycle"]].copy()
    prediction["y_true_soh"] = test_true
    prediction["y_pred_soh"] = test_pred
    prediction["abs_error"] = np.abs(test_pred - test_true)
    prediction.to_csv(OUT / "test_predictions.csv", index=False)
    checkpoint = pipeline.aggregate_by_checkpoint(
        meta_test, test_true, test_pred, OUT / "test_checkpoint_predictions.csv"
    )
    row = {
        "experiment": "controlled_simple_rnn_baseline_rw11",
        "model": "event_window_vanilla_rnn",
        "train_batteries": "RW9+RW10",
        "test_battery": "RW11",
        "window": pipeline.CFG.window,
        "test_windows": int(len(test_true)),
        "n_features": len(pipeline.FEATURES),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "hidden_size": 128,
        "layers": 2,
        "bidirectional": False,
        "best_val_epoch": int(best["epoch"]),
        **{f"event_{key}": value for key, value in test_metrics.items()},
        **{f"checkpoint_{key}": value for key, value in checkpoint.items()},
    }
    (OUT / "metrics.json").write_text(
        json.dumps(
            {
                "pipeline_config": pipeline.asdict(pipeline.CFG),
                "rnn_config": {"input_projection": 64, "hidden_size": 128, "layers": 2, "dropout": 0.10},
                "features": pipeline.FEATURES,
                "row": row,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame([row]).to_csv(OUT / "controlled_simple_rnn_rw11_metrics.csv", index=False)
    make_comparison_figure(row)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
