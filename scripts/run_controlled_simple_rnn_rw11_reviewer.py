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
from torch.utils.data import DataLoader

import run_controlled_bilstm_rw11_reviewer as controlled
import run_full_transformer_loco_reviewer as pipeline


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reviewer_controlled_simple_rnn_rw11"


def make_comparison_figure(rnn_row: dict[str, object]) -> None:
    labels = ["Charge-count\nridge", "Elapsed-time\nridge", "Throughput\nridge", "Vanilla RNN", "Transformer"]
    event_values = [6.3239388179, 6.4443129346, 5.8497431669, float(rnn_row["event_rmse"]), 3.0388083070]
    checkpoint_values = [6.3991515883, 6.5508148138, 6.0601048653, float(rnn_row["checkpoint_rmse"]), 2.5350211038]
    colors = ["#8da0cb", "#8da0cb", "#8da0cb", "#a6d854", "#2f80ed"]
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
    split, meta_test = controlled.prepare_lazy_split(events)
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
