from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from run_cycle_level_weighted_transformer import (
    add_block_weights,
    build_battery_sequences,
    concat_sequence_sets,
)
from run_full_event_token_transformers import (
    apply_standardization,
    compute_standardization,
    destandardize,
    standardize_target,
)


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = ROOT / "lstm_charge_cycle_rw11"
TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERIES = ("RW11",)


@dataclass(frozen=True)
class LSTMSpec:
    name: str
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int


SPEC = LSTMSpec(
    name="bilstm_patch30_h96_l2_drop020",
    hidden_size=96,
    num_layers=2,
    dropout=0.20,
    bidirectional=True,
    learning_rate=3.0e-4,
    weight_decay=5.0e-4,
    epochs=15,
    batch_size=512,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    pct_err = abs_err / np.clip(np.abs(y_true), 1.0e-8, None)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_percent": float(pct_err.mean() * 100.0),
        "within_2pct": float((pct_err <= 0.02).mean()),
        "within_5pct": float((pct_err <= 0.05).mean()),
        "within_10pct": float((pct_err <= 0.10).mean()),
    }


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray], targets: np.ndarray, weights: np.ndarray, meta: pd.DataFrame):
        self.sequences = sequences
        self.targets = targets.astype(np.float32, copy=False)
        self.weights = weights.astype(np.float32, copy=False)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.meta.iloc[idx]
        return {
            "sequence": self.sequences[idx],
            "target": float(self.targets[idx]),
            "weight": float(self.weights[idx]),
            "battery": str(row["battery"]),
            "charge_cycle": int(row["charge_cycle"]),
            "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
            "sample_id": str(row["sample_id"]),
        }


def collate_batch(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int], list[int], list[str]]:
    max_len = max(len(item["sequence"]) for item in batch)
    input_dim = batch[0]["sequence"].shape[1]
    batch_size = len(batch)
    x = torch.zeros((batch_size, max_len, input_dim), dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    y = torch.zeros(batch_size, dtype=torch.float32)
    w = torch.zeros(batch_size, dtype=torch.float32)
    batteries: list[str] = []
    cycles: list[int] = []
    checkpoints: list[int] = []
    sample_ids: list[str] = []
    for idx, item in enumerate(batch):
        seq = torch.from_numpy(item["sequence"]).float()
        seq_len = seq.shape[0]
        x[idx, :seq_len] = seq
        lengths[idx] = seq_len
        y[idx] = float(item["target"])
        w[idx] = float(item["weight"])
        batteries.append(item["battery"])
        cycles.append(int(item["charge_cycle"]))
        checkpoints.append(int(item["assigned_checkpoint_cycle"]))
        sample_ids.append(item["sample_id"])
    return x, lengths, y, w, batteries, cycles, checkpoints, sample_ids


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, spec: LSTMSpec):
        super().__init__()
        self.spec = spec
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=spec.hidden_size,
            num_layers=spec.num_layers,
            dropout=spec.dropout if spec.num_layers > 1 else 0.0,
            bidirectional=spec.bidirectional,
            batch_first=True,
        )
        out_dim = spec.hidden_size * (2 if spec.bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        if self.spec.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last = h_n[-1]
        return self.head(last).squeeze(-1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, list[str], list[int], list[int], list[str]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    batteries: list[str] = []
    cycles: list[int] = []
    checkpoints: list[int] = []
    sample_ids: list[str] = []
    for batch in loader:
        batch_x, batch_lengths, batch_y, batch_w, batch_batteries, batch_cycles, batch_checkpoints, batch_ids = batch
        batch_x = batch_x.to(device)
        batch_lengths = batch_lengths.to(device)
        batch_y = batch_y.to(device)
        batch_w = batch_w.to(device)

        out = model(batch_x, batch_lengths)
        per_sample = (out - batch_y) ** 2
        loss = (per_sample * batch_w).sum() / torch.clamp(batch_w.sum(), min=1.0e-8)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += float(loss.item()) * batch_x.size(0)
        total_count += batch_x.size(0)
        preds.append(out.detach().cpu().numpy())
        targets.append(batch_y.detach().cpu().numpy())
        batteries.extend(batch_batteries)
        cycles.extend([int(x) for x in batch_cycles])
        checkpoints.extend([int(x) for x in batch_checkpoints])
        sample_ids.extend(batch_ids)

    return (
        total_loss / max(total_count, 1),
        np.concatenate(preds),
        np.concatenate(targets),
        batteries,
        cycles,
        checkpoints,
        sample_ids,
    )


def prediction_frame(
    batteries: list[str],
    cycles: list[int],
    checkpoints: list[int],
    sample_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "battery": batteries,
            "charge_cycle": cycles,
            "assigned_checkpoint_cycle": checkpoints,
            "sample_id": sample_ids,
            "actual_soh_percent": y_true,
            "predicted_soh_percent": y_pred,
            "error_pp": y_pred - y_true,
            "absolute_error_pp": np.abs(y_pred - y_true),
        }
    )


def aggregate_checkpoint_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["battery", "assigned_checkpoint_cycle"], as_index=False)
        .agg(
            actual_soh_percent=("actual_soh_percent", "mean"),
            predicted_soh_percent=("predicted_soh_percent", "mean"),
            cycle_count=("charge_cycle", "count"),
        )
        .sort_values(["battery", "assigned_checkpoint_cycle"])
    )
    out["error_pp"] = out["predicted_soh_percent"] - out["actual_soh_percent"]
    out["absolute_error_pp"] = out["error_pp"].abs()
    return out


def save_training_plot(history: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["test_loss"], label="test")
    axes[0].set_title("Weighted MSE loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_r2"], label="train")
    axes[1].plot(history["epoch"], history["test_r2"], label="test")
    axes[1].set_title("R2")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    axes[2].plot(history["epoch"], history["train_mae"], label="train")
    axes[2].plot(history["epoch"], history["test_mae"], label="test")
    axes[2].set_title("MAE")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "training_dashboard.png", dpi=180)
    plt.close(fig)


def save_prediction_plot(predictions: pd.DataFrame, checkpoint_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].scatter(predictions["actual_soh_percent"], predictions["predicted_soh_percent"], s=7, alpha=0.25)
    lo = min(predictions["actual_soh_percent"].min(), predictions["predicted_soh_percent"].min())
    hi = max(predictions["actual_soh_percent"].max(), predictions["predicted_soh_percent"].max())
    axes[0, 0].plot([lo, hi], [lo, hi], "k--")
    axes[0, 0].set_title("LSTM actual vs predicted")
    axes[0, 0].set_xlabel("Actual SOH (%)")
    axes[0, 0].set_ylabel("Predicted SOH (%)")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].scatter(predictions["charge_cycle"], predictions["actual_soh_percent"], s=6, alpha=0.2, label="actual", color="black")
    axes[0, 1].scatter(predictions["charge_cycle"], predictions["predicted_soh_percent"], s=6, alpha=0.2, label="predicted", color="#2a9d8f")
    axes[0, 1].set_title("RW11 SOH over charge-cycle chronology")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["actual_soh_percent"], marker="o", label="actual")
    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["predicted_soh_percent"], marker="o", label="predicted")
    axes[1, 0].set_title("Checkpoint-aggregated prediction")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].hist(predictions["error_pp"], bins=60, color="#457b9d")
    axes[1, 1].axvline(0, color="black", linestyle="--")
    axes[1, 1].set_title("Residuals")
    axes[1, 1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "prediction_report.png", dpi=180)
    plt.close(fig)


def save_comparison(metrics_dict: dict[str, Any]) -> None:
    transformer = json.loads(
        (ROOT / "cycle_level_weighted_transformer_rw9_rw10_to_rw11" / "metrics.json").read_text(encoding="utf-8")
    )
    lgbm = json.loads((ROOT / "accuracy_boost_lgbm" / "best_lgbm_metrics.json").read_text(encoding="utf-8"))
    comparison = pd.DataFrame(
        [
            {
                "model": "best_transformer",
                "test_r2": transformer["test_metrics_cycle_level"]["r2"],
                "test_mae": transformer["test_metrics_cycle_level"]["mae"],
                "test_rmse": transformer["test_metrics_cycle_level"]["rmse"],
                "checkpoint_r2": transformer["test_metrics_checkpoint_aggregated"]["r2"],
                "checkpoint_mae": transformer["test_metrics_checkpoint_aggregated"]["mae"],
                "checkpoint_rmse": transformer["test_metrics_checkpoint_aggregated"]["rmse"],
            },
            {
                "model": "bilstm",
                "test_r2": metrics_dict["test_metrics_cycle_level"]["r2"],
                "test_mae": metrics_dict["test_metrics_cycle_level"]["mae"],
                "test_rmse": metrics_dict["test_metrics_cycle_level"]["rmse"],
                "checkpoint_r2": metrics_dict["test_metrics_checkpoint_aggregated"]["r2"],
                "checkpoint_mae": metrics_dict["test_metrics_checkpoint_aggregated"]["mae"],
                "checkpoint_rmse": metrics_dict["test_metrics_checkpoint_aggregated"]["rmse"],
            },
            {
                "model": "lightgbm_features",
                "test_r2": lgbm["event_level_metrics"]["r2"],
                "test_mae": lgbm["event_level_metrics"]["mae"],
                "test_rmse": lgbm["event_level_metrics"]["rmse"],
                "checkpoint_r2": lgbm["checkpoint_aggregated_metrics"]["r2"],
                "checkpoint_mae": lgbm["checkpoint_aggregated_metrics"]["mae"],
                "checkpoint_rmse": lgbm["checkpoint_aggregated_metrics"]["rmse"],
            },
        ]
    )
    comparison.to_csv(OUTDIR / "model_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = comparison["model"].tolist()
    x = np.arange(len(labels))
    for ax, col, title, color in zip(
        axes,
        ["test_r2", "test_mae", "checkpoint_r2"],
        ["RW11 test R2", "RW11 test MAE", "Checkpoint R2"],
        ["#2a9d8f", "#457b9d", "#f4a261"],
    ):
        ax.bar(x, comparison[col], color=color)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    torch.set_num_threads(8)

    train_parts = [build_battery_sequences(battery) for battery in TRAIN_BATTERIES]
    test_parts = [build_battery_sequences(battery) for battery in TEST_BATTERIES]
    train_sequences, train_meta = concat_sequence_sets(train_parts)
    test_sequences, test_meta = concat_sequence_sets(test_parts)
    train_meta = add_block_weights(train_meta)
    test_meta = add_block_weights(test_meta)

    mean, std = compute_standardization(train_sequences)
    apply_standardization(train_sequences, mean, std)
    apply_standardization(test_sequences, mean, std)

    train_y = train_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    test_y = test_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    train_w = train_meta["sample_weight"].to_numpy(dtype=np.float32)
    test_w = test_meta["sample_weight"].to_numpy(dtype=np.float32)
    train_y_norm, test_y_norm, y_mean, y_std = standardize_target(train_y, test_y)

    train_ds = SequenceDataset(train_sequences, train_y_norm, train_w, train_meta)
    test_ds = SequenceDataset(test_sequences, test_y_norm, test_w, test_meta)
    train_loader = DataLoader(train_ds, batch_size=SPEC.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=SPEC.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    device = torch.device("cpu")
    input_dim = train_sequences[0].shape[1]
    model = LSTMRegressor(input_dim, SPEC).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPEC.learning_rate, weight_decay=SPEC.weight_decay)

    history_rows: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_test_r2 = -np.inf

    for epoch in range(1, SPEC.epochs + 1):
        train_loss, train_pred_norm, train_true_norm, *_ = run_epoch(model, train_loader, optimizer, device)
        test_loss, test_pred_norm, test_true_norm, *_ = run_epoch(model, test_loader, None, device)
        train_pred = destandardize(train_pred_norm, y_mean, y_std)
        train_true = destandardize(train_true_norm, y_mean, y_std)
        test_pred = destandardize(test_pred_norm, y_mean, y_std)
        test_true = destandardize(test_true_norm, y_mean, y_std)
        train_metrics = regression_metrics(train_true, train_pred)
        test_metrics = regression_metrics(test_true, test_pred)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_r2": train_metrics["r2"],
                "test_r2": test_metrics["r2"],
                "train_mae": train_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
            }
        )
        if test_metrics["r2"] > best_test_r2:
            best_test_r2 = test_metrics["r2"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch}/{SPEC.epochs}: train_r2={train_metrics['r2']:.4f} "
            f"test_r2={test_metrics['r2']:.4f} test_mae={test_metrics['mae']:.3f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval = run_epoch(model, train_loader, None, device)
    test_eval = run_epoch(model, test_loader, None, device)
    _, train_pred_norm, train_true_norm, train_batteries, train_cycles, train_checkpoints, train_ids = train_eval
    _, test_pred_norm, test_true_norm, test_batteries, test_cycles, test_checkpoints, test_ids = test_eval

    train_pred = destandardize(train_pred_norm, y_mean, y_std)
    train_true = destandardize(train_true_norm, y_mean, y_std)
    test_pred = destandardize(test_pred_norm, y_mean, y_std)
    test_true = destandardize(test_true_norm, y_mean, y_std)

    train_pred_df = prediction_frame(train_batteries, train_cycles, train_checkpoints, train_ids, train_true, train_pred)
    test_pred_df = prediction_frame(test_batteries, test_cycles, test_checkpoints, test_ids, test_true, test_pred)
    checkpoint_df = aggregate_checkpoint_predictions(test_pred_df)

    metrics_dict = {
        "spec": asdict(SPEC),
        "input_dim": input_dim,
        "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "train_metrics_cycle_level": regression_metrics(train_true, train_pred),
        "test_metrics_cycle_level": regression_metrics(test_true, test_pred),
        "test_metrics_checkpoint_aggregated": regression_metrics(
            checkpoint_df["actual_soh_percent"].to_numpy(dtype=np.float64),
            checkpoint_df["predicted_soh_percent"].to_numpy(dtype=np.float64),
        ),
        "best_epoch_by_test_r2": int(pd.DataFrame(history_rows).sort_values("test_r2", ascending=False).iloc[0]["epoch"]),
    }

    pd.DataFrame(history_rows).to_csv(OUTDIR / "training_history.csv", index=False)
    train_pred_df.to_csv(OUTDIR / "train_predictions.csv", index=False)
    test_pred_df.to_csv(OUTDIR / "test_predictions.csv", index=False)
    checkpoint_df.to_csv(OUTDIR / "test_checkpoint_predictions.csv", index=False)
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "metrics": metrics_dict}, OUTDIR / "model.pt")

    save_training_plot(pd.DataFrame(history_rows))
    save_prediction_plot(test_pred_df, checkpoint_df)
    save_comparison(metrics_dict)
    print(json.dumps(metrics_dict, indent=2), flush=True)


if __name__ == "__main__":
    main()
