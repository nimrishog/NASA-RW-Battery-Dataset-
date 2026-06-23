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


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
FEATURE_DIR = ROOT / "charge engineered features"
OUTDIR = ROOT / "pure_cycle_feature_token_transformer_rw11"
TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERIES = ("RW11",)
CHUNK_WIDTH = 8


@dataclass(frozen=True)
class FeatureTokenSpec:
    name: str
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int


SPECS = [
    FeatureTokenSpec(
        name="pure_feature_tokens_h4_d96_l3_fast",
        d_model=96,
        nhead=4,
        num_layers=3,
        dim_feedforward=192,
        dropout=0.10,
        learning_rate=1.0e-3,
        weight_decay=5.0e-4,
        epochs=20,
        batch_size=4096,
    ),
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


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


def load_feature_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frames = []
    for battery in (*TRAIN_BATTERIES, *TEST_BATTERIES):
        df = pd.read_csv(FEATURE_DIR / f"{battery}_charge_engineered_features.csv")
        df["battery"] = battery
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    exclude = {"battery", "date", "capacity_ah", "soh_percent", "label_source", "mat_step_index", "cycle"}
    feature_names = [
        column
        for column in data.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(data[column])
    ]
    train_df = data[data["battery"].isin(TRAIN_BATTERIES)].copy().reset_index(drop=True)
    test_df = data[data["battery"].isin(TEST_BATTERIES)].copy().reset_index(drop=True)
    return train_df, test_df, feature_names


def add_inverse_label_block_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_block"] = df["soh_percent"].round(8)
    counts = df.groupby(["battery", "label_block"]).size().rename("block_size").reset_index()
    df = df.merge(counts, on=["battery", "label_block"], how="left")
    df["sample_weight"] = 1.0 / df["block_size"].astype(np.float32)
    df["sample_weight"] = df["sample_weight"] / float(df["sample_weight"].mean())
    return df


def standardize_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = train_df[feature_names].to_numpy(dtype=np.float32)
    test_x = test_df[feature_names].to_numpy(dtype=np.float32)
    train_x = np.nan_to_num(train_x, nan=0.0, posinf=0.0, neginf=0.0)
    test_x = np.nan_to_num(test_x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = train_x.mean(axis=0).astype(np.float32)
    std = train_x.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-8, 1.0, std).astype(np.float32)
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def standardize_target(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(train_y.mean(dtype=np.float64))
    std = float(train_y.std(dtype=np.float64))
    if std < 1.0e-8:
        std = 1.0
    return ((train_y - mean) / std).astype(np.float32), ((test_y - mean) / std).astype(np.float32), mean, std


def destandardize(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    return y.astype(np.float32) * std + mean


class FeatureTokenDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray, meta: pd.DataFrame):
        self.features = features.astype(np.float32, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.weights = weights.astype(np.float32, copy=False)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.meta.iloc[idx]
        return {
            "features": self.features[idx],
            "target": float(self.targets[idx]),
            "weight": float(self.weights[idx]),
            "battery": str(row["battery"]),
            "cycle": int(row["cycle"]),
            "label_block": float(row["label_block"]),
        }


def collate_batch(
    batch: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int], list[float]]:
    x = torch.tensor(np.stack([item["features"] for item in batch]), dtype=torch.float32)
    y = torch.tensor([item["target"] for item in batch], dtype=torch.float32)
    w = torch.tensor([item["weight"] for item in batch], dtype=torch.float32)
    batteries = [item["battery"] for item in batch]
    cycles = [int(item["cycle"]) for item in batch]
    labels = [float(item["label_block"]) for item in batch]
    return x, y, w, batteries, cycles, labels


class CycleFeatureTokenTransformer(nn.Module):
    """Pure transformer over full-cycle engineered feature tokens."""

    def __init__(self, feature_count: int, spec: FeatureTokenSpec):
        super().__init__()
        self.feature_count = feature_count
        self.chunk_count = int(math.ceil(feature_count / CHUNK_WIDTH))
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.feature_proj = nn.Linear(CHUNK_WIDTH, spec.d_model)
        self.token_id_embedding = nn.Embedding(self.chunk_count, spec.d_model)
        self.input_norm = nn.LayerNorm(spec.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spec.d_model,
            nhead=spec.nhead,
            dim_feedforward=spec.dim_feedforward,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=spec.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(spec.d_model),
            nn.Linear(spec.d_model, spec.d_model),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        pad_count = self.chunk_count * CHUNK_WIDTH - self.feature_count
        if pad_count:
            x = nn.functional.pad(x, (0, pad_count))
        chunks = x.view(batch_size, self.chunk_count, CHUNK_WIDTH)
        token_ids = torch.arange(self.chunk_count, device=x.device)
        tokens = self.feature_proj(chunks) + self.token_id_embedding(token_ids).unsqueeze(0)
        cls = self.cls.expand(batch_size, -1, -1)
        tokens = self.input_norm(torch.cat([cls, tokens], dim=1))
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0]).squeeze(-1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, list[str], list[int], list[float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    batteries: list[str] = []
    cycles: list[int] = []
    label_blocks: list[float] = []
    with torch.set_grad_enabled(training):
        for x, y, w, batch_battery, batch_cycle, batch_label in loader:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            out = model(x)
            per_sample = (out - y) ** 2
            loss = (per_sample * w).sum() / torch.clamp(w.sum(), min=1.0e-8)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
            total_count += x.size(0)
            preds.append(out.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
            batteries.extend(batch_battery)
            cycles.extend([int(c) for c in batch_cycle])
            label_blocks.extend([float(v) for v in batch_label])
    return total_loss / max(total_count, 1), np.concatenate(preds), np.concatenate(targets), batteries, cycles, label_blocks


def prediction_frame(
    batteries: list[str],
    cycles: list[int],
    label_blocks: list[float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "battery": batteries,
            "cycle": cycles,
            "label_block": label_blocks,
            "actual_soh_percent": y_true,
            "predicted_soh_percent": y_pred,
            "error_pp": y_pred - y_true,
            "absolute_error_pp": np.abs(y_pred - y_true),
        }
    )


def aggregate_label_blocks(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["battery", "label_block"], as_index=False)
        .agg(
            actual_soh_percent=("actual_soh_percent", "mean"),
            predicted_soh_percent=("predicted_soh_percent", "mean"),
            cycle_count=("cycle", "count"),
        )
        .sort_values(["battery", "label_block"], ascending=[True, False])
    )
    out["error_pp"] = out["predicted_soh_percent"] - out["actual_soh_percent"]
    out["absolute_error_pp"] = out["error_pp"].abs()
    return out


def train_one_spec(
    spec: FeatureTokenSpec,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y_norm: np.ndarray,
    test_y_norm: np.ndarray,
    train_w: np.ndarray,
    test_w: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    y_mean: float,
    y_std: float,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec_dir = OUTDIR / spec.name
    spec_dir.mkdir(parents=True, exist_ok=True)
    train_ds = FeatureTokenDataset(train_x, train_y_norm, train_w, train_meta)
    test_ds = FeatureTokenDataset(test_x, test_y_norm, test_w, test_meta)
    train_loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=spec.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    model = CycleFeatureTokenTransformer(train_x.shape[1], spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)

    history_rows: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_r2 = -np.inf
    for epoch in range(1, spec.epochs + 1):
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
        if test_metrics["r2"] > best_r2:
            best_r2 = test_metrics["r2"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"{spec.name} epoch {epoch}/{spec.epochs}: train_r2={train_metrics['r2']:.4f} "
            f"test_r2={test_metrics['r2']:.4f} test_mae={test_metrics['mae']:.3f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    train_eval = run_epoch(model, train_loader, None, device)
    test_eval = run_epoch(model, test_loader, None, device)
    _, train_pred_norm, train_true_norm, train_batteries, train_cycles, train_blocks = train_eval
    _, test_pred_norm, test_true_norm, test_batteries, test_cycles, test_blocks = test_eval
    train_pred = destandardize(train_pred_norm, y_mean, y_std)
    train_true = destandardize(train_true_norm, y_mean, y_std)
    test_pred = destandardize(test_pred_norm, y_mean, y_std)
    test_true = destandardize(test_true_norm, y_mean, y_std)
    train_pred_df = prediction_frame(train_batteries, train_cycles, train_blocks, train_true, train_pred)
    test_pred_df = prediction_frame(test_batteries, test_cycles, test_blocks, test_true, test_pred)
    block_df = aggregate_label_blocks(test_pred_df)
    history = pd.DataFrame(history_rows)
    metrics = {
        "spec": asdict(spec),
        "feature_count": int(train_x.shape[1]),
        "feature_chunk_width": int(CHUNK_WIDTH),
        "feature_token_count": int(math.ceil(train_x.shape[1] / CHUNK_WIDTH)),
        "sequence_tokens_with_cls": int(math.ceil(train_x.shape[1] / CHUNK_WIDTH) + 1),
        "parameter_count": parameter_count(model),
        "train_sample_count": int(len(train_pred_df)),
        "test_sample_count": int(len(test_pred_df)),
        "best_epoch_by_test_r2": int(history.loc[history["test_r2"].idxmax(), "epoch"]),
        "train_metrics_cycle_level": regression_metrics(train_true, train_pred),
        "test_metrics_cycle_level": regression_metrics(test_true, test_pred),
        "test_metrics_label_block_aggregated": regression_metrics(
            block_df["actual_soh_percent"].to_numpy(),
            block_df["predicted_soh_percent"].to_numpy(),
        ),
    }
    history.to_csv(spec_dir / "training_history.csv", index=False)
    train_pred_df.to_csv(spec_dir / "train_predictions.csv", index=False)
    test_pred_df.to_csv(spec_dir / "test_predictions.csv", index=False)
    block_df.to_csv(spec_dir / "test_label_block_predictions.csv", index=False)
    (spec_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "spec": asdict(spec), "metrics": metrics}, spec_dir / "model.pt")
    return metrics, history, test_pred_df, block_df


def save_best_plots(best_name: str, history: pd.DataFrame, pred_df: pd.DataFrame, block_df: pd.DataFrame, comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["test_loss"], label="test")
    axes[0].set_title("Weighted MSE loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(history["epoch"], history["train_r2"], label="train")
    axes[1].plot(history["epoch"], history["test_r2"], label="test")
    axes[1].set_title("R2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(history["epoch"], history["train_mae"], label="train")
    axes[2].plot(history["epoch"], history["test_mae"], label="test")
    axes[2].set_title("MAE")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)
    fig.suptitle(best_name)
    fig.tight_layout()
    fig.savefig(OUTDIR / "best_training_dashboard.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].scatter(pred_df["actual_soh_percent"], pred_df["predicted_soh_percent"], s=7, alpha=0.25)
    lo = min(pred_df["actual_soh_percent"].min(), pred_df["predicted_soh_percent"].min())
    hi = max(pred_df["actual_soh_percent"].max(), pred_df["predicted_soh_percent"].max())
    axes[0, 0].plot([lo, hi], [lo, hi], "k--")
    axes[0, 0].set_title("Actual vs predicted")
    axes[0, 0].set_xlabel("Actual SOH (%)")
    axes[0, 0].set_ylabel("Predicted SOH (%)")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 1].scatter(pred_df["cycle"], pred_df["actual_soh_percent"], s=6, alpha=0.2, label="actual", color="black")
    axes[0, 1].scatter(pred_df["cycle"], pred_df["predicted_soh_percent"], s=6, alpha=0.2, label="predicted", color="#2a9d8f")
    axes[0, 1].set_title("RW11 SOH over cycle")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.25)
    axes[1, 0].plot(block_df["actual_soh_percent"].to_numpy(), marker="o", label="actual")
    axes[1, 0].plot(block_df["predicted_soh_percent"].to_numpy(), marker="o", label="predicted")
    axes[1, 0].set_title("Label-block aggregated SOH")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 1].hist(pred_df["error_pp"], bins=60, color="#457b9d")
    axes[1, 1].axvline(0, color="black", linestyle="--")
    axes[1, 1].set_title("Residuals")
    axes[1, 1].set_xlabel("Prediction error, percentage points")
    axes[1, 1].grid(True, alpha=0.25)
    fig.suptitle(best_name)
    fig.tight_layout()
    fig.savefig(OUTDIR / "best_prediction_report.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = comparison["model"].tolist()
    x = np.arange(len(labels))
    for ax, col, title, color in zip(
        axes,
        ["test_r2", "test_mae", "block_r2"],
        ["RW11 test R2", "RW11 test MAE", "Label-block R2"],
        ["#2a9d8f", "#457b9d", "#f4a261"],
    ):
        ax.bar(x, comparison[col], color=color)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=22, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    torch.set_num_threads(8)
    train_df, test_df, feature_names = load_feature_data()
    train_df = add_inverse_label_block_weights(train_df)
    test_df = add_inverse_label_block_weights(test_df)
    train_x, test_x, feature_mean, feature_std = standardize_features(train_df, test_df, feature_names)
    train_y = train_df["soh_percent"].to_numpy(dtype=np.float32)
    test_y = test_df["soh_percent"].to_numpy(dtype=np.float32)
    train_y_norm, test_y_norm, y_mean, y_std = standardize_target(train_y, test_y)
    train_w = train_df["sample_weight"].to_numpy(dtype=np.float32)
    test_w = test_df["sample_weight"].to_numpy(dtype=np.float32)
    pd.DataFrame({"feature": feature_names, "mean": feature_mean, "std": feature_std}).to_csv(
        OUTDIR / "feature_scaler.csv",
        index=False,
    )
    (OUTDIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    device = torch.device("cpu")
    all_metrics: list[dict[str, Any]] = []
    artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for spec in SPECS:
        set_seed()
        metrics, history, pred_df, block_df = train_one_spec(
            spec,
            train_x,
            test_x,
            train_y_norm,
            test_y_norm,
            train_w,
            test_w,
            train_df,
            test_df,
            y_mean,
            y_std,
            device,
        )
        all_metrics.append(metrics)
        artifacts[spec.name] = (history, pred_df, block_df)

    rows = []
    for metrics in all_metrics:
        rows.append(
            {
                "model": metrics["spec"]["name"],
                "test_r2": metrics["test_metrics_cycle_level"]["r2"],
                "test_mae": metrics["test_metrics_cycle_level"]["mae"],
                "test_rmse": metrics["test_metrics_cycle_level"]["rmse"],
                "block_r2": metrics["test_metrics_label_block_aggregated"]["r2"],
                "block_mae": metrics["test_metrics_label_block_aggregated"]["mae"],
                "best_epoch": metrics["best_epoch_by_test_r2"],
                "parameters": metrics["parameter_count"],
            }
        )
    comparison = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    comparison.to_csv(OUTDIR / "pure_transformer_summary.csv", index=False)
    best_name = str(comparison.loc[0, "model"])
    best_history, best_pred_df, best_block_df = artifacts[best_name]
    save_best_plots(best_name, best_history, best_pred_df, best_block_df, comparison)
    (OUTDIR / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
