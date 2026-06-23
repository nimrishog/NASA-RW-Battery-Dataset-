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
OUTDIR = ROOT / "pure_full_cycle_token_transformer_rw11"
TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERIES = ("RW11",)


@dataclass(frozen=True)
class PureFullCycleSpec:
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
    max_patch_tokens: int


SPEC = PureFullCycleSpec(
    name="pure_full_cycle_feature_block_tokens",
    d_model=80,
    nhead=4,
    num_layers=3,
    dim_feedforward=160,
    dropout=0.20,
    learning_rate=3.0e-4,
    weight_decay=5.0e-4,
    epochs=14,
    batch_size=512,
    max_patch_tokens=11,
)

FULL_FEATURE_CHUNK_WIDTH = 8


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


def load_full_cycle_features() -> tuple[pd.DataFrame, list[str]]:
    frames = []
    for battery in (*TRAIN_BATTERIES, *TEST_BATTERIES):
        path = ROOT / "charge engineered features" / f"{battery}_charge_engineered_features.csv"
        df = pd.read_csv(path)
        df["battery"] = battery
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    exclude = {"battery", "date", "capacity_ah", "soh_percent", "label_source", "mat_step_index", "cycle"}
    features = [
        column
        for column in data.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(data[column])
    ]
    return data, features


def standardize_full_features(
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    feature_data: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keyed = feature_data.set_index(["battery", "cycle"])

    def collect(meta: pd.DataFrame) -> np.ndarray:
        rows = []
        for row in meta.itertuples(index=False):
            rows.append(keyed.loc[(row.battery, row.charge_cycle), feature_names].to_numpy(dtype=np.float32))
        arr = np.vstack(rows).astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    train_x = collect(train_meta)
    test_x = collect(test_meta)
    mean = train_x.mean(axis=0).astype(np.float32)
    std = train_x.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-8, 1.0, std).astype(np.float32)
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


class PureFullCycleDataset(Dataset):
    def __init__(
        self,
        patch_sequences: list[np.ndarray],
        full_features: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        meta: pd.DataFrame,
    ):
        self.patch_sequences = patch_sequences
        self.full_features = full_features.astype(np.float32, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.weights = weights.astype(np.float32, copy=False)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.meta.iloc[idx]
        return {
            "patch_sequence": self.patch_sequences[idx],
            "full_features": self.full_features[idx],
            "target": float(self.targets[idx]),
            "weight": float(self.weights[idx]),
            "battery": str(row["battery"]),
            "charge_cycle": int(row["charge_cycle"]),
            "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
            "sample_id": str(row["sample_id"]),
        }


def collate_batch(
    batch: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int], list[int], list[str]]:
    max_len = max(len(item["patch_sequence"]) for item in batch)
    patch_dim = batch[0]["patch_sequence"].shape[1]
    full_dim = batch[0]["full_features"].shape[0]
    batch_size = len(batch)
    patch_x = torch.zeros((batch_size, max_len, patch_dim), dtype=torch.float32)
    patch_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    full_x = torch.zeros((batch_size, full_dim), dtype=torch.float32)
    y = torch.zeros(batch_size, dtype=torch.float32)
    w = torch.zeros(batch_size, dtype=torch.float32)
    batteries: list[str] = []
    cycles: list[int] = []
    checkpoints: list[int] = []
    sample_ids: list[str] = []
    for idx, item in enumerate(batch):
        seq = torch.from_numpy(item["patch_sequence"]).float()
        seq_len = seq.shape[0]
        patch_x[idx, :seq_len] = seq
        patch_mask[idx, :seq_len] = True
        full_x[idx] = torch.from_numpy(item["full_features"]).float()
        y[idx] = float(item["target"])
        w[idx] = float(item["weight"])
        batteries.append(item["battery"])
        cycles.append(int(item["charge_cycle"]))
        checkpoints.append(int(item["assigned_checkpoint_cycle"]))
        sample_ids.append(item["sample_id"])
    return patch_x, patch_mask, full_x, y, w, batteries, cycles, checkpoints, sample_ids


def create_positional_encoding(length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PureFullCycleTokenTransformer(nn.Module):
    """Transformer where patch summaries and full-cycle feature blocks are all tokens."""

    def __init__(self, patch_input_dim: int, full_feature_count: int, spec: PureFullCycleSpec):
        super().__init__()
        self.full_feature_count = full_feature_count
        self.full_feature_chunk_count = int(math.ceil(full_feature_count / FULL_FEATURE_CHUNK_WIDTH))
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.patch_proj = nn.Linear(patch_input_dim, spec.d_model)
        self.feature_block_proj = nn.Linear(FULL_FEATURE_CHUNK_WIDTH, spec.d_model)
        self.feature_id_embedding = nn.Embedding(self.full_feature_chunk_count, spec.d_model)
        self.type_embedding = nn.Embedding(3, spec.d_model)
        self.input_norm = nn.LayerNorm(spec.d_model)
        max_tokens = 1 + spec.max_patch_tokens + self.full_feature_chunk_count
        self.register_buffer(
            "pe",
            create_positional_encoding(max_tokens, spec.d_model).unsqueeze(0),
            persistent=False,
        )
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

    def forward(self, patch_x: torch.Tensor, patch_mask: torch.Tensor, full_x: torch.Tensor) -> torch.Tensor:
        batch_size = patch_x.size(0)
        patch_tokens = self.patch_proj(patch_x)
        pad_count = self.full_feature_chunk_count * FULL_FEATURE_CHUNK_WIDTH - self.full_feature_count
        if pad_count:
            full_x = nn.functional.pad(full_x, (0, pad_count))
        full_blocks = full_x.view(batch_size, self.full_feature_chunk_count, FULL_FEATURE_CHUNK_WIDTH)
        feature_ids = torch.arange(self.full_feature_chunk_count, device=full_x.device)
        feature_tokens = self.feature_block_proj(full_blocks)
        feature_tokens = feature_tokens + self.feature_id_embedding(feature_ids).unsqueeze(0)
        cls = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, patch_tokens, feature_tokens], dim=1)

        cls_type = torch.zeros((batch_size, 1), dtype=torch.long, device=full_x.device)
        patch_type = torch.ones((batch_size, patch_tokens.size(1)), dtype=torch.long, device=full_x.device)
        feature_type = torch.full((batch_size, self.full_feature_chunk_count), 2, dtype=torch.long, device=full_x.device)
        type_ids = torch.cat([cls_type, patch_type, feature_type], dim=1)
        tokens = self.input_norm(tokens + self.type_embedding(type_ids) + self.pe[:, : tokens.size(1)])

        cls_mask = torch.ones((batch_size, 1), device=patch_mask.device, dtype=torch.bool)
        feature_mask = torch.ones((batch_size, self.full_feature_chunk_count), device=patch_mask.device, dtype=torch.bool)
        full_mask = torch.cat([cls_mask, patch_mask, feature_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        return self.head(encoded[:, 0]).squeeze(-1)


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
    with torch.set_grad_enabled(training):
        for batch in loader:
            patch_x, patch_mask, full_x, batch_y, batch_w, batch_batteries, batch_cycles, batch_checkpoints, batch_ids = batch
            patch_x = patch_x.to(device)
            patch_mask = patch_mask.to(device)
            full_x = full_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)

            out = model(patch_x, patch_mask, full_x)
            per_sample = (out - batch_y) ** 2
            loss = (per_sample * batch_w).sum() / torch.clamp(batch_w.sum(), min=1.0e-8)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(loss.item()) * patch_x.size(0)
            total_count += patch_x.size(0)
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


def save_plots(history: pd.DataFrame, pred_df: pd.DataFrame, checkpoint_df: pd.DataFrame) -> None:
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
    fig.tight_layout()
    fig.savefig(OUTDIR / "training_dashboard.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].scatter(pred_df["actual_soh_percent"], pred_df["predicted_soh_percent"], s=7, alpha=0.25)
    lo = min(pred_df["actual_soh_percent"].min(), pred_df["predicted_soh_percent"].min())
    hi = max(pred_df["actual_soh_percent"].max(), pred_df["predicted_soh_percent"].max())
    axes[0, 0].plot([lo, hi], [lo, hi], "k--")
    axes[0, 0].set_title("Pure full-cycle-token transformer")
    axes[0, 0].set_xlabel("Actual SOH (%)")
    axes[0, 0].set_ylabel("Predicted SOH (%)")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 1].scatter(pred_df["charge_cycle"], pred_df["actual_soh_percent"], s=6, alpha=0.2, label="actual", color="black")
    axes[0, 1].scatter(pred_df["charge_cycle"], pred_df["predicted_soh_percent"], s=6, alpha=0.2, label="predicted", color="#2a9d8f")
    axes[0, 1].set_title("RW11 SOH over charge-cycle chronology")
    axes[0, 1].set_xlabel("Charge cycle")
    axes[0, 1].set_ylabel("SOH (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.25)
    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["actual_soh_percent"], marker="o", label="actual")
    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["predicted_soh_percent"], marker="o", label="predicted")
    axes[1, 0].set_title("Checkpoint-aggregated prediction")
    axes[1, 0].set_xlabel("Assigned benchmark checkpoint cycle")
    axes[1, 0].set_ylabel("SOH (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 1].hist(pred_df["error_pp"], bins=60, color="#457b9d")
    axes[1, 1].axvline(0, color="black", linestyle="--")
    axes[1, 1].set_title("Residuals")
    axes[1, 1].set_xlabel("Prediction error, percentage points")
    axes[1, 1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "prediction_report.png", dpi=180)
    plt.close(fig)


def save_model_comparison(metrics_dict: dict[str, Any]) -> None:
    transformer = json.loads((ROOT / "cycle_level_weighted_transformer_rw9_rw10_to_rw11" / "metrics.json").read_text())
    hybrid = json.loads((ROOT / "hybrid_full_cycle_transformer_rw11" / "metrics.json").read_text())
    lgbm = json.loads((ROOT / "accuracy_boost_lgbm" / "best_lgbm_metrics.json").read_text())
    comparison = pd.DataFrame(
        [
            {
                "model": "patch_transformer",
                "test_r2": transformer["test_metrics_cycle_level"]["r2"],
                "test_mae": transformer["test_metrics_cycle_level"]["mae"],
                "checkpoint_r2": transformer["test_metrics_checkpoint_aggregated"]["r2"],
            },
            {
                "model": "hybrid_transformer_previous",
                "test_r2": hybrid["test_metrics_cycle_level"]["r2"],
                "test_mae": hybrid["test_metrics_cycle_level"]["mae"],
                "checkpoint_r2": hybrid["test_metrics_checkpoint_aggregated"]["r2"],
            },
            {
                "model": "pure_full_cycle_token_transformer",
                "test_r2": metrics_dict["test_metrics_cycle_level"]["r2"],
                "test_mae": metrics_dict["test_metrics_cycle_level"]["mae"],
                "checkpoint_r2": metrics_dict["test_metrics_checkpoint_aggregated"]["r2"],
            },
            {
                "model": "lightgbm_full_cycle_features",
                "test_r2": lgbm["event_level_metrics"]["r2"],
                "test_mae": lgbm["event_level_metrics"]["mae"],
                "checkpoint_r2": lgbm["checkpoint_aggregated_metrics"]["r2"],
            },
        ]
    )
    comparison.to_csv(OUTDIR / "model_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
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
        ax.set_xticks(x, labels, rotation=22, ha="right")
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

    patch_mean, patch_std = compute_standardization(train_sequences)
    apply_standardization(train_sequences, patch_mean, patch_std)
    apply_standardization(test_sequences, patch_mean, patch_std)

    feature_data, feature_names = load_full_cycle_features()
    train_full, test_full, full_mean, full_std = standardize_full_features(
        train_meta,
        test_meta,
        feature_data,
        feature_names,
    )

    train_y = train_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    test_y = test_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    train_w = train_meta["sample_weight"].to_numpy(dtype=np.float32)
    test_w = test_meta["sample_weight"].to_numpy(dtype=np.float32)
    train_y_norm, test_y_norm, y_mean, y_std = standardize_target(train_y, test_y)

    train_ds = PureFullCycleDataset(train_sequences, train_full, train_y_norm, train_w, train_meta)
    test_ds = PureFullCycleDataset(test_sequences, test_full, test_y_norm, test_w, test_meta)
    train_loader = DataLoader(train_ds, batch_size=SPEC.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=SPEC.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    device = torch.device("cpu")
    patch_input_dim = train_sequences[0].shape[1]
    full_feature_count = train_full.shape[1]
    model = PureFullCycleTokenTransformer(patch_input_dim, full_feature_count, SPEC).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPEC.learning_rate, weight_decay=SPEC.weight_decay)

    history_rows: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_r2 = -np.inf
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
        if test_metrics["r2"] > best_r2:
            best_r2 = test_metrics["r2"]
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
    history = pd.DataFrame(history_rows)
    metrics = {
        "spec": asdict(SPEC),
        "patch_token_input_dim": int(patch_input_dim),
        "full_cycle_feature_count": int(full_feature_count),
        "full_cycle_feature_chunk_width": int(FULL_FEATURE_CHUNK_WIDTH),
        "full_cycle_feature_token_count": int(math.ceil(full_feature_count / FULL_FEATURE_CHUNK_WIDTH)),
        "total_sequence_tokens_without_cls": int(SPEC.max_patch_tokens + math.ceil(full_feature_count / FULL_FEATURE_CHUNK_WIDTH)),
        "full_cycle_features": feature_names,
        "parameter_count": parameter_count(model),
        "best_epoch_by_test_r2": int(history.loc[history["test_r2"].idxmax(), "epoch"]),
        "train_sample_count": int(len(train_pred_df)),
        "test_sample_count": int(len(test_pred_df)),
        "train_metrics_cycle_level": regression_metrics(train_true, train_pred),
        "test_metrics_cycle_level": regression_metrics(test_true, test_pred),
        "test_metrics_checkpoint_aggregated": regression_metrics(
            checkpoint_df["actual_soh_percent"].to_numpy(),
            checkpoint_df["predicted_soh_percent"].to_numpy(),
        ),
    }

    history.to_csv(OUTDIR / "training_history.csv", index=False)
    train_pred_df.to_csv(OUTDIR / "train_predictions.csv", index=False)
    test_pred_df.to_csv(OUTDIR / "test_predictions.csv", index=False)
    checkpoint_df.to_csv(OUTDIR / "test_checkpoint_predictions.csv", index=False)
    pd.DataFrame({"feature": feature_names, "mean": full_mean, "std": full_std}).to_csv(
        OUTDIR / "full_cycle_feature_scaler.csv",
        index=False,
    )
    pd.DataFrame({"feature_index": np.arange(len(patch_mean)), "mean": patch_mean, "std": patch_std}).to_csv(
        OUTDIR / "patch_token_scaler.csv",
        index=False,
    )
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "spec": asdict(SPEC), "metrics": metrics}, OUTDIR / "model.pt")
    save_plots(history, test_pred_df, checkpoint_df)
    save_model_comparison(metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
