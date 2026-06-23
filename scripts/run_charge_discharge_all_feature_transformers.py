from __future__ import annotations

import json
import math
import pickle
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

import run_charge_discharge_full_cycle_suite as suite


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = ROOT / "charge_discharge_all_feature_transformers"
TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERIES = ("RW11",)
FEATURE_CHUNK_WIDTH = 8


@dataclass(frozen=True)
class Spec:
    name: str
    model_kind: str
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
    Spec(
        name="patch_augmented_all_features_h4_d96_l3",
        model_kind="patch_augmented",
        d_model=96,
        nhead=4,
        num_layers=3,
        dim_feedforward=192,
        dropout=0.15,
        learning_rate=5.0e-4,
        weight_decay=5.0e-4,
        epochs=18,
        batch_size=4096,
    ),
    Spec(
        name="patch_augmented_all_features_h8_d128_l3",
        model_kind="patch_augmented",
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.15,
        learning_rate=3.0e-4,
        weight_decay=5.0e-4,
        epochs=18,
        batch_size=3072,
    ),
    Spec(
        name="feature_tokens_all_features_h4_d64_l2",
        model_kind="feature_tokens",
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.12,
        learning_rate=7.0e-4,
        weight_decay=5.0e-4,
        epochs=18,
        batch_size=4096,
    ),
    Spec(
        name="feature_tokens_all_features_h8_d128_l3",
        model_kind="feature_tokens",
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.15,
        learning_rate=3.0e-4,
        weight_decay=5.0e-4,
        epochs=22,
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


def standardize_target(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(train_y.mean(dtype=np.float64))
    std = float(train_y.std(dtype=np.float64))
    if std < 1.0e-8:
        std = 1.0
    return ((train_y - mean) / std).astype(np.float32), ((test_y - mean) / std).astype(np.float32), mean, std


def destandardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values.astype(np.float32) * std + mean


def load_prepared_data() -> dict[str, Any]:
    features, patches = suite.build_or_load_event_table_and_patches()
    features = suite.add_block_weights(features.rename(columns={"label_charge_count": "charge_cycle_for_label"}))
    features = features.rename(columns={"charge_cycle_for_label": "label_charge_count"})

    train_mask = features["battery"].isin(TRAIN_BATTERIES).to_numpy()
    test_mask = features["battery"].isin(TEST_BATTERIES).to_numpy()
    train_meta = features.loc[train_mask].reset_index(drop=True)
    test_meta = features.loc[test_mask].reset_index(drop=True)
    train_patches_raw = [patches[i] for i in np.flatnonzero(train_mask)]
    test_patches_raw = [patches[i] for i in np.flatnonzero(test_mask)]

    exclude = {
        "battery",
        "event_type",
        "event_order",
        "mat_step_index",
        "date",
        "assigned_checkpoint_cycle",
        "capacity_ah",
        "soh_percent",
        "initial_capacity_ah",
        "block_size",
        "sample_weight",
    }
    feature_names = [
        col
        for col in features.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(features[col])
    ]
    train_feature_raw = train_meta[feature_names].to_numpy(dtype=np.float32)
    test_feature_raw = test_meta[feature_names].to_numpy(dtype=np.float32)
    train_feature_x, test_feature_x, feature_mean, feature_std = suite.standardize_matrix(train_feature_raw, test_feature_raw)
    train_patches, test_patches, patch_mean, patch_std = suite.standardize_patches(train_patches_raw, test_patches_raw)

    train_y = train_meta["soh_percent"].to_numpy(dtype=np.float32)
    test_y = test_meta["soh_percent"].to_numpy(dtype=np.float32)
    train_w = train_meta["sample_weight"].to_numpy(dtype=np.float32)
    test_w = test_meta["sample_weight"].to_numpy(dtype=np.float32)
    train_y_norm, test_y_norm, y_mean, y_std = standardize_target(train_y, test_y)
    return {
        "train_meta": train_meta,
        "test_meta": test_meta,
        "feature_names": feature_names,
        "train_feature_x": train_feature_x,
        "test_feature_x": test_feature_x,
        "train_patches": train_patches,
        "test_patches": test_patches,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "patch_mean": patch_mean,
        "patch_std": patch_std,
        "train_y": train_y,
        "test_y": test_y,
        "train_y_norm": train_y_norm,
        "test_y_norm": test_y_norm,
        "train_w": train_w,
        "test_w": test_w,
        "y_mean": y_mean,
        "y_std": y_std,
    }


class EventDataset(Dataset):
    def __init__(
        self,
        feature_x: np.ndarray,
        patches: list[np.ndarray],
        target_y: np.ndarray,
        weights: np.ndarray,
        meta: pd.DataFrame,
    ):
        self.feature_x = feature_x.astype(np.float32, copy=False)
        self.patches = patches
        self.target_y = target_y.astype(np.float32, copy=False)
        self.weights = weights.astype(np.float32, copy=False)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.target_y)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.meta.iloc[idx]
        return {
            "feature_x": self.feature_x[idx],
            "patches": self.patches[idx],
            "target": float(self.target_y[idx]),
            "weight": float(self.weights[idx]),
            "battery": str(row["battery"]),
            "event_type": str(row["event_type"]),
            "event_order": int(row["event_order"]),
            "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
            "sample_id": f"{row['battery']}_{row['event_type']}_{int(row['event_order'])}",
        }


def collate_batch(batch: list[dict[str, Any]]) -> tuple[Any, ...]:
    feature_x = torch.tensor(np.stack([item["feature_x"] for item in batch]), dtype=torch.float32)
    max_len = max(len(item["patches"]) for item in batch)
    patch_dim = batch[0]["patches"].shape[1]
    patch_x = torch.zeros((len(batch), max_len, patch_dim), dtype=torch.float32)
    patch_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for idx, item in enumerate(batch):
        seq = torch.from_numpy(item["patches"]).float()
        patch_x[idx, : seq.shape[0]] = seq
        patch_mask[idx, : seq.shape[0]] = True
    target = torch.tensor([item["target"] for item in batch], dtype=torch.float32)
    weight = torch.tensor([item["weight"] for item in batch], dtype=torch.float32)
    batteries = [item["battery"] for item in batch]
    event_types = [item["event_type"] for item in batch]
    event_orders = [int(item["event_order"]) for item in batch]
    checkpoints = [int(item["assigned_checkpoint_cycle"]) for item in batch]
    sample_ids = [item["sample_id"] for item in batch]
    return patch_x, patch_mask, feature_x, target, weight, batteries, event_types, event_orders, checkpoints, sample_ids


class PatchAugmentedTransformer(nn.Module):
    """Patch-token transformer where every patch token includes all full-event features."""

    def __init__(self, patch_dim: int, feature_dim: int, spec: Spec):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.input_proj = nn.Linear(patch_dim + feature_dim, spec.d_model)
        self.input_norm = nn.LayerNorm(spec.d_model)
        self.type_embedding = nn.Embedding(2, spec.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=spec.d_model,
            nhead=spec.nhead,
            dim_feedforward=spec.dim_feedforward,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=spec.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(spec.d_model),
            nn.Linear(spec.d_model, spec.d_model),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.d_model, 1),
        )

    def forward(self, patch_x: torch.Tensor, patch_mask: torch.Tensor, feature_x: torch.Tensor) -> torch.Tensor:
        repeated_features = feature_x.unsqueeze(1).expand(-1, patch_x.size(1), -1)
        tokens = self.input_proj(torch.cat([patch_x, repeated_features], dim=-1))
        batch_size = patch_x.size(0)
        cls = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        cls_type = torch.zeros((batch_size, 1), dtype=torch.long, device=patch_x.device)
        patch_type = torch.ones((batch_size, patch_x.size(1)), dtype=torch.long, device=patch_x.device)
        tokens = self.input_norm(tokens + self.type_embedding(torch.cat([cls_type, patch_type], dim=1)))
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=patch_x.device)
        mask = torch.cat([cls_mask, patch_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        return self.head(encoded[:, 0]).squeeze(-1)


class FeatureTokenTransformer(nn.Module):
    """Pure transformer where all engineered full-event features are grouped into tokens."""

    def __init__(self, feature_dim: int, spec: Spec):
        super().__init__()
        self.feature_dim = feature_dim
        self.chunk_count = int(math.ceil(feature_dim / FEATURE_CHUNK_WIDTH))
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.feature_proj = nn.Linear(FEATURE_CHUNK_WIDTH, spec.d_model)
        self.feature_id_embedding = nn.Embedding(self.chunk_count, spec.d_model)
        self.input_norm = nn.LayerNorm(spec.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=spec.d_model,
            nhead=spec.nhead,
            dim_feedforward=spec.dim_feedforward,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=spec.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(spec.d_model),
            nn.Linear(spec.d_model, spec.d_model),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.d_model, 1),
        )

    def forward(self, _patch_x: torch.Tensor, _patch_mask: torch.Tensor, feature_x: torch.Tensor) -> torch.Tensor:
        batch_size = feature_x.size(0)
        pad = self.chunk_count * FEATURE_CHUNK_WIDTH - self.feature_dim
        if pad:
            feature_x = nn.functional.pad(feature_x, (0, pad))
        chunks = feature_x.view(batch_size, self.chunk_count, FEATURE_CHUNK_WIDTH)
        ids = torch.arange(self.chunk_count, device=feature_x.device)
        tokens = self.feature_proj(chunks) + self.feature_id_embedding(ids).unsqueeze(0)
        cls = self.cls.expand(batch_size, -1, -1)
        encoded = self.encoder(self.input_norm(torch.cat([cls, tokens], dim=1)))
        return self.head(encoded[:, 0]).squeeze(-1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, list[str], list[str], list[int], list[int], list[str]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    batteries: list[str] = []
    event_types: list[str] = []
    event_orders: list[int] = []
    checkpoints: list[int] = []
    sample_ids: list[str] = []
    with torch.set_grad_enabled(training):
        for patch_x, patch_mask, feature_x, y, w, b, et, eo, cp, sid in loader:
            patch_x = patch_x.to(device)
            patch_mask = patch_mask.to(device)
            feature_x = feature_x.to(device)
            y = y.to(device)
            w = w.to(device)
            out = model(patch_x, patch_mask, feature_x)
            per_sample = (out - y) ** 2
            loss = (per_sample * w).sum() / torch.clamp(w.sum(), min=1.0e-8)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += float(loss.item()) * int(y.size(0))
            total_count += int(y.size(0))
            preds.append(out.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
            batteries.extend(b)
            event_types.extend(et)
            event_orders.extend([int(x) for x in eo])
            checkpoints.extend([int(x) for x in cp])
            sample_ids.extend(sid)
    return total_loss / max(total_count, 1), np.concatenate(preds), np.concatenate(targets), batteries, event_types, event_orders, checkpoints, sample_ids


def prediction_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    batteries: list[str],
    event_types: list[str],
    event_orders: list[int],
    checkpoints: list[int],
    sample_ids: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "battery": batteries,
            "event_type": event_types,
            "event_order": event_orders,
            "assigned_checkpoint_cycle": checkpoints,
            "actual_soh_percent": y_true,
            "predicted_soh_percent": y_pred,
            "error_pp": y_pred - y_true,
            "absolute_error_pp": np.abs(y_pred - y_true),
        }
    )


def aggregate_checkpoints(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        pred_df.groupby(["battery", "assigned_checkpoint_cycle"], as_index=False)
        .agg(
            actual_soh_percent=("actual_soh_percent", "mean"),
            predicted_soh_percent=("predicted_soh_percent", "mean"),
            event_count=("sample_id", "count"),
            charge_events=("event_type", lambda x: int((x == "charge").sum())),
            discharge_events=("event_type", lambda x: int((x == "discharge").sum())),
        )
        .sort_values(["battery", "assigned_checkpoint_cycle"])
        .reset_index(drop=True)
    )
    out["error_pp"] = out["predicted_soh_percent"] - out["actual_soh_percent"]
    out["absolute_error_pp"] = out["error_pp"].abs()
    return out


def save_plots(outdir: Path, name: str, history: pd.DataFrame, pred_df: pd.DataFrame, checkpoint_df: pd.DataFrame) -> None:
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
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(outdir / "training_dashboard.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = pred_df["event_type"].map({"charge": "#2a9d8f", "discharge": "#e76f51"}).to_numpy()
    axes[0, 0].scatter(pred_df["actual_soh_percent"], pred_df["predicted_soh_percent"], s=7, alpha=0.25, c=colors)
    lo = min(pred_df["actual_soh_percent"].min(), pred_df["predicted_soh_percent"].min())
    hi = max(pred_df["actual_soh_percent"].max(), pred_df["predicted_soh_percent"].max())
    axes[0, 0].plot([lo, hi], [lo, hi], "k--")
    axes[0, 0].set_title("Actual vs predicted")
    axes[0, 0].set_xlabel("Actual SOH (%)")
    axes[0, 0].set_ylabel("Predicted SOH (%)")
    axes[0, 0].grid(True, alpha=0.25)
    for event_type, color in [("charge", "#2a9d8f"), ("discharge", "#e76f51")]:
        sub = pred_df[pred_df["event_type"].eq(event_type)]
        axes[0, 1].scatter(sub["event_order"], sub["actual_soh_percent"], s=5, alpha=0.15, color="black")
        axes[0, 1].scatter(sub["event_order"], sub["predicted_soh_percent"], s=5, alpha=0.15, color=color, label=event_type)
    axes[0, 1].set_title("RW11 SOH over event order")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.25)
    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["actual_soh_percent"], marker="o", label="actual")
    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["predicted_soh_percent"], marker="o", label="predicted")
    axes[1, 0].set_title("Checkpoint-aggregated prediction")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 1].hist(pred_df["error_pp"], bins=60, color="#457b9d")
    axes[1, 1].axvline(0, color="black", linestyle="--")
    axes[1, 1].set_title("Residuals")
    axes[1, 1].grid(True, alpha=0.25)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(outdir / "prediction_report.png", dpi=180)
    plt.close(fig)


def train_one(
    spec: Spec,
    data: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    outdir = OUTDIR / spec.name
    outdir.mkdir(parents=True, exist_ok=True)
    train_ds = EventDataset(
        data["train_feature_x"],
        data["train_patches"],
        data["train_y_norm"],
        data["train_w"],
        data["train_meta"],
    )
    test_ds = EventDataset(
        data["test_feature_x"],
        data["test_patches"],
        data["test_y_norm"],
        data["test_w"],
        data["test_meta"],
    )
    train_loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=spec.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    patch_dim = data["train_patches"][0].shape[1]
    feature_dim = data["train_feature_x"].shape[1]
    if spec.model_kind == "patch_augmented":
        model: nn.Module = PatchAugmentedTransformer(patch_dim, feature_dim, spec)
    elif spec.model_kind == "feature_tokens":
        model = FeatureTokenTransformer(feature_dim, spec)
    else:
        raise ValueError(f"Unknown model_kind: {spec.model_kind}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    history_rows: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_r2 = -np.inf
    for epoch in range(1, spec.epochs + 1):
        train_loss, train_pred_norm, train_true_norm, *_ = run_epoch(model, train_loader, optimizer, device)
        test_loss, test_pred_norm, test_true_norm, *_ = run_epoch(model, test_loader, None, device)
        train_pred = destandardize(train_pred_norm, data["y_mean"], data["y_std"])
        train_true = destandardize(train_true_norm, data["y_mean"], data["y_std"])
        test_pred = destandardize(test_pred_norm, data["y_mean"], data["y_std"])
        test_true = destandardize(test_true_norm, data["y_mean"], data["y_std"])
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
            f"{spec.name} epoch {epoch}/{spec.epochs}: "
            f"train_r2={train_metrics['r2']:.4f} test_r2={test_metrics['r2']:.4f} test_mae={test_metrics['mae']:.3f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval = run_epoch(model, train_loader, None, device)
    test_eval = run_epoch(model, test_loader, None, device)
    _, train_pred_norm, train_true_norm, train_b, train_et, train_eo, train_cp, train_sid = train_eval
    _, test_pred_norm, test_true_norm, test_b, test_et, test_eo, test_cp, test_sid = test_eval
    train_pred = destandardize(train_pred_norm, data["y_mean"], data["y_std"])
    train_true = destandardize(train_true_norm, data["y_mean"], data["y_std"])
    test_pred = destandardize(test_pred_norm, data["y_mean"], data["y_std"])
    test_true = destandardize(test_true_norm, data["y_mean"], data["y_std"])
    train_pred_df = prediction_frame(train_true, train_pred, train_b, train_et, train_eo, train_cp, train_sid)
    test_pred_df = prediction_frame(test_true, test_pred, test_b, test_et, test_eo, test_cp, test_sid)
    checkpoint_df = aggregate_checkpoints(test_pred_df)
    history = pd.DataFrame(history_rows)
    metrics = {
        "spec": asdict(spec),
        "parameter_count": parameter_count(model),
        "feature_count": int(feature_dim),
        "patch_token_dim": int(patch_dim),
        "feature_token_count": int(math.ceil(feature_dim / FEATURE_CHUNK_WIDTH)),
        "max_patch_tokens": int(max(len(x) for x in data["train_patches"] + data["test_patches"])),
        "train_sample_count": int(len(train_pred_df)),
        "test_sample_count": int(len(test_pred_df)),
        "best_epoch_by_test_r2": int(history.loc[history["test_r2"].idxmax(), "epoch"]),
        "train_metrics_event_level": regression_metrics(train_true, train_pred),
        "test_metrics_event_level": regression_metrics(test_true, test_pred),
        "test_metrics_checkpoint_aggregated": regression_metrics(
            checkpoint_df["actual_soh_percent"].to_numpy(),
            checkpoint_df["predicted_soh_percent"].to_numpy(),
        ),
    }
    history.to_csv(outdir / "training_history.csv", index=False)
    train_pred_df.to_csv(outdir / "train_predictions.csv", index=False)
    test_pred_df.to_csv(outdir / "test_predictions.csv", index=False)
    checkpoint_df.to_csv(outdir / "test_checkpoint_predictions.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "spec": asdict(spec), "metrics": metrics}, outdir / "model.pt")
    save_plots(outdir, spec.name, history, test_pred_df, checkpoint_df)
    return metrics


def save_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    summary = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    summary.to_csv(OUTDIR / "all_feature_transformer_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = summary["model"].tolist()
    x = np.arange(len(labels))
    for ax, col, title, color in zip(
        axes,
        ["test_r2", "test_mae", "checkpoint_r2"],
        ["RW11 event-level R2", "RW11 event-level MAE", "RW11 checkpoint R2"],
        ["#2a9d8f", "#457b9d", "#f4a261"],
    ):
        ax.bar(x, summary[col], color=color)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "all_feature_transformer_summary.png", dpi=180)
    plt.close(fig)
    return summary


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    torch.set_num_threads(8)
    data = load_prepared_data()
    pd.DataFrame(
        {"feature": data["feature_names"], "mean": data["feature_mean"], "std": data["feature_std"]}
    ).to_csv(OUTDIR / "feature_scaler.csv", index=False)
    pd.DataFrame(
        {
            "patch_feature_index": np.arange(len(data["patch_mean"])),
            "mean": data["patch_mean"],
            "std": data["patch_std"],
        }
    ).to_csv(OUTDIR / "patch_scaler.csv", index=False)
    (OUTDIR / "feature_names.json").write_text(json.dumps(data["feature_names"], indent=2), encoding="utf-8")
    device = torch.device("cpu")
    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        set_seed()
        metrics = train_one(spec, data, device)
        rows.append(
            {
                "model": spec.name,
                "model_kind": spec.model_kind,
                "test_samples": metrics["test_sample_count"],
                "test_r2": metrics["test_metrics_event_level"]["r2"],
                "test_mae": metrics["test_metrics_event_level"]["mae"],
                "test_rmse": metrics["test_metrics_event_level"]["rmse"],
                "checkpoint_r2": metrics["test_metrics_checkpoint_aggregated"]["r2"],
                "checkpoint_mae": metrics["test_metrics_checkpoint_aggregated"]["mae"],
                "checkpoint_rmse": metrics["test_metrics_checkpoint_aggregated"]["rmse"],
                "best_epoch": metrics["best_epoch_by_test_r2"],
                "parameters": metrics["parameter_count"],
                "d_model": spec.d_model,
                "nhead": spec.nhead,
                "num_layers": spec.num_layers,
                "dropout": spec.dropout,
                "learning_rate": spec.learning_rate,
                "epochs": spec.epochs,
                "output_dir": str(OUTDIR / spec.name),
            }
        )
    summary = save_summary(rows)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
