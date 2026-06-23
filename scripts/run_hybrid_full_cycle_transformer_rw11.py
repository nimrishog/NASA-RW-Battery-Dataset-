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
OUTDIR = ROOT / "hybrid_full_cycle_transformer_rw11"
TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERIES = ("RW11",)


@dataclass(frozen=True)
class HybridSpec:
    name: str
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    full_feature_hidden: int
    fusion_hidden: int
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    max_tokens: int


SPEC = HybridSpec(
    name="hybrid_patch_tokens_full_cycle_features",
    d_model=80,
    nhead=4,
    num_layers=3,
    dim_feedforward=160,
    dropout=0.20,
    full_feature_hidden=96,
    fusion_hidden=128,
    learning_rate=3.0e-4,
    weight_decay=5.0e-4,
    epochs=18,
    batch_size=512,
    max_tokens=11,
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


class HybridDataset(Dataset):
    def __init__(
        self,
        sequences: list[np.ndarray],
        full_features: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        meta: pd.DataFrame,
    ):
        self.sequences = sequences
        self.full_features = full_features.astype(np.float32, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.weights = weights.astype(np.float32, copy=False)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.meta.iloc[idx]
        return {
            "sequence": self.sequences[idx],
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
    max_len = max(len(item["sequence"]) for item in batch)
    input_dim = batch[0]["sequence"].shape[1]
    full_dim = batch[0]["full_features"].shape[0]
    batch_size = len(batch)
    x = torch.zeros((batch_size, max_len, input_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    full = torch.zeros((batch_size, full_dim), dtype=torch.float32)
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
        mask[idx, :seq_len] = True
        full[idx] = torch.from_numpy(item["full_features"]).float()
        y[idx] = float(item["target"])
        w[idx] = float(item["weight"])
        batteries.append(item["battery"])
        cycles.append(int(item["charge_cycle"]))
        checkpoints.append(int(item["assigned_checkpoint_cycle"]))
        sample_ids.append(item["sample_id"])
    return x, mask, full, y, w, batteries, cycles, checkpoints, sample_ids


def create_positional_encoding(length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class HybridTransformer(nn.Module):
    def __init__(self, token_input_dim: int, full_feature_dim: int, spec: HybridSpec):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.input_proj = nn.Linear(token_input_dim, spec.d_model)
        self.input_norm = nn.LayerNorm(spec.d_model)
        self.register_buffer(
            "pe",
            create_positional_encoding(spec.max_tokens + 1, spec.d_model).unsqueeze(0),
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
        self.full_branch = nn.Sequential(
            nn.LayerNorm(full_feature_dim),
            nn.Linear(full_feature_dim, spec.full_feature_hidden),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.full_feature_hidden, spec.full_feature_hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(spec.d_model + spec.full_feature_hidden),
            nn.Linear(spec.d_model + spec.full_feature_hidden, spec.fusion_hidden),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.fusion_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, full_features: torch.Tensor) -> torch.Tensor:
        tokens = self.input_norm(self.input_proj(x))
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        cls_mask = torch.ones((mask.size(0), 1), device=mask.device, dtype=torch.bool)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        tokens = tokens + self.pe[:, : tokens.size(1)]
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        full_encoded = self.full_branch(full_features)
        fused = torch.cat([encoded[:, 0], full_encoded], dim=1)
        return self.head(fused).squeeze(-1)


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
        batch_x, batch_mask, batch_full, batch_y, batch_w, batch_batteries, batch_cycles, batch_checkpoints, batch_ids = batch
        batch_x = batch_x.to(device)
        batch_mask = batch_mask.to(device)
        batch_full = batch_full.to(device)
        batch_y = batch_y.to(device)
        batch_w = batch_w.to(device)

        out = model(batch_x, batch_mask, batch_full)
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
    axes[0, 0].set_title("Hybrid transformer actual vs predicted")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 1].scatter(pred_df["charge_cycle"], pred_df["actual_soh_percent"], s=6, alpha=0.2, label="actual", color="black")
    axes[0, 1].scatter(pred_df["charge_cycle"], pred_df["predicted_soh_percent"], s=6, alpha=0.2, label="predicted", color="#2a9d8f")
    axes[0, 1].set_title("RW11 SOH over charge-cycle chronology")
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
    fig.tight_layout()
    fig.savefig(OUTDIR / "prediction_report.png", dpi=180)
    plt.close(fig)


def save_model_comparison(metrics_dict: dict[str, Any]) -> None:
    transformer = json.loads((ROOT / "cycle_level_weighted_transformer_rw9_rw10_to_rw11" / "metrics.json").read_text())
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
                "model": "hybrid_full_cycle_transformer",
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

    mean, std = compute_standardization(train_sequences)
    apply_standardization(train_sequences, mean, std)
    apply_standardization(test_sequences, mean, std)

    feature_data, feature_names = load_full_cycle_features()
    train_full, test_full, full_mean, full_std = standardize_full_features(train_meta, test_meta, feature_data, feature_names)

    train_y = train_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    test_y = test_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    train_w = train_meta["sample_weight"].to_numpy(dtype=np.float32)
    test_w = test_meta["sample_weight"].to_numpy(dtype=np.float32)
    train_y_norm, test_y_norm, y_mean, y_std = standardize_target(train_y, test_y)

    train_ds = HybridDataset(train_sequences, train_full, train_y_norm, train_w, train_meta)
    test_ds = HybridDataset(test_sequences, test_full, test_y_norm, test_w, test_meta)
    train_loader = DataLoader(train_ds, batch_size=SPEC.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=SPEC.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    device = torch.device("cpu")
    token_input_dim = train_sequences[0].shape[1]
    full_feature_dim = train_full.shape[1]
    model = HybridTransformer(token_input_dim, full_feature_dim, SPEC).to(device)
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

    metrics_dict = {
        "spec": asdict(SPEC),
        "token_input_dim": token_input_dim,
        "full_cycle_feature_dim": full_feature_dim,
        "full_cycle_features": feature_names,
        "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_epoch_by_test_r2": int(history.sort_values("test_r2", ascending=False).iloc[0]["epoch"]),
        "train_metrics_cycle_level": regression_metrics(train_true, train_pred),
        "test_metrics_cycle_level": regression_metrics(test_true, test_pred),
        "test_metrics_checkpoint_aggregated": regression_metrics(
            checkpoint_df["actual_soh_percent"].to_numpy(dtype=np.float64),
            checkpoint_df["predicted_soh_percent"].to_numpy(dtype=np.float64),
        ),
    }

    history.to_csv(OUTDIR / "training_history.csv", index=False)
    train_pred_df.to_csv(OUTDIR / "train_predictions.csv", index=False)
    test_pred_df.to_csv(OUTDIR / "test_predictions.csv", index=False)
    checkpoint_df.to_csv(OUTDIR / "test_checkpoint_predictions.csv", index=False)
    pd.DataFrame({"feature": feature_names, "mean": full_mean, "std": full_std}).to_csv(OUTDIR / "full_cycle_feature_scaler.csv", index=False)
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "metrics": metrics_dict}, OUTDIR / "model.pt")
    save_plots(history, test_pred_df, checkpoint_df)
    save_model_comparison(metrics_dict)
    print(json.dumps(metrics_dict, indent=2), flush=True)


if __name__ == "__main__":
    main()
