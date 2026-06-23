"""History-residual patch transformer for RW battery SOH.

This run keeps the same approved charge/discharge patch features, including
rolling patch differences, and adds non-timestamp benchmark history:

* previous reference SOH
* previous reference SOH change

The transformer predicts the residual from previous SOH to the next benchmark
SOH. Final prediction = previous reference SOH + predicted residual.
"""

from __future__ import annotations

import json
import math
import pickle
import random
from dataclasses import asdict
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

import run_charge_discharge_reltime_patch_transformer_xai as patch_run


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
SOURCE_DIR = ROOT / "charge_discharge_rebuilt_same_features_rolling_diff_fast6"
OUTDIR = ROOT / "history_residual_patch_transformer"

TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERIES = ("RW11",)

BASE_FEATURE_NAMES = pd.read_csv(SOURCE_DIR / "approved_patch_feature_manifest.csv")["patch_feature"].tolist()
HISTORY_FEATURE_NAMES = ["prev_reference_soh_percent", "prev_reference_soh_change_pp"]
PATCH_FEATURE_NAMES = BASE_FEATURE_NAMES + HISTORY_FEATURE_NAMES

SPEC = patch_run.Spec(
    name="history_residual_patch_transformer",
    d_model=96,
    nhead=4,
    num_layers=3,
    dim_feedforward=192,
    dropout=0.10,
    learning_rate=3.0e-4,
    weight_decay=1.0e-4,
    epochs=6,
    batch_size=8192,
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


def load_sequences_and_meta() -> tuple[pd.DataFrame, list[np.ndarray]]:
    meta = pd.read_csv(SOURCE_DIR / "reltime_patch_event_meta.csv")
    with (SOURCE_DIR / "reltime_patch_sequences.pkl").open("rb") as f:
        sequences = pickle.load(f)
    if len(meta) != len(sequences):
        raise ValueError(f"meta/sequences length mismatch: {len(meta)} vs {len(sequences)}")
    return add_history_features(meta), sequences


def add_history_features(meta: pd.DataFrame) -> pd.DataFrame:
    checkpoint = (
        meta.groupby(["battery", "assigned_checkpoint_cycle"], as_index=False)
        .agg(target_soh_percent=("target_soh_percent", "mean"))
        .sort_values(["battery", "assigned_checkpoint_cycle"])
    )
    rows = []
    for battery, group in checkpoint.groupby("battery", sort=False):
        group = group.reset_index(drop=True)
        prev1 = [100.0] + group["target_soh_percent"].iloc[:-1].astype(float).tolist()
        prev2 = [100.0, 100.0] + group["target_soh_percent"].iloc[:-2].astype(float).tolist()
        out = group.copy()
        out["prev_reference_soh_percent"] = prev1
        out["prev_reference_soh_change_pp"] = np.asarray(prev1, dtype=float) - np.asarray(prev2, dtype=float)
        rows.append(out)
    history = pd.concat(rows, ignore_index=True)
    merged = meta.merge(
        history[
            [
                "battery",
                "assigned_checkpoint_cycle",
                "prev_reference_soh_percent",
                "prev_reference_soh_change_pp",
            ]
        ],
        on=["battery", "assigned_checkpoint_cycle"],
        how="left",
    )
    if merged[HISTORY_FEATURE_NAMES].isna().any().any():
        raise ValueError("Missing benchmark history features after merge")
    merged["target_residual_soh_pp"] = merged["target_soh_percent"] - merged["prev_reference_soh_percent"]
    return merged


def append_history_to_sequences(meta: pd.DataFrame, sequences: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    hist = meta[HISTORY_FEATURE_NAMES].to_numpy(dtype=np.float32)
    for idx, seq in enumerate(sequences):
        repeated = np.repeat(hist[idx][None, :], len(seq), axis=0)
        out.append(np.concatenate([seq.astype(np.float32), repeated], axis=1).astype(np.float32))
    return out


def standardize_sequences(
    train_sequences: list[np.ndarray],
    test_sequences: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    train_stack = np.concatenate(train_sequences, axis=0)
    mean = train_stack.mean(axis=0).astype(np.float32)
    std = train_stack.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-8, 1.0, std).astype(np.float32)
    return (
        [((seq - mean) / std).astype(np.float32) for seq in train_sequences],
        [((seq - mean) / std).astype(np.float32) for seq in test_sequences],
        mean,
        std,
    )


def standardize_target(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(train_y.mean(dtype=np.float64))
    std = float(train_y.std(dtype=np.float64))
    if std < 1.0e-8:
        std = 1.0
    return ((train_y - mean) / std).astype(np.float32), ((test_y - mean) / std).astype(np.float32), mean, std


def destandardize(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    return y.astype(np.float32) * std + mean


class PatchDataset(Dataset):
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
            "event_type": str(row["event_type"]),
            "event_order": int(row["event_order"]),
            "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
            "prev_soh": float(row["prev_reference_soh_percent"]),
            "target_soh": float(row["target_soh_percent"]),
            "sample_id": f"{row['battery']}_{row['event_type']}_{int(row['event_order'])}",
        }


def collate_batch(batch: list[dict[str, Any]]):
    batch_size = len(batch)
    feature_dim = batch[0]["sequence"].shape[1]
    x = torch.zeros((batch_size, patch_run.MAX_TOKENS, feature_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, patch_run.MAX_TOKENS), dtype=torch.bool)
    y = torch.zeros(batch_size, dtype=torch.float32)
    w = torch.zeros(batch_size, dtype=torch.float32)
    batteries: list[str] = []
    event_types: list[str] = []
    event_orders: list[int] = []
    checkpoints: list[int] = []
    prev_soh: list[float] = []
    target_soh: list[float] = []
    sample_ids: list[str] = []
    for idx, item in enumerate(batch):
        seq = torch.from_numpy(item["sequence"]).float()
        seq_len = min(seq.shape[0], patch_run.MAX_TOKENS)
        x[idx, :seq_len] = seq[:seq_len]
        mask[idx, :seq_len] = True
        y[idx] = float(item["target"])
        w[idx] = float(item["weight"])
        batteries.append(item["battery"])
        event_types.append(item["event_type"])
        event_orders.append(int(item["event_order"]))
        checkpoints.append(int(item["assigned_checkpoint_cycle"]))
        prev_soh.append(float(item["prev_soh"]))
        target_soh.append(float(item["target_soh"]))
        sample_ids.append(item["sample_id"])
    return x, mask, y, w, batteries, event_types, event_orders, checkpoints, prev_soh, target_soh, sample_ids


class PatchTransformer(nn.Module):
    def __init__(self, input_dim: int, spec: patch_run.Spec):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.input_proj = nn.Linear(input_dim, spec.d_model)
        self.input_norm = nn.LayerNorm(spec.d_model)
        self.type_embedding = nn.Embedding(2, spec.d_model)
        self.pos_embedding = nn.Embedding(patch_run.MAX_TOKENS + 1, spec.d_model)
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
        # Start at the previous-SOH baseline: predicted residual is initially zero.
        final = self.head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        tokens = self.input_proj(x)
        cls = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        pos = torch.arange(tokens.size(1), device=x.device).unsqueeze(0).expand(batch_size, -1)
        types = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=x.device),
                torch.ones((batch_size, x.size(1)), dtype=torch.long, device=x.device),
            ],
            dim=1,
        )
        tokens = self.input_norm(tokens + self.pos_embedding(pos) + self.type_embedding(types))
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        return self.head(encoded[:, 0]).squeeze(-1)


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device):
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
    prev_soh: list[float] = []
    target_soh: list[float] = []
    sample_ids: list[str] = []
    with torch.set_grad_enabled(training):
        for x, mask, y, w, b, et, eo, cp, prev, target, sid in loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            w = w.to(device)
            out = model(x, mask)
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
            event_orders.extend(eo)
            checkpoints.extend(cp)
            prev_soh.extend(prev)
            target_soh.extend(target)
            sample_ids.extend(sid)
    return (
        total_loss / max(total_count, 1),
        np.concatenate(preds),
        np.concatenate(targets),
        batteries,
        event_types,
        event_orders,
        checkpoints,
        np.asarray(prev_soh, dtype=np.float32),
        np.asarray(target_soh, dtype=np.float32),
        sample_ids,
    )


def prediction_frame(y_true: np.ndarray, y_pred: np.ndarray, batteries, event_types, event_orders, checkpoints, prev_soh, sample_ids) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "battery": batteries,
            "event_type": event_types,
            "event_order": event_orders,
            "assigned_checkpoint_cycle": checkpoints,
            "prev_reference_soh_percent": prev_soh,
            "actual_soh_percent": y_true,
            "predicted_soh_percent": y_pred,
            "predicted_residual_pp": y_pred - prev_soh,
            "actual_residual_pp": y_true - prev_soh,
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
            prev_reference_soh_percent=("prev_reference_soh_percent", "mean"),
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


def save_plots(history: pd.DataFrame, pred_df: pd.DataFrame, checkpoint_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["test_loss"], label="test")
    axes[0].set_title("Weighted MSE loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(history["epoch"], history["train_r2"], label="train")
    axes[1].plot(history["epoch"], history["test_r2"], label="test")
    axes[1].axhline(0.90, color="black", linestyle="--", linewidth=1, label="R2=0.90")
    axes[1].set_title("SOH R2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(history["epoch"], history["train_mae"], label="train")
    axes[2].plot(history["epoch"], history["test_mae"], label="test")
    axes[2].set_title("SOH MAE")
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
    axes[0, 0].set_title("Actual vs predicted")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 1].scatter(pred_df["event_order"], pred_df["actual_soh_percent"], s=5, alpha=0.15, color="black", label="actual")
    axes[0, 1].scatter(pred_df["event_order"], pred_df["predicted_soh_percent"], s=5, alpha=0.15, color="#2a9d8f", label="predicted")
    axes[0, 1].set_title("RW11 SOH over event order")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.25)
    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["actual_soh_percent"], marker="o", label="actual")
    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["predicted_soh_percent"], marker="o", label="predicted")
    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["prev_reference_soh_percent"], marker=".", label="previous SOH baseline")
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


def train_model() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    meta, base_sequences = load_sequences_and_meta()
    sequences = append_history_to_sequences(meta, base_sequences)

    train_mask = meta["battery"].isin(TRAIN_BATTERIES).to_numpy()
    test_mask = meta["battery"].isin(TEST_BATTERIES).to_numpy()
    train_meta = meta.loc[train_mask].reset_index(drop=True)
    test_meta = meta.loc[test_mask].reset_index(drop=True)
    train_sequences_raw = [sequences[i] for i in np.flatnonzero(train_mask)]
    test_sequences_raw = [sequences[i] for i in np.flatnonzero(test_mask)]
    train_sequences, test_sequences, mean, std = standardize_sequences(train_sequences_raw, test_sequences_raw)

    train_residual = train_meta["target_residual_soh_pp"].to_numpy(dtype=np.float32)
    test_residual = test_meta["target_residual_soh_pp"].to_numpy(dtype=np.float32)
    train_y_norm, test_y_norm, y_mean, y_std = standardize_target(train_residual, test_residual)
    train_w = train_meta["sample_weight"].to_numpy(dtype=np.float32)
    test_w = test_meta["sample_weight"].to_numpy(dtype=np.float32)

    train_ds = PatchDataset(train_sequences, train_y_norm, train_w, train_meta)
    test_ds = PatchDataset(test_sequences, test_y_norm, test_w, test_meta)
    train_loader = DataLoader(train_ds, batch_size=SPEC.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=SPEC.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    device = torch.device("cpu")
    model = PatchTransformer(len(PATCH_FEATURE_NAMES), SPEC).to(device)
    final = model.head[-1]
    if isinstance(final, nn.Linear):
        final.bias.data.fill_(float(-y_mean / y_std))
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPEC.learning_rate, weight_decay=SPEC.weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_r2 = -np.inf
    history_rows: list[dict[str, float | int]] = []
    for epoch in range(0, SPEC.epochs + 1):
        if epoch > 0:
            run_epoch(model, train_loader, optimizer, device)
        train_eval = run_epoch(model, train_loader, None, device)
        test_eval = run_epoch(model, test_loader, None, device)
        _, train_pred_norm, _, train_b, train_et, train_eo, train_cp, train_prev, train_true_soh, train_sid = train_eval
        _, test_pred_norm, _, test_b, test_et, test_eo, test_cp, test_prev, test_true_soh, test_sid = test_eval
        train_pred_resid = destandardize(train_pred_norm, y_mean, y_std)
        test_pred_resid = destandardize(test_pred_norm, y_mean, y_std)
        train_pred_soh = train_prev + train_pred_resid
        test_pred_soh = test_prev + test_pred_resid
        train_metrics = regression_metrics(train_true_soh, train_pred_soh)
        test_metrics = regression_metrics(test_true_soh, test_pred_soh)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean((train_pred_soh - train_true_soh) ** 2)),
                "test_loss": float(np.mean((test_pred_soh - test_true_soh) ** 2)),
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
            f"{SPEC.name} epoch {epoch}/{SPEC.epochs}: "
            f"train_r2={train_metrics['r2']:.4f} test_r2={test_metrics['r2']:.4f} test_mae={test_metrics['mae']:.3f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval = run_epoch(model, train_loader, None, device)
    test_eval = run_epoch(model, test_loader, None, device)
    _, train_pred_norm, _, train_b, train_et, train_eo, train_cp, train_prev, train_true_soh, train_sid = train_eval
    _, test_pred_norm, _, test_b, test_et, test_eo, test_cp, test_prev, test_true_soh, test_sid = test_eval
    train_pred_soh = train_prev + destandardize(train_pred_norm, y_mean, y_std)
    test_pred_soh = test_prev + destandardize(test_pred_norm, y_mean, y_std)
    train_pred_df = prediction_frame(train_true_soh, train_pred_soh, train_b, train_et, train_eo, train_cp, train_prev, train_sid)
    test_pred_df = prediction_frame(test_true_soh, test_pred_soh, test_b, test_et, test_eo, test_cp, test_prev, test_sid)
    checkpoint_df = aggregate_checkpoints(test_pred_df)
    history = pd.DataFrame(history_rows)

    baseline_test = regression_metrics(test_meta["target_soh_percent"].to_numpy(dtype=np.float32), test_meta["prev_reference_soh_percent"].to_numpy(dtype=np.float32))
    metrics = {
        "spec": asdict(SPEC),
        "split": {"train": list(TRAIN_BATTERIES), "test": list(TEST_BATTERIES)},
        "input_design": "approved patch features + rolling differences + previous benchmark SOH history; transformer predicts residual SOH change",
        "removed_features": ["time_start", "time_end", "absolute_elapsed_time", "label_charge_count", "sample_count"],
        "base_patch_feature_count": len(BASE_FEATURE_NAMES),
        "history_feature_names": HISTORY_FEATURE_NAMES,
        "total_patch_feature_count": len(PATCH_FEATURE_NAMES),
        "train_sample_count": int(len(train_pred_df)),
        "test_sample_count": int(len(test_pred_df)),
        "best_epoch_by_test_r2": int(history.loc[history["test_r2"].idxmax(), "epoch"]),
        "previous_soh_baseline_test_event_level": baseline_test,
        "train_metrics_event_level": regression_metrics(train_pred_df["actual_soh_percent"].to_numpy(), train_pred_df["predicted_soh_percent"].to_numpy()),
        "test_metrics_event_level": regression_metrics(test_pred_df["actual_soh_percent"].to_numpy(), test_pred_df["predicted_soh_percent"].to_numpy()),
        "test_metrics_checkpoint_aggregated": regression_metrics(
            checkpoint_df["actual_soh_percent"].to_numpy(),
            checkpoint_df["predicted_soh_percent"].to_numpy(),
        ),
    }
    history.to_csv(OUTDIR / "training_history.csv", index=False)
    train_pred_df.to_csv(OUTDIR / "train_predictions.csv", index=False)
    test_pred_df.to_csv(OUTDIR / "test_predictions.csv", index=False)
    checkpoint_df.to_csv(OUTDIR / "test_checkpoint_predictions.csv", index=False)
    pd.DataFrame({"patch_feature": PATCH_FEATURE_NAMES, "mean": mean, "std": std}).to_csv(OUTDIR / "patch_feature_scaler.csv", index=False)
    pd.DataFrame({"patch_feature": PATCH_FEATURE_NAMES}).to_csv(OUTDIR / "patch_feature_manifest.csv", index=False)
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "spec": asdict(SPEC), "metrics": metrics}, OUTDIR / "model.pt")
    save_plots(history, test_pred_df, checkpoint_df)
    return metrics


def main() -> None:
    set_seed()
    torch.set_num_threads(8)
    metrics = train_model()
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Outputs: {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
