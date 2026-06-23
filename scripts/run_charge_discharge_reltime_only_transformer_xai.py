from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import lime.lime_tabular
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

import run_charge_discharge_full_cycle_suite as suite


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
SOURCE_DIR = ROOT / "charge_discharge_full_cycle_suite"
OUTDIR = ROOT / "charge_discharge_reltime_only_transformer_xai"
TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERIES = ("RW11",)
FEATURE_CHUNK_WIDTH = 8
BACKGROUND_SIZE = 128
SHAP_SAMPLES_PER_EVENT_TYPE = 200
LIME_NUM_FEATURES = 15
LIME_NUM_SAMPLES = 5000


@dataclass(frozen=True)
class Spec:
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


SPEC = Spec(
    name="feature_token_transformer_reltime_only",
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.12,
    learning_rate=7.0e-4,
    weight_decay=5.0e-4,
    epochs=22,
    batch_size=4096,
)


DISALLOWED_FEATURES = {
    "label_charge_count",
    "sample_count",
    "time_start",
    "time_end",
}


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


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def standardize_target(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(train_y.mean(dtype=np.float64))
    std = float(train_y.std(dtype=np.float64))
    if std < 1.0e-8:
        std = 1.0
    return ((train_y - mean) / std).astype(np.float32), ((test_y - mean) / std).astype(np.float32), mean, std


def destandardize(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    return y.astype(np.float32) * std + mean


def load_data() -> dict[str, object]:
    features, _ = suite.build_or_load_event_table_and_patches()
    features = suite.add_block_weights(features.rename(columns={"label_charge_count": "charge_cycle_for_label"}))
    features = features.rename(columns={"charge_cycle_for_label": "label_charge_count"})
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
        *DISALLOWED_FEATURES,
    }
    feature_names = [
        col
        for col in features.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(features[col])
    ]

    train_mask = features["battery"].isin(TRAIN_BATTERIES).to_numpy()
    test_mask = features["battery"].isin(TEST_BATTERIES).to_numpy()
    train_meta = features.loc[train_mask].reset_index(drop=True)
    test_meta = features.loc[test_mask].reset_index(drop=True)
    train_raw = np.nan_to_num(train_meta[feature_names].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    test_raw = np.nan_to_num(test_meta[feature_names].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    train_x, test_x, feature_mean, feature_std = suite.standardize_matrix(train_raw, test_raw)
    train_y = train_meta["soh_percent"].to_numpy(dtype=np.float32)
    test_y = test_meta["soh_percent"].to_numpy(dtype=np.float32)
    train_w = train_meta["sample_weight"].to_numpy(dtype=np.float32)
    test_w = test_meta["sample_weight"].to_numpy(dtype=np.float32)
    train_y_norm, test_y_norm, y_mean, y_std = standardize_target(train_y, test_y)
    return {
        "features": features,
        "feature_names": feature_names,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "train_raw": train_raw,
        "test_raw": test_raw,
        "train_x": train_x,
        "test_x": test_x,
        "train_y": train_y,
        "test_y": test_y,
        "train_w": train_w,
        "test_w": test_w,
        "train_y_norm": train_y_norm,
        "test_y_norm": test_y_norm,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


class FeatureDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, meta: pd.DataFrame):
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.weights = weights.astype(np.float32, copy=False)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.meta.iloc[idx]
        return {
            "x": self.x[idx],
            "y": float(self.y[idx]),
            "weight": float(self.weights[idx]),
            "battery": str(row["battery"]),
            "event_type": str(row["event_type"]),
            "event_order": int(row["event_order"]),
            "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
            "sample_id": f"{row['battery']}_{row['event_type']}_{int(row['event_order'])}",
        }


def collate_batch(batch: list[dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str], list[int], list[int], list[str]]:
    x = torch.tensor(np.stack([item["x"] for item in batch]), dtype=torch.float32)
    y = torch.tensor([float(item["y"]) for item in batch], dtype=torch.float32)
    w = torch.tensor([float(item["weight"]) for item in batch], dtype=torch.float32)
    batteries = [str(item["battery"]) for item in batch]
    event_types = [str(item["event_type"]) for item in batch]
    event_orders = [int(item["event_order"]) for item in batch]
    checkpoints = [int(item["assigned_checkpoint_cycle"]) for item in batch]
    sample_ids = [str(item["sample_id"]) for item in batch]
    return x, y, w, batteries, event_types, event_orders, checkpoints, sample_ids


class FeatureTokenTransformer(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        pad = self.chunk_count * FEATURE_CHUNK_WIDTH - self.feature_dim
        if pad:
            x = nn.functional.pad(x, (0, pad))
        chunks = x.view(batch_size, self.chunk_count, FEATURE_CHUNK_WIDTH)
        ids = torch.arange(self.chunk_count, device=x.device)
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
        for x, y, w, b, et, eo, cp, sid in loader:
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
            total_loss += float(loss.item()) * int(y.size(0))
            total_count += int(y.size(0))
            preds.append(out.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
            batteries.extend(b)
            event_types.extend(et)
            event_orders.extend(eo)
            checkpoints.extend(cp)
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


def save_training_plots(history: pd.DataFrame, pred_df: pd.DataFrame, checkpoint_df: pd.DataFrame) -> None:
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
    colors = pred_df["event_type"].map({"charge": "#2a9d8f", "discharge": "#e76f51"}).to_numpy()
    axes[0, 0].scatter(pred_df["actual_soh_percent"], pred_df["predicted_soh_percent"], s=7, alpha=0.25, c=colors)
    lo = min(pred_df["actual_soh_percent"].min(), pred_df["predicted_soh_percent"].min())
    hi = max(pred_df["actual_soh_percent"].max(), pred_df["predicted_soh_percent"].max())
    axes[0, 0].plot([lo, hi], [lo, hi], "k--")
    axes[0, 0].set_title("Actual vs predicted")
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
    fig.tight_layout()
    fig.savefig(OUTDIR / "prediction_report.png", dpi=180)
    plt.close(fig)


def train_model(data: dict[str, object]) -> tuple[FeatureTokenTransformer, dict[str, object], np.ndarray]:
    device = torch.device("cpu")
    train_ds = FeatureDataset(data["train_x"], data["train_y_norm"], data["train_w"], data["train_meta"])  # type: ignore[arg-type]
    test_ds = FeatureDataset(data["test_x"], data["test_y_norm"], data["test_w"], data["test_meta"])  # type: ignore[arg-type]
    train_loader = DataLoader(train_ds, batch_size=SPEC.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=SPEC.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    model = FeatureTokenTransformer(len(data["feature_names"]), SPEC).to(device)  # type: ignore[arg-type]
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPEC.learning_rate, weight_decay=SPEC.weight_decay)
    history_rows: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_r2 = -np.inf
    for epoch in range(1, SPEC.epochs + 1):
        train_loss, train_pred_norm, train_true_norm, *_ = run_epoch(model, train_loader, optimizer, device)
        test_loss, test_pred_norm, test_true_norm, *_ = run_epoch(model, test_loader, None, device)
        train_pred = destandardize(train_pred_norm, data["y_mean"], data["y_std"])  # type: ignore[arg-type]
        train_true = destandardize(train_true_norm, data["y_mean"], data["y_std"])  # type: ignore[arg-type]
        test_pred = destandardize(test_pred_norm, data["y_mean"], data["y_std"])  # type: ignore[arg-type]
        test_true = destandardize(test_true_norm, data["y_mean"], data["y_std"])  # type: ignore[arg-type]
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
            f"{SPEC.name} epoch {epoch}/{SPEC.epochs}: "
            f"train_r2={train_metrics['r2']:.4f} test_r2={test_metrics['r2']:.4f} test_mae={test_metrics['mae']:.3f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval = run_epoch(model, train_loader, None, device)
    test_eval = run_epoch(model, test_loader, None, device)
    _, train_pred_norm, train_true_norm, train_b, train_et, train_eo, train_cp, train_sid = train_eval
    _, test_pred_norm, test_true_norm, test_b, test_et, test_eo, test_cp, test_sid = test_eval
    train_pred = destandardize(train_pred_norm, data["y_mean"], data["y_std"])  # type: ignore[arg-type]
    train_true = destandardize(train_true_norm, data["y_mean"], data["y_std"])  # type: ignore[arg-type]
    test_pred = destandardize(test_pred_norm, data["y_mean"], data["y_std"])  # type: ignore[arg-type]
    test_true = destandardize(test_true_norm, data["y_mean"], data["y_std"])  # type: ignore[arg-type]
    train_pred_df = prediction_frame(train_true, train_pred, train_b, train_et, train_eo, train_cp, train_sid)
    test_pred_df = prediction_frame(test_true, test_pred, test_b, test_et, test_eo, test_cp, test_sid)
    checkpoint_df = aggregate_checkpoints(test_pred_df)
    history = pd.DataFrame(history_rows)
    metrics = {
        "spec": asdict(SPEC),
        "removed_features": sorted(DISALLOWED_FEATURES),
        "feature_count": len(data["feature_names"]),  # type: ignore[arg-type]
        "feature_token_count": int(math.ceil(len(data["feature_names"]) / FEATURE_CHUNK_WIDTH)),  # type: ignore[arg-type]
        "parameter_count": parameter_count(model),
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
    history.to_csv(OUTDIR / "training_history.csv", index=False)
    train_pred_df.to_csv(OUTDIR / "train_predictions.csv", index=False)
    test_pred_df.to_csv(OUTDIR / "test_predictions.csv", index=False)
    checkpoint_df.to_csv(OUTDIR / "test_checkpoint_predictions.csv", index=False)
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "spec": asdict(SPEC), "metrics": metrics}, OUTDIR / "model.pt")
    save_training_plots(history, test_pred_df, checkpoint_df)
    return model, metrics, test_pred


class SohWrapper(nn.Module):
    def __init__(self, model: nn.Module, y_mean: float, y_std: float):
        super().__init__()
        self.model = model
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.model(x) * self.y_std + self.y_mean).unsqueeze(-1)


def transformer_predict_fn(model: nn.Module, y_mean: float, y_std: float) -> Callable[[np.ndarray], np.ndarray]:
    def predict(x: np.ndarray) -> np.ndarray:
        model.eval()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                batch = torch.tensor(x[start : start + 4096], dtype=torch.float32)
                pred = model(batch).detach().cpu().numpy() * y_std + y_mean
                outs.append(pred)
        return np.concatenate(outs).reshape(-1)

    return predict


def clean_shap_values(values: object) -> np.ndarray:
    if isinstance(values, list):
        values = values[0]
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return arr.astype(np.float64, copy=False)


def background_indices(train_meta: pd.DataFrame, size: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    picks = []
    for event_type in ("charge", "discharge"):
        idx = np.flatnonzero(train_meta["event_type"].eq(event_type).to_numpy())
        picks.append(rng.choice(idx, size=min(size // 2, len(idx)), replace=False))
    return np.concatenate(picks)


def stratified_test_indices(test_meta: pd.DataFrame, per_event_type: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    picks = []
    for event_type in ("charge", "discharge"):
        idx = np.flatnonzero(test_meta["event_type"].eq(event_type).to_numpy())
        picks.append(rng.choice(idx, size=min(per_event_type, len(idx)), replace=False))
    return np.concatenate(picks)


def save_shap_outputs(shap_values: np.ndarray, shap_x: np.ndarray, shap_meta: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    xai_dir = OUTDIR / "xai"
    xai_dir.mkdir(exist_ok=True)
    charge_mask = shap_meta["event_type"].eq("charge").to_numpy()
    discharge_mask = shap_meta["event_type"].eq("discharge").to_numpy()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap_combined": np.mean(np.abs(shap_values), axis=0),
            "mean_abs_shap_charge": np.mean(np.abs(shap_values[charge_mask]), axis=0),
            "mean_abs_shap_discharge": np.mean(np.abs(shap_values[discharge_mask]), axis=0),
        }
    ).sort_values("mean_abs_shap_combined", ascending=False)
    importance.to_csv(xai_dir / "global_shap_feature_importance.csv", index=False)

    top = importance.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    y = np.arange(len(top))
    ax.barh(y - 0.2, top["mean_abs_shap_charge"], height=0.2, label="charge", color="#2a9d8f")
    ax.barh(y, top["mean_abs_shap_discharge"], height=0.2, label="discharge", color="#e76f51")
    ax.barh(y + 0.2, top["mean_abs_shap_combined"], height=0.2, label="combined", color="#457b9d")
    ax.set_yticks(y, top["feature"])
    ax.set_title("Global SHAP without absolute time / charge count / sample count")
    ax.set_xlabel("Mean |SHAP| in SOH percentage points")
    ax.legend(frameon=False)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(xai_dir / "global_shap_bar_by_event_type.png", dpi=180)
    plt.close(fig)

    for label, mask in [("combined", np.ones(len(shap_x), dtype=bool)), ("charge", charge_mask), ("discharge", discharge_mask)]:
        plt.figure(figsize=(9, 7))
        shap.summary_plot(shap_values[mask], shap_x[mask], feature_names=feature_names, max_display=20, show=False)
        plt.title(f"SHAP beeswarm ({label})")
        plt.tight_layout()
        plt.savefig(xai_dir / f"global_shap_beeswarm_{label}.png", dpi=180, bbox_inches="tight")
        plt.close()
    return importance


def choose_local_indices(test_meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    abs_err = np.abs(y_pred - y_true)
    for event_type in ("charge", "discharge"):
        idx = np.flatnonzero(test_meta["event_type"].eq(event_type).to_numpy())
        ordered = idx[np.argsort(abs_err[idx])]
        out[event_type] = int(ordered[len(ordered) // 2])
    return out


def run_lime(
    train_x: np.ndarray,
    test_x: np.ndarray,
    test_meta: pd.DataFrame,
    test_y: np.ndarray,
    test_pred: np.ndarray,
    feature_names: list[str],
    predict_fn: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    xai_dir = OUTDIR / "xai"
    local_indices = choose_local_indices(test_meta, test_y, test_pred)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_x,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=True,
        random_state=42,
    )
    rows = []
    for event_type, idx in local_indices.items():
        exp = explainer.explain_instance(
            test_x[idx],
            predict_fn,
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(9, 6)
        plt.title(f"Local LIME: {event_type} event")
        plt.tight_layout()
        fig.savefig(xai_dir / f"local_lime_{event_type}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        exp.save_to_file(str(xai_dir / f"local_lime_{event_type}.html"))
        row = test_meta.iloc[idx]
        for rank, (feature_rule, contribution) in enumerate(exp.as_list(), start=1):
            rows.append(
                {
                    "event_type": event_type,
                    "rank": rank,
                    "feature_rule": feature_rule,
                    "lime_contribution_soh_pp": float(contribution),
                    "battery": row["battery"],
                    "event_order": int(row["event_order"]),
                    "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
                    "actual_soh_percent": float(test_y[idx]),
                    "predicted_soh_percent": float(test_pred[idx]),
                    "absolute_error_pp": float(abs(test_pred[idx] - test_y[idx])),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(xai_dir / "local_lime_explanations.csv", index=False)
    return out


def run_xai(model: FeatureTokenTransformer, data: dict[str, object], test_pred: np.ndarray) -> None:
    xai_dir = OUTDIR / "xai"
    xai_dir.mkdir(exist_ok=True)
    wrapped = SohWrapper(model, data["y_mean"], data["y_std"])  # type: ignore[arg-type]
    wrapped.eval()
    bg_idx = background_indices(data["train_meta"], BACKGROUND_SIZE)  # type: ignore[arg-type]
    shap_idx = stratified_test_indices(data["test_meta"], SHAP_SAMPLES_PER_EVENT_TYPE)  # type: ignore[arg-type]
    explainer = shap.GradientExplainer(
        wrapped,
        torch.tensor(data["train_x"][bg_idx], dtype=torch.float32),  # type: ignore[index]
    )
    shap_values = clean_shap_values(explainer.shap_values(torch.tensor(data["test_x"][shap_idx], dtype=torch.float32)))  # type: ignore[index]
    shap_meta = data["test_meta"].iloc[shap_idx].reset_index(drop=True)  # type: ignore[union-attr]
    importance = save_shap_outputs(shap_values, data["test_x"][shap_idx], shap_meta, data["feature_names"])  # type: ignore[index,arg-type]
    predict_fn = transformer_predict_fn(model, data["y_mean"], data["y_std"])  # type: ignore[arg-type]
    lime_df = run_lime(
        data["train_x"],  # type: ignore[arg-type]
        data["test_x"],  # type: ignore[arg-type]
        data["test_meta"],  # type: ignore[arg-type]
        data["test_y"],  # type: ignore[arg-type]
        test_pred,
        data["feature_names"],  # type: ignore[arg-type]
        predict_fn,
    )
    summary = {
        "top10_shap_features": importance.head(10)["feature"].tolist(),
        "shap_background_train_samples": BACKGROUND_SIZE,
        "shap_test_samples_per_event_type": SHAP_SAMPLES_PER_EVENT_TYPE,
        "lime_num_samples": LIME_NUM_SAMPLES,
        "removed_features": sorted(DISALLOWED_FEATURES),
    }
    (xai_dir / "xai_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_feature_manifest(feature_names: list[str]) -> None:
    rows = []
    family_map = {
        "event_type_discharge": "Original event category",
        "duration_s": "RelTime / event duration",
        "charge_throughput_ah": "Charge throughput",
        "energy_throughput_wh": "Energy throughput",
        "power_mean_w": "Energy / power",
        "power_max_w": "Energy / power",
        "voltage": "Voltage shape",
        "current": "Current statistics",
        "temperature": "Thermal response",
        "time_in_vwin": "Voltage-window timing",
        "time_cross": "Voltage-window timing",
        "dq_dv": "IC / DV features",
        "dv_dq": "IC / DV features",
    }
    for feature in feature_names:
        family = "Distributional statistics"
        for key, value in family_map.items():
            if feature.startswith(key) or key in feature:
                family = value
                break
        rows.append({"feature": feature, "family": family})
    pd.DataFrame(rows).to_csv(OUTDIR / "approved_reltime_only_feature_manifest.csv", index=False)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    torch.set_num_threads(8)
    data = load_data()
    save_feature_manifest(data["feature_names"])  # type: ignore[arg-type]
    pd.DataFrame({"feature": data["feature_names"], "mean": data["feature_mean"], "std": data["feature_std"]}).to_csv(
        OUTDIR / "feature_scaler.csv",
        index=False,
    )
    (OUTDIR / "feature_names.json").write_text(json.dumps(data["feature_names"], indent=2), encoding="utf-8")
    model, metrics, test_pred = train_model(data)
    run_xai(model, data, test_pred)
    print(json.dumps(metrics, indent=2), flush=True)
    print("Outputs:", OUTDIR, flush=True)


if __name__ == "__main__":
    main()
