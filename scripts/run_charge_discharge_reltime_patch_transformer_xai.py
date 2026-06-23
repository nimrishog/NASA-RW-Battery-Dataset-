from __future__ import annotations

import json
import math
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

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
SEGMENT_DIR = ROOT / "charge_discharge_weighted_transformer" / "extracted_segments"
OUTDIR = ROOT / "charge_discharge_reltime_patch_transformer_xai"
TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERIES = ("RW11",)
BATTERIES = (*TRAIN_BATTERIES, *TEST_BATTERIES)
PATCH_SIZE = 30
MAX_TOKENS = 11
BACKGROUND_SIZE = 128
SHAP_SAMPLES_PER_EVENT_TYPE = 160
LIME_NUM_FEATURES = 18
LIME_NUM_SAMPLES = 3500


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
    name="reltime_patch30_transformer",
    d_model=96,
    nhead=4,
    num_layers=3,
    dim_feedforward=192,
    dropout=0.15,
    learning_rate=5.0e-4,
    weight_decay=5.0e-4,
    epochs=18,
    batch_size=4096,
)


PATCH_FEATURE_NAMES = [
    "event_type_discharge",
    "duration_s",
    "rel_start_s",
    "rel_end_s",
    "charge_throughput_ah",
    "energy_throughput_wh",
    "power_mean_w",
    "power_max_w",
    "voltage_start",
    "voltage_end",
    "voltage_delta",
    "voltage_abs_delta",
    "voltage_mean",
    "voltage_std",
    "voltage_min",
    "voltage_max",
    "voltage_slope_vps",
    "dv_dt_mean_vps",
    "dv_dt_median_vps",
    "dv_dt_max_abs_vps",
    "current_mean_a",
    "current_abs_mean_a",
    "current_std_a",
    "current_min_a",
    "current_max_a",
    "temperature_start_c",
    "temperature_end_c",
    "temperature_delta_c",
    "temperature_abs_delta_c",
    "temperature_mean_c",
    "temperature_std_c",
    "dtemp_dt_mean_cps",
    "dtemp_dt_median_cps",
    "dtemp_dt_max_abs_cps",
    "time_in_vwin_3p8_4p0_s",
    "time_in_vwin_4p0_4p1_s",
    "time_cross_3p9_to_4p1_s",
    "dq_dv_median_ahpv",
    "dq_dv_max_ahpv",
    "dq_dv_min_ahpv",
    "dv_dq_median_vpah",
    "dv_dq_max_vpah",
    "dv_dq_min_vpah",
    "voltage_q25",
    "voltage_median",
    "voltage_q75",
    "temperature_q25",
    "temperature_median",
    "temperature_q75",
    "patch_position",
]


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


def safe_nan_stat(values: np.ndarray, fn: Callable[[np.ndarray], float], default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return default
    return float(fn(values))


def patch_features(
    event_type: str,
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    rel_time: np.ndarray,
    patch_idx: int,
    token_count: int,
) -> np.ndarray:
    duration = float(rel_time[-1] - rel_time[0]) if len(rel_time) > 1 else 0.0
    abs_current = np.abs(current)
    power = voltage * abs_current
    if len(voltage) > 1:
        dt = np.clip(np.diff(rel_time).astype(np.float64), 0.0, None)
        charge_ah = float(np.sum(abs_current[:-1] * dt) / 3600.0)
        energy_wh = float(np.sum(voltage[:-1] * abs_current[:-1] * dt) / 3600.0)
        dv = np.diff(voltage).astype(np.float64)
        dtemp = np.diff(temperature).astype(np.float64)
        valid_dt = dt > 1.0e-8
        dv_dt = np.divide(dv, np.where(valid_dt, dt, np.nan))
        dtemp_dt = np.divide(dtemp, np.where(valid_dt, dt, np.nan))
        dq = abs_current[:-1] * dt / 3600.0
        valid_dv = np.abs(dv) > 1.0e-6
        dq_dv = dq[valid_dv] / np.abs(dv[valid_dv]) if np.any(valid_dv) else np.array([0.0])
        valid_dq = dq > 1.0e-10
        dv_dq = np.abs(dv[valid_dq]) / dq[valid_dq] if np.any(valid_dq) else np.array([0.0])
    else:
        charge_ah = 0.0
        energy_wh = 0.0
        dv_dt = np.array([0.0])
        dtemp_dt = np.array([0.0])
        dq_dv = np.array([0.0])
        dv_dq = np.array([0.0])

    rel_step = np.diff(rel_time, prepend=rel_time[0])
    time_in_38_40 = float(np.sum(rel_step[(voltage >= 3.8) & (voltage < 4.0)]))
    time_in_40_41 = float(np.sum(rel_step[(voltage >= 4.0) & (voltage < 4.1)]))
    cross_mask = (voltage >= 3.9) & (voltage <= 4.1)
    time_cross = float(rel_time[cross_mask][-1] - rel_time[cross_mask][0]) if np.sum(cross_mask) > 1 else 0.0

    row = np.array(
        [
            1.0 if event_type == "discharge" else 0.0,
            duration,
            float(rel_time[0]),
            float(rel_time[-1]),
            charge_ah,
            energy_wh,
            float(np.mean(power)),
            float(np.max(power)),
            float(voltage[0]),
            float(voltage[-1]),
            float(voltage[-1] - voltage[0]),
            float(abs(voltage[-1] - voltage[0])),
            float(np.mean(voltage)),
            float(np.std(voltage)),
            float(np.min(voltage)),
            float(np.max(voltage)),
            float((voltage[-1] - voltage[0]) / max(duration, 1.0e-8)),
            safe_nan_stat(dv_dt, np.nanmean),
            safe_nan_stat(dv_dt, np.nanmedian),
            safe_nan_stat(np.abs(dv_dt), np.nanmax),
            float(np.mean(current)),
            float(np.mean(abs_current)),
            float(np.std(current)),
            float(np.min(current)),
            float(np.max(current)),
            float(temperature[0]),
            float(temperature[-1]),
            float(temperature[-1] - temperature[0]),
            float(abs(temperature[-1] - temperature[0])),
            float(np.mean(temperature)),
            float(np.std(temperature)),
            safe_nan_stat(dtemp_dt, np.nanmean),
            safe_nan_stat(dtemp_dt, np.nanmedian),
            safe_nan_stat(np.abs(dtemp_dt), np.nanmax),
            time_in_38_40,
            time_in_40_41,
            time_cross,
            float(np.median(dq_dv)),
            float(np.max(dq_dv)),
            float(np.min(dq_dv)),
            float(np.median(dv_dq)),
            float(np.max(dv_dq)),
            float(np.min(dv_dq)),
            float(np.quantile(voltage, 0.25)),
            float(np.quantile(voltage, 0.50)),
            float(np.quantile(voltage, 0.75)),
            float(np.quantile(temperature, 0.25)),
            float(np.quantile(temperature, 0.50)),
            float(np.quantile(temperature, 0.75)),
            float((patch_idx + 1) / max(token_count, 1)),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)


def build_event_patch_sequence(event_type: str, df: pd.DataFrame) -> np.ndarray:
    voltage = df["voltage"].to_numpy(dtype=np.float64)
    current = df["current"].to_numpy(dtype=np.float64)
    temperature = df["temperature"].to_numpy(dtype=np.float64)
    rel_time = df["relativeTime"].to_numpy(dtype=np.float64)
    token_count = int(math.ceil(len(df) / PATCH_SIZE))
    rows = []
    for patch_idx, start in enumerate(range(0, len(df), PATCH_SIZE)):
        end = min(start + PATCH_SIZE, len(df))
        rows.append(
            patch_features(
                event_type,
                voltage[start:end],
                current[start:end],
                temperature[start:end],
                rel_time[start:end],
                patch_idx,
                token_count,
            )
        )
    return np.stack(rows, axis=0).astype(np.float32)


def build_or_load_sequences() -> tuple[pd.DataFrame, list[np.ndarray]]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUTDIR / "reltime_patch_event_meta.csv"
    seq_path = OUTDIR / "reltime_patch_sequences.pkl"
    if meta_path.exists() and seq_path.exists():
        meta = pd.read_csv(meta_path)
        with seq_path.open("rb") as f:
            sequences = pickle.load(f)
        return meta, sequences

    full_features, _ = suite.build_or_load_event_table_and_patches()
    features = suite.add_block_weights(full_features.rename(columns={"label_charge_count": "charge_cycle_for_label"}))
    features = features.rename(columns={"charge_cycle_for_label": "label_charge_count"})
    key_meta = features[
        [
            "battery",
            "event_type",
            "event_order",
            "assigned_checkpoint_cycle",
            "soh_percent",
            "capacity_ah",
            "sample_weight",
            "block_size",
        ]
    ].copy()
    keyed_rows = {
        (str(row.battery), str(row.event_type), int(row.event_order)): row._asdict()
        for row in key_meta.itertuples(index=False)
    }

    sequences: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for battery in BATTERIES:
        for event_type in ("charge", "discharge"):
            path = SEGMENT_DIR / f"{battery}_{event_type}_random_walk_segments.pkl"
            df = pd.read_pickle(path).sort_values(["event_order", "sample_index"]).reset_index(drop=True)
            for event_order, group in df.groupby("event_order", sort=True):
                key = (battery, event_type, int(event_order))
                if key not in keyed_rows:
                    continue
                info = keyed_rows[key]
                seq = build_event_patch_sequence(event_type, group)
                sequences.append(seq)
                meta_rows.append(
                    {
                        "battery": battery,
                        "event_type": event_type,
                        "event_order": int(event_order),
                        "assigned_checkpoint_cycle": int(info["assigned_checkpoint_cycle"]),
                        "target_soh_percent": float(info["soh_percent"]),
                        "target_capacity_ah": float(info["capacity_ah"]),
                        "sample_weight": float(info["sample_weight"]),
                        "block_size": int(info["block_size"]),
                        "token_count": int(len(seq)),
                        "raw_row_count": int(len(group)),
                    }
                )
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(meta_path, index=False)
    with seq_path.open("wb") as f:
        pickle.dump(sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
    return meta, sequences


def standardize_sequences(
    train_sequences: list[np.ndarray],
    test_sequences: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    train_stack = np.concatenate(train_sequences, axis=0)
    mean = train_stack.mean(axis=0).astype(np.float32)
    std = train_stack.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-8, 1.0, std).astype(np.float32)
    train_out = [((seq - mean) / std).astype(np.float32) for seq in train_sequences]
    test_out = [((seq - mean) / std).astype(np.float32) for seq in test_sequences]
    return train_out, test_out, mean, std


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
            "sample_id": f"{row['battery']}_{row['event_type']}_{int(row['event_order'])}",
        }


def collate_batch(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str], list[int], list[int], list[str]]:
    batch_size = len(batch)
    feature_dim = batch[0]["sequence"].shape[1]
    x = torch.zeros((batch_size, MAX_TOKENS, feature_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, MAX_TOKENS), dtype=torch.bool)
    y = torch.zeros(batch_size, dtype=torch.float32)
    w = torch.zeros(batch_size, dtype=torch.float32)
    batteries: list[str] = []
    event_types: list[str] = []
    event_orders: list[int] = []
    checkpoints: list[int] = []
    sample_ids: list[str] = []
    for idx, item in enumerate(batch):
        seq = torch.from_numpy(item["sequence"]).float()
        seq_len = min(seq.shape[0], MAX_TOKENS)
        x[idx, :seq_len] = seq[:seq_len]
        mask[idx, :seq_len] = True
        y[idx] = float(item["target"])
        w[idx] = float(item["weight"])
        batteries.append(item["battery"])
        event_types.append(item["event_type"])
        event_orders.append(int(item["event_order"]))
        checkpoints.append(int(item["assigned_checkpoint_cycle"]))
        sample_ids.append(item["sample_id"])
    return x, mask, y, w, batteries, event_types, event_orders, checkpoints, sample_ids


class PatchTransformer(nn.Module):
    def __init__(self, input_dim: int, spec: Spec):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.input_proj = nn.Linear(input_dim, spec.d_model)
        self.input_norm = nn.LayerNorm(spec.d_model)
        self.type_embedding = nn.Embedding(2, spec.d_model)
        self.pos_embedding = nn.Embedding(MAX_TOKENS + 1, spec.d_model)
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
        for x, mask, y, w, b, et, eo, cp, sid in loader:
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


def train_model() -> tuple[PatchTransformer, dict[str, Any], dict[str, Any], np.ndarray]:
    meta, sequences = build_or_load_sequences()
    train_mask = meta["battery"].isin(TRAIN_BATTERIES).to_numpy()
    test_mask = meta["battery"].isin(TEST_BATTERIES).to_numpy()
    train_meta = meta.loc[train_mask].reset_index(drop=True)
    test_meta = meta.loc[test_mask].reset_index(drop=True)
    train_sequences_raw = [sequences[i] for i in np.flatnonzero(train_mask)]
    test_sequences_raw = [sequences[i] for i in np.flatnonzero(test_mask)]
    train_sequences, test_sequences, mean, std = standardize_sequences(train_sequences_raw, test_sequences_raw)
    train_y = train_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    test_y = test_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    train_w = train_meta["sample_weight"].to_numpy(dtype=np.float32)
    test_w = test_meta["sample_weight"].to_numpy(dtype=np.float32)
    train_y_norm, test_y_norm, y_mean, y_std = standardize_target(train_y, test_y)

    train_ds = PatchDataset(train_sequences, train_y_norm, train_w, train_meta)
    test_ds = PatchDataset(test_sequences, test_y_norm, test_w, test_meta)
    train_loader = DataLoader(train_ds, batch_size=SPEC.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=SPEC.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    device = torch.device("cpu")
    model = PatchTransformer(len(PATCH_FEATURE_NAMES), SPEC).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPEC.learning_rate, weight_decay=SPEC.weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_r2 = -np.inf
    history_rows: list[dict[str, float | int]] = []
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
    train_pred = destandardize(train_pred_norm, y_mean, y_std)
    train_true = destandardize(train_true_norm, y_mean, y_std)
    test_pred = destandardize(test_pred_norm, y_mean, y_std)
    test_true = destandardize(test_true_norm, y_mean, y_std)
    train_pred_df = prediction_frame(train_true, train_pred, train_b, train_et, train_eo, train_cp, train_sid)
    test_pred_df = prediction_frame(test_true, test_pred, test_b, test_et, test_eo, test_cp, test_sid)
    checkpoint_df = aggregate_checkpoints(test_pred_df)
    history = pd.DataFrame(history_rows)
    metrics = {
        "spec": asdict(SPEC),
        "removed_features": ["time_start", "time_end", "label_charge_count", "absolute_elapsed_time", "sample_count"],
        "patch_feature_names": PATCH_FEATURE_NAMES,
        "patch_feature_count": len(PATCH_FEATURE_NAMES),
        "max_tokens": MAX_TOKENS,
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
    pd.DataFrame({"patch_feature": PATCH_FEATURE_NAMES, "mean": mean, "std": std}).to_csv(OUTDIR / "patch_feature_scaler.csv", index=False)
    pd.DataFrame({"patch_feature": PATCH_FEATURE_NAMES}).to_csv(OUTDIR / "approved_patch_feature_manifest.csv", index=False)
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "spec": asdict(SPEC), "metrics": metrics}, OUTDIR / "model.pt")
    save_plots(history, test_pred_df, checkpoint_df)
    xai_data = {
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "train_y": train_y,
        "test_y": test_y,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return model, metrics, xai_data, test_pred


class SohWrapper(nn.Module):
    def __init__(self, model: nn.Module, y_mean: float, y_std: float):
        super().__init__()
        self.model = model
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.any(torch.abs(x) > 1.0e-8, dim=-1)
        return (self.model(x, mask) * self.y_std + self.y_mean).unsqueeze(-1)


def pad_sequences(sequences: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(sequences), MAX_TOKENS, len(PATCH_FEATURE_NAMES)), dtype=np.float32)
    for idx, seq in enumerate(sequences):
        seq_len = min(len(seq), MAX_TOKENS)
        out[idx, :seq_len] = seq[:seq_len]
    return out


def model_predict_flat(model: nn.Module, y_mean: float, y_std: float) -> Callable[[np.ndarray], np.ndarray]:
    def predict(flat_x: np.ndarray) -> np.ndarray:
        x = flat_x.reshape((-1, MAX_TOKENS, len(PATCH_FEATURE_NAMES))).astype(np.float32)
        mask = np.any(np.abs(x) > 1.0e-8, axis=-1)
        outs: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(x), 1024):
                batch_x = torch.tensor(x[start : start + 1024], dtype=torch.float32)
                batch_mask = torch.tensor(mask[start : start + 1024], dtype=torch.bool)
                pred = model(batch_x, batch_mask).detach().cpu().numpy() * y_std + y_mean
                outs.append(pred)
        return np.concatenate(outs).reshape(-1)

    return predict


def clean_shap_values(values: object) -> np.ndarray:
    if isinstance(values, list):
        values = values[0]
    arr = np.asarray(values)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[:, :, :, 0]
    return arr.astype(np.float64, copy=False)


def run_xai(model: PatchTransformer, xai_data: dict[str, Any], test_pred: np.ndarray) -> None:
    xai_dir = OUTDIR / "xai"
    xai_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    train_meta: pd.DataFrame = xai_data["train_meta"]
    test_meta: pd.DataFrame = xai_data["test_meta"]
    train_padded = pad_sequences(xai_data["train_sequences"])
    test_padded = pad_sequences(xai_data["test_sequences"])

    bg_idx = []
    shap_idx = []
    for event_type in ("charge", "discharge"):
        train_idx = np.flatnonzero(train_meta["event_type"].eq(event_type).to_numpy())
        test_idx = np.flatnonzero(test_meta["event_type"].eq(event_type).to_numpy())
        bg_idx.extend(rng.choice(train_idx, size=min(BACKGROUND_SIZE // 2, len(train_idx)), replace=False).tolist())
        shap_idx.extend(rng.choice(test_idx, size=min(SHAP_SAMPLES_PER_EVENT_TYPE, len(test_idx)), replace=False).tolist())
    bg_idx = np.array(bg_idx, dtype=np.int64)
    shap_idx = np.array(shap_idx, dtype=np.int64)

    wrapped = SohWrapper(model, xai_data["y_mean"], xai_data["y_std"])
    wrapped.eval()
    explainer = shap.GradientExplainer(
        wrapped,
        torch.tensor(train_padded[bg_idx], dtype=torch.float32),
    )
    shap_values = clean_shap_values(explainer.shap_values(torch.tensor(test_padded[shap_idx], dtype=torch.float32)))
    shap_meta = test_meta.iloc[shap_idx].reset_index(drop=True)

    feature_importance = np.mean(np.abs(shap_values), axis=(0, 1))
    charge_mask = shap_meta["event_type"].eq("charge").to_numpy()
    discharge_mask = shap_meta["event_type"].eq("discharge").to_numpy()
    importance = pd.DataFrame(
        {
            "patch_feature": PATCH_FEATURE_NAMES,
            "mean_abs_shap_combined": feature_importance,
            "mean_abs_shap_charge": np.mean(np.abs(shap_values[charge_mask]), axis=(0, 1)),
            "mean_abs_shap_discharge": np.mean(np.abs(shap_values[discharge_mask]), axis=(0, 1)),
        }
    ).sort_values("mean_abs_shap_combined", ascending=False)
    importance.to_csv(xai_dir / "global_shap_patch_feature_importance.csv", index=False)

    top = importance.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    y = np.arange(len(top))
    ax.barh(y - 0.2, top["mean_abs_shap_charge"], height=0.2, label="charge", color="#2a9d8f")
    ax.barh(y, top["mean_abs_shap_discharge"], height=0.2, label="discharge", color="#e76f51")
    ax.barh(y + 0.2, top["mean_abs_shap_combined"], height=0.2, label="combined", color="#457b9d")
    ax.set_yticks(y, top["patch_feature"])
    ax.set_xlabel("Mean |SHAP| aggregated over patch tokens")
    ax.set_title("Patch transformer global SHAP, relTime-only features")
    ax.legend(frameon=False)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(xai_dir / "global_shap_bar_by_event_type.png", dpi=180)
    plt.close(fig)

    heatmap = np.mean(np.abs(shap_values), axis=0)
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(heatmap.T, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(MAX_TOKENS), [f"P{i+1}" for i in range(MAX_TOKENS)])
    ax.set_yticks(np.arange(len(PATCH_FEATURE_NAMES)), PATCH_FEATURE_NAMES, fontsize=6)
    ax.set_title("Mean |SHAP| by patch token and patch feature")
    fig.colorbar(im, ax=ax, label="Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(xai_dir / "global_shap_patch_feature_heatmap.png", dpi=180)
    plt.close(fig)

    # LIME on flattened padded patch sequence.
    flat_train = train_padded.reshape((len(train_padded), -1))
    flat_test = test_padded.reshape((len(test_padded), -1))
    flat_feature_names = [
        f"patch_{patch_idx + 1:02d}_{feature}"
        for patch_idx in range(MAX_TOKENS)
        for feature in PATCH_FEATURE_NAMES
    ]
    predict_fn = model_predict_flat(model, xai_data["y_mean"], xai_data["y_std"])
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        flat_train,
        feature_names=flat_feature_names,
        mode="regression",
        discretize_continuous=True,
        random_state=42,
    )
    test_y = xai_data["test_y"]
    abs_err = np.abs(test_pred - test_y)
    rows = []
    for event_type in ("charge", "discharge"):
        idxs = np.flatnonzero(test_meta["event_type"].eq(event_type).to_numpy())
        ordered = idxs[np.argsort(abs_err[idxs])]
        idx = int(ordered[len(ordered) // 2])
        exp = explainer_lime.explain_instance(
            flat_test[idx],
            predict_fn,
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(9, 6)
        plt.title(f"Patch transformer local LIME: {event_type}")
        plt.tight_layout()
        fig.savefig(xai_dir / f"local_lime_{event_type}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        exp.save_to_file(str(xai_dir / f"local_lime_{event_type}.html"))
        row = test_meta.iloc[idx]
        for rank, (rule, contribution) in enumerate(exp.as_list(), start=1):
            rows.append(
                {
                    "event_type": event_type,
                    "rank": rank,
                    "feature_rule": rule,
                    "lime_contribution_soh_pp": float(contribution),
                    "battery": row["battery"],
                    "event_order": int(row["event_order"]),
                    "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
                    "actual_soh_percent": float(test_y[idx]),
                    "predicted_soh_percent": float(test_pred[idx]),
                    "absolute_error_pp": float(abs(test_pred[idx] - test_y[idx])),
                }
            )
    pd.DataFrame(rows).to_csv(xai_dir / "local_lime_explanations.csv", index=False)
    (xai_dir / "xai_summary.json").write_text(
        json.dumps(
            {
                "shap_background_train_samples": int(len(bg_idx)),
                "shap_test_samples_per_event_type": SHAP_SAMPLES_PER_EVENT_TYPE,
                "lime_num_samples": LIME_NUM_SAMPLES,
                "top10_patch_shap_features": importance.head(10)["patch_feature"].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    torch.set_num_threads(8)
    model, metrics, xai_data, test_pred = train_model()
    run_xai(model, xai_data, test_pred)
    print(json.dumps(metrics, indent=2), flush=True)
    print("Outputs:", OUTDIR, flush=True)


if __name__ == "__main__":
    main()
